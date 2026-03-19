"""
SciX Classifier Celery App

This module defines the `SciXClassifierCelery` class, which extends the `ADSCelery`
class and facilitates the classification and validation of scientific articles for
inclusion in specific collections. It interfaces with SQLAlchemy models and a 
configuration to log, index, validate, and update classification results.

The app uses a predefined set of allowed categories from the config and saves
classification metadata such as models, scores, overrides, and collections into
the database. It handles both automatic classification and manual overrides for 
validation purposes.

Dependencies:
    - SQLAlchemy ORM for database interaction
    - adsputils for configuration and logging
    - ClassifierPipeline.models for ORM models
    - ClassifierPipeline.utilities for helper utilities

Configuration:
    - ALLOWED_CATEGORIES: Set of allowed classification categories
    - CLASSIFICATION_PRETRAINED_MODEL, etc.: Model configuration for classification
"""
import os
import json
import pickle
import zlib
import csv
import time

import ClassifierPipeline.models as models
import ClassifierPipeline.utilities as utils
import ClassifierPipeline.perf_metrics as perf_metrics
from adsputils import get_date, ADSCelery, u2asc
from contextlib import contextmanager
from sqlalchemy import create_engine, desc, and_, or_
from sqlalchemy.orm import scoped_session, sessionmaker
from adsputils import load_config, setup_logging



proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
config = load_config(proj_home=proj_home)
logger = setup_logging('app.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', True))

ALLOWED_CATEGORIES = set(config['ALLOWED_CATEGORIES'])

class SciXClassifierCelery(ADSCelery):
    """
    SciXClassifierCelery

    Inherits from ADSCelery to handle message ingestion, classification processing,
    database indexing, and validation of scientific records.

    Responsibilities:
    - Ingest and parse classification requests
    - Index classification runs and results into the database
    - Manage model and override tracking
    - Update validated collections and handle manual overrides
    """

    def _ensure_runtime_caches(self):
        if not hasattr(self, "_cached_model_metadata_key"):
            self._cached_model_metadata_key = None
        if not hasattr(self, "_cached_model_id"):
            self._cached_model_id = None
        if not hasattr(self, "_run_model_bound"):
            self._run_model_bound = set()

    def _record_context_id(self, record):
        if record is None:
            return None
        return record.get("perf_metrics_context_id")

    def _records_context_id(self, records):
        if not records:
            return None
        return self._record_context_id(records[0])

    def _build_model_metadata(self, run_id=None, context_id=None):
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_build_model_metadata",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            config=config,
        ):
            model_dict = {
                'model': config['CLASSIFICATION_PRETRAINED_MODEL'],
                'revision': config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                'tokenizer': config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER'],
            }
            postprocessing_dict = {
                'ADDITIONAL_EARTH_SCIENCE_PROCESSING': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'],
                'CLASSIFICATION_THRESHOLDS': config['CLASSIFICATION_THRESHOLDS'],
            }
            return json.dumps(model_dict, sort_keys=True), json.dumps(postprocessing_dict, sort_keys=True)

    def _get_or_create_model_id(self, session, run_id=None, context_id=None):
        self._ensure_runtime_caches()
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_get_or_create_model_id",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            config=config,
        ):
            model_json, postprocessing_json = self._build_model_metadata(run_id=run_id, context_id=context_id)
            cache_key = (model_json, postprocessing_json)
            if self._cached_model_metadata_key == cache_key and self._cached_model_id is not None:
                return self._cached_model_id

            model_row = session.query(models.ModelTable).filter(
                and_(models.ModelTable.model == model_json, models.ModelTable.postprocessing == postprocessing_json)
            ).order_by(models.ModelTable.created.desc()).first()

            if model_row is None:
                model_row = models.ModelTable(model=model_json, postprocessing=postprocessing_json)
                session.add(model_row)
                session.flush()

            self._cached_model_metadata_key = cache_key
            self._cached_model_id = model_row.id
            return model_row.id

    def _ensure_run_model(self, session, run_id, model_id, context_id=None):
        self._ensure_runtime_caches()
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_ensure_run_model",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            config=config,
        ):
            if run_id in self._run_model_bound:
                return

            run_row = session.query(models.RunTable).filter(models.RunTable.id == run_id).order_by(models.RunTable.created.desc()).first()
            if run_row is not None and run_row.model_id != model_id:
                run_row.model_id = model_id
            self._run_model_bound.add(run_id)

    def add_record_to_output_file(self, record):
        with perf_metrics.timed_profile(
            category="app_timing",
            name="add_record_to_output_file",
            run_id=record.get("run_id"),
            context_id=self._record_context_id(record),
            record_id=record.get("scix_id") or record.get("bibcode"),
            config=config,
        ):
            utils.add_record_to_output_file(record)

    def _record_key(self, record):
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_record_key",
            run_id=record.get("run_id"),
            context_id=self._record_context_id(record),
            record_id=record.get("scix_id") or record.get("bibcode"),
            config=config,
        ):
            if record.get("scix_id"):
                return ("scix_id", record.get("scix_id"))
            return ("bibcode", record.get("bibcode"))

    def _result_record_key(self, bibcode=None, scix_id=None):
        if scix_id:
            return ("scix_id", scix_id)
        return ("bibcode", bibcode)

    def _prefetch_overrides(self, session, records):
        run_id = records[0].get("run_id") if records else None
        context_id = self._records_context_id(records)
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_prefetch_overrides",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"record_count": len(records)},
            config=config,
        ):
            bibcodes = {record.get("bibcode") for record in records if record.get("bibcode")}
            scix_ids = {record.get("scix_id") for record in records if record.get("scix_id")}
            if not bibcodes and not scix_ids:
                return {"bibcode": {}, "scix_id": {}}

            override_rows = session.query(models.OverrideTable).filter(
                or_(
                    and_(models.OverrideTable.scix_id.in_(list(scix_ids)) if scix_ids else False, models.OverrideTable.scix_id != None),
                    and_(models.OverrideTable.bibcode.in_(list(bibcodes)) if bibcodes else False, models.OverrideTable.bibcode != None),
                )
            ).order_by(models.OverrideTable.created.desc()).all()

            overrides = {"bibcode": {}, "scix_id": {}}
            for row in override_rows:
                if getattr(row, "scix_id", None) and row.scix_id not in overrides["scix_id"]:
                    overrides["scix_id"][row.scix_id] = row
                if getattr(row, "bibcode", None) and row.bibcode not in overrides["bibcode"]:
                    overrides["bibcode"][row.bibcode] = row
            return overrides

    def _prefetch_scores(self, session, batch_specs):
        run_id = batch_specs[0]["record"].get("run_id") if batch_specs else None
        context_id = self._record_context_id(batch_specs[0]["record"]) if batch_specs else None
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_prefetch_scores",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"record_count": len(batch_specs)},
            config=config,
        ):
            run_ids = {spec["record"].get("run_id") for spec in batch_specs if spec["record"].get("run_id") is not None}
            bibcodes = {spec["record"].get("bibcode") for spec in batch_specs if spec["record"].get("bibcode")}
            scix_ids = {spec["record"].get("scix_id") for spec in batch_specs if spec["record"].get("scix_id")}
            if not run_ids:
                return {}

            score_rows = session.query(models.ScoreTable).filter(
                and_(
                    models.ScoreTable.run_id.in_(list(run_ids)),
                    or_(
                        and_(models.ScoreTable.scix_id.in_(list(scix_ids)) if scix_ids else False, models.ScoreTable.scix_id != None),
                        and_(models.ScoreTable.bibcode.in_(list(bibcodes)) if bibcodes else False, models.ScoreTable.bibcode != None),
                    ),
                )
            ).order_by(models.ScoreTable.created.desc()).all()

            dedupe = {}
            for row in score_rows:
                if getattr(row, "scix_id", None):
                    key = ("scix_id", row.scix_id, row.scores, getattr(row, "overrides_id", None), row.run_id)
                else:
                    key = ("bibcode", row.bibcode, row.scores, getattr(row, "overrides_id", None), row.run_id)
                dedupe.setdefault(key, row)
            return dedupe

    def _prefetch_final_collections(self, session, records):
        run_id = records[0].get("run_id") if records else None
        context_id = self._records_context_id(records)
        with perf_metrics.timed_profile(
            category="app_timing",
            name="_prefetch_final_collections",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"record_count": len(records)},
            config=config,
        ):
            bibcodes = {record.get("bibcode") for record in records if record.get("bibcode")}
            scix_ids = {record.get("scix_id") for record in records if record.get("scix_id")}
            if not bibcodes and not scix_ids:
                return {"bibcode": {}, "scix_id": {}}

            final_rows = session.query(models.FinalCollectionTable).filter(
                or_(
                    and_(models.FinalCollectionTable.scix_id.in_(list(scix_ids)) if scix_ids else False, models.FinalCollectionTable.scix_id != None),
                    and_(models.FinalCollectionTable.bibcode.in_(list(bibcodes)) if bibcodes else False, models.FinalCollectionTable.bibcode != None),
                )
            ).order_by(models.FinalCollectionTable.created.desc()).all()

            finals = {"bibcode": {}, "scix_id": {}}
            for row in final_rows:
                if getattr(row, "scix_id", None) and row.scix_id not in finals["scix_id"]:
                    finals["scix_id"][row.scix_id] = row
                if getattr(row, "bibcode", None) and row.bibcode not in finals["bibcode"]:
                    finals["bibcode"][row.bibcode] = row
            return finals

    def index_records_batch(self, records):
        if not records:
            return []
        run_id = records[0].get("run_id")
        context_id = self._records_context_id(records)
        with perf_metrics.timed_profile(
            category="app_timing",
            name="index_records_batch",
            run_id=run_id,
            context_id=context_id,
            record_id=None,
            extra={"record_count": len(records)},
            config=config,
        ):
            batch_specs = []
            stage_start = time.perf_counter()
            with self.session_scope() as session:
                model_id = self._get_or_create_model_id(session, run_id=run_id, context_id=context_id)
                run_ids = sorted({record.get("run_id") for record in records if record.get("run_id") is not None})
                for item_run_id in run_ids:
                    self._ensure_run_model(session, item_run_id, model_id, context_id=context_id)

                overrides = self._prefetch_overrides(session, records)

                for index, record in enumerate(records):
                    if 'operation_step' not in record:
                        record['operation_step'] = 'classify'
                    if 'bibcode' not in record:
                        record['bibcode'] = None
                    if 'scix_id' not in record:
                        record['scix_id'] = None

                    override_row = None
                    if record.get("scix_id"):
                        override_row = overrides["scix_id"].get(record["scix_id"])
                    if override_row is None and record.get("bibcode"):
                        override_row = overrides["bibcode"].get(record["bibcode"])

                    final_collections = override_row.override if override_row is not None else record["collections"]
                    overrides_id = override_row.id if override_row is not None else None
                    scores_dict = {cat: score for cat, score in zip(config['ALLOWED_CATEGORIES'], record['scores'])}
                    score_payload = {
                        'scores': scores_dict,
                        'earth_science_adjustment': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                        'collections': record['collections'],
                    }
                    score_json = json.dumps(score_payload, sort_keys=True)
                    identifier_kind, identifier_value = self._record_key(record)
                    dedupe_key = (identifier_kind, identifier_value, score_json, overrides_id, record.get("run_id"))
                    batch_specs.append({
                        "index": index,
                        "record": record,
                        "final_collections": final_collections,
                        "overrides_id": overrides_id,
                        "score_json": score_json,
                        "dedupe_key": dedupe_key,
                    })

                existing_scores = self._prefetch_scores(session, batch_specs)
                new_score_rows = []
                reused_score_rows = 0
                for spec in batch_specs:
                    existing_score = existing_scores.get(spec["dedupe_key"])
                    if existing_score is not None:
                        spec["score_id"] = existing_score.id
                        reused_score_rows += 1
                        continue
                    record = spec["record"]
                    score_row = models.ScoreTable(
                        bibcode=record['bibcode'],
                        scix_id=record['scix_id'],
                        scores=spec["score_json"],
                        overrides_id=spec["overrides_id"],
                        run_id=record['run_id'],
                    )
                    spec["score_row"] = score_row
                    new_score_rows.append(score_row)

                if new_score_rows:
                    session.add_all(new_score_rows)
                    session.flush()
                    for spec in batch_specs:
                        if spec.get("score_row") is not None:
                            spec["score_id"] = spec["score_row"].id

                existing_finals = self._prefetch_final_collections(session, records)
                new_final_rows = []
                updated_final_rows = 0
                for spec in batch_specs:
                    record = spec["record"]
                    existing_final = None
                    if record.get("scix_id"):
                        existing_final = existing_finals["scix_id"].get(record["scix_id"])
                    if existing_final is None and record.get("bibcode"):
                        existing_final = existing_finals["bibcode"].get(record["bibcode"])

                    if existing_final is not None:
                        existing_final.collection = spec["final_collections"]
                        existing_final.score_id = spec["score_id"]
                        updated_final_rows += 1
                    else:
                        new_final_rows.append(
                            models.FinalCollectionTable(
                                bibcode=record['bibcode'],
                                scix_id=record['scix_id'],
                                collection=spec["final_collections"],
                                score_id=spec["score_id"],
                            )
                        )

                if new_final_rows:
                    session.add_all(new_final_rows)

                session.commit()

            perf_metrics.emit_event(
                stage="index_db",
                run_id=run_ids[0] if len(run_ids) == 1 else None,
                context_id=context_id,
                record_id=None,
                duration_ms=(time.perf_counter() - stage_start) * 1000.0,
                status="ok",
                extra={
                    "record_count": len(records),
                    "batch_mode": True,
                    "batch_size": len(records),
                    "new_score_rows": len(new_score_rows),
                    "reused_score_rows": reused_score_rows,
                    "new_final_rows": len(new_final_rows),
                    "updated_final_rows": updated_final_rows,
                },
                config=config,
            )
            return [(spec["record"], "record_indexed") for spec in batch_specs]

    def index_run(self, perf_metrics_context_id=None):
        """
        Create and persist a new RunTable record in the database.

        Returns:
            str: The ID of the new run
        """
        start = time.perf_counter()
        with self.session_scope() as session:
            run_row = models.RunTable()

            session.add(run_row)
            session.commit()
            logger.info(f'Indexing run {run_row.id}')
            perf_metrics.emit_event(
                stage="app_timing",
                run_id=run_row.id,
                context_id=perf_metrics_context_id,
                record_id=None,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                status="ok",
                extra={"name": "index_run"},
                config=config,
            )
            return run_row.id


    def index_record(self, record):
        """
        Processes and indexes a classification record into the database.
        Supports both automated classification and manual overrides.

        Parameters:
            record (dict): The classification result record to store

        Returns:
            tuple: (record, status), where status is a string message indicating 
                   success, failure, or type of update
        """
        with perf_metrics.timed_profile(
            category="app_timing",
            name="index_record",
            run_id=record.get("run_id"),
            context_id=self._record_context_id(record),
            record_id=record.get("scix_id") or record.get("bibcode"),
            config=config,
        ):
            logger.debug('Indexing record')
            logger.debug(f'Record: {record}')

            if 'operation_step' not in record:
                record['operation_step'] = 'classify'
            if 'bibcode' not in record:
                record['bibcode'] = None
            if 'scix_id' not in record:
                record['scix_id'] = None

            stage_start = time.perf_counter()
            status = "ok"
            with self.session_scope() as session:
                # Initial indexing of automatic classification results
                if record['operation_step'] == 'classify' or record['operation_step'] == 'classify_verify':
                    logger.debug('Indexing new record')
                    context_id = self._record_context_id(record)
                    model_id = self._get_or_create_model_id(session, run_id=record.get('run_id'), context_id=context_id)
                    self._ensure_run_model(session, record['run_id'], model_id, context_id=context_id)

                    check_overrides_query = session.query(models.OverrideTable).filter(or_(and_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.scix_id != None), and_(models.OverrideTable.bibcode == record['bibcode'], models.OverrideTable.bibcode != None))).order_by(models.OverrideTable.created.desc()).first()

                    logger.debug(f'Check Overrides Query: {check_overrides_query}')
                    if check_overrides_query is not None:
                        final_collections = check_overrides_query.override
                        overrides_id = check_overrides_query.id
                    else:
                        final_collections = record['collections']
                        overrides_id = None

                    scores_dict = {cat: score for cat, score in zip(config['ALLOWED_CATEGORIES'], record['scores'])}
                    scores = {'scores': scores_dict,
                              'earth_science_adjustment': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                              'collections': record['collections']}

                    score_row = models.ScoreTable(bibcode=record['bibcode'],
                                                scix_id=record['scix_id'],
                                                scores=json.dumps(scores),
                                                overrides_id=overrides_id,
                                                run_id=record['run_id']
                                                )

                    check_scores_query = session.query(models.ScoreTable).filter(and_(or_(and_(models.ScoreTable.scix_id == record['scix_id'], models.ScoreTable.scix_id != None), and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.bibcode != None)), models.ScoreTable.scores == json.dumps(scores), models.ScoreTable.overrides_id == overrides_id, models.ScoreTable.run_id == record['run_id'])).order_by(models.ScoreTable.created.desc()).first()

                    logger.debug(f'Check Scores Query: {check_scores_query}')
                    if check_scores_query is None:
                        session.add(score_row)
                        session.flush()
                        score_id = score_row.id
                    else:
                        score_id = check_scores_query.id

                    final_collections_row = models.FinalCollectionTable(bibcode=record['bibcode'],
                                                                        scix_id=record['scix_id'],
                                                                        collection=final_collections,
                                                                        score_id=score_id
                                                                        )

                    check_final_collection_query = session.query(models.FinalCollectionTable).filter(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))).order_by(models.FinalCollectionTable.created.desc()).first()

                    logger.debug(f'Check Final Collections Query: {check_final_collection_query}')
                    if check_final_collection_query is None:
                        session.add(final_collections_row)
                    if check_final_collection_query is not None:
                        check_final_collection_query.collection = final_collections
                        check_final_collection_query.score_id = score_id
                    session.commit()

                    result = (record, "record_indexed")
                    perf_metrics.emit_event(
                        stage="index_db",
                        run_id=record.get("run_id"),
                        context_id=self._record_context_id(record),
                        record_id=record.get("scix_id") or record.get("bibcode"),
                        duration_ms=(time.perf_counter() - stage_start) * 1000.0,
                        status=status,
                        extra={"operation_step": record.get("operation_step")},
                        config=config,
                    )
                    return result

                logger.debug('Updating validated record')

                check_overrides_query = session.query(models.OverrideTable).filter(and_(or_(and_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.scix_id != None), and_(models.OverrideTable.bibcode == record['bibcode'], models.OverrideTable.bibcode != None))), models.OverrideTable.override == record['override']).order_by(models.OverrideTable.created.desc()).first()

                allowed = utils.check_is_allowed_category(record['override'])
                empty = utils.check_if_list_single_empty_string(record['override'])

                logger.debug(f'Check overrides query: {check_overrides_query}')
                if check_overrides_query is None and allowed is True:
                    override_row = models.OverrideTable(bibcode=record['bibcode'],
                                                        scix_id=record['scix_id'],
                                                        override=record['override'])
                    session.add(override_row)
                    session.flush()
                    overrides_id = override_row.id

                    update_scores_query = session.query(models.ScoreTable).filter(or_(and_(models.ScoreTable.scix_id == record['scix_id'], models.ScoreTable.scix_id != None), and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.bibcode != None))).order_by(models.ScoreTable.created.desc()).all()
                    logger.debug(f'update_scores_query: {update_scores_query}')
                    for element in update_scores_query:
                        element.overrides_id = overrides_id

                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))).order_by(models.FinalCollectionTable.created.desc()).first()

                    logger.debug(f'update_final_collection_query: {update_final_collection_query}')
                    logger.debug(f'update_final_collection_query with override: {record["override"]}')
                    update_final_collection_query.collection = record['override']
                    update_final_collection_query.validated = True
                    session.commit()
                    success = 'record_validated'

                elif check_overrides_query is None and empty is True:
                    logger.debug(f'Record to update as validated: {record}')
                    update_final_collection_query_False = session.query(models.FinalCollectionTable).filter(and_(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))), models.FinalCollectionTable.validated == False).order_by(models.FinalCollectionTable.created.desc()).first()
                    update_final_collection_query_True = session.query(models.FinalCollectionTable).filter(and_(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))), models.FinalCollectionTable.validated == True).order_by(models.FinalCollectionTable.created.desc()).first()

                    if update_final_collection_query_False is not None:
                        update_final_collection_query_False.validated = True
                        session.commit()
                        success = 'record_validated'
                    if update_final_collection_query_True is not None:
                        success = 'record_previously_validated'

                elif check_overrides_query is not None:
                    logger.info(f'Record {record} already validated')
                    success = 'record_previously_validated'

                else:
                    logger.info(f'Record {record} had other difficulties (re-)validating')
                    success = 'other_failure'

                result = (record, "record_validated")
                perf_metrics.emit_event(
                    stage="index_db",
                    run_id=record.get("run_id"),
                    context_id=self._record_context_id(record),
                    record_id=record.get("scix_id") or record.get("bibcode"),
                    duration_ms=(time.perf_counter() - stage_start) * 1000.0,
                    status=status,
                    extra={"operation_step": record.get("operation_step")},
                    config=config,
                )
                return result

    def query_final_collection_table(self, run_id=None, bibcode=None, scix_id=None, perf_metrics_context_id=None):
        """
        Queries the FinalCollectionTable based on one of run_id, bibcode, or scix_id.

        Parameters:
            run_id (int, optional): ID of the run to query
            bibcode (str, optional): Bibliographic code to query
            scix_id (str, optional): SciX ID to query

        Returns:
            list of dict: List of matched collection records
        """
        with perf_metrics.timed_profile(
            category="app_timing",
            name="query_final_collection_table",
            run_id=run_id,
            context_id=perf_metrics_context_id,
            record_id=scix_id or bibcode,
            config=config,
        ):
            with self.session_scope() as session:
                record_list = []
                if run_id is not None:
                    run_rows = (
                        session.query(models.FinalCollectionTable, models.ScoreTable)
                        .join(models.ScoreTable, models.FinalCollectionTable.score_id == models.ScoreTable.id)
                        .filter(models.ScoreTable.run_id == run_id)
                        .order_by(models.FinalCollectionTable.created.desc())
                        .all()
                    )
                    seen_keys = set()
                    for final_collection_row, score_row in run_rows:
                        out_bibcode = final_collection_row.bibcode or score_row.bibcode
                        out_scix_id = final_collection_row.scix_id or score_row.scix_id
                        dedupe_key = self._result_record_key(bibcode=out_bibcode, scix_id=out_scix_id)
                        if dedupe_key in seen_keys:
                            continue
                        seen_keys.add(dedupe_key)

                        logger.debug(f'Record bibcode: {out_bibcode}, scix_id: {out_scix_id}')
                        out_record = {
                            'bibcode': out_bibcode,
                            'scix_id': out_scix_id,
                            'collections': final_collection_row.collection,
                        }
                        record_list.append(out_record)

                if bibcode is not None:

                    final_collection_query = session.query(models.FinalCollectionTable).filter(models.FinalCollectionTable.bibcode == bibcode).first()

                    out_record = {'bibcode' : bibcode,
                                  'scix_id' : final_collection_query.scix_id,
                                  'collections' : final_collection_query.collection}
                    record_list.append(out_record)

                if scix_id is not None:

                    final_collection_query = session.query(models.FinalCollectionTable).filter(models.FinalCollectionTable.scix_id == scix_id).first()

                    out_record = {'bibcode' : final_collection_query.bibcode,
                                  'scix_id' : scix_id,
                                  'collections' : final_collection_query.collection}
                    record_list.append(out_record)

                logger.debug(f'Record list: {record_list}')
                return record_list

    def update_validated_records(self, run_id):
        """
        Updates the FinalCollectionTable to mark unvalidated records as validated 
        for a specific run ID.

        Parameters:
            run_id (int): ID of the classification run

        Returns:
            tuple: (record_list, success_list) where:
                   - record_list is a list of updated records
                   - success_list is a list of update results
        """
        with perf_metrics.timed_profile(
            category="app_timing",
            name="update_validated_records",
            run_id=run_id,
            context_id=None,
            record_id=None,
            config=config,
        ):
            logger.info(f'Updating run_id: {run_id}')

            record_list = self.query_final_collection_table(run_id=run_id)

            success_list = []
            with self.session_scope() as session:

                for record in record_list:


                    logger.debug(f'Record to update as validated: {record}')
                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))), models.FinalCollectionTable.validated == False).order_by(models.FinalCollectionTable.created.desc()).first()

                    if update_final_collection_query is not None:
                        update_final_collection_query.validated = True
                        session.commit()
                        success_list.append("success")

            logger.debug(f'List of validated records: {record_list}')
            return record_list, success_list
 
