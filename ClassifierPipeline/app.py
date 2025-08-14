import os
import json
import pickle
import zlib
import csv

import ClassifierPipeline.models as models
import ClassifierPipeline.utilities as utils
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
        

    def handle_message_payload(self, message=None, payload=None):
        """
        Handles the message payload
        """

        parsed_message = json.loads(message)
        request_list = parsed_message['classifyRequests']


    def index_run(self):
        """
        Indexes a run into a database

        :param: none 
        :return: str Run table row ID
        """
        with self.session_scope() as session:

            run_row = models.RunTable()

            session.add(run_row)
            session.commit()
            logger.info(f'Indexing run {run_row.id}')

            return run_row.id


    def index_record(self, record):
        """
        Saves a record into a database

        :param: record- dictionar
        :return: boolean - whether record successfuly added
                to the database
        """
        logger.debug('Indexing record')
        logger.debug(f'Record: {record}')

        if 'operation_step' not in record:
            record['operation_step'] = 'classify'
        if 'bibcode' not in record:
            record['bibcode'] = None
        if 'scix_id' not in record:
            record['scix_id'] = None

        with self.session_scope() as session:

            # Initial indexing of automatic classification results
            if record['operation_step'] == 'classify' or record['operation_step'] == 'classify_verify':
                logger.debug('Indexing new record')

                # Model Table
                model = {'model' : config['CLASSIFICATION_PRETRAINED_MODEL'],
                         'revision' : config['CLASSIFICATION_PRETRAINED_MODEL_REVISION'],
                         'tokenizer' : config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']
                         }
                postprocessing = {'ADDITIONAL_EARTH_SCIENCE_PROCESSING' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                                  'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD' : config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD'],
                                  'CLASSIFICATION_THRESHOLDS' : config['CLASSIFICATION_THRESHOLDS']
                                  }
                model_row = models.ModelTable(model=json.dumps(model),
                                              postprocessing=json.dumps(postprocessing)
                                              )

                logger.debug('Checking model query')
                check_model_query = session.query(models.ModelTable).filter(and_(models.ModelTable.model == json.dumps(model), models.ModelTable.postprocessing == json.dumps(postprocessing))).order_by(models.ModelTable.created.desc()).first()

                logger.debug(f'Check Model Query: {check_model_query}')
                if check_model_query is None:
                    session.add(model_row)
                    session.commit()
                    model_id = model_row.id
                else:
                    model_id = check_model_query.id

                # Run Table
                check_run_query = session.query(models.RunTable).filter(models.RunTable.id == record['run_id']).order_by(models.RunTable.created.desc()).first()

                logger.debug(f'Check Run Query: {check_run_query}')
                if check_run_query is not None:

                    if check_run_query.model_id != model_id:
                        check_run_query.model_id = model_id 
                        session.commit()

                # Override Table
                check_overrides_query = session.query(models.OverrideTable).filter(or_(and_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.scix_id != None), and_(models.OverrideTable.bibcode == record['bibcode'], models.OverrideTable.bibcode != None))).order_by(models.OverrideTable.created.desc()).first()


                logger.debug(f'Check Overrides Query: {check_overrides_query}')
                if check_overrides_query is not None:
                    final_collections = check_overrides_query.override
                    overrides_id = check_overrides_query.id
                else:
                    final_collections = record['collections']
                    overrides_id = None

                # Scores Table
                scores_dict = {cat:score for cat, score in zip(config['ALLOWED_CATEGORIES'], record['scores'])}
                scores = {'scores': scores_dict,
                          'earth_science_adjustment': config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
                          'collections': record['collections']}

                score_row = models.ScoreTable(bibcode=record['bibcode'], 
                                            scix_id=record['scix_id'],
                                            scores=json.dumps(scores),
                                            overrides_id = overrides_id,
                                            run_id = record['run_id']
                                            ) 

                # Check if EXACT record is already in the database
                check_scores_query = session.query(models.ScoreTable).filter(and_(or_(and_(models.ScoreTable.scix_id == record['scix_id'], models.ScoreTable.scix_id != None), and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.bibcode != None)), models.ScoreTable.scores == json.dumps(scores), models.ScoreTable.overrides_id == overrides_id, models.ScoreTable.run_id == record['run_id'])).order_by(models.ScoreTable.created.desc()).first()


                logger.debug(f'Check Scores Query: {check_scores_query}')
                if check_scores_query is None:
                    session.add(score_row)
                    session.commit()
                    score_id = score_row.id

                final_collections_row = models.FinalCollectionTable(bibcode = record['bibcode'], 
                                                                        collection = final_collections,
                                                                        score_id = score_row.id
                                                                        )


                # Final Collection Table
                check_final_collection_query = session.query(models.FinalCollectionTable).filter(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))).order_by(models.FinalCollectionTable.created.desc()).first()

                logger.debug(f'Check Final Collections Query: {check_final_collection_query}')
                if check_final_collection_query is None:
                    session.add(final_collections_row)
                    session.commit()
                if check_final_collection_query is not None:
                    check_final_collection_query.final_collection = final_collections
                    session.commit()

                return record, "record_indexed"
                
            else:
                logger.debug('Updating validated record')

                # Check for existing override
                check_overrides_query = session.query(models.OverrideTable).filter(and_(or_(and_(models.OverrideTable.scix_id == record['scix_id'], models.OverrideTable.scix_id != None), and_(models.OverrideTable.bibcode == record['bibcode'], models.OverrideTable.bibcode != None))), models.OverrideTable.override == record['override']).order_by(models.OverrideTable.created.desc()).first()

                allowed = utils.check_is_allowed_category(record['override'])
                empty = utils.check_if_list_single_empty_string(record['override'])

                logger.debug(f'Check overrides query: {check_overrides_query}')
                if check_overrides_query is None and allowed is True:
                    override_row = models.OverrideTable(bibcode=record['bibcode'],
                                                        scix_id=record['scix_id'],
                                                        override=record['override'])
                    session.add(override_row)
                    session.commit()
                    overrides_id = override_row.id

                    update_scores_query = session.query(models.ScoreTable).filter(or_(and_(models.ScoreTable.scix_id == record['scix_id'], models.ScoreTable.scix_id != None), and_(models.ScoreTable.bibcode == record['bibcode'], models.ScoreTable.bibcode != None))).order_by(models.ScoreTable.created.desc()).all()
                    logger.debug(f'update_scores_query: {update_scores_query}')
                    for element in update_scores_query:
                        element.overrides_id = overrides_id
                        session.commit()

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


                if empty is True:
                    logger.debug(f'Record to update as validated: {record}')
                    update_final_collection_query = session.query(models.FinalCollectionTable).filter(and_(or_(and_(models.FinalCollectionTable.scix_id == record['scix_id'], models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record['bibcode'], models.FinalCollectionTable.bibcode != None))), models.FinalCollectionTable.validated == False).order_by(models.FinalCollectionTable.created.desc()).first()

                    if update_final_collection_query is not None:
                        update_final_collection_query.validated = True
                        session.commit()

                return record, "record_validated"

    def query_final_collection_table(self, run_id=None, bibcode=None, scix_id=None):
        """
        Query final collections table based on run_id, bibcode, or scix_id
        
        param: run_id - integer
               bibcode - string
               scix)id - string

        return: list of record dictionaries containing bibcode,
                scix_id, run_id and collections
        """

        with self.session_scope() as session:


            record_list = []
            if run_id is not None:

                run_query = session.query(models.RunTable).filter(models.RunTable.id == run_id).first()
                run_id_query = session.query(models.ScoreTable).filter(models.ScoreTable.run_id == run_query.id).all()

                for record in run_id_query:

                    logger.debug(f'Record bibcode: {record.bibcode}, scix_id: {record.scix_id}')

                    final_collection_query = session.query(models.FinalCollectionTable).filter(or_(and_(models.FinalCollectionTable.scix_id == record.scix_id, models.FinalCollectionTable.scix_id != None), and_(models.FinalCollectionTable.bibcode == record.bibcode, models.FinalCollectionTable.bibcode != None))).order_by(models.FinalCollectionTable.created.desc()).first()

                    out_record = {'bibcode' : record.bibcode,
                                  'scix_id' : record.scix_id,
                                  'collections' : final_collection_query.collection}
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
        Updates validated records in the database

        :param: run_id- integer
        :return: boolean - whether update successful
        """
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
 



