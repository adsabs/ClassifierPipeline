import importlib
import json
import types


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self._first_result = first_result
        self._all_result = all_result if all_result is not None else []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def first(self):
        return self._first_result

    def all(self):
        return self._all_result


class FakeSession:
    def __init__(self, queries=None):
        self.queries = list(queries or [])
        self.added = []
        self.commit_count = 0
        self.flush_count = 0
        self.query_models = []

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added) + 1
        self.added.append(obj)

    def add_all(self, objects):
        for obj in objects:
            self.add(obj)

    def commit(self):
        self.commit_count += 1

    def flush(self):
        self.flush_count += 1
        for index, obj in enumerate(self.added, start=1):
            if getattr(obj, "id", None) is None:
                obj.id = index

    def query(self, *models):
        self.query_models.append(models if len(models) > 1 else models[0])
        return self.queries.pop(0)


class SessionScope:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


def _import_app_module(monkeypatch, base_fake_config, dummy_logger):
    monkeypatch.setattr("adsputils.load_config", lambda proj_home=None: dict(base_fake_config))
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)
    import sys
    sys.modules.pop("ClassifierPipeline.app", None)
    module = importlib.import_module("ClassifierPipeline.app")
    module.config = dict(base_fake_config)
    module.logger = dummy_logger
    module.perf_metrics.emit_event = lambda **kwargs: None
    return module


def _new_app(module, session):
    app = module.SciXClassifierCelery.__new__(module.SciXClassifierCelery)
    app.session_scope = lambda: SessionScope(session)
    return app


def test_index_run_adds_run_row_and_returns_id(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    session = FakeSession()
    app = _new_app(module, session)
    run_id = app.index_run()
    assert run_id == 1
    assert session.commit_count == 1
    assert len(session.added) == 1


def test_index_run_emits_app_timing(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    session = FakeSession()
    app = _new_app(module, session)
    events = []
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    app.index_run()
    assert any(event["stage"] == "app_timing" and event["extra"]["name"] == "index_run" for event in events)


def test_index_record_classify_path_creates_missing_rows(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(first_result=None),  # model
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),  # run
        FakeQuery(first_result=None),  # override
        FakeQuery(first_result=None),  # score
        FakeQuery(first_result=None),  # final collection
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {
        "run_id": 7,
        "bibcode": "B",
        "scix_id": None,
        "collections": ["Astronomy"],
        "scores": [0.9] * 8,
        "operation_step": "classify",
    }
    out_record, status = app.index_record(record)
    assert status == "record_indexed"
    assert out_record is record
    assert session.commit_count == 1
    assert session.flush_count == 2
    assert len(session.added) >= 3


def test_index_record_emits_app_timing(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    events = []
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    record = {
        "run_id": 7,
        "bibcode": "B",
        "scix_id": None,
        "collections": ["Astronomy"],
        "scores": [0.9] * 8,
        "operation_step": "classify",
    }
    app.index_record(record)
    assert any(event["stage"] == "app_timing" and event["extra"]["name"] == "index_record" for event in events)


def test_index_record_classify_path_reuses_existing_model(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    existing_model = types.SimpleNamespace(id=4)
    run_row = types.SimpleNamespace(id=7, model_id=4)
    queries = [
        FakeQuery(first_result=existing_model),
        FakeQuery(first_result=run_row),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    _, status = app.index_record(record)
    assert status == "record_indexed"
    assert session.added and all(not isinstance(item, module.models.ModelTable) for item in session.added)


def test_index_record_classify_path_uses_existing_override(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    override = types.SimpleNamespace(id=5, override=["Other"])
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=override),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    _, status = app.index_record(record)
    assert status == "record_indexed"
    final_collection_row = next(item for item in session.added if isinstance(item, module.models.FinalCollectionTable))
    assert final_collection_row.collection == ["Other"]


def test_index_record_classify_path_skips_duplicate_score_insert(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    existing_score = types.SimpleNamespace(id=9)
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=None),
        FakeQuery(first_result=existing_score),
        FakeQuery(first_result=None),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    _, status = app.index_record(record)
    assert status == "record_indexed"
    assert all(not isinstance(item, module.models.ScoreTable) for item in session.added)


def test_index_record_classify_path_updates_existing_final_collection(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    existing_final = types.SimpleNamespace(collection=None, score_id=None)
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
        FakeQuery(first_result=existing_final),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    _, status = app.index_record(record)
    assert status == "record_indexed"
    assert existing_final.collection == ["Astronomy"]


def test_index_record_classify_path_uses_existing_score_id_for_duplicate(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    existing_score = types.SimpleNamespace(id=9)
    existing_final = types.SimpleNamespace(collection=None, score_id=None)
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=None),
        FakeQuery(first_result=existing_score),
        FakeQuery(first_result=existing_final),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    _, status = app.index_record(record)
    assert status == "record_indexed"
    assert existing_final.score_id == 9
    assert session.commit_count == 1


def test_index_record_classify_path_updates_collection_field_not_final_collection(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    existing_final = types.SimpleNamespace(collection=None, score_id=None, final_collection=None)
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(first_result=None),
        FakeQuery(first_result=None),
        FakeQuery(first_result=existing_final),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"run_id": 7, "bibcode": "B", "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}
    app.index_record(record)
    assert existing_final.collection == ["Astronomy"]
    assert existing_final.final_collection is None


def test_index_record_validation_path_inserts_override_and_updates_records(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    score_row = types.SimpleNamespace(overrides_id=None)
    final_row = types.SimpleNamespace(collection=None, validated=False)
    queries = [
        FakeQuery(first_result=None),  # existing override
        FakeQuery(all_result=[score_row]),  # scores
        FakeQuery(first_result=final_row),  # final collection
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"bibcode": "B", "scix_id": None, "override": ["Astronomy"], "operation_step": "validate"}
    _, status = app.index_record(record)
    assert status == "record_validated"
    assert score_row.overrides_id == 1
    assert final_row.collection == ["Astronomy"]
    assert final_row.validated is True
    assert session.commit_count == 1


def test_index_record_validation_path_marks_unvalidated_without_override(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    false_row = types.SimpleNamespace(validated=False)
    true_row = None
    queries = [
        FakeQuery(first_result=None),
        FakeQuery(first_result=false_row),
        FakeQuery(first_result=true_row),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"bibcode": "B", "scix_id": None, "override": [""], "operation_step": "validate"}
    _, status = app.index_record(record)
    assert status == "record_validated"
    assert false_row.validated is True
    assert session.commit_count == 1


def test_index_record_validation_path_returns_previously_validated(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(first_result=types.SimpleNamespace(id=1)),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    record = {"bibcode": "B", "scix_id": None, "override": ["Astronomy"], "operation_step": "validate"}
    _, status = app.index_record(record)
    assert status == "record_validated"
    assert session.commit_count == 0


def test_get_or_create_model_id_uses_cache_on_warm_path(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    session = FakeSession([FakeQuery(first_result=None)])
    app = _new_app(module, session)
    first_id = app._get_or_create_model_id(session)
    second_id = app._get_or_create_model_id(session)
    assert first_id == second_id == 1
    assert session.query_models == [module.models.ModelTable]


def test_ensure_run_model_skips_requery_for_bound_run(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    run_row = types.SimpleNamespace(id=7, model_id=None)
    session = FakeSession([FakeQuery(first_result=run_row)])
    app = _new_app(module, session)
    app._ensure_run_model(session, 7, 3)
    app._ensure_run_model(session, 7, 3)
    assert run_row.model_id == 3
    assert session.query_models == [module.models.RunTable]


def test_index_records_batch_inserts_scores_and_final_rows_in_one_commit(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    events = []
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    session = FakeSession([
        FakeQuery(first_result=None),  # model
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),  # run
        FakeQuery(all_result=[]),  # overrides
        FakeQuery(all_result=[]),  # scores
        FakeQuery(all_result=[]),  # finals
    ])
    app = _new_app(module, session)
    records = [
        {"run_id": 7, "bibcode": "B1", "scix_id": None, "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"},
        {"run_id": 7, "bibcode": "B2", "scix_id": None, "collections": ["Astronomy"], "scores": [0.8] * 8, "operation_step": "classify"},
    ]
    results = app.index_records_batch(records)
    assert [status for _, status in results] == ["record_indexed", "record_indexed"]
    assert session.commit_count == 1
    assert session.flush_count == 2
    index_db_events = [event for event in events if event["stage"] == "index_db"]
    assert index_db_events[0]["extra"]["record_count"] == 2


def test_index_records_batch_reuses_existing_score_rows(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    score_payload = {
        'scores': {cat: 0.9 for cat in base_fake_config["ALLOWED_CATEGORIES"]},
        'earth_science_adjustment': base_fake_config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'],
        'collections': ["Astronomy"],
    }
    existing_score = types.SimpleNamespace(id=9, bibcode="B1", scix_id=None, scores=json.dumps(score_payload, sort_keys=True), overrides_id=None, run_id=7)
    existing_final = types.SimpleNamespace(collection=None, score_id=None)
    session = FakeSession([
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(all_result=[]),
        FakeQuery(all_result=[existing_score]),
        FakeQuery(all_result=[types.SimpleNamespace(bibcode="B1", scix_id=None, collection=None, score_id=None, created=1)]),
    ])
    app = _new_app(module, session)
    records = [{"run_id": 7, "bibcode": "B1", "scix_id": None, "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"}]
    results = app.index_records_batch(records)
    assert results[0][1] == "record_indexed"
    assert all(not isinstance(item, module.models.ScoreTable) for item in session.added if not isinstance(item, module.models.ModelTable))


def test_index_records_batch_preserves_input_order(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    session = FakeSession([
        FakeQuery(first_result=None),
        FakeQuery(first_result=types.SimpleNamespace(id=7, model_id=None)),
        FakeQuery(all_result=[]),
        FakeQuery(all_result=[]),
        FakeQuery(all_result=[]),
    ])
    app = _new_app(module, session)
    records = [
        {"run_id": 7, "bibcode": "B1", "scix_id": None, "collections": ["Astronomy"], "scores": [0.9] * 8, "operation_step": "classify"},
        {"run_id": 7, "bibcode": "B2", "scix_id": None, "collections": ["Astronomy"], "scores": [0.8] * 8, "operation_step": "classify"},
    ]
    results = app.index_records_batch(records)
    assert [record["bibcode"] for record, _ in results] == ["B1", "B2"]


def test_query_final_collection_table_by_run_id(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(
            all_result=[
                (
                    types.SimpleNamespace(bibcode="B", scix_id="S", collection=["Astronomy"], created=2),
                    types.SimpleNamespace(bibcode="B", scix_id="S"),
                )
            ]
        ),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]


def test_query_final_collection_table_by_run_id_dedupes_duplicate_final_rows(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(
            all_result=[
                (
                    types.SimpleNamespace(bibcode="B", scix_id=None, collection=["Astronomy"], created=5),
                    types.SimpleNamespace(bibcode="B", scix_id=None),
                ),
                (
                    types.SimpleNamespace(bibcode="B", scix_id=None, collection=["Physics"], created=1),
                    types.SimpleNamespace(bibcode="B", scix_id=None),
                ),
            ]
        ),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [{"bibcode": "B", "scix_id": None, "collections": ["Astronomy"]}]


def test_query_final_collection_table_by_run_id_prefers_scix_id_for_dedupe(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(
            all_result=[
                (
                    types.SimpleNamespace(bibcode="B1", scix_id="S1", collection=["Astronomy"], created=5),
                    types.SimpleNamespace(bibcode="B1", scix_id="S1"),
                ),
                (
                    types.SimpleNamespace(bibcode="B2", scix_id="S1", collection=["Physics"], created=4),
                    types.SimpleNamespace(bibcode="B2", scix_id="S1"),
                ),
            ]
        ),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [{"bibcode": "B1", "scix_id": "S1", "collections": ["Astronomy"]}]


def test_query_final_collection_table_by_run_id_preserves_distinct_records(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(
            all_result=[
                (
                    types.SimpleNamespace(bibcode="B1", scix_id=None, collection=["Astronomy"], created=5),
                    types.SimpleNamespace(bibcode="B1", scix_id=None),
                ),
                (
                    types.SimpleNamespace(bibcode="B2", scix_id=None, collection=["Physics"], created=4),
                    types.SimpleNamespace(bibcode="B2", scix_id=None),
                ),
            ]
        ),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [
        {"bibcode": "B1", "scix_id": None, "collections": ["Astronomy"]},
        {"bibcode": "B2", "scix_id": None, "collections": ["Physics"]},
    ]


def test_query_final_collection_table_by_run_id_uses_score_linkage_not_latest_identifier_lookup(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(
            all_result=[
                (
                    types.SimpleNamespace(bibcode="B", scix_id=None, collection=["Astronomy"], created=2, score_id=7),
                    types.SimpleNamespace(id=7, bibcode="B", scix_id=None, run_id=7),
                )
            ]
        ),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [{"bibcode": "B", "scix_id": None, "collections": ["Astronomy"]}]


def test_query_final_collection_table_by_bibcode(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [FakeQuery(first_result=types.SimpleNamespace(scix_id="S", collection=["Astronomy"]))]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(bibcode="B") == [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]


def test_query_final_collection_table_by_scix_id(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [FakeQuery(first_result=types.SimpleNamespace(bibcode="B", collection=["Astronomy"]))]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(scix_id="S") == [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]


def test_update_validated_records_marks_records(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    pending = types.SimpleNamespace(validated=False)
    queries = [FakeQuery(first_result=pending)]
    session = FakeSession(queries)
    app = _new_app(module, session)
    app.query_final_collection_table = lambda run_id=None: [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]
    records, successes = app.update_validated_records(7)
    assert records == [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]
    assert successes == ["success"]
    assert pending.validated is True
