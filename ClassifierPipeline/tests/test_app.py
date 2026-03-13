import importlib
import types


class FakeQuery:
    def __init__(self, first_result=None, all_result=None):
        self._first_result = first_result
        self._all_result = all_result if all_result is not None else []

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
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

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added) + 1
        self.added.append(obj)

    def commit(self):
        self.commit_count += 1

    def query(self, model):
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
    assert session.commit_count >= 3
    assert len(session.added) >= 3


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
    existing_final = types.SimpleNamespace(final_collection=None)
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
    assert existing_final.final_collection == ["Astronomy"]


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


def test_query_final_collection_table_by_run_id(monkeypatch, base_fake_config, dummy_logger):
    module = _import_app_module(monkeypatch, base_fake_config, dummy_logger)
    queries = [
        FakeQuery(first_result=types.SimpleNamespace(id=7)),
        FakeQuery(all_result=[types.SimpleNamespace(bibcode="B", scix_id="S")]),
        FakeQuery(first_result=types.SimpleNamespace(collection=["Astronomy"])),
    ]
    session = FakeSession(queries)
    app = _new_app(module, session)
    assert app.query_final_collection_table(run_id=7) == [{"bibcode": "B", "scix_id": "S", "collections": ["Astronomy"]}]


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
