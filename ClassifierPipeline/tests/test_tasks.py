import importlib
import sys
import types


class IdentityDecorator:
    def __call__(self, *args, **kwargs):
        def wrapper(func):
            func.delay = func
            return func
        return wrapper


def _import_tasks_module(monkeypatch, base_fake_config, dummy_logger):
    fake_app_instance = types.SimpleNamespace(
        exchange="exchange",
        conf=types.SimpleNamespace(CELERY_QUEUES=()),
        index_run=lambda: "RUNID",
        index_record=lambda record: (record, "record_indexed"),
        add_record_to_output_file=lambda record: None,
        query_final_collection_table=lambda **kwargs: [],
        update_validated_records=lambda run_id: ([], []),
        forward_message=lambda message: None,
        task=IdentityDecorator(),
    )

    class FakeCeleryFactory:
        def __call__(self, *args, **kwargs):
            return fake_app_instance

    fake_app_module = types.ModuleType("ClassifierPipeline.app")
    fake_app_module.SciXClassifierCelery = FakeCeleryFactory()

    fake_utils = types.ModuleType("ClassifierPipeline.utilities")
    fake_utils.classifyRequestRecordList_to_list = lambda message: message if isinstance(message, list) else [message]
    fake_utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    fake_utils.prepare_output_file = lambda path: None
    fake_utils.classify_record_from_scores = lambda record: record
    fake_utils.return_fake_data = lambda record: {**record, "categories": ["fake"], "scores": [1.0]}
    fake_utils.add_record_to_output_file = lambda record: None
    fake_utils.dict_to_ClassifyResponseRecord = lambda message: {"wrapped": message}

    class FakeClassifier:
        def batch_score_SciX_categories(self, texts):
            return [["Astronomy"]], [[0.99]]

    fake_classifier_module = types.ModuleType("ClassifierPipeline.classifier")
    fake_classifier_module.Classifier = FakeClassifier

    monkeypatch.setattr("adsputils.load_config", lambda *args, **kwargs: dict(base_fake_config))
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setattr("kombu.Queue", lambda name, exchange, routing_key=None: (name, exchange, routing_key))

    import ClassifierPipeline

    monkeypatch.setitem(sys.modules, "ClassifierPipeline.app", fake_app_module)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.utilities", fake_utils)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.classifier", fake_classifier_module)
    monkeypatch.setattr(ClassifierPipeline, "app", fake_app_module, raising=False)
    monkeypatch.setattr(ClassifierPipeline, "utilities", fake_utils, raising=False)
    monkeypatch.setattr(ClassifierPipeline, "classifier", fake_classifier_module, raising=False)
    sys.modules.pop("ClassifierPipeline.tasks", None)
    module = importlib.import_module("ClassifierPipeline.tasks")
    module.config = dict(base_fake_config)
    module.logger = dummy_logger
    return module, fake_app_instance


def test_record_identifier_prefers_scix_id(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    assert module._record_identifier({"scix_id": "S", "bibcode": "B"}) == "S"


def test_record_identifier_uses_bibcode_fallback(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    assert module._record_identifier({"bibcode": "B"}) == "B"


def test_task_update_record_creates_run_id_and_output_file(monkeypatch, base_fake_config, dummy_logger):
    module, fake_app = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    calls = {"prepared": [], "forwarded": [], "events": []}
    module.app.index_run = lambda: "RUNID"
    module.utils.prepare_output_file = lambda path: calls["prepared"].append(path)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.task_send_input_record_to_classifier = lambda message: calls["forwarded"].append(message)
    module.perf_metrics.emit_event = lambda **kwargs: calls["events"].append(kwargs)

    result = module.task_update_record({"bibcode": "B", "title": "T", "abstract": "A"})
    assert result == {"run_id": "RUNID", "records_submitted": 1}
    assert calls["prepared"]
    assert calls["forwarded"][0][0]["text"] == "T A"
    assert calls["events"][0]["stage"] == "ingest_enqueue"


def test_task_update_record_reuses_existing_run_id(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.prepare_output_file = lambda path: None
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.perf_metrics.emit_event = lambda **kwargs: None
    module.task_send_input_record_to_classifier = lambda message: None
    result = module.task_update_record({"run_id": "EXISTING", "bibcode": "B", "title": "T", "abstract": "A"})
    assert result["run_id"] == "EXISTING"


def test_task_update_record_uses_delay_when_enabled(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.config["DELAY_MESSAGE"] = True
    module.utils.prepare_output_file = lambda path: None
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.perf_metrics.emit_event = lambda **kwargs: None
    recorded = []

    def capture(message):
        recorded.append(message)

    capture.delay = capture
    module.task_send_input_record_to_classifier = capture
    module.task_update_record({"bibcode": "B", "title": "T", "abstract": "A"})
    assert recorded


def test_task_send_input_record_to_classifier_real_inference(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.config["FAKE_DATA"] = False
    monkeypatch.delenv("PERF_FORCE_FAKE_DATA", raising=False)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.classify_record_from_scores = lambda record: {**record, "classified": True}
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    events = []
    forwarded = []
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    module.task_index_classified_record = lambda message: forwarded.append(message)
    module.classifier = types.SimpleNamespace(batch_score_SciX_categories=lambda texts: ([["Astronomy"]], [[0.9]]))
    message = {"bibcode": "B", "title": "T", "abstract": "A", "run_id": "R"}
    module.task_send_input_record_to_classifier(message)
    assert forwarded[0][0]["categories"] == ["Astronomy"]
    assert forwarded[0][0]["scores"] == [0.9]
    assert events[0]["stage"] == "classify"


def test_task_send_input_record_to_classifier_fake_data_config(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.config["FAKE_DATA"] = True
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.return_fake_data = lambda record: {**record, "categories": ["fake"], "scores": [1.0]}
    module.utils.classify_record_from_scores = lambda record: record
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.perf_metrics.emit_event = lambda **kwargs: None
    forwarded = []
    module.task_index_classified_record = lambda message: forwarded.append(message)
    module.task_send_input_record_to_classifier({"bibcode": "B", "title": "T", "abstract": "A", "run_id": "R"})
    assert forwarded[0][0]["categories"] == ["fake"]


def test_task_send_input_record_to_classifier_fake_data_env_override(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.config["FAKE_DATA"] = False
    monkeypatch.setenv("PERF_FORCE_FAKE_DATA", "true")
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.return_fake_data = lambda record: {**record, "categories": ["fake"], "scores": [1.0]}
    module.utils.classify_record_from_scores = lambda record: record
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.perf_metrics.emit_event = lambda **kwargs: None
    forwarded = []
    module.task_index_classified_record = lambda message: forwarded.append(message)
    module.task_send_input_record_to_classifier({"bibcode": "B", "title": "T", "abstract": "A", "run_id": "R"})
    assert forwarded[0][0]["categories"] == ["fake"]


def test_task_send_input_record_to_classifier_fake_data_record_override(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.config["FAKE_DATA"] = False
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.utils.return_fake_data = lambda record: {**record, "categories": ["fake"], "scores": [1.0]}
    module.utils.classify_record_from_scores = lambda record: record
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.perf_metrics.emit_event = lambda **kwargs: None
    forwarded = []
    module.task_index_classified_record = lambda message: forwarded.append(message)
    module.task_send_input_record_to_classifier({"bibcode": "B", "title": "T", "abstract": "A", "run_id": "R", "fake_data": True})
    assert forwarded[0][0]["categories"] == ["fake"]


def test_task_index_classified_record_classify_verify_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.index_record = lambda record: (record, "record_indexed")
    module.perf_metrics.emit_event = lambda **kwargs: None
    calls = []
    module.utils.add_record_to_output_file = lambda record: calls.append(record)
    module.task_index_classified_record({"bibcode": "B", "operation_step": "classify_verify", "run_id": "R"})
    assert calls and calls[0]["bibcode"] == "B"


def test_task_index_classified_record_classify_path(monkeypatch, base_fake_config, dummy_logger):
    module, fake_app = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.index_record = lambda record: (record, "record_indexed")
    module.perf_metrics.emit_event = lambda **kwargs: None
    app_calls = []
    resend_calls = []
    module.app.add_record_to_output_file = lambda record: app_calls.append(record)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.task_resend_to_master = lambda message: resend_calls.append(message)
    module.task_index_classified_record({"bibcode": "B", "operation_step": "classify", "run_id": "R"})
    assert app_calls and resend_calls


def test_task_index_classified_record_validated_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.index_record = lambda record: (record, "record_validated")
    module.perf_metrics.emit_event = lambda **kwargs: None
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    resend_calls = []
    module.task_resend_to_master = lambda message: resend_calls.append(message)
    module.task_index_classified_record({"bibcode": "B", "operation_step": "validate", "run_id": "R"})
    assert resend_calls


def test_out_message_converts_and_forwards(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    forwarded = []
    module.utils.dict_to_ClassifyResponseRecord = lambda message: {"wrapped": message}
    module.app.forward_message = lambda message: forwarded.append(message)
    module.out_message({"bibcode": "B"})
    assert forwarded == [{"wrapped": {"bibcode": "B"}}]


def test_task_message_to_master_dict_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    calls = []
    events = []
    module.out_message = lambda message: calls.append(message)
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    module.task_message_to_master({"bibcode": "B", "run_id": "R"})
    assert calls == [{"bibcode": "B", "run_id": "R"}]
    assert events[0]["stage"] == "forward"


def test_task_message_to_master_list_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    calls = []
    events = []
    payload = [{"bibcode": "B1", "run_id": "R"}, {"bibcode": "B2", "run_id": "R"}]
    module.out_message = lambda message: calls.append(message)
    module.perf_metrics.emit_event = lambda **kwargs: events.append(kwargs)
    module.task_message_to_master(payload)
    assert calls == [payload, payload]
    assert len(events) == 2


def test_task_resend_to_master_bibcode_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.query_final_collection_table = lambda **kwargs: [{"bibcode": "B"}]
    forwarded = []
    module.task_message_to_master = lambda message: forwarded.append(message)
    module.task_resend_to_master({"bibcode": "B"})
    assert forwarded == [{"bibcode": "B"}]


def test_task_resend_to_master_scix_id_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.query_final_collection_table = lambda **kwargs: [{"scix_id": "S"}]
    forwarded = []
    module.task_message_to_master = lambda message: forwarded.append(message)
    module.task_resend_to_master({"scix_id": "S"})
    assert forwarded == [{"scix_id": "S"}]


def test_task_resend_to_master_run_id_path(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.query_final_collection_table = lambda **kwargs: [{"bibcode": "B"}]
    forwarded = []
    module.task_message_to_master = lambda message: forwarded.append(message)
    module.task_resend_to_master({"run_id": "R"})
    assert forwarded == [{"bibcode": "B"}]


def test_task_update_validated_records_forwards_successes(monkeypatch, base_fake_config, dummy_logger):
    module, _ = _import_tasks_module(monkeypatch, base_fake_config, dummy_logger)
    module.utils.classifyRequestRecordList_to_list = lambda message: [dict(message)]
    module.app.update_validated_records = lambda run_id: ([{"bibcode": "B"}], ["success"])
    forwarded = []
    module.task_message_to_master = lambda message: forwarded.append(message)
    module.task_update_validated_records({"run_id": "R"})
    assert forwarded == [{"bibcode": "B"}]
