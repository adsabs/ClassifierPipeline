import importlib
import sys
import types

import pytest


def _import_run_module(monkeypatch, dummy_logger):
    fake_tasks = types.ModuleType("ClassifierPipeline.tasks")
    fake_tasks.task_update_record = lambda message: None
    fake_tasks.task_update_record.delay = fake_tasks.task_update_record
    fake_tasks.task_update_validated_records = lambda message: None
    fake_tasks.task_index_classified_record = lambda message: None
    fake_tasks.task_resend_to_master = lambda message: None
    fake_tasks.prepare_pre_ingest_run = lambda filename: ("PRE-RUN-ID", f"/prepared/{filename or 'pre-ingest'}_classified.tsv")

    fake_utils = types.ModuleType("ClassifierPipeline.utilities")
    fake_utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    fake_utils.classifyRequestRecordList_to_list = lambda message: message

    monkeypatch.setattr("adsputils.load_config", lambda proj_home=None: {"LOGGING_LEVEL": "INFO", "LOG_STDOUT": False, "DELAY_MESSAGE": False, "TEST_INPUT_DATA": "", "PRE_INGEST_OUTPUT_PREFIX": "input-text"})
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.utilities", fake_utils)
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def test_batch_pre_ingest_records_skips_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("title\tabstract\nTitle 1\tAbstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/pre.tsv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_get_pre_ingest_delimiter_uses_extension_defaults(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    monkeypatch.setattr(module, "sniff_pre_ingest_delimiter", lambda records_path: None)

    assert module.get_pre_ingest_delimiter("records.csv") == ","
    assert module.get_pre_ingest_delimiter("records.CSV") == ","
    assert module.get_pre_ingest_delimiter("records.tsv") == "\t"
    assert module.get_pre_ingest_delimiter("records.txt") == "\t"


def test_get_pre_ingest_delimiter_prefers_explicit_override(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)

    assert module.get_pre_ingest_delimiter("records.tsv", delimiter="csv") == ","
    assert module.get_pre_ingest_delimiter("records.csv", delimiter="tsv") == "\t"
    assert module.get_pre_ingest_delimiter("records.txt", delimiter=r"\t") == "\t"


def test_normalize_pre_ingest_delimiter_rejects_unsupported_value(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)

    with pytest.raises(ValueError, match="Unsupported delimiter"):
        module.normalize_pre_ingest_delimiter("pipe")


def test_get_pre_ingest_delimiter_uses_sniffed_value_when_no_override(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    monkeypatch.setattr(module, "sniff_pre_ingest_delimiter", lambda records_path: ",")

    assert module.get_pre_ingest_delimiter("records.txt") == ","


def test_batch_pre_ingest_records_accepts_no_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("Title 1\tAbstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/pre.tsv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_batch_pre_ingest_records_tsv_regression_guard(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("title\tabstract\nTitle 1\tAbstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured[0][0]["title"] == "Title 1"
    assert captured[0][0]["abstract"] == "Abstract 1"


def test_batch_pre_ingest_records_accepts_csv_with_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.csv"
    records.write_text("title,abstract\nTitle 1,Abstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/pre.csv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_batch_pre_ingest_records_accepts_csv_without_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.csv"
    records.write_text("Title 1,Abstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/pre.csv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_batch_pre_ingest_records_accepts_txt_with_sniffed_comma(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.txt"
    records.write_text("title,abstract\nTitle 1,Abstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/pre.txt_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_batch_pre_ingest_records_accepts_explicit_delimiter_override(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "misnamed.tsv"
    records.write_text("title,abstract\nTitle 1,Abstract 1\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10, delimiter="csv")

    assert captured == [[{"title": "Title 1", "abstract": "Abstract 1", "operation_step": "pre_ingest", "output_path": "/prepared/misnamed.tsv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_batch_pre_ingest_records_prepares_output_once_and_reuses_run_id(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("title\tabstract\nTitle 1\tAbstract 1\nTitle 2\tAbstract 2\n")
    captured = []
    prepared = []
    module.prepare_pre_ingest_run = lambda filename: (prepared.append(filename) or ("PRE-RUN-ID", f"/prepared/{filename}_classified.tsv"))
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=1)

    assert prepared == ["pre.tsv"]
    assert [batch[0]["run_id"] for batch in captured] == ["PRE-RUN-ID", "PRE-RUN-ID"]


def test_batch_pre_ingest_records_honors_exact_batch_size_without_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("T1\tA1\nT2\tA2\nT3\tA3\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=2)

    assert [len(batch) for batch in captured] == [2, 1]
    assert [row["title"] for row in captured[0]] == ["T1", "T2"]
    assert [row["title"] for row in captured[1]] == ["T3"]


def test_batch_pre_ingest_records_treats_title_word_as_data_when_second_column_not_header(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    records = tmp_path / "pre.tsv"
    records.write_text("title\tActually an abstract\n")
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.batch_pre_ingest_records(str(records), batch_size=10)

    assert captured == [[{"title": "title", "abstract": "Actually an abstract", "operation_step": "pre_ingest", "output_path": "/prepared/pre.tsv_classified.tsv", "run_id": "PRE-RUN-ID", "output_prepared": True}]]


def test_pre_ingest_row_to_dictionary_rejects_short_rows(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    with pytest.raises(ValueError, match="Expected 2 columns, got 1"):
        module.pre_ingest_row_to_dictionary(["only-title"])


def test_batch_pre_ingest_records_reports_delimiter_help_on_short_rows(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    records = tmp_path / "pre.tsv"
    records.write_text("title,abstract\nTitle 1,Abstract 1\n")

    with pytest.raises(ValueError, match="Supply --delimiter"):
        module.batch_pre_ingest_records(str(records), batch_size=10, delimiter="tsv")


def test_queue_pre_ingest_input_text_routes_as_abstract(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.queue_pre_ingest_input_text("sample body")

    assert captured == [[{"title": "", "abstract": "sample body", "operation_step": "pre_ingest", "output_path": "input-text"}]]


def test_queue_pre_ingest_input_text_uses_config_default_prefix(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    module.config["PRE_INGEST_OUTPUT_PREFIX"] = "configured-prefix"
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.queue_pre_ingest_input_text("sample body")

    assert captured[0][0]["output_path"] == "configured-prefix"


def test_queue_pre_ingest_input_text_uses_override_prefix(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    module.utils.list_to_ClassifyRequestRecordList = lambda payload: payload
    captured = []
    module.task_update_record.delay = lambda message: captured.append(message)

    module.queue_pre_ingest_input_text("sample body", output_prefix="custom-prefix")

    assert captured[0][0]["output_path"] == "custom-prefix"


def test_main_rejects_input_text_without_pre_ingest(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)

    with pytest.raises(SystemExit):
        module.main(["--input-text", "sample body"])


def test_main_rejects_output_prefix_without_pre_ingest(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)

    with pytest.raises(SystemExit):
        module.main(["--output-prefix", "custom-prefix"])


def test_main_rejects_delimiter_without_pre_ingest(monkeypatch, dummy_logger, capsys):
    module = _import_run_module(monkeypatch, dummy_logger)

    with pytest.raises(SystemExit):
        module.main(["--delimiter", "csv"])

    captured = capsys.readouterr()
    assert "`--delimiter` is only supported with `--pre-ingest`." in captured.err


def test_main_rejects_unsupported_delimiter_value(monkeypatch, dummy_logger, tmp_path, capsys):
    module = _import_run_module(monkeypatch, dummy_logger)
    records = tmp_path / "pre.txt"
    records.write_text("title,abstract\n")

    with pytest.raises(SystemExit):
        module.main(["--pre-ingest", "--records", str(records), "--delimiter", "pipe"])

    captured = capsys.readouterr()
    assert "Unsupported delimiter 'pipe'." in captured.err
    assert "Use one of: comma, csv, tab, tsv, ',', '\\t'." in captured.err


def test_main_rejects_pre_ingest_without_exactly_one_input_source(monkeypatch, dummy_logger, tmp_path, capsys):
    module = _import_run_module(monkeypatch, dummy_logger)
    records = tmp_path / "pre.tsv"
    records.write_text("title\tabstract\n")

    with pytest.raises(SystemExit):
        module.main(["--pre-ingest"])

    captured = capsys.readouterr()
    assert "`--pre-ingest` requires exactly one of `--records` or `--input-text`." in captured.err

    with pytest.raises(SystemExit):
        module.main(["--pre-ingest", "--records", str(records), "--input-text", "sample body"])

    captured = capsys.readouterr()
    assert "`--pre-ingest` requires exactly one of `--records` or `--input-text`." in captured.err


def test_main_rejects_delimiter_with_input_text(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)

    with pytest.raises(SystemExit):
        module.main(["--pre-ingest", "--input-text", "sample body", "--delimiter", "csv"])


def test_main_routes_input_text_through_pre_ingest_with_override(monkeypatch, dummy_logger):
    module = _import_run_module(monkeypatch, dummy_logger)
    captured = []
    module.queue_pre_ingest_input_text = lambda text, output_prefix=None: captured.append((text, output_prefix))

    module.main(["--pre-ingest", "--input-text", "sample body", "--output-prefix", "custom-prefix"])

    assert captured == [("sample body", "custom-prefix")]


def test_main_routes_records_through_pre_ingest_with_override(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    records = tmp_path / "pre.tsv"
    records.write_text("title\tabstract\n")
    captured = []
    module.batch_pre_ingest_records = lambda records_path, batch_size=500, output_prefix=None, delimiter=None: captured.append((records_path, batch_size, output_prefix, delimiter))

    module.main(["--pre-ingest", "--records", str(records), "--output-prefix", "custom-prefix"])

    assert captured == [(str(records), 500, "custom-prefix", None)]


def test_main_routes_records_through_pre_ingest_with_delimiter(monkeypatch, dummy_logger, tmp_path):
    module = _import_run_module(monkeypatch, dummy_logger)
    records = tmp_path / "pre.txt"
    records.write_text("title,abstract\n")
    captured = []
    module.batch_pre_ingest_records = lambda records_path, batch_size=500, output_prefix=None, delimiter=None: captured.append((records_path, batch_size, output_prefix, delimiter))

    module.main(["--pre-ingest", "--records", str(records), "--delimiter", "csv"])

    assert captured == [(str(records), 500, None, "csv")]
