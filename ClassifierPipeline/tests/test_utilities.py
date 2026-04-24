import importlib
import types


def _import_utilities(monkeypatch, base_fake_config, dummy_logger):
    monkeypatch.setattr("adsputils.load_config", lambda proj_home=None: dict(base_fake_config))
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)
    import sys
    sys.modules.pop("ClassifierPipeline.utilities", None)
    module = importlib.import_module("ClassifierPipeline.utilities")
    monkeypatch.setattr(module, "config", dict(base_fake_config), raising=True)
    monkeypatch.setattr(module, "logger", dummy_logger, raising=True)
    module.reset_output_buffers_for_tests()
    return module


class FakeListMessage:
    def __init__(self, field_name):
        self.field_name = field_name
        setattr(self, field_name, [])


def test_classify_record_from_scores_basic(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    record = {
        "categories": base_fake_config["ALLOWED_CATEGORIES"],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
    }
    out = module.classify_record_from_scores(record.copy())
    assert out["collections"] == ["Astronomy", "Other Physics", "Other"]
    assert out["collection_scores"] == [0.61, 0.35, 0.21]


def test_classify_record_from_scores_earth_science_override(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    record = {
        "categories": base_fake_config["ALLOWED_CATEGORIES"],
        "scores": [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.5, 0.0],
    }
    out = module.classify_record_from_scores(record)
    assert "Earth Science" in out["collections"]
    assert "Other" not in out["collections"]


def test_prepare_output_file_writes_header(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    text = output.read_text()
    assert text.startswith("bibcode\tscix_id\trun_id\ttitle\tcollections")
    assert "astrophysics_score" in text
    assert "gross_collection" in text


def test_add_record_to_output_file_appends_row(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    record = {
        "bibcode": "B",
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy", "Other"],
        "collection_scores": [0.61, 0.21],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
        "output_path": str(output),
    }
    module.add_record_to_output_file(record)
    module.flush_output_file(str(output))
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[1].startswith("B\tS\tR\tTitle\tAstronomy, Other")
    columns = lines[1].split("\t")
    assert columns[6:10] == ["0.61", "0.2", "0.4", "0.61"]
    assert columns[10:15] == ["0.5", "0.1", "0.35", "0.21", "0.5"]
    assert columns[15:17] == ["0.1", "astronomy"]


def test_add_record_to_output_file_buffers_until_threshold(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    record = {
        "bibcode": "B",
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy"],
        "collection_scores": [0.61],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
        "output_path": str(output),
    }
    for index in range(24):
        current = dict(record, bibcode=f"B{index}")
        module.add_record_to_output_file(current)
    assert output.read_text().strip().splitlines() == [
        "bibcode\tscix_id\trun_id\ttitle\tcollections\tcollection_scores\tastrophysics_score\theliophysics_score\tplanetary_science_score\tastronomy_score\tearth_science_score\tbiology_score\tphysics_score\tother_score\tgeneral_score\tgarbage_score\tgross_collection\toverride"
    ]


def test_add_record_to_output_file_flushes_at_threshold(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    record = {
        "bibcode": "B",
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy"],
        "collection_scores": [0.61],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
        "output_path": str(output),
    }
    for index in range(25):
        current = dict(record, bibcode=f"B{index}")
        module.add_record_to_output_file(current)
    assert len(output.read_text().strip().splitlines()) == 26


def test_flush_output_file_flushes_remaining_rows(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    record = {
        "bibcode": "B",
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy"],
        "collection_scores": [0.61],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
        "output_path": str(output),
    }
    module.add_record_to_output_file(record)
    module.flush_output_file(str(output))
    assert len(output.read_text().strip().splitlines()) == 2


def test_prepare_output_file_resets_existing_buffer(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.prepare_output_file(str(output))
    record = {
        "bibcode": "B",
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy"],
        "collection_scores": [0.61],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
        "output_path": str(output),
    }
    module.add_record_to_output_file(record)
    module.prepare_output_file(str(output))
    module.flush_output_file(str(output))
    assert len(output.read_text().strip().splitlines()) == 1


def test_ensure_output_file_creates_header_without_truncating_existing_rows(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = tmp_path / "out.tsv"
    module.ensure_output_file(str(output))
    initial_lines = output.read_text().strip().splitlines()
    assert len(initial_lines) == 1

    with open(output, "a", newline="") as handle:
        handle.write("B\tS\tR\tTitle\n")

    module.ensure_output_file(str(output))
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2


def test_build_output_row_uses_blank_identifiers_and_derived_scores(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    row = module.build_output_row(
        {
            "title": "Title",
            "run_id": "R",
            "collections": ["Other Physics"],
            "collection_scores": [0.81],
            "scores": [0.12, 0.23, 0.34, 0.91, 0.45, 0.81, 0.67, 0.05],
        }
    )
    assert row[:4] == ["", "", "R", "Title"]
    assert row[6:10] == [0.12, 0.23, 0.34, 0.34]
    assert row[10:15] == [0.91, 0.45, 0.81, 0.67, 0.91]
    assert row[15:17] == [0.05, "general"]


def test_build_output_row_gross_collection_tie_breaks_to_astronomy(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    row = module.build_output_row(
        {
            "title": "Tie",
            "run_id": "R",
            "collections": [],
            "collection_scores": [],
            "scores": [0.5, 0.4, 0.3, 0.5, 0.4, 0.5, 0.2, 0.1],
        }
    )
    assert row[9] == 0.5
    assert row[12] == 0.5
    assert row[14] == 0.5
    assert row[16] == "astronomy"


def test_safe_score_logs_debug_and_returns_zero_on_bad_payload(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    debug_messages = []
    monkeypatch.setattr(
        module,
        "logger",
        types.SimpleNamespace(
            debug=lambda message: debug_messages.append(message),
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        ),
        raising=True,
    )
    assert module._safe_score(["bad"], 0) == 0.0
    assert debug_messages
    assert "Unable to parse score at index 0" in debug_messages[0]


def test_buffering_is_isolated_per_output_path(monkeypatch, base_fake_config, dummy_logger, tmp_path):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output_one = tmp_path / "one.tsv"
    output_two = tmp_path / "two.tsv"
    module.prepare_output_file(str(output_one))
    module.prepare_output_file(str(output_two))
    base_record = {
        "scix_id": "S",
        "run_id": "R",
        "title": "Title",
        "collections": ["Astronomy"],
        "collection_scores": [0.61],
        "scores": [0.61, 0.2, 0.4, 0.5, 0.1, 0.35, 0.21, 0.1],
    }
    module.add_record_to_output_file(dict(base_record, bibcode="B1", output_path=str(output_one)))
    module.add_record_to_output_file(dict(base_record, bibcode="B2", output_path=str(output_two)))
    module.flush_output_file(str(output_one))
    assert len(output_one.read_text().strip().splitlines()) == 2
    assert len(output_two.read_text().strip().splitlines()) == 1


def test_check_is_allowed_category_true(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_is_allowed_category(["astronomy", "Earth Science"]) is True


def test_check_is_allowed_category_false(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_is_allowed_category(["astronomy", "Nope"]) is False


def test_check_if_list_single_empty_string_true(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_if_list_single_empty_string([""]) is True


def test_check_if_list_single_empty_string_false_cases(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_if_list_single_empty_string([]) is False
    assert module.check_if_list_single_empty_string(["", "x"]) is False
    assert module.check_if_list_single_empty_string("") is False


def test_return_fake_data_sets_expected_keys(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    output = module.return_fake_data({})
    assert {"categories", "scores", "model", "postprocessing"}.issubset(output.keys())
    assert len(output["categories"]) == len(output["scores"])


def test_filter_allowed_fields_request_mode(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    filtered = module.filter_allowed_fields({"bibcode": "B", "title": "T", "output_prepared": True, "extra": "x"})
    assert filtered == {"bibcode": "B", "title": "T", "output_prepared": True}


def test_filter_allowed_fields_response_mode(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    filtered = module.filter_allowed_fields({"bibcode": "B", "collections": ["A"], "scores": [1]}, response=True)
    assert filtered == {"bibcode": "B", "collections": ["A"]}


def test_dict_to_request_record_uses_filtered_payload(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    monkeypatch.setattr(module, "ClassifyRequestRecord", lambda: types.SimpleNamespace())
    monkeypatch.setattr(module, "ParseDict", lambda payload, message: {"payload": payload, "message": message})
    out = module.dict_to_ClassifyRequestRecord({"bibcode": "B", "extra": "x"})
    assert out["payload"] == {"bibcode": "B"}


def test_list_to_request_record_list_builds_payload(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    monkeypatch.setattr(module, "ClassifyRequestRecordList", lambda: FakeListMessage("classify_requests"))
    monkeypatch.setattr(module, "ParseDict", lambda payload, message: {"payload": payload, "message": message})
    out = module.list_to_ClassifyRequestRecordList([{"bibcode": "B", "extra": "x"}])
    assert out["payload"] == {"classify_requests": [{"bibcode": "B"}]}


def test_dict_to_response_record_uses_filtered_payload(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    monkeypatch.setattr(module, "ClassifyResponseRecord", lambda: types.SimpleNamespace())
    monkeypatch.setattr(module, "ParseDict", lambda payload, message: {"payload": payload, "message": message})
    out = module.dict_to_ClassifyResponseRecord({"bibcode": "B", "collections": ["A"], "scores": [1]})
    assert out["payload"] == {"bibcode": "B", "collections": ["A"]}


def test_list_to_response_record_list_builds_payload(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    monkeypatch.setattr(module, "ClassifyResponseRecordList", lambda: FakeListMessage("classifyResponses"))
    monkeypatch.setattr(module, "ParseDict", lambda payload, message: {"payload": payload, "message": message})
    out = module.list_to_ClassifyResponseRecordList([{"bibcode": "B", "collections": ["A"], "scores": [1]}])
    assert out["payload"] == {"classifyResponses": [{"bibcode": "B", "collections": ["A"]}]}


def test_classifyRequestRecordList_to_list_roundtrip(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)

    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    message = types.SimpleNamespace(classify_requests=[Request(bibcode="B"), Request(scix_id="S")])
    monkeypatch.setattr(module, "MessageToDict", lambda message, preserving_proto_field_name=True: dict(message.__dict__))
    assert module.classifyRequestRecordList_to_list(message) == [{"bibcode": "B"}, {"scix_id": "S"}]


def test_check_identifier_scix_id(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_identifier("scix:AB12-CD34-EF56") == "scix_id"


def test_check_identifier_bibcode_like(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_identifier("2019J....123..456AB") == "bibcode"


def test_check_identifier_invalid(monkeypatch, base_fake_config, dummy_logger):
    module = _import_utilities(monkeypatch, base_fake_config, dummy_logger)
    assert module.check_identifier("short") is None
