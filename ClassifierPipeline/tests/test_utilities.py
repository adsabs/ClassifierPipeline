import os
import sys
import types
import tempfile
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------- Build lightweight mocks for external deps BEFORE importing utilities ----------

def _make_mock_adsputils():
    m = types.ModuleType("adsputils")

    def load_config(*args, **kwargs):
        return {
            "LOGGING_LEVEL": "DEBUG",
            "LOG_STDOUT": True,
            "CLASSIFICATION_THRESHOLDS": [0.6, 0.5, 0.55, 0.5, 0.7, 0.65, 0.4, 0.8],
            "ADDITIONAL_EARTH_SCIENCE_PROCESSING": "inactive",
            "ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD": 0.66,
            "CLASSIFICATION_PRETRAINED_MODEL": "unit/model",
            "CLASSIFICATION_PRETRAINED_MODEL_REVISION": "rev-1",
            "CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER": "tok-1",
            "ALLOWED_CATEGORIES": [
                "Astronomy", "Heliophysics", "Planetary Science", "Earth Science",
                "NASA-funded Biophysics", "Other Physics", "Other", "Text Garbage"
            ],
        }

    class _Logger:
        def info(self, *a, **k): pass
        def debug(self, *a, **k): pass

    def setup_logging(*args, **kwargs):
        return _Logger()


    def get_date(): return "1970-01-01"
    def ADSCelery(): return None
    def u2asc(s): return s

    m.load_config = load_config
    m.setup_logging = setup_logging
    m.get_date = get_date
    m.ADSCelery = ADSCelery
    m.u2asc = u2asc
    return m

def _make_mock_adsmsg():
    m = types.ModuleType("adsmsg")

    class ClassifyRequestRecord:  # empty shell
        pass

    class ClassifyRequestRecordList:
        def __init__(self):
            self.classify_requests = []

    class ClassifyResponseRecord:
        pass

    class ClassifyResponseRecordList:
        def __init__(self):
            self.classifyResponses = []   # camelCase JSON name
            self.classify_responses = []  # snake_case convenience

    m.ClassifyRequestRecord = ClassifyRequestRecord
    m.ClassifyRequestRecordList = ClassifyRequestRecordList
    m.ClassifyResponseRecord = ClassifyResponseRecord
    m.ClassifyResponseRecordList = ClassifyResponseRecordList
    return m

def _make_mock_pb_json_format():
    m = types.ModuleType("google.protobuf.json_format")

    def ParseDict(d, message):
        # request list
        if hasattr(message, "classify_requests") and ("classifyRequests" in d or "classify_requests" in d):
            items = d.get("classifyRequests", d.get("classify_requests"))
            message.classify_requests = [SimpleNamespace(**item) for item in items]
            message._raw = d
            return message
        # response list
        if hasattr(message, "classifyResponses") and "classifyResponses" in d:
            items = d["classifyResponses"]
            setattr(message, "classifyResponses", [SimpleNamespace(**item) for item in items])
            setattr(message, "classify_responses", [SimpleNamespace(**item) for item in items])
            message._raw = d
            return message
        # single messages
        for k, v in d.items():
            setattr(message, k, v)
        message._raw = d
        return message

    def MessageToDict(obj, preserving_proto_field_name=False):
        if isinstance(obj, SimpleNamespace):
            return vars(obj).copy()
        if hasattr(obj, "_raw"):
            return dict(obj._raw)
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

    m.ParseDict = ParseDict
    m.MessageToDict = MessageToDict
    m.Parse = MagicMock(name="Parse")
    return m

# Inject mocks
sys.modules.setdefault("adsputils", _make_mock_adsputils())
sys.modules.setdefault("adsmsg", _make_mock_adsmsg())
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules.setdefault("google.protobuf", types.ModuleType("google.protobuf"))
sys.modules.setdefault("google.protobuf.json_format", _make_mock_pb_json_format())

# ---------- Import utilities.py by path (robust to PYTHONPATH) ----------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent  # assumes tests/ is sibling to utilities.py
UTILITIES_PATH = PROJECT_DIR / "utilities.py"

if not UTILITIES_PATH.exists():
    # Fallback: try package-style path if repo is a package (ClassifierPipeline/utilities.py)
    pkg_utilities = PROJECT_DIR / "ClassifierPipeline" / "utilities.py"
    if pkg_utilities.exists():
        UTILITIES_PATH = pkg_utilities

spec = importlib.util.spec_from_file_location("utilities", str(UTILITIES_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load utilities.py from {UTILITIES_PATH}")
utilities = importlib.util.module_from_spec(spec)
sys.modules["utilities"] = utilities  # so relative imports resolve against this name
spec.loader.exec_module(utilities)

# ------------------------------ TESTS ------------------------------

class UtilitiesTestCase(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    # -------- file I/O --------

    def test_prepare_output_file_writes_header(self):
        output_path = os.path.join(self.tmpdir.name, "out.tsv")
        utilities.prepare_output_file(output_path)
        with open(output_path, "r", newline="") as f:
            text = f.read().rstrip("\n")
        # current header: bibcode, scix_id, run_id, title, ...
        self.assertTrue(
            text.startswith("bibcode\tscix_id\trun_id\ttitle\tcollections\tcollection_scores"),
            text
        )

    def test_add_record_to_output_file_appends_row(self):
        output_path = os.path.join(self.tmpdir.name, "out.tsv")
        utilities.prepare_output_file(output_path)
        record = {
            "bibcode": "2000Test.....A....1",
            "scix_id": "scix:abcd-1234-ef56",
            "run_id": "run-1",
            "title": "A Title",
            "collections": ["Astronomy", "Other Physics"],
            "collection_scores": [0.61, 0.70],
            "scores": [0.611, 0.5, 0.2, 0.3, 0.1, 0.701, 0.2, 0.01],
            "output_path": output_path,
        }
        utilities.add_record_to_output_file(record)
        with open(output_path, "r", newline="") as f:
            lines = [line.rstrip("\n") for line in f]
        self.assertEqual(len(lines), 2)
        header = lines[0].split("\t")
        row = lines[1].split("\t")
        row = [c.rstrip("\r") for c in row]
        self.assertEqual(header[:4], ["bibcode", "scix_id", "run_id", "title"])
        self.assertEqual(row[0], record["bibcode"])
        self.assertEqual(row[1], record["scix_id"])
        self.assertEqual(row[2], record["run_id"])
        self.assertEqual(row[3], record["title"])
        self.assertEqual(row[4], "Astronomy, Other Physics")
        self.assertEqual(row[5], "0.61, 0.7")
        self.assertEqual(row[6], "0.61")
        self.assertEqual(row[13], "0.01")  # garbage_score
        self.assertEqual(row[14], "")      # override

    # -------- thresholds & classification --------

    def test_classify_record_from_scores_basic(self):
        record = {
            "scores": [0.61, 0.50, 0.30, 0.51, 0.2, 0.7, 0.41, 0.79],
            "categories": ["Astronomy","Heliophysics","Planetary Science","Earth Science",
                           "NASA-funded Biophysics","Other Physics","Other","Text Garbage"]
        }
        out = utilities.classify_record_from_scores(record.copy())
        # self.assertEqual(out["collections"], ["Astronomy", "Heliophysics", "Earth Science", "Other Physics", "Other"])
        # self.assertEqual(out["collection_scores"], [0.61, 0.5, 0.51, 0.7, 0.41])
        self.assertEqual(out["collections"], ["Astronomy", "Earth Science", "Other Physics", "Other"])
        self.assertEqual(out["collection_scores"], [0.61, 0.51, 0.7, 0.41])

    def test_classify_record_from_scores_earth_science_override(self):
        utilities.config["ADDITIONAL_EARTH_SCIENCE_PROCESSING"] = "active"
        utilities.config["ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD"] = 0.66
        try:
            record = {
                "scores": [0.0, 0.0, 0.0, 0.70, 0.0, 0.0, 0.50, 0.0],
                "categories": ["Astronomy","Heliophysics","Planetary Science","Earth Science",
                               "NASA-funded Biophysics","Other Physics","Other","Text Garbage"]
            }
            out = utilities.classify_record_from_scores(record.copy())
            self.assertIn("Earth Science", out["collections"])
            self.assertNotIn("Other", out["collections"])
        finally:
            utilities.config["ADDITIONAL_EARTH_SCIENCE_PROCESSING"] = "inactive"

    # -------- category helpers --------

    def test_check_is_allowed_category(self):
        self.assertTrue(utilities.check_is_allowed_category(["Astronomy", "Earth Science"]))
        self.assertTrue(utilities.check_is_allowed_category(["astronomy", "earth science"]))
        self.assertFalse(utilities.check_is_allowed_category(["Astronomy", "NotARealCategory"]))

    def test_check_if_list_single_empty_string(self):
        self.assertTrue(utilities.check_if_list_single_empty_string([""]))
        self.assertFalse(utilities.check_if_list_single_empty_string([]))
        self.assertFalse(utilities.check_if_list_single_empty_string(["x"]))
        self.assertFalse(utilities.check_if_list_single_empty_string(""))

    # -------- fake data --------

    def test_return_fake_data(self):
        utilities.config["CLASSIFICATION_PRETRAINED_MODEL"] = "m"
        utilities.config["CLASSIFICATION_PRETRAINED_MODEL_REVISION"] = "r"
        utilities.config["CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER"] = "t"
        out = utilities.return_fake_data({})
        self.assertEqual(len(out["categories"]), 8)
        self.assertEqual(out["scores"], [0.5] * 8)
        self.assertEqual(out["model"]["model"], "m")
        self.assertIn("CLASSIFICATION_THRESHOLDS", out["postprocessing"])

    # -------- protobuf-ish conversions --------

    def _len_response_list(self, msg):
        if hasattr(msg, "classify_responses"):
            return len(getattr(msg, "classify_responses"))
        if hasattr(msg, "classifyResponses"):
            return len(getattr(msg, "classifyResponses"))
        return 0

    def test_list_to_ClassifyRequestRecordList_roundtrip(self):
        lst_msg = utilities.list_to_ClassifyRequestRecordList([
            {"bibcode": "b1", "title": "t1"},
            {"bibcode": "b2", "title": "t2"},
        ])
        out_list = utilities.classifyRequestRecordList_to_list(lst_msg)
        self.assertEqual(out_list, [{"bibcode": "b1", "title": "t1"},
                                    {"bibcode": "b2", "title": "t2"}])

    def test_dict_and_list_to_ClassifyResponseRecordList(self):
        msg = utilities.dict_to_ClassifyResponseRecord({
            "bibcode": "b3", "scix_id": "s3", "status": 0, "collections": ["Astronomy"]
        })
        self.assertTrue(hasattr(msg, "bibcode"))
        lst = utilities.list_to_ClassifyResponseRecordList([
            {"bibcode": "b3", "scix_id": "s3", "status": 0, "collections": ["Astronomy"]},
            {"bibcode": "b4", "scix_id": "s4", "status": 1, "collections": []},
        ])
        self.assertEqual(self._len_response_list(lst), 2)

    # -------- identifier parsing --------

    def test_check_identifier_scix(self):
        self.assertEqual(utilities.check_identifier("scix:abcd-1234-ef56"), "scix_id")

    def test_check_identifier_bibcode_like(self):
        # 19-char bibcode-like string (non-scix) -> "bibcode"
        self.assertEqual(utilities.check_identifier("2019ApJ....1234A01A"), "bibcode")

    def test_check_identifier_invalid_length(self):
        self.assertIsNone(utilities.check_identifier("too_short"))


if __name__ == "__main__":
    unittest.main()

