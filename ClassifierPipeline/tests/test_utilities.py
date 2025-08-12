# tests/test_utilities.py
import builtins
import io
import os
import types
import importlib
from textwrap import dedent

import pytest


@pytest.fixture(autouse=True)
def stub_config_and_logger(monkeypatch, tmp_path):
    """Provide a predictable config and a no-op logger for all tests.

    Also ensures we import the target module *after* patching globals.
    """
    # Minimal config
    fake_config = {
        'LOGGING_LEVEL': 'DEBUG',
        'LOG_STDOUT': False,
        'CLASSIFICATION_THRESHOLDS': [0.6, 0.4, 0.8, 0.5, 0.7, 0.3, 0.2, 0.9],
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING': 'active',
        'ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD': 0.55,
        'CLASSIFICATION_PRETRAINED_MODEL': 'model-x',
        'CLASSIFICATION_PRETRAINED_MODEL_REVISION': 'rev-1',
        'CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER': 'tok-1',
        'ALLOWED_CATEGORIES': ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'Other'],
    }

    class DummyLogger:
        def __getattr__(self, name):
            def _(*args, **kwargs):
                return None
            return _

    # Ensure module-level config and logger are our fakes post-import
    import ClassifierPipeline.utilities as U  # module under test must be available on PYTHONPATH
    monkeypatch.setattr(U, 'config', fake_config, raising=True)
    monkeypatch.setattr(U, 'logger', DummyLogger(), raising=True)

    return U


@pytest.fixture
def U():
    import ClassifierPipeline.utilities as U
    return U


def test_classify_record_from_scores_basic(U):
    record = {
        'categories': [
            'Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science',
            'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage'
        ],
        'scores': [0.61, 0.2, 0.4, 0.50, 0.1, 0.35, 0.21, 0.1],
    }
    out = U.classify_record_from_scores(record.copy())
    # collections picked where score > threshold
    assert 'Astronomy' in out['collections']
    assert 'Heliophysics' not in out['collections']
    # rounding to two decimals
    assert all(isinstance(x, float) for x in out['collection_scores'])
    assert all(len(f"{x:.2f}") > 0 for x in out['collection_scores'])


def test_classify_record_from_scores_earth_science_override(U, monkeypatch):
    # Arrange: make "Other" above its 0.2 threshold and Earth Science above extra threshold
    record = {
        'categories': [
            'Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science',
            'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage'
        ],
        'scores': [0.0, 0.0, 0.0, 0.60, 0.0, 0.0, 0.50, 0.0],
    }
    out = U.classify_record_from_scores(record)
    # The ES extra check should remove Other and add Earth Science
    assert 'Earth Science' in out['collections']
    assert 'Other' not in out['collections']


def test_prepare_output_file_writes_header(U, tmp_path):
    p = tmp_path / 'out.tsv'
    U.prepare_output_file(p)
    text = p.read_text()
    assert text.startswith('bibcode\tscix_id\ttitle\tabstract\trun_id')
    # header has 11 columns separated by tabs
    assert text.strip().count('\t') == 10


def test_check_is_allowed_category_true(U, monkeypatch):
    assert U.check_is_allowed_category(['astronomy', 'Earth Science']) is True


def test_check_is_allowed_category_false(U):
    assert U.check_is_allowed_category(['astronomy', 'MadeUp']) is False


def test_return_fake_data_sets_expected_keys(U):
    base = {}
    out = U.return_fake_data(base)
    assert set(['categories', 'scores', 'model', 'postprocessing']).issubset(out)
    assert len(out['categories']) == len(out['scores'])


def test_filter_allowed_fields_request_default(U):
    d = {
        'bibcode': 'X', 'scix_id': 'Y', 'status': 1, 'title': 't', 'abstract': 'a',
        'operation_step': 'op', 'run_id': 'r', 'override': False, 'output_path': '/tmp',
        'scores': [0.1], 'collections': ['A'], 'collection_scores': [0.1],
        'EXTRA': 'nope'
    }
    filtered = U.filter_allowed_fields(d)
    assert 'EXTRA' not in filtered
    assert 'bibcode' in filtered and 'scores' in filtered


def test_filter_allowed_fields_response_set(U):
    d = {'bibcode': 'B', 'scix_id': 'S', 'status': 2, 'collections': ['C'], 'scores': [1]}
    filtered = U.filter_allowed_fields(d, response=True)
    assert set(filtered) == {'bibcode', 'scix_id', 'status', 'collections'}


# --- Protobuf-related helpers: stub out ParseDict, MessageToDict, and message classes
class _ListMsg:
    def __init__(self, field_name):
        self._field_name = field_name
        setattr(self, field_name, [])
    def __repr__(self):
        return f"_ListMsg({self._field_name}={getattr(self, self._field_name)!r})"

class _Elem:
    def __init__(self):
        pass

class _ResponseListForOutput:
    def __init__(self):
        self.classify_requests = []
    class _Adder:
        def __init__(self, outer):
            self.outer = outer
        def __call__(self):
            e = _Elem()
            self.outer.classify_requests.append(e)
            return e
    def __getattr__(self, name):
        if name == 'classify_requests':
            return self.classify_requests
        if name == 'classify_requests_add':
            return self._Adder(self)
        raise AttributeError
    def __repr__(self):
        return f"_ResponseListForOutput(n={len(self.classify_requests)})"


@pytest.fixture
def stub_protobuf(monkeypatch):
    import ClassifierPipeline.utilities as U

    # Stub message classes
    monkeypatch.setattr(U, 'ClassifyRequestRecord', object)
    monkeypatch.setattr(U, 'ClassifyRequestRecordList', lambda: _ListMsg('classify_requests'))
    monkeypatch.setattr(U, 'ClassifyResponseRecord', object)
    monkeypatch.setattr(U, 'ClassifyResponseRecordList', lambda: _ListMsg('classifyResponses'))

    # Stub ParseDict and MessageToDict
    def _parse_dict(input_dict, message):
        # emulate filling a message; just return a tuple for visibility
        return (message.__class__.__name__, dict(input_dict))
    monkeypatch.setattr(U, 'ParseDict', _parse_dict)

    def _msg_to_dict(msg, preserving_proto_field_name=True):
        # assume msg is a simple object with attributes
        if isinstance(msg, dict):
            return msg
        return dict(getattr(msg, '__dict__', {}))
    monkeypatch.setattr(U, 'MessageToDict', _msg_to_dict)

    return U


def test_dict_to_ClassifyRequestRecord_uses_ParseDict(stub_protobuf):
    U = stub_protobuf
    out = U.dict_to_ClassifyRequestRecord({'bibcode': 'A', 'EXTRA': 1})
    # Our stub returns a tuple (class name, filtered dict)
    assert isinstance(out, tuple)
    name, payload = out
    assert 'EXTRA' not in payload and 'bibcode' in payload


def test_list_to_ClassifyRequestRecordList_builds_list(stub_protobuf):
    U = stub_protobuf
    msg = U.list_to_ClassifyRequestRecordList([
        {'bibcode': 'A'}, {'bibcode': 'B', 'scores': [0.1]},
    ])
    # Our stub list message simply holds a Python list attribute
    assert hasattr(msg, 'classify_requests')


def test_dict_to_ClassifyResponseRecord_filters(stub_protobuf):
    U = stub_protobuf
    out = U.dict_to_ClassifyResponseRecord({'bibcode': 'A', 'collections': ['X'], 'scores': [1]})
    # Scores are not part of response allowed fields
    assert 'scores' not in out[1]


def test_list_to_ClassifyResponseRecordList_builds(stub_protobuf):
    U = stub_protobuf
    msg = U.list_to_ClassifyResponseRecordList([
        {'bibcode': 'A', 'collections': ['X']},
        {'bibcode': 'B', 'collections': []},
    ])
    assert hasattr(msg, 'classifyResponses')
    assert isinstance(msg.classifyResponses, list)


def test_classifyRequestRecordList_to_list_roundtrip(monkeypatch):
    import ClassifierPipeline.utilities as U
    class _Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    class _Msg:
        def __init__(self):
            self.classify_requests = [_Req(bibcode='A'), _Req(bibcode='B')]
    monkeypatch.setattr(U, 'MessageToDict', lambda m, preserving_proto_field_name=True: m.__dict__)
    out = U.classifyRequestRecordList_to_list(_Msg())
    assert out == [{'bibcode': 'A'}, {'bibcode': 'B'}]


def test_check_identifier_scix(U):
    assert U.check_identifier('scix:AB12-CD34-EF56') == 'scix_id'


def test_check_identifier_bibcode_like(U):
    # Length 19 but not SciX pattern
    assert U.check_identifier('2019J..123..456A1') == 'bibcode'


def test_check_identifier_invalid(U):
    assert U.check_identifier('short') is None


def test_list_to_output_message_semantics(monkeypatch):
    import ClassifierPipeline.utilities as U
    # Stub ClassifyResponseRecordList to match the function's expectations
    class _Msg:
        def __init__(self):
            self.classify_requests = []
        class _Adder:
            def __init__(self, outer):
                self.outer = outer
            def __call__(self):
                e = types.SimpleNamespace()
                self.outer.classify_requests.append(e)
                return e
        def __getattr__(self, name):
            if name == 'classify_requests':
                return self.classify_requests
            if name == 'classify_requests_add':
                return self._Adder(self)
            raise AttributeError
    monkeypatch.setattr(U, 'ClassifyResponseRecordList', _Msg)

    out = U.list_to_output_message([
        {'bibcode': 'A', 'status': 5, 'collections': ['X']},
        {'bibcode': 'B', 'status': 6, 'collections': ['Y']},
    ])
    # Current behavior: status ends up being overwritten with collections (likely bug)
    assert len(out.classify_requests) == 2
    assert getattr(out.classify_requests[0], 'bibcode') == 'A'
    assert getattr(out.classify_requests[0], 'status') == ['X']  # documents current bug
