import pytest
from types import SimpleNamespace
# from classifier import Classifier   # adjust import-path if renamed
from ClassifierPipeline.classifier import Classifier   # adjust import-path if renamed
# from ..classifier import Classifier   # adjust import-path if renamed


# ---------------------------------------------------------------------- #
# Lightweight fakes
# ---------------------------------------------------------------------- #
class DummyTokenizer(SimpleNamespace):
    cls_token_id: int = 101
    sep_token_id: int = 102
    pad_token_id: int = 0


@pytest.fixture(scope="module")
def clf():
    c = Classifier()
    # Monkey-patch heavy objects with cheap stand-ins
    c.tokenizer = DummyTokenizer()
    c.model = None         # not touched by helper methods
    return c


# ---------------------------------------------------------------------- #
# input_ids_splitter
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("length", "stride", "exp_chunks"),
    [
        (  10,  5, 2),   # shorter than window → still ≥ 1 chunk
        (1000, 255, 3),  # hand-checked against floor(1000/255)
        (510, 510, 1),   # exactly one full window
    ],
)
def test_splitter_chunk_count(clf, length, stride, exp_chunks):
    ids = list(range(length))
    chunks = clf.input_ids_splitter(ids, window_stride=stride)
    assert len(chunks) == exp_chunks


def test_splitter_window_size(clf):
    ids = list(range(800))
    chunks = clf.input_ids_splitter(ids, window_size=510, window_stride=255)
    # First window is full, last may be shorter
    assert len(chunks[0]) == 510
    assert len(chunks[-1]) <= 510


# ---------------------------------------------------------------------- #
# add_special_tokens_split_input_ids
# ---------------------------------------------------------------------- #
def test_add_special_tokens_padding(clf):
    ids = list(range(600))
    chunks = clf.input_ids_splitter(ids, window_size=510, window_stride=500)
    windows = clf.add_special_tokens_split_input_ids(chunks, clf.tokenizer)

    # All windows equal length after padding
    lengths = {len(w) for w in windows}
    assert len(lengths) == 1

    # Check special tokens
    first = windows[0]
    assert first[0] == clf.tokenizer.cls_token_id
    assert first[-1] == clf.tokenizer.sep_token_id

