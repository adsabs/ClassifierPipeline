"""
These tests heavily mock external dependencies (Hugging Face models/tokenizers,
Torch model inference, and ADS utilities) so the suite runs fast and deterministically.

Assumptions
-----------
- The production code typically lives in `classifier.py` or `ClassifierPipeline/classifier.py`.
  The test auto-detects both. You can also set an env var `MODULE_UNDER_TEST` to an
  explicit module path (e.g., `ClassifierPipeline.classifier`).
- PyTorch is available at import time (the application imports `tensor`/`no_grad`).

What is covered
---------------
- `Classifier.input_ids_splitter` behavior across edge cases.
- `Classifier.add_special_tokens_split_input_ids` correctness for CLS/SEP/PAD and padding.
- `Classifier.batch_score_SciX_categories` for `max`, `mean`, and a custom combiner,
  including thresholding and that the tokenizer is called with `add_special_tokens=False`.

The tests create in-memory stub modules for `adsputils` and
`ClassifierPipeline.astrobert_classification` **before** importing the module under test,
which avoids loading real configs or models.
"""
from __future__ import annotations

import os
import importlib
import sys
import types
import unittest
from typing import Dict, List

import torch

# ---------------------------
# Lightweight stubs installed before importing the SUT
# ---------------------------
class _DummyLogger:
    def info(self, *_, **__):
        pass

    def debug(self, *_, **__):
        pass

    def exception(self, *_, **__):
        pass


def _make_adsputils_stub() -> types.ModuleType:
    mod = types.ModuleType("adsputils")

    def load_config(*_, **__):
        return {}

    def setup_logging(*_, **__):
        return _DummyLogger()

    mod.load_config = load_config
    mod.setup_logging = setup_logging
    return mod


def _install_classifierpipeline_stub() -> None:
    """Install a stub ONLY for the *child* module
    `ClassifierPipeline.astrobert_classification` without clobbering the real
    `ClassifierPipeline` package if it exists.
    """
    try:
        parent = importlib.import_module("ClassifierPipeline")
    except ModuleNotFoundError:
        parent = types.ModuleType("ClassifierPipeline")
        # Mark as a package so submodules can exist
        parent.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ClassifierPipeline"] = parent

    child = types.ModuleType("ClassifierPipeline.astrobert_classification")

    # Shared, mutable state for tests to control model outputs
    child.LOGITS_MAP: Dict[int, torch.Tensor] = {}

    class MockTokenizer:
        """Very small tokenizer stub returning pre-seeded token id lists per text.

        Usage in tests: set `classifier_instance.tokenizer.token_map[text] = [ids...]`.
        """

        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0

        def __init__(self):
            self.token_map: Dict[str, List[int]] = {}
            self.calls: List[dict] = []

        def __call__(self, texts: List[str], add_special_tokens: bool = False, **_):
            self.calls.append({"texts": list(texts), "add_special_tokens": add_special_tokens})
            try:
                input_ids = [self.token_map[t] for t in texts]
            except KeyError as exc:
                raise AssertionError(
                    f"Tokenizer mapping not provided for text: {exc} — set token_map in your test"
                )
            return {"input_ids": input_ids}

    class DummyModel:
        """Returns an object with a `.logits` tensor based on batch length.

        The logits tensor is selected from `LOGITS_MAP[len(batch)]` when present;
        otherwise defaults to zeros with shape (batch_len, 3).
        """

        def __init__(self, logits_map: Dict[int, torch.Tensor]):
            self._logits_map = logits_map

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                batch_len = int(input_ids.shape[0])
            else:  # list/sequence
                batch_len = len(input_ids)

            logits = self._logits_map.get(batch_len)
            if logits is None:
                logits = torch.zeros((batch_len, 3), dtype=torch.float32)

            class _Out:
                def __init__(self, logits_tensor: torch.Tensor):
                    self.logits = logits_tensor

            return _Out(logits)

    class AstroBERTClassification:  # noqa: N801 - mirrors prod symbol
        def __init__(self):
            self.tokenizer = MockTokenizer()
            self.model = DummyModel(child.LOGITS_MAP)
            self.labels = ["A", "B", "C"]
            self.id2label = {i: l for i, l in enumerate(self.labels)}
            self.label2id = {l: i for i, l in enumerate(self.labels)}

    child.AstroBERTClassification = AstroBERTClassification

    # Attach child without replacing any other submodules (like real `tests`)
    setattr(parent, "astrobert_classification", child)
    sys.modules[child.__name__] = child


# Helper to import the module under test from common layouts

def _import_module_under_test():
    env_target = os.getenv("MODULE_UNDER_TEST")
    candidates = []
    if env_target:
        candidates.append(env_target)
    # If tests live inside a package (e.g., ClassifierPipeline.tests), try that root
    root_pkg = (__package__ or "").split(".")[0] or None
    if root_pkg:
        candidates.append(f"{root_pkg}.classifier")
    # Fallback to a top-level module
    candidates.append("classifier")

    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            last_err = e
    # If all imports failed, surface the last error (most informative)
    raise last_err

# Inject stubs so importing the module under test won't pull heavy deps
sys.modules["adsputils"] = _make_adsputils_stub()
_install_classifierpipeline_stub()

# Import the module under test (after stubbing)
classifier = _import_module_under_test()


class ClassifierUnitTests(unittest.TestCase):
    """Unit tests for core behaviors of the `Classifier` class.

    The instantiated classifier uses fully mocked tokenizer/model from the stubbed
    `ClassifierPipeline.astrobert_classification` module.
    """

    def setUp(self):
        # Fresh instance per test to avoid cross-test state
        self.clf = classifier.Classifier()

    # -----------------------------
    # input_ids_splitter
    # -----------------------------
    def test_input_ids_splitter_with_overlap_and_tail(self):
        """Splits respect window/stride and include a shorter trailing chunk."""
        input_ids = list(range(600))
        splits = self.clf.input_ids_splitter(input_ids, window_size=510, window_stride=255)
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 510)
        self.assertEqual(len(splits[1]), 345)  # 600 - 255

    def test_input_ids_splitter_handles_empty_input(self):
        """Empty input still yields a single (empty) split per implementation."""
        splits = self.clf.input_ids_splitter([], window_size=510, window_stride=255)
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0], [])

    # -----------------------------
    # add_special_tokens_split_input_ids
    # -----------------------------
    def test_add_special_tokens_and_padding(self):
        """Ensures CLS/SEP are added and last sequence padded to first's length."""
        # Arrange: simple split and known token ids
        splits = [[1, 2, 3], [4]]
        tok = self.clf.tokenizer
        tok.cls_token_id, tok.sep_token_id, tok.pad_token_id = 101, 102, 0

        # Act
        result = self.clf.add_special_tokens_split_input_ids(splits, tok)

        # First row gets CLS/SEP
        self.assertEqual(result[0], [101, 1, 2, 3, 102])
        # Second row gets CLS/SEP then pads to match first row length
        self.assertEqual(result[1], [101, 4, 102, 0, 0])
        # All rows equal length
        self.assertTrue(len({len(r) for r in result}) == 1)

    # -----------------------------
    # batch_score_SciX_categories
    # -----------------------------
    def test_batch_score_max_combiner_with_thresholds(self):
        """`max` combiner merges split predictions and applies thresholds per label."""
        # Arrange
        tok = self.clf.tokenizer
        tok.token_map = {"t1": list(range(8))}  # -> with window 4/4 => 2 splits

        # shape: (2 splits, 3 labels)
        logits = torch.tensor([[0.0, 2.0, -2.0], [0.0, -2.0, 2.0]], dtype=torch.float32)
        sys.modules["ClassifierPipeline.astrobert_classification"].LOGITS_MAP[2] = logits

        # Act
        cats, scores = self.clf.batch_score_SciX_categories(
            ["t1"], score_combiner="max", score_thresholds=[0.6, 0.6, 0.6], window_size=4, window_stride=4
        )

        # Assert categories
        self.assertEqual(cats, [["B", "C"]])

        # Assert scores equal to sigmoid(logits).max(dim=0)
        expected = torch.sigmoid(logits).max(dim=0)[0].tolist()
        for got, exp in zip(scores[0], expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_batch_score_mean_combiner_defaults_to_all_labels(self):
        """When thresholds are None (-> zeros), all labels with non-negative scores pass."""
        tok = self.clf.tokenizer
        tok.token_map = {"single": list(range(4))}  # 1 split with window 4/4

        logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        sys.modules["ClassifierPipeline.astrobert_classification"].LOGITS_MAP[1] = logits

        cats, scores = self.clf.batch_score_SciX_categories(
            ["single"], score_combiner="mean", window_size=4, window_stride=4
        )

        # With thresholds defaulting to 0.0, sigmoid(0) == 0.5 -> all labels
        self.assertEqual(cats[0], self.clf.labels)
        expected = torch.sigmoid(logits).mean(dim=0).tolist()
        for got, exp in zip(scores[0], expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_batch_score_custom_combiner_min(self):
        """Supports a custom combiner function operating on the predictions tensor."""
        tok = self.clf.tokenizer
        tok.token_map = {"doc": list(range(8))}  # -> 2 splits

        logits = torch.tensor([[0.0, 2.0, -2.0], [0.0, -2.0, 2.0]], dtype=torch.float32)
        sys.modules["ClassifierPipeline.astrobert_classification"].LOGITS_MAP[2] = logits

        combiner = lambda preds: preds.min(dim=0)[0]
        cats, scores = self.clf.batch_score_SciX_categories(
            ["doc"], score_combiner=combiner, score_thresholds=[0.4, 0.4, 0.4], window_size=4, window_stride=4
        )

        # min(sigmoid(...)) across splits yields 0.5, ~0.119, ~0.119 -> only label 'A' >= 0.4
        self.assertEqual(cats, [["A"]])

        expected = torch.sigmoid(logits).min(dim=0)[0].tolist()
        for got, exp in zip(scores[0], expected):
            self.assertAlmostEqual(got, exp, places=6)

    def test_tokenizer_called_without_special_tokens(self):
        """The tokenizer must be called with `add_special_tokens=False` as in prod code."""
        tok = self.clf.tokenizer
        tok.token_map = {"check": list(range(4))}
        logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        sys.modules["ClassifierPipeline.astrobert_classification"].LOGITS_MAP[1] = logits

        self.clf.batch_score_SciX_categories(["check"], window_size=4, window_stride=4)
        self.assertTrue(tok.calls, "Tokenizer was not called")
        self.assertFalse(tok.calls[-1]["add_special_tokens"])  # must be False


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)

