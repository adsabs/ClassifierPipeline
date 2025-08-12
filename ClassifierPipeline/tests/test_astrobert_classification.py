"""
Unit tests for astrobert_classification

These tests *mock out* the expensive Hugging Face `from_pretrained` calls so
that they run quickly and without network/GPU dependencies.
"""

from unittest import mock
import importlib  # still used for reload in the singleton test
import pytest

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def stub_hf(monkeypatch):
    """Replace model + tokenizer loading with cheap stubs, *before* import."""

    class _Dummy:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *args, **kwargs):
            # mimic HF output shape
            class _Out:
                logits = [[0.1]]
            return _Out()

    # Patch the HF classes directly so the patched methods are in effect
    # even during the module import of astrobert_classification.
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **k: _Dummy(),
        raising=True,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda *a, **k: _Dummy(),
        raising=True,
    )

    yield  # monkeypatch fixture will undo patches automatically


# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #
def test_module_imports(stub_hf):
    """Module should import without accessing the network."""
    import ClassifierPipeline.astrobert_classification as astrobert_classification
    assert hasattr(astrobert_classification, "AstroBERTClassification")
    assert astrobert_classification.AstroBERTClassification.labels  # not empty


def test_mappings_consistency(stub_hf):
    """id2label and label2id must be perfect inverses."""
    import ClassifierPipeline.astrobert_classification as astrobert_classification
    mapping = astrobert_classification.AstroBERTClassification
    assert {mapping.id2label[k] for k in mapping.id2label} == set(mapping.labels)
    for label in mapping.labels:
        idx = mapping.label2id[label]
        assert mapping.id2label[idx] == label


def test_singleton(stub_hf):
    """
    Re-importing the module should not create another model instance.

    We verify by checking object identity.
    """
    import ClassifierPipeline.astrobert_classification as astrobert_classification
    mod1 = astrobert_classification
    # Force a re-import via importlib.reload
    mod2 = importlib.reload(astrobert_classification)

    assert mod1.AstroBERTClassification.model is mod2.AstroBERTClassification.model
    assert mod1.AstroBERTClassification.tokenizer is mod2.AstroBERTClassification.tokenizer

