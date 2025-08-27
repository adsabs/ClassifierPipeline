import unittest
from unittest import mock
import importlib

# ---------------------------------------------------------------------------- #
# Dummy classes to mock transformers behavior                                 #
# ---------------------------------------------------------------------------- #
class _Dummy:
    def __init__(self, *_, **__):
        pass

    def __call__(self, *args, **kwargs):
        class _Out:
            logits = [[0.1]]
        return _Out()


class TestAstroBERTClassification(unittest.TestCase):
    def setUp(self):
        patcher_tokenizer = mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_Dummy(),
        )
        patcher_model = mock.patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=_Dummy(),
        )

        self.mock_tokenizer = patcher_tokenizer.start()
        self.mock_model = patcher_model.start()

        self.addCleanup(patcher_tokenizer.stop)
        self.addCleanup(patcher_model.stop)

    def test_module_imports(self):
        import ClassifierPipeline.astrobert_classification as astrobert_classification
        self.assertTrue(hasattr(astrobert_classification, "AstroBERTClassification"))
        self.assertTrue(bool(astrobert_classification.AstroBERTClassification.labels))

    def test_mappings_consistency(self):
        import ClassifierPipeline.astrobert_classification as astrobert_classification
        mapping = astrobert_classification.AstroBERTClassification

        self.assertEqual(
            {mapping.id2label[k] for k in mapping.id2label}, set(mapping.labels)
        )
        for label in mapping.labels:
            idx = mapping.label2id[label]
            self.assertEqual(mapping.id2label[idx], label)

    def test_singleton(self):
        import ClassifierPipeline.astrobert_classification as astrobert_classification
        mod1 = astrobert_classification
        mod2 = importlib.reload(astrobert_classification)

        self.assertIs(mod1.AstroBERTClassification.model, mod2.AstroBERTClassification.model)
        self.assertIs(mod1.AstroBERTClassification.tokenizer, mod2.AstroBERTClassification.tokenizer)


if __name__ == "__main__":
    unittest.main()

