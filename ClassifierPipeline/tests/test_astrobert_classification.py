import importlib
import sys
import types


def _import_module(monkeypatch, base_fake_config, dummy_logger):
    tokenizer_calls = []
    model_calls = []
    warning_calls = []

    class DummyTokenizer:
        pass

    class DummyModel:
        pass

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            return DummyTokenizer()

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model_calls.append((args, kwargs))
            return DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = AutoModelForSequenceClassification
    fake_transformers.TokenClassificationPipeline = object
    fake_transformers.logging = types.SimpleNamespace(set_verbosity_warning=lambda: warning_calls.append(True))

    monkeypatch.setattr("adsputils.load_config", lambda proj_home=None: dict(base_fake_config))
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("ClassifierPipeline.astrobert_classification", None)
    module = importlib.import_module("ClassifierPipeline.astrobert_classification")
    return module, tokenizer_calls, model_calls, warning_calls, DummyTokenizer, DummyModel


def test_import_loads_model_and_tokenizer_from_config(monkeypatch, base_fake_config, dummy_logger):
    module, tokenizer_calls, model_calls, warning_calls, DummyTokenizer, DummyModel = _import_module(
        monkeypatch, base_fake_config, dummy_logger
    )
    assert len(tokenizer_calls) == 1
    assert len(model_calls) == 1
    _, tok_kwargs = tokenizer_calls[0]
    _, model_kwargs = model_calls[0]
    assert tok_kwargs["pretrained_model_name_or_path"] == base_fake_config["CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER"]
    assert tok_kwargs["revision"] == base_fake_config["CLASSIFICATION_PRETRAINED_MODEL_REVISION"]
    assert tok_kwargs["do_lower_case"] is False
    assert model_kwargs["pretrained_model_name_or_path"] == base_fake_config["CLASSIFICATION_PRETRAINED_MODEL"]
    assert model_kwargs["revision"] == base_fake_config["CLASSIFICATION_PRETRAINED_MODEL_REVISION"]
    assert model_kwargs["num_labels"] == len(base_fake_config["ALLOWED_CATEGORIES"])
    assert model_kwargs["problem_type"] == "multi_label_classification"
    assert warning_calls == [True]
    assert isinstance(module.AstroBERTClassification.tokenizer, DummyTokenizer)
    assert isinstance(module.AstroBERTClassification.model, DummyModel)


def test_label_mappings_are_consistent(monkeypatch, base_fake_config, dummy_logger):
    module, _, _, _, _, _ = _import_module(monkeypatch, base_fake_config, dummy_logger)
    assert module.labels == base_fake_config["ALLOWED_CATEGORIES"]
    assert len(module.id2label) == len(module.label2id) == len(module.labels)
    for index, label in module.id2label.items():
        assert module.label2id[label] == index


def test_class_wrapper_exposes_loaded_singletons(monkeypatch, base_fake_config, dummy_logger):
    module, _, _, _, _, _ = _import_module(monkeypatch, base_fake_config, dummy_logger)
    wrapper = module.AstroBERTClassification
    assert wrapper.tokenizer is module.tokenizer
    assert wrapper.model is module.model
    assert wrapper.labels == module.labels
    assert wrapper.id2label == module.id2label
    assert wrapper.label2id == module.label2id


def test_transformers_warning_logging_is_configured(monkeypatch, base_fake_config, dummy_logger):
    _, _, _, warning_calls, _, _ = _import_module(monkeypatch, base_fake_config, dummy_logger)
    assert warning_calls == [True]
