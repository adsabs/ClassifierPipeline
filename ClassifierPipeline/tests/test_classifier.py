import importlib
import math
import sys
import types

import pytest


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def sigmoid(self):
        if self.values and isinstance(self.values[0], list):
            return FakeTensor([[1.0 / (1.0 + math.exp(-value)) for value in row] for row in self.values])
        return FakeTensor([1.0 / (1.0 + math.exp(-value)) for value in self.values])

    def mean(self, dim=0):
        columns = list(zip(*self.values))
        return FakeTensor([sum(column) / len(column) for column in columns])

    def max(self, dim=0):
        columns = list(zip(*self.values))
        return FakeTensor([max(column) for column in columns]), None

    def tolist(self):
        return self.values

    def __iter__(self):
        return iter(self.values)


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __init__(self, token_ids=None):
        self.token_ids = token_ids or [[1, 2, 3]]
        self.calls = []

    def __call__(self, texts, add_special_tokens=False):
        self.calls.append({"texts": texts, "add_special_tokens": add_special_tokens})
        return {"input_ids": self.token_ids}


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeOutput(self.outputs.pop(0))


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _import_classifier(monkeypatch, base_fake_config, dummy_logger, fake_wrapper):
    monkeypatch.setattr("adsputils.load_config", lambda proj_home=None: dict(base_fake_config))
    monkeypatch.setattr("adsputils.setup_logging", lambda *args, **kwargs: dummy_logger)

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: FakeNoGrad()
    fake_torch.tensor = lambda values: FakeTensor(values)

    fake_astrobert = types.ModuleType("ClassifierPipeline.astrobert_classification")
    fake_astrobert.AstroBERTClassification = fake_wrapper

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.astrobert_classification", fake_astrobert)
    sys.modules.pop("ClassifierPipeline.classifier", None)
    return importlib.import_module("ClassifierPipeline.classifier")


def test_init_copies_astrobert_wrapper_state(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer()
    model = FakeModel([FakeTensor([[0.0]])])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = model
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    assert classifier.tokenizer is tokenizer
    assert classifier.model is model
    assert classifier.labels == ["Astronomy"]


def test_input_ids_splitter_short_input(monkeypatch, base_fake_config, dummy_logger):
    class FakeWrapper:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel([FakeTensor([[0.0]])])
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    assert classifier.input_ids_splitter([1, 2, 3], window_size=5, window_stride=2) == [[1, 2, 3]]


def test_input_ids_splitter_long_input(monkeypatch, base_fake_config, dummy_logger):
    class FakeWrapper:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel([FakeTensor([[0.0]])])
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    assert classifier.input_ids_splitter([1, 2, 3, 4, 5, 6], window_size=4, window_stride=2) == [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6]]


def test_add_special_tokens_and_padding(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer()

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = FakeModel([FakeTensor([[0.0]])])
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    result = classifier.add_special_tokens_split_input_ids([[1, 2], [3]], tokenizer)
    assert result[0] == [101, 1, 2, 102]
    assert result[1] == [101, 3, 102, 0]


def test_batch_score_with_max_combiner(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3], [4, 5]])
    model = FakeModel([
        FakeTensor([[-1.3862943611, 2.1972245773], [1.3862943611, -2.1972245773]]),
        FakeTensor([[0.8472978604, -0.8472978604]]),
    ])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = model
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    categories, scores = classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        score_thresholds=[0.5, 0.5],
        window_size=3,
        window_stride=2,
    )
    assert tokenizer.calls[0]["add_special_tokens"] is False
    assert categories == [["Astronomy", "Heliophysics"], ["Astronomy"]]
    assert scores[0] == pytest.approx([0.8, 0.9], rel=1e-5)
    assert scores[1] == pytest.approx([0.7, 0.3], rel=1e-5)


def test_batch_score_with_mean_combiner(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3]])
    model = FakeModel([
        FakeTensor([[-1.3862943611, 2.1972245773], [1.3862943611, -2.1972245773]]),
    ])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = model
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    categories, scores = classifier.batch_score_SciX_categories(
        ["doc1"],
        score_combiner="mean",
        score_thresholds=[0.6, 0.4],
        window_size=3,
        window_stride=2,
    )
    assert scores[0] == pytest.approx([0.5, 0.5], rel=1e-5)
    assert categories == [["Heliophysics"]]


def test_batch_score_applies_thresholds(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2]])
    model = FakeModel([FakeTensor([[0.4473122180, -1.3862943611]])])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = model
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    categories, scores = classifier.batch_score_SciX_categories(["doc"], score_thresholds=[0.6, 0.3])
    assert categories == [["Astronomy"]]
    assert scores[0][0] == pytest.approx(0.61, rel=1e-5)


def test_batch_score_propagates_model_error(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2]])

    class ErrorModel:
        def __call__(self, **kwargs):
            raise RuntimeError("boom")

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = ErrorModel()
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    with pytest.raises(RuntimeError, match="boom"):
        classifier.batch_score_SciX_categories(["doc"])


def test_batch_score_emits_classifier_timing_and_shape_metrics(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3], [4, 5, 6, 7, 8]])
    model = FakeModel([
        FakeTensor([[0.0, 1.0], [1.0, 0.0]]),
        FakeTensor([[0.5, -0.5], [0.25, -0.25]]),
    ])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = model
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    captured = []

    def fake_emit_event(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(module.perf_metrics, "emit_event", fake_emit_event)

    classifier = module.Classifier()
    classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        run_id="run-1",
        configured_record_batch_size=10,
        window_size=3,
        window_stride=2,
    )

    timing_names = {
        event["extra"]["name"]
        for event in captured
        if event["stage"] == "classifier_timing"
    }
    assert timing_names == {
        "tokenizer_call",
        "input_splitting",
        "special_token_padding",
        "model_forward",
        "post_sigmoid_aggregation",
    }

    shape_events = {
        event["extra"]["name"]: event["duration_ms"]
        for event in captured
        if event["stage"] == "classifier_batch_shape"
    }
    assert shape_events["configured_record_batch_size"] == 10.0
    assert shape_events["total_chunks"] == 3.0
    assert shape_events["effective_chunk_batch_size"] == 3.0
    assert shape_events["max_chunks_per_record"] == 2.0
    assert shape_events["mean_chunks_per_record"] == 1.5
    assert shape_events["max_tokenized_length"] == 5.0
    assert shape_events["padded_tensor_rows"] == 2.0
    assert shape_events["padded_tensor_cols"] == 5.0
