import importlib
import math
import sys
import types

import pytest


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor(self.values[item])
        return self.values[item]

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


class AttentionAwareFakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"].values
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.values
        logits = []
        for row_index, row in enumerate(input_ids):
            if attention_mask is None:
                active_tokens = list(row)
            else:
                active_tokens = [
                    token_id
                    for token_id, is_active in zip(row, attention_mask[row_index])
                    if is_active
                ]
            token_sum = sum(active_tokens)
            token_count = len(active_tokens)
            logits.append([float(token_sum) / 10.0, -float(token_count)])
        return FakeOutput(FakeTensor(logits))


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def assert_nested_approx(actual, expected, rel=1e-6):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row, rel=rel)


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
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [4, 5]])
    model = FakeModel([
        FakeTensor([
            [0.8472978604, -0.8472978604],
            [-1.3862943611, 2.1972245773],
            [1.3862943611, -2.1972245773],
        ]),
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
        model_inference_batch_size=2,
    )
    assert tokenizer.calls[0]["add_special_tokens"] is False
    assert categories == [["Astronomy", "Heliophysics"], ["Astronomy"]]
    assert scores[0] == pytest.approx([0.8, 0.9], rel=1e-5)
    assert scores[1] == pytest.approx([0.7, 0.3], rel=1e-5)


def test_batch_score_with_mean_combiner(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5]])
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
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5], [0.25, -0.25]]),
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
    assert shape_events["model_inference_batch_size"] == 16.0
    assert shape_events["total_chunks"] == 3.0
    assert shape_events["effective_chunk_batch_size"] == 3.0
    assert shape_events["max_chunks_per_record"] == 2.0
    assert shape_events["mean_chunks_per_record"] == 1.5
    assert shape_events["max_tokenized_length"] == 5.0
    assert shape_events["padded_tensor_rows"] == 2.0
    assert shape_events["padded_tensor_cols"] == 5.0
    assert shape_events["micro_batch_count"] == 1.0
    assert shape_events["max_micro_batch_records"] == 2.0
    assert shape_events["mean_micro_batch_records"] == 2.0
    assert shape_events["max_micro_batch_rows"] == 3.0
    assert shape_events["mean_micro_batch_rows"] == 3.0
    assert shape_events["grouping_applied"] == 1.0
    assert shape_events["mean_grouped_record_width"] == 5.0
    assert shape_events["max_grouped_record_width"] == 5.0
    assert shape_events["mean_micro_batch_padded_cols"] == 5.0
    assert shape_events["max_micro_batch_padded_cols"] == 5.0
    assert shape_events["mean_micro_batch_width_span"] == 0.0
    assert shape_events["max_micro_batch_width_span"] == 0.0
    assert shape_events["attention_mask_applied"] == 1.0


def test_group_prepared_records_by_length_sorts_by_width_then_index(monkeypatch, base_fake_config, dummy_logger):
    class FakeWrapper:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel([FakeTensor([[0.0]])])
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    prepared = [
        {"original_index": 2, "split_input_ids_with_tokens": [[1, 2, 3, 4]], "chunk_count": 1, "max_row_width": 4, "total_row_tokens": 4},
        {"original_index": 0, "split_input_ids_with_tokens": [[1, 2]], "chunk_count": 1, "max_row_width": 2, "total_row_tokens": 2},
        {"original_index": 1, "split_input_ids_with_tokens": [[1, 2]], "chunk_count": 1, "max_row_width": 2, "total_row_tokens": 2},
        {"original_index": 3, "split_input_ids_with_tokens": [[1, 2, 3], [4]], "chunk_count": 2, "max_row_width": 3, "total_row_tokens": 4},
    ]
    grouped = classifier._group_prepared_records_by_length(prepared)
    assert [record["original_index"] for record in grouped] == [0, 1, 3, 2]


def test_batch_score_uses_internal_micro_batches(monkeypatch, base_fake_config, dummy_logger):
    base_fake_config["MODEL_INFERENCE_BATCH_SIZE"] = 2
    tokenizer = FakeTokenizer(token_ids=[[1], [2], [3], [4], [5]])
    model = FakeModel([
        FakeTensor([[0.1, 0.2], [0.3, 0.4]]),
        FakeTensor([[0.5, 0.6], [0.7, 0.8]]),
        FakeTensor([[0.9, 1.0]]),
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
    classifier.batch_score_SciX_categories(["d1", "d2", "d3", "d4", "d5"])
    assert len(model.calls) == 3


def test_batch_score_preserves_output_order_with_micro_batches(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1], [2], [3]])
    model = FakeModel([
        FakeTensor([[3.0, -3.0], [-3.0, 3.0]]),
        FakeTensor([[0.5, -0.5]]),
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
        ["doc1", "doc2", "doc3"],
        score_thresholds=[0.5, 0.5],
        model_inference_batch_size=2,
    )
    assert categories == [["Astronomy"], ["Heliophysics"], ["Astronomy"]]
    assert scores[0][0] > scores[0][1]
    assert scores[1][1] > scores[1][0]
    assert scores[2][0] > scores[2][1]


def test_batch_score_micro_batches_support_multichunk_records(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [7]])
    model = FakeModel([
        FakeTensor([[3.0, -3.0], [-3.0, 3.0], [0.5, -0.5]]),
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
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    assert categories == [["Astronomy", "Heliophysics"], ["Astronomy"]]
    assert scores[0] == pytest.approx([0.6224593312, 0.9525741268], rel=1e-5)
    assert scores[1][0] > scores[1][1]


def test_batch_score_pads_mixed_width_rows_within_micro_batch(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4], [7]])
    model = FakeModel([
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
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
    classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    input_rows = model.calls[0]["input_ids"].values
    widths = {len(row) for row in input_rows}
    assert widths == {5}


def test_batch_score_preserves_output_order_after_length_grouping(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [8], [9, 10, 11, 12]])
    model = FakeModel([
        FakeTensor([[-3.0, 3.0], [3.0, -3.0], [0.25, -0.25]]),
        FakeTensor([[0.5, -0.5], [0.75, -0.75]]),
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
        ["long-doc", "short-doc", "mid-doc"],
        score_thresholds=[0.5, 0.5],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    assert categories == [["Astronomy"], ["Heliophysics"], ["Astronomy"]]
    assert scores[0][0] > scores[0][1]
    assert scores[1][1] > scores[1][0]
    assert scores[2][0] > scores[2][1]


def test_batch_score_groups_mixed_width_records_into_lower_span_micro_batches(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [6], [7, 8, 9, 10], [11]])
    model = FakeModel([
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
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
    monkeypatch.setattr(module.perf_metrics, "emit_event", lambda **kwargs: captured.append(kwargs))
    classifier = module.Classifier()
    classifier.batch_score_SciX_categories(
        ["long-a", "short-a", "mid-a", "short-b"],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    shape_events = {
        event["extra"]["name"]: event["duration_ms"]
        for event in captured
        if event["stage"] == "classifier_batch_shape"
    }
    assert shape_events["mean_micro_batch_width_span"] < 2.5
    assert shape_events["max_micro_batch_width_span"] <= 2.0


def test_batch_score_length_grouping_supports_multichunk_records(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [7], [8, 9, 10, 11, 12]])
    model = FakeModel([
        FakeTensor([[0.5, -0.5], [3.0, -3.0], [0.25, -0.25]]),
        FakeTensor([[-3.0, 3.0], [-0.5, 0.5]]),
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
        ["long-a", "short", "long-b"],
        score_thresholds=[0.5, 0.5],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    assert categories == [["Astronomy"], ["Astronomy"], ["Heliophysics"]]
    assert scores[0][0] > scores[0][1]
    assert scores[1][0] > scores[1][1]
    assert scores[2][1] > scores[2][0]


def test_batch_score_micro_batch_forward_passes_attention_mask(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4], [7]])
    model = FakeModel([
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
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
    classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    assert "input_ids" in model.calls[0]
    assert "attention_mask" in model.calls[0]
    assert len(model.calls[0]["input_ids"].values) == len(model.calls[0]["attention_mask"].values)
    assert all(
        len(input_row) == len(mask_row)
        for input_row, mask_row in zip(model.calls[0]["input_ids"].values, model.calls[0]["attention_mask"].values)
    )


def test_attention_mask_zeroes_padded_positions_in_micro_batch(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4], [7]])
    model = FakeModel([
        FakeTensor([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]]),
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
    classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )
    input_rows = model.calls[0]["input_ids"].values
    mask_rows = model.calls[0]["attention_mask"].values
    for input_row, mask_row in zip(input_rows, mask_rows):
        for token_id, mask_value in zip(input_row, mask_row):
            expected = 0 if token_id == tokenizer.pad_token_id else 1
            assert mask_value == expected


def test_batched_scores_match_unbatched_scores_for_mixed_width_records(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4], [7]])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = AttentionAwareFakeModel()
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    batched_classifier = module.Classifier()
    batched_categories, batched_scores = batched_classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )

    unbatched_classifier = module.Classifier()
    unbatched_categories, unbatched_scores = unbatched_classifier.batch_score_SciX_categories(
        ["doc1", "doc2"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=1,
        window_size=3,
        window_stride=2,
    )
    assert_nested_approx(batched_scores, unbatched_scores, rel=1e-6)
    assert batched_categories == unbatched_categories


def test_length_grouped_batched_scores_match_unbatched_scores(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [8], [9, 10, 11, 12]])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = AttentionAwareFakeModel()
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    grouped_classifier = module.Classifier()
    grouped_categories, grouped_scores = grouped_classifier.batch_score_SciX_categories(
        ["long-doc", "short-doc", "mid-doc"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )

    unbatched_classifier = module.Classifier()
    unbatched_categories, unbatched_scores = unbatched_classifier.batch_score_SciX_categories(
        ["long-doc", "short-doc", "mid-doc"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=1,
        window_size=3,
        window_stride=2,
    )
    assert_nested_approx(grouped_scores, unbatched_scores, rel=1e-6)
    assert grouped_categories == unbatched_categories


def test_multichunk_batched_scores_match_unbatched_scores(monkeypatch, base_fake_config, dummy_logger):
    tokenizer = FakeTokenizer(token_ids=[[1, 2, 3, 4, 5], [7], [8, 9, 10, 11, 12]])

    class FakeWrapper:
        def __init__(self):
            self.tokenizer = tokenizer
            self.model = AttentionAwareFakeModel()
            self.labels = ["Astronomy", "Heliophysics"]
            self.id2label = {0: "Astronomy", 1: "Heliophysics"}
            self.label2id = {"Astronomy": 0, "Heliophysics": 1}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    batched_classifier = module.Classifier()
    batched_categories, batched_scores = batched_classifier.batch_score_SciX_categories(
        ["long-a", "short", "long-b"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=2,
        window_size=3,
        window_stride=2,
    )

    unbatched_classifier = module.Classifier()
    unbatched_categories, unbatched_scores = unbatched_classifier.batch_score_SciX_categories(
        ["long-a", "short", "long-b"],
        score_thresholds=[0.0, 0.0],
        model_inference_batch_size=1,
        window_size=3,
        window_stride=2,
    )
    assert_nested_approx(batched_scores, unbatched_scores, rel=1e-6)
    assert batched_categories == unbatched_categories


def test_batch_score_model_inference_batch_size_falls_back_to_default(monkeypatch, base_fake_config, dummy_logger):
    class FakeWrapper:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.model = FakeModel([FakeTensor([[0.0]])])
            self.labels = ["Astronomy"]
            self.id2label = {0: "Astronomy"}
            self.label2id = {"Astronomy": 0}

    module = _import_classifier(monkeypatch, base_fake_config, dummy_logger, FakeWrapper)
    classifier = module.Classifier()
    module.config["MODEL_INFERENCE_BATCH_SIZE"] = 0
    assert classifier._resolve_model_inference_batch_size() == 16
    module.config["MODEL_INFERENCE_BATCH_SIZE"] = -1
    assert classifier._resolve_model_inference_batch_size() == 16
    module.config["MODEL_INFERENCE_BATCH_SIZE"] = "bad"
    assert classifier._resolve_model_inference_batch_size() == 16
    assert classifier._resolve_model_inference_batch_size(requested_size=8) == 8
