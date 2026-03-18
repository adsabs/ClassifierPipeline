import json
import importlib
from types import SimpleNamespace

import pytest

import ClassifierPipeline.perf_metrics as perf_metrics


def test_percentile_interpolation():
    values = [10, 20, 30, 40]
    assert perf_metrics.percentile(values, 50) == 25.0
    assert perf_metrics.percentile(values, 95) == 38.5


def test_aggregate_events_counts_and_latency():
    events = [
        {"ts": 1.0, "stage": "ingest_enqueue", "duration_ms": 5.0, "status": "ok", "extra": {"record_count": 2}},
        {"ts": 2.0, "stage": "classify", "duration_ms": 15.0, "status": "ok", "extra": {}},
        {"ts": 2.5, "stage": "task_timing", "duration_ms": 12.0, "status": "ok", "extra": {"name": "task_update_record"}},
        {"ts": 2.6, "stage": "app_timing", "duration_ms": 8.0, "status": "ok", "extra": {"name": "index_run"}},
        {"ts": 2.7, "stage": "classifier_timing", "duration_ms": 4.0, "status": "ok", "extra": {"name": "tokenizer_call"}},
        {"ts": 2.8, "stage": "classifier_batch_shape", "duration_ms": 12.0, "status": "ok", "extra": {"name": "total_chunks"}},
        {"ts": 3.0, "stage": "index", "duration_ms": 25.0, "status": "ok", "extra": {}},
        {"ts": 4.0, "stage": "forward", "duration_ms": 10.0, "status": "error", "extra": {}},
    ]

    summary = perf_metrics.aggregate_events(events, expected_records=2)

    assert summary["counts"]["records_submitted"] == 2
    assert summary["counts"]["records_indexed"] == 1
    assert summary["counts"]["records_forwarded"] == 1
    assert summary["counts"]["failures"] == 1
    assert summary["status"] == "incomplete"

    classify_stats = summary["latency_ms"]["classify"]
    assert classify_stats["count"] == 1
    assert classify_stats["p95"] == 15.0
    assert summary["task_timing_ms"]["task_update_record"]["p95"] == 12.0
    assert summary["app_timing_ms"]["index_run"]["p95"] == 8.0
    assert summary["classifier_timing_ms"]["tokenizer_call"]["p95"] == 4.0
    assert summary["classifier_batch_shapes"]["total_chunks"]["p95"] == 12.0


def test_emit_event_uses_registered_run_metrics_context(tmp_path, monkeypatch):
    importlib.reload(perf_metrics)
    context_dir = tmp_path / "context"
    events_path = tmp_path / "events.jsonl"
    config = {"PERF_METRICS_ENABLED": False, "PERF_METRICS_CONTEXT_DIR": str(context_dir)}
    monkeypatch.setenv("PERF_METRICS_CONTEXT_DIR", str(context_dir))
    monkeypatch.delenv("PERF_METRICS_PATH", raising=False)
    monkeypatch.delenv("PERF_METRICS_ENABLED", raising=False)

    perf_metrics.register_run_metrics_context(
        run_id=123,
        enabled=True,
        path=str(events_path),
        context_id="ctx-1",
        config=config,
        context_dir=str(context_dir),
    )
    perf_metrics.emit_event(
        stage="task_timing",
        run_id=123,
        context_id="ctx-1",
        record_id=None,
        duration_ms=12.5,
        extra={"name": "task_update_record"},
        config=config,
    )

    resolved = perf_metrics.resolve_run_metrics_context(123, config=config, context_id="ctx-1")
    payloads = perf_metrics.load_events(str(resolved["path"]), run_id=123, context_id="ctx-1")
    assert len(payloads) == 1
    assert payloads[0]["stage"] == "task_timing"
    assert payloads[0]["extra"]["name"] == "task_update_record"
    assert payloads[0]["context_id"] == "ctx-1"


def test_load_events_filters_by_context_id(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"stage": "classify", "run_id": "1", "context_id": "ctx-a"}),
                json.dumps({"stage": "classify", "run_id": "1", "context_id": "ctx-b"}),
                json.dumps({"stage": "classify", "run_id": "2", "context_id": "ctx-a"}),
            ]
        )
        + "\n"
    )

    payloads = perf_metrics.load_events(str(path), run_id=1, context_id="ctx-a")
    assert len(payloads) == 1
    assert payloads[0]["context_id"] == "ctx-a"


def test_aggregate_events_normalizes_batched_classify_latency():
    events = [
        {"ts": 1.0, "stage": "classify", "duration_ms": 40.0, "status": "ok", "extra": {"record_count": 4, "batch_mode": True}},
    ]

    summary = perf_metrics.aggregate_events(events)
    assert summary["latency_ms"]["classify"]["p95"] == 10.0


def test_aggregate_events_records_classify_batch_latency_and_sizes():
    events = [
        {"ts": 1.0, "stage": "classify", "duration_ms": 40.0, "status": "ok", "extra": {"record_count": 4, "batch_mode": True}},
    ]

    summary = perf_metrics.aggregate_events(events)
    assert summary["batch_latency_ms"]["classify"]["p95"] == 40.0
    assert summary["batch_sizes"]["classify"]["p95"] == 4.0


def test_aggregate_events_normalizes_batched_index_db_latency():
    events = [
        {"ts": 1.0, "stage": "index_db", "duration_ms": 100.0, "status": "ok", "extra": {"record_count": 10, "batch_mode": True}},
    ]
    summary = perf_metrics.aggregate_events(events)
    assert summary["latency_ms"]["index_db"]["p95"] == 10.0
    assert summary["batch_latency_ms"]["index_db"]["p95"] == 100.0
    assert summary["batch_sizes"]["index_db"]["p95"] == 10.0


def test_evaluate_gate_pass_and_fail():
    baseline = {
        "throughput": {"overall_records_per_minute": 100.0, "load_adjusted_records_per_minute": 105.0},
        "latency_ms": {"classify": {"p95": 100.0}, "index": {"p95": 50.0}},
        "system_load": {"summary": {"normalized_load_1m": {"mean": 1.05}, "memory_available_ratio": {"mean": 0.4}}},
    }
    candidate_pass = {
        "throughput": {"overall_records_per_minute": 120.0, "load_adjusted_records_per_minute": 126.0},
        "latency_ms": {"classify": {"p95": 105.0}, "index": {"p95": 52.0}},
        "system_load": {"summary": {"normalized_load_1m": {"mean": 1.1}, "memory_available_ratio": {"mean": 0.35}}},
    }
    candidate_fail = {
        "throughput": {"overall_records_per_minute": 103.0, "load_adjusted_records_per_minute": 103.0},
        "latency_ms": {"classify": {"p95": 130.0}, "index": {"p95": 55.0}},
        "system_load": {"summary": {"normalized_load_1m": {"mean": 0.9}, "memory_available_ratio": {"mean": 0.32}}},
    }

    gate_pass = perf_metrics.evaluate_gate(
        candidate=candidate_pass,
        baseline=baseline,
        min_throughput_improvement_pct=5.0,
        p95_regression_limit_pct=10.0,
    )
    assert gate_pass["pass"] is True
    assert gate_pass["load_adjusted_throughput_delta_pct"] == pytest.approx(20.0)
    assert gate_pass["system_load_delta"]["candidate_mean_normalized_load_1m"] == pytest.approx(1.1)

    gate_fail = perf_metrics.evaluate_gate(
        candidate=candidate_fail,
        baseline=baseline,
        min_throughput_improvement_pct=5.0,
        p95_regression_limit_pct=10.0,
    )
    assert gate_fail["pass"] is False
    assert gate_fail["reasons"]


def test_read_linux_meminfo(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       1024 kB\nMemAvailable:    256 kB\n")

    result = perf_metrics._read_linux_meminfo(str(meminfo))
    assert result["memory_total_bytes"] == 1024 * 1024
    assert result["memory_available_bytes"] == 256 * 1024
    assert result["memory_available_ratio"] == 0.25


def test_read_macos_memory(monkeypatch):
    responses = {
        ("sysctl", "-n", "hw.memsize"): SimpleNamespace(returncode=0, stdout="4096\n"),
        ("vm_stat",): SimpleNamespace(
            returncode=0,
            stdout="Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 1.\nPages inactive: 2.\nPages speculative: 3.\n",
        ),
    }

    def fake_run(args, capture_output, text, check):
        return responses[tuple(args)]

    monkeypatch.setattr(perf_metrics.subprocess, "run", fake_run)
    result = perf_metrics._read_macos_memory()
    assert result["memory_total_bytes"] == 4096
    assert result["memory_available_bytes"] == 6 * 4096
    assert result["memory_available_ratio"] == 6.0


def test_aggregate_system_samples_and_adjustment():
    samples = [
        {
            "platform": "linux",
            "cpu_count": 4,
            "memory_probe": "linux_meminfo",
            "normalized_load_1m": 0.5,
            "normalized_load_5m": 0.4,
            "normalized_load_15m": 0.3,
            "memory_available_ratio": 0.25,
        },
        {
            "platform": "linux",
            "cpu_count": 4,
            "memory_probe": "linux_meminfo",
            "normalized_load_1m": 1.5,
            "normalized_load_5m": 1.0,
            "normalized_load_15m": 0.8,
            "memory_available_ratio": 0.5,
        },
    ]
    system_load = perf_metrics.aggregate_system_samples(samples, enabled=True, sample_interval_s=2.0)
    assert system_load["collection"]["sample_count"] == 2
    assert system_load["summary"]["normalized_load_1m"]["mean"] == pytest.approx(1.0)
    assert system_load["summary"]["memory_available_ratio"]["mean"] == pytest.approx(0.375)

    summary = {"throughput": {"overall_records_per_minute": 200.0}, "system_load": system_load}
    perf_metrics.apply_system_load_adjustment(summary)
    assert summary["throughput"]["host_load_adjustment_factor"] == pytest.approx(1.0)
    assert summary["throughput"]["load_adjusted_records_per_minute"] == pytest.approx(200.0)

    summary["system_load"]["summary"]["normalized_load_1m"]["mean"] = 1.5
    perf_metrics.apply_system_load_adjustment(summary)
    assert summary["throughput"]["host_load_adjustment_factor"] == pytest.approx(1.5)
    assert summary["throughput"]["load_adjusted_records_per_minute"] == pytest.approx(300.0)


def test_render_markdown_includes_system_load(tmp_path):
    output_path = tmp_path / "report.md"
    summary = {
        "status": "complete",
        "throughput": {
            "overall_records_per_minute": 100.0,
            "host_load_adjustment_factor": 1.2,
            "load_adjusted_records_per_minute": 120.0,
        },
        "duration_s": {"wall_clock": 10.0},
        "counts": {"records_submitted": 10, "records_indexed": 10, "records_forwarded": 10, "failures": 0},
        "latency_ms": {},
        "task_timing_ms": {"task_update_record": {"count": 1, "p50": 11.0, "p95": 11.0, "p99": 11.0, "mean": 11.0, "min": 11.0, "max": 11.0}},
        "app_timing_ms": {"index_run": {"count": 1, "p50": 7.0, "p95": 7.0, "p99": 7.0, "mean": 7.0, "min": 7.0, "max": 7.0}},
        "classifier_timing_ms": {"tokenizer_call": {"count": 1, "p50": 3.0, "p95": 3.0, "p99": 3.0, "mean": 3.0, "min": 3.0, "max": 3.0}},
        "classifier_batch_shapes": {"total_chunks": {"count": 1, "p50": 12.0, "p95": 12.0, "p99": 12.0, "mean": 12.0, "min": 12.0, "max": 12.0}},
        "batch_latency_ms": {"classify": {"mean": 25.0, "p95": 40.0}, "index_db": {"mean": 35.0, "p95": 50.0}},
        "batch_sizes": {"classify": {"mean": 100.0, "p95": 120.0}, "index_db": {"mean": 90.0, "p95": 110.0}},
        "system_load": {
            "collection": {"sample_count": 3, "sample_interval_s": 1.0, "platform": "linux", "memory_probe": "linux_meminfo"},
            "summary": {
                "normalized_load_1m": {"mean": 1.1, "p95": 1.4},
                "normalized_load_5m": {"mean": 0.9},
                "memory_available_ratio": {"mean": 0.3, "min": 0.2},
            },
        },
        "gate": {"pass": True, "throughput_delta_pct": 5.0, "load_adjusted_throughput_delta_pct": 10.0, "reasons": []},
    }

    perf_metrics.render_markdown(summary, str(output_path))
    content = output_path.read_text()
    assert "## System Load" in content
    assert "Load-Adjusted Throughput" in content
    assert "## Task Timing (ms)" in content
    assert "## App Function Timing (ms)" in content
    assert "## Classifier Timing (ms)" in content
    assert "## Classifier Batch Shapes" in content
    assert "## Batch Metrics" in content
