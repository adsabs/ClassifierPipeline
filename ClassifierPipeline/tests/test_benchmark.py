import json
import sys
import types
from types import SimpleNamespace

from ClassifierPipeline import benchmark


def test_parse_csv_list_int_and_str():
    assert benchmark._parse_csv_list("50,100,250", int) == [50, 100, 250]
    assert benchmark._parse_csv_list("real,fake", str) == ["real", "fake"]


def test_cmd_compare_outputs_gate(tmp_path, capsys):
    baseline = {
        "throughput": {"overall_records_per_minute": 100.0, "load_adjusted_records_per_minute": 100.0},
        "latency_ms": {"classify": {"p95": 100.0}},
    }
    candidate = {
        "throughput": {"overall_records_per_minute": 120.0, "load_adjusted_records_per_minute": 120.0},
        "latency_ms": {"classify": {"p95": 105.0}},
    }

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "compare.json"

    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))

    args = SimpleNamespace(
        baseline=str(baseline_path),
        candidate=str(candidate_path),
        output=str(output_path),
        p95_regression_limit_pct=10.0,
        min_throughput_improvement_pct=5.0,
    )

    rc = benchmark.cmd_compare(args)
    assert rc == 0
    assert output_path.exists()

    out = capsys.readouterr().out
    assert '"pass": true' in out.lower()


def test_run_case_includes_system_load(monkeypatch):
    class DummyTask:
        def delay(self, message):
            return None

    class DummyApp:
        def index_run(self):
            return 123

    dummy_tasks = types.ModuleType("ClassifierPipeline.tasks")
    dummy_tasks.app = DummyApp()
    dummy_tasks.task_update_record = DummyTask()

    dummy_utils = types.ModuleType("ClassifierPipeline.utilities")
    dummy_utils.list_to_ClassifyRequestRecordList = lambda payload: payload

    monkeypatch.setattr(benchmark, "_read_dataset", lambda path: [{"bibcode": "B", "title": "T", "abstract": "A"}])
    monkeypatch.setattr(benchmark, "_poll_run_completion", lambda **kwargs: {"complete": True, "records_indexed": 1, "elapsed_s": 1.0})
    monkeypatch.setattr(benchmark.perf_metrics, "load_events", lambda path, run_id=None: [])
    monkeypatch.setattr(
        benchmark.perf_metrics,
        "aggregate_events",
        lambda events, started_at=None, ended_at=None, expected_records=None: {
            "counts": {"records_submitted": 0, "records_indexed": 0, "records_forwarded": 0, "failures": 0},
            "throughput": {},
            "latency_ms": {},
            "duration_s": {"wall_clock": max(1.0, float(ended_at) - float(started_at))},
            "errors": {},
            "status": "complete",
        },
    )
    monkeypatch.setattr(
        benchmark.perf_metrics,
        "collect_system_sample",
        lambda: {
            "ts": 1.0,
            "platform": "linux",
            "cpu_count": 4,
            "memory_probe": "linux_meminfo",
            "normalized_load_1m": 1.5,
            "normalized_load_5m": 1.0,
            "normalized_load_15m": 0.8,
            "memory_available_ratio": 0.25,
        },
    )

    import ClassifierPipeline
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.tasks", dummy_tasks)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.utilities", dummy_utils)
    monkeypatch.setattr(ClassifierPipeline, "tasks", dummy_tasks, raising=False)
    monkeypatch.setattr(ClassifierPipeline, "utilities", dummy_utils, raising=False)

    summary = benchmark._run_case(
        records_path="dataset.csv",
        mode="fake",
        batch_size=1,
        timeout_s=1,
        poll_interval_s=0.1,
        operation_step="classify_verify",
        events_path="events.jsonl",
        system_sample_interval_s=0.01,
        system_load_enabled=True,
    )

    assert "system_load" in summary
    assert summary["system_load"]["collection"]["enabled"] is True
    assert summary["throughput"]["load_adjusted_records_per_minute"] >= summary["throughput"]["overall_records_per_minute"]


def test_run_case_disable_system_load(monkeypatch):
    class DummyTask:
        def delay(self, message):
            return None

    class DummyApp:
        def index_run(self):
            return 123

    dummy_tasks = types.ModuleType("ClassifierPipeline.tasks")
    dummy_tasks.app = DummyApp()
    dummy_tasks.task_update_record = DummyTask()

    dummy_utils = types.ModuleType("ClassifierPipeline.utilities")
    dummy_utils.list_to_ClassifyRequestRecordList = lambda payload: payload

    monkeypatch.setattr(benchmark, "_read_dataset", lambda path: [{"bibcode": "B", "title": "T", "abstract": "A"}])
    monkeypatch.setattr(benchmark, "_poll_run_completion", lambda **kwargs: {"complete": True, "records_indexed": 1, "elapsed_s": 1.0})
    monkeypatch.setattr(benchmark.perf_metrics, "load_events", lambda path, run_id=None: [])
    monkeypatch.setattr(
        benchmark.perf_metrics,
        "aggregate_events",
        lambda events, started_at=None, ended_at=None, expected_records=None: {
            "counts": {"records_submitted": 0, "records_indexed": 0, "records_forwarded": 0, "failures": 0},
            "throughput": {},
            "latency_ms": {},
            "duration_s": {"wall_clock": max(1.0, float(ended_at) - float(started_at))},
            "errors": {},
            "status": "complete",
        },
    )

    import ClassifierPipeline
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.tasks", dummy_tasks)
    monkeypatch.setitem(sys.modules, "ClassifierPipeline.utilities", dummy_utils)
    monkeypatch.setattr(ClassifierPipeline, "tasks", dummy_tasks, raising=False)
    monkeypatch.setattr(ClassifierPipeline, "utilities", dummy_utils, raising=False)

    summary = benchmark._run_case(
        records_path="dataset.csv",
        mode="fake",
        batch_size=1,
        timeout_s=1,
        poll_interval_s=0.1,
        operation_step="classify_verify",
        events_path="events.jsonl",
        system_load_enabled=False,
    )

    assert summary["system_load"]["collection"]["enabled"] is False
    assert summary["throughput"]["host_load_adjustment_factor"] == 1.0
