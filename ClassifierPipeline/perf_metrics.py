"""Performance metrics helpers for benchmark/profiling workflows.

This module is intentionally stdlib-only and file-backed so it can be used from
multiple Celery worker processes without introducing extra infrastructure.
"""

import json
import os
import platform
import re
import subprocess
import time
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "active"}


def metrics_enabled(config: Optional[dict] = None) -> bool:
    env_value = os.getenv("PERF_METRICS_ENABLED")
    if env_value is not None:
        return _as_bool(env_value)
    if config is None:
        return False
    return _as_bool(config.get("PERF_METRICS_ENABLED", False))


def metrics_path(config: Optional[dict] = None) -> Optional[str]:
    env_path = os.getenv("PERF_METRICS_PATH")
    if env_path:
        return env_path
    if config is None:
        return None
    config_path = config.get("PERF_METRICS_PATH")
    if config_path:
        return config_path
    output_dir = config.get("PERF_METRICS_OUTPUT_DIR")
    if output_dir:
        return os.path.join(output_dir, "perf_events.jsonl")
    return None


def metrics_context_dir(config: Optional[dict] = None) -> Optional[str]:
    env_dir = os.getenv("PERF_METRICS_CONTEXT_DIR")
    if env_dir:
        return env_dir
    if config is not None:
        config_dir = config.get("PERF_METRICS_CONTEXT_DIR")
        if config_dir:
            return config_dir
    base_path = metrics_path(config=config)
    if base_path:
        return os.path.join(os.path.dirname(base_path), "perf_run_context")
    return None


def _run_context_path(
    run_id: Any,
    config: Optional[dict] = None,
    context_dir: Optional[str] = None,
    context_id: Optional[str] = None,
) -> Optional[str]:
    if run_id is None:
        return None
    directory = context_dir or metrics_context_dir(config=config)
    if not directory:
        return None
    if context_id:
        return os.path.join(directory, f"run_{run_id}_{context_id}.json")
    return os.path.join(directory, f"run_{run_id}.json")


def register_run_metrics_context(
    run_id: Any,
    enabled: bool,
    path: Optional[str],
    context_id: Optional[str] = None,
    config: Optional[dict] = None,
    context_dir: Optional[str] = None,
) -> None:
    try:
        targets = []
        target = _run_context_path(run_id, config=config, context_dir=context_dir, context_id=context_id)
        if target:
            targets.append(target)
        generic_target = _run_context_path(run_id, config=config, context_dir=context_dir)
        if generic_target and generic_target not in targets:
            targets.append(generic_target)
        if not targets:
            return
        payload = {
            "enabled": bool(enabled),
            "path": path,
            "context_id": context_id,
            "updated_at": time.time(),
        }
        for current_target in targets:
            directory = os.path.dirname(current_target)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(current_target, "w") as handle:
                json.dump(payload, handle, sort_keys=True)
    except Exception:
        return


def resolve_run_metrics_context(
    run_id: Any,
    config: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> Dict[str, Any]:
    target = _run_context_path(run_id, config=config, context_id=context_id)
    if not target or not os.path.exists(target):
        if context_id is not None:
            target = _run_context_path(run_id, config=config)
    if not target or not os.path.exists(target):
        return {"enabled": None, "path": None}
    try:
        with open(target, "r") as handle:
            payload = json.load(handle)
        return {
            "enabled": payload.get("enabled"),
            "path": payload.get("path"),
            "context_id": payload.get("context_id"),
        }
    except Exception:
        return {"enabled": None, "path": None, "context_id": None}


def emit_event(
    stage: str,
    run_id: Optional[Any] = None,
    context_id: Optional[str] = None,
    record_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    status: str = "ok",
    extra: Optional[dict] = None,
    config: Optional[dict] = None,
    path: Optional[str] = None,
) -> None:
    """Append a single event to the metrics event stream.

    This function is intentionally best-effort and will never raise.
    """
    try:
        run_context = (
            resolve_run_metrics_context(run_id, config=config, context_id=context_id)
            if run_id is not None
            else {"enabled": None, "path": None, "context_id": None}
        )
        resolved_context_id = context_id or run_context.get("context_id")
        enabled = metrics_enabled(config=config)
        if run_context.get("enabled") is not None:
            enabled = bool(run_context.get("enabled"))
        if not enabled:
            return

        target_path = path or run_context.get("path") or metrics_path(config=config)
        if not target_path:
            return

        payload = {
            "ts": time.time(),
            "stage": stage,
            "run_id": str(run_id) if run_id is not None else None,
            "context_id": resolved_context_id,
            "record_id": record_id,
            "duration_ms": float(duration_ms) if duration_ms is not None else None,
            "status": status,
            "extra": extra or {},
        }

        directory = os.path.dirname(target_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(target_path, "a") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
    except Exception:
        # Metrics emission must never break the pipeline.
        return


@contextmanager
def timed_stage(
    stage: str,
    run_id: Optional[Any] = None,
    context_id: Optional[str] = None,
    record_id: Optional[str] = None,
    status: str = "ok",
    extra: Optional[dict] = None,
    config: Optional[dict] = None,
    path: Optional[str] = None,
):
    start = time.perf_counter()
    outcome = status
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        emit_event(
            stage=stage,
            run_id=run_id,
            context_id=context_id,
            record_id=record_id,
            duration_ms=duration_ms,
            status=outcome,
            extra=extra,
            config=config,
            path=path,
        )


@contextmanager
def timed_profile(
    category: str,
    name: str,
    run_id: Optional[Any] = None,
    context_id: Optional[str] = None,
    record_id: Optional[str] = None,
    status: str = "ok",
    extra: Optional[dict] = None,
    config: Optional[dict] = None,
    path: Optional[str] = None,
):
    payload_extra = {"name": name}
    if extra:
        payload_extra.update(extra)
    start = time.perf_counter()
    outcome = status
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        emit_event(
            stage=category,
            run_id=run_id,
            context_id=context_id,
            record_id=record_id,
            duration_ms=(time.perf_counter() - start) * 1000.0,
            status=outcome,
            extra=payload_extra,
            config=config,
            path=path,
        )


def profiled_function(
    category: str,
    name: Optional[str] = None,
    run_id_getter=None,
    context_id_getter=None,
    record_id_getter=None,
    extra_getter=None,
    config_getter=None,
):
    def decorator(func):
        profile_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            run_id = run_id_getter(*args, **kwargs) if run_id_getter else None
            context_id = context_id_getter(*args, **kwargs) if context_id_getter else None
            record_id = record_id_getter(*args, **kwargs) if record_id_getter else None
            extra = extra_getter(*args, **kwargs) if extra_getter else None
            config = config_getter(*args, **kwargs) if config_getter else None
            with timed_profile(
                category=category,
                name=profile_name,
                run_id=run_id,
                context_id=context_id,
                record_id=record_id,
                extra=extra,
                config=config,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def load_events(path: str, run_id: Optional[Any] = None, context_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    run_id_str = str(run_id) if run_id is not None else None
    context_id_str = str(context_id) if context_id is not None else None
    output: List[Dict[str, Any]] = []

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if run_id_str is not None and str(payload.get("run_id")) != run_id_str:
                continue
            if context_id_str is not None and str(payload.get("context_id")) != context_id_str:
                continue
            output.append(payload)

    return output


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    data = sorted(float(v) for v in values)
    if not data:
        return None

    if pct <= 0:
        return data[0]
    if pct >= 100:
        return data[-1]

    idx = (len(data) - 1) * (pct / 100.0)
    lower = int(idx)
    upper = min(lower + 1, len(data) - 1)
    weight = idx - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _numeric_stats(values: List[float], include_p99: bool = True) -> Dict[str, Any]:
    if not values:
        output = {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
        }
        if include_p99:
            output["p99"] = None
        return output

    total = sum(values)
    output = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": total / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }
    if include_p99:
        output["p99"] = percentile(values, 99)
    return output


def _duration_stats(values: List[float]) -> Dict[str, Any]:
    return _numeric_stats(values, include_p99=True)


def _read_linux_meminfo(path: str = "/proc/meminfo") -> Optional[Dict[str, float]]:
    try:
        values: Dict[str, int] = {}
        with open(path, "r") as handle:
            for line in handle:
                parts = line.split(":")
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                match = re.search(r"(\d+)", parts[1])
                if match:
                    values[key] = int(match.group(1))

        total_kib = values.get("MemTotal")
        available_kib = values.get("MemAvailable")
        if total_kib is None or available_kib is None or total_kib <= 0:
            return None

        total_bytes = int(total_kib) * 1024
        available_bytes = int(available_kib) * 1024
        return {
            "memory_total_bytes": total_bytes,
            "memory_available_bytes": available_bytes,
            "memory_available_ratio": float(available_bytes) / float(total_bytes),
            "memory_probe": "linux_meminfo",
        }
    except Exception:
        return None


def _read_macos_memory() -> Optional[Dict[str, float]]:
    try:
        total_proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
        )
        if total_proc.returncode != 0:
            return None
        total_bytes = int((total_proc.stdout or "").strip())
        if total_bytes <= 0:
            return None

        vm_proc = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=False,
        )
        if vm_proc.returncode != 0:
            return None

        page_size = 4096
        page_size_match = re.search(r"page size of (\d+) bytes", vm_proc.stdout or "")
        if page_size_match:
            page_size = int(page_size_match.group(1))

        pages: Dict[str, int] = {}
        for line in (vm_proc.stdout or "").splitlines():
            match = re.match(r"([^:]+):\s+(\d+)\.", line.strip())
            if match:
                pages[match.group(1)] = int(match.group(2))

        available_pages = (
            pages.get("Pages free", 0)
            + pages.get("Pages inactive", 0)
            + pages.get("Pages speculative", 0)
        )
        available_bytes = int(available_pages) * int(page_size)
        return {
            "memory_total_bytes": int(total_bytes),
            "memory_available_bytes": available_bytes,
            "memory_available_ratio": float(available_bytes) / float(total_bytes),
            "memory_probe": "macos_vm_stat",
        }
    except Exception:
        return None


def _host_memory_snapshot() -> Dict[str, Optional[float]]:
    system = platform.system().lower()
    if system == "linux":
        snapshot = _read_linux_meminfo()
        if snapshot:
            return snapshot
        return {
            "memory_total_bytes": None,
            "memory_available_bytes": None,
            "memory_available_ratio": None,
            "memory_probe": "linux_meminfo_unavailable",
        }
    if system == "darwin":
        snapshot = _read_macos_memory()
        if snapshot:
            return snapshot
        return {
            "memory_total_bytes": None,
            "memory_available_bytes": None,
            "memory_available_ratio": None,
            "memory_probe": "macos_vm_stat_unavailable",
        }
    return {
        "memory_total_bytes": None,
        "memory_available_bytes": None,
        "memory_available_ratio": None,
        "memory_probe": "unsupported",
    }


def _host_load_snapshot() -> Dict[str, Optional[float]]:
    cpu_count = os.cpu_count()
    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1 = load5 = load15 = None

    def _normalize(value: Optional[float]) -> Optional[float]:
        if value is None or not cpu_count:
            return None
        return float(value) / float(cpu_count)

    return {
        "platform": platform.system().lower(),
        "cpu_count": cpu_count,
        "loadavg_1m": float(load1) if load1 is not None else None,
        "loadavg_5m": float(load5) if load5 is not None else None,
        "loadavg_15m": float(load15) if load15 is not None else None,
        "normalized_load_1m": _normalize(load1),
        "normalized_load_5m": _normalize(load5),
        "normalized_load_15m": _normalize(load15),
    }


def collect_system_sample() -> Dict[str, Optional[float]]:
    sample: Dict[str, Optional[float]] = {"ts": time.time()}
    sample.update(_host_load_snapshot())
    sample.update(_host_memory_snapshot())
    return sample


def aggregate_system_samples(samples: List[Dict[str, Any]], enabled: bool = True, sample_interval_s: float = 1.0) -> Dict[str, Any]:
    platform_name = platform.system().lower()
    cpu_count = os.cpu_count()
    memory_probe = "unsupported"
    if samples:
        platform_name = str(samples[0].get("platform") or platform_name)
        cpu_count = samples[0].get("cpu_count")
        memory_probe = str(samples[0].get("memory_probe") or memory_probe)

    summary: Dict[str, Any] = {}
    for key in (
        "normalized_load_1m",
        "normalized_load_5m",
        "normalized_load_15m",
        "memory_available_ratio",
    ):
        values = [float(sample[key]) for sample in samples if sample.get(key) is not None]
        summary[key] = _numeric_stats(values, include_p99=False)

    return {
        "collection": {
            "enabled": bool(enabled),
            "sample_interval_s": float(sample_interval_s),
            "sample_count": len(samples),
            "platform": platform_name,
            "cpu_count": cpu_count,
            "memory_probe": memory_probe,
        },
        "samples": samples,
        "summary": summary,
    }


def apply_system_load_adjustment(summary: Dict[str, Any]) -> Dict[str, Any]:
    throughput = summary.setdefault("throughput", {})
    raw = throughput.get("overall_records_per_minute")
    mean_load = (
        (((summary.get("system_load", {}) or {}).get("summary", {}) or {}).get("normalized_load_1m", {}) or {}).get("mean")
    )
    factor = max(1.0, float(mean_load)) if mean_load is not None else 1.0
    throughput["host_load_adjustment_factor"] = factor
    throughput["load_adjusted_records_per_minute"] = (float(raw) * factor) if raw is not None else None
    return summary


def aggregate_events(
    events: List[Dict[str, Any]],
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    expected_records: Optional[int] = None,
) -> Dict[str, Any]:
    stages: Dict[str, List[float]] = {}
    batch_latencies: Dict[str, List[float]] = {}
    batch_sizes: Dict[str, List[float]] = {}
    stage_errors: Dict[str, int] = {}
    task_timings: Dict[str, List[float]] = {}
    app_timings: Dict[str, List[float]] = {}
    classifier_timings: Dict[str, List[float]] = {}
    classifier_batch_shapes: Dict[str, List[float]] = {}

    submitted = 0
    indexed = 0
    forwarded = 0
    failure_count = 0

    event_timestamps = [e.get("ts") for e in events if e.get("ts") is not None]

    for event in events:
        stage = event.get("stage", "unknown")
        status = event.get("status", "ok")
        duration = event.get("duration_ms")

        if status != "ok":
            failure_count += 1
            stage_errors[stage] = stage_errors.get(stage, 0) + 1

        extra = event.get("extra", {}) or {}
        if duration is not None:
            duration_value = float(duration)
            if stage == "task_timing":
                task_name = str(extra.get("name") or "unknown")
                task_timings.setdefault(task_name, []).append(duration_value)
            elif stage == "app_timing":
                function_name = str(extra.get("name") or "unknown")
                app_timings.setdefault(function_name, []).append(duration_value)
            elif stage == "classifier_timing":
                timing_name = str(extra.get("name") or "unknown")
                classifier_timings.setdefault(timing_name, []).append(duration_value)
            elif stage == "classifier_batch_shape":
                shape_name = str(extra.get("name") or "unknown")
                classifier_batch_shapes.setdefault(shape_name, []).append(duration_value)
            elif stage in {"classify", "index_db"}:
                record_count = int(extra.get("record_count", 0) or 0)
                normalized_duration = duration_value / record_count if record_count > 0 else duration_value
                stages.setdefault(stage, []).append(normalized_duration)
                if extra.get("batch_mode") or record_count > 1:
                    batch_latencies.setdefault(stage, []).append(duration_value)
                    if record_count > 0:
                        batch_sizes.setdefault(stage, []).append(float(record_count))
            else:
                stages.setdefault(stage, []).append(duration_value)

        if stage == "ingest_enqueue":
            submitted += int(extra.get("record_count", 1))
        elif stage == "index":
            indexed += 1
        elif stage in {"forward", "forward_message"}:
            forwarded += 1

    if started_at is None:
        started_at = min(event_timestamps) if event_timestamps else None
    if ended_at is None:
        ended_at = max(event_timestamps) if event_timestamps else None

    wall_duration_s: Optional[float]
    if started_at is None or ended_at is None:
        wall_duration_s = None
    else:
        wall_duration_s = max(0.0, float(ended_at) - float(started_at))

    throughput = None
    if wall_duration_s and wall_duration_s > 0:
        throughput = (indexed / wall_duration_s) * 60.0

    latency_ms = {stage: _duration_stats(values) for stage, values in stages.items()}
    batch_latency_ms = {stage: _duration_stats(values) for stage, values in batch_latencies.items()}
    batch_size_stats = {stage: _numeric_stats(values, include_p99=True) for stage, values in batch_sizes.items()}

    status = "complete"
    backlog = None
    if expected_records is not None:
        backlog = max(0, int(expected_records) - int(indexed))
        if backlog > 0:
            status = "incomplete"

    return {
        "counts": {
            "records_submitted": submitted,
            "records_indexed": indexed,
            "records_forwarded": forwarded,
            "failures": failure_count,
            "backlog": backlog,
        },
        "throughput": {
            "overall_records_per_minute": throughput,
        },
        "latency_ms": latency_ms,
        "task_timing_ms": {name: _duration_stats(values) for name, values in task_timings.items()},
        "app_timing_ms": {name: _duration_stats(values) for name, values in app_timings.items()},
        "classifier_timing_ms": {name: _duration_stats(values) for name, values in classifier_timings.items()},
        "classifier_batch_shapes": {name: _numeric_stats(values, include_p99=True) for name, values in classifier_batch_shapes.items()},
        "batch_latency_ms": batch_latency_ms,
        "batch_sizes": batch_size_stats,
        "duration_s": {
            "wall_clock": wall_duration_s,
        },
        "errors": {
            "by_stage": stage_errors,
        },
        "status": status,
    }


def evaluate_gate(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
    min_throughput_improvement_pct: float,
    p95_regression_limit_pct: float,
) -> Dict[str, Any]:
    cand_t = (candidate.get("throughput", {}) or {}).get("overall_records_per_minute")
    base_t = (baseline.get("throughput", {}) or {}).get("overall_records_per_minute")
    cand_adj_t = (candidate.get("throughput", {}) or {}).get("load_adjusted_records_per_minute")
    base_adj_t = (baseline.get("throughput", {}) or {}).get("load_adjusted_records_per_minute")

    throughput_delta_pct = None
    load_adjusted_throughput_delta_pct = None
    throughput_pass = False
    if cand_t is not None and base_t not in (None, 0):
        throughput_delta_pct = ((cand_t - base_t) / base_t) * 100.0
        throughput_pass = throughput_delta_pct >= min_throughput_improvement_pct
    if cand_adj_t is not None and base_adj_t not in (None, 0):
        load_adjusted_throughput_delta_pct = ((cand_adj_t - base_adj_t) / base_adj_t) * 100.0

    p95_deltas: Dict[str, Optional[float]] = {}
    p95_pass = True

    cand_lat = candidate.get("latency_ms", {}) or {}
    base_lat = baseline.get("latency_ms", {}) or {}

    common_stages = sorted(set(cand_lat.keys()) & set(base_lat.keys()))
    for stage in common_stages:
        cand_p95 = (cand_lat.get(stage) or {}).get("p95")
        base_p95 = (base_lat.get(stage) or {}).get("p95")

        if cand_p95 is None or base_p95 in (None, 0):
            p95_deltas[stage] = None
            continue

        delta_pct = ((cand_p95 - base_p95) / base_p95) * 100.0
        p95_deltas[stage] = delta_pct
        if delta_pct > p95_regression_limit_pct:
            p95_pass = False

    gate_pass = throughput_pass and p95_pass
    reasons: List[str] = []
    if not throughput_pass:
        reasons.append("throughput improvement below threshold")
    if not p95_pass:
        reasons.append("p95 latency regression exceeded threshold")

    base_sys = (baseline.get("system_load", {}) or {}).get("summary", {}) or {}
    cand_sys = (candidate.get("system_load", {}) or {}).get("summary", {}) or {}

    return {
        "pass": gate_pass,
        "throughput_delta_pct": throughput_delta_pct,
        "load_adjusted_throughput_delta_pct": load_adjusted_throughput_delta_pct,
        "p95_delta_pct_by_stage": p95_deltas,
        "system_load_delta": {
            "baseline_mean_normalized_load_1m": ((base_sys.get("normalized_load_1m") or {}).get("mean")),
            "candidate_mean_normalized_load_1m": ((cand_sys.get("normalized_load_1m") or {}).get("mean")),
            "baseline_mean_memory_available_ratio": ((base_sys.get("memory_available_ratio") or {}).get("mean")),
            "candidate_mean_memory_available_ratio": ((cand_sys.get("memory_available_ratio") or {}).get("mean")),
        },
        "reasons": reasons,
        "thresholds": {
            "min_throughput_improvement_pct": min_throughput_improvement_pct,
            "p95_regression_limit_pct": p95_regression_limit_pct,
        },
    }


def write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _fmt(value: Optional[float], places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def render_markdown(summary: Dict[str, Any], output_path: str) -> None:
    latency = summary.get("latency_ms", {}) or {}
    task_timing = summary.get("task_timing_ms", {}) or {}
    app_timing = summary.get("app_timing_ms", {}) or {}
    classifier_timing = summary.get("classifier_timing_ms", {}) or {}
    classifier_batch_shapes = summary.get("classifier_batch_shapes", {}) or {}
    batch_latency = summary.get("batch_latency_ms", {}) or {}
    batch_sizes = summary.get("batch_sizes", {}) or {}
    counts = summary.get("counts", {}) or {}
    throughput_info = summary.get("throughput", {}) or {}
    throughput = throughput_info.get("overall_records_per_minute")
    load_adjusted_throughput = throughput_info.get("load_adjusted_records_per_minute")
    load_adjustment_factor = throughput_info.get("host_load_adjustment_factor")
    duration = (summary.get("duration_s", {}) or {}).get("wall_clock")
    system_load = summary.get("system_load", {}) or {}
    system_load_collection = system_load.get("collection", {}) or {}
    system_load_summary = system_load.get("summary", {}) or {}
    gate = summary.get("gate", {}) or {}
    runtime_metadata = summary.get("runtime_metadata", {}) or {}

    lines = [
        "# Throughput Benchmark Report",
        "",
        "## Run Config",
        "",
    ]

    run_metadata = summary.get("run_metadata", {}) or {}
    for key in sorted(run_metadata.keys()):
        lines.append(f"- **{key}**: `{run_metadata[key]}`")

    if runtime_metadata:
        lines.extend([
            "",
            "## Runtime Metadata",
            "",
            f"- **device**: `{runtime_metadata.get('device', 'n/a')}`",
            f"- **torch_num_threads**: `{runtime_metadata.get('torch_num_threads', 'n/a')}`",
            f"- **torch_num_interop_threads**: `{runtime_metadata.get('torch_num_interop_threads', 'n/a')}`",
            f"- **tokenizer_parallelism**: `{runtime_metadata.get('tokenizer_parallelism', 'n/a')}`",
            f"- **omp_num_threads**: `{runtime_metadata.get('omp_num_threads', 'n/a')}`",
            f"- **mkl_num_threads**: `{runtime_metadata.get('mkl_num_threads', 'n/a')}`",
        ])

    lines.extend([
        "",
        "## Top-Line Results",
        "",
        f"- **Status**: `{summary.get('status', 'unknown')}`",
        f"- **Throughput**: `{_fmt(throughput)} records/min`",
        f"- **Load-Adjusted Throughput**: `{_fmt(load_adjusted_throughput)} records/min`",
        f"- **Host Load Adjustment Factor**: `{_fmt(load_adjustment_factor)}`",
        f"- **Wall Duration**: `{_fmt(duration)} s`",
        f"- **Submitted**: `{counts.get('records_submitted', 0)}`",
        f"- **Indexed**: `{counts.get('records_indexed', 0)}`",
        f"- **Forwarded**: `{counts.get('records_forwarded', 0)}`",
        f"- **Failures**: `{counts.get('failures', 0)}`",
        "",
        "## Stage Latency (ms)",
        "",
        "| Stage | Count | p50 | p95 | p99 | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    slowest_stage = None
    slowest_p95 = None

    for stage in sorted(latency.keys()):
        stats = latency[stage]
        p95 = stats.get("p95")
        if p95 is not None and (slowest_p95 is None or p95 > slowest_p95):
            slowest_stage = stage
            slowest_p95 = p95

        lines.append(
            "| {stage} | {count} | {p50} | {p95} | {p99} | {mean} | {min_v} | {max_v} |".format(
                stage=stage,
                count=stats.get("count", 0),
                p50=_fmt(stats.get("p50")),
                p95=_fmt(stats.get("p95")),
                p99=_fmt(stats.get("p99")),
                mean=_fmt(stats.get("mean")),
                min_v=_fmt(stats.get("min")),
                max_v=_fmt(stats.get("max")),
            )
        )

    lines.extend([
        "",
        "## Bottleneck",
        "",
        f"- Highest p95 stage: `{slowest_stage or 'n/a'}` ({_fmt(slowest_p95)} ms)",
        "",
    ])

    if task_timing:
        lines.extend([
            "## Task Timing (ms)",
            "",
            "| Task | Count | p50 | p95 | p99 | Mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in sorted(task_timing.keys()):
            stats = task_timing[name]
            lines.append(
                "| {name} | {count} | {p50} | {p95} | {p99} | {mean} | {min_v} | {max_v} |".format(
                    name=name,
                    count=stats.get("count", 0),
                    p50=_fmt(stats.get("p50")),
                    p95=_fmt(stats.get("p95")),
                    p99=_fmt(stats.get("p99")),
                    mean=_fmt(stats.get("mean")),
                    min_v=_fmt(stats.get("min")),
                    max_v=_fmt(stats.get("max")),
                )
            )
        lines.append("")

    if app_timing:
        lines.extend([
            "## App Function Timing (ms)",
            "",
            "| Function | Count | p50 | p95 | p99 | Mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in sorted(app_timing.keys()):
            stats = app_timing[name]
            lines.append(
                "| {name} | {count} | {p50} | {p95} | {p99} | {mean} | {min_v} | {max_v} |".format(
                    name=name,
                    count=stats.get("count", 0),
                    p50=_fmt(stats.get("p50")),
                    p95=_fmt(stats.get("p95")),
                    p99=_fmt(stats.get("p99")),
                    mean=_fmt(stats.get("mean")),
                    min_v=_fmt(stats.get("min")),
                    max_v=_fmt(stats.get("max")),
                )
            )
        lines.append("")

    if classifier_timing:
        lines.extend([
            "## Classifier Timing (ms)",
            "",
            "| Step | Count | p50 | p95 | p99 | Mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in sorted(classifier_timing.keys()):
            stats = classifier_timing[name]
            lines.append(
                "| {name} | {count} | {p50} | {p95} | {p99} | {mean} | {min_v} | {max_v} |".format(
                    name=name,
                    count=stats.get("count", 0),
                    p50=_fmt(stats.get("p50")),
                    p95=_fmt(stats.get("p95")),
                    p99=_fmt(stats.get("p99")),
                    mean=_fmt(stats.get("mean")),
                    min_v=_fmt(stats.get("min")),
                    max_v=_fmt(stats.get("max")),
                )
            )
        lines.append("")

    if classifier_batch_shapes:
        lines.extend([
            "## Classifier Batch Shapes",
            "",
            "| Metric | Count | p50 | p95 | p99 | Mean | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in sorted(classifier_batch_shapes.keys()):
            stats = classifier_batch_shapes[name]
            lines.append(
                "| {name} | {count} | {p50} | {p95} | {p99} | {mean} | {min_v} | {max_v} |".format(
                    name=name,
                    count=stats.get("count", 0),
                    p50=_fmt(stats.get("p50")),
                    p95=_fmt(stats.get("p95")),
                    p99=_fmt(stats.get("p99")),
                    mean=_fmt(stats.get("mean")),
                    min_v=_fmt(stats.get("min")),
                    max_v=_fmt(stats.get("max")),
                )
            )
        lines.append("")

    if batch_latency or batch_sizes:
        classify_batch_latency = batch_latency.get("classify", {}) or {}
        classify_batch_sizes = batch_sizes.get("classify", {}) or {}
        index_db_batch_latency = batch_latency.get("index_db", {}) or {}
        index_db_batch_sizes = batch_sizes.get("index_db", {}) or {}
        lines.extend([
            "## Batch Metrics",
            "",
            f"- **Classify Batch Size Mean**: `{_fmt(classify_batch_sizes.get('mean'))}`",
            f"- **Classify Batch Size p95**: `{_fmt(classify_batch_sizes.get('p95'))}`",
            f"- **Classify Batch Latency Mean**: `{_fmt(classify_batch_latency.get('mean'))} ms`",
            f"- **Classify Batch Latency p95**: `{_fmt(classify_batch_latency.get('p95'))} ms`",
            f"- **Index DB Batch Size Mean**: `{_fmt(index_db_batch_sizes.get('mean'))}`",
            f"- **Index DB Batch Size p95**: `{_fmt(index_db_batch_sizes.get('p95'))}`",
            f"- **Index DB Batch Latency Mean**: `{_fmt(index_db_batch_latency.get('mean'))} ms`",
            f"- **Index DB Batch Latency p95**: `{_fmt(index_db_batch_latency.get('p95'))} ms`",
            "",
        ])

    lines.extend([
        "## System Load",
        "",
        f"- **Sample Count**: `{system_load_collection.get('sample_count', 0)}`",
        f"- **Sample Interval**: `{_fmt(system_load_collection.get('sample_interval_s'))} s`",
        f"- **Platform**: `{system_load_collection.get('platform', 'n/a')}`",
        f"- **Memory Probe**: `{system_load_collection.get('memory_probe', 'n/a')}`",
        f"- **Normalized Load (1m) Mean**: `{_fmt(((system_load_summary.get('normalized_load_1m') or {}).get('mean')))}`",
        f"- **Normalized Load (1m) p95**: `{_fmt(((system_load_summary.get('normalized_load_1m') or {}).get('p95')))}`",
        f"- **Normalized Load (5m) Mean**: `{_fmt(((system_load_summary.get('normalized_load_5m') or {}).get('mean')))}`",
        f"- **Memory Available Ratio Mean**: `{_fmt(((system_load_summary.get('memory_available_ratio') or {}).get('mean')))}`",
        f"- **Memory Available Ratio Min**: `{_fmt(((system_load_summary.get('memory_available_ratio') or {}).get('min')))}`",
        "",
        "## Gate",
        "",
    ])

    if gate:
        lines.append(f"- **Pass**: `{gate.get('pass')}`")
        lines.append(f"- **Throughput Delta %**: `{_fmt(gate.get('throughput_delta_pct'))}`")
        lines.append(f"- **Load-Adjusted Throughput Delta %**: `{_fmt(gate.get('load_adjusted_throughput_delta_pct'))}`")
        reasons = gate.get("reasons", []) or ["none"]
        lines.append(f"- **Reasons**: {', '.join(reasons)}")
    else:
        lines.append("- Gate not evaluated")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
