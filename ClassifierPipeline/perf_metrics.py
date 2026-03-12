"""Performance metrics helpers for benchmark/profiling workflows.

This module is intentionally stdlib-only and file-backed so it can be used from
multiple Celery worker processes without introducing extra infrastructure.
"""

import json
import os
import time
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


def emit_event(
    stage: str,
    run_id: Optional[Any] = None,
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
        if not metrics_enabled(config=config):
            return

        target_path = path or metrics_path(config=config)
        if not target_path:
            return

        payload = {
            "ts": time.time(),
            "stage": stage,
            "run_id": str(run_id) if run_id is not None else None,
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
            record_id=record_id,
            duration_ms=duration_ms,
            status=outcome,
            extra=extra,
            config=config,
            path=path,
        )


def load_events(path: str, run_id: Optional[Any] = None) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    run_id_str = str(run_id) if run_id is not None else None
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


def _duration_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    total = sum(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": total / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def aggregate_events(
    events: List[Dict[str, Any]],
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    expected_records: Optional[int] = None,
) -> Dict[str, Any]:
    stages: Dict[str, List[float]] = {}
    stage_errors: Dict[str, int] = {}

    submitted = 0
    indexed = 0
    forwarded = 0
    failure_count = 0

    event_timestamps = [e.get("ts") for e in events if e.get("ts") is not None]

    for event in events:
        stage = event.get("stage", "unknown")
        status = event.get("status", "ok")
        duration = event.get("duration_ms")

        if duration is not None:
            stages.setdefault(stage, []).append(float(duration))

        if status != "ok":
            failure_count += 1
            stage_errors[stage] = stage_errors.get(stage, 0) + 1

        extra = event.get("extra", {}) or {}
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

    throughput_delta_pct = None
    throughput_pass = False
    if cand_t is not None and base_t not in (None, 0):
        throughput_delta_pct = ((cand_t - base_t) / base_t) * 100.0
        throughput_pass = throughput_delta_pct >= min_throughput_improvement_pct

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

    return {
        "pass": gate_pass,
        "throughput_delta_pct": throughput_delta_pct,
        "p95_delta_pct_by_stage": p95_deltas,
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
    counts = summary.get("counts", {}) or {}
    throughput = (summary.get("throughput", {}) or {}).get("overall_records_per_minute")
    duration = (summary.get("duration_s", {}) or {}).get("wall_clock")
    gate = summary.get("gate", {}) or {}

    lines = [
        "# Throughput Benchmark Report",
        "",
        "## Run Config",
        "",
    ]

    run_metadata = summary.get("run_metadata", {}) or {}
    for key in sorted(run_metadata.keys()):
        lines.append(f"- **{key}**: `{run_metadata[key]}`")

    lines.extend([
        "",
        "## Top-Line Results",
        "",
        f"- **Status**: `{summary.get('status', 'unknown')}`",
        f"- **Throughput**: `{_fmt(throughput)} records/min`",
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
        "## Gate",
        "",
    ])

    if gate:
        lines.append(f"- **Pass**: `{gate.get('pass')}`")
        lines.append(f"- **Throughput Delta %**: `{_fmt(gate.get('throughput_delta_pct'))}`")
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
