"""Benchmark CLI for production-like throughput profiling.

Usage examples:
  python -m ClassifierPipeline.benchmark run --dataset ClassifierPipeline/tests/stub_data/stub_new_records.csv --mode fake
  python -m ClassifierPipeline.benchmark sweep --dataset ClassifierPipeline/tests/stub_data/stub_new_records.csv
  python -m ClassifierPipeline.benchmark compare --baseline baseline.json --candidate candidate.json
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from adsputils import load_config

import ClassifierPipeline.perf_metrics as perf_metrics


DEFAULT_BATCH_SIZES = [50, 100, 250, 500]
DEFAULT_MODES = ["real", "fake"]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_csv_list(value: str, cast_type=int) -> List:
    if not value:
        return []
    output = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        output.append(cast_type(item))
    return output


def _safe_git_commit() -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        return None
    return None


def _read_dataset(records_path: str) -> List[Dict[str, str]]:
    import ClassifierPipeline.utilities as utils

    records: List[Dict[str, str]] = []
    possible_headers = {"bibcode", "scixid", "scix_id"}

    with open(records_path, "r") as handle:
        reader = csv.reader(handle)
        first_row = next(reader)
        data_rows = []
        if str(first_row[0]).lower() not in possible_headers:
            data_rows.append(first_row)
        data_rows.extend(list(reader))

    for row in data_rows:
        if len(row) < 3:
            continue
        identifier = str(row[0]).strip()
        title = str(row[1])
        abstract = str(row[2])

        record = {"title": title, "abstract": abstract}
        identifier_type = utils.check_identifier(identifier)
        if identifier_type == "bibcode":
            record["bibcode"] = identifier
        elif identifier_type == "scix_id":
            record["scix_id"] = identifier
        else:
            # invalid identifier; skip
            continue

        records.append(record)

    return records


def _chunks(items: List[dict], chunk_size: int) -> Iterable[List[dict]]:
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def _poll_run_completion(run_id: int, expected_records: int, timeout_s: int, poll_interval_s: float) -> Dict[str, object]:
    import ClassifierPipeline.tasks as tasks

    start = time.time()
    last_indexed = 0
    while (time.time() - start) < timeout_s:
        records = tasks.app.query_final_collection_table(run_id=run_id)
        indexed = len(records)
        last_indexed = indexed
        if indexed >= expected_records:
            return {
                "complete": True,
                "records_indexed": indexed,
                "elapsed_s": time.time() - start,
            }
        time.sleep(poll_interval_s)

    return {
        "complete": False,
        "records_indexed": last_indexed,
        "elapsed_s": time.time() - start,
    }


def _run_case(
    records_path: str,
    mode: str,
    batch_size: int,
    timeout_s: int,
    poll_interval_s: float,
    operation_step: str,
    events_path: str,
    worker_profile: Optional[str] = None,
    queue_names: Optional[str] = None,
) -> Dict[str, object]:
    import ClassifierPipeline.tasks as tasks
    import ClassifierPipeline.utilities as utils

    # Profiling flags consumed by task workers.
    os.environ["PERF_METRICS_ENABLED"] = "true"
    os.environ["PERF_METRICS_PATH"] = events_path
    os.environ["PERF_FORCE_FAKE_DATA"] = "true" if mode == "fake" else "false"

    records = _read_dataset(records_path)
    if not records:
        raise RuntimeError(f"No valid records found in dataset: {records_path}")

    run_id = tasks.app.index_run()
    submitted = 0

    start_wall = time.time()
    for chunk in _chunks(records, batch_size):
        payload = []
        for record in chunk:
            item = dict(record)
            item["run_id"] = run_id
            item["operation_step"] = operation_step
            payload.append(item)

        message = utils.list_to_ClassifyRequestRecordList(payload)
        tasks.task_update_record.delay(message)
        submitted += len(payload)

    completion = _poll_run_completion(
        run_id=run_id,
        expected_records=submitted,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    end_wall = time.time()

    events = perf_metrics.load_events(events_path, run_id=run_id)
    summary = perf_metrics.aggregate_events(
        events,
        started_at=start_wall,
        ended_at=end_wall,
        expected_records=submitted,
    )

    summary["run_metadata"] = {
        "run_id": run_id,
        "dataset": records_path,
        "mode": mode,
        "batch_size": batch_size,
        "timeout_s": timeout_s,
        "operation_step": operation_step,
        "worker_profile": worker_profile,
        "queue_names": queue_names,
        "git_commit": _safe_git_commit(),
        "timestamp_utc": _utc_timestamp(),
    }

    summary["counts"]["records_submitted"] = submitted
    summary["counts"]["records_indexed"] = completion["records_indexed"]
    summary["status"] = "complete" if completion["complete"] else "incomplete"
    summary.setdefault("errors", {})
    summary["errors"].update({
        "timeout": not completion["complete"],
        "completion_elapsed_s": completion["elapsed_s"],
    })

    return summary


def _summary_filename(prefix: str, suffix: str) -> str:
    return f"{prefix}_{_utc_timestamp()}_{suffix}"


def _write_run_artifacts(summary: Dict[str, object], output_dir: str, prefix: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    run_id = (summary.get("run_metadata", {}) or {}).get("run_id", "unknown")
    mode = (summary.get("run_metadata", {}) or {}).get("mode", "unknown")
    batch = (summary.get("run_metadata", {}) or {}).get("batch_size", "unknown")
    stem = _summary_filename(prefix, f"run{run_id}_{mode}_b{batch}")

    json_path = os.path.join(output_dir, f"{stem}.json")
    md_path = os.path.join(output_dir, f"{stem}.md")

    perf_metrics.write_json(json_path, summary)
    perf_metrics.render_markdown(summary, md_path)

    return {"json": json_path, "markdown": md_path}


def cmd_run(args) -> int:
    config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
    output_dir = args.output_dir or config.get("PERF_METRICS_OUTPUT_DIR", os.path.join("logs", "benchmarks"))
    events_path = args.events_path or os.path.join(output_dir, "perf_events.jsonl")

    summary = _run_case(
        records_path=args.dataset,
        mode=args.mode,
        batch_size=args.batch_size,
        timeout_s=args.timeout,
        poll_interval_s=args.poll_interval,
        operation_step=args.operation_step,
        events_path=events_path,
        worker_profile=args.worker_profile,
        queue_names=args.queue_names,
    )

    if args.baseline:
        baseline = json.load(open(args.baseline, "r"))
        summary["gate"] = perf_metrics.evaluate_gate(
            candidate=summary,
            baseline=baseline,
            min_throughput_improvement_pct=args.min_throughput_improvement_pct,
            p95_regression_limit_pct=args.p95_regression_limit_pct,
        )

    artifacts = _write_run_artifacts(summary, output_dir=output_dir, prefix="benchmark")
    print(json.dumps({
        "status": summary.get("status"),
        "throughput": (summary.get("throughput", {}) or {}).get("overall_records_per_minute"),
        "json": artifacts["json"],
        "markdown": artifacts["markdown"],
        "gate": summary.get("gate"),
    }, indent=2, sort_keys=True))

    if summary.get("status") != "complete":
        return 2
    if summary.get("gate") and not summary["gate"].get("pass", False):
        return 3
    return 0


def _run_warmup_case(case_args: dict) -> None:
    try:
        _run_case(**case_args)
    except Exception:
        # Warmup failures should not hide measured run behavior.
        return


def cmd_sweep(args) -> int:
    config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))
    output_dir = args.output_dir or config.get("PERF_METRICS_OUTPUT_DIR", os.path.join("logs", "benchmarks"))
    os.makedirs(output_dir, exist_ok=True)

    sweep_config = {}
    if args.config:
        with open(args.config, "r") as handle:
            sweep_config = json.load(handle)

    batch_sizes = _parse_csv_list(args.batch_sizes, int) or sweep_config.get("batch_sizes") or DEFAULT_BATCH_SIZES
    modes = _parse_csv_list(args.modes, str) or sweep_config.get("modes") or DEFAULT_MODES

    timeout_s = args.timeout if args.timeout is not None else sweep_config.get("timeout", 900)
    poll_interval_s = args.poll_interval if args.poll_interval is not None else sweep_config.get("poll_interval", 2.0)
    operation_step = args.operation_step or sweep_config.get("operation_step", "classify_verify")

    events_path = args.events_path or os.path.join(output_dir, "perf_events.jsonl")

    results = []
    for mode in modes:
        for batch_size in batch_sizes:
            case_args = {
                "records_path": args.dataset,
                "mode": mode,
                "batch_size": int(batch_size),
                "timeout_s": int(timeout_s),
                "poll_interval_s": float(poll_interval_s),
                "operation_step": operation_step,
                "events_path": events_path,
                "worker_profile": args.worker_profile,
                "queue_names": args.queue_names,
            }

            if args.warmup:
                _run_warmup_case(case_args)

            summary = _run_case(**case_args)
            artifacts = _write_run_artifacts(summary, output_dir=output_dir, prefix="benchmark_case")
            results.append({
                "mode": mode,
                "batch_size": batch_size,
                "status": summary.get("status"),
                "throughput": (summary.get("throughput", {}) or {}).get("overall_records_per_minute"),
                "json": artifacts["json"],
                "markdown": artifacts["markdown"],
            })

    sweep_summary = {
        "run_metadata": {
            "dataset": args.dataset,
            "batch_sizes": batch_sizes,
            "modes": modes,
            "timeout_s": timeout_s,
            "poll_interval_s": poll_interval_s,
            "operation_step": operation_step,
            "warmup": bool(args.warmup),
            "worker_profile": args.worker_profile,
            "queue_names": args.queue_names,
            "git_commit": _safe_git_commit(),
            "timestamp_utc": _utc_timestamp(),
        },
        "results": results,
    }

    sweep_json = os.path.join(output_dir, _summary_filename("benchmark_sweep", "summary") + ".json")
    perf_metrics.write_json(sweep_json, sweep_summary)

    print(json.dumps({"results": results, "sweep_json": sweep_json}, indent=2, sort_keys=True))
    return 0


def cmd_compare(args) -> int:
    baseline = json.load(open(args.baseline, "r"))
    candidate = json.load(open(args.candidate, "r"))

    gate = perf_metrics.evaluate_gate(
        candidate=candidate,
        baseline=baseline,
        min_throughput_improvement_pct=args.min_throughput_improvement_pct,
        p95_regression_limit_pct=args.p95_regression_limit_pct,
    )

    output = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "gate": gate,
    }

    if args.output:
        perf_metrics.write_json(args.output, output)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if gate.get("pass") else 3


def build_parser() -> argparse.ArgumentParser:
    config = load_config(proj_home=os.path.realpath(os.path.join(os.path.dirname(__file__), "../")))

    parser = argparse.ArgumentParser(description="Throughput benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_p95_reg = float(config.get("PERF_P95_REGRESSION_LIMIT_PCT", 10.0))
    default_tp_impr = float(config.get("PERF_MIN_THROUGHPUT_IMPROVEMENT_PCT", 5.0))

    run_parser = subparsers.add_parser("run", help="Run one benchmark configuration")
    run_parser.add_argument("--dataset", required=True, help="CSV dataset path")
    run_parser.add_argument("--mode", choices=["real", "fake"], default="real")
    run_parser.add_argument("--batch-size", type=int, default=100)
    run_parser.add_argument("--timeout", type=int, default=900)
    run_parser.add_argument("--poll-interval", type=float, default=2.0)
    run_parser.add_argument("--operation-step", default="classify_verify")
    run_parser.add_argument("--events-path", default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--worker-profile", default="")
    run_parser.add_argument("--queue-names", default="")
    run_parser.add_argument("--baseline", default=None)
    run_parser.add_argument("--p95-regression-limit-pct", type=float, default=default_p95_reg)
    run_parser.add_argument("--min-throughput-improvement-pct", type=float, default=default_tp_impr)
    run_parser.set_defaults(func=cmd_run)

    sweep_parser = subparsers.add_parser("sweep", help="Run fixed/matrix benchmark sweep")
    sweep_parser.add_argument("--dataset", required=True, help="CSV dataset path")
    sweep_parser.add_argument("--config", default=None, help="Optional sweep config JSON")
    sweep_parser.add_argument("--batch-sizes", default=",".join(str(x) for x in DEFAULT_BATCH_SIZES))
    sweep_parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    sweep_parser.add_argument("--timeout", type=int, default=None)
    sweep_parser.add_argument("--poll-interval", type=float, default=None)
    sweep_parser.add_argument("--operation-step", default="classify_verify")
    sweep_parser.add_argument("--events-path", default=None)
    sweep_parser.add_argument("--output-dir", default=None)
    sweep_parser.add_argument("--worker-profile", default="")
    sweep_parser.add_argument("--queue-names", default="")
    sweep_parser.add_argument("--warmup", action="store_true", default=True)
    sweep_parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    sweep_parser.set_defaults(func=cmd_sweep)

    compare_parser = subparsers.add_parser("compare", help="Compare candidate vs baseline metrics JSON")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", default=None)
    compare_parser.add_argument("--p95-regression-limit-pct", type=float, default=default_p95_reg)
    compare_parser.add_argument("--min-throughput-improvement-pct", type=float, default=default_tp_impr)
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
