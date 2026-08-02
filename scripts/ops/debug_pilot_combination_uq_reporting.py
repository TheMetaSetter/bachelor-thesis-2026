from __future__ import annotations

"""Debug a pilot combination run for metrics and UQ reporting.

This script reruns evaluation-only for one thesis offline run, then inspects the
resulting artifacts and prints a concise JSON diagnostics payload. It is meant
for the user's pilot-combination debugging flow, not for batch execution.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_METRIC_KEYS = ("vus_pr", "affiliation_f1", "vus_roc")
REQUIRED_UQ_KEYS = (
    "point_anomaly_score_variance_mean",
    "point_anomaly_score_variance_p95",
    "window_anomaly_score_variance_mean",
    "classification_probability_variance_mean",
    "classification_variance_mean",
    "continuous_retrieval_variance_point_mean",
    "continuous_retrieval_variance_window_mean",
    "discrete_retrieval_variance_point_mean",
    "discrete_retrieval_variance_window_mean",
    "reconstruction_variance_point_mean",
    "reconstruction_variance_window_mean",
    "reconstruction_variance_full_mean",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_trace_path(run_root: Path) -> Path | None:
    return _first_existing_path(
        [
            run_root
            / "two_stage"
            / "stage_b_fusion_finetuning"
            / "traces"
            / "test_traces.json",
            run_root / "traces" / "test_traces.json",
            run_root
            / "two_stage"
            / "stage_b_fusion_finetuning"
            / "traces"
            / "evaluation_traces.json",
            run_root / "traces" / "evaluation_traces.json",
        ]
    )


def _summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    uncertainty_history = trace.get("uncertainty_history")
    stochastic_query = trace.get("stochastic_query")
    mc_sample_histories = trace.get("mc_sample_histories") or {}
    return {
        "batch_index": trace.get("batch_index"),
        "entity_ids": trace.get("entity_ids"),
        "has_uncertainty_history": uncertainty_history is not None,
        "uncertainty_history_keys": sorted(list(uncertainty_history.keys()))
        if isinstance(uncertainty_history, dict)
        else [],
        "has_stochastic_query": stochastic_query is not None,
        "stochastic_query_keys": sorted(list(stochastic_query.keys()))
        if isinstance(stochastic_query, dict)
        else [],
        "sample_retention_policy": trace.get("sample_retention_policy"),
        "mc_sample_histories_non_null": {
            key_name: value is not None
            for key_name, value in mc_sample_histories.items()
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one evaluation-only pilot combination and print UQ diagnostics."
    )
    parser.add_argument(
        "--run-root",
        required=True,
        help="Run root, for example outputs/benchmark/smd/thesis/O0/machine_1_6/seed36.",
    )
    parser.add_argument(
        "--experiment-config",
        help=(
            "Experiment config path. Defaults to the Stage B generated config "
            "under the given run root."
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        help=(
            "Checkpoint path. Defaults to the Stage B best checkpoint under the "
            "given run root."
        ),
    )
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        help="Protocol config for the evaluation rerun.",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python interpreter to use for the rerun command.",
    )
    parser.add_argument(
        "--log-path",
        help="Where to write the evaluation command output. Defaults under outputs/tmux_logs.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    run_root = _resolve_path(args.run_root, repo_root)
    experiment_config = (
        _resolve_path(args.experiment_config, repo_root)
        if args.experiment_config
        else run_root
        / "two_stage"
        / "generated_configs"
        / "02_stage_b_fusion_finetuning.yaml"
    )
    checkpoint_path = (
        _resolve_path(args.checkpoint_path, repo_root)
        if args.checkpoint_path
        else run_root
        / "two_stage"
        / "stage_b_fusion_finetuning"
        / "checkpoints"
        / "best.pt"
    )
    protocol_config = _resolve_path(args.protocol_config, repo_root)
    log_path = (
        _resolve_path(args.log_path, repo_root)
        if args.log_path
        else repo_root
        / "outputs"
        / "tmux_logs"
        / "debug_pilot_combination_uq_reporting.txt"
    )

    command = [
        str(_resolve_path(args.python_bin, repo_root)),
        "-m",
        "scripts.run_thesis_offline_benchmark",
        "--experiment-config",
        str(experiment_config),
        "--protocol-config",
        str(protocol_config),
        "--evaluation-only",
        "--checkpoint-path",
        str(checkpoint_path),
    ]

    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "run_root": str(run_root),
                "experiment_config": str(experiment_config),
                "checkpoint_path": str(checkpoint_path),
                "protocol_config": str(protocol_config),
                "log_path": str(log_path),
                "command": command,
                "env": {
                    "THESIS_CONSOLE_QUIET": "1",
                    "THESIS_DEBUG_UQ_TRACE": "1",
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["THESIS_CONSOLE_QUIET"] = "1"
    env["THESIS_DEBUG_UQ_TRACE"] = "1"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)

    metrics_path = (
        run_root / "two_stage" / "stage_b_fusion_finetuning" / "evaluation_metrics.json"
    )
    uq_summary_path = run_root / "metrics" / "uq_summary.json"
    trace_path = _find_trace_path(run_root)
    report: dict[str, Any] = {
        "run_root": str(run_root),
        "artifact_exists": {
            "evaluation_metrics.json": metrics_path.exists(),
            "uq_summary.json": uq_summary_path.exists(),
            "trace_path": str(trace_path) if trace_path is not None else None,
        },
    }

    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        report["metrics"] = {key: metrics.get(key) for key in REQUIRED_METRIC_KEYS}
        report["metric_missing"] = [
            key for key in REQUIRED_METRIC_KEYS if key not in metrics
        ]
    else:
        report["metric_missing"] = list(REQUIRED_METRIC_KEYS)

    if uq_summary_path.exists():
        uq_summary = _load_json(uq_summary_path)
        test_split = dict(uq_summary.get("splits", {}).get("test", {}))
        uncertainty_summary = dict(test_split.get("uncertainty_summary", {}))
        trace_audit = dict(test_split.get("trace_audit", {}))
        report["uq_values"] = {
            key: uncertainty_summary.get(key) for key in REQUIRED_UQ_KEYS
        }
        report["uq_missing"] = [
            key for key in REQUIRED_UQ_KEYS if uncertainty_summary.get(key) is None
        ]
        report["trace_audit"] = trace_audit
    else:
        report["uq_missing"] = list(REQUIRED_UQ_KEYS)

    if trace_path is not None:
        trace_payloads = _load_json(trace_path)
        report["trace_file"] = {
            "path": str(trace_path),
            "num_traces": len(trace_payloads)
            if isinstance(trace_payloads, list)
            else None,
            "first_trace": (
                _summarize_trace(trace_payloads[0])
                if isinstance(trace_payloads, list) and trace_payloads
                else None
            ),
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
