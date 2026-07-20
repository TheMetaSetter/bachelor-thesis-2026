from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.ops.prune_raw_trace_artifacts import prune_raw_trace_artifacts


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
DEFAULT_PROTOCOL_CONFIG = Path("configs/protocol/smd_window20_cleanval_q99_ewma09.yaml")
DEFAULT_PYTHON_BIN = Path(".venv/bin/python")
DEFAULT_RUN_ROOTS = (
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed36",
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed36",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed8",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: Path | str, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def _default_run_roots(repo_root: Path) -> list[Path]:
    return [_resolve_path(run_root, repo_root) for run_root in DEFAULT_RUN_ROOTS]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def _verify_run(run_root: Path) -> dict[str, Any]:
    metrics_path = (
        run_root / "two_stage" / "stage_b_fusion_finetuning" / "evaluation_metrics.json"
    )
    uq_summary_path = run_root / "metrics" / "uq_summary.json"
    report_path = run_root / "benchmark" / "thesis_offline_benchmark_report.json"
    missing_paths = [
        str(path)
        for path in (metrics_path, uq_summary_path, report_path)
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(f"missing required artifact(s): {missing_paths}")

    metrics = _load_json(metrics_path)
    uq_summary = _load_json(uq_summary_path)
    test_split = dict(uq_summary.get("splits", {}).get("test", {}))
    uncertainty_summary = dict(test_split.get("uncertainty_summary", {}))
    trace_audit = dict(test_split.get("trace_audit", {}))

    metric_missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if metric_missing:
        raise ValueError(f"missing metric keys for {run_root}: {metric_missing}")

    uq_missing = [
        key for key in REQUIRED_UQ_KEYS if uncertainty_summary.get(key) is None
    ]
    if uq_missing:
        raise ValueError(f"missing UQ summary values for {run_root}: {uq_missing}")

    if not trace_audit.get("any_uncertainty_history", False):
        raise ValueError(f"trace audit missing uncertainty history for {run_root}")
    if not trace_audit.get("any_mc_sample_history", False):
        raise ValueError(f"trace audit missing MC sample history for {run_root}")

    return {
        "run_root": str(run_root),
        "metric_values": {key: metrics[key] for key in REQUIRED_METRIC_KEYS},
        "uq_values": {key: uncertainty_summary[key] for key in REQUIRED_UQ_KEYS},
        "trace_audit": trace_audit,
    }


def _delete_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _cleanup_heavy_artifacts(run_root: Path, *, keep_compact_traces: bool) -> dict[str, Any]:
    removed: list[str] = []
    if keep_compact_traces:
        compaction_report = prune_raw_trace_artifacts(root_dir=run_root, dry_run=False)
        removed.extend(compaction_report["compacted"])
    else:
        for pattern in (
            "traces/*_traces.json",
            "scores/*.npz",
            "retention/**/traces.json",
            "retention/**/score*.npz",
            "retention/**/scores/*.npz",
            "two_stage/stage_b_fusion_finetuning/traces/*_traces.json",
            "two_stage/stage_b_fusion_finetuning/scores/*.npz",
            "two_stage/stage_b_fusion_finetuning/retention/**/traces.json",
            "two_stage/stage_b_fusion_finetuning/retention/**/score*.npz",
            "two_stage/stage_b_fusion_finetuning/retention/**/scores/*.npz",
        ):
            for path in sorted(run_root.glob(pattern)):
                if _delete_path(path):
                    removed.append(str(path))

    for relative_path in (
        "two_stage/stage_b_fusion_finetuning/evaluation_records.json",
        "two_stage/stage_b_fusion_finetuning/evaluation_curves.json",
    ):
        path = run_root / relative_path
        if _delete_path(path):
            removed.append(str(path))

    return {
        "run_root": str(run_root),
        "keep_compact_traces": keep_compact_traces,
        "removed_count": len(removed),
        "removed": removed,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-evaluate thesis offline runs one by one, verify summary files, then prune heavy artifacts."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        help="Run root to process. May be repeated. If omitted, the 18 thesis rerun roots are used.",
    )
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_CONFIG),
        help="Protocol config to use for every evaluation-only rerun.",
    )
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON_BIN),
        help="Python interpreter to use for the evaluation command.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/tmux_logs",
        help="Directory where per-run logs are written.",
    )
    parser.add_argument(
        "--keep-compact-traces",
        action="store_true",
        help="Keep compacted trace JSON files after verification instead of deleting them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running evaluation or pruning.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = _repo_root()
    python_bin = _resolve_path(args.python_bin, repo_root)
    protocol_config = _resolve_path(args.protocol_config, repo_root)
    log_dir = _resolve_path(args.log_dir, repo_root)
    run_roots = (
        [_resolve_path(run_root, repo_root) for run_root in args.run_roots]
        if args.run_roots
        else _default_run_roots(repo_root)
    )

    print(json.dumps(
        {
            "repo_root": str(repo_root),
            "python_bin": str(python_bin),
            "protocol_config": str(protocol_config),
            "run_count": len(run_roots),
            "keep_compact_traces": bool(args.keep_compact_traces),
            "dry_run": bool(args.dry_run),
        },
        indent=2,
        sort_keys=True,
    ))

    if args.dry_run:
        for run_root in run_roots:
            print(
                json.dumps(
                    {
                        "run_root": str(run_root),
                        "experiment_config": str(
                            run_root
                            / "two_stage"
                            / "generated_configs"
                            / "02_stage_b_fusion_finetuning.yaml"
                        ),
                        "checkpoint_path": str(
                            run_root
                            / "two_stage"
                            / "stage_b_fusion_finetuning"
                            / "checkpoints"
                            / "best.pt"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return

    summary: list[dict[str, Any]] = []
    for index, run_root in enumerate(run_roots, start=1):
        experiment_config = (
            run_root / "two_stage" / "generated_configs" / "02_stage_b_fusion_finetuning.yaml"
        )
        checkpoint_path = (
            run_root
            / "two_stage"
            / "stage_b_fusion_finetuning"
            / "checkpoints"
            / "best.pt"
        )
        eval_log = log_dir / f"re_eval_{run_root.as_posix().replace('/', '_')}.txt"
        print(f"[{index}/{len(run_roots)}] evaluating {run_root}")
        _run_command(
            [
                str(python_bin),
                "-m",
                "scripts.run_thesis_offline_benchmark",
                "--experiment-config",
                str(experiment_config),
                "--protocol-config",
                str(protocol_config),
                "--evaluation-only",
                "--checkpoint-path",
                str(checkpoint_path),
            ],
            cwd=repo_root,
            log_path=eval_log,
        )
        verified = _verify_run(run_root)
        print(
            f"[{index}/{len(run_roots)}] verified metric and UQ summaries for {run_root}"
        )
        cleanup_report = _cleanup_heavy_artifacts(
            run_root,
            keep_compact_traces=bool(args.keep_compact_traces),
        )
        print(
            f"[{index}/{len(run_roots)}] pruned {cleanup_report['removed_count']} heavy artifacts for {run_root}"
        )
        summary.append(verified | cleanup_report)

    summary_path = log_dir / "re_evaluate_and_prune_thesis_runs_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
