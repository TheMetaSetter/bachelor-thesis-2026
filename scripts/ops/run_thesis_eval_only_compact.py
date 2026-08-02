from __future__ import annotations

"""Run thesis evaluation-only reruns into a short, compact output tree.

This wrapper keeps the source Stage B checkpoint tree untouched and writes each
rerun into a short alias path such as `outputs/eval18/o0_m1_6_s6/`.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.ops.prune_raw_trace_artifacts import prune_raw_trace_artifacts


DEFAULT_PROTOCOL_CONFIG = Path("configs/protocol/smd_window20_cleanval_q99_ewma09.yaml")
DEFAULT_PYTHON_BIN = Path(".venv/bin/python")
DEFAULT_OUTPUT_ROOT = Path("outputs/eval18")
DEFAULT_RUN_ROOTS = (
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed8",
    "outputs/benchmark/smd/thesis/O0/machine_3_9/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_1_6/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_4/seed36",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed6",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed8",
    "outputs/benchmark/smd/thesis/O1/machine_3_9/seed36",
)
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


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return loaded


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _source_run_identity(run_root: Path) -> tuple[str, str, int]:
    parts = run_root.resolve().parts
    try:
        outputs_index = parts.index("outputs")
    except ValueError as exc:
        raise ValueError(f"run root must contain outputs/: {run_root}") from exc

    tail = parts[outputs_index + 1 :]
    if (
        len(tail) < 6
        or tail[0] != "benchmark"
        or tail[1] != "smd"
        or tail[2] != "thesis"
    ):
        raise ValueError(
            "run root must look like outputs/benchmark/smd/thesis/<variant>/<entity>/seed<seed>"
        )

    variant = tail[3]
    entity = tail[4]
    seed_part = tail[5]
    if not seed_part.startswith("seed"):
        raise ValueError(f"invalid seed segment in run root: {run_root}")
    try:
        seed = int(seed_part.removeprefix("seed"))
    except ValueError as exc:
        raise ValueError(f"invalid seed value in run root: {run_root}") from exc
    return variant, entity, seed


def _alias_for_run(run_root: Path) -> str:
    variant, entity, seed = _source_run_identity(run_root)
    entity_alias = entity.replace("machine_", "m")
    return f"{variant.lower()}_{entity_alias}_s{seed}"


def _compact_config_path(output_root: Path, alias: str) -> Path:
    return output_root / alias / "generated_configs" / "stage_b_eval_only.yaml"


def _compact_output_root(output_root: Path, alias: str) -> Path:
    return output_root / alias


def _compact_log_path(repo_root: Path, alias: str) -> Path:
    return repo_root / "outputs" / "tmux_logs" / f"eval18_{alias}.txt"


def _prepare_compact_config(
    *,
    source_config_path: Path,
    compact_output_root: Path,
) -> dict[str, Any]:
    config = _load_yaml(source_config_path)
    config["output_dir"] = str(compact_output_root)
    config["checkpoint_dir"] = str(compact_output_root / "checkpoints")
    logging_config = dict(config.get("logging") or {})
    if logging_config:
        logging_config["wandb_run_name"] = f"eval18_{compact_output_root.name}"
        config["logging"] = logging_config
    return config


def _verify_compact_run(
    *,
    compact_output_root: Path,
) -> dict[str, Any]:
    metrics_path = (
        compact_output_root
        / "two_stage"
        / "stage_b_fusion_finetuning"
        / "evaluation_metrics.json"
    )
    uq_summary_path = compact_output_root / "metrics" / "uq_summary.json"
    report_path = (
        compact_output_root / "benchmark" / "thesis_offline_benchmark_report.json"
    )
    trace_path = compact_output_root / "traces" / "test_traces.json"

    report: dict[str, Any] = {
        "compact_output_root": str(compact_output_root),
        "artifact_exists": {
            "evaluation_metrics.json": metrics_path.exists(),
            "uq_summary.json": uq_summary_path.exists(),
            "thesis_offline_benchmark_report.json": report_path.exists(),
            "test_traces.json": trace_path.exists(),
        },
    }

    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        report["metric_values"] = {
            key: metrics.get(key) for key in REQUIRED_METRIC_KEYS
        }
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
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run thesis evaluation-only reruns into a compact alias tree and keep the paths short."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        dest="run_roots",
        help=(
            "Source benchmark run root. If omitted, the 18 thesis combinations are used."
        ),
    )
    parser.add_argument(
        "--skip-run-root",
        action="append",
        dest="skip_run_roots",
        help=(
            "Source benchmark run root to exclude from the default 18-combination set."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Compact output root that will hold the short alias directories.",
    )
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_CONFIG),
        help="Protocol config to use for every evaluation-only rerun.",
    )
    parser.add_argument(
        "--python-bin",
        default=str(DEFAULT_PYTHON_BIN),
        help="Python interpreter to use for the rerun command.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write compact configs and the manifest; do not run evaluation.",
    )
    parser.add_argument(
        "--keep-compact-traces",
        action="store_true",
        help="Keep compact trace JSON after verification instead of compacting them further.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow reusing an existing compact output root.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = _repo_root()
    python_bin = _resolve_path(args.python_bin, repo_root)
    protocol_config = _resolve_path(args.protocol_config, repo_root)
    output_root = _resolve_path(args.output_root, repo_root)
    run_roots = (
        [_resolve_path(run_root, repo_root) for run_root in args.run_roots]
        if args.run_roots
        else [_resolve_path(run_root, repo_root) for run_root in DEFAULT_RUN_ROOTS]
    )
    if args.skip_run_roots:
        skip_run_roots = {
            _resolve_path(run_root, repo_root).resolve()
            for run_root in args.skip_run_roots
        }
        run_roots = [
            run_root
            for run_root in run_roots
            if run_root.resolve() not in skip_run_roots
        ]

    compact_entries: list[dict[str, Any]] = []
    for run_root in run_roots:
        alias = _alias_for_run(run_root)
        source_config = (
            run_root
            / "two_stage"
            / "generated_configs"
            / "02_stage_b_fusion_finetuning.yaml"
        )
        source_checkpoint = (
            run_root
            / "two_stage"
            / "stage_b_fusion_finetuning"
            / "checkpoints"
            / "best.pt"
        )
        compact_run_root = _compact_output_root(output_root, alias)
        compact_config_path = _compact_config_path(output_root, alias)
        compact_log_path = _compact_log_path(repo_root, alias)
        if compact_run_root.exists() and not args.overwrite_existing:
            raise FileExistsError(
                f"compact output root already exists: {compact_run_root}"
            )

        if not source_config.exists():
            raise FileNotFoundError(f"missing source config: {source_config}")
        if not source_checkpoint.exists():
            raise FileNotFoundError(f"missing source checkpoint: {source_checkpoint}")

        compact_config = _prepare_compact_config(
            source_config_path=source_config,
            compact_output_root=compact_run_root,
        )
        _dump_yaml(compact_config_path, compact_config)

        entry: dict[str, Any] = {
            "source_run_root": str(run_root),
            "alias": alias,
            "compact_output_root": str(compact_run_root),
            "compact_config_path": str(compact_config_path),
            "compact_log_path": str(compact_log_path),
            "checkpoint_path": str(source_checkpoint),
            "protocol_config": str(protocol_config),
        }

        if args.prepare_only:
            entry["status"] = "prepared"
            compact_entries.append(entry)
            continue

        command = [
            str(python_bin),
            "-m",
            "scripts.run_thesis_offline_benchmark",
            "--experiment-config",
            str(compact_config_path),
            "--protocol-config",
            str(protocol_config),
            "--evaluation-only",
            "--checkpoint-path",
            str(source_checkpoint),
        ]
        compact_log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["THESIS_CONSOLE_QUIET"] = "1"
        with compact_log_path.open("w", encoding="utf-8") as log_file:
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

        verify_report = _verify_compact_run(compact_output_root=compact_run_root)
        entry["verification"] = verify_report
        if not args.keep_compact_traces:
            entry["trace_prune"] = prune_raw_trace_artifacts(
                root_dir=compact_run_root,
                dry_run=False,
            )
        else:
            entry["trace_prune"] = {"skipped": True}
        compact_entries.append(entry)

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "run_count": len(run_roots),
        "prepare_only": args.prepare_only,
        "keep_compact_traces": args.keep_compact_traces,
        "runs": compact_entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "output_root": str(output_root),
                "run_count": len(run_roots),
                "prepare_only": args.prepare_only,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
