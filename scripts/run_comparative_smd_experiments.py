from __future__ import annotations

"""Sequential mixed-method launcher for comparative SMD experiments.

This script intentionally stays as a thin subprocess coordinator. It validates
resolved configs, derives the run family from config semantics, writes a durable
manifest, then delegates the actual work to existing entrypoints.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_BASELINE_MODEL_NAMES = {
    "redlamp_mlp_baseline",
    "redlamp_cnn_baseline",
}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mixed baseline and thesis comparative SMD experiments"
    )
    parser.add_argument("--config-paths", nargs="+", required=True)
    parser.add_argument("--smoke-config-paths", nargs="*", default=[])
    parser.add_argument(
        "--report-dir",
        default="outputs/comparative_smd_reports/default",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def normalize_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def resolve_dataset_root(resolved_experiment_config: dict[str, Any]) -> Path:
    dataset_root = Path(str(resolved_experiment_config["data"]["root_dir"]))
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    return dataset_root.resolve()


def validate_dataset_roots(resolved_experiment_configs: list[dict[str, Any]]) -> None:
    for resolved_experiment_config in resolved_experiment_configs:
        dataset_root = resolve_dataset_root(resolved_experiment_config)
        if not dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist for {resolved_experiment_config['experiment_name']}: {dataset_root}"
            )


def validate_unique_artifact_paths(
    resolved_experiment_configs: list[dict[str, Any]],
) -> None:
    seen_output_dirs: dict[Path, str] = {}
    seen_checkpoint_dirs: dict[Path, str] = {}
    for resolved_experiment_config in resolved_experiment_configs:
        experiment_name = str(resolved_experiment_config["experiment_name"])
        output_dir = Path(str(resolved_experiment_config["output_dir"]))
        checkpoint_dir = Path(str(resolved_experiment_config["checkpoint_dir"]))
        if output_dir in seen_output_dirs:
            raise ValueError(
                f"Duplicate output_dir detected: {output_dir} is shared by "
                f"{seen_output_dirs[output_dir]} and {experiment_name}"
            )
        if checkpoint_dir in seen_checkpoint_dirs:
            raise ValueError(
                f"Duplicate checkpoint_dir detected: {checkpoint_dir} is shared by "
                f"{seen_checkpoint_dirs[checkpoint_dir]} and {experiment_name}"
            )
        seen_output_dirs[output_dir] = experiment_name
        seen_checkpoint_dirs[checkpoint_dir] = experiment_name


def resolve_stage_family(resolved_experiment_config: dict[str, Any]) -> str:
    model_name = str(resolved_experiment_config["model"]["model_name"])
    if model_name == "thesis_multitask" and "three_stage" in resolved_experiment_config:
        return "thesis_three_stage"
    if model_name in SUPPORTED_BASELINE_MODEL_NAMES:
        return "baseline_single_stage"
    raise ValueError(
        "Unsupported comparative run family for model "
        f"{model_name}. Expected thesis three-stage or supported baseline."
    )


def _validate_single_entity_contract(resolved_experiment_config: dict[str, Any]) -> str:
    entity_ids = resolved_experiment_config["data"].get("entity_ids")
    if not isinstance(entity_ids, list) or len(entity_ids) != 1:
        raise ValueError(
            "Comparative SMD runs require exactly one entity_id per experiment config"
        )
    entity_id = entity_ids[0]
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("Comparative SMD entity_id must be a non-empty string")
    return entity_id


def _build_thesis_three_stage_commands(config_path: Path) -> list[list[str]]:
    return [
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_three_stage_offline_pretraining.py"),
            "--experiment-config",
            str(config_path),
        ]
    ]


def _build_baseline_single_stage_commands(
    config_path: Path,
    resolved_experiment_config: dict[str, Any],
) -> list[list[str]]:
    checkpoint_path = (
        Path(str(resolved_experiment_config["checkpoint_dir"])) / "best.pt"
    )
    return [
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train.py"),
            "--experiment-config",
            str(config_path),
        ],
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
            "--experiment-config",
            str(config_path),
            "--checkpoint-path",
            str(checkpoint_path),
        ],
    ]


def _build_run_record(
    *,
    config_path: Path,
    resolved_experiment_config: dict[str, Any],
    run_stage: str,
) -> dict[str, Any]:
    stage_family = resolve_stage_family(resolved_experiment_config)
    entity_id = _validate_single_entity_contract(resolved_experiment_config)
    output_dir = Path(str(resolved_experiment_config["output_dir"]))
    if stage_family == "thesis_three_stage":
        commands = _build_thesis_three_stage_commands(config_path)
    else:
        commands = _build_baseline_single_stage_commands(
            config_path,
            resolved_experiment_config,
        )

    return {
        "run_id": f"{run_stage}:{resolved_experiment_config['experiment_name']}",
        "run_stage": run_stage,
        "experiment_name": str(resolved_experiment_config["experiment_name"]),
        "stage_family": stage_family,
        "model_name": str(resolved_experiment_config["model"]["model_name"]),
        "entity_id": entity_id,
        "seed": int(resolved_experiment_config["seed"]),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(resolved_experiment_config["checkpoint_dir"]),
        "evaluation_metrics_path": str(output_dir / "evaluation_metrics.json"),
        "evaluation_records_path": str(output_dir / "evaluation_records.json"),
        "evaluation_curves_path": str(output_dir / "evaluation_curves.json"),
        "commands": commands,
    }


def _load_run_records(
    config_paths: list[str | Path],
    run_stage: str,
) -> list[dict[str, Any]]:
    run_records: list[dict[str, Any]] = []
    for config_path in config_paths:
        normalized_config_path = normalize_config_path(config_path)
        resolved_experiment_config = load_experiment_config(normalized_config_path)
        resolved_experiment_config["_config_path"] = str(normalized_config_path)
        run_records.append(
            _build_run_record(
                config_path=normalized_config_path,
                resolved_experiment_config=resolved_experiment_config,
                run_stage=run_stage,
            )
        )
    return run_records


def build_comparative_run_plan(
    *,
    config_paths: list[str | Path],
    smoke_config_paths: list[str | Path],
    report_dir: str | Path,
) -> dict[str, Any]:
    smoke_runs = _load_run_records(smoke_config_paths, run_stage="smoke")
    main_runs = _load_run_records(config_paths, run_stage="main")
    all_runs = smoke_runs + main_runs

    resolved_experiment_configs = [
        load_experiment_config(normalize_config_path(run_record["config_path"]))
        for run_record in all_runs
    ]
    validate_unique_artifact_paths(resolved_experiment_configs)
    validate_dataset_roots(resolved_experiment_configs)

    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = REPOSITORY_ROOT / resolved_report_dir
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    run_plan = {
        "created_at_utc": _utc_now_iso(),
        "report_dir": str(resolved_report_dir),
        "manifest_path": str(resolved_report_dir / "comparative_manifest.json"),
        "execution_report_path": str(
            resolved_report_dir / "comparative_execution_report.json"
        ),
        "smoke_runs": smoke_runs,
        "main_runs": main_runs,
    }
    Path(run_plan["manifest_path"]).write_text(
        json.dumps(run_plan, indent=2),
        encoding="utf-8",
    )
    return run_plan


def _write_execution_report(
    execution_report_path: str | Path,
    execution_report: dict[str, Any],
) -> None:
    Path(execution_report_path).write_text(
        json.dumps(execution_report, indent=2),
        encoding="utf-8",
    )


def _iter_run_groups(run_plan: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("smoke", list(run_plan["smoke_runs"])),
        ("main", list(run_plan["main_runs"])),
    ]


def execute_comparative_run_plan(
    run_plan: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    execution_report = {
        "started_at_utc": _utc_now_iso(),
        "manifest_path": run_plan["manifest_path"],
        "execution_report_path": run_plan["execution_report_path"],
        "dry_run": dry_run,
        "completed_run_ids": [],
        "executed_run_ids": [],
        "status": "running",
    }

    if dry_run:
        execution_report["status"] = "dry_run"
        execution_report["finished_at_utc"] = _utc_now_iso()
        _write_execution_report(run_plan["execution_report_path"], execution_report)
        return execution_report

    try:
        for _, run_records in _iter_run_groups(run_plan):
            for run_record in run_records:
                execution_report["executed_run_ids"].append(run_record["run_id"])
                for command in run_record["commands"]:
                    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
                execution_report["completed_run_ids"].append(run_record["run_id"])
    except subprocess.CalledProcessError as error:
        execution_report["status"] = "failed"
        execution_report["failed_at_utc"] = _utc_now_iso()
        if execution_report["executed_run_ids"]:
            execution_report["failed_run_id"] = execution_report["executed_run_ids"][-1]
        execution_report["failed_command"] = list(error.cmd)
        execution_report["failed_return_code"] = int(error.returncode)
        _write_execution_report(run_plan["execution_report_path"], execution_report)
        raise

    execution_report["status"] = "completed"
    execution_report["finished_at_utc"] = _utc_now_iso()
    _write_execution_report(run_plan["execution_report_path"], execution_report)
    return execution_report


def _print_run_records(run_records: list[dict[str, Any]], stage_name: str) -> None:
    for run_record in run_records:
        print(
            json.dumps(
                {
                    "stage_name": stage_name,
                    "run_id": run_record["run_id"],
                    "stage_family": run_record["stage_family"],
                    "entity_id": run_record["entity_id"],
                    "seed": run_record["seed"],
                    "commands": run_record["commands"],
                }
            )
        )


def main() -> None:
    args = parse_args()
    run_plan = build_comparative_run_plan(
        config_paths=list(args.config_paths),
        smoke_config_paths=list(args.smoke_config_paths),
        report_dir=args.report_dir,
    )
    _print_run_records(run_plan["smoke_runs"], stage_name="smoke")
    _print_run_records(run_plan["main_runs"], stage_name="main")

    if args.preflight_only:
        return
    execute_comparative_run_plan(run_plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
