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

import yaml

from src.core.config import load_experiment_config


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_BASELINE_MODEL_NAMES = {
    "redlamp_baseline",
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
    parser.add_argument(
        "--data-num-workers-override",
        type=int,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
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


def _normalize_artifact_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


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
        "device": str(resolved_experiment_config.get("device", "missing")),
        "entity_id": entity_id,
        "seed": int(resolved_experiment_config["seed"]),
        "config_path": str(config_path),
        "original_config_path": str(
            resolved_experiment_config.get("_original_config_path", config_path)
        ),
        "dataset_root": str(resolve_dataset_root(resolved_experiment_config)),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(resolved_experiment_config["checkpoint_dir"]),
        "evaluation_metrics_path": str(output_dir / "evaluation_metrics.json"),
        "evaluation_records_path": str(output_dir / "evaluation_records.json"),
        "evaluation_curves_path": str(output_dir / "evaluation_curves.json"),
        "data_num_workers": resolved_experiment_config["data"].get("num_workers"),
        "data_num_workers_override": resolved_experiment_config.get(
            "_data_num_workers_override"
        ),
        "commands": commands,
    }


def _load_run_records(
    config_paths: list[str | Path],
    run_stage: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_records: list[dict[str, Any]] = []
    resolved_experiment_configs: list[dict[str, Any]] = []
    for config_path in config_paths:
        normalized_config_path = normalize_config_path(config_path)
        resolved_experiment_config = load_experiment_config(normalized_config_path)
        resolved_experiment_config["_original_config_path"] = str(
            normalized_config_path
        )
        resolved_experiment_configs.append(resolved_experiment_config)
        run_records.append(
            _build_run_record(
                config_path=normalized_config_path,
                resolved_experiment_config=resolved_experiment_config,
                run_stage=run_stage,
            )
        )
    return run_records, resolved_experiment_configs


def _materialize_worker_override_configs(
    *,
    run_records: list[dict[str, Any]],
    resolved_experiment_configs: list[dict[str, Any]],
    report_dir: Path,
    data_num_workers_override: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generated_configs_dir = report_dir / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)
    overridden_run_records: list[dict[str, Any]] = []
    overridden_configs: list[dict[str, Any]] = []

    for run_record, resolved_experiment_config in zip(
        run_records,
        resolved_experiment_configs,
        strict=True,
    ):
        overridden_config = json.loads(json.dumps(resolved_experiment_config))
        overridden_config = {
            key: value
            for key, value in overridden_config.items()
            if not str(key).startswith("_")
        }
        for reference_field in [
            "data_config_path",
            "model_config_path",
            "task_config_path",
        ]:
            if reference_field in overridden_config:
                overridden_config[reference_field] = str(
                    normalize_config_path(str(overridden_config[reference_field]))
                )
        overridden_config.setdefault("data_overrides", {})
        overridden_config["data_overrides"]["num_workers"] = int(
            data_num_workers_override
        )
        overridden_config.setdefault("data", {})
        overridden_config["data"]["num_workers"] = int(data_num_workers_override)
        generated_config_path = (
            generated_configs_dir / f"{run_record['run_id'].replace(':', '__')}.yaml"
        )
        generated_config_path.write_text(
            yaml.safe_dump(overridden_config, sort_keys=False),
            encoding="utf-8",
        )
        overridden_configs.append(overridden_config)
        overridden_runtime_config = dict(overridden_config)
        overridden_runtime_config["_original_config_path"] = str(
            run_record["config_path"]
        )
        overridden_runtime_config["_data_num_workers_override"] = int(
            data_num_workers_override
        )
        overridden_run_records.append(
            _build_run_record(
                config_path=generated_config_path,
                resolved_experiment_config=overridden_runtime_config,
                run_stage=str(run_record["run_stage"]),
            )
        )
    return overridden_run_records, overridden_configs


def build_comparative_run_plan(
    *,
    config_paths: list[str | Path],
    smoke_config_paths: list[str | Path],
    report_dir: str | Path,
    data_num_workers_override: int | None = None,
) -> dict[str, Any]:
    smoke_runs, smoke_configs = _load_run_records(smoke_config_paths, run_stage="smoke")
    main_runs, main_configs = _load_run_records(config_paths, run_stage="main")

    resolved_report_dir = Path(report_dir)
    if not resolved_report_dir.is_absolute():
        resolved_report_dir = REPOSITORY_ROOT / resolved_report_dir
    resolved_report_dir.mkdir(parents=True, exist_ok=True)

    if data_num_workers_override is not None:
        smoke_runs, smoke_configs = _materialize_worker_override_configs(
            run_records=smoke_runs,
            resolved_experiment_configs=smoke_configs,
            report_dir=resolved_report_dir,
            data_num_workers_override=data_num_workers_override,
        )
        main_runs, main_configs = _materialize_worker_override_configs(
            run_records=main_runs,
            resolved_experiment_configs=main_configs,
            report_dir=resolved_report_dir,
            data_num_workers_override=data_num_workers_override,
        )

    all_runs = smoke_runs + main_runs
    resolved_experiment_configs = smoke_configs + main_configs
    validate_unique_artifact_paths(resolved_experiment_configs)
    validate_dataset_roots(resolved_experiment_configs)

    run_plan = {
        "created_at_utc": _utc_now_iso(),
        "report_dir": str(resolved_report_dir),
        "manifest_path": str(resolved_report_dir / "comparative_manifest.json"),
        "execution_report_path": str(
            resolved_report_dir / "comparative_execution_report.json"
        ),
        "data_num_workers_override": data_num_workers_override,
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


def _iter_run_groups(
    run_plan: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("smoke", list(run_plan["smoke_runs"])),
        ("main", list(run_plan["main_runs"])),
    ]


def _load_existing_execution_report(
    execution_report_path: str | Path,
) -> dict[str, Any] | None:
    report_path = Path(execution_report_path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def _run_has_required_artifacts(run_record: dict[str, Any]) -> bool:
    checkpoint_path = _normalize_artifact_path(run_record["checkpoint_dir"]) / "best.pt"
    metrics_path = _normalize_artifact_path(run_record["evaluation_metrics_path"])
    records_path = _normalize_artifact_path(run_record["evaluation_records_path"])
    curves_path = _normalize_artifact_path(run_record["evaluation_curves_path"])
    if str(run_record["stage_family"]) == "baseline_single_stage":
        return all(
            path.exists()
            for path in [checkpoint_path, metrics_path, records_path, curves_path]
        )

    three_stage_execution_report_path = (
        _normalize_artifact_path(run_record["output_dir"])
        / "three_stage"
        / "three_stage_execution_report.json"
    )
    return all(
        path.exists()
        for path in [
            checkpoint_path,
            metrics_path,
            records_path,
            curves_path,
            three_stage_execution_report_path,
        ]
    )


def execute_comparative_run_plan(
    run_plan: dict[str, Any],
    *,
    dry_run: bool,
    skip_completed: bool = False,
) -> dict[str, Any]:
    existing_execution_report = (
        _load_existing_execution_report(run_plan["execution_report_path"])
        if skip_completed and not dry_run
        else None
    )
    existing_completed_run_ids = set(
        existing_execution_report.get("completed_run_ids", [])
        if existing_execution_report is not None
        else []
    )
    execution_report = {
        "started_at_utc": _utc_now_iso(),
        "manifest_path": run_plan["manifest_path"],
        "execution_report_path": run_plan["execution_report_path"],
        "dry_run": dry_run,
        "completed_run_ids": [],
        "executed_run_ids": [],
        "skipped_run_ids": [],
        "skip_completed": skip_completed,
        "resumed_from_existing_report": existing_execution_report is not None,
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
                if (
                    skip_completed
                    and run_record["run_id"] in existing_completed_run_ids
                    and _run_has_required_artifacts(run_record)
                ):
                    execution_report["completed_run_ids"].append(run_record["run_id"])
                    execution_report["skipped_run_ids"].append(run_record["run_id"])
                    continue
                execution_report["executed_run_ids"].append(run_record["run_id"])
                for command in run_record["commands"]:
                    command_to_run = list(command)
                    if (
                        skip_completed
                        and str(run_record["stage_family"]) == "thesis_three_stage"
                        and "--skip-completed" not in command_to_run
                    ):
                        command_to_run.append("--skip-completed")
                    subprocess.run(command_to_run, cwd=REPOSITORY_ROOT, check=True)
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
        data_num_workers_override=args.data_num_workers_override,
    )
    _print_run_records(run_plan["smoke_runs"], stage_name="smoke")
    _print_run_records(run_plan["main_runs"], stage_name="main")

    if args.preflight_only:
        return
    execute_comparative_run_plan(
        run_plan,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
    )


if __name__ == "__main__":
    main()
