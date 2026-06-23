from __future__ import annotations

"""Preflight and plan builder for three-stage offline pre-training.

This first slice formalizes the stage contract, validates the exact training
epoch budget, and exposes a deterministic phase plan that later execution code
can build on without re-encoding the schedule in multiple places.
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml

from scripts.train import build_model_from_experiment_config, register_runtime_components
from src.core.config import load_experiment_config
from src.core.console import console_print


THREE_STAGE_PHASE_FIELD_ORDER: list[tuple[str, str]] = [
    ("stage1_classification", "stage1_classification_epochs"),
    ("stage1_reconstruction", "stage1_reconstruction_epochs"),
    ("stage2_recovery", "stage2_recovery_epochs"),
    ("stage3_prototype_warmup", "stage3_prototype_warmup_epochs"),
    ("multitask_pretraining", "multitask_pretraining_epochs"),
]

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def compute_three_stage_total_training_epochs(three_stage_config: dict[str, Any]) -> int:
    return sum(
        int(three_stage_config[field_name])
        for _, field_name in THREE_STAGE_PHASE_FIELD_ORDER
    )


def validate_three_stage_epoch_budget(experiment_config: dict[str, Any]) -> None:
    if "three_stage" not in experiment_config:
        raise ValueError("Experiment config must define three_stage for this runner")

    three_stage_config = experiment_config["three_stage"]
    computed_total_training_epochs = compute_three_stage_total_training_epochs(
        three_stage_config
    )
    expected_total_training_epochs = int(
        three_stage_config["expected_total_training_epochs"]
    )
    if computed_total_training_epochs != expected_total_training_epochs:
        exact_budget_suffix = ""
        if expected_total_training_epochs == 300:
            exact_budget_suffix = " The main three-stage experiment must total exactly 300 epochs."
        raise ValueError(
            "Three-stage training epochs must sum to expected_total_training_epochs. "
            f"Got total={computed_total_training_epochs}, "
            f"expected_total_training_epochs={expected_total_training_epochs}."
            f"{exact_budget_suffix}"
        )
    if int(experiment_config["epochs"]) != expected_total_training_epochs:
        raise ValueError(
            "Experiment epochs must match three_stage.expected_total_training_epochs. "
            f"Got epochs={experiment_config['epochs']}, "
            "expected_total_training_epochs="
            f"{expected_total_training_epochs}."
        )
    if expected_total_training_epochs == 300 and computed_total_training_epochs != 300:
        raise ValueError("The main three-stage experiment must total exactly 300 epochs")


def build_three_stage_training_plan(
    experiment_config: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_three_stage_epoch_budget(experiment_config)
    three_stage_config = experiment_config["three_stage"]
    training_plan: list[dict[str, Any]] = []
    current_global_epoch_start = 1
    for phase_name, field_name in THREE_STAGE_PHASE_FIELD_ORDER:
        phase_epochs = int(three_stage_config[field_name])
        phase_record = {
            "phase_name": phase_name,
            "epochs": phase_epochs,
            "global_epoch_start": current_global_epoch_start,
            "global_epoch_end": current_global_epoch_start + phase_epochs - 1,
        }
        training_plan.append(phase_record)
        current_global_epoch_start += phase_epochs
    return training_plan


def _to_stage_experiment_name(base_experiment_name: str, phase_name: str) -> str:
    return f"{base_experiment_name}__{phase_name}"


def _to_stage_output_dir(base_output_dir: str, phase_name: str) -> str:
    return str(Path(base_output_dir) / "three_stage" / phase_name)


def _to_stage_initialization_checkpoint_path(
    base_output_dir: str,
    phase_name: str,
) -> str | None:
    three_stage_root = Path(base_output_dir) / "three_stage"
    if phase_name == "stage2_recovery":
        return str(three_stage_root / "initializations" / "stage2_recovery_init.pt")
    if phase_name == "stage3_prototype_warmup":
        return str(three_stage_root / "stage2_recovery" / "checkpoints" / "best.pt")
    if phase_name == "multitask_pretraining":
        return str(
            three_stage_root / "stage3_prototype_warmup" / "checkpoints" / "best.pt"
        )
    return None


def _resolve_repo_config_reference(config_reference: str) -> str:
    reference_path = Path(config_reference)
    if reference_path.is_absolute():
        return str(reference_path)
    return str((REPOSITORY_ROOT / reference_path).resolve())


def _build_stage_experiment_config(
    experiment_config: dict[str, Any],
    phase_record: dict[str, Any],
) -> dict[str, Any]:
    stage_config = copy.deepcopy(experiment_config)
    phase_name = str(phase_record["phase_name"])
    stage_experiment_name = _to_stage_experiment_name(
        str(experiment_config["experiment_name"]),
        phase_name,
    )
    stage_output_dir = Path(_to_stage_output_dir(str(experiment_config["output_dir"]), phase_name))
    stage_checkpoint_dir = stage_output_dir / "checkpoints"
    stage_config["experiment_name"] = stage_experiment_name
    stage_config["output_dir"] = str(stage_output_dir)
    stage_config["checkpoint_dir"] = str(stage_checkpoint_dir)
    stage_config["epochs"] = int(phase_record["epochs"])
    for reference_field in [
        "data_config_path",
        "model_config_path",
        "task_config_path",
    ]:
        if reference_field in stage_config:
            stage_config[reference_field] = _resolve_repo_config_reference(
                str(stage_config[reference_field])
            )
    stage_config["three_stage_phase"] = phase_name
    stage_config["three_stage_global_epoch_start"] = int(
        phase_record["global_epoch_start"]
    )
    stage_config["three_stage_global_epoch_end"] = int(phase_record["global_epoch_end"])
    model_overrides = copy.deepcopy(stage_config.get("model_overrides", {}))
    model_overrides["training_phase"] = phase_name
    model_overrides["freeze_memories_after_initialization"] = bool(
        stage_config["three_stage"]["freeze_memories_after_initialization"]
    )
    model_overrides["freeze_recovered_zipped_encoder_during_warmup"] = bool(
        stage_config["three_stage"]["freeze_recovered_zipped_encoder_during_warmup"]
    )
    model_overrides["discrete_memory_label_source"] = str(
        stage_config["three_stage"]["discrete_memory_label_source"]
    )
    stage_config["model_overrides"] = model_overrides
    initialization_checkpoint_path = _to_stage_initialization_checkpoint_path(
        str(experiment_config["output_dir"]),
        phase_name,
    )
    if initialization_checkpoint_path is not None:
        stage_config["initialization_checkpoint_path"] = initialization_checkpoint_path
    logging_config = copy.deepcopy(stage_config.get("logging", {}))
    logging_config["wandb_job_type"] = phase_name
    logging_config["wandb_run_name"] = stage_experiment_name
    stage_config["logging"] = logging_config
    return stage_config


def _stage_manifest_root(experiment_config: dict[str, Any]) -> Path:
    return Path(str(experiment_config["output_dir"])) / "three_stage"


def materialize_three_stage_run_manifest(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    training_plan = build_three_stage_training_plan(experiment_config)
    manifest_root = _stage_manifest_root(experiment_config)
    generated_configs_dir = manifest_root / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    training_stages: list[dict[str, Any]] = []
    for phase_index, phase_record in enumerate(training_plan, start=1):
        phase_name = str(phase_record["phase_name"])
        stage_config = _build_stage_experiment_config(experiment_config, phase_record)
        stage_config_path = generated_configs_dir / f"{phase_index:02d}_{phase_name}.yaml"
        with stage_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage_config, handle, sort_keys=False)
        training_stages.append(
            {
                "phase_name": phase_name,
                "epochs": int(phase_record["epochs"]),
                "global_epoch_start": int(phase_record["global_epoch_start"]),
                "global_epoch_end": int(phase_record["global_epoch_end"]),
                "config_path": str(stage_config_path),
                "output_dir": stage_config["output_dir"],
                "checkpoint_dir": stage_config["checkpoint_dir"],
                "initialization_checkpoint_path": stage_config.get(
                    "initialization_checkpoint_path"
                ),
                "best_checkpoint_path": str(
                    Path(stage_config["checkpoint_dir"]) / "best.pt"
                ),
            }
        )

    evaluation_config_path = generated_configs_dir / "06_evaluation_reference.yaml"
    with evaluation_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(experiment_config, handle, sort_keys=False)

    manifest = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "total_training_epochs": compute_three_stage_total_training_epochs(
            experiment_config["three_stage"]
        ),
        "training_stages": training_stages,
        "evaluation": {
            "config_path": str(evaluation_config_path),
            "checkpoint_path": training_stages[-1]["best_checkpoint_path"],
        },
    }
    manifest_path = manifest_root / "three_stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_three_stage_execution_commands(
    manifest: dict[str, Any],
) -> dict[str, list[list[str]] | list[str]]:
    training_commands: list[list[str]] = []
    for stage_record in manifest["training_stages"]:
        training_commands.append(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "train.py"),
                "--experiment-config",
                str(stage_record["config_path"]),
            ]
        )
    evaluation_command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "evaluate.py"),
        "--experiment-config",
        str(manifest["evaluation"]["config_path"]),
        "--checkpoint-path",
        str(manifest["evaluation"]["checkpoint_path"]),
    ]
    return {
        "training": training_commands,
        "evaluation": evaluation_command,
    }


def _resolve_manifest_root_from_manifest(manifest: dict[str, Any]) -> Path:
    first_stage_config_path = Path(str(manifest["training_stages"][0]["config_path"]))
    return first_stage_config_path.parent.parent


def _prepare_stage2_recovery_initialization_checkpoint(
    manifest: dict[str, Any],
) -> Path:
    stage_records_by_phase = {
        stage_record["phase_name"]: stage_record
        for stage_record in manifest["training_stages"]
    }
    classification_checkpoint_path = Path(
        str(stage_records_by_phase["stage1_classification"]["best_checkpoint_path"])
    )
    reconstruction_checkpoint_path = Path(
        str(stage_records_by_phase["stage1_reconstruction"]["best_checkpoint_path"])
    )
    stage2_record = stage_records_by_phase["stage2_recovery"]
    initialization_checkpoint_path = Path(
        str(stage2_record["initialization_checkpoint_path"])
    )
    initialization_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    classification_checkpoint = torch.load(
        classification_checkpoint_path,
        map_location="cpu",
    )
    reconstruction_checkpoint = torch.load(
        reconstruction_checkpoint_path,
        map_location="cpu",
    )
    stage2_config = load_experiment_config(stage2_record["config_path"])
    register_runtime_components()
    stage2_model = build_model_from_experiment_config(stage2_config)
    zipped_state_dict = stage2_model.state_dict()
    classification_state_dict = classification_checkpoint["model_state_dict"]
    reconstruction_state_dict = reconstruction_checkpoint["model_state_dict"]

    for state_key in list(zipped_state_dict.keys()):
        if state_key.startswith("encoder."):
            zipped_state_dict[state_key] = 0.5 * (
                classification_state_dict[state_key]
                + reconstruction_state_dict[state_key]
            )
        elif state_key.startswith("classification_head."):
            zipped_state_dict[state_key] = classification_state_dict[state_key]
        elif state_key.startswith("reconstruction_head."):
            zipped_state_dict[state_key] = reconstruction_state_dict[state_key]

    stage2_model.load_state_dict(zipped_state_dict)
    initialization_payload = {
        "model_state_dict": stage2_model.state_dict(),
        "optimizer_state_dict": {},
        "scaler_state_dict": classification_checkpoint["scaler_state_dict"],
        "config": stage2_config,
        "epoch": 0,
        "metric_history": [],
        "extra_state": {
            "memory_initialized": False,
            "memory_training_enabled": False,
            "memory_ready_for_initialization": False,
            "memory_initialization_epoch": None,
        },
    }
    torch.save(initialization_payload, initialization_checkpoint_path)
    console_print(
        "THREE_STAGE",
        "Prepared Stage 2 recovery initialization checkpoint",
        initialization_checkpoint_path=initialization_checkpoint_path,
        classification_checkpoint_path=classification_checkpoint_path,
        reconstruction_checkpoint_path=reconstruction_checkpoint_path,
        zipping_approximation="parameter_average_identity_matching",
    )
    return initialization_checkpoint_path


def execute_three_stage_plan(
    manifest: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    command_plan = build_three_stage_execution_commands(manifest)
    manifest_root = _resolve_manifest_root_from_manifest(manifest)
    execution_report_path = manifest_root / "three_stage_execution_report.json"
    planned_stage_names = [
        stage_record["phase_name"] for stage_record in manifest["training_stages"]
    ] + ["evaluation"]
    completed_stage_names: list[str] = []
    executed_stage_names: list[str] = []

    execution_report: dict[str, Any] = {
        "experiment_name": manifest["experiment_name"],
        "dry_run": dry_run,
        "status": "dry_run" if dry_run else "completed",
        "started_at_utc": _utc_now_iso(),
        "planned_stage_names": planned_stage_names,
        "executed_stage_names": planned_stage_names if dry_run else executed_stage_names,
        "completed_stage_names": planned_stage_names if dry_run else completed_stage_names,
        "manifest_path": str(manifest_root / "three_stage_manifest.json"),
        "execution_report_path": str(execution_report_path),
        "server_preflight_summary_path": str(
            manifest_root / "server_preflight_summary.json"
        ),
        "stage2_recovery_initialization_checkpoint_path": str(
            manifest_root / "initializations" / "stage2_recovery_init.pt"
        ),
        "evaluation_checkpoint_path": str(manifest["evaluation"]["checkpoint_path"]),
        "training_commands": command_plan["training"],
        "evaluation_command": command_plan["evaluation"],
    }

    try:
        if not dry_run:
            for stage_record, training_command in zip(
                manifest["training_stages"],
                command_plan["training"],
            ):
                phase_name = str(stage_record["phase_name"])
                executed_stage_names.append(phase_name)
                if phase_name == "stage2_recovery":
                    _prepare_stage2_recovery_initialization_checkpoint(manifest)
                subprocess.run(training_command, cwd=REPOSITORY_ROOT, check=True)
                completed_stage_names.append(phase_name)

            executed_stage_names.append("evaluation")
            subprocess.run(command_plan["evaluation"], cwd=REPOSITORY_ROOT, check=True)
            completed_stage_names.append("evaluation")

        execution_report["executed_stage_names"] = (
            planned_stage_names if dry_run else executed_stage_names
        )
        execution_report["completed_stage_names"] = (
            planned_stage_names if dry_run else completed_stage_names
        )
        execution_report["finished_at_utc"] = _utc_now_iso()
        execution_report_path.write_text(
            json.dumps(execution_report, indent=2),
            encoding="utf-8",
        )
        return execution_report
    except subprocess.CalledProcessError as error:
        execution_report["status"] = "failed"
        execution_report["executed_stage_names"] = list(executed_stage_names)
        execution_report["completed_stage_names"] = list(completed_stage_names)
        execution_report["failed_stage_name"] = (
            executed_stage_names[-1] if executed_stage_names else None
        )
        execution_report["failed_command"] = list(error.cmd)
        execution_report["failed_returncode"] = int(error.returncode)
        execution_report["failed_at_utc"] = _utc_now_iso()
        execution_report_path.write_text(
            json.dumps(execution_report, indent=2),
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight the finalized three-stage offline pre-training plan"
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to the three-stage experiment config",
    )
    parser.add_argument(
        "--print-plan-json",
        action="store_true",
        help="Print the resolved three-stage training plan as JSON",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config and print plan without execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest and command plan without running stage subprocesses",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_config)
    training_plan = build_three_stage_training_plan(experiment_config)
    manifest = materialize_three_stage_run_manifest(experiment_config)
    console_print(
        "THREE_STAGE",
        "Validated three-stage training plan",
        experiment_name=experiment_config["experiment_name"],
        total_training_epochs=compute_three_stage_total_training_epochs(
            experiment_config["three_stage"]
        ),
        phases=[phase["phase_name"] for phase in training_plan],
        manifest_path=_stage_manifest_root(experiment_config) / "three_stage_manifest.json",
    )
    if args.print_plan_json:
        print(json.dumps(manifest, indent=2))
    if not args.preflight_only:
        execution_report = execute_three_stage_plan(manifest, dry_run=args.dry_run)
        console_print(
            "THREE_STAGE",
            "Completed three-stage execution orchestration",
            dry_run=execution_report["dry_run"],
            executed_stage_names=execution_report["executed_stage_names"],
            execution_report_path=_resolve_manifest_root_from_manifest(manifest)
            / "three_stage_execution_report.json",
        )


if __name__ == "__main__":
    main()
