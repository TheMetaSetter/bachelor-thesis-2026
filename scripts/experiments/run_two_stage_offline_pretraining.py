from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from scripts.cli.train import (
    build_model_from_experiment_config,
    register_runtime_components,
)
from src.core.config import (
    TWO_STAGE_A_EPOCHS_KEY,
    TWO_STAGE_B_EPOCHS_KEY,
    load_experiment_config,
)
from src.core.console import console_print
from src.core.registry import build_dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TWO_STAGE_A_PHASE_NAME = "stage_a_multitask_pretraining"
TWO_STAGE_B_PHASE_NAME = "stage_b_fusion_finetuning"
TWO_STAGE_PHASE_FIELD_ORDER: list[tuple[str, str]] = [
    (TWO_STAGE_A_PHASE_NAME, TWO_STAGE_A_EPOCHS_KEY),
    (TWO_STAGE_B_PHASE_NAME, TWO_STAGE_B_EPOCHS_KEY),
]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def compute_two_stage_total_training_epochs(two_stage_config: dict[str, Any]) -> int:
    return sum(
        int(two_stage_config[field_name])
        for _, field_name in TWO_STAGE_PHASE_FIELD_ORDER
    )


def validate_two_stage_epoch_budget(experiment_config: dict[str, Any]) -> None:
    if "two_stage" not in experiment_config:
        raise ValueError("Experiment config must define two_stage for this runner")

    two_stage_config = experiment_config["two_stage"]
    computed_total_training_epochs = compute_two_stage_total_training_epochs(
        two_stage_config
    )
    expected_total_training_epochs = int(
        two_stage_config["expected_total_training_epochs"]
    )
    if computed_total_training_epochs != expected_total_training_epochs:
        raise ValueError(
            "Two-stage training epochs must sum to expected_total_training_epochs. "
            f"Got total={computed_total_training_epochs}, "
            f"expected_total_training_epochs={expected_total_training_epochs}."
        )
    if int(experiment_config["epochs"]) != expected_total_training_epochs:
        raise ValueError(
            "Experiment epochs must match two_stage.expected_total_training_epochs. "
            f"Got epochs={experiment_config['epochs']}, "
            f"expected_total_training_epochs={expected_total_training_epochs}."
        )


def build_two_stage_training_plan(
    experiment_config: dict[str, Any],
) -> list[dict[str, Any]]:
    # Offline pre-training is the large phase.
    # Stage A and Stage B are the two stages inside that phase.

    # In this method, there will be also online test-time adaptation phase,
    # after offline pre-training.

    validate_two_stage_epoch_budget(experiment_config)
    two_stage_config = experiment_config["two_stage"]
    training_plan: list[dict[str, Any]] = []
    current_global_epoch_start = 1
    for stage_name, field_name in TWO_STAGE_PHASE_FIELD_ORDER:
        stage_epochs = int(two_stage_config[field_name])
        stage_record = {
            "stage_name": stage_name,
            "epochs": stage_epochs,
            "global_epoch_start": current_global_epoch_start,
            "global_epoch_end": current_global_epoch_start + stage_epochs - 1,
        }
        training_plan.append(stage_record)
        current_global_epoch_start += stage_epochs
    return training_plan


def _to_stage_output_dir(base_output_dir: str, stage_name: str) -> str:
    return str(Path(base_output_dir) / "two_stage" / stage_name)


def _resolve_repo_config_reference(config_reference: str) -> str:
    reference_path = Path(config_reference)
    if reference_path.is_absolute():
        return str(reference_path)
    return str((REPOSITORY_ROOT / reference_path).resolve())


def _build_stage_experiment_config(
    experiment_config: dict[str, Any],
    stage_record: dict[str, Any],
) -> dict[str, Any]:
    stage_config = copy.deepcopy(experiment_config)
    stage_name = str(stage_record["stage_name"])
    stage_config["experiment_name"] = (
        f"{experiment_config['experiment_name']}__{stage_name}"
    )
    stage_output_dir = Path(
        _to_stage_output_dir(str(experiment_config["output_dir"]), stage_name)
    )
    stage_config["output_dir"] = str(stage_output_dir)
    stage_config["checkpoint_dir"] = str(stage_output_dir / "checkpoints")
    stage_config["epochs"] = int(stage_record["epochs"])
    for reference_field in [
        "data_config_path",
        "model_config_path",
        "task_config_path",
    ]:
        if reference_field in stage_config:
            stage_config[reference_field] = _resolve_repo_config_reference(
                str(stage_config[reference_field])
            )
    stage_config.pop("two_stage_phase", None)
    stage_config.pop("two_stage_global_epoch_start", None)
    stage_config.pop("two_stage_global_epoch_end", None)
    stage_config["stage_name"] = stage_name
    stage_config["stage_global_epoch_start"] = int(stage_record["global_epoch_start"])
    stage_config["stage_global_epoch_end"] = int(stage_record["global_epoch_end"])
    stage_config["model"].pop("training_phase", None)
    stage_config["model"]["stage_name"] = stage_name
    model_overrides = copy.deepcopy(stage_config.get("model_overrides", {}))
    model_overrides["training_phase"] = stage_name
    model_overrides["stage_name"] = stage_name
    stage_config["model_overrides"] = model_overrides
    logging_config = copy.deepcopy(stage_config.get("logging", {}))
    logging_config["wandb_job_type"] = stage_name
    logging_config["wandb_run_name"] = stage_config["experiment_name"]
    stage_config["logging"] = logging_config
    if stage_name == TWO_STAGE_B_PHASE_NAME:
        stage_config["initialization_checkpoint_path"] = str(
            Path(str(experiment_config["output_dir"]))
            / "two_stage"
            / "initializations"
            / "stage_b_init.pt"
        )
    return stage_config


def _stage_manifest_root(experiment_config: dict[str, Any]) -> Path:
    return Path(str(experiment_config["output_dir"])) / "two_stage"


def materialize_two_stage_run_manifest(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    training_plan = build_two_stage_training_plan(experiment_config)
    manifest_root = _stage_manifest_root(experiment_config)
    generated_configs_dir = manifest_root / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    stage_records: list[dict[str, Any]] = []
    for stage_index, stage_record in enumerate(training_plan, start=1):
        stage_name = str(stage_record["stage_name"])
        stage_config = _build_stage_experiment_config(experiment_config, stage_record)
        stage_config_path = (
            generated_configs_dir / f"{stage_index:02d}_{stage_name}.yaml"
        )
        with stage_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage_config, handle, sort_keys=False)
        stage_records.append(
            {
                **stage_record,
                "config_path": str(stage_config_path),
                "checkpoint_dir": stage_config["checkpoint_dir"],
                "best_checkpoint_path": str(
                    Path(stage_config["checkpoint_dir"]) / "best.pt"
                ),
                "initialization_checkpoint_path": stage_config.get(
                    "initialization_checkpoint_path"
                ),
            }
        )

    evaluation = {
        "config_path": stage_records[-1]["config_path"],
        "checkpoint_path": stage_records[-1]["best_checkpoint_path"],
    }
    manifest = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "manifest_version": 1,
        "created_at_utc": _utc_now_iso(),
        "manifest_root": str(manifest_root),
        "training_stages": stage_records,
        "total_training_epochs": compute_two_stage_total_training_epochs(
            experiment_config["two_stage"]
        ),
        "evaluation": evaluation,
    }
    manifest_path = manifest_root / "two_stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location="cpu")


def _load_stage_a_state_into_stage_b_model(
    *,
    model: torch.nn.Module,
    stage_a_state_dict: dict[str, Any],
) -> None:
    load_result = model.load_state_dict(stage_a_state_dict, strict=False)
    allowed_unexpected_keys = {
        "discrete_assignment.weight",
        "discrete_assignment.bias",
    }
    unexpected_keys = set(load_result.unexpected_keys)
    missing_keys = set(load_result.missing_keys)
    if unexpected_keys - allowed_unexpected_keys or missing_keys:
        raise RuntimeError(
            "Unexpected checkpoint mismatch while preparing Stage B initialization: "
            f"missing_keys={sorted(missing_keys)}, "
            f"unexpected_keys={sorted(unexpected_keys)}"
        )


def _prepare_stage_b_initialization_checkpoint(manifest: dict[str, Any]) -> Path:
    stage_a_record = manifest["training_stages"][0]
    stage_b_record = manifest["training_stages"][1]
    stage_b_config = load_experiment_config(stage_b_record["config_path"])
    stage_a_checkpoint_path = Path(str(stage_a_record["best_checkpoint_path"]))
    initialization_checkpoint_path = Path(
        str(stage_b_record["initialization_checkpoint_path"])
    )
    initialization_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    register_runtime_components()
    model = build_model_from_experiment_config(stage_b_config)
    stage_a_checkpoint = _load_checkpoint_payload(stage_a_checkpoint_path)
    _load_stage_a_state_into_stage_b_model(
        model=model,
        stage_a_state_dict=stage_a_checkpoint["model_state_dict"],
    )
    if hasattr(model, "load_checkpoint_extra_state"):
        model.load_checkpoint_extra_state(stage_a_checkpoint.get("extra_state"))

    initialization_data_config = copy.deepcopy(stage_b_config["data"])
    initialization_data_config["shuffle_train"] = False
    initialization_data_config["num_workers"] = 0
    data_bundle = build_dataset(
        initialization_data_config["dataset_name"], initialization_data_config
    )
    init_device = str(stage_b_config.get("device", "cpu"))
    if init_device == "cuda" and not torch.cuda.is_available():
        init_device = "cpu"
    model.to(init_device)
    if not model.maybe_initialize_memories_from_loader(
        train_loader=data_bundle["loaders"]["train"],
        device=init_device,
    ):
        raise RuntimeError(
            "Stage B initialization checkpoint could not initialize memories"
        )

    initialization_payload = dict(stage_a_checkpoint)
    initialization_payload["model_state_dict"] = model.state_dict()
    if hasattr(model, "get_checkpoint_extra_state"):
        initialization_payload["extra_state"] = model.get_checkpoint_extra_state()
    initialization_payload["config"] = stage_b_config
    torch.save(initialization_payload, initialization_checkpoint_path)
    return initialization_checkpoint_path


def build_two_stage_execution_commands(manifest: dict[str, Any]) -> dict[str, Any]:
    training_commands: list[list[str]] = []
    for stage_record in manifest["training_stages"]:
        training_commands.append(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--experiment-config",
                str(stage_record["config_path"]),
            ]
        )
    evaluation_command = [
        sys.executable,
        "-m",
        "scripts.evaluate",
        "--experiment-config",
        str(manifest["evaluation"]["config_path"]),
        "--checkpoint-path",
        str(manifest["evaluation"]["checkpoint_path"]),
    ]
    return {"training": training_commands, "evaluation": evaluation_command}


def execute_two_stage_plan(
    manifest: dict[str, Any],
    dry_run: bool = False,
    skip_completed: bool = False,
) -> dict[str, Any]:
    command_plan = build_two_stage_execution_commands(manifest)
    manifest_root = Path(str(manifest["manifest_root"]))
    execution_report_path = manifest_root / "two_stage_execution_report.json"
    started_at_utc = _utc_now_iso()
    executed_stage_names: list[str] = []
    completed_stage_names: list[str] = []
    skipped_stage_names: list[str] = []
    existing_execution_report: dict[str, Any] | None = None

    if skip_completed and not dry_run and execution_report_path.exists():
        existing_execution_report = json.loads(
            execution_report_path.read_text(encoding="utf-8")
        )
        if str(existing_execution_report.get("status")) == "completed":
            return {
                "manifest_path": str(manifest_root / "two_stage_manifest.json"),
                "execution_report_path": str(execution_report_path),
                "started_at_utc": started_at_utc,
                "finished_at_utc": _utc_now_iso(),
                "dry_run": dry_run,
                "skip_completed": skip_completed,
                "resumed_from_existing_report": True,
                "status": "skipped_completed",
                "executed_stage_names": [],
                "completed_stage_names": list(
                    existing_execution_report.get("completed_stage_names", [])
                ),
                "skipped_stage_names": list(
                    existing_execution_report.get("completed_stage_names", [])
                ),
                "stage_b_initialization_checkpoint_path": str(
                    manifest_root / "initializations" / "stage_b_init.pt"
                ),
                "evaluation_checkpoint_path": str(
                    manifest["evaluation"]["checkpoint_path"]
                ),
            }

    training_stage_records = list(manifest["training_stages"])
    training_commands = list(command_plan["training"])

    if dry_run:
        for stage_record in training_stage_records:
            executed_stage_names.append(str(stage_record["stage_name"]))
            completed_stage_names.append(str(stage_record["stage_name"]))
    else:
        subprocess.run(training_commands[0], check=True)
        first_stage_name = str(training_stage_records[0]["stage_name"])
        executed_stage_names.append(first_stage_name)
        completed_stage_names.append(first_stage_name)
        _prepare_stage_b_initialization_checkpoint(manifest)
        for stage_record, command in zip(
            training_stage_records[1:],
            training_commands[1:],
            strict=True,
        ):
            stage_name = str(stage_record["stage_name"])
            executed_stage_names.append(stage_name)
            completed_stage_names.append(stage_name)
            subprocess.run(command, check=True)

    executed_stage_names.append("evaluation")
    completed_stage_names.append("evaluation")
    if not dry_run:
        subprocess.run(command_plan["evaluation"], check=True)

    execution_report = {
        "manifest_path": str(manifest_root / "two_stage_manifest.json"),
        "execution_report_path": str(execution_report_path),
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "dry_run": dry_run,
        "skip_completed": skip_completed,
        "resumed_from_existing_report": existing_execution_report is not None,
        "status": "dry_run" if dry_run else "completed",
        "executed_stage_names": executed_stage_names,
        "completed_stage_names": completed_stage_names,
        "skipped_stage_names": skipped_stage_names,
        "stage_b_initialization_checkpoint_path": str(
            manifest_root / "initializations" / "stage_b_init.pt"
        ),
        "evaluation_checkpoint_path": str(manifest["evaluation"]["checkpoint_path"]),
    }
    execution_report_path.write_text(
        json.dumps(execution_report, indent=2), encoding="utf-8"
    )
    return execution_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-stage offline pretraining thesis experiment"
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_config)
    manifest = materialize_two_stage_run_manifest(experiment_config)
    execution_report = execute_two_stage_plan(
        manifest,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
    )
    console_print(
        "TWO_STAGE",
        "Completed two-stage orchestration",
        dry_run=execution_report["dry_run"],
        manifest_path=execution_report["manifest_path"],
    )


if __name__ == "__main__":
    main()
