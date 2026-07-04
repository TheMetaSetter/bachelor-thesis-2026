from __future__ import annotations

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

from scripts.train import (
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


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
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
    return sum(int(two_stage_config[field_name]) for _, field_name in TWO_STAGE_PHASE_FIELD_ORDER)


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
    validate_two_stage_epoch_budget(experiment_config)
    two_stage_config = experiment_config["two_stage"]
    training_plan: list[dict[str, Any]] = []
    current_global_epoch_start = 1
    for phase_name, field_name in TWO_STAGE_PHASE_FIELD_ORDER:
        phase_epochs = int(two_stage_config[field_name])
        phase_record = {
            "phase_name": phase_name,
            "epochs": phase_epochs,
            "global_epoch_start": current_global_epoch_start,
            "global_epoch_end": current_global_epoch_start + phase_epochs - 1,
        }
        training_plan.append(phase_record)
        current_global_epoch_start += phase_epochs
    return training_plan


def _to_stage_output_dir(base_output_dir: str, phase_name: str) -> str:
    return str(Path(base_output_dir) / "two_stage" / phase_name)


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
    stage_config["experiment_name"] = f"{experiment_config['experiment_name']}__{phase_name}"
    stage_output_dir = Path(
        _to_stage_output_dir(str(experiment_config["output_dir"]), phase_name)
    )
    stage_config["output_dir"] = str(stage_output_dir)
    stage_config["checkpoint_dir"] = str(stage_output_dir / "checkpoints")
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
    stage_config["two_stage_phase"] = phase_name
    stage_config["two_stage_global_epoch_start"] = int(
        phase_record["global_epoch_start"]
    )
    stage_config["two_stage_global_epoch_end"] = int(phase_record["global_epoch_end"])
    stage_config["model"]["training_phase"] = phase_name
    logging_config = copy.deepcopy(stage_config.get("logging", {}))
    logging_config["wandb_job_type"] = phase_name
    logging_config["wandb_run_name"] = stage_config["experiment_name"]
    stage_config["logging"] = logging_config
    if phase_name == TWO_STAGE_B_PHASE_NAME:
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

    training_stages: list[dict[str, Any]] = []
    for phase_index, phase_record in enumerate(training_plan, start=1):
        phase_name = str(phase_record["phase_name"])
        stage_config = _build_stage_experiment_config(experiment_config, phase_record)
        stage_config_path = (
            generated_configs_dir / f"{phase_index:02d}_{phase_name}.yaml"
        )
        with stage_config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(stage_config, handle, sort_keys=False)
        training_stages.append(
            {
                **phase_record,
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
        "config_path": training_stages[-1]["config_path"],
        "checkpoint_path": training_stages[-1]["best_checkpoint_path"],
    }
    manifest = {
        "experiment_name": str(experiment_config["experiment_name"]),
        "manifest_version": 1,
        "created_at_utc": _utc_now_iso(),
        "manifest_root": str(manifest_root),
        "training_stages": training_stages,
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
    model.load_state_dict(stage_a_checkpoint["model_state_dict"])
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
        train_loader=data_bundle["train_loader"],
        device=init_device,
    ):
        raise RuntimeError("Stage B initialization checkpoint could not initialize memories")

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
    return {"training": training_commands, "evaluation": evaluation_command}


def execute_two_stage_plan(manifest: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    command_plan = build_two_stage_execution_commands(manifest)
    manifest_root = Path(str(manifest["manifest_root"]))
    execution_report_path = manifest_root / "two_stage_execution_report.json"
    started_at_utc = _utc_now_iso()
    executed_stage_names: list[str] = []

    if not dry_run:
        _prepare_stage_b_initialization_checkpoint(manifest)

    for stage_record, command in zip(
        manifest["training_stages"],
        command_plan["training"],
        strict=True,
    ):
        executed_stage_names.append(str(stage_record["phase_name"]))
        if dry_run:
            continue
        subprocess.run(command, check=True)

    executed_stage_names.append("evaluation")
    if not dry_run:
        subprocess.run(command_plan["evaluation"], check=True)

    execution_report = {
        "manifest_path": str(manifest_root / "two_stage_manifest.json"),
        "execution_report_path": str(execution_report_path),
        "started_at_utc": started_at_utc,
        "finished_at_utc": _utc_now_iso(),
        "dry_run": dry_run,
        "executed_stage_names": executed_stage_names,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_config = load_experiment_config(args.experiment_config)
    manifest = materialize_two_stage_run_manifest(experiment_config)
    execution_report = execute_two_stage_plan(manifest, dry_run=args.dry_run)
    console_print(
        "TWO_STAGE",
        "Completed two-stage orchestration",
        dry_run=execution_report["dry_run"],
        manifest_path=execution_report["manifest_path"],
    )


if __name__ == "__main__":
    main()
