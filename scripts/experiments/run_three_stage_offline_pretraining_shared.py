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

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
import yaml

from scripts.cli.train import (
    build_model_from_experiment_config,
    register_runtime_components,
)
from src.core.config import (
    STAGE3_WARMUP_EPOCHS_CANONICAL_KEY,
    STAGE3_WARMUP_EPOCHS_LEGACY_KEY,
)
from src.core.config import load_experiment_config
from src.core.console import console_print
from src.core.registry import build_dataset

# Legacy phase label kept only for compatibility with older artifacts.
STAGE3_PHASE_LEGACY_NAME = "stage3_prototype_warmup"
STAGE3_PHASE_CANONICAL_NAME = "stage3_memory_initialization_and_fusion_warmup"

THREE_STAGE_PHASE_FIELD_ORDER: list[tuple[str, str]] = [
    ("stage1_classification", "stage1_classification_epochs"),
    ("stage1_reconstruction", "stage1_reconstruction_epochs"),
    ("stage2_recovery", "stage2_recovery_epochs"),
    (STAGE3_PHASE_CANONICAL_NAME, STAGE3_WARMUP_EPOCHS_CANONICAL_KEY),
    ("multitask_pretraining", "multitask_pretraining_epochs"),
]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ZIPPING_ACTIVATION_BATCHES = 8
STATISTICAL_PROCEDURE_NAMES = [
    "stage2_mtz_parameter_zipping",
    "stage3_memory_initialization",
]


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _build_semantic_stage_metadata(phase_name: str) -> dict[str, Any]:
    if phase_name == STAGE3_PHASE_CANONICAL_NAME:
        return {
            "semantic_stage_label": "Stage 3: Memory Initialization and Fusion Warm-Up",
            "memory_initialization_substep": True,
            "fusion_warmup_substep": True,
        }
    return {
        "semantic_stage_label": phase_name,
        "memory_initialization_substep": False,
        "fusion_warmup_substep": False,
    }


def compute_three_stage_total_training_epochs(
    three_stage_config: dict[str, Any],
) -> int:
    return sum(
        int(three_stage_config[field_name])
        for _, field_name in THREE_STAGE_PHASE_FIELD_ORDER
    )


def _optimizer_training_phase_names() -> list[str]:
    return [phase_name for phase_name, _ in THREE_STAGE_PHASE_FIELD_ORDER]


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
            exact_budget_suffix = (
                " The main three-stage experiment must total exactly 300 epochs."
            )
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
        raise ValueError(
            "The main three-stage experiment must total exactly 300 epochs"
        )


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
    if phase_name == STAGE3_PHASE_CANONICAL_NAME:
        return str(three_stage_root / "stage2_recovery" / "checkpoints" / "best.pt")
    if phase_name == "multitask_pretraining":
        return str(
            three_stage_root / STAGE3_PHASE_CANONICAL_NAME / "checkpoints" / "best.pt"
        )
    return None


def _resolve_repo_config_reference(config_reference: str) -> str:
    reference_path = Path(config_reference)
    if reference_path.is_absolute():
        return str(reference_path)
    return str((REPOSITORY_ROOT / reference_path).resolve())


