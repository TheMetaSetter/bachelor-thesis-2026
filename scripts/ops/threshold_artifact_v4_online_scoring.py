"""Resolve the frozen A0 scoring config used by V4 threshold recalibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_experiment_config


DEFAULT_ONLINE_EXPERIMENT_CONFIG_DIRECTORY = Path(
    "configs/experiment/online_benchmark/thesis"
)


@dataclass(frozen=True)
class StageBInventoryEntry:
    experiment_config_path: Path
    offline_variant: str
    entity_id: str
    seed: int
    threshold_artifact_v3_path: Path
    stage_b_best_checkpoint_path: Path
    threshold_artifact_v4_path: Path
    audit_path: Path


def entry_as_report_value(entry: StageBInventoryEntry) -> dict[str, str | int]:
    return {
        "experiment_config_path": str(entry.experiment_config_path),
        "offline_variant": entry.offline_variant,
        "entity_id": entry.entity_id,
        "seed": entry.seed,
        "threshold_artifact_v3_path": str(entry.threshold_artifact_v3_path),
        "stage_b_best_checkpoint_path": str(entry.stage_b_best_checkpoint_path),
        "threshold_artifact_v4_path": str(entry.threshold_artifact_v4_path),
        "audit_path": str(entry.audit_path),
    }


def load_a0_scoring_config(
    entry: StageBInventoryEntry, window_size: int
) -> dict[str, Any]:
    """Load the matching A0 config and bind it to this Stage-B checkpoint."""
    entity_token = entry.entity_id.replace("-", "_")
    config_name = (
        f"smd__thesis__online__{entry.offline_variant}_A0__{entity_token}"
        f"__w{window_size}__seed{entry.seed}__main.yaml"
    )
    config_path = DEFAULT_ONLINE_EXPERIMENT_CONFIG_DIRECTORY / config_name
    online_config = load_experiment_config(config_path)
    online_config["task"]["reference_checkpoint_path"] = str(
        entry.stage_b_best_checkpoint_path
    )
    online_config["online_variant"] = "A0"
    return online_config
