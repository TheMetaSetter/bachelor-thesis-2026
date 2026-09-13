"""Resolve the frozen A0 scoring config used by V4 threshold recalibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_experiment_config


DEFAULT_ONLINE_EXPERIMENT_CONFIG_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "experiment"
    / "online_benchmark"
    / "thesis"
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


def _a0_config_name(entry: StageBInventoryEntry, window_size: int) -> str:
    entity_token = entry.entity_id.replace("-", "_")
    return (
        f"smd__thesis__online__{entry.offline_variant}_A0__{entity_token}"
        f"__w{window_size}__seed{entry.seed}__main.yaml"
    )


def resolve_a0_scoring_config_path(
    entry: StageBInventoryEntry, window_size: int
) -> Path:
    """Find the generated A0 config, or use the static config fallback."""
    entity_token = entry.entity_id.replace("-", "_")
    generated_path = entry.experiment_config_path.resolve().with_name(
        f"on-thesis-{entry.offline_variant}-A0-{entity_token}-s{entry.seed}.yaml"
    )
    static_path = DEFAULT_ONLINE_EXPERIMENT_CONFIG_DIRECTORY / _a0_config_name(
        entry, window_size
    )
    for candidate in (generated_path, static_path):
        if candidate.is_file():
            return candidate
    checked_paths = "\n".join(f"- {path}" for path in (generated_path, static_path))
    raise FileNotFoundError(
        "A0 scoring config does not exist. Checked:\n" + checked_paths
    )


def load_a0_scoring_config(
    entry: StageBInventoryEntry, window_size: int
) -> dict[str, Any]:
    """Load the matching A0 config and bind it to this Stage-B checkpoint."""
    config_path = resolve_a0_scoring_config_path(entry, window_size)
    online_config = load_experiment_config(config_path)
    online_config["task"]["reference_checkpoint_path"] = str(
        entry.stage_b_best_checkpoint_path
    )
    online_config["online_variant"] = "A0"
    return online_config
