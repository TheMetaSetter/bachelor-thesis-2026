from __future__ import annotations

from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ONLINE_BENCHMARK_METADATA_FIELDS = (
    "offline_variant",
    "entity_id",
    "seed",
    "benchmark_mode",
    "stage_name",
)
STAGE_B_CHECKPOINT_STAGE_NAME = "stage_b_fusion_finetuning"
STAGE_B_CHECKPOINT_NAME = "best.pt"


def _require_non_empty_string(field_name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"task.{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(field_name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"task.{field_name} must be a non-negative integer")
    return int(value)


def _find_stage_b_checkpoint_candidates(stage_root: Path) -> list[Path]:
    return sorted(stage_root.glob("*/checkpoints/best.pt"))


def _resolve_checkpoint_from_metadata(task_config: dict[str, Any]) -> Path:
    missing_fields = [
        field_name
        for field_name in ONLINE_BENCHMARK_METADATA_FIELDS
        if field_name not in task_config
    ]
    if missing_fields:
        raise ValueError(
            "online_adaptation task config is missing required metadata keys: "
            f"{missing_fields}"
        )

    offline_variant = _require_non_empty_string(
        "offline_variant", task_config["offline_variant"]
    )
    entity_id = _require_non_empty_string("entity_id", task_config["entity_id"])
    seed = _require_non_negative_int("seed", task_config["seed"])
    benchmark_mode = _require_non_empty_string(
        "benchmark_mode", task_config["benchmark_mode"]
    )
    stage_name = _require_non_empty_string("stage_name", task_config["stage_name"])

    benchmark_root = {
        "main": "benchmark",
        "smoke": "benchmark_smoke",
    }.get(benchmark_mode)
    if benchmark_root is None:
        raise ValueError("task.benchmark_mode must be one of: main, smoke")

    entity_token = entity_id.replace("-", "_")
    stage_root = (
        REPOSITORY_ROOT
        / "outputs"
        / benchmark_root
        / "smd"
        / "thesis"
        / offline_variant
        / entity_token
        / f"seed{seed}"
        / "two_stage"
    )
    candidates = _find_stage_b_checkpoint_candidates(stage_root)
    matching_candidates = [
        candidate
        for candidate in candidates
        if candidate.parent.parent.name == stage_name
    ]
    if len(matching_candidates) == 1:
        return matching_candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "No Stage B checkpoint matches online benchmark metadata: "
            f"offline_variant={offline_variant}, entity_id={entity_id}, seed={seed}, "
            f"benchmark_mode={benchmark_mode}, stage_name={stage_name}"
        )
    if not matching_candidates:
        raise FileNotFoundError(
            "No Stage B checkpoint matches the requested stage_name: "
            f"{stage_name}. Candidates found: {[str(path) for path in candidates]}"
        )
    raise ValueError(
        "Ambiguous Stage B checkpoint metadata matched multiple candidates: "
        f"{[str(path) for path in matching_candidates]}"
    )


def resolve_stage_b_checkpoint(experiment_config: dict[str, Any]) -> Path:
    task_config = dict(experiment_config.get("task", {}))
    reference_checkpoint_path = task_config.get("reference_checkpoint_path")
    if isinstance(reference_checkpoint_path, str) and reference_checkpoint_path:
        requested_path = Path(reference_checkpoint_path)
        if requested_path.is_file():
            return requested_path
        raise FileNotFoundError(
            "Configured reference_checkpoint_path does not exist: "
            f"{requested_path}. Use the canonical Stage B checkpoint path or "
            "provide complete online benchmark metadata."
        )
    return _resolve_checkpoint_from_metadata(task_config)


def resolve_threshold_artifact(experiment_config: dict[str, Any]) -> Path:
    """Resolve the explicit offline artifact selected for this online run."""
    task_config = dict(experiment_config.get("task", {}))
    configured_path = task_config.get("threshold_artifact_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("task.threshold_artifact_path must be a non-empty string")
    artifact_path = Path(configured_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Configured threshold_artifact_path does not exist: " f"{artifact_path}"
        )
    return artifact_path
