from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from scripts.cli.train import (
    build_model_from_experiment_config,
    register_runtime_components,
    run_training_experiment,
)
from scripts.experiments.run_two_stage_offline_pretraining import (
    prepare_stage_b_initialization_checkpoint,
)
from src.core.config import load_experiment_config
from src.engine.checkpoint import CheckpointManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_DIRECTORY = (
    REPOSITORY_ROOT / "configs/experiment/offline_benchmark/thesis"
)
DIRECT_OUTPUT_ROOT = "outputs/benchmark/smd"
CLOUD_REPOSITORY_ROOT = Path("/root/bachelor-thesis-2026")
OFFLINE_VARIANTS = ("O0", "O1")
ENTITIES = ("machine_1_6", "machine_3_4", "machine_3_9")
SEEDS = (6, 8, 36)


def build_run_configs() -> list[Path]:
    return [
        BASELINE_CONFIG_DIRECTORY
        / f"smd__thesis__offline__{offline_variant}__{entity}__w20__seed{seed}__main.yaml"
        for offline_variant in OFFLINE_VARIANTS
        for entity in ENTITIES
        for seed in SEEDS
    ]


def _parse_baseline_identity(config_path: Path) -> tuple[str, str, int]:
    name_parts = config_path.stem.split("__")
    if len(name_parts) != 8 or name_parts[2] != "offline":
        raise ValueError(f"Unexpected baseline config name: {config_path.name}")
    offline_variant = name_parts[3]
    entity = name_parts[4]
    seed_text = name_parts[6]
    if not seed_text.startswith("seed"):
        raise ValueError(f"Unexpected seed in config name: {config_path.name}")
    return offline_variant, entity, int(seed_text.removeprefix("seed"))


def build_stage_a_source_checkpoint_path(config_path: Path) -> Path:
    offline_variant, entity, seed = _parse_baseline_identity(config_path)
    return (
        Path("outputs/benchmark/smd/thesis")
        / offline_variant
        / entity
        / f"seed{seed}"
        / "two_stage"
        / "stage_a_multitask_pretraining/checkpoints/best.pt"
    )


def build_stage_b_initialization_checkpoint_path(
    direct_config: dict[str, Any],
) -> Path:
    return (
        Path(str(direct_config["output_dir"]))
        / "initializations"
        / "stage_b_init.pt"
    )


def build_direct_experiment_config(config_path: Path) -> dict[str, Any]:
    """Build one Stage-B-only direct-routing config from a main config."""
    baseline_config = load_experiment_config(config_path)
    offline_variant, entity, seed = _parse_baseline_identity(config_path)

    direct_config = dict(baseline_config)
    direct_config.update(
        {
            "experiment_name": (
                "smd__thesis__offline__direct_branch_routing__"
                f"{offline_variant}__{entity}__w20__seed{seed}__stage_b"
            ),
            "experiment_variant": "direct_branch_routing_v1",
            "epochs": 5,
            "output_dir": (
                f"{DIRECT_OUTPUT_ROOT}/"
                f"{entity}/seed{seed}/"
                f"thesis_direct_branch_routing_{offline_variant}/offline/stage_b"
            ),
            "checkpoint_dir": (
                f"{DIRECT_OUTPUT_ROOT}/"
                f"{entity}/seed{seed}/"
                f"thesis_direct_branch_routing_{offline_variant}/offline/stage_b/checkpoints"
            ),
        }
    )
    direct_config["initialization_checkpoint_path"] = str(
        build_stage_b_initialization_checkpoint_path(direct_config)
    )
    direct_config.pop("two_stage", None)

    direct_model_config = dict(direct_config["model"])
    direct_model_config.update(
        {
            "training_phase": "stage_b_fusion_finetuning",
            "fusion_mode": "direct_branch_routing",
        }
    )
    direct_config["model"] = direct_model_config

    direct_logging_config = dict(direct_config.get("logging", {}))
    direct_logging_config["wandb_run_name"] = direct_config["experiment_name"]
    direct_config["logging"] = direct_logging_config
    return direct_config


def _repository_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_direct_config(config: dict[str, Any]) -> None:
    if config["device"] != "cuda":
        raise ValueError(f"GPU run requires device=cuda: {config['experiment_name']}")
    if config["model"].get("fusion_mode") != "direct_branch_routing":
        raise ValueError(f"Expected direct_branch_routing: {config['experiment_name']}")
    if config["model"].get("training_phase") != "stage_b_fusion_finetuning":
        raise ValueError(f"Expected Stage B training: {config['experiment_name']}")
    if "two_stage" in config:
        raise ValueError(
            f"Direct routing run must not include two_stage: {config['experiment_name']}"
        )


def _cloud_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else CLOUD_REPOSITORY_ROOT / path


def ensure_stage_b_initialization_checkpoint(
    *,
    stage_b_config: dict[str, Any],
    stage_a_source_checkpoint_path: Path,
) -> Path:
    source_path = _repository_path(str(stage_a_source_checkpoint_path))
    target_path = _repository_path(
        str(build_stage_b_initialization_checkpoint_path(stage_b_config))
    )
    if source_path.resolve() == target_path.resolve():
        raise ValueError("Stage A source and Stage B initialization paths must differ")
    if not target_path.exists():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing Stage A best checkpoint: {source_path}"
            )
        prepare_stage_b_initialization_checkpoint(
            stage_b_config=stage_b_config,
            stage_a_checkpoint_path=source_path,
            initialization_checkpoint_path=target_path,
        )

    payload = torch.load(target_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Stage B initialization checkpoint must contain a mapping")
    payload_config = payload.get("config", {})
    if not isinstance(payload_config, dict):
        raise ValueError("Stage B initialization checkpoint config must be a mapping")
    if payload_config.get("experiment_name") != stage_b_config.get("experiment_name"):
        raise ValueError(
            "Existing Stage B initialization checkpoint has a different experiment identity"
        )

    register_runtime_components()
    model = build_model_from_experiment_config(stage_b_config)
    CheckpointManager(target_path.parent).load_checkpoint(
        target_path,
        model,
        strict=True,
    )
    return target_path


def main() -> None:
    direct_runs: list[tuple[Path, dict[str, Any], Path]] = []
    for config_path in build_run_configs():
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
        direct_config = build_direct_experiment_config(config_path)
        source_path = build_stage_a_source_checkpoint_path(config_path)
        direct_runs.append((config_path, direct_config, source_path))

    print(f"Prepared {len(direct_runs)} direct-routing Stage B runs.")
    for _, direct_config, source_path in direct_runs:
        _validate_direct_config(direct_config)
        print(
            f"{direct_config['experiment_name']}: "
            f"stage_a_source={_cloud_path(str(source_path))} "
            f"stage_b_init={_cloud_path(direct_config['initialization_checkpoint_path'])} "
            f"output={_cloud_path(direct_config['output_dir'])}"
        )

    for _, direct_config, source_path in direct_runs:
        ensure_stage_b_initialization_checkpoint(
            stage_b_config=direct_config,
            stage_a_source_checkpoint_path=source_path,
        )
        run_training_experiment(direct_config)


if __name__ == "__main__":
    main()
