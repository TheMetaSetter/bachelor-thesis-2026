from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.cli.train import run_training_experiment
from src.core.config import load_experiment_config


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
            "initialization_checkpoint_path": (
                "outputs/benchmark/smd/thesis/"
                f"{offline_variant}/{entity}/seed{seed}/two_stage/"
                "stage_a_multitask_pretraining/checkpoints/best.pt"
            ),
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
    checkpoint_path = _repository_path(config["initialization_checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing Stage A best checkpoint for {config['experiment_name']}: "
            f"{checkpoint_path}"
        )


def _cloud_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else CLOUD_REPOSITORY_ROOT / path


def main() -> None:
    direct_configs = []
    for config_path in build_run_configs():
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config does not exist: {config_path}")
        direct_configs.append(build_direct_experiment_config(config_path))

    print(f"Prepared {len(direct_configs)} direct-routing Stage B runs.")
    for direct_config in direct_configs:
        _validate_direct_config(direct_config)
        print(
            f"{direct_config['experiment_name']}: "
            f"init={_cloud_path(direct_config['initialization_checkpoint_path'])} "
            f"output={_cloud_path(direct_config['output_dir'])}"
        )

    for direct_config in direct_configs:
        run_training_experiment(direct_config)


if __name__ == "__main__":
    main()
