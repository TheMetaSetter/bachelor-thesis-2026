from __future__ import annotations

from pathlib import Path

from scripts.cli.train import run_training_experiment
from src.core.config import load_experiment_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs/experiment/offline_ablation/thesis"
EXPERIMENT_CONFIGS = tuple(
    CONFIG_DIRECTORY / f"smd__thesis__offline__direct_branch_routing__{entity}__w20__seed6__stage_b.yaml"
    for entity in ("machine_1_6", "machine_3_4", "machine_3_9")
)


def build_run_configs() -> list[Path]:
    return list(EXPERIMENT_CONFIGS)


def main() -> None:
    for config_path in build_run_configs():
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

        experiment_config = load_experiment_config(config_path)
        model_config = experiment_config["model"]
        if experiment_config["device"] != "cuda":
            raise ValueError(f"GPU run requires device=cuda: {config_path}")
        if model_config.get("fusion_mode") != "direct_branch_routing":
            raise ValueError(f"Expected direct_branch_routing: {config_path}")
        if model_config.get("training_phase") != "stage_b_fusion_finetuning":
            raise ValueError(f"Expected Stage B training: {config_path}")
        if "two_stage" in experiment_config:
            raise ValueError(f"Direct routing run must not include two_stage: {config_path}")

        run_training_experiment(experiment_config)


if __name__ == "__main__":
    main()
