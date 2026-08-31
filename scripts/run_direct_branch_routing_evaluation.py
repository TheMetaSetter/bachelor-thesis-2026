from __future__ import annotations

"""Evaluate the three direct-routing runs for machine_3_4."""

from pathlib import Path

from scripts.cli.evaluate import run_evaluation_experiment
from scripts.run_direct_branch_routing_full import build_direct_experiment_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG_DIRECTORY = (
    REPOSITORY_ROOT / "configs/experiment/offline_benchmark/thesis"
)
OFFLINE_VARIANT = "O0"
ENTITY = "machine_3_4"
SEEDS = (6, 8, 36)


def build_baseline_config_path(seed: int) -> Path:
    return (
        BASELINE_CONFIG_DIRECTORY
        / f"smd__thesis__offline__{OFFLINE_VARIANT}__{ENTITY}__w20__seed{seed}__main.yaml"
    )


def build_checkpoint_path(seed: int) -> Path:
    return (
        REPOSITORY_ROOT
        / "outputs/benchmark/smd"
        / ENTITY
        / f"seed{seed}"
        / f"thesis_direct_branch_routing_{OFFLINE_VARIANT}"
        / "offline/stage_b/checkpoints/best.pt"
    )


def prepare_evaluation_config(experiment_config: dict[str, object]) -> dict[str, object]:
    evaluation_config = dict(experiment_config)
    logging_config = dict(evaluation_config.get("logging", {}))
    logging_config.update(
        {
            "use_wandb": True,
            "wandb_mode": "online",
            "wandb_job_type": "evaluate",
        }
    )
    logging_config.setdefault(
        "wandb_run_name",
        f"{evaluation_config['experiment_name']}-evaluate",
    )
    evaluation_config["logging"] = logging_config
    return evaluation_config


def main() -> None:
    for seed in SEEDS:
        config_path = build_baseline_config_path(seed)
        checkpoint_path = build_checkpoint_path(seed)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing experiment config: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing direct-routing checkpoint: {checkpoint_path}")

        experiment_config = prepare_evaluation_config(
            build_direct_experiment_config(config_path)
        )
        print(f"Evaluating seed{seed}: {checkpoint_path}")
        run_evaluation_experiment(experiment_config, str(checkpoint_path))


if __name__ == "__main__":
    main()
