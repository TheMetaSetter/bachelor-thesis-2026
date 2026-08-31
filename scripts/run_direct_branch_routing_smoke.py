from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.cli.train import run_training_experiment
from scripts.run_direct_branch_routing_full import (
    BASELINE_CONFIG_DIRECTORY,
    _cloud_path,
    _validate_direct_config,
    build_direct_experiment_config,
)


BASELINE_CONFIG_PATH = (
    BASELINE_CONFIG_DIRECTORY
    / "smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml"
)
SMOKE_OUTPUT_DIR = (
    "outputs/benchmark_smoke/smd/machine_1_6/seed6/"
    "thesis_direct_branch_routing_O0/offline/stage_b"
)


def build_smoke_experiment_config() -> dict[str, Any]:
    config = build_direct_experiment_config(BASELINE_CONFIG_PATH)
    config.update(
        {
            "experiment_name": (
                "smd__thesis__offline__direct_branch_routing__"
                "O0__machine_1_6__w20__seed6__smoke"
            ),
            "experiment_variant": "direct_branch_routing_smoke_v1",
            "epochs": 1,
            "output_dir": SMOKE_OUTPUT_DIR,
            "checkpoint_dir": f"{SMOKE_OUTPUT_DIR}/checkpoints",
        }
    )

    smoke_data_config = dict(config["data"])
    smoke_data_config.update(
        {
            "batch_size": 256,
            "max_train_windows": 2048,
            "max_val_windows": 2048,
            "max_test_windows": 2048,
        }
    )
    config["data"] = smoke_data_config

    smoke_evaluation_config = dict(config.get("evaluation", {}))
    smoke_evaluation_config.update(
        {
            "vus_max_buffer_size": 10,
            "vus_num_thresholds": 20,
        }
    )
    config["evaluation"] = smoke_evaluation_config

    smoke_logging_config = dict(config.get("logging", {}))
    smoke_logging_config.update(
        {
            "use_wandb": False,
            "wandb_mode": "disabled",
            "wandb_run_name": config["experiment_name"],
            "wandb_tags": ["offline-ablation", "direct-branch-routing", "smoke"],
        }
    )
    config["logging"] = smoke_logging_config
    return config


def main() -> None:
    config = build_smoke_experiment_config()
    _validate_direct_config(config)
    print(f"init={_cloud_path(config['initialization_checkpoint_path'])}")
    print(f"output={_cloud_path(config['output_dir'])}")
    run_training_experiment(config)


if __name__ == "__main__":
    main()
