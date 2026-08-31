from __future__ import annotations

import argparse
from pathlib import Path

from src.core.config import load_experiment_config

from scripts.experiments.run_two_stage_offline_pretraining import (
    prepare_stage_b_initialization_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a direct-routing Stage B initialization checkpoint from "
            "an existing Stage A checkpoint. This command does not run Stage A."
        )
    )
    parser.add_argument(
        "--stage-b-config",
        required=True,
        type=Path,
        help="Path to the standalone direct-routing Stage B experiment config.",
    )
    parser.add_argument(
        "--stage-a-checkpoint",
        required=True,
        type=Path,
        help="Path to the existing Stage A best checkpoint.",
    )
    parser.add_argument(
        "--output-checkpoint",
        required=True,
        type=Path,
        help="Path where the Stage B initialization checkpoint will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_b_config = load_experiment_config(args.stage_b_config)
    model_config = stage_b_config.get("model", {})
    if model_config.get("training_phase") != "stage_b_fusion_finetuning":
        raise ValueError(
            "Bridge config must use training_phase=stage_b_fusion_finetuning"
        )
    if model_config.get("fusion_mode") != "direct_branch_routing":
        raise ValueError("Bridge config must use fusion_mode=direct_branch_routing")
    if "two_stage" in stage_b_config:
        raise ValueError("Bridge config must not define two_stage")
    output_path = prepare_stage_b_initialization_checkpoint(
        stage_b_config=stage_b_config,
        stage_a_checkpoint_path=args.stage_a_checkpoint,
        initialization_checkpoint_path=args.output_checkpoint,
    )
    print(f"stage_b_initialization_checkpoint={output_path}")


if __name__ == "__main__":
    main()
