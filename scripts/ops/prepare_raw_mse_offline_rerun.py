"""Prepare the approved 18 raw-MSE runs; this command never starts training."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.core.config import load_experiment_config


SOURCE_CONFIGS = Path("scripts/configs/experiment/offline_benchmark/thesis")


def prepare_configs(destination: Path, run_id: str) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for entity in ("machine_1_6", "machine_3_4", "machine_3_9"):
        for variant in ("O0", "O1"):
            for seed in (6, 8, 36):
                name = f"smd__thesis__offline__{variant}__{entity}__w20__seed{seed}__main"
                config = yaml.safe_load((SOURCE_CONFIGS / f"{name}.yaml").read_text())
                config["experiment_name"] = f"{name}__{run_id}"
                base = Path("outputs/benchmark/smd") / entity / f"seed{seed}" / f"thesis_{variant}_{run_id}" / "offline"
                config["output_dir"] = str(base)
                config["checkpoint_dir"] = str(base / "stage_b_fusion_finetuning")
                config["reconstruction_loss_space"] = "raw_input"
                config["evaluation"].update(score_space="raw_input", point_score_transform="identity")
                config["logging"]["wandb_run_name"] = config["experiment_name"]
                config["logging"]["wandb_tags"].extend([run_id, "raw-input-mse-training"])
                config["logging"]["enable_reconstruction_diagnostics"] = True
                config["model_overrides"] = dict(config.get("model_overrides", {}))
                # The parent config is also used to rebuild the final Stage-B model.
                config["model_overrides"]["training_phase"] = "stage_b_fusion_finetuning"
                path = destination / f"{variant}__{entity}__seed{seed}.yaml"
                with path.open("x", encoding="utf-8") as handle:
                    yaml.safe_dump(config, handle, sort_keys=False)
                load_experiment_config(path)
                paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="raw_mse_20260905")
    args = parser.parse_args()
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_id):
        parser.error("run-id must use only letters, digits, underscores or hyphens")
    destination = Path("configs/generated") / args.run_id
    for path in prepare_configs(destination, args.run_id):
        print(path)


if __name__ == "__main__":
    main()
