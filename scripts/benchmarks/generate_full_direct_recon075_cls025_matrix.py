"""Generate the isolated 18-cell offline ablation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG_ROOT = (
    REPOSITORY_ROOT / "configs" / "experiment" / "offline_benchmark" / "thesis"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs"
    / "benchmark_full_direct_recon075_cls025"
    / "generated_configs"
)
MODEL_CONFIG_PATH = (
    "configs/model/"
    "thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml"
)
VARIANTS = ("O0", "O1")
ENTITIES = ("machine_1_6", "machine_3_4", "machine_3_9")
SEEDS = (6, 8, 36)


def _source_config_path(variant: str, entity: str, seed: int) -> Path:
    return SOURCE_CONFIG_ROOT / (
        f"smd__thesis__offline__{variant}__{entity}__w20__seed{seed}__main.yaml"
    )


def _generated_config_name(variant: str, entity: str, seed: int) -> str:
    return (
        f"smd__thesis__offline__{variant}_recon075_cls025_direct__"
        f"{entity}__w20__seed{seed}__main.yaml"
    )


def _experiment_name(variant: str, entity: str, seed: int) -> str:
    return _generated_config_name(variant, entity, seed).removesuffix(".yaml")


def _model_overrides(variant: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "lambda_recon": 0.75,
        "lambda_cls": 0.25,
        "fusion_mode": "direct_branch_routing",
    }
    if variant == "O0":
        overrides["enable_score_loss"] = False
        return overrides
    overrides.update(
        {
            "enable_score_loss": True,
            "score_loss_granularity": "point",
            "score_loss_type": "pointwise_balanced_bce_logits",
            "score_loss_target": "synthetic_anomaly_mask",
            "score_loss_normalization": "train_batch_normal_tokens_detached_mean_std",
            "score_loss_reduction": "pointwise_binary_balanced_mean",
        }
    )
    return overrides


def build_config(variant: str, entity: str, seed: int, output_root: Path) -> dict[str, Any]:
    source_path = _source_config_path(variant, entity, seed)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source config: {source_path}")
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping in source config: {source_path}")

    experiment_name = _experiment_name(variant, entity, seed)
    run_root = output_root.parent / "smd" / "thesis" / variant / entity / f"seed{seed}"
    config.update(
        {
            "experiment_name": experiment_name,
            "experiment_variant": (
                "two_stage_base_recon075_cls025_direct_v1"
                if variant == "O0"
                else "two_stage_point_score_supervised_recon075_cls025_direct_v1"
            ),
            "model_config_path": MODEL_CONFIG_PATH,
            "output_dir": str(run_root),
            "checkpoint_dir": str(run_root / "checkpoints"),
            "reconstruction_loss_space": "normalized_input",
            "data_overrides": {"num_workers": 12},
            "model_overrides": _model_overrides(variant),
        }
    )
    logging_config = dict(config.get("logging", {}))
    logging_config.update(
        {
            "use_wandb": True,
            "wandb_mode": "online",
            "wandb_run_name": experiment_name,
        }
    )
    logging_config["wandb_tags"] = list(logging_config.get("wandb_tags", [])) + [
        "full-matrix",
        "recon075-cls025",
        "direct-branch-routing",
    ]
    config["logging"] = logging_config
    return config


def generate_configs(output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for variant in VARIANTS:
        for entity in ENTITIES:
            for seed in SEEDS:
                config = build_config(variant, entity, seed, output_root)
                output_path = output_root / _generated_config_name(variant, entity, seed)
                output_path.write_text(
                    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                generated_paths.append(output_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    for path in generate_configs(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
