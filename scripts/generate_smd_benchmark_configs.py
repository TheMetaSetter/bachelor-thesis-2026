from __future__ import annotations

"""Generate THESIS offline benchmark configs for the SMD entities.

₍^. .^₎⟆ Config matrix path

entity + seed + variant + run_mode
  -> one experiment YAML
  -> one benchmark wrapper input
  -> one fair offline run
"""

import argparse
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_BENCHMARK_CONFIG_DIR = (
    REPOSITORY_ROOT / "configs" / "experiment" / "offline_benchmark" / "thesis"
)
WINDOW_SIZE = 20
BENCHMARK_ENTITY_IDS = ("machine_1_6", "machine_3_4", "machine_3_9")
BENCHMARK_SEEDS = (6, 8, 36)
BENCHMARK_VARIANTS = ("O0", "O1")


def _entity_token(entity_id: str) -> str:
    return entity_id.replace("-", "_")


def _data_config_path(entity_id: str) -> str:
    return f"configs/data/smd_benchmark_{entity_id}_window20.yaml"


def _variant_experiment_name(
    *,
    variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> str:
    mode = "smoke" if smoke else "main"
    return (
        f"smd__thesis__offline__{variant}__{_entity_token(entity_id)}"
        f"__w{WINDOW_SIZE}__seed{seed}__{mode}"
    )


def _variant_output_dir(
    *,
    variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> str:
    root_dir = "benchmark_smoke" if smoke else "benchmark"
    return (
        f"outputs/{root_dir}/smd/thesis/{variant}/{_entity_token(entity_id)}/seed{seed}"
    )


def _variant_epochs(smoke: bool) -> dict[str, int]:
    if smoke:
        return {
            "epochs": 2,
            "stage_a_multitask_epochs": 1,
            "stage_b_fusion_finetuning_epochs": 1,
        }
    return {
        "epochs": 30,
        "stage_a_multitask_epochs": 25,
        "stage_b_fusion_finetuning_epochs": 5,
    }


def _variant_logging(
    smoke: bool, variant: str, entity_id: str, seed: int
) -> dict[str, Any]:
    wandb_mode = "disabled" if smoke else "online"
    use_wandb = not smoke
    tags = [
        "offline-benchmark",
        "thesis",
        variant.lower(),
        entity_id,
        f"seed{seed}",
        f"window{WINDOW_SIZE}",
    ]
    if smoke:
        tags.append("smoke")
    return {
        "use_wandb": use_wandb,
        "wandb_project": "bachelor-thesis-2026",
        "wandb_mode": wandb_mode,
        "wandb_run_name": _variant_experiment_name(
            variant=variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "wandb_tags": tags,
        "log_hard_prediction_ratio": not smoke,
        "log_row_normalized_confusion_matrix": not smoke,
        "diagnostics_stages_for_classification": ["train", "val_synth"],
    }


def _variant_model_overrides(variant: str) -> dict[str, Any]:
    if variant == "O0":
        return {}
    return {
        "enable_score_loss": True,
        "score_loss_granularity": "point",
        "score_loss_type": "pointwise_balanced_bce_logits",
        "score_loss_target": "synthetic_anomaly_mask",
        "score_loss_normalization": "train_batch_normal_tokens_detached_mean_std",
        "score_loss_reduction": "pointwise_binary_balanced_mean",
    }


def build_offline_benchmark_config(
    *,
    variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    epochs = _variant_epochs(smoke)
    config: dict[str, Any] = {
        "experiment_name": _variant_experiment_name(
            variant=variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "seed": seed,
        "experiment_variant": (
            "two_stage_base_v1"
            if variant == "O0"
            else "two_stage_point_score_supervised_v1"
        ),
        "device": "cpu" if smoke else "cuda",
        "output_dir": _variant_output_dir(
            variant=variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "checkpoint_dir": (
            f"{_variant_output_dir(variant=variant, entity_id=entity_id, seed=seed, smoke=smoke)}/checkpoints"
        ),
        "data_config_path": _data_config_path(entity_id),
        "model_config_path": "configs/model/thesis_multitask_two_stage_window20.yaml",
        "task_config_path": (
            "configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml"
        ),
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
            "scheduler": {
                "scheduler_name": "cosine",
                "warmup_epochs": 1 if smoke else 5,
                "warmup_start_lr": 0.001,
                "cosine_end_lr": 0.0,
                "cosine_after_warmup": True,
            },
        },
        "checkpoint_monitor_metric": "val_synth_vus_pr",
        "epochs": epochs["epochs"],
        "evaluation": {
            "vus_max_buffer_size": 10 if smoke else 20,
            "vus_num_thresholds": 20 if smoke else 200,
        },
        "logging": _variant_logging(smoke, variant, entity_id, seed),
        "two_stage": {
            "expected_total_training_epochs": epochs["epochs"],
            "stage_a_multitask_epochs": epochs["stage_a_multitask_epochs"],
            "stage_b_fusion_finetuning_epochs": epochs[
                "stage_b_fusion_finetuning_epochs"
            ],
            "discrete_memory_label_source": "synthetic_train_labels",
            "freeze_encoder_and_memories_in_stage_b": True,
        },
    }
    model_overrides = _variant_model_overrides(variant)
    if model_overrides:
        config["model_overrides"] = model_overrides
    if smoke:
        config["data_overrides"] = {
            "batch_size": 8,
            "num_workers": 0,
            "max_train_windows": 16,
            "max_val_windows": 8,
            "max_test_windows": 16,
        }
    return config


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def generate_thesis_offline_benchmark_configs() -> list[Path]:
    generated_paths: list[Path] = []
    for variant in BENCHMARK_VARIANTS:
        for entity_id in BENCHMARK_ENTITY_IDS:
            for seed in BENCHMARK_SEEDS:
                for smoke in (False, True):
                    config = build_offline_benchmark_config(
                        variant=variant,
                        entity_id=entity_id,
                        seed=seed,
                        smoke=smoke,
                    )
                    mode = "smoke" if smoke else "main"
                    config_path = OFFLINE_BENCHMARK_CONFIG_DIR / (
                        f"smd__thesis__offline__{variant}__{_entity_token(entity_id)}"
                        f"__w{WINDOW_SIZE}__seed{seed}__{mode}.yaml"
                    )
                    _write_config(config_path, config)
                    generated_paths.append(config_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-count", action="store_true")
    args = parser.parse_args()
    generated_paths = generate_thesis_offline_benchmark_configs()
    if args.print_count:
        print(len(generated_paths))
    else:
        for path in generated_paths:
            print(path)


if __name__ == "__main__":
    main()
