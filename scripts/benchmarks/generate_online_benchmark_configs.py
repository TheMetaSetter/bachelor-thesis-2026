from __future__ import annotations

"""Generate THESIS online benchmark configs for the SMD entities.

₍^. .^₎⟆ Online config path

offline checkpoint reference + online adaptation contract
  -> one experiment YAML
  -> one benchmark wrapper input
  -> one fair online run
"""

import argparse
from pathlib import Path
from typing import Any

from scripts.benchmarks._config_generation_helpers import (
    entity_token,
    write_yaml_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ONLINE_BENCHMARK_CONFIG_DIR = (
    REPOSITORY_ROOT / "configs" / "experiment" / "online_benchmark" / "thesis"
)
WINDOW_SIZE = 20
BENCHMARK_ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
BENCHMARK_SEEDS = (6, 8, 36)
BENCHMARK_OFFLINE_VARIANTS = ("O0", "O1")
BENCHMARK_ONLINE_VARIANTS = ("A0", "A1", "A2")
STAGE_B_CHECKPOINT_STAGE_NAME = "stage_b_fusion_finetuning"


def _entity_token(entity_id: str) -> str:
    return entity_token(entity_id)


def _benchmark_mode(smoke: bool) -> str:
    return "smoke" if smoke else "main"


def _experiment_name(
    *,
    offline_variant: str,
    online_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> str:
    return (
        f"smd__thesis__online__{offline_variant}_{online_variant}"
        f"__{_entity_token(entity_id)}__w{WINDOW_SIZE}__seed{seed}"
        f"__{_benchmark_mode(smoke)}"
    )


def _output_dir(
    *,
    entity_id: str,
    offline_variant: str,
    online_variant: str,
    seed: int,
    smoke: bool,
) -> str:
    root_dir = "benchmark_smoke" if smoke else "benchmark"
    return (
        f"outputs/{root_dir}/online/smd/thesis/{offline_variant}_{online_variant}"
        f"/{_entity_token(entity_id)}/seed{seed}"
    )


def _variant_experiment_name(online_variant: str) -> str:
    return f"online_tta_{online_variant.lower()}_v1"


def _threshold_artifact_path(
    *, offline_variant: str, entity_id: str, seed: int, smoke: bool
) -> str:
    root_dir = "benchmark_smoke" if smoke else "benchmark"
    artifact_filename = (
        "thresholds.json" if smoke else "thresholds_v4_recalibrated.json"
    )
    return (
        f"outputs/{root_dir}/smd/thesis/{offline_variant}/"
        f"{_entity_token(entity_id)}/seed{seed}/thresholds/{artifact_filename}"
    )


def _variant_data_config_path(entity_id: str) -> str:
    return f"configs/data/smd_benchmark_{_entity_token(entity_id)}_window20.yaml"


def _model_overrides() -> dict[str, Any]:
    return {
        "input_dim": 38,
        "encoder_dim": 64,
        "hidden_dim": 32,
        "projector_hidden_dim": 64,
        "projector_dropout": 0.0,
        "enable_prototype_alignment": False,
        "lambda_align": 1.0,
        "lambda_proto": 0.1,
        "lambda_anchor": 0.001,
        "score_source": "projected_hidden",
    }


def _task_overrides(
    *,
    offline_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    max_online_steps = 16 if smoke else None
    checkpoint_every_n_steps = 8 if smoke else 50
    return {
        "offline_variant": offline_variant,
        "entity_id": entity_id,
        "seed": seed,
        "benchmark_mode": _benchmark_mode(smoke),
        "stage_name": STAGE_B_CHECKPOINT_STAGE_NAME,
        "threshold_artifact_path": _threshold_artifact_path(
            offline_variant=offline_variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "warm_start_projector": False,
        "target_param_group": "projector_params",
        "clean_stream_only": True,
        "max_online_steps": max_online_steps,
        "log_every_n_steps": 1,
        "checkpoint_every_n_steps": checkpoint_every_n_steps,
        "reset_policy": "disabled",
        "reset_alignment_threshold": 0.0,
    }


def _data_overrides() -> dict[str, Any]:
    return {
        "batch_size": 1,
        "num_workers": 12,
    }


def build_online_benchmark_config(
    *,
    offline_variant: str,
    online_variant: str,
    entity_id: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "experiment_name": _experiment_name(
            offline_variant=offline_variant,
            online_variant=online_variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "seed": seed,
        "experiment_variant": _variant_experiment_name(online_variant),
        "device": "cuda",
        "output_dir": _output_dir(
            entity_id=entity_id,
            offline_variant=offline_variant,
            online_variant=online_variant,
            seed=seed,
            smoke=smoke,
        ),
        "checkpoint_dir": (
            f"{_output_dir(entity_id=entity_id, offline_variant=offline_variant, online_variant=online_variant, seed=seed, smoke=smoke)}/checkpoints"
        ),
        "data_config_path": _variant_data_config_path(entity_id),
        "model_config_path": "configs/model/online_adaptation.yaml",
        "task_config_path": "configs/task/online_adaptation.yaml",
        "data_overrides": {
            "window_size": WINDOW_SIZE,
            "stride": 1,
            "train_stride": 1,
            "val_stride": WINDOW_SIZE,
            "test_stride": WINDOW_SIZE,
            "shuffle_train": False,
            **_data_overrides(),
        },
        "model_overrides": _model_overrides(),
        "task_overrides": _task_overrides(
            offline_variant=offline_variant,
            entity_id=entity_id,
            seed=seed,
            smoke=smoke,
        ),
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "epochs": 1,
        "evaluation": {
            "vus_max_buffer_size": 10 if smoke else 20,
            "vus_num_thresholds": 20 if smoke else 200,
            "retention_policy": "retain_for_eda",
        },
        "logging": {
            "use_wandb": False,
            "wandb_project": "bachelor-thesis-2026",
            "wandb_mode": "disabled",
            "wandb_run_name": _experiment_name(
                offline_variant=offline_variant,
                online_variant=online_variant,
                entity_id=entity_id,
                seed=seed,
                smoke=smoke,
            ),
            "wandb_tags": [
                "online-benchmark",
                "thesis",
                offline_variant.lower(),
                online_variant.lower(),
                entity_id,
                f"seed{seed}",
                f"window{WINDOW_SIZE}",
                _benchmark_mode(smoke),
            ],
            "log_hard_prediction_ratio": False,
            "log_row_normalized_confusion_matrix": False,
            "diagnostics_stages_for_classification": ["train", "val_synth"],
        },
    }
    return config


def generate_thesis_online_benchmark_configs() -> list[Path]:
    generated_paths: list[Path] = []
    for offline_variant in BENCHMARK_OFFLINE_VARIANTS:
        for online_variant in BENCHMARK_ONLINE_VARIANTS:
            for entity_id in BENCHMARK_ENTITY_IDS:
                for seed in BENCHMARK_SEEDS:
                    for smoke in (False, True):
                        config = build_online_benchmark_config(
                            offline_variant=offline_variant,
                            online_variant=online_variant,
                            entity_id=entity_id,
                            seed=seed,
                            smoke=smoke,
                        )
                        config_path = ONLINE_BENCHMARK_CONFIG_DIR / (
                            f"smd__thesis__online__{offline_variant}_{online_variant}"
                            f"__{_entity_token(entity_id)}__w{WINDOW_SIZE}"
                            f"__seed{seed}__{_benchmark_mode(smoke)}.yaml"
                        )
                        write_yaml_config(config_path, config)
                        generated_paths.append(config_path)
    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-count", action="store_true")
    args = parser.parse_args()
    generated_paths = generate_thesis_online_benchmark_configs()
    if args.print_count:
        print(len(generated_paths))
    else:
        for path in generated_paths:
            print(path)


if __name__ == "__main__":
    main()
