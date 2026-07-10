from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import (
    load_experiment_config,
    load_yaml_config,
    validate_experiment_config,
)
from src.data.augment import REDLAMP_ANOMALY_FAMILIES
from src.models.redlamp_baseline import RedLampBaseline


TRAINING_POLICY_EXPERIMENT_CONFIGS = [
    "configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-100ep__w100__seed7__default.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-100ep-usage-kaggle__w100__seed7__kaggle.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-continuous-only__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-discrete-only__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-fused__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-augmentation__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-covariance__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-diversity__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-gate__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-usage__w100__seed7__default.yaml",
    "configs/experiment/ablation/smd__thesis_multitask__multitask-no-variance__w100__seed7__default.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-full__w100__seed7__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11__w100__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-100ep__w100__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-300ep__w100__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-window10-binary__w10__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-full-vus-pr-a__w100__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-full-vus-pr-b__w100__seed11__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed23__w100__seed23__rtx3090.yaml",
    "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed47__w100__seed47__rtx3090.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-rtx3090-smoke__w100__seed7__smoke.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-rtx3090-smoke-seed11__w100__seed11__smoke.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-rtx3090-smoke-seed23__w100__seed23__smoke.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-rtx3090-smoke-seed47__w100__seed47__smoke.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-smoke__w100__seed7__smoke.yaml",
    "configs/experiment/smoke/smd__thesis_multitask__multitask-usage-kaggle-smoke__w100__seed7__smoke.yaml",
    "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-lr1e-3__w20__seed11__default.yaml",
    "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml",
    "configs/experiment/smoke/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml",
    "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-alt__w20__seed11__default.yaml",
    "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml",
    "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml",
    "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-nobootstrap__w20__seed11__default.yaml",
    "configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml",
]


def test_load_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__thesis_multitask__vertical-slice__w100__seed7__default.yaml"
    )
    assert loaded_config["data"]["window_size"] == 100
    assert loaded_config["model"]["model_name"] == "reconstruction_mlp_ae"


def test_load_online_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml"
    )
    assert loaded_config["model"]["model_name"] == "online_adaptation"
    assert loaded_config["task"]["target_param_group"] == "projector_params"
    assert (
        loaded_config["task"]["reference_checkpoint_path"]
        == "outputs/smd_multitask/checkpoints/best.pt"
    )


@pytest.mark.parametrize(
    "experiment_config_path",
    TRAINING_POLICY_EXPERIMENT_CONFIGS,
)
def test_target_training_configs_use_shared_adamw_scheduler_val_synth_vus_pr_policy(
    experiment_config_path: str,
) -> None:
    loaded_config = load_experiment_config(experiment_config_path)
    scheduler_config = loaded_config["optimizer"]["scheduler"]

    assert loaded_config["optimizer"]["optimizer_name"] == "adamw"
    assert loaded_config["optimizer"]["learning_rate"] == 0.001
    assert scheduler_config["scheduler_name"] in {"cosine", "reduce_on_plateau"}
    if scheduler_config["scheduler_name"] == "cosine":
        assert scheduler_config["warmup_epochs"] == 5
        assert scheduler_config["warmup_start_lr"] == 0.001
        assert scheduler_config["cosine_end_lr"] == 0.0
        assert scheduler_config["cosine_after_warmup"] is True
    else:
        assert scheduler_config["monitor_metric"] == "val_synth_vus_pr"
    assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"


def test_multitask_config_accepts_memory_bootstrap_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: memory-plan-smoke",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/test",
                "checkpoint_dir: outputs/test/checkpoints",
                "data:",
                "  dataset_name: smd",
                "  root_dir: data/ServerMachineDataset",
                "  window_size: 100",
                "  stride: 10",
                "  batch_size: 2",
                "  num_workers: 0",
                "  validation_split_ratio: 0.2",
                "model:",
                "  model_name: thesis_multitask",
                "  input_dim: 38",
                "  encoder_dim: 64",
                "  hidden_dim: 16",
                "  mlp_num_linear_layers: 3",
                "  num_classes: 2",
                "  dropout: 0.0",
                "  continuous_num_prototypes: 4",
                "  discrete_codebook_size: 8",
                "  gumbel_temperature: 1.0",
                "  temperature_start: 1.0",
                "  temperature_end: 1.0",
                "  temperature_anneal_fraction: 1.0",
                "  alpha_logit_init: 0.0",
                "  beta_logit_init: 0.0",
                "  lambda_cls: 1.0",
                "  lambda_div: 0.0",
                "  lambda_var: 0.0",
                "  lambda_cov: 0.0",
                "  lambda_use: 0.0",
                "  lambda_gate: 0.0",
                "  usage_lambda_start: 0.0",
                "  usage_lambda_end: 0.0",
                "  usage_lambda_schedule_fraction: 1.0",
                "  variance_floor_gamma: 1.0",
                "  gate_barrier_margin: 0.25",
                "  bootstrap_encoder_epochs: 10",
                "  discrete_ema_decay: 0.99",
                "  memory_norm_epsilon: 1.0e-6",
                "  memory_initialization_batches: 16",
                "  memory_initialization_with_synthetic_windows: true",
                "task:",
                "  task_name: multitask_tsad",
                "  use_synthetic_augmentation: true",
                "  use_synthetic_validation: true",
                "  synthetic_validation_seed: 7",
                "  anomaly_probability: 0.5",
                "  min_segment_fraction: 0.1",
                "  max_segment_fraction: 0.2",
                "  spike_scale: 3.0",
                "  freeze_fusion_for_epochs: 0",
                "  warmup_alpha_value: 0.0",
                "  warmup_beta_value: 0.0",
                f"  anomaly_families: {list(REDLAMP_ANOMALY_FAMILIES)}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 12",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_yaml_config(config_path)
    validate_experiment_config(loaded_config)

    assert loaded_config["model"]["bootstrap_encoder_epochs"] == 10
    assert loaded_config["model"]["discrete_ema_decay"] == 0.99
    assert loaded_config["model"]["memory_norm_epsilon"] == 1.0e-6
    assert loaded_config["model"]["memory_initialization_batches"] == 16
    assert (
        loaded_config["model"]["memory_initialization_with_synthetic_windows"] is True
    )


def test_load_multitask_ablation_config_applies_overrides() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/ablation/smd__thesis_multitask__multitask-continuous-only__w100__seed7__default.yaml"
    )
    assert loaded_config["model"]["lambda_gate"] == 0.01
    assert loaded_config["task"]["freeze_fusion_for_epochs"] == 3
    assert loaded_config["task"]["warmup_alpha_value"] == 0.0
    assert loaded_config["task"]["warmup_beta_value"] == 0.0
    assert loaded_config["task"]["anomaly_families"] == list(REDLAMP_ANOMALY_FAMILIES)


def test_load_seed_specific_rtx3090_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11__w100__seed11__rtx3090.yaml"
    )
    assert loaded_config["seed"] == 11
    assert loaded_config["model"]["mlp_num_linear_layers"] == 3
    assert loaded_config["output_dir"] == "outputs/smd_multitask_rtx3090_seed11"
    assert loaded_config["model"]["num_classes"] == 12
    assert loaded_config["task"]["classification_label_mode"] == "redlamp_multiclass"


def test_load_single_entity_rtx3090_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-100ep__w100__seed11__rtx3090.yaml"
    )
    assert loaded_config["data"]["entity_ids"] == ["machine-2-1"]
    assert loaded_config["optimizer"]["scheduler"]["scheduler_name"] == "cosine"
    assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"


def test_load_single_entity_window10_binary_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-window10-binary__w10__seed11__rtx3090.yaml"
    )

    assert loaded_config["data"]["entity_ids"] == ["machine-2-1"]
    assert loaded_config["data"]["window_size"] == 10
    assert loaded_config["data"]["stride"] == 10


def test_validate_experiment_config_rejects_missing_required_top_level_section() -> (
    None
):
    with pytest.raises(
        ValueError, match="Experiment config is missing required section: task"
    ):
        validate_experiment_config(
            {
                "experiment_name": "missing-task",
                "seed": 7,
                "device": "cpu",
                "output_dir": "outputs/test",
                "checkpoint_dir": "outputs/test/checkpoints",
                "data": {"dataset_name": "smd"},
                "model": {"model_name": "reconstruction_mlp_ae"},
                "optimizer": {"learning_rate": 1.0e-3, "weight_decay": 0.0},
                "epochs": 1,
            }
        )


def test_multitask_config_accepts_fixed_synthetic_train_seed_when_train_is_not_shuffled() -> (
    None
):
    experiment_config = {
        "experiment_name": "fixed-train-synth-ok",
        "seed": 7,
        "device": "cpu",
        "output_dir": "outputs/fixed-train-synth-ok",
        "checkpoint_dir": "outputs/fixed-train-synth-ok/checkpoints",
        "data": {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 20,
            "stride": 1,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": False,
        },
        "model": {
            "model_name": "redlamp_baseline",
            "input_dim": 38,
            "window_size": 20,
            "latent_dim": 16,
            "encoder_family": "cnn_simple",
            "mlp_num_linear_layers": 3,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 8,
            "cnn_dropout": 0.1,
            "classifier_dim": 8,
            "num_classes": 12,
            "dropout": 0.1,
            "lambda_recon": 0.9,
            "lambda_cls": 0.1,
            "use_label_refurbishment": True,
            "refurbishment_alpha": 0.1,
            "refurbishment_beta": 0.01,
            "enable_gradient_conflict_profiling": False,
            "gradient_profiling_scope": "encoder_all",
            "gradient_focus_layer_name": "encoder_last_affine",
            "gradient_log_every_n_steps": 1,
            "gradient_ema_alpha": 0.1,
            "gradient_sma_window": 50,
            "gradient_profile_include_bias": False,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": True,
            "use_synthetic_validation": True,
            "synthetic_train_seed": 17,
            "synthetic_validation_seed": 7,
            "classification_label_mode": "redlamp_multiclass",
            "freeze_fusion_for_epochs": 0,
            "warmup_alpha_value": 0.5,
            "warmup_beta_value": 0.5,
            "anomaly_probability": 0.5,
            "train_balance_classes": True,
            "min_segment_fraction": 0.2,
            "max_segment_fraction": 0.3,
            "spike_scale": 3.0,
            "anomaly_visibility_boost": 1.5,
            "anomaly_families": list(REDLAMP_ANOMALY_FAMILIES),
        },
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
        },
        "epochs": 1,
        "checkpoint_monitor_metric": "val_synth_vus_pr",
    }

    validate_experiment_config(experiment_config)


def test_multitask_config_rejects_fixed_synthetic_train_seed_when_train_is_shuffled() -> (
    None
):
    experiment_config = {
        "experiment_name": "fixed-train-synth-bad",
        "seed": 7,
        "device": "cpu",
        "output_dir": "outputs/fixed-train-synth-bad",
        "checkpoint_dir": "outputs/fixed-train-synth-bad/checkpoints",
        "data": {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 20,
            "stride": 1,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": True,
        },
        "model": {
            "model_name": "redlamp_baseline",
            "input_dim": 38,
            "window_size": 20,
            "latent_dim": 16,
            "encoder_family": "cnn_simple",
            "mlp_num_linear_layers": 3,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 8,
            "cnn_dropout": 0.1,
            "classifier_dim": 8,
            "num_classes": 12,
            "dropout": 0.1,
            "lambda_recon": 0.9,
            "lambda_cls": 0.1,
            "use_label_refurbishment": True,
            "refurbishment_alpha": 0.1,
            "refurbishment_beta": 0.01,
            "enable_gradient_conflict_profiling": False,
            "gradient_profiling_scope": "encoder_all",
            "gradient_focus_layer_name": "encoder_last_affine",
            "gradient_log_every_n_steps": 1,
            "gradient_ema_alpha": 0.1,
            "gradient_sma_window": 50,
            "gradient_profile_include_bias": False,
        },
        "task": {
            "task_name": "multitask_tsad",
            "use_synthetic_augmentation": True,
            "use_synthetic_validation": True,
            "synthetic_train_seed": 17,
            "synthetic_validation_seed": 7,
            "classification_label_mode": "redlamp_multiclass",
            "freeze_fusion_for_epochs": 0,
            "warmup_alpha_value": 0.5,
            "warmup_beta_value": 0.5,
            "anomaly_probability": 0.5,
            "train_balance_classes": True,
            "min_segment_fraction": 0.2,
            "max_segment_fraction": 0.3,
            "spike_scale": 3.0,
            "anomaly_visibility_boost": 1.5,
            "anomaly_families": list(REDLAMP_ANOMALY_FAMILIES),
        },
        "optimizer": {
            "optimizer_name": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
        },
        "epochs": 1,
        "checkpoint_monitor_metric": "val_synth_vus_pr",
    }

    with pytest.raises(
        ValueError,
        match="synthetic_train_seed requires data.shuffle_train=false",
    ):
        validate_experiment_config(experiment_config)


def test_validate_config_accepts_adamw_cosine_gradient_clipping_and_vus_monitor() -> (
    None
):
    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml"
    )
    loaded_config["optimizer"]["optimizer_name"] = "adamw"
    loaded_config["optimizer"]["gradient_clip_norm"] = 1.0
    loaded_config["optimizer"]["scheduler"] = {
        "scheduler_name": "cosine",
        "warmup_epochs": 5,
        "warmup_start_lr": 1.0e-4,
        "cosine_end_lr": 0.0,
        "cosine_after_warmup": True,
    }
    loaded_config["checkpoint_monitor_metric"] = "val_vus_pr"

    validate_experiment_config(loaded_config)


def test_validate_config_rejects_legacy_val_realistic_vus_pr_checkpoint_monitor() -> (
    None
):
    loaded_config = load_experiment_config(
        "configs/experiment/benchmark/baseline/"
        "smd__redlamp_baseline__benchmark-machine_1_6__w20__seed6__main.yaml"
    )
    loaded_config["checkpoint_monitor_metric"] = "val_realistic_vus_pr"

    with pytest.raises(
        ValueError,
        match="checkpoint_monitor_metric must be one of",
    ):
        validate_experiment_config(loaded_config)


def test_multitask_validation_defaults_to_balanced_redlamp_multiclass_when_omitted() -> (
    None
):
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml"
    )
    loaded_config["task"].pop("classification_label_mode")
    loaded_config["task"].pop("train_balance_classes")

    validate_experiment_config(loaded_config)

    assert loaded_config["task"]["classification_label_mode"] == "redlamp_multiclass"
    assert loaded_config["task"]["train_balance_classes"] is True


def test_multitask_validation_keeps_explicit_binary_opt_in_when_requested() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-seed11-machine-2-1-window10-binary__w10__seed11__rtx3090.yaml"
    )

    validate_experiment_config(loaded_config)

    assert loaded_config["model"]["num_classes"] == 2
    assert loaded_config["task"]["classification_label_mode"] == "binary"
    assert loaded_config["task"]["train_balance_classes"] is True


def test_validate_config_accepts_reconstruction_diagnostics_logging_fields() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2-small-100ep__w20__seed11__default.yaml"
    )
    assert loaded_config["logging"]["enable_reconstruction_diagnostics"] is True
    assert loaded_config["logging"]["diagnostics_log_interval_steps"] == 1
    assert loaded_config["logging"]["diagnostics_include_grad_norm"] is False

    validate_experiment_config(loaded_config)


def test_validate_config_rejects_invalid_reconstruction_diagnostics_logging_fields() -> (
    None
):
    loaded_config = load_experiment_config(
        "configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2-small-100ep__w20__seed11__default.yaml"
    )
    loaded_config["logging"]["diagnostics_log_interval_steps"] = 0
    with pytest.raises(
        ValueError,
        match="logging.diagnostics_log_interval_steps must be a positive integer when provided",
    ):
        validate_experiment_config(loaded_config)


def test_validate_config_rejects_invalid_optimizer_name() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml"
    )
    loaded_config["optimizer"]["optimizer_name"] = "sgd"

    with pytest.raises(ValueError, match="optimizer.optimizer_name"):
        validate_experiment_config(loaded_config)


def test_validate_config_rejects_non_positive_gradient_clip_norm() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml"
    )
    loaded_config["optimizer"]["gradient_clip_norm"] = 0.0

    with pytest.raises(ValueError, match="optimizer.gradient_clip_norm"):
        validate_experiment_config(loaded_config)


def test_load_explicit_redlamp_adamw_cosine_configs() -> None:
    for config_path in [
        "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-lr1e-3__w20__seed11__default.yaml",
        "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-alt__w20__seed11__default.yaml",
    ]:
        loaded_config = load_experiment_config(config_path)

        assert loaded_config["optimizer"]["optimizer_name"] == "adamw"
        assert loaded_config["optimizer"]["scheduler"]["scheduler_name"] in {
            "cosine",
            "reduce_on_plateau",
        }
        assert loaded_config["optimizer"]["learning_rate"] == 0.001
        assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"
        assert loaded_config["epochs"] == 300


def test_load_explicit_redlamp_cosine_val_synth_vus_pr_config() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/scale/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr__w20__seed68__default.yaml"
    )

    assert loaded_config["optimizer"]["optimizer_name"] == "adamw"
    assert (
        loaded_config["optimizer"]["scheduler"]["scheduler_name"] == "reduce_on_plateau"
    )
    assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"
    assert loaded_config["epochs"] == 300


def test_load_explicit_redlamp_cosine_val_synth_vus_pr_smoke_config() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smoke/smd__redlamp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-val-vus-pr-smoke__w20__seed11__smoke.yaml"
    )

    assert loaded_config["optimizer"]["optimizer_name"] == "adamw"
    assert loaded_config["optimizer"]["scheduler"]["scheduler_name"] == "cosine"
    assert loaded_config["checkpoint_monitor_metric"] == "val_synth_vus_pr"
    assert loaded_config["epochs"] == 1
    assert loaded_config["data"]["max_train_windows"] == 64
    assert loaded_config["data"]["max_val_windows"] == 32
    assert loaded_config["logging"]["use_wandb"] is True
    assert loaded_config["logging"]["wandb_mode"] == "online"


@pytest.mark.parametrize(
    "experiment_config_path,expected_model_name",
    [
        (
            "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml",
            "thesis_multitask",
        ),
        (
            "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml",
            "redlamp_baseline",
        ),
    ],
)
def test_load_redlamp_multiclass_window20_configs(
    experiment_config_path: str,
    expected_model_name: str,
) -> None:
    loaded_config = load_experiment_config(experiment_config_path)

    assert loaded_config["data_config_path"] == (
        "configs/data/smd_rtx3090_machine_2_1_20.yaml"
    )
    assert loaded_config["task_config_path"] == (
        "configs/task/multitask_tsad_redlamp_multiclass_window20.yaml"
    )
    assert loaded_config["model"]["model_name"] == expected_model_name
    assert loaded_config["data"]["window_size"] == 20
    assert loaded_config["data"]["stride"] == 20
    assert loaded_config["model"]["num_classes"] == 12
    assert loaded_config["model"]["mlp_num_linear_layers"] == 3
    assert loaded_config["task"]["classification_label_mode"] == "redlamp_multiclass"
    assert loaded_config["task"]["train_balance_classes"] is True
    assert loaded_config["task"]["anomaly_families"] == list(REDLAMP_ANOMALY_FAMILIES)
    assert loaded_config["logging"]["use_wandb"] is True
    assert loaded_config["logging"]["wandb_mode"] == "online"


@pytest.mark.parametrize(
    "experiment_config_path",
    [
        "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml",
        "configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-nobootstrap__w20__seed11__default.yaml",
        "configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml",
    ],
)
def test_load_thesis_multiclass_bootstrap_train_configs_enable_wandb(
    experiment_config_path: str,
) -> None:
    loaded_config = load_experiment_config(experiment_config_path)

    assert loaded_config["logging"]["use_wandb"] is True
    assert loaded_config["logging"]["wandb_mode"] == "online"


def test_load_multitask_smoke_config_keeps_wandb_disabled() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smoke/smd__thesis_multitask__multitask-smoke__w100__seed7__smoke.yaml"
    )

    assert loaded_config["logging"]["use_wandb"] is False
    assert loaded_config["logging"]["wandb_mode"] == "disabled"


def test_legacy_redlamp_baseline_experiment_path_constructs_canonical_model() -> None:
    from scripts.train import (
        build_model_from_experiment_config,
        register_runtime_components,
    )

    loaded_config = load_experiment_config(
        "configs/experiment/baseline/smd__redlamp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml"
    )
    register_runtime_components()

    model = build_model_from_experiment_config(loaded_config)

    assert isinstance(model, RedLampBaseline)
    assert model.window_size == 20
    assert model.num_classes == 12


def test_thesis_model_defaults_to_enabled_label_refurbishment_when_omitted(
    tmp_path: Path,
) -> None:
    from scripts.train import (
        build_model_from_experiment_config,
        register_runtime_components,
    )

    data_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    experiment_path = tmp_path / "experiment.yaml"
    data_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "window_size: 20",
                "stride: 10",
                "batch_size: 2",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        )
    )
    model_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 16",
                "mlp_num_linear_layers: 3",
                "num_classes: 12",
                "dropout: 0.0",
                "continuous_num_prototypes: 4",
                "discrete_codebook_size: 8",
                "gumbel_temperature: 1.0",
                "temperature_start: 1.0",
                "temperature_end: 1.0",
                "temperature_anneal_fraction: 1.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
                "lambda_recon: 1.0",
                "lambda_cls: 1.0",
                "lambda_div: 0.0",
                "lambda_var: 0.0",
                "lambda_cov: 0.0",
                "lambda_use: 0.0",
                "lambda_gate: 0.0",
                "usage_lambda_start: 0.0",
                "usage_lambda_end: 0.0",
                "usage_lambda_schedule_fraction: 1.0",
                "variance_floor_gamma: 1.0",
                "gate_barrier_margin: 0.25",
                "bootstrap_encoder_epochs: 0",
                "discrete_ema_decay: 0.99",
                "memory_norm_epsilon: 1.0e-6",
                "memory_initialization_batches: 16",
                "memory_initialization_with_synthetic_windows: true",
            ]
        )
    )
    task_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "classification_label_mode: redlamp_multiclass",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "anomaly_probability: 0.5",
                "train_balance_classes: false",
                "min_segment_fraction: 0.2",
                "max_segment_fraction: 0.3",
                "spike_scale: 3.0",
                "anomaly_visibility_boost: 1.5",
                "anomaly_families:",
                "  - spike",
                "  - noise",
                "  - scale",
            ]
        )
    )
    experiment_path.write_text(
        "\n".join(
            [
                "experiment_name: thesis-default-refurbishment",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/test",
                "checkpoint_dir: outputs/test/checkpoints",
                "epochs: 1",
                f"data_config_path: {data_path}",
                f"model_config_path: {model_path}",
                f"task_config_path: {task_path}",
                "optimizer:",
                "  optimizer_name: adamw",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "logging:",
                "  use_wandb: false",
                "  wandb_mode: disabled",
            ]
        )
    )

    loaded_config = load_experiment_config(str(experiment_path))
    validate_experiment_config(loaded_config)
    register_runtime_components()

    model = build_model_from_experiment_config(loaded_config)

    assert model.use_label_refurbishment is True


def test_load_experiment_config_injects_window_size_into_thesis_model(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    data_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "entity_ids: [machine-2-1]",
                "window_size: 20",
                "stride: 20",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
                "shuffle_train: true",
            ]
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "mlp_num_linear_layers: 3",
                "num_classes: 2",
                "dropout: 0.1",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.5",
                "temperature_start: 1.5",
                "temperature_end: 0.7",
                "temperature_anneal_fraction: 0.8",
                "temperature_hold_fraction: 0.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
                "use_label_refurbishment: false",
                "refurbishment_alpha: 0.0",
                "refurbishment_beta: 0.0",
                "reconstruction_normal_only: false",
                "lambda_cls: 1.0",
                "lambda_div: 0.0",
                "lambda_var: 0.0",
                "lambda_cov: 0.0",
                "lambda_use: 0.0",
                "lambda_gate: 0.0",
                "usage_lambda_start: 0.0",
                "usage_lambda_end: 0.0",
                "usage_lambda_schedule_fraction: 1.0",
                "variance_floor_gamma: 1.0",
                "gate_barrier_margin: 0.25",
                "bootstrap_encoder_epochs: 0",
                "discrete_ema_decay: 0.99",
                "memory_norm_epsilon: 1.0e-6",
                "memory_initialization_batches: 1",
                "memory_initialization_with_synthetic_windows: false",
            ]
        ),
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "classification_label_mode: binary",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "train_balance_classes: false",
                "anomaly_families: [spike, flip]",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path.write_text(
        "\n".join(
            [
                "experiment_name: unit_test_window_size_injection",
                "seed: 11",
                "device: cpu",
                "output_dir: outputs/unit_test_window_size_injection",
                "checkpoint_dir: outputs/unit_test_window_size_injection/checkpoints",
                f"data_config_path: {data_path}",
                f"model_config_path: {model_path}",
                f"task_config_path: {task_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(experiment_path)

    assert loaded_config["model"]["window_size"] == 20


def test_load_experiment_config_rejects_thesis_model_window_size_mismatch(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    data_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "entity_ids: [machine-2-1]",
                "window_size: 20",
                "stride: 20",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
                "shuffle_train: true",
            ]
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "window_size: 10",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "mlp_num_linear_layers: 3",
                "num_classes: 2",
                "dropout: 0.1",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.5",
                "temperature_start: 1.5",
                "temperature_end: 0.7",
                "temperature_anneal_fraction: 0.8",
                "temperature_hold_fraction: 0.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
                "use_label_refurbishment: false",
                "refurbishment_alpha: 0.0",
                "refurbishment_beta: 0.0",
                "reconstruction_normal_only: false",
                "lambda_cls: 1.0",
                "lambda_div: 0.0",
                "lambda_var: 0.0",
                "lambda_cov: 0.0",
                "lambda_use: 0.0",
                "lambda_gate: 0.0",
                "usage_lambda_start: 0.0",
                "usage_lambda_end: 0.0",
                "usage_lambda_schedule_fraction: 1.0",
                "variance_floor_gamma: 1.0",
                "gate_barrier_margin: 0.25",
                "bootstrap_encoder_epochs: 0",
                "discrete_ema_decay: 0.99",
                "memory_norm_epsilon: 1.0e-6",
                "memory_initialization_batches: 1",
                "memory_initialization_with_synthetic_windows: false",
            ]
        ),
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "classification_label_mode: binary",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "train_balance_classes: false",
                "anomaly_families: [spike, flip]",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path.write_text(
        "\n".join(
            [
                "experiment_name: unit_test_window_size_mismatch",
                "seed: 11",
                "device: cpu",
                "output_dir: outputs/unit_test_window_size_mismatch",
                "checkpoint_dir: outputs/unit_test_window_size_mismatch/checkpoints",
                f"data_config_path: {data_path}",
                f"model_config_path: {model_path}",
                f"task_config_path: {task_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="model.window_size must match data.window_size"
    ):
        load_experiment_config(experiment_path)
