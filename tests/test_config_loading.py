from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import (
    load_experiment_config,
    load_yaml_config,
    validate_experiment_config,
)
from src.data.augment import REDLAMP_ANOMALY_FAMILIES


def test_load_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_vertical_slice.yaml")
    assert loaded_config["data"]["window_size"] == 100
    assert loaded_config["model"]["model_name"] == "reconstruction_mlp_ae"


def test_load_online_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smd_online_adaptation.yaml"
    )
    assert loaded_config["model"]["model_name"] == "online_adaptation"
    assert loaded_config["task"]["target_param_group"] == "projector_params"
    assert (
        loaded_config["task"]["reference_checkpoint_path"]
        == "outputs/smd_multitask/checkpoints/best.pt"
    )


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
        "configs/experiment/smd_multitask_continuous_only.yaml"
    )
    assert loaded_config["model"]["lambda_gate"] == 0.01
    assert loaded_config["task"]["freeze_fusion_for_epochs"] == 3
    assert loaded_config["task"]["warmup_alpha_value"] == 0.0
    assert loaded_config["task"]["warmup_beta_value"] == 0.0
    assert loaded_config["task"]["anomaly_families"] == list(REDLAMP_ANOMALY_FAMILIES)


def test_load_seed_specific_rtx3090_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smd_multitask_rtx3090_seed11.yaml"
    )
    assert loaded_config["seed"] == 11
    assert loaded_config["model"]["mlp_num_linear_layers"] == 3
    assert loaded_config["output_dir"] == "outputs/smd_multitask_rtx3090_seed11"


def test_load_single_entity_rtx3090_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smd_multitask_rtx3090_seed11_machine_2_1_val_synth_pr_auc.yaml"
    )
    assert loaded_config["data"]["entity_ids"] == ["machine-2-1"]
    assert (
        loaded_config["optimizer"]["scheduler"]["monitor_metric"] == "val_synth_pr_auc"
    )


def test_load_multitask_experiment_config_rejects_invalid_mlp_num_linear_layers(
    tmp_path: Path,
) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "mlp_num_linear_layers: 1",
                "num_classes: 2",
                "dropout: 0.0",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.0",
                "temperature_start: 1.0",
                "temperature_end: 1.0",
                "temperature_anneal_fraction: 1.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
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
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                f"anomaly_families: {list(REDLAMP_ANOMALY_FAMILIES)}",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_multitask_depth",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_multitask_depth",
                "checkpoint_dir: outputs/invalid_multitask_depth/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mlp_num_linear_layers"):
        load_experiment_config(experiment_config_path)


def test_load_experiment_config_rejects_invalid_data_entity_ids(tmp_path: Path) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "entity_ids: machine-2-1",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: reconstruction_mlp_ae",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 16",
                "dropout: 0.1",
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: reconstruction",
                "loss_name: mse",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_data_entity_ids",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_data_entity_ids",
                "checkpoint_dir: outputs/invalid_data_entity_ids/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.entity_ids"):
        load_experiment_config(experiment_config_path)


def test_load_experiment_config_rejects_missing_required_keys(tmp_path: Path) -> None:
    invalid_experiment_path = tmp_path / "invalid_experiment.yaml"
    invalid_experiment_path.write_text("experiment_name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(invalid_experiment_path)


def test_load_online_experiment_config_rejects_invalid_target_param_group(
    tmp_path: Path,
) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: online_adaptation",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "projector_hidden_dim: 64",
                "projector_dropout: 0.0",
                "enable_prototype_alignment: false",
                "lambda_align: 1.0",
                "lambda_proto: 0.1",
                "lambda_anchor: 0.001",
                "score_source: projected_hidden",
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: online_adaptation",
                "reference_checkpoint_path: outputs/smd_multitask/checkpoints/best.pt",
                "warm_start_projector: false",
                "target_param_group: invalid_group",
                "clean_stream_only: true",
                "max_online_steps: 4",
                "log_every_n_steps: 1",
                "checkpoint_every_n_steps: 2",
                "view_noise_std: 0.01",
                "view_dropout_probability: 0.0",
                "reset_policy: disabled",
                "reset_alignment_threshold: 0.0",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_online_adaptation",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_online_adaptation",
                "checkpoint_dir: outputs/invalid_online_adaptation/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_param_group"):
        load_experiment_config(experiment_config_path)


def test_load_multitask_experiment_config_rejects_invalid_anomaly_families(
    tmp_path: Path,
) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "num_classes: 2",
                "dropout: 0.0",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.0",
                "temperature_start: 1.0",
                "temperature_end: 1.0",
                "temperature_anneal_fraction: 1.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
                "lambda_cls: 1.0",
                "enable_diversity_loss: false",
                "enable_variance_loss: false",
                "enable_covariance_loss: false",
                "enable_usage_loss: false",
                "enable_gate_loss: false",
                "lambda_div: 0.0",
                "lambda_var: 0.0",
                "lambda_cov: 0.0",
                "lambda_use: 0.0",
                "lambda_gate: 0.0",
                "variance_floor_gamma: 1.0",
                "gate_barrier_margin: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "use_synthetic_augmentation: true",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "anomaly_families: invalid_family",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_multitask_families",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_multitask_families",
                "checkpoint_dir: outputs/invalid_multitask_families/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="anomaly_families"):
        load_experiment_config(experiment_config_path)


def test_load_multitask_experiment_config_rejects_invalid_temperature_hold_fraction(
    tmp_path: Path,
) -> None:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"
    experiment_config_path = tmp_path / "experiment.yaml"

    data_config_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "window_size: 100",
                "stride: 10",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "num_classes: 2",
                "dropout: 0.0",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.0",
                "temperature_start: 1.0",
                "temperature_end: 1.0",
                "temperature_anneal_fraction: 1.0",
                "temperature_hold_fraction: 1.2",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
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
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "anomaly_families:",
                "  - spike",
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_multitask_temperature_hold",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_multitask_temperature_hold",
                "checkpoint_dir: outputs/invalid_multitask_temperature_hold/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="temperature_hold_fraction"):
        load_experiment_config(experiment_config_path)


def test_load_experiment_config_accepts_valid_reduce_on_plateau_scheduler() -> None:
    loaded_config = load_experiment_config(
        "configs/experiment/smd_multitask_rtx3090_full.yaml"
    )
    scheduler_config = loaded_config["optimizer"]["scheduler"]

    assert scheduler_config["scheduler_name"] == "reduce_on_plateau"
    assert scheduler_config["monitor_metric"] == "val_loss"
    assert scheduler_config["patience"] == 20


def test_load_experiment_config_accepts_valid_val_synth_pr_auc_scheduler(
    tmp_path: Path,
) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: valid_scheduler_monitor_val_synth_pr_auc",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/valid_scheduler_monitor_val_synth_pr_auc",
                "checkpoint_dir: outputs/valid_scheduler_monitor_val_synth_pr_auc/checkpoints",
                f"data_config_path: {Path('configs/data/smd_smoke.yaml').resolve()}",
                f"model_config_path: {Path('configs/model/thesis_multitask.yaml').resolve()}",
                f"task_config_path: {Path('configs/task/multitask_tsad.yaml').resolve()}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  scheduler:",
                "    scheduler_name: reduce_on_plateau",
                "    monitor_metric: val_synth_pr_auc",
                "    factor: 0.5",
                "    patience: 2",
                "    threshold: 0.0001",
                "    threshold_mode: rel",
                "    cooldown: 0",
                "    min_lr: 1.0e-5",
                "epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(experiment_config_path)

    assert (
        loaded_config["optimizer"]["scheduler"]["monitor_metric"] == "val_synth_pr_auc"
    )


def test_load_experiment_config_accepts_label_refurbishment_and_masking_fields(
    tmp_path: Path,
) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
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
                "use_label_refurbishment: true",
                "refurbishment_alpha: 0.2",
                "refurbishment_beta: 0.1",
                "reconstruction_normal_only: true",
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
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: valid_label_refurbishment_and_masking",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/valid_label_refurbishment_and_masking",
                "checkpoint_dir: outputs/valid_label_refurbishment_and_masking/checkpoints",
                f"data_config_path: {Path('configs/data/smd_smoke.yaml').resolve()}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {Path('configs/task/multitask_tsad.yaml').resolve()}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(experiment_config_path)

    assert loaded_config["model"]["use_label_refurbishment"] is True
    assert loaded_config["model"]["refurbishment_alpha"] == 0.2
    assert loaded_config["model"]["refurbishment_beta"] == 0.1
    assert loaded_config["model"]["reconstruction_normal_only"] is True


def test_load_experiment_config_rejects_invalid_scheduler_monitor_metric(
    tmp_path: Path,
) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_scheduler_monitor_metric",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_scheduler_monitor_metric",
                "checkpoint_dir: outputs/invalid_scheduler_monitor_metric/checkpoints",
                f"data_config_path: {Path('configs/data/smd_smoke.yaml').resolve()}",
                f"model_config_path: {Path('configs/model/thesis_multitask.yaml').resolve()}",
                f"task_config_path: {Path('configs/task/multitask_tsad.yaml').resolve()}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  scheduler:",
                "    scheduler_name: reduce_on_plateau",
                "    monitor_metric: val_synth_loss",
                "    factor: 0.5",
                "    patience: 2",
                "    threshold: 0.0001",
                "    threshold_mode: rel",
                "    cooldown: 0",
                "    min_lr: 1.0e-5",
                "epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="monitor_metric"):
        load_experiment_config(experiment_config_path)


def test_load_experiment_config_rejects_invalid_refurbishment_alpha(
    tmp_path: Path,
) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
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
                "use_label_refurbishment: true",
                "refurbishment_alpha: 1.2",
                "refurbishment_beta: 0.1",
                "reconstruction_normal_only: true",
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
            ]
        ),
        encoding="utf-8",
    )
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_refurbishment_alpha",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_refurbishment_alpha",
                "checkpoint_dir: outputs/invalid_refurbishment_alpha/checkpoints",
                f"data_config_path: {Path('configs/data/smd_smoke.yaml').resolve()}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {Path('configs/task/multitask_tsad.yaml').resolve()}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refurbishment_alpha"):
        load_experiment_config(experiment_config_path)


def test_load_experiment_config_rejects_scheduler_min_lr_above_learning_rate(
    tmp_path: Path,
) -> None:
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: invalid_scheduler_min_lr",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/invalid_scheduler_min_lr",
                "checkpoint_dir: outputs/invalid_scheduler_min_lr/checkpoints",
                f"data_config_path: {Path('configs/data/smd_smoke.yaml').resolve()}",
                f"model_config_path: {Path('configs/model/thesis_multitask.yaml').resolve()}",
                f"task_config_path: {Path('configs/task/multitask_tsad.yaml').resolve()}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  scheduler:",
                "    scheduler_name: reduce_on_plateau",
                "    monitor_metric: val_loss",
                "    factor: 0.5",
                "    patience: 2",
                "    threshold: 0.0001",
                "    threshold_mode: rel",
                "    cooldown: 0",
                "    min_lr: 0.01",
                "epochs: 3",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="min_lr"):
        load_experiment_config(experiment_config_path)
