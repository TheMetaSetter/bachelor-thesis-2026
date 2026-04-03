from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config
from src.data.augment import REDLAMP_ANOMALY_FAMILIES


def test_load_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_vertical_slice.yaml")
    assert loaded_config["data"]["window_size"] == 100
    assert loaded_config["model"]["model_name"] == "reconstruction_mlp_ae"


def test_load_online_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_online_adaptation.yaml")
    assert loaded_config["model"]["model_name"] == "online_adaptation"
    assert loaded_config["task"]["target_param_group"] == "projector_params"
    assert loaded_config["task"]["reference_checkpoint_path"] == "outputs/smd_multitask/checkpoints/best.pt"


def test_load_multitask_ablation_config_applies_overrides() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_multitask_continuous_only.yaml")
    assert loaded_config["model"]["lambda_gate"] == 0.01
    assert loaded_config["task"]["freeze_fusion_for_epochs"] == 3
    assert loaded_config["task"]["warmup_alpha_value"] == 0.0
    assert loaded_config["task"]["warmup_beta_value"] == 0.0
    assert loaded_config["task"]["anomaly_families"] == list(REDLAMP_ANOMALY_FAMILIES)


def test_load_experiment_config_rejects_missing_required_keys(tmp_path: Path) -> None:
    invalid_experiment_path = tmp_path / "invalid_experiment.yaml"
    invalid_experiment_path.write_text("experiment_name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_config(invalid_experiment_path)


def test_load_online_experiment_config_rejects_invalid_target_param_group(tmp_path: Path) -> None:
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


def test_load_multitask_experiment_config_rejects_invalid_anomaly_families(tmp_path: Path) -> None:
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
