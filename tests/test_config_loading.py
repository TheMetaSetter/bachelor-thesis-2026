from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config


def test_load_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_vertical_slice.yaml")
    assert loaded_config["data"]["window_size"] == 100
    assert loaded_config["model"]["model_name"] == "reconstruction_mlp_ae"


def test_load_online_experiment_config_reads_valid_yaml() -> None:
    loaded_config = load_experiment_config("configs/experiment/smd_online_adaptation.yaml")
    assert loaded_config["model"]["model_name"] == "online_adaptation"
    assert loaded_config["task"]["target_param_group"] == "projector_params"


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
                "reference_checkpoint_path: outputs/smd_vertical_slice/checkpoints/best.pt",
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
