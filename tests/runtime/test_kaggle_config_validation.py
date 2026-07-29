from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import load_experiment_config


def _write_common_yaml_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_config_path = tmp_path / "data.yaml"
    model_config_path = tmp_path / "model.yaml"
    task_config_path = tmp_path / "task.yaml"

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
                "download: false",
                "skip_existing_download: true",
                "annotate_cleaning_metadata: false",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path.write_text(
        "\n".join(
            [
                "model_name: redlamp_baseline",
                "input_dim: 38",
                "dropout: 0.0",
                "latent_dim: 16",
                "classifier_dim: 8",
                "num_classes: 12",
            ]
        ),
        encoding="utf-8",
    )
    task_config_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "use_synthetic_augmentation: false",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "freeze_fusion_for_epochs: 0",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "anomaly_families: [spike]",
            ]
        ),
        encoding="utf-8",
    )
    return data_config_path, model_config_path, task_config_path


def test_load_experiment_config_accepts_optional_kaggle_logging_keys(
    tmp_path: Path,
) -> None:
    data_config_path, model_config_path, task_config_path = _write_common_yaml_files(
        tmp_path
    )
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: kaggle_enabled",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/kaggle_enabled",
                "checkpoint_dir: outputs/kaggle_enabled/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
                "logging:",
                "  mirror_best_checkpoint_to_kaggle: true",
                "  kaggle_dataset_handle: user/dataset",
                "  kaggle_version_notes: automated update",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(experiment_config_path)
    assert loaded_config["logging"]["mirror_best_checkpoint_to_kaggle"] is True
    assert loaded_config["logging"]["kaggle_dataset_handle"] == "user/dataset"


def test_load_experiment_config_rejects_kaggle_mirroring_without_dataset_handle(
    tmp_path: Path,
) -> None:
    data_config_path, model_config_path, task_config_path = _write_common_yaml_files(
        tmp_path
    )
    experiment_config_path = tmp_path / "experiment.yaml"
    experiment_config_path.write_text(
        "\n".join(
            [
                "experiment_name: kaggle_missing_handle",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/kaggle_missing_handle",
                "checkpoint_dir: outputs/kaggle_missing_handle/checkpoints",
                f"data_config_path: {data_config_path}",
                f"model_config_path: {model_config_path}",
                f"task_config_path: {task_config_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
                "logging:",
                "  mirror_best_checkpoint_to_kaggle: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="kaggle_dataset_handle"):
        load_experiment_config(experiment_config_path)
