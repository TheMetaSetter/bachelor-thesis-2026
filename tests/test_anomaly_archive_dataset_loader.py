from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import validate_experiment_config
from src.data import load_anomaly_archive_data
from src.data.datasets.anomaly_archive import AnomalyArchiveDatasetParser
from src.data.loaders import build_anomaly_archive_dataset_bundle


def test_anomaly_archive_parser_splits_staffiii_file_into_three_sequences() -> None:
    parser = AnomalyArchiveDatasetParser(
        file_path=Path(
            "data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt"
        ),
        validation_split_ratio=0.2,
        comparison_mode="pre_vs_anomaly",
    )

    parsed_sequences = parser.parse()

    assert set(parsed_sequences) == {"train", "val", "test"}
    assert len(parsed_sequences["train"]) == 1
    assert len(parsed_sequences["val"]) == 1
    assert len(parsed_sequences["test"]) == 1
    assert parsed_sequences["train"][0]["x"].ndim == 2
    assert parsed_sequences["train"][0]["x"].shape[1] == 1
    assert parsed_sequences["test"][0]["point_labels"].sum().item() > 0


def test_anomaly_archive_dataset_bundle_builds_redlamp_ready_windows() -> None:
    bundle = build_anomaly_archive_dataset_bundle(
        {
            "dataset_name": "anomaly_archive",
            "file_path": "data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt",
            "window_size": 20,
            "stride": 10,
            "batch_size": 8,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": False,
            "comparison_mode": "pre_vs_anomaly",
            "max_train_windows": 8,
            "max_val_windows": 4,
            "max_test_windows": 4,
        }
    )

    batch = next(iter(bundle["loaders"]["train"]))

    assert bundle["dataset_name"] == "anomaly_archive"
    assert bundle["datasets"]["train"]
    assert batch["x"].ndim == 3
    assert batch["x"].shape[2] == 1
    assert batch["point_labels"].shape == batch["x"].shape[:2]


def test_public_api_exposes_anomaly_archive_loader() -> None:
    public_bundle = load_anomaly_archive_data(
        file_path="data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt",
        window_size=20,
        stride=10,
        batch_size=4,
        validation_split_ratio=0.2,
        num_workers=0,
        shuffle_train=False,
        comparison_mode="pre_vs_anomaly",
        max_train_windows=4,
        max_val_windows=2,
        max_test_windows=2,
    )

    assert public_bundle["dataset_name"] == "anomaly_archive"
    assert public_bundle.loaders["train"] is not None


def test_validate_experiment_config_allows_anomaly_archive_dataset() -> None:
    experiment_config = {
        "experiment_name": "staffiii_redlamp_mlp_baseline",
        "seed": 11,
        "device": "cpu",
        "output_dir": "outputs/staffiii_redlamp_mlp_baseline",
        "checkpoint_dir": "outputs/staffiii_redlamp_mlp_baseline/checkpoints",
        "data": {
            "dataset_name": "anomaly_archive",
            "file_path": "data/AnomalyArchive/219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt",
            "window_size": 20,
            "stride": 10,
            "batch_size": 8,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": False,
            "comparison_mode": "pre_vs_anomaly",
        },
        "model": {
            "model_name": "redlamp_mlp_baseline",
            "input_dim": 1,
            "window_size": 20,
            "latent_dim": 16,
            "encoder_family": "mlp",
            "mlp_num_linear_layers": 3,
            "cnn_num_layers": 3,
            "cnn_kernel_size": 3,
            "cnn_hidden_channels": 8,
            "cnn_dropout": 0.1,
            "classifier_dim": 8,
            "num_classes": 12,
            "dropout": 0.1,
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
            "synthetic_validation_seed": 7,
            "classification_label_mode": "redlamp_multiclass",
            "freeze_fusion_for_epochs": 0,
            "warmup_alpha_value": 0.0,
            "warmup_beta_value": 0.0,
            "anomaly_probability": 0.5,
            "train_balance_classes": False,
            "val_realistic": True,
            "val_realistic_source": "test_same_scope",
            "val_anomaly_rate_override": None,
            "min_segment_fraction": 0.2,
            "max_segment_fraction": 0.3,
            "spike_scale": 3.0,
            "anomaly_visibility_boost": 1.5,
            "anomaly_families": ["spike"],
        },
        "optimizer": {
            "optimizer_name": "adam",
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 0.5,
        },
        "epochs": 1,
    }

    validate_experiment_config(experiment_config)
