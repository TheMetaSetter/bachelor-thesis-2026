from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.redlamp_baseline import RedLampBaseline


def _build_batch(batch_size: int = 4) -> dict[str, object]:
    return {
        "x": torch.randn(batch_size, 20, 4),
        "point_labels": torch.zeros(batch_size, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(batch_size)],
    }


class _SingleEntityValidationDataset:
    def __init__(self) -> None:
        self.sequences = [
            {
                "x": torch.zeros(20, 4),
                "point_labels": torch.zeros(20, dtype=torch.long),
                "meta": {"entity_id": "machine-0"},
            }
        ]


class _SingleEntityValidationDataLoader:
    def __init__(self) -> None:
        self.dataset = _SingleEntityValidationDataset()
        self.batch = {
            "x": torch.randn(1, 20, 4),
            "point_labels": torch.zeros(1, 20, dtype=torch.long),
            "mask": None,
            "timestamps": None,
            "meta": [
                {
                    "entity_id": "machine-0",
                    "start_index": 0,
                    "end_index": 20,
                }
            ],
        }

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter([self.batch])


def test_trainer_logs_clean_and_auxiliary_realistic_validation_metrics_separately_for_baseline(
    tmp_path: Path,
) -> None:
    model = RedLampBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
        lambda_recon=0.9,
        lambda_cls=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )
    batch = _build_batch()
    val_loader = _SingleEntityValidationDataLoader()

    try:
        outputs = trainer.train(
            train_loader=[batch],
            val_loader=val_loader,
            scaler_state={
                "feature_mean": torch.zeros(4),
                "feature_std": torch.ones(4),
            },
            config={"experiment_name": "redlamp-validation-alignment-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    epoch_metrics = outputs["metric_history"][0]

    assert "train_classification_loss" in epoch_metrics
    assert "val_loss" in epoch_metrics
    assert "val_reconstruction_loss" in epoch_metrics
    assert "val_classification_loss" not in epoch_metrics
    assert "val_classification_accuracy" not in epoch_metrics
    assert "val_synth_loss" in epoch_metrics
    assert "val_synth_classification_loss" in epoch_metrics
    assert "val_synth_classification_accuracy" in epoch_metrics
    assert "val_synth_accuracy" in epoch_metrics
    assert "val_synth_macro_f1" in epoch_metrics
    assert "val_synth_pr_auc_pointwise" in epoch_metrics
    assert "val_synth_roc_auc_pointwise" in epoch_metrics
    assert "val_synth_vus_pr" in epoch_metrics
    assert "val_synth_loss" not in epoch_metrics


def test_trainer_falls_back_to_synthetic_validation_rate_for_anomaly_archive(
    tmp_path: Path,
) -> None:
    model = RedLampBaseline(
        input_dim=1,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=0.37,
        lambda_recon=0.9,
        lambda_cls=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )

    try:
        anomaly_rate = trainer._resolve_realistic_validation_anomaly_rate(
            config={
                "data": {
                    "dataset_name": "anomaly_archive",
                    "file_path": (
                        "data/AnomalyArchive/"
                        "219_UCR_Anomaly_STAFFIIIDatabase_41612_64632_64852.txt"
                    ),
                    "window_size": 20,
                    "stride": 10,
                },
                "task": {
                    "val_synth_source": "test_same_scope",
                    "anomaly_probability": 0.37,
                },
            }
        )
    finally:
        experiment_logger.close()

    assert anomaly_rate == pytest.approx(0.37)
