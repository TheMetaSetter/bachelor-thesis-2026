from __future__ import annotations

from pathlib import Path

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from src.models.thesis_multitask import ThesisMultitaskModel


def _build_batch(batch_size: int = 4) -> dict[str, object]:
    return {
        "x": torch.randn(batch_size, 100, 38),
        "point_labels": torch.zeros(batch_size, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": f"machine-{index}"} for index in range(batch_size)],
    }


class _SingleEntityValidationDataset:
    def __init__(self) -> None:
        self.sequences = [
            {
                "x": torch.zeros(100, 38),
                "point_labels": torch.zeros(100, dtype=torch.long),
                "meta": {"entity_id": "machine-0"},
            }
        ]


class _SingleEntityValidationDataLoader:
    def __init__(self) -> None:
        self.dataset = _SingleEntityValidationDataset()
        self.batch = {
            "x": torch.randn(1, 100, 38),
            "point_labels": torch.zeros(1, 100, dtype=torch.long),
            "mask": None,
            "timestamps": None,
            "meta": [
                {
                    "entity_id": "machine-0",
                    "start_index": 0,
                    "end_index": 100,
                }
            ],
        }

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter([self.batch])


def _build_model(**overrides: object) -> ThesisMultitaskModel:
    model_kwargs: dict[str, object] = {
        "input_dim": 38,
        "window_size": 100,
        "encoder_dim": 64,
        "hidden_dim": 16,
        "num_classes": 2,
        "dropout": 0.0,
        "continuous_enabled": True,
        "continuous_num_prototypes": 4,
        "discrete_enabled": True,
        "discrete_codebook_size": 8,
        "gumbel_temperature": 1.5,
        "temperature_start": 1.5,
        "temperature_end": 0.7,
        "temperature_anneal_fraction": 0.8,
        "temperature_hold_fraction": 0.25,
        "alpha_logit_init": 0.0,
        "beta_logit_init": 0.0,
        "lambda_cls": 1.0,
        "lambda_div": 0.0,
        "lambda_var": 0.0,
        "lambda_cov": 0.0,
        "lambda_use": 0.0,
        "lambda_gate": 0.0,
        "usage_lambda_start": 0.2,
        "usage_lambda_end": 0.2,
        "usage_lambda_schedule_fraction": 0.5,
        "use_synthetic_augmentation": True,
        "use_synthetic_validation": True,
        "synthetic_validation_seed": 123,
        "anomaly_probability": 1.0,
        "min_segment_fraction": 0.1,
        "max_segment_fraction": 0.2,
        "spike_scale": 3.0,
    }
    model_kwargs.update(overrides)
    return ThesisMultitaskModel(**model_kwargs)


def test_clean_validation_omits_classification_metrics() -> None:
    model = _build_model()
    step_output = model.validation_step(_build_batch())

    assert step_output["log"]["val_loss"] >= 0.0
    assert step_output["log"]["val_reconstruction_loss"] >= 0.0
    assert step_output["log"]["val_usage_lambda"] == 0.2
    assert "val_classification_loss" not in step_output["log"]
    assert "val_classification_accuracy" not in step_output["log"]
    assert (
        torch.count_nonzero(step_output["batch"]["classification_labels"]).item() == 0
    )


def test_synthetic_validation_is_deterministic_after_rng_reset() -> None:
    model = _build_model()
    batch = _build_batch()

    model.prepare_synthetic_validation_epoch()
    first_step = model.synthetic_validation_step(batch)
    model.prepare_synthetic_validation_epoch()
    second_step = model.synthetic_validation_step(batch)

    assert torch.equal(first_step["batch"]["x"], second_step["batch"]["x"])
    assert torch.equal(
        first_step["batch"]["classification_labels"],
        second_step["batch"]["classification_labels"],
    )
    assert torch.equal(
        first_step["batch"]["synthetic_anomaly_mask"],
        second_step["batch"]["synthetic_anomaly_mask"],
    )
    assert (
        first_step["batch"]["augmentation_metadata"]
        == second_step["batch"]["augmentation_metadata"]
    )
    assert "val_synth_classification_loss" in first_step["log"]
    assert "val_synth_classification_accuracy" in first_step["log"]
    assert "synthetic_anomaly_mask" in first_step["batch"]
    assert (
        first_step["batch"]["synthetic_anomaly_mask"].shape
        == first_step["batch"]["x"].shape[:2]
    )


def test_trainer_logs_clean_and_auxiliary_realistic_validation_metrics_separately(
    tmp_path: Path,
) -> None:
    model = _build_model()
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
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "validation-alignment-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    epoch_metrics = outputs["metric_history"][0]

    assert "train_classification_loss" in epoch_metrics
    assert "val_loss" in epoch_metrics
    assert "val_reconstruction_loss" in epoch_metrics
    assert "val_usage_lambda" in epoch_metrics
    assert "val_classification_loss" not in epoch_metrics
    assert "val_classification_accuracy" not in epoch_metrics
    assert "val_realistic_loss" in epoch_metrics
    assert "val_realistic_classification_loss" in epoch_metrics
    assert "val_realistic_classification_accuracy" in epoch_metrics
    assert "val_realistic_roc_auc" in epoch_metrics
    assert "val_realistic_pr_auc" in epoch_metrics
    assert "val_realistic_pr_auc_pointwise" in epoch_metrics
    assert "val_realistic_vus_pr" in epoch_metrics
    assert epoch_metrics["train_usage_lambda"] == 0.2
    assert epoch_metrics["val_usage_lambda"] == 0.2
    assert epoch_metrics["val_realistic_usage_lambda"] == 0.2
    assert epoch_metrics["val_loss"] != epoch_metrics["val_realistic_loss"]
