from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer
from tests.runtime.test_learning_rate_scheduler import (
    DummyPlateauModel,
    _SingleEntityDataLoader,
    _build_batch,
    _build_scheduler_experiment_config,
)
from scripts.train import build_scheduler_from_experiment_config


def test_trainer_reduces_learning_rate_after_clean_validation_plateau(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(val_loss_sequence=[1.0, 1.0, 1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(patience=1),
    )
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_monitor_metric=scheduler_monitor_metric,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )
    batch = _build_batch()

    try:
        outputs = trainer.train(
            train_loader=[batch],
            val_loader=[batch],
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "scheduler-plateau-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert metric_history[0]["optimizer_lr"] == 0.1
    assert metric_history[1]["optimizer_lr"] == 0.1
    assert metric_history[2]["optimizer_lr"] == 0.05
    assert metric_history[2]["scheduler_lr_reduced"] == 1.0
    assert metric_history[2]["scheduler_monitor_val_loss"] == 1.0
    assert metric_history[2]["optimizer_lr_group_0"] == 0.05


def test_trainer_tracks_iteration_level_cosine_learning_rates(tmp_path: Path) -> None:
    model = DummyPlateauModel(val_loss_sequence=[1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        cosine_scheduler_config={
            "base_learning_rate": 0.1,
            "total_epochs": 1,
            "warmup_epochs": 0,
            "warmup_start_lr": 0.01,
            "cosine_end_lr": 0.0,
            "cosine_after_warmup": True,
        },
    )
    batch = _build_batch()

    try:
        outputs = trainer.train(
            train_loader=[batch, batch],
            val_loader=[batch],
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "cosine-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    epoch_metrics = outputs["metric_history"][0]
    assert epoch_metrics["optimizer_lr_start"] == 0.1
    assert epoch_metrics["optimizer_lr_end"] < epoch_metrics["optimizer_lr_start"]
    assert epoch_metrics["optimizer_lr_min"] == epoch_metrics["optimizer_lr_end"]
    assert epoch_metrics["optimizer_lr_max"] == epoch_metrics["optimizer_lr_start"]


def test_trainer_applies_gradient_clipping_when_configured(
    monkeypatch, tmp_path: Path
) -> None:
    model = DummyPlateauModel(val_loss_sequence=[1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    called_max_norms: list[float] = []

    def _fake_clip_grad_norm_(parameters, max_norm: float):
        list(parameters)
        called_max_norms.append(max_norm)
        return torch.tensor(2.5)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _fake_clip_grad_norm_)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        gradient_clip_norm=1.0,
    )

    try:
        outputs = trainer.train(
            train_loader=[_build_batch()],
            val_loader=[_build_batch()],
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "clip-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    assert called_max_norms == [1.0]
    assert outputs["metric_history"][0]["gradient_norm_max"] == 2.5
    assert outputs["metric_history"][0]["gradient_clipped_steps"] == 1.0


def test_trainer_adds_val_vus_pr_before_checkpoint_selection(tmp_path: Path) -> None:
    model = DummyPlateauModel(val_loss_sequence=[1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    validation_batch = {
        "x": torch.randn(1, 2, 1),
        "point_labels": torch.tensor([[0, 1]], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [
            {
                "entity_id": "machine-a",
                "start_index": 0,
                "end_index": 2,
            }
        ],
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        validation_evaluator_config={
            "vus_max_buffer_size": 1,
            "vus_num_thresholds": 10,
        },
        checkpoint_monitor_metric="val_vus_pr",
    )

    try:
        outputs = trainer.train(
            train_loader=[_build_batch()],
            val_loader=_SingleEntityDataLoader(validation_batch),
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "val-vus-pr-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    assert "val_vus_pr" in outputs["metric_history"][0]
    assert outputs["best_checkpoint_path"] is not None


def test_trainer_logs_val_synth_vus_pr_alongside_synthetic_classification_metrics(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(val_loss_sequence=[1.0], val_synth_loss_sequence=[1.0])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    validation_batch = {
        "x": torch.randn(1, 2, 1),
        "point_labels": torch.tensor([[0, 1]], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [
            {
                "entity_id": "machine-a",
                "start_index": 0,
                "end_index": 2,
            }
        ],
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        validation_evaluator_config={
            "vus_max_buffer_size": 1,
            "vus_num_thresholds": 10,
        },
    )

    try:
        outputs = trainer.train(
            train_loader=[_build_batch()],
            val_loader=_SingleEntityDataLoader(validation_batch),
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "val-synth-vus-pr-test"},
            epochs=1,
        )
    finally:
        experiment_logger.close()

    epoch_metrics = outputs["metric_history"][0]
    assert "val_synth_pr_auc" in epoch_metrics
    assert "val_synth_pr_auc_pointwise" in epoch_metrics
    assert "val_synth_vus_pr" in epoch_metrics
    assert "val_synth_threshold" in epoch_metrics


def test_trainer_tracks_best_checkpoint_from_val_synth_vus_pr(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.8, 0.7, 0.6],
        val_synth_loss_sequence=[1.0, 1.0, 1.0],
        val_synth_point_scores_sequence=[
            torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            torch.tensor([[0.3, 0.7]], dtype=torch.float32),
        ],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    validation_batch = {
        "x": torch.randn(1, 2, 1),
        "point_labels": torch.tensor([[0, 1]], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [
            {
                "entity_id": "machine-a",
                "start_index": 0,
                "end_index": 2,
            }
        ],
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        validation_evaluator_config={
            "vus_max_buffer_size": 1,
            "vus_num_thresholds": 10,
        },
        checkpoint_monitor_metric="val_synth_vus_pr",
    )
    val_synth_vus_pr_sequence = iter([0.20, 0.90, 0.50])

    def fake_aggregate_reconstructed_pointwise_metrics(
        self,
        *,
        data_loader: Any,
        batch_payloads: list[dict[str, Any]],
        stage_name: str,
        threshold: float | None = None,
    ) -> dict[str, float]:
        assert threshold is None  # This fixture exercises the historical score path.
        del data_loader, batch_payloads
        if stage_name != "val_synth":
            return {}
        metric_value = next(val_synth_vus_pr_sequence)
        return {
            "val_synth_pr_auc_pointwise": metric_value,
            "val_synth_vus_pr": metric_value,
            "val_synth_threshold": 0.5,
        }

    trainer._aggregate_reconstructed_pointwise_metrics = MethodType(
        fake_aggregate_reconstructed_pointwise_metrics,
        trainer,
    )

    try:
        outputs = trainer.train(
            train_loader=[_build_batch()],
            val_loader=_SingleEntityDataLoader(validation_batch),
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "best-val-synth-vus-pr-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    best_checkpoint = torch.load(outputs["best_checkpoint_path"], map_location="cpu")

    assert best_checkpoint["epoch"] == 2
    assert (
        best_checkpoint["metric_history"][-1]["val_synth_vus_pr"]
        >= best_checkpoint["metric_history"][0]["val_synth_vus_pr"]
    )


def test_trainer_supports_cosine_runtime_with_val_synth_vus_pr_checkpoint_monitor(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[1.0, 1.0],
        val_synth_loss_sequence=[1.0, 1.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    validation_batch = {
        "x": torch.randn(1, 2, 1),
        "point_labels": torch.tensor([[0, 1]], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [
            {
                "entity_id": "machine-a",
                "start_index": 0,
                "end_index": 2,
            }
        ],
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
        cosine_scheduler_config={
            "base_learning_rate": 0.1,
            "total_epochs": 2,
            "warmup_epochs": 0,
            "warmup_start_lr": 0.1,
            "cosine_end_lr": 0.0,
            "cosine_after_warmup": True,
        },
        validation_evaluator_config={
            "vus_max_buffer_size": 1,
            "vus_num_thresholds": 10,
        },
        checkpoint_monitor_metric="val_synth_vus_pr",
    )

    try:
        outputs = trainer.train(
            train_loader=[_build_batch(), _build_batch()],
            val_loader=_SingleEntityDataLoader(validation_batch),
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "cosine-val-synth-vus-pr-test"},
            epochs=2,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert "val_synth_vus_pr" in metric_history[-1]
    assert metric_history[-1]["optimizer_lr"] < 0.1
    assert outputs["best_checkpoint_path"] is not None


def test_trainer_ignores_val_synth_metrics_for_scheduler_stepping(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[1.0, 0.9, 0.8],
        val_synth_loss_sequence=[1.0, 5.0, 10.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(patience=0),
    )
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_monitor_metric=scheduler_monitor_metric,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )
    batch = _build_batch()

    try:
        outputs = trainer.train(
            train_loader=[batch],
            val_loader=[batch],
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "scheduler-ignore-val-synth-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert [epoch_metrics["optimizer_lr"] for epoch_metrics in metric_history] == [
        0.1,
        0.1,
        0.1,
    ]
    assert [
        epoch_metrics["scheduler_lr_reduced"] for epoch_metrics in metric_history
    ] == [0.0, 0.0, 0.0]
    assert metric_history[-1]["val_synth_loss"] == 10.0
    assert metric_history[-1]["scheduler_monitor_val_loss"] == 0.8
