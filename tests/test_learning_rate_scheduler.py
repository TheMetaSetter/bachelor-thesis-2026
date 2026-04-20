from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from scripts.train import build_scheduler_from_experiment_config
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer


class DummyPlateauModel(nn.Module):
    def __init__(
        self,
        *,
        val_loss_sequence: list[float],
        val_synth_loss_sequence: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.val_loss_sequence = val_loss_sequence
        self.val_synth_loss_sequence = val_synth_loss_sequence or list(
            val_loss_sequence
        )
        self.validation_step_index = 0
        self.synthetic_validation_step_index = 0

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        loss = self.scalar * 0.0 + 1.0
        return {
            "loss": loss,
            "log": {"train_loss": float(loss.detach().cpu())},
            "outputs": {"aux": {}},
            "batch": batch,
        }

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        val_loss = float(self.val_loss_sequence[self.validation_step_index])
        self.validation_step_index += 1
        loss = self.scalar * 0.0 + val_loss
        return {
            "loss": loss,
            "log": {
                "val_loss": val_loss,
                "val_reconstruction_loss": val_loss,
            },
            "outputs": {"aux": {}},
            "batch": batch,
        }

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        val_synth_loss = float(
            self.val_synth_loss_sequence[self.synthetic_validation_step_index]
        )
        self.synthetic_validation_step_index += 1
        loss = self.scalar * 0.0 + val_synth_loss
        synthetic_batch = dict(batch)
        synthetic_batch["classification_labels"] = torch.tensor(
            [0, 1], dtype=torch.long
        )
        return {
            "loss": loss,
            "log": {
                "val_synth_loss": val_synth_loss,
                "val_synth_classification_loss": val_synth_loss,
                "val_synth_classification_accuracy": 0.5,
            },
            "outputs": {
                "logits": torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float32),
                "aux": {},
            },
            "batch": synthetic_batch,
        }


def _build_batch() -> dict[str, Any]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def _build_scheduler_experiment_config(
    *,
    patience: int = 1,
    cooldown: int = 0,
    min_lr: float = 1.0e-5,
    monitor_metric: str = "val_loss",
) -> dict[str, Any]:
    return {
        "optimizer": {
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "scheduler": {
                "scheduler_name": "reduce_on_plateau",
                "monitor_metric": monitor_metric,
                "factor": 0.5,
                "patience": patience,
                "threshold": 0.0,
                "threshold_mode": "rel",
                "cooldown": cooldown,
                "min_lr": min_lr,
            },
        }
    }


def test_scheduler_builder_returns_none_without_scheduler_config() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        {"optimizer": {"learning_rate": 0.1, "weight_decay": 0.0}},
    )

    assert scheduler is None
    assert scheduler_monitor_metric is None


def test_scheduler_builder_returns_reduce_on_plateau_instance() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(),
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler_monitor_metric == "val_loss"


def test_scheduler_builder_supports_val_synth_roc_auc_monitor() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(monitor_metric="val_synth_roc_auc"),
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler_monitor_metric == "val_synth_roc_auc"


def test_scheduler_builder_supports_val_synth_pr_auc_monitor() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(monitor_metric="val_synth_pr_auc"),
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler_monitor_metric == "val_synth_pr_auc"


def test_scheduler_respects_min_lr_floor() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.025,
    )

    for _ in range(8):
        scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == 0.025


def test_scheduler_respects_cooldown_between_reductions() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        threshold=0.0,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1.0e-5,
    )

    scheduler.step(1.0)
    scheduler.step(1.0)
    learning_rate_after_first_reduction = optimizer.param_groups[0]["lr"]
    scheduler.step(1.0)
    learning_rate_during_cooldown = optimizer.param_groups[0]["lr"]
    scheduler.step(1.0)
    learning_rate_after_second_reduction = optimizer.param_groups[0]["lr"]

    assert learning_rate_after_first_reduction == 0.05
    assert learning_rate_during_cooldown == learning_rate_after_first_reduction
    assert learning_rate_after_second_reduction == 0.025


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


def test_trainer_can_step_scheduler_from_val_synth_roc_auc(tmp_path: Path) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[1.0, 1.0, 1.0],
        val_synth_loss_sequence=[1.0, 1.0, 1.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(
            patience=1,
            monitor_metric="val_synth_roc_auc",
        ),
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
            config={"experiment_name": "scheduler-val-synth-roc-auc-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert "val_synth_roc_auc" in metric_history[-1]
    assert (
        metric_history[-1]["scheduler_monitor_val_synth_roc_auc"]
        == metric_history[-1]["val_synth_roc_auc"]
    )


def test_trainer_can_step_scheduler_from_val_synth_pr_auc(tmp_path: Path) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[1.0, 1.0, 1.0],
        val_synth_loss_sequence=[1.0, 1.0, 1.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(
            patience=1,
            monitor_metric="val_synth_pr_auc",
        ),
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
            config={"experiment_name": "scheduler-val-synth-pr-auc-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert "val_synth_pr_auc" in metric_history[-1]
    assert (
        metric_history[-1]["scheduler_monitor_val_synth_pr_auc"]
        == metric_history[-1]["val_synth_pr_auc"]
    )
