from __future__ import annotations

from pathlib import Path
from typing import Any
from types import MethodType

import torch
from torch import nn

from scripts.train import (
    build_scheduler_from_experiment_config,
    compute_candi_style_cosine_learning_rate,
)
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer


class DummyPlateauModel(nn.Module):
    def __init__(
        self,
        *,
        val_loss_sequence: list[float],
        val_synth_loss_sequence: list[float] | None = None,
        val_synth_point_scores_sequence: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.val_loss_sequence = val_loss_sequence
        self.val_synth_loss_sequence = val_synth_loss_sequence or list(
            val_loss_sequence
        )
        self.val_synth_point_scores_sequence = val_synth_point_scores_sequence or [
            torch.tensor([[0.1, 0.9]], dtype=torch.float32)
            for _ in self.val_synth_loss_sequence
        ]
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
        current_step_index = self.synthetic_validation_step_index
        val_synth_loss = float(self.val_synth_loss_sequence[current_step_index])
        self.synthetic_validation_step_index += 1
        loss = self.scalar * 0.0 + val_synth_loss
        synthetic_batch = dict(batch)
        synthetic_batch["classification_labels"] = torch.tensor(
            [0, 1], dtype=torch.long
        )
        synthetic_batch["synthetic_anomaly_mask"] = torch.tensor(
            [[0, 1]],
            dtype=torch.long,
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
                "point_scores": self.val_synth_point_scores_sequence[
                    current_step_index
                ],
                "aux": {},
            },
            "batch": synthetic_batch,
        }

    def realistic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.synthetic_validation_step(batch)

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        point_scores = batch["point_labels"].float() + 0.1
        return {
            "outputs": {
                "point_scores": point_scores,
                "window_scores": point_scores.mean(dim=1),
                "aux": {},
            }
        }


def _build_batch() -> dict[str, Any]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


class _SingleEntitySequenceDataset:
    def __init__(self) -> None:
        self.sequences = [
            {
                "x": torch.zeros(2, 1),
                "point_labels": torch.tensor([0, 1], dtype=torch.long),
                "meta": {"entity_id": "machine-a"},
            }
        ]


class _SingleEntityDataLoader:
    def __init__(self, batch: dict[str, Any]) -> None:
        self.dataset = _SingleEntitySequenceDataset()
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __iter__(self):
        return iter([self.batch])


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


def test_compute_candi_style_cosine_learning_rate_uses_fractional_progress() -> None:
    first_batch_learning_rate = compute_candi_style_cosine_learning_rate(
        base_learning_rate=0.1,
        current_progress=0.0,
        total_epochs=4,
        warmup_epochs=1,
        warmup_start_lr=0.01,
        cosine_end_lr=0.0,
        cosine_after_warmup=True,
    )
    second_batch_learning_rate = compute_candi_style_cosine_learning_rate(
        base_learning_rate=0.1,
        current_progress=0.5,
        total_epochs=4,
        warmup_epochs=1,
        warmup_start_lr=0.01,
        cosine_end_lr=0.0,
        cosine_after_warmup=True,
    )
    late_learning_rate = compute_candi_style_cosine_learning_rate(
        base_learning_rate=0.1,
        current_progress=3.75,
        total_epochs=4,
        warmup_epochs=1,
        warmup_start_lr=0.01,
        cosine_end_lr=0.0,
        cosine_after_warmup=True,
    )

    assert first_batch_learning_rate == 0.01
    assert second_batch_learning_rate != first_batch_learning_rate
    assert late_learning_rate < second_batch_learning_rate
    assert late_learning_rate < 0.01


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


def test_scheduler_builder_supports_val_synth_loss_monitor() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(monitor_metric="val_synth_loss"),
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler_monitor_metric == "val_synth_loss"


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
