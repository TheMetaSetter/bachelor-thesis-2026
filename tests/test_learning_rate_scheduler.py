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
        self.val_synth_point_scores_sequence = (
            val_synth_point_scores_sequence
            or [
                torch.tensor([[0.1, 0.9]], dtype=torch.float32)
                for _ in self.val_synth_loss_sequence
            ]
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
    model = DummyPlateauModel(
        val_loss_sequence=[1.0], val_synth_loss_sequence=[1.0]
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
    ) -> dict[str, float]:
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


def test_trainer_can_step_scheduler_from_val_synth_loss(tmp_path: Path) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.5, 0.4, 0.3],
        val_synth_loss_sequence=[1.0, 1.0, 1.0],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(
            patience=1,
            monitor_metric="val_synth_loss",
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
            config={"experiment_name": "scheduler-val-synth-loss-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert metric_history[0]["optimizer_lr"] == 0.1
    assert metric_history[1]["optimizer_lr"] == 0.1
    assert metric_history[2]["optimizer_lr"] == 0.05
    assert metric_history[2]["scheduler_lr_reduced"] == 1.0
    assert (
        metric_history[2]["scheduler_monitor_val_synth_loss"]
        == metric_history[2]["val_synth_loss"]
    )


def test_trainer_tracks_best_checkpoint_from_scheduler_monitor_metric(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.8, 0.7, 0.6],
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
    val_synth_pr_auc_sequence = iter([0.20, 0.90, 0.50])

    def fake_aggregate_multitask_classification_metrics(
        self,
        *,
        logits_history: list[torch.Tensor],
        label_history: list[torch.Tensor],
        forward_pass_seconds_history: list[float],
        stage_name: str,
    ) -> dict[str, float]:
        del logits_history, label_history, forward_pass_seconds_history
        if stage_name == "val_synth":
            return {"val_synth_pr_auc": next(val_synth_pr_auc_sequence)}
        return {}

    trainer._aggregate_multitask_classification_metrics = MethodType(
        fake_aggregate_multitask_classification_metrics,
        trainer,
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
            config={"experiment_name": "scheduler-monitor-best-checkpoint-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    best_checkpoint = torch.load(outputs["best_checkpoint_path"], map_location="cpu")

    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["metric_history"][-1]["val_synth_pr_auc"] == 0.90
    assert best_checkpoint["metric_history"][-1]["val_loss"] == 0.7


def test_trainer_tracks_best_checkpoint_from_val_synth_loss(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.8, 0.7, 0.6],
        val_synth_loss_sequence=[0.9, 0.2, 0.5],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(
            patience=1,
            monitor_metric="val_synth_loss",
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
            config={"experiment_name": "scheduler-monitor-best-val-synth-loss-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    best_checkpoint = torch.load(outputs["best_checkpoint_path"], map_location="cpu")

    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["metric_history"][-1]["val_synth_loss"] == 0.2
    assert best_checkpoint["metric_history"][-1]["val_loss"] == 0.7


def test_trainer_tracks_best_checkpoint_from_val_loss_without_scheduler(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.8, 0.5, 0.6],
        val_synth_loss_sequence=[1.0, 1.0, 1.0],
    )
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
            config={"experiment_name": "val-loss-best-checkpoint-test"},
            epochs=3,
        )
    finally:
        experiment_logger.close()

    best_checkpoint = torch.load(outputs["best_checkpoint_path"], map_location="cpu")

    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["metric_history"][-1]["val_loss"] == 0.5
