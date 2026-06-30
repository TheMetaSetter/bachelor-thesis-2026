from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.engine.trainer import Trainer, build_checkpoint_evaluation_metadata
from src.models.base_model import BaseModel


class _NaNValidationModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch_size, window_size, _ = batch["x"].shape
        point_scores = torch.zeros(batch_size, window_size, dtype=batch["x"].dtype)
        return {
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {"forward_pass_seconds": 0.0},
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        loss = self.scalar.pow(2)
        outputs = self.forward(batch)
        return {
            "loss": loss,
            "log": {
                "train_loss": float(loss.detach().cpu()),
                "train_reconstruction_loss": float(loss.detach().cpu()),
            },
            "outputs": outputs,
            "loss_terms": {
                "total_loss": loss,
                "reconstruction_loss": loss,
            },
            "batch": batch,
        }

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        loss = self.scalar.new_tensor(0.0)
        outputs = self.forward(batch)
        return {
            "loss": loss,
            "log": {
                "val_loss": float("nan"),
                "val_reconstruction_loss": 0.0,
            },
            "outputs": outputs,
            "loss_terms": {
                "total_loss": loss,
                "reconstruction_loss": loss,
            },
            "batch": batch,
        }

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        loss = self.scalar.new_tensor(0.0)
        outputs = self.forward(batch)
        return {
            "loss": loss,
            "log": {"test_loss": 0.0},
            "outputs": outputs,
            "loss_terms": {"total_loss": loss},
            "batch": batch,
        }


def _build_batch() -> dict[str, Any]:
    return {
        "x": torch.randn(2, 4, 3),
        "point_labels": torch.zeros(2, 4, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-a"}, {"entity_id": "unit-b"}],
    }


def test_trainer_saves_final_and_fallback_best_checkpoint_when_monitor_is_nan(
    tmp_path: Path,
) -> None:
    model = _NaNValidationModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    checkpoint_manager = CheckpointManager(tmp_path / "checkpoints")
    experiment_logger = ExperimentLogger(
        tmp_path / "outputs",
        experiment_config={"experiment_name": "nan-checkpoint-fallback"},
        logging_config={"use_wandb": False},
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scheduler_monitor_metric=None,
        checkpoint_manager=checkpoint_manager,
        experiment_logger=experiment_logger,
        device="cpu",
    )

    outputs = trainer.train(
        train_loader=[_build_batch()],
        val_loader=[_build_batch()],
        scaler_state={},
        config={
            "experiment_name": "nan-checkpoint-fallback",
            "task": {},
        },
        epochs=1,
    )

    assert outputs["final_checkpoint_path"] is not None
    assert outputs["best_checkpoint_path"] is not None
    assert Path(outputs["final_checkpoint_path"]).exists()
    assert Path(outputs["best_checkpoint_path"]).exists()
    assert Path(outputs["final_checkpoint_path"]).name == "final.pt"
    assert Path(outputs["best_checkpoint_path"]).name == "best.pt"


def test_build_checkpoint_evaluation_metadata_uses_matching_validation_threshold() -> (
    None
):
    metadata = build_checkpoint_evaluation_metadata(
        checkpoint_monitor_metric="val_synth_vus_pr",
        epoch_metrics={
            "val_synth_threshold": 0.125,
            "val_synth_vus_pr": 0.8,
        },
        base_extra_state={"memory_initialized": True},
    )

    assert metadata == {
        "memory_initialized": True,
        "evaluation_threshold": 0.125,
        "evaluation_threshold_metric_name": "val_synth_threshold",
        "evaluation_threshold_source": "checkpoint::val_synth_threshold",
    }


def test_build_checkpoint_evaluation_metadata_preserves_base_state_when_threshold_missing() -> (
    None
):
    metadata = build_checkpoint_evaluation_metadata(
        checkpoint_monitor_metric="val_loss",
        epoch_metrics={"val_loss": 1.0},
        base_extra_state={"memory_initialized": False},
    )

    assert metadata == {"memory_initialized": False}
