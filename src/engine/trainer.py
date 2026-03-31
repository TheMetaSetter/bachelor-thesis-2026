from __future__ import annotations

from typing import Any

import torch

from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Any,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: CheckpointManager,
        experiment_logger: ExperimentLogger,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.experiment_logger = experiment_logger
        self.device = device
        self.metric_history: list[dict[str, Any]] = []

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epochs: int,
    ) -> dict[str, Any]:
        best_val_loss = float("inf")
        best_checkpoint_path = None

        self.model.to(self.device)
        for epoch_index in range(epochs):
            self.model.train()
            train_losses: list[float] = []
            for train_batch in train_loader:
                batch_on_device = self._move_batch_to_device(train_batch)
                step_output = self.task.training_step(self.model, batch_on_device)
                loss = step_output["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            self.model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for val_batch in val_loader:
                    batch_on_device = self._move_batch_to_device(val_batch)
                    step_output = self.task.validation_step(self.model, batch_on_device)
                    val_losses.append(float(step_output["loss"].detach().cpu()))

            epoch_metrics = {
                "epoch": epoch_index + 1,
                "train_loss": sum(train_losses) / max(len(train_losses), 1),
                "val_loss": sum(val_losses) / max(len(val_losses), 1),
            }
            self.metric_history.append(epoch_metrics)
            self.experiment_logger.log_metrics(epoch_metrics)

            if epoch_metrics["val_loss"] <= best_val_loss:
                best_val_loss = epoch_metrics["val_loss"]
                best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    checkpoint_name="best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scaler_state=scaler_state,
                    config=config,
                    epoch=epoch_index + 1,
                    metric_history=self.metric_history,
                )

        return {
            "best_checkpoint_path": best_checkpoint_path,
            "metric_history": self.metric_history,
        }

