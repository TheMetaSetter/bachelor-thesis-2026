from __future__ import annotations
"""Epoch-based training loop shared by the offline models.

The engine stays intentionally small. A new reader should notice that this file
does not know model-specific losses; it only moves batches, calls stage methods,
logs metrics, and saves checkpoints.
"""

from typing import Any

import torch

from src.core.console import console_print, summarize_batch
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.models.base_model import BaseModel


class Trainer:
    def __init__(
        self,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: CheckpointManager,
        experiment_logger: ExperimentLogger,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.experiment_logger = experiment_logger
        self.device = device
        self.metric_history: list[dict[str, Any]] = []

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Keeping device transfer in one helper avoids repeating the same logic
        # in every script and keeps the model files focused on model behavior.
        batch_on_device = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        console_print("TRAIN", "Moved batch to device", device=self.device, **summarize_batch(batch_on_device))
        return batch_on_device

    def _aggregate_logs(self, batch_logs: list[dict[str, float]]) -> dict[str, float]:
        if not batch_logs:
            return {}

        aggregated_logs: dict[str, float] = {}
        for key in batch_logs[0]:
            aggregated_logs[key] = sum(batch_log[key] for batch_log in batch_logs) / len(batch_logs)
        return aggregated_logs

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epochs: int,
    ) -> dict[str, Any]:
        # The trainer owns loop mechanics only. The model owns the meaning of a
        # training step, including optional schedules such as fusion warm-up.
        best_val_loss = float("inf")
        best_checkpoint_path = None

        self.model.to(self.device)
        console_print(
            "TRAIN",
            "Starting offline training loop",
            device=self.device,
            epochs=epochs,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
        )
        for epoch_index in range(epochs):
            # Call model-owned training step
            self.model.train()
            if hasattr(self.model, "set_epoch_context"):
                # Some models need epoch context to update schedules without
                # creating a second training codepath in the engine.
                self.model.set_epoch_context(epoch_index=epoch_index, total_epochs=epochs)
            
            train_logs: list[dict[str, float]] = []
            console_print("TRAIN", "Starting epoch", epoch=epoch_index + 1)
            for train_batch_index, train_batch in enumerate(train_loader, start=1):
                batch_on_device = self._move_batch_to_device(train_batch)
                console_print("TRAIN", "Processing training batch", epoch=epoch_index + 1, batch_index=train_batch_index)
                step_output = self.model.training_step(batch_on_device)
                loss = step_output["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                console_print(
                    "TRAIN",
                    "Completed optimizer step",
                    epoch=epoch_index + 1,
                    batch_index=train_batch_index,
                    loss=float(loss.detach().cpu()),
                    step_log=step_output["log"],
                )
                train_logs.append(step_output["log"])

            self.model.eval()
            val_logs: list[dict[str, float]] = []
            with torch.no_grad():
                for val_batch_index, val_batch in enumerate(val_loader, start=1):
                    batch_on_device = self._move_batch_to_device(val_batch)
                    console_print("VAL", "Processing validation batch", epoch=epoch_index + 1, batch_index=val_batch_index)
                    step_output = self.model.validation_step(batch_on_device)
                    console_print(
                        "VAL",
                        "Completed validation batch",
                        epoch=epoch_index + 1,
                        batch_index=val_batch_index,
                        step_log=step_output["log"],
                    )
                    val_logs.append(step_output["log"])

            epoch_metrics = {"epoch": epoch_index + 1}
            epoch_metrics.update(self._aggregate_logs(train_logs))
            epoch_metrics.update(self._aggregate_logs(val_logs))
            self.metric_history.append(epoch_metrics)
            self.experiment_logger.log_metrics(epoch_metrics)
            console_print("TRAIN", "Completed epoch", epoch=epoch_index + 1, epoch_metrics=epoch_metrics)

            current_val_loss = float(epoch_metrics.get("val_loss", float("inf")))
            if current_val_loss <= best_val_loss:
                best_val_loss = current_val_loss
                console_print("CHECKPOINT", "Validation loss improved; saving best checkpoint", epoch=epoch_index + 1, best_val_loss=best_val_loss)
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
