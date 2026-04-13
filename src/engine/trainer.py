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
from src.metrics.pointwise import compute_binary_classification_metrics
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

    def _aggregate_multitask_classification_metrics(
        self,
        *,
        logits_history: list[torch.Tensor],
        label_history: list[torch.Tensor],
        forward_pass_seconds_history: list[float],
        stage_name: str,
    ) -> dict[str, float]:
        if not logits_history or not label_history:
            return {}

        concatenated_logits = torch.cat(logits_history, dim=0)
        concatenated_labels = torch.cat(label_history, dim=0)
        classification_metrics = compute_binary_classification_metrics(
            logits=concatenated_logits,
            labels=concatenated_labels,
        )
        prefixed_metrics = {
            f"{stage_name}_{metric_name}": metric_value
            for metric_name, metric_value in classification_metrics.items()
        }
        if forward_pass_seconds_history:
            prefixed_metrics[f"{stage_name}_forward_pass_seconds_mean"] = (
                sum(forward_pass_seconds_history) / len(forward_pass_seconds_history)
            )
        return prefixed_metrics

    def _run_validation_epoch(
        self,
        *,
        val_loader: Any,
        epoch_index: int,
        stage_name: str,
        step_method_name: str,
    ) -> tuple[list[dict[str, float]], list[torch.Tensor], list[torch.Tensor], list[float]]:
        stage_logs: list[dict[str, float]] = []
        logits_history: list[torch.Tensor] = []
        label_history: list[torch.Tensor] = []
        forward_pass_seconds_history: list[float] = []
        step_method = getattr(self.model, step_method_name)

        with torch.no_grad():
            for val_batch_index, val_batch in enumerate(val_loader, start=1):
                batch_on_device = self._move_batch_to_device(val_batch)
                console_print(stage_name.upper(), "Processing validation batch", epoch=epoch_index + 1, batch_index=val_batch_index)
                step_output = step_method(batch_on_device)
                console_print(
                    stage_name.upper(),
                    "Completed validation batch",
                    epoch=epoch_index + 1,
                    batch_index=val_batch_index,
                    step_log=step_output["log"],
                )
                stage_logs.append(step_output["log"])
                if (
                    step_output["outputs"].get("logits") is not None
                    and "classification_labels" in step_output["batch"]
                    and f"{stage_name}_classification_loss" in step_output["log"]
                ):
                    logits_history.append(step_output["outputs"]["logits"].detach().cpu())
                    label_history.append(
                        step_output["batch"]["classification_labels"].detach().cpu()
                    )
                if "forward_pass_seconds" in step_output["outputs"]["aux"]:
                    forward_pass_seconds_history.append(
                        float(step_output["outputs"]["aux"]["forward_pass_seconds"])
                    )

        return stage_logs, logits_history, label_history, forward_pass_seconds_history

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
            train_logits_history: list[torch.Tensor] = []
            train_label_history: list[torch.Tensor] = []
            train_forward_pass_seconds_history: list[float] = []
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
                if (
                    step_output["outputs"].get("logits") is not None
                    and "classification_labels" in step_output["batch"]
                ):
                    train_logits_history.append(step_output["outputs"]["logits"].detach().cpu())
                    train_label_history.append(
                        step_output["batch"]["classification_labels"].detach().cpu()
                    )
                if "forward_pass_seconds" in step_output["outputs"]["aux"]:
                    train_forward_pass_seconds_history.append(
                        float(step_output["outputs"]["aux"]["forward_pass_seconds"])
                    )

            self.model.eval()
            val_logs, _, _, val_forward_pass_seconds_history = self._run_validation_epoch(
                val_loader=val_loader,
                epoch_index=epoch_index,
                stage_name="val",
                step_method_name="validation_step",
            )
            val_synth_logs: list[dict[str, float]] = []
            val_synth_logits_history: list[torch.Tensor] = []
            val_synth_label_history: list[torch.Tensor] = []
            val_synth_forward_pass_seconds_history: list[float] = []
            if hasattr(self.model, "synthetic_validation_step"):
                if hasattr(self.model, "prepare_synthetic_validation_epoch"):
                    self.model.prepare_synthetic_validation_epoch()
                (
                    val_synth_logs,
                    val_synth_logits_history,
                    val_synth_label_history,
                    val_synth_forward_pass_seconds_history,
                ) = self._run_validation_epoch(
                    val_loader=val_loader,
                    epoch_index=epoch_index,
                    stage_name="val_synth",
                    step_method_name="synthetic_validation_step",
                )

            epoch_metrics = {"epoch": epoch_index + 1}
            epoch_metrics.update(self._aggregate_logs(train_logs))
            epoch_metrics.update(self._aggregate_logs(val_logs))
            epoch_metrics.update(self._aggregate_logs(val_synth_logs))
            epoch_metrics.update(
                self._aggregate_multitask_classification_metrics(
                    logits_history=train_logits_history,
                    label_history=train_label_history,
                    forward_pass_seconds_history=train_forward_pass_seconds_history,
                    stage_name="train",
                )
            )
            epoch_metrics.update(
                self._aggregate_multitask_classification_metrics(
                    logits_history=val_synth_logits_history,
                    label_history=val_synth_label_history,
                    forward_pass_seconds_history=val_synth_forward_pass_seconds_history,
                    stage_name="val_synth",
                )
            )
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
