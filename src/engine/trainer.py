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
        scheduler: Any | None,
        scheduler_monitor_metric: str | None,
        checkpoint_manager: CheckpointManager,
        experiment_logger: ExperimentLogger,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor_metric = scheduler_monitor_metric
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
        console_print(
            "TRAIN",
            "Moved batch to device",
            device=self.device,
            **summarize_batch(batch_on_device),
        )
        return batch_on_device

    def _aggregate_logs(self, batch_logs: list[dict[str, float]]) -> dict[str, float]:
        if not batch_logs:
            return {}

        aggregated_logs: dict[str, float] = {}
        for key in batch_logs[0]:
            aggregated_logs[key] = sum(
                batch_log[key] for batch_log in batch_logs
            ) / len(batch_logs)
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
            prefixed_metrics[f"{stage_name}_forward_pass_seconds_mean"] = sum(
                forward_pass_seconds_history
            ) / len(forward_pass_seconds_history)
        return prefixed_metrics

    def _get_optimizer_learning_rates(self) -> list[float]:
        return [
            float(parameter_group["lr"])
            for parameter_group in self.optimizer.param_groups
        ]

    def _step_learning_rate_scheduler(
        self, epoch_metrics: dict[str, Any]
    ) -> dict[str, float]:
        learning_rate_metrics: dict[str, float] = {}
        current_learning_rates = self._get_optimizer_learning_rates()
        learning_rate_metrics["optimizer_lr"] = current_learning_rates[0]
        for group_index, learning_rate in enumerate(current_learning_rates):
            learning_rate_metrics[f"optimizer_lr_group_{group_index}"] = learning_rate

        if self.scheduler is None:
            learning_rate_metrics["scheduler_lr_reduced"] = 0.0
            return learning_rate_metrics

        if self.scheduler_monitor_metric is None:
            raise ValueError(
                "scheduler_monitor_metric must be set when scheduler is configured"
            )
        if self.scheduler_monitor_metric not in epoch_metrics:
            raise KeyError(
                f"Scheduler monitor metric '{self.scheduler_monitor_metric}' is missing from epoch metrics"
            )

        monitor_value = float(epoch_metrics[self.scheduler_monitor_metric])
        learning_rate_metrics[f"scheduler_monitor_{self.scheduler_monitor_metric}"] = (
            monitor_value
        )
        previous_learning_rates = list(current_learning_rates)
        self.scheduler.step(monitor_value)
        updated_learning_rates = self._get_optimizer_learning_rates()
        learning_rate_metrics["optimizer_lr"] = updated_learning_rates[0]
        for group_index, learning_rate in enumerate(updated_learning_rates):
            learning_rate_metrics[f"optimizer_lr_group_{group_index}"] = learning_rate
        learning_rate_metrics["scheduler_lr_reduced"] = float(
            any(
                updated_learning_rate < previous_learning_rate
                for updated_learning_rate, previous_learning_rate in zip(
                    updated_learning_rates, previous_learning_rates
                )
            )
        )
        console_print(
            "TRAIN",
            "Stepped learning rate scheduler",
            scheduler_monitor_metric=self.scheduler_monitor_metric,
            monitor_value=monitor_value,
            previous_learning_rates=previous_learning_rates,
            updated_learning_rates=updated_learning_rates,
            lr_reduced=learning_rate_metrics["scheduler_lr_reduced"],
        )
        return learning_rate_metrics

    def _resolve_best_checkpoint_monitor(self) -> tuple[str, str]:
        checkpoint_monitor_metric = (
            "val_loss"
            if self.scheduler_monitor_metric is None
            else self.scheduler_monitor_metric
        )
        checkpoint_monitor_modes = {
            "val_loss": "min",
            "val_synth_roc_auc": "max",
            "val_synth_pr_auc": "max",
        }
        if checkpoint_monitor_metric not in checkpoint_monitor_modes:
            raise ValueError(
                f"Unsupported checkpoint monitor metric: {checkpoint_monitor_metric}"
            )
        return checkpoint_monitor_metric, checkpoint_monitor_modes[
            checkpoint_monitor_metric
        ]

    def _build_initial_best_checkpoint_value(self, monitor_mode: str) -> float:
        if monitor_mode == "min":
            return float("inf")
        if monitor_mode == "max":
            return float("-inf")
        raise ValueError(f"Unsupported checkpoint monitor mode: {monitor_mode}")

    def _is_best_checkpoint_metric_improved(
        self,
        *,
        candidate_metric_value: float,
        best_metric_value: float,
        monitor_mode: str,
    ) -> bool:
        if monitor_mode == "min":
            return candidate_metric_value <= best_metric_value
        if monitor_mode == "max":
            return candidate_metric_value >= best_metric_value
        raise ValueError(f"Unsupported checkpoint monitor mode: {monitor_mode}")

    def _run_validation_epoch(
        self,
        *,
        val_loader: Any,
        epoch_index: int,
        stage_name: str,
        step_method_name: str,
    ) -> tuple[
        list[dict[str, float]], list[torch.Tensor], list[torch.Tensor], list[float]
    ]:
        stage_logs: list[dict[str, float]] = []
        logits_history: list[torch.Tensor] = []
        label_history: list[torch.Tensor] = []
        forward_pass_seconds_history: list[float] = []
        step_method = getattr(self.model, step_method_name)

        with torch.no_grad():
            for val_batch_index, val_batch in enumerate(val_loader, start=1):
                batch_on_device = self._move_batch_to_device(val_batch)
                console_print(
                    stage_name.upper(),
                    "Processing validation batch",
                    epoch=epoch_index + 1,
                    batch_index=val_batch_index,
                )
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
                    logits_history.append(
                        step_output["outputs"]["logits"].detach().cpu()
                    )
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
        (
            best_checkpoint_monitor_metric,
            best_checkpoint_monitor_mode,
        ) = self._resolve_best_checkpoint_monitor()
        best_checkpoint_metric_value = self._build_initial_best_checkpoint_value(
            best_checkpoint_monitor_mode
        )
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
                self.model.set_epoch_context(
                    epoch_index=epoch_index, total_epochs=epochs
                )
            if hasattr(self.model, "maybe_initialize_memories_from_loader"):
                memory_initialized = self.model.maybe_initialize_memories_from_loader(
                    train_loader=train_loader,
                    device=self.device,
                )
                console_print(
                    "TRAIN",
                    "Checked prototype memory initialization hook",
                    epoch=epoch_index + 1,
                    memory_initialized=memory_initialized,
                )

            train_logs: list[dict[str, float]] = []
            train_logits_history: list[torch.Tensor] = []
            train_label_history: list[torch.Tensor] = []
            train_forward_pass_seconds_history: list[float] = []
            console_print("TRAIN", "Starting epoch", epoch=epoch_index + 1)
            for train_batch_index, train_batch in enumerate(train_loader, start=1):
                batch_on_device = self._move_batch_to_device(train_batch)
                console_print(
                    "TRAIN",
                    "Processing training batch",
                    epoch=epoch_index + 1,
                    batch_index=train_batch_index,
                )
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
                    train_logits_history.append(
                        step_output["outputs"]["logits"].detach().cpu()
                    )
                    train_label_history.append(
                        step_output["batch"]["classification_labels"].detach().cpu()
                    )
                if "forward_pass_seconds" in step_output["outputs"]["aux"]:
                    train_forward_pass_seconds_history.append(
                        float(step_output["outputs"]["aux"]["forward_pass_seconds"])
                    )

            self.model.eval()
            val_logs, _, _, val_forward_pass_seconds_history = (
                self._run_validation_epoch(
                    val_loader=val_loader,
                    epoch_index=epoch_index,
                    stage_name="val",
                    step_method_name="validation_step",
                )
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
            epoch_metrics.update(self._step_learning_rate_scheduler(epoch_metrics))
            self.metric_history.append(epoch_metrics)
            self.experiment_logger.log_metrics(epoch_metrics)
            console_print(
                "TRAIN",
                "Completed epoch",
                epoch=epoch_index + 1,
                epoch_metrics=epoch_metrics,
            )

            if best_checkpoint_monitor_metric not in epoch_metrics:
                raise KeyError(
                    f"Best checkpoint monitor metric '{best_checkpoint_monitor_metric}' is missing from epoch metrics"
                )
            current_checkpoint_metric_value = float(
                epoch_metrics[best_checkpoint_monitor_metric]
            )
            if self._is_best_checkpoint_metric_improved(
                candidate_metric_value=current_checkpoint_metric_value,
                best_metric_value=best_checkpoint_metric_value,
                monitor_mode=best_checkpoint_monitor_mode,
            ):
                best_checkpoint_metric_value = current_checkpoint_metric_value
                console_print(
                    "CHECKPOINT",
                    "Checkpoint monitor improved; saving best checkpoint",
                    epoch=epoch_index + 1,
                    checkpoint_monitor_metric=best_checkpoint_monitor_metric,
                    checkpoint_monitor_mode=best_checkpoint_monitor_mode,
                    checkpoint_monitor_value=current_checkpoint_metric_value,
                    best_checkpoint_metric_value=best_checkpoint_metric_value,
                )
                best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    checkpoint_name="best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler_state=scaler_state,
                    config=config,
                    epoch=epoch_index + 1,
                    metric_history=self.metric_history,
                    extra_state=(
                        self.model.get_checkpoint_extra_state()
                        if hasattr(self.model, "get_checkpoint_extra_state")
                        else None
                    ),
                )

        return {
            "best_checkpoint_path": best_checkpoint_path,
            "metric_history": self.metric_history,
        }
