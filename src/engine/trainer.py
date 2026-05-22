from __future__ import annotations

"""Epoch-based training loop shared by the offline models.

The engine stays intentionally small. A new reader should notice that this file
does not know model-specific losses; it only moves batches, calls stage methods,
logs metrics, and saves checkpoints.
"""

from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_batch
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import (
    Evaluator,
    reconstruct_pointwise_records_from_window_payload,
    select_point_score_threshold,
)
from src.engine.logger import ExperimentLogger
from src.metrics.pointwise import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_pointwise_metrics,
)
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
        cosine_scheduler_config: dict[str, Any] | None = None,
        gradient_clip_norm: float | None = None,
        validation_evaluator_config: dict[str, Any] | None = None,
        checkpoint_monitor_metric: str | None = None,
        enable_reconstruction_diagnostics: bool = False,
        diagnostics_log_interval_steps: int = 1,
        diagnostics_include_grad_norm: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor_metric = scheduler_monitor_metric
        self.checkpoint_manager = checkpoint_manager
        self.experiment_logger = experiment_logger
        self.device = device
        self.cosine_scheduler_config = cosine_scheduler_config
        self.gradient_clip_norm = gradient_clip_norm
        self.validation_evaluator_config = validation_evaluator_config
        self.checkpoint_monitor_metric = checkpoint_monitor_metric
        self.enable_reconstruction_diagnostics = enable_reconstruction_diagnostics
        self.diagnostics_log_interval_steps = diagnostics_log_interval_steps
        self.diagnostics_include_grad_norm = diagnostics_include_grad_norm
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

    def _aggregate_reconstruction_diagnostics(
        self, *, batch_logs: list[dict[str, float]]
    ) -> dict[str, float]:
        if not self.enable_reconstruction_diagnostics or not batch_logs:
            return {}

        diagnostic_metrics: dict[str, float] = {}
        for stage_name in ("train", "val", "val_synth", "test"):
            reconstruction_loss_key = f"{stage_name}_reconstruction_loss"
            if reconstruction_loss_key not in batch_logs[0]:
                continue
            reconstruction_losses = np.asarray(
                [
                    float(batch_log[reconstruction_loss_key])
                    for batch_log in batch_logs
                    if reconstruction_loss_key in batch_log
                ],
                dtype=np.float64,
            )
            if reconstruction_losses.size == 0:
                continue

            mean_loss = float(np.mean(reconstruction_losses))
            std_loss = float(np.std(reconstruction_losses))
            p50_loss = float(np.percentile(reconstruction_losses, 50))
            p95_loss = float(np.percentile(reconstruction_losses, 95))

            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_std"] = (
                std_loss
            )
            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_min"] = (
                float(np.min(reconstruction_losses))
            )
            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_max"] = (
                float(np.max(reconstruction_losses))
            )
            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_p90"] = (
                float(np.percentile(reconstruction_losses, 90))
            )
            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_p95"] = (
                p95_loss
            )
            diagnostic_metrics[f"diag/recon/{stage_name}_reconstruction_loss_cv"] = (
                std_loss / (mean_loss + 1.0e-12)
            )
            diagnostic_metrics[
                f"diag/recon/{stage_name}_reconstruction_loss_p95_to_p50"
            ] = p95_loss / (p50_loss + 1.0e-12)
        return diagnostic_metrics

    def _include_reconstruction_diagnostics_for_step(self, *, step_index: int) -> bool:
        if not self.enable_reconstruction_diagnostics:
            return False
        return step_index % self.diagnostics_log_interval_steps == 0

    def _build_filtered_step_log(
        self, *, step_log: dict[str, float], step_index: int
    ) -> dict[str, float]:
        if self._include_reconstruction_diagnostics_for_step(step_index=step_index):
            return step_log
        return {
            metric_name: metric_value
            for metric_name, metric_value in step_log.items()
            if not metric_name.startswith("diag/recon/")
        }

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
        num_classes = concatenated_logits.shape[-1]
        if num_classes == 2:
            classification_metrics = compute_binary_classification_metrics(
                logits=concatenated_logits,
                labels=concatenated_labels,
            )
        else:
            classification_metrics = compute_multiclass_classification_metrics(
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

    def _set_optimizer_learning_rate(self, new_learning_rate: float) -> None:
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = new_learning_rate

    def _step_cosine_learning_rate_scheduler(
        self,
        *,
        epoch_index: int,
        train_batch_index: int,
        num_training_batches: int,
    ) -> float:
        if self.cosine_scheduler_config is None:
            raise ValueError("cosine_scheduler_config must be set for cosine stepping")

        from scripts.train import compute_candi_style_cosine_learning_rate

        current_progress = epoch_index + float(train_batch_index) / num_training_batches
        updated_learning_rate = compute_candi_style_cosine_learning_rate(
            base_learning_rate=float(
                self.cosine_scheduler_config["base_learning_rate"]
            ),
            current_progress=current_progress,
            total_epochs=int(self.cosine_scheduler_config["total_epochs"]),
            warmup_epochs=int(self.cosine_scheduler_config["warmup_epochs"]),
            warmup_start_lr=float(self.cosine_scheduler_config["warmup_start_lr"]),
            cosine_end_lr=float(self.cosine_scheduler_config["cosine_end_lr"]),
            cosine_after_warmup=bool(
                self.cosine_scheduler_config["cosine_after_warmup"]
            ),
        )
        self._set_optimizer_learning_rate(updated_learning_rate)
        return updated_learning_rate

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
        checkpoint_monitor_metric = self.checkpoint_monitor_metric
        if checkpoint_monitor_metric is None:
            checkpoint_monitor_metric = (
                "val_loss"
                if self.scheduler_monitor_metric is None
                else self.scheduler_monitor_metric
            )
        checkpoint_monitor_modes = {
            "val_loss": "min",
            "val_synth_loss": "min",
            "val_synth_roc_auc": "max",
            "val_synth_pr_auc": "max",
            "val_synth_vus_pr": "max",
            "val_vus_pr": "max",
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
        pointwise_label_batch_key: str | None = None,
    ) -> tuple[
        list[dict[str, float]],
        list[torch.Tensor],
        list[torch.Tensor],
        list[float],
        list[dict[str, Any]],
    ]:
        stage_logs: list[dict[str, float]] = []
        logits_history: list[torch.Tensor] = []
        label_history: list[torch.Tensor] = []
        forward_pass_seconds_history: list[float] = []
        pointwise_payloads: list[dict[str, Any]] = []
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
                filtered_step_log = self._build_filtered_step_log(
                    step_log=step_output["log"],
                    step_index=val_batch_index,
                )
                console_print(
                    stage_name.upper(),
                    "Completed validation batch",
                    epoch=epoch_index + 1,
                    batch_index=val_batch_index,
                    step_log=filtered_step_log,
                )
                stage_logs.append(filtered_step_log)
                if (
                    step_output["outputs"].get("logits") is not None
                    and "classification_labels" in step_output["batch"]
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
                if (
                    pointwise_label_batch_key is not None
                    and step_output["outputs"].get("point_scores") is not None
                    and pointwise_label_batch_key in step_output["batch"]
                ):
                    pointwise_payloads.append(
                        {
                            "meta": step_output["batch"]["meta"],
                            "point_scores": step_output["outputs"]["point_scores"]
                            .detach()
                            .cpu(),
                            "point_labels": step_output["batch"][
                                pointwise_label_batch_key
                            ]
                            .detach()
                            .cpu(),
                        }
                    )

        return (
            stage_logs,
            logits_history,
            label_history,
            forward_pass_seconds_history,
            pointwise_payloads,
        )

    def _aggregate_reconstructed_pointwise_metrics(
        self,
        *,
        data_loader: Any,
        batch_payloads: list[dict[str, Any]],
        stage_name: str,
    ) -> dict[str, float]:
        if not batch_payloads:
            return {}
        if not hasattr(data_loader, "dataset") or not hasattr(
            data_loader.dataset, "sequences"
        ):
            return {}

        sequences_by_entity = Evaluator._build_sequences_by_entity(data_loader)
        reconstructed_records = reconstruct_pointwise_records_from_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payloads=batch_payloads,
        )
        concatenated_scores = np.concatenate(
            [record["point_scores"].numpy() for record in reconstructed_records],
            axis=0,
        )
        concatenated_labels = np.concatenate(
            [record["point_labels"].numpy() for record in reconstructed_records],
            axis=0,
        )
        threshold = select_point_score_threshold(concatenated_scores, quantile=0.95)
        pointwise_metrics = compute_pointwise_metrics(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
            threshold=threshold,
            vus_max_buffer_size=(
                0
                if self.validation_evaluator_config is None
                else self.validation_evaluator_config.get("vus_max_buffer_size")
            ),
            vus_num_thresholds=(
                200
                if self.validation_evaluator_config is None
                else int(
                    self.validation_evaluator_config.get("vus_num_thresholds", 200)
                )
            ),
        )
        pointwise_metrics["threshold"] = threshold
        return {
            f"{stage_name}_{metric_name}_pointwise": metric_value
            if metric_name not in {"vus_pr", "threshold"}
            else metric_value
            for metric_name, metric_value in pointwise_metrics.items()
            if metric_name not in {"vus_pr", "threshold"}
        } | {
            f"{stage_name}_vus_pr": pointwise_metrics["vus_pr"],
            f"{stage_name}_threshold": pointwise_metrics["threshold"],
        }

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
            batch_learning_rates: list[float] = []
            gradient_norm_history: list[float] = []
            clipped_step_count = 0
            console_print("TRAIN", "Starting epoch", epoch=epoch_index + 1)
            for train_batch_index, train_batch in enumerate(train_loader, start=1):
                if self.cosine_scheduler_config is not None:
                    batch_learning_rates.append(
                        self._step_cosine_learning_rate_scheduler(
                            epoch_index=epoch_index,
                            train_batch_index=train_batch_index - 1,
                            num_training_batches=len(train_loader),
                        )
                    )
                batch_on_device = self._move_batch_to_device(train_batch)
                console_print(
                    "TRAIN",
                    "Processing training batch",
                    epoch=epoch_index + 1,
                    batch_index=train_batch_index,
                )
                step_output = self.model.training_step(batch_on_device)
                filtered_step_log = self._build_filtered_step_log(
                    step_log=step_output["log"],
                    step_index=train_batch_index,
                )
                loss = step_output["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm is not None:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_norm,
                    )
                    gradient_norm_value = float(gradient_norm.detach().cpu())
                    gradient_norm_history.append(gradient_norm_value)
                    if gradient_norm_value > self.gradient_clip_norm:
                        clipped_step_count += 1
                self.optimizer.step()
                console_print(
                    "TRAIN",
                    "Completed optimizer step",
                    epoch=epoch_index + 1,
                    batch_index=train_batch_index,
                    loss=float(loss.detach().cpu()),
                    step_log=filtered_step_log,
                )
                train_logs.append(filtered_step_log)
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
            (
                val_logs,
                val_logits_history,
                val_label_history,
                val_forward_pass_seconds_history,
                _val_pointwise_payloads,
            ) = self._run_validation_epoch(
                val_loader=val_loader,
                epoch_index=epoch_index,
                stage_name="val",
                step_method_name="validation_step",
            )
            val_synth_logs: list[dict[str, float]] = []
            val_synth_logits_history: list[torch.Tensor] = []
            val_synth_label_history: list[torch.Tensor] = []
            val_synth_forward_pass_seconds_history: list[float] = []
            val_synth_pointwise_payloads: list[dict[str, Any]] = []
            if hasattr(self.model, "synthetic_validation_step"):
                if hasattr(self.model, "prepare_synthetic_validation_epoch"):
                    self.model.prepare_synthetic_validation_epoch()
                (
                    val_synth_logs,
                    val_synth_logits_history,
                    val_synth_label_history,
                    val_synth_forward_pass_seconds_history,
                    val_synth_pointwise_payloads,
                ) = self._run_validation_epoch(
                    val_loader=val_loader,
                    epoch_index=epoch_index,
                    stage_name="val_synth",
                    step_method_name="synthetic_validation_step",
                    pointwise_label_batch_key="synthetic_anomaly_mask",
                )

            epoch_metrics = {"epoch": epoch_index + 1}
            epoch_metrics.update(self._aggregate_logs(train_logs))
            epoch_metrics.update(self._aggregate_logs(val_logs))
            epoch_metrics.update(self._aggregate_logs(val_synth_logs))
            epoch_metrics.update(
                self._aggregate_reconstruction_diagnostics(batch_logs=train_logs)
            )
            epoch_metrics.update(
                self._aggregate_reconstruction_diagnostics(batch_logs=val_logs)
            )
            epoch_metrics.update(
                self._aggregate_reconstruction_diagnostics(batch_logs=val_synth_logs)
            )
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
                    logits_history=val_logits_history,
                    label_history=val_label_history,
                    forward_pass_seconds_history=val_forward_pass_seconds_history,
                    stage_name="val",
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
            epoch_metrics.update(
                self._aggregate_reconstructed_pointwise_metrics(
                    data_loader=val_loader,
                    batch_payloads=val_synth_pointwise_payloads,
                    stage_name="val_synth",
                )
            )
            if self.validation_evaluator_config is not None:
                validation_evaluation_outputs = Evaluator(
                    device=self.device,
                    vus_max_buffer_size=self.validation_evaluator_config.get(
                        "vus_max_buffer_size"
                    ),
                    vus_num_thresholds=int(
                        self.validation_evaluator_config.get("vus_num_thresholds", 200)
                    ),
                ).evaluate(model=self.model, data_loader=val_loader)
                epoch_metrics.update(
                    {
                        f"val_{metric_name}": metric_value
                        for metric_name, metric_value in validation_evaluation_outputs[
                            "metrics"
                        ].items()
                    }
                )
            if batch_learning_rates:
                epoch_metrics["optimizer_lr_start"] = batch_learning_rates[0]
                epoch_metrics["optimizer_lr_end"] = batch_learning_rates[-1]
                epoch_metrics["optimizer_lr_min"] = min(batch_learning_rates)
                epoch_metrics["optimizer_lr_max"] = max(batch_learning_rates)
            if gradient_norm_history:
                epoch_metrics["gradient_norm_max"] = max(gradient_norm_history)
                epoch_metrics["gradient_norm_last"] = gradient_norm_history[-1]
                epoch_metrics["gradient_clipped_steps"] = float(clipped_step_count)
                if self.enable_reconstruction_diagnostics and (
                    self.diagnostics_include_grad_norm
                ):
                    epoch_metrics["diag/grad/train_gradient_norm_mean"] = float(
                        sum(gradient_norm_history) / len(gradient_norm_history)
                    )
                    epoch_metrics["diag/grad/train_gradient_norm_std"] = float(
                        np.std(np.asarray(gradient_norm_history, dtype=np.float64))
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
