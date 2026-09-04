from __future__ import annotations

from typing import Any

import torch

from src.core.console import console_print, summarize_batch
from src.data.stream import OnlineWindowBatcher
from src.engine.checkpoint import CheckpointManager
from src.engine.logger import ExperimentLogger
from src.models.online_adaptation import OnlineAdaptationModel


class OnlineLoop:
    def __init__(
        self,
        model: OnlineAdaptationModel,
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
        batch_on_device = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        console_print(
            "ONLINE",
            "Moved online batch to device",
            device=self.device,
            **summarize_batch(batch_on_device),
        )
        return batch_on_device

    def _measure_update_norm(
        self,
        before_parameters: list[torch.Tensor],
        after_parameters: list[torch.Tensor],
    ) -> float:
        update_norm = 0.0
        for before_parameter, after_parameter in zip(
            before_parameters, after_parameters
        ):
            update_norm += float(
                torch.sum((after_parameter - before_parameter) ** 2).detach().cpu()
            )
        return update_norm**0.5

    def _build_checkpoint_extra_state(
        self,
        online_batcher: OnlineWindowBatcher,
    ) -> dict[str, Any]:
        return {
            "stream_state_dict": online_batcher.state_dict(),
            "projector_anchor_state_dict": self.model.get_projector_anchor_state_dict(),
            "target_param_group": self.model.target_param_group,
            "online_metric_history": self.metric_history,
            "reset_policy_state": {
                "reset_policy": self.model.reset_policy,
                "reset_alignment_threshold": self.model.reset_alignment_threshold,
            },
        }

    def run(
        self,
        online_batcher: OnlineWindowBatcher,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        max_online_steps: int,
        log_every_n_steps: int,
        checkpoint_every_n_steps: int,
    ) -> dict[str, Any]:
        # step 1: Prepare the model and output state for online adaptation.
        self.model.to(self.device)
        records: list[dict[str, Any]] = []
        final_checkpoint_path = None
        console_print(
            "ONLINE",
            "Starting online adaptation loop",
            device=self.device,
            max_online_steps=max_online_steps,
            log_every_n_steps=log_every_n_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
        )

        # step 2: Process the stream one causal batch at a time.
        for step_index, batch in enumerate(online_batcher, start=1):
            # step 3: Stop when the configured online-step limit is reached.
            if step_index > max_online_steps:
                break

            # step 4: Move the current batch to the selected device.
            batch_on_device = self._move_batch_to_device(batch)
            console_print("ONLINE", "Processing online step", step_index=step_index)

            # step 5: Measure scores before changing model parameters.
            self.model.eval()
            with torch.no_grad():
                pre_outputs = self.model.forward(batch_on_device)
            pre_window_score_mean = float(
                pre_outputs["window_scores"].mean().detach().cpu()
            )

            # step 6: Snapshot the parameters that online adaptation may change.
            trainable_parameters = [
                parameter.detach().clone()
                for parameter in self.model.get_parameter_group(
                    self.model.target_param_group
                )
            ]

            # step 7: Compute the adaptation loss for the current batch.
            self.model.train()
            step_output = self.model.training_step(batch_on_device)
            loss = step_output["loss"]

            # step 8: Apply one optimizer update.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step 9: Measure scores and parameter changes after adaptation.
            self.model.eval()
            with torch.no_grad():
                post_outputs = self.model.forward(batch_on_device)
            post_window_score_mean = float(
                post_outputs["window_scores"].mean().detach().cpu()
            )

            updated_parameters = [
                parameter.detach().clone()
                for parameter in self.model.get_parameter_group(
                    self.model.target_param_group
                )
            ]
            update_norm = self._measure_update_norm(
                trainable_parameters, updated_parameters
            )

            # step 10: Record and periodically log online metrics.
            step_metrics = {
                "online/step": step_index,
                "online/pre_window_score_mean": pre_window_score_mean,
                "online/post_window_score_mean": post_window_score_mean,
                "online/update_norm": update_norm,
                "online/alignment_loss": float(
                    post_outputs["aux"]["alignment_loss"].detach().cpu()
                ),
                "online/prototype_alignment_loss": float(
                    post_outputs["aux"]["prototype_alignment_loss"].detach().cpu()
                ),
                "online/projector_drift": float(
                    post_outputs["aux"]["projector_drift"].detach().cpu()
                ),
            }
            self.metric_history.append(step_metrics)
            if step_index % log_every_n_steps == 0:
                self.experiment_logger.log_metrics(step_metrics)
            console_print(
                "ONLINE",
                "Completed online step",
                step_index=step_index,
                step_metrics=step_metrics,
            )

            records.append(
                {
                    "step": step_index,
                    "entity_ids": [meta["entity_id"] for meta in batch["meta"]],
                    "stream_steps": [meta["stream_step"] for meta in batch["meta"]],
                    "start_indices": [meta["start_index"] for meta in batch["meta"]],
                    "end_indices": [meta["end_index"] for meta in batch["meta"]],
                    "pre_window_score_mean": pre_window_score_mean,
                    "post_window_score_mean": post_window_score_mean,
                    "alignment_loss": step_metrics["online/alignment_loss"],
                    "projector_drift": step_metrics["online/projector_drift"],
                }
            )

            # step 11: Save a periodic checkpoint when the interval is reached.
            if step_index % checkpoint_every_n_steps == 0:
                console_print(
                    "CHECKPOINT",
                    "Saving periodic online checkpoint",
                    step_index=step_index,
                )
                final_checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    checkpoint_name=f"online_step_{step_index}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=None,
                    scaler_state=scaler_state,
                    config=config,
                    epoch=step_index,
                    metric_history=self.metric_history,
                    extra_state=self._build_checkpoint_extra_state(online_batcher),
                )

        # step 12: Save the final online checkpoint after the stream ends.
        console_print(
            "CHECKPOINT",
            "Saving final online checkpoint",
            total_steps=len(self.metric_history),
        )
        final_checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_name="online_final.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            scaler_state=scaler_state,
            config=config,
            epoch=len(self.metric_history),
            metric_history=self.metric_history,
            extra_state=self._build_checkpoint_extra_state(online_batcher),
        )

        return {
            "final_checkpoint_path": final_checkpoint_path,
            "metric_history": self.metric_history,
            "records": records,
        }
