from __future__ import annotations

"""Training step helpers for the thesis multitask model."""

from typing import Any

import torch

from src.core.console import console_print, summarize_label_distribution
from src.models.thesis_multitask_impl.thesis_multitask_components import TWO_STAGE_A_PHASE_NAME


class ThesisMultitaskLossStepMixin:
    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        include_classification_metrics: bool,
    ) -> dict[str, float]:
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            discrete_usage_top1 = 0.0
            discrete_usage_entropy = 0.0
            discrete_usage_concentration = 0.0
            discrete_usage_active_codes = 0.0
        else:
            average_usage = assignment_probabilities.mean(dim=(0, 1))
            average_usage = average_usage / average_usage.sum().clamp_min(self.epsilon)
            discrete_usage_top1 = float(average_usage.max().detach().cpu())
            discrete_usage_entropy = float(
                (
                    -(
                        average_usage * torch.log(average_usage.clamp_min(self.epsilon))
                    ).sum()
                )
                .detach()
                .cpu()
            )
            discrete_usage_concentration = float(
                torch.sum(average_usage.pow(2)).detach().cpu()
            )
            discrete_usage_active_codes = float(
                torch.sum(
                    (
                        average_usage > (1.0 / max(self.discrete_codebook_size * 2, 1))
                    ).float()
                )
                .detach()
                .cpu()
            )
        stage_log = {
            f"{stage_name}_loss": float(loss_terms["total_loss"].detach().cpu()),
            f"{stage_name}_reconstruction_loss": float(
                loss_terms["reconstruction_loss"].detach().cpu()
            ),
            f"{stage_name}_diversity_loss": float(
                loss_terms["diversity_loss"].detach().cpu()
            ),
            f"{stage_name}_variance_loss": float(
                loss_terms["variance_loss"].detach().cpu()
            ),
            f"{stage_name}_covariance_loss": float(
                loss_terms["covariance_loss"].detach().cpu()
            ),
            f"{stage_name}_usage_loss": float(loss_terms["usage_loss"].detach().cpu()),
            f"{stage_name}_gate_loss": float(loss_terms["gate_loss"].detach().cpu()),
            f"{stage_name}_contrastive_loss": float(
                loss_terms["contrastive_loss"].detach().cpu()
            ),
            f"{stage_name}_alpha": float(outputs["aux"]["alpha"].mean().detach().cpu()),
            f"{stage_name}_beta": float(outputs["aux"]["beta"].mean().detach().cpu()),
            f"{stage_name}_alpha_std": float(
                outputs["aux"]["alpha"].std(unbiased=False).detach().cpu()
            ),
            f"{stage_name}_beta_std": float(
                outputs["aux"]["beta"].std(unbiased=False).detach().cpu()
            ),
            f"{stage_name}_continuous_norm": float(
                outputs["aux"]["continuous_branch"]["prototype_context"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_norm": float(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
                .norm(dim=-1)
                .mean()
                .detach()
                .cpu()
            ),
            f"{stage_name}_discrete_usage_top1": discrete_usage_top1,
            f"{stage_name}_discrete_usage_entropy": discrete_usage_entropy,
            f"{stage_name}_discrete_usage_concentration": discrete_usage_concentration,
            f"{stage_name}_discrete_usage_active_codes": discrete_usage_active_codes,
            f"{stage_name}_temperature": float(self.gumbel_temperature),
            f"{stage_name}_usage_lambda": float(self.current_usage_lambda),
            f"{stage_name}_warmup_active": float(self.schedule_state["warmup_active"]),
            f"{stage_name}_memory_initialized": float(
                outputs["aux"]["memory"]["memory_initialized"]
            ),
            f"{stage_name}_memory_training_enabled": float(
                outputs["aux"]["memory"]["memory_training_enabled"]
            ),
            f"{stage_name}_memory_ready_for_initialization": float(
                outputs["aux"]["memory"]["memory_ready_for_initialization"]
            ),
            f"{stage_name}_memory_mode": float(
                outputs["aux"]["memory"]["train_memory_mode"]
            ),
        }
        uncertainty = outputs["aux"].get("uncertainty")
        if uncertainty is not None:
            stage_log[f"diag/uncertainty/{stage_name}_point_score_variance_mean"] = float(
                uncertainty["point_anomaly_score_variance"].mean().detach().cpu()
            )
            stage_log[f"diag/uncertainty/{stage_name}_window_score_variance_mean"] = float(
                uncertainty["window_anomaly_score_variance"].mean().detach().cpu()
            )
            stage_log[f"diag/uncertainty/{stage_name}_reconstruction_variance_mean"] = float(
                uncertainty["reconstruction_variance_full"].mean().detach().cpu()
            )
            stage_log[
                f"diag/uncertainty/{stage_name}_classification_variance_mean"
            ] = float(
                uncertainty["classification_variance_mean"].mean().detach().cpu()
                if uncertainty.get("classification_variance_mean") is not None
                else 0.0
            )
        if stage_name in {"train", "val_synth"}:
            cka_reconstruction_mean = float(
                outputs["aux"]["fusion"]["cka_reconstruction_mean"]
            )
            cka_reconstruction_std = float(
                outputs["aux"]["fusion"]["cka_reconstruction_std"]
            )
            stage_log[f"{stage_name}_cka_reconstruction_mean"] = cka_reconstruction_mean
            stage_log[f"{stage_name}_cka_reconstruction_std"] = cka_reconstruction_std
            stage_log[f"diag/cka/{stage_name}_reconstruction_mean"] = (
                cka_reconstruction_mean
            )
            stage_log[f"diag/cka/{stage_name}_reconstruction_std"] = (
                cka_reconstruction_std
            )
            if self.enable_classification_path:
                cka_classification_mean = float(
                    outputs["aux"]["fusion"]["cka_classification_mean"]
                )
                cka_classification_std = float(
                    outputs["aux"]["fusion"]["cka_classification_std"]
                )
                stage_log[f"{stage_name}_cka_classification_mean"] = (
                    cka_classification_mean
                )
                stage_log[f"{stage_name}_cka_classification_std"] = (
                    cka_classification_std
                )
                stage_log[f"diag/cka/{stage_name}_classification_mean"] = (
                    cka_classification_mean
                )
                stage_log[f"diag/cka/{stage_name}_classification_std"] = (
                    cka_classification_std
                )
        if include_classification_metrics and outputs.get("logits") is not None:
            predicted_labels = torch.argmax(outputs["logits"], dim=-1)
            classification_accuracy = float(
                (predicted_labels == batch["classification_labels"])
                .float()
                .mean()
                .detach()
                .cpu()
            )
            stage_log[f"{stage_name}_classification_loss"] = float(
                loss_terms["classification_loss"].detach().cpu()
            )
            stage_log[f"{stage_name}_classification_accuracy"] = classification_accuracy
        reconstruction_diagnostics = self._compute_reconstruction_diagnostics(
            outputs=outputs,
            batch=batch,
        )
        for metric_name, metric_value in reconstruction_diagnostics.items():
            stage_log[f"diag/recon/{stage_name}_{metric_name}"] = metric_value
        return stage_log

    def _shared_step(
        self,
        batch: dict[str, Any],
        stage_name: str,
        *,
        classification_weight: float,
        include_classification_metrics: bool,
    ) -> dict[str, Any]:
        contrastive_pair = self._prepare_contrastive_pair_batches(batch, stage_name)
        if contrastive_pair is None:
            prepared_batch = self._prepare_batch(batch, stage_name)
            contrastive_loss = self._zero_loss(prepared_batch["x"])
        else:
            clean_batch, augmented_batch = contrastive_pair
            clean_outputs = self.forward(clean_batch, stage_name="val")
            prepared_batch = augmented_batch
            prepared_batch["paired_hidden_for_fusion"] = clean_outputs[
                "hidden"
            ].detach()
            contrastive_loss = self._compute_two_view_contrastive_loss(
                anchor_hidden=clean_outputs["hidden"],
                positive_hidden=self.encoder(prepared_batch)["hidden"],
                synthetic_anomaly_mask=prepared_batch["synthetic_anomaly_mask"],
            )

        outputs = self.forward(prepared_batch, stage_name=stage_name)
        reconstruction_loss = self._compute_reconstruction_loss(outputs, prepared_batch)
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        optional_loss_values = self._compute_optional_loss_terms(outputs)
        score_loss, score_loss_diagnostics = self._compute_point_score_loss(
            outputs,
            prepared_batch,
        )
        if score_loss is None:
            score_loss = self._zero_loss(outputs["recon"])
            score_loss_was_skipped = (
                self.enable_score_loss and self.training_phase == TWO_STAGE_A_PHASE_NAME
            )
        else:
            score_loss_was_skipped = False
        if self.enable_score_loss and self.training_phase == TWO_STAGE_A_PHASE_NAME:
            if score_loss_was_skipped:
                if not hasattr(self, "_score_loss_skipped_batches"):
                    self._score_loss_skipped_batches = 0
                self._score_loss_skipped_batches += 1
                classification_branch_loss = classification_loss
            else:
                classification_branch_loss = 0.5 * (classification_loss + score_loss)
        else:
            classification_branch_loss = classification_loss

        total_loss = self._compute_total_loss(
            reconstruction_loss=reconstruction_loss,
            classification_loss=classification_branch_loss,
            optional_loss_values=optional_loss_values,
            reconstruction_weight=self._phase_reconstruction_weight(),
            classification_weight=(
                min(self._phase_classification_weight(), classification_weight)
                if self.enable_classification_path
                else 0.0
            ),
        )
        if self._phase_uses_contrastive_objective():
            total_loss = (
                total_loss + self._phase_contrastive_weight() * contrastive_loss
            )

        loss_terms = {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
            "score_loss": score_loss,
            "contrastive_loss": contrastive_loss,
            **optional_loss_values,
        }
        console_print(
            stage_name.upper(),
            "Completed multitask stage step",
            batch_size=prepared_batch["x"].shape[0],
            total_loss=float(total_loss.detach().cpu()),
            reconstruction_loss=float(reconstruction_loss.detach().cpu()),
            classification_loss=float(classification_loss.detach().cpu()),
            score_loss=float(score_loss.detach().cpu()),
            diversity_loss=float(optional_loss_values["diversity_loss"].detach().cpu()),
            variance_loss=float(optional_loss_values["variance_loss"].detach().cpu()),
            covariance_loss=float(
                optional_loss_values["covariance_loss"].detach().cpu()
            ),
            usage_loss=float(optional_loss_values["usage_loss"].detach().cpu()),
            gate_loss=float(optional_loss_values["gate_loss"].detach().cpu()),
            contrastive_loss=float(contrastive_loss.detach().cpu()),
            score_loss_skipped_batches=float(
                getattr(self, "_score_loss_skipped_batches", 0)
            ),
            classification_label_distribution=(
                summarize_label_distribution(prepared_batch["classification_labels"])
                if self.enable_classification_path
                else {}
            ),
            alpha=float(outputs["aux"]["alpha"].mean().detach().cpu()),
            beta=float(outputs["aux"]["beta"].mean().detach().cpu()),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        stage_log = self._build_stage_log(
            stage_name,
            outputs,
            loss_terms,
            prepared_batch,
            include_classification_metrics=include_classification_metrics,
        )
        stage_log[f"{stage_name}_score_loss"] = float(score_loss.detach().cpu())
        if self.enable_score_loss and self.training_phase == TWO_STAGE_A_PHASE_NAME:
            stage_log[f"{stage_name}_score_loss_skipped_batches"] = float(
                getattr(self, "_score_loss_skipped_batches", 0)
            )
            for diagnostic_name, diagnostic_value in score_loss_diagnostics.items():
                stage_log[f"diag/score/{stage_name}_{diagnostic_name}"] = float(
                    diagnostic_value.detach().cpu()
                )
        if stage_name == "train":
            self._gradient_profile_train_step_count += 1
            should_log_gradient_conflict = (
                self.enable_gradient_conflict_profiling
                and self._gradient_profile_train_step_count
                % self.gradient_log_every_n_steps
                == 0
            )
            if should_log_gradient_conflict:
                gradient_conflict_logs = self._profile_encoder_gradient_conflict(
                    reconstruction_loss=reconstruction_loss,
                    classification_loss=classification_loss,
                )
                stage_log.update(gradient_conflict_logs)
        return {
            "loss": total_loss,
            "log": stage_log,
            "outputs": outputs,
            "loss_terms": loss_terms,
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="train",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val",
            classification_weight=0.0,
            include_classification_metrics=False,
        )

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="val_synth",
            classification_weight=self.lambda_cls,
            include_classification_metrics=True,
        )

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(
            batch=batch,
            stage_name="test",
            classification_weight=0.0,
            include_classification_metrics=False,
        )
