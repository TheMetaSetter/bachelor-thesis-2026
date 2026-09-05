from __future__ import annotations

"""Core loss computations for the thesis multitask model."""

import math
from typing import Any, Callable

import torch
import torch.nn.functional as F

from src.models.thesis_multitask_impl.thesis_multitask_components import (
    TWO_STAGE_A_PHASE_NAME,
)


class ThesisMultitaskLossCoreMixin:
    def _normalize_branch_tokens(
        self, branch_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_hidden = self.branch_layer_norm(branch_hidden).reshape(
            -1, self.hidden_dim
        )
        feature_mean = normalized_hidden.mean(dim=0, keepdim=True)
        feature_std = normalized_hidden.std(dim=0, unbiased=False, keepdim=True)
        standardized_hidden = (normalized_hidden - feature_mean) / (
            feature_std + self.epsilon
        )
        return normalized_hidden, standardized_hidden

    def _compute_reconstruction_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        squared_reconstruction_error = self.reconstruction_squared_error(outputs, batch)
        if not self.reconstruction_normal_only or "synthetic_anomaly_mask" not in batch:
            return torch.mean(squared_reconstruction_error)

        normal_time_step_mask = self._build_normal_time_step_mask(
            batch, squared_reconstruction_error
        )
        expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
            squared_reconstruction_error
        )
        active_normal_cells = torch.count_nonzero(expanded_normal_mask)
        if int(active_normal_cells.item()) == 0:
            return torch.mean(squared_reconstruction_error)

        return (
            torch.sum(squared_reconstruction_error * expanded_normal_mask)
            / expanded_normal_mask.sum()
        )

    def _compute_reconstruction_diagnostics(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> dict[str, float]:
        squared_reconstruction_error = self.reconstruction_squared_error(outputs, batch)
        per_window_error = squared_reconstruction_error.mean(dim=(1, 2))
        reconstruction = outputs["recon"]
        target = batch["x"]
        samples = (outputs.get("aux", {}).get("stochastic_query") or {}).get(
            "reconstruction_samples"
        )
        if isinstance(samples, torch.Tensor):
            reconstruction, target = samples, target.unsqueeze(1)
        diagnostics = {
            "normalized_input_recon_mse_mean": float(
                (reconstruction - target).square().mean().detach().cpu()
            ),
            "recon_mse_mean": float(torch.mean(per_window_error).detach().cpu()),
            "recon_mse_std": float(
                torch.std(per_window_error, unbiased=False).detach().cpu()
            ),
            "active_normal_cells": float(squared_reconstruction_error.numel()),
            "normal_cell_ratio": 1.0,
            "synthetic_cell_ratio": 0.0,
            "fallback_to_full_mse_flag": 0.0,
        }
        if "synthetic_anomaly_mask" not in batch:
            return diagnostics

        normal_time_step_mask = self._build_normal_time_step_mask(
            batch, squared_reconstruction_error
        )
        expanded_normal_mask = normal_time_step_mask.unsqueeze(-1).expand_as(
            squared_reconstruction_error
        )
        active_normal_cells = float(torch.count_nonzero(expanded_normal_mask).item())
        total_cells = float(expanded_normal_mask.numel())
        normal_cell_ratio = active_normal_cells / max(total_cells, 1.0)
        synthetic_cell_ratio = 1.0 - normal_cell_ratio
        fallback_to_full_mse_flag = 0.0
        if self.reconstruction_normal_only and active_normal_cells <= 0.0:
            fallback_to_full_mse_flag = 1.0

        diagnostics["active_normal_cells"] = active_normal_cells
        diagnostics["normal_cell_ratio"] = normal_cell_ratio
        diagnostics["synthetic_cell_ratio"] = synthetic_cell_ratio
        diagnostics["fallback_to_full_mse_flag"] = fallback_to_full_mse_flag
        return diagnostics

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if outputs.get("logits") is None:
            return self._zero_loss(outputs["recon"])
        if self.use_label_refurbishment:
            target_probabilities = self._build_refurbished_classification_targets(
                batch["classification_labels"],
                outputs["logits"].dtype,
            )
            log_probabilities = F.log_softmax(outputs["logits"], dim=-1)
            return torch.mean(
                torch.sum(-target_probabilities * log_probabilities, dim=-1)
            )

        return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())

    def _build_normal_time_step_mask(
        self,
        batch: dict[str, Any],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        anomaly_mask = batch["synthetic_anomaly_mask"].to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        normal_time_step_mask = 1.0 - anomaly_mask
        return torch.clamp(normal_time_step_mask, min=0.0, max=1.0)

    def _build_refurbished_classification_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        hard_labels = classification_labels.long()
        target_probabilities = F.one_hot(
            hard_labels,
            num_classes=self.num_classes,
        ).to(dtype=target_dtype)

        if self.classification_label_mode == "binary":
            if self.num_classes != 2:
                raise ValueError("Binary label refurbishment requires num_classes == 2")
            target_probabilities[:, 0] = torch.where(
                hard_labels == 0,
                1.0 - self.refurbishment_beta,
                self.refurbishment_alpha,
            )
            target_probabilities[:, 1] = torch.where(
                hard_labels == 0,
                self.refurbishment_beta,
                1.0 - self.refurbishment_alpha,
            )
            return target_probabilities

        target_probabilities = torch.where(
            target_probabilities > 0.0,
            1.0
            - (
                self.refurbishment_alpha
                + self.refurbishment_beta * self.num_classes
                - self.refurbishment_beta
            ),
            self.refurbishment_beta,
        )
        target_probabilities[:, 0] = target_probabilities[:, 0] + (
            self.refurbishment_alpha
        )
        return target_probabilities / target_probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.epsilon)

    def _build_refurbished_binary_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._build_refurbished_classification_targets(
            classification_labels=classification_labels,
            target_dtype=target_dtype,
        )

    def _compute_cross_branch_diversity_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        continuous_hidden = outputs["aux"]["continuous_branch"]["prototype_context"]
        discrete_hidden = outputs["aux"]["discrete_branch"]["quantized_hidden"]
        _, standardized_continuous = self._normalize_branch_tokens(continuous_hidden)
        _, standardized_discrete = self._normalize_branch_tokens(discrete_hidden)
        num_tokens = standardized_continuous.shape[0]
        cross_branch_correlation = (
            standardized_continuous.T @ standardized_discrete / num_tokens
        )
        return cross_branch_correlation.pow(2).mean()

    def _compute_variance_floor_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        variance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
            ]
            normalized_hidden, _ = self._normalize_branch_tokens(branch_hidden)
            feature_std = normalized_hidden.std(dim=0, unbiased=False)
            variance_losses.append(
                F.relu(self.variance_floor_gamma - feature_std).pow(2).mean()
            )
        return torch.stack(variance_losses).sum()

    def _compute_covariance_reduction_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        covariance_losses: list[torch.Tensor] = []
        for branch_name in ["continuous_branch", "discrete_branch"]:
            branch_hidden = outputs["aux"][branch_name][
                "prototype_context"
                if branch_name == "continuous_branch"
                else "quantized_hidden"
            ]
            _, standardized_hidden = self._normalize_branch_tokens(branch_hidden)
            num_tokens = standardized_hidden.shape[0]
            covariance_matrix = standardized_hidden.T @ standardized_hidden / num_tokens
            diagonal_matrix = torch.diag(torch.diag(covariance_matrix))
            off_diagonal_matrix = covariance_matrix - diagonal_matrix
            if self.hidden_dim == 1:
                covariance_losses.append(self._zero_loss(branch_hidden))
            else:
                covariance_losses.append(
                    off_diagonal_matrix.pow(2).sum()
                    / (self.hidden_dim * (self.hidden_dim - 1))
                )
        return torch.stack(covariance_losses).sum()

    def _compute_prototype_usage_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        assignment_probabilities = outputs["aux"]["discrete_branch"][
            "assignment_probabilities"
        ]
        if assignment_probabilities is None or self.discrete_codebook_size <= 0:
            return self._zero_loss(outputs["hidden"])
        average_usage = assignment_probabilities.mean(dim=(0, 1))
        target_usage = torch.full_like(average_usage, 1.0 / self.discrete_codebook_size)
        return torch.sum((average_usage - target_usage) ** 2)

    def _compute_gate_regularization_loss(
        self, outputs: dict[str, Any]
    ) -> torch.Tensor:
        alpha = outputs["aux"]["alpha"]
        beta = outputs["aux"]["beta"]
        max_entropy = math.log(2.0)
        alpha_clamped = torch.clamp(alpha, self.epsilon, 1.0 - self.epsilon)
        beta_clamped = torch.clamp(beta, self.epsilon, 1.0 - self.epsilon)
        alpha_entropy = -(
            alpha_clamped * torch.log(alpha_clamped)
            + (1.0 - alpha_clamped) * torch.log(1.0 - alpha_clamped)
        )
        beta_entropy = -(
            beta_clamped * torch.log(beta_clamped)
            + (1.0 - beta_clamped) * torch.log(1.0 - beta_clamped)
        )
        alpha_penalty = 1.0 - alpha_entropy / max_entropy
        beta_penalty = 1.0 - beta_entropy / max_entropy
        return 0.5 * (alpha_penalty + beta_penalty).mean()

    def _point_mask_from_synthetic_mask(
        self, synthetic_anomaly_mask: torch.Tensor
    ) -> torch.Tensor:
        if synthetic_anomaly_mask.ndim == 2:
            return synthetic_anomaly_mask.bool()
        if synthetic_anomaly_mask.ndim == 3:
            return synthetic_anomaly_mask.bool().any(dim=-1)
        raise ValueError("synthetic_anomaly_mask must have shape [B, L] or [B, L, C]")

    def _compute_point_score_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        """
        This method calculates balanced point-level anomaly score loss.
        This loss pulls anomaly score of normal points to be lower than anomalous points.
        """

        diagnostics: dict[str, torch.Tensor] = {
            "point_score_normal_count": torch.tensor(0, device=outputs["recon"].device),
            "point_score_anomaly_count": torch.tensor(
                0, device=outputs["recon"].device
            ),
        }

        # step 1: Check whether this loss variant is active.
        if not self.enable_score_loss:
            return None, diagnostics
        if self.training_phase != TWO_STAGE_A_PHASE_NAME:
            return None, diagnostics
        if "synthetic_anomaly_mask" not in batch:
            return None, diagnostics
        if self.score_loss_granularity != "point":
            return None, diagnostics
        if self.score_loss_target != "synthetic_anomaly_mask":
            return None, diagnostics
        if self.score_loss_type not in {
            "pointwise_balanced_bce_logits",
            "pointwise_balanced_reconstruction_score",
        }:
            return None, diagnostics

        # step 2: Compute one reconstruction error for every point.
        pointwise_reconstruction_error = self.reconstruction_squared_error(
            outputs, batch
        ).mean(dim=-1)

        # step 3: Build normal and anomalous point masks.
        anomaly_mask = self._point_mask_from_synthetic_mask(
            batch["synthetic_anomaly_mask"]
        )
        normal_mask = ~anomaly_mask

        # step 4: Require both classes so both balanced loss terms are defined.
        normal_count = normal_mask.sum()
        anomaly_count = anomaly_mask.sum()
        diagnostics["point_score_normal_count"] = normal_count.detach()
        diagnostics["point_score_anomaly_count"] = anomaly_count.detach()
        if int(normal_count.item()) == 0 or int(anomaly_count.item()) == 0:
            return None, diagnostics

        # step 5: Select reconstruction errors from normal points.
        normal_scores = pointwise_reconstruction_error[normal_mask]

        # step 6: Estimate normal-score mean and standard deviation.
        score_mean = normal_scores.mean().detach()
        score_std = normal_scores.std(unbiased=False).detach().clamp_min(self.epsilon)

        # step 7: Normalize every point score using normal-point statistics.
        normalized_scores = (pointwise_reconstruction_error - score_mean) / score_std

        # step 8: Use 0 for normal points and 1 for anomalous points.
        point_targets = anomaly_mask.float()

        # step 9: Treat normalized reconstruction scores as BCE logits.
        loss_per_token = F.binary_cross_entropy_with_logits(
            normalized_scores,
            point_targets,
            reduction="none",
        )

        # step 10: Average the loss separately for normal and anomalous points.
        loss_normal = loss_per_token[normal_mask].mean()
        loss_anomaly = loss_per_token[anomaly_mask].mean()

        # step 11: Give the normal and anomalous groups equal weight.
        score_loss = 0.5 * loss_normal + 0.5 * loss_anomaly

        with torch.no_grad():
            anomaly_scores = pointwise_reconstruction_error[anomaly_mask]
            diagnostics.update(
                {
                    "point_score_normal_mean": normal_scores.mean(),
                    "point_score_normal_std": normal_scores.std(unbiased=False),
                    "point_score_anomaly_mean": anomaly_scores.mean(),
                    "point_score_anomaly_std": anomaly_scores.std(unbiased=False),
                    "point_score_gap_mean": anomaly_scores.mean()
                    - normal_scores.mean(),
                    "point_score_gap_extreme": anomaly_scores.min()
                    - normal_scores.max(),
                }
            )

        return score_loss, diagnostics

    def _compute_optional_loss_terms(
        self, outputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        optional_loss_values: dict[str, torch.Tensor] = {}
        if not self._phase_uses_prototype_path():
            for loss_name in self.optional_loss_configs:
                optional_loss_values[loss_name] = self._zero_loss(outputs["hidden"])
            return optional_loss_values
        for loss_name, loss_config in self.optional_loss_configs.items():
            compute_fn: Callable[[dict[str, Any]], torch.Tensor] = loss_config[
                "compute_fn"
            ]
            if loss_config["enabled"]:
                optional_loss_values[loss_name] = compute_fn(outputs)
            else:
                optional_loss_values[loss_name] = self._zero_loss(outputs["hidden"])
        return optional_loss_values
