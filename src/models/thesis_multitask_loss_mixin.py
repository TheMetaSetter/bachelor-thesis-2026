from __future__ import annotations

"""Mixin extracted from the thesis multitask model.

This file keeps constructor and configuration plumbing together so the main
model file can stay below the code-size limit without changing runtime
behavior.
"""

import math
import time
from collections import OrderedDict, deque
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.console import (
    console_print,
    print_parameter_summary,
    summarize_batch,
    summarize_label_distribution,
    summarize_tensor,
)
from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import SyntheticAnomalyInjector
from src.models.thesis_multitask_components import (
    STAGE3_PHASE_CANONICAL_NAME,
    STAGE3_PHASE_LEGACY_NAME,
    TWO_STAGE_A_PHASE_NAME,
    TWO_STAGE_B_PHASE_NAME,
    TWO_STAGE_PHASE_NAMES,
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    MultitaskArchitectureConfig,
    MultitaskWindowEncoder,
    ObjectiveConfig,
    MemoryInitializationConfig,
    PrototypeBranchConfig,
    ScheduleAndWarmupConfig,
    SyntheticAnomalyConfig,
    ThesisMultitaskModelConfig,
    build_multilayer_perceptron,
)


class ThesisMultitaskLossMixin:
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
        squared_reconstruction_error = (outputs["recon"] - batch["x"]) ** 2
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
        squared_reconstruction_error = (outputs["recon"] - batch["x"]) ** 2
        per_window_error = squared_reconstruction_error.mean(dim=(1, 2))
        diagnostics = {
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
        # This discourages the two branches from collapsing onto the same signal.
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

    # Version 1 with pure simple loss (only classification and reconstruction) was trained.
    # Based on real diagnostic, this usage loss will be added to prevent over-centralizing in few discrete prototypes.
    # So in version 2, the final loss function has 3 terms: one for classification, one for reconstruction
    # and one for usage of discrete prototypes.
    def _compute_prototype_usage_loss(self, outputs: dict[str, Any]) -> torch.Tensor:
        # Usage balancing is the main protection against dead or ignored codes.
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
        # Gate entropy regularization keeps the fusion scalars from collapsing
        # too confidently unless the data actually supports that decision.
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
        diagnostics: dict[str, torch.Tensor] = {
            "point_score_normal_count": torch.tensor(0, device=outputs["recon"].device),
            "point_score_anomaly_count": torch.tensor(
                0, device=outputs["recon"].device
            ),
        }
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

        pointwise_reconstruction_error = ((outputs["recon"] - batch["x"]) ** 2).mean(
            dim=-1
        )
        anomaly_mask = self._point_mask_from_synthetic_mask(
            batch["synthetic_anomaly_mask"]
        )
        normal_mask = ~anomaly_mask

        normal_count = normal_mask.sum()
        anomaly_count = anomaly_mask.sum()
        diagnostics["point_score_normal_count"] = normal_count.detach()
        diagnostics["point_score_anomaly_count"] = anomaly_count.detach()
        if int(normal_count.item()) == 0 or int(anomaly_count.item()) == 0:
            return None, diagnostics

        normal_scores = pointwise_reconstruction_error[normal_mask]
        score_mean = normal_scores.mean().detach()
        score_std = normal_scores.std(unbiased=False).detach().clamp_min(self.epsilon)
        normalized_scores = (pointwise_reconstruction_error - score_mean) / score_std
        point_targets = anomaly_mask.float()
        loss_per_token = F.binary_cross_entropy_with_logits(
            normalized_scores,
            point_targets,
            reduction="none",
        )
        loss_normal = loss_per_token[normal_mask].mean()
        loss_anomaly = loss_per_token[anomaly_mask].mean()
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

    def _compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
        optional_loss_values: dict[str, torch.Tensor],
        reconstruction_weight: float,
        classification_weight: float,
    ) -> torch.Tensor:
        # The weighted sum is intentionally explicit so readers can map each
        # `lambda_*` config field directly to one line of the objective. The
        # default beginning of training is still the small objective
        # `lambda_recon * L_recon + lambda_cls * L_cls`.
        total_loss = (
            reconstruction_weight * reconstruction_loss
            + classification_weight * classification_loss
        )
        for loss_name, loss_value in optional_loss_values.items():
            loss_weight = self.optional_loss_configs[loss_name]["weight"]
            if loss_name == "usage_loss":
                loss_weight = self.current_usage_lambda
            total_loss = total_loss + loss_weight * loss_value
        return total_loss

    def _get_encoder_profiled_parameters(self) -> OrderedDict[str, nn.Parameter]:
        profiled_parameters: OrderedDict[str, nn.Parameter] = OrderedDict()
        encoder_layers: Any = (
            self.encoder.network if hasattr(self.encoder, "network") else self.encoder
        )
        if hasattr(encoder_layers, "network"):
            encoder_layers = encoder_layers.network
        affine_layer_indices = [
            layer_index
            for layer_index, layer_module in enumerate(encoder_layers)
            if isinstance(layer_module, (nn.Linear, nn.Conv1d))
        ]
        for layer_index in affine_layer_indices:
            layer_module = encoder_layers[layer_index]
            layer_type = "linear" if isinstance(layer_module, nn.Linear) else "conv"
            weight_key = f"encoder.{layer_type}{layer_index}.weight"
            profiled_parameters[weight_key] = layer_module.weight
            if self.gradient_profile_include_bias and layer_module.bias is not None:
                bias_key = f"encoder.{layer_type}{layer_index}.bias"
                profiled_parameters[bias_key] = layer_module.bias
        if len(profiled_parameters) == 0:
            raise ValueError("No encoder parameters available for gradient profiling")
        return profiled_parameters

    def _flatten_tensor_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1)

    def _compute_cosine_similarity(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
    ) -> float:
        gradient_ce_flattened = self._flatten_tensor_for_metrics(gradient_ce)
        gradient_mse_flattened = self._flatten_tensor_for_metrics(gradient_mse)
        dot_product = torch.dot(gradient_ce_flattened, gradient_mse_flattened)
        norm_ce = torch.linalg.vector_norm(gradient_ce_flattened)
        norm_mse = torch.linalg.vector_norm(gradient_mse_flattened)
        cosine_similarity = dot_product / ((norm_ce * norm_mse).clamp_min(1.0e-12))
        return float(cosine_similarity.detach().cpu())

    def _compute_preservation_ratio(
        self,
        gradient_ce: torch.Tensor,
        gradient_mse: torch.Tensor,
        gradient_total: torch.Tensor,
    ) -> float:
        norm_ce = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_ce)
        )
        norm_mse = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_mse)
        )
        norm_total = torch.linalg.vector_norm(
            self._flatten_tensor_for_metrics(gradient_total)
        )
        preservation_ratio = norm_total / (norm_ce + norm_mse).clamp_min(1.0e-12)
        return float(preservation_ratio.detach().cpu())

    def _extract_layerwise_gradients(
        self,
        loss: torch.Tensor,
        encoder_parameters: list[nn.Parameter],
    ) -> list[torch.Tensor]:
        gradients = torch.autograd.grad(
            loss,
            encoder_parameters,
            retain_graph=True,
            allow_unused=False,
        )
        return [gradient.detach().clone() for gradient in gradients]

    def _update_ema(self, metric_key: str, metric_value: float) -> float:
        previous_ema = self._gradient_profile_ema_state.get(metric_key)
        if previous_ema is None:
            updated_ema = metric_value
        else:
            updated_ema = (
                self.gradient_ema_alpha * metric_value
                + (1.0 - self.gradient_ema_alpha) * previous_ema
            )
        self._gradient_profile_ema_state[metric_key] = updated_ema
        return updated_ema

    def _update_sma(self, metric_key: str, metric_value: float) -> float:
        if metric_key not in self._gradient_profile_sma_buffers:
            self._gradient_profile_sma_buffers[metric_key] = deque(
                maxlen=self.gradient_sma_window
            )
        self._gradient_profile_sma_buffers[metric_key].append(metric_value)
        sma_value = sum(self._gradient_profile_sma_buffers[metric_key]) / len(
            self._gradient_profile_sma_buffers[metric_key]
        )
        return float(sma_value)

    def _resolve_focus_layer_parameter_name(self) -> str:
        if self.gradient_focus_layer_name in {
            "encoder_last_linear",
            "encoder_last_affine",
        }:
            parameter_names = list(self._encoder_profiled_parameters.keys())
            return parameter_names[-1]
        raise ValueError(
            f"Unsupported gradient_focus_layer_name: {self.gradient_focus_layer_name}"
        )

    def _build_gradient_conflict_log_dict(
        self,
        raw_metrics: dict[str, float],
    ) -> dict[str, float]:
        gradient_conflict_logs: dict[str, float] = {}
        for raw_key, raw_value in raw_metrics.items():
            gradient_conflict_logs[raw_key] = raw_value
            metric_suffix = raw_key.split("train_gradconf_raw/", 1)[1]
            ema_key = f"train_gradconf_ema/{metric_suffix}"
            sma_key = f"train_gradconf_sma/{metric_suffix}"
            gradient_conflict_logs[ema_key] = self._update_ema(raw_key, raw_value)
            gradient_conflict_logs[sma_key] = self._update_sma(raw_key, raw_value)
        return gradient_conflict_logs

    def _profile_encoder_gradient_conflict(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
    ) -> dict[str, float]:
        encoder_parameter_items = list(self._encoder_profiled_parameters.items())
        encoder_parameter_names = [
            parameter_name for parameter_name, _ in encoder_parameter_items
        ]
        encoder_parameters = [
            parameter_tensor for _, parameter_tensor in encoder_parameter_items
        ]
        weighted_classification_loss = self.lambda_cls * classification_loss
        weighted_reconstruction_loss = self.lambda_recon * reconstruction_loss
        gradients_ce = self._extract_layerwise_gradients(
            weighted_classification_loss,
            encoder_parameters,
        )
        gradients_mse = self._extract_layerwise_gradients(
            weighted_reconstruction_loss,
            encoder_parameters,
        )

        raw_metrics: dict[str, float] = {}
        for layer_name, gradient_ce, gradient_mse in zip(
            encoder_parameter_names, gradients_ce, gradients_mse
        ):
            gradient_total = gradient_ce + gradient_mse
            norm_ce = float(
                torch.linalg.vector_norm(self._flatten_tensor_for_metrics(gradient_ce))
                .detach()
                .cpu()
            )
            norm_mse = float(
                torch.linalg.vector_norm(self._flatten_tensor_for_metrics(gradient_mse))
                .detach()
                .cpu()
            )
            norm_total = float(
                torch.linalg.vector_norm(
                    self._flatten_tensor_for_metrics(gradient_total)
                )
                .detach()
                .cpu()
            )
            cosine_similarity = self._compute_cosine_similarity(
                gradient_ce, gradient_mse
            )
            preservation_ratio = self._compute_preservation_ratio(
                gradient_ce=gradient_ce,
                gradient_mse=gradient_mse,
                gradient_total=gradient_total,
            )
            metric_prefix = f"train_gradconf_raw/{layer_name}"
            raw_metrics[f"{metric_prefix}/cosine_sim"] = cosine_similarity
            raw_metrics[f"{metric_prefix}/r_ratio"] = preservation_ratio
            raw_metrics[f"{metric_prefix}/norm_ce"] = norm_ce
            raw_metrics[f"{metric_prefix}/norm_mse"] = norm_mse
            raw_metrics[f"{metric_prefix}/norm_total"] = norm_total
            if layer_name == self._resolve_focus_layer_parameter_name():
                focus_prefix = "train_gradconf_raw/focus"
                raw_metrics[f"{focus_prefix}/cosine_sim"] = cosine_similarity
                raw_metrics[f"{focus_prefix}/r_ratio"] = preservation_ratio
                raw_metrics[f"{focus_prefix}/norm_ce"] = norm_ce
                raw_metrics[f"{focus_prefix}/norm_mse"] = norm_mse
                raw_metrics[f"{focus_prefix}/norm_total"] = norm_total
        return self._build_gradient_conflict_log_dict(raw_metrics)

    def _build_stage_log(
        self,
        stage_name: str,
        outputs: dict[str, Any],
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        include_classification_metrics: bool,
    ) -> dict[str, float]:
        # These logs are part of the branch-collapse observability surface, not
        # just convenience metrics. They are meant to support ablation reading.
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
        # This is the one place where the actual multitask training objective is assembled.

        # Chuẩn bị batch dữ liệu nghĩa là tải các mẫu dữ liệu lên từ
        # dataset và tiêm bất thường nhân tạo vào nếu cần.
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

        # Đưa các mẫu dữ liệu qua mạng để tính toán ra kết quả
        outputs = self.forward(prepared_batch, stage_name=stage_name)

        # Tính toán các hàm loss thành phần
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

        # Tính toán hàm loss tổng
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
