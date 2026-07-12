from __future__ import annotations

"""Gradient profiling helpers for the thesis multitask model."""

from collections import OrderedDict, deque
from typing import Any

import torch
import torch.nn as nn


class ThesisMultitaskLossGradientMixin:
    def _compute_total_loss(
        self,
        reconstruction_loss: torch.Tensor,
        classification_loss: torch.Tensor,
        optional_loss_values: dict[str, torch.Tensor],
        reconstruction_weight: float,
        classification_weight: float,
    ) -> torch.Tensor:
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
