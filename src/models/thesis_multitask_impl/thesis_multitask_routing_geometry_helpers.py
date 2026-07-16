from __future__ import annotations

"""Geometry and Monte Carlo helpers for the thesis multitask routing path."""

from typing import Any

import torch
import torch.nn.functional as F

from src.models.thesis_multitask_impl.thesis_multitask_components import QueryBundle


def _build_sampled_fusion_hidden(
    self: Any,
    continuous_samples: torch.Tensor,
    discrete_samples: torch.Tensor,
    *,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if continuous_samples.shape != discrete_samples.shape:
        raise ValueError("continuous_samples and discrete_samples must match shape")
    if continuous_samples.ndim != 4:
        raise ValueError("sample tensors must have rank 4")
    if alpha.ndim != 1 or beta.ndim != 1:
        raise ValueError("alpha and beta must have rank 1")
    if (
        alpha.shape[0] != continuous_samples.shape[0]
        or beta.shape[0] != continuous_samples.shape[0]
    ):
        raise ValueError("alpha and beta must match batch size")

    if self.fusion_mode == "task_specific_concat_projection":
        concatenated_hidden = torch.cat([continuous_samples, discrete_samples], dim=-1)
        flattened_hidden = concatenated_hidden.reshape(
            concatenated_hidden.shape[0] * concatenated_hidden.shape[1],
            concatenated_hidden.shape[2],
            concatenated_hidden.shape[3],
        )
        hidden_reconstruction = self.reconstruction_concat_projection(
            flattened_hidden
        ).reshape(
            concatenated_hidden.shape[0],
            concatenated_hidden.shape[1],
            concatenated_hidden.shape[2],
            self.hidden_dim,
        )
        hidden_classification = self.classification_concat_projection(
            flattened_hidden
        ).reshape(
            concatenated_hidden.shape[0],
            concatenated_hidden.shape[1],
            concatenated_hidden.shape[2],
            self.hidden_dim,
        )
        return hidden_reconstruction, hidden_classification

    alpha_expanded = alpha.view(-1, 1, 1, 1)
    beta_expanded = beta.view(-1, 1, 1, 1)
    hidden_reconstruction = (
        beta_expanded * discrete_samples + (1.0 - beta_expanded) * continuous_samples
    )
    hidden_classification = (
        alpha_expanded * discrete_samples + (1.0 - alpha_expanded) * continuous_samples
    )
    return hidden_reconstruction, hidden_classification


def _variance_from_samples(
    self: Any,
    sample_tensor: torch.Tensor,
    *,
    correction: int | None = None,
) -> torch.Tensor:
    if sample_tensor.ndim < 2:
        raise ValueError("sample_tensor must have at least rank 2")
    if sample_tensor.shape[1] < 2:
        variance = torch.zeros_like(sample_tensor[:, 0])
    else:
        variance = sample_tensor.var(
            dim=1,
            correction=self.variance_correction if correction is None else correction,
        )
    return variance


def _build_monte_carlo_uncertainty(
    self: Any,
    *,
    reconstruction_samples: torch.Tensor,
    continuous_samples: torch.Tensor,
    discrete_samples: torch.Tensor,
    point_score_samples: torch.Tensor,
    window_score_samples: torch.Tensor,
    classification_probability_samples: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    reconstruction_variance_full = self._variance_from_samples(reconstruction_samples)
    reconstruction_variance_point = reconstruction_variance_full.mean(dim=-1)
    reconstruction_variance_window = reconstruction_variance_point.mean(dim=-1)

    continuous_variance_full = self._variance_from_samples(continuous_samples)
    continuous_retrieval_variance_point = continuous_variance_full.mean(dim=-1)
    continuous_retrieval_variance_window = continuous_retrieval_variance_point.mean(
        dim=-1
    )

    discrete_variance_full = self._variance_from_samples(discrete_samples)
    discrete_retrieval_variance_point = discrete_variance_full.mean(dim=-1)
    discrete_retrieval_variance_window = discrete_retrieval_variance_point.mean(dim=-1)

    point_anomaly_score_variance = self._variance_from_samples(point_score_samples)
    window_anomaly_score_variance = self._variance_from_samples(window_score_samples)

    classification_probability_variance = None
    classification_variance_mean = None
    if classification_probability_samples is not None:
        classification_probability_variance = self._variance_from_samples(
            classification_probability_samples
        )
        classification_variance_mean = classification_probability_variance.mean(dim=-1)

    return {
        "point_anomaly_score_variance": point_anomaly_score_variance,
        "window_anomaly_score_variance": window_anomaly_score_variance,
        "continuous_retrieval_variance_point": continuous_retrieval_variance_point,
        "continuous_retrieval_variance_window": continuous_retrieval_variance_window,
        "discrete_retrieval_variance_point": discrete_retrieval_variance_point,
        "discrete_retrieval_variance_window": discrete_retrieval_variance_window,
        "reconstruction_variance_point": reconstruction_variance_point,
        "reconstruction_variance_window": reconstruction_variance_window,
        "reconstruction_variance_full": reconstruction_variance_full,
        "classification_probability_variance": classification_probability_variance,
        "classification_variance_mean": classification_variance_mean,
    }


def _build_monte_carlo_forward_outputs(
    self: Any,
    batch: dict[str, Any],
    *,
    fusion_outputs: dict[str, Any],
    query_bundle: QueryBundle,
    continuous_samples: torch.Tensor,
    discrete_samples: torch.Tensor,
    discrete_topk_ids: torch.Tensor,
) -> dict[str, Any]:
    batch_size, num_samples, window_size, hidden_dim = continuous_samples.shape
    if window_size != self.window_size or hidden_dim != self.hidden_dim:
        raise ValueError("sample tensors must match model window and hidden sizes")
    sampled_hidden_reconstruction, sampled_hidden_classification = (
        self._build_sampled_fusion_hidden(
            continuous_samples,
            discrete_samples,
            alpha=fusion_outputs["alpha"],
            beta=fusion_outputs["beta"],
        )
    )
    flattened_reconstruction_hidden = sampled_hidden_reconstruction.reshape(
        batch_size * num_samples,
        window_size,
        hidden_dim,
    )
    flattened_classification_hidden = sampled_hidden_classification.reshape(
        batch_size * num_samples,
        window_size * hidden_dim,
    )
    input_dim = batch["x"].shape[-1]
    reconstruction_samples = self.reconstruction_head(
        flattened_reconstruction_hidden
    ).reshape(batch_size, num_samples, window_size, input_dim)
    logits_samples = None
    classification_probability_samples = None
    if self.enable_classification_path:
        logits_samples = self.classification_head(
            flattened_classification_hidden
        ).reshape(batch_size, num_samples, self.num_classes)
        classification_probability_samples = torch.softmax(
            logits_samples,
            dim=-1,
        )
    point_score_samples = torch.mean(
        (reconstruction_samples - batch["x"].unsqueeze(1)) ** 2,
        dim=-1,
    )
    window_score_samples = point_score_samples.mean(dim=-1)
    reconstruction_mean = reconstruction_samples.mean(dim=1)
    point_score_mean = point_score_samples.mean(dim=1)
    window_score_mean = window_score_samples.mean(dim=1)
    if classification_probability_samples is not None:
        class_probabilities = classification_probability_samples.mean(dim=1)
        logits = torch.log(
            class_probabilities.clamp_min(torch.finfo(class_probabilities.dtype).eps)
        )
    else:
        class_probabilities = None
        logits = None
    uncertainty = self._build_monte_carlo_uncertainty(
        reconstruction_samples=reconstruction_samples,
        continuous_samples=continuous_samples,
        discrete_samples=discrete_samples,
        point_score_samples=point_score_samples,
        window_score_samples=window_score_samples,
        classification_probability_samples=classification_probability_samples,
    )
    stochastic_query = {
        "schema_version": 3,
        "enabled": True,
        "num_samples": num_samples,
        "continuous_temperature": self.continuous_temperature,
        "discrete_temperature": self.discrete_temperature,
        "continuous_retrieved_samples": continuous_samples,
        "discrete_retrieved_samples": discrete_samples,
        "discrete_topk_ids": discrete_topk_ids,
        "reconstruction_samples": reconstruction_samples,
        "classification_probability_samples": classification_probability_samples,
        "point_score_samples": point_score_samples,
        "window_score_samples": window_score_samples,
        "return_mc_samples": self.return_mc_samples,
        "sample_retention_policy": self.sample_retention_policy,
    }
    if logits_samples is not None:
        stochastic_query["logits_samples"] = logits_samples
    outputs = {
        "recon": reconstruction_mean,
        "logits": logits,
        "point_scores": point_score_mean,
        "window_scores": window_score_mean,
    }
    aux = {
        "class_probabilities": class_probabilities,
        "uncertainty": uncertainty,
        "deterministic_geometry": {
            "hidden_reconstruction": fusion_outputs["hidden_reconstruction"],
            "hidden_classification": fusion_outputs["hidden_classification"],
            "alpha": fusion_outputs["alpha"],
            "beta": fusion_outputs["beta"],
            "fusion": fusion_outputs["aux"],
        },
        "stochastic_query": stochastic_query,
    }
    return {
        "outputs": outputs,
        "aux": aux,
        "sample_outputs": {
            "reconstruction_samples": reconstruction_samples,
            "logits_samples": logits_samples,
            "classification_probability_samples": classification_probability_samples,
        },
    }


def _discrete_prototype_lookup(
    self: Any,
    hidden: torch.Tensor,
    *,
    stage_name: str,
    active_codebook: torch.Tensor | None = None,
    precomputed_assignment_logits: torch.Tensor | None = None,
    precomputed_assignment_probabilities: torch.Tensor | None = None,
) -> dict[str, Any]:
    normalized_hidden = self._normalize_hidden_for_memory(hidden)
    discrete_hidden = normalized_hidden
    assignment_logits = precomputed_assignment_logits
    assignment_probabilities = precomputed_assignment_probabilities
    code_indices = None
    memory_bypass_active = self._should_bypass_memory_for_stage(stage_name)
    normalized_codebook = active_codebook
    if normalized_codebook is None and self.discrete_codebook is not None:
        normalized_codebook = self._normalize_memory_vectors(self.discrete_codebook)

    if normalized_codebook is not None and not memory_bypass_active:
        if assignment_logits is None or assignment_probabilities is None:
            if self.discrete_query_mode == "cosine_topk":
                assignment_logits = torch.einsum(
                    "blh,kh->blk",
                    normalized_hidden,
                    normalized_codebook,
                )
                topk_value_count = min(
                    self.discrete_topk,
                    int(normalized_codebook.shape[0]),
                )
                topk_logits, topk_indices = torch.topk(
                    assignment_logits,
                    k=topk_value_count,
                    dim=-1,
                )
                topk_weights = torch.softmax(
                    topk_logits / self.discrete_query_temperature,
                    dim=-1,
                )
                assignment_probabilities = torch.zeros_like(assignment_logits)
                assignment_probabilities.scatter_(
                    dim=-1,
                    index=topk_indices,
                    src=topk_weights,
                )
            else:
                if self.discrete_assignment is None:
                    raise ValueError("discrete_assignment is not available")
                assignment_logits = self.discrete_assignment(normalized_hidden)
                assignment_probabilities = F.gumbel_softmax(
                    assignment_logits,
                    tau=self.gumbel_temperature,
                    hard=False,
                    dim=-1,
                )
        discrete_hidden = torch.einsum(
            "blk,kh->blh",
            assignment_probabilities,
            normalized_codebook,
        )
        discrete_hidden = self._normalize_hidden_for_memory(discrete_hidden)
        code_indices = torch.argmax(assignment_probabilities, dim=-1)

    return {
        "hidden": hidden,
        "quantized_hidden": discrete_hidden,
        "assignment_logits": assignment_logits,
        "assignment_probabilities": assignment_probabilities,
        "code_indices": code_indices,
        "aux": {
            "branch_name": "discrete",
            "enabled": self.discrete_memory_enabled,
            "codebook_size": self.discrete_codebook_size,
            "temperature": self.gumbel_temperature,
            "query_mode": self.discrete_query_mode,
            "topk": self.discrete_topk,
            "query_temperature": self.discrete_query_temperature,
            "memory_bypass_active": memory_bypass_active,
            "memory_initialized": self.memory_initialized,
        },
    }


def _compute_fusion_outputs(
    self: Any,
    continuous_hidden: torch.Tensor,
    discrete_hidden: torch.Tensor,
    *,
    base_hidden: torch.Tensor | None = None,
    paired_hidden: torch.Tensor | None = None,
) -> dict[str, Any]:
    if self.fusion_mode == "task_specific_concat_projection":
        concatenated_hidden = torch.cat([continuous_hidden, discrete_hidden], dim=-1)
        hidden_reconstruction = self.reconstruction_concat_projection(
            concatenated_hidden
        )
        hidden_classification = self.classification_concat_projection(
            concatenated_hidden
        )
        alpha = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        beta = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        cka_reconstruction = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        cka_classification = continuous_hidden.new_zeros(continuous_hidden.shape[0])
        return {
            "hidden_reconstruction": hidden_reconstruction,
            "hidden_classification": hidden_classification,
            "alpha": alpha,
            "beta": beta,
            "aux": {
                "fusion_mode": "task_specific_concat_projection",
                "alpha": 0.0,
                "beta": 0.0,
                "alpha_std": 0.0,
                "beta_std": 0.0,
                "alpha_logit": float(self.alpha_logit.detach().cpu()),
                "beta_logit": float(self.beta_logit.detach().cpu()),
                "cka_reconstruction_mean": 0.0,
                "cka_reconstruction_std": 0.0,
                "cka_classification_mean": 0.0,
                "cka_classification_std": 0.0,
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }
    if self.active_alpha_override is None:
        alpha_scalar = torch.sigmoid(self.alpha_logit)
    else:
        alpha_scalar = continuous_hidden.new_tensor(float(self.active_alpha_override))
    if self.active_beta_override is None:
        beta_scalar = torch.sigmoid(self.beta_logit)
    else:
        beta_scalar = continuous_hidden.new_tensor(float(self.active_beta_override))

    alpha = alpha_scalar.expand(continuous_hidden.shape[0])
    beta = beta_scalar.expand(continuous_hidden.shape[0])
    cka_reconstruction = continuous_hidden.new_zeros(continuous_hidden.shape[0])
    cka_classification = continuous_hidden.new_zeros(continuous_hidden.shape[0])
    if (
        self.enable_cka_gated_fusion
        and base_hidden is not None
        and paired_hidden is not None
    ):
        cka_reconstruction = self._compute_batch_linear_cka_scores(
            base_hidden,
            continuous_hidden,
        )
        cka_classification = self._compute_batch_linear_cka_scores(
            paired_hidden,
            discrete_hidden,
        )
        cka_features = torch.stack([cka_reconstruction, cka_classification], dim=-1)
        alpha = torch.sigmoid(self.classification_fusion_gate(cka_features)).squeeze(-1)
        beta = torch.sigmoid(self.reconstruction_fusion_gate(cka_features)).squeeze(-1)
    alpha_expanded = alpha.view(-1, 1, 1)
    beta_expanded = beta.view(-1, 1, 1)

    hidden_reconstruction = (
        beta_expanded * discrete_hidden + (1.0 - beta_expanded) * continuous_hidden
    )
    hidden_classification = (
        alpha_expanded * discrete_hidden + (1.0 - alpha_expanded) * continuous_hidden
    )

    return {
        "hidden_reconstruction": hidden_reconstruction,
        "hidden_classification": hidden_classification,
        "alpha": alpha,
        "beta": beta,
        "aux": {
            "fusion_mode": "learnable_sigmoid_scalars",
            "alpha": float(alpha.mean().detach().cpu()),
            "beta": float(beta.mean().detach().cpu()),
            "alpha_std": float(alpha.std(unbiased=False).detach().cpu()),
            "beta_std": float(beta.std(unbiased=False).detach().cpu()),
            "alpha_logit": float(alpha_scalar.detach().cpu()),
            "beta_logit": float(beta_scalar.detach().cpu()),
            "cka_reconstruction_mean": float(cka_reconstruction.mean().detach().cpu()),
            "cka_reconstruction_std": float(
                cka_reconstruction.std(unbiased=False).detach().cpu()
            ),
            "cka_classification_mean": float(cka_classification.mean().detach().cpu()),
            "cka_classification_std": float(
                cka_classification.std(unbiased=False).detach().cpu()
            ),
            "warmup_active": self.schedule_state["warmup_active"],
            "temperature": self.gumbel_temperature,
        },
    }
