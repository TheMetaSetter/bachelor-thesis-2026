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
    QueryBundle,
    PrototypeBranchConfig,
    ScheduleAndWarmupConfig,
    SyntheticAnomalyConfig,
    ThesisMultitaskModelConfig,
    build_multilayer_perceptron,
)


class ThesisMultitaskRoutingMixin:
    def _gumbel_noise_from_uniform(self, uniform: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(uniform.dtype).eps
        clamped_uniform = uniform.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(clamped_uniform))

    def _sample_gumbel_noise(self, reference: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand(
            reference.shape,
            device=reference.device,
            dtype=reference.dtype,
        )
        return self._gumbel_noise_from_uniform(uniform)

    def build_stochastic_queries(
        self,
        hidden: torch.Tensor,
        *,
        stage_name: str,
        active_memory_bank: torch.Tensor | None = None,
        active_codebook: torch.Tensor | None = None,
    ) -> QueryBundle:
        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        memory_bypass_active = self._should_bypass_memory_for_stage(stage_name)

        continuous_memory_bank = None
        continuous_logits = None
        if active_memory_bank is not None and not memory_bypass_active:
            continuous_memory_bank = self._normalize_memory_vectors(active_memory_bank)
            continuous_logits = torch.einsum(
                "blh,kh->blk",
                normalized_hidden,
                continuous_memory_bank,
            ) / math.sqrt(self.hidden_dim)

        discrete_codebook = None
        discrete_logits = None
        if active_codebook is not None and not memory_bypass_active:
            discrete_codebook = self._normalize_memory_vectors(active_codebook)
        elif self.discrete_codebook is not None and not memory_bypass_active:
            discrete_codebook = self._normalize_memory_vectors(self.discrete_codebook)

        if not memory_bypass_active:
            if self.discrete_query_mode == "cosine_topk":
                if discrete_codebook is not None:
                    discrete_logits = torch.einsum(
                        "blh,kh->blk",
                        normalized_hidden,
                        discrete_codebook,
                    )
            else:
                if self.discrete_assignment is None:
                    raise ValueError("discrete_assignment is not available")
                discrete_logits = self.discrete_assignment(normalized_hidden)

        return QueryBundle(
            hidden=hidden,
            normalized_hidden=normalized_hidden,
            continuous_memory_bank=continuous_memory_bank,
            discrete_codebook=discrete_codebook,
            continuous_logits=continuous_logits,
            discrete_logits=discrete_logits,
            memory_bypass_active=memory_bypass_active,
            discrete_query_mode=self.discrete_query_mode,
            continuous_temperature=self.continuous_temperature,
            discrete_temperature=self.discrete_temperature,
            discrete_topk=self.discrete_topk,
        )

    def sample_continuous_retrieval(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if (
            query_bundle.continuous_memory_bank is None
            or query_bundle.continuous_logits is None
            or query_bundle.memory_bypass_active
        ):
            return query_bundle.hidden.unsqueeze(1).expand(
                -1, num_samples, -1, -1
            ).contiguous()

        sampled_logits = query_bundle.continuous_logits.unsqueeze(1).expand(
            -1, num_samples, -1, -1
        )
        if self.stochastic_inference:
            sampled_logits = sampled_logits + self._sample_gumbel_noise(sampled_logits)
        sampled_logits = sampled_logits / query_bundle.continuous_temperature
        sample_weights = torch.softmax(sampled_logits, dim=-1)
        sample_hidden = torch.einsum(
            "bmlk,kh->bmlh",
            sample_weights,
            query_bundle.continuous_memory_bank,
        )
        return self._normalize_hidden_for_memory(sample_hidden)

    def sample_discrete_retrieval(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if (
            query_bundle.discrete_codebook is None
            or query_bundle.discrete_logits is None
            or query_bundle.memory_bypass_active
        ):
            return query_bundle.hidden.unsqueeze(1).expand(
                -1, num_samples, -1, -1
            ).contiguous()

        if query_bundle.discrete_query_mode == "cosine_topk":
            topk_value_count = min(
                query_bundle.discrete_topk,
                int(query_bundle.discrete_logits.shape[-1]),
            )
            topk_logits, topk_indices = torch.topk(
                query_bundle.discrete_logits,
                k=topk_value_count,
                dim=-1,
            )
            topk_logits = topk_logits.unsqueeze(1).expand(-1, num_samples, -1, -1)
            topk_indices = topk_indices.unsqueeze(1).expand(-1, num_samples, -1, -1)
            assignment_probabilities = torch.zeros_like(
                query_bundle.discrete_logits.unsqueeze(1).expand(
                    -1, num_samples, -1, -1
                )
            )
            assignment_probabilities.scatter_(
                dim=-1,
                index=topk_indices,
                src=torch.softmax(
                    topk_logits / query_bundle.discrete_temperature,
                    dim=-1,
                ),
            )
        else:
            sampled_logits = query_bundle.discrete_logits.unsqueeze(1).expand(
                -1, num_samples, -1, -1
            )
            if self.stochastic_inference:
                sampled_logits = sampled_logits + self._sample_gumbel_noise(
                    sampled_logits
                )
            sampled_logits = sampled_logits / query_bundle.discrete_temperature
            assignment_probabilities = torch.softmax(sampled_logits, dim=-1)

        sample_hidden = torch.einsum(
            "bmlk,kh->bmlh",
            assignment_probabilities,
            query_bundle.discrete_codebook,
        )
        return self._normalize_hidden_for_memory(sample_hidden)

    def sample_discrete_topk_ids(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if query_bundle.discrete_logits is None:
            return query_bundle.hidden.new_zeros(
                query_bundle.hidden.shape[0],
                num_samples,
                query_bundle.hidden.shape[1],
                3,
                dtype=torch.long,
            )
        topk_count = min(3, int(query_bundle.discrete_logits.shape[-1]))
        topk_indices = torch.topk(
            query_bundle.discrete_logits,
            k=topk_count,
            dim=-1,
        ).indices
        if topk_count < 3:
            repeat_count = 3 - topk_count
            topk_indices = torch.cat(
                [topk_indices, topk_indices[..., -1:].expand(-1, -1, repeat_count)],
                dim=-1,
            )
        return topk_indices.unsqueeze(1).expand(-1, num_samples, -1, -1).contiguous()

    def _build_sampled_fusion_hidden(
        self,
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
        if alpha.shape[0] != continuous_samples.shape[0] or beta.shape[0] != continuous_samples.shape[0]:
            raise ValueError("alpha and beta must match batch size")

        if self.fusion_mode == "task_specific_concat_projection":
            concatenated_hidden = torch.cat(
                [continuous_samples, discrete_samples], dim=-1
            )
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
            beta_expanded * discrete_samples
            + (1.0 - beta_expanded) * continuous_samples
        )
        hidden_classification = (
            alpha_expanded * discrete_samples
            + (1.0 - alpha_expanded) * continuous_samples
        )
        return hidden_reconstruction, hidden_classification

    def _variance_from_samples(
        self,
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
        self,
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
        continuous_retrieval_variance_window = (
            continuous_retrieval_variance_point.mean(dim=-1)
        )

        discrete_variance_full = self._variance_from_samples(discrete_samples)
        discrete_retrieval_variance_point = discrete_variance_full.mean(dim=-1)
        discrete_retrieval_variance_window = discrete_retrieval_variance_point.mean(
            dim=-1
        )

        point_anomaly_score_variance = self._variance_from_samples(point_score_samples)
        window_anomaly_score_variance = self._variance_from_samples(window_score_samples)

        classification_probability_variance = None
        classification_variance_mean = None
        if classification_probability_samples is not None:
            classification_probability_variance = self._variance_from_samples(
                classification_probability_samples
            )
            classification_variance_mean = classification_probability_variance.mean(
                dim=-1
            )

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
        self,
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
            logits = torch.log(class_probabilities.clamp_min(torch.finfo(class_probabilities.dtype).eps))
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

    def prepare_synthetic_training_epoch(self) -> None:
        self.synthetic_anomaly_injector.reset_rng()

    def prepare_synthetic_validation_epoch(self) -> None:
        self.synthetic_validation_injector.reset_rng()

    def _continuous_prototype_lookup(
        self,
        hidden: torch.Tensor,
        *,
        stage_name: str,
        active_memory_bank: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # The continuous branch keeps a soft weighted prototype mixture.
        normalized_hidden = self._normalize_hidden_for_memory(hidden)
        continuous_hidden = normalized_hidden
        attention_logits = None
        attention_weights = None
        memory_bypass_active = self._should_bypass_memory_for_stage(stage_name)
        memory_bank_for_read = active_memory_bank

        if memory_bank_for_read is not None and not memory_bypass_active:
            attention_logits = torch.einsum(
                "blh,kh->blk",
                normalized_hidden,
                memory_bank_for_read,
            ) / math.sqrt(self.hidden_dim)
            attention_weights = torch.softmax(attention_logits, dim=-1)
            continuous_hidden = torch.einsum(
                "blk,kh->blh",
                attention_weights,
                memory_bank_for_read,
            )
            continuous_hidden = self._normalize_hidden_for_memory(continuous_hidden)

        return {
            "hidden": hidden,
            "prototype_context": continuous_hidden,
            "prototype_logits": attention_logits,
            "prototype_weights": attention_weights,
            "aux": {
                "branch_name": "continuous",
                "enabled": self.continuous_memory_enabled,
                "num_prototypes": self.continuous_num_prototypes,
                "memory_bypass_active": memory_bypass_active,
                "memory_initialized": self.memory_initialized,
            },
        }

    def _discrete_prototype_lookup(
        self,
        hidden: torch.Tensor,
        *,
        stage_name: str,
        active_codebook: torch.Tensor | None = None,
        precomputed_assignment_logits: torch.Tensor | None = None,
        precomputed_assignment_probabilities: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # The discrete branch keeps a quantized codebook view of the same tokens.
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
        self,
        continuous_hidden: torch.Tensor,
        discrete_hidden: torch.Tensor,
        *,
        base_hidden: torch.Tensor | None = None,
        paired_hidden: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        # Fusion is expressed as exact limiting cases of the same equations.
        # That is why continuous-only and discrete-only ablations need no second model.
        if self.fusion_mode == "task_specific_concat_projection":
            concatenated_hidden = torch.cat(
                [continuous_hidden, discrete_hidden], dim=-1
            )
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
            alpha_scalar = continuous_hidden.new_tensor(
                float(self.active_alpha_override)
            )
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
            cka_features = torch.stack(
                [cka_reconstruction, cka_classification],
                dim=-1,
            )
            alpha = torch.sigmoid(
                self.classification_fusion_gate(cka_features)
            ).squeeze(-1)
            beta = torch.sigmoid(self.reconstruction_fusion_gate(cka_features)).squeeze(
                -1
            )
        alpha_expanded = alpha.view(-1, 1, 1)
        beta_expanded = beta.view(-1, 1, 1)

        # Beta là mức độ mà tác vụ tái tạo chuỗi (reconstruction) sử dụng
        # nhánh các vec-tơ nguyên mẫu rời rạc (discrete prototype).
        # Mình kì vọng giá trị này sẽ nhỏ hơn alpha.
        hidden_reconstruction = (
            beta_expanded * discrete_hidden + (1.0 - beta_expanded) * continuous_hidden
        )

        # Alpha là mức độ mà tác vụ phân loại (classification) sử dụng nhánh các
        # vec-tơ nguyên mẫu rời rạc.
        # Mình kì vọng giá trị này sẽ lớn hơn beta.
        hidden_classification = (
            alpha_expanded * discrete_hidden
            + (1.0 - alpha_expanded) * continuous_hidden
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
                "cka_reconstruction_mean": float(
                    cka_reconstruction.mean().detach().cpu()
                ),
                "cka_reconstruction_std": float(
                    cka_reconstruction.std(unbiased=False).detach().cpu()
                ),
                "cka_classification_mean": float(
                    cka_classification.mean().detach().cpu()
                ),
                "cka_classification_std": float(
                    cka_classification.std(unbiased=False).detach().cpu()
                ),
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }

    def _compute_linear_cka_score(
        self,
        lhs_tokens: torch.Tensor,
        rhs_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if lhs_tokens.shape[0] <= 1:
            return lhs_tokens.new_zeros(())
        centered_lhs = lhs_tokens - lhs_tokens.mean(dim=0, keepdim=True)
        centered_rhs = rhs_tokens - rhs_tokens.mean(dim=0, keepdim=True)
        lhs_gram = centered_lhs @ centered_lhs.T
        rhs_gram = centered_rhs @ centered_rhs.T
        hsic_lhs_rhs = torch.sum(lhs_gram * rhs_gram)
        hsic_lhs_lhs = torch.sum(lhs_gram * lhs_gram)
        hsic_rhs_rhs = torch.sum(rhs_gram * rhs_gram)
        return hsic_lhs_rhs / torch.sqrt(
            hsic_lhs_lhs * hsic_rhs_rhs + self.cka_eps
        ).clamp_min(self.cka_eps)

    def _compute_batch_linear_cka_scores(
        self,
        lhs_hidden: torch.Tensor,
        rhs_hidden: torch.Tensor,
    ) -> torch.Tensor:
        sample_scores: list[torch.Tensor] = []
        for sample_index in range(lhs_hidden.shape[0]):
            sample_scores.append(
                self._compute_linear_cka_score(
                    lhs_hidden[sample_index],
                    rhs_hidden[sample_index],
                )
            )
        return torch.stack(sample_scores)

    def _compute_two_view_contrastive_loss(
        self,
        anchor_hidden: torch.Tensor,
        positive_hidden: torch.Tensor,
        synthetic_anomaly_mask: torch.Tensor,
    ) -> torch.Tensor:
        normal_token_mask = (synthetic_anomaly_mask == 0).reshape(-1)
        if int(normal_token_mask.sum().item()) == 0:
            return self._zero_loss(anchor_hidden)
        anchor_tokens = anchor_hidden.reshape(-1, self.hidden_dim)[normal_token_mask]
        positive_tokens = positive_hidden.reshape(-1, self.hidden_dim)[
            normal_token_mask
        ]
        normalized_anchors = F.normalize(anchor_tokens, dim=-1, eps=self.epsilon)
        normalized_positives = F.normalize(positive_tokens, dim=-1, eps=self.epsilon)
        logits = (normalized_anchors @ normalized_positives.T) / max(
            self.contrastive_temperature, self.epsilon
        )
        targets = torch.arange(
            logits.shape[0],
            device=logits.device,
            dtype=torch.long,
        )
        return F.cross_entropy(logits, targets)

    def _clone_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        cloned_batch: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cloned_batch[key] = value.clone()
            elif isinstance(value, list):
                cloned_batch[key] = [
                    dict(item) if isinstance(item, dict) else item for item in value
                ]
            else:
                cloned_batch[key] = value
        return cloned_batch

    def _prepare_clean_batch(
        self, batch: dict[str, Any], stage_name: str
    ) -> dict[str, Any]:
        # The model owns augmentation timing because synthetic supervision is
        # part of the multitask objective, not a separate preprocessing pipeline.
        if (
            "classification_labels" in batch
            and "synthetic_anomaly_mask" in batch
            and "augmentation_metadata" in batch
        ):
            console_print(
                stage_name.upper(),
                "Received pre-augmented multitask batch",
                **summarize_batch(batch),
                classification_label_distribution=summarize_label_distribution(
                    batch["classification_labels"]
                ),
            )
            return self._clone_batch(batch)

        if stage_name == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)

        prepared_batch = self._clone_batch(batch)
        batch_size, window_size, _ = prepared_batch["x"].shape
        prepared_batch["classification_labels"] = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["classification_class_names"] = (
            self._classification_class_names()
        )
        prepared_batch["synthetic_anomaly_mask"] = torch.zeros(
            batch_size,
            window_size,
            dtype=torch.long,
            device=prepared_batch["x"].device,
        )
        prepared_batch["augmentation_metadata"] = [
            {
                "is_synthetic_anomaly": False,
                "anomaly_family": "clean",
                "anomaly_family_index": None,
                "start_index": None,
                "end_index": None,
                "affected_channels": [],
                "family_parameters_by_channel": {},
            }
            for _ in range(batch_size)
        ]
        if prepared_batch["point_labels"] is None:
            prepared_batch["point_labels"] = prepared_batch[
                "synthetic_anomaly_mask"
            ].clone()
        console_print(
            stage_name.upper(),
            "Prepared clean multitask batch",
            **summarize_batch(prepared_batch),
            classification_label_distribution=summarize_label_distribution(
                prepared_batch["classification_labels"]
            ),
        )
        return prepared_batch

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if stage_name == "val_synth" and self.use_synthetic_validation:
            return self.synthetic_validation_injector.augment_batch(batch)
        return self._prepare_clean_batch(batch, stage_name)

    def _prepare_contrastive_pair_batches(
        self,
        batch: dict[str, Any],
        stage_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if not self.enable_two_view_contrastive:
            return None
        if not self._phase_uses_contrastive_objective():
            return None
        if stage_name not in {"train", "val_synth"}:
            return None
        clean_batch = self._prepare_clean_batch(
            self._clone_batch(batch), stage_name="val"
        )
        augmented_batch = self._prepare_batch(self._clone_batch(batch), stage_name)
        return clean_batch, augmented_batch

    def forward(
        self,
        batch: dict[str, Any],
        stage_name: str = "train",
    ) -> dict[str, Any]:
        """
        Hàm forward này là hàm tính toán quan trọng nhất của mô hình
        trong giai đoạn offline pre-training theo proposal của phương pháp.
        """

        # The forward pass is the main representation story of the thesis model:
        # encode once, build two branch views, fuse per task, then score.
        validate_batch(batch)
        console_print(
            "MODEL", "Multitask forward input batch", **summarize_batch(batch)
        )
        forward_start_time = time.perf_counter()

        # Truyền lô dữ liệu qua encoder để mã hoá
        # Thu được các vec-tơ ẩn (hidden vector)
        # Một vec-tơ ẩn cho mỗi bước thời gian (timestep)
        encoder_outputs = self.encoder(batch)
        hidden = encoder_outputs["hidden"]
        anomaly_mask = batch.get("synthetic_anomaly_mask")
        normal_token_mask = None
        anomaly_token_mask = None
        if anomaly_mask is not None and self.enable_two_view_contrastive:
            normal_token_mask = anomaly_mask == 0
            anomaly_token_mask = anomaly_mask == 1
        if self._phase_uses_prototype_path():
            if (
                self.continuous_prototype_bank is not None
                and self._should_update_memory(stage_name)
            ):
                active_continuous_memory_bank = self._update_continuous_memory_bank(
                    hidden,
                    token_mask=normal_token_mask,
                )
            elif self.continuous_prototype_bank is not None:
                active_continuous_memory_bank = self._normalize_memory_vectors(
                    self.continuous_prototype_bank
                )
            else:
                active_continuous_memory_bank = None
            if self.discrete_codebook is not None and self._should_update_memory(
                stage_name
            ):
                self._update_discrete_codebook_memory(
                    hidden,
                    token_mask=anomaly_token_mask,
                )
                active_assignment_logits = None
                active_assignment_probabilities = None
                active_discrete_codebook = self._normalize_memory_vectors(
                    self.discrete_codebook
                )
            elif self.discrete_codebook is not None:
                active_assignment_logits = None
                active_assignment_probabilities = None
                active_discrete_codebook = self._normalize_memory_vectors(
                    self.discrete_codebook
                )
            else:
                active_assignment_logits = None
                active_assignment_probabilities = None
                active_discrete_codebook = None

            # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu liên tục
            continuous_outputs = self._continuous_prototype_lookup(
                hidden,
                stage_name=stage_name,
                active_memory_bank=active_continuous_memory_bank,
            )

            # Tái tạo các vec-tơ ẩn sử dụng các vec-tơ nguyên mẫu rời rạc
            discrete_outputs = self._discrete_prototype_lookup(
                hidden,
                stage_name=stage_name,
                active_codebook=active_discrete_codebook,
                precomputed_assignment_logits=active_assignment_logits,
                precomputed_assignment_probabilities=active_assignment_probabilities,
            )

            # Kết hợp vec-tơ ẩn từ hai nhánh lại với nhau
            fusion_outputs = self._compute_fusion_outputs(
                continuous_hidden=continuous_outputs["prototype_context"],
                discrete_hidden=discrete_outputs["quantized_hidden"],
                base_hidden=hidden,
                paired_hidden=batch.get("paired_hidden_for_fusion"),
            )
            stochastic_query = None
            monte_carlo_forward_outputs = None
            if self.stochastic_inference:
                query_bundle = self.build_stochastic_queries(
                    hidden,
                    stage_name=stage_name,
                    active_memory_bank=active_continuous_memory_bank,
                    active_codebook=active_discrete_codebook,
                )
                continuous_retrieved_samples = self.sample_continuous_retrieval(
                    query_bundle,
                    self.monte_carlo_samples,
                )
                discrete_retrieved_samples = self.sample_discrete_retrieval(
                    query_bundle,
                    self.monte_carlo_samples,
                )
                discrete_topk_ids = self.sample_discrete_topk_ids(
                    query_bundle,
                    self.monte_carlo_samples,
                )
                stochastic_query = {
                    "schema_version": 3,
                    "enabled": True,
                    "num_samples": self.monte_carlo_samples,
                    "continuous_temperature": self.continuous_temperature,
                    "discrete_temperature": self.discrete_temperature,
                    "continuous_retrieved_samples": continuous_retrieved_samples,
                    "discrete_retrieved_samples": discrete_retrieved_samples,
                    "discrete_topk_ids": discrete_topk_ids,
                }
                if not self.training:
                    monte_carlo_forward_outputs = (
                        self._build_monte_carlo_forward_outputs(
                            batch,
                            fusion_outputs=fusion_outputs,
                            query_bundle=query_bundle,
                            continuous_samples=continuous_retrieved_samples,
                            discrete_samples=discrete_retrieved_samples,
                            discrete_topk_ids=discrete_topk_ids,
                        )
                    )
        else:
            active_continuous_memory_bank = None
            active_discrete_codebook = None
            (
                continuous_outputs,
                discrete_outputs,
                fusion_outputs,
            ) = self._build_phase_passthrough_outputs(hidden)
            stochastic_query = None

        # Lấy ra vec-tơ kết hợp dùng cho từng tác vụ
        hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
        hidden_classification = fusion_outputs["hidden_classification"]

        # Truyền vec-tơ kết hợp dùng cho
        # tác vụ tái tạo qua mạng tái tạo (reconstruction head)
        recon = self.reconstruction_head(hidden_reconstruction)

        # Xét vec-tơ kết hợp dùng cho tác vụ phân loại.
        # RedLamp dùng toàn bộ các vec-tơ ẩn theo thời gian bằng cách trải phẳng
        # cửa sổ thành một biểu diễn cấp-window trước classifier head.
        if hidden_classification.shape[1] != self.window_size:
            raise ValueError(
                "hidden_classification must have window dimension "
                f"{self.window_size}, but received {hidden_classification.shape[1]}"
            )
        flattened_classification_hidden = hidden_classification.reshape(
            hidden_classification.shape[0],
            self.window_size * self.hidden_dim,
        )
        logits = None
        class_probabilities = None
        monte_carlo_uncertainty = None
        point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)
        window_scores = point_scores.mean(dim=1)
        if not self.training and monte_carlo_forward_outputs is not None:
            recon = monte_carlo_forward_outputs["outputs"]["recon"]
            logits = monte_carlo_forward_outputs["outputs"]["logits"]
            point_scores = monte_carlo_forward_outputs["outputs"]["point_scores"]
            window_scores = monte_carlo_forward_outputs["outputs"]["window_scores"]
            class_probabilities = monte_carlo_forward_outputs["aux"].get(
                "class_probabilities"
            )
            stochastic_query = monte_carlo_forward_outputs["aux"].get(
                "stochastic_query"
            )
            monte_carlo_uncertainty = monte_carlo_forward_outputs["aux"].get(
                "uncertainty"
            )
        elif self.enable_classification_path:
            logits = self.classification_head(flattened_classification_hidden)
            class_probabilities = torch.softmax(logits, dim=-1)

        # Chuẩn bị đầu ra theo thoả thuận (contract) đã được thiết lập.
        # Xem: src/core/contracts.py
        outputs = {
            "hidden": hidden,
            "pooled": flattened_classification_hidden,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": window_scores,
            "aux": {
                "encoder": encoder_outputs["aux"],
                "continuous_branch": continuous_outputs,
                "discrete_branch": discrete_outputs,
                "fusion": fusion_outputs["aux"],
                "active_continuous_memory_bank": active_continuous_memory_bank,
                "active_discrete_codebook": active_discrete_codebook,
                "hidden_reconstruction": hidden_reconstruction,
                "hidden_classification": hidden_classification,
                "alpha": fusion_outputs["alpha"],
                "beta": fusion_outputs["beta"],
                "class_probabilities": class_probabilities,
                "deterministic_geometry": {
                    "hidden_reconstruction": hidden_reconstruction,
                    "hidden_classification": hidden_classification,
                    "alpha": fusion_outputs["alpha"],
                    "beta": fusion_outputs["beta"],
                    "fusion": fusion_outputs["aux"],
                },
                "classification_class_names": self._classification_class_names(),
                "memory": self.get_memory_lifecycle_state(),
                "stochastic_query": stochastic_query,
                "uncertainty": monte_carlo_uncertainty,
                "retention": {
                    "return_mc_samples": self.return_mc_samples,
                    "sample_retention_policy": self.sample_retention_policy,
                },
                "forward_pass_seconds": time.perf_counter() - forward_start_time,
            },
        }
        validate_model_outputs(outputs)
        console_print(
            "MODEL",
            "Multitask forward outputs",
            hidden=summarize_tensor(outputs["hidden"]),
            pooled=summarize_tensor(outputs["pooled"]),
            recon=summarize_tensor(outputs["recon"]),
            logits=summarize_tensor(outputs["logits"]),
            point_scores=summarize_tensor(outputs["point_scores"]),
            window_scores=summarize_tensor(outputs["window_scores"]),
            continuous_hidden=summarize_tensor(
                outputs["aux"]["continuous_branch"]["prototype_context"]
            ),
            discrete_hidden=summarize_tensor(
                outputs["aux"]["discrete_branch"]["quantized_hidden"]
            ),
            assignment_probabilities=summarize_tensor(
                outputs["aux"]["discrete_branch"]["assignment_probabilities"]
            ),
            hidden_reconstruction=summarize_tensor(
                outputs["aux"]["hidden_reconstruction"]
            ),
            hidden_classification=summarize_tensor(
                outputs["aux"]["hidden_classification"]
            ),
            forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
        )
        return outputs
