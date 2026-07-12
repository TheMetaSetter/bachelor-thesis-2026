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
from src.models.thesis_multitask_routing_forward_helpers import (
    _prepare_clean_batch as routing_prepare_clean_batch,
    forward as routing_forward,
)
from src.models.thesis_multitask_routing_geometry_helpers import (
    _build_monte_carlo_forward_outputs as routing_build_monte_carlo_forward_outputs,
    _build_monte_carlo_uncertainty as routing_build_monte_carlo_uncertainty,
    _build_sampled_fusion_hidden as routing_build_sampled_fusion_hidden,
    _compute_fusion_outputs as routing_compute_fusion_outputs,
    _discrete_prototype_lookup as routing_discrete_prototype_lookup,
    _variance_from_samples as routing_variance_from_samples,
)
from src.models.thesis_multitask_routing_helpers import (
    build_stochastic_queries as routing_build_stochastic_queries,
    sample_continuous_retrieval as routing_sample_continuous_retrieval,
    sample_discrete_retrieval as routing_sample_discrete_retrieval,
    sample_discrete_topk_ids as routing_sample_discrete_topk_ids,
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
        return routing_build_stochastic_queries(
            self,
            hidden,
            stage_name=stage_name,
            active_memory_bank=active_memory_bank,
            active_codebook=active_codebook,
        )

    def sample_continuous_retrieval(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        return routing_sample_continuous_retrieval(self, query_bundle, num_samples)

    def sample_discrete_retrieval(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        return routing_sample_discrete_retrieval(self, query_bundle, num_samples)

    def sample_discrete_topk_ids(
        self,
        query_bundle: QueryBundle,
        num_samples: int,
    ) -> torch.Tensor:
        return routing_sample_discrete_topk_ids(self, query_bundle, num_samples)

    def _build_sampled_fusion_hidden(
        self,
        continuous_samples: torch.Tensor,
        discrete_samples: torch.Tensor,
        *,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return routing_build_sampled_fusion_hidden(
            self,
            continuous_samples,
            discrete_samples,
            alpha=alpha,
            beta=beta,
        )

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
        return routing_build_monte_carlo_uncertainty(
            self,
            reconstruction_samples=reconstruction_samples,
            continuous_samples=continuous_samples,
            discrete_samples=discrete_samples,
            point_score_samples=point_score_samples,
            window_score_samples=window_score_samples,
            classification_probability_samples=classification_probability_samples,
        )

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
        return routing_build_monte_carlo_forward_outputs(
            self,
            batch,
            fusion_outputs=fusion_outputs,
            query_bundle=query_bundle,
            continuous_samples=continuous_samples,
            discrete_samples=discrete_samples,
            discrete_topk_ids=discrete_topk_ids,
        )

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
        return routing_discrete_prototype_lookup(
            self,
            hidden,
            stage_name=stage_name,
            active_codebook=active_codebook,
            precomputed_assignment_logits=precomputed_assignment_logits,
            precomputed_assignment_probabilities=precomputed_assignment_probabilities,
        )

    def _compute_fusion_outputs(
        self,
        continuous_hidden: torch.Tensor,
        discrete_hidden: torch.Tensor,
        *,
        base_hidden: torch.Tensor | None = None,
        paired_hidden: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return routing_compute_fusion_outputs(
            self,
            continuous_hidden,
            discrete_hidden,
            base_hidden=base_hidden,
            paired_hidden=paired_hidden,
        )

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
        return routing_prepare_clean_batch(self, batch, stage_name)

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
        return routing_forward(self, batch, stage_name=stage_name)
