from __future__ import annotations

"""Sampling helpers for the thesis multitask routing path."""

import math
from typing import Any

import torch

from src.models.thesis_multitask_components import QueryBundle


def build_stochastic_queries(
    self: Any,
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
    self: Any,
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
    self: Any,
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
            sampled_logits = sampled_logits + self._sample_gumbel_noise(sampled_logits)
        sampled_logits = sampled_logits / query_bundle.discrete_temperature
        assignment_probabilities = torch.softmax(sampled_logits, dim=-1)

    sample_hidden = torch.einsum(
        "bmlk,kh->bmlh",
        assignment_probabilities,
        query_bundle.discrete_codebook,
    )
    return self._normalize_hidden_for_memory(sample_hidden)


def sample_discrete_topk_ids(
    self: Any,
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
