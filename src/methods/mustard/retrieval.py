"""Retrieve continuous prototypes and sparse codewords."""

from collections.abc import Mapping
from math import sqrt

import torch
from torch.nn import functional as F


def _retrieve_branch(query, bank, settings, epsilon, *, sparse, generator=None):
    if bank is None:
        return query, None
    logits = query @ bank.T
    if sparse:
        logits, indices = logits.topk(settings["top_k"], dim=-1)
    else:
        logits = logits / sqrt(query.shape[-1])
    if generator is not None:
        uniform = torch.rand(logits.shape, generator=generator, dtype=logits.dtype)
        uniform = uniform.to(logits.device).clamp(min=epsilon, max=1.0 - epsilon)
        logits = logits - torch.log(-torch.log(uniform))
    weights = (logits / settings["temperature"]).softmax(dim=-1)
    if sparse:
        weights = torch.zeros((*logits.shape[:-1], len(bank)), device=query.device, dtype=query.dtype).scatter(
            -1, indices, weights
        )
    return F.normalize(weights @ bank, dim=-1, eps=epsilon), weights


def retrieve(features: torch.Tensor, memory: Mapping[str, torch.Tensor], settings: dict) -> dict:
    """Return the normalized query and the two task representations."""
    if features.ndim != 3:
        raise ValueError("retrieval requires [batch, time, hidden] features")
    epsilon = settings["memory"]["normalization_epsilon"]
    query = F.normalize(features, dim=-1, eps=epsilon)
    generator = settings.get("generator")
    result = {"query": query}
    for branch in ("continuous", "discrete"):
        bank = memory.get(branch)
        if bank is None or not settings[branch].get("enabled", True):
            context, weights = features, None
        else:
            context, weights = _retrieve_branch(
                query, bank, settings[branch], epsilon, sparse=branch == "discrete", generator=generator if branch == "continuous" else None
            )
        result[branch] = context
        result[f"{branch}_weights"] = weights
    return result


def _kmeans(tokens: torch.Tensor, count: int, epsilon: float) -> torch.Tensor:
    """Fit normalized clusters with deterministic farthest-first seeds."""
    if tokens.ndim != 2 or len(tokens) == 0 or count < 1:
        raise ValueError("memory initialization requires tokens and a positive center count")
    tokens = F.normalize(tokens, dim=-1, eps=epsilon)
    if len(tokens) <= count:
        return tokens[torch.arange(count, device=tokens.device) % len(tokens)]
    first = ((tokens - tokens.mean(dim=0)) ** 2).sum(dim=-1).argmin()
    indices = [first]
    minimum = ((tokens - tokens[first]) ** 2).sum(dim=-1)
    for _ in range(1, count):
        next_index = minimum.argmax()
        indices.append(next_index)
        minimum = torch.minimum(minimum, ((tokens - tokens[next_index]) ** 2).sum(dim=-1))
    centers = tokens[torch.stack(indices)]
    for _ in range(10):
        assignments = torch.cdist(tokens, centers).argmin(dim=-1)
        for index in range(count):
            members = tokens[assignments == index]
            if len(members):
                centers[index] = F.normalize(members.mean(dim=0), dim=0, eps=epsilon)
    return centers
