"""Signature-based token verification for MUSTARD online updates."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def nearest_discrete_codeword(
    hidden: torch.Tensor, codebook: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest discrete codeword index and cosine distance per token."""
    if hidden.ndim != 3 or codebook.ndim != 2 or hidden.shape[-1] != codebook.shape[-1]:
        raise ValueError("hidden must be [B,L,H] and codebook [K,H]")
    distances = 1.0 - F.normalize(hidden, dim=-1) @ F.normalize(codebook, dim=-1).T
    values, ids = distances.min(dim=-1)
    return ids, values


def filter_known_anomaly_tokens(
    hidden: torch.Tensor,
    codebook: torch.Tensor,
    anomalous_codeword_mask: torch.Tensor,
    anomaly_radii: torch.Tensor,
) -> torch.Tensor:
    """Mask tokens that fall within the radius of a known anomalous codeword."""
    ids, distances = nearest_discrete_codeword(hidden, codebook)
    if anomalous_codeword_mask.shape != (codebook.shape[0],) or anomaly_radii.shape != (
        codebook.shape[0],
    ):
        raise ValueError("codeword mask and radii must have shape [K]")
    return anomalous_codeword_mask.to(device=ids.device)[ids] & (
        distances <= anomaly_radii.to(distances.device)[ids]
    )


def ordered_continuous_signature(
    hidden: torch.Tensor, continuous_prototypes: torch.Tensor, topk: int = 3
) -> torch.Tensor:
    """Return [B, L, topk] prototype indices ordered by cosine similarity."""
    if not 1 <= topk <= continuous_prototypes.shape[0]:
        raise ValueError("topk must be between 1 and the number of prototypes")
    similarity = (
        F.normalize(hidden, dim=-1) @ F.normalize(continuous_prototypes, dim=-1).T
    )
    return similarity.topk(topk, dim=-1, largest=True, sorted=True).indices


def find_recurrent_signatures(
    windows: Sequence[tuple[int, int, torch.Tensor]],
) -> set[tuple[int, ...]]:
    """Return signatures that appear in at least two non-overlapping windows."""
    occurrences: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for start, stop, signatures in windows:
        for row in signatures.reshape(-1, signatures.shape[-1]).tolist():
            occurrences.setdefault(tuple(row), []).append((start, stop))
    return {
        sig
        for sig, spans in occurrences.items()
        if any(
            left_stop <= right_start
            for index, (_, left_stop) in enumerate(spans)
            for right_start, _ in spans[index + 1 :]
        )
    }


def build_pnn_token_mask(
    signatures: torch.Tensor,
    recurrent_signatures: set[tuple[int, ...]],
    known_anomaly_mask: torch.Tensor,
) -> torch.Tensor:
    """Select tokens with recurrent signatures that are not known anomalies."""
    batch_size, length, _ = signatures.shape
    mask = torch.zeros(
        (batch_size, length), dtype=torch.bool, device=known_anomaly_mask.device
    )
    if recurrent_signatures:
        for b in range(batch_size):
            for t, row in enumerate(signatures[b].tolist()):
                if tuple(row) in recurrent_signatures:
                    mask[b, t] = True
    if mask.shape != known_anomaly_mask.shape:
        raise ValueError("known_anomaly_mask shape must match signature tokens")
    return mask & ~known_anomaly_mask.to(dtype=torch.bool)
