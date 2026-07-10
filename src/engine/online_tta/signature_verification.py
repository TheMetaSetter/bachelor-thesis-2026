"""Read-only prototype and recurrent-signature helpers for online PNN."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import torch
import torch.nn.functional as F


def nearest_discrete_codeword(hidden: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden.ndim != 3 or codebook.ndim != 2 or hidden.shape[-1] != codebook.shape[-1]:
        raise ValueError("hidden must be [B,L,H] and codebook [K,H]")
    distances = 1.0 - F.normalize(hidden, dim=-1) @ F.normalize(codebook, dim=-1).T
    values, ids = distances.min(dim=-1)
    return ids, values


def filter_known_anomaly_tokens(hidden: torch.Tensor, codebook: torch.Tensor, anomalous_codeword_mask: torch.Tensor, anomaly_radii: torch.Tensor) -> torch.Tensor:
    ids, distances = nearest_discrete_codeword(hidden, codebook)
    if anomalous_codeword_mask.shape != (codebook.shape[0],) or anomaly_radii.shape != (codebook.shape[0],):
        raise ValueError("codeword mask and radii must have shape [K]")
    return anomalous_codeword_mask.to(device=ids.device)[ids] & (distances <= anomaly_radii.to(distances.device)[ids])


def ordered_continuous_signature(hidden: torch.Tensor, continuous_prototypes: torch.Tensor, topk: int = 3) -> list[list[tuple[int, ...]]]:
    if not 1 <= topk <= continuous_prototypes.shape[0]:
        raise ValueError("topk must be between 1 and the number of prototypes")
    distances = 1.0 - F.normalize(hidden, dim=-1) @ F.normalize(continuous_prototypes, dim=-1).T
    ids = distances.topk(topk, dim=-1, largest=False).indices
    return [[tuple(int(v) for v in row) for row in batch] for batch in ids.tolist()]


@dataclass(frozen=True)
class SignatureWindow:
    entity_id: str
    start: int
    end: int
    signatures: list[list[tuple[int, ...]]]


def find_recurrent_signatures(window_signatures: Sequence[SignatureWindow]) -> set[tuple[int, ...]]:
    counts: dict[tuple[int, ...], list[tuple[str, int, int]]] = {}
    for window in window_signatures:
        for token in window.signatures:
            for signature in token:
                counts.setdefault(signature, []).append((window.entity_id, window.start, window.end))
    return {signature for signature, occurrences in counts.items() if any(
        left[0] == right[0] and left[2] <= right[1] or left[0] != right[0]
        for index, left in enumerate(occurrences) for right in occurrences[index + 1:]
    )}


def build_pnn_token_mask(signatures: list[list[tuple[int, ...]]], recurrent_signatures: set[tuple[int, ...]], known_anomaly_mask: torch.Tensor) -> torch.Tensor:
    mask = torch.tensor(
        [[tuple(token) in recurrent_signatures for token in batch] for batch in signatures],
        dtype=torch.bool,
        device=known_anomaly_mask.device,
    )
    if mask.shape != known_anomaly_mask.shape:
        raise ValueError("known_anomaly_mask shape must match signature tokens")
    return mask & ~known_anomaly_mask.to(dtype=torch.bool)
