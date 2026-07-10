"""Read-only prototype and recurrent-signature helpers for online PNN."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrototypeVerificationMetadata:
    """Detached codebook metadata required for safe anomaly filtering."""

    codebook: torch.Tensor
    anomalous_codeword_mask: torch.Tensor
    anomaly_radii: torch.Tensor

    def __post_init__(self) -> None:
        codebook_size = self.codebook.shape[0]
        if self.codebook.ndim != 2:
            raise ValueError("codebook must have shape [K, H]")
        if not torch.is_floating_point(self.codebook):
            raise TypeError("codebook must use a floating dtype")
        if self.anomalous_codeword_mask.shape != (codebook_size,):
            raise ValueError("anomalous_codeword_mask must have shape [K]")
        if self.anomalous_codeword_mask.dtype != torch.bool:
            raise TypeError("anomalous_codeword_mask must use bool dtype")
        if self.anomaly_radii.shape != (codebook_size,):
            raise ValueError("anomaly_radii must have shape [K]")
        if not torch.is_floating_point(self.anomaly_radii):
            raise TypeError("anomaly_radii must use a floating dtype")
        if (self.anomaly_radii < 0).any().item():
            raise ValueError("anomaly_radii must be non-negative")

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "PrototypeVerificationMetadata":
        if getattr(model, "verification_metadata_source", "") in {
            "",
            "uninitialized",
            "disabled",
        }:
            raise AttributeError(
                "online reference model checkpoint lacks calibrated "
                "anomalous_codeword_mask and anomaly_radii metadata"
            )
        codebook = getattr(model, "discrete_codebook", None)
        mask = getattr(model, "anomalous_codeword_mask", None)
        radii = getattr(model, "anomaly_radii", None)
        if not all(
            isinstance(value, torch.Tensor) for value in (codebook, mask, radii)
        ):
            raise AttributeError(
                "online reference model must expose discrete_codebook, "
                "anomalous_codeword_mask, and anomaly_radii"
            )
        return cls(codebook.detach(), mask.detach().bool(), radii.detach())


def nearest_discrete_codeword(
    hidden: torch.Tensor, codebook: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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
) -> list[list[tuple[int, ...]]]:
    if not 1 <= topk <= continuous_prototypes.shape[0]:
        raise ValueError("topk must be between 1 and the number of prototypes")
    distances = (
        1.0 - F.normalize(hidden, dim=-1) @ F.normalize(continuous_prototypes, dim=-1).T
    )
    ids = distances.topk(topk, dim=-1, largest=False).indices
    return [[tuple(int(v) for v in row) for row in batch] for batch in ids.tolist()]


@dataclass(frozen=True)
class SignatureWindow:
    entity_id: str
    start: int
    end: int
    signatures: list[list[tuple[int, ...]]]


def signature_window_to_dict(window: SignatureWindow) -> dict[str, object]:
    """Convert one signature window to checkpoint-safe primitives."""
    return {
        "entity_id": window.entity_id,
        "start": window.start,
        "end": window.end,
        "signatures": [
            [list(signature) for signature in token] for token in window.signatures
        ],
    }


def signature_window_from_dict(payload: dict[str, object]) -> SignatureWindow:
    """Restore one signature window from checkpoint-safe primitives."""
    raw_signatures = payload.get("signatures", [])
    signatures = [
        [tuple(int(value) for value in signature) for signature in token]
        for token in raw_signatures  # type: ignore[union-attr]
    ]
    return SignatureWindow(
        str(payload["entity_id"]),
        int(payload["start"]),
        int(payload["end"]),
        signatures,
    )


def find_recurrent_signatures(
    window_signatures: Sequence[SignatureWindow],
) -> set[tuple[int, ...]]:
    counts: dict[tuple[int, ...], list[tuple[str, int, int]]] = {}
    for window in window_signatures:
        for token in window.signatures:
            for signature in token:
                counts.setdefault(signature, []).append(
                    (window.entity_id, window.start, window.end)
                )
    return {
        signature
        for signature, occurrences in counts.items()
        if any(
            left[0] == right[0] and left[2] <= right[1] or left[0] != right[0]
            for index, left in enumerate(occurrences)
            for right in occurrences[index + 1 :]
        )
    }


def build_pnn_token_mask(
    signatures: list[list[tuple[int, ...]]],
    recurrent_signatures: set[tuple[int, ...]],
    known_anomaly_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.tensor(
        [
            [tuple(token) in recurrent_signatures for token in batch]
            for batch in signatures
        ],
        dtype=torch.bool,
        device=known_anomaly_mask.device,
    )
    if mask.shape != known_anomaly_mask.shape:
        raise ValueError("known_anomaly_mask shape must match signature tokens")
    return mask & ~known_anomaly_mask.to(dtype=torch.bool)
