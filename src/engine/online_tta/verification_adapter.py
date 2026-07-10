"""Pure prototype filters used by one shared verification buffer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.engine.online_tta.signature_verification import (
    PrototypeVerificationMetadata,
    SignatureWindow,
    build_pnn_token_mask,
    filter_known_anomaly_tokens,
    find_recurrent_signatures,
    ordered_continuous_signature,
)


@dataclass(frozen=True)
class VerificationResult:
    """Describe one entry after buffer-wide prototype verification."""

    adapted: bool
    pseudo_normal_points: int
    reason: str
    pnn_mask: torch.Tensor


def build_entry_batch(entry: dict[str, Any], device: str) -> dict[str, Any]:
    """Rebuild an unlabeled online batch from one serialized buffer entry."""
    x_tensor = torch.as_tensor(entry["window"], dtype=torch.float32, device=device)
    if x_tensor.ndim != 2:
        raise ValueError("verification entry window must have shape [L, C]")
    x_tensor = x_tensor.unsqueeze(0)
    meta = {
        "entity_id": str(entry["entity_id"]),
        "start_index": int(entry["window_start"]),
        "end_index": int(entry["window_end"]),
        "stream_step": int(entry["stream_step"]),
    }
    return {
        "x": x_tensor,
        "view_a": x_tensor,
        "view_b": x_tensor,
        "point_labels": None,
        "mask": None,
        "timestamps": None,
        "meta": [meta],
    }


def _score_verification_entry(
    model: torch.nn.Module, entry: dict[str, Any], device: str
) -> tuple[torch.Tensor, list[list[tuple[int, ...]]]]:
    batch = build_entry_batch(entry, device)
    model.eval()
    with torch.no_grad():
        outputs = model.forward(batch)
    hidden = outputs["aux"]["projected_hidden"].detach()
    reference_model = model.reference_encoder.model
    metadata = PrototypeVerificationMetadata.from_model(reference_model)
    known_anomaly = filter_known_anomaly_tokens(
        hidden,
        metadata.codebook.to(device),
        metadata.anomalous_codeword_mask.to(device),
        metadata.anomaly_radii.to(device),
    )
    prototypes = reference_model.continuous_prototype_bank
    signatures = ordered_continuous_signature(hidden, prototypes.to(device), topk=3)
    return known_anomaly, signatures


def verify_buffer_entries(
    model: torch.nn.Module, entries: list[dict[str, Any]], device: str
) -> dict[str, VerificationResult]:
    """Run discrete-radius then recurrent-signature filters over all entries."""
    scored: list[tuple[dict[str, Any], torch.Tensor, list[list[tuple[int, ...]]]]] = []
    signature_windows: list[SignatureWindow] = []
    for entry in entries:
        known_anomaly, signatures = _score_verification_entry(model, entry, device)
        scored.append((entry, known_anomaly, signatures))
        signature_windows.append(
            SignatureWindow(
                str(entry["entity_id"]),
                int(entry["window_start"]),
                int(entry["window_end"]),
                signatures,
            )
        )
    recurrent = find_recurrent_signatures(signature_windows)
    results: dict[str, VerificationResult] = {}
    for entry, known_anomaly, signatures in scored:
        pnn_mask = build_pnn_token_mask(signatures, recurrent, known_anomaly)
        count = int(pnn_mask.sum().item())
        results[str(entry["entry_id"])] = VerificationResult(
            adapted=count > 0,
            pseudo_normal_points=count,
            reason="recurrent_signature" if count else "no_recurrent_signature",
            pnn_mask=pnn_mask.detach(),
        )
    return results
