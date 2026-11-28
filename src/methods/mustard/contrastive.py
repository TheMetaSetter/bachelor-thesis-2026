"""Vectorized multi-positive InfoNCE loss for MUSTARD A2 online updates."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_inputs(
    projected_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    anomalous_codewords: torch.Tensor,
    temperature: float,
) -> None:
    if reference_hidden.shape != projected_hidden.shape or reference_hidden.ndim != 3:
        raise ValueError("hidden tensors must share shape [B, L, H]")
    if (
        anomalous_codewords.ndim != 2
        or anomalous_codewords.shape[1] != projected_hidden.shape[-1]
        or anomalous_codewords.shape[0] == 0
    ):
        raise ValueError("anomalous_codewords must have non-empty shape [K_anom, H]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")


def _build_masks(
    anchor_ids: torch.Tensor,
    anchor_count: int,
    num_tokens: int,
    num_negatives: int,
    recurrent_signature_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = anchor_ids.device
    # Keys layout: [source_tokens (N), anchor_tokens (A), negatives (K_neg)]
    total_keys = num_tokens + anchor_count + num_negatives
    pos_mask = torch.zeros((anchor_count, total_keys), dtype=torch.bool, device=device)
    pos_mask[torch.arange(anchor_count, device=device), anchor_ids] = True
    if recurrent_signature_ids is not None:
        flat_sigs = recurrent_signature_ids.reshape(num_tokens, -1)[anchor_ids]
        same_sig = (flat_sigs[:, None, :] == flat_sigs[None, :, :]).all(dim=-1)
        same_sig.fill_diagonal_(False)
        pos_mask[:, num_tokens : num_tokens + anchor_count] = same_sig
    neg_mask = torch.zeros((anchor_count, total_keys), dtype=torch.bool, device=device)
    neg_mask[:, num_tokens + anchor_count :] = True
    return pos_mask, neg_mask


def compute_token_multi_positive_info_nce(
    projected_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    anomalous_codewords: torch.Tensor,
    pnn_mask: torch.Tensor | None = None,
    recurrent_signature_ids: torch.Tensor | None = None,
    known_anomaly_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull verified tokens toward source/same-signature positives and away from anomalies."""
    _validate_inputs(
        projected_hidden, reference_hidden, anomalous_codewords, temperature
    )
    device = projected_hidden.device
    shape_bl = projected_hidden.shape[:2]
    anchor_mask = (
        torch.ones(shape_bl, dtype=torch.bool, device=device)
        if pnn_mask is None
        else pnn_mask.to(device=device, dtype=torch.bool)
    )
    anchor_ids = anchor_mask.flatten().nonzero(as_tuple=False).squeeze(-1)
    if anchor_ids.numel() == 0:
        return projected_hidden.sum() * 0.0

    projected = F.normalize(projected_hidden, dim=-1).reshape(
        -1, projected_hidden.shape[-1]
    )
    source = F.normalize(reference_hidden.detach(), dim=-1).reshape_as(projected)
    anchors = projected[anchor_ids]

    negatives = [F.normalize(anomalous_codewords.detach(), dim=-1)]
    if known_anomaly_mask is not None:
        anom = known_anomaly_mask.to(device=device, dtype=torch.bool).flatten()
        if anom.any():
            negatives.extend([projected[anom].detach(), source[anom]])
    neg_bank = torch.cat(negatives, dim=0)

    keys = torch.cat([source, anchors.detach(), neg_bank], dim=0)
    logits = (anchors @ keys.T) / temperature
    pos_mask, neg_mask = _build_masks(
        anchor_ids, len(anchors), len(source), len(neg_bank), recurrent_signature_ids
    )
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    all_logits = logits.masked_fill(~(pos_mask | neg_mask), float("-inf"))
    return (
        torch.logsumexp(all_logits, dim=-1) - torch.logsumexp(pos_logits, dim=-1)
    ).mean()
