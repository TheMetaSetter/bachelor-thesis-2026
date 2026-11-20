from __future__ import annotations

import torch
import torch.nn.functional as F


def _require_mask_shape(
    mask: torch.Tensor,
    expected_shape: tuple[int, int],
    context: str,
) -> torch.Tensor:
    if mask.shape != expected_shape:
        raise ValueError(f"{context} must have shape {list(expected_shape)}")
    return mask.to(dtype=torch.bool)


def compute_token_multi_positive_info_nce(
    projected_hidden: torch.Tensor,
    reference_hidden: torch.Tensor,
    anomalous_codewords: torch.Tensor,
    pnn_mask: torch.Tensor | None = None,
    recurrent_signature_ids: torch.Tensor | None = None,
    known_anomaly_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Use source/same-signature positives and anomaly-memory negatives."""
    if reference_hidden.shape != projected_hidden.shape or reference_hidden.ndim != 3:
        raise ValueError("hidden tensors must share shape [B, L, H]")
    if anomalous_codewords.ndim != 2 or anomalous_codewords.shape[1] != projected_hidden.shape[-1]:
        raise ValueError("anomalous_codewords must have shape [K_anom, H]")
    if anomalous_codewords.shape[0] == 0:
        raise ValueError("at least one anomalous codeword is required")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    anchor_mask = torch.ones(
        projected_hidden.shape[:2], dtype=torch.bool, device=projected_hidden.device
    )
    if pnn_mask is not None:
        anchor_mask = _require_mask_shape(
            pnn_mask.to(device=projected_hidden.device),
            projected_hidden.shape[:2],
            "pnn_mask",
        )
    if recurrent_signature_ids is not None:
        recurrent_signature_ids = recurrent_signature_ids.to(
            device=projected_hidden.device
        )
        if (
            recurrent_signature_ids.ndim != 3
            or recurrent_signature_ids.shape[:2] != projected_hidden.shape[:2]
        ):
            raise ValueError("recurrent_signature_ids must have shape [B, L, T]")
    if known_anomaly_mask is not None:
        known_anomaly_mask = _require_mask_shape(
            known_anomaly_mask.to(device=projected_hidden.device),
            projected_hidden.shape[:2],
            "known_anomaly_mask",
        )
    if not anchor_mask.any():
        return projected_hidden.sum() * 0.0
    projected = F.normalize(projected_hidden, dim=-1)
    source = F.normalize(reference_hidden.detach(), dim=-1)
    anomaly_keys = F.normalize(anomalous_codewords.detach(), dim=-1)
    return _mean_token_loss(
        projected, source, anomaly_keys, anchor_mask, recurrent_signature_ids, known_anomaly_mask, temperature,
    )


def _mean_token_loss(projected, source, anomaly_keys, anchor_mask, recurrent_signature_ids,
                     known_anomaly_mask, temperature):
    losses = []
    for batch_id, token_id in anchor_mask.nonzero(as_tuple=False).tolist():
        anchor = projected[batch_id, token_id]
        positives = [source[batch_id, token_id]]
        if recurrent_signature_ids is not None:
            signature = recurrent_signature_ids[batch_id, token_id]
            matches = (recurrent_signature_ids == signature).all(dim=-1) & anchor_mask
            matches[batch_id, token_id] = False
            positives.extend(projected[matches].detach().unbind(0))
        negatives = [anomaly_keys]
        if known_anomaly_mask is not None and known_anomaly_mask.any():
            negatives.extend(
                [
                    projected[known_anomaly_mask].detach(),
                    source[known_anomaly_mask],
                ]
            )
        positive_logits = anchor @ torch.stack(positives).T / temperature
        negative_logits = anchor @ torch.cat(negatives, dim=0).T / temperature
        losses.append(
            torch.logsumexp(torch.cat([positive_logits, negative_logits]), dim=0)
            - torch.logsumexp(positive_logits, dim=0)
        )
    return torch.stack(losses).mean()
