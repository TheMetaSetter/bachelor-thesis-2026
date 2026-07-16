from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    squared_error = (prediction - target) ** 2
    if mask is None:
        return torch.mean(squared_error)

    mask_tensor = mask.to(dtype=prediction.dtype)
    if mask_tensor.shape != squared_error.shape:
        if mask_tensor.dim() == squared_error.dim() - 1:
            mask_tensor = mask_tensor.unsqueeze(-1)
        mask_tensor = mask_tensor.expand_as(squared_error)

    normalizer = torch.sum(mask_tensor).clamp_min(1.0)
    return torch.sum(squared_error * mask_tensor) / normalizer


def compute_a1_pnn_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _masked_mean_squared_error(reconstruction, target, mask)


def compute_a2_hard_old_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return _masked_mean_squared_error(reconstruction, target, mask)


def compute_hard_old_hinge_loss(score: torch.Tensor, b_window: float) -> torch.Tensor:
    """Return mean ``relu(score - B_window)^2`` without label-dependent state."""
    return torch.relu(score - float(b_window)).square().mean()


def _require_mask_shape(
    mask: torch.Tensor,
    expected_shape: tuple[int, int],
    context: str,
) -> torch.Tensor:
    if mask.shape != expected_shape:
        raise ValueError(f"{context} must have shape {list(expected_shape)}")
    return mask.to(dtype=torch.bool)


def compute_masked_pnn_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    pnn_mask: torch.Tensor,
) -> torch.Tensor:
    if reconstruction.shape != target.shape or reconstruction.ndim != 3:
        raise ValueError("reconstruction and target must have shape [B, L, C]")
    if pnn_mask.shape != reconstruction.shape[:2]:
        raise ValueError("pnn_mask must have shape [B, L]")
    return _masked_mean_squared_error(reconstruction, target, pnn_mask)


def compute_a2_online_contrastive_loss(
    reference_hidden: torch.Tensor,
    projected_hidden: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    pooled_reference = F.normalize(reference_hidden.mean(dim=1), dim=-1)
    pooled_projected = F.normalize(projected_hidden.mean(dim=1), dim=-1)
    similarity_logits = pooled_projected @ pooled_reference.T / temperature
    labels = torch.arange(similarity_logits.shape[0], device=similarity_logits.device)
    return 0.5 * (
        F.cross_entropy(similarity_logits, labels)
        + F.cross_entropy(similarity_logits.T, labels)
    )


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
    if (
        anomalous_codewords.ndim != 2
        or anomalous_codewords.shape[1] != projected_hidden.shape[-1]
    ):
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
    projected = F.normalize(projected_hidden, dim=-1)
    source = F.normalize(reference_hidden.detach(), dim=-1)
    anomaly_keys = F.normalize(anomalous_codewords.detach(), dim=-1)
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
    if not losses:
        return projected_hidden.sum() * 0.0
    return torch.stack(losses).mean()
