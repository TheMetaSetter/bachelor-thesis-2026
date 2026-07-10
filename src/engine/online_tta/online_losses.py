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
    reference_hidden: torch.Tensor,
    projected_hidden: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrast every projected token against same-position positives."""
    if reference_hidden.shape != projected_hidden.shape or reference_hidden.ndim != 3:
        raise ValueError("hidden tensors must share shape [B, L, H]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    reference = F.normalize(reference_hidden, dim=-1).reshape(
        -1, reference_hidden.shape[-1]
    )
    projected = F.normalize(projected_hidden, dim=-1).reshape_as(reference)
    logits = projected @ reference.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
