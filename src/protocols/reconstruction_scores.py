from __future__ import annotations

from typing import Any

import torch

from src.data.scalers import SequenceStandardScaler


def _validate_score_inputs(
    input_scaled: torch.Tensor, reconstruction: torch.Tensor
) -> torch.Tensor:
    if not isinstance(input_scaled, torch.Tensor):
        raise TypeError("input_scaled must be a torch.Tensor")
    if input_scaled.ndim != 3:
        raise ValueError("input_scaled must have shape [B, L, D]")
    if not isinstance(reconstruction, torch.Tensor):
        raise TypeError("reconstruction must be a torch.Tensor")
    if reconstruction.ndim not in {3, 4}:
        raise ValueError("reconstruction must have shape [B, L, D] or [B, M, L, D]")
    if reconstruction.ndim == 3:
        reconstruction = reconstruction.unsqueeze(1)
    if reconstruction.shape[0] != input_scaled.shape[0] or reconstruction.shape[2:] != input_scaled.shape[1:]:
        raise ValueError("reconstruction batch, window, and feature shapes must match input_scaled")
    if not torch.isfinite(input_scaled.float()).all().item():
        raise ValueError("input_scaled must contain only finite values")
    if not torch.isfinite(reconstruction.float()).all().item():
        raise ValueError("reconstruction must contain only finite values")
    return reconstruction


def score_reconstruction(
    input_scaled: torch.Tensor,
    reconstruction: torch.Tensor,
    scaler: SequenceStandardScaler,
) -> dict[str, torch.Tensor]:
    """Return raw-input and normalized point/window reconstruction MSE.

    The MC path computes MSE for each reconstruction sample first, then averages
    those MSE values. It never computes MSE from the mean reconstruction.
    """
    reconstruction_samples = _validate_score_inputs(input_scaled, reconstruction)
    raw_input = scaler.inverse_transform_tensor(input_scaled)
    raw_reconstruction = scaler.inverse_transform_tensor(
        reconstruction_samples.reshape(-1, *reconstruction_samples.shape[2:])
    ).reshape_as(reconstruction_samples)

    normalized_point_mse_samples = (
        input_scaled.unsqueeze(1) - reconstruction_samples
    ).square().mean(dim=-1)
    raw_point_mse_samples = (
        raw_input.unsqueeze(1) - raw_reconstruction
    ).square().mean(dim=-1)
    scores = {
        "raw_input_point_mse": raw_point_mse_samples.mean(dim=1),
        "raw_input_window_mse": raw_point_mse_samples.mean(dim=(1, 2)),
        "normalized_input_point_mse": normalized_point_mse_samples.mean(dim=1),
        "normalized_input_window_mse": normalized_point_mse_samples.mean(dim=(1, 2)),
    }
    if not all(torch.isfinite(score.float()).all().item() for score in scores.values()):
        raise ValueError("reconstruction scores must contain only finite values")
    return scores
