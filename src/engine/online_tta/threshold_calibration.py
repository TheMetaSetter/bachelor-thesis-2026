"""Pure score and threshold calculations for entity-scoped online calibration."""
from __future__ import annotations

from collections.abc import Sequence
import torch


def compute_input_window_score(reconstruction: torch.Tensor, input_window: torch.Tensor) -> torch.Tensor:
    if reconstruction.shape != input_window.shape or reconstruction.ndim != 3:
        raise ValueError("reconstruction and input_window must both have shape [B, L, C]")
    return (reconstruction - input_window).square().mean(dim=(1, 2))


def compute_latent_window_score(model_outputs: dict[str, object]) -> torch.Tensor:
    for key in ("latent_window_score", "memory_score", "latent_score"):
        value = model_outputs.get(key)
        if isinstance(value, torch.Tensor):
            return value if value.ndim == 1 else value.reshape(value.shape[0], -1).mean(dim=1)
    raise KeyError("online model outputs must expose an explicit latent or memory score")


def aggregate_endpoint_ewma(current: float, previous: float | None, current_weight: float, previous_weight: float) -> float:
    if previous is None:
        return float(current)
    total = current_weight + previous_weight
    if total <= 0:
        raise ValueError("EWMA weights must have a positive sum")
    return float((current * current_weight + previous * previous_weight) / total)


def quantile_threshold(scores: torch.Tensor | Sequence[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    values = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if values.numel() == 0:
        raise ValueError("cannot compute a threshold from empty scores")
    return float(torch.quantile(values, quantile).item())
