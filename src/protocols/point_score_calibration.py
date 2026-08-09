from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


POINT_SCORE_TRANSFORM_NAME = "shifted-and-scaled logistic sigmoid"
POINT_SCORE_TAU_ESTIMATOR = "mad_based_robust_scale"
POINT_SCORE_MAD_NORMALIZER = 0.6745


@dataclass(frozen=True)
class PointScoreCalibration:
    """Immutable calibration state for the official THESIS point score."""

    center: float
    tau: float
    transform_name: str = POINT_SCORE_TRANSFORM_NAME
    tau_estimator: str = POINT_SCORE_TAU_ESTIMATOR
    mad_normalizer: float = POINT_SCORE_MAD_NORMALIZER

    def __post_init__(self) -> None:
        if not np.isfinite(self.center):
            raise ValueError("point score calibration center must be finite")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("point score calibration tau must be finite and positive")
        if self.transform_name != POINT_SCORE_TRANSFORM_NAME:
            raise ValueError("unsupported point score transform")
        if self.tau_estimator != POINT_SCORE_TAU_ESTIMATOR:
            raise ValueError("unsupported point score tau estimator")
        if float(self.mad_normalizer) != POINT_SCORE_MAD_NORMALIZER:
            raise ValueError("point score MAD normalizer must be 0.6745")

    def to_artifact_fields(self) -> dict[str, Any]:
        return {
            "point_score_transform": self.transform_name,
            "point_score_c": float(self.center),
            "point_score_tau": float(self.tau),
            "point_score_tau_estimator": self.tau_estimator,
            "point_score_mad_normalizer": float(self.mad_normalizer),
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "PointScoreCalibration":
        return cls(
            center=float(artifact["point_score_c"]),
            tau=float(artifact["point_score_tau"]),
            transform_name=str(artifact["point_score_transform"]),
            tau_estimator=str(artifact["point_score_tau_estimator"]),
            mad_normalizer=float(artifact["point_score_mad_normalizer"]),
        )


def _as_finite_numpy_scores(raw_point_mse: Any) -> np.ndarray:
    score_array = np.asarray(
        raw_point_mse.detach().cpu().numpy()
        if isinstance(raw_point_mse, torch.Tensor)
        else raw_point_mse,
        dtype=np.float64,
    )
    if score_array.size == 0:
        raise ValueError("raw point MSE must not be empty")
    if not np.isfinite(score_array).all():
        raise ValueError("raw point MSE must contain only finite values")
    return score_array


def fit_mad_logistic_calibration(raw_point_mse: Any) -> PointScoreCalibration:
    """Fit c and tau from one entity's raw clean-validation point MSE."""
    score_array = _as_finite_numpy_scores(raw_point_mse)
    center = float(np.median(score_array))
    mad = float(np.median(np.abs(score_array - center)))
    tau = mad / POINT_SCORE_MAD_NORMALIZER
    return PointScoreCalibration(center=center, tau=tau)


def transform_point_scores(
    raw_point_mse: Any,
    calibration: PointScoreCalibration,
) -> np.ndarray | torch.Tensor:
    """Apply q = sigmoid((e - c) / tau) without changing the input shape."""
    if not isinstance(calibration, PointScoreCalibration):
        raise TypeError("calibration must be a PointScoreCalibration")
    if isinstance(raw_point_mse, torch.Tensor):
        output_dtype = (
            raw_point_mse.dtype
            if raw_point_mse.is_floating_point()
            else torch.get_default_dtype()
        )
        score_tensor = raw_point_mse.to(dtype=output_dtype)
        if score_tensor.numel() == 0:
            raise ValueError("raw point MSE must not be empty")
        if not bool(torch.isfinite(score_tensor).all()):
            raise ValueError("raw point MSE must contain only finite values")
        return torch.sigmoid(
            (score_tensor - calibration.center) / calibration.tau
        )
    score_array = _as_finite_numpy_scores(raw_point_mse)
    transformed = 1.0 / (
        1.0 + np.exp(-np.clip((score_array - calibration.center) / calibration.tau, -700.0, 700.0))
    )
    return transformed
