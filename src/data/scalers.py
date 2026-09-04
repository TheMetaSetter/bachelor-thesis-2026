from __future__ import annotations

from typing import Any

import torch

from src.core.contracts import validate_raw_sequence


class SequenceStandardScaler:
    def __init__(self, epsilon: float = 1.0e-3) -> None:
        self.epsilon = epsilon
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None
        self.feature_active_mask: torch.Tensor | None = None

    def fit(self, sequences: list[dict[str, Any]]) -> None:
        if not sequences:
            raise ValueError("Cannot fit scaler on empty sequence list")
        for sequence in sequences:
            validate_raw_sequence(sequence)

        stacked_training_points = torch.cat(
            [sequence["x"] for sequence in sequences], dim=0
        )
        self.feature_mean = stacked_training_points.mean(dim=0)
        raw_feature_std = stacked_training_points.std(dim=0, unbiased=False)
        self.feature_std = raw_feature_std
        self.feature_active_mask = raw_feature_std > 0.0

    def _resolve_active_feature_std(self) -> torch.Tensor:
        if self.feature_std is None:
            raise RuntimeError("Scaler must be fit before transform")
        return torch.clamp(self.feature_std, min=self.epsilon)

    def inverse_transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        """Restore scaled values to the original sensor-value space."""
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if (
            self.feature_mean is None
            or self.feature_std is None
            or self.feature_active_mask is None
        ):
            raise RuntimeError("Scaler must be fit before inverse transform")
        if values.ndim == 0 or values.shape[-1] != self.feature_mean.shape[0]:
            raise ValueError("values feature dimension must match fitted scaler")

        restored_values = values.clone()
        active_mask = self.feature_active_mask.to(device=values.device)
        if bool(active_mask.any()):
            feature_mean = self.feature_mean.to(
                device=values.device, dtype=values.dtype
            )
            active_feature_std = self._resolve_active_feature_std().to(
                device=values.device, dtype=values.dtype
            )
            restored_values[..., active_mask] = (
                values[..., active_mask] * active_feature_std[active_mask]
                + feature_mean[active_mask]
            )
        return restored_values

    def transform_sequence(self, sequence: dict[str, Any]) -> dict[str, Any]:
        validate_raw_sequence(sequence)
        if (
            self.feature_mean is None
            or self.feature_std is None
            or self.feature_active_mask is None
        ):
            raise RuntimeError("Scaler must be fit before transform")

        transformed_sequence = dict(sequence)
        transformed_x = sequence["x"].clone()
        active_mask = self.feature_active_mask.to(device=transformed_x.device)
        if bool(active_mask.any()):
            scaled_feature_std = self._resolve_active_feature_std().to(
                device=transformed_x.device
            )
            feature_mean = self.feature_mean.to(device=transformed_x.device)
            transformed_x[:, active_mask] = (
                sequence["x"][:, active_mask] - feature_mean[active_mask]
            ) / scaled_feature_std[active_mask]
        transformed_sequence["x"] = transformed_x
        transformed_sequence["meta"] = dict(sequence["meta"])
        return transformed_sequence

    def transform_sequences(
        self, sequences: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [self.transform_sequence(sequence) for sequence in sequences]

    def fit_transform_sequences(
        self, sequences: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.fit(sequences)
        return self.transform_sequences(sequences)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "feature_active_mask": self.feature_active_mask,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epsilon = float(state_dict["epsilon"])
        self.feature_mean = state_dict["feature_mean"]
        self.feature_std = state_dict["feature_std"]
        feature_active_mask = state_dict.get("feature_active_mask")
        if feature_active_mask is None:
            self.feature_active_mask = torch.ones_like(self.feature_std, dtype=torch.bool)
        else:
            self.feature_active_mask = feature_active_mask.to(dtype=torch.bool)
