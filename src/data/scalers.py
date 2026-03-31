from __future__ import annotations

from typing import Any

import torch

from src.core.contracts import validate_raw_sequence


class SequenceStandardScaler:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.feature_mean: torch.Tensor | None = None
        self.feature_std: torch.Tensor | None = None

    def fit(self, sequences: list[dict[str, Any]]) -> None:
        if not sequences:
            raise ValueError("Cannot fit scaler on empty sequence list")
        for sequence in sequences:
            validate_raw_sequence(sequence)

        stacked_training_points = torch.cat([sequence["x"] for sequence in sequences], dim=0)
        self.feature_mean = stacked_training_points.mean(dim=0)
        raw_feature_std = stacked_training_points.std(dim=0, unbiased=False)
        self.feature_std = torch.clamp(raw_feature_std, min=self.epsilon)

    def transform_sequence(self, sequence: dict[str, Any]) -> dict[str, Any]:
        validate_raw_sequence(sequence)
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Scaler must be fit before transform")

        transformed_sequence = dict(sequence)
        transformed_sequence["x"] = (sequence["x"] - self.feature_mean) / self.feature_std
        transformed_sequence["meta"] = dict(sequence["meta"])
        return transformed_sequence

    def transform_sequences(self, sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.transform_sequence(sequence) for sequence in sequences]

    def state_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.epsilon = float(state_dict["epsilon"])
        self.feature_mean = state_dict["feature_mean"]
        self.feature_std = state_dict["feature_std"]

