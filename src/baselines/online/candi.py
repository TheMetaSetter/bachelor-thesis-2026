from __future__ import annotations

"""Lightweight online CANDI-style streaming baseline.

The goal here is not to re-implement the whole external codebase. The goal is
to keep the benchmark contract clear: score stride-1 windows, calibrate on clean
validation, and adapt only on the windows the CANDI policy would consider safe
enough.
"""

from typing import Any
from pathlib import Path

import numpy as np

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase


class CANDIStreamingBaseline(AdaptiveStreamingBaselineBase):
    method_name = "candi"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        input_dim: int | None = None,
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "main",
        seed: int = 0,
        adaptation_momentum: float = 0.02,
        encoder_family: str = "cnn_simple",
        encoder_dim: int = 128,
        cnn_num_layers: int = 3,
        cnn_kernel_size: int = 3,
        cnn_hidden_channels: int = 64,
        cnn_dropout: float = 0.1,
        pretrained_encoder_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__(
            train_sequence=train_sequence,
            input_dim=input_dim,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
            adaptation_momentum=adaptation_momentum,
            encoder_family=encoder_family,
            encoder_dim=encoder_dim,
            cnn_num_layers=cnn_num_layers,
            cnn_kernel_size=cnn_kernel_size,
            cnn_hidden_channels=cnn_hidden_channels,
            cnn_dropout=cnn_dropout,
            pretrained_encoder_checkpoint=pretrained_encoder_checkpoint,
        )

    def _should_update(
        self,
        *,
        triage_decision: str,
        raw_point_score: float,
        ewma_point_score: float,
        threshold_value: float,
    ) -> bool:
        del raw_point_score, ewma_point_score, threshold_value
        return triage_decision in {"gray_zone", "pnn_candidate"}

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "policy": "update_on_gray_zone_and_pnn_candidate",
            "adaptation_momentum": self.adaptation_momentum,
            "online_variant": self.online_variant,
            "seed": self.seed,
            "window_size": self.window_size,
            "threshold_quantile": self.threshold_quantile,
            "update_policy": "adaptive_reference_statistics",
            **self._backbone_metadata(),
        }
