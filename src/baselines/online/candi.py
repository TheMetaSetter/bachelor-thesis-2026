from __future__ import annotations

"""Lightweight online CANDI-style streaming baseline.

The goal here is not to re-implement the whole external codebase. The goal is
to keep the benchmark contract clear: score stride-1 windows, calibrate on clean
validation, and adapt only on the windows the CANDI policy would consider safe
enough.
"""

from typing import Any

import numpy as np

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase


class CANDIStreamingBaseline(AdaptiveStreamingBaselineBase):
    method_name = "candi"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
        adaptation_momentum: float = 0.02,
    ) -> None:
        super().__init__(
            train_sequence=train_sequence,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            online_variant=online_variant,
            seed=seed,
            adaptation_momentum=adaptation_momentum,
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
        }
