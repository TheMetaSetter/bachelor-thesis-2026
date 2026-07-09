from __future__ import annotations

"""Lightweight online M2N2-style streaming baseline."""

from typing import Any

import numpy as np

from src.baselines.online.adaptive import AdaptiveStreamingBaselineBase


class M2N2StreamingBaseline(AdaptiveStreamingBaselineBase):
    method_name = "m2n2"

    def __init__(
        self,
        *,
        train_sequence: np.ndarray,
        window_size: int = 20,
        threshold_quantile: float = 0.99,
        online_variant: str = "A0",
        seed: int = 0,
        adaptation_momentum: float = 0.01,
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
        del triage_decision
        return (
            raw_point_score <= threshold_value and ewma_point_score <= threshold_value
        )

    def _method_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method_name,
            "policy": "update_on_non_anomalous_windows",
            "adaptation_momentum": self.adaptation_momentum,
            "online_variant": self.online_variant,
        }
