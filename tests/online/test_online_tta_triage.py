from __future__ import annotations

import torch

from src.engine.online_tta.online_engine_window_metrics import (
    _update_online_window_buffers,
)
from src.engine.online_tta.triage import classify_online_window


class _CapturingVerificationBuffer:
    def __init__(self) -> None:
        self.entry = None

    def try_admit(self, entry: dict[str, object]) -> bool:
        self.entry = entry
        return True


def test_online_tta_triage_assigns_strong_anomaly_first() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.95,
            latent_window_score=0.95,
            thresholds=thresholds,
        )
        == "strong_anomaly"
    )


def test_online_tta_triage_assigns_hard_old_normality() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.3,
            latent_window_score=0.1,
            thresholds=thresholds,
        )
        == "hard_old_normality"
    )


def test_online_tta_triage_assigns_normal_and_gray_zone() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.2,
            latent_window_score=0.7,
            thresholds=thresholds,
        )
        == "normal"
    )
    assert (
        classify_online_window(
            input_window_score=0.5,
            latent_window_score=0.5,
            thresholds=thresholds,
        )
        == "gray_zone"
    )


def test_gray_zone_buffer_preserves_model_output_score_identity() -> None:
    buffer = _CapturingVerificationBuffer()
    batch = {
        "x": torch.zeros(1, 2, 1),
        "meta": [
            {
                "stream_step": 3,
                "start_index": 2,
                "end_index": 4,
                "entity_id": "machine-1-6",
            }
        ],
    }

    admitted, rejected = _update_online_window_buffers(
        batch_on_device=batch,
        raw_point_score=0.4,
        input_window_score=0.4,
        latent_window_score=0.5,
        triage_decision="gray_zone",
        verification_buffer=buffer,
        score_space="model_output",
    )

    assert (admitted, rejected) == (True, False)
    assert buffer.entry is not None
    assert buffer.entry["score_space"] == "model_output"
    assert buffer.entry["point_score_transform"] is None
