from __future__ import annotations

import pytest

from src.engine.online_tta.online_engine_window_metrics import (
    _update_online_window_buffers,
)
from src.engine.online_tta.verification_buffer import VerificationBuffer


def test_verification_buffer_rejects_overlapping_windows() -> None:
    buffer = VerificationBuffer(max_size=4, non_overlap_gap=0)
    assert buffer.admit(window_start=0, window_end=20)
    buffer.add({"window_start": 0, "window_end": 20})
    assert not buffer.admit(window_start=10, window_end=30)
    assert buffer.admit(window_start=20, window_end=40)


def _verification_batch() -> dict[str, object]:
    import torch

    return {
        "x": torch.zeros(1, 20, 2),
        "meta": [
            {
                "stream_step": 3,
                "start_index": 10,
                "end_index": 30,
                "entity_id": "machine-1-6",
            }
        ],
    }


def test_gray_zone_admission_uses_canonical_verification_ttl() -> None:
    buffer = VerificationBuffer(max_size=4, non_overlap_gap=0)

    admitted, rejected = _update_online_window_buffers(
        batch_on_device=_verification_batch(),
        raw_point_score=1.0,
        input_window_score=1.0,
        latent_window_score=1.0,
        triage_decision="gray_zone",
        verification_buffer=buffer,
    )

    assert (admitted, rejected) == (True, False)
    assert buffer.items()[0]["ttl_remaining"] == 2


@pytest.mark.parametrize(
    "triage_decision", ["normal", "hard_old_normality", "strong_anomaly"]
)
def test_non_gray_decisions_do_not_admit_verification_entries(
    triage_decision: str,
) -> None:
    buffer = VerificationBuffer(max_size=4, non_overlap_gap=0)

    admitted, rejected = _update_online_window_buffers(
        batch_on_device=_verification_batch(),
        raw_point_score=1.0,
        input_window_score=1.0,
        latent_window_score=1.0,
        triage_decision=triage_decision,
        verification_buffer=buffer,
    )

    assert (admitted, rejected) == (False, False)
    assert len(buffer) == 0
