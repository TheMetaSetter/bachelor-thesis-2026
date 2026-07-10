from __future__ import annotations

from src.engine.online_tta.ttl_buffer import TTLBuffer
from src.engine.online_tta.verification_buffer import VerificationBuffer


def test_verification_buffer_rejects_overlapping_windows() -> None:
    buffer = VerificationBuffer(max_size=4, non_overlap_gap=0)
    assert buffer.admit(window_start=0, window_end=20)
    buffer.add({"window_start": 0, "window_end": 20})
    assert not buffer.admit(window_start=10, window_end=30)
    assert buffer.admit(window_start=20, window_end=40)


def test_ttl_buffer_expires_old_items() -> None:
    buffer = TTLBuffer(ttl_steps=2)
    buffer.add("window-a", current_step=1)
    buffer.add("window-b", current_step=2)

    assert buffer.contains("window-a")
    assert buffer.contains("window-b")

    buffer.expire(current_step=4)

    assert not buffer.contains("window-a")
    assert not buffer.contains("window-b")
