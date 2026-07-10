"""Verification-cycle orchestration separated from stream scoring."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.engine.online_tta.verification_buffer import VerificationBuffer


class VerificationCycleController:
    """Run a callback once for a full verification buffer."""

    def __init__(self, buffer: VerificationBuffer, capacity: int = 8) -> None:
        self.buffer = buffer
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("verification capacity must be positive")

    def maybe_run(
        self, verify: Callable[[dict[str, Any]], bool]
    ) -> dict[str, int] | None:
        if len(self.buffer) < self.capacity or not self.buffer.should_verify():
            return None
        entries = self.buffer.items()
        for entry in entries:
            self.buffer.mark_verification_result(
                str(entry["entry_id"]), bool(verify(dict(entry)))
            )
        return self.buffer.finish_verification_cycle()
