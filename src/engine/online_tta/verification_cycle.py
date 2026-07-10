"""Verification-cycle orchestration separated from stream scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_adapter import VerificationResult


class VerificationCycleController:
    """Run a callback once for a full verification buffer."""

    def __init__(self, buffer: VerificationBuffer, capacity: int = 8) -> None:
        self.buffer = buffer
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("verification capacity must be positive")

    def maybe_run(
        self,
        verify: Callable[[list[dict[str, Any]]], dict[str, VerificationResult]],
    ) -> dict[str, int] | None:
        if len(self.buffer) < self.capacity or not self.buffer.should_verify():
            return None
        entries = self.buffer.items()
        results = verify(entries)
        for entry in entries:
            entry_id = str(entry["entry_id"])
            if entry_id not in results:
                raise ValueError(f"verification result missing entry {entry_id}")
            self.buffer.mark_verification_result(
                entry_id, bool(results[entry_id].adapted)
            )
        return self.buffer.finish_verification_cycle()
