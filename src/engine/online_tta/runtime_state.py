"""Serializable state owned by one online entity stream."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OnlineRuntimeState:
    """Keep entity-specific online state separate from model parameters."""

    entity_id: str
    online_variant: str
    threshold_artifact: dict[str, Any]
    signature_history: list[dict[str, Any]] = field(default_factory=list)
    verification_entries: list[dict[str, Any]] = field(default_factory=list)
    hard_old_intervals: list[tuple[int, int]] = field(default_factory=list)
    state_version: int = 1

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")
        if self.online_variant not in {"A0", "A1", "A2"}:
            raise ValueError("online_variant must be A0, A1, or A2")
        artifact_entity = str(self.threshold_artifact.get("entity_id", ""))
        if artifact_entity != self.entity_id:
            raise ValueError("threshold artifact entity does not match runtime entity")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "entity_id": self.entity_id,
            "online_variant": self.online_variant,
            "threshold_artifact": self.threshold_artifact,
            "signature_history": list(self.signature_history),
            "verification_entries": list(self.verification_entries),
            "hard_old_intervals": [list(interval) for interval in self.hard_old_intervals],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OnlineRuntimeState":
        intervals = [tuple(int(value) for value in interval) for interval in payload.get("hard_old_intervals", [])]
        return cls(
            entity_id=str(payload["entity_id"]),
            online_variant=str(payload["online_variant"]),
            threshold_artifact=dict(payload["threshold_artifact"]),
            signature_history=list(payload.get("signature_history", [])),
            verification_entries=list(payload.get("verification_entries", [])),
            hard_old_intervals=intervals,
            state_version=int(payload.get("state_version", 1)),
        )


def validate_resume_state(
    payload: dict[str, Any], entity_id: str, online_variant: str
) -> OnlineRuntimeState:
    """Validate identity before any model or buffer mutation."""
    state = OnlineRuntimeState.from_dict(payload)
    if state.entity_id != entity_id:
        raise ValueError("checkpoint entity does not match requested entity")
    if state.online_variant != online_variant:
        raise ValueError("checkpoint online variant does not match requested variant")
    return state


def restore_online_runtime_state(
    state: OnlineRuntimeState,
    verification_buffer: Any,
    hard_old_guard: Any,
) -> None:
    """Restore mutable containers after identity validation."""
    verification_buffer.clear()
    for entry in state.verification_entries:
        verification_buffer.add(entry)
    for interval in state.hard_old_intervals:
        hard_old_guard.add(interval)
