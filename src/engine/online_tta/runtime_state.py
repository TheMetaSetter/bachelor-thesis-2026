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
            "hard_old_intervals": [
                list(interval) for interval in self.hard_old_intervals
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OnlineRuntimeState":
        intervals = [
            tuple(int(value) for value in interval)
            for interval in payload.get("hard_old_intervals", [])
        ]
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
    signature_history: list[Any] | None = None,
) -> None:
    """Restore mutable containers after identity validation."""
    verification_buffer.clear()
    for entry in state.verification_entries:
        verification_buffer.add(entry)
    for interval in state.hard_old_intervals:
        hard_old_guard.add(interval)
    if signature_history is not None:
        from src.engine.online_tta.signature_verification import (
            signature_window_from_dict,
        )

        signature_history.clear()
        signature_history.extend(
            signature_window_from_dict(entry) for entry in state.signature_history
        )


def resume_online_runtime(
    *,
    checkpoint_manager: Any,
    checkpoint_path: str,
    model: Any,
    entity_id: str,
    online_variant: str,
    verification_buffer: Any,
    hard_old_guard: Any,
    signature_history: list[Any] | None = None,
) -> OnlineRuntimeState:
    """Load model state, validate identity, and restore mutable containers."""
    checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path, model, optimizer=None, scheduler=None
    )
    extra_state = checkpoint.get("extra_state", {})
    state_payload = extra_state.get("online_runtime_state")
    if state_payload is None:
        state_payload = {
            "entity_id": entity_id,
            "online_variant": extra_state.get("online_variant", online_variant),
            "threshold_artifact": extra_state.get("threshold_artifact", {}),
            "verification_entries": extra_state.get("verification_buffer_entries", []),
            "hard_old_intervals": extra_state.get("hard_old_guard_intervals", []),
        }
    state = validate_resume_state(state_payload, entity_id, online_variant)
    restore_online_runtime_state(
        state, verification_buffer, hard_old_guard, signature_history
    )
    return state
