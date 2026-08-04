"""Serializable state owned by one online entity stream."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clone_dict_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


@dataclass
class OnlineRuntimeState:
    """Keep entity-specific online state separate from model parameters.

    The hidden assumption here is one runtime object per entity stream. That is
    why the entity identity, threshold artifact identity, cursor, and EWMA
    history live together in one serializable payload.
    """

    entity_id: str
    online_variant: str
    threshold_artifact: dict[str, Any]
    stream_cursor: int = 0
    active_ewma_point_scores: dict[str, float] = field(default_factory=dict)
    verification_entries: list[dict[str, Any]] = field(default_factory=list)
    verification_history: list[dict[str, Any]] = field(default_factory=list)
    hard_old_intervals: list[tuple[int, int]] = field(default_factory=list)
    checkpoint_path: str = ""
    threshold_artifact_path: str = ""
    state_version: int = 2
    runtime_schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")
        if self.online_variant not in {"A0", "A1", "A2"}:
            raise ValueError("online_variant must be A0, A1, or A2")
        if not isinstance(self.threshold_artifact, dict):
            raise TypeError("threshold_artifact must be a mapping")
        artifact_entity = str(self.threshold_artifact.get("entity_id", ""))
        if artifact_entity != self.entity_id:
            raise ValueError("threshold artifact entity does not match runtime entity")
        if "thresholds" not in self.threshold_artifact:
            raise ValueError("threshold_artifact must contain thresholds")
        if int(self.stream_cursor) < 0:
            raise ValueError("stream_cursor must be non-negative")
        if int(self.state_version) != 2:
            raise ValueError("state_version must be 2")
        if int(self.runtime_schema_version) != 2:
            raise ValueError("runtime schema version must be 2")
        self.active_ewma_point_scores = {
            str(index): float(score)
            for index, score in self.active_ewma_point_scores.items()
        }
        if self.checkpoint_path and not isinstance(self.checkpoint_path, str):
            raise TypeError("checkpoint_path must be a string")
        if self.threshold_artifact_path and not isinstance(
            self.threshold_artifact_path, str
        ):
            raise TypeError("threshold_artifact_path must be a string")
        for interval in self.hard_old_intervals:
            if len(interval) != 2:
                raise ValueError("hard_old_intervals entries must be 2-tuples")
            start, end = (int(interval[0]), int(interval[1]))
            if end <= start:
                raise ValueError("hard_old_intervals must contain proper intervals")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "runtime_schema_version": self.runtime_schema_version,
            "entity_id": self.entity_id,
            "online_variant": self.online_variant,
            "threshold_artifact": self.threshold_artifact,
            "stream_cursor": int(self.stream_cursor),
            "active_ewma_point_scores": dict(self.active_ewma_point_scores),
            "verification_entries": _clone_dict_list(self.verification_entries),
            "verification_history": _clone_dict_list(self.verification_history),
            "hard_old_intervals": [
                [int(interval[0]), int(interval[1])]
                for interval in self.hard_old_intervals
            ],
            "checkpoint_path": self.checkpoint_path,
            "threshold_artifact_path": self.threshold_artifact_path,
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
            stream_cursor=int(payload.get("stream_cursor", 0)),
            active_ewma_point_scores=dict(payload.get("active_ewma_point_scores", {})),
            verification_entries=list(payload.get("verification_entries", [])),
            verification_history=list(payload.get("verification_history", [])),
            hard_old_intervals=intervals,
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            threshold_artifact_path=str(payload.get("threshold_artifact_path", "")),
            state_version=int(payload.get("state_version", 1)),
            runtime_schema_version=int(payload.get("runtime_schema_version", 1)),
        )

    def advance_cursor(self, step_count: int = 1) -> None:
        """Advance the causal cursor by the number of processed windows."""
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        self.stream_cursor += int(step_count)

    def replace_active_ewma_point_scores(self, scores: dict[int, float]) -> None:
        self.active_ewma_point_scores = {
            str(index): float(score) for index, score in scores.items()
        }

    def append_verification_entry(self, entry: dict[str, Any]) -> None:
        self.verification_entries.append(dict(entry))

    def append_verification_history(self, record: dict[str, Any]) -> None:
        self.verification_history.append(dict(record))

    def append_hard_old_interval(self, interval: tuple[int, int]) -> None:
        start, end = int(interval[0]), int(interval[1])
        if end <= start:
            raise ValueError("hard_old interval end must exceed start")
        self.hard_old_intervals.append((start, end))


def build_online_runtime_state(
    *,
    entity_id: str,
    online_variant: str,
    threshold_artifact: dict[str, Any],
    checkpoint_path: str = "",
    threshold_artifact_path: str = "",
) -> OnlineRuntimeState:
    """Construct a clean runtime state for a fresh online stream."""
    return OnlineRuntimeState(
        entity_id=entity_id,
        online_variant=online_variant,
        threshold_artifact=threshold_artifact,
        checkpoint_path=checkpoint_path,
        threshold_artifact_path=threshold_artifact_path,
    )


def validate_resume_state(
    payload: dict[str, Any],
    entity_id: str,
    online_variant: str,
    *,
    threshold_artifact: dict[str, Any] | None = None,
    runtime_schema_version: int = 2,
) -> OnlineRuntimeState:
    """Validate identity before any model or buffer mutation."""
    state = OnlineRuntimeState.from_dict(payload)
    if state.entity_id != entity_id:
        raise ValueError("checkpoint entity does not match requested entity")
    if state.online_variant != online_variant:
        raise ValueError("checkpoint online variant does not match requested variant")
    if state.runtime_schema_version != int(runtime_schema_version):
        raise ValueError("checkpoint runtime schema does not match requested schema")
    if (
        threshold_artifact is not None
        and state.threshold_artifact != threshold_artifact
    ):
        raise ValueError(
            "checkpoint threshold artifact does not match requested artifact"
        )
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


def resume_online_runtime(
    *,
    checkpoint_manager: Any,
    checkpoint_path: str,
    model: Any,
    entity_id: str,
    online_variant: str,
    verification_buffer: Any,
    hard_old_guard: Any,
) -> OnlineRuntimeState:
    """Load model state, validate identity, and restore mutable containers."""
    checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path, model, optimizer=None, scheduler=None
    )
    extra_state = checkpoint.get("extra_state", {})
    state_payload = extra_state.get("online_runtime_state")
    if state_payload is None:
        raise ValueError("legacy online runtime state is not supported by schema version 2")
    state = validate_resume_state(
        state_payload,
        entity_id,
        online_variant,
        threshold_artifact=extra_state.get("threshold_artifact"),
        runtime_schema_version=2,
    )
    restore_online_runtime_state(
        state,
        verification_buffer,
        hard_old_guard,
    )
    return state
