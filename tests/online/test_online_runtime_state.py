import pytest

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.runtime_state import (
    OnlineRuntimeState,
    restore_online_runtime_state,
    validate_resume_state,
)
from src.engine.online_tta.verification_buffer import VerificationBuffer


def _artifact(entity_id: str) -> dict[str, object]:
    return {"entity_id": entity_id, "thresholds": {"online_ewma_point": {"value": 1.0}}}


def test_runtime_state_roundtrip_validates_identity() -> None:
    state = OnlineRuntimeState("machine-1-6", "A2", _artifact("machine-1-6"))
    restored = validate_resume_state(state.to_dict(), "machine-1-6", "A2")
    assert restored.to_dict() == state.to_dict()


def test_runtime_state_rejects_wrong_entity_before_restore() -> None:
    state = OnlineRuntimeState("machine-1-6", "A0", _artifact("machine-1-6"))
    with pytest.raises(ValueError, match="entity"):
        validate_resume_state(state.to_dict(), "machine-3-4", "A0")


def test_runtime_state_restores_live_containers() -> None:
    state = OnlineRuntimeState(
        "machine-1-6", "A2", _artifact("machine-1-6"),
        verification_entries=[{"entry_id": "e1", "window_start": 0, "window_end": 2}],
        hard_old_intervals=[(4, 6)],
    )
    buffer = VerificationBuffer()
    guard = NonOverlapGuard()
    restore_online_runtime_state(state, buffer, guard)
    assert len(buffer) == 1
    assert guard.intervals() == [(4, 6)]
