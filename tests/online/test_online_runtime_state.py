import pytest

from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.runtime_state import (
    OnlineRuntimeState,
    resume_online_runtime,
    restore_online_runtime_state,
    validate_resume_state,
)
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.signature_verification import SignatureWindow
from src.engine.online_tta.verification_adapter import VerificationResult
from src.engine.online_tta.verification_cycle import VerificationCycleController
from src.engine.online_tta.triage import classify_online_window
import torch


def _artifact(entity_id: str) -> dict[str, object]:
    return {"entity_id": entity_id, "thresholds": {"online_ewma_point": {"value": 1.0}}}


def test_runtime_state_roundtrip_validates_identity() -> None:
    state = OnlineRuntimeState(
        "machine-1-6",
        "A2",
        _artifact("machine-1-6"),
        stream_cursor=7,
        previous_ewma_score=2.5,
        signature_history=[
            {
                "entity_id": "machine-1-6",
                "start": 0,
                "end": 2,
                "signatures": [[[0, 1, 2]]],
            }
        ],
        recurrent_signatures=[{"signature": [0, 1, 2]}],
        verification_history=[{"step": 1, "decision": "gray_zone"}],
        checkpoint_path="/tmp/online_final.pt",
        threshold_artifact_path="/tmp/thresholds/machine-1-6.json",
    )
    restored = validate_resume_state(state.to_dict(), "machine-1-6", "A2")
    assert restored.to_dict() == state.to_dict()
    assert restored.stream_cursor == 7
    assert restored.previous_ewma_score == 2.5
    assert restored.recurrent_signatures == [{"signature": [0, 1, 2]}]
    assert restored.verification_history == [{"step": 1, "decision": "gray_zone"}]


def test_runtime_state_rejects_wrong_entity_before_restore() -> None:
    state = OnlineRuntimeState("machine-1-6", "A0", _artifact("machine-1-6"))
    with pytest.raises(ValueError, match="entity"):
        validate_resume_state(state.to_dict(), "machine-3-4", "A0")


def test_runtime_state_restores_live_containers() -> None:
    state = OnlineRuntimeState(
        "machine-1-6",
        "A2",
        _artifact("machine-1-6"),
        verification_entries=[{"entry_id": "e1", "window_start": 0, "window_end": 2}],
        hard_old_intervals=[(4, 6)],
    )
    buffer = VerificationBuffer()
    guard = NonOverlapGuard()
    restore_online_runtime_state(state, buffer, guard)
    assert len(buffer) == 1
    assert guard.intervals() == [(4, 6)]


def test_resume_online_runtime_ignores_obsolete_ttl_buffer_size() -> None:
    state = OnlineRuntimeState(
        "machine-1-6",
        "A2",
        _artifact("machine-1-6"),
        verification_entries=[{"entry_id": "e1", "window_start": 0, "window_end": 2}],
    )

    class _CheckpointManager:
        def load_checkpoint(self, checkpoint_path, model, optimizer, scheduler):
            return {
                "extra_state": {
                    "online_runtime_state": state.to_dict(),
                    "threshold_artifact": _artifact("machine-1-6"),
                    "ttl_buffer_size": 999,
                }
            }

    buffer = VerificationBuffer()
    guard = NonOverlapGuard()
    resumed = resume_online_runtime(
        checkpoint_manager=_CheckpointManager(),
        checkpoint_path="/tmp/old_online_final.pt",
        model=object(),
        entity_id="machine-1-6",
        online_variant="A2",
        verification_buffer=buffer,
        hard_old_guard=guard,
    )

    assert resumed.to_dict() == state.to_dict()
    assert len(buffer) == 1
    assert buffer.items()[0]["entry_id"] == "e1"
    assert buffer.items()[0]["ttl_remaining"] == 2


def test_runtime_state_restores_signature_history() -> None:
    state = OnlineRuntimeState(
        "machine-1-6",
        "A2",
        _artifact("machine-1-6"),
        signature_history=[
            {
                "entity_id": "machine-1-6",
                "start": 0,
                "end": 2,
                "signatures": [[[0, 1, 2], [1, 2, 3]]],
            }
        ],
    )
    history: list[SignatureWindow] = []
    restore_online_runtime_state(
        state,
        VerificationBuffer(),
        NonOverlapGuard(),
        history,
        recurrent_signatures=[],
        verification_history=[],
    )
    assert history[0].signatures == [[(0, 1, 2), (1, 2, 3)]]


def test_runtime_state_rejects_schema_mismatch() -> None:
    state = OnlineRuntimeState("machine-1-6", "A2", _artifact("machine-1-6"))
    payload = state.to_dict()
    payload["runtime_schema_version"] = 2
    with pytest.raises(ValueError, match="runtime schema"):
        validate_resume_state(payload, "machine-1-6", "A2")


def test_resumed_next_event_matches_uninterrupted_execution() -> None:
    entries = [
        {"entry_id": f"e{i}", "window_start": i * 3, "window_end": i * 3 + 2}
        for i in range(7)
    ]
    state = OnlineRuntimeState(
        "machine-1-6", "A2", _artifact("machine-1-6"), verification_entries=entries
    )
    original, resumed = VerificationBuffer(), VerificationBuffer()
    original_guard, resumed_guard = NonOverlapGuard(), NonOverlapGuard()
    restore_online_runtime_state(state, original, original_guard)
    restore_online_runtime_state(
        OnlineRuntimeState.from_dict(state.to_dict()), resumed, resumed_guard
    )
    next_entry = {"entry_id": "e7", "window_start": 21, "window_end": 23}
    original.try_admit(next_entry)
    resumed.try_admit(next_entry)

    def unresolved(current_entries):
        return {
            entry["entry_id"]: VerificationResult(
                False, 0, "unresolved", torch.zeros(1, 2, dtype=torch.bool)
            )
            for entry in current_entries
        }

    first = VerificationCycleController(original).maybe_run(unresolved)
    second = VerificationCycleController(resumed).maybe_run(unresolved)
    thresholds = {
        "input_window_threshold": 1.0,
        "latent_window_low_threshold": 0.5,
        "latent_window_high_threshold": 0.9,
    }
    assert first == second and original.items() == resumed.items()
    assert classify_online_window(1.2, 0.7, thresholds) == "gray_zone"
