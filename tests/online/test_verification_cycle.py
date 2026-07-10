from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController


def test_verification_cycle_runs_at_capacity_and_ticks_ttl_once() -> None:
    buffer = VerificationBuffer(verification_capacity=2)
    buffer.try_admit({"entry_id": "a", "window_start": 0, "window_end": 2})
    buffer.try_admit({"entry_id": "b", "window_start": 3, "window_end": 5})
    result = VerificationCycleController(buffer, capacity=2).maybe_run(lambda _: False)
    assert result == {"remaining": 2, "removed": 0}
    assert buffer.items()[0]["ttl_remaining"] == 1
