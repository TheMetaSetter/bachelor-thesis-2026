from __future__ import annotations

from src.engine.online_tta.timing_debug import OnlineTtaTimingLogger


def _batch() -> dict[str, object]:
    return {
        "meta": [
            {
                "entity_id": "machine-1-6",
                "start_index": 10,
                "end_index": 30,
            }
        ]
    }


def test_timing_logger_is_silent_when_disabled(capsys) -> None:
    logger = OnlineTtaTimingLogger(enabled=False, device="cpu")
    logger.set_window(_batch())

    assert logger.measure("prepare_event", lambda: 7) == 7
    assert capsys.readouterr().out == ""


def test_timing_logger_prints_component_and_window_when_enabled(capsys) -> None:
    logger = OnlineTtaTimingLogger(enabled=True, device="cpu")
    logger.set_window(_batch())

    assert logger.measure("prepare_event", lambda: 7) == 7
    output = capsys.readouterr().out
    assert "[online-tta-timing]" in output
    assert "entity=machine-1-6" in output
    assert "window=[10,30)" in output
    assert "component=prepare_event" in output
    assert "elapsed_ms=" in output
