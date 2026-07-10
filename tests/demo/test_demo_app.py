from __future__ import annotations

from pathlib import Path

from demo.app import run_demo


def test_run_demo_uses_replay_helpers(tmp_path: Path, monkeypatch) -> None:
    offline_state = object()
    online_state = object()
    calls: list[str] = []

    monkeypatch.setattr(
        "demo.app.build_offline_replay_state",
        lambda path: calls.append(f"offline:{path}") or offline_state,
    )
    monkeypatch.setattr(
        "demo.app.build_online_replay_state",
        lambda path: calls.append(f"online:{path}") or online_state,
    )
    monkeypatch.setattr(
        "demo.app.plot_offline_replay",
        lambda state, output_path: calls.append("plot_offline") or Path(output_path),
    )
    monkeypatch.setattr(
        "demo.app.plot_online_replay",
        lambda state, output_path: calls.append("plot_online") or Path(output_path),
    )

    outputs = run_demo(
        offline_report_path="offline.json",
        online_report_path="online.json",
        output_dir=str(tmp_path / "demo"),
    )

    assert outputs["offline_replay"].endswith("offline_replay.png")
    assert outputs["online_replay"].endswith("online_replay.png")
    assert calls == [
        "offline:offline.json",
        "plot_offline",
        "online:online.json",
        "plot_online",
    ]
