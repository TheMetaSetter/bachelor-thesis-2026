from pathlib import Path

import pytest

from scripts.ops.preflight_full_benchmark_matrix import build_preflight_report


def test_full_benchmark_matrix_preflight_is_complete_and_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_checkpoint_path = tmp_path / "best.pt"
    fake_checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        "scripts.ops.preflight_full_benchmark_matrix.resolve_stage_b_checkpoint",
        lambda config: fake_checkpoint_path,
    )

    report = build_preflight_report()

    assert report["status"] == "ready"
    assert report["offline"] == {"thesis": 18, "redlamp": 9, "traditional": 27}
    assert report["online"] == {"thesis": 54, "baselines": 81}
    assert report["threshold_safety"] == {
        "offline_source": "clean_validation",
        "online_source": "clean_validation",
        "test_label_usage": "metrics_only",
        "point_adjustment": False,
    }
