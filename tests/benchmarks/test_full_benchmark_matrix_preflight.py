from scripts.ops.preflight_full_benchmark_matrix import build_preflight_report


def test_full_benchmark_matrix_preflight_is_complete_and_safe() -> None:
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
