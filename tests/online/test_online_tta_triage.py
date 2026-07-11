from __future__ import annotations

from src.engine.online_tta.triage import classify_online_window


def test_online_tta_triage_assigns_strong_anomaly_first() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.95,
            latent_window_score=0.95,
            thresholds=thresholds,
        )
        == "strong_anomaly"
    )


def test_online_tta_triage_assigns_hard_old_normality() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.3,
            latent_window_score=0.1,
            thresholds=thresholds,
        )
        == "hard_old_normality"
    )


def test_online_tta_triage_assigns_normal_and_gray_zone() -> None:
    thresholds = {
        "input_window_threshold": 0.2,
        "latent_window_low_threshold": 0.4,
        "latent_window_high_threshold": 0.8,
    }

    assert (
        classify_online_window(
            input_window_score=0.2,
            latent_window_score=0.7,
            thresholds=thresholds,
        )
        == "normal"
    )
    assert (
        classify_online_window(
            input_window_score=0.5,
            latent_window_score=0.5,
            thresholds=thresholds,
        )
        == "gray_zone"
    )
