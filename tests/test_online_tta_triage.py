from __future__ import annotations

from src.engine.online_tta.triage import classify_online_window


def test_online_tta_triage_assigns_strong_anomaly_first() -> None:
    thresholds = {
        "strong_anomaly_threshold": 0.8,
        "pnn_candidate_input_threshold": 0.4,
        "pnn_candidate_latent_threshold": 0.6,
        "hard_old_normality_threshold": 0.2,
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
        "strong_anomaly_threshold": 0.8,
        "pnn_candidate_input_threshold": 0.4,
        "pnn_candidate_latent_threshold": 0.6,
        "hard_old_normality_threshold": 0.2,
    }

    assert (
        classify_online_window(
            input_window_score=0.1,
            latent_window_score=0.1,
            thresholds=thresholds,
        )
        == "hard_old_normality"
    )


def test_online_tta_triage_assigns_pnn_candidate_and_gray_zone() -> None:
    thresholds = {
        "strong_anomaly_threshold": 0.8,
        "pnn_candidate_input_threshold": 0.4,
        "pnn_candidate_latent_threshold": 0.6,
        "hard_old_normality_threshold": 0.2,
    }

    assert (
        classify_online_window(
            input_window_score=0.3,
            latent_window_score=0.7,
            thresholds=thresholds,
        )
        == "pnn_candidate"
    )
    assert (
        classify_online_window(
            input_window_score=0.5,
            latent_window_score=0.5,
            thresholds=thresholds,
        )
        == "gray_zone"
    )
