from __future__ import annotations

from typing import Any

import numpy as np

from src.baselines.online import (
    CANDIStreamingBaseline,
    IForestStreamingBaseline,
    KMeansADStreamingBaseline,
    M2N2StreamingBaseline,
    StumpyChannelABStreamingBaseline,
)


def _build_sequence(entity_id: str, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    length = 32
    channels = 3
    return {
        "x": rng.normal(size=(length, channels)).astype(np.float64),
        "point_labels": np.zeros(length, dtype=np.int64),
        "mask": np.ones((length, channels), dtype=np.float64),
        "timestamps": np.arange(length, dtype=np.int64),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "sequence_length": length,
            "source_sequence_length": length,
        },
    }


def _protocol_config() -> dict[str, Any]:
    return {
        "window_size": 8,
        "offline_tail_policy": "end_align",
        "offline_threshold_split": "clean_validation",
        "offline_threshold_quantile": 0.9,
        "online_window_stride": 1,
        "online_threshold_split": "clean_validation",
        "online_threshold_quantile": 0.9,
        "online_ewma_current_weight": 0.9,
        "online_ewma_previous_weight": 0.1,
        "test_label_usage": "metrics_only",
        "point_adjustment": False,
    }


def test_online_streaming_baselines_calibrate_and_run() -> None:
    train_sequence = _build_sequence("machine-1-6", seed=0)["x"]
    clean_validation_sequences = [_build_sequence("machine-1-6", seed=1)]
    test_sequence = _build_sequence("machine-1-6", seed=2)
    protocol_config = _protocol_config()

    baselines = [
        CANDIStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="A0",
            seed=7,
        ),
        M2N2StreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="A0",
            seed=7,
        ),
        StumpyChannelABStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
        ),
        KMeansADStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
            n_clusters=2,
        ),
        IForestStreamingBaseline(
            train_sequence=train_sequence,
            window_size=8,
            online_variant="main",
            seed=7,
            n_estimators=10,
        ),
    ]

    for baseline in baselines:
        expected_variant = baseline.online_variant
        calibration = baseline.calibrate(
            clean_validation_sequences=clean_validation_sequences,
            protocol_config=protocol_config,
            device="cpu",
        )
        metric_history, records = baseline.run_sequence(
            sequence=test_sequence,
            threshold_value=float(calibration["threshold_value"]),
            protocol_config=protocol_config,
            device="cpu",
        )
        assert calibration["threshold_source"] == "clean_validation_stride1_ewma"
        assert metric_history
        assert records
        assert records[0]["online_variant"] == expected_variant
        for metric in metric_history:
            assert "online/verification_buffer_size" in metric
            assert "online/ttl_buffer_size" not in metric
            assert "online/raw_point_score" in metric
            assert "online/prediction" in metric
            assert "online/did_update" in metric
