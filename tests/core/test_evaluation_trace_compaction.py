from __future__ import annotations

from src.core.evaluation_trace_compaction import compact_evaluation_trace_payload


def test_compact_evaluation_trace_payload_keeps_uq_summary_and_drops_raw_samples() -> (
    None
):
    trace_payload = {
        "batch_index": 7,
        "entity_ids": ["machine-3-9"],
        "point_score_summary": {"mean": 0.12},
        "window_score_summary": {"mean": 0.34},
        "point_score_history": [0.1, 0.2],
        "window_score_history": [0.3, 0.4],
        "uncertainty_history": {
            "point_anomaly_score_variance": [0.5, 0.6],
            "window_anomaly_score_variance": [0.7],
        },
        "sample_retention_policy": "retain_for_eda",
        "deterministic_geometry": {"hidden_reconstruction": [1.0]},
        "stochastic_query": {
            "schema_version": 3,
            "enabled": True,
            "num_samples": 10,
            "continuous_temperature": 0.9,
            "discrete_temperature": 0.8,
            "return_mc_samples": False,
            "sample_retention_policy": "none",
            "point_score_samples": [1.0, 2.0],
            "window_score_samples": [3.0, 4.0],
            "reconstruction_samples": [[[5.0]]],
        },
        "mc_sample_histories": {
            "point_score_samples": [1.0, 2.0],
            "window_score_samples": [3.0, 4.0],
        },
    }

    compacted = compact_evaluation_trace_payload(trace_payload)

    assert compacted["batch_index"] == 7
    assert compacted["entity_ids"] == ["machine-3-9"]
    assert compacted["point_score_history"] == [0.1, 0.2]
    assert compacted["window_score_history"] == [0.3, 0.4]
    assert compacted["uncertainty_history"] == {
        "point_anomaly_score_variance": [0.5, 0.6],
        "window_anomaly_score_variance": [0.7],
    }
    assert compacted["sample_retention_policy"] == "retain_for_eda"
    assert compacted["stochastic_query"] == {
        "schema_version": 3,
        "enabled": True,
        "num_samples": 10,
        "continuous_temperature": 0.9,
        "discrete_temperature": 0.8,
        "return_mc_samples": False,
        "sample_retention_policy": "none",
    }
    assert "deterministic_geometry" not in compacted
    assert "mc_sample_histories" not in compacted
