from __future__ import annotations

import pytest

from src.core.uq_summary import build_uq_summary_payload, validate_uq_summary_payload


def test_build_uq_summary_payload_compacts_split_statistics() -> None:
    payload = build_uq_summary_payload(
        benchmark_kind="offline",
        experiment_name="pytest-thesis-offline",
        method_name="THESIS",
        variant_name="O0",
        entity_id="machine_1_6",
        seed=6,
        stage_name="stage_b_fusion_finetuning",
        checkpoint_path="/tmp/best.pt",
        checkpoint_sha256="abc123",
        experiment_config_path="/tmp/experiment.yaml",
        protocol_config_path="/tmp/protocol.yaml",
        output_dir="/tmp/output",
        run_scalar_logs={
            "query/continuous_temperature": 0.9,
            "query/discrete_temperature": 0.9,
            "query/num_samples_train": 10,
            "query/num_samples_eval": 10,
            "query/continuous_weight_entropy_mean": None,
            "query/discrete_topk_weight_entropy_mean": None,
        },
        split_inputs={
            "clean_validation": {
                "point_scores": [0.1, 0.2, 0.3],
                "traces": [
                    {
                        "sample_retention_policy": "retain_for_eda",
                        "window_score_history": [0.4, 0.5],
                        "uncertainty_history": {
                            "point_anomaly_score_variance": [0.1, 0.2],
                            "window_anomaly_score_variance": [0.3],
                            "classification_probability_variance": [0.4, 0.5],
                            "classification_variance_mean": [0.6],
                        },
                        "mc_sample_histories": {
                            "point_score_samples": [1.0],
                            "window_score_samples": [1.0],
                            "reconstruction_samples": [1.0],
                            "classification_probability_samples": [1.0],
                        },
                    }
                ],
            },
            "synthetic_validation": {
                "point_scores": [0.6, 0.7],
                "traces": [],
            },
            "test": {
                "point_scores": [0.8, 0.9],
                "traces": [
                    {
                        "sample_retention_policy": "retain_for_eda",
                        "window_score_history": [0.2, 0.3],
                        "uncertainty_history": {
                            "point_anomaly_score_variance": [0.7],
                            "window_anomaly_score_variance": [0.8],
                            "reconstruction_variance_full": [[0.9, 1.0]],
                        },
                        "mc_sample_histories": {},
                    }
                ],
            },
        },
    )

    validate_uq_summary_payload(payload)
    assert payload["schema_version"] == 1
    assert payload["run"]["variant_name"] == "O0"
    assert payload["splits"]["clean_validation"]["num_traces"] == 1
    assert (
        payload["splits"]["clean_validation"]["trace_audit"][
            "uncertainty_history_non_null_count"
        ]
        == 1
    )
    assert payload["splits"]["clean_validation"]["point_score_summary"][
        "mean"
    ] == pytest.approx(0.2)
    assert payload["splits"]["clean_validation"]["window_score_summary"][
        "mean"
    ] == pytest.approx(0.45)
    assert payload["splits"]["clean_validation"]["uncertainty_summary"][
        "point_anomaly_score_variance_mean"
    ] == pytest.approx(0.15)


def test_validate_uq_summary_payload_rejects_raw_trace_fields() -> None:
    payload = {
        "schema_version": 1,
        "created_at_utc": "2026-07-20T00:00:00Z",
        "run": {
            "benchmark_kind": "offline",
            "experiment_name": "pytest-thesis-offline",
            "method_name": "THESIS",
            "variant_name": "O0",
            "entity_id": "machine_1_6",
            "seed": 6,
            "stage_name": "stage_b_fusion_finetuning",
            "checkpoint_path": "/tmp/best.pt",
            "checkpoint_sha256": None,
            "experiment_config_path": "/tmp/experiment.yaml",
            "protocol_config_path": "/tmp/protocol.yaml",
            "output_dir": "/tmp/output",
        },
        "run_scalar_logs": {},
        "splits": {
            "clean_validation": {
                "num_traces": 1,
                "sample_retention_policy": "retain_for_eda",
                "trace_audit": {
                    "any_uncertainty_history": False,
                    "uncertainty_history_non_null_count": 0,
                    "any_mc_sample_history": False,
                    "mc_histories_non_null_count": {
                        "point_score_samples": 0,
                        "window_score_samples": 0,
                        "reconstruction_samples": 0,
                        "classification_probability_samples": 0,
                    },
                },
                "point_score_summary": {
                    "mean": 0.1,
                    "std": 0.0,
                    "min": 0.1,
                    "p50": 0.1,
                    "p95": 0.1,
                    "max": 0.1,
                },
                "window_score_summary": {
                    "mean": 0.1,
                    "std": 0.0,
                    "min": 0.1,
                    "p50": 0.1,
                    "p95": 0.1,
                    "max": 0.1,
                },
                "uncertainty_summary": {
                    "point_anomaly_score_variance_mean": None,
                    "point_anomaly_score_variance_p95": None,
                    "window_anomaly_score_variance_mean": None,
                    "continuous_retrieval_variance_point_mean": None,
                    "continuous_retrieval_variance_window_mean": None,
                    "discrete_retrieval_variance_point_mean": None,
                    "discrete_retrieval_variance_window_mean": None,
                    "reconstruction_variance_point_mean": None,
                    "reconstruction_variance_window_mean": None,
                    "reconstruction_variance_full_mean": None,
                    "classification_probability_variance_mean": None,
                    "classification_variance_mean": None,
                },
                "stochastic_query": {},
            },
            "synthetic_validation": {
                "num_traces": 0,
                "sample_retention_policy": None,
                "trace_audit": {
                    "any_uncertainty_history": False,
                    "uncertainty_history_non_null_count": 0,
                    "any_mc_sample_history": False,
                    "mc_histories_non_null_count": {
                        "point_score_samples": 0,
                        "window_score_samples": 0,
                        "reconstruction_samples": 0,
                        "classification_probability_samples": 0,
                    },
                },
                "point_score_summary": {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                "window_score_summary": {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                "uncertainty_summary": {
                    "point_anomaly_score_variance_mean": None,
                    "point_anomaly_score_variance_p95": None,
                    "window_anomaly_score_variance_mean": None,
                    "continuous_retrieval_variance_point_mean": None,
                    "continuous_retrieval_variance_window_mean": None,
                    "discrete_retrieval_variance_point_mean": None,
                    "discrete_retrieval_variance_window_mean": None,
                    "reconstruction_variance_point_mean": None,
                    "reconstruction_variance_window_mean": None,
                    "reconstruction_variance_full_mean": None,
                    "classification_probability_variance_mean": None,
                    "classification_variance_mean": None,
                },
            },
            "test": {
                "num_traces": 0,
                "sample_retention_policy": None,
                "trace_audit": {
                    "any_uncertainty_history": False,
                    "uncertainty_history_non_null_count": 0,
                    "any_mc_sample_history": False,
                    "mc_histories_non_null_count": {
                        "point_score_samples": 0,
                        "window_score_samples": 0,
                        "reconstruction_samples": 0,
                        "classification_probability_samples": 0,
                    },
                },
                "point_score_summary": {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                "window_score_summary": {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                "uncertainty_summary": {
                    "point_anomaly_score_variance_mean": None,
                    "point_anomaly_score_variance_p95": None,
                    "window_anomaly_score_variance_mean": None,
                    "continuous_retrieval_variance_point_mean": None,
                    "continuous_retrieval_variance_window_mean": None,
                    "discrete_retrieval_variance_point_mean": None,
                    "discrete_retrieval_variance_window_mean": None,
                    "reconstruction_variance_point_mean": None,
                    "reconstruction_variance_window_mean": None,
                    "reconstruction_variance_full_mean": None,
                    "classification_probability_variance_mean": None,
                    "classification_variance_mean": None,
                },
            },
        },
    }

    with pytest.raises(ValueError, match="must not contain raw key"):
        validate_uq_summary_payload(payload)
