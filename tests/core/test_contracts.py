from __future__ import annotations

import pytest
import torch

from src.core.contracts import validate_model_outputs, validate_online_batch


def test_validate_model_outputs_accepts_full_spec_v3_sample_histories() -> None:
    # Ý tưởng chính ở đây là: contract phải chấp nhận output có đủ phần
    # stochastic samples, uncertainty, và deterministic geometry mà THESIS dùng
    # cho EDA sau thí nghiệm.
    outputs = {
        "hidden": torch.randn(2, 20, 16),
        "pooled": torch.randn(2, 320),
        "recon": torch.randn(2, 20, 38),
        "logits": torch.randn(2, 2),
        "point_scores": torch.randn(2, 20),
        "window_scores": torch.randn(2),
        "aux": {
            "stochastic_query": {
                "schema_version": 3,
                "enabled": True,
                "num_samples": 10,
                "continuous_temperature": 0.9,
                "discrete_temperature": 0.9,
                "continuous_retrieved_samples": torch.randn(2, 10, 20, 16),
                "discrete_retrieved_samples": torch.randn(2, 10, 20, 16),
                "discrete_topk_ids": torch.randint(0, 8, (2, 10, 20, 3)),
                "reconstruction_samples": torch.randn(2, 10, 20, 38),
                "classification_probability_samples": torch.randn(2, 10, 2),
                "point_score_samples": torch.randn(2, 10, 20),
                "window_score_samples": torch.randn(2, 10),
            },
            "uncertainty": {
                "point_anomaly_score_variance": torch.randn(2, 20),
                "window_anomaly_score_variance": torch.randn(2),
                "reconstruction_variance_full": torch.randn(2, 20, 38),
                "classification_probability_variance": torch.randn(2, 2),
            },
            "deterministic_geometry": {
                "nearest_codeword_ids": torch.randint(0, 8, (2, 20)),
                "nearest_codeword_distances": torch.randn(2, 20),
                "known_anomaly_mask": torch.zeros(2, 20, dtype=torch.bool),
                "continuous_signature_ids": torch.randint(0, 8, (2, 20, 3)),
                "latent_window_score": torch.randn(2),
            },
        },
    }

    validate_model_outputs(outputs)


def test_validate_online_batch_requires_single_window_and_no_secondary_views() -> None:
    batch = {
        "x": torch.zeros(1, 20, 3),
        "point_labels": None,
        "mask": None,
        "timestamps": torch.arange(20).unsqueeze(0),
        "absolute_indices": torch.arange(20).unsqueeze(0),
        "meta": [{"entity_id": "machine-1-6", "start_index": 0, "end_index": 20}],
    }

    validate_online_batch(batch)
    assert "view_a" not in batch
    assert "view_b" not in batch


def test_validate_online_batch_rejects_non_causal_absolute_indices() -> None:
    batch = {
        "x": torch.zeros(1, 3, 1),
        "point_labels": None,
        "mask": None,
        "timestamps": torch.arange(3).unsqueeze(0),
        "absolute_indices": torch.tensor([[3, 3, 4]]),
        "meta": [{"entity_id": "machine-1-6", "start_index": 3, "end_index": 6}],
    }

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_online_batch(batch)


def test_validate_model_outputs_rejects_mismatched_sample_axis() -> None:
    outputs = {
        "hidden": torch.randn(2, 20, 16),
        "pooled": torch.randn(2, 320),
        "recon": torch.randn(2, 20, 38),
        "logits": torch.randn(2, 2),
        "point_scores": torch.randn(2, 20),
        "window_scores": torch.randn(2),
        "aux": {
            "stochastic_query": {
                "schema_version": 3,
                "enabled": True,
                "num_samples": 10,
                "continuous_temperature": 0.9,
                "discrete_temperature": 0.9,
                "continuous_retrieved_samples": torch.randn(2, 9, 20, 16),
                "discrete_retrieved_samples": torch.randn(2, 10, 20, 16),
                "discrete_topk_ids": torch.randint(0, 8, (2, 10, 20, 3)),
                "reconstruction_samples": torch.randn(2, 10, 20, 38),
                "classification_probability_samples": torch.randn(2, 10, 2),
                "point_score_samples": torch.randn(2, 10, 20),
                "window_score_samples": torch.randn(2, 10),
            }
        },
    }

    with pytest.raises(ValueError, match="sample axis equal to num_samples"):
        validate_model_outputs(outputs)


def test_validate_model_outputs_rejects_non_finite_uncertainty() -> None:
    outputs = {
        "hidden": torch.randn(2, 20, 16),
        "pooled": torch.randn(2, 320),
        "recon": torch.randn(2, 20, 38),
        "logits": torch.randn(2, 2),
        "point_scores": torch.randn(2, 20),
        "window_scores": torch.randn(2),
        "aux": {
            "stochastic_query": {
                "schema_version": 3,
                "enabled": True,
                "num_samples": 10,
                "continuous_temperature": 0.9,
                "discrete_temperature": 0.9,
                "continuous_retrieved_samples": torch.randn(2, 10, 20, 16),
                "discrete_retrieved_samples": torch.randn(2, 10, 20, 16),
                "discrete_topk_ids": torch.randint(0, 8, (2, 10, 20, 3)),
                "reconstruction_samples": torch.randn(2, 10, 20, 38),
                "classification_probability_samples": torch.randn(2, 10, 2),
                "point_score_samples": torch.randn(2, 10, 20),
                "window_score_samples": torch.randn(2, 10),
            },
            "uncertainty": {
                "point_anomaly_score_variance": torch.tensor(
                    [[0.1, float("nan")], [0.2, 0.3]]
                ),
                "window_anomaly_score_variance": torch.randn(2),
                "reconstruction_variance_full": torch.randn(2, 20, 38),
                "classification_probability_variance": torch.randn(2, 2),
            },
        },
    }

    with pytest.raises(ValueError, match="finite values"):
        validate_model_outputs(outputs)
