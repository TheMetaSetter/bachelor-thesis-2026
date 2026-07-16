from __future__ import annotations

import numpy as np
import torch

from src.engine.evaluator import Evaluator


def test_evaluator_trace_payload_keeps_histories_and_mc_samples() -> None:
    step_output = {
        "outputs": {
            "window_scores": torch.tensor([0.4, 0.6], dtype=torch.float32),
            "aux": {
                "uncertainty": {
                    "point_anomaly_score_variance": torch.tensor([0.1, 0.2]),
                    "window_anomaly_score_variance": torch.tensor([0.3, 0.4]),
                    "reconstruction_variance_full": torch.tensor(
                        [[0.5, 0.6], [0.7, 0.8]]
                    ),
                },
                "deterministic_geometry": {
                    "latent_window_score": torch.tensor([0.9, 1.0])
                },
                "stochastic_query": {
                    "sample_retention_policy": "retain_for_eda",
                    "point_score_samples": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                    "window_score_samples": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                    "reconstruction_samples": torch.tensor([[[1.0, 2.0]]]),
                    "classification_probability_samples": torch.tensor([[[0.2, 0.8]]]),
                },
            },
        }
    }
    trace_payload = Evaluator._build_trace_payload(
        batch_index=3,
        batch_meta=[{"entity_id": "machine-1-6"}],
        step_output=step_output,
        point_scores=torch.tensor([[0.11, 0.22]], dtype=torch.float32),
    )

    assert trace_payload["batch_index"] == 3
    assert trace_payload["entity_ids"] == ["machine-1-6"]
    assert trace_payload["sample_retention_policy"] == "retain_for_eda"
    assert np.allclose(trace_payload["point_score_history"], [[0.11, 0.22]])
    assert np.allclose(trace_payload["window_score_history"], [0.4, 0.6])
    assert np.allclose(
        trace_payload["mc_sample_histories"]["point_score_samples"],
        [[0.1, 0.2], [0.3, 0.4]],
    )
    assert np.allclose(
        trace_payload["mc_sample_histories"]["window_score_samples"],
        [[0.5, 0.6], [0.7, 0.8]],
    )
    assert np.allclose(
        trace_payload["mc_sample_histories"]["reconstruction_samples"],
        [[[1.0, 2.0]]],
    )
    assert np.allclose(
        trace_payload["uncertainty_history"]["point_anomaly_score_variance"],
        [0.1, 0.2],
    )
