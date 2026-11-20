import numpy as np
import pytest

from src.scoring import aggregate_points, calibrate, reconstruction_scores, score_batch


EWMA = {
    "offline_overlap_reduction": "last_ewma_estimate", "offline_threshold_quantile": 0.95,
    "offline_threshold_split": "validation", "point_score_transform": "identity",
    "ewma_previous_weight": 0.25, "ewma_current_weight": 0.75,
}


def test_raw_units_and_stochastic_latent_samples():
    x = np.array([[[0.0, 1.0]]])
    reconstruction = np.array([[[1.0, 2.0]]])
    settings = {"score_space": "raw_input", "scaler": {"center": [2, 7], "divisor": [2, 3]}}
    np.testing.assert_allclose(reconstruction_scores(x, reconstruction, settings), [[6.5]])
    query = np.array([[[1.0, 0.0]]])
    samples = np.array([[[[0.0, 1.0]], [[0.0, -1.0]]]])
    np.testing.assert_allclose(reconstruction_scores(query, samples, {"score_space": "latent"}), [[1.0]])


def test_q95_uses_final_pointwise_ewma_and_rejects_test_calibration():
    raw = {"score_kind": "window", "raw_scores": [[0.1, 0.2], [0.6, 0.3]],
           "indices": [[0, 1], [1, 2]], "source_split": "validation", "labels": [[0, 0], [0, np.nan]],
           "length": 3}
    identity = {"checkpoint_sha256": "ckpt"}
    result = calibrate(raw, EWMA | {"score_space": "latent"}, identity=identity)
    assert result["point_threshold"] == pytest.approx(np.quantile([0.1, 0.5, 0.3], 0.95))
    assert result["unknown_label_windows"] == 1
    with pytest.raises(ValueError, match="source"):
        calibrate(raw | {"source_split": "test"}, EWMA, identity=identity)


def test_transform_guards_zero_scale():
    raw = {"score_kind": "point", "raw_scores": [1.0, 1.0], "indices": [0, 1],
           "source_split": "validation", "length": 2}
    settings = EWMA | {"score_space": "raw_input", "point_score_transform": "validation_mad_logistic"}
    with pytest.raises(ValueError, match="scale"):
        calibrate(raw, settings, identity={"checkpoint_sha256": "ckpt"})


def test_aggregation_preserves_uncovered_positions():
    records = iter([{"score_kind": "window", "raw_scores": [[1.0, 2.0], [6.0, 8.0]],
                     "indices": [[0, 1], [1, 2]]}])
    result = aggregate_points(records, length=4, settings={"offline_overlap_reduction": "mean"})
    np.testing.assert_allclose(result["scores"], [1, 4, 8, np.nan], equal_nan=True)
    assert result["coverage"].tolist() == [True, True, True, False]


def test_online_score_state_is_read_only_and_finalized_indices_stay_fixed():
    record = {"score_kind": "window", "raw_scores": [[0.2, 0.4]], "indices": [[3, 4]]}
    settings = EWMA | {"phase": "online"}
    first, state = score_batch(record, None, settings)
    later = record | {"raw_scores": [[0.8, 0.6]], "indices": [[4, 5]]}
    second, next_state = score_batch(later, None, settings, state=state)
    assert first["scores"].tolist() == [[0.2, 0.4]]
    assert second["scores"].tolist() == [[pytest.approx(0.7), 0.6]]
    assert next_state["pending"][4] == pytest.approx(0.7)
    assert state["pending"][4] == 0.4
    assert 3 not in next_state["pending"]
    with pytest.raises(ValueError, match="order"):
        score_batch(record, None, settings, state=next_state)
