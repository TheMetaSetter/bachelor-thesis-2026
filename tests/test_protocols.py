from types import SimpleNamespace

import numpy as np
import pytest

from src import protocols


CONFIG = {
    "dataset": "smd",
    "entity": "machine",
    "method_name": "mustard",
    "variant": "A1",
    "phase": "online",
    "seed": 6,
    "data": {"window_size": 2},
    "method": {"phases": ["online"]},
    "evaluation": {"score_space": "latent", "point_score_transform": "identity"},
    "profile_paths": {},
    "prepared_identity": {"dataset": "smd", "entity": "machine"},
}
BATCH = {
    "x": np.ones((1, 2, 1)),
    "indices": np.array([[4, 5]]),
    "segment_ids": np.array([0]),
    "identity": {"dataset": "smd", "entity": "machine", "split": "test"},
}


def test_run_rejects_wrong_phase_and_result_identity():
    protocols.validate_run_input(CONFIG)
    with pytest.raises(ValueError, match="phase"):
        protocols.validate_run_input(CONFIG | {"phase": "offline"})
    result = {
        key: CONFIG[key]
        for key in ("dataset", "entity", "method_name", "variant", "phase", "seed")
    }
    protocols.validate_run_output(result | {"status": "running"}, CONFIG)
    with pytest.raises(ValueError, match="entity"):
        protocols.validate_run_output(
            result | {"status": "failed", "entity": "other"}, CONFIG
        )


def test_run_rejects_misaligned_evaluation():
    with pytest.raises(ValueError, match="align"):
        protocols.validate_run_input(
            CONFIG | {"kind": "evaluation", "scores": [0.2], "labels": [0, 1]}
        )


def test_data_rejects_nonfinite_prepared_values_and_disordered_indices():
    split = {
        "x": np.array([[1.0], [2.0]]),
        "labels": np.array([np.nan, 0]),
        "source_file": ["raw", "raw"],
        "source_row": np.array([0, 1]),
        "identity": {"dataset": "smd", "entity": "machine", "split": "test"},
    }
    protocols.validate_data_input(split, CONFIG)
    with pytest.raises(ValueError, match="finite"):
        protocols.validate_data_input(
            split | {"x": np.array([[1.0], [np.inf]])}, CONFIG
        )
    protocols.validate_data_output(BATCH, CONFIG)
    with pytest.raises(ValueError, match="indices"):
        protocols.validate_data_output(BATCH | {"indices": np.array([[5, 4]])}, CONFIG)


def test_scoring_rejects_test_calibration_and_nonfinite_scores():
    with pytest.raises(ValueError, match="source"):
        protocols.validate_calibration_input(
            {"kind": "calibration", "source_split": "test"}, CONFIG
        )
    scores = {
        "score_kind": "window",
        "raw_scores": np.array([[0.1, 0.2]]),
        "indices": BATCH["indices"],
    }
    protocols.validate_calibration_output(scores, CONFIG)
    with pytest.raises(ValueError, match="finite"):
        protocols.validate_calibration_output(
            scores | {"raw_scores": np.array([[np.nan, 0.2]])}, CONFIG
        )


def test_scoring_rejects_mismatched_threshold_and_bad_triage_order():
    calibration = {
        "kind": "calibration",
        "source_split": "validation",
        "score_space": "raw_input",
        "point_threshold": 0.5,
        "checkpoint_sha256": "checkpoint",
    }
    with pytest.raises(ValueError, match="score space"):
        protocols.validate_calibration_output(calibration, CONFIG)
    calibration = calibration | {
        "score_space": "latent",
        "latent_low": 0.6,
        "latent_high": 0.2,
    }
    with pytest.raises(ValueError, match="order"):
        protocols.validate_calibration_output(calibration, CONFIG)


def test_update_rejects_labels_and_disallowed_changes():
    method = SimpleNamespace(update=lambda: None, optimizer=object())
    context = {
        "allow_update": True,
        "recorded_score": 0.1,
        "allowed_parameters": ["projector"],
    }
    protocols.validate_update_input(method, BATCH, context)
    with pytest.raises(ValueError, match="labels"):
        protocols.validate_update_input(
            method, BATCH | {"labels": np.array([[0, 1]])}, context
        )
    result = {"did_update": True, "loss": 0.2, "changed_parameters": ["encoder"]}
    with pytest.raises(ValueError, match="allowed"):
        protocols.validate_update_output(result, context)
    with pytest.raises(ValueError, match="finite"):
        protocols.validate_update_output(result | {"loss": float("inf")}, context)
