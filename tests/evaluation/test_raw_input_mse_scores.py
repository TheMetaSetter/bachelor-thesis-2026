from __future__ import annotations

import pytest
import torch

from src.data.scalers import SequenceStandardScaler
from src.core.contracts import validate_evaluation_record
from src.engine.evaluator import (
    Evaluator,
    reconstruct_pointwise_records_from_window_payload,
)
from src.protocols.reconstruction_scores import score_reconstruction


def _fit_scaler() -> SequenceStandardScaler:
    scaler = SequenceStandardScaler()
    scaler.fit(
        [
            {
                "x": torch.tensor(
                    [[10.0, 5.0, 7.0], [14.0, 5.0, 9.0]], dtype=torch.float32
                ),
                "point_labels": None,
                "mask": None,
                "timestamps": None,
                "meta": {
                    "dataset_name": "toy",
                    "entity_id": "machine-1",
                    "split": "train",
                    "num_channels": 3,
                    "sequence_length": 2,
                },
            }
        ]
    )
    return scaler


def _fit_one_feature_scaler() -> SequenceStandardScaler:
    scaler = SequenceStandardScaler()
    scaler.fit(
        [
            {
                "x": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
                "point_labels": None,
                "mask": None,
                "timestamps": None,
                "meta": {
                    "dataset_name": "toy",
                    "entity_id": "machine-1",
                    "split": "train",
                    "num_channels": 1,
                    "sequence_length": 2,
                },
            }
        ]
    )
    return scaler


def test_inverse_transform_restores_active_features_and_keeps_inactive_features() -> (
    None
):
    scaler = _fit_scaler()
    scaled_values = torch.tensor([[0.0, 99.0, -1.0]], dtype=torch.float32)

    restored_values = scaler.inverse_transform_tensor(scaled_values)

    assert torch.allclose(restored_values, torch.tensor([[12.0, 99.0, 7.0]]))
    assert torch.equal(scaled_values, torch.tensor([[0.0, 99.0, -1.0]]))


def test_inverse_transform_rejects_unfitted_scaler_and_wrong_feature_dimension() -> (
    None
):
    with pytest.raises(RuntimeError, match="fit"):
        SequenceStandardScaler().inverse_transform_tensor(torch.zeros(1, 1))

    with pytest.raises(ValueError, match="feature dimension"):
        _fit_scaler().inverse_transform_tensor(torch.zeros(1, 2))


def test_score_reconstruction_returns_raw_and_normalized_point_and_window_mse() -> None:
    scaler = _fit_scaler()
    input_scaled = torch.tensor(
        [[[0.0, 99.0, -1.0], [1.0, 99.0, 1.0]]], dtype=torch.float32
    )
    reconstruction_scaled = torch.tensor(
        [[[0.5, 99.0, -1.5], [0.0, 99.0, 0.0]]], dtype=torch.float32
    )

    scores = score_reconstruction(input_scaled, reconstruction_scaled, scaler)

    assert scores["raw_input_point_mse"].shape == (1, 2)
    assert scores["raw_input_window_mse"].shape == (1,)
    assert torch.allclose(
        scores["raw_input_point_mse"], torch.tensor([[5.0 / 12.0, 5.0 / 3.0]])
    )
    assert torch.allclose(scores["raw_input_window_mse"], torch.tensor([25.0 / 24.0]))
    assert torch.allclose(
        scores["normalized_input_point_mse"],
        torch.tensor([[1.0 / 6.0, 2.0 / 3.0]]),
    )
    assert torch.allclose(
        scores["normalized_input_window_mse"], torch.tensor([5.0 / 12.0])
    )


def test_score_reconstruction_averages_per_sample_mse_not_mean_reconstruction() -> None:
    scaler = _fit_scaler()
    input_scaled = torch.zeros(1, 1, 3)
    reconstruction_samples = torch.tensor(
        [[[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]]], dtype=torch.float32
    )

    scores = score_reconstruction(input_scaled, reconstruction_samples, scaler)

    assert torch.allclose(
        scores["normalized_input_point_mse"], torch.tensor([[5.0 / 3.0]])
    )
    assert torch.allclose(
        scores["normalized_input_window_mse"], torch.tensor([5.0 / 3.0])
    )


def test_score_reconstruction_rejects_non_finite_values() -> None:
    scaler = _fit_scaler()
    with pytest.raises(ValueError, match="finite"):
        score_reconstruction(
            torch.tensor([[[float("nan"), 0.0, 0.0]]]),
            torch.zeros(1, 1, 3),
            scaler,
        )


def test_reconstruct_raw_and_normalized_scores_averages_overlap_independently() -> None:
    sequences_by_entity = {
        "machine-1": {
            "x": torch.zeros(4, 1),
            "point_labels": torch.tensor([0, 1, 0, 1]),
            "meta": {"entity_id": "machine-1"},
        }
    }
    records = reconstruct_pointwise_records_from_window_payload(
        sequences_by_entity=sequences_by_entity,
        batch_payloads=[
            {
                "meta": [{"entity_id": "machine-1", "start_index": 0, "end_index": 3}],
                "point_scores": torch.tensor([[1.0, 2.0, 3.0]]),
                "raw_input_point_mse": torch.tensor([[10.0, 20.0, 30.0]]),
                "normalized_input_point_mse": torch.tensor([[1.0, 2.0, 3.0]]),
                "point_labels": torch.tensor([[0, 1, 0]]),
            },
            {
                "meta": [{"entity_id": "machine-1", "start_index": 1, "end_index": 4}],
                "point_scores": torch.tensor([[4.0, 5.0, 6.0]]),
                "raw_input_point_mse": torch.tensor([[40.0, 50.0, 60.0]]),
                "normalized_input_point_mse": torch.tensor([[4.0, 5.0, 6.0]]),
                "point_labels": torch.tensor([[1, 0, 1]]),
            },
        ],
    )

    record = records[0]
    assert torch.equal(
        record["raw_input_point_mse"], torch.tensor([10.0, 30.0, 40.0, 60.0])
    )
    assert torch.equal(
        record["normalized_input_point_mse"], torch.tensor([1.0, 3.0, 4.0, 6.0])
    )
    assert torch.equal(record["covered_point_mask"], torch.ones(4, dtype=torch.bool))


class _RawEvaluationModel(torch.nn.Module):
    def test_step(self, batch):
        reconstruction = batch["x"] + torch.tensor(
            [[[1.0], [0.0], [1.0]]], dtype=batch["x"].dtype
        )
        return {
            "outputs": {
                "recon": reconstruction,
                "point_scores": torch.zeros(batch["x"].shape[:2]),
                "window_scores": torch.zeros(batch["x"].shape[0]),
                "aux": {},
            }
        }

    def to(self, device):
        return self

    def eval(self):
        return self


class _RawEvaluationDataset:
    def __init__(self):
        self.sequences = [
            {
                "x": torch.zeros(3, 1),
                "point_labels": torch.tensor([0, 1, 0]),
                "meta": {"entity_id": "machine-1"},
            }
        ]


class _RawEvaluationLoader:
    def __init__(self):
        self.dataset = _RawEvaluationDataset()

    def __len__(self):
        return 1

    def __iter__(self):
        yield {
            "x": torch.zeros(1, 3, 1),
            "point_labels": torch.tensor([[0, 1, 0]]),
            "mask": torch.ones(1, 3, 1),
            "timestamps": torch.arange(3).unsqueeze(0),
            "meta": [{"entity_id": "machine-1", "start_index": 0, "end_index": 3}],
        }


class _SyntheticEvaluationModel(_RawEvaluationModel):
    def synthetic_validation_step(self, batch):
        prepared_batch = dict(batch)
        prepared_batch["x"] = batch["x"] + 2.0
        prepared_batch["synthetic_anomaly_mask"] = torch.tensor(
            [[0, 1, 0]], dtype=torch.long, device=batch["x"].device
        )
        return {
            "outputs": {
                "recon": prepared_batch["x"] + 1.0,
                "point_scores": torch.zeros(batch["x"].shape[:2]),
                "window_scores": torch.zeros(batch["x"].shape[0]),
                "aux": {},
            },
            "batch": prepared_batch,
        }


def test_evaluator_uses_raw_input_mse_for_threshold_and_prediction() -> None:
    scaler = _fit_one_feature_scaler()
    result = Evaluator(device="cpu").evaluate(
        model=_RawEvaluationModel(),
        data_loader=_RawEvaluationLoader(),
        point_score_threshold=0.1,
        threshold_source="clean_validation",
        score_space="raw_input",
        scaler=scaler,
        window_score_threshold=0.1,
    )

    record = result["records"][0]
    assert torch.allclose(
        record["raw_input_point_mse"], torch.tensor([0.25, 0.0, 0.25])
    )
    assert torch.allclose(
        record["normalized_input_point_mse"], torch.tensor([1.0, 0.0, 1.0])
    )
    assert torch.equal(record["window_labels"], torch.tensor([1]))
    assert torch.equal(record["point_predictions"], torch.tensor([1, 0, 1]))
    assert torch.equal(record["window_predictions"], torch.ones(1, dtype=torch.long))
    assert result["metrics"]["threshold"] == 0.1


def test_evaluator_scores_after_synthetic_preparation_and_uses_synthetic_labels() -> (
    None
):
    result = Evaluator(device="cpu").evaluate(
        model=_SyntheticEvaluationModel(),
        data_loader=_RawEvaluationLoader(),
        score_space="raw_input",
        scaler=_fit_one_feature_scaler(),
        point_score_threshold=0.1,
        window_score_threshold=0.1,
        evaluation_stage="val_synth",
    )

    record = result["records"][0]
    assert torch.equal(record["point_labels"], torch.tensor([0, 1, 0]))
    assert torch.equal(record["window_labels"], torch.tensor([1]))


def test_evaluation_record_validates_optional_raw_and_normalized_scores() -> None:
    record = {
        "entity_id": "machine-1-6",
        "point_scores": torch.zeros(3),
        "point_labels": torch.tensor([0, 1, 0]),
        "num_points": 3,
        "raw_input_point_mse": torch.ones(3),
        "normalized_input_point_mse": torch.full((3,), 2.0),
    }
    validate_evaluation_record(record)

    record["raw_input_point_mse"] = torch.ones(2)
    with pytest.raises(ValueError, match="raw_input_point_mse"):
        validate_evaluation_record(record)
