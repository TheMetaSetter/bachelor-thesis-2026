from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.engine.evaluator import (
    Evaluator,
    reconstruct_pointwise_records_from_window_payload,
    select_point_score_threshold,
)
from src.metrics.pointwise import (
    compute_binary_classification_metrics,
    compute_pointwise_curve_payload,
    compute_pointwise_metrics,
)
from src.models.base_model import BaseModel


class _ToyEvaluationModel(BaseModel):
    def __init__(self, batch_point_scores: list[torch.Tensor]) -> None:
        super().__init__()
        self.batch_point_scores = batch_point_scores
        self.next_batch_index = 0

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        point_scores = self.batch_point_scores[self.next_batch_index]
        self.next_batch_index += 1
        return {
            "outputs": {
                "point_scores": point_scores,
                "window_scores": point_scores.mean(dim=1),
                "aux": {"forward_pass_seconds": 0.1},
            }
        }


class _ToyEvaluationDataset:
    def __init__(self) -> None:
        self.sequences = [
            {
                "x": torch.zeros(4, 1, dtype=torch.float32),
                "point_labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
                "meta": {"entity_id": "machine-1"},
            }
        ]


class _ToyEvaluationDataLoader:
    def __init__(self) -> None:
        self.dataset = _ToyEvaluationDataset()
        self.batches = [
            {
                "x": torch.zeros(1, 3, 1, dtype=torch.float32),
                "point_labels": torch.tensor([[0, 1, 0]], dtype=torch.long),
                "mask": torch.ones(1, 3, 1, dtype=torch.float32),
                "timestamps": torch.arange(3).unsqueeze(0),
                "meta": [
                    {
                        "entity_id": "machine-1",
                        "start_index": 0,
                        "end_index": 3,
                    }
                ],
            },
            {
                "x": torch.zeros(1, 3, 1, dtype=torch.float32),
                "point_labels": torch.tensor([[1, 0, 1]], dtype=torch.long),
                "mask": torch.ones(1, 3, 1, dtype=torch.float32),
                "timestamps": torch.arange(1, 4).unsqueeze(0),
                "meta": [
                    {
                        "entity_id": "machine-1",
                        "start_index": 1,
                        "end_index": 4,
                    }
                ],
            },
        ]

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def test_select_point_score_threshold_ignores_zero_mass_when_positive_scores_exist() -> (
    None
):
    point_scores = np.array([0.0, 0.0, 0.0, 0.2, 0.4, 0.8], dtype=np.float32)

    threshold = select_point_score_threshold(point_scores, quantile=0.5)

    assert threshold > 0.0


def test_evaluator_averages_overlapping_window_point_scores() -> None:
    model = _ToyEvaluationModel(
        batch_point_scores=[
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            torch.tensor([[3.0, 5.0, 7.0]], dtype=torch.float32),
        ]
    )
    data_loader = _ToyEvaluationDataLoader()

    evaluation_outputs = Evaluator(device="cpu").evaluate(
        model=model,
        data_loader=data_loader,
    )

    record = evaluation_outputs["records"][0]
    expected_scores = torch.tensor([1.0, 2.5, 4.0, 7.0], dtype=torch.float32)

    assert record["entity_id"] == "machine-1"
    assert torch.allclose(record["point_scores"], expected_scores)
    assert evaluation_outputs["metrics"]["forward_pass_seconds_mean"] == 0.1
    assert evaluation_outputs["metrics"]["raw_num_points"] == 4
    assert evaluation_outputs["metrics"]["evaluated_num_points"] == 4
    assert evaluation_outputs["metrics"]["is_truncated_evaluation"] == 0.0


def test_reconstruct_pointwise_records_from_window_payload_averages_overlaps() -> None:
    sequences_by_entity = {
        "machine-1": {
            "x": torch.zeros(4, 1, dtype=torch.float32),
            "point_labels": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            "meta": {"entity_id": "machine-1"},
        }
    }
    batch_payloads = [
        {
            "meta": [
                {
                    "entity_id": "machine-1",
                    "start_index": 0,
                    "end_index": 3,
                }
            ],
            "point_scores": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            "point_labels": torch.tensor([[0, 1, 0]], dtype=torch.long),
        },
        {
            "meta": [
                {
                    "entity_id": "machine-1",
                    "start_index": 1,
                    "end_index": 4,
                }
            ],
            "point_scores": torch.tensor([[3.0, 5.0, 7.0]], dtype=torch.float32),
            "point_labels": torch.tensor([[1, 0, 1]], dtype=torch.long),
        },
    ]

    reconstructed_records = reconstruct_pointwise_records_from_window_payload(
        sequences_by_entity=sequences_by_entity,
        batch_payloads=batch_payloads,
    )

    assert len(reconstructed_records) == 1
    assert reconstructed_records[0]["entity_id"] == "machine-1"
    assert torch.allclose(
        reconstructed_records[0]["point_scores"],
        torch.tensor([1.0, 2.5, 4.0, 7.0], dtype=torch.float32),
    )
    assert torch.equal(
        reconstructed_records[0]["point_labels"],
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    assert reconstructed_records[0]["num_points"] == 4


def test_evaluator_concatenates_entity_records_into_one_global_pointwise_vector() -> (
    None
):
    reconstructed_records = [
        {
            "entity_id": "machine-a",
            "point_scores": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "point_labels": torch.tensor([0, 1], dtype=torch.long),
            "num_points": 2,
        },
        {
            "entity_id": "machine-b",
            "point_scores": torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
            "point_labels": torch.tensor([1, 0, 0], dtype=torch.long),
            "num_points": 3,
        },
    ]

    concatenated_scores = np.concatenate(
        [record["point_scores"].numpy() for record in reconstructed_records],
        axis=0,
    )
    concatenated_labels = np.concatenate(
        [record["point_labels"].numpy() for record in reconstructed_records],
        axis=0,
    )

    assert np.allclose(concatenated_scores, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert concatenated_labels.tolist() == [0, 1, 1, 0, 0]


def test_compute_pointwise_metrics_uses_strict_threshold_comparison() -> None:
    point_labels = np.array([0, 1, 0], dtype=np.int64)
    point_scores = np.array([0.0, 0.2, 0.0], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.0,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0


def test_compute_pointwise_metrics_includes_vus_pr() -> None:
    point_labels = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    point_scores = np.array([0.0, 0.9, 0.8, 0.1, 0.2, 0.95], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.5,
        vus_max_buffer_size=2,
        vus_num_thresholds=20,
    )

    assert "vus_pr" in metrics
    assert "vus_roc" in metrics
    assert "affiliation_f1" in metrics
    assert "threshold" in metrics
    assert metrics["vus_pr"] >= 0.0
    assert metrics["vus_pr"] <= 1.0
    assert metrics["vus_roc"] >= 0.0
    assert metrics["vus_roc"] <= 1.0
    assert metrics["affiliation_f1"] >= 0.0
    assert metrics["affiliation_f1"] <= 1.0
    assert metrics["threshold"] == 0.5


def test_evaluator_accepts_vus_configuration() -> None:
    evaluator = Evaluator(
        device="cpu",
        vus_max_buffer_size=20,
        vus_num_thresholds=50,
    )

    assert evaluator.vus_max_buffer_size == 20
    assert evaluator.vus_num_thresholds == 50


def test_compute_binary_classification_metrics_reports_expected_values() -> None:
    logits = torch.tensor(
        [
            [4.0, 0.1],
            [0.1, 4.0],
            [3.0, 0.2],
            [0.2, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    metrics = compute_binary_classification_metrics(logits=logits, labels=labels)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["fpr"] == 0.0


def test_pointwise_metric_helpers_handle_single_class_labels_safely() -> None:
    point_labels = np.zeros(4, dtype=np.int64)
    point_scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=point_scores,
        threshold=0.5,
    )
    curves = compute_pointwise_curve_payload(
        point_labels=point_labels,
        point_scores=point_scores,
    )

    assert np.isnan(metrics["roc_auc"])
    assert not np.isnan(metrics["fpr"])
    assert "roc_curve" in curves
    assert "pr_curve" in curves
