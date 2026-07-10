from __future__ import annotations

import torch

from scripts.visualize_evaluation_results import (
    build_visualization_payload,
    save_entity_evaluation_visualization,
)


def test_evaluation_visualization_writes_entity_artifact(tmp_path) -> None:
    raw_sequence = {
        "x": torch.randn(40, 4),
        "point_labels": torch.zeros(40, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "entity_id": "machine-a",
            "split": "test",
        },
    }
    evaluation_record = {
        "entity_id": "machine-a",
        "point_scores": [0.1] * 10 + [0.9] * 5 + [0.2] * 25,
        "point_labels": [0] * 8 + [1] * 7 + [0] * 25,
        "num_points": 40,
    }

    output_path = save_entity_evaluation_visualization(
        raw_sequence=raw_sequence,
        evaluation_record=evaluation_record,
        threshold=0.5,
        output_path=tmp_path / "evaluation_plot.png",
        channels_to_plot=3,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_visualization_payload_uses_raw_sequence_labels_as_ground_truth() -> None:
    raw_sequence = {
        "x": torch.randn(6, 2),
        "point_labels": torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "entity_id": "machine-a",
            "split": "test",
        },
    }
    evaluation_record = {
        "entity_id": "machine-a",
        "point_scores": [0.1, 0.2, 0.9, 0.8, 0.1, 0.1],
        "point_labels": [0, 0, 0, 0, 0, 0],
        "num_points": 6,
        "evaluated_start_index": 0,
        "evaluated_end_index": 6,
        "evaluated_num_points": 6,
        "raw_num_points": 6,
    }

    payload = build_visualization_payload(
        raw_sequence=raw_sequence,
        evaluation_record=evaluation_record,
        threshold=0.5,
    )

    assert payload["ground_truth_mask"].tolist() == [0, 0, 1, 1, 0, 0]
    assert payload["predicted_mask"].tolist() == [0, 0, 1, 1, 0, 0]


def test_build_visualization_payload_uses_covered_point_mask_when_present() -> None:
    raw_sequence = {
        "x": torch.randn(6, 2),
        "point_labels": torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "entity_id": "machine-a",
            "split": "test",
        },
    }
    evaluation_record = {
        "entity_id": "machine-a",
        "point_scores": [0.1, 0.2, 0.9, 0.8, 0.1, 0.1],
        "point_labels": [0, 0, 0, 0, 0, 0],
        "covered_point_mask": [1, 1, 1, 0, 0, 0],
        "num_points": 6,
        "evaluated_start_index": 0,
        "evaluated_end_index": 6,
        "evaluated_num_points": 3,
        "raw_num_points": 6,
    }

    payload = build_visualization_payload(
        raw_sequence=raw_sequence,
        evaluation_record=evaluation_record,
        threshold=0.5,
    )

    assert payload["is_truncated"] is True
    assert payload["covered_point_mask"].tolist() == [1, 1, 1, 0, 0, 0]
    assert payload["coverage_spans"] == [(0, 3)]
