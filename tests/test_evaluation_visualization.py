from __future__ import annotations

import torch

from scripts.visualize_evaluation_results import save_entity_evaluation_visualization


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
