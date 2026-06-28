from __future__ import annotations

import pytest
import torch

from src.data.split_protocol import (
    describe_label_regime,
    summarize_split_point_labels,
    validate_benchmark_test_labels,
)


def _build_sequence(labels: list[int]) -> dict[str, object]:
    x = torch.zeros(len(labels), 1, dtype=torch.float32)
    return {
        "x": x,
        "point_labels": torch.tensor(labels, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "synthetic",
            "entity_id": "entity-1",
            "split": "test",
            "num_channels": 1,
            "sequence_length": len(labels),
        },
    }


def test_describe_label_regime_distinguishes_three_cases() -> None:
    assert describe_label_regime([0, 0, 0]) == "all_zero"
    assert describe_label_regime([1, 1, 1]) == "all_one"
    assert describe_label_regime([0, 1, 0]) == "mixed"


def test_summarize_split_point_labels_counts_mixed_test_labels() -> None:
    summary = summarize_split_point_labels(
        [_build_sequence([0, 0, 1, 1, 0, 0])],
    )

    assert summary["label_regime"] == "mixed"
    assert summary["n_pos"] == 2
    assert summary["n_neg"] == 4
    assert summary["positive_ratio"] == pytest.approx(2 / 6)


def test_validate_benchmark_test_labels_rejects_single_class_test_split() -> None:
    with pytest.raises(ValueError, match="must contain both normal and anomalous"):
        validate_benchmark_test_labels(
            dataset_name="anomaly_archive",
            split_sequences=[_build_sequence([1, 1, 1, 1])],
        )
