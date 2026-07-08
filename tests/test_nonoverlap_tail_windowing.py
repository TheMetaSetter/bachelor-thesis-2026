from __future__ import annotations

import torch

from src.data.window import slice_sequence_into_windows
from src.protocols.point_scores import build_nonoverlap_tail_window_starts


def _raw_sequence(sequence_length: int = 95) -> dict:
    return {
        "x": torch.arange(sequence_length * 2, dtype=torch.float32).reshape(
            sequence_length, 2
        ),
        "point_labels": torch.zeros(sequence_length, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-1-6",
            "split": "test",
            "num_channels": 2,
            "sequence_length": sequence_length,
        },
    }


def test_nonoverlap_starts_cover_exact_sequence() -> None:
    starts = build_nonoverlap_tail_window_starts(sequence_length=100, window_size=20)

    assert starts == [0, 20, 40, 60, 80]


def test_nonoverlap_starts_add_one_end_aligned_tail_window() -> None:
    starts = build_nonoverlap_tail_window_starts(sequence_length=95, window_size=20)

    assert starts == [0, 20, 40, 60, 75]


def test_end_align_windowing_marks_only_tail_window_as_overlap_tail() -> None:
    windows = slice_sequence_into_windows(
        _raw_sequence(),
        window_size=20,
        stride=20,
        tail_policy="end_align",
    )

    assert [window["meta"]["start_index"] for window in windows] == [0, 20, 40, 60, 75]
    assert [window["meta"]["end_index"] for window in windows] == [20, 40, 60, 80, 95]
    assert [window["meta"]["is_tail_window"] for window in windows] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert all(window["meta"]["tail_policy"] == "end_align" for window in windows)
