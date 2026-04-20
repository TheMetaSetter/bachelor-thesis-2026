from __future__ import annotations

import torch

from src.data.window import Windowizer, slice_sequence_into_windows


def test_windowizer_preserves_shape_and_metadata() -> None:
    raw_sequence = {
        "x": torch.arange(60, dtype=torch.float32).reshape(20, 3),
        "point_labels": torch.zeros(20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-x",
            "split": "train",
            "num_channels": 3,
            "sequence_length": 20,
        },
    }

    windows = slice_sequence_into_windows(raw_sequence, window_size=5, stride=5)

    assert len(windows) == 4
    assert windows[0]["x"].shape == (5, 3)
    assert windows[0]["meta"]["start_index"] == 0
    assert windows[0]["meta"]["end_index"] == 5
    assert all(window["meta"]["entity_id"] == "machine-x" for window in windows)


def test_windowizer_transform_combines_windows_without_crossing_entities() -> None:
    first_sequence = {
        "x": torch.ones(10, 2),
        "point_labels": torch.zeros(10, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-a",
            "split": "train",
            "num_channels": 2,
            "sequence_length": 10,
        },
    }
    second_sequence = {
        "x": 2 * torch.ones(10, 2),
        "point_labels": torch.zeros(10, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-b",
            "split": "train",
            "num_channels": 2,
            "sequence_length": 10,
        },
    }
    windowizer = Windowizer(window_size=5, stride=5)

    windows = windowizer.transform([first_sequence, second_sequence])

    assert len(windows) == 4
    assert [window["meta"]["entity_id"] for window in windows] == [
        "machine-a",
        "machine-a",
        "machine-b",
        "machine-b",
    ]
