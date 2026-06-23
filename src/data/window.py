from __future__ import annotations

from typing import Any

import torch

from src.core.contracts import validate_raw_sequence, validate_window


def _clone_optional_tensor(optional_tensor: torch.Tensor | None) -> torch.Tensor | None:
    if optional_tensor is None:
        return None
    return optional_tensor.clone()


def slice_sequence_into_windows(
    raw_sequence: dict[str, Any],
    window_size: int = 100,
    stride: int = 10,
) -> list[dict[str, Any]]:
    validate_raw_sequence(raw_sequence)
    full_sequence = raw_sequence["x"]
    sequence_length = full_sequence.shape[0]
    if window_size > sequence_length:
        return []

    sliced_windows: list[dict[str, Any]] = []
    for start_index in range(0, sequence_length - window_size + 1, stride):
        end_index = start_index + window_size
        window = {
            "x": full_sequence[start_index:end_index].clone(),
            "point_labels": None
            if raw_sequence["point_labels"] is None
            else raw_sequence["point_labels"][start_index:end_index].clone(),
            "mask": None
            if raw_sequence["mask"] is None
            else raw_sequence["mask"][start_index:end_index].clone(),
            "timestamps": None
            if raw_sequence["timestamps"] is None
            else raw_sequence["timestamps"][start_index:end_index].clone(),
            "meta": {
                "dataset_name": raw_sequence["meta"]["dataset_name"],
                "entity_id": raw_sequence["meta"]["entity_id"],
                "split": raw_sequence["meta"]["split"],
                "start_index": start_index,
                "end_index": end_index,
                "window_size": window_size,
                "series_id": raw_sequence["meta"].get(
                    "series_id",
                    (
                        f"{raw_sequence['meta']['dataset_name']}:"
                        f"{raw_sequence['meta']['split']}:"
                        f"{raw_sequence['meta']['entity_id']}"
                    ),
                ),
                "absolute_start_index": start_index,
                "absolute_end_index": end_index,
                "source_sequence_length": int(
                    raw_sequence["meta"].get(
                        "source_sequence_length",
                        raw_sequence["meta"]["sequence_length"],
                    )
                ),
            },
        }
        validate_window(window)
        sliced_windows.append(window)

    return sliced_windows


class Windowizer:
    def __init__(self, window_size: int = 100, stride: int = 10) -> None:
        self.window_size = window_size
        self.stride = stride

    def transform(self, sequences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_windows: list[dict[str, Any]] = []
        for sequence in sequences:
            all_windows.extend(
                slice_sequence_into_windows(
                    raw_sequence=sequence,
                    window_size=self.window_size,
                    stride=self.stride,
                )
            )
        return all_windows
