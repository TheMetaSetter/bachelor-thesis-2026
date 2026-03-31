from __future__ import annotations

from typing import Any

from src.data.window import Windowizer, slice_sequence_into_windows


class DatasetStream:
    def __init__(self, raw_sequence: dict[str, Any], window_size: int = 100, stride: int = 10) -> None:
        self.raw_sequence = raw_sequence
        self.windowizer = Windowizer(window_size=window_size, stride=stride)
        self.point_cursor = 0
        self.window_cursor = 0
        self.windows = slice_sequence_into_windows(raw_sequence, window_size=window_size, stride=stride)

    def next_point(self) -> dict[str, Any] | None:
        if self.point_cursor >= self.raw_sequence["x"].shape[0]:
            return None
        point_index = self.point_cursor
        self.point_cursor += 1
        return {
            "x": self.raw_sequence["x"][point_index],
            "point_label": None
            if self.raw_sequence["point_labels"] is None
            else self.raw_sequence["point_labels"][point_index],
            "mask": None if self.raw_sequence["mask"] is None else self.raw_sequence["mask"][point_index],
            "timestamp": None
            if self.raw_sequence["timestamps"] is None
            else self.raw_sequence["timestamps"][point_index],
            "meta": {
                "entity_id": self.raw_sequence["meta"]["entity_id"],
                "split": self.raw_sequence["meta"]["split"],
                "point_index": point_index,
            },
        }

    def next_window(self) -> dict[str, Any] | None:
        if self.window_cursor >= len(self.windows):
            return None
        window = self.windows[self.window_cursor]
        self.window_cursor += 1
        return window

    def reset(self) -> None:
        self.point_cursor = 0
        self.window_cursor = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "point_cursor": self.point_cursor,
            "window_cursor": self.window_cursor,
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.point_cursor = int(state_dict["point_cursor"])
        self.window_cursor = int(state_dict["window_cursor"])

