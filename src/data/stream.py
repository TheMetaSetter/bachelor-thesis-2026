from __future__ import annotations
"""Sequential window streaming for the first online adaptation slice.

The offline data path exposes shuffled batches. This file instead preserves time
order so the online loop can process windows as a stream. A new reader should
notice that the online path extends the offline window contract rather than
replacing it.
"""

from dataclasses import dataclass
from typing import Any

import torch

from src.core.contracts import validate_online_batch, validate_window
from src.data.collate import collate_windows


@dataclass
class SequenceCursor:
    # The cursor is small but important: it is the piece of state that lets the
    # online stream resume from checkpoints without inventing a second protocol.
    position: int = 0

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.position = int(state_dict["position"])

    def reset(self) -> None:
        self.position = 0


class SMDOnlineStream:
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        window_size: int,
        stride: int,
        clean_stream_only: bool = True,
        max_windows: int | None = None,
    ) -> None:
        # The first accepted online slice is intentionally conservative. The
        # stream therefore supports only clean sequential windows right now.
        if not clean_stream_only:
            raise ValueError("The first online adaptation slice supports only clean_stream_only=True")
        self.sequences = sequences
        self.window_size = window_size
        self.stride = stride
        self.cursor = SequenceCursor(position=0)
        self.index_records: list[tuple[int, int, int]] = []

        for sequence_index, sequence in enumerate(sequences):
            sequence_length = int(sequence["x"].shape[0])
            if sequence_length < window_size:
                continue
            for start_index in range(0, sequence_length - window_size + 1, stride):
                end_index = start_index + window_size
                self.index_records.append((sequence_index, start_index, end_index))
                if max_windows is not None and len(self.index_records) >= max_windows:
                    return

    def __len__(self) -> int:
        return len(self.index_records)

    def has_next(self) -> bool:
        return self.cursor.position < len(self.index_records)

    def next_window(self) -> dict[str, Any]:
        # Each yielded window still looks like an offline window, plus stream
        # metadata that the online loop can serialize and resume later.
        if not self.has_next():
            raise StopIteration("No more windows remain in the online stream")

        stream_step = self.cursor.position
        sequence_index, start_index, end_index = self.index_records[self.cursor.position]
        sequence = self.sequences[sequence_index]
        self.cursor.position += 1

        window = {
            "x": sequence["x"][start_index:end_index].clone(),
            "point_labels": None
            if sequence["point_labels"] is None
            else sequence["point_labels"][start_index:end_index].clone(),
            "mask": None if sequence["mask"] is None else sequence["mask"][start_index:end_index].clone(),
            "timestamps": None
            if sequence["timestamps"] is None
            else sequence["timestamps"][start_index:end_index].clone(),
            "meta": {
                "dataset_name": sequence["meta"]["dataset_name"],
                "entity_id": sequence["meta"]["entity_id"],
                "split": sequence["meta"]["split"],
                "start_index": start_index,
                "end_index": end_index,
                "window_size": self.window_size,
                "stream_step": stream_step,
            },
        }
        validate_window(window)
        return window

    def state_dict(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor.state_dict(),
            "window_size": self.window_size,
            "stride": self.stride,
            "num_index_records": len(self.index_records),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.cursor.load_state_dict(state_dict["cursor"])

    def reset(self) -> None:
        self.cursor.reset()


class OnlineWindowBatcher:
    def __init__(
        self,
        stream: SMDOnlineStream,
        batch_size: int,
        view_noise_std: float = 0.0,
        view_dropout_probability: float = 0.0,
    ) -> None:
        # The batcher adds only the two online views. Everything else is reused
        # from the same collated window structure as the offline pipeline.
        self.stream = stream
        self.batch_size = batch_size
        self.view_noise_std = view_noise_std
        self.view_dropout_probability = view_dropout_probability

    def _build_view(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        # View construction stays lightweight in the first online slice because
        # the main experiment question is adaptation, not aggressive augmentation.
        view_tensor = batch_tensor.clone()
        if self.view_noise_std > 0.0:
            view_tensor = view_tensor + torch.randn_like(view_tensor) * self.view_noise_std
        if self.view_dropout_probability > 0.0:
            keep_mask = torch.rand_like(view_tensor).ge(self.view_dropout_probability)
            view_tensor = view_tensor * keep_mask.to(view_tensor.dtype)
        return view_tensor

    def next_batch(self) -> dict[str, Any]:
        # The final batch is validated against the extended online contract
        # before the model ever sees it.
        windows: list[dict[str, Any]] = []
        while len(windows) < self.batch_size and self.stream.has_next():
            windows.append(self.stream.next_window())

        if not windows:
            raise StopIteration("No more batches remain in the online batcher")

        batch = collate_windows(windows)
        batch["view_a"] = self._build_view(batch["x"])
        batch["view_b"] = self._build_view(batch["x"])
        validate_online_batch(batch)
        return batch

    def __iter__(self) -> Any:
        while True:
            try:
                yield self.next_batch()
            except StopIteration:
                return

    def state_dict(self) -> dict[str, Any]:
        return {
            "stream_state_dict": self.stream.state_dict(),
            "batch_size": self.batch_size,
            "view_noise_std": self.view_noise_std,
            "view_dropout_probability": self.view_dropout_probability,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.stream.load_state_dict(state_dict["stream_state_dict"])

    def reset(self) -> None:
        self.stream.reset()
