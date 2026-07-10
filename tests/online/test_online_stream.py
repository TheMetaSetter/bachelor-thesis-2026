from __future__ import annotations

from typing import Any

import torch

from src.data.stream import OnlineWindowBatcher, SMDOnlineStream


def _build_sequence(
    entity_id: str, sequence_length: int = 130, num_channels: int = 38
) -> dict[str, Any]:
    return {
        "x": torch.randn(sequence_length, num_channels),
        "point_labels": torch.zeros(sequence_length, dtype=torch.long),
        "mask": torch.ones(sequence_length, num_channels),
        "timestamps": torch.arange(sequence_length),
        "meta": {
            "dataset_name": "smd",
            "entity_id": entity_id,
            "split": "test",
            "num_channels": num_channels,
            "sequence_length": sequence_length,
        },
    }


def test_online_stream_emits_monotonic_windows() -> None:
    stream = SMDOnlineStream(
        sequences=[_build_sequence("machine-1"), _build_sequence("machine-2")],
        window_size=100,
        stride=10,
        clean_stream_only=True,
    )

    collected_steps: list[int] = []
    while stream.has_next():
        window = stream.next_window()
        collected_steps.append(window["meta"]["stream_step"])
        assert window["x"].shape == (100, 38)

    assert collected_steps == list(range(len(collected_steps)))


def test_online_batcher_restores_stream_state() -> None:
    stream = SMDOnlineStream(
        sequences=[_build_sequence("machine-1"), _build_sequence("machine-2")],
        window_size=100,
        stride=10,
        clean_stream_only=True,
    )
    batcher = OnlineWindowBatcher(
        stream=stream,
        batch_size=2,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
    )

    first_batch = batcher.next_batch()
    saved_state = batcher.state_dict()

    restored_stream = SMDOnlineStream(
        sequences=[_build_sequence("machine-1"), _build_sequence("machine-2")],
        window_size=100,
        stride=10,
        clean_stream_only=True,
    )
    restored_batcher = OnlineWindowBatcher(
        stream=restored_stream,
        batch_size=2,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
    )
    restored_batcher.load_state_dict(saved_state)
    restored_batch = restored_batcher.next_batch()

    assert first_batch["view_a"].shape[-2:] == (100, 38)
    assert first_batch["view_b"].shape[-2:] == (100, 38)
    assert (
        restored_batch["meta"][0]["stream_step"]
        == saved_state["stream_state_dict"]["cursor"]["position"]
    )
