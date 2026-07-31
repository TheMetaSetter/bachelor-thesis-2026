import torch

from src.data.stream import SMDOnlineStream
from src.engine.online_tta.online_engine_run import _select_online_stream_sequence


def test_selected_online_stream_keeps_entity_global_window_indices() -> None:
    sequence = {
        "x": torch.zeros(100, 2),
        "point_labels": torch.zeros(100),
        "mask": None,
        "timestamps": None,
        "meta": {
            "dataset_name": "smd",
            "entity_id": "machine-1-6",
            "split": "test",
            "sequence_length": 100,
        },
    }

    selected = _select_online_stream_sequence(
        sequence, absolute_start_index=56, absolute_end_index=91
    )
    stream = SMDOnlineStream(
        sequences=[selected], window_size=20, stride=1, clean_stream_only=True
    )

    assert len(stream) == 16
    assert stream.next_window()["meta"]["start_index"] == 56
    last_window = stream._build_window(selected, 15, 35, stream_step=15)
    assert last_window["meta"]["end_index"] == 91
