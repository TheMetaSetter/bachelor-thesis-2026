from __future__ import annotations

from src.data.loaders import build_smd_dataset_bundle


def test_machine_3_4_stride1_windows_expose_overlap_source_identity_metadata() -> None:
    data_bundle = build_smd_dataset_bundle(
        {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "entity_ids": ["machine-3-4"],
            "window_size": 20,
            "stride": 1,
            "batch_size": 2,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": False,
            "max_train_windows": 3,
            "max_val_windows": 2,
            "max_test_windows": 2,
        }
    )

    first_window = data_bundle["datasets"]["train"][0]
    second_window = data_bundle["datasets"]["train"][1]

    assert first_window["meta"]["entity_id"] == "machine-3-4"
    assert first_window["meta"]["series_id"] == "smd:train:machine-3-4"
    assert second_window["meta"]["series_id"] == first_window["meta"]["series_id"]
    assert first_window["meta"]["absolute_start_index"] == 0
    assert first_window["meta"]["absolute_end_index"] == 20
    assert second_window["meta"]["absolute_start_index"] == 1
    assert second_window["meta"]["absolute_end_index"] == 21
    assert first_window["meta"]["source_sequence_length"] > 21
