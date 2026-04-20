from __future__ import annotations

from src.data import load_smd_data
from src.data.loaders import build_smd_dataset_bundle


def test_public_data_api_matches_existing_bundle_lengths() -> None:
    public_bundle = load_smd_data(
        root="data/ServerMachineDataset",
        batch_size=8,
        max_train_windows=16,
        max_val_windows=8,
        max_test_windows=8,
    )
    builder_bundle = build_smd_dataset_bundle(
        {
            "dataset_name": "smd",
            "root_dir": "data/ServerMachineDataset",
            "window_size": 100,
            "stride": 10,
            "batch_size": 8,
            "num_workers": 0,
            "validation_split_ratio": 0.2,
            "shuffle_train": True,
            "max_train_windows": 16,
            "max_val_windows": 8,
            "max_test_windows": 8,
        }
    )

    assert len(public_bundle.datasets["train"]) == len(
        builder_bundle["datasets"]["train"]
    )
    assert len(public_bundle.datasets["val"]) == len(builder_bundle["datasets"]["val"])
    assert len(public_bundle.datasets["test"]) == len(
        builder_bundle["datasets"]["test"]
    )


def test_public_data_api_returns_batch_contract_and_attribute_access() -> None:
    public_bundle = load_smd_data(
        root="data/ServerMachineDataset",
        batch_size=4,
        max_train_windows=8,
        max_val_windows=4,
        max_test_windows=4,
    )

    batch = next(iter(public_bundle.loaders["train"]))

    assert public_bundle["dataset_name"] == "smd"
    assert batch["x"].ndim == 3
    assert batch["x"].shape[0] == 4
    assert batch["point_labels"].shape == batch["x"].shape[:2]
