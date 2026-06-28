from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.data.download import get_smd_dataset_root
from src.data.loaders import build_anomaly_archive_dataset_bundle
from src.data.loaders import build_smd_dataset_bundle
from src.data.public_types import PublicDataBundle


def _build_smd_data_config(
    *,
    root: str = "data/ServerMachineDataset",
    window_size: int = 100,
    stride: int = 10,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    num_workers: int | str = 0,
    min_num_workers: int = 4,
    shuffle_train: bool = True,
    download: bool = False,
    skip_existing_download: bool = True,
    annotate_cleaning_metadata: bool = False,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
) -> dict[str, Any]:
    normalized_root = str(get_smd_dataset_root(root))
    data_config: dict[str, Any] = {
        "dataset_name": "smd",
        "root_dir": normalized_root,
        "window_size": window_size,
        "stride": stride,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "min_num_workers": min_num_workers,
        "validation_split_ratio": validation_split_ratio,
        "shuffle_train": shuffle_train,
        "download": download,
        "skip_existing_download": skip_existing_download,
        "annotate_cleaning_metadata": annotate_cleaning_metadata,
    }
    if max_train_windows is not None:
        data_config["max_train_windows"] = max_train_windows
    if max_val_windows is not None:
        data_config["max_val_windows"] = max_val_windows
    if max_test_windows is not None:
        data_config["max_test_windows"] = max_test_windows
    return data_config


def _coerce_public_bundle(bundle: dict[str, Any]) -> PublicDataBundle:
    return PublicDataBundle(
        dataset_name=bundle["dataset_name"],
        parser=bundle["parser"],
        scaler=bundle["scaler"],
        raw_sequences=bundle["raw_sequences"],
        scaled_sequences=bundle["scaled_sequences"],
        datasets=bundle["datasets"],
        loaders=bundle["loaders"],
    )


def _build_anomaly_archive_data_config(
    *,
    file_path: str,
    window_size: int = 100,
    stride: int = 10,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    num_workers: int | str = 0,
    min_num_workers: int = 4,
    shuffle_train: bool = True,
    annotate_cleaning_metadata: bool = False,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
) -> dict[str, Any]:
    data_config: dict[str, Any] = {
        "dataset_name": "anomaly_archive",
        "file_path": file_path,
        "window_size": window_size,
        "stride": stride,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "min_num_workers": min_num_workers,
        "validation_split_ratio": validation_split_ratio,
        "shuffle_train": shuffle_train,
        "annotate_cleaning_metadata": annotate_cleaning_metadata,
    }
    if max_train_windows is not None:
        data_config["max_train_windows"] = max_train_windows
    if max_val_windows is not None:
        data_config["max_val_windows"] = max_val_windows
    if max_test_windows is not None:
        data_config["max_test_windows"] = max_test_windows
    return data_config


def load_smd_data(
    *,
    root: str = "data/ServerMachineDataset",
    window_size: int = 100,
    stride: int = 10,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    num_workers: int | str = 0,
    min_num_workers: int = 4,
    shuffle_train: bool = True,
    download: bool = False,
    skip_existing_download: bool = True,
    annotate_cleaning_metadata: bool = False,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
) -> PublicDataBundle:
    data_config = _build_smd_data_config(
        root=root,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        validation_split_ratio=validation_split_ratio,
        num_workers=num_workers,
        min_num_workers=min_num_workers,
        shuffle_train=shuffle_train,
        download=download,
        skip_existing_download=skip_existing_download,
        annotate_cleaning_metadata=annotate_cleaning_metadata,
        max_train_windows=max_train_windows,
        max_val_windows=max_val_windows,
        max_test_windows=max_test_windows,
    )
    return _coerce_public_bundle(build_smd_dataset_bundle(data_config))


def load_anomaly_archive_data(
    *,
    file_path: str,
    window_size: int = 100,
    stride: int = 10,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    num_workers: int | str = 0,
    min_num_workers: int = 4,
    shuffle_train: bool = True,
    annotate_cleaning_metadata: bool = False,
    max_train_windows: int | None = None,
    max_val_windows: int | None = None,
    max_test_windows: int | None = None,
) -> PublicDataBundle:
    data_config = _build_anomaly_archive_data_config(
        file_path=file_path,
        window_size=window_size,
        stride=stride,
        batch_size=batch_size,
        validation_split_ratio=validation_split_ratio,
        num_workers=num_workers,
        min_num_workers=min_num_workers,
        shuffle_train=shuffle_train,
        annotate_cleaning_metadata=annotate_cleaning_metadata,
        max_train_windows=max_train_windows,
        max_val_windows=max_val_windows,
        max_test_windows=max_test_windows,
    )
    return _coerce_public_bundle(build_anomaly_archive_dataset_bundle(data_config))


def point_labels_to_window_labels(point_labels: torch.Tensor) -> torch.Tensor:
    if point_labels.ndim != 2:
        raise ValueError(
            f"point_labels must have shape [batch, window], got {tuple(point_labels.shape)}"
        )
    return (point_labels.sum(dim=1) > 0).long()


def flatten_windows_for_baseline(batch_x: torch.Tensor) -> np.ndarray:
    if batch_x.ndim != 3:
        raise ValueError(
            f"batch_x must have shape [batch, window, channels], got {tuple(batch_x.shape)}"
        )
    flattened_windows = batch_x.reshape(batch_x.shape[0], -1)
    return flattened_windows.detach().cpu().numpy()
