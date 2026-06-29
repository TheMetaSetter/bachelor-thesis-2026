from __future__ import annotations

"""Offline SMD data path from full entity sequences to fixed-size windows.

A fresher should read this file before reading the models. It shows how raw SMD
sequences are parsed, scaled, windowed, and finally exposed as the batch
contract consumed by the baseline and multitask models.
"""

import os
from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.core.console import console_print
from src.data.cleaning import SequenceCleaningPipeline
from src.data.collate import collate_windows
from src.data.base import BaseDatasetBuilder
from src.data.datasets.anomaly_archive import AnomalyArchiveDatasetParser
from src.data.datasets.smd import SMDDatasetParser
from src.data.download import download_smd_dataset, get_smd_dataset_root
from src.data.scalers import SequenceStandardScaler
from src.data.split_protocol import validate_benchmark_test_labels


def _resolve_data_loader_num_workers(data_config: dict[str, Any]) -> int:
    # Kaggle and server runs often need a portable "use the machine" option
    # instead of a hard-coded worker count. "auto" means use all visible CPU
    # workers with a configurable floor.
    configured_num_workers = data_config.get("num_workers", 0)
    if isinstance(configured_num_workers, str):
        if configured_num_workers != "auto":
            raise ValueError("data.num_workers must be an integer or the string 'auto'")
        visible_cpu_count = os.cpu_count() or 0
        minimum_num_workers = int(data_config.get("min_num_workers", 4))
        resolved_num_workers = max(visible_cpu_count, minimum_num_workers)
        console_print(
            "DATA",
            "Resolved auto data loader workers",
            visible_cpu_count=visible_cpu_count,
            minimum_num_workers=minimum_num_workers,
            resolved_num_workers=resolved_num_workers,
        )
        return resolved_num_workers
    resolved_num_workers = int(configured_num_workers)
    console_print(
        "DATA",
        "Resolved explicit data loader workers",
        resolved_num_workers=resolved_num_workers,
    )
    return resolved_num_workers


def _resolve_smd_root_dir(data_config: dict[str, Any]) -> str:
    # Keep the loader on the same root-resolution codepath as the public data
    # API so environment overrides such as SMD_ROOT_DIR apply everywhere.
    resolved_root_dir = str(get_smd_dataset_root(data_config["root_dir"]))
    console_print(
        "DATA",
        "Resolved SMD root directory for loader builder",
        resolved_root_dir=resolved_root_dir,
    )
    return resolved_root_dir


def _resolve_split_stride(data_config: dict[str, Any], split_name: str) -> int:
    # Keep one shared fallback stride for backward compatibility, but allow
    # explicit split-specific overrides where benchmark protocol needs denser
    # coverage on validation or test timelines.
    override_field_name = f"{split_name}_stride"
    override_stride = data_config.get(override_field_name)
    if override_stride is not None:
        return int(override_stride)
    return int(data_config["stride"])


def _build_window_datasets(
    *,
    scaled_sequences: dict[str, list[dict[str, Any]]],
    data_config: dict[str, Any],
) -> tuple[dict[str, Dataset], int]:
    resolved_num_workers = _resolve_data_loader_num_workers(data_config)
    datasets = {
        split_name: WindowDataset(
            sequences=split_sequences,
            window_size=int(data_config["window_size"]),
            stride=_resolve_split_stride(data_config, split_name),
            max_windows=data_config.get(f"max_{split_name}_windows"),
        )
        for split_name, split_sequences in scaled_sequences.items()
    }
    console_print(
        "DATA",
        "Built window datasets",
        train_stride=_resolve_split_stride(data_config, "train"),
        val_stride=_resolve_split_stride(data_config, "val"),
        test_stride=_resolve_split_stride(data_config, "test"),
        train_windows=len(datasets["train"]),
        val_windows=len(datasets["val"]),
        test_windows=len(datasets["test"]),
    )
    return datasets, resolved_num_workers


def _build_loaders_from_datasets(
    *,
    datasets: dict[str, Dataset],
    data_config: dict[str, Any],
    resolved_num_workers: int,
) -> dict[str, DataLoader]:
    batch_size = int(data_config["batch_size"])
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=bool(data_config.get("shuffle_train", True)),
            num_workers=resolved_num_workers,
            persistent_workers=resolved_num_workers > 0,
            collate_fn=collate_windows,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=resolved_num_workers,
            persistent_workers=resolved_num_workers > 0,
            collate_fn=collate_windows,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=resolved_num_workers,
            persistent_workers=resolved_num_workers > 0,
            collate_fn=collate_windows,
        ),
    }
    console_print(
        "DATA",
        "Built data loaders",
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        resolved_num_workers=resolved_num_workers,
        persistent_workers=resolved_num_workers > 0,
    )
    return loaders


def _build_dataset_bundle_from_sequences(
    *,
    dataset_name: str,
    parser: Any,
    cleaned_sequences: dict[str, list[dict[str, Any]]],
    data_config: dict[str, Any],
) -> dict[str, Any]:
    console_print(
        "DATA",
        "Cleaned dataset split sequences",
        dataset_name=dataset_name,
        train_sequences=len(cleaned_sequences["train"]),
        val_sequences=len(cleaned_sequences["val"]),
        test_sequences=len(cleaned_sequences["test"]),
    )
    validate_benchmark_test_labels(
        dataset_name=dataset_name,
        split_sequences=cleaned_sequences["test"],
    )
    scaler = SequenceStandardScaler()
    scaler.fit(cleaned_sequences["train"])
    scaled_sequences = {
        split_name: scaler.transform_sequences(split_sequences)
        for split_name, split_sequences in cleaned_sequences.items()
    }

    datasets, resolved_num_workers = _build_window_datasets(
        scaled_sequences=scaled_sequences,
        data_config=data_config,
    )
    loaders = _build_loaders_from_datasets(
        datasets=datasets,
        data_config=data_config,
        resolved_num_workers=resolved_num_workers,
    )
    return {
        "dataset_name": dataset_name,
        "parser": parser,
        "scaler": scaler,
        "raw_sequences": cleaned_sequences,
        "scaled_sequences": scaled_sequences,
        "datasets": datasets,
        "loaders": loaders,
    }


def rebuild_dataset_bundle_with_scaler_state(
    *,
    data_bundle: dict[str, Any],
    data_config: dict[str, Any],
    scaler_state_dict: dict[str, Any],
) -> dict[str, Any]:
    raw_sequences = data_bundle.get("raw_sequences")
    if raw_sequences is None:
        raise ValueError(
            "Cannot rebuild dataset bundle with checkpoint scaler state because "
            "raw_sequences are missing"
        )
    scaler = SequenceStandardScaler()
    scaler.load_state_dict(scaler_state_dict)
    scaled_sequences = {
        split_name: scaler.transform_sequences(split_sequences)
        for split_name, split_sequences in raw_sequences.items()
    }
    datasets, resolved_num_workers = _build_window_datasets(
        scaled_sequences=scaled_sequences,
        data_config=data_config,
    )
    loaders = _build_loaders_from_datasets(
        datasets=datasets,
        data_config=data_config,
        resolved_num_workers=resolved_num_workers,
    )
    rebuilt_bundle = dict(data_bundle)
    rebuilt_bundle["scaler"] = scaler
    rebuilt_bundle["scaled_sequences"] = scaled_sequences
    rebuilt_bundle["datasets"] = datasets
    rebuilt_bundle["loaders"] = loaders
    return rebuilt_bundle


class WindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        window_size: int,
        stride: int,
        max_windows: int | None = None,
    ) -> None:
        # The dataset stores only index triples so windows are materialized on
        # demand without copying every possible slice up front.
        self.sequences = sequences
        self.window_size = window_size
        self.stride = stride
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Each item already matches the window contract expected by the collate
        # function, so the engine never has to know where the window came from.
        sequence_index, start_index, end_index = self.index_records[index]
        sequence = self.sequences[sequence_index]
        return {
            "x": sequence["x"][start_index:end_index].clone(),
            "point_labels": None
            if sequence["point_labels"] is None
            else sequence["point_labels"][start_index:end_index].clone(),
            "mask": None
            if sequence["mask"] is None
            else sequence["mask"][start_index:end_index].clone(),
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
                "series_id": sequence["meta"].get(
                    "series_id",
                    (
                        f"{sequence['meta']['dataset_name']}:"
                        f"{sequence['meta']['split']}:"
                        f"{sequence['meta']['entity_id']}"
                    ),
                ),
                "absolute_start_index": start_index,
                "absolute_end_index": end_index,
                "source_sequence_length": int(
                    sequence["meta"].get(
                        "source_sequence_length",
                        sequence["meta"]["sequence_length"],
                    )
                ),
            },
        }


class SMDDatasetBuilder(BaseDatasetBuilder):
    def build(self, data_config: dict[str, Any]) -> dict[str, Any]:
        # The builder returns more than dataloaders on purpose: research code
        # often needs access to the parser, scaler, and scaled sequences later.
        resolved_root_dir = _resolve_smd_root_dir(data_config)
        console_print(
            "DATA",
            "Building SMD dataset bundle",
            resolved_root_dir=resolved_root_dir,
            window_size=data_config["window_size"],
            train_stride=_resolve_split_stride(data_config, "train"),
            val_stride=_resolve_split_stride(data_config, "val"),
            test_stride=_resolve_split_stride(data_config, "test"),
            batch_size=data_config["batch_size"],
        )
        if bool(data_config.get("download", False)):
            download_smd_dataset(
                root_dir=resolved_root_dir,
                skip_existing_download=bool(
                    data_config.get("skip_existing_download", True)
                ),
            )
        parser = SMDDatasetParser(
            root_dir=resolved_root_dir,
            validation_split_ratio=float(data_config["validation_split_ratio"]),
            entity_ids=data_config.get("entity_ids"),
        )
        parsed_sequences = parser.parse()
        cleaning_pipeline = SequenceCleaningPipeline(
            annotate_metadata=bool(data_config.get("annotate_cleaning_metadata", False))
        )
        cleaned_sequences = cleaning_pipeline.transform_splits(parsed_sequences)
        return _build_dataset_bundle_from_sequences(
            dataset_name="smd",
            parser=parser,
            cleaned_sequences=cleaned_sequences,
            data_config=data_config,
        )


class AnomalyArchiveDatasetBuilder(BaseDatasetBuilder):
    def build(self, data_config: dict[str, Any]) -> dict[str, Any]:
        file_path = data_config.get("file_path")
        if file_path is None:
            raise ValueError("anomaly_archive data config requires file_path")
        console_print(
            "DATA",
            "Building AnomalyArchive dataset bundle",
            file_path=file_path,
            window_size=data_config["window_size"],
            train_stride=_resolve_split_stride(data_config, "train"),
            val_stride=_resolve_split_stride(data_config, "val"),
            test_stride=_resolve_split_stride(data_config, "test"),
            batch_size=data_config["batch_size"],
        )
        parser = AnomalyArchiveDatasetParser(
            file_path=file_path,
            validation_split_ratio=float(data_config["validation_split_ratio"]),
        )
        parsed_sequences = parser.parse()
        cleaning_pipeline = SequenceCleaningPipeline(
            annotate_metadata=bool(data_config.get("annotate_cleaning_metadata", False))
        )
        cleaned_sequences = cleaning_pipeline.transform_splits(parsed_sequences)
        return _build_dataset_bundle_from_sequences(
            dataset_name="anomaly_archive",
            parser=parser,
            cleaned_sequences=cleaned_sequences,
            data_config=data_config,
        )


def build_smd_dataset_bundle(data_config: dict[str, Any]) -> dict[str, Any]:
    # This is the registry-facing entrypoint used by the active scripts.
    return SMDDatasetBuilder().build(data_config)


def build_anomaly_archive_dataset_bundle(data_config: dict[str, Any]) -> dict[str, Any]:
    return AnomalyArchiveDatasetBuilder().build(data_config)


def build_smd_dataloaders(data_config: dict[str, Any]) -> dict[str, Any]:
    return build_smd_dataset_bundle(data_config)
