from __future__ import annotations
"""Offline SMD data path from full entity sequences to fixed-size windows.

A fresher should read this file before reading the models. It shows how raw SMD
sequences are parsed, scaled, windowed, and finally exposed as the batch
contract consumed by the baseline and multitask models.
"""

import os
from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.data.cleaning import SequenceCleaningPipeline
from src.data.collate import collate_windows
from src.data.base import BaseDatasetBuilder
from src.data.datasets.smd import SMDDatasetParser
from src.data.download import download_smd_dataset
from src.data.scalers import SequenceStandardScaler


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
        return max(visible_cpu_count, minimum_num_workers)
    return int(configured_num_workers)


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
            },
        }


class SMDDatasetBuilder(BaseDatasetBuilder):
    def build(self, data_config: dict[str, Any]) -> dict[str, Any]:
        # The builder returns more than dataloaders on purpose: research code
        # often needs access to the parser, scaler, and scaled sequences later.
        if bool(data_config.get("download", False)):
            download_smd_dataset(
                root_dir=data_config["root_dir"],
                skip_existing_download=bool(data_config.get("skip_existing_download", True)),
            )
        parser = SMDDatasetParser(
            root_dir=data_config["root_dir"],
            validation_split_ratio=float(data_config["validation_split_ratio"]),
        )
        parsed_sequences = parser.parse()
        cleaning_pipeline = SequenceCleaningPipeline(
            annotate_metadata=bool(data_config.get("annotate_cleaning_metadata", False))
        )
        cleaned_sequences = cleaning_pipeline.transform_splits(parsed_sequences)

        scaler = SequenceStandardScaler()
        scaler.fit(cleaned_sequences["train"])
        scaled_sequences = {
            split_name: scaler.transform_sequences(split_sequences)
            for split_name, split_sequences in cleaned_sequences.items()
        }

        resolved_num_workers = _resolve_data_loader_num_workers(data_config)

        datasets = {
            split_name: WindowDataset(
                sequences=split_sequences,
                window_size=int(data_config["window_size"]),
                stride=int(data_config["stride"]),
                max_windows=data_config.get(f"max_{split_name}_windows"),
            )
            for split_name, split_sequences in scaled_sequences.items()
        }

        loaders = {
            "train": DataLoader(
                datasets["train"],
                batch_size=int(data_config["batch_size"]),
                shuffle=bool(data_config.get("shuffle_train", True)),
                num_workers=resolved_num_workers,
                persistent_workers=resolved_num_workers > 0,
                collate_fn=collate_windows,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=int(data_config["batch_size"]),
                shuffle=False,
                num_workers=resolved_num_workers,
                persistent_workers=resolved_num_workers > 0,
                collate_fn=collate_windows,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=int(data_config["batch_size"]),
                shuffle=False,
                num_workers=resolved_num_workers,
                persistent_workers=resolved_num_workers > 0,
                collate_fn=collate_windows,
            ),
        }

        return {
            "dataset_name": "smd",
            "parser": parser,
            "scaler": scaler,
            "raw_sequences": cleaned_sequences,
            "scaled_sequences": scaled_sequences,
            "datasets": datasets,
            "loaders": loaders,
        }


def build_smd_dataset_bundle(data_config: dict[str, Any]) -> dict[str, Any]:
    # This is the registry-facing entrypoint used by the active scripts.
    return SMDDatasetBuilder().build(data_config)


def build_smd_dataloaders(data_config: dict[str, Any]) -> dict[str, Any]:
    return build_smd_dataset_bundle(data_config)
