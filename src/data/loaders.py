from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Dataset

from src.data.collate import collate_windows
from src.data.base import BaseDatasetBuilder
from src.data.datasets.smd import SMDDatasetParser
from src.data.scalers import SequenceStandardScaler


class WindowDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        window_size: int,
        stride: int,
        max_windows: int | None = None,
    ) -> None:
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
        parser = SMDDatasetParser(
            root_dir=data_config["root_dir"],
            validation_split_ratio=float(data_config["validation_split_ratio"]),
        )
        parsed_sequences = parser.parse()

        scaler = SequenceStandardScaler()
        scaler.fit(parsed_sequences["train"])
        scaled_sequences = {
            split_name: scaler.transform_sequences(split_sequences)
            for split_name, split_sequences in parsed_sequences.items()
        }

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
                num_workers=int(data_config.get("num_workers", 0)),
                collate_fn=collate_windows,
            ),
            "val": DataLoader(
                datasets["val"],
                batch_size=int(data_config["batch_size"]),
                shuffle=False,
                num_workers=int(data_config.get("num_workers", 0)),
                collate_fn=collate_windows,
            ),
            "test": DataLoader(
                datasets["test"],
                batch_size=int(data_config["batch_size"]),
                shuffle=False,
                num_workers=int(data_config.get("num_workers", 0)),
                collate_fn=collate_windows,
            ),
        }

        return {
            "dataset_name": "smd",
            "parser": parser,
            "scaler": scaler,
            "raw_sequences": parsed_sequences,
            "scaled_sequences": scaled_sequences,
            "datasets": datasets,
            "loaders": loaders,
        }


def build_smd_dataset_bundle(data_config: dict[str, Any]) -> dict[str, Any]:
    return SMDDatasetBuilder().build(data_config)


def build_smd_dataloaders(data_config: dict[str, Any]) -> dict[str, Any]:
    return build_smd_dataset_bundle(data_config)
