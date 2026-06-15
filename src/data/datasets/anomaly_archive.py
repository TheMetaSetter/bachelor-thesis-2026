from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_tensor
from src.core.contracts import validate_raw_sequence
from src.data.base import BaseSequenceParser


_ANOMALY_ARCHIVE_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>\d+)_UCR_Anomaly_(?P<series_name>.+)_(?P<start_index>\d+)"
    r"_(?P<anomaly_start_index>\d+)_(?P<anomaly_end_index>\d+)\.txt$"
)


class AnomalyArchiveDatasetParser(BaseSequenceParser):
    def __init__(
        self,
        file_path: str | Path,
        validation_split_ratio: float = 0.2,
        comparison_mode: str = "pre_vs_anomaly",
        inclusive_anomaly_end: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.validation_split_ratio = validation_split_ratio
        self.comparison_mode = comparison_mode
        self.inclusive_anomaly_end = inclusive_anomaly_end

    def _parse_filename(self) -> dict[str, int | str]:
        file_name = self.file_path.name
        match = _ANOMALY_ARCHIVE_FILENAME_PATTERN.match(file_name)
        if match is None:
            raise ValueError(
                "AnomalyArchive file name must follow "
                "<prefix>_UCR_Anomaly_<series>_<start>_<anomaly_start>_<anomaly_end>.txt"
            )
        return {
            "series_name": match.group("series_name"),
            "start_index": int(match.group("start_index")),
            "anomaly_start_index": int(match.group("anomaly_start_index")),
            "anomaly_end_index": int(match.group("anomaly_end_index")),
        }

    def _load_values(self) -> np.ndarray:
        if not self.file_path.exists():
            raise FileNotFoundError(f"AnomalyArchive file does not exist: {self.file_path}")
        loaded_values = np.fromstring(
            self.file_path.read_text(encoding="utf-8"),
            sep=" ",
            dtype=np.float32,
        )
        if loaded_values.size == 0:
            raise ValueError(f"No numeric values found in {self.file_path}")
        return loaded_values

    def _build_raw_sequence(
        self,
        *,
        values: np.ndarray,
        split: str,
        entity_id: str,
        point_labels: torch.Tensor | None,
    ) -> dict[str, Any]:
        value_tensor = torch.from_numpy(values.astype(np.float32, copy=False)).unsqueeze(1)
        raw_sequence = {
            "x": value_tensor,
            "point_labels": point_labels,
            "mask": None,
            "timestamps": None,
            "meta": {
                "dataset_name": "anomaly_archive",
                "entity_id": entity_id,
                "split": split,
                "series_name": entity_id,
                "source_file_name": self.file_path.name,
                "start_index": 0,
                "end_index": int(value_tensor.shape[0]),
                "num_channels": 1,
                "sequence_length": int(value_tensor.shape[0]),
            },
        }
        validate_raw_sequence(raw_sequence)
        return raw_sequence

    def parse(self) -> dict[str, list[dict[str, Any]]]:
        metadata = self._parse_filename()
        values = self._load_values()
        anomaly_start_index = int(metadata["anomaly_start_index"])
        anomaly_end_index = int(metadata["anomaly_end_index"])
        if anomaly_start_index <= 0 or anomaly_start_index >= values.size:
            raise ValueError("anomaly_start_index must lie within the loaded series")
        if anomaly_end_index <= anomaly_start_index:
            raise ValueError("anomaly_end_index must be greater than anomaly_start_index")

        if self.comparison_mode == "pre_vs_anomaly":
            train_region_values = values[:anomaly_start_index]
            anomaly_stop_index = (
                anomaly_end_index + 1 if self.inclusive_anomaly_end else anomaly_end_index
            )
            test_values = values[anomaly_start_index:anomaly_stop_index]
        elif self.comparison_mode == "pre_vs_post":
            train_region_values = values[:anomaly_start_index]
            test_values = values[anomaly_end_index:]
        else:
            raise ValueError(
                "comparison_mode must be either 'pre_vs_post' or 'pre_vs_anomaly'"
            )

        if train_region_values.size == 0:
            raise ValueError("Training region is empty after applying the annotation split")
        if test_values.size == 0:
            raise ValueError("Testing region is empty after applying the annotation split")

        validation_length = max(
            1, int(train_region_values.size * self.validation_split_ratio)
        )
        train_length = train_region_values.size - validation_length
        if train_length < 1:
            raise ValueError("Validation split ratio leaves no training data")

        train_values = train_region_values[:train_length]
        val_values = train_region_values[train_length:]

        train_sequences = [
            self._build_raw_sequence(
                values=train_values,
                split="train",
                entity_id=str(metadata["series_name"]),
                point_labels=torch.zeros(train_values.size, dtype=torch.long),
            )
        ]
        val_sequences = [
            self._build_raw_sequence(
                values=val_values,
                split="val",
                entity_id=str(metadata["series_name"]),
                point_labels=torch.zeros(val_values.size, dtype=torch.long),
            )
        ]
        test_point_labels = (
            torch.ones(test_values.size, dtype=torch.long)
            if self.comparison_mode == "pre_vs_anomaly"
            else torch.zeros(test_values.size, dtype=torch.long)
        )
        test_sequences = [
            self._build_raw_sequence(
                values=test_values,
                split="test",
                entity_id=str(metadata["series_name"]),
                point_labels=test_point_labels,
            )
        ]
        console_print(
            "DATA",
            "Completed AnomalyArchive parsing",
            file_path=self.file_path,
            train_tensor=summarize_tensor(train_sequences[0]["x"]),
            val_tensor=summarize_tensor(val_sequences[0]["x"]),
            test_tensor=summarize_tensor(test_sequences[0]["x"]),
            comparison_mode=self.comparison_mode,
        )
        return {"train": train_sequences, "val": val_sequences, "test": test_sequences}

