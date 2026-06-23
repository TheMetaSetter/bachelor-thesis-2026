from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_tensor
from src.core.contracts import validate_raw_sequence
from src.data.base import BaseSequenceParser


class SMDDatasetParser(BaseSequenceParser):
    def __init__(
        self,
        root_dir: str | Path,
        validation_split_ratio: float = 0.2,
        entity_ids: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.validation_split_ratio = validation_split_ratio
        self.entity_ids = entity_ids
        self.train_dir = self.root_dir / "train"
        self.test_dir = self.root_dir / "test"
        self.test_label_dir = self.root_dir / "test_label"

    def _load_feature_matrix(self, file_path: Path) -> torch.Tensor:
        loaded_array = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        return torch.from_numpy(loaded_array)

    def _load_label_vector(self, file_path: Path) -> torch.Tensor:
        loaded_array = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        return torch.from_numpy(loaded_array).long()

    def _build_raw_sequence(
        self,
        x_tensor: torch.Tensor,
        entity_id: str,
        split: str,
        point_labels: torch.Tensor | None,
    ) -> dict[str, Any]:
        raw_sequence = {
            "x": x_tensor,
            "point_labels": point_labels,
            "mask": None,
            "timestamps": None,
            "meta": {
                "dataset_name": "smd",
                "entity_id": entity_id,
                "split": split,
                "series_id": f"smd:{split}:{entity_id}",
                "num_channels": int(x_tensor.shape[1]),
                "sequence_length": int(x_tensor.shape[0]),
                "source_sequence_length": int(x_tensor.shape[0]),
            },
        }
        validate_raw_sequence(raw_sequence)
        return raw_sequence

    def parse(self) -> dict[str, list[dict[str, Any]]]:
        train_files = sorted(self.train_dir.glob("*.txt"))
        test_files = sorted(self.test_dir.glob("*.txt"))
        label_files = sorted(self.test_label_dir.glob("*.txt"))
        train_files_by_entity = {
            train_file.stem: train_file for train_file in train_files
        }
        test_files_by_entity = {test_file.stem: test_file for test_file in test_files}
        label_files_by_entity = {
            label_file.stem: label_file for label_file in label_files
        }
        console_print(
            "DATA",
            "Parsing SMD split files",
            root_dir=self.root_dir,
            train_count=len(train_files),
            test_count=len(test_files),
            label_count=len(label_files),
            entity_ids=self.entity_ids,
        )

        if self.entity_ids is None and (
            len(train_files) != 28 or len(test_files) != 28 or len(label_files) != 28
        ):
            raise ValueError("SMD parser expected 28 machine files per split")

        if self.entity_ids is None:
            selected_entity_ids = sorted(train_files_by_entity.keys())
        else:
            selected_entity_ids = list(self.entity_ids)
            if not selected_entity_ids:
                raise ValueError(
                    "SMD parser requires at least one entity_id when filtering is enabled"
                )
            for entity_id in selected_entity_ids:
                if entity_id not in train_files_by_entity:
                    raise ValueError(
                        f"Requested SMD entity is missing from train split: {entity_id}"
                    )
                if entity_id not in test_files_by_entity:
                    raise ValueError(
                        f"Requested SMD entity is missing from test split: {entity_id}"
                    )
                if entity_id not in label_files_by_entity:
                    raise ValueError(
                        f"Requested SMD entity is missing from test_label split: {entity_id}"
                    )

        train_sequences: list[dict[str, Any]] = []
        val_sequences: list[dict[str, Any]] = []
        test_sequences: list[dict[str, Any]] = []

        for entity_id in selected_entity_ids:
            train_file = train_files_by_entity[entity_id]
            test_file = test_files_by_entity[entity_id]
            label_file = label_files_by_entity[entity_id]

            train_tensor = self._load_feature_matrix(train_file)
            test_tensor = self._load_feature_matrix(test_file)
            test_labels = self._load_label_vector(label_file)
            console_print(
                "DATA",
                "Loaded SMD entity files",
                entity_id=entity_id,
                train_tensor=summarize_tensor(train_tensor),
                test_tensor=summarize_tensor(test_tensor),
                test_labels=summarize_tensor(test_labels),
            )
            if test_tensor.shape[0] != test_labels.shape[0]:
                raise ValueError(
                    f"Test labels do not match test sequence length for {entity_id}"
                )

            validation_length = max(
                1, int(train_tensor.shape[0] * self.validation_split_ratio)
            )
            train_cutoff = train_tensor.shape[0] - validation_length
            if train_cutoff < 1:
                raise ValueError(
                    f"Validation split ratio leaves no training data for {entity_id}"
                )

            train_sequences.append(
                self._build_raw_sequence(
                    x_tensor=train_tensor[:train_cutoff].clone(),
                    entity_id=entity_id,
                    split="train",
                    point_labels=torch.zeros(train_cutoff, dtype=torch.long),
                )
            )
            val_sequences.append(
                self._build_raw_sequence(
                    x_tensor=train_tensor[train_cutoff:].clone(),
                    entity_id=entity_id,
                    split="val",
                    point_labels=torch.zeros(validation_length, dtype=torch.long),
                )
            )
            test_sequences.append(
                self._build_raw_sequence(
                    x_tensor=test_tensor.clone(),
                    entity_id=entity_id,
                    split="test",
                    point_labels=test_labels.clone(),
                )
            )

        parsed_splits = {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences,
        }
        console_print(
            "DATA",
            "Completed SMD parsing",
            selected_entity_ids=selected_entity_ids,
            train_sequences=len(train_sequences),
            val_sequences=len(val_sequences),
            test_sequences=len(test_sequences),
        )
        return parsed_splits


def compute_smd_test_window_anomaly_rate(
    *,
    root_dir: str | Path,
    window_size: int,
    stride: int,
    entity_ids: list[str] | None,
    use_all_entities: bool,
) -> float:
    if use_all_entities:
        resolved_test_dir = Path(root_dir) / "test"
        resolved_entity_ids = sorted(
            file_path.stem for file_path in resolved_test_dir.glob("*.txt")
        )
    else:
        resolved_entity_ids = entity_ids
    parser = SMDDatasetParser(
        root_dir=root_dir,
        validation_split_ratio=0.2,
        entity_ids=resolved_entity_ids,
    )
    parsed_splits = parser.parse()
    test_sequences = parsed_splits["test"]
    total_windows = 0
    anomalous_windows = 0

    for sequence in test_sequences:
        point_labels = sequence["point_labels"]
        if point_labels is None:
            continue
        sequence_length = int(point_labels.shape[0])
        if sequence_length < window_size:
            continue
        for start_index in range(0, sequence_length - window_size + 1, stride):
            end_index = start_index + window_size
            total_windows += 1
            window_has_anomaly = bool(
                torch.count_nonzero(point_labels[start_index:end_index]).item() > 0
            )
            if window_has_anomaly:
                anomalous_windows += 1

    if total_windows == 0:
        raise ValueError(
            "Cannot derive SMD test window anomaly rate because zero windows were generated"
        )
    anomaly_rate = anomalous_windows / total_windows
    console_print(
        "DATA",
        "Computed SMD test window anomaly rate",
        window_size=window_size,
        stride=stride,
        use_all_entities=use_all_entities,
        selected_entity_ids=resolved_entity_ids,
        total_windows=total_windows,
        anomalous_windows=anomalous_windows,
        anomaly_rate=anomaly_rate,
    )
    return float(anomaly_rate)
