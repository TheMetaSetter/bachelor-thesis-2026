from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.core.contracts import validate_raw_sequence
from src.data.base import BaseSequenceParser


class SMDDatasetParser(BaseSequenceParser):
    def __init__(self, root_dir: str | Path, validation_split_ratio: float = 0.2) -> None:
        self.root_dir = Path(root_dir)
        self.validation_split_ratio = validation_split_ratio
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
                "num_channels": int(x_tensor.shape[1]),
                "sequence_length": int(x_tensor.shape[0]),
            },
        }
        validate_raw_sequence(raw_sequence)
        return raw_sequence

    def parse(self) -> dict[str, list[dict[str, Any]]]:
        train_files = sorted(self.train_dir.glob("*.txt"))
        test_files = sorted(self.test_dir.glob("*.txt"))
        label_files = sorted(self.test_label_dir.glob("*.txt"))

        if len(train_files) != 28 or len(test_files) != 28 or len(label_files) != 28:
            raise ValueError("SMD parser expected 28 machine files per split")

        train_sequences: list[dict[str, Any]] = []
        val_sequences: list[dict[str, Any]] = []
        test_sequences: list[dict[str, Any]] = []

        for train_file, test_file, label_file in zip(train_files, test_files, label_files):
            entity_id = train_file.stem
            if test_file.stem != entity_id or label_file.stem != entity_id:
                raise ValueError(f"Mismatched SMD files for entity: {entity_id}")

            train_tensor = self._load_feature_matrix(train_file)
            test_tensor = self._load_feature_matrix(test_file)
            test_labels = self._load_label_vector(label_file)
            if test_tensor.shape[0] != test_labels.shape[0]:
                raise ValueError(f"Test labels do not match test sequence length for {entity_id}")

            validation_length = max(1, int(train_tensor.shape[0] * self.validation_split_ratio))
            train_cutoff = train_tensor.shape[0] - validation_length
            if train_cutoff < 1:
                raise ValueError(f"Validation split ratio leaves no training data for {entity_id}")

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

        return {
            "train": train_sequences,
            "val": val_sequences,
            "test": test_sequences,
        }

