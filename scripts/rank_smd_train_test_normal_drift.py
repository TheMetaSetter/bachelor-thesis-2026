from __future__ import annotations

"""Rank SMD entities by drift between raw train and normal-only test points.

This script stays intentionally SMD-specific and outside the generic runtime.
Its only job is selection analysis for experiment setup, not model training.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.anomaly_archive_kl import estimate_histogram_kl_divergence
from src.data.download import get_smd_dataset_root


@dataclass(frozen=True)
class SMDNormalDriftRankingRow:
    entity_id: str
    num_train_points: int
    num_test_normal_points: int
    num_channels: int
    mean_kl_test_to_train: float
    max_kl_test_to_train: float
    top5_mean_kl_test_to_train: float
    per_channel_kl_test_to_train: list[float]


def _load_feature_matrix(file_path: Path) -> np.ndarray:
    feature_matrix = np.loadtxt(file_path, delimiter=",", dtype=np.float64)
    if feature_matrix.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix in {file_path}")
    return feature_matrix


def _load_label_vector(file_path: Path) -> np.ndarray:
    label_vector = np.loadtxt(file_path, delimiter=",", dtype=np.int64)
    label_vector = np.asarray(label_vector).reshape(-1)
    return label_vector


def _iter_smd_entity_ids(dataset_root: Path) -> Iterable[str]:
    for train_file in sorted((dataset_root / "train").glob("*.txt")):
        yield train_file.stem


def _compute_channelwise_kl_test_to_train(
    *,
    train_matrix: np.ndarray,
    test_normal_matrix: np.ndarray,
    bins: int,
    smoothing: float,
) -> list[float]:
    if train_matrix.shape[1] != test_normal_matrix.shape[1]:
        raise ValueError(
            "Train and test matrices must have the same number of channels"
        )

    per_channel_scores: list[float] = []
    for channel_index in range(train_matrix.shape[1]):
        per_channel_scores.append(
            estimate_histogram_kl_divergence(
                left_values=test_normal_matrix[:, channel_index],
                right_values=train_matrix[:, channel_index],
                bins=bins,
                smoothing=smoothing,
            )
        )
    return per_channel_scores


def build_smd_normal_drift_ranking(
    *,
    root_dir: str | Path,
    bins: int = 64,
    smoothing: float = 1.0e-12,
) -> list[SMDNormalDriftRankingRow]:
    dataset_root = get_smd_dataset_root(root_dir)
    ranking_rows: list[SMDNormalDriftRankingRow] = []

    for entity_id in _iter_smd_entity_ids(dataset_root):
        train_matrix = _load_feature_matrix(dataset_root / "train" / f"{entity_id}.txt")
        test_matrix = _load_feature_matrix(dataset_root / "test" / f"{entity_id}.txt")
        test_labels = _load_label_vector(
            dataset_root / "test_label" / f"{entity_id}.txt"
        )

        if test_matrix.shape[0] != test_labels.shape[0]:
            raise ValueError(
                f"Test matrix length and label length disagree for {entity_id}"
            )

        normal_test_mask = test_labels == 0
        test_normal_matrix = test_matrix[normal_test_mask]
        if test_normal_matrix.shape[0] == 0:
            raise ValueError(f"Entity {entity_id} has no normal test points")

        per_channel_kl_scores = _compute_channelwise_kl_test_to_train(
            train_matrix=train_matrix,
            test_normal_matrix=test_normal_matrix,
            bins=bins,
            smoothing=smoothing,
        )
        sorted_scores = sorted(per_channel_kl_scores, reverse=True)
        topk = min(5, len(sorted_scores))
        ranking_rows.append(
            SMDNormalDriftRankingRow(
                entity_id=entity_id,
                num_train_points=int(train_matrix.shape[0]),
                num_test_normal_points=int(test_normal_matrix.shape[0]),
                num_channels=int(train_matrix.shape[1]),
                mean_kl_test_to_train=float(np.mean(per_channel_kl_scores)),
                max_kl_test_to_train=float(np.max(per_channel_kl_scores)),
                top5_mean_kl_test_to_train=float(np.mean(sorted_scores[:topk])),
                per_channel_kl_test_to_train=per_channel_kl_scores,
            )
        )

    ranking_rows.sort(
        key=lambda row: (
            row.mean_kl_test_to_train,
            row.top5_mean_kl_test_to_train,
            row.max_kl_test_to_train,
            row.entity_id,
        ),
        reverse=True,
    )
    return ranking_rows


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/ServerMachineDataset")
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--smoothing", type=float, default=1.0e-12)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    ranking_rows = build_smd_normal_drift_ranking(
        root_dir=args.root_dir,
        bins=args.bins,
        smoothing=args.smoothing,
    )
    serializable_rows = [asdict(ranking_row) for ranking_row in ranking_rows]

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(serializable_rows, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(serializable_rows, indent=2))


if __name__ == "__main__":
    main()
