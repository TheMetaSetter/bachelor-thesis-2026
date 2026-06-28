from __future__ import annotations

from typing import Any

import numpy as np


def describe_label_regime(point_labels: np.ndarray | list[int]) -> str:
    label_array = np.asarray(point_labels, dtype=np.int64).reshape(-1)
    unique_labels = sorted(np.unique(label_array).tolist())
    if unique_labels == [0]:
        return "all_zero"
    if unique_labels == [1]:
        return "all_one"
    return "mixed"


def summarize_split_point_labels(
    split_sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    concatenated_labels: list[np.ndarray] = []
    for sequence in split_sequences:
        point_labels = sequence["point_labels"]
        if point_labels is None:
            continue
        concatenated_labels.append(point_labels.detach().cpu().numpy())

    if not concatenated_labels:
        return {
            "label_regime": "unlabeled",
            "n_pos": 0,
            "n_neg": 0,
            "positive_ratio": 0.0,
        }

    label_array = np.concatenate(concatenated_labels, axis=0)
    n_pos = int(np.count_nonzero(label_array))
    n_total = int(label_array.shape[0])
    return {
        "label_regime": describe_label_regime(label_array),
        "n_pos": n_pos,
        "n_neg": int(n_total - n_pos),
        "positive_ratio": 0.0 if n_total == 0 else float(n_pos / n_total),
    }


def validate_benchmark_test_labels(
    *,
    dataset_name: str,
    split_sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    label_summary = summarize_split_point_labels(split_sequences)
    if label_summary["label_regime"] != "mixed":
        raise ValueError(
            f"{dataset_name} benchmark test split must contain both normal and "
            "anomalous timesteps on the reconstructed timeline."
        )
    return label_summary
