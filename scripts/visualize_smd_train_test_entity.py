from __future__ import annotations

"""Visualize one SMD entity with separate train and test panels."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "ServerMachineDataset"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "documents"
    / "logs"
    / "07-04-2026"
    / "research"
    / "smd-machine-3-9-train-test.png"
)


def labels_to_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")

    spans: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, value in enumerate(labels.astype(int).tolist()):
        if value != 0 and start_index is None:
            start_index = index
        elif value == 0 and start_index is not None:
            spans.append((start_index, index))
            start_index = None
    if start_index is not None:
        spans.append((start_index, int(labels.shape[0])))
    return spans


def load_smd_entity_series(
    dataset_root: str | Path,
    entity_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    train_path = root / "train" / f"{entity_id}.txt"
    test_path = root / "test" / f"{entity_id}.txt"
    label_path = root / "test_label" / f"{entity_id}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing test_label file: {label_path}")

    train_values = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test_values = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(label_path, delimiter=",", dtype=np.int64)

    if train_values.ndim != 2:
        raise ValueError("train values must be two-dimensional [time, channel]")
    if test_values.ndim != 2:
        raise ValueError("test values must be two-dimensional [time, channel]")
    if test_labels.ndim != 1:
        raise ValueError("test labels must be one-dimensional [time]")
    if test_values.shape[0] != test_labels.shape[0]:
        raise ValueError("test sequence length must match test_label length")

    return train_values, test_values, test_labels


def save_smd_train_test_visualization(
    *,
    train_values: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    entity_id: str,
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    color_min = float(
        np.percentile(np.concatenate([train_values, test_values], axis=0), 1)
    )
    color_max = float(
        np.percentile(np.concatenate([train_values, test_values], axis=0), 99)
    )
    anomaly_spans = labels_to_spans(test_labels)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        constrained_layout=True,
        sharey=True,
    )

    train_image = axes[0].imshow(
        train_values.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        interpolation="nearest",
    )
    axes[0].set_title(f"{entity_id} | train")
    axes[0].set_ylabel("Channel")

    axes[1].imshow(
        test_values.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=color_min,
        vmax=color_max,
        interpolation="nearest",
    )
    for start_index, end_index in anomaly_spans:
        axes[1].axvspan(start_index, end_index, color="red", alpha=0.18)
    axes[1].set_title(
        f"{entity_id} | test | anomaly_spans={len(anomaly_spans)} | anomalous_points={int(test_labels.sum())}"
    )
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Channel")

    colorbar = figure.colorbar(train_image, ax=axes, shrink=0.92)
    colorbar.set_label("Normalized sensor value")

    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--entity-id", default="machine-3-9")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    train_values, test_values, test_labels = load_smd_entity_series(
        dataset_root=args.dataset_root,
        entity_id=args.entity_id,
    )
    output_path = save_smd_train_test_visualization(
        train_values=train_values,
        test_values=test_values,
        test_labels=test_labels,
        entity_id=args.entity_id,
        output_path=args.output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
