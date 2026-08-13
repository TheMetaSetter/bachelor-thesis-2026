from __future__ import annotations

"""Visualize SMD train/test sequences using the locked benchmark data path."""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.smd import SMDDatasetParser  # noqa: E402
from src.data.loaders import WindowDataset  # noqa: E402
from src.data.scalers import SequenceStandardScaler  # noqa: E402


ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "documents"
    / "logs"
    / "08-09-2026"
    / "research"
    / "smd-benchmark-train-test-line-w20.png"
)


def _load_data_config(entity_id: str, config_dir: Path) -> dict[str, Any]:
    config_path = (
        config_dir / f"smd_benchmark_{entity_id.replace('-', '_')}_window20.yaml"
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in benchmark config: {config_path}")
    return config


def _load_entity_payload(
    entity_id: str,
    config_dir: Path,
) -> dict[str, Any]:
    config = _load_data_config(entity_id, config_dir)
    parser = SMDDatasetParser(
        root_dir=PROJECT_ROOT / config["root_dir"],
        validation_split_ratio=float(config["validation_split_ratio"]),
        entity_ids=[entity_id],
    )
    parsed_splits = parser.parse()
    scaler = SequenceStandardScaler()
    scaler.fit(parsed_splits["train"])
    scaled_splits = {
        split_name: scaler.transform_sequences(sequences)
        for split_name, sequences in parsed_splits.items()
    }
    train_sequence = scaled_splits["train"][0]
    test_sequence = scaled_splits["test"][0]
    train_dataset = WindowDataset(
        sequences=[train_sequence],
        window_size=int(config["window_size"]),
        stride=int(config["train_stride"]),
    )
    test_dataset = WindowDataset(
        sequences=[test_sequence],
        window_size=int(config["window_size"]),
        stride=int(config["test_stride"]),
    )
    return {
        "entity_id": entity_id,
        "config": config,
        "train_values": train_sequence["x"].detach().cpu().numpy(),
        "test_values": test_sequence["x"].detach().cpu().numpy(),
        "test_labels": test_sequence["point_labels"].detach().cpu().numpy(),
        "train_windows": len(train_dataset),
        "test_windows": len(test_dataset),
        "validation_points": int(parsed_splits["val"][0]["x"].shape[0]),
    }


def _labels_to_spans(labels: np.ndarray) -> list[tuple[int, int]]:
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


def _line_limits(payloads: list[dict[str, Any]]) -> tuple[float, float]:
    values = np.concatenate(
        [payload["train_values"].ravel() for payload in payloads]
        + [payload["test_values"].ravel() for payload in payloads]
    )
    absolute_limit = float(np.nanpercentile(np.abs(values), 99.0))
    absolute_limit = min(max(absolute_limit, 1.0), 5.0)
    return -absolute_limit, absolute_limit


def _plot_entity(
    axis_train: Any,
    axis_test: Any,
    payload: dict[str, Any],
    line_min: float,
    line_max: float,
) -> None:
    entity_id = payload["entity_id"]
    config = payload["config"]
    train_values = payload["train_values"]
    test_values = payload["test_values"]
    train_time = np.arange(train_values.shape[0])
    test_time = np.arange(test_values.shape[0])

    axis_train.plot(
        train_time,
        train_values,
        color="tab:blue",
        alpha=0.16,
        linewidth=0.45,
    )
    axis_train.plot(
        train_time,
        np.mean(train_values, axis=1),
        color="black",
        linewidth=1.0,
    )
    axis_train.set_title(
        f"{entity_id} | benchmark train\n"
        f"points={len(payload['train_values']):,} "
        f"| w={config['window_size']} | stride={config['train_stride']} "
        f"| windows={payload['train_windows']:,}",
        fontsize=9,
        pad=2,
    )
    axis_train.set_ylabel("Standardized value")
    axis_train.set_xlabel("Time index")
    axis_train.set_ylim(line_min, line_max)
    axis_train.grid(alpha=0.18, linewidth=0.4)

    axis_test.plot(
        test_time,
        test_values,
        color="tab:blue",
        alpha=0.16,
        linewidth=0.45,
    )
    (mean_line,) = axis_test.plot(
        test_time,
        np.mean(test_values, axis=1),
        color="black",
        linewidth=1.0,
        label="channel mean",
    )
    for start_index, end_index in _labels_to_spans(payload["test_labels"]):
        axis_test.axvspan(start_index, end_index, color="black", alpha=0.18)
    window_stride = int(config["test_stride"])
    for boundary in range(0, len(payload["test_values"]), window_stride * 100):
        axis_test.axvline(boundary, color="black", alpha=0.20, linewidth=0.35)
    axis_test.set_title(
        f"{entity_id} | benchmark test\n"
        f"points={len(payload['test_values']):,} "
        f"| w={config['window_size']} | stride={config['test_stride']} "
        f"| windows={payload['test_windows']:,} "
        f"| anomaly_points={int(payload['test_labels'].sum()):,}",
        fontsize=9,
        pad=2,
    )
    axis_test.set_xlabel("Time index")
    axis_test.set_ylabel("Standardized value")
    axis_test.set_ylim(line_min, line_max)
    axis_test.grid(alpha=0.18, linewidth=0.4)
    axis_test.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="tab:blue",
                alpha=0.55,
                linewidth=1.0,
                label="38 channels",
            ),
            mean_line,
            Patch(facecolor="black", alpha=0.18, label="test_label anomaly"),
        ],
        loc="upper right",
        fontsize=8,
    )


def save_visualization(
    payloads: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    line_min, line_max = _line_limits(payloads)
    figure, axes = plt.subplots(
        len(payloads),
        2,
        figsize=(18, 5.2 * len(payloads)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, payload in enumerate(payloads):
        _plot_entity(
            axes[row_index, 0],
            axes[row_index, 1],
            payload,
            line_min,
            line_max,
        )
    figure.suptitle(
        "SMD benchmark train/test line plot | w=20 | train-fit standardization | "
        "validation tail omitted | 38 channels + channel mean",
        fontsize=12,
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs" / "data"))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    payloads = [
        _load_entity_payload(entity_id, Path(args.config_dir))
        for entity_id in ENTITY_IDS
    ]
    output_path = save_visualization(payloads, Path(args.output_path))
    print(output_path)
    for payload in payloads:
        print(
            f"{payload['entity_id']}: train_points={len(payload['train_values'])}, "
            f"validation_points={payload['validation_points']}, "
            f"test_points={len(payload['test_values'])}, "
            f"train_windows={payload['train_windows']}, "
            f"test_windows={payload['test_windows']}"
        )


if __name__ == "__main__":
    main()
