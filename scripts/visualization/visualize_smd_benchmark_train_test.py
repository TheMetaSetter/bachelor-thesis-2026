from __future__ import annotations

"""Create one full-timeline train/test plot for each selected SMD entity."""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.smd import SMDDatasetParser  # noqa: E402


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "ServerMachineDataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "documents" / "logs" / "08-09-2026" / "research"
SELECTED_ENTITY_IDS = ("machine-1-6", "machine-3-4", "machine-3-9")
NUM_CHANNEL_GROUPS = 3
TIME_START = 10_000
TIME_END = 15_000
CHANNEL_COLORS = ("tab:blue", "tab:orange", "tab:green")


def _load_entity_payloads(
    dataset_root: Path,
    entity_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    parser = SMDDatasetParser(
        root_dir=dataset_root,
        validation_split_ratio=0.2,
        entity_ids=list(entity_ids),
    )
    parsed_splits = parser.parse()
    test_by_entity = {
        sequence["meta"]["entity_id"]: sequence for sequence in parsed_splits["test"]
    }
    payloads: list[dict[str, Any]] = []
    for train_sequence in parsed_splits["train"]:
        entity_id = str(train_sequence["meta"]["entity_id"])
        full_train_values = train_sequence["x"].detach().cpu().numpy()
        full_test_values = test_by_entity[entity_id]["x"].detach().cpu().numpy()
        channel_mean = np.mean(full_train_values, axis=0)
        channel_std = np.std(full_train_values, axis=0, ddof=0)
        ordered_channels = np.argsort(channel_mean)
        scale_groups = np.array_split(ordered_channels, NUM_CHANNEL_GROUPS)
        channel_indices: list[int] = []
        channel_labels: list[str] = []
        for group_index, group_channels in enumerate(scale_groups, start=1):
            selected_position = int(np.argmax(channel_std[group_channels]))
            selected_channel = int(group_channels[selected_position])
            channel_indices.append(selected_channel)
            channel_labels.append(f"scale group {group_index} | ch {selected_channel}")
        payloads.append(
            {
                "entity_id": entity_id,
                "train_std": float(np.std(full_train_values, ddof=0)),
                "channel_indices": channel_indices,
                "channel_labels": channel_labels,
                "train_values": full_train_values[TIME_START:TIME_END][
                    :, channel_indices
                ],
                "test_values": full_test_values[TIME_START:TIME_END][
                    :, channel_indices
                ],
            }
        )
    return payloads


def _line_limits(payload: dict[str, Any]) -> tuple[float, float]:
    plotted_values = np.concatenate(
        [payload["train_values"].ravel(), payload["test_values"].ravel()]
    )
    lower = float(np.nanmin(plotted_values))
    upper = float(np.nanmax(plotted_values))
    padding = max((upper - lower) * 0.05, 0.1)
    return lower - padding, upper + padding


def _plot_split(
    axis: Any,
    payload: dict[str, Any],
    split_name: str,
    line_limits: tuple[float, float],
    show_legend: bool,
) -> None:
    values = payload[f"{split_name}_values"]
    time_indices = np.arange(TIME_START, TIME_START + values.shape[0])
    for channel_position, channel_label in enumerate(payload["channel_labels"]):
        axis.plot(
            time_indices,
            values[:, channel_position],
            color=CHANNEL_COLORS[channel_position],
            linewidth=0.7,
            label=channel_label if show_legend else None,
        )
    axis.set_title(f"{payload['entity_id']} | {split_name}", loc="left")
    axis.set_ylabel("Raw value")
    axis.set_xlim(TIME_START, TIME_END)
    axis.set_ylim(*line_limits)
    axis.grid(alpha=0.2, linewidth=0.4)
    if show_legend:
        axis.legend(loc="upper right")


def save_entity_visualization(
    payload: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=False,
        constrained_layout=True,
    )
    line_limits = _line_limits(payload)
    _plot_split(axes[0], payload, "train", line_limits, show_legend=True)
    _plot_split(axes[1], payload, "test", line_limits, show_legend=False)
    axes[1].set_xlabel("Time index")
    output_path = output_dir / f"smd-benchmark-train-test-{payload['entity_id']}.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    selected_payloads = _load_entity_payloads(
        Path(args.dataset_root),
        SELECTED_ENTITY_IDS,
    )
    output_dir = Path(args.output_dir)

    for payload in selected_payloads:
        print(
            f"entity={payload['entity_id']}, "
            f"channels={payload['channel_labels']}, "
            f"selected_train_std={payload['train_std']:.6f}"
        )
        print(save_entity_visualization(payload, output_dir))


if __name__ == "__main__":
    main()
