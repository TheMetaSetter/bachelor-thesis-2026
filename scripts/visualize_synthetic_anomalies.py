from __future__ import annotations

"""Visualization helper for inspecting one synthetic anomaly family at a time."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# Add src to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.core.registry import build_dataset, register_dataset
from src.data.collate import collate_windows
from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.data.loaders import (
    build_anomaly_archive_dataset_bundle,
    build_smd_dataset_bundle,
)


def select_channels_to_plot(
    clean_window: torch.Tensor,
    metadata: dict,
    num_channels: int,
    affected_channels: list[int],
    max_affected_channels: int = 3,
    max_total_channels: int = 5,
    min_boundary_delta: float = 0.01,
    max_boundary_delta: float = 0.1,
) -> list[int]:
    visible_affected_channels = select_visible_affected_channels(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=num_channels,
        affected_channels=affected_channels,
        min_boundary_delta=min_boundary_delta,
        max_boundary_delta=max_boundary_delta,
    )

    selected_channels: list[int] = []
    seen_channels: set[int] = set()

    for channel_index in visible_affected_channels:
        if channel_index in seen_channels:
            continue
        selected_channels.append(channel_index)
        seen_channels.add(channel_index)
        if len(selected_channels) >= min(max_affected_channels, max_total_channels):
            break

    if len(selected_channels) < min(max_total_channels, num_channels):
        for channel_index in range(num_channels):
            if channel_index in seen_channels:
                continue
            selected_channels.append(channel_index)
            seen_channels.add(channel_index)
            if len(selected_channels) >= min(max_total_channels, num_channels):
                break

    return selected_channels


def select_visible_affected_channels(
    clean_window: torch.Tensor,
    metadata: dict,
    num_channels: int,
    affected_channels: list[int],
    min_boundary_delta: float = 0.01,
    max_boundary_delta: float = 0.1,
) -> list[int]:
    def boundary_delta(channel_index: int) -> float:
        start_index = metadata["start_index"]
        end_index = metadata["end_index"]
        if start_index is None or end_index is None:
            return float("inf")
        if start_index < 0 or end_index <= start_index:
            return float("inf")
        if end_index > clean_window.shape[0]:
            return float("inf")
        if channel_index < 0 or channel_index >= clean_window.shape[1]:
            return float("inf")
        start_value = clean_window[start_index, channel_index].item()
        end_value = clean_window[end_index - 1, channel_index].item()
        return abs(float(end_value - start_value))

    visible_affected_channels = [
        channel_index
        for channel_index in affected_channels
        if 0 <= channel_index < num_channels
        and min_boundary_delta < boundary_delta(channel_index) <= max_boundary_delta
    ]
    visible_affected_channels.sort(
        key=lambda channel_index: boundary_delta(channel_index), reverse=True
    )
    return visible_affected_channels


def plot_synthetic_anomaly_mask(axis, anomaly_mask: torch.Tensor) -> None:
    mask_length = int(anomaly_mask.shape[0])
    axis.imshow(
        anomaly_mask.unsqueeze(0).numpy(),
        aspect="auto",
        cmap="Reds",
        interpolation="nearest",
        origin="lower",
        extent=(0, mask_length, 0, 1),
    )
    axis.set_xlim(0, mask_length)
    axis.set_ylim(0, 1)
    axis.set_title("Synthetic anomaly mask")
    axis.set_yticks([])


def save_synthetic_anomaly_visualization(
    clean_batch: dict,
    augmented_batch: dict,
    output_path: str | Path,
    sample_index: int = 0,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    clean_window = clean_batch["x"][sample_index].detach().cpu()
    augmented_window = augmented_batch["x"][sample_index].detach().cpu()
    anomaly_mask = (
        augmented_batch["synthetic_anomaly_mask"][sample_index].detach().cpu()
    )
    metadata = augmented_batch["augmentation_metadata"][sample_index]
    visible_affected_channels = select_visible_affected_channels(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=clean_window.shape[1],
        affected_channels=list(metadata.get("affected_channels", [])),
    )

    channels_to_plot = select_channels_to_plot(
        clean_window=clean_window,
        metadata=metadata,
        num_channels=clean_window.shape[1],
        affected_channels=list(metadata.get("affected_channels", [])),
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    affected_channels = set(visible_affected_channels)
    for channel_index in channels_to_plot:
        line_label = (
            f"channel {channel_index} (affected)"
            if channel_index in affected_channels
            else f"channel {channel_index} (context)"
        )
        axes[0].plot(
            clean_window[:, channel_index].numpy(),
            label=line_label,
        )
        axes[1].plot(
            augmented_window[:, channel_index].numpy(),
            label=line_label,
        )

    axes[0].set_title("Clean window")
    axes[1].set_title(
        (
            f"Augmented window: {metadata['anomaly_family']} "
            f"(index={metadata.get('anomaly_family_index')}) "
            f"segment_length={metadata.get('segment_length')} "
            f"boost={metadata.get('visibility_boost_factor')} "
            f"visible_channels={channels_to_plot} "
            f"filter=(0.01, 0.1]"
        )
    )
    plot_synthetic_anomaly_mask(axes[2], anomaly_mask)

    for axis in axes[:2]:
        axis.legend(loc="upper right")
        if metadata["start_index"] is not None and metadata["end_index"] is not None:
            axis.axvspan(
                metadata["start_index"], metadata["end_index"], color="red", alpha=0.15
            )

    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    return output_file


def build_demo_batch(experiment_config_path: str | None = None) -> dict:
    if experiment_config_path is None:
        return {
            "x": torch.randn(4, 100, 3),
            "point_labels": torch.zeros(4, 100, dtype=torch.long),
            "mask": None,
            "timestamps": None,
            "meta": [{"entity_id": f"demo-{index}"} for index in range(4)],
        }

    experiment_config = load_experiment_config(experiment_config_path)
    register_dataset("smd", build_smd_dataset_bundle)
    register_dataset("anomaly_archive", build_anomaly_archive_dataset_bundle)
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"], experiment_config["data"]
    )
    train_dataset = data_bundle["datasets"]["train"]
    num_demo_windows = min(4, len(train_dataset))
    demo_windows = [train_dataset[index] for index in range(num_demo_windows)]
    return collate_windows(demo_windows)


def build_visualization_injector(
    experiment_config_path: str | None,
    anomaly_family: str | None,
) -> SyntheticAnomalyInjector:
    if experiment_config_path is None:
        if anomaly_family is None:
            return SyntheticAnomalyInjector(anomaly_probability=1.0)
        return SyntheticAnomalyInjector(
            anomaly_probability=1.0,
            anomaly_families=(anomaly_family,),
        )

    experiment_config = load_experiment_config(experiment_config_path)
    task_config = experiment_config["task"]
    configured_families = tuple(task_config.get("anomaly_families", REDLAMP_ANOMALY_FAMILIES))
    selected_families = configured_families if anomaly_family is None else (anomaly_family,)
    return SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        min_segment_fraction=float(task_config["min_segment_fraction"]),
        max_segment_fraction=float(task_config["max_segment_fraction"]),
        spike_scale=float(task_config["spike_scale"]),
        anomaly_visibility_boost=float(task_config.get("anomaly_visibility_boost", 1.5)),
        anomaly_families=selected_families,
        classification_label_mode=str(task_config.get("classification_label_mode", "redlamp_multiclass")),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument(
        "--output-path",
        default="outputs/synthetic_anomaly_visualization/sample.png",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--anomaly-family",
        choices=list(REDLAMP_ANOMALY_FAMILIES),
        default=None,
        help="Force one anomaly family for deterministic inspection.",
    )
    args = parser.parse_args()

    clean_batch = build_demo_batch(args.experiment_config)
    injector = build_visualization_injector(
        experiment_config_path=args.experiment_config,
        anomaly_family=args.anomaly_family,
    )
    augmented_batch = injector.augment_batch(clean_batch)
    saved_path = save_synthetic_anomaly_visualization(
        clean_batch=clean_batch,
        augmented_batch=augmented_batch,
        output_path=args.output_path,
        sample_index=args.sample_index,
    )
    print(saved_path)


if __name__ == "__main__":
    main()
