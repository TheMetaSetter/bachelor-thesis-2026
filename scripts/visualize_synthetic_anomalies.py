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
from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.data.loaders import build_smd_dataset_bundle


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
    anomaly_mask = augmented_batch["synthetic_anomaly_mask"][sample_index].detach().cpu()
    metadata = augmented_batch["augmentation_metadata"][sample_index]

    channels_to_plot = min(3, clean_window.shape[1])
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    for channel_index in range(channels_to_plot):
        axes[0].plot(clean_window[:, channel_index].numpy(), label=f"channel {channel_index}")
        axes[1].plot(augmented_window[:, channel_index].numpy(), label=f"channel {channel_index}")

    axes[0].set_title("Clean window")
    axes[1].set_title(
        (
            f"Augmented window: {metadata['anomaly_family']} "
            f"(index={metadata.get('anomaly_family_index')}) "
            f"channels={metadata['affected_channels']}"
        )
    )
    axes[2].imshow(anomaly_mask.unsqueeze(0).numpy(), aspect="auto", cmap="Reds")
    axes[2].set_title("Synthetic anomaly mask")
    axes[2].set_yticks([])

    for axis in axes[:2]:
        axis.legend(loc="upper right")
        if metadata["start_index"] is not None and metadata["end_index"] is not None:
            axis.axvspan(metadata["start_index"], metadata["end_index"], color="red", alpha=0.15)

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
    data_bundle = build_dataset(experiment_config["data"]["dataset_name"], experiment_config["data"])
    return next(iter(data_bundle["loaders"]["train"]))


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
    if args.anomaly_family is None:
        injector = SyntheticAnomalyInjector(anomaly_probability=1.0)
    else:
        injector = SyntheticAnomalyInjector(
            anomaly_probability=1.0,
            anomaly_families=(args.anomaly_family,),
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
