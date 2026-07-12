from __future__ import annotations

"""Short visual/statistical comparison for legacy vs visible synthetic profiles.

⸜(｡˃ ᵕ ˂ )⸝♡ Where this script fits

clean windows
  -> legacy synthetic profile
  -> visible synthetic profile
  -> JSON summary + side-by-side figure
  -> human selects the main benchmark profile
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from scripts.visualization.visualize_synthetic_anomalies import build_demo_batch
from src.data.collate import collate_windows
from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.data.download import resolve_repo_relative_path
from src.data.loaders import build_smd_dataset_bundle
from src.protocols.synthetic_profile import injector_kwargs_from_synthetic_profile


SMD_ENTITY_CONFIG_PATHS = {
    "machine-1-6": "../configs/data/smd_benchmark_machine_1_6_window20.yaml",
    "machine-3-4": "../configs/data/smd_benchmark_machine_3_4_window20.yaml",
    "machine-3-9": "../configs/data/smd_benchmark_machine_3_9_window20.yaml",
}


def _resolve_runtime_path(path: str | Path) -> Path:
    return resolve_repo_relative_path(path)


def _load_profile(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(_resolve_runtime_path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(_resolve_runtime_path(path).read_text(encoding="utf-8"))


def _resolve_visualization_seed(
    requested_seed: int | None,
    rng: Any | None = None,
) -> int:
    if requested_seed is not None:
        return int(requested_seed)
    active_rng = rng if rng is not None else np.random.default_rng()
    return int(active_rng.integers(0, 2**31 - 1))


def _select_random_window_indices(
    dataset_lengths: dict[str, int],
    rng: Any,
) -> dict[str, int]:
    return {
        entity_id: int(rng.integers(0, dataset_length))
        for entity_id, dataset_length in dataset_lengths.items()
    }


def build_random_smd_entity_window_batch(
    *,
    entity_ids: list[str],
    split_name: str,
    seed: int,
) -> dict[str, Any]:
    """Build one random real SMD window per requested entity.

    ₍^. .^₎⟆ Notebook data path

    3 SMD entities
      -> one random window from each entity
      -> collated clean batch
      -> synthetic profile comparison
    """
    rng = np.random.default_rng(seed)
    datasets_by_entity = {}
    for entity_id in entity_ids:
        data_config = _load_yaml(SMD_ENTITY_CONFIG_PATHS[entity_id])
        data_config["num_workers"] = 0
        data_config["batch_size"] = 1
        data_bundle = build_smd_dataset_bundle(data_config)
        datasets_by_entity[entity_id] = data_bundle["datasets"][split_name]

    window_indices = _select_random_window_indices(
        {entity_id: len(dataset) for entity_id, dataset in datasets_by_entity.items()},
        rng,
    )
    windows = [
        datasets_by_entity[entity_id][window_indices[entity_id]]
        for entity_id in entity_ids
    ]
    return collate_windows(windows)


def _ensure_demo_batch_has_enough_channels(
    batch: dict[str, Any],
    min_channels: int = 6,
) -> dict[str, Any]:
    current_channels = int(batch["x"].shape[2])
    if current_channels >= min_channels:
        return batch

    repeated_batch = dict(batch)
    repeat_count = (min_channels + current_channels - 1) // current_channels
    repeated_x = batch["x"].repeat(1, 1, repeat_count)[:, :, :min_channels]
    repeated_batch["x"] = repeated_x
    return repeated_batch


def _build_injector(profile: dict[str, Any], seed: int) -> SyntheticAnomalyInjector:
    kwargs = injector_kwargs_from_synthetic_profile(profile)
    kwargs.pop("window_size")
    return SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        anomaly_families=REDLAMP_ANOMALY_FAMILIES,
        deterministic_seed=seed,
        train_balance_classes=True,
        classification_label_mode="redlamp_multiclass",
        **kwargs,
    )


def _build_family_injector(
    profile: dict[str, Any],
    family_name: str,
    seed: int,
) -> SyntheticAnomalyInjector:
    kwargs = injector_kwargs_from_synthetic_profile(profile)
    kwargs.pop("window_size")
    return SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        anomaly_families=(family_name,),
        deterministic_seed=seed,
        train_balance_classes=False,
        classification_label_mode="redlamp_multiclass",
        **kwargs,
    )


def _build_family_gallery_batches(
    *,
    profile: dict[str, Any],
    clean_batch: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    gallery_batches: list[dict[str, Any]] = []
    for family_index, family_name in enumerate(REDLAMP_ANOMALY_FAMILIES):
        family_seed = seed + family_index
        augmented_batch = _build_family_injector(
            profile,
            family_name,
            family_seed,
        ).augment_batch(clean_batch)
        _attach_source_window_metadata(augmented_batch)
        gallery_batches.append(
            {
                "family_name": family_name,
                "seed": family_seed,
                "batch": augmented_batch,
            }
        )
    return gallery_batches


def _mean_abs_delta(
    clean_batch: dict[str, Any], augmented_batch: dict[str, Any]
) -> float:
    delta = torch.abs(augmented_batch["x"] - clean_batch["x"])
    return float(delta.mean().detach().cpu())


def _profile_summary(
    profile_name: str,
    clean_batch: dict[str, Any],
    augmented_batch: dict[str, Any],
) -> dict[str, Any]:
    metadata = augmented_batch["augmentation_metadata"]
    family_counts: dict[str, int] = {}
    for item in metadata:
        family_name = str(item["anomaly_family"])
        family_counts[family_name] = family_counts.get(family_name, 0) + 1
    return {
        "profile_name": profile_name,
        "mean_abs_delta": _mean_abs_delta(clean_batch, augmented_batch),
        "mask_positive_points": int(
            augmented_batch["synthetic_anomaly_mask"].sum().detach().cpu()
        ),
        "family_counts": family_counts,
    }


def _injected_point_indices(
    augmented_batch: dict[str, Any],
    sample_index: int,
) -> list[int]:
    mask = augmented_batch["synthetic_anomaly_mask"][sample_index].detach().cpu()
    return [int(index.item()) for index in torch.nonzero(mask > 0, as_tuple=False)]


def _build_sample_plot_annotation(
    profile_name: str,
    batch: dict[str, Any],
    sample_index: int,
) -> dict[str, Any]:
    metadata = batch["augmentation_metadata"][sample_index]
    anomaly_family = metadata["anomaly_family"]
    start_index = metadata["start_index"]
    end_index = metadata["end_index"]
    entity_id = metadata.get("entity_id", "unknown")
    source_start_index = metadata.get("source_start_index", "unknown")
    return {
        "title": (
            f"{profile_name} | entity={entity_id} "
            f"| window_start={source_start_index} | anomaly={anomaly_family} "
            f"| segment=[{start_index}, {end_index})"
        ),
        "injected_point_indices": _injected_point_indices(batch, sample_index),
        "affected_channels": list(metadata.get("affected_channels", [])),
    }


def _attach_source_window_metadata(batch: dict[str, Any]) -> None:
    for metadata, source_meta in zip(
        batch["augmentation_metadata"],
        batch["meta"],
        strict=True,
    ):
        metadata["entity_id"] = source_meta.get("entity_id", "unknown")
        metadata["source_start_index"] = int(source_meta.get("start_index", -1))
        metadata["source_end_index"] = int(source_meta.get("end_index", -1))


def _select_most_visible_sample_channel(
    clean_batch: dict[str, Any],
    augmented_batch: dict[str, Any],
) -> tuple[int, int]:
    clean = clean_batch["x"].detach().cpu()
    augmented = augmented_batch["x"].detach().cpu()
    masks = augmented_batch["synthetic_anomaly_mask"].detach().cpu().bool()
    best_score = -1.0
    best_sample_index = 0
    best_channel_index = 0

    for sample_index in range(clean.shape[0]):
        masked_points = masks[sample_index]
        if not bool(masked_points.any()):
            continue
        delta = torch.abs(augmented[sample_index] - clean[sample_index])
        masked_delta = delta[masked_points]
        channel_scores = masked_delta.mean(dim=0)
        score, channel_index = torch.max(channel_scores, dim=0)
        if float(score.item()) > best_score:
            best_score = float(score.item())
            best_sample_index = sample_index
            best_channel_index = int(channel_index.item())
    return best_sample_index, best_channel_index


def _select_most_visible_channels(
    clean_values: torch.Tensor,
    augmented_values: torch.Tensor,
    injected_mask: torch.Tensor,
    max_channels: int = 3,
) -> list[int]:
    if not bool(injected_mask.any()):
        return list(range(min(max_channels, clean_values.shape[1])))

    delta = torch.abs(augmented_values - clean_values)
    masked_delta = delta[injected_mask]
    channel_scores = masked_delta.mean(dim=0)
    sorted_indices = torch.argsort(channel_scores, descending=True)
    visible_channels: list[int] = []
    for channel_index in sorted_indices.tolist():
        if float(channel_scores[channel_index].item()) <= 0.0:
            continue
        visible_channels.append(int(channel_index))
        if len(visible_channels) == max_channels:
            break
    if visible_channels:
        return visible_channels
    return list(range(min(max_channels, clean_values.shape[1])))


def _augment_until_three_visible_channels(
    profile: dict[str, Any],
    clean_batch: dict[str, Any],
    seed: int,
    max_attempts: int = 25,
) -> tuple[dict[str, Any], int]:
    for attempt_index in range(max_attempts):
        candidate_seed = seed + attempt_index
        candidate_batch = _build_injector(profile, candidate_seed).augment_batch(
            clean_batch
        )
        _attach_source_window_metadata(candidate_batch)
        sample_index, _ = _select_most_visible_sample_channel(
            clean_batch,
            candidate_batch,
        )
        injected_mask = candidate_batch["synthetic_anomaly_mask"][sample_index].bool()
        channels = _select_most_visible_channels(
            clean_batch["x"][sample_index].detach().cpu(),
            candidate_batch["x"][sample_index].detach().cpu(),
            injected_mask.detach().cpu(),
        )
        if len(channels) >= 3:
            return candidate_batch, candidate_seed
    return candidate_batch, candidate_seed


def _plot_profile_window(
    axis: Any,
    values: torch.Tensor,
    annotation: dict[str, Any],
    channel_indices: list[int],
) -> None:
    x_values = list(range(values.shape[0]))
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    injected_indices = annotation["injected_point_indices"]
    for color_index, channel_index in enumerate(channel_indices):
        color = colors[color_index % len(colors)]
        axis.plot(
            x_values,
            values[:, channel_index].numpy(),
            color=color,
            linewidth=1.8,
            label=f"ch {channel_index}",
        )
        if injected_indices:
            y_values = values[injected_indices, channel_index].numpy()
            axis.scatter(injected_indices, y_values, color=color, s=24, zorder=3)
    for point_index in injected_indices:
        axis.axvline(point_index, color="#d62728", alpha=0.06, linewidth=1)
    axis.set_title(annotation["title"])
    axis.set_ylabel("value")
    axis.legend(loc="upper right")


def _plot_clean_reference_channels(
    axis: Any,
    values: torch.Tensor,
    channels: list[int],
    title_prefix: str,
) -> None:
    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for color_index, channel_index in enumerate(channels):
        axis.plot(
            list(range(values.shape[0])),
            values[:, channel_index].numpy(),
            color=colors[color_index % len(colors)],
            linewidth=1.6,
            label=f"ch {channel_index}",
        )
    axis.set_title(f"{title_prefix} clean reference | 3 injected channels")
    axis.set_ylabel("value")
    axis.legend(loc="upper right")


def _plot_family_gallery(
    clean_batch: dict[str, Any],
    gallery_batches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_batch["x"].detach().cpu()
    figure_height = max(22 * 2.1, 20)
    figure, axes = plt.subplots(
        len(gallery_batches) * 2,
        1,
        figsize=(13, figure_height),
        constrained_layout=True,
    )
    for family_index, gallery_item in enumerate(gallery_batches):
        augmented_batch = gallery_item["batch"]
        sample_index, _ = _select_most_visible_sample_channel(
            clean_batch,
            augmented_batch,
        )
        augmented = augmented_batch["x"].detach().cpu()
        mask = augmented_batch["synthetic_anomaly_mask"][sample_index].detach().cpu()
        channels = _select_most_visible_channels(
            clean[sample_index],
            augmented[sample_index],
            mask.bool(),
        )
        annotation = _build_sample_plot_annotation(
            "Synthetic",
            augmented_batch,
            sample_index,
        )
        clean_axis = axes[family_index * 2]
        anomaly_axis = axes[family_index * 2 + 1]
        _plot_clean_reference_channels(
            clean_axis,
            clean[sample_index],
            channels,
            f"{gallery_item['family_name']} clean reference",
        )
        _plot_profile_window(
            anomaly_axis,
            augmented[sample_index],
            annotation,
            channels,
        )
        anomaly_axis.set_xlabel("point index inside window")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_first_sample(
    clean_batch: dict[str, Any],
    legacy_batch: dict[str, Any],
    visible_batch: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_sample, _ = _select_most_visible_sample_channel(
        clean_batch,
        legacy_batch,
    )
    visible_sample, _ = _select_most_visible_sample_channel(
        clean_batch,
        visible_batch,
    )
    clean = clean_batch["x"].detach().cpu()
    legacy = legacy_batch["x"].detach().cpu()
    visible = visible_batch["x"].detach().cpu()
    legacy_annotation = _build_sample_plot_annotation(
        "Legacy profile",
        legacy_batch,
        legacy_sample,
    )
    visible_annotation = _build_sample_plot_annotation(
        "Visible profile",
        visible_batch,
        visible_sample,
    )
    legacy_mask = legacy_batch["synthetic_anomaly_mask"][legacy_sample].detach().cpu()
    visible_mask = (
        visible_batch["synthetic_anomaly_mask"][visible_sample].detach().cpu()
    )
    legacy_channels = _select_most_visible_channels(
        clean[legacy_sample],
        legacy[legacy_sample],
        legacy_mask.bool(),
    )
    visible_channels = _select_most_visible_channels(
        clean[visible_sample],
        visible[visible_sample],
        visible_mask.bool(),
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 7), constrained_layout=True)
    _plot_clean_reference_channels(
        axes[0, 0],
        clean[legacy_sample],
        legacy_channels,
        "Legacy",
    )
    _plot_profile_window(
        axes[1, 0],
        legacy[legacy_sample],
        legacy_annotation,
        legacy_channels,
    )
    _plot_clean_reference_channels(
        axes[0, 1],
        clean[visible_sample],
        visible_channels,
        "Visible",
    )
    _plot_profile_window(
        axes[1, 1],
        visible[visible_sample],
        visible_annotation,
        visible_channels,
    )
    for column_index in range(2):
        axes[1, column_index].set_xlabel("point index inside window")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    base_seed = _resolve_visualization_seed(args.seed)
    if bool(args.use_smd_entities):
        clean_batch = build_random_smd_entity_window_batch(
            entity_ids=list(SMD_ENTITY_CONFIG_PATHS),
            split_name=str(args.split),
            seed=base_seed,
        )
    else:
        clean_batch = _ensure_demo_batch_has_enough_channels(
            build_demo_batch(args.experiment_config),
            min_channels=12,
        )
    legacy_profile = _load_profile(args.legacy_profile)
    visible_profile = _load_profile(args.visible_profile)
    output_dir = _resolve_runtime_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_mode == "full_taxonomy":
        selected_profile = visible_profile
        gallery_batches = _build_family_gallery_batches(
            profile=selected_profile,
            clean_batch=clean_batch,
            seed=base_seed,
        )
        summary = {
            "requested_seed": args.seed,
            "base_visualization_seed": base_seed,
            "source": "smd_entities" if bool(args.use_smd_entities) else "demo_batch",
            "split": args.split if bool(args.use_smd_entities) else None,
            "plot_mode": args.plot_mode,
            "profile_name": selected_profile["profile_name"],
            "families": [
                {"family_name": item["family_name"], "seed": item["seed"]}
                for item in gallery_batches
            ],
        }
        summary_path = output_dir / "synthetic_profile_comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
        _plot_family_gallery(
            clean_batch,
            gallery_batches,
            output_dir / "synthetic_profile_comparison_first_sample.png",
        )
        return summary

    legacy_batch, legacy_seed = _augment_until_three_visible_channels(
        legacy_profile,
        clean_batch,
        base_seed,
    )
    visible_batch, visible_seed = _augment_until_three_visible_channels(
        visible_profile,
        clean_batch,
        base_seed,
    )

    summary = {
        "requested_seed": args.seed,
        "base_visualization_seed": base_seed,
        "experiment_config": args.experiment_config,
        "source": "smd_entities" if bool(args.use_smd_entities) else "demo_batch",
        "split": args.split if bool(args.use_smd_entities) else None,
        "plot_mode": args.plot_mode,
        "visualization_seeds": {
            "legacy": legacy_seed,
            "visible": visible_seed,
        },
        "legacy": _profile_summary("legacy", clean_batch, legacy_batch),
        "visible": _profile_summary("visible", clean_batch, visible_batch),
    }
    summary_path = output_dir / "synthetic_profile_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), "utf-8")
    _plot_first_sample(
        clean_batch,
        legacy_batch,
        visible_batch,
        output_dir / "synthetic_profile_comparison_first_sample.png",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", default=None)
    parser.add_argument(
        "--legacy-profile",
        default="configs/protocol/synthetic_redlamp12_legacy_window20.yaml",
    )
    parser.add_argument(
        "--visible-profile",
        default="configs/protocol/synthetic_redlamp12_visible_window20.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix the visualization seed. Omit this to show a different window each run.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/synthetic_profile_comparison",
    )
    parser.add_argument(
        "--use-smd-entities",
        action="store_true",
        help="Sample one real SMD window from each benchmark entity.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to sample when --use-smd-entities is enabled.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["profile_comparison", "full_taxonomy"],
        default="profile_comparison",
    )
    summary = run_comparison(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
