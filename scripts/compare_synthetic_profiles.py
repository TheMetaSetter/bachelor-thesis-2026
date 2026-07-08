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
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from scripts.visualize_synthetic_anomalies import build_demo_batch
from src.data.augment import REDLAMP_ANOMALY_FAMILIES, SyntheticAnomalyInjector
from src.protocols.synthetic_profile import injector_kwargs_from_synthetic_profile


def _load_profile(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _mean_abs_delta(clean_batch: dict[str, Any], augmented_batch: dict[str, Any]) -> float:
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


def _plot_first_sample(
    clean_batch: dict[str, Any],
    legacy_batch: dict[str, Any],
    visible_batch: dict[str, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_batch["x"][0].detach().cpu()
    legacy = legacy_batch["x"][0].detach().cpu()
    visible = visible_batch["x"][0].detach().cpu()

    figure, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(clean[:, 0].numpy())
    axes[0].set_title("Clean")
    axes[1].plot(legacy[:, 0].numpy())
    axes[1].set_title("Legacy profile")
    axes[2].plot(visible[:, 0].numpy())
    axes[2].set_title("Visible profile")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    clean_batch = build_demo_batch(args.experiment_config)
    legacy_profile = _load_profile(Path(args.legacy_profile))
    visible_profile = _load_profile(Path(args.visible_profile))

    legacy_batch = _build_injector(legacy_profile, args.seed).augment_batch(clean_batch)
    visible_batch = _build_injector(visible_profile, args.seed).augment_batch(clean_batch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "experiment_config": args.experiment_config,
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        default="outputs/synthetic_profile_comparison",
    )
    summary = run_comparison(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
