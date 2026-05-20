from __future__ import annotations

"""Visualize training metric timelines from metrics.jsonl.

This script is intentionally small and explicit so thesis experiments can quickly
inspect train and val_synth trends, including Exp2 CKA diagnostics.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics_jsonl(metrics_path: str | Path) -> list[dict[str, float]]:
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {path}")

    parsed_rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parsed_rows.append(json.loads(stripped))
    if not parsed_rows:
        raise ValueError(f"metrics.jsonl is empty: {path}")
    return parsed_rows


def _extract_series(rows: list[dict[str, float]], metric_name: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        if metric_name not in row:
            continue
        epoch_value = int(row.get("epoch", len(epochs) + 1))
        metric_value = float(row[metric_name])
        epochs.append(epoch_value)
        values.append(metric_value)
    return epochs, values


def _plot_metric_rows(
    rows: list[dict[str, float]],
    metric_names: list[str],
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(13, 2.6 * len(metric_names)),
        constrained_layout=True,
        sharex=True,
    )
    if len(metric_names) == 1:
        axes = [axes]

    for axis, metric_name in zip(axes, metric_names):
        epochs, values = _extract_series(rows, metric_name)
        if not values:
            axis.text(
                0.5,
                0.5,
                f"Missing metric: {metric_name}",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.set_ylabel(metric_name)
            axis.grid(True, alpha=0.25)
            continue

        axis.plot(epochs, values, marker="o", markersize=3, linewidth=1.2)
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Epoch")
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path",
        default="outputs/smd_thesis_multitask_redlamp_multiclass_window20_exp2/metrics.jsonl",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/smd_thesis_multitask_redlamp_multiclass_window20_exp2/training_metric_plots.png",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=None,
        help="Repeat this flag to select metrics manually.",
    )
    args = parser.parse_args()

    default_metrics = [
        "train_loss",
        "val_synth_vus_pr",
        "train_contrastive_loss",
        "val_synth_contrastive_loss",
        "train_cka_reconstruction_mean",
        "train_cka_classification_mean",
        "val_synth_cka_reconstruction_mean",
        "val_synth_cka_classification_mean",
    ]
    metric_names = args.metrics if args.metrics else default_metrics
    rows = _load_metrics_jsonl(args.metrics_path)
    saved_path = _plot_metric_rows(rows, metric_names, args.output_path)
    print(saved_path)


if __name__ == "__main__":
    main()
