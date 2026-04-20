from __future__ import annotations

"""Render per-entity evaluation figures from saved anomaly scores.

This script is the reader-facing visualization entrypoint for offline anomaly
evaluation. It takes the saved evaluation records, loads the matching raw test
series, then writes one figure per entity so a fresher can inspect both the
timeline and the anomaly score without going back to notebooks.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import load_experiment_config
from src.data.datasets.smd import SMDDatasetParser


def _load_json_file(json_path: str | Path) -> Any:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_positive_spans(binary_values: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start_index: int | None = None
    for index, value in enumerate(binary_values.astype(int).tolist()):
        if value == 1 and start_index is None:
            start_index = index
        elif value == 0 and start_index is not None:
            spans.append((start_index, index))
            start_index = None
    if start_index is not None:
        spans.append((start_index, int(binary_values.shape[0])))
    return spans


def save_entity_evaluation_visualization(
    raw_sequence: dict[str, Any],
    evaluation_record: dict[str, Any],
    threshold: float,
    output_path: str | Path,
    channels_to_plot: int = 3,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    raw_values = raw_sequence["x"].detach().cpu()
    point_scores = torch.tensor(evaluation_record["point_scores"], dtype=torch.float32)
    point_labels = torch.tensor(evaluation_record["point_labels"], dtype=torch.long)

    if raw_values.shape[0] != point_scores.shape[0]:
        raise ValueError(
            "raw sequence length must match evaluation point_scores length"
        )

    predicted_mask = (point_scores > threshold).numpy().astype(np.int64)
    ground_truth_mask = point_labels.numpy().astype(np.int64)
    predicted_spans = _collect_positive_spans(predicted_mask)
    ground_truth_spans = _collect_positive_spans(ground_truth_mask)

    plotted_channels = min(channels_to_plot, raw_values.shape[1])
    figure, axes = plt.subplots(
        1 + plotted_channels,
        1,
        figsize=(16, 3 + 2.5 * plotted_channels),
        constrained_layout=True,
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    time_index = np.arange(raw_values.shape[0])
    score_axis = axes[0]
    score_axis.plot(
        time_index,
        point_scores.numpy(),
        color="navy",
        linewidth=1.0,
        label="point score",
    )
    score_axis.axhline(
        threshold, color="black", linestyle="--", linewidth=1.0, label="threshold"
    )
    score_axis.fill_between(
        time_index,
        threshold,
        point_scores.numpy(),
        where=predicted_mask.astype(bool),
        color="red",
        alpha=0.22,
        label="predicted anomaly",
    )
    score_axis.set_ylabel("Score")
    score_axis.set_title(
        f"{evaluation_record['entity_id']} | threshold={threshold:.4f} | red=predicted, blue=ground truth"
    )
    score_axis.legend(
        handles=[
            score_axis.lines[0],
            score_axis.lines[1],
            Patch(facecolor="red", alpha=0.22, label="predicted anomaly"),
            Patch(facecolor="royalblue", alpha=0.16, label="ground truth anomaly"),
        ],
        loc="upper right",
    )

    for channel_offset in range(plotted_channels):
        axis = axes[1 + channel_offset]
        axis.plot(
            time_index,
            raw_values[:, channel_offset].numpy(),
            color="steelblue",
            linewidth=0.9,
        )
        axis.set_ylabel(f"ch {channel_offset}")

    for axis in axes:
        for start_index, end_index in ground_truth_spans:
            axis.axvspan(start_index, end_index, color="royalblue", alpha=0.16)
        for start_index, end_index in predicted_spans:
            axis.axvspan(start_index, end_index, color="red", alpha=0.12)

    axes[-1].set_xlabel("Time index")
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    return output_file


def render_evaluation_visualizations(
    experiment_config_path: str,
    evaluation_records_path: str | Path | None = None,
    evaluation_metrics_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    channels_to_plot: int = 3,
    entity_ids: list[str] | None = None,
) -> list[Path]:
    experiment_config = load_experiment_config(experiment_config_path)
    base_output_dir = Path(experiment_config["output_dir"])
    records_path = (
        Path(evaluation_records_path)
        if evaluation_records_path is not None
        else base_output_dir / "evaluation_records.json"
    )
    metrics_path = (
        Path(evaluation_metrics_path)
        if evaluation_metrics_path is not None
        else base_output_dir / "evaluation_metrics.json"
    )
    render_dir = (
        Path(output_dir)
        if output_dir is not None
        else base_output_dir / "evaluation_plots"
    )

    evaluation_records = _load_json_file(records_path)
    evaluation_metrics = _load_json_file(metrics_path)
    threshold = float(evaluation_metrics["threshold"])

    parser = SMDDatasetParser(
        root_dir=experiment_config["data"]["root_dir"],
        validation_split_ratio=float(
            experiment_config["data"]["validation_split_ratio"]
        ),
    )
    raw_test_sequences = parser.parse()["test"]
    sequence_by_entity_id = {
        sequence["meta"]["entity_id"]: sequence for sequence in raw_test_sequences
    }

    requested_entity_ids = set(entity_ids) if entity_ids else None
    saved_paths: list[Path] = []
    for evaluation_record in evaluation_records:
        entity_id = evaluation_record["entity_id"]
        if requested_entity_ids is not None and entity_id not in requested_entity_ids:
            continue
        if entity_id not in sequence_by_entity_id:
            raise ValueError(f"Missing raw test sequence for entity_id: {entity_id}")

        saved_paths.append(
            save_entity_evaluation_visualization(
                raw_sequence=sequence_by_entity_id[entity_id],
                evaluation_record=evaluation_record,
                threshold=threshold,
                output_path=render_dir / f"{entity_id}.png",
                channels_to_plot=channels_to_plot,
            )
        )
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/smd_multitask_smoke.yaml",
    )
    parser.add_argument("--evaluation-records", default=None)
    parser.add_argument("--evaluation-metrics", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--channels-to-plot", type=int, default=3)
    parser.add_argument(
        "--entity-id",
        action="append",
        dest="entity_ids",
        default=None,
        help="Repeat this flag to render only selected entities.",
    )
    args = parser.parse_args()

    saved_paths = render_evaluation_visualizations(
        experiment_config_path=args.experiment_config,
        evaluation_records_path=args.evaluation_records,
        evaluation_metrics_path=args.evaluation_metrics,
        output_dir=args.output_dir,
        channels_to_plot=args.channels_to_plot,
        entity_ids=args.entity_ids,
    )
    for saved_path in saved_paths:
        print(saved_path)


if __name__ == "__main__":
    main()
