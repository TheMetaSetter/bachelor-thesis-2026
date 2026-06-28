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

from src.analysis.evaluation_protocol_audit import (
    describe_label_regime,
)
from src.core.console import console_print
from src.core.config import load_experiment_config
from src.data.datasets.anomaly_archive import AnomalyArchiveDatasetParser
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


def _load_test_sequences_for_experiment_config(
    experiment_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    data_config = experiment_config["data"]
    dataset_name = data_config["dataset_name"]
    if dataset_name == "smd":
        parser = SMDDatasetParser(
            root_dir=data_config["root_dir"],
            validation_split_ratio=float(data_config["validation_split_ratio"]),
            entity_ids=data_config.get("entity_ids"),
        )
    elif dataset_name == "anomaly_archive":
        parser = AnomalyArchiveDatasetParser(
            file_path=data_config["file_path"],
            validation_split_ratio=float(data_config["validation_split_ratio"]),
        )
    else:
        raise ValueError(f"Unsupported dataset_name for visualization: {dataset_name}")

    return {
        sequence["meta"]["entity_id"]: sequence for sequence in parser.parse()["test"]
    }


def build_visualization_payload(
    raw_sequence: dict[str, Any],
    evaluation_record: dict[str, Any],
    threshold: float,
    benchmark_comparability: str | None = None,
) -> dict[str, Any]:
    raw_values = raw_sequence["x"].detach().cpu()
    raw_point_labels = raw_sequence["point_labels"]
    if raw_point_labels is None:
        raise ValueError("raw_sequence must include point_labels for visualization")
    point_scores = torch.tensor(evaluation_record["point_scores"], dtype=torch.float32)
    ground_truth_mask = raw_point_labels.detach().cpu().numpy().astype(np.int64)
    predicted_mask = (point_scores > threshold).numpy().astype(np.int64)
    evaluated_start_index = int(evaluation_record.get("evaluated_start_index", 0))
    evaluated_end_index = int(
        evaluation_record.get("evaluated_end_index", raw_values.shape[0])
    )
    raw_num_points = int(evaluation_record.get("raw_num_points", raw_values.shape[0]))
    evaluated_num_points = int(
        evaluation_record.get("evaluated_num_points", raw_values.shape[0])
    )
    is_truncated = evaluated_num_points < raw_num_points
    return {
        "raw_values": raw_values,
        "point_scores": point_scores,
        "ground_truth_mask": ground_truth_mask,
        "predicted_mask": predicted_mask,
        "ground_truth_spans": _collect_positive_spans(ground_truth_mask),
        "predicted_spans": _collect_positive_spans(predicted_mask),
        "evaluated_start_index": evaluated_start_index,
        "evaluated_end_index": evaluated_end_index,
        "evaluated_num_points": evaluated_num_points,
        "raw_num_points": raw_num_points,
        "is_truncated": is_truncated,
        "label_regime": describe_label_regime(ground_truth_mask),
        "benchmark_comparability": benchmark_comparability,
    }


def save_entity_evaluation_visualization(
    raw_sequence: dict[str, Any],
    evaluation_record: dict[str, Any],
    threshold: float,
    output_path: str | Path,
    channels_to_plot: int = 3,
    benchmark_comparability: str | None = None,
    strict_coverage: bool = False,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = build_visualization_payload(
        raw_sequence=raw_sequence,
        evaluation_record=evaluation_record,
        threshold=threshold,
        benchmark_comparability=benchmark_comparability,
    )
    raw_values = payload["raw_values"]
    point_scores = payload["point_scores"]

    if raw_values.shape[0] != point_scores.shape[0]:
        raise ValueError(
            "raw sequence length must match evaluation point_scores length"
        )
    if strict_coverage and payload["is_truncated"]:
        raise ValueError(
            "Visualization requested for a truncated evaluation artifact. "
            "Disable strict_coverage to render a forensic plot instead."
        )
    if payload["is_truncated"]:
        console_print(
            "VIZ",
            "Rendering truncated evaluation coverage",
            entity_id=evaluation_record["entity_id"],
            evaluated_num_points=payload["evaluated_num_points"],
            raw_num_points=payload["raw_num_points"],
        )

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
    if payload["is_truncated"]:
        score_axis.axvspan(
            payload["evaluated_start_index"],
            payload["evaluated_end_index"],
            color="gold",
            alpha=0.10,
            label="evaluated coverage",
        )
    score_axis.fill_between(
        time_index,
        threshold,
        point_scores.numpy(),
        where=payload["predicted_mask"].astype(bool),
        color="red",
        alpha=0.22,
        label="predicted anomaly",
    )
    score_axis.set_ylabel("Score")
    score_axis.set_title(
        f"{evaluation_record['entity_id']} | threshold={threshold:.4f} | "
        "red=predicted, blue=raw ground truth"
    )
    legend_handles = [
        score_axis.lines[0],
        score_axis.lines[1],
        Patch(facecolor="red", alpha=0.22, label="predicted anomaly"),
        Patch(facecolor="royalblue", alpha=0.16, label="raw ground truth anomaly"),
    ]
    if payload["is_truncated"]:
        legend_handles.append(
            Patch(facecolor="gold", alpha=0.10, label="evaluated coverage")
        )
    score_axis.legend(handles=legend_handles, loc="upper right")

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
        for start_index, end_index in payload["ground_truth_spans"]:
            axis.axvspan(start_index, end_index, color="royalblue", alpha=0.16)
        for start_index, end_index in payload["predicted_spans"]:
            axis.axvspan(start_index, end_index, color="red", alpha=0.12)

    note_lines = [
        f"label_regime={payload['label_regime']}",
        f"evaluated_points={payload['evaluated_num_points']}/{payload['raw_num_points']}",
        f"truncated={payload['is_truncated']}",
    ]
    if payload["benchmark_comparability"] is not None:
        note_lines.append(
            f"benchmark_comparability={payload['benchmark_comparability']}"
        )
    score_axis.text(
        0.01,
        0.02,
        "\n".join(note_lines),
        transform=score_axis.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "lightgray"},
    )

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
    sequence_by_entity_id = _load_test_sequences_for_experiment_config(
        experiment_config
    )
    benchmark_comparability = (
        "non_comparable"
        if bool(int(evaluation_metrics.get("is_truncated_evaluation", 0)))
        else "benchmark_comparable"
    )

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
                benchmark_comparability=benchmark_comparability,
            )
        )
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment/smoke/smd__thesis_multitask__multitask-smoke__w100__seed7__smoke.yaml",
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
