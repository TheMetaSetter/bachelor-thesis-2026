from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.protocols.threshold_artifact import (
    load_threshold_artifact,
    validate_threshold_artifact,
)


def load_raw_score_arrays(
    score_path: Path, threshold_artifact: dict[str, Any]
) -> dict[str, np.ndarray]:
    validate_threshold_artifact(threshold_artifact)
    if threshold_artifact.get("score_space") != "raw_input":
        raise ValueError("histogram requires a raw_input threshold artifact")
    if threshold_artifact.get("point_score_transform") != "identity":
        raise ValueError("histogram requires the identity point-score transform")

    with np.load(score_path, allow_pickle=False) as score_archive:
        arrays = {
            field_name: np.asarray(score_archive[field_name])
            for field_name in (
                "raw_input_point_mse",
                "point_labels",
                "raw_input_window_mse",
                "window_labels",
            )
        }
    for field_name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{field_name} must be one-dimensional")
        if "mse" in field_name and not np.isfinite(values).all():
            raise ValueError(f"{field_name} must contain only finite values")
    if arrays["raw_input_point_mse"].shape != arrays["point_labels"].shape:
        raise ValueError("point score and point label shapes must match")
    if arrays["raw_input_window_mse"].shape != arrays["window_labels"].shape:
        raise ValueError("window score and window label shapes must match")
    for field_name in ("point_labels", "window_labels"):
        if not np.isin(arrays[field_name], [0, 1]).all():
            raise ValueError(f"{field_name} must contain only labels 0 and 1")
    return arrays


def _summary_for_values(
    values: np.ndarray, labels: np.ndarray, threshold: float
) -> dict[str, Any]:
    normal_values = values[labels == 0]
    anomalous_values = values[labels == 1]
    return {
        "count": int(values.size),
        "normal_count": int(normal_values.size),
        "anomalous_count": int(anomalous_values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "threshold": float(threshold),
        "above_threshold_count": int(np.count_nonzero(values > threshold)),
    }


def build_histogram_summary(
    arrays: dict[str, np.ndarray],
    *,
    point_threshold: float,
    window_threshold: float,
) -> dict[str, dict[str, Any]]:
    return {
        "point": _summary_for_values(
            arrays["raw_input_point_mse"], arrays["point_labels"], point_threshold
        ),
        "window": _summary_for_values(
            arrays["raw_input_window_mse"], arrays["window_labels"], window_threshold
        ),
    }


def _shared_histogram_bins(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
        return np.linspace(minimum - padding, maximum + padding, 21)
    return np.histogram_bin_edges(values, bins=40)


def plot_raw_input_mse_histograms(
    *,
    arrays: dict[str, np.ndarray],
    threshold_artifact: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    point_threshold = float(threshold_artifact["thresholds"]["offline_point"]["value"])
    window_threshold = float(threshold_artifact["thresholds"]["input_window"]["value"])
    summary = build_histogram_summary(
        arrays, point_threshold=point_threshold, window_threshold=window_threshold
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_id = str(threshold_artifact["entity_id"])
    figure_path = output_dir / f"{entity_id}_raw_input_mse_histograms.png"
    summary_path = output_dir / f"{entity_id}_raw_input_mse_histograms.json"

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    point_values = arrays["raw_input_point_mse"]
    window_values = arrays["raw_input_window_mse"]
    point_bins = _shared_histogram_bins(point_values)
    window_bins = _shared_histogram_bins(window_values)
    axes[0].hist(
        point_values[arrays["point_labels"] == 0],
        bins=point_bins,
        alpha=0.7,
        label="normal point",
    )
    axes[0].hist(
        point_values[arrays["point_labels"] == 1],
        bins=point_bins,
        alpha=0.7,
        label="anomalous point",
    )
    axes[0].axvline(
        point_threshold,
        color="black",
        linestyle="--",
        label=f"threshold={point_threshold:.4g}",
    )
    axes[0].set_title("Point-level raw input MSE")
    axes[0].set_xlabel("raw input MSE")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].hist(
        window_values[arrays["window_labels"] == 0],
        bins=window_bins,
        alpha=0.7,
        label="normal window",
    )
    axes[1].hist(
        window_values[arrays["window_labels"] == 1],
        bins=window_bins,
        alpha=0.7,
        label="anomalous window",
    )
    axes[1].axvline(
        window_threshold,
        color="black",
        linestyle="--",
        label=f"threshold={window_threshold:.4g}",
    )
    axes[1].set_title("Window-level raw input MSE")
    axes[1].set_xlabel("raw input MSE")
    axes[1].set_ylabel("count")
    axes[1].legend()
    figure.suptitle(f"{entity_id} | score_space=raw_input | transform=identity")
    figure.savefig(
        figure_path,
        dpi=160,
        metadata={"score_space": "raw_input", "entity_id": entity_id},
    )
    plt.close(figure)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return figure_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-artifact", required=True, type=Path)
    parser.add_argument("--threshold-artifact", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    threshold_artifact = load_threshold_artifact(args.threshold_artifact)
    arrays = load_raw_score_arrays(args.score_artifact, threshold_artifact)
    figure_path, summary_path = plot_raw_input_mse_histograms(
        arrays=arrays,
        threshold_artifact=threshold_artifact,
        output_dir=args.output_dir,
    )
    print(
        json.dumps({"figure": str(figure_path), "summary": str(summary_path)}, indent=2)
    )


if __name__ == "__main__":
    main()
