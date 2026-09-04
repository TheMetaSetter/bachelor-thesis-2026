"""Plot direct-branch-routing anomaly scores for machine-3-4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (6, 8, 36)


def load_run(raw_dir: Path, seed: int) -> tuple[dict, np.ndarray]:
    metrics_path = raw_dir / f"seed{seed}_evaluation_metrics.json"
    records_path = raw_dir / f"seed{seed}_evaluation_records.json"
    metrics = json.loads(metrics_path.read_text())
    records = json.loads(records_path.read_text())
    record = records[0]
    scores = np.asarray(record["point_scores"], dtype=float)
    covered = np.asarray(record["covered_point_mask"], dtype=bool)
    scores = scores[covered]
    return metrics, scores


def summarize(seed: int, metrics: dict, scores: np.ndarray) -> dict:
    threshold = float(metrics["threshold"])
    return {
        "entity": "machine-3-4",
        "method": "thesis_direct_branch_routing_O0",
        "seed": seed,
        "num_scores": int(scores.size),
        "threshold": threshold,
        "score_min": float(np.min(scores)),
        "score_q01": float(np.quantile(scores, 0.01)),
        "score_q25": float(np.quantile(scores, 0.25)),
        "score_median": float(np.median(scores)),
        "score_mean": float(np.mean(scores)),
        "score_q75": float(np.quantile(scores, 0.75)),
        "score_q95": float(np.quantile(scores, 0.95)),
        "score_q99": float(np.quantile(scores, 0.99)),
        "score_max": float(np.max(scores)),
        "score_std": float(np.std(scores)),
        "above_threshold_count": int(np.count_nonzero(scores > threshold)),
        "above_threshold_ratio": float(np.mean(scores > threshold)),
        "protocol_status": metrics.get("protocol_status"),
        "benchmark_comparability": metrics.get("benchmark_comparability"),
    }


def plot_runs(raw_dir: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [(seed, *load_run(raw_dir, seed)) for seed in SEEDS]
    summaries = [summarize(seed, metrics, scores) for seed, metrics, scores in loaded]

    fig, axes = plt.subplots(1, len(loaded), figsize=(14, 4.4), sharey=True)
    for ax, (seed, metrics, scores), summary in zip(axes, loaded, summaries):
        positive_scores = scores[scores > 0]
        bins = np.geomspace(positive_scores.min(), positive_scores.max(), 60)
        ax.hist(positive_scores, bins=bins, color="#4472C4", alpha=0.82)
        ax.axvline(
            summary["threshold"],
            color="#C00000",
            linewidth=2,
            label=f"threshold = {summary['threshold']:.3f}",
        )
        ax.set_xscale("log")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("anomaly score (log scale)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.03,
            0.96,
            "median = "
            f"{summary['score_median']:.3f}\n"
            "q95 = "
            f"{summary['score_q95']:.3f}\n"
            f"> threshold = {summary['above_threshold_ratio']:.1%}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axes[0].set_ylabel("number of evaluated points")
    fig.suptitle("Machine-3-4 direct branch routing: anomaly score distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "anomaly_score_distributions.png", dpi=180)
    plt.close(fig)

    (output_dir / "anomaly_score_distribution_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    with (output_dir / "anomaly_score_distribution_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    raw_dir = repository_root / "outputs/analysis/machine_3_4_direct_branch_routing/raw"
    output_dir = repository_root / "outputs/analysis/machine_3_4_direct_branch_routing"
    summaries = plot_runs(raw_dir, output_dir)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
