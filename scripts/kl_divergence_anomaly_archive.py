from __future__ import annotations

"""Compute KL divergence between train and test segments in AnomalyArchive."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.analysis.anomaly_archive_kl import (
    build_kl_report,
    compute_ks_drift_report,
    load_time_series_values,
    parse_anomaly_archive_filename,
    rank_anomaly_archive_by_kl,
    rank_anomaly_archive_by_ks,
    split_series_by_annotation,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/AnomalyArchive/053_UCR_Anomaly_DISTORTEDWalkingAceleration1_1500_2764_2995.txt"),
        help="Path to one AnomalyArchive time-series file.",
    )
    parser.add_argument(
        "--comparison-mode",
        choices=("pre_vs_post", "pre_vs_anomaly"),
        default="pre_vs_post",
        help="How to define the test segment from the annotation.",
    )
    parser.add_argument("--bins", type=int, default=64, help="Histogram bin count.")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1e-12,
        help="Additive smoothing used in the discrete KL estimate.",
    )
    parser.add_argument(
        "--inclusive-anomaly-end",
        action="store_true",
        help="Treat anomaly_end_index as inclusive when comparison-mode is pre_vs_anomaly.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rank every file in data/AnomalyArchive instead of a single file.",
    )
    parser.add_argument(
        "--stat-test",
        choices=("kl", "ks"),
        default="kl",
        help="Statistic to compute when --all is set.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold used for KS test with BH correction.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/AnomalyArchive"),
        help="Root directory containing AnomalyArchive files when --all is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("documents/logs/06-15-2026/research/anomaly_archive_kl"),
        help="Directory for ranking tables and plots.",
    )
    return parser.parse_args()


def _plot_ecdf(axis: plt.Axes, values: np.ndarray, label: str) -> None:
    sorted_values = np.sort(values)
    cumulative = np.arange(1, sorted_values.size + 1, dtype=np.float64) / sorted_values.size
    axis.step(sorted_values, cumulative, where="post", label=label)


def _plot_histogram_and_ecdf(
    train_values: np.ndarray,
    test_values: np.ndarray,
    output_path: Path,
    title: str,
    bins: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].hist(train_values, bins=bins, density=True, alpha=0.7, color="#1f77b4")
    axes[0, 0].set_title("Train histogram")
    axes[0, 1].hist(test_values, bins=bins, density=True, alpha=0.7, color="#d62728")
    axes[0, 1].set_title("Test histogram")

    _plot_ecdf(axes[1, 0], train_values, "train")
    _plot_ecdf(axes[1, 0], test_values, "test")
    axes[1, 0].set_title("ECDF comparison")
    axes[1, 0].legend()

    combined_values = np.concatenate([train_values, test_values])
    combined_min = float(np.min(combined_values))
    combined_max = float(np.max(combined_values))
    bins_edges = np.linspace(combined_min, combined_max, bins + 1)
    axes[1, 1].hist(
        train_values,
        bins=bins_edges,
        density=True,
        alpha=0.5,
        label="train",
        color="#1f77b4",
    )
    axes[1, 1].hist(
        test_values,
        bins=bins_edges,
        density=True,
        alpha=0.5,
        label="test",
        color="#d62728",
    )
    axes[1, 1].set_title("Shared-bin histogram")
    axes[1, 1].legend()

    figure.suptitle(title)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _write_ranking_csv(output_path: Path, ranking_entries) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "rank",
                "file_name",
                "series_name",
                "total_length",
                "train_length",
                "test_length",
                "kl_train_to_test",
                "kl_test_to_train",
                "symmetric_kl",
            ]
        )
        for rank, entry in enumerate(ranking_entries, start=1):
            report = entry.report
            writer.writerow(
                [
                    rank,
                    report.file_path.name,
                    report.series_name or "",
                    report.total_length,
                    report.train_length,
                    report.test_length,
                    f"{report.kl_train_to_test:.12f}",
                    f"{report.kl_test_to_train:.12f}",
                    f"{entry.symmetric_kl:.12f}",
                ]
            )
    return output_path


def _write_significance_csv(output_path: Path, ranking_entries) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "rank",
                "file_name",
                "series_name",
                "total_length",
                "train_length",
                "test_length",
                "ks_statistic",
                "p_value",
                "bh_adjusted_p_value",
                "is_significant",
            ]
        )
        for rank, report in enumerate(ranking_entries, start=1):
            writer.writerow(
                [
                    rank,
                    report.file_path.name,
                    report.series_name or "",
                    report.total_length,
                    report.train_length,
                    report.test_length,
                    f"{report.ks_statistic:.12f}",
                    f"{report.p_value:.12e}",
                    f"{report.bh_adjusted_p_value:.12e}",
                    "yes" if report.is_significant else "no",
                ]
            )
    return output_path


def main() -> None:
    arguments = parse_arguments()
    if arguments.all:
        if arguments.stat_test == "kl":
            ranking_entries = rank_anomaly_archive_by_kl(
                root_directory=arguments.root,
                comparison_mode=arguments.comparison_mode,
                bins=arguments.bins,
                smoothing=arguments.smoothing,
                inclusive_anomaly_end=arguments.inclusive_anomaly_end,
            )
            ranking_csv_path = _write_ranking_csv(
                arguments.output_dir / "anomaly_archive_kl_ranking.csv",
                ranking_entries,
            )
            top_report = ranking_entries[0].report
            top_values = load_time_series_values(top_report.file_path)
            parsed = parse_anomaly_archive_filename(top_report.file_path)
            if parsed is None:
                midpoint = top_values.size // 2
                train_values = top_values[:midpoint]
                test_values = top_values[midpoint:]
            else:
                train_values, test_values = split_series_by_annotation(
                    top_values,
                    anomaly_start_index=parsed.anomaly_start_index,
                    anomaly_end_index=parsed.anomaly_end_index,
                    comparison_mode=arguments.comparison_mode,
                    inclusive_anomaly_end=arguments.inclusive_anomaly_end,
                )
            plot_path = _plot_histogram_and_ecdf(
                train_values=train_values,
                test_values=test_values,
                output_path=arguments.output_dir / "top_kl_hist_ecdf.png",
                title=(
                    f"Top KL series: {top_report.file_path.name} | "
                    f"KL(train||test)={top_report.kl_train_to_test:.6f}"
                ),
                bins=arguments.bins,
            )

            print(f"ranked_files: {len(ranking_entries)}")
            print(f"ranking_csv: {ranking_csv_path}")
            print(f"top_file: {top_report.file_path}")
            print(f"top_series: {top_report.series_name}")
            print(f"top_kl_train_to_test: {top_report.kl_train_to_test:.6f}")
            print(f"top_kl_test_to_train: {top_report.kl_test_to_train:.6f}")
            print(f"top_symmetric_kl: {ranking_entries[0].symmetric_kl:.6f}")
            print(f"plot_path: {plot_path}")
        else:
            significance_reports = rank_anomaly_archive_by_ks(
                root_directory=arguments.root,
                comparison_mode=arguments.comparison_mode,
                inclusive_anomaly_end=arguments.inclusive_anomaly_end,
                alpha=arguments.alpha,
            )
            significance_csv_path = _write_significance_csv(
                arguments.output_dir / "anomaly_archive_ks_significance.csv",
                significance_reports,
            )
            significant_reports = [report for report in significance_reports if report.is_significant]

            print(f"ranked_files: {len(significance_reports)}")
            print(f"significance_csv: {significance_csv_path}")
            print(f"alpha: {arguments.alpha}")
            print(f"significant_count: {len(significant_reports)}")
            for report in significant_reports[:20]:
                print(
                    f"{report.file_path.name}\t"
                    f"ks={report.ks_statistic:.6f}\t"
                    f"p={report.p_value:.3e}\t"
                    f"bh_p={report.bh_adjusted_p_value:.3e}"
                )
        return

    report = build_kl_report(
        file_path=arguments.file,
        comparison_mode=arguments.comparison_mode,
        bins=arguments.bins,
        smoothing=arguments.smoothing,
        inclusive_anomaly_end=arguments.inclusive_anomaly_end,
    )

    print(f"file: {report.file_path}")
    if report.series_name is not None:
        print(f"series: {report.series_name}")
    print(f"total_length: {report.total_length}")
    print(f"train_length: {report.train_length}")
    print(f"test_length: {report.test_length}")
    print(f"comparison_mode: {report.comparison_mode}")
    print(f"kl(train || test): {report.kl_train_to_test:.6f}")
    print(f"kl(test || train): {report.kl_test_to_train:.6f}")


if __name__ == "__main__":
    main()
