from __future__ import annotations

"""KL-divergence helpers for AnomalyArchive time-series files."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy import stats


_ANOMALY_ARCHIVE_FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>\d+)_UCR_Anomaly_(?P<series_name>.+)_(?P<start_index>\d+)"
    r"_(?P<anomaly_start_index>\d+)_(?P<anomaly_end_index>\d+)\.txt$"
)


@dataclass(frozen=True)
class AnomalyArchiveMetadata:
    series_name: str
    start_index: int
    anomaly_start_index: int
    anomaly_end_index: int


def parse_anomaly_archive_filename(
    file_path: str | Path,
) -> AnomalyArchiveMetadata | None:
    file_name = Path(file_path).name
    match = _ANOMALY_ARCHIVE_FILENAME_PATTERN.match(file_name)
    if match is None:
        return None

    return AnomalyArchiveMetadata(
        series_name=match.group("series_name"),
        start_index=int(match.group("start_index")),
        anomaly_start_index=int(match.group("anomaly_start_index")),
        anomaly_end_index=int(match.group("anomaly_end_index")),
    )


def load_time_series_values(file_path: str | Path) -> np.ndarray:
    values = np.fromstring(Path(file_path).read_text(encoding="utf-8"), sep=" ")
    if values.size == 0:
        raise ValueError(f"No numeric values found in {file_path}")
    return values


def split_series_by_annotation(
    values: np.ndarray,
    anomaly_start_index: int | None,
    anomaly_end_index: int | None,
    comparison_mode: str = "pre_vs_post",
    inclusive_anomaly_end: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if anomaly_start_index is None or anomaly_end_index is None:
        raise ValueError("Annotation indices are required for annotation-based split.")

    if comparison_mode == "pre_vs_post":
        train_values = values[:anomaly_start_index]
        test_values = values[anomaly_end_index:]
    elif comparison_mode == "pre_vs_anomaly":
        anomaly_stop_index = (
            anomaly_end_index + 1 if inclusive_anomaly_end else anomaly_end_index
        )
        train_values = values[:anomaly_start_index]
        test_values = values[anomaly_start_index:anomaly_stop_index]
    else:
        raise ValueError(
            "comparison_mode must be either 'pre_vs_post' or 'pre_vs_anomaly'."
        )

    if train_values.size == 0:
        raise ValueError("Training segment is empty after split.")
    if test_values.size == 0:
        raise ValueError("Testing segment is empty after split.")

    return train_values, test_values


def estimate_histogram_kl_divergence(
    left_values: np.ndarray,
    right_values: np.ndarray,
    bins: int = 64,
    smoothing: float = 1e-12,
) -> float:
    if left_values.size == 0 or right_values.size == 0:
        raise ValueError("Both value sets must be non-empty.")
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive.")

    combined_min = float(min(np.min(left_values), np.min(right_values)))
    combined_max = float(max(np.max(left_values), np.max(right_values)))
    if combined_min == combined_max:
        return 0.0

    bin_edges = np.linspace(combined_min, combined_max, bins + 1)
    left_histogram, _ = np.histogram(left_values, bins=bin_edges)
    right_histogram, _ = np.histogram(right_values, bins=bin_edges)

    left_probabilities = left_histogram.astype(np.float64) + smoothing
    right_probabilities = right_histogram.astype(np.float64) + smoothing
    left_probabilities /= left_probabilities.sum()
    right_probabilities /= right_probabilities.sum()

    return float(
        np.sum(left_probabilities * np.log(left_probabilities / right_probabilities))
    )


@dataclass(frozen=True)
class KLDivergenceReport:
    file_path: Path
    series_name: str | None
    total_length: int
    train_length: int
    test_length: int
    kl_train_to_test: float
    kl_test_to_train: float
    comparison_mode: str


@dataclass(frozen=True)
class KLDivergenceRankingEntry:
    report: KLDivergenceReport
    symmetric_kl: float


@dataclass(frozen=True)
class DriftSignificanceReport:
    file_path: Path
    series_name: str | None
    total_length: int
    train_length: int
    test_length: int
    ks_statistic: float
    p_value: float
    bh_adjusted_p_value: float
    is_significant: bool
    comparison_mode: str


def iter_anomaly_archive_files(root_directory: str | Path) -> Iterable[Path]:
    root_path = Path(root_directory)
    yield from sorted(root_path.glob("*.txt"))


def build_kl_report(
    file_path: str | Path,
    comparison_mode: str = "pre_vs_post",
    bins: int = 64,
    smoothing: float = 1e-12,
    inclusive_anomaly_end: bool = False,
) -> KLDivergenceReport:
    path = Path(file_path)
    metadata = parse_anomaly_archive_filename(path)
    series_values = load_time_series_values(path)

    if metadata is None:
        midpoint = series_values.size // 2
        train_values = series_values[:midpoint]
        test_values = series_values[midpoint:]
        series_name = None
    else:
        train_values, test_values = split_series_by_annotation(
            series_values,
            anomaly_start_index=metadata.anomaly_start_index,
            anomaly_end_index=metadata.anomaly_end_index,
            comparison_mode=comparison_mode,
            inclusive_anomaly_end=inclusive_anomaly_end,
        )
        series_name = metadata.series_name

    kl_train_to_test = estimate_histogram_kl_divergence(
        train_values,
        test_values,
        bins=bins,
        smoothing=smoothing,
    )
    kl_test_to_train = estimate_histogram_kl_divergence(
        test_values,
        train_values,
        bins=bins,
        smoothing=smoothing,
    )

    return KLDivergenceReport(
        file_path=path,
        series_name=series_name,
        total_length=int(series_values.size),
        train_length=int(train_values.size),
        test_length=int(test_values.size),
        kl_train_to_test=kl_train_to_test,
        kl_test_to_train=kl_test_to_train,
        comparison_mode=comparison_mode,
    )


def rank_anomaly_archive_by_kl(
    root_directory: str | Path,
    comparison_mode: str = "pre_vs_post",
    bins: int = 64,
    smoothing: float = 1e-12,
    inclusive_anomaly_end: bool = False,
) -> list[KLDivergenceRankingEntry]:
    ranking_entries: list[KLDivergenceRankingEntry] = []
    for file_path in iter_anomaly_archive_files(root_directory):
        report = build_kl_report(
            file_path=file_path,
            comparison_mode=comparison_mode,
            bins=bins,
            smoothing=smoothing,
            inclusive_anomaly_end=inclusive_anomaly_end,
        )
        ranking_entries.append(
            KLDivergenceRankingEntry(
                report=report,
                symmetric_kl=0.5 * (report.kl_train_to_test + report.kl_test_to_train),
            )
        )

    ranking_entries.sort(
        key=lambda entry: (
            entry.report.kl_train_to_test,
            entry.symmetric_kl,
            entry.report.file_path.name,
        ),
        reverse=True,
    )
    return ranking_entries


def compute_ks_drift_report(
    file_path: str | Path,
    comparison_mode: str = "pre_vs_post",
    inclusive_anomaly_end: bool = False,
) -> DriftSignificanceReport:
    path = Path(file_path)
    metadata = parse_anomaly_archive_filename(path)
    series_values = load_time_series_values(path)

    if metadata is None:
        midpoint = series_values.size // 2
        train_values = series_values[:midpoint]
        test_values = series_values[midpoint:]
        series_name = None
    else:
        train_values, test_values = split_series_by_annotation(
            series_values,
            anomaly_start_index=metadata.anomaly_start_index,
            anomaly_end_index=metadata.anomaly_end_index,
            comparison_mode=comparison_mode,
            inclusive_anomaly_end=inclusive_anomaly_end,
        )
        series_name = metadata.series_name

    ks_result = stats.ks_2samp(
        train_values, test_values, alternative="two-sided", mode="auto"
    )
    return DriftSignificanceReport(
        file_path=path,
        series_name=series_name,
        total_length=int(series_values.size),
        train_length=int(train_values.size),
        test_length=int(test_values.size),
        ks_statistic=float(ks_result.statistic),
        p_value=float(ks_result.pvalue),
        bh_adjusted_p_value=float(ks_result.pvalue),
        is_significant=False,
        comparison_mode=comparison_mode,
    )


def rank_anomaly_archive_by_ks(
    root_directory: str | Path,
    comparison_mode: str = "pre_vs_post",
    inclusive_anomaly_end: bool = False,
    alpha: float = 0.05,
) -> list[DriftSignificanceReport]:
    reports = [
        compute_ks_drift_report(
            file_path=file_path,
            comparison_mode=comparison_mode,
            inclusive_anomaly_end=inclusive_anomaly_end,
        )
        for file_path in iter_anomaly_archive_files(root_directory)
    ]
    if not reports:
        return []

    adjusted_p_values = _benjamini_hochberg_adjustment(reports)

    ranking_reports = []
    for report_index, report in enumerate(reports):
        bh_adjusted_p_value = adjusted_p_values[report_index]
        ranking_reports.append(
            DriftSignificanceReport(
                file_path=report.file_path,
                series_name=report.series_name,
                total_length=report.total_length,
                train_length=report.train_length,
                test_length=report.test_length,
                ks_statistic=report.ks_statistic,
                p_value=report.p_value,
                bh_adjusted_p_value=bh_adjusted_p_value,
                is_significant=bh_adjusted_p_value <= alpha,
                comparison_mode=report.comparison_mode,
            )
        )

    ranking_reports.sort(
        key=lambda report: (
            report.bh_adjusted_p_value,
            report.p_value,
            report.ks_statistic,
            report.file_path.name,
        )
    )
    return ranking_reports


def _benjamini_hochberg_adjustment(
    reports: list[DriftSignificanceReport],
) -> list[float]:
    sorted_indices = sorted(
        range(len(reports)), key=lambda index: reports[index].p_value
    )
    adjusted_p_values = [0.0] * len(reports)
    running_min = 1.0
    for reverse_rank, report_index in enumerate(reversed(sorted_indices), start=1):
        raw_p_value = reports[report_index].p_value
        bh_value = min(
            raw_p_value * len(reports) / (len(reports) - reverse_rank + 1), 1.0
        )
        running_min = min(running_min, bh_value)
        adjusted_p_values[report_index] = running_min
    return adjusted_p_values
