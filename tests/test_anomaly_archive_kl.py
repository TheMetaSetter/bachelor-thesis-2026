from __future__ import annotations

import numpy as np

from src.analysis.anomaly_archive_kl import (
    build_kl_report,
    compute_ks_drift_report,
    estimate_histogram_kl_divergence,
    parse_anomaly_archive_filename,
    rank_anomaly_archive_by_kl,
    rank_anomaly_archive_by_ks,
    split_series_by_annotation,
)


def test_parse_anomaly_archive_filename_extracts_annotation_fields() -> None:
    metadata = parse_anomaly_archive_filename(
        "053_UCR_Anomaly_DISTORTEDWalkingAceleration1_1500_2764_2995.txt"
    )

    assert metadata is not None
    assert metadata.series_name == "DISTORTEDWalkingAceleration1"
    assert metadata.start_index == 1500
    assert metadata.anomaly_start_index == 2764
    assert metadata.anomaly_end_index == 2995


def test_split_series_by_annotation_uses_pre_and_post_segments() -> None:
    values = np.arange(10, dtype=np.float64)

    train_values, test_values = split_series_by_annotation(
        values,
        anomaly_start_index=3,
        anomaly_end_index=7,
    )

    assert np.array_equal(train_values, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(test_values, np.array([7.0, 8.0, 9.0]))


def test_histogram_kl_is_zero_for_identical_inputs() -> None:
    values = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

    kl_value = estimate_histogram_kl_divergence(values, values, bins=4)

    assert kl_value == 0.0


def test_build_kl_report_works_on_shortest_archive_series() -> None:
    report = build_kl_report(
        "data/AnomalyArchive/053_UCR_Anomaly_DISTORTEDWalkingAceleration1_1500_2764_2995.txt"
    )

    assert report.total_length == 6684
    assert report.train_length == 2764
    assert report.test_length == 3689
    assert report.kl_train_to_test >= 0.0
    assert report.kl_test_to_train >= 0.0


def test_rank_anomaly_archive_by_kl_orders_larger_shift_first(tmp_path) -> None:
    first_file = tmp_path / "001_UCR_Anomaly_ExampleA_0_2_4.txt"
    second_file = tmp_path / "002_UCR_Anomaly_ExampleB_0_2_4.txt"
    first_file.write_text("0 0 0 10 10 10", encoding="utf-8")
    second_file.write_text("1 1 1 1 1 1", encoding="utf-8")

    ranking_entries = rank_anomaly_archive_by_kl(tmp_path)

    assert ranking_entries[0].report.file_path.name == first_file.name
    assert (
        ranking_entries[0].report.kl_train_to_test
        >= ranking_entries[1].report.kl_train_to_test
    )


def test_ks_drift_report_flags_identical_segments_as_not_significant(tmp_path) -> None:
    file_path = tmp_path / "001_UCR_Anomaly_Example_0_2_4.txt"
    file_path.write_text("1 1 1 1 1 1", encoding="utf-8")

    report = compute_ks_drift_report(file_path)

    assert report.ks_statistic == 0.0
    assert report.p_value == 1.0
    assert report.is_significant is False


def test_rank_anomaly_archive_by_ks_applies_bh_correction(tmp_path) -> None:
    shifted_file = tmp_path / "001_UCR_Anomaly_Shifted_0_2_4.txt"
    stable_file = tmp_path / "002_UCR_Anomaly_Stable_0_2_4.txt"
    shifted_file.write_text("0 0 0 10 10 10", encoding="utf-8")
    stable_file.write_text("1 1 1 1 1 1", encoding="utf-8")

    ranking_reports = rank_anomaly_archive_by_ks(tmp_path, alpha=0.05)

    assert ranking_reports[0].file_path.name == shifted_file.name
    assert (
        ranking_reports[0].bh_adjusted_p_value <= ranking_reports[1].bh_adjusted_p_value
    )
