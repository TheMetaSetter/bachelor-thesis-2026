from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.export_post_anomaly_suffix_counts import (
    build_iops_suffix_row,
    build_label_based_suffix_row,
    build_swat_suffix_row,
    build_ucr_suffix_row,
)


def test_build_ucr_suffix_row_uses_filename_anomaly_end_index(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "123_UCR_Anomaly_DemoSeries_10_40_55.txt"
    file_path.write_text(" ".join(str(index) for index in range(100)), encoding="utf-8")

    row = build_ucr_suffix_row(file_path)

    assert row["series_id"] == "DemoSeries"
    assert row["total_length"] == 100
    assert row["anomaly_end_index"] == 55
    assert row["post_anomaly_suffix_len"] == 45
    assert row["num_anomalous_points"] == 15


def test_build_label_based_suffix_row_uses_last_anomalous_point() -> None:
    row = build_label_based_suffix_row(
        series_id="machine-1-1",
        source_file=Path("machine-1-1.txt"),
        labels=[0, 1, 0, 1, 0, 0],
        annotation_type="pointwise_test_labels",
        notes="last_nonzero_label_in_test_stream",
    )

    assert row["total_length"] == 6
    assert row["last_anomalous_index"] == 3
    assert row["post_anomaly_suffix_len"] == 2
    assert row["num_anomalous_points"] == 2


def test_build_swat_suffix_row_uses_last_attack_label() -> None:
    frame = pd.DataFrame(
        {
            "Normal/Attack": ["Normal", "Attack", "Normal", "Attack", "Normal"],
        }
    )

    row = build_swat_suffix_row(
        series_id="attack.csv",
        source_file=Path("attack.csv"),
        frame=frame,
    )

    assert row["total_length"] == 5
    assert row["last_anomalous_index"] == 3
    assert row["post_anomaly_suffix_len"] == 1
    assert row["num_anomalous_points"] == 2


def test_build_iops_suffix_row_reads_second_column_as_label() -> None:
    frame = pd.DataFrame(
        [
            [37.15, 0],
            [36.74, 1],
            [37.59, 0],
            [37.80, 1],
            [37.10, 0],
        ]
    )

    row = build_iops_suffix_row(
        series_id="KPI-demo",
        source_file=Path("KPI-demo.test.out"),
        frame=frame,
    )

    assert row["total_length"] == 5
    assert row["last_anomalous_index"] == 3
    assert row["post_anomaly_suffix_len"] == 1
    assert row["num_anomalous_points"] == 2


def test_build_label_based_suffix_row_marks_series_without_anomaly() -> None:
    row = build_label_based_suffix_row(
        series_id="normal-only",
        source_file=Path("normal.csv"),
        labels=[0, 0, 0, 0],
        annotation_type="pointwise_test_labels",
        notes="last_nonzero_label_in_test_stream",
    )

    assert row["last_anomalous_index"] is None
    assert row["post_anomaly_suffix_len"] == 4
    assert row["num_anomalous_points"] == 0
    assert row["notes"] == "no_anomaly_in_series"
