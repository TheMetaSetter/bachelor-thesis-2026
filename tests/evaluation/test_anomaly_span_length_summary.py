from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from scripts.summarize_anomaly_span_lengths import (
    build_anomaly_archive_span_rows,
    build_iops_span_rows,
    build_nasa_span_rows,
    build_span_rows_from_labels,
    build_smd_span_rows,
    build_swat_span_rows,
    labels_to_spans,
    run_span_summary,
)


def test_labels_to_spans_returns_empty_for_normal_only_series() -> None:
    assert labels_to_spans([0, 0, 0, 0]) == []


def test_labels_to_spans_extracts_single_half_open_span() -> None:
    assert labels_to_spans([0, 1, 1, 1, 0]) == [(1, 4)]


def test_labels_to_spans_extracts_multiple_disjoint_spans() -> None:
    assert labels_to_spans([0, 1, 1, 0, 1, 0, 1, 1]) == [(1, 3), (4, 5), (6, 8)]


def test_labels_to_spans_handles_spans_at_series_boundaries() -> None:
    assert labels_to_spans([1, 1, 0, 0, 1]) == [(0, 2), (4, 5)]


def test_build_span_rows_from_labels_keeps_span_metadata() -> None:
    rows = build_span_rows_from_labels(
        dataset_name="demo",
        series_id="series-1",
        source_file=Path("demo.txt"),
        labels=[0, 1, 1, 0, 1],
        split="test",
        entity_id="entity-1",
    )

    assert len(rows) == 2
    assert rows[0]["span_start_index"] == 1
    assert rows[0]["span_end_index_exclusive"] == 3
    assert rows[0]["span_length"] == 2
    assert rows[0]["span_ordinal_in_series"] == 1
    assert rows[0]["entity_id"] == "entity-1"
    assert rows[1]["span_start_index"] == 4
    assert rows[1]["span_end_index_exclusive"] == 5
    assert rows[1]["span_ordinal_in_series"] == 2


def test_build_anomaly_archive_span_rows_uses_filename_interval(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "123_UCR_Anomaly_DemoSeries_10_40_55.txt"
    file_path.write_text(" ".join(str(index) for index in range(100)), encoding="utf-8")

    rows = build_anomaly_archive_span_rows(file_path)

    assert len(rows) == 1
    assert rows[0]["dataset_name"] == "anomaly_archive"
    assert rows[0]["series_id"] == "DemoSeries"
    assert rows[0]["span_start_index"] == 40
    assert rows[0]["span_end_index_exclusive"] == 55
    assert rows[0]["span_length"] == 15


def test_build_nasa_span_rows_uses_half_open_anomaly_sequences(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "labeled_anomalies.csv"
    metadata_path.write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                'A-1,SMAP,"[[2, 5], [8, 9]]","[point, point]",12',
            ]
        ),
        encoding="utf-8",
    )

    rows = build_nasa_span_rows(metadata_path)

    assert [row["span_length"] for row in rows] == [3, 1]
    assert rows[0]["span_start_index"] == 2
    assert rows[0]["span_end_index_exclusive"] == 5
    assert rows[1]["span_start_index"] == 8
    assert rows[1]["span_end_index_exclusive"] == 9
    assert rows[0]["spacecraft"] == "SMAP"


def test_build_smd_span_rows_extracts_contiguous_segments(tmp_path: Path) -> None:
    label_dir = tmp_path / "ServerMachineDataset" / "test_label"
    label_dir.mkdir(parents=True)
    (label_dir / "machine-1-1.txt").write_text("0,1,1,0,1,0", encoding="utf-8")

    rows = build_smd_span_rows(tmp_path / "ServerMachineDataset")

    assert [
        (row["span_start_index"], row["span_end_index_exclusive"]) for row in rows
    ] == [
        (1, 3),
        (4, 5),
    ]


def test_build_iops_span_rows_reads_second_column_as_label(tmp_path: Path) -> None:
    dataset_root = tmp_path / "IOPS"
    dataset_root.mkdir()
    (dataset_root / "KPI-demo.test.out").write_text(
        "\n".join(["1.0,0", "2.0,1", "3.0,1", "4.0,0"]),
        encoding="utf-8",
    )

    rows = build_iops_span_rows(dataset_root)

    assert len(rows) == 1
    assert rows[0]["series_id"] == "KPI-demo"
    assert rows[0]["span_start_index"] == 1
    assert rows[0]["span_end_index_exclusive"] == 3


def test_build_swat_span_rows_uses_merged_csv_attack_column(tmp_path: Path) -> None:
    dataset_root = tmp_path / "SWaT"
    dataset_root.mkdir()
    pd.DataFrame(
        {
            "Normal/Attack": ["Normal", "Attack", "Attack", "Normal", "Attack"],
        }
    ).to_csv(dataset_root / "merged.csv", index=False)

    rows = build_swat_span_rows(dataset_root)

    assert [
        (row["span_start_index"], row["span_end_index_exclusive"]) for row in rows
    ] == [
        (1, 3),
        (4, 5),
    ]


def test_run_span_summary_writes_csv_and_markdown_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    output_dir = tmp_path / "outputs"

    anomaly_archive_root = data_root / "AnomalyArchive"
    anomaly_archive_root.mkdir(parents=True)
    (anomaly_archive_root / "123_UCR_Anomaly_DemoSeries_10_40_55.txt").write_text(
        " ".join(str(index) for index in range(100)),
        encoding="utf-8",
    )

    nasa_root = data_root / "NASA"
    nasa_root.mkdir()
    (nasa_root / "labeled_anomalies.csv").write_text(
        "\n".join(
            [
                "chan_id,spacecraft,anomaly_sequences,class,num_values",
                'A-1,SMAP,"[[2, 5], [8, 9]]","[point, point]",12',
            ]
        ),
        encoding="utf-8",
    )

    smd_label_dir = data_root / "ServerMachineDataset" / "test_label"
    smd_label_dir.mkdir(parents=True)
    (smd_label_dir / "machine-1-1.txt").write_text("0,1,1,0,1,0", encoding="utf-8")

    iops_root = data_root / "IOPS"
    iops_root.mkdir()
    (iops_root / "KPI-demo.test.out").write_text(
        "\n".join(["1.0,0", "2.0,1", "3.0,1", "4.0,0"]),
        encoding="utf-8",
    )

    swat_root = data_root / "SWaT"
    swat_root.mkdir()
    pd.DataFrame(
        {
            "Normal/Attack": ["Normal", "Attack", "Attack", "Normal", "Attack"],
        }
    ).to_csv(swat_root / "merged.csv", index=False)

    run_span_summary(data_root=data_root, output_dir=output_dir)

    span_csv = output_dir / "anomaly_span_lengths.csv"
    summary_csv = output_dir / "anomaly_span_length_summary.csv"
    summary_md = output_dir / "research-anomaly-span-length-summary.md"

    assert span_csv.exists()
    assert summary_csv.exists()
    assert summary_md.exists()

    with span_csv.open(encoding="utf-8", newline="") as handle:
        span_rows = list(csv.DictReader(handle))
    assert len(span_rows) == 8

    with summary_csv.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert {row["dataset_name"] for row in summary_rows} == {
        "anomaly_archive",
        "nasa",
        "smd",
        "iops",
        "swat",
    }
    assert "mean_span_length" in summary_rows[0]
    assert "p95" in summary_rows[0]

    markdown_text = summary_md.read_text(encoding="utf-8")
    assert "SWaT uses merged.csv" in markdown_text
    assert "NASA uses half-open intervals [start, end)." in markdown_text
