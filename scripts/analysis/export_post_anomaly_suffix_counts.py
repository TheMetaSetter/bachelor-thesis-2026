from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.anomaly_archive import _ANOMALY_ARCHIVE_FILENAME_PATTERN

OUTPUT_DIR = PROJECT_ROOT / "documents" / "logs" / "06-27-2026" / "research"
DEFAULT_COLUMNS = [
    "series_id",
    "source_file",
    "total_length",
    "annotation_type",
    "anomaly_end_index",
    "last_anomalous_index",
    "post_anomaly_suffix_len",
    "num_anomalous_points",
    "notes",
]


def _normalize_binary_labels(labels: Iterable[object]) -> np.ndarray:
    label_array = np.asarray(list(labels))
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    try:
        normalized = label_array.astype(np.int64, copy=False)
    except ValueError as error:
        raise ValueError("labels must be coercible to integer values") from error
    return normalized


def build_label_based_suffix_row(
    *,
    series_id: str,
    source_file: Path,
    labels: Iterable[object],
    annotation_type: str,
    notes: str,
) -> dict[str, object]:
    normalized_labels = _normalize_binary_labels(labels)
    anomalous_indices = np.flatnonzero(normalized_labels != 0)
    total_length = int(normalized_labels.shape[0])
    if anomalous_indices.size == 0:
        return {
            "series_id": series_id,
            "source_file": str(source_file),
            "total_length": total_length,
            "annotation_type": annotation_type,
            "anomaly_end_index": None,
            "last_anomalous_index": None,
            "post_anomaly_suffix_len": total_length,
            "num_anomalous_points": 0,
            "notes": "no_anomaly_in_series",
        }

    last_anomalous_index = int(anomalous_indices[-1])
    return {
        "series_id": series_id,
        "source_file": str(source_file),
        "total_length": total_length,
        "annotation_type": annotation_type,
        "anomaly_end_index": None,
        "last_anomalous_index": last_anomalous_index,
        "post_anomaly_suffix_len": total_length - last_anomalous_index - 1,
        "num_anomalous_points": int(anomalous_indices.shape[0]),
        "notes": notes,
    }


def build_ucr_suffix_row(file_path: Path) -> dict[str, object]:
    match = _ANOMALY_ARCHIVE_FILENAME_PATTERN.match(file_path.name)
    if match is None:
        raise ValueError(
            "AnomalyArchive file name must follow "
            "<prefix>_UCR_Anomaly_<series>_<start>_<anomaly_start>_<anomaly_end>.txt"
        )
    values = np.fromstring(file_path.read_text(encoding="utf-8"), sep=" ", dtype=float)
    if values.size == 0:
        raise ValueError(f"No numeric values found in {file_path}")
    anomaly_start_index = int(match.group("anomaly_start_index"))
    anomaly_end_index = int(match.group("anomaly_end_index"))
    return {
        "series_id": match.group("series_name"),
        "source_file": str(file_path),
        "total_length": int(values.shape[0]),
        "annotation_type": "filename_anomaly_interval",
        "anomaly_end_index": anomaly_end_index,
        "last_anomalous_index": None,
        "post_anomaly_suffix_len": int(values.shape[0]) - anomaly_end_index,
        "num_anomalous_points": anomaly_end_index - anomaly_start_index,
        "notes": "anomaly_end_index_from_filename",
    }


def build_swat_suffix_row(
    *,
    series_id: str,
    source_file: Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    label_column = next(
        (column for column in frame.columns if str(column).strip() == "Normal/Attack"),
        None,
    )
    if label_column is None:
        raise ValueError(
            f"SWaT file is missing the Normal/Attack column: {source_file}"
        )
    labels = frame[label_column].astype(str).str.strip().eq("Attack").astype(int)
    return build_label_based_suffix_row(
        series_id=series_id,
        source_file=source_file,
        labels=labels.to_numpy(),
        annotation_type="attack_label_column",
        notes="last_attack_label_in_csv",
    )


def build_iops_suffix_row(
    *,
    series_id: str,
    source_file: Path,
    frame: pd.DataFrame,
) -> dict[str, object]:
    if frame.shape[1] != 2:
        raise ValueError(
            f"IOPS test file must have exactly 2 columns [Value, Label]: {source_file}"
        )
    return build_label_based_suffix_row(
        series_id=series_id,
        source_file=source_file,
        labels=frame.iloc[:, 1].to_numpy(),
        annotation_type="pointwise_label_column",
        notes="last_nonzero_label_in_test_stream",
    )


def extract_ucr_rows(dataset_root: Path) -> list[dict[str, object]]:
    return [
        build_ucr_suffix_row(file_path)
        for file_path in sorted(dataset_root.glob("*.txt"))
    ]


def extract_smd_rows(dataset_root: Path) -> list[dict[str, object]]:
    label_dir = dataset_root / "test_label"
    rows: list[dict[str, object]] = []
    for label_file in sorted(label_dir.glob("*.txt")):
        entity_id = label_file.stem
        labels = np.loadtxt(label_file, delimiter=",", dtype=np.int64, ndmin=1)
        rows.append(
            build_label_based_suffix_row(
                series_id=entity_id,
                source_file=label_file,
                labels=labels,
                annotation_type="pointwise_test_labels",
                notes="last_nonzero_label_in_test_stream",
            )
        )
    return rows


def extract_swat_rows(dataset_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in sorted(dataset_root.glob("*.csv")):
        frame = pd.read_csv(file_path)
        rows.append(
            build_swat_suffix_row(
                series_id=file_path.name,
                source_file=file_path,
                frame=frame,
            )
        )
    return rows


def extract_iops_rows(dataset_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in sorted(dataset_root.glob("*.test.out")):
        frame = pd.read_csv(file_path, header=None)
        rows.append(
            build_iops_suffix_row(
                series_id=file_path.name.removesuffix(".test.out"),
                source_file=file_path,
                frame=frame,
            )
        )
    return rows


def _write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in DEFAULT_COLUMNS})


def _print_summary(dataset_name: str, rows: list[dict[str, object]]) -> None:
    suffix_lengths = [int(row["post_anomaly_suffix_len"]) for row in rows]
    zero_suffix_count = sum(length == 0 for length in suffix_lengths)
    print(
        f"{dataset_name}: exported={len(rows)}, "
        f"min_suffix={min(suffix_lengths)}, "
        f"max_suffix={max(suffix_lengths)}, "
        f"zero_suffix={zero_suffix_count}"
    )


def main() -> None:
    dataset_exports = [
        (
            "UCR",
            PROJECT_ROOT / "data" / "AnomalyArchive",
            OUTPUT_DIR / "ucr_post_anomaly_suffix_counts.csv",
            extract_ucr_rows,
        ),
        (
            "SMD",
            PROJECT_ROOT / "data" / "ServerMachineDataset",
            OUTPUT_DIR / "smd_post_anomaly_suffix_counts.csv",
            extract_smd_rows,
        ),
        (
            "SWaT",
            PROJECT_ROOT / "data" / "SWaT",
            OUTPUT_DIR / "swat_post_anomaly_suffix_counts.csv",
            extract_swat_rows,
        ),
        (
            "IOPS",
            PROJECT_ROOT / "data" / "IOPS",
            OUTPUT_DIR / "iops_post_anomaly_suffix_counts.csv",
            extract_iops_rows,
        ),
    ]

    for dataset_name, dataset_root, output_path, extractor in dataset_exports:
        rows = extractor(dataset_root)
        _write_rows(output_path, rows)
        _print_summary(dataset_name, rows)
        print(output_path)


if __name__ == "__main__":
    main()
