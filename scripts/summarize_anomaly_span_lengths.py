from __future__ import annotations

import argparse
import ast
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "documents" / "logs" / "07-03-2026" / "research"
DEFAULT_DATASETS = ("anomaly_archive", "nasa", "smd", "iops", "swat")
SPAN_COLUMNS = [
    "dataset_name",
    "series_id",
    "source_file",
    "span_start_index",
    "span_end_index_exclusive",
    "span_length",
    "span_ordinal_in_series",
    "split",
    "entity_id",
    "spacecraft",
]
SUMMARY_COLUMNS = [
    "dataset_name",
    "num_series",
    "num_spans",
    "num_anomalous_points",
    "mean_span_length",
    "median_span_length",
    "std_span_length",
    "min_span_length",
    "max_span_length",
    "p25",
    "p75",
    "p90",
    "p95",
    "num_zero_anomaly_series",
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


def labels_to_spans(labels: Iterable[object]) -> list[tuple[int, int]]:
    normalized_labels = _normalize_binary_labels(labels)
    spans: list[tuple[int, int]] = []
    start_index: int | None = None

    for index, label in enumerate(normalized_labels.tolist()):
        is_anomalous = int(label) != 0
        if is_anomalous and start_index is None:
            start_index = index
        elif not is_anomalous and start_index is not None:
            spans.append((start_index, index))
            start_index = None

    if start_index is not None:
        spans.append((start_index, int(normalized_labels.shape[0])))
    return spans


def build_span_rows_from_labels(
    *,
    dataset_name: str,
    series_id: str,
    source_file: Path,
    labels: Iterable[object],
    split: str = "test",
    entity_id: str | None = None,
    spacecraft: str | None = None,
) -> list[dict[str, object]]:
    spans = labels_to_spans(labels)
    rows: list[dict[str, object]] = []
    for span_ordinal, (start_index, end_index_exclusive) in enumerate(spans, start=1):
        rows.append(
            {
                "dataset_name": dataset_name,
                "series_id": series_id,
                "source_file": str(source_file),
                "span_start_index": start_index,
                "span_end_index_exclusive": end_index_exclusive,
                "span_length": end_index_exclusive - start_index,
                "span_ordinal_in_series": span_ordinal,
                "split": split,
                "entity_id": entity_id,
                "spacecraft": spacecraft,
            }
        )
    return rows


def build_anomaly_archive_span_rows(file_path: Path) -> list[dict[str, object]]:
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
    return [
        {
            "dataset_name": "anomaly_archive",
            "series_id": match.group("series_name"),
            "source_file": str(file_path),
            "span_start_index": anomaly_start_index,
            "span_end_index_exclusive": anomaly_end_index,
            "span_length": anomaly_end_index - anomaly_start_index,
            "span_ordinal_in_series": 1,
            "split": "test",
            "entity_id": match.group("series_name"),
            "spacecraft": None,
        }
    ]


def build_nasa_span_rows(metadata_path: Path) -> list[dict[str, object]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"NASA metadata file does not exist: {metadata_path}")
    metadata_frame = pd.read_csv(metadata_path)
    required_columns = {
        "chan_id",
        "spacecraft",
        "anomaly_sequences",
        "num_values",
    }
    missing_columns = required_columns.difference(metadata_frame.columns)
    if missing_columns:
        raise ValueError(
            f"NASA metadata file is missing required columns {sorted(missing_columns)}: {metadata_path}"
        )

    rows: list[dict[str, object]] = []
    for metadata_row in metadata_frame.to_dict(orient="records"):
        chan_id = str(metadata_row["chan_id"])
        spacecraft = str(metadata_row["spacecraft"])
        anomaly_sequences = ast.literal_eval(str(metadata_row["anomaly_sequences"]))
        for span_ordinal, interval in enumerate(anomaly_sequences, start=1):
            start_index = int(interval[0])
            end_index_exclusive = int(interval[1])
            rows.append(
                {
                    "dataset_name": "nasa",
                    "series_id": chan_id,
                    "source_file": str(metadata_path),
                    "span_start_index": start_index,
                    "span_end_index_exclusive": end_index_exclusive,
                    "span_length": end_index_exclusive - start_index,
                    "span_ordinal_in_series": span_ordinal,
                    "split": "test",
                    "entity_id": chan_id,
                    "spacecraft": spacecraft,
                }
            )
    return rows


def build_smd_span_rows(dataset_root: Path) -> list[dict[str, object]]:
    label_dir = dataset_root / "test_label"
    if not label_dir.exists():
        raise FileNotFoundError(f"SMD test_label directory does not exist: {label_dir}")
    rows: list[dict[str, object]] = []
    for label_file in sorted(label_dir.glob("*.txt")):
        labels = np.loadtxt(label_file, delimiter=",", dtype=np.int64, ndmin=1)
        rows.extend(
            build_span_rows_from_labels(
                dataset_name="smd",
                series_id=label_file.stem,
                source_file=label_file,
                labels=labels,
                split="test",
                entity_id=label_file.stem,
            )
        )
    return rows


def build_iops_span_rows(dataset_root: Path) -> list[dict[str, object]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"IOPS directory does not exist: {dataset_root}")
    rows: list[dict[str, object]] = []
    for file_path in sorted(dataset_root.glob("*.test.out")):
        frame = pd.read_csv(file_path, header=None)
        if frame.shape[1] != 2:
            raise ValueError(
                f"IOPS test file must have exactly 2 columns [Value, Label]: {file_path}"
            )
        rows.extend(
            build_span_rows_from_labels(
                dataset_name="iops",
                series_id=file_path.name.removesuffix(".test.out"),
                source_file=file_path,
                labels=frame.iloc[:, 1].to_numpy(),
                split="test",
                entity_id=file_path.name.removesuffix(".test.out"),
            )
        )
    return rows


def build_swat_span_rows(dataset_root: Path) -> list[dict[str, object]]:
    merged_file = dataset_root / "merged.csv"
    if not merged_file.exists():
        raise FileNotFoundError(f"SWaT merged.csv does not exist: {merged_file}")
    frame = pd.read_csv(merged_file)
    label_column = next(
        (column for column in frame.columns if str(column).strip() == "Normal/Attack"),
        None,
    )
    if label_column is None:
        raise ValueError(f"SWaT file is missing the Normal/Attack column: {merged_file}")
    labels = frame[label_column].astype(str).str.strip().eq("Attack").astype(int)
    return build_span_rows_from_labels(
        dataset_name="swat",
        series_id="merged.csv",
        source_file=merged_file,
        labels=labels.to_numpy(),
        split="test",
        entity_id="merged.csv",
    )


def _register_zero_anomaly_series(
    *,
    dataset_name: str,
    series_id: str,
    source_file: Path,
    split: str,
    entity_id: str | None = None,
    spacecraft: str | None = None,
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "series_id": series_id,
        "source_file": str(source_file),
        "split": split,
        "entity_id": entity_id,
        "spacecraft": spacecraft,
    }


def collect_series_registry(data_root: Path, selected_datasets: Iterable[str]) -> list[dict[str, object]]:
    normalized_datasets = tuple(selected_datasets)
    series_rows: list[dict[str, object]] = []

    if "anomaly_archive" in normalized_datasets:
        dataset_root = data_root / "AnomalyArchive"
        if not dataset_root.exists():
            raise FileNotFoundError(f"AnomalyArchive directory does not exist: {dataset_root}")
        for file_path in sorted(dataset_root.glob("*.txt")):
            match = _ANOMALY_ARCHIVE_FILENAME_PATTERN.match(file_path.name)
            if match is None:
                raise ValueError(f"Invalid AnomalyArchive file name: {file_path}")
            series_id = str(match.group("series_name"))
            series_rows.append(
                _register_zero_anomaly_series(
                    dataset_name="anomaly_archive",
                    series_id=series_id,
                    source_file=file_path,
                    split="test",
                    entity_id=series_id,
                )
            )

    if "nasa" in normalized_datasets:
        metadata_path = data_root / "NASA" / "labeled_anomalies.csv"
        metadata_frame = pd.read_csv(metadata_path)
        for metadata_row in metadata_frame.to_dict(orient="records"):
            chan_id = str(metadata_row["chan_id"])
            series_rows.append(
                _register_zero_anomaly_series(
                    dataset_name="nasa",
                    series_id=chan_id,
                    source_file=metadata_path,
                    split="test",
                    entity_id=chan_id,
                    spacecraft=str(metadata_row["spacecraft"]),
                )
            )

    if "smd" in normalized_datasets:
        label_dir = data_root / "ServerMachineDataset" / "test_label"
        if not label_dir.exists():
            raise FileNotFoundError(f"SMD test_label directory does not exist: {label_dir}")
        for label_file in sorted(label_dir.glob("*.txt")):
            series_rows.append(
                _register_zero_anomaly_series(
                    dataset_name="smd",
                    series_id=label_file.stem,
                    source_file=label_file,
                    split="test",
                    entity_id=label_file.stem,
                )
            )

    if "iops" in normalized_datasets:
        dataset_root = data_root / "IOPS"
        if not dataset_root.exists():
            raise FileNotFoundError(f"IOPS directory does not exist: {dataset_root}")
        for file_path in sorted(dataset_root.glob("*.test.out")):
            series_id = file_path.name.removesuffix(".test.out")
            series_rows.append(
                _register_zero_anomaly_series(
                    dataset_name="iops",
                    series_id=series_id,
                    source_file=file_path,
                    split="test",
                    entity_id=series_id,
                )
            )

    if "swat" in normalized_datasets:
        merged_file = data_root / "SWaT" / "merged.csv"
        if not merged_file.exists():
            raise FileNotFoundError(f"SWaT merged.csv does not exist: {merged_file}")
        series_rows.append(
            _register_zero_anomaly_series(
                dataset_name="swat",
                series_id="merged.csv",
                source_file=merged_file,
                split="test",
                entity_id="merged.csv",
            )
        )

    return series_rows


def collect_span_rows(data_root: Path, selected_datasets: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dataset_set = tuple(selected_datasets)
    if "anomaly_archive" in dataset_set:
        dataset_root = data_root / "AnomalyArchive"
        if not dataset_root.exists():
            raise FileNotFoundError(f"AnomalyArchive directory does not exist: {dataset_root}")
        for file_path in sorted(dataset_root.glob("*.txt")):
            rows.extend(build_anomaly_archive_span_rows(file_path))
    if "nasa" in dataset_set:
        rows.extend(build_nasa_span_rows(data_root / "NASA" / "labeled_anomalies.csv"))
    if "smd" in dataset_set:
        rows.extend(build_smd_span_rows(data_root / "ServerMachineDataset"))
    if "iops" in dataset_set:
        rows.extend(build_iops_span_rows(data_root / "IOPS"))
    if "swat" in dataset_set:
        rows.extend(build_swat_span_rows(data_root / "SWaT"))
    return rows


def summarize_span_rows(
    span_rows: list[dict[str, object]],
    series_registry: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    span_frame = pd.DataFrame(span_rows)
    registry_frame = pd.DataFrame(series_registry)
    for dataset_name in sorted(registry_frame["dataset_name"].unique().tolist()):
        dataset_registry = registry_frame.loc[registry_frame["dataset_name"] == dataset_name]
        dataset_spans = span_frame.loc[span_frame["dataset_name"] == dataset_name] if not span_frame.empty else pd.DataFrame()
        span_lengths = dataset_spans["span_length"].astype(float).to_numpy() if not dataset_spans.empty else np.asarray([], dtype=float)
        anomalous_series = (
            set(dataset_spans["series_id"].tolist()) if not dataset_spans.empty else set()
        )
        num_series = int(dataset_registry["series_id"].nunique())
        num_zero_anomaly_series = num_series - len(anomalous_series)
        if span_lengths.size == 0:
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "num_series": num_series,
                    "num_spans": 0,
                    "num_anomalous_points": 0,
                    "mean_span_length": None,
                    "median_span_length": None,
                    "std_span_length": None,
                    "min_span_length": None,
                    "max_span_length": None,
                    "p25": None,
                    "p75": None,
                    "p90": None,
                    "p95": None,
                    "num_zero_anomaly_series": num_zero_anomaly_series,
                }
            )
            continue
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "num_series": num_series,
                "num_spans": int(span_lengths.shape[0]),
                "num_anomalous_points": int(span_lengths.sum()),
                "mean_span_length": float(np.mean(span_lengths)),
                "median_span_length": float(np.median(span_lengths)),
                "std_span_length": float(np.std(span_lengths, ddof=0)),
                "min_span_length": int(np.min(span_lengths)),
                "max_span_length": int(np.max(span_lengths)),
                "p25": float(np.percentile(span_lengths, 25)),
                "p75": float(np.percentile(span_lengths, 75)),
                "p90": float(np.percentile(span_lengths, 90)),
                "p95": float(np.percentile(span_lengths, 95)),
                "num_zero_anomaly_series": num_zero_anomaly_series,
            }
        )
    return summary_rows


def _write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def _write_markdown_summary(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    selected_datasets: Iterable[str],
) -> None:
    lines = [
        "# Research: Anomaly span length summary",
        "",
        "This note summarizes anomaly span lengths extracted directly from local datasets under `data/`.",
        "",
        "## Assumptions",
        "",
        "- SWaT uses merged.csv.",
        "- NASA uses half-open intervals [start, end).",
        "- AnomalyArchive uses half-open filename intervals [anomaly_start_index, anomaly_end_index).",
        "- SMD uses ServerMachineDataset test_label files only.",
        "",
        "## Dataset summary",
        "",
        "| dataset | num_series | num_spans | num_anomalous_points | mean | median | min | max | zero_anomaly_series |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset_name} | {num_series} | {num_spans} | {num_anomalous_points} | {mean_span_length} | {median_span_length} | {min_span_length} | {max_span_length} | {num_zero_anomaly_series} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Selected datasets: {', '.join(selected_datasets)}",
            "- Series without anomaly spans are omitted from the span-level CSV but still counted in dataset-level summary.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_dataset_selection(dataset_argument: str | None) -> tuple[str, ...]:
    if dataset_argument is None or dataset_argument.strip() == "":
        return DEFAULT_DATASETS
    selected_datasets = tuple(
        dataset_name.strip().lower()
        for dataset_name in dataset_argument.split(",")
        if dataset_name.strip()
    )
    unknown_datasets = sorted(set(selected_datasets).difference(DEFAULT_DATASETS))
    if unknown_datasets:
        raise ValueError(
            f"Unsupported dataset names {unknown_datasets}. Supported datasets: {list(DEFAULT_DATASETS)}"
        )
    return selected_datasets


def run_span_summary(
    *,
    data_root: Path,
    output_dir: Path,
    selected_datasets: Iterable[str] | None = None,
) -> None:
    normalized_selected_datasets = tuple(selected_datasets or DEFAULT_DATASETS)
    series_registry = collect_series_registry(data_root, normalized_selected_datasets)
    span_rows = collect_span_rows(data_root, normalized_selected_datasets)
    summary_rows = summarize_span_rows(span_rows, series_registry)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "anomaly_span_lengths.csv", SPAN_COLUMNS, span_rows)
    _write_csv(output_dir / "anomaly_span_length_summary.csv", SUMMARY_COLUMNS, summary_rows)
    _write_markdown_summary(
        output_dir / "research-anomaly-span-length-summary.md",
        summary_rows,
        normalized_selected_datasets,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize anomaly span lengths for selected datasets under data/."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV and markdown outputs will be written.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names: anomaly_archive,nasa,smd,iops,swat",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    selected_datasets = parse_dataset_selection(args.datasets)
    run_span_summary(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        selected_datasets=selected_datasets,
    )


if __name__ == "__main__":
    main()
