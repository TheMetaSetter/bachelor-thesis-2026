from __future__ import annotations

"""Build the online performance payload for Table 3.

The online runtime stores one score record per causal window. This script joins
those records with the SMD test labels at the same absolute endpoint indices,
then computes the three report metrics with the repository implementation.
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.metrics.pointwise import compute_pointwise_metrics


METRIC_NAMES = ("vus_pr", "affiliation_f1", "vus_roc")
DEFAULT_WINDOW_SIZE = 20
DEFAULT_VUS_MAX_BUFFER_SIZE = 20
DEFAULT_VUS_NUM_THRESHOLDS = 200
DEFAULT_WORKERS = 8

STREAM_RANGES = {
    "machine_1_6": {
        "label_file": "machine-1-6.txt",
        "absolute_start_index": 146,
        "absolute_end_index": 2200,
    },
    "machine_3_4": {
        "label_file": "machine-3-4.txt",
        "absolute_start_index": 2634,
        "absolute_end_index": 6116,
    },
    "machine_3_9": {
        "label_file": "machine-3-9.txt",
        "absolute_start_index": 1099,
        "absolute_end_index": 10807,
    },
}

METHOD_LABELS = {
    "thesis": "THESIS",
    "m2n2": "M2N2",
    "candi": "CANDI",
    "iforest": "Isolation Forest",
    "kmeans_ad": "KMeansAD",
    "stumpy": "StumPy",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _job_sort_key(job: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(job["method"]),
        str(job.get("offline_variant") or ""),
        str(job["online_variant"]),
        f'{job["entity_id"]}:{job["seed"]}',
    )


def _build_job(manifest_record: dict[str, Any], reporting_root: Path) -> dict[str, Any]:
    entity_id = str(manifest_record["entity_id"])
    if entity_id not in STREAM_RANGES:
        raise ValueError(f"No absolute range configured for {entity_id!r}")
    staged_path = reporting_root / "online_metrics_99" / manifest_record["staged_name"]
    return {
        **manifest_record,
        "local_metric_path": str(staged_path),
        "stream_range": STREAM_RANGES[entity_id],
    }


def _load_online_arrays(
    job: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = _load_json(Path(job["local_metric_path"]))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty metric list: {job['local_metric_path']}")

    start = int(job["stream_range"]["absolute_start_index"])
    end = int(job["stream_range"]["absolute_end_index"])
    window_size = DEFAULT_WINDOW_SIZE
    expected_steps = max(0, end - start - window_size + 1)
    if len(rows) != expected_steps:
        raise ValueError(
            f"Unexpected step count for {job['staged_name']}: "
            f"expected {expected_steps}, found {len(rows)}"
        )

    score_array = np.asarray(
        [row["online/ewma_point_score"] for row in rows], dtype=np.float64
    )
    threshold_values = np.asarray(
        [row["online/threshold"] for row in rows], dtype=np.float64
    )
    prediction_array = np.asarray(
        [row["online/prediction"] for row in rows], dtype=np.int64
    )
    if not np.isfinite(score_array).all() or not np.isfinite(threshold_values).all():
        raise ValueError(f"Non-finite online score or threshold: {job['staged_name']}")
    return rows, score_array, threshold_values, prediction_array


def _load_point_labels(
    job: dict[str, Any],
    *,
    point_start: int,
    point_end: int,
) -> tuple[np.ndarray, Path]:
    label_path = (
        Path("data/ServerMachineDataset/test_label")
        / job["stream_range"]["label_file"]
    )
    full_labels = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
    return full_labels[point_start:point_end], label_path


def _compute_metric_values(
    point_labels: np.ndarray,
    score_array: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | None]:
    metrics = compute_pointwise_metrics(
        point_labels=point_labels,
        point_scores=score_array,
        threshold=threshold,
        vus_max_buffer_size=DEFAULT_VUS_MAX_BUFFER_SIZE,
        vus_num_thresholds=DEFAULT_VUS_NUM_THRESHOLDS,
    )
    return {name: _finite_or_none(metrics[name]) for name in METRIC_NAMES}


def _build_run_record(
    job: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    score_array: np.ndarray,
    threshold_values: np.ndarray,
    prediction_array: np.ndarray,
    point_labels: np.ndarray,
    label_path: Path,
    metric_values: dict[str, float | None],
) -> dict[str, Any]:
    start = int(job["stream_range"]["absolute_start_index"])
    end = int(job["stream_range"]["absolute_end_index"])
    point_start = start + DEFAULT_WINDOW_SIZE - 1

    threshold = float(threshold_values[0])
    expected_predictions = (score_array > threshold).astype(np.int64)
    return {
        "method": job["method"],
        "method_family": job["method_family"],
        "method_label": METHOD_LABELS.get(job["method"], job["method"]),
        "offline_variant": job.get("offline_variant"),
        "online_variant": job.get("online_variant") or "main",
        "entity_id": job["entity_id"],
        "seed": job["seed"],
        "source_metric_file": job["source_path"],
        "staged_metric_file": job["staged_name"],
        "metric_score_field": "online/ewma_point_score",
        "stream": {
            "absolute_start_index": start,
            "absolute_end_index": end,
            "point_start_index": point_start,
            "point_end_index_exclusive": end,
            "window_size": DEFAULT_WINDOW_SIZE,
            "expected_steps": end - start - DEFAULT_WINDOW_SIZE + 1,
            "processed_steps": len(rows),
        },
        "labels": {
            "label_file": str(label_path),
            "positive_count": int(point_labels.sum()),
            "negative_count": int((point_labels == 0).sum()),
            "unique_label_count": int(np.unique(point_labels).size),
        },
        "threshold": {
            "value": threshold,
            "is_constant": bool(np.allclose(threshold_values, threshold)),
            "prediction_mismatch_count": int(
                np.count_nonzero(expected_predictions != prediction_array)
            ),
        },
        "metrics": metric_values,
        "status": "complete",
    }


def _compute_run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    rows, score_array, threshold_values, prediction_array = _load_online_arrays(job)
    start = int(job["stream_range"]["absolute_start_index"])
    end = int(job["stream_range"]["absolute_end_index"])
    point_start = start + DEFAULT_WINDOW_SIZE - 1
    point_labels, label_path = _load_point_labels(
        job, point_start=point_start, point_end=end
    )
    if len(point_labels) != len(score_array):
        raise ValueError(f"Label/score length mismatch: {job['staged_name']}")
    metric_values = _compute_metric_values(
        point_labels,
        score_array,
        threshold=float(threshold_values[0]),
    )
    return _build_run_record(
        job,
        rows=rows,
        score_array=score_array,
        threshold_values=threshold_values,
        prediction_array=prediction_array,
        point_labels=point_labels,
        label_path=label_path,
        metric_values=metric_values,
    )


def _summarize_records(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...],
    standard_deviation_name: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups.setdefault(key, []).append(record)

    summaries = []
    for key, grouped_records in sorted(groups.items(), key=lambda item: item[0]):
        metric_summary = {}
        for metric_name in METRIC_NAMES:
            values = [
                record["metrics"][metric_name]
                for record in grouped_records
                if record["metrics"][metric_name] is not None
            ]
            metric_summary[metric_name] = {
                "mean": _finite_or_none(mean(values)) if values else None,
                standard_deviation_name: (
                    _finite_or_none(pstdev(values)) if len(values) > 1 else 0.0
                ),
                "count": len(values),
            }
        summary = {field: value for field, value in zip(group_fields, key, strict=True)}
        summary["method_label"] = grouped_records[0]["method_label"]
        summary["run_count"] = len(grouped_records)
        summary["metrics"] = metric_summary
        summaries.append(summary)
    return summaries


def build_report(
    *,
    reporting_root: Path,
    manifest_path: Path,
    workers: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    manifest_records = list(manifest["files"])
    if len(manifest_records) != 99:
        raise ValueError(f"Expected 99 online manifest records, found {len(manifest_records)}")
    jobs = sorted(
        [_build_job(record, reporting_root) for record in manifest_records],
        key=_job_sort_key,
    )

    if workers == 1:
        records = [_compute_run_metrics(job) for job in jobs]
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(_compute_run_metrics, jobs))
        except PermissionError:
            # Some restricted environments do not allow multiprocessing
            # semaphores. Threads preserve the same independent run contract.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                records = list(executor.map(_compute_run_metrics, jobs))

    records.sort(key=_job_sort_key)
    return {
        "schema_version": "online_table3_metrics.v1",
        "report_name": "Table 3 - online benchmark performance",
        "source_revision": manifest.get("source_revision"),
        "source_manifest": str(manifest_path),
        "comparison_scope": {
            "dataset": "SMD",
            "methods": sorted({record["method"] for record in records}),
            "run_count": len(records),
            "entity_count": len({record["entity_id"] for record in records}),
            "seed_values": sorted({record["seed"] for record in records}),
        },
        "metric_protocol": {
            "score_field": "online/ewma_point_score",
            "threshold_field": "online/threshold",
            "prediction_field": "online/prediction",
            "window_size": DEFAULT_WINDOW_SIZE,
            "online_window_stride": 1,
            "vus_max_buffer_size": DEFAULT_VUS_MAX_BUFFER_SIZE,
            "vus_num_thresholds": DEFAULT_VUS_NUM_THRESHOLDS,
            "test_labels": "SMD test_label sliced at causal endpoints",
            "point_adjustment": False,
        },
        "stream_ranges": STREAM_RANGES,
        "records": records,
        "summary_by_method_variant": _summarize_records(
            records,
            ("method", "offline_variant", "online_variant"),
            "population_std_across_runs",
        ),
        "summary_by_method_variant_entity": _summarize_records(
            records,
            ("method", "offline_variant", "online_variant", "entity_id"),
            "population_std_across_seeds",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reporting-root",
        type=Path,
        default=Path("reporting/online_phase_tables"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "reporting/online_phase_tables/online_metrics_99/"
            "online_metrics_99_manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    report = build_report(
        reporting_root=args.reporting_root,
        manifest_path=args.manifest,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(f"records={len(report['records'])}")
    print(f"method_variant_rows={len(report['summary_by_method_variant'])}")
    print(f"method_variant_entity_rows={len(report['summary_by_method_variant_entity'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
