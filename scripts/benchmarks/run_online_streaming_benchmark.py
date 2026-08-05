from __future__ import annotations

"""Online streaming benchmark launcher for the baseline sweep.

₍^. .^₎⟆ Benchmark flow

benchmark config + protocol config
  -> load scaled SMD splits
  -> fit / initialize baseline
  -> calibrate on clean validation
  -> run on test stream
  -> write shared report and threshold artifacts
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.baselines.online import (
    CANDIStreamingBaseline,
    IForestStreamingBaseline,
    KMeansADStreamingBaseline,
    M2N2StreamingBaseline,
    OnlineStreamingBaselineProtocol,
    StumpyChannelABStreamingBaseline,
)
from src.core.config import load_yaml_config
from src.core.registry import build_dataset
from src.core.runtime_components import register_evaluation_runtime_components
from src.protocols.online_stream_range import select_online_stream_sequence
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)
from src.protocols.smd_benchmark_protocol import validate_protocol_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_BUILDERS: dict[str, Callable[..., OnlineStreamingBaselineProtocol]] = {
    "candi": CANDIStreamingBaseline,
    "m2n2": M2N2StreamingBaseline,
    "stumpy": StumpyChannelABStreamingBaseline,
    "kmeans_ad": KMeansADStreamingBaseline,
    "iforest": IForestStreamingBaseline,
}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_json_config(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    if not path.is_absolute():
        path = (REPOSITORY_ROOT / path).resolve()
    return load_yaml_config(path)


def _apply_data_overrides(
    data_config: dict[str, Any], data_overrides: dict[str, Any] | None
) -> dict[str, Any]:
    if not data_overrides:
        return data_config
    merged_data_config = dict(data_config)
    for key, value in data_overrides.items():
        merged_data_config[key] = value
    return merged_data_config


def _truncate_sequence_to_max_online_steps(
    sequence: dict[str, Any],
    *,
    window_size: int,
    max_online_steps: int | None,
) -> dict[str, Any]:
    if max_online_steps is None or max_online_steps <= 0:
        return sequence
    sequence_x = _to_numpy(sequence["x"], dtype=np.float64)
    if sequence_x.shape[0] <= window_size:
        return sequence
    max_points = min(sequence_x.shape[0], window_size + max_online_steps - 1)
    truncated_sequence = dict(sequence)
    truncated_sequence["x"] = sequence_x[:max_points]
    if "point_labels" in sequence:
        point_labels = _to_numpy(sequence["point_labels"], dtype=np.int64).reshape(-1)
        truncated_sequence["point_labels"] = point_labels[:max_points]
    if "mask" in sequence and sequence["mask"] is not None:
        mask = _to_numpy(sequence["mask"], dtype=np.int64).reshape(-1)
        truncated_sequence["mask"] = mask[:max_points]
    if "timestamps" in sequence and sequence["timestamps"] is not None:
        truncated_sequence["timestamps"] = np.asarray(sequence["timestamps"])[
            :max_points
        ]
    metadata = dict(sequence.get("meta", {}))
    metadata["sequence_length"] = max_points
    absolute_start_index = int(metadata.get("absolute_start_index", 0))
    metadata["absolute_end_index"] = absolute_start_index + max_points
    truncated_sequence["meta"] = metadata
    return truncated_sequence


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(path)


def _to_numpy(array_like: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=dtype)


def _resolve_split_sequences(
    data_bundle: dict[str, Any],
    split_name: str,
) -> list[dict[str, Any]]:
    sequence_groups = data_bundle.get("scaled_sequences") or data_bundle.get(
        "raw_sequences"
    )
    if sequence_groups is None:
        raise ValueError(
            "Dataset bundle does not contain scaled_sequences or raw_sequences"
        )
    split_sequences = sequence_groups.get(split_name)
    if split_sequences is None and split_name == "train":
        split_sequences = sequence_groups.get("val")
    if split_sequences is None:
        raise KeyError(f"Missing split {split_name!r} in dataset bundle")
    if not split_sequences:
        raise ValueError(f"Split {split_name!r} is empty")
    return split_sequences


def _single_sequence(
    split_sequences: list[dict[str, Any]], split_name: str
) -> dict[str, Any]:
    if len(split_sequences) != 1:
        raise ValueError(
            f"Online streaming benchmark expects exactly one sequence in split {split_name!r}"
        )
    return split_sequences[0]


def _instantiate_baseline(
    baseline_name: str,
    baseline_kwargs: dict[str, Any],
) -> OnlineStreamingBaselineProtocol:
    if baseline_name not in BASELINE_BUILDERS:
        raise ValueError(
            f"Unknown baseline_name {baseline_name!r}. Supported baselines: {sorted(BASELINE_BUILDERS)}"
        )
    builder = BASELINE_BUILDERS[baseline_name]
    return builder(**baseline_kwargs)


def _normalize_online_records(
    records: list[dict[str, Any]],
    online_variant: str,
) -> list[dict[str, Any]]:
    normalized_records: list[dict[str, Any]] = []
    for record in records:
        normalized_record = dict(record)
        normalized_record.setdefault("online_variant", online_variant)
        normalized_record.setdefault("did_update", False)
        normalized_records.append(normalized_record)
    return normalized_records


def _build_threshold_artifact_from_calibration(
    *,
    calibration: dict[str, Any],
    method_name: str,
    variant_name: str,
    entity_id: str,
    seed: int,
    window_size: int,
    protocol_config: dict[str, Any],
) -> dict[str, Any]:
    threshold_value = float(calibration["threshold_value"])
    validation_point_scores = np.asarray(
        calibration.get("validation_point_scores", []),
        dtype=np.float64,
    ).reshape(-1)
    validation_ewma_scores = np.asarray(
        calibration.get("validation_ewma_scores", []),
        dtype=np.float64,
    ).reshape(-1)
    if validation_point_scores.size == 0:
        validation_point_scores = np.asarray([threshold_value], dtype=np.float64)
    if validation_ewma_scores.size == 0:
        validation_ewma_scores = np.asarray([threshold_value], dtype=np.float64)
    offline_quantile = float(
        protocol_config.get(
            "offline_threshold_quantile",
            protocol_config["online_threshold_quantile"],
        )
    )
    return build_threshold_artifact(
        method_name=method_name,
        variant_name=variant_name,
        entity_id=entity_id,
        seed=seed,
        window_size=window_size,
        offline_point_threshold=float(
            np.nanquantile(validation_point_scores, offline_quantile)
        ),
        online_ewma_point_threshold=threshold_value,
        quantile=offline_quantile,
        ewma_current_weight=float(protocol_config["online_ewma_current_weight"]),
        ewma_previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        created_by="scripts/run_online_streaming_benchmark.py",
        config_path="benchmark_config",
    )


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "online_streaming_benchmark_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def run_online_streaming_benchmark(
    *,
    benchmark_config_path: str,
    protocol_config_path: str,
    dry_run: bool,
) -> dict[str, Any]:
    benchmark_config = _load_json_config(benchmark_config_path)
    resolved_protocol_config_path = protocol_config_path or benchmark_config.get(
        "protocol_config_path",
        "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    protocol_config = _load_json_config(resolved_protocol_config_path)
    validate_protocol_config(protocol_config)

    output_dir = Path(str(benchmark_config["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = (REPOSITORY_ROOT / output_dir).resolve()

    report: dict[str, Any] = {
        "benchmark_status": "dry_run" if dry_run else "completed",
        "created_at_utc": _utc_now_iso(),
        "benchmark_config_path": benchmark_config_path,
        "protocol_config_path": str(resolved_protocol_config_path),
        "benchmark_config": benchmark_config,
        "protocol": protocol_config,
        "artifact_paths": {},
    }

    if dry_run:
        report["report_path"] = str(_write_report(output_dir, report))
        return report

    register_evaluation_runtime_components()
    data_config = _load_json_config(benchmark_config["data_config_path"])
    data_config = _apply_data_overrides(
        data_config, benchmark_config.get("data_overrides")
    )
    data_bundle = build_dataset(data_config["dataset_name"], data_config)

    train_sequence = _single_sequence(
        _resolve_split_sequences(data_bundle, "train"),
        "train",
    )
    clean_validation_sequences = _resolve_split_sequences(data_bundle, "val")
    test_sequences = _resolve_split_sequences(data_bundle, "test")

    baseline_name = str(benchmark_config["baseline_name"])
    online_variant = str(benchmark_config.get("online_variant", "A0"))
    seed = int(benchmark_config.get("seed", 0))
    baseline_kwargs = dict(benchmark_config.get("baseline_kwargs", {}))
    baseline_kwargs.setdefault(
        "train_sequence", _to_numpy(train_sequence["x"], dtype=np.float64)
    )
    baseline_kwargs.setdefault("window_size", int(protocol_config["window_size"]))
    baseline_kwargs.setdefault(
        "threshold_quantile", float(protocol_config["online_threshold_quantile"])
    )
    baseline_kwargs.setdefault("online_variant", online_variant)
    baseline_kwargs.setdefault("seed", seed)
    baseline = _instantiate_baseline(baseline_name, baseline_kwargs)

    max_online_steps_override = benchmark_config.get("task_overrides", {}).get(
        "max_online_steps"
    )
    max_online_steps = (
        int(max_online_steps_override)
        if max_online_steps_override is not None
        else None
    )

    calibration = baseline.calibrate(
        clean_validation_sequences=clean_validation_sequences,
        protocol_config=protocol_config,
        device=str(benchmark_config.get("device", "cpu")),
    )
    entity_id = str(
        clean_validation_sequences[0]["meta"].get(
            "entity_id",
            train_sequence.get("meta", {}).get("entity_id", "unknown"),
        )
    )
    threshold_artifact = calibration.get("threshold_artifact")
    if threshold_artifact is None:
        threshold_artifact = _build_threshold_artifact_from_calibration(
            calibration=calibration,
            method_name=baseline_name,
            variant_name=online_variant,
            entity_id=entity_id,
            seed=seed,
            window_size=int(protocol_config["window_size"]),
            protocol_config=protocol_config,
        )
    threshold_path = output_dir / "thresholds" / "online_thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)

    metric_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    stream_selections: list[dict[str, Any]] = []
    task_overrides = dict(benchmark_config.get("task_overrides", {}))
    for sequence in test_sequences:
        selected_sequence = select_online_stream_sequence(
            sequence,
            absolute_start_index=task_overrides.get("absolute_start_index"),
            absolute_end_index=task_overrides.get("absolute_end_index"),
        )
        limited_sequence = _truncate_sequence_to_max_online_steps(
            selected_sequence,
            window_size=int(protocol_config["window_size"]),
            max_online_steps=max_online_steps,
        )
        selected_metadata = dict(limited_sequence.get("meta", {}))
        stream_selections.append(
            {
                "entity_id": str(selected_metadata.get("entity_id", entity_id)),
                "source_sequence_length": int(
                    selected_metadata.get(
                        "source_sequence_length", sequence["x"].shape[0]
                    )
                ),
                "absolute_start_index": int(
                    selected_metadata.get("absolute_start_index", 0)
                ),
                "absolute_end_index": int(
                    selected_metadata.get(
                        "absolute_end_index", limited_sequence["x"].shape[0]
                    )
                ),
                "sequence_length": int(
                    selected_metadata.get(
                        "sequence_length", limited_sequence["x"].shape[0]
                    )
                ),
            }
        )
        sequence_metric_history, sequence_records = baseline.run_sequence(
            sequence=limited_sequence,
            threshold_value=float(calibration["threshold_value"]),
            protocol_config=protocol_config,
            device=str(benchmark_config.get("device", "cpu")),
        )
        metric_history.extend(sequence_metric_history)
        records.extend(sequence_records)

    normalized_records = _normalize_online_records(records, online_variant)
    metrics_path = output_dir / "online_metrics.json"
    records_path = output_dir / "online_records.json"
    _write_json(metrics_path, metric_history)
    _write_json(records_path, normalized_records)

    report["entity_id"] = entity_id
    report["seed"] = seed
    report["baseline_name"] = baseline_name
    report["online_variant"] = online_variant
    report["online_execution"] = {
        "benchmark_status": "completed",
        "created_at_utc": _utc_now_iso(),
        "baseline_name": baseline_name,
        "online_variant": online_variant,
        "threshold_artifact": threshold_artifact,
        "threshold_artifact_path": str(threshold_path),
        "threshold_value": float(calibration["threshold_value"]),
        "threshold_source": calibration["threshold_source"],
        "stream_selections": stream_selections,
        "metric_history": metric_history,
        "records": normalized_records,
        "online_metrics_path": str(metrics_path),
        "online_records_path": str(records_path),
    }
    report["artifact_paths"] = {
        "thresholds": str(threshold_path),
        "metrics": str(metrics_path),
        "records": str(records_path),
    }
    report["report_path"] = str(_write_report(output_dir, report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_online_streaming_benchmark(
        benchmark_config_path=args.benchmark_config,
        protocol_config_path=args.protocol_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
