from __future__ import annotations

"""Offline benchmark launcher for the fair traditional baselines.

₍^. .^₎⟆ Launcher flow

benchmark config + protocol config
  -> load scaled SMD splits
  -> fit one baseline on train
  -> calibrate on clean validation
  -> write shared score and threshold artifacts
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.baselines.traditional import (
    IForestWindowBaseline,
    KMeansADWindowBaseline,
    StumpyChannelABFrozenTrainRef,
    TraditionalBaselineProtocol,
)
from src.core.config import load_yaml_config
from src.core.registry import build_dataset
from src.metrics.pointwise import compute_pointwise_metrics
from src.protocols.point_scores import ewma_scores
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASELINE_BUILDERS: dict[str, Callable[..., TraditionalBaselineProtocol]] = {
    "stumpy_channel_ab": StumpyChannelABFrozenTrainRef,
    "kmeans_ad": KMeansADWindowBaseline,
    "iforest": IForestWindowBaseline,
}


def register_evaluation_runtime_components() -> None:
    from src.core.runtime_components import (
        register_evaluation_runtime_components as _register_evaluation_runtime_components,
    )

    return _register_evaluation_runtime_components()


def validate_protocol_config(protocol_config: dict[str, Any]) -> None:
    from src.protocols.smd_benchmark_protocol import (
        validate_protocol_config as _validate_protocol_config,
    )

    return _validate_protocol_config(protocol_config, require_score_identity=False)


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


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return str(path)


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return str(path)


def _to_numpy(array_like: Any, *, dtype: Any) -> np.ndarray:
    if hasattr(array_like, "detach"):
        array_like = array_like.detach().cpu().numpy()
    return np.asarray(array_like, dtype=dtype)


def _single_sequence(
    split_sequences: list[dict[str, Any]], split_name: str
) -> dict[str, Any]:
    if len(split_sequences) != 1:
        raise ValueError(
            f"Offline benchmark expects exactly one sequence in split {split_name!r}"
        )
    return split_sequences[0]


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
    if split_sequences is None:
        raise KeyError(f"Missing split {split_name!r} in dataset bundle")
    if not split_sequences:
        raise ValueError(f"Split {split_name!r} is empty")
    return split_sequences


def _build_split_payload(
    *,
    baseline: TraditionalBaselineProtocol,
    split_sequence: dict[str, Any],
) -> dict[str, np.ndarray]:
    x = _to_numpy(split_sequence["x"], dtype=np.float64)
    point_labels = _to_numpy(split_sequence["point_labels"], dtype=np.int64).reshape(-1)
    point_scores = np.asarray(baseline.score_sequence(x), dtype=np.float64).reshape(-1)
    if point_scores.shape != point_labels.shape:
        raise ValueError(
            "Baseline score_sequence output must match point_labels length"
        )
    covered_point_mask = np.isfinite(point_scores)
    return {
        "point_scores": point_scores,
        "point_labels": point_labels,
        "covered_point_mask": covered_point_mask,
    }


def _score_validation_split(
    *,
    baseline: TraditionalBaselineProtocol,
    data_bundle: dict[str, Any],
    split_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    split_sequence = _single_sequence(
        _resolve_split_sequences(data_bundle, split_name), split_name
    )
    payload = _build_split_payload(baseline=baseline, split_sequence=split_sequence)
    return payload, dict(split_sequence.get("meta", {}))


def _build_metrics(
    *,
    point_labels: np.ndarray,
    point_scores: np.ndarray,
    threshold: float,
    protocol_config: dict[str, Any],
) -> dict[str, Any]:
    covered_mask = np.isfinite(point_scores)
    return compute_pointwise_metrics(
        point_labels=point_labels[covered_mask],
        point_scores=point_scores[covered_mask],
        threshold=threshold,
        vus_max_buffer_size=int(protocol_config["window_size"]),
        vus_num_thresholds=200,
    )


def _ewma_threshold(
    point_scores: np.ndarray,
    protocol_config: dict[str, Any],
) -> float:
    smoothed_scores = ewma_scores(
        point_scores=np.asarray(point_scores, dtype=np.float64).reshape(-1),
        current_weight=float(protocol_config["online_ewma_current_weight"]),
        previous_weight=float(protocol_config["online_ewma_previous_weight"]),
    )
    covered_mask = np.isfinite(smoothed_scores)
    if not np.any(covered_mask):
        raise ValueError(
            "EWMA validation scores must contain at least one finite value"
        )
    return float(
        np.nanquantile(
            smoothed_scores[covered_mask],
            float(protocol_config["online_threshold_quantile"]),
        )
    )


def _instantiate_baseline(
    baseline_name: str,
    baseline_kwargs: dict[str, Any],
) -> TraditionalBaselineProtocol:
    if baseline_name not in BASELINE_BUILDERS:
        raise ValueError(
            f"Unknown baseline_name {baseline_name!r}. "
            f"Supported baselines: {sorted(BASELINE_BUILDERS)}"
        )
    builder = BASELINE_BUILDERS[baseline_name]
    return builder(**baseline_kwargs)


def run_offline_benchmark(
    *,
    benchmark_config_path: str,
    protocol_config_path: str | None = None,
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

    report = {
        "benchmark_status": "dry_run" if dry_run else "completed",
        "created_at_utc": _utc_now_iso(),
        "benchmark_config_path": benchmark_config_path,
        "protocol_config_path": str(resolved_protocol_config_path),
        "benchmark_config": benchmark_config,
        "protocol": protocol_config,
        "artifact_paths": {},
    }
    report_path = output_dir / "benchmark" / "offline_benchmark_report.json"

    if dry_run:
        report["report_path"] = str(_write_json(report_path, report))
        return report

    register_evaluation_runtime_components()
    data_config = _load_json_config(benchmark_config["data_config_path"])
    data_config = _apply_data_overrides(
        data_config, benchmark_config.get("data_overrides")
    )
    data_bundle = build_dataset(data_config["dataset_name"], data_config)

    baseline_name = str(benchmark_config["baseline_name"])
    baseline_kwargs = dict(benchmark_config.get("baseline_kwargs", {}))
    baseline_kwargs.setdefault("window_size", int(protocol_config["window_size"]))
    baseline_kwargs.setdefault(
        "threshold_quantile", float(protocol_config["offline_threshold_quantile"])
    )
    baseline = _instantiate_baseline(baseline_name, baseline_kwargs)

    train_sequence = _single_sequence(
        _resolve_split_sequences(data_bundle, "train"),
        "train",
    )
    baseline.fit(_to_numpy(train_sequence["x"], dtype=np.float64))

    clean_validation_sequence = _single_sequence(
        _resolve_split_sequences(data_bundle, "val"),
        "val",
    )
    calibration = baseline.calibrate(
        _to_numpy(clean_validation_sequence["x"], dtype=np.float64)
    )
    # Re-score clean validation after calibration so the score artifact and
    # threshold artifact share the exact same contract.
    clean_validation_payload, clean_validation_meta = _score_validation_split(
        baseline=baseline,
        data_bundle=data_bundle,
        split_name="val",
    )
    synthetic_validation_split = "val_synth"
    if synthetic_validation_split not in (data_bundle.get("scaled_sequences") or {}):
        synthetic_validation_split = "val"
    synthetic_validation_payload, synthetic_validation_meta = _score_validation_split(
        baseline=baseline,
        data_bundle=data_bundle,
        split_name=synthetic_validation_split,
    )
    test_payload, test_meta = _score_validation_split(
        baseline=baseline,
        data_bundle=data_bundle,
        split_name="test",
    )

    entity_id = str(
        clean_validation_meta.get(
            "entity_id",
            train_sequence.get("meta", {}).get("entity_id", "unknown_entity"),
        )
    )
    seed = int(benchmark_config.get("seed", 0))
    window_size = int(protocol_config["window_size"])
    offline_point_threshold = float(calibration["threshold"])
    online_ewma_point_threshold = _ewma_threshold(
        clean_validation_payload["point_scores"],
        protocol_config,
    )

    threshold_artifact = build_threshold_artifact(
        method_name=baseline_name,
        variant_name=baseline_name,
        entity_id=entity_id,
        seed=seed,
        window_size=window_size,
        offline_point_threshold=offline_point_threshold,
        online_ewma_point_threshold=online_ewma_point_threshold,
        quantile=float(protocol_config["offline_threshold_quantile"]),
        ewma_current_weight=float(protocol_config["online_ewma_current_weight"]),
        ewma_previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        created_by="scripts/run_offline_benchmark.py",
        config_path=str(benchmark_config_path),
    )
    threshold_path = output_dir / "thresholds" / "thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)

    clean_scores_path = output_dir / "scores" / "clean_validation_point_scores.npz"
    synthetic_scores_path = (
        output_dir / "scores" / "synthetic_validation_point_scores.npz"
    )
    test_scores_path = output_dir / "scores" / "test_point_scores.npz"
    _write_npz(clean_scores_path, clean_validation_payload)
    _write_npz(synthetic_scores_path, synthetic_validation_payload)
    _write_npz(test_scores_path, test_payload)

    test_metrics = _build_metrics(
        point_labels=test_payload["point_labels"],
        point_scores=test_payload["point_scores"],
        threshold=offline_point_threshold,
        protocol_config=protocol_config,
    )
    metrics_path = output_dir / "metrics" / "offline_metrics.json"
    _write_json(metrics_path, test_metrics)

    report["entity_id"] = entity_id
    report["seed"] = seed
    report["baseline_name"] = baseline_name
    report["method_metadata"] = calibration["method_metadata"]
    report["thresholds"] = threshold_artifact["thresholds"]
    report["artifact_paths"] = {
        "thresholds": str(threshold_path),
        "clean_validation_scores": str(clean_scores_path),
        "synthetic_validation_scores": str(synthetic_scores_path),
        "test_scores": str(test_scores_path),
        "metrics": str(metrics_path),
        "report": str(report_path),
    }
    report["offline_metrics"] = test_metrics
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_offline_benchmark(
        benchmark_config_path=args.benchmark_config,
        protocol_config_path=args.protocol_config,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
