from __future__ import annotations

"""Compact UQ summary artifact helpers.

The benchmark wrappers export this file alongside `evaluation_metrics.json`
so the project can keep summary statistics after raw trace payloads are
deleted for disk-space control.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json
import numpy as np


_REQUIRED_SPLIT_NAMES = ("clean_validation", "synthetic_validation", "test")
_SUMMARY_KEYS = ("mean", "std", "min", "p50", "p95", "max")
_UNCERTAINTY_KEYS = (
    "point_anomaly_score_variance",
    "window_anomaly_score_variance",
    "continuous_retrieval_variance_point",
    "continuous_retrieval_variance_window",
    "discrete_retrieval_variance_point",
    "discrete_retrieval_variance_window",
    "reconstruction_variance_point",
    "reconstruction_variance_window",
    "reconstruction_variance_full",
    "classification_probability_variance",
    "classification_variance_mean",
)
_MC_SAMPLE_HISTORY_KEYS = (
    "point_score_samples",
    "window_score_samples",
    "reconstruction_samples",
    "classification_probability_samples",
)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _flatten_numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: list[float] = []
        for nested_value in value.values():
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened = []
        for nested_value in value:
            flattened.extend(_flatten_numeric_values(nested_value))
        return flattened
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    return [float(item) for item in array[np.isfinite(array)]]


def _summary_from_value(value: Any) -> dict[str, float | None]:
    finite_values = np.asarray(_flatten_numeric_values(value), dtype=float)
    if finite_values.size == 0:
        return {key: None for key in _SUMMARY_KEYS}
    return {
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(ddof=0)),
        "min": float(finite_values.min()),
        "p50": float(np.quantile(finite_values, 0.5)),
        "p95": float(np.quantile(finite_values, 0.95)),
        "max": float(finite_values.max()),
    }


def _first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _split_trace_audit(traces: list[dict[str, Any]]) -> dict[str, Any]:
    mc_histories_non_null_count = dict.fromkeys(_MC_SAMPLE_HISTORY_KEYS, 0)
    uncertainty_history_non_null_count = 0
    for trace in traces:
        uncertainty_history = trace.get("uncertainty_history")
        if uncertainty_history is not None:
            uncertainty_history_non_null_count += 1
        mc_sample_histories = trace.get("mc_sample_histories") or {}
        for key_name in mc_histories_non_null_count:
            if mc_sample_histories.get(key_name) is not None:
                mc_histories_non_null_count[key_name] += 1
    return {
        "any_uncertainty_history": uncertainty_history_non_null_count > 0,
        "uncertainty_history_non_null_count": uncertainty_history_non_null_count,
        "any_mc_sample_history": any(
            count > 0 for count in mc_histories_non_null_count.values()
        ),
        "mc_histories_non_null_count": mc_histories_non_null_count,
    }


def _collect_trace_values(
    traces: list[dict[str, Any]],
    key_name: str,
    nested_key_name: str | None = None,
) -> list[Any]:
    collected_values: list[Any] = []
    for trace in traces:
        value = trace.get(key_name)
        if nested_key_name is not None and isinstance(value, dict):
            value = value.get(nested_key_name)
        if value is not None:
            collected_values.append(value)
    return collected_values


def _split_uq_summary(
    *,
    point_scores: Any,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_audit = _split_trace_audit(traces)
    window_score_values = _collect_trace_values(traces, "window_score_history")
    uncertainty_histories = _collect_trace_values(traces, "uncertainty_history")
    uncertainty_summary: dict[str, Any] = {}
    for field_name in _UNCERTAINTY_KEYS:
        field_values: list[Any] = []
        for uncertainty_history in uncertainty_histories:
            if isinstance(uncertainty_history, dict):
                field_values.append(uncertainty_history.get(field_name))
        if field_name == "point_anomaly_score_variance":
            field_summary = _summary_from_value(field_values)
            uncertainty_summary["point_anomaly_score_variance_mean"] = field_summary[
                "mean"
            ]
            uncertainty_summary["point_anomaly_score_variance_p95"] = field_summary[
                "p95"
            ]
            continue
        if field_name == "classification_probability_variance":
            uncertainty_summary["classification_probability_variance_mean"] = (
                _summary_from_value(field_values)["mean"]
            )
            continue
        if field_name == "classification_variance_mean":
            uncertainty_summary["classification_variance_mean"] = _summary_from_value(
                field_values
            )["mean"]
            continue
        uncertainty_summary[f"{field_name}_mean"] = _summary_from_value(
            field_values
        )["mean"]
    return {
        "num_traces": len(traces),
        "sample_retention_policy": _first_non_null(
            [trace.get("sample_retention_policy") for trace in traces]
        ),
        "trace_audit": trace_audit,
        "point_score_summary": _summary_from_value(point_scores),
        "window_score_summary": _summary_from_value(window_score_values),
        "uncertainty_summary": uncertainty_summary,
    }


def build_uq_summary_payload(
    *,
    benchmark_kind: str,
    experiment_name: str,
    method_name: str,
    variant_name: str,
    entity_id: str,
    seed: int,
    stage_name: str,
    checkpoint_path: str,
    checkpoint_sha256: str | None,
    experiment_config_path: str,
    protocol_config_path: str,
    output_dir: str,
    run_scalar_logs: dict[str, Any],
    split_inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "created_at_utc": _utc_now_iso(),
        "run": {
            "benchmark_kind": benchmark_kind,
            "experiment_name": experiment_name,
            "method_name": method_name,
            "variant_name": variant_name,
            "entity_id": entity_id,
            "seed": int(seed),
            "stage_name": stage_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "experiment_config_path": experiment_config_path,
            "protocol_config_path": protocol_config_path,
            "output_dir": output_dir,
        },
        "run_scalar_logs": dict(run_scalar_logs),
        "splits": {},
    }
    for split_name, split_input in split_inputs.items():
        payload["splits"][split_name] = _split_uq_summary(
            point_scores=split_input.get("point_scores"),
            traces=list(split_input.get("traces") or []),
        )
    validate_uq_summary_payload(payload)
    return payload


def validate_uq_summary_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("uq_summary payload must be a mapping")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("uq_summary schema_version must be 1")
    if not isinstance(payload.get("created_at_utc"), str):
        raise TypeError("uq_summary.created_at_utc must be a string")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise TypeError("uq_summary.run must be a mapping")
    required_run_keys = (
        "benchmark_kind",
        "experiment_name",
        "method_name",
        "variant_name",
        "entity_id",
        "seed",
        "stage_name",
        "checkpoint_path",
        "experiment_config_path",
        "protocol_config_path",
        "output_dir",
    )
    for key_name in required_run_keys:
        if key_name not in run:
            raise ValueError(f"uq_summary.run missing required field: {key_name}")
    if not isinstance(payload.get("run_scalar_logs"), dict):
        raise TypeError("uq_summary.run_scalar_logs must be a mapping")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise TypeError("uq_summary.splits must be a mapping")
    for split_name in _REQUIRED_SPLIT_NAMES:
        if split_name not in splits:
            raise ValueError(f"uq_summary.splits missing required split: {split_name}")
    raw_keys = {
        "stochastic_query",
        "uncertainty_history",
        "mc_sample_histories",
        "deterministic_geometry",
    }
    for split_name, split in splits.items():
        if not isinstance(split, dict):
            raise TypeError(f"uq_summary.splits['{split_name}'] must be a mapping")
        if int(split.get("num_traces", 0)) < 0:
            raise ValueError(f"uq_summary.splits['{split_name}'].num_traces must be >= 0")
        if not isinstance(split.get("trace_audit"), dict):
            raise TypeError(
                f"uq_summary.splits['{split_name}'].trace_audit must be a mapping"
            )
        for raw_key in raw_keys:
            if raw_key in split:
                raise ValueError(
                    f"uq_summary.splits['{split_name}'] must not contain raw key {raw_key}"
                )
        for summary_name in ("point_score_summary", "window_score_summary", "uncertainty_summary"):
            summary = split.get(summary_name)
            if not isinstance(summary, dict):
                raise TypeError(
                    f"uq_summary.splits['{split_name}'].{summary_name} must be a mapping"
                )
            for value_name, value in summary.items():
                if value is not None and not isinstance(value, (int, float)):
                    raise TypeError(
                        f"uq_summary.splits['{split_name}'].{summary_name}.{value_name} must be numeric or null"
                    )
        trace_audit = split["trace_audit"]
        if not isinstance(trace_audit.get("any_uncertainty_history"), bool):
            raise TypeError(
                f"uq_summary.splits['{split_name}'].trace_audit.any_uncertainty_history must be a boolean"
            )
        if not isinstance(trace_audit.get("uncertainty_history_non_null_count"), int):
            raise TypeError(
                f"uq_summary.splits['{split_name}'].trace_audit.uncertainty_history_non_null_count must be an integer"
            )
        if not isinstance(trace_audit.get("any_mc_sample_history"), bool):
            raise TypeError(
                f"uq_summary.splits['{split_name}'].trace_audit.any_mc_sample_history must be a boolean"
            )
        mc_histories_non_null_count = trace_audit.get("mc_histories_non_null_count")
        if not isinstance(mc_histories_non_null_count, dict):
            raise TypeError(
                f"uq_summary.splits['{split_name}'].trace_audit.mc_histories_non_null_count must be a mapping"
            )
        for key_name in _MC_SAMPLE_HISTORY_KEYS:
            if key_name not in mc_histories_non_null_count:
                raise ValueError(
                    f"uq_summary.splits['{split_name}'].trace_audit.mc_histories_non_null_count missing {key_name}"
                )
            if not isinstance(mc_histories_non_null_count[key_name], int):
                raise TypeError(
                    f"uq_summary.splits['{split_name}'].trace_audit.mc_histories_non_null_count['{key_name}'] must be an integer"
                )


def write_uq_summary_json(path: str | Path, payload: dict[str, Any]) -> Path:
    validate_uq_summary_payload(payload)
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path
