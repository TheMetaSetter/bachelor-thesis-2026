from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_threshold_artifact(
    *,
    method_name: str,
    variant_name: str,
    entity_id: str,
    seed: int,
    window_size: int,
    offline_point_threshold: float,
    online_ewma_point_threshold: float,
    quantile: float,
    ewma_current_weight: float,
    ewma_previous_weight: float,
    created_by: str,
    config_path: str,
    input_window_threshold: float | None = None,
    latent_window_low_threshold: float | None = None,
    latent_window_high_threshold: float | None = None,
) -> dict[str, Any]:
    thresholds = {
            "offline_point": {
                "value": float(offline_point_threshold),
                "source_split": "clean_validation",
                "score_rule": "nonoverlap_tail_average",
                "quantile": float(quantile),
            },
            "online_ewma_point": {
                "value": float(online_ewma_point_threshold),
                "source_split": "clean_validation",
                "score_rule": "stride1_causal_endpoint_ewma",
                "quantile": float(quantile),
                "ewma_current_weight": float(ewma_current_weight),
                "ewma_previous_weight": float(ewma_previous_weight),
            },
    }
    if input_window_threshold is not None:
        thresholds["input_window"] = {
            "value": float(input_window_threshold), "source_split": "clean_validation",
            "score_rule": "window_mean_squared_error", "quantile": 0.99,
        }
    if latent_window_low_threshold is not None or latent_window_high_threshold is not None:
        if latent_window_low_threshold is None or latent_window_high_threshold is None:
            raise ValueError("latent window thresholds must be supplied together")
        if latent_window_low_threshold > latent_window_high_threshold:
            raise ValueError("latent window low threshold must not exceed high threshold")
        thresholds["latent_window_low"] = {
            "value": float(latent_window_low_threshold), "source_split": "clean_validation",
            "score_rule": "latent_memory_distance", "quantile": 0.95,
        }
        thresholds["latent_window_high"] = {
            "value": float(latent_window_high_threshold), "source_split": "clean_validation",
            "score_rule": "latent_memory_distance", "quantile": 0.99,
        }
    return {
        "artifact_version": 2,
        "method_name": method_name,
        "variant_name": variant_name,
        "entity_id": entity_id,
        "seed": int(seed),
        "window_size": int(window_size),
        "thresholds": thresholds,
        "provenance": {
            "test_label_usage": "metrics_only",
            "created_by": created_by,
            "config_path": config_path,
        },
    }


def write_threshold_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_threshold_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
