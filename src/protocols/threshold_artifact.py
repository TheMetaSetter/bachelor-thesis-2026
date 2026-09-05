from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.core.config_aliases import normalize_variance_correction_value


THRESHOLD_ARTIFACT_SCHEMA_VERSION = 5
HISTORICAL_THESIS_SCHEMA_VERSION = 4

_REQUIRED_ONLINE_THRESHOLDS = {
    "online_ewma_point",
    "input_window",
    "latent_window_low",
    "latent_window_high",
}
_OFFLINE_POINT_SOURCE_SPLITS = {
    "clean_validation",
    "synthetic_validation_normal",
}


def validate_threshold_artifact(artifact: dict[str, Any]) -> None:
    required_keys = {
        "schema_version",
        "method_name",
        "variant_name",
        "entity_id",
        "seed",
        "window_size",
        "calibration_split",
        "stochastic_inference",
        "monte_carlo_samples",
        "continuous_temperature",
        "discrete_temperature",
        "score_reduction",
        "variance_correction",
        "numeric_precision",
        "return_mc_samples",
        "sample_retention_policy",
        "offline_point_threshold_nonoverlap",
        "online_point_threshold_ewma",
        "offline_stride",
        "online_stride",
        "ewma_current_weight",
        "ewma_previous_weight",
        "provenance",
        "thresholds",
    }
    missing_keys = sorted(required_keys - set(artifact))
    if missing_keys:
        raise ValueError(f"threshold artifact is missing required keys: {missing_keys}")
    schema_version = int(artifact["schema_version"])
    if schema_version not in {
        3,
        HISTORICAL_THESIS_SCHEMA_VERSION,
        THRESHOLD_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError("threshold artifact schema_version must be one of: 3, 4, 5")
    if schema_version in {
        HISTORICAL_THESIS_SCHEMA_VERSION,
        THRESHOLD_ARTIFACT_SCHEMA_VERSION,
    } and (
        not isinstance(artifact.get("checkpoint_sha256"), str)
        or not artifact["checkpoint_sha256"]
    ):
        raise ValueError(
            "threshold artifact checkpoint_sha256 must be a non-empty string"
        )
    requires_point_score_calibration = (
        schema_version == HISTORICAL_THESIS_SCHEMA_VERSION
        and artifact["method_name"] == "THESIS"
    )
    requires_raw_score_identity = schema_version == THRESHOLD_ARTIFACT_SCHEMA_VERSION
    if requires_point_score_calibration:
        calibration_fields = {
            "point_score_transform",
            "point_score_c",
            "point_score_tau",
            "point_score_tau_estimator",
            "point_score_mad_normalizer",
        }
        missing_calibration_fields = sorted(calibration_fields - set(artifact))
        if missing_calibration_fields:
            raise ValueError(
                "THESIS schema v4 artifact is missing point-score calibration fields: "
                f"{missing_calibration_fields}"
            )
        if artifact["point_score_transform"] != ("shifted-and-scaled logistic sigmoid"):
            raise ValueError("unsupported THESIS point score transform")
        if artifact["point_score_tau_estimator"] != "mad_based_robust_scale":
            raise ValueError("unsupported THESIS point score tau estimator")
        for field_name in ["point_score_c", "point_score_tau"]:
            if not math.isfinite(float(artifact[field_name])):
                raise ValueError(f"threshold artifact {field_name} must be finite")
        if float(artifact["point_score_tau"]) <= 0.0:
            raise ValueError("threshold artifact point_score_tau must be positive")
        if float(artifact["point_score_mad_normalizer"]) != 0.6745:
            raise ValueError(
                "threshold artifact point_score_mad_normalizer must be 0.6745"
            )
    if requires_raw_score_identity:
        raw_identity_fields = {
            "score_space",
            "point_score_transform",
            "point_score_definition",
            "window_score_definition",
        }
        missing_raw_identity_fields = sorted(raw_identity_fields - set(artifact))
        if missing_raw_identity_fields:
            raise ValueError(
                "raw schema v5 artifact is missing score identity fields: "
                f"{missing_raw_identity_fields}"
            )
        if artifact["score_space"] != "raw_input":
            raise ValueError("raw schema v5 artifact score_space must be raw_input")
        if artifact["point_score_transform"] != "identity":
            raise ValueError(
                "raw schema v5 artifact point_score_transform must be identity"
            )
        if artifact["point_score_definition"] != "raw_input_point_mse":
            raise ValueError(
                "raw schema v5 artifact has unsupported point score definition"
            )
        if artifact["window_score_definition"] != "raw_input_window_mse":
            raise ValueError(
                "raw schema v5 artifact has unsupported window score definition"
            )
        calibration_fields = {
            "point_score_c",
            "point_score_tau",
            "point_score_tau_estimator",
            "point_score_mad_normalizer",
        }
        if calibration_fields.intersection(artifact):
            raise ValueError(
                "raw schema v5 artifact must not contain sigmoid calibration fields"
            )
    if not isinstance(artifact["entity_id"], str) or not artifact["entity_id"]:
        raise ValueError("threshold artifact entity_id must be a non-empty string")
    if not isinstance(artifact["method_name"], str) or not artifact["method_name"]:
        raise ValueError("threshold artifact method_name must be a non-empty string")
    if not isinstance(artifact["variant_name"], str) or not artifact["variant_name"]:
        raise ValueError("threshold artifact variant_name must be a non-empty string")
    if not isinstance(artifact["seed"], int) or artifact["seed"] < 0:
        raise ValueError("threshold artifact seed must be a non-negative integer")
    if not isinstance(artifact["window_size"], int) or artifact["window_size"] <= 0:
        raise ValueError("threshold artifact window_size must be a positive integer")
    if artifact["calibration_split"] != "clean_validation":
        raise ValueError(
            "threshold artifact calibration_split must be clean_validation"
        )
    if not isinstance(artifact["stochastic_inference"], bool):
        raise TypeError("threshold artifact stochastic_inference must be boolean")
    if not isinstance(artifact["monte_carlo_samples"], int) or (
        artifact["monte_carlo_samples"] <= 0
    ):
        raise ValueError(
            "threshold artifact monte_carlo_samples must be a positive integer"
        )
    for field_name in [
        "continuous_temperature",
        "discrete_temperature",
        "ewma_current_weight",
        "ewma_previous_weight",
    ]:
        if float(artifact[field_name]) <= 0.0:
            raise ValueError(f"threshold artifact {field_name} must be positive")
    if artifact["score_reduction"] not in {"mean", "median", "sum"}:
        raise ValueError(
            "threshold artifact score_reduction must be one of: mean, median, sum"
        )
    if int(artifact["variance_correction"]) not in {0, 1}:
        raise ValueError("threshold artifact variance_correction must be 0 or 1")
    if artifact["numeric_precision"] not in {"fp16", "fp32", "fp64"}:
        raise ValueError(
            "threshold artifact numeric_precision must be one of: fp16, fp32, fp64"
        )
    if not isinstance(artifact["return_mc_samples"], bool):
        raise TypeError("threshold artifact return_mc_samples must be boolean")
    if artifact["sample_retention_policy"] not in {
        "none",
        "retain_all",
        "retain_for_eda",
    }:
        raise ValueError(
            "threshold artifact sample_retention_policy must be one of: "
            "none, retain_all, retain_for_eda"
        )
    provenance = artifact["provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("threshold artifact provenance must be a mapping")
    for provenance_key in ["created_by", "config_path"]:
        provenance_value = provenance.get(provenance_key)
        if not isinstance(provenance_value, str) or not provenance_value:
            raise ValueError(
                f"threshold artifact provenance.{provenance_key} must be a non-empty string"
            )
    if provenance.get("calibration_split") != artifact["calibration_split"]:
        raise ValueError(
            "threshold artifact provenance.calibration_split must match calibration_split"
        )
    if provenance.get("checkpoint_sha256") != artifact.get("checkpoint_sha256"):
        raise ValueError(
            "threshold artifact provenance.checkpoint_sha256 must match checkpoint_sha256"
        )
    if provenance.get("resolved_config_sha256") != artifact.get(
        "resolved_config_sha256"
    ):
        raise ValueError(
            "threshold artifact provenance.resolved_config_sha256 must match resolved_config_sha256"
        )
    if requires_point_score_calibration:
        for field_name in [
            "point_score_transform",
            "point_score_c",
            "point_score_tau",
            "point_score_tau_estimator",
            "point_score_mad_normalizer",
        ]:
            if provenance.get(field_name) != artifact.get(field_name):
                raise ValueError(
                    "threshold artifact provenance calibration field must match "
                    f"{field_name}"
                )
    if requires_raw_score_identity:
        for field_name in [
            "score_space",
            "point_score_transform",
            "point_score_definition",
            "window_score_definition",
        ]:
            if provenance.get(field_name) != artifact.get(field_name):
                raise ValueError(
                    "threshold artifact provenance raw identity field must match "
                    f"{field_name}"
                )
    if "checkpoint_sha256" in artifact and artifact["checkpoint_sha256"] is not None:
        if (
            not isinstance(artifact["checkpoint_sha256"], str)
            or not artifact["checkpoint_sha256"]
        ):
            raise ValueError(
                "threshold artifact checkpoint_sha256 must be a non-empty string"
            )
    if (
        "resolved_config_sha256" in artifact
        and artifact["resolved_config_sha256"] is not None
    ):
        if (
            not isinstance(artifact["resolved_config_sha256"], str)
            or not artifact["resolved_config_sha256"]
        ):
            raise ValueError(
                "threshold artifact resolved_config_sha256 must be a non-empty string"
            )
    if (
        not isinstance(artifact["offline_stride"], int)
        or artifact["offline_stride"] <= 0
    ):
        raise ValueError("threshold artifact offline_stride must be a positive integer")
    if not isinstance(artifact["online_stride"], int) or artifact["online_stride"] <= 0:
        raise ValueError("threshold artifact online_stride must be a positive integer")
    if artifact["online_stride"] != 1:
        raise ValueError("threshold artifact online_stride must be 1")
    if artifact["offline_stride"] != artifact["window_size"]:
        raise ValueError(
            "threshold artifact offline_stride must match window_size for non-overlap calibration"
        )
    thresholds = artifact["thresholds"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise TypeError("threshold artifact thresholds must be a non-empty mapping")
    missing_thresholds = sorted(_REQUIRED_ONLINE_THRESHOLDS - set(thresholds))
    if (
        schema_version
        in {HISTORICAL_THESIS_SCHEMA_VERSION, THRESHOLD_ARTIFACT_SCHEMA_VERSION}
        and missing_thresholds
    ):
        raise ValueError(
            "threshold artifact is missing required online thresholds: "
            f"{missing_thresholds}"
        )
    for threshold_name, threshold_record in thresholds.items():
        if not isinstance(threshold_record, dict):
            raise TypeError(
                f"threshold artifact threshold {threshold_name} must be a mapping"
            )
        for threshold_key in ["value", "source_split", "score_rule", "quantile"]:
            if threshold_key not in threshold_record:
                raise ValueError(
                    f"threshold artifact threshold {threshold_name} is missing {threshold_key}"
                )
            if threshold_key == "quantile" and not (
                0.0 < float(threshold_record[threshold_key]) <= 1.0
            ):
                raise ValueError(
                    f"threshold artifact threshold {threshold_name} quantile must be in (0, 1]"
                )
        if not isinstance(threshold_record["value"], (int, float)):
            raise TypeError(
                f"threshold artifact threshold {threshold_name} value must be numeric"
            )
        if (
            not isinstance(threshold_record["score_rule"], str)
            or not threshold_record["score_rule"]
        ):
            raise ValueError(
                f"threshold artifact threshold {threshold_name} score_rule must be a non-empty string"
            )
        if (
            not isinstance(threshold_record["source_split"], str)
            or not threshold_record["source_split"]
        ):
            raise ValueError(
                f"threshold artifact threshold {threshold_name} source_split must be a non-empty string"
            )
        if (
            threshold_name == "offline_point"
            and threshold_record["source_split"] not in _OFFLINE_POINT_SOURCE_SPLITS
        ):
            raise ValueError(
                "threshold artifact offline_point source_split must be one of "
                f"{sorted(_OFFLINE_POINT_SOURCE_SPLITS)!r}"
            )
        if (
            "ewma_current_weight" in threshold_record
            or "ewma_previous_weight" in threshold_record
        ):
            if threshold_name != "online_ewma_point":
                raise ValueError(
                    "EWMA weights may only be attached to online_ewma_point threshold"
                )
            if (
                float(threshold_record["ewma_current_weight"]) <= 0.0
                or float(threshold_record["ewma_previous_weight"]) <= 0.0
            ):
                raise ValueError("EWMA threshold weights must be positive")
    if (
        schema_version
        in {HISTORICAL_THESIS_SCHEMA_VERSION, THRESHOLD_ARTIFACT_SCHEMA_VERSION}
        and thresholds["online_ewma_point"]["score_rule"]
        != "stride1_causal_window_vector_ewma"
    ):
        raise ValueError("online_ewma_point must use stride1_causal_window_vector_ewma")
    if schema_version in {
        HISTORICAL_THESIS_SCHEMA_VERSION,
        THRESHOLD_ARTIFACT_SCHEMA_VERSION,
    } and float(thresholds["latent_window_low"]["value"]) > float(
        thresholds["latent_window_high"]["value"]
    ):
        raise ValueError("latent window low threshold must not exceed high threshold")


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
    calibration_split: str = "clean_validation",
    offline_point_threshold_source_split: str | None = None,
    stochastic_inference: bool = True,
    monte_carlo_samples: int = 10,
    continuous_temperature: float = 0.9,
    discrete_temperature: float = 0.9,
    score_reduction: str = "mean",
    variance_correction: int | str = 1,
    numeric_precision: str = "fp32",
    return_mc_samples: bool = False,
    sample_retention_policy: str = "none",
    offline_stride: int = 20,
    online_stride: int = 1,
    checkpoint_sha256: str | None = None,
    resolved_config_sha256: str | None = None,
    input_window_threshold: float | None = None,
    latent_window_low_threshold: float | None = None,
    latent_window_high_threshold: float | None = None,
    latent_window_low_quantile: float = 0.75,
    point_score_c: float | None = None,
    point_score_tau: float | None = None,
    point_score_transform: str = "shifted-and-scaled logistic sigmoid",
    point_score_tau_estimator: str = "mad_based_robust_scale",
    point_score_mad_normalizer: float = 0.6745,
    score_space: str = "model_output",
) -> dict[str, Any]:
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    is_raw_input_protocol = score_space == "raw_input"
    if score_space not in {"model_output", "raw_input"}:
        raise ValueError("score_space must be model_output or raw_input")
    offline_point_source = (
        calibration_split
        if offline_point_threshold_source_split is None
        else str(offline_point_threshold_source_split)
    )
    if offline_point_source not in _OFFLINE_POINT_SOURCE_SPLITS:
        raise ValueError(
            "offline_point_threshold_source_split must be one of "
            f"{sorted(_OFFLINE_POINT_SOURCE_SPLITS)!r}"
        )
    is_historical_thesis_v4 = method_name == "THESIS" and not is_raw_input_protocol
    if is_historical_thesis_v4 and (
        not isinstance(checkpoint_sha256, str) or not checkpoint_sha256
    ):
        raise ValueError("checkpoint_sha256 must be a non-empty string")
    if method_name == "THESIS" and (
        input_window_threshold is None
        or latent_window_low_threshold is None
        or latent_window_high_threshold is None
    ):
        raise ValueError("THESIS schema versions 4 and 5 require all triage thresholds")
    if is_historical_thesis_v4 and (point_score_c is None or point_score_tau is None):
        raise ValueError("THESIS schema version 4 requires point score calibration")
    if is_historical_thesis_v4:
        if not math.isfinite(float(point_score_c)):
            raise ValueError("point_score_c must be finite")
        if not math.isfinite(float(point_score_tau)) or float(point_score_tau) <= 0.0:
            raise ValueError("point_score_tau must be finite and positive")
    if (
        latent_window_low_threshold is not None
        and latent_window_high_threshold is not None
        and latent_window_low_threshold > latent_window_high_threshold
    ):
        raise ValueError("latent window low threshold must not exceed high threshold")
    variance_correction_value = normalize_variance_correction_value(variance_correction)
    thresholds = {
        "offline_point": {
            "value": float(offline_point_threshold),
            "source_split": offline_point_source,
            "score_rule": "nonoverlap_tail_average",
            "quantile": float(quantile),
        },
        "online_ewma_point": {
            "value": float(online_ewma_point_threshold),
            "source_split": calibration_split,
            "score_rule": (
                "stride1_causal_window_vector_ewma"
                if is_historical_thesis_v4 or is_raw_input_protocol
                else "stride1_causal_endpoint_ewma"
            ),
            "quantile": float(quantile),
            "ewma_current_weight": float(ewma_current_weight),
            "ewma_previous_weight": float(ewma_previous_weight),
        },
    }
    if input_window_threshold is not None:
        thresholds["input_window"] = {
            "value": float(input_window_threshold),
            "source_split": calibration_split,
            "score_rule": (
                "raw_input_window_mse"
                if is_raw_input_protocol
                else "window_mean_squared_error"
            ),
            "quantile": 0.99,
        }
    if (
        latent_window_low_threshold is not None
        and latent_window_high_threshold is not None
    ):
        thresholds["latent_window_low"] = {
            "value": float(latent_window_low_threshold),
            "source_split": calibration_split,
            "score_rule": "latent_memory_distance",
            "quantile": float(latent_window_low_quantile),
        }
        thresholds["latent_window_high"] = {
            "value": float(latent_window_high_threshold),
            "source_split": calibration_split,
            "score_rule": "latent_memory_distance",
            "quantile": 0.99,
        }
    artifact = {
        "artifact_version": (
            THRESHOLD_ARTIFACT_SCHEMA_VERSION
            if is_raw_input_protocol
            else HISTORICAL_THESIS_SCHEMA_VERSION
            if is_historical_thesis_v4
            else 3
        ),
        "schema_version": (
            THRESHOLD_ARTIFACT_SCHEMA_VERSION
            if is_raw_input_protocol
            else HISTORICAL_THESIS_SCHEMA_VERSION
            if is_historical_thesis_v4
            else 3
        ),
        "method_name": method_name,
        "variant_name": variant_name,
        "entity_id": entity_id,
        "seed": int(seed),
        "window_size": int(window_size),
        "calibration_split": calibration_split,
        "stochastic_inference": bool(stochastic_inference),
        "monte_carlo_samples": int(monte_carlo_samples),
        "continuous_temperature": float(continuous_temperature),
        "discrete_temperature": float(discrete_temperature),
        "score_reduction": score_reduction,
        "variance_correction": variance_correction_value,
        "numeric_precision": numeric_precision,
        "return_mc_samples": bool(return_mc_samples),
        "sample_retention_policy": sample_retention_policy,
        "offline_point_threshold_nonoverlap": float(offline_point_threshold),
        "online_point_threshold_ewma": float(online_ewma_point_threshold),
        "ewma_current_weight": float(ewma_current_weight),
        "ewma_previous_weight": float(ewma_previous_weight),
        "offline_stride": int(offline_stride),
        "online_stride": int(online_stride),
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "thresholds": thresholds,
        "provenance": {
            "created_by": created_by,
            "config_path": config_path,
            "calibration_split": calibration_split,
            "threshold_method": method_name,
            "threshold_variant": variant_name,
            "test_label_usage": "metrics_only",
            "score_reduction": score_reduction,
            "variance_correction": variance_correction_value,
            "numeric_precision": numeric_precision,
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
        },
    }
    if is_raw_input_protocol:
        artifact.update(
            {
                "score_space": "raw_input",
                "point_score_transform": "identity",
                "point_score_definition": "raw_input_point_mse",
                "window_score_definition": "raw_input_window_mse",
            }
        )
        artifact["provenance"].update(
            {
                "score_space": "raw_input",
                "point_score_transform": "identity",
                "point_score_definition": "raw_input_point_mse",
                "window_score_definition": "raw_input_window_mse",
            }
        )
    elif is_historical_thesis_v4:
        artifact.update(
            {
                "point_score_transform": point_score_transform,
                "point_score_c": float(point_score_c),
                "point_score_tau": float(point_score_tau),
                "point_score_tau_estimator": point_score_tau_estimator,
                "point_score_mad_normalizer": float(point_score_mad_normalizer),
            }
        )
        artifact["provenance"].update(
            {
                "point_score_transform": point_score_transform,
                "point_score_c": float(point_score_c),
                "point_score_tau": float(point_score_tau),
                "point_score_tau_estimator": point_score_tau_estimator,
                "point_score_mad_normalizer": float(point_score_mad_normalizer),
            }
        )
    return artifact


def write_threshold_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    validate_threshold_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_threshold_artifact(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_threshold_artifact(artifact)
    return artifact
