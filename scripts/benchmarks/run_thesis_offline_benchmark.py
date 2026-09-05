from __future__ import annotations

"""THESIS offline benchmark wrapper.

₍₍⚞(˶˃ ꒳ ˂˶)⚟⁾⁾ How this wrapper fits

experiment config + protocol config
  -> validate locked fairness rules
  -> materialize existing two-stage plan
  -> execute existing two-stage runner
  -> write one benchmark report

This file does not train a model by itself. It delegates training to
`scripts/run_two_stage_offline_pretraining.py` so there is only one owner for
Stage A and Stage B behavior.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.core.config import load_experiment_config
from src.core.artifact_integrity import (
    build_retention_bundle_manifest,
    sha256_file,
    write_retention_bundle_manifest,
)
from src.core.evaluation_trace_compaction import compact_evaluation_trace_payloads
from src.core.uq_summary import (
    build_uq_summary_payload,
    write_uq_summary_json,
)
from src.core.registry import build_dataset
from src.core.runtime_components import register_online_runtime_components
from src.data.loaders import rebuild_dataset_bundle_with_scaler_state
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import Evaluator
from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
    select_synthetic_validation_normal_point_threshold,
)
from src.engine.online_tta.online_calibration import collect_stride1_online_scores
from src.engine.online_tta.online_engine_shared import (
    _build_model_from_experiment_config,
)
from scripts.ops.threshold_artifact_v4_online_scoring import (
    StageBInventoryEntry,
    load_a0_scoring_config,
)
from src.protocols.point_scores import ewma_scores
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)
from src.protocols.point_score_calibration import fit_mad_logistic_calibration


def build_model_from_experiment_config(experiment_config: dict[str, Any]) -> Any:
    from scripts.cli.train import (
        build_model_from_experiment_config as _build_model_from_experiment_config,
    )

    return _build_model_from_experiment_config(experiment_config)


def materialize_two_stage_run_manifest(
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    from scripts.experiments.run_two_stage_offline_pretraining import (
        materialize_two_stage_run_manifest as _materialize_two_stage_run_manifest,
    )

    return _materialize_two_stage_run_manifest(experiment_config)


def execute_two_stage_plan(
    manifest: dict[str, Any],
    dry_run: bool,
    skip_completed: bool,
) -> dict[str, Any]:
    from scripts.experiments.run_two_stage_offline_pretraining import (
        execute_two_stage_plan as _execute_two_stage_plan,
    )

    return _execute_two_stage_plan(
        manifest,
        dry_run=dry_run,
        skip_completed=skip_completed,
    )


def validate_two_stage_epoch_budget(experiment_config: dict[str, Any]) -> None:
    from scripts.experiments.run_two_stage_offline_pretraining import (
        validate_two_stage_epoch_budget as _validate_two_stage_epoch_budget,
    )

    return _validate_two_stage_epoch_budget(experiment_config)


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


def _load_yaml_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    report_dir = output_dir / "benchmark"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "thesis_offline_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return report_path


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return str(path)


def _write_score_npz(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "point_scores": np.asarray(payload["point_scores"], dtype=float),
        "point_labels": np.asarray(payload["point_labels"], dtype=np.int64),
        "covered_point_mask": np.asarray(payload["covered_point_mask"], dtype=bool),
    }
    optional_fields = {
        "raw_input_point_mse": float,
        "normalized_input_point_mse": float,
        "point_predictions": np.int64,
        "raw_input_window_mse": float,
        "normalized_input_window_mse": float,
        "window_labels": np.int64,
        "window_predictions": np.int64,
    }
    for field_name, dtype in optional_fields.items():
        if field_name in payload:
            arrays[field_name] = np.asarray(payload[field_name], dtype=dtype)
    np.savez(path, **arrays)
    return str(path)


def _write_trace_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    return str(path)


def _build_run_scalar_logs(experiment_config: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(experiment_config.get("model", {}))
    return {
        "query/continuous_temperature": model_config.get("continuous_temperature"),
        "query/discrete_temperature": model_config.get("discrete_temperature"),
        "query/num_samples_train": model_config.get("monte_carlo_samples"),
        "query/num_samples_eval": model_config.get("monte_carlo_samples"),
        "query/continuous_weight_entropy_mean": model_config.get(
            "continuous_weight_entropy_mean"
        ),
        "query/discrete_topk_weight_entropy_mean": model_config.get(
            "discrete_topk_weight_entropy_mean"
        ),
    }


def _build_uq_summary_inputs(
    artifact_inputs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "clean_validation": {
            "point_scores": artifact_inputs["clean_validation"]["point_scores"],
            "traces": artifact_inputs["clean_validation_traces"],
        },
        "synthetic_validation": {
            "point_scores": artifact_inputs["synthetic_validation"]["point_scores"],
            "traces": artifact_inputs["synthetic_validation_traces"],
        },
        "test": {
            "point_scores": artifact_inputs["test"]["point_scores"],
            "traces": artifact_inputs["test_traces"],
        },
    }


def _resolve_retention_policy(experiment_config: dict[str, Any]) -> str:
    evaluation_config = dict(experiment_config.get("evaluation", {}))
    return str(evaluation_config.get("retention_policy", "retain_for_eda"))


def _summarize_loaded_checkpoint_contract(
    *,
    checkpoint_path: str,
    loaded_checkpoint: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    relevant_fields = [
        "stochastic_inference",
        "monte_carlo_samples",
        "continuous_temperature",
        "discrete_temperature",
        "variance_correction",
        "return_mc_samples",
        "sample_retention_policy",
    ]
    checkpoint_metadata = dict(loaded_checkpoint.get("checkpoint_metadata") or {})
    extra_state = dict(loaded_checkpoint.get("extra_state") or {})
    model_flags = {
        field_name: getattr(model, field_name, None) for field_name in relevant_fields
    }
    checkpoint_metadata_flags = {
        field_name: checkpoint_metadata.get(field_name)
        for field_name in relevant_fields
        if field_name in checkpoint_metadata
    }
    extra_state_flags = {
        field_name: extra_state.get(field_name)
        for field_name in relevant_fields
        if field_name in extra_state
    }
    return {
        "checkpoint_path": str(checkpoint_path),
        "has_checkpoint_metadata": bool(checkpoint_metadata),
        "has_extra_state": bool(extra_state),
        "checkpoint_metadata_flags": checkpoint_metadata_flags,
        "extra_state_flags": extra_state_flags,
        "model_flags": model_flags,
        "metadata_mismatches": {
            field_name: {
                "checkpoint": checkpoint_metadata.get(field_name),
                "model": model_flags[field_name],
            }
            for field_name in relevant_fields
            if field_name in checkpoint_metadata
            and checkpoint_metadata.get(field_name) != model_flags[field_name]
        },
        "extra_state_mismatches": {
            field_name: {
                "checkpoint": extra_state.get(field_name),
                "model": model_flags[field_name],
            }
            for field_name in relevant_fields
            if field_name in extra_state
            and extra_state.get(field_name) != model_flags[field_name]
        },
    }


def _summarize_trace_payloads(trace_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sample_keys = [
        "point_score_samples",
        "window_score_samples",
        "reconstruction_samples",
        "classification_probability_samples",
    ]
    mc_histories_non_null_count = {
        key_name: sum(
            1
            for trace_payload in trace_payloads
            if trace_payload.get("mc_sample_histories", {}).get(key_name) is not None
        )
        for key_name in sample_keys
    }
    return {
        "num_traces": len(trace_payloads),
        "any_uncertainty_history": any(
            trace_payload.get("uncertainty_history") is not None
            for trace_payload in trace_payloads
        ),
        "uncertainty_history_non_null_count": sum(
            1
            for trace_payload in trace_payloads
            if trace_payload.get("uncertainty_history") is not None
        ),
        "mc_histories_non_null_count": mc_histories_non_null_count,
        "any_mc_sample_history": any(
            count > 0 for count in mc_histories_non_null_count.values()
        ),
        "first_sample_retention_policy": (
            trace_payloads[0].get("sample_retention_policy") if trace_payloads else None
        ),
    }


def _summarize_metric_variance_keys(metrics: dict[str, Any]) -> dict[str, Any]:
    variance_metric_keys = sorted(
        key_name
        for key_name, value in metrics.items()
        if "variance" in key_name and isinstance(value, (int, float))
    )
    return {
        "has_variance_metrics": bool(variance_metric_keys),
        "variance_metric_keys": variance_metric_keys,
    }


def collect_offline_artifact_inputs(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    manifest: dict[str, Any],
    execution_report: dict[str, Any],
) -> dict[str, Any]:
    register_evaluation_runtime_components()
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"],
        experiment_config["data"],
    )
    model = build_model_from_experiment_config(experiment_config)
    checkpoint_path = str(manifest["evaluation"]["checkpoint_path"])
    checkpoint_payload = _load_evaluation_checkpoint(
        experiment_config,
        manifest,
        model,
    )
    checkpoint_audit = _summarize_loaded_checkpoint_contract(
        checkpoint_path=checkpoint_path,
        loaded_checkpoint=checkpoint_payload,
        model=model,
    )
    data_bundle = _maybe_rebuild_with_checkpoint_scaler(
        data_bundle,
        experiment_config,
        checkpoint_payload,
    )
    evaluator = _build_evaluator(experiment_config)
    split_outputs = _evaluate_offline_benchmark_splits(
        evaluator=evaluator,
        model=model,
        loaders=data_bundle["loaders"],
        protocol_config=protocol_config,
        scaler=data_bundle.get("scaler"),
    )
    return {
        "entity_id": _require_single_entity_id(split_outputs["test"], "test"),
        "seed": int(experiment_config.get("seed", 0)),
        "variant_name": str(experiment_config.get("offline_variant", "O0")),
        "model": model,
        "checkpoint_path": checkpoint_path,
        "scaler": data_bundle.get("scaler"),
        "clean_validation_sequences": data_bundle.get("scaled_sequences", {}).get(
            "val", []
        ),
        "device": str(experiment_config["device"]),
        "point_score_calibration": split_outputs["point_score_calibration"],
        "offline_point_threshold": float(split_outputs["offline_point_threshold"]),
        "offline_point_threshold_source": str(
            split_outputs["offline_point_threshold_source"]
        ),
        "clean_validation": split_outputs["clean_validation_payload"],
        "clean_validation_traces": split_outputs["clean_validation"].get("traces", []),
        "synthetic_validation": _evaluation_outputs_to_score_payload(
            split_outputs["synthetic_validation"],
        ),
        "synthetic_validation_traces": split_outputs["synthetic_validation"].get(
            "traces",
            [],
        ),
        "test": _evaluation_outputs_to_score_payload(split_outputs["test"]),
        "test_traces": split_outputs["test"].get("traces", []),
        "offline_metrics": dict(split_outputs["test"]["metrics"]),
        "variance_trace_audit": {
            "checkpoint": checkpoint_audit,
            "metrics": _summarize_metric_variance_keys(
                split_outputs["test"]["metrics"]
            ),
            "traces": {
                "clean_validation": _summarize_trace_payloads(
                    split_outputs["clean_validation"].get("traces", [])
                ),
                "synthetic_validation": _summarize_trace_payloads(
                    split_outputs["synthetic_validation"].get("traces", [])
                ),
                "test": _summarize_trace_payloads(
                    split_outputs["test"].get("traces", [])
                ),
            },
            "retention": {
                "retention_policy": str(
                    experiment_config.get("evaluation", {}).get(
                        "retention_policy", "retain_for_eda"
                    )
                ),
                "inspection_ready": bool(
                    experiment_config.get("evaluation", {}).get(
                        "retention_policy", "retain_for_eda"
                    )
                    == "retain_for_eda"
                ),
            },
        },
    }


def _load_evaluation_checkpoint(
    experiment_config: dict[str, Any],
    manifest: dict[str, Any],
    model: Any,
) -> dict[str, Any]:
    checkpoint_path = manifest["evaluation"]["checkpoint_path"]
    checkpoint_manager = CheckpointManager(experiment_config["checkpoint_dir"])
    return checkpoint_manager.load_checkpoint(checkpoint_path, model, strict=False)


def _maybe_rebuild_with_checkpoint_scaler(
    data_bundle: dict[str, Any],
    experiment_config: dict[str, Any],
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    if "raw_sequences" not in data_bundle:
        return data_bundle
    scaler_state = checkpoint_payload.get("scaler_state_dict")
    if scaler_state is None:
        return data_bundle
    return rebuild_dataset_bundle_with_scaler_state(
        data_bundle=data_bundle,
        data_config=experiment_config["data"],
        scaler_state_dict=scaler_state,
    )


def _build_evaluator(experiment_config: dict[str, Any]) -> Evaluator:
    evaluation_config = dict(experiment_config.get("evaluation", {}))
    return Evaluator(
        device=str(experiment_config["device"]),
        vus_max_buffer_size=evaluation_config.get("vus_max_buffer_size"),
        vus_num_thresholds=int(evaluation_config.get("vus_num_thresholds", 200)),
    )


def _evaluate_offline_benchmark_splits(
    *,
    evaluator: Evaluator,
    model: Any,
    loaders: dict[str, Any],
    protocol_config: dict[str, Any],
    scaler: Any | None = None,
) -> dict[str, Any]:
    score_space = str(protocol_config.get("score_space", "model_output"))
    raw_protocol = score_space == "raw_input" and scaler is not None
    if score_space == "raw_input" and scaler is None:
        raise ValueError("raw_input offline evaluation requires a fitted scaler")
    if raw_protocol:
        clean_outputs = evaluator.evaluate(
            model,
            loaders["val"],
            score_space="raw_input",
            scaler=scaler,
        )
        clean_payload = _evaluation_outputs_to_score_payload(clean_outputs)
        clean_threshold = select_clean_validation_point_threshold(
            clean_payload["point_scores"],
            quantile=float(protocol_config["offline_threshold_quantile"]),
        )
        clean_window_threshold = float(
            np.quantile(
                clean_payload["raw_input_window_mse"],
                float(protocol_config["B_window_quantile"]),
            )
        )
        point_source = protocol_config.get(
            "offline_point_threshold_source_split", "clean_validation"
        )
        if point_source == "synthetic_validation_normal":
            synthetic_calibration_outputs = _evaluate_named_split(
                evaluator,
                model,
                loaders,
                split_name="val_synth",
                fallback_split_name="val",
                point_score_threshold=None,
                window_score_threshold=clean_window_threshold,
                score_space="raw_input",
                scaler=scaler,
            )
            synthetic_calibration_payload = _evaluation_outputs_to_score_payload(
                synthetic_calibration_outputs
            )
            selected_point_threshold = (
                select_synthetic_validation_normal_point_threshold(
                    synthetic_calibration_payload["point_scores"],
                    synthetic_calibration_payload["point_labels"],
                    quantile=float(protocol_config["offline_threshold_quantile"]),
                )
            )
            threshold_source = "synthetic_validation_normal_quantile"
            synthetic_outputs = _evaluate_named_split(
                evaluator,
                model,
                loaders,
                split_name="val_synth",
                fallback_split_name="val",
                point_score_threshold=selected_point_threshold,
                threshold_source=threshold_source,
                window_score_threshold=clean_window_threshold,
                score_space="raw_input",
                scaler=scaler,
            )
        else:
            selected_point_threshold = clean_threshold
            threshold_source = "clean_validation_quantile"
            synthetic_outputs = _evaluate_named_split(
                evaluator,
                model,
                loaders,
                split_name="val_synth",
                fallback_split_name="val",
                point_score_threshold=selected_point_threshold,
                window_score_threshold=clean_window_threshold,
                score_space="raw_input",
                scaler=scaler,
            )
        test_outputs = evaluator.evaluate(
            model,
            loaders["test"],
            point_score_threshold=selected_point_threshold,
            threshold_source=threshold_source,
            window_score_threshold=clean_window_threshold,
            score_space="raw_input",
            scaler=scaler,
        )
        return {
            "clean_validation": clean_outputs,
            "clean_validation_payload": clean_payload,
            "raw_clean_validation_payload": clean_payload,
            "point_score_calibration": None,
            "synthetic_validation": synthetic_outputs,
            "test": test_outputs,
            "offline_point_threshold": selected_point_threshold,
            "offline_point_threshold_source": point_source,
        }
    raw_clean_outputs = evaluator.evaluate(model, loaders["val"])
    _require_single_entity_id(raw_clean_outputs, "clean_validation")
    raw_clean_payload = _evaluation_outputs_to_score_payload(raw_clean_outputs)
    calibration = fit_mad_logistic_calibration(raw_clean_payload["point_scores"])
    if not hasattr(model, "set_point_score_calibration"):
        raise TypeError("THESIS offline model must support point-score calibration")
    model.set_point_score_calibration(calibration)
    clean_outputs = evaluator.evaluate(model, loaders["val"])
    clean_payload = _evaluation_outputs_to_score_payload(clean_outputs)
    clean_threshold = select_clean_validation_point_threshold(
        clean_payload["point_scores"],
        quantile=float(protocol_config["offline_threshold_quantile"]),
    )
    synthetic_outputs = _evaluate_named_split(
        evaluator,
        model,
        loaders,
        split_name="val_synth",
        fallback_split_name="val",
        point_score_threshold=clean_threshold,
    )
    test_outputs = evaluator.evaluate(
        model,
        loaders["test"],
        point_score_threshold=clean_threshold,
        threshold_source="clean_validation_quantile",
    )
    return {
        "clean_validation": clean_outputs,
        "clean_validation_payload": clean_payload,
        "raw_clean_validation_payload": raw_clean_payload,
        "point_score_calibration": calibration,
        "synthetic_validation": synthetic_outputs,
        "test": test_outputs,
        "offline_point_threshold": clean_threshold,
        "offline_point_threshold_source": "clean_validation",
    }


def _evaluate_named_split(
    evaluator: Evaluator,
    model: Any,
    loaders: dict[str, Any],
    *,
    split_name: str,
    fallback_split_name: str,
    point_score_threshold: float | None,
    threshold_source: str | None = None,
    window_score_threshold: float | None = None,
    score_space: str = "model_output",
    scaler: Any | None = None,
) -> dict[str, Any]:
    loader = loaders.get(split_name, loaders[fallback_split_name])
    kwargs: dict[str, Any] = {}
    if point_score_threshold is not None:
        kwargs["point_score_threshold"] = point_score_threshold
        kwargs["threshold_source"] = threshold_source or "clean_validation_quantile"
    if score_space == "raw_input":
        kwargs.update(
            {
                "score_space": score_space,
                "scaler": scaler,
                "window_score_threshold": window_score_threshold,
                "evaluation_stage": "val_synth",
            }
        )
    return evaluator.evaluate(model, loader, **kwargs)


def _evaluation_outputs_to_score_payload(
    evaluation_outputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    score_arrays: list[np.ndarray] = []
    raw_score_arrays: list[np.ndarray] = []
    normalized_score_arrays: list[np.ndarray] = []
    prediction_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    mask_arrays: list[np.ndarray] = []
    for record in evaluation_outputs["records"]:
        mask = np.asarray(record["covered_point_mask"], dtype=bool)
        point_scores = np.asarray(record["point_scores"], dtype=float)[mask]
        score_arrays.append(point_scores)
        if "raw_input_point_mse" in record:
            raw_score_arrays.append(
                np.asarray(record["raw_input_point_mse"], dtype=float)[mask]
            )
        if "normalized_input_point_mse" in record:
            normalized_score_arrays.append(
                np.asarray(record["normalized_input_point_mse"], dtype=float)[mask]
            )
        if "point_predictions" in record:
            prediction_arrays.append(
                np.asarray(record["point_predictions"], dtype=np.int64)[mask]
            )
        label_arrays.append(np.asarray(record["point_labels"], dtype=np.int64)[mask])
        mask_arrays.append(np.ones(int(mask.sum()), dtype=bool))
    payload = {
        "point_scores": np.concatenate(score_arrays),
        "point_labels": np.concatenate(label_arrays),
        "covered_point_mask": np.concatenate(mask_arrays),
    }
    if raw_score_arrays:
        payload["raw_input_point_mse"] = np.concatenate(raw_score_arrays)
    if normalized_score_arrays:
        payload["normalized_input_point_mse"] = np.concatenate(normalized_score_arrays)
    if prediction_arrays:
        payload["point_predictions"] = np.concatenate(prediction_arrays)
    window_records = evaluation_outputs.get("window_records", [])
    if window_records:
        payload["raw_input_window_mse"] = np.asarray(
            [item["raw_input_window_mse"] for item in window_records], dtype=float
        )
        payload["normalized_input_window_mse"] = np.asarray(
            [item["normalized_input_window_mse"] for item in window_records],
            dtype=float,
        )
        payload["window_labels"] = np.asarray(
            [item["window_label"] for item in window_records], dtype=np.int64
        )
        if all("window_prediction" in item for item in window_records):
            payload["window_predictions"] = np.asarray(
                [item["window_prediction"] for item in window_records],
                dtype=np.int64,
            )
    return payload


def _first_entity_id(evaluation_outputs: dict[str, Any]) -> str:
    records = evaluation_outputs["records"]
    if not records:
        return "unknown"
    return str(records[0]["entity_id"])


def _require_single_entity_id(
    evaluation_outputs: dict[str, Any], split_name: str
) -> str:
    entity_ids = {
        str(record["entity_id"]) for record in evaluation_outputs.get("records", [])
    }
    if len(entity_ids) != 1:
        raise ValueError(
            f"THESIS {split_name} calibration artifact requires exactly one entity, "
            f"got {sorted(entity_ids)}"
        )
    return next(iter(entity_ids))


def _build_thresholds(
    artifact_inputs: dict[str, Any],
    protocol_config: dict[str, Any],
    experiment_config_path: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    raw_protocol = (
        str(protocol_config.get("score_space", "model_output")) == "raw_input"
    )
    clean_scores = np.asarray(
        artifact_inputs["clean_validation"].get(
            "raw_input_point_mse",
            artifact_inputs["clean_validation"]["point_scores"],
        ),
        dtype=float,
    )
    quantile = float(protocol_config["offline_threshold_quantile"])
    checkpoint_path = Path(str(artifact_inputs.get("checkpoint_path", "")))
    entry = StageBInventoryEntry(
        experiment_config_path=Path(experiment_config_path),
        offline_variant=str(artifact_inputs["variant_name"]),
        entity_id=str(artifact_inputs["entity_id"]),
        seed=int(artifact_inputs["seed"]),
        threshold_artifact_v3_path=Path(),
        stage_b_best_checkpoint_path=checkpoint_path,
        threshold_artifact_v4_path=Path(),
        audit_path=Path(),
    )
    register_online_runtime_components()
    online_config = load_a0_scoring_config(
        entry,
        int(protocol_config["window_size"]),
    )
    if raw_protocol:
        online_model = artifact_inputs["model"]
    else:
        online_model = _build_model_from_experiment_config(online_config)
        online_model.set_point_score_calibration(
            artifact_inputs["point_score_calibration"]
        )
    if hasattr(online_model, "to"):
        online_model.to(str(artifact_inputs["device"]))
    online_calibration = collect_stride1_online_scores(
        model=online_model,
        clean_validation_sequences=artifact_inputs["clean_validation_sequences"],
        window_size=int(protocol_config["window_size"]),
        batch_size=1,
        view_noise_std=0.0,
        view_dropout_probability=0.0,
        device=artifact_inputs["device"],
        current_weight=float(protocol_config["online_ewma_current_weight"]),
        previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        scaler=artifact_inputs.get("scaler") if raw_protocol else None,
    )
    selected_offline_threshold = float(
        artifact_inputs.get(
            "offline_point_threshold",
            select_clean_validation_point_threshold(
                clean_scores,
                quantile=quantile,
            ),
        )
    )
    if not np.isfinite(selected_offline_threshold):
        raise ValueError("offline point threshold must be finite")
    builder_kwargs: dict[str, Any] = {
        "method_name": "THESIS",
        "variant_name": str(artifact_inputs["variant_name"]),
        "entity_id": str(artifact_inputs["entity_id"]),
        "seed": int(artifact_inputs["seed"]),
        "window_size": int(protocol_config["window_size"]),
        "offline_point_threshold": selected_offline_threshold,
        "online_ewma_point_threshold": select_online_ewma_threshold(
            np.asarray(online_calibration["ewma"], dtype=float),
            quantile=float(protocol_config["online_threshold_quantile"]),
        ),
        "quantile": quantile,
        "ewma_current_weight": float(protocol_config["online_ewma_current_weight"]),
        "ewma_previous_weight": float(protocol_config["online_ewma_previous_weight"]),
        "created_by": "scripts/run_thesis_offline_benchmark.py",
        "config_path": experiment_config_path,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": sha256_file(experiment_config_path),
        "input_window_threshold": float(
            np.quantile(
                np.asarray(online_calibration["input_window"], dtype=float),
                float(protocol_config["B_window_quantile"]),
            )
        ),
        "latent_window_low_threshold": float(
            np.quantile(
                np.asarray(online_calibration["latent_window"], dtype=float),
                float(protocol_config["A_low_quantile"]),
            )
        ),
        "latent_window_high_threshold": float(
            np.quantile(
                np.asarray(online_calibration["latent_window"], dtype=float),
                float(protocol_config["A_high_quantile"]),
            )
        ),
        "latent_window_low_quantile": float(protocol_config["A_low_quantile"]),
        "score_space": "raw_input" if raw_protocol else "model_output",
    }
    offline_point_source = artifact_inputs.get(
        "offline_point_threshold_source", "clean_validation"
    )
    if offline_point_source != "clean_validation":
        builder_kwargs["offline_point_threshold_source_split"] = str(
            offline_point_source
        )
    if not raw_protocol:
        builder_kwargs.update(
            {
                "point_score_c": float(
                    artifact_inputs["point_score_calibration"].center
                ),
                "point_score_tau": float(
                    artifact_inputs["point_score_calibration"].tau
                ),
            }
        )
    return build_threshold_artifact(**builder_kwargs)


def _export_offline_artifacts(
    *,
    output_dir: Path,
    artifact_inputs: dict[str, Any],
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    experiment_config_path: str,
    protocol_config_path: str,
    manifest: dict[str, Any],
) -> dict[str, str]:
    checkpoint_path = manifest.get("evaluation", {}).get("checkpoint_path")
    if not checkpoint_path or not Path(str(checkpoint_path)).is_file():
        raise FileNotFoundError(
            "offline artifact export requires Stage B best checkpoint"
        )
    checkpoint_sha256 = sha256_file(str(checkpoint_path))
    threshold_artifact = _build_thresholds(
        artifact_inputs,
        protocol_config,
        experiment_config_path,
        checkpoint_sha256,
    )
    threshold_path = output_dir / "thresholds" / "thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)
    uq_summary_payload = build_uq_summary_payload(
        benchmark_kind="offline",
        experiment_name=str(experiment_config.get("experiment_name")),
        method_name="THESIS",
        variant_name=str(artifact_inputs["variant_name"]),
        entity_id=str(artifact_inputs["entity_id"]),
        seed=int(artifact_inputs["seed"]),
        stage_name=str(
            experiment_config.get("stage_name")
            or experiment_config.get("model", {}).get("stage_name")
            or "stage_b_fusion_finetuning"
        ),
        checkpoint_path=str(checkpoint_path) if checkpoint_path else "",
        checkpoint_sha256=checkpoint_sha256,
        experiment_config_path=experiment_config_path,
        protocol_config_path=protocol_config_path,
        output_dir=str(output_dir),
        run_scalar_logs=_build_run_scalar_logs(experiment_config),
        split_inputs=_build_uq_summary_inputs(artifact_inputs),
    )
    uq_summary_path = output_dir / "metrics" / "uq_summary.json"
    write_uq_summary_json(uq_summary_path, uq_summary_payload)
    compacted_clean_validation_traces = compact_evaluation_trace_payloads(
        artifact_inputs["clean_validation_traces"]
    )
    compacted_synthetic_validation_traces = compact_evaluation_trace_payloads(
        artifact_inputs["synthetic_validation_traces"]
    )
    compacted_test_traces = compact_evaluation_trace_payloads(
        artifact_inputs["test_traces"]
    )
    return {
        "thresholds": str(threshold_path),
        "uq_summary": str(uq_summary_path),
        "clean_validation_scores": _write_score_npz(
            output_dir / "scores" / "clean_validation_point_scores.npz",
            artifact_inputs["clean_validation"],
        ),
        "clean_validation_traces": _write_trace_json(
            output_dir / "traces" / "clean_validation_traces.json",
            compacted_clean_validation_traces,
        ),
        "synthetic_validation_scores": _write_score_npz(
            output_dir / "scores" / "synthetic_validation_point_scores.npz",
            artifact_inputs["synthetic_validation"],
        ),
        "synthetic_validation_traces": _write_trace_json(
            output_dir / "traces" / "synthetic_validation_traces.json",
            compacted_synthetic_validation_traces,
        ),
        "test_scores": _write_score_npz(
            output_dir / "scores" / "test_point_scores.npz",
            artifact_inputs["test"],
        ),
        "test_traces": _write_trace_json(
            output_dir / "traces" / "test_traces.json",
            compacted_test_traces,
        ),
        "offline_metrics": _write_json(
            output_dir / "metrics" / "offline_metrics.json",
            artifact_inputs["offline_metrics"],
        ),
        "resolved_protocol": _write_json(
            output_dir / "protocol" / "resolved_protocol.json",
            protocol_config,
        ),
    }


from scripts.benchmarks._internal.run_thesis_offline_benchmark_helpers import (
    _export_offline_retention_bundle,
)


def _build_evaluation_only_run(
    experiment_config: dict[str, Any],
    checkpoint_path: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = {
        "manifest_root": str(output_dir / "evaluation_only"),
        "evaluation": {"checkpoint_path": checkpoint_path},
        "evaluation_only": True,
    }
    execution_report = {
        "manifest_path": None,
        "execution_report_path": None,
        "started_at_utc": _utc_now_iso(),
        "finished_at_utc": _utc_now_iso(),
        "dry_run": False,
        "skip_completed": False,
        "resumed_from_existing_report": False,
        "status": "evaluation_only",
        "executed_stage_names": [],
        "completed_stage_names": [],
        "skipped_stage_names": [],
        "evaluation_only": True,
        "checkpoint_path": checkpoint_path,
    }
    return manifest, execution_report


def run_thesis_offline_benchmark(
    *,
    experiment_config_path: str,
    protocol_config_path: str,
    dry_run: bool,
    skip_completed: bool,
    evaluation_only: bool = False,
    checkpoint_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    if output_dir is not None and not str(output_dir).strip():
        raise ValueError("output_dir must be a non-empty path")
    if output_dir is not None and not evaluation_only:
        raise ValueError("output_dir is only supported with --evaluation-only")
    experiment_config = load_experiment_config(experiment_config_path)
    protocol_config = _load_yaml_config(protocol_config_path)
    retention_policy = _resolve_retention_policy(experiment_config)
    effective_output_dir = Path(
        str(output_dir) if output_dir is not None else str(experiment_config["output_dir"])
    )

    validate_protocol_config(protocol_config)
    if (
        output_dir is not None
        and evaluation_only
        and protocol_config.get("offline_point_threshold_source_split")
        == "synthetic_validation_normal"
        and effective_output_dir.resolve()
        == Path(str(experiment_config["output_dir"])).resolve()
    ):
        raise ValueError(
            "synthetic validation reruns require a new rerun root, not the legacy output root"
        )
    if evaluation_only:
        if dry_run:
            raise ValueError("--dry-run cannot be combined with --evaluation-only")
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint-path is required when --evaluation-only is set"
            )
        manifest, execution_report = _build_evaluation_only_run(
            experiment_config,
            checkpoint_path,
            effective_output_dir,
        )
    else:
        validate_two_stage_epoch_budget(experiment_config)
        manifest = materialize_two_stage_run_manifest(experiment_config)
        execution_report = execute_two_stage_plan(
            manifest,
            dry_run=dry_run,
            skip_completed=skip_completed,
        )
    artifact_paths: dict[str, str] = {}
    retention_artifact_paths: dict[str, str] = {}
    variance_trace_audit: dict[str, Any] | None = None
    if not dry_run:
        artifact_inputs = collect_offline_artifact_inputs(
            experiment_config=experiment_config,
            protocol_config=protocol_config,
            manifest=manifest,
            execution_report=execution_report,
        )
        variance_trace_audit = artifact_inputs.get("variance_trace_audit")
        artifact_paths = _export_offline_artifacts(
            output_dir=effective_output_dir,
            artifact_inputs=artifact_inputs,
            experiment_config=experiment_config,
            protocol_config=protocol_config,
            experiment_config_path=experiment_config_path,
            protocol_config_path=protocol_config_path,
            manifest=manifest,
        )
        retention_artifact_paths = _export_offline_retention_bundle(
            output_dir=effective_output_dir,
            artifact_inputs=artifact_inputs,
            artifact_paths=artifact_paths,
            manifest=manifest,
            execution_report=execution_report,
            experiment_config=experiment_config,
            experiment_config_path=experiment_config_path,
            protocol_config=protocol_config,
            protocol_config_path=protocol_config_path,
            retention_policy=retention_policy,
        )
    report = {
        "benchmark_status": "dry_run" if dry_run else execution_report["status"],
        "evaluation_only": evaluation_only,
        "checkpoint_path": checkpoint_path,
        "artifact_paths": artifact_paths,
        "retention_policy": retention_policy,
        "retention_artifact_paths": retention_artifact_paths,
        "variance_trace_audit": variance_trace_audit,
        "created_at_utc": _utc_now_iso(),
        "experiment_config_path": experiment_config_path,
        "protocol_config_path": protocol_config_path,
        "protocol": protocol_config,
        "two_stage_manifest": manifest,
        "two_stage_execution": execution_report,
    }
    report_path = _write_report(effective_output_dir, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    report = run_thesis_offline_benchmark(
        experiment_config_path=args.experiment_config,
        protocol_config_path=args.protocol_config,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
        evaluation_only=args.evaluation_only,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
