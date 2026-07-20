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
from src.core.uq_summary import (
    build_uq_summary_payload,
    write_uq_summary_json,
)
from src.core.registry import build_dataset
from src.data.loaders import rebuild_dataset_bundle_with_scaler_state
from src.engine.checkpoint import CheckpointManager
from src.engine.evaluator import Evaluator
from src.engine.thresholding import (
    select_clean_validation_point_threshold,
    select_online_ewma_threshold,
)
from src.protocols.point_scores import ewma_scores
from src.protocols.threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)


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

    return _validate_protocol_config(protocol_config)


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
    np.savez(
        path,
        point_scores=np.asarray(payload["point_scores"], dtype=float),
        point_labels=np.asarray(payload["point_labels"], dtype=np.int64),
        covered_point_mask=np.asarray(payload["covered_point_mask"], dtype=bool),
    )
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
    )
    return {
        "entity_id": _first_entity_id(split_outputs["test"]),
        "seed": int(experiment_config.get("seed", 0)),
        "variant_name": str(experiment_config.get("offline_variant", "O0")),
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
) -> dict[str, Any]:
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
        "synthetic_validation": synthetic_outputs,
        "test": test_outputs,
    }


def _evaluate_named_split(
    evaluator: Evaluator,
    model: Any,
    loaders: dict[str, Any],
    *,
    split_name: str,
    fallback_split_name: str,
    point_score_threshold: float,
) -> dict[str, Any]:
    loader = loaders.get(split_name, loaders[fallback_split_name])
    return evaluator.evaluate(
        model,
        loader,
        point_score_threshold=point_score_threshold,
        threshold_source="clean_validation_quantile",
    )


def _evaluation_outputs_to_score_payload(
    evaluation_outputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    score_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    mask_arrays: list[np.ndarray] = []
    for record in evaluation_outputs["records"]:
        mask = np.asarray(record["covered_point_mask"], dtype=bool)
        score_arrays.append(np.asarray(record["point_scores"], dtype=float)[mask])
        label_arrays.append(np.asarray(record["point_labels"], dtype=np.int64)[mask])
        mask_arrays.append(np.ones(int(mask.sum()), dtype=bool))
    return {
        "point_scores": np.concatenate(score_arrays),
        "point_labels": np.concatenate(label_arrays),
        "covered_point_mask": np.concatenate(mask_arrays),
    }


def _first_entity_id(evaluation_outputs: dict[str, Any]) -> str:
    records = evaluation_outputs["records"]
    if not records:
        return "unknown"
    return str(records[0]["entity_id"])


def _build_thresholds(
    artifact_inputs: dict[str, Any],
    protocol_config: dict[str, Any],
    experiment_config_path: str,
) -> dict[str, Any]:
    clean_scores = np.asarray(
        artifact_inputs["clean_validation"]["point_scores"],
        dtype=float,
    )
    quantile = float(protocol_config["offline_threshold_quantile"])
    online_scores = ewma_scores(
        clean_scores,
        current_weight=float(protocol_config["online_ewma_current_weight"]),
        previous_weight=float(protocol_config["online_ewma_previous_weight"]),
    )
    return build_threshold_artifact(
        method_name="THESIS",
        variant_name=str(artifact_inputs["variant_name"]),
        entity_id=str(artifact_inputs["entity_id"]),
        seed=int(artifact_inputs["seed"]),
        window_size=int(protocol_config["window_size"]),
        offline_point_threshold=select_clean_validation_point_threshold(
            clean_scores,
            quantile=quantile,
        ),
        online_ewma_point_threshold=select_online_ewma_threshold(
            online_scores,
            quantile=float(protocol_config["online_threshold_quantile"]),
        ),
        quantile=quantile,
        ewma_current_weight=float(protocol_config["online_ewma_current_weight"]),
        ewma_previous_weight=float(protocol_config["online_ewma_previous_weight"]),
        created_by="scripts/run_thesis_offline_benchmark.py",
        config_path=experiment_config_path,
    )


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
    threshold_artifact = _build_thresholds(
        artifact_inputs,
        protocol_config,
        experiment_config_path,
    )
    threshold_path = output_dir / "thresholds" / "thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)
    checkpoint_path = manifest.get("evaluation", {}).get("checkpoint_path")
    checkpoint_sha256 = None
    if checkpoint_path and Path(str(checkpoint_path)).is_file():
        checkpoint_sha256 = sha256_file(str(checkpoint_path))
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
    return {
        "thresholds": str(threshold_path),
        "uq_summary": str(uq_summary_path),
        "clean_validation_scores": _write_score_npz(
            output_dir / "scores" / "clean_validation_point_scores.npz",
            artifact_inputs["clean_validation"],
        ),
        "clean_validation_traces": _write_trace_json(
            output_dir / "traces" / "clean_validation_traces.json",
            artifact_inputs["clean_validation_traces"],
        ),
        "synthetic_validation_scores": _write_score_npz(
            output_dir / "scores" / "synthetic_validation_point_scores.npz",
            artifact_inputs["synthetic_validation"],
        ),
        "synthetic_validation_traces": _write_trace_json(
            output_dir / "traces" / "synthetic_validation_traces.json",
            artifact_inputs["synthetic_validation_traces"],
        ),
        "test_scores": _write_score_npz(
            output_dir / "scores" / "test_point_scores.npz",
            artifact_inputs["test"],
        ),
        "test_traces": _write_trace_json(
            output_dir / "traces" / "test_traces.json",
            artifact_inputs["test_traces"],
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


def run_thesis_offline_benchmark(
    *,
    experiment_config_path: str,
    protocol_config_path: str,
    dry_run: bool,
    skip_completed: bool,
    evaluation_only: bool = False,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    experiment_config = load_experiment_config(experiment_config_path)
    protocol_config = _load_yaml_config(protocol_config_path)
    retention_policy = _resolve_retention_policy(experiment_config)

    validate_protocol_config(protocol_config)
    if evaluation_only:
        if dry_run:
            raise ValueError("--dry-run cannot be combined with --evaluation-only")
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint-path is required when --evaluation-only is set"
            )
        manifest = {
            "manifest_root": str(
                Path(experiment_config["output_dir"]) / "evaluation_only"
            ),
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
            output_dir=Path(str(experiment_config["output_dir"])),
            artifact_inputs=artifact_inputs,
            experiment_config=experiment_config,
            protocol_config=protocol_config,
            experiment_config_path=experiment_config_path,
            protocol_config_path=protocol_config_path,
            manifest=manifest,
        )
        retention_artifact_paths = _export_offline_retention_bundle(
            output_dir=Path(str(experiment_config["output_dir"])),
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
    report_path = _write_report(Path(str(experiment_config["output_dir"])), report)
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
    args = parser.parse_args()
    report = run_thesis_offline_benchmark(
        experiment_config_path=args.experiment_config,
        protocol_config_path=args.protocol_config,
        dry_run=args.dry_run,
        skip_completed=args.skip_completed,
        evaluation_only=args.evaluation_only,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
