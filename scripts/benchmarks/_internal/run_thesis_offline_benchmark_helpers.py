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

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.core.config import load_experiment_config
from src.core.artifact_integrity import (
    build_retention_bundle_manifest,
    sha256_file,
    write_retention_bundle_manifest,
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


def _resolve_retention_policy(experiment_config: dict[str, Any]) -> str:
    evaluation_config = dict(experiment_config.get("evaluation", {}))
    return str(evaluation_config.get("retention_policy", "retain_for_eda"))


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
    checkpoint_payload = _load_evaluation_checkpoint(
        experiment_config,
        manifest,
        model,
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
    protocol_config: dict[str, Any],
    experiment_config_path: str,
) -> dict[str, str]:
    threshold_artifact = _build_thresholds(
        artifact_inputs,
        protocol_config,
        experiment_config_path,
    )
    threshold_path = output_dir / "thresholds" / "thresholds.json"
    write_threshold_artifact(threshold_artifact, threshold_path)
    return {
        "thresholds": str(threshold_path),
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


def _export_offline_retention_bundle(
    *,
    output_dir: Path,
    artifact_inputs: dict[str, Any],
    artifact_paths: dict[str, str],
    manifest: dict[str, Any],
    execution_report: dict[str, Any],
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    protocol_config_path: str,
    retention_policy: str,
) -> dict[str, str]:
    entity_id = str(artifact_inputs["entity_id"])
    retention_root = output_dir / "retention" / entity_id / "offline"
    bundle_paths: dict[str, str] = {}
    resolved_config_sha256 = CheckpointManager._stable_json_digest(experiment_config)
    checkpoint_sha256 = None
    candidate_checkpoint = manifest.get("evaluation", {}).get("checkpoint_path")
    if candidate_checkpoint:
        candidate_path = Path(str(candidate_checkpoint))
        if candidate_path.is_file():
            checkpoint_sha256 = sha256_file(candidate_path)
    summary_payload = {
        "bundle_type": "offline_thesis_retention",
        "bundle_schema_version": 1,
        "entity_id": entity_id,
        "retention_policy": retention_policy,
        "compression": "none",
        "experiment_name": experiment_config.get("experiment_name"),
        "protocol_config_path": protocol_config_path,
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "artifact_paths": dict(artifact_paths),
        "evaluation_split_sizes": {
            "clean_validation": int(
                len(np.asarray(artifact_inputs["clean_validation"]["point_scores"]))
            ),
            "synthetic_validation": int(
                len(np.asarray(artifact_inputs["synthetic_validation"]["point_scores"]))
            ),
            "test": int(len(np.asarray(artifact_inputs["test"]["point_scores"]))),
        },
        "offline_metrics": dict(artifact_inputs["offline_metrics"]),
        "two_stage_execution": dict(execution_report),
        "inspection_ready": retention_policy == "retain_for_eda",
    }
    bundle_paths["summary"] = _write_json(
        retention_root / "retention_summary.json",
        summary_payload,
    )
    if retention_policy == "retain_for_eda":
        bundle_paths["clean_validation_traces"] = _write_trace_json(
            retention_root / "clean_validation_traces.json",
            artifact_inputs["clean_validation_traces"],
        )
        bundle_paths["synthetic_validation_traces"] = _write_trace_json(
            retention_root / "synthetic_validation_traces.json",
            artifact_inputs["synthetic_validation_traces"],
        )
        bundle_paths["test_traces"] = _write_trace_json(
            retention_root / "test_traces.json",
            artifact_inputs["test_traces"],
        )
        bundle_paths["clean_validation_scores"] = _write_score_npz(
            retention_root / "clean_validation_point_scores.npz",
            artifact_inputs["clean_validation"],
        )
        bundle_paths["synthetic_validation_scores"] = _write_score_npz(
            retention_root / "synthetic_validation_point_scores.npz",
            artifact_inputs["synthetic_validation"],
        )
        bundle_paths["test_scores"] = _write_score_npz(
            retention_root / "test_point_scores.npz",
            artifact_inputs["test"],
        )
        bundle_paths["offline_metrics"] = _write_json(
            retention_root / "offline_metrics.json",
            artifact_inputs["offline_metrics"],
        )
        bundle_paths["resolved_protocol"] = _write_json(
            retention_root / "resolved_protocol.json",
            protocol_config,
        )
    manifest_identity = {
        "entity_id": entity_id,
        "experiment_name": str(experiment_config.get("experiment_name")),
        "retention_policy": retention_policy,
    }
    manifest = build_retention_bundle_manifest(
        bundle_paths,
        manifest_identity,
        provenance={
            "checkpoint_sha256": checkpoint_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "protocol_config_path": protocol_config_path,
        },
        retention_policy=retention_policy,
        compression="none",
        export_scope="entity",
    )
    manifest_path = write_retention_bundle_manifest(
        retention_root / "retention_bundle_manifest.json",
        manifest,
    )
    bundle_paths["manifest"] = str(manifest_path)
    bundle_paths["checkpoint_sha256"] = (
        str(checkpoint_sha256) if checkpoint_sha256 else ""
    )
    bundle_paths["resolved_config_sha256"] = resolved_config_sha256
    bundle_paths["retention_root"] = str(retention_root)
    return bundle_paths
