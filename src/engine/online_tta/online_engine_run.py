from __future__ import annotations

"""End-to-end online benchmark execution for THESIS."""

from pathlib import Path
from typing import Any

from src.core.artifact_integrity import (
    build_artifact_manifest,
    verify_artifact_manifest,
    sha256_file,
    write_artifact_manifest,
)
from src.core.registry import build_dataset
from src.core.runtime_components import register_online_runtime_components
from src.core.seed import seed_everything
from src.engine.checkpoint import CheckpointManager
from src.engine.online_tta.non_overlap_guard import NonOverlapGuard
from src.engine.online_tta.online_calibration import (
    build_online_stream as _build_online_stream,
)
from src.engine.online_tta.online_engine_shared import (
    _build_model_from_experiment_config,
    _build_optimizer_from_experiment_config,
    _build_threshold_artifact_from_scores,
    _sync_online_runtime_state,
    _utc_now_iso,
    _validate_single_window_online_batch,
    _write_json,
    calibrate_entity_threshold_artifacts,
)
from src.engine.online_tta.online_engine_window_core import (
    _build_triage_thresholds,
    _process_online_window,
)
from src.engine.online_tta.signature_verification import (
    SignatureWindow,
    find_recurrent_signatures,
)
from src.engine.online_tta.timing_debug import OnlineTtaTimingLogger
from src.engine.online_tta.verification_buffer import VerificationBuffer
from src.engine.online_tta.verification_cycle import VerificationCycleController
from src.engine.online_tta.runtime_state import build_online_runtime_state
from src.engine.online_tta.online_optimizer import (
    assert_only_projector_is_trainable,
)
from src.protocols.threshold_artifact import write_threshold_artifact


def _resolve_max_online_steps(value: Any) -> int | None:
    if value is None:
        return None
    resolved_value = int(value)
    if resolved_value <= 0:
        return None
    return resolved_value


def _select_online_stream_sequence(
    sequence: dict[str, Any],
    *,
    absolute_start_index: int | None,
    absolute_end_index: int | None,
) -> dict[str, Any]:
    if absolute_start_index is None and absolute_end_index is None:
        return sequence
    if absolute_start_index is None or absolute_end_index is None:
        raise ValueError(
            "absolute_start_index and absolute_end_index must be set together"
        )

    source_length = int(sequence["x"].shape[0])
    if not 0 <= absolute_start_index < absolute_end_index <= source_length:
        raise ValueError(
            "Online stream range must satisfy "
            f"0 <= start < end <= {source_length}, got "
            f"[{absolute_start_index}, {absolute_end_index})"
        )

    selected_sequence = dict(sequence)

    # TODO: Giải thích vòng for này. ==> Tối ưu thêm.
    for field_name in ("x", "point_labels", "mask", "timestamps"):
        value = sequence.get(field_name)
        selected_sequence[field_name] = (
            None
            if value is None
            else value[absolute_start_index:absolute_end_index].clone()
        )

    selected_sequence["meta"] = {
        **sequence["meta"],
        "sequence_length": absolute_end_index - absolute_start_index,
        "source_sequence_length": source_length,
        "absolute_start_index": absolute_start_index,
        "absolute_end_index": absolute_end_index,
    }

    return selected_sequence


def _build_dry_run_online_context(*, online_variant: str) -> dict[str, Any]:
    return {
        "benchmark_status": "dry_run",
        "created_at_utc": _utc_now_iso(),
        "online_variant": online_variant,
        "threshold_artifact": None,
        "threshold_artifact_path": None,
        "final_checkpoint_path": None,
        "metric_history": [],
        "records": [],
        "online_metrics_path": None,
        "online_records_path": None,
    }


def _persist_threshold_artifacts(
    threshold_artifacts: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    threshold_paths: dict[str, str] = {}
    for entity_id, artifact in threshold_artifacts.items():
        path = output_dir / "thresholds" / entity_id / "online_thresholds.json"
        write_threshold_artifact(artifact, path)
        threshold_paths[entity_id] = str(path)
    return threshold_paths


def _build_runtime_online_context(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
) -> dict[str, Any]:
    data_bundle = build_dataset(
        experiment_config["data"]["dataset_name"], experiment_config["data"]
    )
    model = _build_model_from_experiment_config(experiment_config)
    optimizer = _build_optimizer_from_experiment_config(model, experiment_config)

    assert_only_projector_is_trainable(model)

    # Calibration happens before the streaming loop, so the model must already
    # live on the target device here; otherwise validation windows reach CUDA
    # tensors while the encoder weights still sit on CPU.
    model.to(str(experiment_config["device"]))

    checkpoint_manager = CheckpointManager(
        Path(str(experiment_config["checkpoint_dir"]))
    )
    reference_checkpoint_path = str(
        experiment_config.get("task", {}).get("reference_checkpoint_path", "")
    )
    reference_checkpoint_sha256 = None
    if reference_checkpoint_path and Path(reference_checkpoint_path).is_file():
        reference_checkpoint_sha256 = sha256_file(reference_checkpoint_path)

    threshold_artifacts = calibrate_entity_threshold_artifacts(
        model=model,
        clean_validation_sequences=data_bundle["scaled_sequences"]["val"],
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        device=str(experiment_config["device"]),
        checkpoint_sha256=reference_checkpoint_sha256,
    )
    output_dir = Path(str(experiment_config["output_dir"]))
    threshold_paths = _persist_threshold_artifacts(threshold_artifacts, output_dir)
    first_entity = next(iter(threshold_artifacts))
    threshold_artifact = threshold_artifacts[first_entity]
    threshold_path = threshold_paths[first_entity]

    batch_size = int(experiment_config["data"]["batch_size"])
    if batch_size != 1:
        raise ValueError(
            "online benchmark startup expects batch_size=1 so each step is one causal window"
        )

    runtime_state = build_online_runtime_state(
        entity_id=str(threshold_artifact["entity_id"]),
        online_variant=online_variant,
        threshold_artifact=threshold_artifact,
        checkpoint_path=str(checkpoint_manager.checkpoint_dir / "online_final.pt"),
        threshold_artifact_path=str(threshold_path),
    )

    # TODO: Có hơi nhiều field quá không?
    return {
        "benchmark_status": "completed",
        "created_at_utc": _utc_now_iso(),
        "online_variant": online_variant,
        "data_bundle": data_bundle,
        "model": model,
        "optimizer": optimizer,
        "checkpoint_manager": checkpoint_manager,
        "threshold_artifact": threshold_artifact,
        "threshold_artifacts": threshold_artifacts,
        "threshold_paths": threshold_paths,
        "threshold_artifact_path": str(threshold_path),
        "reference_checkpoint_path": reference_checkpoint_path,
        "reference_checkpoint_sha256": reference_checkpoint_sha256,
        "threshold_value": float(
            threshold_artifact["thresholds"]["online_ewma_point"]["value"]
        ),
        "runtime_state": runtime_state,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "view_noise_std": float(experiment_config["task"].get("view_noise_std", 0.0)),
        "view_dropout_probability": float(
            experiment_config["task"].get("view_dropout_probability", 0.0)
        ),
        "device": str(experiment_config["device"]),
        "max_online_steps": _resolve_max_online_steps(
            experiment_config["task"].get("max_online_steps")
        ),
        "debug_timing": bool(experiment_config["task"].get("debug_timing", False)),
        "verification_buffer": VerificationBuffer(max_size=64, non_overlap_gap=0),
        "hard_old_guard": NonOverlapGuard(max_size=1),
        "signature_history": [],
    }


def _run_online_sequence(
    *,
    model,
    optimizer,
    sequence: dict[str, Any],
    online_variant: str,
    threshold_value: float,
    protocol_config: dict[str, Any],
    batch_size: int,
    view_noise_std: float,
    view_dropout_probability: float,
    device: str,
    verification_buffer: VerificationBuffer,
    runtime_state=None,
    max_online_steps: int | None,
    hard_old_guard: NonOverlapGuard | None = None,
    signature_history: list[SignatureWindow] | None = None,
    threshold_artifact: dict[str, Any] | None = None,
    timing_logger: OnlineTtaTimingLogger | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if hard_old_guard is None:
        hard_old_guard = NonOverlapGuard(max_size=1)
    if signature_history is None:
        signature_history = []
    if runtime_state is None:
        runtime_state = build_online_runtime_state(
            entity_id=str(sequence["meta"]["entity_id"]),
            online_variant=online_variant,
            threshold_artifact={
                "entity_id": str(sequence["meta"]["entity_id"]),
                "thresholds": {},
            },
        )
    from src.engine.online_tta import online_engine as public_online_engine

    batcher = public_online_engine._build_online_stream(
        sequences=[sequence],
        window_size=int(protocol_config.get("window_size", 1)),
        batch_size=batch_size,
        view_noise_std=view_noise_std,
        view_dropout_probability=view_dropout_probability,
    )
    metric_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    previous_ewma_score: float | None = None
    verification_controller = VerificationCycleController(verification_buffer)
    ewma_current_weight = float(protocol_config["online_ewma_current_weight"])
    ewma_previous_weight = float(protocol_config["online_ewma_previous_weight"])
    triage_thresholds = _build_triage_thresholds(threshold_value, threshold_artifact)
    if timing_logger is None:
        timing_logger = OnlineTtaTimingLogger(enabled=False, device=device)

    for batch in batcher:
        _validate_single_window_online_batch(batch)
        if max_online_steps is not None and len(metric_history) >= max_online_steps:
            break
        from src.engine.online_tta import online_engine as public_online_engine

        timing_logger.set_window(batch)
        
        previous_ewma_score, metric, record = (
            public_online_engine._process_online_window(
                model=model,
                optimizer=optimizer,
                batch=batch,
                online_variant=online_variant,
                threshold_value=threshold_value,
                ewma_current_weight=ewma_current_weight,
                ewma_previous_weight=ewma_previous_weight,
                triage_thresholds=triage_thresholds,
                verification_buffer=verification_buffer,
                previous_ewma_score=previous_ewma_score,
                device=device,
                signature_history=signature_history,
                verification_controller=verification_controller,
                hard_old_guard=hard_old_guard,
                timing_logger=timing_logger,
            )
        )
        metric["online/step"] = len(metric_history) + 1
        _sync_online_runtime_state(
            runtime_state=runtime_state,
            previous_ewma_score=previous_ewma_score,
            signature_history=signature_history,
            recurrent_signatures=find_recurrent_signatures(signature_history),
            record=record,
            hard_old_guard=hard_old_guard,
            verification_buffer=verification_buffer,
        )
        records.append(record)
        metric_history.append(metric)

    return metric_history, records


def _build_online_execution_context(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return _build_dry_run_online_context(online_variant=online_variant)
    return _build_runtime_online_context(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
    )


def _run_online_execution_sequences(
    *,
    context: dict[str, Any],
    protocol_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    max_online_steps_limit = _resolve_max_online_steps(context.get("max_online_steps"))
    for sequence in context["data_bundle"]["scaled_sequences"]["test"]:
        if (
            max_online_steps_limit is not None
            and len(metric_history) >= max_online_steps_limit
        ):
            break
        entity_id = str(sequence["meta"]["entity_id"])
        artifact = context["threshold_artifacts"].get(entity_id)
        if artifact is None:
            raise KeyError(f"No threshold artifact for test entity {entity_id}")
        from src.engine.online_tta import online_engine as public_online_engine

        sequence_metric_history, sequence_records = (
            public_online_engine._run_online_sequence(
                model=context["model"],
                optimizer=context["optimizer"],
                sequence=sequence,
                online_variant=context["online_variant"],
                threshold_value=float(
                    artifact["thresholds"]["online_ewma_point"]["value"]
                ),
                protocol_config=protocol_config,
                batch_size=context["batch_size"],
                view_noise_std=context["view_noise_std"],
                view_dropout_probability=context["view_dropout_probability"],
                device=context["device"],
                verification_buffer=context["verification_buffer"],
                runtime_state=context.get("runtime_state"),
                hard_old_guard=context["hard_old_guard"],
                signature_history=context["signature_history"],
                max_online_steps=(
                    None
                    if max_online_steps_limit is None
                    else max_online_steps_limit - len(metric_history)
                ),
                threshold_artifact=artifact,
                timing_logger=OnlineTtaTimingLogger(
                    enabled=bool(context.get("debug_timing", False)),
                    device=context["device"],
                ),
            )
        )
        metric_history.extend(sequence_metric_history)
        records.extend(sequence_records)
    return metric_history, records


def _finalize_online_execution(
    *,
    context: dict[str, Any],
    experiment_config: dict[str, Any],
    metric_history: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = context["output_dir"]
    window_size = int(experiment_config["data"]["window_size"])
    expected_windows = sum(
        max(0, int(sequence["x"].shape[0]) - window_size + 1)
        for sequence in context["data_bundle"]["scaled_sequences"]["test"]
    )
    smoke_limit = context.get("max_online_steps")
    expected_processed = (
        min(expected_windows, int(smoke_limit))
        if smoke_limit is not None and int(smoke_limit) > 0
        else expected_windows
    )
    coverage_status = "complete" if len(records) == expected_processed else "incomplete"
    metrics_path = _write_json(output_dir / "online_metrics.json", metric_history)
    records_path = _write_json(output_dir / "online_records.json", records)
    final_checkpoint_path = context["checkpoint_manager"].save_checkpoint(
        checkpoint_name="online_final.pt",
        model=context["model"],
        optimizer=None,
        scheduler=None,
        scaler_state=context["data_bundle"]["scaler"].state_dict(),
        config=experiment_config,
        epoch=len(metric_history),
        metric_history=metric_history,
        extra_state={
            "threshold_artifact": context["threshold_artifact"],
            "threshold_artifact_path": context["threshold_artifact_path"],
            "online_variant": context["online_variant"],
            "stream_cursor": context["runtime_state"].stream_cursor,
            "previous_ewma_score": context["runtime_state"].previous_ewma_score,
            "signature_history": context["runtime_state"].signature_history,
            "recurrent_signatures": context["runtime_state"].recurrent_signatures,
            "verification_buffer_size": len(context["verification_buffer"]),
            "verification_buffer_entries": context["verification_buffer"].items(),
            "verification_history": context["runtime_state"].verification_history,
            "hard_old_guard_intervals": context["hard_old_guard"].intervals(),
            "online_runtime_state": context["runtime_state"].to_dict(),
        },
    )
    artifact_identity = {
        "entity_id": str(context["threshold_artifact"]["entity_id"]),
        "online_variant": str(context["online_variant"]),
        "experiment_name": str(experiment_config["experiment_name"]),
    }
    artifact_manifest = build_artifact_manifest(
        {
            "checkpoint": final_checkpoint_path,
            "metrics": metrics_path,
            "records": records_path,
            "threshold": context["threshold_artifact_path"],
        },
        identity=artifact_identity,
        provenance={
            "threshold_artifact_path": context["threshold_artifact_path"],
            "threshold_artifact_sha256": CheckpointManager._stable_json_digest(
                context["threshold_artifact"]
            ),
            "resolved_experiment_config_sha256": CheckpointManager._stable_json_digest(
                experiment_config
            ),
            "online_variant": context["online_variant"],
            "reference_checkpoint_path": context["reference_checkpoint_path"],
            "reference_checkpoint_sha256": context["reference_checkpoint_sha256"],
        },
    )
    artifact_manifest_path = write_artifact_manifest(
        output_dir / "online_artifact_manifest.json", artifact_manifest
    )
    expected_provenance = artifact_manifest.get("provenance")
    artifact_integrity_status = (
        "verified"
        if verify_artifact_manifest(
            artifact_manifest,
            artifact_identity,
            expected_provenance=expected_provenance,
        )
        else "failed"
    )
    execution_complete = (
        coverage_status == "complete" and artifact_integrity_status == "verified"
    )
    return {
        "benchmark_status": "completed" if execution_complete else "failed",
        "experiment_status": "complete" if execution_complete else "incomplete",
        "matrix_status": "matrix_ready",
        "runtime_protocol_status": "full_spec_v2",
        "stream_coverage_status": coverage_status,
        "artifact_integrity_status": artifact_integrity_status,
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_path": str(artifact_manifest_path),
        "metric_availability_status": "recorded",
        "expected_windows": expected_windows,
        "processed_windows": len(records),
        "created_at_utc": context["created_at_utc"],
        "online_variant": context["online_variant"],
        "threshold_artifact": context["threshold_artifact"],
        "threshold_artifact_path": context["threshold_artifact_path"],
        "final_checkpoint_path": str(final_checkpoint_path),
        "metric_history": metric_history,
        "records": records,
        "online_metrics_path": metrics_path,
        "online_records_path": records_path,
        "threshold_source": "clean_validation_stride1_ewma",
    }


def run_thesis_online_tta_experiment(
    *,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    online_variant: str,
    dry_run: bool,
) -> dict[str, Any]:

    # TODO: Làm sao để không cần gán cứng tên biến thể của thí nghiệm nữa?
    # online hay offline variant là các biến thể.
    if online_variant not in {"A0", "A1", "A2"}:
        raise ValueError("online_variant must be one of: A0, A1, A2")

    seed_everything(int(experiment_config["seed"]))

    register_online_runtime_components()
    context = _build_online_execution_context(
        experiment_config=experiment_config,
        protocol_config=protocol_config,
        online_variant=online_variant,
        dry_run=dry_run,
    )
    if context["benchmark_status"] == "dry_run":
        return context

    # Chọn đoạn con của test series mà thí nghiệm online
    # sẽ stream trên đó.
    # Thường là đoạn có chứa anomaly spans.
    sequence = _select_online_stream_sequence(
        context["data_bundle"]["scaled_sequences"]["test"][0],
        absolute_start_index=experiment_config["task"].get("absolute_start_index"),
        absolute_end_index=experiment_config["task"].get("absolute_end_index"),
    )

    context["data_bundle"]["scaled_sequences"]["test"] = [sequence]

    # TODO: Đơn giản hoá lời gọi hàm này lại.
    # Số lượng argument đang quá nhiều.
    metric_history, records = _run_online_sequence(
        # Mô hình và cơ chế cập nhật tham số.
        model=context["model"],
        optimizer=context["optimizer"],

        # Chuỗi dữ liệu đầu vào và cách tạo batch.
        sequence=sequence,
        batch_size=context["batch_size"],

        # Biến thể, giao thức và ngưỡng quyết định online.
        online_variant=context["online_variant"],
        threshold_value=context["threshold_value"],
        threshold_artifact=context["threshold_artifact"],
        protocol_config=protocol_config,

        # Cấu hình tạo hai view của mỗi cửa sổ.
        view_noise_std=context["view_noise_std"],
        view_dropout_probability=context["view_dropout_probability"],

        # Trạng thái có thể thay đổi trong quá trình chạy online.
        verification_buffer=context["verification_buffer"],
        runtime_state=context.get("runtime_state"),
        hard_old_guard=context["hard_old_guard"],
        signature_history=context["signature_history"],

        # Môi trường, giới hạn thực thi và đo thời gian.
        device=context["device"],
        max_online_steps=_resolve_max_online_steps(context.get("max_online_steps")),
        timing_logger=OnlineTtaTimingLogger(
            enabled=bool(context["debug_timing"]),
            device=context["device"],
        ),
    )

    return _finalize_online_execution(
        context=context,
        experiment_config=experiment_config,
        metric_history=metric_history,
        records=records,
    )