from __future__ import annotations

r"""Evaluation loop that merges overlapping window scores back to entity timelines.

This file is part of the final evaluation step in the anomaly detection pipeline.
The model scores many short windows of time, and these window-level scores are
merged back to the original full timeline for each entity (for example, a
machine or a server). That way, we can measure how well the model found
anomalies on the real timeline instead of just on short fragments.

How the flow works in this codebase:

  data_loader -> batch of windows -> model.test_step -> point_scores
        \\___________________________________________/
                              merge back to full timeline

Cute diagram for the flow:

  (｡•́‿•̀｡)ﾉ  DATA SERIES
           \  [full entity timeline]
            v
        +----------------+      +------------------+      +------------------+
        | windowed input | ---> | model predicts   | ---> | merge window     |
        |  segment 1     |      | point_scores     |      | scores back into |
        +----------------+      +------------------+      | full timeline    |
            ^  ^  ^                                       +------------------+
            |  |  |                                         |
    overlapping windows                      final metrics   v
            |  |  |                                         |
        +----------------+      +------------------+      +------------------+
        | windowed input | ---> | model predicts   | ---> | compute metrics  |
        |  segment 2     |      | point_scores     |      | on covered points|
        +----------------+      +------------------+      +------------------+

The evaluator is intentionally model-agnostic. It only needs the model to
produce `point_scores` and `window_scores` for each batch of windows.
"""

from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_batch, summarize_tensor
from src.core.contracts import validate_evaluation_record
from src.data.split_protocol import describe_label_regime
from src.data.api import point_labels_to_window_labels
from src.metrics.pointwise import (
    compute_pointwise_curve_payload,
    compute_pointwise_metrics,
)
from src.engine.thresholding import (
    resolve_evaluation_threshold,
    # select_point_score_threshold,
)
from src.models.base_model import BaseModel
from src.protocols.reconstruction_scores import score_reconstruction


def _describe_benchmark_comparability(
    *,
    label_regime: str,
    is_truncated_evaluation: bool,
) -> tuple[str, str]:
    """Decide whether the results are comparable to benchmark evaluations.

    If the evaluation is truncated or the label regime is not the expected mixed
    benchmark format, then the results should be marked as non-comparable.
    """
    if is_truncated_evaluation:
        return "non_comparable", "truncated_smoke_evaluation"
    if label_regime != "mixed":
        return "non_comparable", "single_class_test_labels"
    return "benchmark_comparable", "benchmark_comparable_full_timeline"


def _build_entity_raw_point_labels(
    sequence_by_entity: dict[str, Any],
    fallback_dtype: torch.dtype,
) -> torch.Tensor:
    """Get the true labels for the full sequence, or use zeros if missing.

    In this codebase, each entity may have a full timeline stored in
    `sequence_by_entity`. If the raw labels are missing, we create a zero
    tensor so the evaluator still has a valid fallback.
    """
    raw_point_labels = sequence_by_entity.get("point_labels")
    if raw_point_labels is None:
        sequence_length = int(sequence_by_entity["x"].shape[0])
        return torch.zeros(sequence_length, dtype=fallback_dtype)
    return raw_point_labels.detach().cpu().clone().to(fallback_dtype)


def _validate_window_payload(
    batch_payload: dict[str, Any],
) -> tuple[Any, Any, Any, torch.Tensor | None, torch.Tensor | None]:
    batch_meta = batch_payload["meta"]
    point_scores = batch_payload["point_scores"]
    point_labels = batch_payload["point_labels"]
    if point_scores.ndim != 2 or point_labels.ndim != 2:
        raise ValueError("window scores and labels must have shape [B, L]")
    if point_scores.shape != point_labels.shape:
        raise ValueError("window scores and labels must share the same shape")
    if len(batch_meta) != point_scores.shape[0]:
        raise ValueError("window metadata length must match batch size")
    raw_scores = batch_payload.get("raw_input_point_mse")
    normalized_scores = batch_payload.get("normalized_input_point_mse")
    for name, scores in {
        "raw_input_point_mse": raw_scores,
        "normalized_input_point_mse": normalized_scores,
    }.items():
        if scores is not None and (
            not isinstance(scores, torch.Tensor) or scores.shape != point_scores.shape
        ):
            raise ValueError(f"{name} must have shape [B, L] matching point_scores")
    if (raw_scores is None) != (normalized_scores is None):
        raise ValueError("raw and normalized point scores must be provided together")
    return batch_meta, point_scores, point_labels, raw_scores, normalized_scores


def _initialize_entity_accumulators(
    *,
    entity_id: str,
    sequence_by_entity: dict[str, Any],
    point_labels: torch.Tensor,
    score_sums: dict[str, torch.Tensor],
    score_counts: dict[str, torch.Tensor],
    entity_labels: dict[str, torch.Tensor],
    raw_score_sums: dict[str, torch.Tensor] | None = None,
    normalized_score_sums: dict[str, torch.Tensor] | None = None,
) -> None:
    sequence_length = int(sequence_by_entity["x"].shape[0])
    score_sums[entity_id] = torch.zeros(sequence_length, dtype=torch.float32)
    score_counts[entity_id] = torch.zeros(sequence_length, dtype=torch.float32)
    entity_labels[entity_id] = _build_entity_raw_point_labels(
        sequence_by_entity, fallback_dtype=point_labels.dtype
    )
    if raw_score_sums is not None:
        raw_score_sums[entity_id] = torch.zeros(sequence_length, dtype=torch.float32)
    if normalized_score_sums is not None:
        normalized_score_sums[entity_id] = torch.zeros(
            sequence_length, dtype=torch.float32
        )


def accumulate_pointwise_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payload: dict[str, Any],
    entity_score_sums: dict[str, torch.Tensor],
    entity_score_counts: dict[str, torch.Tensor],
    entity_point_labels: dict[str, torch.Tensor],
    entity_raw_score_sums: dict[str, torch.Tensor] | None = None,
    entity_normalized_score_sums: dict[str, torch.Tensor] | None = None,
) -> None:
    """Add window-level scores back onto the full entity timeline.

    Each batch contains many overlapping windows from the same entity.
    This function keeps a running sum and count so overlapping points
    can later be averaged cleanly.
    """
    (
        batch_meta,
        point_scores,
        point_labels,
        raw_input_point_mse,
        normalized_input_point_mse,
    ) = _validate_window_payload(batch_payload)

    # For each window inside the batch, add scores back to the full entity timeline.

    for window_index, meta in enumerate(batch_meta):
        entity_id = meta["entity_id"]
        start_index = int(meta["start_index"])
        end_index = int(meta["end_index"])
        if entity_id not in entity_score_sums:
            _initialize_entity_accumulators(
                entity_id=entity_id,
                sequence_by_entity=sequences_by_entity[entity_id],
                point_labels=point_labels,
                score_sums=entity_score_sums,
                score_counts=entity_score_counts,
                entity_labels=entity_point_labels,
                raw_score_sums=entity_raw_score_sums,
                normalized_score_sums=entity_normalized_score_sums,
            )

        entity_score_sums[entity_id][start_index:end_index] += point_scores[
            window_index
        ]
        entity_score_counts[entity_id][start_index:end_index] += 1.0
        if raw_input_point_mse is not None and entity_raw_score_sums is not None:
            entity_raw_score_sums[entity_id][start_index:end_index] += (
                raw_input_point_mse[window_index]
            )
        if (
            normalized_input_point_mse is not None
            and entity_normalized_score_sums is not None
        ):
            entity_normalized_score_sums[entity_id][start_index:end_index] += (
                normalized_input_point_mse[window_index]
            )
        entity_point_labels[entity_id][start_index:end_index] = torch.maximum(
            entity_point_labels[entity_id][start_index:end_index],
            point_labels[window_index].to(entity_point_labels[entity_id].dtype),
        )


def reconstruct_pointwise_records_from_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn all window predictions into full entity evaluation records.

    This merges overlapping window scores for each entity and keeps a
    mask showing which time points were actually covered by at least one
    window.
    """
    entity_score_sums: dict[str, torch.Tensor] = {}
    entity_score_counts: dict[str, torch.Tensor] = {}
    entity_point_labels: dict[str, torch.Tensor] = {}
    has_raw_scores = any("raw_input_point_mse" in payload for payload in batch_payloads)
    entity_raw_score_sums = {} if has_raw_scores else None
    entity_normalized_score_sums = {} if has_raw_scores else None

    for batch_payload in batch_payloads:
        accumulate_pointwise_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payload=batch_payload,
            entity_score_sums=entity_score_sums,
            entity_score_counts=entity_score_counts,
            entity_point_labels=entity_point_labels,
            entity_raw_score_sums=entity_raw_score_sums,
            entity_normalized_score_sums=entity_normalized_score_sums,
        )

    evaluation_records: list[dict[str, Any]] = []
    for entity_id, score_sum in entity_score_sums.items():
        evaluation_record = _build_reconstructed_evaluation_record(
            entity_id=entity_id,
            score_sum=score_sum,
            raw_counts=entity_score_counts[entity_id],
            point_labels=entity_point_labels[entity_id],
            raw_score_sum=(
                None
                if entity_raw_score_sums is None
                else entity_raw_score_sums[entity_id]
            ),
            normalized_score_sum=(
                None
                if entity_normalized_score_sums is None
                else entity_normalized_score_sums[entity_id]
            ),
        )
        validate_evaluation_record(evaluation_record)
        evaluation_records.append(evaluation_record)

    return evaluation_records


def _build_reconstructed_evaluation_record(
    *,
    entity_id: str,
    score_sum: torch.Tensor,
    raw_counts: torch.Tensor,
    point_labels: torch.Tensor,
    raw_score_sum: torch.Tensor | None = None,
    normalized_score_sum: torch.Tensor | None = None,
) -> dict[str, Any]:
    averaged_scores = score_sum / torch.clamp(raw_counts, min=1.0)
    covered_indices = torch.nonzero(raw_counts > 0.0, as_tuple=False).reshape(-1)
    if covered_indices.numel() == 0:
        start_index, end_index, num_evaluated_points = 0, 0, 0
    else:
        start_index = int(covered_indices[0].item())
        end_index = int(covered_indices[-1].item()) + 1
        num_evaluated_points = int(covered_indices.numel())
    record = {
        "entity_id": entity_id,
        "point_scores": averaged_scores,
        "point_labels": point_labels,
        "covered_point_mask": raw_counts > 0.0,
        "num_points": int(averaged_scores.shape[0]),
        "evaluated_start_index": start_index,
        "evaluated_end_index": end_index,
        "evaluated_num_points": num_evaluated_points,
        "raw_num_points": int(averaged_scores.shape[0]),
    }
    if raw_score_sum is not None and normalized_score_sum is not None:
        record["raw_input_point_mse"] = raw_score_sum / torch.clamp(raw_counts, min=1.0)
        record["normalized_input_point_mse"] = normalized_score_sum / torch.clamp(
            raw_counts, min=1.0
        )
        record["score_space"] = "raw_input"
        record["point_score_transform"] = "identity"
    return record


def extract_covered_pointwise_arrays(
    evaluation_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collect only the actual covered points for metric calculation.

    Some parts of the entity timeline may not be inside any window, so
    we only evaluate on points that were scored by the model.
    """
    covered_point_scores: list[np.ndarray] = []
    covered_point_labels: list[np.ndarray] = []

    for evaluation_record in evaluation_records:
        point_scores = evaluation_record["point_scores"]
        point_labels = evaluation_record["point_labels"]
        covered_point_mask = evaluation_record.get("covered_point_mask")
        if covered_point_mask is None:
            covered_point_scores.append(point_scores.numpy())
            covered_point_labels.append(point_labels.numpy())
            continue
        covered_indices = covered_point_mask.to(dtype=torch.bool)
        covered_point_scores.append(point_scores[covered_indices].numpy())
        covered_point_labels.append(point_labels[covered_indices].numpy())

    if not covered_point_scores:
        raise ValueError("Cannot compute metrics from zero evaluation records")
    num_covered_points = sum(
        int(score_array.shape[0]) for score_array in covered_point_scores
    )
    if num_covered_points == 0:
        raise ValueError("Cannot compute metrics from zero covered evaluation points")

    return (
        np.concatenate(covered_point_scores, axis=0),
        np.concatenate(covered_point_labels, axis=0),
    )


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(inner) for inner in value]
    return value


def _extract_raw_reconstruction(step_output: dict[str, Any]) -> torch.Tensor:
    outputs = step_output["outputs"]
    stochastic_query = outputs["aux"].get("stochastic_query") or {}
    reconstruction_samples = stochastic_query.get("reconstruction_samples")
    if isinstance(reconstruction_samples, torch.Tensor):
        return reconstruction_samples
    reconstruction = outputs.get("recon")
    if not isinstance(reconstruction, torch.Tensor):
        raise ValueError("raw_input scoring requires reconstruction tensors")
    return reconstruction


def _build_window_score_records(
    *,
    batch_meta: list[dict[str, Any]],
    raw_window_scores: torch.Tensor,
    normalized_window_scores: torch.Tensor,
    point_labels: torch.Tensor,
    window_score_threshold: float | None,
) -> list[dict[str, Any]]:
    window_labels = point_labels_to_window_labels(point_labels)
    records = []
    for index, meta in enumerate(batch_meta):
        record = {
            "entity_id": str(meta["entity_id"]),
            "start_index": int(meta["start_index"]),
            "end_index": int(meta["end_index"]),
            "raw_input_window_mse": float(raw_window_scores[index]),
            "normalized_input_window_mse": float(normalized_window_scores[index]),
            "window_label": int(window_labels[index]),
            "score_space": "raw_input",
            "point_score_transform": "identity",
        }
        if window_score_threshold is not None:
            record["window_prediction"] = int(
                record["raw_input_window_mse"] > window_score_threshold
            )
        records.append(record)
    return records


class Evaluator:
    def __init__(
        self,
        device: str = "cpu",
        vus_max_buffer_size: int | None = None,
        vus_num_thresholds: int = 200,
    ) -> None:
        self.device = device
        self.vus_max_buffer_size = vus_max_buffer_size
        self.vus_num_thresholds = vus_num_thresholds

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Only tensors need device placement; metadata stays as plain Python
        # values so the aggregation code can read entity ids and indices easily.
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _run_model_on_batch(
        self,
        model: BaseModel,
        batch_index: int,
        batch_on_device: dict[str, Any],
        evaluation_stage: str = "test",
    ) -> dict[str, Any]:
        console_print(
            "EVAL",
            "Evaluating batch",
            batch_index=batch_index,
            **summarize_batch(batch_on_device),
        )

        # Tính anomaly score cho từng điểm dữ liệu (timestep)
        # trong từng window của batch.
        method_name = {
            "val_synth": "synthetic_validation_step",
            "val": "validation_step",
            "test": "test_step",
        }.get(evaluation_stage, f"{evaluation_stage}_step")
        step_method = getattr(model, method_name, None)
        if step_method is None:
            raise AttributeError(
                f"model does not expose evaluation stage method: {method_name}"
            )
        return step_method(batch_on_device)

    @staticmethod
    def _remember_forward_pass_seconds(
        step_output: dict[str, Any],
        forward_pass_seconds_history: list[float],
    ) -> None:
        output_aux = step_output["outputs"]["aux"]
        if "forward_pass_seconds" in output_aux:
            forward_pass_seconds_history.append(
                float(output_aux["forward_pass_seconds"])
            )

    @staticmethod
    def _log_batch_outputs(
        batch_index: int,
        step_output: dict[str, Any],
        point_scores: torch.Tensor,
    ) -> None:
        uncertainty = step_output["outputs"]["aux"].get("uncertainty")
        uncertainty_summary = {}
        if uncertainty is not None:
            uncertainty_summary = {
                "point_score_variance_mean": float(
                    uncertainty["point_anomaly_score_variance"].mean().detach().cpu()
                ),
                "window_score_variance_mean": float(
                    uncertainty["window_anomaly_score_variance"].mean().detach().cpu()
                ),
                "reconstruction_variance_mean": float(
                    uncertainty["reconstruction_variance_full"].mean().detach().cpu()
                ),
            }
        console_print(
            "EVAL",
            "Produced evaluation batch outputs",
            batch_index=batch_index,
            point_scores=summarize_tensor(point_scores),
            window_scores=summarize_tensor(step_output["outputs"]["window_scores"]),
            **uncertainty_summary,
        )

    @staticmethod
    def _detach_uncertainty_to_cpu(
        uncertainty: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if uncertainty is None:
            return None
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in uncertainty.items()
        }

    @staticmethod
    def _build_trace_payload(
        *,
        batch_index: int,
        batch_meta: list[dict[str, Any]],
        step_output: dict[str, Any],
        point_scores: torch.Tensor,
    ) -> dict[str, Any]:
        stochastic_query = step_output["outputs"]["aux"].get("stochastic_query") or {}
        return {
            "batch_index": batch_index,
            "entity_ids": [meta["entity_id"] for meta in batch_meta],
            "point_score_summary": summarize_tensor(point_scores),
            "window_score_summary": summarize_tensor(
                step_output["outputs"]["window_scores"]
            ),
            "point_score_history": _json_safe_value(point_scores),
            "window_score_history": _json_safe_value(
                step_output["outputs"]["window_scores"]
            ),
            "uncertainty_history": _json_safe_value(
                step_output["outputs"]["aux"].get("uncertainty")
            ),
            "deterministic_geometry": _json_safe_value(
                step_output["outputs"]["aux"].get("deterministic_geometry")
            ),
            "stochastic_query": _json_safe_value(stochastic_query),
            "sample_retention_policy": stochastic_query.get("sample_retention_policy"),
            "mc_sample_histories": {
                "point_score_samples": _json_safe_value(
                    stochastic_query.get("point_score_samples")
                ),
                "window_score_samples": _json_safe_value(
                    stochastic_query.get("window_score_samples")
                ),
                "reconstruction_samples": _json_safe_value(
                    stochastic_query.get("reconstruction_samples")
                ),
                "classification_probability_samples": _json_safe_value(
                    stochastic_query.get("classification_probability_samples")
                ),
            },
        }

    @staticmethod
    def _build_sequences_by_entity(data_loader: Any) -> dict[str, dict[str, Any]]:
        return {
            sequence["meta"]["entity_id"]: sequence
            for sequence in data_loader.dataset.sequences
        }

    def evaluate(
        self,
        model: BaseModel,
        data_loader: Any,
        point_score_threshold: float | None = None,
        threshold_source: str | None = None,
        *,
        score_space: str = "model_output",
        scaler: Any | None = None,
        window_score_threshold: float | None = None,
        evaluation_stage: str = "test",
    ) -> dict[str, Any]:
        if score_space not in {"model_output", "raw_input"}:
            raise ValueError("score_space must be model_output or raw_input")
        if score_space == "raw_input" and scaler is None:
            raise ValueError("raw_input scoring requires a fitted scaler")
        # Window-level scores are accumulated back onto each entity because the
        # downstream metrics should be interpreted on the original timeline.
        model.to(self.device)
        model.eval()
        console_print(
            "EVAL",
            "Starting evaluation",
            device=self.device,
            num_batches=len(data_loader),
        )
        forward_pass_seconds_history: list[float] = []
        sequences_by_entity = self._build_sequences_by_entity(data_loader)
        pointwise_batch_payloads: list[dict[str, Any]] = []
        trace_payloads: list[dict[str, Any]] = []
        window_records: list[dict[str, Any]] = []

        with torch.no_grad():
            # Với mỗi batch dữ liệu đọc được từ data_loader,
            for batch_index, batch in enumerate(data_loader, start=1):
                batch_on_device = self._move_batch_to_device(batch)
                step_output = self._run_model_on_batch(
                    model=model,
                    batch_index=batch_index,
                    batch_on_device=batch_on_device,
                    evaluation_stage=evaluation_stage,
                )
                scoring_batch = step_output.get("batch", batch_on_device)
                point_labels = scoring_batch.get("point_labels")
                synthetic_labels = scoring_batch.get("synthetic_anomaly_mask")
                if evaluation_stage == "val_synth" and synthetic_labels is not None:
                    point_labels = synthetic_labels
                    if point_labels.ndim == 3:
                        point_labels = point_labels.any(dim=-1).long()
                if not isinstance(point_labels, torch.Tensor):
                    raise ValueError("evaluation stage must provide point_labels")
                if score_space == "raw_input":
                    scores = score_reconstruction(
                        scoring_batch["x"],
                        _extract_raw_reconstruction(step_output),
                        scaler,
                    )
                    point_scores = scores["raw_input_point_mse"].detach().cpu()
                    normalized_point_scores = (
                        scores["normalized_input_point_mse"].detach().cpu()
                    )
                    raw_window_scores = scores["raw_input_window_mse"].detach().cpu()
                    normalized_window_scores = (
                        scores["normalized_input_window_mse"].detach().cpu()
                    )
                    window_records.extend(
                        _build_window_score_records(
                            batch_meta=batch["meta"],
                            raw_window_scores=raw_window_scores,
                            normalized_window_scores=normalized_window_scores,
                            point_labels=point_labels.detach().cpu(),
                            window_score_threshold=window_score_threshold,
                        )
                    )
                else:
                    point_scores = step_output["outputs"]["point_scores"].detach().cpu()
                    normalized_point_scores = None
                self._remember_forward_pass_seconds(
                    step_output=step_output,
                    forward_pass_seconds_history=forward_pass_seconds_history,
                )
                self._log_batch_outputs(
                    batch_index=batch_index,
                    step_output=step_output,
                    point_scores=point_scores,
                )
                trace_payloads.append(
                    self._build_trace_payload(
                        batch_index=batch_index,
                        batch_meta=batch["meta"],
                        step_output=step_output,
                        point_scores=point_scores,
                    )
                )
                batch_payload = {
                    "meta": batch["meta"],
                    "point_scores": point_scores,
                    "point_labels": point_labels.detach().cpu(),
                    "uncertainty": self._detach_uncertainty_to_cpu(
                        step_output["outputs"]["aux"].get("uncertainty")
                    ),
                }
                if normalized_point_scores is not None:
                    batch_payload["raw_input_point_mse"] = point_scores
                    batch_payload["normalized_input_point_mse"] = (
                        normalized_point_scores
                    )
                pointwise_batch_payloads.append(batch_payload)

        # After the loop, we have all window-level scores for every entity.
        # Next we reconstruct the full entity timelines from those windows.
        evaluation_records = reconstruct_pointwise_records_from_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payloads=pointwise_batch_payloads,
        )
        # Metric computation should use only the timesteps that were actually
        # covered by at least one evaluation window. Full-length raw labels stay
        # on each record for audit and visualization.
        concatenated_scores, concatenated_labels = extract_covered_pointwise_arrays(
            evaluation_records
        )

        threshold, resolved_threshold_source = resolve_evaluation_threshold(
            concatenated_scores,
            point_score_threshold=point_score_threshold,
            threshold_source=threshold_source,
            quantile=0.99,
        )
        if score_space == "raw_input":
            for record in evaluation_records:
                record["point_predictions"] = (
                    record["raw_input_point_mse"] > threshold
                ).long()
                entity_windows = [
                    item
                    for item in window_records
                    if item["entity_id"] == record["entity_id"]
                ]
                record["window_labels"] = torch.tensor(
                    [item["window_label"] for item in entity_windows],
                    dtype=torch.long,
                )
                record["window_predictions"] = torch.tensor(
                    [item.get("window_prediction", 0) for item in entity_windows],
                    dtype=torch.long,
                )

        # Tính toán các độ đo pointwise (pointwise metric)
        metrics = compute_pointwise_metrics(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
            threshold=threshold,
            vus_max_buffer_size=self.vus_max_buffer_size,
            vus_num_thresholds=self.vus_num_thresholds,
        )

        metrics["threshold"] = threshold
        metrics["raw_num_points"] = float(
            sum(int(record["raw_num_points"]) for record in evaluation_records)
        )
        metrics["evaluated_num_points"] = float(
            sum(int(record["evaluated_num_points"]) for record in evaluation_records)
        )
        metrics["num_entities_evaluated"] = float(len(evaluation_records))
        metrics["is_truncated_evaluation"] = float(
            1.0
            if any(
                int(record["evaluated_num_points"]) < int(record["raw_num_points"])
                for record in evaluation_records
            )
            else 0.0
        )
        label_regime = describe_label_regime(concatenated_labels)
        benchmark_comparability, protocol_status = _describe_benchmark_comparability(
            label_regime=label_regime,
            is_truncated_evaluation=bool(metrics["is_truncated_evaluation"]),
        )
        metrics["label_regime"] = label_regime
        metrics["benchmark_comparability"] = benchmark_comparability
        metrics["protocol_status"] = protocol_status
        metrics["threshold_source"] = resolved_threshold_source
        if score_space == "raw_input":
            metrics["score_space"] = "raw_input"
            metrics["point_score_transform"] = "identity"
            if window_score_threshold is not None:
                metrics["window_threshold"] = float(window_score_threshold)
        if forward_pass_seconds_history:
            metrics["forward_pass_seconds_mean"] = sum(
                forward_pass_seconds_history
            ) / len(forward_pass_seconds_history)
        curves = compute_pointwise_curve_payload(
            point_labels=concatenated_labels,
            point_scores=concatenated_scores,
        )
        console_print(
            "EVAL",
            "Completed evaluation",
            num_records=len(evaluation_records),
            concatenated_scores_length=len(concatenated_scores),
            concatenated_labels_length=len(concatenated_labels),
            threshold=threshold,
            metrics=metrics,
            roc_curve_points=len(curves["roc_curve"]["x"]),
            pr_curve_points=len(curves["pr_curve"]["x"]),
        )

        return {
            "metrics": metrics,
            "records": evaluation_records,
            "curves": curves,
            "traces": trace_payloads,
            "window_records": window_records,
        }
