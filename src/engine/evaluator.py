from __future__ import annotations

"""Evaluation loop that merges overlapping window scores back to entity timelines.

A new reader should pair this file with the batch and output contracts. The
evaluator relies on the fixed `point_scores` field so it can stay model-agnostic.
"""

from typing import Any

import numpy as np
import torch

from src.core.console import console_print, summarize_batch, summarize_tensor
from src.core.contracts import validate_evaluation_record
from src.data.split_protocol import describe_label_regime
from src.metrics.pointwise import (
    compute_pointwise_curve_payload,
    compute_pointwise_metrics,
)
from src.models.base_model import BaseModel


def select_point_score_threshold(
    point_scores: np.ndarray, quantile: float = 0.95
) -> float:
    """
    Hàm này có nhiệm vụ chọn threshold
    để biến anomaly score liên tục
    thành dự đoán nhị phân.
    """

    # The smoke runs can produce many exact zeros, so selecting a threshold from
    # only the positive support avoids the "everything is anomalous" failure
    # mode in plots and thresholded metrics.
    positive_scores = point_scores[point_scores > 0.0]
    reference_scores = positive_scores if positive_scores.size > 0 else point_scores
    threshold = float(np.quantile(reference_scores, quantile))

    if threshold <= 0.0 and positive_scores.size > 0:
        threshold = float(np.min(positive_scores))
    return threshold


def _describe_benchmark_comparability(
    *,
    label_regime: str,
    is_truncated_evaluation: bool,
) -> tuple[str, str]:
    if is_truncated_evaluation:
        return "non_comparable", "truncated_smoke_evaluation"
    if label_regime != "mixed":
        return "non_comparable", "single_class_test_labels"
    return "benchmark_comparable", "benchmark_comparable_full_timeline"


def _build_entity_raw_point_labels(
    sequence_by_entity: dict[str, Any],
    fallback_dtype: torch.dtype,
) -> torch.Tensor:
    raw_point_labels = sequence_by_entity.get("point_labels")
    if raw_point_labels is None:
        sequence_length = int(sequence_by_entity["x"].shape[0])
        return torch.zeros(sequence_length, dtype=fallback_dtype)
    return raw_point_labels.detach().cpu().clone().to(fallback_dtype)


def accumulate_pointwise_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payload: dict[str, Any],
    entity_score_sums: dict[str, torch.Tensor],
    entity_score_counts: dict[str, torch.Tensor],
    entity_point_labels: dict[str, torch.Tensor],
) -> None:
    batch_meta = batch_payload["meta"]
    point_scores = batch_payload["point_scores"]
    point_labels = batch_payload["point_labels"]

    if point_scores.ndim != 2:
        raise ValueError("batch_payload['point_scores'] must have shape [B, L]")
    if point_labels.ndim != 2:
        raise ValueError("batch_payload['point_labels'] must have shape [B, L]")
    if point_scores.shape != point_labels.shape:
        raise ValueError(
            "batch_payload['point_scores'] and batch_payload['point_labels'] must share the same shape"
        )
    if len(batch_meta) != point_scores.shape[0]:
        raise ValueError("batch_payload['meta'] length must match batch size")

    for window_index, meta in enumerate(batch_meta):
        entity_id = meta["entity_id"]
        start_index = int(meta["start_index"])
        end_index = int(meta["end_index"])
        sequence_length = int(sequences_by_entity[entity_id]["x"].shape[0])

        if entity_id not in entity_score_sums:
            entity_score_sums[entity_id] = torch.zeros(
                sequence_length,
                dtype=torch.float32,
            )
            entity_score_counts[entity_id] = torch.zeros(
                sequence_length,
                dtype=torch.float32,
            )
            entity_point_labels[entity_id] = _build_entity_raw_point_labels(
                sequences_by_entity[entity_id],
                fallback_dtype=point_labels.dtype,
            )

        entity_score_sums[entity_id][start_index:end_index] += point_scores[
            window_index
        ]
        entity_score_counts[entity_id][start_index:end_index] += 1.0
        entity_point_labels[entity_id][start_index:end_index] = torch.maximum(
            entity_point_labels[entity_id][start_index:end_index],
            point_labels[window_index].to(entity_point_labels[entity_id].dtype),
        )


def reconstruct_pointwise_records_from_window_payload(
    *,
    sequences_by_entity: dict[str, dict[str, Any]],
    batch_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_score_sums: dict[str, torch.Tensor] = {}
    entity_score_counts: dict[str, torch.Tensor] = {}
    entity_point_labels: dict[str, torch.Tensor] = {}

    for batch_payload in batch_payloads:
        accumulate_pointwise_window_payload(
            sequences_by_entity=sequences_by_entity,
            batch_payload=batch_payload,
            entity_score_sums=entity_score_sums,
            entity_score_counts=entity_score_counts,
            entity_point_labels=entity_point_labels,
        )

    evaluation_records: list[dict[str, Any]] = []
    for entity_id, score_sum in entity_score_sums.items():
        raw_counts = entity_score_counts[entity_id]
        counts = torch.clamp(raw_counts, min=1.0)
        averaged_scores = score_sum / counts
        covered_indices = torch.nonzero(raw_counts > 0.0, as_tuple=False).reshape(-1)
        if covered_indices.numel() == 0:
            evaluated_start_index = 0
            evaluated_end_index = 0
            evaluated_num_points = 0
        else:
            evaluated_start_index = int(covered_indices[0].item())
            evaluated_end_index = int(covered_indices[-1].item()) + 1
            evaluated_num_points = int(covered_indices.numel())
        evaluation_record = {
            "entity_id": entity_id,
            "point_scores": averaged_scores,
            "point_labels": entity_point_labels[entity_id],
            "covered_point_mask": raw_counts > 0.0,
            "num_points": int(averaged_scores.shape[0]),
            "evaluated_start_index": evaluated_start_index,
            "evaluated_end_index": evaluated_end_index,
            "evaluated_num_points": evaluated_num_points,
            "raw_num_points": int(averaged_scores.shape[0]),
        }
        validate_evaluation_record(evaluation_record)
        evaluation_records.append(evaluation_record)

    return evaluation_records


def extract_covered_pointwise_arrays(
    evaluation_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
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
    ) -> dict[str, Any]:
        console_print(
            "EVAL",
            "Evaluating batch",
            batch_index=batch_index,
            **summarize_batch(batch_on_device),
        )

        # Tính anomaly score cho từng điểm dữ liệu (timestep)
        # trong từng window của batch.
        return model.test_step(batch_on_device)

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
        console_print(
            "EVAL",
            "Produced evaluation batch outputs",
            batch_index=batch_index,
            point_scores=summarize_tensor(point_scores),
            window_scores=summarize_tensor(step_output["outputs"]["window_scores"]),
        )

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
    ) -> dict[str, Any]:
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

        with torch.no_grad():
            # Với mỗi batch dữ liệu đọc được từ data_loader,
            for batch_index, batch in enumerate(data_loader, start=1):
                batch_on_device = self._move_batch_to_device(batch)
                step_output = self._run_model_on_batch(
                    model=model,
                    batch_index=batch_index,
                    batch_on_device=batch_on_device,
                )
                point_scores = step_output["outputs"]["point_scores"].detach().cpu()
                self._remember_forward_pass_seconds(
                    step_output=step_output,
                    forward_pass_seconds_history=forward_pass_seconds_history,
                )
                self._log_batch_outputs(
                    batch_index=batch_index,
                    step_output=step_output,
                    point_scores=point_scores,
                )
                pointwise_batch_payloads.append(
                    {
                        "meta": batch["meta"],
                        "point_scores": point_scores,
                        "point_labels": batch["point_labels"].detach().cpu(),
                    }
                )

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

        if point_score_threshold is None:
            # Chọn threshold cho anomaly score
            # Một timestep có anomaly score vượt ngưỡng threshold này thì tính là anomaly
            threshold = select_point_score_threshold(concatenated_scores, quantile=0.95)
            resolved_threshold_source = "positive_support_quantile_0.95"
        else:
            threshold = float(point_score_threshold)
            resolved_threshold_source = threshold_source or "provided_threshold"

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
        }
