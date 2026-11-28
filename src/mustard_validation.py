"""Measure MUSTARD on separate calibration and held-out validation windows."""

import numpy as np
import torch

from .evaluation_metrics import _uncertainty_value, compute_metrics
from .offline import _scored_batches
from .prepared_data import _window_schedule

CORE_METRICS = ("segment_mean_vus_pr", "segment_mean_affiliation_f1", "fpr")
POINT_VARIANCE = "point_anomaly_score_variance"


def assume_normal_validation(split: dict) -> dict:
    """Use normal pseudo-labels without changing the prepared data."""
    return split | {"labels": np.zeros(len(split["x"]), dtype=np.int8)}


def split_validation(split: dict) -> tuple[dict, dict]:
    """Split by source position. Keep the prepared split identity."""
    length = len(split["x"])
    midpoint = length // 2
    parts = []
    for start, stop in ((0, midpoint), (midpoint, length)):
        part = split.copy()
        for field in ("x", "labels", "timestamps"):
            if field in split:
                part[field] = split[field][start:stop]
        if "segments" in split:
            part["segments"] = [
                (max(start, left) - start, min(stop, right) - start)
                for left, right in split["segments"]
                if max(start, left) < min(stop, right)
            ]
        parts.append(part)
    return parts[0], parts[1]


def _validation_records(method, split, config, options, *, synthetic=None):
    settings = config["data"] | {"limits": config.get("limits", {})}
    if (
        len(split["x"]) == 0
        or next(_window_schedule(split, settings, "offline", config["seed"]), None)
        is None
    ):
        return
    yield from _scored_batches(
        method, split, config, synthetic=synthetic, score_options=options
    )


def _score_options(config, space, fields=()):
    return {
        "score_space": space,
        "generator": torch.Generator().manual_seed(config["seed"]),
        "uncertainty_fields": fields,
    }


def synthetic_records(method, held_out, config, space, *, uncertainty_fields=()):
    """Regenerate the same windows and masks for each evaluation."""
    seed = config["seed"]
    recipe = dict(method.config["synthetic"])
    options = _score_options(config, space, uncertainty_fields)
    yield from _validation_records(
        method,
        held_out,
        config,
        options,
        synthetic=(recipe, torch.Generator().manual_seed(seed)),
    )


def _epoch_threshold(records, quantile):
    scores = [record["raw_scores"].ravel() for record, _ in records]
    return float(np.quantile(np.concatenate(scores), quantile)) if scores else None


def _synthetic_metrics(records, settings, threshold):
    requested = set(CORE_METRICS)
    requested.update(set(settings["metrics"]) & {"segment_mean_vus_roc", "recall"})
    windows, masks = [], []
    for record, mask in records:
        windows.extend(record["raw_scores"])
        masks.extend(mask)
    if not windows:
        return dict.fromkeys(requested)
    if threshold is None:
        requested &= {"segment_mean_vus_pr", "segment_mean_vus_roc"}
    width = len(windows[0])
    length = len(windows) * width
    scores = {
        "scores": np.concatenate(windows),
        "segments": [(start, start + width) for start in range(0, length, width)],
    }
    result = compute_metrics(
        scores,
        np.concatenate(masks),
        {
            "metrics": sorted(requested),
            "threshold": threshold,
            "vus_max_buffer_size": settings.get("vus_max_buffer_size", 20),
            "vus_num_thresholds": settings.get("vus_num_thresholds", 200),
        },
    )
    return dict.fromkeys(CORE_METRICS) | result["metrics"]


def evaluate_epoch(method, validation: dict, config: dict, *, stage: str) -> dict:
    """Use unsmoothed window scores to measure one completed epoch."""
    calibration, held_out = split_validation(assume_normal_validation(validation))
    settings = config["evaluation"]
    space = "raw_input" if stage == "stage_a" else settings["score_space"]
    options = _score_options(config, space)
    threshold = _epoch_threshold(
        _validation_records(method, calibration, config, options),
        settings["offline_threshold_quantile"],
    )
    fpr, reason = None, "no_eligible_calibration_scores"
    if threshold is not None:
        predictions = [
            record["raw_scores"].ravel() > threshold
            for record, _ in _validation_records(method, held_out, config, options)
        ]
        count = sum(len(values) for values in predictions)
        if count:
            fpr = sum(int(values.sum()) for values in predictions) / count
        reason = None if count else "no_held_out_scores"
    synthetic = synthetic_records(method, held_out, config, space)
    return {
        "score_space": space,
        "threshold": threshold,
        "held_out_fpr": fpr,
        "reason": reason,
        "synthetic_metrics": _synthetic_metrics(synthetic, settings, threshold),
    }


def synthetic_uncertainty(method, validation, config):
    """Summarize the selected checkpoint's synthetic validation scores."""
    _, held_out = split_validation(validation)
    batches = synthetic_records(
        method,
        held_out,
        config,
        config["evaluation"]["score_space"],
        uncertainty_fields=(POINT_VARIANCE,),
    )
    values = [
        record["uncertainty"][POINT_VARIANCE].ravel()
        for record, _ in batches
        if POINT_VARIANCE in record.get("uncertainty", {})
    ]
    data = {POINT_VARIANCE: np.concatenate(values)} if values else {}
    mean, metadata = _uncertainty_value(f"{POINT_VARIANCE}_mean", data)
    return {"mean": mean, "value_count": metadata.get("value_count", 0)}
