"""Calculate point scores and calibrate them before test evaluation."""

from collections.abc import Iterator

import numpy as np

from . import protocols


def reconstruction_scores(x, reconstruction, settings: dict):
    """Reduce feature error, then reduce stochastic samples if present."""
    shape = tuple(x.shape)
    if len(shape) != 3 or not all(shape):
        raise ValueError("reconstruction scoring requires [batch, time, feature]")
    sampled = reconstruction.ndim == 4
    if sampled:
        if reconstruction.shape[1] < 1 or shape != (
            reconstruction.shape[0],
            *reconstruction.shape[2:],
        ):
            raise ValueError("reconstruction sample shape does not match the input")
        x = x[:, None]
    elif tuple(reconstruction.shape) != shape:
        raise ValueError("reconstruction shape does not match the input")
    space = settings["score_space"]

    residual = reconstruction - x
    
    if space == "raw_input":
        divisor = settings["scaler"]["divisor"]
        if np.shape(divisor) != (shape[-1],) or not np.all(np.asarray(divisor) > 0):
            raise ValueError(
                "raw input scoring requires a positive divisor for each feature"
            )
        if isinstance(residual, np.ndarray):
            divisor = np.asarray(divisor)
        else:
            divisor = residual.new_tensor(divisor)
        # The saved affine center cancels in the reconstruction difference.
        residual = residual * divisor
    elif space not in {"normalized_input", "latent"}:
        raise ValueError(f"unsupported reconstruction score space: {space}")
    
    error = (residual**2).mean(-1)

    return error.mean(1) if sampled else error


def _fit_transform(scores, name):
    if name == "identity":
        return {"name": name}
    center = float(np.median(scores))
    if name == "validation_mad_logistic":
        scale = float(np.median(np.abs(scores - center)) / 0.6745)
    elif name == "validation_median_iqr":
        lower, upper = np.quantile(scores, [0.25, 0.75])
        scale = max(float(upper - lower), 1e-8)
    else:
        raise ValueError(f"unsupported score transform: {name}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("calibration score transform requires a finite positive scale")
    return {"name": name, "center": center, "scale": scale}


def _transform(scores, transform):
    if transform["name"] == "identity":
        return scores
    scaled = (scores - transform["center"]) / transform["scale"]
    if transform["name"] == "validation_mad_logistic":
        return 1 / (1 + np.exp(-np.clip(scaled, -700, 700)))
    return scaled


def _eligible_record(record):
    values = np.asarray(record["raw_scores"], dtype=float)
    indices = np.asarray(record["indices"])
    if "source_window_labels" in record:
        labels = np.asarray(record["source_window_labels"], dtype=float)
        aligned = labels.ndim == 2 and len(labels) == len(values)
    else:
        labels = np.asarray(
            record.get("labels", np.full(values.shape, np.nan)), dtype=float
        )
        aligned = labels.shape == values.shape
    if not aligned or not np.all(np.isnan(labels) | np.isin(labels, [0, 1])):
        raise ValueError("calibration labels must align and be binary or unknown")
    anomalies = labels == 1
    unknown = np.isnan(labels)
    if labels.ndim == 2:
        anomalies = anomalies.any(axis=1)
        unknown = unknown.any(axis=1)
    selected = ~anomalies
    if not selected.any():
        raise ValueError("calibration has no eligible scores")
    eligible = dict(record, raw_scores=values[selected], indices=indices[selected])
    counts = {
        "selected_windows": int(selected.sum()),
        "excluded_windows": int(anomalies.sum()),
        "unknown_label_windows": int((unknown & selected).sum()),
    }
    return eligible, counts


def calibrate(scores: dict, settings: dict, *, identity: dict) -> dict:
    """Fit the point transform and threshold with an approved split."""
    phase = settings.get("phase", "offline")
    source = scores.get("source_split")
    protocols.validate_calibration_input(
        {"kind": "calibration", "source_split": source}, {}
    )
    expected = settings.get(f"{phase}_threshold_split", "validation")
    expected = (
        settings.get("offline_point_threshold_source", expected)
        if phase == "offline"
        else expected
    )
    if source != expected:
        raise ValueError(f"calibration: source {source!r} does not match {expected!r}")
    if (
        phase == "online"
        and settings.get("online_point_threshold_source") == "offline_parent"
    ):
        raise ValueError(
            "online point calibration must load the offline parent threshold"
        )
    protocols.validate_calibration_input(scores, {})
    selected, counts = _eligible_record(scores)
    transform = _fit_transform(
        selected["raw_scores"], settings.get("point_score_transform", "identity")
    )
    selected["raw_scores"] = _transform(selected["raw_scores"], transform)
    point = aggregate_points(
        iter([selected]), length=scores["length"], settings=settings
    )
    quantile = settings[f"{phase}_threshold_quantile"]
    if not 0 < quantile < 1:
        raise ValueError("calibration quantile must be between zero and one")
    result = {
        **identity,
        **counts,
        "kind": "calibration",
        "source_split": source,
        "score_space": settings["score_space"],
        "transform": transform,
        "point_threshold": float(
            np.quantile(point["scores"][point["coverage"]], quantile)
        ),
        "quantile": quantile,
        "overlap_reduction": _overlap_policy(settings),
        "ewma_previous_weight": settings.get("ewma_previous_weight", 0.25),
        "ewma_current_weight": settings.get("ewma_current_weight", 0.75),
        "normal_data_assumption": counts["unknown_label_windows"] > 0,
    }
    protocols.validate_calibration_output(result, {"evaluation": settings})
    return result


def _overlap_policy(settings):
    if settings.get("phase") == "online":
        return "last_ewma_estimate"
    return settings.get("offline_overlap_reduction", "mean")


def _ewma_rows(indices, values, settings, state):
    pending = dict(state.get("pending", {}))
    last_start = state.get("last_start", -1)
    previous_weight = settings.get("ewma_previous_weight", 0.25)
    current_weight = settings.get("ewma_current_weight", 0.75)
    scores = np.empty(values.shape, dtype=float)
    for row, (positions, observations) in enumerate(zip(indices, values, strict=True)):
        start = int(positions[0])
        if start <= last_start or np.any(np.diff(positions) <= 0):
            raise ValueError("point EWMA requires strict window order")
        updated = {}
        for column, (index, observed) in enumerate(
            zip(positions, observations, strict=True)
        ):
            index = int(index)
            value = float(observed)
            estimate = (
                previous_weight * pending[index] + current_weight * value
                if index in pending
                else value
            )
            scores[row, column] = estimate
            updated[index] = estimate
        pending = updated
        last_start = start
    return scores, {"last_start": last_start, "pending": pending}


def score_batch(
    raw: dict, calibration: dict | None, settings: dict, *, state: dict | None = None
):
    """Apply the selected score transform and return a new scoring state."""
    protocols.validate_calibration_input(raw, {})
    if (
        calibration is not None
        and calibration["score_space"] != settings["score_space"]
    ):
        raise ValueError("calibration: score space mismatch")
    transform = (
        calibration["transform"] if calibration is not None else {"name": "identity"}
    )
    values = _transform(np.asarray(raw["raw_scores"], dtype=float), transform)
    indices = np.asarray(raw["indices"])
    new_state = dict(state or {})
    if len(values) and _overlap_policy(settings) == "last_ewma_estimate":
        if values.ndim == 2:
            values, new_state = _ewma_rows(indices, values, settings, new_state)
        elif indices[0] <= new_state.get("last_start", -1) or np.any(
            np.diff(indices) <= 0
        ):
            raise ValueError("point endpoints must follow strict stream order")
        else:
            new_state = {"last_start": int(indices[-1]), "pending": {}}
    result = dict(raw, scores=values, indices=indices)
    if calibration is not None:
        result["predictions"] = values > calibration["point_threshold"]
    return result, new_state


def aggregate_points(batches: Iterator[dict], *, length: int, settings: dict) -> dict:
    """Aggregate aligned scores and leave uncovered positions undefined."""
    totals = np.zeros(length, dtype=float)
    counts = np.zeros(length, dtype=int)
    state = None
    policy = _overlap_policy(settings)
    for batch in batches:
        current = batch
        if "scores" not in current:
            current, state = score_batch(current, None, settings, state=state)
        indices = np.asarray(current["indices"]).ravel()
        values = np.asarray(current["scores"]).ravel()
        if (
            len(indices) != len(values)
            or np.any(indices < 0)
            or np.any(indices >= length)
        ):
            raise ValueError(
                "points: scores or indices do not align with the split range"
            )
        if policy == "mean":
            np.add.at(totals, indices, values)
        else:
            for index, value in zip(indices, values, strict=True):
                totals[index] = value
        np.add.at(counts, indices, 1)
    coverage = counts > 0
    points = np.full(length, np.nan)
    points[coverage] = (
        totals[coverage] / counts[coverage] if policy == "mean" else totals[coverage]
    )
    return {"scores": points, "coverage": coverage}
