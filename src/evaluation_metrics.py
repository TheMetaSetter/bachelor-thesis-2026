"""Compute point metrics and metrics for continuous eligible segments."""

import numpy as np


_POINT_NAMES = {"precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"}
_PREFIX = "segment_mean_"
_BUDGETS = ("0.001", "0.005", "0.01")
_RANGE_NAMES = {"vus_roc", "vus_pr", "affiliation_f1"} | {
    f"vus_{kind}_at_fpr_{budget}" for kind in ("pr", "roc") for budget in _BUDGETS
}
_UNCERTAINTY = {
    "point_anomaly_score_variance",
    "window_anomaly_score_variance",
    "continuous_retrieval_variance_point",
    "continuous_retrieval_variance_window",
    "discrete_retrieval_variance_point",
    "discrete_retrieval_variance_window",
    "reconstruction_variance_point",
    "reconstruction_variance_window",
    "reconstruction_variance_full",
    "classification_probability_variance",
    "classification_variance_mean",
}
_STATISTICS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "p50": lambda values: np.quantile(values, 0.5),
    "p95": lambda values: np.quantile(values, 0.95),
}
_UNCERTAINTY_NAMES = {
    f"{field}_{stat}" for field in _UNCERTAINTY for stat in _STATISTICS
}


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _event_bounds(mask):
    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [(int(left), int(right)) for left, right in transitions.reshape(-1, 2)]


def _point_metrics(names, scores, labels, threshold):
    result = {}
    if names - {"roc_auc", "pr_auc"}:
        if not np.isfinite(threshold):
            raise ValueError("threshold-dependent metrics require a finite threshold")
        predicted = scores > threshold
        tp = np.count_nonzero(predicted & labels)
        fp = np.count_nonzero(predicted & ~labels)
        positive = np.count_nonzero(labels)
        denominators = {
            "precision": tp + fp,
            "recall": positive,
            "f1": positive + tp + fp,
            "fpr": len(labels) - positive,
        }
        numerators = {"precision": tp, "recall": tp, "f1": 2 * tp, "fpr": fp}
        result.update(
            {
                key: _divide(numerators[key], denominators[key])
                for key in names & denominators.keys()
            }
        )
    if names & {"pr_auc", "roc_auc"}:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if "pr_auc" in names:
            result["pr_auc"] = (
                float(average_precision_score(labels, scores)) if labels.any() else None
            )
        if "roc_auc" in names:
            result["roc_auc"] = (
                float(roc_auc_score(labels, scores))
                if labels.any() and not labels.all()
                else None
            )
    return result


def _range_metrics(names, scores, labels, settings):
    if labels.all() or not labels.any():
        return dict.fromkeys(names)
    output = {}
    buffer_size = settings.get("vus_max_buffer_size", 20)
    thresholds = settings.get("vus_num_thresholds", 200)
    if names & {"vus_pr", "vus_roc"}:
        from vus.utils.metrics import metricor

        _, _, _, _, roc, pr = metricor().RangeAUC_volume_opt(
            labels.astype(int), scores, buffer_size, thresholds
        )
        output.update(
            {
                key: value
                for key, value in {"vus_pr": pr, "vus_roc": roc}.items()
                if key in names
            }
        )
    if any("_at_fpr_" in key for key in names):
        from .budgeted_vus import compute_budgeted_vus_metrics

        values = compute_budgeted_vus_metrics(
            labels.astype(int), scores, buffer_size, thresholds
        )
        output.update(
            {
                name: value
                for kind, entries in values.items()
                for budget, value in entries.items()
                if (name := f"{kind}_at_fpr_{budget}") in names
            }
        )
    if "affiliation_f1" in names:
        from affiliation.metrics import pr_from_events

        threshold = settings.get("threshold", np.nan)
        if not np.isfinite(threshold):
            raise ValueError("affiliation F1 requires a finite threshold")
        predicted = _event_bounds(scores > threshold)
        if predicted:
            values = pr_from_events(predicted, _event_bounds(labels), (0, len(labels)))
            precision, recall = values["precision"], values["recall"]
            output["affiliation_f1"] = (
                _divide(2 * precision * recall, precision + recall) or 0.0
            )
        else:
            output["affiliation_f1"] = 0.0
    return {
        key: float(value) if np.isfinite(value) else None
        for key, value in output.items()
    }


def _validated_arrays(record, labels):
    scores = np.asarray(record["scores"], dtype=float)
    labels = np.asarray(labels, dtype=float)
    coverage = np.asarray(record.get("coverage", ~np.isnan(scores)))
    if (
        scores.ndim != 1
        or labels.shape != scores.shape
        or coverage.shape != scores.shape
    ):
        raise ValueError("metrics: scores, labels, and coverage must align")
    if coverage.dtype != bool:
        raise ValueError("metrics: coverage must be boolean")
    if np.any(np.isinf(scores)) or not np.isfinite(scores[coverage]).all():
        raise ValueError("metrics: covered scores must be finite")
    known = ~np.isnan(labels)
    if not np.all(np.isin(labels[known], [0, 1])):
        raise ValueError("metrics: labels must be binary or unknown")
    eligible = known & coverage
    counts = {
        "total_observations": len(scores),
        "covered_observations": int(coverage.sum()),
        "eligible_observations": int(eligible.sum()),
        "unknown_labels": int((~known).sum()),
    }
    return scores, labels == 1, eligible, counts


def _summarize_segments(names, scores, labels, eligible, bounds, settings):
    segment_records = []
    for left, right in bounds:
        for start, stop in _event_bounds(eligible[left:right]):
            start, stop = start + left, stop + left
            y = labels[start:stop]
            values = _range_metrics(names, scores[start:stop], y, settings)
            reason = (
                "missing_class"
                if y.all() or not y.any()
                else "nonfinite_reference_value"
            )
            segment_records.append(
                {
                    "bounds": [start, stop],
                    "observations": stop - start,
                    "metrics": values,
                    "missing": {key: reason for key in values if values[key] is None},
                }
            )
    return segment_records


def _range_summary(name, segments, count):
    values = [
        (segment["observations"], segment["metrics"][name])
        for segment in segments
        if segment["metrics"][name] is not None
    ]
    included = sum(n for n, _ in values)
    eligible = sum(segment["observations"] for segment in segments)
    metadata = {
        "total_segments": len(segments),
        "contributing_segments": len(values),
        "undefined_segments": len(segments) - len(values),
        "eligible_observations": eligible,
        "contributing_observations": included,
        "omitted_observations": eligible - included,
        "coverage": _divide(included, count),
    }
    if not values:
        metadata["reason"] = "no_defined_segments"
    return _divide(sum(n * value for n, value in values), included), metadata


def _uncertainty_value(name, uncertainties):
    field, statistic = name.rsplit("_", 1)
    if field not in uncertainties:
        return None, {"reason": "not_provided_by_method"}
    values = np.asarray(uncertainties[field], dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, {"reason": "no_finite_uncertainty_values"}
    return float(_STATISTICS[statistic](finite)), {"value_count": len(finite)}


def _segments(record, length):
    bounds = record.get("segments", [(0, length)] if length else [])
    cursor = 0
    for left, right in bounds:
        if left != cursor or not left < right <= length:
            raise ValueError("metrics: invalid source segment bounds")
        cursor = right
    if cursor != length:
        raise ValueError("metrics: source segments must cover the split")
    return bounds


def compute_metrics(scores: dict, labels, settings: dict) -> dict:
    """Return unrounded metrics with eligibility and segment records."""
    requested = set(settings["metrics"])
    requested.update(
        f"{field}_{statistic}"
        for field in settings.get("uncertainty_fields", ())
        for statistic in settings.get("uncertainty_statistics", ())
    )
    supported = (
        _POINT_NAMES
        | _RANGE_NAMES
        | {_PREFIX + name for name in _RANGE_NAMES}
        | _UNCERTAINTY_NAMES
    )
    if requested - supported:
        raise ValueError(f"metrics: unknown names {sorted(requested - supported)}")
    values, binary, eligible, counts = _validated_arrays(scores, labels)
    metrics = _point_metrics(
        requested & _POINT_NAMES,
        values[eligible],
        binary[eligible],
        settings.get("threshold", np.nan),
    )
    metadata = {
        key: (
            {"reason": "undefined_for_eligible_labels_or_predictions"}
            if value is None
            else {}
        )
        for key, value in metrics.items()
    }
    if "pr_auc" in metadata:
        metadata["pr_auc"]["definition"] = "average_precision"
    names = {key.removeprefix(_PREFIX) for key in requested} & _RANGE_NAMES
    bounds = _segments(scores, len(values))
    segments = (
        _summarize_segments(names, values, binary, eligible, bounds, settings)
        if names
        else []
    )
    for name in requested & _RANGE_NAMES:
        metrics[name] = segments[0]["metrics"][name] if len(segments) == 1 else None
        metadata[name] = (
            {}
            if metrics[name] is not None
            else {"reason": "undefined_or_discontinuous"}
        )
    for name in requested - _RANGE_NAMES:
        if name.startswith(_PREFIX):
            metrics[name], metadata[name] = _range_summary(
                name.removeprefix(_PREFIX), segments, len(values)
            )
    for name in requested & _UNCERTAINTY_NAMES:
        metrics[name], metadata[name] = _uncertainty_value(
            name, scores.get("uncertainty", {})
        )
    return {
        "metrics": metrics,
        "eligibility": counts,
        "segments": segments,
        "metric_metadata": metadata,
    }
