import numpy as np
import pytest
from affiliation.metrics import pr_from_events
from vus.utils.metrics import metricor

from src.evaluation_metrics import compute_metrics
from src.budgeted_vus import compute_budgeted_vus_metrics


Y = np.array([0, 0, 1, 1, 0, 0])
S = np.array([0.1, 0.6, 0.8, 0.7, 0.05, 0.2])
SETTINGS = {"threshold": 0.5, "vus_max_buffer_size": 2, "vus_num_thresholds": 20}
UNCERTAINTY = [
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
]


def test_detection_and_reference_range_values():
    budgets = compute_budgeted_vus_metrics(Y, S, 2, 20)
    _, _, _, _, roc, pr = metricor().RangeAUC_volume_opt(Y, S, 2, 20)
    values = pr_from_events([(1, 4)], [(2, 4)], (0, len(Y)))
    p, r = values["precision"], values["recall"]
    expected = {
        "precision": 2 / 3,
        "recall": 1,
        "f1": 0.8,
        "fpr": 0.25,
        "roc_auc": 1,
        "pr_auc": 1,
        "vus_pr": pr,
        "vus_roc": roc,
        "affiliation_f1": 2 * p * r / (p + r),
    }
    expected.update(
        {
            f"{kind}_at_fpr_{budget}": value
            for kind, entries in budgets.items()
            for budget, value in entries.items()
        }
    )
    expected.update(
        {
            f"segment_mean_{key}": value
            for key, value in list(expected.items())
            if key.startswith(("vus_", "affiliation_"))
        }
    )
    result = compute_metrics({"scores": S}, Y, SETTINGS | {"metrics": list(expected)})
    assert result["metrics"] == pytest.approx(expected, abs=1e-10, rel=1e-8)


@pytest.mark.parametrize("field", UNCERTAINTY)
def test_uncertainty_summary(field):
    expected = {
        "mean": 2,
        "std": np.sqrt(2 / 3),
        "min": 1,
        "p50": 2,
        "p95": 2.9,
        "max": 3,
    }
    metrics = {f"{field}_{key}": value for key, value in expected.items()}
    result = compute_metrics(
        {"scores": S, "uncertainty": {field: [1, 2, 3, np.nan]}},
        Y,
        {"metrics": list(metrics)},
    )
    assert result["metrics"] == pytest.approx(metrics)


def test_gap_unknown_labels_and_uncovered_scores_are_kept_separate():
    labels = np.array([0, 1, 0, np.nan, 0, 0, 1, 0, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.6, 0.9, 0.2, np.nan])
    coverage = np.array([True] * 8 + [False])
    metrics = ["fpr", "affiliation_f1", "segment_mean_affiliation_f1"]
    result = compute_metrics(
        {"scores": scores, "coverage": coverage, "segments": [(0, 5), (5, 9)]},
        labels,
        SETTINGS | {"metrics": metrics},
    )
    assert [segment["bounds"] for segment in result["segments"]] == [
        [0, 3],
        [4, 5],
        [5, 8],
    ]
    assert result["metrics"]["fpr"] == pytest.approx(1 / 5)
    assert result["metrics"]["affiliation_f1"] is None
    summary = result["metric_metadata"]["segment_mean_affiliation_f1"]
    assert summary["contributing_segments"] == 2
    assert summary["omitted_observations"] == 1


def test_constants_empty_classes_and_nonfinite_scores():
    settings = SETTINGS | {"metrics": ["roc_auc", "pr_auc", "precision", "fpr"]}
    constant = np.full(6, 0.5)
    result = compute_metrics({"scores": constant}, Y, settings)
    assert result["metrics"]["roc_auc"] == 0.5
    assert result["metrics"]["precision"] is None
    assert result["metrics"]["fpr"] == 0
    normal = compute_metrics({"scores": constant}, np.zeros(6), settings)
    assert normal["metrics"]["roc_auc"] is None and normal["metrics"]["pr_auc"] is None
    with pytest.raises(ValueError, match="finite"):
        compute_metrics({"scores": np.array([np.inf] * 6)}, Y, settings)


def test_unavailable_reference_is_an_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "affiliation.metrics", None)
    with pytest.raises(ImportError, match="affiliation"):
        compute_metrics({"scores": S}, Y, SETTINGS | {"metrics": ["affiliation_f1"]})


def test_evaluation_returns_configured_uncertainty_summaries():
    field = "point_anomaly_score_variance"
    scores = {
        "scores": np.array([0.3, 0.7]),
        "uncertainty": {field: np.array([0.2, 0.4])},
    }
    settings = {
        "metrics": ["fpr"],
        "threshold": 0.5,
        "uncertainty_fields": [field],
        "uncertainty_statistics": ["mean", "max"],
    }
    result = compute_metrics(scores, np.array([0, 1]), settings)
    assert result["metrics"][f"{field}_mean"] == pytest.approx(0.3)
    assert result["metrics"][f"{field}_max"] == pytest.approx(0.4)


def test_variance_summary_uses_point_scores_from_stochastic_passes():
    import torch
    from src.methods.mustard.score_outputs import uncertainty

    field = "point_anomaly_score_variance"
    metric = field + "_mean"
    errors = torch.tensor([[[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]]])
    sample = torch.zeros(1, 2, 1)
    passes = [
        dict.fromkeys(("continuous", "discrete", "reconstruction", "logits"), sample)
    ] * 3
    for count, expected in ((3, 2.0), (1, None)):
        fields = uncertainty(passes[:count], errors[:, :count], [field], correction=1)
        result = compute_metrics(
            {"scores": [0.0, 1.0], "uncertainty": fields}, [0, 1], {"metrics": [metric]}
        )
        assert result["metrics"][metric] == expected
