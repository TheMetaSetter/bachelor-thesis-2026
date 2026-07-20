from __future__ import annotations

"""Inspect a thesis run's reporting artifacts with a compact JSON summary."""

import argparse
import json
from pathlib import Path
from typing import Any


UQ_KEYS = (
    "point_anomaly_score_variance_mean",
    "point_anomaly_score_variance_p95",
    "window_anomaly_score_variance_mean",
    "classification_probability_variance_mean",
    "classification_variance_mean",
    "continuous_retrieval_variance_point_mean",
    "continuous_retrieval_variance_window_mean",
    "discrete_retrieval_variance_point_mean",
    "discrete_retrieval_variance_window_mean",
    "reconstruction_variance_point_mean",
    "reconstruction_variance_window_mean",
    "reconstruction_variance_full_mean",
)


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    metrics_path = run_root / "two_stage" / "stage_b_fusion_finetuning" / "evaluation_metrics.json"
    uq_path = run_root / "metrics" / "uq_summary.json"
    trace_path = run_root / "traces" / "test_traces.json"

    metrics = _load_json(metrics_path)
    uq_summary = _load_json(uq_path)
    traces = _load_json(trace_path)

    test_split = (uq_summary or {}).get("splits", {}).get("test", {})
    uncertainty_summary = test_split.get("uncertainty_summary", {})
    trace_audit = test_split.get("trace_audit", {})

    first_trace = traces[0] if isinstance(traces, list) and traces else None

    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "metrics": {k: metrics.get(k) for k in ("vus_pr", "affiliation_f1", "vus_roc")}
                if isinstance(metrics, dict)
                else None,
                "uq_missing": [k for k in UQ_KEYS if uncertainty_summary.get(k) is None],
                "uq_non_null": [k for k in UQ_KEYS if uncertainty_summary.get(k) is not None],
                "trace_audit": {
                    "any_uncertainty_history": trace_audit.get("any_uncertainty_history"),
                    "any_mc_sample_history": trace_audit.get("any_mc_sample_history"),
                    "uncertainty_history_non_null_count": trace_audit.get(
                        "uncertainty_history_non_null_count"
                    ),
                    "mc_histories_non_null_count": trace_audit.get(
                        "mc_histories_non_null_count"
                    ),
                },
                "trace_first": {
                    "keys": sorted(first_trace.keys()) if isinstance(first_trace, dict) else None,
                    "stochastic_query_keys": sorted(first_trace.get("stochastic_query", {}).keys())
                    if isinstance(first_trace, dict)
                    and isinstance(first_trace.get("stochastic_query"), dict)
                    else None,
                    "has_uncertainty_history": first_trace.get("uncertainty_history") is not None
                    if isinstance(first_trace, dict)
                    else None,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
