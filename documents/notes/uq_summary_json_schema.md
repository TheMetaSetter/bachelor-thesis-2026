# UQ Summary JSON Schema

Date: 2026-07-20

Purpose:
- Companion artifact to `evaluation_metrics.json`.
- Store only summary statistics and audit counts.
- Keep the file small enough that raw `*_traces.json` and Monte Carlo tensors can be deleted later.

Non-goals:
- Do not store raw tensors.
- Do not store per-window histories.
- Do not duplicate the full `stochastic_query` or `uncertainty_history` payload.

## Top-level shape

```json
{
  "schema_version": 1,
  "created_at_utc": "2026-07-20T12:34:56Z",
  "run": {
    "benchmark_kind": "offline",
    "experiment_name": "string",
    "method_name": "string",
    "variant_name": "O0",
    "entity_id": "machine_1_6",
    "seed": 6,
    "stage_name": "stage_b_fusion_finetuning",
    "checkpoint_path": "/abs/path/to/checkpoint.pt",
    "checkpoint_sha256": "hex-or-null",
    "experiment_config_path": "/abs/path/to/config.yaml",
    "protocol_config_path": "/abs/path/to/protocol.yaml",
    "output_dir": "/abs/path/to/output"
  },
  "run_scalar_logs": {
    "query/continuous_temperature": 0.9,
    "query/discrete_temperature": 0.9,
    "query/num_samples_train": 10,
    "query/num_samples_eval": 10,
    "query/continuous_weight_entropy_mean": 0.0,
    "query/discrete_topk_weight_entropy_mean": 0.0
  },
  "splits": {
    "clean_validation": { "...": "..." },
    "synthetic_validation": { "...": "..." },
    "test": { "...": "..." }
  }
}
```

## `run`

This object identifies the benchmark run once, so the split blocks stay small.

Required fields:
- `benchmark_kind`: `offline` or `online`
- `experiment_name`
- `method_name`
- `variant_name`
- `entity_id`
- `seed`
- `stage_name`
- `checkpoint_path`
- `experiment_config_path`
- `protocol_config_path`
- `output_dir`

Optional fields:
- `checkpoint_sha256`

## `run_scalar_logs`

These are run-level scalar summaries that do not depend on the split.

Recommended keys:
- `query/continuous_temperature`
- `query/discrete_temperature`
- `query/num_samples_train`
- `query/num_samples_eval`
- `query/continuous_weight_entropy_mean`
- `query/discrete_topk_weight_entropy_mean`

If a run cannot compute one of these values, store `null`.

## Per-split schema

Each split under `splits` uses the same shape.

```json
{
  "num_traces": 3,
  "sample_retention_policy": "retain_for_eda",
  "trace_audit": {
    "any_uncertainty_history": true,
    "uncertainty_history_non_null_count": 3,
    "any_mc_sample_history": true,
    "mc_histories_non_null_count": {
      "point_score_samples": 3,
      "window_score_samples": 3,
      "reconstruction_samples": 3,
      "classification_probability_samples": 3
    }
  },
  "point_score_summary": {
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "window_score_summary": {
    "mean": 0.0,
    "std": 0.0,
    "min": 0.0,
    "p50": 0.0,
    "p95": 0.0,
    "max": 0.0
  },
  "uncertainty_summary": {
    "point_anomaly_score_variance_mean": 0.0,
    "point_anomaly_score_variance_p95": 0.0,
    "window_anomaly_score_variance_mean": 0.0,
    "continuous_retrieval_variance_point_mean": 0.0,
    "continuous_retrieval_variance_window_mean": 0.0,
    "discrete_retrieval_variance_point_mean": 0.0,
    "discrete_retrieval_variance_window_mean": 0.0,
    "reconstruction_variance_point_mean": 0.0,
    "reconstruction_variance_window_mean": 0.0,
    "reconstruction_variance_full_mean": 0.0,
    "classification_probability_variance_mean": 0.0,
    "classification_variance_mean": 0.0
  }
}
```

### Field notes

- `num_traces` is the number of trace records exported for that split.
- `trace_audit.any_uncertainty_history` tells whether at least one trace kept the `uncertainty_history` block.
- `trace_audit.any_mc_sample_history` tells whether at least one trace kept MC sample histories.
- `point_score_summary` and `window_score_summary` are the compact summaries needed for reporting and quick QA.
- `uncertainty_summary` should contain only scalar summaries, never tensor payloads.

### Required summary fields for reporting

For the offline-phase table and the validation-vs-test variance comparison, keep at least:
- `point_score_summary.mean`
- `window_score_summary.mean`
- `uncertainty_summary.point_anomaly_score_variance_mean`
- `uncertainty_summary.window_anomaly_score_variance_mean`
- `uncertainty_summary.classification_variance_mean`
- `uncertainty_summary.reconstruction_variance_point_mean`
- `uncertainty_summary.reconstruction_variance_window_mean`
- `uncertainty_summary.continuous_retrieval_variance_point_mean`
- `uncertainty_summary.continuous_retrieval_variance_window_mean`
- `uncertainty_summary.discrete_retrieval_variance_point_mean`
- `uncertainty_summary.discrete_retrieval_variance_window_mean`

## Explicit exclusions

These must not appear in `uq_summary.json`:
- `stochastic_query`
- `uncertainty_history`
- `mc_sample_histories`
- `deterministic_geometry`
- raw MC sample tensors
- per-window arrays of scores or labels

## Suggested interpretation

`evaluation_metrics.json` remains the canonical file for final benchmark metrics such as `vus_pr`, `affiliation_f1`, and `vus_roc`.

`uq_summary.json` is only for compact UQ reporting and provenance checks after the heavy trace files have been removed.
