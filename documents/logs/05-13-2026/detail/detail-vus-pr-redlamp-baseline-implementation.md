---
date: 2026-05-13 14:18:06 +07 +0700
researcher: Codex
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation notes for VUS-PR exact-naive metric and RedLamp baseline setup"
tags: [detail, implementation, time-series, anomaly-detection, metrics, vus-pr]
status: complete
source_detail: documents/logs/05-13-2026/detail/detail-vus-pr-redlamp-baseline.md
---

# Detail: VUS-PR RedLamp Baseline Implementation Notes

## Implemented Scope

- Added exact-naive VUS-PR helpers in `src/metrics/pointwise.py`.
- Added optional `vus_pr` emission through `compute_pointwise_metrics`.
- Added evaluator-level VUS settings in `src/engine/evaluator.py`.
- Passed evaluation VUS settings from `scripts/evaluate.py`.
- Added explicit VUS settings to:
  - `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`
  - `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`
- Added `pr_auc` and `vus_pr` fields to ablation summary rows.
- Added focused tests for VUS range extraction, threshold-aware labels, VUS behavior, evaluator wiring, and ablation summary fields.

## Final Test Commands

```bash
./.venv/bin/pytest -q tests/test_vus_pr_metric.py
```

Result: `5 passed`.

```bash
./.venv/bin/pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py tests/test_config_loading.py tests/test_ablation_runner.py
```

Result: `36 passed, 4 warnings`.

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py tests/test_multitask_shapes.py tests/test_one_redlamp_mlp_train_step.py tests/test_one_multitask_train_step.py
```

Result: `8 passed`.

```bash
./.venv/bin/python scripts/run_multiseed_experiments.py --config-paths configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml --preflight-only
```

Result: preflight passed.

```bash
./.venv/bin/python scripts/run_multiseed_experiments.py --config-paths configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml --preflight-only
```

Result: preflight passed.

## Observed Metric Keys

When `vus_max_buffer_size` is configured, pointwise evaluation metrics include:

- `roc_auc`
- `pr_auc`
- `precision`
- `recall`
- `f1`
- `fpr`
- `vus_pr`
- `threshold`

The ablation summary row now includes both `pr_auc` and `vus_pr` alongside the existing thresholded fields.

## Deviations and Compatibility Notes

- The implementation uses `np.trapezoid` instead of `np.trapz` because the repository virtual environment uses NumPy 2.x, where `np.trapz` is unavailable.
- The PR area helper applies a precision envelope and clamps precision, recall, and area into `[0, 1]`. This keeps the exact-naive VUS-PR output in the valid metric range and preserves the acceptance criterion that a perfect ranking returns `1.0`.
- No model, data loader, online adaptation, or projector code was modified.
