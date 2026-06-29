---
date: 2026-06-29T00:00:00+07:00
researcher: Codex
git_commit: ad1fce42fc447fcf8a2180153b0c8590edeec541
branch: dev
repository: bachelor-thesis-2026
topic: "Why some test time-points do not appear in any test window"
tags: [research, time-series, anomaly-detection, windowing, evaluation]
status: complete
last_updated: 2026-06-29
last_updated_by: Codex
---

# Research: Why some test time-points do not appear in any test window

**Date**: 2026-06-29 00:00:00 +07:00
**Researcher**: Codex
**Git Commit**: `ad1fce42fc447fcf8a2180153b0c8590edeec541`
**Branch**: `dev`

## Research Question

Why do some time-points in the testing series never appear in any testing window, so the model never gets a chance to compute scores on them?

## Summary

The main reason is the current window-generation rule in [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:176). It creates windows only at start indices

`0, stride, 2*stride, ...`

until the largest start index that still allows a full window of length `window_size`. If `(sequence_length - window_size)` is not divisible by `stride`, then the last few points at the tail of the test sequence are left uncovered. Those tail points never enter any window.

A second and more severe reason appears when `max_test_windows` is configured. In that case, the builder stops early after a fixed number of windows and leaves the remaining future timeline completely unevaluated.

The evaluator then reconstructs a full-length record anyway. For uncovered points, it keeps zero score and zero label placeholders inside the reconstructed timeline instead of removing them from metric computation. This means the codebase already knows the evaluated coverage length, but the pointwise score vector still has the full raw test length.

## Detailed Findings

### Data Preparation

The active window builder is `WindowDataset` in [src/data/loaders.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:176). Its loop is:

```python
for start_index in range(0, sequence_length - window_size + 1, stride):
    end_index = start_index + window_size
    self.index_records.append((sequence_index, start_index, end_index))
```

This rule means:

1. each window must have exactly `window_size` points
2. each new window start moves forward by exactly `stride`
3. the builder does not create a special last window to force coverage of the final suffix

So if:

`remainder = (sequence_length - window_size) % stride`

and `remainder != 0`, then the uncovered tail length is exactly `remainder`.

For a toy example:

- `sequence_length = 31`
- `window_size = 20`
- `stride = 10`
- valid starts are `0` and `10`
- the last window is `[10, 30)`
- point `30` is never included

This is not caused by the model. It happens before the model runs.

### STAFFIII Concrete Evidence

The current evaluation command logged by the user points to:

- [configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/experiment/scale/anomaly_archive__redlamp_mlp_baseline__staffiii-window20-adamw-cosine-warmup10-vus-pr-confmat__w20__seed11__default.yaml:12)
- which currently resolves to [configs/data/anomaly_archive_staffiii_full.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/data/anomaly_archive_staffiii_full.yaml:1)

For this active full config:

- `window_size = 20`
- `stride = 10`
- `max_test_windows` is not set

Using the real parser in [src/data/datasets/anomaly_archive.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:90), the STAFFIII test sequence has:

- raw test length = `258228`
- actual generated test windows = `25821`
- last generated window = `[258200, 258220)`
- last possible full-coverage-forcing start would be `258208`
- uncovered suffix length = `8`

So the final test indices `258220` through `258227` never appear in any test window.

These uncovered points are normal points in this particular series tail, but they are still unevaluated points.

### Smoke or Truncated Config Case

There is also a separate smoke config:

- [configs/data/anomaly_archive_staffiii.yaml](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/configs/data/anomaly_archive_staffiii.yaml:1)

This one sets:

- `max_test_windows: 32`

Under the same `window_size = 20`, `stride = 10`, that means:

- last kept window = `[310, 330)`
- only the first `330` test points are covered
- `257898` test points are left uncovered

So there are two distinct uncovered-point mechanisms:

1. small uncovered tail due to stride remainder, even in full evaluation
2. massive truncation due to `max_test_windows`, in smoke evaluation

### Evaluation Reconstruction

The evaluator accumulates overlapping window scores back into entity timelines in [src/engine/evaluator.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:57).

Two details matter:

1. `entity_score_counts` is zero for points never covered by any window.
2. reconstruction then does `counts = torch.clamp(raw_counts, min=1.0)` and `averaged_scores = score_sum / counts`.

That means an uncovered point gets:

- `score_sum = 0`
- `count = 1` after clamping
- reconstructed score = `0`

The reconstructed record still keeps:

- `num_points = full raw test length`
- `raw_num_points = full raw test length`
- `evaluated_num_points = only the number of covered points`

So the codebase preserves a distinction between raw length and evaluated length, but it still stores a full-length point-score vector.

### Audit Logic

The audit layer already detects truncation from dataset coverage metadata in [src/analysis/evaluation_protocol_audit.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py:75).

However, its current `is_truncated` flag is tied to:

```python
configured_limit is not None and any(...)
```

So it explicitly warns on `max_test_windows` truncation, but it does not treat the small stride-remainder tail as truncated unless a window cap was configured. In other words, the code already knows the evaluated end index and raw end index, but the audit warning is currently stronger for capped truncation than for remainder-based tail loss.

### Tests that Lock Current Behavior

The current behavior is already captured by tests:

- [tests/test_evaluation_protocol_audit.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_evaluation_protocol_audit.py:74) verifies that uncovered suffix points remain in the reconstructed full-length record with zero score and zero label.
- [tests/test_evaluation_protocol_audit.py](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/tests/test_evaluation_protocol_audit.py:115) verifies that capped test coverage is flagged as truncated in the audit report.

## Evaluation

The question here is not whether the model predicts well. The question is whether every intended test point is actually evaluated.

The current answer is:

- not always
- because some points are never windowized
- and the evaluator later reconstructs a full-length vector anyway

This can affect downstream metrics because zero-scored uncovered points are still present in reconstructed pointwise records.

## Code References

- [src/data/loaders.py:73](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:73) - dataset builder passes `max_*_windows` into `WindowDataset`
- [src/data/loaders.py:176](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:176) - window start loop that can leave a remainder tail
- [src/engine/evaluator.py:100](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:100) - score accumulation over covered ranges only
- [src/engine/evaluator.py:131](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:131) - uncovered counts are clamped to `1.0`
- [src/analysis/evaluation_protocol_audit.py:138](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py:138) - truncation warning depends on configured window cap
- [src/data/datasets/anomaly_archive.py:105](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:105) - test sequence is built as future data after `start_index`

## Pipeline Documentation

For the active pipeline, the order is:

1. parse full raw sequences into `train`, `val`, `test`
2. fit scaler on train sequences only
3. transform train, val, and test sequences
4. create fixed-size overlapping windows
5. run the model on test windows only
6. merge overlapping scores back to pointwise entity timelines
7. concatenate entity timelines for pointwise metrics

The missing-point issue appears between steps 4 and 6.

## Historical Context (from documents/)

The design documents emphasize a fixed window contract and overlap-aware evaluation. The current repository follows that architecture. The uncovered-point problem is therefore not a mismatch between the design documents and the runtime path. It is a consequence of the exact window start rule currently implemented.

## Visual Evidence

See:

- [test-window-coverage-investigation.png](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-29-2026/research/test-window-coverage-investigation.png)

That figure shows:

- a toy sequence where the final point is uncovered
- the real STAFFIII tail, where the last 8 test points are uncovered under the active full config

## Open Questions

1. Should the benchmark protocol require strict full coverage of the raw test timeline, even when `stride` does not divide the test suffix cleanly?
2. Should uncovered points be excluded from metric vectors instead of remaining as zero-scored placeholders?
3. Should remainder-based tail loss be flagged by the audit layer even when `max_test_windows` is not configured?
