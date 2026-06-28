---
date: 2026-06-27 14:05:00 +07 +0700
researcher: Artificial Intelligence Agent
topic: "Forensic loader and metric protocol audit after exact-run investigation"
status: complete
---

# Research: Forensic Loader and Metric Protocol Audit After Exact-Run Investigation

## Scope

This note continues the earlier exact-run investigation and broadens it into a stricter audit of the currently active dataset loaders, overlap reconstruction path, degenerate metric behavior, and reference-codebase protocol alignment. The goal is not to propose a redesign yet. The goal is to separate:

1. true protocol problems,
2. true implementation bugs,
3. mathematically expected degenerate metrics,
4. reader-facing tooling bugs that can mislead interpretation.

## Executive Findings

Two different failure modes are now strongly verified.

First, for `anomaly_archive` with `comparison_mode: pre_vs_anomaly`, the test split is all-positive by construction. That alone is enough to explain the reported bundle `precision = 1`, `recall = 0.05`, `pr_auc = 1`, `roc_auc = NaN`, `vus_pr = NaN`, without needing a model bug.

Second, for SMD smoke runs that use `max_test_windows`, the evaluation covers only an early prefix of the test timeline. That can produce all-zero reconstructed labels inside the evaluated coverage even though the raw test sequence contains real anomalies later.

These two failure modes are different in nature:

- `anomaly_archive pre_vs_anomaly` is a dataset-slicing protocol issue.
- `smd smoke max_test_windows` is an evaluation-coverage truncation issue.

Both can generate weird metrics, but for different reasons.

## Active Runtime Audit

The active runtime registry still supports only `smd` and `anomaly_archive` in [`src/core/config.py:299-313`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/core/config.py:299).

The shared loader path in [`src/data/loaders.py:135-174`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/data/loaders.py:135) does three things in a clean order:

1. parse raw split sequences,
2. fit `SequenceStandardScaler` on `train` only,
3. transform all splits, then windowize.

This part is aligned with standard no-leakage practice.

The key divergence starts later, at split semantics and test-window materialization.

## SMD Audit

The SMD parser in [`src/data/datasets/smd.py:61-166`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/data/datasets/smd.py:61) is semantically reasonable:

- `train` and `val` are split from the official SMD train series,
- `test` is the official test series,
- raw point labels are attached only to test.

This is the normal protocol shape.

However, `WindowDataset` in [`src/data/loaders.py:177-199`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/data/loaders.py:177) appends windows in increasing time order and stops immediately when `max_windows` is reached.

That means `max_test_windows` is not a harmless sample cap. It is a **prefix cut** on the test timeline.

For the inspected smoke config:

- raw test length of `machine-1-6` is `23689`,
- test windows materialized are only `64`,
- with `window_size = 20` and `stride = 1`, the evaluated coverage is only the earliest prefix up to around index `82`.

This is why the saved smoke `evaluation_records.json` can show all-zero labels even when the raw parser says the test series contains anomalies later.

So for SMD:

- the parser is not the main problem,
- the smoke truncation policy is the main protocol distortion.

## AnomalyArchive Audit

The AnomalyArchive parser in [`src/data/datasets/anomaly_archive.py:106-160`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:106) is where the severe semantic issue lives.

Under `pre_vs_anomaly`:

- `train_region_values = values[:anomaly_start_index]`
- `test_values = values[anomaly_start_index:anomaly_end_index]`
- `test_point_labels = torch.ones(test_values.size, dtype=torch.long)`

So the test set is not a future segment containing both normal and anomalous points. It is just the anomaly segment itself. The test labels are therefore all ones by construction.

Under `pre_vs_post`:

- test is cut from the post-anomaly segment,
- labels become all zero.

So both modes are protocol-specific slices, not ordinary anomaly-detection test splits.

This is the direct root cause of the weird STAFFIII metric bundle.

## Reconstruction and Metric Audit

The evaluator reconstructs window outputs back to timelines in [`src/engine/evaluator.py:44-140`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:44).

Three verified behaviors matter:

1. overlap scores are averaged correctly over covered timesteps,
2. uncovered suffix timesteps can remain inside the full-length record with zero score and zero label,
3. after per-entity reconstruction, entity arrays are concatenated into one global pointwise vector before metric computation in [`src/engine/evaluator.py:255-282`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py:255).

The metric helper in [`src/metrics/pointwise.py:381-478`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/src/metrics/pointwise.py:381) explicitly returns `NaN` for `vus_pr`, `vus_roc`, and `affiliation_f1` when there is only one unique label class.

That means:

- `roc_auc = NaN` for all-one or all-zero labels is expected,
- `vus_pr = NaN` for all-one or all-zero labels is expected,
- `affiliation_f1 = NaN` for all-one or all-zero labels is expected.

These are not evidence of broken formulas. They are evidence of a broken evaluation regime.

## Reference Codebase Cross-Check

The inspected reference loaders consistently follow the usual pattern more closely than the current `anomaly_archive` runtime.

In CANDI, the generic loader in [`bsc-thesis-ref-codebases/CANDI-main/datasets/build.py:114-138`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/datasets/build.py:114):

- splits `train` and `val` from training data,
- keeps `test` as a separate held-out series,
- fits scaler on train,
- applies transform to val and test.

For SMD in the same codebase, [`bsc-thesis-ref-codebases/CANDI-main/datasets/build.py:185-219`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/bsc-thesis-ref-codebases/CANDI-main/datasets/build.py:185) also uses train/test separation from official files and never replaces the whole test timeline with just the anomaly span.

CARLA’s SMD and SWAT loaders are older and use window-level labels rather than pointwise reconstruction, but they still preserve the core idea:

- train is built from train split,
- test is built from test split,
- scaling stats come from train-side data,
- the test timeline still comes from the official test partition rather than from an isolated anomaly-only slice.

So the strongest protocol mismatch against the references is not the shared scaling design. It is specifically the AnomalyArchive comparison-mode slicing.

## Secondary Tooling Finding

The new forensic helper script added earlier, [`scripts/forensic_audit_run.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC%20GIA%20TPHCM/%C4%90H%20KHOA%20HỌC%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/scripts/forensic_audit_run.py), has a reader-facing bug in its conclusion paragraph: it currently hard-codes AnomalyArchive wording even when the inspected run is SMD. The evidence sections above it are still useful, but the final paragraph is not yet dataset-aware.

This is a tooling bug in the audit helper, not a root-cause bug in the data or metric pipeline.

## Conclusions

The current state of the investigation supports four high-confidence conclusions.

First, the weird STAFFIII metric bundle is primarily explained by `anomaly_archive pre_vs_anomaly` producing an all-positive test set.

Second, the weird SMD smoke metrics are primarily explained by `max_test_windows` truncating evaluation to an early prefix of the test timeline.

Third, the pointwise metric formulas themselves are not the main culprit in these cases. They are reacting in mathematically expected ways to degenerate label regimes.

Fourth, the codebase still needs a clearer distinction between:

- full-timeline evaluation,
- truncated smoke evaluation,
- protocol-special research slices such as `pre_vs_anomaly` and `pre_vs_post`.

Without that distinction, readers can easily mistake a protocol artifact for a model result.
