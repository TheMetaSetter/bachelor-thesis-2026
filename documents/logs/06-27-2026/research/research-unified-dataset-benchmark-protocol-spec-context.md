---
date: 2026-06-27 20:48:21 +0700
researcher: TheMetaSetter
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Research context for a unified dataset benchmark protocol spec"
tags: [research, time-series, anomaly-detection, loaders, evaluation]
status: complete
last_updated: 2026-06-27
last_updated_by: TheMetaSetter
---

# Research: Research context for a unified dataset benchmark protocol spec

**Date**: 2026-06-27 20:48:21 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `89a598f643cf0c20b0ab540b926e6b71f27e975f`  
**Branch**: `dev`

## Research Question

Before writing a unified benchmark protocol specification for all datasets, what is the current loader, split, scaling, windowization, reconstruction, and metric behavior of the repository as it exists today?

## Summary

The current repository already has a fairly clear shared pipeline for offline window-based anomaly detection, but its split semantics are not yet unified at the benchmark-protocol level.

At runtime, the active dataset registry still supports only `smd` and `anomaly_archive` in [`src/core/config.py:299-313`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:299). The shared loader path does parse raw split sequences, fit `SequenceStandardScaler` on `train` only, transform all splits, and then windowize. That part is well aligned with leakage-safe practice.

The main protocol problem is later in the pipeline. The repository still permits `anomaly_archive` comparison-mode slicing that replaces a full labeled future test segment with anomaly-only or post-anomaly-only slices. The repository also allows split-local window truncation via `max_test_windows`, which can silently turn full-timeline evaluation into early-prefix evaluation.

So the correct foundation for the future spec is this:

1. the shared scale-then-window workflow is mostly sound,
2. the benchmark split contract is not yet unified,
3. `comparison_mode` is the strongest remaining protocol outlier,
4. smoke truncation must not be treated as ordinary benchmark evaluation.

## Detailed Findings

### Data Preparation

The codebase exposes a shared bundle builder in [`src/data/loaders.py:135-174`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:135). The order is:

1. clean parsed raw sequences,
2. fit `SequenceStandardScaler` on `train` only,
3. transform `train`, `val`, and `test`,
4. create `WindowDataset` objects per split,
5. expose PyTorch loaders.

This means the current repository already follows the important no-leakage normalization rule: fit on training data only, then apply the transform to validation and test.

The windowizer itself is in [`src/data/loaders.py:177-240`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:177). It builds windows by iterating forward in time with:

- a fixed `window_size`,
- a fixed `stride`,
- optional hard caps such as `max_test_windows`.

Each emitted window carries `start_index`, `end_index`, `entity_id`, `series_id`, and `source_sequence_length` in metadata. This is useful because later reconstruction logic can map window outputs back onto original split-local time indices.

### Dataset Split Semantics

The current config contract still allows only two runtime dataset names, `smd` and `anomaly_archive`, in [`src/core/config.py:269-313`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:269). It also still explicitly allows:

- `comparison_mode`,
- `inclusive_anomaly_end`,
- `max_train_windows`,
- `max_val_windows`,
- `max_test_windows`.

For `smd`, the split shape is semantically normal. Earlier audit work already established that the parser uses official train series for `train` and `val`, and official test series for `test`. The main risk for `smd` is therefore not split meaning, but truncation after split construction.

For `anomaly_archive`, the current parser in [`src/data/datasets/anomaly_archive.py:94-179`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:94) is the protocol outlier:

- under `pre_vs_anomaly`, `test_values = values[anomaly_start_index:anomaly_stop_index]`,
- under `pre_vs_post`, `test_values = values[anomaly_end_index:]`,
- test labels become all ones or all zeros respectively.

So as implemented today, `comparison_mode` does not define a normal future test timeline. It defines a special slice regime.

### Window Labels and Timeline Reconstruction

The evaluator is explicitly pointwise and overlap-aware. It reconstructs window outputs back onto the original split-local test timeline in [`src/engine/evaluator.py:44-142`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:44).

The implementation does three things:

1. sum point scores over overlapping windows,
2. count how many windows covered each time index,
3. average the score at each covered timestep.

Labels are aggregated by `torch.maximum` over overlapping windows. This means if any covering window marks a timestep as anomalous, the reconstructed timestep label stays anomalous.

This reconstruction logic matches the current pointwise-evaluation design. It also means that a window whose label vector is all ones is not automatically a problem. It becomes a problem only if the entire reconstructed test timeline is all ones because the split was built that way.

### Metric Computation

After reconstruction, the evaluator concatenates entity-level records into one global pointwise vector before metric computation in [`src/engine/evaluator.py:255-324`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:255).

Threshold selection is currently score-quantile based:

- use the 95th percentile of positive scores if they exist,
- otherwise use the 95th percentile of all scores,
- if that threshold collapses to zero but positive scores exist, use the minimum positive score.

Pointwise metrics are then computed in [`src/metrics/pointwise.py:542-607`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/metrics/pointwise.py:542). The metric path now includes diagnostics such as:

- `unique_labels`,
- `n_pos`,
- `n_neg`,
- `positive_ratio`,
- score summary fields,
- threshold,
- `affiliation_f1`,
- `vus_pr`,
- `vus_roc`.

The helper uses `_safe_metric`, so single-class label regimes naturally produce `NaN` for metrics such as ROC-AUC when the metric is undefined. Therefore weird bundles such as:

- `precision = 1`,
- `recall` very low,
- `pr_auc = 1`,
- `roc_auc = NaN`,
- `vus_pr = NaN`

can be mathematically expected when the split itself is degenerate.

### Audit and Reporting Surfaces

The repository already contains a protocol-audit helper in [`src/analysis/evaluation_protocol_audit.py:128-399`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py:128). It already records:

- split label regimes,
- evaluated points versus raw points,
- truncated coverage,
- scaler fit scope,
- comparison-mode explanation,
- warnings for single-class test labels.

This matters for the future spec because the repository is no longer missing audit infrastructure entirely. The stronger problem is that the data contract still allows protocol-special modes to sit inside the normal runtime path.

## Code References

- [`src/core/config.py:269`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:269) - active data config keys still include `comparison_mode` and window caps
- [`src/core/config.py:299`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py:299) - active runtime datasets are only `smd` and `anomaly_archive`
- [`src/data/loaders.py:65`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:65) - shared window dataset builder path
- [`src/data/loaders.py:150`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:150) - scaler is fit on `train` only before windowization
- [`src/data/loaders.py:177`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py:177) - windowization and `max_windows` truncation behavior
- [`src/data/datasets/anomaly_archive.py:106`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:106) - `pre_vs_anomaly` anomaly-only test slicing
- [`src/data/datasets/anomaly_archive.py:114`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py:114) - `pre_vs_post` post-anomaly-only test slicing
- [`src/engine/evaluator.py:44`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:44) - overlap-aware pointwise reconstruction
- [`src/engine/evaluator.py:255`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py:255) - concatenation of entity-level records into one global pointwise vector
- [`src/metrics/pointwise.py:542`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/metrics/pointwise.py:542) - pointwise metrics and diagnostics
- [`src/analysis/evaluation_protocol_audit.py:207`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py:207) - audit report generation

## Pipeline Documentation

As of this snapshot, the repository’s implemented offline evaluation pipeline is:

1. parse raw sequences into split-local timelines,
2. clean sequences,
3. fit scaler on `train` only,
4. transform all splits,
5. cut overlapping windows inside each split,
6. run the model to produce per-window point scores,
7. average overlap scores back to split-local pointwise timelines,
8. concatenate entity-level timelines,
9. choose one global score threshold,
10. compute pointwise and range-based metrics.

This pipeline is coherent if the splits themselves are coherent. Therefore the strongest remaining benchmark-spec responsibility is to define valid split semantics for all dataset families and to prohibit protocol-special slices from pretending to be ordinary benchmark test sets.

## Historical Context (from documents/)

The research log already contains direct evidence that:

- `pre_vs_anomaly` explains the previously observed all-positive STAFFIII behavior,
- `max_test_windows` explains truncated SMD smoke coverage,
- reference codebases more often preserve full future test segments than anomaly-only slices.

See:

- [`documents/logs/06-27-2026/research/research-forensic-loader-and-metric-protocol-audit.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-27-2026/research/research-forensic-loader-and-metric-protocol-audit.md)
- [`documents/logs/06-27-2026/research/anomaly_archive_staffiii_pre_vs_anomaly__forensic_audit_v2.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-27-2026/research/anomaly_archive_staffiii_pre_vs_anomaly__forensic_audit_v2.md)
- [`documents/logs/06-27-2026/research/smd_smoke_redlamp_machine_1_6_seed6__forensic_audit_v2.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-27-2026/research/smd_smoke_redlamp_machine_1_6_seed6__forensic_audit_v2.md)

## Open Questions

1. Should the future unified spec support only the datasets already active in the runtime registry first, or should it define a broader family-level contract immediately for `SMD`, `SWaT`, `IOPS`, `NASA`, and `UCR/AnomalyArchive` even before all are wired into `src/core/config.py`?
2. Should benchmark-incompatible artifacts such as truncated smoke evaluations be rejected at runtime, or preserved with explicit non-comparable protocol status fields?
3. Should the future spec define a minimum acceptable test condition such as “at least one positive test point and at least one window,” or should it require a stronger mixed-label full test timeline condition for benchmark mode?
