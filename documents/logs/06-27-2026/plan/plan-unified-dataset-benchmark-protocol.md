---
date: 2026-06-27 21:00:00 +0700
researcher: TheMetaSetter
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Plan for a unified dataset benchmark protocol"
tags: [plan, time-series, anomaly-detection, datasets, evaluation]
status: draft
last_updated: 2026-06-27
last_updated_by: TheMetaSetter
---

# Plan: Unified Dataset Benchmark Protocol

## Current State

- The repository already has a stable shared offline pipeline for anomaly detection experiments: parse raw sequences, clean them, fit `SequenceStandardScaler` on `train` only, transform all splits, slice windows inside each split, reconstruct pointwise test scores, and compute pointwise metrics. The decisive runtime path is centered on [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py), [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py), and [`src/metrics/pointwise.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/metrics/pointwise.py).
- The repository already has stable parser and builder abstractions in [`src/data/base.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/base.py). This is a good foundation for dataset-family-specific parsing under one shared benchmark protocol.
- The repository already has protocol-audit infrastructure in [`src/analysis/evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py), but that audit layer is currently partly shaped around `anomaly_archive`-specific comparison modes.
- The strongest protocol mismatch is that `comparison_mode` is still part of the active data contract in [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py), and the `anomaly_archive` parser still constructs anomaly-only or post-anomaly-only test slices in [`src/data/datasets/anomaly_archive.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py).
- The current runtime registry still supports only `smd` and `anomaly_archive`, but the codebase and local datasets already contain broader families such as `UCR`, `IOPS`, `SWaT`, and `NASA`. Therefore the future benchmark protocol should be defined at the family-contract level, not at the current registry snapshot level.

## Design Options

### Option A: Narrow runtime cleanup only

This option removes `comparison_mode`, hardens `smd` and `anomaly_archive`, and stops there.

Advantages:
- It is the smallest immediate code change.
- It minimizes short-term regression risk.
- It is easier to test in the current runtime because only two dataset names are active.

Disadvantages:
- It hardcodes today’s registry snapshot into the design.
- It does not produce a durable protocol surface for `SWaT`, `IOPS`, `NASA`, `UCR`, or future datasets.
- It will likely force another redesign when new dataset families are wired into the runtime.

### Option B: Unified benchmark protocol with dataset-family adapters

This option defines one benchmark contract for all dataset families and keeps dataset-specific logic only in parser adapters.

Advantages:
- It matches the user’s stated goal that the design must remain valid for future datasets published later.
- It keeps the shared pipeline stable while localizing dataset differences to parser logic and split-spec extraction.
- It is the cleanest long-term match to the existing `BaseSequenceParser` and `BaseDatasetBuilder` structure.

Disadvantages:
- It requires a broader design pass now.
- It will touch more config, audit, and public API surfaces.
- It needs more tests because the protocol must be proven on multiple dataset shapes.

### Option C: Full loader framework rewrite

This option would introduce a more abstract dataset framework with richer generic split objects, benchmark-policy classes, and a larger registry surface.

Advantages:
- It could model many dataset families under one explicit type system.
- It might reduce repeated code once many families are active.

Disadvantages:
- It violates the codebase’s current preference for small, direct, readable extensions.
- It adds abstraction cost before the benchmark contract itself is fully stabilized.
- It increases the probability of architecture drift away from the current repo style.

## Recommended Approach

Option B is the strongest fit.

The repository already has the right “thin waist” for this solution:

- dataset-specific parsing through parser classes,
- shared scale-and-window builder flow,
- shared evaluator,
- shared audit/report surfaces.

So the plan should not rewrite the framework. It should define one benchmark protocol that every dataset family must map into.

## Target Benchmark Contract

The unified protocol should define these rules for every dataset family:

1. `train`, `val`, and `test` are time-ordered split-local raw timelines.
2. `train` and `val` must come from the past relative to `test`.
3. `test` must be a future labeled timeline segment, not an anomaly-only slice and not a post-anomaly-only slice.
4. The protocol does not require every test window to contain both normal and anomalous points.
5. The protocol does require the reconstructed test timeline to be meaningful for pointwise anomaly detection.
6. The scaler fit scope is always `train_only_before_windowing`.
7. Windowization always happens after split parsing and after scaling.
8. Evaluation is defined on the reconstructed pointwise test timeline, not on standalone window labels.

This contract should be family-agnostic. A dataset family may differ in:

- where its raw train/test data come from,
- how labels are encoded,
- whether validation is derived from train or provided explicitly,
- whether the anomaly annotations are pointwise labels, interval metadata, or file-name metadata.

But every family must still map into the same runtime split contract above.

## Proposed Programming Scope

### 1. Remove protocol-special slicing from the benchmark runtime

Modify:

- [`src/data/datasets/anomaly_archive.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py)
- [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py)
- [`src/data/api.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/api.py)
- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py)

Planned contract change:

- remove `comparison_mode` and `inclusive_anomaly_end` from benchmark-facing data config,
- define one benchmark split path for `anomaly_archive`,
- keep any old analysis-only slicing logic outside the main runtime path if historical research scripts still need it.

### 2. Introduce one explicit split-semantics interface

Add a small, direct helper module, for example:

- `src/data/split_protocol.py`

Purpose:

- define the repository’s benchmark split contract in one place,
- expose validation helpers that all dataset parsers can call,
- avoid re-encoding the benchmark meaning separately inside each parser.

This module should stay simple. It does not need an inheritance-heavy framework. It only needs stable, explicit validation utilities such as:

- `validate_train_val_test_temporal_order(...)`
- `validate_labeled_test_timeline(...)`
- `describe_protocol_status(...)`

### 3. Keep dataset-family parsing separate from benchmark normalization and windowization

Preserve:

- `BaseSequenceParser` in [`src/data/base.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/base.py)
- the shared builder flow in [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/data/loaders.py)

Extend later through parser files such as:

- `src/data/datasets/swat.py`
- `src/data/datasets/iops.py`
- `src/data/datasets/nasa.py`
- a revised `src/data/datasets/anomaly_archive.py`

The spec should require every new dataset parser to emit the same raw split object shape already used today:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

### 4. Distinguish benchmark mode from smoke truncation mode

Modify:

- [`src/analysis/evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py)
- [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/engine/evaluator.py)

Planned additions:

- one explicit `protocol_status` field,
- one explicit `is_truncated_evaluation` boolean-style field retained through artifacts,
- one explicit `benchmark_comparability` field such as:
  - `full_timeline_comparable`
  - `truncated_smoke_not_comparable`
  - `invalid_test_regime`

### 5. Expand the registry in a forward-compatible way

Modify later:

- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/src/core/config.py)

Direction:

- replace the current narrow assumption “only `smd` and `anomaly_archive` exist” with a dataset registry that can safely grow,
- but do this with a small explicit set of parser registrations rather than a dynamic plugin framework.

This keeps the codebase simple while still honoring the requirement that more dataset families may be published later.

## Engineering Principles to Preserve

- Separation of concerns:
  dataset parsers define raw split semantics; the shared loader defines scaling and windowization; the evaluator defines reconstruction and scoring; the audit layer defines reporting.

- Single responsibility:
  no parser should also own generic threshold logic, no evaluator should reinterpret raw file-level dataset metadata, and no metric helper should silently repair broken split semantics.

- Composition over inheritance:
  use one shared protocol-validation helper and multiple parser adapters instead of a deep hierarchy of dataset-policy subclasses.

- Stable interfaces:
  keep the existing raw-sequence contract, batch contract, and evaluation-record contract stable wherever possible.

## Risk and Mitigation

- Risk: the implementation overfits to the current active registry and makes future dataset onboarding harder.
  Mitigation: write the spec at the dataset-family contract level and keep parser logic separate from shared benchmark logic.

- Risk: removing `comparison_mode` breaks old notes, tests, or one-off scripts.
  Mitigation: confine benchmark cleanup to the main runtime path first, and move any old slice analysis into clearly analysis-only modules if it must survive.

- Risk: smoke configs continue to be interpreted as benchmark results.
  Mitigation: require explicit protocol-status fields in saved artifacts and audit reports.

- Risk: developers begin rejecting datasets too aggressively because some test windows are all-zero or all-one.
  Mitigation: define validity at the reconstructed test timeline level, not at the individual window level.

- Risk: future families such as `IOPS`, `NASA`, or new public datasets have different annotation forms.
  Mitigation: make label extraction dataset-specific but require all families to emit the same raw split object and the same benchmark semantics.

- Risk: the spec becomes too abstract and diverges from the current repository style.
  Mitigation: keep the implementation centered on the current parser/builder/evaluator modules, using only one small additional split-protocol helper.

## Test Plan

The implementation plan should require suspicious, protocol-focused tests before code changes:

1. `anomaly_archive` benchmark parser test:
   verify that benchmark-mode `test` is a full future labeled segment, not an anomaly-only segment.

2. `smd` truncation audit test:
   verify that `max_test_windows` produces a clearly marked non-comparable smoke artifact.

3. shared split-protocol validation tests:
   verify temporal ordering, non-empty train, non-empty test, and labeled-test invariants.

4. future dataset-family contract tests:
   small synthetic fixtures for `UCR`-style interval metadata, `IOPS`-style label columns, and `NASA`-style anomaly intervals.

5. reconstruction integrity tests:
   verify that pointwise reconstruction and label propagation still align by index after protocol cleanup.

6. regression tests for public API:
   verify that `src/data/api.py` still produces coherent bundles after removing benchmark-invalid config keys.

## Validation Procedures

- Run focused config-load tests for the revised data config contract.
- Run protocol-audit tests that assert new status fields and warnings.
- Run evaluator tests that confirm reconstruction behavior is unchanged except for protocol metadata.
- Run at least one end-to-end offline smoke evaluation with a non-truncated main config and one truncated smoke config to prove the artifact distinction.

## Open Questions

1. Should the benchmark spec require every valid `test` timeline to contain at least one anomalous point, or should “all-normal official test split” remain valid for certain future dataset families if the raw benchmark itself is designed that way?
2. Should benchmark-incompatible datasets or sequences fail immediately, or should they be allowed only under a clearly marked research or smoke protocol status?
3. When `UCR/AnomalyArchive` is reintroduced under the unified contract, should the repository use the file’s official train/test boundary directly, or should it permit one configurable but benchmark-safe test-context extension rule around the anomaly as a secondary mode outside the main benchmark?

## Recommended Next Step

Move to `prompts/3_structure_prompt.md` and turn this plan into a staged implementation outline. The outline should decide the edit order across:

- config contract cleanup,
- dataset parser refactor,
- audit/report updates,
- test-first coverage for future dataset-family onboarding.
