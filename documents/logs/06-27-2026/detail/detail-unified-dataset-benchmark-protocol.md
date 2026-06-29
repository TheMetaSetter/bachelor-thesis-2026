---
date: 2026-06-29 10:30:00 +0700
planner: Codex
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed plan for the unified dataset benchmark protocol"
tags: [detail, datasets, benchmark, loaders, evaluation, protocol, tests]
status: in_progress
last_updated: 2026-06-29
last_updated_by: Codex
source_structure: documents/logs/06-27-2026/structure/structure-unified-dataset-benchmark-protocol.md
source_plan: documents/logs/06-27-2026/plan/plan-unified-dataset-benchmark-protocol.md
source_research: documents/logs/06-27-2026/research/research-unified-dataset-benchmark-protocol-spec-context.md
---

# Detail: Unified dataset benchmark protocol

## Goal

Implement one simple, uniform benchmark contract for all dataset families in this repository: `train` and `val` must come from the past, `test` must be a future labeled timeline, reconstructed `test` must contain both `normal` and `anomaly`, and truncated smoke runs must remain runnable but must be marked `non-comparable`. The implementation must fix the current `anomaly_archive` protocol mismatch, preserve the existing scale-then-window pipeline, and stay easy to extend to future datasets such as `iccad`.

## Locked Decisions

The following rules are already settled and must be treated as fixed requirements during implementation:

1. A benchmark-valid `test` split must contain both `normal` and `anomaly` on the reconstructed pointwise timeline.
2. `train` and `val` must be temporally earlier than `test`.
3. `smoke` or truncated evaluations may remain runnable, but they must be labeled `non-comparable` and must not be treated as benchmark results.
4. `UCR / AnomalyArchive` must no longer use anomaly-only or post-only test slices inside the benchmark runtime.
5. The implementation must remain simple. Parser-specific logic should stay inside parser files, and the shared loader/evaluator path should only gain the smallest needed extensions.
6. The next benchmark-default stride policy is split-specific rather than one-size-fits-all:
   - `train_stride` may stay larger than `1` for computational practicality.
   - `val_stride` should be `1` when `val` is used as realistic checkpoint-selection evidence.
   - `test_stride` should be `1` for benchmark-comparable evaluation.
7. The simplest implementation path is to keep the existing `data.stride` as a backward-compatible fallback, then allow thin split-specific overrides such as `train_stride`, `val_stride`, and `test_stride`.
8. A benchmark-comparable `val` or `test` run must not leave a stride-remainder uncovered suffix on the reconstructed timeline.

## Progress Snapshot

As of `2026-06-29`, the plan is no longer at the original starting point. The codebase has already moved forward, so this detail document must be read as an in-progress execution log rather than a fresh greenfield plan.

- Phase 1 is effectively completed in the benchmark runtime:
  - benchmark `comparison_mode` and `inclusive_anomaly_end` were removed from the active config path.
- Phase 2 is effectively completed:
  - `src/data/split_protocol.py` exists and benchmark test labels are validated in the shared loader path.
- Phase 3 is effectively completed for benchmark runtime:
  - `anomaly_archive` now builds one future mixed-label `test` timeline instead of anomaly-only or post-only slices.
- Phase 4 is partially completed:
  - evaluator metrics and audit artifacts already expose `benchmark_comparability` and `protocol_status`.
- Phase 5 is now substantially completed:
  - suspicious reconstruction behavior is now evidenced and tested,
  - the evaluator computes metrics from actually covered points only,
  - saved evaluation artifacts persist `covered_point_mask`,
  - and the audit layer distinguishes capped smoke truncation from stride-remainder coverage loss.
- The highest-priority remaining work is now:
  - finish migrating all intended benchmark experiment configs onto the new benchmark-safe data/task settings,
  - keep smoke configs explicitly non-comparable,
  - and treat future dataset onboarding such as `SWaT`, `IOPS`, `NASA`, or `iccad` as separate parser/runtime work rather than assuming those families are already runnable.

## Current Code Reality

The detail plan is grounded in the current runtime:

- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/core/config.py) no longer allows benchmark `comparison_mode` or `inclusive_anomaly_end` in the active runtime path.
- [`src/data/datasets/anomaly_archive.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py) now builds one benchmark-safe future mixed-label `test` split.
- [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/loaders.py) already has the correct high-level order: parse, clean, fit scaler on `train`, transform all splits, then windowize.
- [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/loaders.py) now supports split-specific stride overrides with `data.stride` as the fallback.
- [`src/data/split_protocol.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/split_protocol.py) already centralizes the mixed-label benchmark test rule.
- [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py) now exposes `raw_num_points`, `evaluated_num_points`, `benchmark_comparability`, and `protocol_status`.
- [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py) now preserves raw point labels per entity and computes metric vectors from covered timesteps only.
- [`src/analysis/evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py) is already benchmark-generic in its public language.
- The main unresolved protocol bug is now narrower:
  - legacy experiment configs outside the new benchmark family still mostly point at `val_realistic` and older stride settings,
  - benchmark-safe configs should use split-specific coverage policy explicitly,
  - and future dataset families still need their own parser/runtime onboarding before they can join the same benchmark contract.
- The future dataset directory [`data/ibm-cloud-console-anomaly-dataset-iccad`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/data/ibm-cloud-console-anomaly-dataset-iccad) already shows a realistic future onboarding shape:
  - `anomaly_windows.csv` provides anomaly intervals,
  - `location_downtime.csv` provides location downtime intervals,
  - `unpivoted_data.parquet` stores row-wise metric events,
  - `pivoted_data_all.parquet` stores a wide multivariate matrix.

## Design Principles for This Implementation

This detail plan should preserve only the patterns that already help this codebase:

- **Composition over inheritance**: keep parsers separate, and add one thin shared validation helper instead of building a dataset-policy framework.
- **Adapter pattern**: each dataset parser is an adapter from raw files to the existing raw split contract `x`, `point_labels`, `mask`, `timestamps`, `meta`.
- **Thin registry**: keep explicit dataset registrations in config and builder code; do not create a dynamic plugin system.
- **Evaluator stays dataset-agnostic**: the evaluator should not learn dataset-specific split meaning.
- **Test-first for suspicious cases**: write failing tests for protocol violations before editing runtime code.

## Phase 1 - Lock the benchmark config contract

Status: effectively completed in the active benchmark runtime.

### Phase summary

This phase removes benchmark-invalid config semantics from the public runtime surface. The thesis objective here is to ensure that benchmark meaning is decided once, in config and parser contracts, before metrics and visualizations are trusted.

### Files to modify

- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/core/config.py)
- [`src/data/api.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/api.py)
- dataset YAMLs and tests that currently pass `comparison_mode`

### Planned edits

1. In `src/core/config.py`:
   - remove `comparison_mode` and `inclusive_anomaly_end` from `allowed_data_keys`
   - keep `max_train_windows`, `max_val_windows`, and `max_test_windows`, but document them as runtime caps that may make a run `non-comparable`
   - keep explicit dataset names for now, but prepare the validator so adding `iccad` later is one small extension

2. In `src/data/api.py`:
   - delete `comparison_mode` and `inclusive_anomaly_end` from `_build_anomaly_archive_data_config`
   - delete them from `load_anomaly_archive_data(...)`
   - keep the API shape otherwise stable

3. In configs/tests:
   - update any tests or helper configs that still pass `comparison_mode="pre_vs_anomaly"`
   - replace them with benchmark-safe configs

### Tests to write first

- Update [`tests/test_anomaly_archive_dataset_loader.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/tests/test_anomaly_archive_dataset_loader.py):
  - assert the parser no longer accepts benchmark `comparison_mode`
  - assert `test` labels are mixed rather than all-one or all-zero on benchmark-safe fixtures
- Add a config validation test:
  - passing `comparison_mode` in `data` must now fail loudly

### Acceptance criteria

- A benchmark experiment config containing `comparison_mode` or `inclusive_anomaly_end` fails validation.
- Public anomaly archive loading no longer exposes those arguments.
- No benchmark-facing loader path can silently construct anomaly-only or post-only test slices.

## Phase 2 - Introduce one thin split-protocol helper

Status: effectively completed.

### Phase summary

This phase centralizes benchmark validity rules in one small module so every dataset parser can be checked by the same logic. The thesis objective here is correctness and future extensibility, not architecture expansion.

### Files to create or modify

- Create [`src/data/split_protocol.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/split_protocol.py)
- Modify [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/loaders.py)
- Possibly modify [`src/data/base.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/base.py) only if a tiny hook is necessary

### Planned helper interface

`src/data/split_protocol.py` should stay very small and expose functions similar to:

- `summarize_split_label_regime(split_sequences) -> dict`
- `validate_benchmark_split_contract(parsed_sequences, dataset_name) -> dict`
- `validate_benchmark_test_labels(point_labels, dataset_name, entity_id) -> None`
- `describe_protocol_status(*, raw_test_num_points, evaluated_num_points, has_both_classes, is_truncated) -> dict`

The important point is that this is not a new class hierarchy. It is only a small shared rule module.

### Planned edits in loader path

1. After parser output is produced but before scaling:
   - validate split existence: `train`, `val`, `test`
   - validate non-empty `train` and `test`
   - validate labeled `test`
   - validate `test` contains both 0 and 1 in raw point labels for benchmark-valid families

2. After windowization:
   - preserve existing metadata
   - make truncation visible as protocol metadata, not hidden behavior

### Tests to write first

- Create `tests/test_split_protocol.py` with focused synthetic cases:
  - mixed-label future `test` passes
  - all-one `test` fails benchmark validation
  - all-zero `test` fails benchmark validation
  - empty `test` fails
  - missing labels fail

### Acceptance criteria

- There is exactly one shared place defining benchmark split validity.
- The loader path rejects benchmark-invalid parsed splits before training/evaluation proceeds.
- The new helper stays small and function-based rather than introducing a framework.

## Phase 3 - Rewrite `anomaly_archive` around the benchmark contract

Status: effectively completed for the benchmark runtime.

### Phase summary

This phase is the scientific core of the fix. It replaces the current special-slice parser behavior with one benchmark-safe future test timeline that includes both normal and anomaly whenever the raw series permits it.

### Files to modify

- [`src/data/datasets/anomaly_archive.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/datasets/anomaly_archive.py)
- [`tests/test_anomaly_archive_dataset_loader.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/tests/test_anomaly_archive_dataset_loader.py)

### Planned parsing rule

For the benchmark runtime, the parser should no longer ask “pre-vs-anomaly or pre-vs-post?”. Instead it should build:

- a past region for `train` and `val`
- one future `test` region that includes both:
  - some normal timesteps
  - the anomaly span

The simplest benchmark-safe default is:

- treat the filename metadata as the official temporal annotation,
- build `test` from a future segment that includes the anomaly interval and at least one normal region,
- fail loudly if the raw series cannot satisfy the mixed-label benchmark rule.

The detail implementation should avoid inventing many knobs here. One benchmark path is enough.

### Important metadata to preserve

Each raw sequence meta should clearly retain:

- `series_name`
- `source_file_name`
- `sequence_length`
- split-local `start_index` and `end_index`
- raw anomaly span indices if useful for debugging and visualization

### Tests to write first

- Rewrite the parser test so it asserts:
  - `train`, `val`, `test` each exist
  - `test.point_labels` contains both 0 and 1
  - `train` and `val` labels are all zero unless contaminated labels are explicitly supported later
- Add a failure test for a pathological file where no mixed-label future test can be built

### Acceptance criteria

- The anomaly archive parser no longer contains `pre_vs_anomaly` / `pre_vs_post` benchmark branches.
- Benchmark-safe anomaly archive `test` labels are mixed.
- Any anomaly archive file that cannot satisfy the benchmark rule fails loudly instead of yielding degenerate metrics.

## Phase 4 - Harden evaluation metadata and protocol audit

Status: partially completed.

### Phase summary

This phase makes benchmark validity visible in artifacts. The thesis objective here is that every run should explain exactly what was evaluated, even when the run is only a smoke run.

### Files to modify

- [`src/analysis/evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/analysis/evaluation_protocol_audit.py)
- [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py)
- [`src/metrics/pointwise.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/metrics/pointwise.py) only if wording/metadata additions are needed

### Planned edits

1. Remove anomaly-archive comparison-mode explanations from the audit layer.
2. Replace them with benchmark-generic protocol fields such as:
   - `protocol_status`
   - `benchmark_comparability`
   - `label_regime`
   - `raw_test_num_points`
   - `evaluated_num_points`
   - `is_truncated_evaluation`
   - `truncation_reason`

3. In the evaluator:
   - keep overlap averaging logic unchanged unless a real bug is found
   - keep entity concatenation behavior explicit in metadata
   - add a clear `benchmark_comparability` field to returned metrics

4. In audit markdown:
   - explain truncated smoke runs in plain language
   - explain that mixed-label benchmark validity is the scientific requirement

### Tests to write first

- Update [`tests/test_evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/tests/test_evaluation_protocol_audit.py):
  - replace comparison-mode assertions with benchmark-contract assertions
  - assert truncated SMD runs are marked `non_comparable`
  - assert full mixed-label runs are marked `full_timeline_comparable`

- Update evaluator tests:
  - assert metrics include comparability metadata
  - assert truncation flags remain consistent with record coverage

### Acceptance criteria

- Evaluation artifacts always say whether a run is benchmark-comparable.
- Truncated runs remain runnable but cannot be mistaken for benchmark results.
- Audit language no longer depends on removed anomaly archive modes.

## Phase 5 - Preserve current reconstruction logic, but test it more aggressively

Status: partially completed, but now materially safer than before.

### Phase summary

This phase does not assume the overlap reconstruction is wrong. It treats it as a suspicious component and locks its behavior with sharper tests. The thesis objective is to trust reconstruction only after index-level evidence is preserved.

### Files to modify

- [`src/engine/evaluator.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/engine/evaluator.py) only if metadata is missing
- [`tests/test_evaluator_thresholding.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/tests/test_evaluator_thresholding.py)
- [`tests/test_evaluation_protocol_audit.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/tests/test_evaluation_protocol_audit.py)

### Planned tests

1. Synthetic overlap test:
   - known window scores
   - known overlap counts
   - exact expected reconstructed point scores

2. Coverage-hole test:
   - confirm uncovered suffix remains outside evaluated coverage metadata
   - confirm uncovered points cannot silently pass as full benchmark coverage

3. Multi-entity test:
   - confirm concatenation across entities is explicit and measured
   - confirm per-entity records still preserve local coverage

4. Mixed-label validity test:
   - confirm benchmark validity is checked on the reconstructed timeline, not per-window labels

### Acceptance criteria

- Reconstruction behavior is index-verified by tests.
- Mixed-label benchmark validity is defined at the reconstructed timeline level.
- The codebase can explain exactly how many raw test points were actually covered.

## Phase 5A - Add split-specific stride with the smallest possible extension

### Phase summary

This is now the next immediate implementation slice. The goal is to preserve the current loader design while allowing benchmark-comparable `val` and `test` to use full-coverage window stepping without forcing the same computational cost on `train`.

Status: implemented in runtime, pending config rollout.

### Files to modify

- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/core/config.py)
- [`src/data/api.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/api.py)
- [`src/data/loaders.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/src/data/loaders.py)
- tests around config validation and loader behavior
- only the active benchmark configs that should become comparable under the new rule

### Planned edits

1. Keep `data.stride` as the existing default.
2. Add optional `data.train_stride`, `data.val_stride`, and `data.test_stride`.
3. Resolve each split stride with a simple fallback:
   - `resolved_train_stride = train_stride if provided else stride`
   - `resolved_val_stride = val_stride if provided else stride`
   - `resolved_test_stride = test_stride if provided else stride`
4. Keep `WindowDataset` unchanged except for receiving the already-resolved stride.
5. Update benchmark-target configs so `val` and `test` use `1` where comparability is intended.

### Tests to write first

1. Config validation test:
   - split-specific stride keys are accepted when positive and `<= window_size`
   - invalid split-specific stride values fail loudly
2. Loader resolution test:
   - when `val_stride` or `test_stride` is provided, the corresponding split uses that stride instead of the global fallback
3. Benchmark-coverage test:
   - with `test_stride=1` and no `max_test_windows`, a synthetic test split should have no stride-remainder uncovered suffix

### Acceptance criteria

- Existing configs that only use `data.stride` still work.
- Benchmark configs can opt into `val_stride=1` and `test_stride=1` without a loader redesign.
- The implementation stays function-based and local to config and loader code.

## Phase 5B - Stop treating uncovered points as ordinary evaluated points

### Phase summary

This phase is still required even after split-specific stride is added, because smoke or capped runs can still leave uncovered regions. The scientific rule is that pointwise metrics should be based on actually covered points, while full raw length should remain visible for audit and visualization.

Status: partially implemented in evaluator runtime. Full artifact persistence and downstream forensic propagation are still pending.

### Planned direction

1. Preserve full raw timeline metadata for plots and audit.
2. Preserve explicit evaluated coverage indices or a coverage mask in reconstructed records.
3. Compute benchmark metrics from covered points only when evaluation is truncated.
4. Keep truncated runs marked `non_comparable` even after this cleanup.

## Phase 6 - Prepare future dataset onboarding, including `iccad`

### Phase summary

This phase does not require implementing every future dataset immediately. Its purpose is to ensure the unified protocol can onboard them with small parser additions instead of another redesign.

### Files to plan for

- future parser: `src/data/datasets/iccad.py`
- future tests: `tests/test_iccad_dataset_loader.py`
- future config registration: `src/core/config.py`
- future public API extension: `src/data/api.py`

### ICCAD-specific observations from the current data directory

The dataset folder already suggests a realistic mapping strategy:

- [`anomaly_windows.csv`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/data/ibm-cloud-console-anomaly-dataset-iccad/anomaly_windows.csv) has anomaly interval metadata.
- [`location_downtime.csv`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20luận%20to%CC%82%CC%81t%20nghiệp/bachelor-thesis-2026/data/ibm-cloud-console-anomaly-dataset-iccad/location_downtime.csv) has location downtime intervals.
- `unpivoted_data.parquet` contains columns like `interval_start`, `location`, `kind`, `host`, `method`, `statusCode`, `endpoint`, `aggregated_stats_name`, `aggregated_stats_value`.
- `pivoted_data_all.parquet` is already a wide multivariate matrix with `interval_start` plus many metric columns.

### Planned onboarding rule for `iccad`

When `iccad` is implemented later, its parser must:

1. choose one concrete series definition, for example one location-level or one pivoted multivariate timeline,
2. derive pointwise labels from interval metadata,
3. emit `train`, `val`, `test` splits under the same benchmark contract,
4. fail loudly if a chosen series cannot produce a future mixed-label `test`.

This means the benchmark protocol remains stable even when the raw source format is completely different from `SMD` or `UCR`.

### Acceptance criteria

- The detail plan leaves a direct path to implement `iccad` without changing the shared benchmark contract.
- Future dataset onboarding remains parser-driven, not framework-driven.

## Test Execution Order

The implementation should follow this order strictly:

1. Write failing config validation tests.
2. Write failing split-protocol unit tests.
3. Rewrite anomaly archive parser tests to the new benchmark rule.
4. Update audit/evaluator tests for `non_comparable` and mixed-label benchmark semantics.
5. Implement runtime code changes phase by phase.
6. Run focused `pytest` subsets first.
7. Run one integration-style loader/evaluator path for `smd`.
8. Run one integration-style loader/evaluator path for benchmark-safe anomaly archive.

## Recommended Test Commands

Use `.venv/bin/python -m pytest` or `pytest` depending on the existing repo pattern, with the repo preference leaning toward `.venv/bin/python`.

Initial focused subsets:

```bash
.venv/bin/python -m pytest -q \
  tests/test_anomaly_archive_dataset_loader.py \
  tests/test_evaluation_protocol_audit.py \
  tests/test_evaluator_thresholding.py \
  tests/test_public_data_api.py \
  tests/test_smoke_loader_limits.py
```

Then add the new split-protocol test file:

```bash
.venv/bin/python -m pytest -q tests/test_split_protocol.py
```

## Risks and Mitigations

### Risk 1 - Benchmark fix accidentally becomes another abstraction project

Mitigation:
- create only one helper module
- keep parser classes as the only dataset-specific adapters
- do not introduce dataset-policy class trees

### Risk 2 - AnomalyArchive still hides single-class test regimes under a new name

Mitigation:
- enforce the mixed-label benchmark rule in the parser and shared validator
- write tests that fail on all-one and all-zero test labels

### Risk 3 - Smoke results still leak into benchmark summaries

Mitigation:
- persist `benchmark_comparability`
- persist `truncation_reason`
- require summary/report code to surface these fields

### Risk 4 - Future datasets such as `iccad` force another redesign

Mitigation:
- freeze the benchmark contract now
- treat each future dataset as a parser-mapping problem

## Final Acceptance Criteria

The work is complete only when all of the following are true:

1. The benchmark runtime no longer supports `pre_vs_anomaly` or `pre_vs_post`.
2. Benchmark-valid `test` splits are enforced to contain both normal and anomaly points on the reconstructed timeline.
3. Truncated smoke runs remain executable but are always marked `non_comparable`.
4. The audit artifact and evaluation metadata clearly expose comparability, coverage, and label regime.
5. `SMD` still works under the shared pipeline without benchmark-semantic regression.
6. `AnomalyArchive` is benchmark-safe under the new parser contract.
7. The code structure remains simple enough that a future `iccad` parser can be added without changing the shared contract.

## Recommended Immediate Implementation Scope

For the first coding pass, implement only:

- Phase 1
- Phase 2
- Phase 3
- the metadata part of Phase 4
- the tests in Phase 5 that directly guard these changes

Do not implement `iccad` runtime support in the same pass unless the benchmark contract is already stable. The correct practical order is:

1. fix the shared benchmark meaning,
2. prove it on `smd` and `anomaly_archive`,
3. then onboard `iccad` as a new parser using that frozen contract.
