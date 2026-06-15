---
date: 2026-06-15 00:00 +07:00
researcher: Artificial Intelligence Agent
git_commit: 866c225ed69e4c618718551558077669090411e4
branch: dev
repository: bachelor-thesis-2026
topic: "Sequence dataset loader strategy with normalize-before-windowing"
tags: [detail, time-series, anomaly-detection, loader, normalization, windowing]
status: draft
last_updated: 2026-06-15
last_updated_by: Artificial Intelligence Agent
---

# Detailed Plan: Sequence Dataset Loader Strategy with Normalize-Before-Windowing

## Scope

This detailed plan covers a data-layer refactor only. The objective is to preserve the current SMD runtime contract while making it possible to add annotation-aware sequence datasets without creating a one-off custom loader.

The plan enforces one hard rule:

- normalize full sequences first
- then slice overlapping windows

This ordering must remain consistent across all offline dataset builders and any future dataset strategy built on the same shared pipeline.

## Current Implementation Status

The repository already contains the core pieces needed for this refactor:

- `src/data/loaders.py` currently implements the full SMD pipeline end to end.
- `src/data/scalers.py` already performs train-only fitting and full-sequence normalization.
- `src/data/window.py` already provides a reusable sliding-window helper with overlap semantics.
- `src/core/registry.py` already dispatches dataset builders by name.
- `src/core/config.py` already centralizes config validation and dataset-name gating.

So the remaining work is not to invent a new data system. The remaining work is to factor the current behavior into a shared builder plus one dataset strategy, while keeping the observable runtime contract unchanged.

## Phase 1: Freeze the Existing Contracts and Define the Shared Dataset Strategy Interface

### Phase summary

The first phase establishes the smallest possible abstraction boundary for the dataset layer. The goal is to keep the current runtime stable while defining one shared interface that can support both SMD and future annotation-aware archives.

### File-level edits

#### `src/data/base.py`

Add a minimal strategy interface for sequence datasets without introducing deep inheritance.

Planned edit content:

- keep `BaseDatasetBuilder` unchanged as the shared builder contract;
- add a small abstract base class or protocol-like interface for a dataset strategy, such as `BaseSequenceDatasetStrategy`;
- require the strategy to own raw parsing and split resolution for one dataset family;
- keep the interface intentionally small, for example:
  - `build(data_config: dict[str, Any]) -> dict[str, Any]`
  - or `load_sequences(...)` plus `split_sequences(...)` if the implementation needs a slightly finer decomposition;
- avoid forcing strategies to know about training loops, checkpointing, or model code.

#### `src/core/contracts.py`

Freeze the sequence and batch contract used by the shared loader path.

Planned edit content:

- keep the current raw sequence contract intact;
- ensure `validate_raw_sequence(...)` continues to enforce `[T, D]` full sequences before windowing;
- keep `validate_batch(...)` unchanged so all models continue to consume the same `[B, L, D]` structure;
- do not add dataset-specific batch branches.

#### `src/core/config.py`

Prepare the config gate for one additional dataset name and optional normalization settings.

Planned edit content:

- keep the current SMD config schema valid;
- add the new dataset name only after the shared builder is ready;
- add a normalization mode key only if it is explicitly needed, for example `normalization: standard` or `normalization: minmax`;
- keep unsupported keys failing early.

### Interface and contract definitions

- **Dataset strategy contract**: one strategy per dataset family should own raw file parsing and split boundary resolution.
- **Raw sequence contract**: each full sequence must expose `x`, optional labels, optional mask, optional timestamps, and metadata.
- **Batch contract**: the loader must still emit fixed windows with tensor shape `[B, L, D]` and the existing metadata structure.
- **Encoder, model, task, engine contracts**: no changes in this phase. They remain frozen dependencies and must continue to consume the same batch shape.
- **Normalization contract**: the loader must normalize the full sequence before any window helper or window dataset sees the data.
- **Overlap contract**: the shared slicing behavior must continue to support stride-based overlap, including the `0-19`, `1-20`, `2-21` pattern when `window_size=20` and `stride=1`.

### Acceptance criteria

- The repository still validates the existing SMD config without changing any model or engine code.
- The shared strategy interface exists in one place and is small enough to read in one pass.
- No training script changes are needed in this phase.
- The batch contract remains unchanged after refactoring the interface.
- The documented normalization-before-windowing rule is explicit in the code path and in the tests.

## Phase 2: Refactor the Shared Dataset Builder to Normalize Before Windowing

### Phase summary

This phase moves the loader logic toward a shared builder implementation while keeping the current SMD path behaviorally stable. The emphasis is on preserving the exact order of preprocessing stages and removing dataset-specific hard-coding from the builder body.

### File-level edits

#### `src/data/loaders.py`

Refactor the current SMD-specific builder into a shared builder that can call a dataset strategy.

Planned edit content:

- preserve the current `WindowDataset` behavior or extract only the reusable index enumeration logic into a helper if that improves readability;
- keep the sequence of operations exactly as:
  1. parse raw sequences
  2. clean metadata
  3. fit `SequenceStandardScaler` on train only
  4. transform all splits
  5. build overlapping windows
  6. construct `DataLoader` objects
- move the hard-coded SMD parse/split steps behind the strategy interface;
- keep the returned bundle keys unchanged:
  - `dataset_name`
  - `parser`
  - `scaler`
  - `raw_sequences`
  - `scaled_sequences`
  - `datasets`
  - `loaders`
- preserve `collate_windows` and the current `shuffle_train`, `num_workers`, and `persistent_workers` behavior.

#### `src/data/scalers.py`

Keep standardization as the default normalization path and make any alternative mode explicit.

Planned edit content:

- keep the current `SequenceStandardScaler` behavior unchanged for the default path;
- if min-max normalization is needed, add it as a clearly named option rather than a silent behavioral branch;
- make sure full sequences are transformed before `WindowDataset` or `slice_sequence_into_windows(...)` runs;
- keep state dict round-tripping stable.

#### `src/data/window.py`

Keep window slicing as the canonical overlapping-window utility.

Planned edit content:

- preserve `slice_sequence_into_windows(...)` as the reference implementation for overlapping windows;
- keep stride-based overlap semantics explicit and testable;
- ensure the helper remains aligned with the shared builder if the builder stops duplicating window enumeration logic;
- do not alter the meaning of `window_size` or `stride`.

#### `src/data/cleaning.py`

Keep cleaning conservative.

Planned edit content:

- retain metadata validation and annotation-only cleaning behavior;
- do not introduce dataset-specific heuristics in the cleaning layer;
- keep cleaning before normalization so malformed metadata fails before data statistics are computed.

### Design pattern application

- **Composition over inheritance**: the shared builder composes a dataset strategy instead of subclassing one builder per dataset family.
- **Registry pattern**: `src/core/registry.py` remains the single entrypoint for dataset dispatch.
- **Strategy pattern**: one dataset strategy object encapsulates parse-and-split behavior for each dataset family.
- **Template Method**: the shared builder keeps the invariant preprocessing order fixed.

### Acceptance criteria

- For the existing SMD path, normalized sequences are still produced before any window slicing occurs.
- Overlapping windows for `window_size=20` and `stride=1` still correspond to `0-19`, `1-20`, `2-21`, and so on.
- The SMD bundle returned by `build_dataset(...)` still exposes the same top-level keys and remains consumable by current entrypoints.
- The refactor does not introduce additional model or engine dependencies.
- The shared builder and strategy are readable from top to bottom without needing to inspect the trainer.

## Phase 3: Add One Dataset Strategy for an Annotation-Aware Archive

### Phase summary

This phase adds the first non-SMD dataset strategy using the shared builder. The strategy should own both parsing and split boundary resolution so the rest of the pipeline can stay generic.

### File-level edits

#### `src/data/datasets/<dataset_name>.py`

Add one dataset-family module for the first annotation-aware archive.

Planned edit content:

- implement a single strategy class, such as `AnnotationAwareSequenceDatasetStrategy`;
- make the strategy parse the raw file format used by the target archive;
- make the strategy resolve the split boundary from filename annotations or annotation markers;
- return the same bundle shape as the SMD builder path;
- keep the strategy readable and self-contained in one file.

#### `src/data/loaders.py`

Register the new strategy and route it through the shared builder.

Planned edit content:

- add a dataset-name-to-strategy mapping, or register the strategy through the existing registry path;
- keep the shared builder free of dataset-specific if/else blocks beyond dispatch;
- preserve the same output contract after dispatch.

#### `src/core/config.py`

Allow the new dataset name and only the config keys that the strategy truly needs.

Planned edit content:

- extend the supported dataset name set;
- keep validation strict for unsupported fields;
- if the dataset uses split metadata from filenames, keep that behavior explicit in config or strategy rather than inferred implicitly.

#### `src/data/api.py`

Expose the new dataset through the notebook-facing wrapper only if the public API should support it immediately.

Planned edit content:

- keep `PublicDataBundle` unchanged unless the new dataset needs an additional field;
- preserve the mapping-like contract for downstream analysis notebooks.

### Interface and contract definitions

- **Dataset strategy output**: a dataset strategy must return train / validation / test sequences or an equivalent raw-sequence bundle that the shared builder can transform without special casing.
- **Split contract**: the strategy must define whether it uses `pre_vs_post` or another explicit boundary mode.
- **Window contract**: after normalization, the same fixed-length overlapping window logic must be applied.

### Acceptance criteria

- The new dataset can be built through `build_dataset(...)` without changing the trainer.
- Split boundaries are deterministic and come from the annotation source selected by the strategy.
- The loader still emits normalized full sequences before any window creation step.
- The new strategy does not require a separate custom loader entrypoint outside the shared builder.
- The first dataset strategy remains single-file and self-contained.

## Phase 4: Add Tests and Validation for Ordering, Shape, and Registry Behavior

### Phase summary

This phase validates that the refactor is actually stable, not just structurally cleaner. The tests should confirm normalization order, overlap behavior, bundle shape, and registry integration.

### File-level edits

#### `tests/test_windowizer.py`

Extend or add tests for overlap semantics.

Planned edit content:

- verify that `slice_sequence_into_windows(...)` returns overlapping slices in the expected order;
- verify that `stride=1` yields windows shifted by exactly one timestep;
- verify that empty output is returned when the sequence is shorter than the window size.

#### `tests/test_data_cleaning_pipeline.py` or a new loader-focused test file

Add a small test for normalize-before-windowing.

Planned edit content:

- construct a tiny synthetic sequence bundle;
- fit the scaler on train data only;
- transform the sequence and then window it;
- assert that window materialization sees normalized values rather than raw values.

#### `tests/test_public_data_api.py`

Keep the public wrapper aligned with the bundle contract.

Planned edit content:

- verify the wrapper still exposes the same mapping keys;
- verify that the new dataset, if exposed publicly, returns a compatible bundle.

#### `tests/test_config_loading.py`

Extend config validation tests.

Planned edit content:

- confirm the new dataset name is accepted only when intended;
- confirm unsupported config keys still fail;
- confirm normalization-related config values are parsed and preserved if added.

#### New loader smoke test file if needed

Add a minimal bundle construction smoke test.

Planned edit content:

- build the dataset bundle end to end using a tiny fixture;
- verify that train / validation / test loaders are present;
- verify that a single batch matches the current shape contract.

### Test plan

- **Unit tests for data shapes**: validate raw sequence shape, window shape, and batch shape.
- **Integration test for a single training step**: run only if the refactor changes the bundle contract enough to justify a smoke pass through `scripts/train.py`; otherwise keep the scope at data-layer smoke tests.
- **Registry test**: confirm the new dataset name is resolved through `src/core/registry.py`.
- **Ordering test**: confirm normalization is applied before any windowing helper runs.

### Acceptance criteria

- The smallest meaningful pytest subset passes after the refactor.
- The loader outputs remain shape-compatible with existing models and the engine.
- The new dataset strategy is reachable through the registry.
- The tests fail if normalization is accidentally moved after windowing.
- The tests are small enough to run as a loader smoke suite before any full training run.

## Cross-Cutting Risk Mitigation

### Prototype redundancy

This slice does not modify prototype branches, but the refactor must not force any model-side fallback path or duplicate batch schema. Mitigation is to keep the data contract unchanged so prototype modules remain untouched.

### Fusion collapse

This slice does not change fusion logic. Mitigation is to avoid introducing any dataset-specific metadata that would alter model-side branch selection or loss weighting.

### Adaptation contamination

This slice does not change online adaptation. Mitigation is to keep the offline and online data contracts aligned so the online path can continue to consume the same normalized window format.

### Projector drift

This slice does not change the projector. Mitigation is to leave the online adaptation implementation untouched and verify the shared batch format still matches its expectations.

### Evaluation metric inflation

This slice does not change evaluation metrics. Mitigation is to keep the evaluation code paths stable and ensure the new dataset does not bypass the existing window reconstruction and metric aggregation behavior.

## Overall Acceptance Criteria

The detailed plan is complete when all of the following are true:

1. The data path still normalizes full sequences before windowing.
2. The shared builder can dispatch through one dataset strategy without custom loader duplication.
3. The current SMD path remains behaviorally stable.
4. The first annotation-aware dataset can be added without changing trainer or evaluator logic.
5. Tests cover window overlap, normalization order, registry lookup, and bundle shape.
6. No model, task, or engine contract is modified as a side effect of the loader refactor.
7. The plan is implementable in a minimal vertical slice, with no requirement to touch model files for this specific loader task.

## Recommended Execution Order

1. Freeze the interface in `src/data/base.py`, `src/core/contracts.py`, and `src/core/config.py`.
2. Refactor the shared builder in `src/data/loaders.py`.
3. Keep normalization behavior explicit in `src/data/scalers.py`.
4. Add the first dataset strategy in `src/data/datasets/<dataset_name>.py`.
5. Extend tests in `tests/`.
6. Run a small smoke build and verify the new dataset path still fits the existing training entrypoints.

## Final Recommendation

Keep the implementation as small as possible:

- one shared dataset builder
- one dataset strategy per dataset family
- one normalization path
- one overlapping-window helper

That is the simplest design that still gives the codebase a stable long-term maintenance path.
