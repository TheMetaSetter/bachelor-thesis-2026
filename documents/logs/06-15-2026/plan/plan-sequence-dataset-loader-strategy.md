---
date: 2026-06-15 00:00 +07:00
researcher: Artificial Intelligence Agent
git_commit: 866c225ed69e4c618718551558077669090411e4
branch: dev
repository: bachelor-thesis-2026
topic: "Sequence dataset loader strategy with normalize-before-windowing"
tags: [plan, time-series, anomaly-detection, loader, normalization, windowing]
status: draft
last_updated: 2026-06-15
last_updated_by: Artificial Intelligence Agent
---

# Plan: Sequence Dataset Loader Strategy with Normalize-Before-Windowing

## Current State

- The active offline dataset path is already centralized in `src/data/loaders.py`, where the SMD pipeline parses sequences, validates them, cleans metadata, fits a scaler on the training split, transforms all splits, and then builds fixed-length windows and `DataLoader` objects.
- The normalization step already happens before windowing in the current SMD implementation. `SequenceStandardScaler` in `src/data/scalers.py` fits on train sequences and transforms full sequences before `WindowDataset` materializes windows.
- Window overlap is already implemented as a sliding slice with stride control in `src/data/loaders.py` and `src/data/window.py`, so the current behavior naturally supports `0-19`, `1-20`, `2-21`, and similar overlapping windows when `window_size=20` and `stride=1`.
- The registry contract is already established in `src/core/registry.py`, and the experiment gate is centralized in `src/core/config.py`.
- The offline entrypoints `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py` all consume the dataset bundle through the registry, so changing the dataset abstraction in one place can keep the rest of the runtime stable.
- The public notebook-facing wrapper in `src/data/api.py` already exposes the dataset bundle as a stable mapping, which should be preserved.

## Design Choice

The simplest and most maintainable direction is to keep a single shared sequence dataset pipeline and introduce one dataset-level strategy object per dataset family. The strategy should own both raw parsing and split boundary resolution for that dataset. This avoids a larger abstraction tree while still preventing a one-off custom loader from becoming a maintenance burden.

This choice preserves the current SMD loader as the reference implementation and keeps the shared path:

`parse -> clean -> normalize -> window -> collate -> DataLoader`

No stage should move normalization after windowing.

## Recommended Implementation Slice

### Option selected

Use a **minimal shared builder + one dataset strategy** approach.

This is the best fit because:

- it keeps the codepath count low;
- it preserves the current SMD runtime contract;
- it allows future datasets to join the same pipeline without duplicating preprocessing logic;
- it matches the repository preference for readability and single-responsibility files.

## Files to Modify

### 1. `src/data/loaders.py`

Refactor the current SMD-specific builder into a shared builder shape that can call a dataset strategy.

Planned changes:

- keep `WindowDataset` behavior or move only the reusable window enumeration logic into a shared helper if needed;
- preserve the order `clean -> scaler.fit(train) -> scaler.transform(all) -> window`;
- add a shared builder entrypoint that receives a dataset strategy instead of hard-coding SMD behavior;
- keep the returned bundle contract stable: `dataset_name`, `parser`, `scaler`, `raw_sequences`, `scaled_sequences`, `datasets`, `loaders`.

### 2. `src/data/scalers.py`

Keep standardization as the default normalization mechanism.

Planned changes:

- add a small, explicit normalization mode if min-max normalization is required later;
- keep the current `fit` and `transform` behavior unchanged for the default path;
- ensure scaling is always applied on full sequences before window extraction.

### 3. `src/data/window.py`

Keep window slicing as the canonical overlapping-window utility.

Planned changes:

- preserve `slice_sequence_into_windows(...)` as the readable reference implementation;
- align any shared builder logic with this helper rather than duplicating stride logic;
- keep stride-based overlap semantics unchanged.

### 4. `src/data/base.py`

Keep the abstract builder contract small.

Planned changes:

- reuse the existing `BaseDatasetBuilder`;
- add only the minimal abstraction needed for a dataset strategy if the shared builder needs it;
- avoid deep inheritance.

### 5. `src/core/config.py`

Expand the config gate in one central place.

Planned changes:

- allow the new dataset name only after the shared builder path is ready;
- add normalization-related config keys only if they are needed and explicit;
- keep validation strict so unsupported keys fail early.

### 6. `src/core/registry.py`

Keep the registry pattern intact.

Planned changes:

- register the new dataset builder under a stable dataset name;
- avoid special casing in the training scripts;
- keep `build_dataset(name, ...)` as the sole dispatch entrypoint.

### 7. `src/data/api.py`

Keep the public notebook-facing interface synchronized with the shared builder.

Planned changes:

- expose the new dataset through the same bundle shape if notebook usage is needed;
- preserve the existing `PublicDataBundle` mapping contract.

### 8. `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_online_adaptation.py`

These should remain mostly unchanged.

Planned changes:

- verify they still resolve the dataset bundle through the registry;
- only touch them if a new config key or bundle field is strictly necessary;
- do not introduce dataset-specific branching here.

## Data Flow Contract

The implementation must preserve the following order:

1. load raw sequences
2. validate and clean sequence metadata
3. fit normalization on training data only
4. transform train / validation / test full sequences
5. slice transformed sequences into overlapping windows
6. collate windows into batches

This order is the key requirement from the latest research note and must not be reversed.

## Test Plan

Add or extend small tests so the loader contract is pressure-tested before the dataset is used in experiments.

### Loader and normalization tests

- verify that normalization happens before windowing on a small synthetic sequence;
- verify that `SequenceStandardScaler` still fits on train only;
- verify that overlapping windows are enumerated correctly for stride 1;
- verify that window counts match expectation for a short synthetic sequence.

### Contract tests

- verify the dataset bundle still exposes the same keys;
- verify the returned batch shape contract remains `[B, L, D]`;
- verify registry lookup works for the new dataset name;
- verify config validation rejects unsupported dataset values.

### Smoke test

- build the dataset bundle once end-to-end with a tiny sample file;
- ensure the output can be consumed by the current training entrypoint without changing trainer logic.

## Validation Procedure

1. Run the existing loader and window tests first to confirm the current SMD path still behaves the same.
2. Add the new dataset-specific test case with one annotated file and confirm the split boundary is respected.
3. Run the smallest available pytest subset around data loading and config validation.
4. Confirm that a training dry run still reaches dataset construction before any model changes are attempted.

## Risk and Mitigation

- Risk: moving abstractions too early may add complexity without reuse value. Mitigation: keep only one dataset strategy and one shared builder.
- Risk: normalization order may drift across codepaths. Mitigation: keep normalization in the shared builder before any window materialization and cover it with a dedicated test.
- Risk: overlap semantics may diverge between offline and future online paths. Mitigation: reuse the same sliding-window helper and keep stride semantics explicit in tests.
- Risk: config or registry changes may leak dataset-specific logic into scripts. Mitigation: centralize changes in `src/core/config.py` and `src/core/registry.py` only.

## Open Questions

- Should the first supported dataset after SMD be an annotation-aware archive such as STAFFIII / AnomalyArchive, or should the loader refactor land independently of the dataset addition?
- Do you want to keep standardization as the default normalization and add min-max only as an optional mode, or should both be exposed immediately?
- Should the new dataset strategy return train / val / test directly, or should it return raw annotated sequences and let the shared builder perform the final split?

## Recommended Next Step

Proceed with the shared builder refactor first, while keeping SMD behavior unchanged. After that, add the first dataset strategy on top of the stable pipeline and validate normalization-before-windowing with a small test file.
