# Sequence Dataset Loader Strategy Design

Date: 2026-06-15
Status: Draft
Scope: Loader architecture for SMD and future sequence datasets with annotation-aware splits

## Purpose

This note defines a reusable dataset-loading architecture for the thesis codebase. The goal is to avoid one-off custom loaders for each new time-series archive while keeping the existing SMD loader contract stable.

The preferred outcome is:

- one shared sequence dataset pipeline
- dataset-specific behavior isolated in small strategy objects
- one registry-facing entrypoint per dataset family
- no changes to trainer or evaluator semantics when a new dataset is added

This design follows `documents/design/design_starter.md` and the repository rule that the data path should stay thin, readable, and configuration-driven.

## Design Goals

1. Keep the current SMD pipeline usable as the reference implementation.
2. Support datasets whose train/test boundary is defined by filename annotation or explicit annotation markers.
3. Avoid duplicating preprocessing, scaling, and windowing logic across datasets.
4. Keep loader contracts stable so future models and experiments can reuse the same batch shape.
5. Make new dataset support a matter of adding a parser or split strategy, not rewriting the whole pipeline.

## Non-Goals

- Rewriting the trainer.
- Introducing a new batch schema.
- Moving dataset logic into model files.
- Adding dataset-specific hacks inside the training loop.

## Proposed Architecture

The data layer should be organized around a shared pipeline with small pluggable parts.

### Shared Pipeline

The common runtime steps are:

1. parse raw sequences
2. clean or normalize metadata
3. split into train / val / test
4. fit scaler on train only
5. transform all splits
6. windowize sequences
7. build PyTorch `Dataset` and `DataLoader`

This pipeline is invariant across datasets and should remain in one builder abstraction.

### Strategy Points

The dataset-specific variation should be isolated into the following strategies:

- parser strategy: how raw files are read and converted into full sequences
- split strategy: how boundaries between train, validation, anomaly, and test are computed
- root resolution strategy: how the dataset root is discovered
- download strategy: whether the dataset needs fetch logic
- window policy strategy: whether window slicing needs special rules

For STAFFIII / AnomalyArchive, the main variation is the split strategy because the boundary may come from the filename or an annotation block.

## Recommended Pattern

Use a combination of:

- `Registry` for dataset lookup by name
- `Template Method` for the shared dataset-building pipeline
- `Strategy` for dataset-specific parsing and splitting

This is better than a fully custom loader because the shared steps remain centralized and testable.

## Contracts

### Dataset Builder Contract

Every dataset builder should expose the same output shape:

```python
{
    "dataset_name": str,
    "parser": object,
    "scaler": object,
    "raw_sequences": dict[str, list[dict]],
    "scaled_sequences": dict[str, list[dict]],
    "datasets": dict[str, Dataset],
    "loaders": dict[str, DataLoader],
}
```

This keeps downstream code agnostic to the dataset family.

### Batch Contract

The batch format should remain compatible with the current SMD path:

```python
{
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The model and trainer should not need a second dataset-specific batch format.

## Split Policy

The split strategy must be explicit.

For annotation-aware archives:

- if filename encodes anomaly start/end, use that as the primary boundary source
- if an annotation block exists in the file, prefer that over inference from raw position
- if both exist and conflict, fail loudly unless a config flag explicitly selects one source

The split strategy should support at least these modes:

- `pre_vs_post`: train is the prefix before anomaly start, test is the suffix after anomaly end
- `pre_vs_anomaly`: train is the prefix before anomaly start, test is the annotated anomaly segment itself

The default should be the least ambiguous mode for evaluation reproducibility, which is `pre_vs_post`.

## Error Handling

The loader should fail early and explicitly for:

- unsupported `dataset_name`
- missing annotation boundary
- conflicting annotation sources
- sequence length shorter than window size
- unsupported config keys

Errors should be descriptive enough that a user can correct the config without tracing through the entire code path.

## Testing Plan

Minimal tests should cover:

1. registry lookup for the new dataset name
2. parser output shape for one sample file
3. split strategy on a file with annotation in filename
4. window counts for train / val / test
5. batch shape from the resulting DataLoader
6. checkpoint-free smoke build of the dataset bundle

The tests should be small and deterministic. They should not require rewriting the model side.

## Implementation Guidance

The code should stay readable under the repo’s `1 model - 1 file` and “least amount of codepaths” constraints.

Recommended file responsibilities:

- `src/data/loaders.py`: shared dataset builder logic and registry-facing entrypoints
- `src/data/datasets/<dataset_name>.py`: dataset-specific parser
- `src/data/splits.py` or similar: split strategies
- `src/data/window.py`: windowing logic if more datasets need reuse
- `src/data/base.py`: builder and parser abstract base classes

For the STAFFIII / AnomalyArchive case, the first extension should be a parser + split strategy, not a full custom loader branch.

## Decision

Adopt the shared pipeline with strategy-based extension points.

Do not add a one-off custom loader for STAFFIII / AnomalyArchive unless a future dataset genuinely breaks the shared pipeline contract.
