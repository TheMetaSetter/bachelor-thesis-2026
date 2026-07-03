---
date: 2026-07-01 14:40:19 +07
researcher: TheMetaSetter
git_commit: 0fd6abfb39ed704bf41a83845ec826b577ebdd94
branch: dev
repository: bachelor-thesis-2026
topic: "Whether validation can use non-overlapping windows with stride equal to window size"
tags: [research, validation, stride, windowing, overlap]
status: complete
last_updated: 2026-07-01
last_updated_by: TheMetaSetter
---

# Research: Whether validation can use non-overlapping windows with stride equal to window size

**Date**: 2026-07-01 14:40:19 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: `0fd6abfb39ed704bf41a83845ec826b577ebdd94`  
**Branch**: `dev`

## Research Question

Can the validation set use `stride = window_size` so that validation windows do not overlap? If so, how should that be implemented in the current codebase, and what are the consequences for validation metrics and coverage?

## Summary

Yes. The current loader already supports split-specific stride values, so validation can use `stride = window_size` without changing the windowing architecture. The simplest implementation is to set `val_stride` to the same value as `window_size` in the relevant data config files. For the active SMD benchmark, that means changing the benchmark data YAMLs from `val_stride: 1` to `val_stride: 20` when `window_size: 20`.

This change makes validation windows non-overlapping. The evaluator and training validation loop remain compatible because they already reconstruct pointwise labels and scores from window payloads and then compute metrics only on covered points.

The main caveat is coverage. `WindowDataset` uses a stepwise `range(0, sequence_length - window_size + 1, stride)` loop, so when `stride = window_size`, any trailing remainder shorter than one full window is dropped. The code does not automatically add an end-aligned final window. Therefore, non-overlapping validation is safe for the current pipeline, but it may leave uncovered tail points and reduce the number of evaluated timesteps.

## Detailed Findings

### Data Preparation

The split-specific stride is already explicit in the loader:

- `src/data/loaders.py:66-74`
- `src/data/loaders.py:87-97`
- `src/data/api.py:18-54`
- `src/data/api.py:80-111`

The loader resolves stride with this precedence:

1. `train_stride`, `val_stride`, or `test_stride` if that split-specific field is present;
2. otherwise the shared `stride` fallback.

For the active SMD benchmark configs, the current values are:

- `window_size: 20`
- `train_stride: 10`
- `val_stride: 1`
- `test_stride: 1`

Relevant config files:

- `configs/data/smd_benchmark_machine_1_6_window20.yaml`
- `configs/data/smd_benchmark_machine_3_4_window20.yaml`
- `configs/data/smd_benchmark_machine_3_9_window20.yaml`

Because the loader already reads `val_stride`, no code change is required to enable non-overlapping validation. A configuration change is sufficient.

### Modeling and Training

The training loop uses the same validation loader for both the clean validation stage and the synthetic validation stage:

- `src/engine/trainer.py:683-718`

That means setting `val_stride = window_size` affects both:

- the clean `val` stage;
- the `val_synth` stage.

The validation epochs already call the synthetic augmentation reset hook before `val_synth`, so validation augmentation remains deterministic across epochs:

- `src/engine/trainer.py:702-717`
- `src/models/redlamp_baseline.py:280-284`
- `src/models/thesis_multitask.py:1863-1867`

### Evaluation

The evaluator reconstructs pointwise scores and labels from overlapping or non-overlapping window payloads:

- `src/engine/evaluator.py:68-199`
- `src/engine/evaluator.py:270-360`

Its reconstruction logic already works for `stride = window_size` because it simply sums scores into the covered interval and counts coverage. If windows do not overlap, each covered timestep will have count `1`, so averaging becomes trivial.

The metric path then uses only covered points:

$$
\text{metrics} \leftarrow \text{compute on covered timesteps only}
$$

This is already the existing behavior in:

- `src/engine/evaluator.py:170-199`

So non-overlapping validation does not break the metric code. It only changes how many timesteps are covered.

### Consequences of `stride = window_size`

If `stride = window_size`, then:

1. Validation windows become non-overlapping.
2. Reconstruction becomes simpler because each covered timestep comes from exactly one window.
3. Coverage may shrink if the raw sequence length is not an exact multiple of the window size.
4. The trailing remainder of the sequence is not evaluated unless the windowing logic is extended to add a final end-aligned window.

This is especially relevant because the active evaluator marks truncated coverage when some timesteps are not covered:

- `src/engine/evaluator.py:153-165`
- `src/engine/evaluator.py:347-360`

### Proposed Implementation

The smallest implementation is:

1. Keep the loader architecture unchanged.
2. Set `val_stride` equal to `window_size` in the validation data config for the benchmark you want to simplify.
3. Leave `train_stride` and `test_stride` unchanged unless you intentionally want the same behavior there.
4. Verify that the validation metrics still have acceptable coverage after the change.

For the active SMD benchmark with `window_size: 20`, the config change is:

```yaml
val_stride: 20
```

in each of the active SMD benchmark data YAMLs.

### Optional Follow-Up if Full Coverage Is Required

If the goal is not only to avoid overlap but also to cover the entire validation timeline, then the current code would need an additional windowing rule to append a final end-aligned window when the length is not divisible by the window size. That is not currently implemented.

## Code References

- `src/data/loaders.py:66-97` - split-specific stride resolution and loader construction
- `src/data/loaders.py:231-289` - `WindowDataset` window indexing
- `src/data/window.py:16-57` - sequence slicing into windows
- `src/data/api.py:18-54` - public SMD loader config fields
- `src/data/api.py:80-111` - public anomaly archive loader config fields
- `src/engine/trainer.py:683-718` - validation and synthetic validation share the same loader
- `src/engine/evaluator.py:68-199` - pointwise reconstruction and covered-point extraction
- `src/engine/evaluator.py:270-360` - metric computation and truncation bookkeeping

## Pipeline Documentation

The current validation pipeline is:

1. Build windows from the validation sequence using the configured stride.
2. Feed those windows to the clean validation stage and to the synthetic validation stage.
3. Reconstruct pointwise scores and labels from the windows.
4. Compute metrics only on the covered timesteps.

With `val_stride = window_size`, step 1 becomes non-overlapping for validation, while steps 2 to 4 remain unchanged.

## Historical Context (from documents/)

The repository research notes already state that overlap semantics matter because the evaluator reconstructs scores by averaging overlapping windows. The current code also already supports split-specific stride settings, so validation non-overlap is a protocol choice, not an architectural change.

## Open Questions

1. If the validation sequence length is not divisible by the window size, do you prefer to accept the truncated tail or to extend the windowing logic to append a final end-aligned window?
2. Should the non-overlapping setting apply only to clean validation, or also to synthetic validation, given that both stages currently share the same validation loader?
