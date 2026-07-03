---
date: 2026-07-01 14:40:19 +07
researcher: TheMetaSetter
git_commit: 0fd6abfb39ed704bf41a83845ec826b577ebdd94
branch: dev
repository: bachelor-thesis-2026
topic: "Synthetic anomaly overlap handling and label granularity"
tags: [research, synthetic-anomaly, overlap, labels, windowing]
status: complete
last_updated: 2026-07-01
last_updated_by: TheMetaSetter
---

# Research: Synthetic anomaly overlap handling and label granularity

**Date**: 2026-07-01 14:40:19 +07  
**Researcher**: TheMetaSetter  
**Git Commit**: `0fd6abfb39ed704bf41a83845ec826b577ebdd94`  
**Branch**: `dev`

## Research Question

If the same original timestep is covered by multiple windows, and each window injects a different synthetic anomaly at that timestep, how does the current code handle the overlap? When those windows are merged back to the original timeline, what is the final value at the overlapped timestep, and is the final synthetic label anomaly or normal? At what granularity is the synthetic anomaly label assigned?

## Summary

The current implementation treats each window as an independent sample during augmentation. A window is cloned, corrupted, and labeled without any cross-window reconciliation at augmentation time. Therefore, if two overlapping windows contain the same original timestep, the code does not try to force those two copies to agree during injection.

The reconciliation step happens later, during evaluation-style reconstruction. There, pointwise scores are averaged across all windows that cover the same timestep, while pointwise labels are merged with a maximum operation. Since labels are binary, that maximum acts like a logical OR: if any covering window marks a timestep as anomalous, the reconstructed timestep label becomes anomalous.

So the answer is:

- overlap is not resolved during synthetic injection;
- reconstructed scores are averaged across overlapping windows;
- reconstructed labels are the maximum across overlapping windows;
- the final reconstructed synthetic label is anomaly if any covering window says anomaly, otherwise normal;
- synthetic supervision is assigned at both levels, but with different fields:
  - `classification_labels` is window-level;
  - `synthetic_anomaly_mask` is timestep-level inside each window, and `point_labels` is the per-timestep label field used by the trainer and evaluator.

## Detailed Findings

### Data Preparation

Windows are created by slicing one raw sequence into overlapping windows. The windowizer and loader both use a start-index loop of the form `range(0, sequence_length - window_size + 1, stride)`, so the same raw timestep can appear in multiple windows when `stride < window_size`.

Relevant code:

- `src/data/window.py:16-57`
- `src/data/loaders.py:231-289`

Each returned window is a fresh clone:

- `x` is sliced and cloned per window;
- `point_labels` is sliced and cloned per window;
- `meta` keeps `start_index`, `end_index`, `absolute_start_index`, and `absolute_end_index`.

This means overlapping windows are separate samples, not shared views into one mutable timeline.

### Modeling and Training

Synthetic augmentation is handled inside `SyntheticAnomalyInjector`.

Relevant code:

- `src/data/augment.py:146-156`
- `src/data/augment.py:800-864`
- `src/data/augment.py:866-931`
- `src/models/redlamp_baseline.py:252-309`
- `src/models/thesis_multitask.py:1044-1063`
- `src/models/thesis_multitask.py:1863-1867`
- `src/engine/trainer.py:603-717`

Important points:

1. `augment_batch()` loops over windows one by one.
2. For each anomalous window, `_inject_single_window()` selects one contiguous subsequence inside that window and corrupts the cloned `x`.
3. The code does not look across neighboring windows when injecting anomalies.
4. The window-level class target is stored in `classification_labels`.
5. The per-timestep synthetic mask for that window is stored in `synthetic_anomaly_mask`.
6. The per-timestep label field is written as:

```python
augmented_batch["point_labels"] = torch.maximum(
    original_point_labels.clone(), anomaly_masks
)
```

when original point labels exist, or as `anomaly_masks` when they do not.

That means synthetic labels are available at two levels:

- `classification_labels`: one scalar per window;
- `synthetic_anomaly_mask` / `point_labels`: one label per timestep inside the window.

### Evaluation

The evaluator reconstructs overlapping windows back to an entity timeline.

Relevant code:

- `src/engine/evaluator.py:68-199`
- `src/engine/evaluator.py:270-360`

For each entity and timestep:

$$
s_t = \frac{\sum_{w \ni t} s_{w,t}}{\sum_{w \ni t} 1}
$$

where `s_{w,t}` is the point score contributed by window `w` at timestep `t`.

For labels, the code uses:

$$
y_t = \max_{w \ni t} y_{w,t}
$$

Because labels are binary, this is an OR operation in practice.

Therefore:

- if one overlapping window says anomaly at that timestep, the reconstructed label is anomaly;
- only if all covering windows say normal does the reconstructed label stay normal.

The evaluator keeps:

- `point_scores`: the full reconstructed score timeline;
- `point_labels`: the full reconstructed label timeline;
- `covered_point_mask`: which timesteps were actually covered by at least one window.

Metrics are then computed only on the covered portion of the reconstructed timeline.

### Direct Answers to the Three Questions

1. **If multiple overlapping windows inject different synthetic anomalies at the same original timestep, how is it handled?**

   Each window is augmented independently. There is no cross-window conflict resolution during injection. The same original timestep may therefore have different corrupted copies in different windows.

2. **When those windows are merged back to the original timeline, what happens at the overlapped timestep?**

   The reconstructed score is the average across all covering windows. The reconstructed label is the maximum across all covering windows, so any anomalous covering window makes the final label anomalous.

3. **Is synthetic anomaly label assigned at window level or per timestep inside the window?**

   Both exist, but they serve different purposes:

   - `classification_labels` is window-level.
   - `synthetic_anomaly_mask` is timestep-level inside each window.
   - `point_labels` is the per-timestep label field that the trainer and evaluator consume.

## Code References

- `src/data/window.py:16-57` - overlapping window slicing and local metadata
- `src/data/loaders.py:231-289` - `WindowDataset` materialization and per-window cloning
- `src/data/augment.py:146-156` - deterministic RNG reset
- `src/data/augment.py:800-864` - balanced or Bernoulli synthetic class selection
- `src/data/augment.py:866-931` - window-wise synthetic augmentation and label writing
- `src/models/redlamp_baseline.py:252-309` - baseline synthetic injector wiring
- `src/models/thesis_multitask.py:1044-1063` - thesis synthetic injector wiring
- `src/models/thesis_multitask.py:1863-1867` - epoch-level RNG reset hook
- `src/engine/trainer.py:603-717` - training and `val_synth` hooks
- `src/engine/evaluator.py:68-199` - overlap reconstruction by sum/count and maximum label merge

## Pipeline Documentation

The current synthetic pipeline is window-centric:

1. Build overlapping windows from the original sequence.
2. Clone each window independently.
3. Inject synthetic anomalies inside each window.
4. Assign a scalar `classification_labels` value to the whole window.
5. Assign a timestep-level `synthetic_anomaly_mask` inside the same window.
6. During reconstruction, average overlapping scores and take the maximum label at each timestep.

There is no code path that stitches the raw corrupted `x` values back into one global timeline tensor. The only stitched objects are point scores and point labels for evaluation and visualization.

## Historical Context (from documents/)

The repository design docs already emphasize a thin waist between windowed data and model outputs, with overlapping windows reconstructed later on the evaluator side. The current implementation follows that contract directly.

## Open Questions

None for the three questions above. The code is explicit:

- overlap is independent during injection;
- score aggregation uses averaging;
- label aggregation uses maximum;
- classification supervision is window-level;
- synthetic mask supervision is timestep-level inside each window.
