---
date: 2026-06-10 00:00:00 +07:00
researcher: Artificial Intelligence Agent
git_commit: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
branch: dev
repository: bachelor-thesis-2026
topic: "Generalize boundary-visibility behavior across synthetic anomaly families at window size 20"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-10
last_updated_by: Artificial Intelligence Agent
---

# Research: Generalize boundary-visibility behavior across synthetic anomaly families at window size 20

**Date**: 2026-06-10 00:00:00 +07:00
**Researcher**: Artificial Intelligence Agent
**Git Commit**: 1a2825cbfe400a9f3ce280d83f2da8f26c39daba
**Branch**: dev

## Research Question
Generalize whether the remaining synthetic anomaly families can exhibit the same kind of visually weak boundary change that was observed for `flip` under the current short-window setup.

## Summary
The thesis injector applies one contiguous injected segment per affected channel, using a half-open interval `[start_index, end_index)`. Under the active `window20` task config, the segment length is sampled from `min_segment_fraction: 0.1` and `max_segment_fraction: 0.2`, so the injected subsequence is typically 2-4 time steps long. Families that change temporal order or the whole segment trajectory are more likely to show a visible boundary mismatch, while point-like or level-like families may require different inspection criteria.

## Detailed Findings

### Data Preparation
- The injector samples `start_index` and `end_index` from the current window and then writes back only into the slice `start_index:end_index`.
- Each affected channel is processed independently inside `_inject_single_window`, but the same time span is reused across all affected channels in that window.

### Modeling and Training
- This note only documents synthetic anomaly injection behavior.
- The owning file is `src/data/augment.py`.

### Evaluation
- The visualizer filter `|x[end_index-1] - x[start_index]| in (0.01, 0.1]` is useful for `flip`-like families, but it is not a universal visibility criterion for all anomaly families.

## Code References
- `src/data/augment.py:190-273` - segment sampling and in-place update semantics.
- `src/data/augment.py:276-331` - `spike` and `flip`.
- `src/data/augment.py:345-528` - `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`.
- `src/data/augment.py:567-600` - `upsidedown` and `mixture`.
- `src/data/augment.py:654-700` - per-channel injection loop in `_inject_single_window`.

## Pipeline Documentation
Under the current `window20` task config, the injected segment is short enough that the following qualitative grouping is useful:

| Family group | Boundary-visibility behavior |
| --- | --- |
| `flip`, `speedup`, `wander`, `contextual`, `upsidedown` | More likely to show boundary mismatch because they alter order or shift the whole subsequence trajectory. |
| `cutoff`, `average`, `scale` | Often visible as level/amplitude changes, but not guaranteed by a boundary-delta filter alone. |
| `spike`, `noise` | Primarily local or pointwise corruption; boundary mismatch is not the main diagnostic. |
| `mixture` | Depends on the selected primitive components. |

## Historical Context (from documents/)
- The current task config keeps `min_segment_fraction: 0.1` and `max_segment_fraction: 0.2` for `window20`.
- Earlier research notes already identified `src/data/augment.py` as the single owning surface for synthetic anomaly injection.

## Open Questions
- For families other than `flip`, a dedicated visualization rule may be needed if the goal is to inspect the anomaly as a human-readable boundary change rather than as a generic corruption mask.
