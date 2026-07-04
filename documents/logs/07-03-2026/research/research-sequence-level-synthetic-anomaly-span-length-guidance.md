---
date: 2026-07-03 19:41:12 +0700
researcher: TheMetaSetter
git_commit: 31277c7afdd9e2d8dec3c39bf9c497fe1afab051
branch: dev
repository: bachelor-thesis-2026
topic: "Synthetic anomaly span-length guidance for RedLamp families"
tags: [research, time-series, anomaly-detection, synthetic-anomaly, sequence-level-augmentation, redlamp]
status: draft
last_updated: 2026-07-03
last_updated_by: Codex
---

# Research: Synthetic Anomaly Span-Length Guidance For RedLamp Families

## Research Question

This note records the discussion around two related questions:

1. If the current RedLamp-style synthetic anomaly families are injected inside already-cut windows, what span lengths make each family visually and semantically clear?
2. If synthetic anomalies are injected into the original sequence first and windows are cut afterward, what span lengths should each family use so the resulting windows still show the intended anomaly class clearly?

The discussion focuses on the 11 active RedLamp anomaly families used in the thesis codebase:

1. `spike`
2. `flip`
3. `speedup`
4. `noise`
5. `cutoff`
6. `average`
7. `scale`
8. `wander`
9. `contextual`
10. `upsidedown`
11. `mixture`

## Current Runtime Context

The active implementation currently performs synthetic anomaly injection after windowization.

The current order is:

1. parse raw split sequences;
2. clean sequences;
3. fit scaler on clean train sequences;
4. transform train, validation, and test sequences;
5. cut sequences into fixed-length windows;
6. collate windows into a batch shaped `[B, L, D]`;
7. inject synthetic anomalies into individual windows inside the model-side training or synthetic-validation path.

Therefore, the current implementation is window-local augmentation, not sequence-level augmentation.

In the current injector, every anomaly family receives one sampled contiguous segment inside the current window. The segment length is sampled from:

```text
min_segment_length = int(window_size * min_segment_fraction)
max_segment_length = int(window_size * max_segment_fraction)
```

With the active benchmark-style setting `window_size = 20` and `min_segment_fraction = 0.2`, `max_segment_fraction = 0.3`, each anomalous window receives a synthetic span of about `4-6` timesteps.

This is useful for compact window-local corruption, but it is not equally expressive for all 11 families. Some families are naturally short and local, while others need a longer temporal context to become visually and semantically clear.

## Why One Shared Span Range Is Too Crude

The main conclusion is that a single global span-length range should not be treated as ideal for all RedLamp families.

Different families expose different kinds of abnormality:

- `spike` is point-like.
- `noise` is local stochastic corruption.
- `cutoff` and `average` are local flattening or replacement patterns.
- `scale` changes local amplitude.
- `flip`, `speedup`, and `upsidedown` need visible temporal structure inside the segment.
- `wander` needs enough duration for drift to accumulate.
- `contextual` needs enough normal context outside the segment for the injected region to be meaningfully out of context.
- `mixture` needs enough room for multiple primitive mechanisms to appear without collapsing into indistinguishable noise.

Because of this, the right span length is family-dependent.

## If Injection Happens After Windowing

If injection happens after windowing, the span must fit inside one window. This means the ideal range is constrained by `L = window_size`.

The following ranges describe visually useful values when anomalies are created inside a single already-cut window.

| Family | Ideal span length to show the class clearly | Practical note |
|---|---:|---|
| `spike` | `1-3` timesteps | This is a point anomaly. If stretched too much, it stops looking like a spike and starts looking like a burst or noisy segment. |
| `noise` | `5-30` timesteps | Needs several consecutive points to look like local noise, but should not be so long that it becomes an entirely noisy regime. |
| `cutoff` | `5-50` timesteps | Needs enough duration to show zeroing or hold behavior. Too short can look like a small local drop. |
| `average` | `5-50` timesteps | Needs enough duration to show a flattened mean segment. Too short can be confused with cutoff or scale. |
| `scale` | `8-60` timesteps | Needs enough points to show amplitude compression or expansion around the segment center. |
| `flip` | `8-40` timesteps | Reverse-subsequence behavior needs internal shape. With only a few points, reversal is hard to see. |
| `speedup` | `10-60` timesteps | Temporal compression needs enough samples to reveal a changed temporal rate. |
| `wander` | `20-200+` timesteps | Drift or random-walk behavior needs time to accumulate. This family is weak under very short windows. |
| `contextual` | `10-80` timesteps | Needs a segment that differs from outside-window context. If the whole window is too small, contextual contrast is weak. |
| `upsidedown` | `8-60` timesteps | Inversion needs visible shape. Too short can look like mild scale or noise. |
| `mixture` | `10-100+` timesteps | Needs enough duration for multiple primitive anomaly components to be visible. |

For the two common window sizes discussed in the project, the practical window-local ranges are:

| Family | Recommended range with `L = 20` | Recommended range with `L = 100` |
|---|---:|---:|
| `spike` | `1-3` | `1-3` |
| `noise` | `4-8` | `10-30` |
| `cutoff` | `4-10` | `10-50` |
| `average` | `4-10` | `10-50` |
| `scale` | `5-12` | `15-60` |
| `flip` | `6-16` | `15-40` |
| `speedup` | `8-18` | `20-60` |
| `wander` | `12-20`, but still short | `30-100` |
| `contextual` | `8-16`, keeping outside context | `20-70` |
| `upsidedown` | `6-16` | `15-60` |
| `mixture` | `10-20` | `25-100` |

The practical implication is clear: with `window_size = 20`, the current span range `4-6` is acceptable for short local families, but it is too restrictive for families that need longer temporal structure.

For `L = 20`, the families that remain relatively compatible are:

- `spike`
- `noise`
- `cutoff`
- `average`
- `scale`

The families that are more constrained or less expressive under `L = 20` are:

- `wander`
- `speedup`
- `contextual`
- `mixture`
- partly `flip`
- partly `upsidedown`

## If Injection Happens Before Windowing

If synthetic anomalies are injected into the original sequence before windowing, span length should be defined on the original timeline, not on the already-cut window.

This changes the design logic.

The goal is no longer merely to fit an anomaly inside one window. The goal is to create an event on the original timeline such that, after sliding-window extraction, multiple windows see consistent slices of the same event.

For a window size `L`, a useful rule is:

- short point-like anomalies can remain much shorter than `L`;
- shape-based anomalies should usually be at least `0.5L`;
- regime, drift, contextual, and mixture anomalies should often be `1L` to `10L+`.

This means that if `L = 20`, many sequence-level synthetic anomalies should be longer than 20 timesteps. If `L = 100`, many should be longer than 100 timesteps.

### Recommended Sequence-Level Ranges

| Family | Ideal span length if injected before windowing | With `L = 20` | With `L = 100` |
|---|---:|---:|---:|
| `spike` | `1-3` points, or several spikes inside a region of `5-20` | `1-3`, or burst `5-10` | `1-3`, or burst `10-20` |
| `noise` | `0.5L-2L` | `10-40` | `50-200` |
| `cutoff` | `0.5L-3L` | `10-60` | `50-300` |
| `average` | `0.5L-3L` | `10-60` | `50-300` |
| `scale` | `1L-4L` | `20-80` | `100-400` |
| `flip` | `0.5L-2L` | `10-40` | `50-200` |
| `speedup` | `1L-4L` | `20-80` | `100-400` |
| `wander` | `2L-10L+` | `40-200+` | `200-1000+` |
| `contextual` | `1L-5L`, with normal context before and after | `20-100` | `100-500` |
| `upsidedown` | `0.5L-3L` | `10-60` | `50-300` |
| `mixture` | `2L-8L` | `40-160` | `200-800` |

## Interpretation By Family

### `spike`

`spike` should stay short.

The cleanest form is `1-3` points. If the desired event is longer, it should be modeled as a burst containing several sparse spikes inside a local event region, not as one long fully anomalous spike segment.

If `spike` is stretched over many consecutive timesteps, the visual identity changes. It becomes closer to noisy burst or level shift instead of a point anomaly.

### `noise`

`noise` needs a short-to-medium contiguous region.

At least several points are needed so the viewer can distinguish random local disturbance from isolated spike behavior. For sequence-level injection, `0.5L-2L` is a useful range because several resulting windows will contain a meaningful noisy region.

### `cutoff`

`cutoff` needs enough duration to show a sensor-like loss, zeroing, or hold behavior.

For window-local injection, a few points may be visible but not always persuasive. For sequence-level injection, `0.5L-3L` is more suitable because multiple windows can show the cutoff region as a stable event.

### `average`

`average` creates a flattened segment at the local mean.

It needs enough duration for the flatness to be visible. If too short, it may look like a small perturbation rather than a clear anomaly family. Sequence-level spans around `0.5L-3L` make the flattened region more recognizable.

### `scale`

`scale` changes local amplitude around the segment center.

This class benefits from at least one full window length under sequence-level injection, because the window should include enough shape before and during the amplitude change. A useful range is `1L-4L`.

### `flip`

`flip` reverses a subsequence.

This only makes sense if the segment contains enough internal temporal structure. With too few points, reversing the segment has weak visual effect. A useful sequence-level range is `0.5L-2L`.

### `speedup`

`speedup` compresses temporal dynamics.

It needs enough duration for temporal-rate change to be visible. Very short spans make speedup hard to distinguish from interpolation artifacts or local distortion. A useful sequence-level range is `1L-4L`.

### `wander`

`wander` should be treated as a long-span anomaly.

The family is based on cumulative drift-like behavior, so it needs time to accumulate. With `L = 20`, a `4-6` timestep span is usually too short to show the intended behavior. A better sequence-level range is `2L-10L+`.

### `contextual`

`contextual` depends on comparison with the surrounding context.

This class should preserve normal context before and after the injected span. If the span covers the entire visible context, it becomes harder to identify the injected region as contextually abnormal. A useful sequence-level range is `1L-5L`, with explicit care that the event is not placed so close to sequence boundaries that context disappears.

### `upsidedown`

`upsidedown` inverts the segment around its local mean.

It needs enough shape to make inversion visible. Short spans can be confused with scale or noise. A useful sequence-level range is `0.5L-3L`.

### `mixture`

`mixture` combines multiple primitive anomaly mechanisms.

It should generally be longer than its component families, because if several components are applied inside a very short segment, the result becomes hard to interpret. A useful sequence-level range is `2L-8L`.

## Why Visualization Is Still Needed

The recommended ranges above are engineering priors, not final empirical proof.

Human visual inspection is still useful because the phrase "the anomaly class is clear" is partly a perceptual and data-dependent criterion. A span length that is clear on one SMD entity may be less clear on another entity if the base signal is flat, noisy, periodic, or already volatile.

However, the inspection should not be casual. It should be a structured visual audit.

For window-local injection, useful plots should show:

1. clean window;
2. augmented window;
3. difference curve;
4. synthetic anomaly mask;
5. class name and span length.

For sequence-level injection before windowing, useful plots should show:

1. original timeline before injection;
2. full timeline after injection;
3. the injected span mask on the original timeline;
4. several windows cut from the same injected timeline:
   - a window before the span;
   - a window touching the start of the span;
   - a window centered inside the span;
   - a window touching the end of the span;
   - a window after the span.

This is important because sequence-level injection introduces a new question: not only whether the full injected event looks right, but also whether the downstream windows cut from that event preserve enough class identity for training.

## Practical Recommendation

The most practical next step is not to immediately rewrite the augmentation engine.

The safer path is:

1. keep the current window-local injector as the active benchmark path;
2. create a separate sequence-level synthetic visualization notebook or script;
3. generate a grid over:
   - anomaly family;
   - span-length bucket;
   - entity;
   - window size;
   - stride;
4. inspect the plots by human judgment;
5. only then lock family-specific span-length ranges into a future sequence-level synthetic augmentation spec.

This keeps the current benchmark stable while creating evidence for whether sequence-level synthetic anomaly injection is worth implementing.

## Summary

The core conclusion is:

> If synthetic anomalies are injected after windowing, many RedLamp families are forced into short, window-local corruptions. If they are injected before windowing, the span-length design should move to the original timeline and should become family-specific.

For `window_size = 20`, the current short span range is acceptable for local families such as `spike`, `noise`, `cutoff`, `average`, and partly `scale`. It is not ideal for long-context families such as `wander`, `contextual`, `speedup`, and `mixture`.

For sequence-level injection, the rough default should be:

- `spike`: `1-3` points, or sparse burst regions;
- medium local families: about `0.5L-4L`;
- long-context families: about `2L-10L+`.

Before committing these ranges to runtime config, the project should generate visual examples and use human inspection to calibrate which ranges best preserve each synthetic anomaly class.
