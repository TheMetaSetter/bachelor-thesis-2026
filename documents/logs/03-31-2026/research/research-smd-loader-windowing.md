---
date: 2026-03-31 15:37:10 +0700
researcher: Artificial Intelligence Agent
git_commit: 7779c876c7da79c961ec7ac18f710620d5172533
branch: dev
repository: bachelor-thesis-2026
topic: "Minimum runnable SMD vertical slice for loader, stream, windowing, baseline, and evaluation"
tags: [research, smd, loader, stream, windowing, baseline, evaluation]
status: complete
last_updated: 2026-03-31
last_updated_by: Artificial Intelligence Agent
---

# Research: Minimum runnable SMD vertical slice for loader, stream, windowing, baseline, and evaluation

**Date**: 2026-03-31 15:37:10 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 7779c876c7da79c961ec7ac18f710620d5172533  
**Branch**: dev

## Research Question
For Phase 1, what is the smallest correct SMD-first pipeline that should be implemented now so the thesis codebase becomes executable end to end?

## Executive Answer
The minimum runnable vertical slice should use **only SMD**, preserve the thesis contracts, and defer every higher-risk idea.

The recommended Phase 1 implementation is:

```text
raw SMD machine files
-> per-machine dataset parser
-> sequential DatasetStream
-> Windowizer with L=100 and default stride=10
-> simple reconstruction baseline
-> point and window anomaly scores
-> pointwise evaluation metrics
```

The key implementation decisions are:

- Use `data/ServerMachineDataset/` as the **canonical native source** for Phase 1.
- Keep dataset parsing and windowing as separate modules.
- Keep the native axis order as `[time, channel]` and batched windows as `[B, L, D]`.
- Use `window_size=100` and `stride=10` as the default Phase 1 setting.
- Fit one global `StandardScaler` on the training split only, then apply it to validation and test.
- Use a **boring reconstruction baseline** first, preferably a small MLP autoencoder that reconstructs `[B, L, D]`.
- Keep anomaly scoring **explicitly modifiable** behind a stable scorer interface. For Phase 1, use per-timestep reconstruction error as the default `point_scores` and derive `window_scores` from those point scores.
- Report honest pointwise metrics first: ROC-AUC, PR-AUC, precision, recall, and F1. Do not rely only on point-adjusted metrics.

The main non-goals for Phase 1 are:

- no online adaptation
- no prototype modules
- no synthetic anomaly injection
- no drift injection
- no multi-dataset support beyond the reusable SMD-first interfaces

## Why this is the correct minimum

### 1. The design documents already define the Phase 1 shape
The design documents repeatedly say the first milestone is a vertical slice, not the full thesis model. The agreed first milestone is:

```text
SMD loader -> encoder adapter -> simple head -> train/eval loop
```

That is explicit in `documents/design/idea.md`. The same documents also fix the native contracts:

- windows have length `L = 100`
- raw sequences are `[T, D]`
- model batches are `[B, L, D]`
- streaming must expose `next_point()` and `next_window()`

So the Phase 1 task is not open-ended research anymore. The local design already narrows it to an executable SMD path with stable interfaces.

### 2. The native implementation should preserve SMD machine boundaries from the start
SMD is not one homogeneous long series. The design notes explicitly warn that SMD contains multiple machine subsets and that code organization, checkpoint naming, and metric aggregation should respect that structure.

That makes the raw per-machine files under `data/ServerMachineDataset/` the safer canonical source for native code. They preserve `entity_id` naturally and prevent windows from crossing machine boundaries. Using only the concatenated `data/SMD/*.npy` files would be slightly simpler at first, but it hides machine identity and makes later streaming and per-machine evaluation harder.

### 3. The data-layer split is already specified
`documents/design/design_starter.md` explicitly says:

- dataset parsing is one concern
- window construction is another concern

So Phase 1 should not copy the TSLib `SMDSegLoader` structure directly, because that reference mixes loading, scaling, splitting, and window slicing inside one class. The RedLamp reference matches the desired separation better, even though its tensor orientation differs from the thesis-facing `[L, D]` contract.

## Local Findings

### Data assets present in this repository
The repository already contains both local SMD forms:

- `data/SMD/SMD_train.npy` with shape `(708405, 38)`
- `data/SMD/SMD_test.npy` with shape `(708420, 38)`
- `data/SMD/SMD_test_label.npy` with shape `(708420,)`
- `data/ServerMachineDataset/train/*.txt`, `test/*.txt`, and `test_label/*.txt` for 28 machines

The concatenated arrays and the raw machine files are consistent with each other. The total length across the raw machine files matches the lengths of the `.npy` arrays. The test labels contain `29,444` positive anomaly points.

### Reference behavior from TSLib
The local TSLib reference for SMD:

- reads `SMD_train.npy`, `SMD_test.npy`, and `SMD_test_label.npy`
- fits a `StandardScaler` on the train array
- transforms train and test
- uses the last twenty percent of train as validation
- performs windowing inside the dataset class
- defaults to `step=100`

This is useful as a quick baseline reference, but it is not a good direct template for the thesis-native data layer because it hides the parser/windowizer boundary and also uses test labels as placeholder labels for training and validation windows.

### Reference behavior from RedLamp
The local RedLamp SMD reference:

- reads raw machine files one machine at a time
- preserves per-machine entities
- optionally creates validation splits per machine
- separates loading from rolling-window creation

This is closer to the desired thesis data-flow, but its internal arrays are arranged as `[feature, time]` and its batches are `[B, D, L]`, so a native adapter is still needed.

## Concrete Phase 1 Recommendation

### 1. Canonical SMD parser
Implement one native SMD parser that reads from `data/ServerMachineDataset/` and emits one `raw_sequence` dictionary per machine:

```python
raw_sequence = {
    "x": FloatTensor[T, D],
    "point_labels": Optional[IntTensor[T]],
    "mask": Optional[BoolTensor[T, D]],
    "timestamps": Optional[Tensor[T]],
    "meta": {
        "dataset_name": "smd",
        "entity_id": "machine-1-1",
        "split": "train" | "val" | "test",
        "num_channels": 38,
        "sequence_length": T,
    },
}
```

For Phase 1:

- train sequences come from `train/*.txt`
- test sequences come from `test/*.txt`
- test labels come from `test_label/*.txt`
- timestamps can be `None`
- mask can default to all-true if no missingness handling is needed yet

The `.npy` files under `data/SMD/` should still be kept as a **verification source**, not the main parser target. They are useful to confirm that concatenated lengths match expected totals and to smoke-test scaling logic.

### 2. Validation split
Use a **per-machine** validation split from the tail of each training machine sequence. The exact ratio is less important than keeping the split simple and reproducible. For Phase 1, the cleanest choice is:

- last `10%` of each training machine for validation
- first `90%` of each training machine for training

This follows the more machine-aware RedLamp behavior and avoids creating one validation block that spans concatenated machine boundaries.

### 3. Scaling
Fit one global scaler on all training timesteps from all training machines after the train/validation split has been made. Then:

- transform the training slices
- transform the validation slices with the same scaler
- transform the test sequences with the same scaler

This keeps the implementation simple while respecting the basic rule that test statistics must not leak into fitting.

### 4. Sequential stream
Implement a lightweight `DatasetStream` for Phase 1, even though this phase is still offline.

It should:

- iterate one machine sequence point by point
- expose `next_point()`
- expose `next_window()` by delegating to the windowizer
- expose `reset()`
- emit metadata fields that keep `entity_id`, `split`, and `time_index` visible

This satisfies the design contract now and avoids a later rewrite when online adaptation is added.

For Phase 1, this stream does **not** need River integration, drift injection, or any online optimizer logic. It only needs to prove that the dataset can be consumed sequentially in the same schema that future streaming experiments will use.

### 5. Windowizer
Use:

- `window_size = 100`
- `stride = 10`
- no padding by default

This is the best local default for Phase 1 because:

- `L=100` is fixed throughout the design docs
- `stride=10` is the only explicit config example in the thesis design
- overlapping windows are more useful than `step=100` for a trainable baseline and later point-score aggregation

On the local concatenated array lengths, `L=100` and `stride=10` would produce:

- `70,831` train windows
- `70,833` test windows

By contrast, `step=100` would produce only `7,084` train windows and `7,084` test windows. That is simpler, but it is much coarser and less useful for point-level scoring.

The native window object should remain:

```python
window = {
    "x": FloatTensor[L, D],
    "point_labels": Optional[IntTensor[L]],
    "mask": Optional[BoolTensor[L, D]],
    "timestamps": Optional[Tensor[L]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "start_index": int,
        "end_index": int,
        "window_size": 100,
        "stride": 10,
    },
}
```

### 6. Baseline model
The best Phase 1 baseline is a **small reconstruction autoencoder** in one self-contained model file.

A good minimum choice is:

- flatten `[B, L, D]` to `[B, L * D]`
- MLP encoder
- small bottleneck
- MLP decoder
- reshape back to `[B, L, D]`

This is not novel, but it is correct for the stated Phase 1 goal:

- train/eval loop becomes easy to implement
- anomaly score is obvious
- output contract fits the later thesis model contract
- all training and inference logic can remain in one file, as required by `codebase_preferences.md`

This baseline is preferable to adding forecasting heads, prototypes, or augmentation now, because those add branching decisions before the codebase has even proven that SMD loading and evaluation work.

### 7. Anomaly score
The anomaly scoring mechanism should be treated as a **replaceable module**, not as a hard-coded property of the first baseline.

For Phase 1, the default scorer for the reconstruction baseline should be:

- `point_scores[t] = mean squared reconstruction error across channels at timestep t`
- `window_score = mean(point_scores within the window)`

For overlapping test windows, aggregate point scores back onto the original timeline by averaging all window contributions that cover each timestep.

This yields:

- one score per test point for honest pointwise metrics
- one score per window for debugging and later window-level reporting

However, the implementation should not assume reconstruction error is permanent. The score builder should be easy to swap later for alternatives such as:

- forecasting residuals
- latent-distance scores
- energy-based scores
- classifier logits or probabilities
- hybrid combinations of several scores

The stable part of Phase 1 is the output contract:

- the model or scorer must produce `point_scores` when point-level scoring is available
- the evaluator must accept `point_scores` and `window_scores` without depending on how they were computed
- threshold selection and metric computation must operate on the score outputs, not on baseline-specific reconstruction internals

### 8. Evaluation
Phase 1 evaluation should stay honest and minimal.

Report:

- pointwise ROC-AUC
- pointwise PR-AUC
- pointwise precision
- pointwise recall
- pointwise F1

For thresholded metrics, set the threshold **without using test labels**. The cleanest Phase 1 rule is:

- compute normal-reference scores on train or validation-normal windows
- choose a fixed percentile or `mean + k * std`
- apply that threshold unchanged on test

Do not begin with point-adjusted F1 as the main result. The design notes already warn that some anomaly-detection reporting conventions can inflate numbers. If point-adjusted metrics are added later, they should be secondary.

## What should not be implemented yet

The following are out of scope for this Phase 1 slice and should be deferred:

- prototype memory modules
- continuous/discrete fusion
- synthetic anomaly injection
- CARLA-style anomaly taxonomy training
- dual-encoder online adaptation
- projector alignment losses
- drift scenarios
- all-dataset abstraction work beyond the interfaces needed for SMD

Adding them before the SMD path is executable would violate the stated Phase 1 goal.

## Acceptance Checks

At the end of Phase 1, the codebase should be able to answer the user’s required questions with a concrete yes:

### 1. Can I load SMD correctly?
Yes, if the parser reads all 28 machines, preserves `entity_id`, and returns `[T, D]` sequences with aligned test labels.

### 2. Can I iterate over it sequentially?
Yes, if `DatasetStream.next_point()` walks each machine in time order and exposes `is_start_of_sequence` and `is_end_of_sequence`.

### 3. Can I form windows of length 100?
Yes, if the windowizer emits `[100, 38]` windows and aligned label slices without crossing machine boundaries.

### 4. Can I run a full train/eval loop?
Yes, if one reconstruction baseline trains on training windows, validates on held-out windows, saves checkpoints, reloads them, and runs inference on test windows.

### 5. Can I compute anomaly metrics?
Yes, if test-time point scores are reassembled onto the original test timeline and the evaluator reports threshold-free and thresholded pointwise metrics.

If any of those five checks still fail, then Phase 1 is not complete and no higher-level thesis modules should be added yet.

## Recommended build order

1. Implement `datasets/smd.py` with per-machine raw parsing.
2. Implement `scalers.py` and fit-transform logic using train split only.
3. Implement `streams/dataset_stream.py` for sequential access.
4. Implement `window.py` to slice `[T, D]` into `[L, D]`.
5. Implement a PyTorch dataset/loader wrapper that uses the same window contract.
6. Implement one reconstruction baseline in one model file.
7. Implement evaluator code that converts model outputs into point scores and metrics.
8. Add minimal pytest coverage for loader shapes, one forward/backward pass, and checkpoint save/load.

## Code References

- `documents/design/idea.md:6` - fixes `L = 100`.
- `documents/design/idea.md:97` - states the first milestone is the minimal vertical slice `SMD loader -> encoder adapter -> simple head -> train/eval loop`.
- `documents/design/idea.md:171` - warns against misleading anomaly metrics.
- `documents/design/idea.md:175` - states SMD machine structure must be respected from the start.
- `documents/design/design_starter.md:162` - requires separation between dataset parsing and window construction.
- `documents/design/design_starter.md:171` - names `datasets/smd.py`, `window.py`, `scalers.py`, and `loaders.py` as the intended data-layer modules.
- `documents/design/design_starter.md:313` - provides the explicit config example with `window_size: 100` and `stride: 10`.
- `documents/design/design_starter.md:583` - defines the `raw_sequence` contract as `[T, D]`.
- `documents/design/design_starter.md:611` - defines the streaming contract with `next_point()` and `next_window()`.
- `documents/design/design_starter.md:640` - defines the native window contract as `[L, D]`.
- `documents/design/design_starter.md:668` - defines the model batch contract as `[B, L, D]`.
- `documents/design/design_starter.md:705` - defines the evaluation record contract.
- `documents/design/stream_design.md:162` - defines the intended `DatasetStream -> Windowizer -> ...` flow.
- `documents/design/stream_design.md:195` - defines `next_point()`, `next_window()`, `reset()`, and `state_dict()` as the unified stream interface.
- `documents/design/stream_design.md:263` - says the recommended plan is to start with SMD and a minimal vertical slice.
- `bsc-thesis-ref-codebases/Time-Series-Library/data_provider/data_loader.py:603` - `SMDSegLoader` shows the local reference for SMD scaling and simple window slicing.
- `bsc-thesis-ref-codebases/Time-Series-Library/data_provider/data_loader.py:626` - fits the scaler on training data.
- `bsc-thesis-ref-codebases/Time-Series-Library/data_provider/data_loader.py:647` - shows the reference loader’s mixed train/test-label behavior, which should not be copied directly.
- `bsc-thesis-ref-codebases/RedLamp/loaders/load.py:268` - `load_smd` shows per-machine raw parsing.
- `bsc-thesis-ref-codebases/RedLamp/loaders/load.py:342` - shows per-machine validation splitting.
- `bsc-thesis-ref-codebases/RedLamp/loaders/load.py:363` - shows per-machine test loading with aligned labels.
- `bsc-thesis-ref-codebases/RedLamp/loaders/loader.py:79` - shows separate rolling-window creation.
- `bsc-thesis-ref-codebases/RedLamp/loaders/loader.py:109` - shows windows are created after parsing, not inside the parser.

## Final Recommendation
The Phase 1 implementation should be intentionally narrow:

- one dataset: SMD
- one parser path: raw per-machine files
- one stream contract
- one windowizer with `L=100`, `stride=10`
- one baseline: reconstruction autoencoder
- one honest evaluator

That is enough to make the codebase runnable. It is also enough to validate the core thesis interfaces before any prototype or online-adaptation logic is added.
