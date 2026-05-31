---
date: 2026-03-31 16:10:00 +0700
planner: Artificial Intelligence Agent
git_commit: 7779c876c7da79c961ec7ac18f710620d5172533
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for the minimum runnable SMD vertical slice"
tags: [plan, smd, loader, stream, windowing, baseline, evaluation]
status: complete
last_updated: 2026-03-31
last_updated_by: Artificial Intelligence Agent
source_research: documents/logs/03-31-2026/researches/research-smd-loader-windowing.md
---

# Plan: Implementation plan for the minimum runnable SMD vertical slice

## Current State

- The repository is still design-first. At the time of planning, the repository contains design documents, prompts, notebooks, datasets, and reference codebases, but it does not yet contain `src/`, `configs/`, `scripts/`, or `tests/` implementation directories.
- The thesis intent is already stable enough to guide implementation. `documents/design/idea.md` fixes the first milestone as `SMD loader -> encoder adapter -> simple head -> train/eval loop`, fixes the native window length at `L = 100`, and requires a thesis-facing hidden representation of shape `[B, L, H]`.
- The engineering contracts are already specified in `documents/design/design_starter.md`. The codebase must preserve time-major tensors, with raw sequences as `[T, D]`, windows as `[L, D]`, and batched model input as `[B, L, D]`. The required stable schemas for `raw_sequence`, `stream_point`, `window`, `batch`, `outputs`, and `evaluation_record` are already documented.
- The local SMD assets are sufficient to implement the first runnable slice now. The repository contains both `data/ServerMachineDataset/` with 28 per-machine raw files and `data/SMD/*.npy` concatenated arrays. The research note establishes that the raw per-machine files preserve entity boundaries and should be treated as the canonical parsing source, while the concatenated arrays should be retained only for verification.
- The repository guidance in `codebase_preferences.md` imposes non-negotiable engineering constraints. The implementation must prioritize readability, keep model logic self-contained, follow a one-model-per-file rule, and add focused `pytest` coverage for data shapes, one forward and backward pass, checkpoint save and load, and configuration initialization.

## Design Options

### Option A: Strict minimum vertical slice with a simple reconstruction baseline

This option implements only the modules required to make SMD runnable end to end:

- one native SMD parser;
- one scaler utility;
- one sequential stream wrapper;
- one windowizer;
- one PyTorch dataset and data loader wrapper;
- one reconstruction model file;
- one reconstruction task;
- one evaluator with pointwise metrics.

This option best satisfies the thesis milestone and minimizes early coupling. It is the preferred implementation path for the current repository state.

### Option B: Encoder-adapter-first vertical slice with a stronger abstraction boundary

This option implements the same runnable slice as Option A, but introduces an explicit `BaseEncoder` and `BaseModel` boundary immediately, even though only one simple reconstruction baseline exists initially.

This option is slightly more expensive now, but it better preserves the later thesis requirement that every backbone, including future adapter-wrapped encoders, must expose `hidden: [B, L, H]`. This option should be adopted in a lightweight form, not as a deep inheritance hierarchy.

### Option C: Stream-first infrastructure with River-oriented online scaffolding

This option expands the first slice to include early online-oriented abstractions such as `state_dict()`, scenario hooks, and partial River integration before the offline SMD baseline is proven.

This option does not align well with the current repository maturity. It introduces engineering complexity before the codebase can yet demonstrate correct parsing, batching, training, checkpointing, and evaluation on SMD. It should therefore be deferred.

## Selected Approach

The recommended approach is **Option A with the minimal abstraction discipline of Option B**.

In practice, this means the first implementation should remain intentionally narrow and SMD-only, but it should still freeze the following interfaces from the beginning:

- batch contract: `batch["x"]` is always `[B, L, D]`;
- encoder contract: every model must expose `outputs["hidden"]` as `[B, L, H]`;
- output contract: reconstruction, point scores, and window scores remain separate keys in the output dictionary.

This hybrid approach aligns best with the thesis vision because it achieves an executable vertical slice now while preventing future encoder, prototype, and online-adaptation work from rewriting the data and model boundaries.

## Risk And Mitigation

- Risk: early implementation drifts away from the documented native contracts.
  Mitigation: create typed helper validators in `src/core/contracts.py` that check keys and tensor shapes for `raw_sequence`, `window`, `batch`, and `outputs` during development and testing.
- Risk: windows may accidentally cross machine boundaries if the parser collapses SMD into one long sequence.
  Mitigation: parse `data/ServerMachineDataset/` one machine at a time and construct training, validation, and test windows per entity only.
- Risk: validation leakage may occur if scaling is fit before the train/validation split.
  Mitigation: split each machine sequence first, fit the scaler only on the retained training portions, and reuse that fitted scaler unchanged for validation and test.
- Risk: the first baseline may hard-code reconstruction-specific assumptions into the evaluator.
  Mitigation: define a scorer interface that produces `point_scores` and `window_scores` independently of how the model was trained.
- Risk: checkpointing is omitted because the first milestone appears small.
  Mitigation: treat checkpoint save and load as part of the vertical slice acceptance criteria, not as optional infrastructure.
- Risk: introducing River or synthetic anomaly injection too early may block progress.
  Mitigation: expose `DatasetStream.reset()` and `DatasetStream.state_dict()` now, but defer River integration, drift injectors, and augmentation until the offline SMD slice is stable.
- Risk: future thesis modules may force major file restructuring if the initial codebase is too ad hoc.
  Mitigation: adopt the directory skeleton from `documents/design/design_starter.md` now, but only populate the files needed for the first slice.

## Open Questions

- Should the first reconstruction baseline use a flattened MLP autoencoder or a sequence-native LSTM autoencoder? The research note recommends an MLP autoencoder as the lowest-risk starting point, but the final choice should be validated against readability and training stability.
- Should the first configuration loader be plain YAML only, or plain YAML plus small dataclass wrappers? The former is simpler; the latter offers better validation. The recommended choice for Phase 1 is plain YAML with explicit validation functions.
- Should validation metrics be computed only pointwise, or should window-level debugging metrics also be logged from the start? The recommended approach is pointwise metrics as the official result and window-level metrics as debugging output only.

## Detailed Implementation Plan

### 1. Establish the initial repository structure

Add the following directories and files:

```text
configs/
  data/smd.yaml
  model/reconstruction_mlp_ae.yaml
  task/reconstruction.yaml
  experiment/smd_vertical_slice.yaml
src/
  core/
    config.py
    contracts.py
    registry.py
    seed.py
  data/
    base.py
    scalers.py
    window.py
    collate.py
    loaders.py
    stream.py
    datasets/
      smd.py
  models/
    base_model.py
    reconstruction_mlp_ae.py
  tasks/
    base_task.py
    reconstruction_task.py
  metrics/
    pointwise.py
  engine/
    trainer.py
    evaluator.py
    checkpoint.py
    logger.py
scripts/
  train.py
  evaluate.py
tests/
  test_smd_dataset_shapes.py
  test_windowizer.py
  test_model_shapes.py
  test_one_train_step.py
  test_checkpoint_roundtrip.py
  test_config_loading.py
```

This structure should be treated as the minimum viable skeleton. It follows `documents/design/design_starter.md` while avoiding premature addition of prototype modules, contrastive losses, drift injectors, and multitask heads.

### 2. Implement contract enforcement before feature expansion

Create `src/core/contracts.py` with explicit validation helpers:

- `validate_raw_sequence(raw_sequence: dict) -> None`
- `validate_window(window: dict) -> None`
- `validate_batch(batch: dict) -> None`
- `validate_model_outputs(outputs: dict) -> None`

These helpers should check:

- required keys exist;
- tensors use the correct axis order;
- optional fields are either `None` or have the documented shape;
- metadata contains `dataset_name`, `entity_id`, and `split`;
- model outputs always contain `hidden`, `aux`, and the fixed optional keys even when some values are `None`.

This step is necessary because the current repository does not yet contain code, so the easiest time to prevent schema drift is before additional modules depend on informal assumptions.

### 3. Implement the SMD parser in `src/data/datasets/smd.py`

Define a class `SMDDatasetParser` with the following public methods:

- `load_train_sequences() -> list[dict]`
- `load_validation_sequences(validation_ratio: float) -> list[dict]`
- `load_test_sequences() -> list[dict]`
- `build_splits(validation_ratio: float) -> dict[str, list[dict]]`

Implementation requirements:

- Read from `data/ServerMachineDataset/train/*.txt`, `test/*.txt`, and `test_label/*.txt`.
- Preserve one entity per file and derive `entity_id` from the filename.
- Emit the raw sequence contract with `x: FloatTensor[T, D]` and `point_labels: Optional[IntTensor[T]]`.
- Create validation data from the tail of each training entity, with `validation_ratio=0.1` as the default in configuration.
- Ensure test labels align exactly with the test sequence length.
- Keep `timestamps=None` for Phase 1 because SMD is order-indexed.
- Set `mask` to an all-true tensor or `None`, but do not change key names across splits.

The parser must not perform scaling, windowing, or PyTorch batching. This separation is explicitly required by the design documents.

### 4. Implement scaling in `src/data/scalers.py`

Define a small class `SequenceStandardScaler` with:

- `fit(sequences: list[dict]) -> None`
- `transform_sequence(raw_sequence: dict) -> dict`
- `transform_sequences(sequences: list[dict]) -> list[dict]`
- `state_dict() -> dict`
- `load_state_dict(state: dict) -> None`

Implementation requirements:

- Fit on concatenated training timesteps only after the train/validation split has been created.
- Store feature-wise `mean` and `std`.
- Apply the same scaler instance to train, validation, and test.
- Preserve metadata and labels unchanged when transforming.

The scaler state must be checkpointable because later experiment reproducibility depends on exact preprocessing recovery.

### 5. Implement streaming and window construction as separate modules

Create `src/data/stream.py` and define `DatasetStream` with:

- `__init__(self, raw_sequence: dict, windowizer: "Windowizer | None" = None)`
- `next_point(self) -> dict | None`
- `next_window(self) -> dict | None`
- `reset(self) -> None`
- `state_dict(self) -> dict`
- `load_state_dict(self, state: dict) -> None`

Create `src/data/window.py` and define:

- `Windowizer`
- `slice_sequence_into_windows(raw_sequence: dict, window_size: int, stride: int) -> list[dict]`

Implementation requirements:

- `DatasetStream.next_point()` must emit the documented `stream_point` schema.
- `DatasetStream.next_window()` must delegate to the `Windowizer` and emit the documented `window` schema.
- `Windowizer` must preserve labels, masks, timestamps, and metadata while slicing.
- Windows must never cross entity boundaries.
- Default configuration must be `window_size=100` and `stride=10`.
- No padding should be used in Phase 1.

This preserves the future online contract now without adding online adaptation logic prematurely.

### 6. Implement batching in `src/data/collate.py` and `src/data/loaders.py`

Add:

- `WindowDataset`, a thin PyTorch dataset over a prebuilt list of windows;
- `build_smd_dataloaders(config: dict) -> dict[str, DataLoader]`;
- `collate_windows(batch_windows: list[dict]) -> dict`.

Implementation requirements:

- Convert a list of native windows into the batched contract:
  `x: [B, L, D]`, `point_labels: [B, L]` when available, `mask: [B, L, D]` when available, and `meta: list[dict]`.
- Preserve `None` for missing optional fields rather than fabricating placeholders with incompatible semantics.
- Expose separate train, validation, and test data loaders from the same parser and scaler pipeline.

### 7. Implement the first baseline as one self-contained model file

Create `src/models/reconstruction_mlp_ae.py` and define `ReconstructionMLPAutoencoder`.

This file should contain:

- model configuration parsing;
- module definitions;
- `forward(batch: dict) -> dict`;
- helper methods for reconstruction loss inputs if needed.

Implementation requirements:

- Input: `batch["x"]` with shape `[B, L, D]`.
- Hidden representation: produce `outputs["hidden"]` with shape `[B, L, H]`, even if the internal model flattens the window first.
- Reconstruction: produce `outputs["recon"]` with shape `[B, L, D]`.
- Point scores: compute `outputs["point_scores"]` as per-timestep mean squared error across channels.
- Window scores: compute `outputs["window_scores"]` as the mean of point scores across the window.
- Auxiliary outputs: return `outputs["aux"] = {}` by default.

This file must remain self-contained to satisfy the repository’s one-model-per-file preference.

### 8. Implement the task layer with stable responsibilities

Create:

- `src/tasks/base_task.py`
- `src/tasks/reconstruction_task.py`

Define `ReconstructionTask` with:

- `training_step(model, batch) -> dict`
- `validation_step(model, batch) -> dict`
- `test_step(model, batch) -> dict`

Responsibilities:

- call the model;
- compute reconstruction loss from `batch["x"]` and `outputs["recon"]`;
- log scalar metrics needed by the trainer;
- return standardized output dictionaries without embedding dataset-specific assumptions.

The task layer must own loss computation. The model must own forward computation. This separation follows the single-responsibility guidance in the design documents.

### 9. Implement training, evaluation, checkpointing, and logging

Create:

- `src/engine/trainer.py`
- `src/engine/evaluator.py`
- `src/engine/checkpoint.py`
- `src/engine/logger.py`

Define:

- `Trainer`
- `Evaluator`
- `CheckpointManager`

Implementation requirements for `Trainer`:

- consume a `model`, `task`, optimizer, and data loaders;
- run epoch-based offline training;
- evaluate on validation data after each epoch;
- save the best checkpoint according to a validation criterion.

Implementation requirements for `CheckpointManager`:

- save and load the model state, optimizer state, scaler state, configuration, epoch, and any metric history needed for reproducibility.

Implementation requirements for `Evaluator`:

- convert batched outputs into window-level evaluation records;
- reassemble overlapping `point_scores` back onto each original test timeline by averaging contributions from all covering windows;
- compute pointwise ROC-AUC, PR-AUC, precision, recall, and F1;
- select the anomaly threshold from validation-normal or training-normal score statistics only, not from test labels.

The evaluator must write a serialization-friendly structure consistent with the documented `evaluation_record` contract so results can later be logged to Weights and Biases without schema redesign.

### 10. Implement configuration files and configuration loading

Create the following YAML files:

- `configs/data/smd.yaml`
- `configs/model/reconstruction_mlp_ae.yaml`
- `configs/task/reconstruction.yaml`
- `configs/experiment/baseline/smd__thesis_multitask__vertical-slice__w100__seed7__default.yaml`

Recommended contents:

- `configs/data/smd.yaml`: data root, validation ratio, window size, stride, batch size, and number of workers;
- `configs/model/reconstruction_mlp_ae.yaml`: `window_size`, `num_channels`, hidden width, bottleneck width, dropout if used;
- `configs/task/reconstruction.yaml`: learning rate, weight decay, epoch count, threshold rule, and checkpoint metric;
- `configs/experiment/baseline/smd__thesis_multitask__vertical-slice__w100__seed7__default.yaml`: references to the data, model, and task configs, plus seed and output directory.

Create `src/core/config.py` with:

- `load_yaml_config(path: str) -> dict`
- `load_experiment_config(path: str) -> dict`
- `validate_experiment_config(config: dict) -> None`

This keeps the first slice readable and avoids the overhead of a larger configuration framework.

### 11. Add registry only where it reduces coupling

Create `src/core/registry.py` with simple registries for:

- datasets;
- models;
- tasks.

The registry must remain minimal. It should map names to classes and factory callables, but it must not evolve into a heavy plugin architecture during Phase 1.

### 12. Add entry scripts for the vertical slice

Create:

- `scripts/train.py`
- `scripts/evaluate.py`

`scripts/train.py` should:

- load the experiment config;
- set the random seed;
- build parser, scaler, windows, loaders, model, task, optimizer, and trainer;
- run training and save the best checkpoint.

`scripts/evaluate.py` should:

- load the saved checkpoint and scaler state;
- rebuild the test pipeline;
- run evaluation;
- write metrics and evaluation records to disk.

If a third script is added later, `scripts/predict.py` may be introduced, but it is not necessary for the first vertical slice.

## Interface Enforcement

### Batch contract

The batch contract should be enforced in both `collate_windows` and `validate_batch`:

- `batch["x"]` must be a floating tensor of shape `[B, L, D]`;
- `batch["point_labels"]` must be either `None` or `[B, L]`;
- `batch["mask"]` must be either `None` or `[B, L, D]`;
- `batch["timestamps"]` must be either `None` or `[B, L]`;
- `batch["meta"]` must be a list of metadata dictionaries with one entry per batch item.

### Encoder contract

The encoder or model contract should be enforced in `validate_model_outputs`:

- `outputs["hidden"]` is mandatory and must be `[B, L, H]`;
- `outputs["recon"]` is optional but, when present, must be `[B, L, D]`;
- `outputs["point_scores"]` is optional but, when present, must be `[B, L]`;
- `outputs["window_scores"]` is optional but, when present, must be `[B]`.

Although the first baseline is not a separate encoder adapter, its `forward` method must already satisfy the thesis-facing hidden representation contract.

### Model output contract

The model output contract should be frozen in `src/models/base_model.py` and validated everywhere downstream:

```python
outputs = {
    "hidden": ...,
    "pooled": ...,
    "recon": ...,
    "logits": ...,
    "window_scores": ...,
    "point_scores": ...,
    "aux": ...,
}
```

Downstream code must consume these named outputs rather than inspect model internals. This is necessary so later prototype models, classifier heads, or encoder adapters can replace the baseline without rewriting the evaluator.

## Validation And Testing Plan

Add the following `pytest` coverage:

- `tests/test_smd_dataset_shapes.py`
  Verify that the parser reads all SMD entities, preserves `entity_id`, and emits `[T, D]` sequences with aligned labels.
- `tests/test_windowizer.py`
  Verify that `Windowizer` produces `[L, D]` windows, correct `start_index` and `end_index`, and never crosses entity boundaries.
- `tests/test_model_shapes.py`
  Verify that the reconstruction baseline accepts `[B, L, D]` and returns `hidden`, `recon`, `point_scores`, and `window_scores` with the documented shapes.
- `tests/test_one_train_step.py`
  Run one forward pass and one backward pass on one synthetic or reduced SMD batch.
- `tests/test_checkpoint_roundtrip.py`
  Verify that saving and reloading restores model parameters, optimizer state, and scaler state consistently.
- `tests/test_config_loading.py`
  Verify that YAML experiment configuration files load correctly and reject invalid keys or missing required fields.

These tests should remain small and fast. They are intended to validate contract correctness, not to benchmark full training quality.

## Validation Procedures

The first vertical slice should be considered complete only if all of the following checks succeed:

1. The parser reads all 28 SMD machines from `data/ServerMachineDataset/` and preserves per-machine metadata.
2. The training, validation, and test splits are reproducible and do not leak test statistics into scaling.
3. The windowizer produces `[100, 38]` windows with `stride=10`.
4. The reconstruction baseline can train for at least one epoch without shape or contract failures.
5. A checkpoint can be saved and reloaded successfully.
6. Test-time window scores can be reassembled into point-level scores on the original timeline.
7. The evaluator reports pointwise ROC-AUC, PR-AUC, precision, recall, and F1 using a threshold chosen without test-label leakage.

## Recommended Build Order

1. Create the directory skeleton, configuration loader, and contract validators.
2. Implement the SMD parser and per-machine train/validation/test split logic.
3. Implement the scaler and persistable scaler state.
4. Implement `DatasetStream` and `Windowizer`.
5. Implement the PyTorch window dataset, collate function, and data loaders.
6. Implement the self-contained reconstruction baseline model file.
7. Implement `ReconstructionTask`, `Trainer`, `CheckpointManager`, and `Evaluator`.
8. Add entry scripts for training and evaluation.
9. Add focused `pytest` coverage and verify the acceptance checks.

## Deferred Work

The following modules should be explicitly deferred until the first slice passes all acceptance checks:

- pretrained encoder adapters such as MOMENT or TimesNet;
- continuous and discrete prototype modules;
- task-specific fusion;
- synthetic anomaly injection;
- CARLA-style anomaly-type supervision;
- River integration;
- drift injectors and online adaptation;
- multi-dataset generalization beyond the interfaces already frozen for SMD.

## Final Recommendation

The codebase should now be built as an intentionally narrow, readable, and fully testable SMD-first vertical slice. The implementation should privilege contract stability and end-to-end executability over novelty.

The strongest alignment with the thesis vision is achieved by proving the following path first: per-machine SMD parsing, train-only scaling, sequential stream exposure, fixed-length windowing, one self-contained reconstruction baseline, checkpointable training, and honest pointwise evaluation. Once this path is stable, the repository will have the correct foundation for encoder adapters, prototype modules, and online adaptation without requiring a structural rewrite.
