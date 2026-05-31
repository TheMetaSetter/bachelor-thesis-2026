---
date: 2026-04-02 00:46:19 +0700
planner: TheMetaSetter
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for Phase 4 online adaptation"
tags: [detail, time-series, anomaly-detection, online-adaptation, phase-4]
status: complete
last_updated: 2026-04-02
last_updated_by: TheMetaSetter
source_plan: documents/logs/04-02-2026/plan/plan-phase-4-online-adaptation-implementation.md
source_research: documents/logs/04-02-2026/research/research-phase-4-online-adaptation-current-state-and-design-target.md
---

# Detail: Detailed implementation plan for Phase 4 online adaptation

## Overview

This document translates the Phase 4 research note and the corresponding implementation plan into a programming-level execution sequence for the current repository. The repository now contains a conservative Phase 4 scaffold, so this detail pass must preserve two constraints simultaneously:

- it must describe the now-implemented projector-first online slice faithfully;
- it must keep the broader pre-Phase-4 gate semantics for any later widening beyond that slice.

The governing implementation principles remain:

- strictly follow `codebase_preferences.md`;
- preserve one-model-one-file ownership;
- preserve the existing batch and model-output contracts;
- use composition over inheritance;
- keep the online adaptation boundary narrow and explicit;
- keep the first Phase 4 slice conservative: clean stream, projector-first updates, and checkpointable online state.

The intended first online adaptation boundary is:

$$
\text{reference\_params} \text{ frozen}, \qquad
\text{online\_encoder\_params} \text{ frozen initially}, \qquad
\text{projector\_params} \text{ trainable}.
$$

Accordingly, the first Phase 4 implementation must not silently broaden adaptation to encoder layers, drift injection, or NGD-style optimization.

## Global Contracts And Implementation Rules

### Runtime layers

The Phase 4 runtime should continue to respect the four stable layers already used by the repository:

1. Configuration
2. Data
3. Model
4. Engine

The online loop is an engine addition, not a replacement for the offline trainer. The stream surface is a data-layer addition, not a model concern. The online loss and projector logic remain inside the owning model file.

### Dataset and batch contracts

The current base batch contract remains valid and must not be broken:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The online path extends this contract to:

```python
online_batch = {
    "x": Tensor[B, L, D],
    "view_a": Tensor[B, L, D],
    "view_b": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The new `meta` dictionaries must include:

- `stream_step`
- `entity_id`
- `split`
- `start_index`
- `end_index`

### Encoder and model-output contracts

Both online encoders must preserve the thesis-facing hidden-state contract:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor],
    "point_scores": Optional[Tensor[B, L]],
    "window_scores": Optional[Tensor[B]],
    "aux": dict,
}
```

Online-only tensors must live in `outputs["aux"]`. The top-level field names must remain stable so that checkpointing, serialization, and downstream analysis stay readable.

### Design pattern rules

- Composition over inheritance remains the primary rule across data and engine code.
- The adapter pattern is required for wrapping the offline multitask encoder into the online adaptation model.
- The earlier strategy pattern for tasks remains out of the active online runtime. The online model owns `training_step`, `validation_step`, and `test_step` directly.
- Registry-based construction remains required for datasets and models. The online runtime must not introduce a second ad hoc constructor path.

## Phase 0 - Verify and close the pre-Phase-4 gate

### Phase summary

This phase exists because the older repository documentation stated that Phase 4 should not begin until the offline gate was passed. The current repository now includes the conservative online scaffold, so the thesis objective of this phase is to verify that the scaffold sits on top of a sufficiently closed offline base rather than to pretend the online files do not exist.

### File-level edits

This phase may require edits to the existing offline path before any online file is added:

```text
src/models/reconstruction_mlp_ae.py
src/models/thesis_multitask.py
src/engine/trainer.py
src/engine/evaluator.py
scripts/train.py
scripts/evaluate.py
scripts/visualize_synthetic_anomalies.py
tests/test_synthetic_anomaly_visualization.py
configs/experiment/
src/tasks/
src/losses/
src/models/modules/
```

### Explicit edit content

- Verify that `scripts/train.py` and `scripts/evaluate.py` use `build_dataset(...)` as the only active dataset-construction path.
- Verify that `src/models/reconstruction_mlp_ae.py` and `src/models/thesis_multitask.py` own their stage logic directly.
- Verify that the active offline path no longer depends on model-specific logic under `src/tasks/`, `src/losses/`, or `src/models/modules/`.
- Verify that synthetic anomaly visualization exists as a maintained script-level surface.
- Verify that the ablation-readiness items already documented for continuous-only, discrete-only, fused, logging, and scheduling are either complete or closed in the same branch before Phase 4 code begins.

### Interface and contract definitions

No new online interface is allowed to become active until the gate is passed. The only valid contracts in this phase are the current offline batch, model-output, and step-output contracts.

### Design pattern application

This phase preserves the registry/factory path and the model-owned step-method pattern. It explicitly blocks reintroduction of the older task-split architecture into the active path.

### Risk mitigation

- Prototype redundancy and fusion collapse remain offline risks at this stage. The mitigation is to require the ablation-readiness logging and limiting-case tests before online code is merged.
- Adaptation contamination is prevented by not adding online code before the gate is closed.
- Evaluation metric inflation is prevented by refusing to mix incomplete online instrumentation with the current offline evaluator.

### Test plan and validation steps

Run the documented pre-Phase-4 regression suite and the newer ablation-readiness checks before the online slice is treated as an accepted active path:

```bash
pytest -q tests/test_config_loading.py tests/test_smd_dataset_shapes.py tests/test_windowizer.py tests/test_model_shapes.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py tests/test_registry.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_synthetic_anomaly_injection.py tests/test_synthetic_anomaly_visualization.py
```

### Acceptance criteria

Phase 0 is complete only if all of the following are true:

- the full offline regression set passes;
- the active path is registry-driven;
- the active path is one-model-one-file for the offline models;
- maintained anomaly visualization exists;
- the existing online adaptation files are justified by the now-implemented offline ablation gate rather than by a bypass around it.

## Phase 1 - Add the online configuration family and stream data surface

### Phase summary

This phase introduces the minimum configuration and data abstractions needed for online adaptation without changing the existing offline runtime. The thesis objective is to expose a sequential online window stream that can later feed the conservative projector-first adaptation model.

### File-level edits

The following files should be added or modified:

```text
configs/model/online_adaptation.yaml
configs/task/online_adaptation.yaml
configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml
src/core/config.py
src/core/registry.py
src/core/contracts.py
src/data/stream.py
tests/test_online_stream.py
tests/test_config_loading.py
tests/test_registry.py
```

### Explicit edit content

`configs/model/online_adaptation.yaml`

- Define architecture-facing fields only:
  - `model_name`
  - `input_dim`
  - `encoder_dim`
  - `hidden_dim`
  - `projector_hidden_dim`
  - `projector_dropout`
  - `enable_prototype_alignment`
  - `lambda_align`
  - `lambda_proto`
  - `lambda_anchor`
  - `score_source`

`configs/task/online_adaptation.yaml`

- Define runtime policy fields:
  - `task_name`
  - `reference_checkpoint_path`
  - `warm_start_projector`
  - `target_param_group`
  - `clean_stream_only`
  - `max_online_steps`
  - `log_every_n_steps`
  - `checkpoint_every_n_steps`
  - `view_noise_std`
  - `view_dropout_probability`
  - `reset_policy`
  - `reset_alignment_threshold`

`configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml`

- Keep the current experiment-config shape with:
  - `data_config_path`
  - `model_config_path`
  - `task_config_path`
  - `optimizer`
  - `output_dir`
  - `checkpoint_dir`
  - `device`
  - `seed`

`src/core/config.py`

- Extend supported model names to include `online_adaptation`.
- Extend supported task names to include `online_adaptation`.
- Add validation for:
  - `target_param_group`
  - online step limits
  - online view-noise parameters
  - optional reset-policy configuration
- Preserve backward compatibility with the offline experiment family.

`src/core/contracts.py`

- Add `validate_online_batch(batch)`.
- Keep `validate_batch(batch)` unchanged and call it first from the online validator.
- Validate that `view_a` and `view_b` have the same shape as `x`.

`src/data/stream.py`

- Add `SequenceCursor`.
- Add `SMDOnlineStream`.
- Add `OnlineWindowBatcher`.
- Reuse the existing SMD parser, scaler, and window conventions rather than inventing a second SMD parsing path.
- Emit clean-stream windows only in the first slice. Do not add drift injectors yet.

### Interface and contract definitions

The new stream surface should expose:

```python
next_window() -> dict[str, Any]
state_dict() -> dict[str, Any]
load_state_dict(state: dict[str, Any]) -> None
reset() -> None
```

`OnlineWindowBatcher` should yield validated `online_batch` dictionaries that preserve the current batch keys plus `view_a` and `view_b`.

### Design pattern application

- Composition over inheritance: the stream classes should wrap the existing parser and scaler outputs rather than subclassing the current dataset classes deeply.
- Registry/factory usage: add the online model name to the registry but do not create a separate registry system for online-only code.
- Adapter pattern is not yet introduced in this phase; this phase prepares the data and config side only.

### Risk mitigation

- Adaptation contamination is mitigated by restricting this phase to clean sequential windows only.
- Evaluation metric inflation is mitigated by not changing the offline evaluator in this phase.
- Fusion collapse and prototype redundancy remain mitigated by the earlier offline gate; this phase must not weaken the existing offline logging surfaces.

### Test plan and validation steps

- `tests/test_online_stream.py` must verify:
  - monotonic stream order;
  - preserved window length;
  - valid `view_a` and `view_b` shapes;
  - serializable stream cursor state.
- `tests/test_config_loading.py` must verify valid and invalid online config cases.
- `tests/test_registry.py` must verify that `online_adaptation` registers and builds correctly.

### Acceptance criteria

Phase 1 is complete only if:

- the online experiment config loads successfully;
- the stream surface emits contract-valid online batches;
- online stream state can be saved and restored;
- no offline test regresses.

## Phase 2 - Implement the self-contained online adaptation model

### Phase summary

This phase introduces the first actual Phase 4 model slice. The thesis objective is to align an online representation to a frozen reference representation through a small residual projector, while keeping the adapted parameter set minimal and explicit.

### File-level edits

The following files should be added or modified:

```text
src/models/online_adaptation.py
src/models/base_model.py
src/core/contracts.py
src/core/registry.py
tests/test_online_adaptation_step.py
tests/test_registry.py
```

### Explicit edit content

`src/models/online_adaptation.py`

- Add `ThesisMultitaskEncoderAdapter`.
- Add `ResidualProjector`.
- Add `OnlineAdaptationModel`.

`ThesisMultitaskEncoderAdapter`

- Load the trained `ThesisMultitaskModel` checkpoint or its state source.
- Expose `hidden: [B, L, H]` and optional `pooled: [B, H]`.
- Provide frozen access to the offline reconstruction and scoring surfaces required by the online model.

`ResidualProjector`

- Implement a residual projector:

$$
g(u) = u + F(u)
$$

- Initialize the final layer near zero so that the projector begins close to identity.
- Keep the module small and serializable.

`OnlineAdaptationModel`

- Own:
  - frozen reference encoder adapter;
  - explicit online encoder adapter;
  - residual projector;
  - online alignment loss;
  - optional prototype-alignment loss;
  - optional anchor regularization loss;
  - model-owned stage methods.
- Compute online-specific tensors into `outputs["aux"]`.
- Keep `outputs["point_scores"]` and `outputs["window_scores"]` compatible with the offline scoring interpretation.

`src/models/base_model.py`

- Preserve the existing abstract model API.
- Do not add online-only abstract methods. The online model must fit the same `forward`, `training_step`, `validation_step`, and `test_step` interface.

### Interface and contract definitions

The online model must preserve the top-level output contract and place online-only tensors under:

```python
outputs["aux"] = {
    "reference_hidden": ...,
    "online_hidden": ...,
    "projected_hidden": ...,
    "alignment_loss": ...,
    "prototype_alignment_loss": ...,
    "anchor_loss": ...,
    "projector_drift": ...,
    "target_param_group": ...,
}
```

The first active adaptation boundary is:

- `reference_params` frozen;
- `online_encoder_params` frozen;
- `projector_params` trainable.

### Design pattern application

- Adapter pattern: `ThesisMultitaskEncoderAdapter` adapts the existing offline multitask model to the online hidden-state contract.
- Composition over inheritance: `OnlineAdaptationModel` composes two encoder adapters and a projector instead of subclassing the offline multitask model.
- Strategy pattern for tasks remains inactive. The online model owns its stage logic.

### Risk mitigation

- Adaptation contamination is mitigated by projector-only updates and clean-stream batches.
- Projector drift is mitigated by residual initialization, warm-start support, and explicit drift logging.
- Prototype redundancy and fusion collapse are mitigated by freezing the offline scoring geometry instead of retraining the offline prototype surfaces during the first slice.
- Evaluation metric inflation is mitigated by separating alignment losses from anomaly-score reporting inside `aux`.

### Test plan and validation steps

- `tests/test_online_adaptation_step.py` must verify:
  - `projector_params` change after one training step;
  - `reference_params` do not change;
  - `online_encoder_params` do not change in the first slice;
  - output keys and shapes remain valid.

### Acceptance criteria

Phase 2 is complete only if:

- one online adaptation step executes without contract failure;
- only the intended parameter group changes;
- the online model remains one self-contained file;
- no offline model file is broken by the online addition.

## Phase 3 - Add the online loop, checkpoint extension, and runtime script

### Phase summary

This phase turns the online model into a runnable Phase 4 path. The thesis objective is to expose a checkpointable, inspectable online adaptation loop that processes the sequential SMD stream one step at a time.

### File-level edits

The following files should be added or modified:

```text
src/engine/online_loop.py
src/engine/checkpoint.py
src/engine/logger.py
scripts/run_online_adaptation.py
tests/test_online_state_roundtrip.py
tests/test_online_adaptation_step.py
```

### Explicit edit content

`src/engine/online_loop.py`

- Add class `OnlineLoop`.
- Responsibilities:
  - iterate over `SMDOnlineStream`;
  - call `model.training_step(batch)` on each online batch;
  - update only the configured target parameter group;
  - log pre-update and post-update metrics;
  - save periodic checkpoints;
  - emit serialized online records in stream order.

`src/engine/checkpoint.py`

- Extend `save_checkpoint(...)` with optional `extra_state`.
- Persist:
  - `stream_state_dict`
  - `projector_anchor_state_dict`
  - `target_param_group`
  - `online_metric_history`
  - `reset_policy_state`
- Keep existing offline checkpoint behavior unchanged.

`src/engine/logger.py`

- Reuse `metrics.jsonl`.
- Keep one logger implementation.
- Standardize online metric names:
  - `online/alignment_loss`
  - `online/prototype_alignment_loss`
  - `online/projector_drift`
  - `online/update_norm`
  - `online/window_score_mean`

`scripts/run_online_adaptation.py`

- Load `configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml`.
- Register the SMD dataset builder, `thesis_multitask`, and `online_adaptation`.
- Build the data bundle, stream surface, and online model.
- Load the offline checkpoint defined by `reference_checkpoint_path`.
- Construct an optimizer over the configured target parameter group only.
- Execute `OnlineLoop`.
- Write:
  - `online_metrics.json`
  - `online_records.json`
  - online checkpoints

### Interface and contract definitions

The online loop should return a summary dictionary shaped like:

```python
{
    "final_checkpoint_path": Path,
    "metric_history": list[dict[str, Any]],
    "records_path": Path,
}
```

Checkpoint round-tripping must preserve both model state and stream position.

### Design pattern application

- Engine separation: the offline trainer stays offline-only; the online loop is a separate engine class.
- Registry/factory usage remains the script entry pattern.
- Composition remains visible: `OnlineLoop` receives a stream, a model, an optimizer, and a checkpoint manager rather than subclassing the offline trainer.

### Risk mitigation

- Projector drift is mitigated by checkpointing anchor state and logging drift every online step.
- Adaptation contamination remains mitigated by keeping the reset policy disabled by default in the first slice.
- Evaluation metric inflation is mitigated by storing stream-order records and logging per-step metrics rather than only final aggregates.

### Test plan and validation steps

- `tests/test_online_state_roundtrip.py` must verify:
  - model round-trip;
  - optimizer round-trip;
  - stream-state round-trip;
  - projector-anchor round-trip;
  - online metric-history round-trip.

### Acceptance criteria

Phase 3 is complete only if:

- the online loop can run a reduced SMD stream end to end;
- checkpoints restore the stream cursor and projector anchor correctly;
- online metrics are appended to `metrics.jsonl`;
- the online script writes its summary artifacts without breaking offline scripts.

## Phase 4 - Validate the first online adaptation slice and hold the boundary

### Phase summary

This phase validates the completed first slice and explicitly prevents scope drift. The thesis objective is to demonstrate that the repository can now execute a conservative Phase 4 path without silently expanding into drift injection, encoder unfreezing, or NGD experimentation.

This phase should also be read as the current repository translation of the
earlier generic roadmap. The foundational online scaffolding is no longer the
missing work. The remaining later-slice streaming scope is drift injection,
non-adaptive online baselines under drift, broader adaptation policies, and
eventual NGD-style expansion once the accepted first slice is stable.

### File-level edits

This phase is mostly validation-oriented and may touch:

```text
tests/test_online_stream.py
tests/test_online_adaptation_step.py
tests/test_online_state_roundtrip.py
configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml
documents/logs/04-02-2026/
```

### Explicit edit content

- Keep `clean_stream_only: true` in the first accepted online experiment.
- Keep `target_param_group: projector_params` in the first accepted online experiment.
- Keep reset-policy execution disabled by default even if checkpoint fields already support it.
- Do not add drift injectors, encoder unfreezing, or NGD-specific optimizer branches in this phase.

### Interface and contract definitions

The accepted first-slice behavior is:

- clean sequential SMD windows;
- frozen reference encoder;
- frozen online encoder;
- trainable residual projector;
- serializable online state;
- stream-order artifact writing.

Anything broader than that belongs to a later Phase 4 sub-slice.

### Design pattern application

- Scope discipline is itself part of the architecture. The first online slice remains a narrow extension, not a second research codepath explosion.

### Risk mitigation

- Prototype redundancy and fusion collapse remain monitored through the existing offline ablation surface and are not re-optimized during the first online slice.
- Adaptation contamination remains constrained by clean-stream-only execution.
- Projector drift remains observable through explicit logging and anchor comparison.
- Evaluation metric inflation remains controlled by pre-update versus post-update metric logging and stream-order serialization.

### Test plan and validation steps

Run the following minimum validation stack:

```bash
pytest -q tests/test_online_stream.py tests/test_online_adaptation_step.py tests/test_online_state_roundtrip.py tests/test_config_loading.py tests/test_registry.py
```

Then run one smoke experiment through `scripts/run_online_adaptation.py` on a reduced SMD stream and verify:

1. the stream cursor advances monotonically;
2. only projector parameters change;
3. `online/alignment_loss` is logged;
4. `online/projector_drift` is logged;
5. `online_records.json` is written in stream order;
6. restoring from checkpoint resumes from the saved stream position.

### Acceptance criteria

Phase 4 is complete only if:

- the conservative online path runs end to end on SMD;
- the online state round-trips correctly;
- the online metrics and records are written as planned;
- the offline runtime path remains unchanged and test-clean;
- no deferred Phase 4 feature was pulled into the first merge silently.

## Completion Standard

This detailed plan is complete when it provides a measurable path from the current offline-only repository to a first accepted Phase 4 implementation slice with the following exact properties:

- the offline gate is passed first;
- the online runtime is projector-first and clean-stream-only;
- `src/models/online_adaptation.py` owns all online model logic directly;
- `src/engine/online_loop.py` owns the sequential online runtime;
- the batch, encoder, and output contracts remain readable and explicit;
- the repository can checkpoint and restore online state without creating a second incompatible infrastructure path.
