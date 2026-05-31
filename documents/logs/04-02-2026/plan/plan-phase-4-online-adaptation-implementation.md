---
date: 2026-04-02 00:40:35 +0700
planner: TheMetaSetter
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan for Phase 4 online adaptation"
tags: [plan, time-series, anomaly-detection, online-adaptation, phase-4]
status: complete
last_updated: 2026-04-02
last_updated_by: TheMetaSetter
source_research: documents/logs/04-02-2026/research/research-phase-4-online-adaptation-current-state-and-design-target.md
---

# Plan: Implementation plan for Phase 4 online adaptation

## Current State

- The active repository path now includes both the ablation-ready offline multitask path and the first conservative Phase 4 slice. `src/models/thesis_multitask.py` owns the multitask encoder, prototype branches, fusion logic, schedule-aware objective surface, and offline stage logic. `src/models/online_adaptation.py` and `src/engine/online_loop.py` now own the first projector-first online runtime.
- The stable runtime contracts already exist in `src/core/contracts.py`. The active batch contract is centered on `x`, `point_labels`, `mask`, `timestamps`, and `meta`. The active model-output contract is centered on `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- `scripts/train.py` and `scripts/evaluate.py` now use registry-driven dataset construction through `build_dataset(...)`, which satisfies one of the major pre-Phase-4 gate conditions.
- The repository now contains `src/models/online_adaptation.py`, `src/engine/online_loop.py`, online-specific configs, online tests, online checkpoint state, and online monitoring code. The remaining issue is no longer code absence. It is keeping the accepted online scope conservative and tied to the correct offline multitask checkpoint flow.
- `src/core/config.py` now validates both the offline multitask ablation family and the online adaptation family, including config overrides, schedule controls, and online target-parameter-group checks.
- The latest detail and research documents should no longer treat Phase 4 as absent. The correct current framing is that the pre-Phase-4 offline gate has been implemented for the first accepted slice, while broader online expansion beyond projector-first clean-stream adaptation remains deferred.

## Design Options

### Option A: Conservative projector-first online adaptation on a clean SMD stream

This option adds the smallest valid Phase 4 slice. It reuses a trained offline `ThesisMultitaskModel` checkpoint as the frozen reference source, clones its encoder into an online path, trains only a lightweight residual projector at first, streams clean SMD windows sequentially, and logs online alignment and anomaly-scoring behavior without introducing drift injection or broad parameter unfreezing in the first slice.

This option aligns most closely with `documents/design/idea.md`, `documents/design/design_starter.md`, `documents/design/stream_design.md`, and `codebase_preferences.md`.

### Option B: Streaming-plus-drift Phase 4 from the first implementation slice

This option implements the same dual-encoder projector design, but it also introduces custom drift injectors and mixed streaming scenarios immediately.

This option is viable later, but it is not the correct first implementation slice. It combines two sources of complexity at once: online adaptation mechanics and controlled non-stationarity simulation.

### Option C: Broad online adaptation boundary from the first slice

This option begins by adapting the projector and selected encoder layers together, and it prepares NGD-style optimization immediately.

This option should not be selected for the first Phase 4 implementation. It weakens the intended optimization boundary, increases update variance, and makes it harder to diagnose whether failures come from the online loop, the projector, or the encoder itself.

## Selected Approach

The recommended approach is **Option A**.

The first Phase 4 implementation should therefore be a clean-stream, projector-first, checkpointable online adaptation slice. It should satisfy the design contract while keeping the adapted parameter set minimal:

- `reference_params` are frozen always;
- `online_encoder_params` exist explicitly but remain frozen in the first slice;
- `projector_params` are the only trainable parameter group in the first slice;
- drift injection, trigger-based reset policies, encoder unfreezing, and NGD-style preconditioning remain deferred until the projector-first slice is stable.

## Risk And Mitigation

- Risk: the implementation may bypass the documented pre-Phase-4 gate and add broader online code on top of an unstable offline path.
  Mitigation: keep the accepted online runtime limited to the now-implemented projector-first slice, and treat drift injection, encoder unfreezing, and NGD-style optimization as later follow-on work.
- Risk: the online path may violate the existing batch and output contracts.
  Mitigation: preserve the current top-level batch keys and output keys. Add online-specific tensors such as `view_a` and `view_b` only as validated extensions, and place online-specific artifacts in `outputs["aux"]`.
- Risk: online updates may adapt to anomalous windows rather than genuine shift.
  Mitigation: begin with projector-only adaptation, clean streaming only, low learning rate, and explicit logging of alignment loss, score stability, and update norm. Defer more aggressive adaptation boundaries.
- Risk: the projector may drift away from its intended alignment role.
  Mitigation: use a residual projector initialized near identity, warm-start it offline from the same reference space, keep an anchor copy in checkpoint state, and log projector drift from that anchor on every online step.
- Risk: adding a separate online architecture may duplicate large portions of the offline model.
  Mitigation: make the online model reuse the offline encoder and frozen scoring surfaces through a small adapter boundary, but keep all online-specific training logic inside `src/models/online_adaptation.py`.
- Risk: online checkpointing may fork into an incompatible second serialization path.
  Mitigation: extend `CheckpointManager` into a generic extra-state serializer rather than creating a second checkpoint manager.
- Risk: online metrics may look better than they are because evaluation and adaptation are mixed carelessly.
  Mitigation: log pre-update and post-update metrics separately for each online batch and serialize stream-order records rather than only final aggregates.

## Open Questions

- The design documents define future drift injection surfaces for Phase 4 and later. The first implementation slice should defer them. If that deferral is later revised, the online data layer will need a second implementation slice rather than being expanded silently.
- The design documents mention trigger-based reset policies, but they do not define final thresholds. The first implementation slice should therefore include reset-policy state in the checkpoint contract while keeping the default reset policy disabled.
- The repository does not yet expose a generic encoder-adapter abstraction. This plan resolves that by using a thesis-model-specific adapter inside `src/models/online_adaptation.py` for the first slice.

## Implementation Plan

### 1. Enforce the pre-Phase-4 gate before adding online code

Perform an explicit gate check at the start of the Phase 4 branch. The branch must not add online files until the following are true:

- `scripts/train.py` and `scripts/evaluate.py` use registry-driven dataset construction only.
- `src/models/reconstruction_mlp_ae.py` and `src/models/thesis_multitask.py` own their active stage logic directly.
- the active multitask path no longer depends on model-specific logic in `src/tasks/`, `src/losses/`, or `src/models/modules/`;
- synthetic anomaly visualization exists and is maintained;
- the ablation-readiness checklist conditions are either already satisfied or explicitly closed in the same branch before online files are added.

If any item above fails at branch start, fix that item first and do not add Phase 4 files until the gate is documented as passed.

### 2. Add the online experiment family without disturbing the offline one

Add the following configuration files:

```text
configs/model/online_adaptation.yaml
configs/task/online_adaptation.yaml
configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml
```

Use the existing `data/model/task` experiment shape so that the online experiment family remains readable beside the offline family.

`configs/model/online_adaptation.yaml` should define architecture-facing fields only:

- `model_name: online_adaptation`
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

`configs/task/online_adaptation.yaml` should define runtime and adaptation policy:

- `task_name: online_adaptation`
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

`configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml` should mirror the current experiment pattern, reference the new data/model/task configs, and keep optimizer settings explicit for the online loop.

Modify `src/core/config.py` so that:

- `online_adaptation` is accepted as both a supported model name and a supported task name;
- the online experiment family validates the new task fields;
- the optimizer section validates an explicit `target_param_group`;
- the offline experiment family remains backward compatible.

### 3. Extend the data layer with a sequential online stream surface

Add `src/data/stream.py` and keep it dedicated to the online path.

Implement the following classes there:

- `SequenceCursor`
- `SMDOnlineStream`
- `OnlineWindowBatcher`

Responsibilities:

- `SMDOnlineStream` reuses the existing SMD parser and scaler state and emits windows in strict sequence order without future look-ahead.
- `OnlineWindowBatcher` groups the sequential windows into online mini-batches while preserving the standard batch keys.
- `SequenceCursor` carries serializable stream position state for checkpoint round-tripping.

The online batch contract should extend the current batch contract rather than replace it:

```python
batch = {
    "x": Tensor[B, L, D],
    "view_a": Tensor[B, L, D],
    "view_b": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

Modify `src/core/contracts.py` to add `validate_online_batch(batch)`. It should call `validate_batch(batch)` first, then validate `view_a` and `view_b` as rank-3 tensors matching `x.shape`.

The `meta` dictionaries for online batches should add:

- `stream_step`
- `entity_id`
- `split`
- `start_index`
- `end_index`

The first implementation slice should stream clean SMD windows only. Drift injectors remain deferred to a later Phase 4 extension and should not be included in the first merge.

### 4. Implement the self-contained online model file

Add `src/models/online_adaptation.py`. Keep all online-specific model logic in this file.

Implement the following classes in that file:

- `ThesisMultitaskEncoderAdapter`
- `ResidualProjector`
- `OnlineAdaptationModel`

`ThesisMultitaskEncoderAdapter` should:

- load a trained `ThesisMultitaskModel` checkpoint;
- expose the thesis-facing hidden-state contract `hidden: [B, L, H]`;
- provide frozen access to the offline scoring surfaces needed for reconstruction scores, classification logits, and prototype geometry.

`ResidualProjector` should:

- implement `g(u) = u + F(u)`;
- initialize the final linear layer near zero so that the projector begins near identity;
- expose a small `state_dict()` that can be anchored and round-tripped independently.

`OnlineAdaptationModel` should own:

- the frozen reference encoder adapter;
- the explicit online encoder adapter;
- the residual projector;
- the online loss computation;
- the online scoring path;
- the model-owned stage methods used by the online loop.

The online model-output contract should preserve the existing top-level shape:

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

The online-specific artifacts should live under `aux`:

- `reference_hidden`
- `online_hidden`
- `projected_hidden`
- `alignment_loss`
- `prototype_alignment_loss`
- `anchor_loss`
- `projector_drift`
- `target_param_group`

The first implementation slice should use the following adaptation boundary:

- freeze the reference encoder always;
- freeze the online encoder in the first slice;
- update only `projector_params`;
- compute anomaly scores from the projected online representation passed through the frozen offline scoring surfaces so that online scoring stays compatible with offline geometry.

### 5. Add a dedicated online loop rather than overloading the epoch trainer

Add `src/engine/online_loop.py` with class `OnlineLoop`.

`OnlineLoop` should:

- iterate over `SMDOnlineStream` in sequence order;
- call `model.training_step(batch)` for each online adaptation step;
- update only the optimizer parameter group named in config;
- log pre-update and post-update metrics per step;
- save periodic online checkpoints;
- return a final summary containing metric history, final checkpoint path, and serialized online records.

Do not fold this logic into `src/engine/trainer.py`. The epoch trainer should remain the offline engine. The online loop is a separate runtime layer because it advances a stream cursor and online state rather than epoch counters.

### 6. Extend checkpointing and logging without creating parallel infrastructure

Modify `src/engine/checkpoint.py` so that `save_checkpoint(...)` accepts an optional `extra_state` mapping and persists it beside the current offline fields. The online loop should use that generic extension rather than a separate checkpoint manager.

The online `extra_state` mapping should include:

- `stream_state_dict`
- `projector_anchor_state_dict`
- `target_param_group`
- `online_metric_history`
- `reset_policy_state`

Keep `src/engine/logger.py` as the single logging surface. The online loop should reuse `metrics.jsonl` and prefix online keys consistently, for example:

- `online/alignment_loss`
- `online/prototype_alignment_loss`
- `online/projector_drift`
- `online/update_norm`
- `online/window_score_mean`

### 7. Add the online entry script

Add `scripts/run_online_adaptation.py`.

This script should:

1. load `configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml`;
2. register the SMD dataset builder, the offline multitask model, and the new online adaptation model;
3. build the SMD data bundle and online stream surface;
4. load the offline multitask checkpoint referenced by `reference_checkpoint_path`;
5. instantiate `OnlineAdaptationModel` from that checkpoint;
6. build an optimizer over the configured target parameter group only;
7. run `OnlineLoop`;
8. write `online_metrics.json`, `online_records.json`, and an online checkpoint under the configured output directory.

Keep `scripts/train.py` and `scripts/evaluate.py` unchanged except for any shared registration helpers they need.

### 8. Add the first online test suite

Add the following tests:

```text
tests/test_online_stream.py
tests/test_online_adaptation_step.py
tests/test_online_state_roundtrip.py
tests/test_config_loading.py
tests/test_registry.py
```

Test responsibilities:

- `tests/test_online_stream.py`
  Verify strict sequence order, window-size preservation, cursor-state round-tripping, and the presence of `view_a` and `view_b`.
- `tests/test_online_adaptation_step.py`
  Verify that one online step changes `projector_params`, leaves `reference_params` unchanged, leaves `online_encoder_params` unchanged in the first slice, and returns contract-valid outputs.
- `tests/test_online_state_roundtrip.py`
  Verify that model state, optimizer state, stream cursor, projector anchor state, and online metric history survive save and load.
- `tests/test_config_loading.py`
  Extend the existing config tests so that `online_adaptation` model and task configs validate successfully and bad `target_param_group` values fail loudly.
- `tests/test_registry.py`
  Extend the registry tests to cover `online_adaptation`.

### 9. Keep the first Phase 4 slice intentionally narrow

The first merge should explicitly defer the following:

- drift injectors;
- encoder unfreezing;
- NGD-style optimization;
- trigger-based reset execution beyond a disabled default state;
- extra online evaluation scripts beyond `scripts/run_online_adaptation.py`.

Those features belong to later Phase 4 sub-slices. They should not be mixed into the first online merge.

## Interface Enforcement

### Batch contract

- Preserve the current base batch keys.
- Add `view_a` and `view_b` only as validated online extensions.
- Keep `x` as the canonical window tensor so downstream utilities still recognize the batch shape.

### Encoder contract

- Both the frozen reference encoder and the online encoder must expose `hidden: [B, L, H]` and optional `pooled: [B, H]`.
- The online model must treat any encoder-specific internal representation as private and re-expose only the thesis-facing hidden representation.

### Model output contract

- Preserve the existing top-level output dictionary so offline and online scores remain serializable through the same downstream style.
- Place online-only analysis tensors in `aux`.
- Continue to validate outputs through `validate_model_outputs`.

## Validation Procedures

Validate Phase 4 in the following order:

1. Re-run the pre-Phase-4 gate regression suite and document that it passes.
2. Run the new online unit tests.
3. Run a smoke online adaptation job on a reduced SMD stream with projector-only updates.
4. Confirm that:
   - only projector parameters change;
   - the stream cursor advances monotonically and restores from checkpoint;
   - online metrics are written to `metrics.jsonl`;
   - `online_records.json` is written in stream order;
   - `online/alignment_loss` and `online/projector_drift` are both logged.

## Recommended Build Order

1. Gate verification and any remaining gate closures.
2. Config-loader extension for the online experiment family.
3. `src/data/stream.py` and online batch validation.
4. `src/models/online_adaptation.py`.
5. `src/engine/online_loop.py`.
6. Generic checkpoint-manager extension.
7. `scripts/run_online_adaptation.py`.
8. Online tests and smoke validation.

## Completion Standard

Phase 4 is complete for the first implementation slice when the repository can take a trained offline `ThesisMultitaskModel` checkpoint, stream clean SMD windows sequentially, adapt only a residual projector online, checkpoint and restore online state, and emit online metrics and records without breaking the existing offline runtime path.
