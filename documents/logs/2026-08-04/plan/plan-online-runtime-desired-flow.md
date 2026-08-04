---
date: 2026-08-04 17:10:00 +07:00
planner: OpenAI Codex
topic: "Implement the desired online runtime flow"
status: ready
revision: 4be64456d6aa652457a0702154bae0d9b742a803
branch: dev
related_research: documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md
---

# Implementation Plan: Implement the desired online runtime flow

## Summary

The online runner currently recalibrates thresholds, keeps only endpoint
scalars, builds a preliminary `pnn_mask` before triage, and runs verification
before the current window's update. The implementation must follow the desired
order from the research report:

```text
load stage_b_best_checkpoint and threshold_artifact
  -> receive causal_window with absolute_indices
  -> create source_hidden/projected_hidden
  -> create window_point_scores
  -> update current_window_ewma_point_scores by absolute index
  -> create window_point_predictions
  -> classify triage_region
  -> run or admit the current online action
  -> run verification_cycle for buffered entries
  -> emit online_event_record and UI event
  -> save online_runtime_state
```

The plan keeps the existing `VerificationResult` and `was_adapted` runtime
machinery. It does not add a separate verification-result object. It also keeps
the simple sliding-window rule: a point is updated while it still appears in a
later window; after that, its latest prediction remains unchanged.

## Request

Use the research report and `prompts/2_plan_prompt.md` to list programming tasks
for each affected code area. The implementation must use the canonical online
ontology names, keep `recurrent_signature_set` cycle-local, create no
`online_mlp_projector` for A0, use the current score when a point has no prior
EWMA value, and send each new event directly to the UI callback.

## Current state

The executable path is:

```text
python -m scripts.run_thesis_online_benchmark
  -> scripts.benchmarks.run_thesis_online_benchmark.main
  -> run_thesis_online_benchmark
  -> run_thesis_online_tta_experiment
  -> _run_online_sequence
  -> _process_online_window
```

`_build_runtime_online_context` in
`src/engine/online_tta/online_engine_run.py` currently calibrates thresholds
from clean validation data at online startup. `_score_online_window` in
`src/engine/online_tta/online_engine_window_metrics.py` keeps only the final
point score and one scalar EWMA value. `_prepare_online_window_event` builds
`pnn_mask` and appends to `signature_history` before it classifies the window.
`_admit_and_verify_online_window` runs the verification cycle before
`_execute_window_event_step` handles the current window.

`OnlineRuntimeState` stores `previous_ewma_score` and `signature_history`, and
`_normalize_online_records` can invent `did_update=True` for a missing field.
The demo reads scalar records after the run and has no direct event callback.

## Desired end state

1. `online_tta_phase` loads the same `stage_b_best_checkpoint` and
   `threshold_artifact` produced by the offline phase. It validates entity,
   window size, EWMA weights, schema, and checkpoint hash before changing
   stream state. It does not recalibrate thresholds on the online stream.
2. Every `causal_window` carries increasing `absolute_indices`. The runner
   keeps score, EWMA, and prediction vectors indexed by those values.
3. A first-seen point uses its current `window_point_scores` value as its EWMA
   value. A later window updates that point with the configured EWMA weights.
4. A0 uses `source_hidden`, has no projector or optimizer, and stops after
   `window_point_predictions`. A1 and A2 create the projector only when needed.
5. `triage_region` is classified before verification work. A2 handles an
   accepted `hard_old_normality` update before `verification_cycle`; `gray_zone`
   creates a `verification_entry`; normal and strong-anomaly windows do neither.
6. `hard_old_interval_guard` never changes `triage_region`. If it rejects an
   interval, the region remains `hard_old_normality` and `did_update` is false.
7. `recurrent_signature_set` is rebuilt inside one `verification_cycle` and is
   not persisted as a long-lived `signature_history`.
8. `online_event_record`, metrics, checkpoints, retention files, and the live UI
   use vector fields. Existing scalar baseline records remain readable only at
   the boundary where compatibility is required.
9. The completion report identifies the implemented protocol as
   `full_spec_v3`.

## Scope

### In scope

- Threshold artifact creation, resolution, validation, and online loading.
- A0 model construction and optimizer ownership.
- Absolute-index batch contracts and vector score/EWMA/prediction state.
- Triage, hard-old guard, verification ordering, and loss parameter wiring.
- Runtime state, checkpoint/resume, event records, metrics, retention, demo,
  and direct UI callback.
- Focused unit tests, integration tests, and one online smoke run.

### Out of scope

- Changing the mathematical formulas in `compute_hard_old_hinge_loss` or
  `compute_masked_pnn_reconstruction_loss`; the report says those formulas are
  already correct.
- Removing `VerificationResult`, `was_adapted`, or the existing verification
  buffer lifecycle.
- Replacing the four-region `classify_online_window` rule.
- Broad benchmark sweeps or deleting existing artifacts.

## Evidence

- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:16-21` — four groups of runtime changes required by the research.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:70-88` — required runtime order.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:120-214` — threshold, A0, absolute-index, vector-score, triage, and guard changes.
- `documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md:215-380` — verification order, loss wiring, runtime state, records, demo, and protocol status changes.
- `documents/spec/online_tta_terminology_ontology.md:1-35` — offline-first ownership and canonical names.
- `documents/spec/online_tta_terminology_ontology.md:116-182` — threshold, causal-window, vector-score, EWMA, and prediction contracts.
- `src/engine/online_tta/online_engine_run.py:129-315` — current startup and sequence loop.
- `src/engine/online_tta/online_engine_window_core.py:53-324` — current per-window order and guard behavior.
- `src/engine/online_tta/online_engine_window_metrics.py:33-273` — current scalar scoring, preliminary PNN mask, verification update, and output construction.
- `src/engine/online_tta/runtime_state.py:13-270` — current scalar runtime schema and resume path.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:499-556` — current offline threshold artifact creation and checkpoint-hash timing.
- `tests/online/test_full_spec_online_contract.py` and `tests/online/test_online_runtime_state.py` — existing contract and resume tests to extend.

## Implementation approach

Use small, dependency-ordered changes. First make the artifact and tensor
contracts explicit. Then change the score/state path, then reorder the window
orchestrator, and only after that update records and demo consumers. Keep the
verification API stable while removing only the obsolete global
`signature_history` path. Each phase must leave focused tests runnable.

## Phase 1: Lock artifact and configuration contracts

### Goal

Make the offline artifact the only source of online thresholds and make its
identity checkable.

### Changes

#### 1. Offline threshold artifact contents

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py`
- **Symbol:** `_build_thresholds`, `_export_offline_artifacts`
- **Change:** Compute `online_point_ewma_threshold` using the same absolute-index
  overlapping-window score path that online inference will use. Add the
  canonical `input_window_threshold`, `latent_window_low_threshold`, and
  `latent_window_high_threshold` records. Compute and store the Stage B
  checkpoint hash before writing the artifact. Keep `variant_name` owned by the
  offline run; do not use it to represent `online_variant`.
- **Reason:** The online phase must consume offline calibration rather than
  recalibrate from clean validation data at startup.
- **Dependencies:** The score/EWMA helper from Phase 3 must be shared by
  offline artifact creation and online inference.

#### 2. Artifact schema and loading

- **File:** `src/protocols/threshold_artifact.py`
- **Symbol:** `validate_threshold_artifact`, `build_threshold_artifact`,
  `load_threshold_artifact`
- **Change:** Extend validation and serialization for the four canonical online
  thresholds and checkpoint identity. Reject missing, mismatched, or stale
  schema values before the stream starts.
- **Reason:** A malformed artifact must fail before model or buffer mutation.
- **Dependencies:** Keep compatibility with the existing baseline artifact only
  where its current schema explicitly permits it; do not silently map an old
  scalar to a new vector threshold.

#### 3. Artifact and checkpoint resolution

- **File:** `src/engine/online_tta/checkpoint_resolution.py` and
  `scripts/benchmarks/run_thesis_online_benchmark.py`
- **Symbol:** `resolve_stage_b_checkpoint`, `run_thesis_online_benchmark`
- **Change:** Resolve the offline `threshold_artifact` using the same offline
  variant, entity, seed, and benchmark identity used for
  `stage_b_best_checkpoint`. Add the approved artifact-path field to the
  configuration contract before using it; do not invent an unapproved
  canonical name.
- **Reason:** The checkpoint and artifact must come from one offline run.
- **Dependencies:** The ontology and config validation must define this field
  before implementation proceeds.

#### 4. Online startup

- **File:** `src/engine/online_tta/online_engine_run.py`
- **Symbol:** `_build_runtime_online_context`
- **Change:** Load and validate the resolved artifact before building
  `online_runtime_state`. Remove `calibrate_entity_threshold_artifacts` and
  `_persist_threshold_artifacts` from normal THESIS online startup. Keep the
  artifact path and identity in runtime context.
- **Reason:** Online must not create a new calibration decision.
- **Dependencies:** Phase 1 artifact schema and Phase 5 runtime schema.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_threshold_artifact.py tests/online/test_online_entrypoint.py` — the online path loads and validates the offline artifact and does not recalibrate.

#### Manual

- [ ] Run one A2 smoke configuration and inspect the report — its artifact path, entity, window size, EWMA weights, and checkpoint hash match the offline run.

### Risks

- Existing artifacts may not contain the new threshold records. Reject them with
  a clear schema error or route them through an explicitly marked legacy path;
  never reinterpret them silently.

## Phase 2: Make A0 and loss/configuration branches explicit

### Goal

Ensure A0 has no projector or optimizer, and ensure A1/A2 use the correct
threshold and loss parameters.

### Changes

#### 1. Conditional projector construction

- **File:** `src/models/online_impl/online_adaptation.py`
- **Symbol:** `OnlineAdaptationModel.__init__`, projector helper methods
- **Change:** Accept the already-resolved `online_variant` or an equivalent
  explicit construction flag. For A0, do not create
  `online_mlp_projector`, projector alias, anchor state, or projector parameter
  group. Preserve `forward_source`. For A1/A2, keep the current projector path.
- **Reason:** A0 is inference-only and must not own online trainable state.
- **Dependencies:** `_build_runtime_online_context` and optimizer construction.

#### 2. Optimizer ownership

- **File:** `src/engine/online_tta/online_engine_run.py` and
  `src/engine/online_tta/online_engine_window_core.py`
- **Symbol:** `_build_runtime_online_context`, `_execute_window_event_step`
- **Change:** Build an optimizer only for A1/A2. Do not create a new optimizer
  per window for A0 or for a skipped action.
- **Reason:** Avoids invalid projector access and makes the update boundary
  visible in the call chain.

#### 3. Model configuration

- **Files:** `configs/model/online_adaptation.yaml`,
  `src/core/config_model_validation.py`,
  `scripts/benchmarks/generate_online_benchmark_configs.py`, generated online
  experiment configs
- **Symbol:** online adaptation model fields and validation allow-lists
- **Change:** Add `lambda_online_contrastive` to the model-owned configuration,
  validate it as non-negative, and pass it through the generated configs.
- **Reason:** A2 currently hardcodes `0.1`, which can disagree with the chosen
  model configuration.
- **Dependencies:** Preserve existing `lambda_align`, `lambda_proto`, and
  `lambda_anchor` meanings.

#### 4. Loss branch

- **File:** `src/engine/online_tta/online_engine_step.py`
- **Symbol:** `_run_online_variant_update`, `execute_online_tta_step`
- **Change:** Pass `input_window_threshold` separately from
  `online_point_ewma_threshold`. Use `input_window_threshold` for
  `hard_old_reconstruction_loss`. Use configured
  `lambda_online_contrastive`. Remove the A1 fallback that updates without a
  non-empty `pnn_mask`; A1 updates only on a verified non-empty mask. Keep the
  internal `pnn_verified` compatibility value only until the update API can
  receive the mask condition directly; never persist it as `triage_region`.
- **Reason:** The current code mixes threshold meanings and can run an A1 update
  without verified PNN points.
- **Dependencies:** Phase 4 supplies the final `pnn_mask` after triage.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_tta_variants.py tests/online/test_full_spec_online_losses.py tests/online/test_online_tta_trainable_surface.py` — A0 has no projector/optimizer and A1/A2 use the intended branches and weights.

#### Manual

- [ ] Inspect model parameter summaries for A0 and A2 — A0 lists no projector; A2 lists only projector parameters as trainable.

### Risks

- Removing projector attributes can break generic callers. Keep A0-specific
  branches explicit and make any accessor fail clearly instead of fabricating
  an empty optimizer.

## Phase 3: Add absolute indices and vector score state

### Goal

Make each point addressable across overlapping windows and compute the desired
vectors before adaptation.

### Changes

#### 1. Window indices

- **Files:** `src/data/stream.py`, `src/data/collate.py`,
  `src/core/contracts.py`
- **Symbols:** `SMDOnlineStream._build_window`, `collate_windows`,
  `validate_window`, `validate_batch`, `validate_online_batch`
- **Change:** Emit `absolute_indices: LongTensor[L]` for each
  `causal_window`, collate it as `[B, L]`, and validate dtype, shape, length,
  and strictly increasing order. Keep `start_index` and `end_index` as metadata
  for compatibility, but use `absolute_indices` for point state.
- **Reason:** Local position `t` is not stable when windows overlap.
- **Dependencies:** `build_entry_batch` must reconstruct the same indices.

#### 2. Verification entry batches

- **File:** `src/engine/online_tta/verification_adapter.py`
- **Symbol:** `build_entry_batch`
- **Change:** Rebuild `absolute_indices` from the stored entry interval and
  validate the resulting batch before verification.
- **Reason:** A buffered entry must use the same point identity as the live
  stream.

#### 3. Shared point-score state

- **File:** `src/protocols/point_scores.py`
- **Symbol:** existing EWMA helpers and new vector helper (proposed symbol)
- **Change:** Add a small mapping-based helper that merges
  `window_point_scores` by `causal_window.absolute_indices`, computes
  `current_window_ewma_point_scores`, and applies this rule per point:

  ```text
  if previous value is absent: current value = current score
  else: current value = previous_weight * previous value
                         + current_weight * current score
  ```

  Return the current window's vector plus the updated absolute-index state.
- **Reason:** A first-seen point must not be treated as previous EWMA zero, and
  overlapping windows must update the same point.
- **Dependencies:** Offline threshold calibration must call the same helper.

#### 4. Online scoring path

- **Files:** `src/engine/online_tta/online_engine_window_metrics.py`,
  `src/engine/online_tta/online_engine_run.py`
- **Symbols:** `_extract_online_window_scores`, `_score_online_window`,
  `_run_online_sequence`
- **Change:** Return `window_point_scores [L]`, input/latent window scores,
  `current_window_ewma_point_scores [L]`, and the per-point state. Remove
  `previous_ewma_score: float | None` from the active vector path.
- **Reason:** Endpoint scalar names are not aliases of the canonical window
  vectors.
- **Dependencies:** Phase 3 index contract and Phase 4 triage thresholds.

#### 5. Prediction before update

- **Files:** `src/engine/online_tta/online_engine_window_core.py`,
  `src/engine/online_tta/online_engine_step.py`
- **Symbols:** `_prepare_online_window_event`, `_build_event_window_outputs`,
  `_build_step_record`
- **Change:** Create `window_point_predictions [L]` from the current EWMA vector
  and `online_point_ewma_threshold` before any model update. Keep the latest
  absolute-index prediction in state while the point remains in later windows.
  Stop passing `input_window_score` as `raw_point_score`; preserve the actual
  endpoint only in legacy scalar fields.
- **Reason:** The record currently mixes input-window and endpoint values and
  calculates prediction inside the update function.
- **Dependencies:** Phase 6 record schema.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_stream.py tests/online/test_full_spec_online_contract.py tests/online/test_online_ewma_threshold.py` — batches contain correct absolute indices and first-seen/overlap EWMA values.
- [ ] Add a focused integration test (proposed `tests/online/test_online_window_flow.py`) — a later overlapping window updates existing points and a point outside later windows keeps its latest prediction.

#### Manual

- [ ] Print two overlapping windows from one entity — the shared absolute indices receive two successive EWMA/prediction updates.

### Risks

- A vector state can grow with the stream. Keep only the point state needed for
  active overlapping windows plus the latest output required by the record/UI;
  do not add a separate finalization table.

## Phase 4: Reorder triage, guard, update, and verification

### Goal

Make the per-window call order match the desired runtime flow.

### Changes

#### 1. Per-window orchestration

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbols:** `_process_online_window`, `_prepare_online_window_event`,
  `_admit_and_verify_online_window`, `_execute_window_event_step`
- **Change:** Replace the current `prepare -> buffer_and_verification -> update`
  sequence with `score/EWMA/prediction -> triage -> current action -> buffer
  admission -> verification_cycle -> output`. Split the current combined
  admission/verification function into two short functions.
- **Reason:** Verification currently runs before the current window's A2
  action.
- **Dependencies:** Phase 3 vector event and Phase 5 buffer state.

#### 2. Triage classification

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbol:** `_classify_event_window`
- **Change:** Call `classify_online_window` without a guard. Keep the returned
  four-region `triage_region` unchanged.
- **Reason:** `hard_old_interval_guard` is an admission guard, not a triage
  classifier.

#### 3. Hard-old action

- **Files:** `src/engine/online_tta/online_engine_window_core.py`,
  `src/engine/online_tta/non_overlap_guard.py`
- **Symbol:** `_execute_window_event_step`, `NonOverlapGuard.accept/add`
- **Change:** For A2 and `triage_region=hard_old_normality`, check
  `hard_old_interval_guard` immediately before the update. On rejection, keep
  the region, set `did_update=False`, and do not admit a verification entry.
  Call `add(interval)` only after a successful update. Keep the guard's current
  interval behavior.
- **Reason:** The current code changes hard-old windows to `gray_zone`.

#### 4. Verification path

- **Files:** `src/engine/online_tta/online_engine_window_metrics.py`,
  `src/engine/online_tta/verification_adapter.py`,
  `src/engine/online_tta/verification_cycle.py`
- **Symbols:** `_update_online_window_buffers`, `_verify_and_adapt_entries`,
  `verify_buffer_entries`, `VerificationCycleController.maybe_run`
- **Change:** Admit only current `gray_zone` windows. Run
  `verification_cycle` only from `verification_buffer`. Let that cycle be the
  only path that creates `recurrent_signature_set` and `pnn_mask`. Keep one
  `VerificationResult` per `verification_entry`, including `adapted` and
  `was_adapted` bookkeeping. Do not add a common `verification_outcome` field.
- **Reason:** A preliminary global signature history incorrectly mixes current
  windows with buffered entries.
- **Dependencies:** The internal `pnn_verified` branch remains temporary and
  noncanonical until the step API accepts the mask condition directly.

#### 5. A0 early exit

- **File:** `src/engine/online_tta/online_engine_window_core.py`
- **Symbol:** `_process_online_window`
- **Change:** After vector prediction and runtime-state point update, return for
  A0. Do not classify triage, admit entries, run verification, or call
  `execute_online_tta_step` for A0.
- **Reason:** A0 is inference-only.

#### 6. Cycle-local signatures

- **Files:** `src/engine/online_tta/online_engine_run.py`,
  `src/engine/online_tta/online_engine_shared.py`,
  `src/engine/online_tta/runtime_state.py`
- **Symbol:** `signature_history` and recurrent-signature state plumbing
- **Change:** Remove the global live `signature_history` path. Build and discard
  `recurrent_signature_set` inside each verification cycle. Do not persist
  signatures from normal, hard-old, or strong-anomaly windows.
- **Reason:** `signature_history` is not an ontology-confirmed alias of
  `recurrent_signature_set` and the chosen design is cycle-local.

### Verification

#### Automated

- [ ] Add spy-based order checks to the proposed window-flow integration test — triage precedes PNN creation and A2 hard-old update precedes verification.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_tta_triage.py tests/online/test_online_verification_buffer.py tests/online/test_verification_cycle.py tests/online/test_online_tta_variants.py` — four regions, guard rejection, per-entry verification, and A0 behavior remain correct.

#### Manual

- [ ] Run one A2 stream with a gray-zone entry and one hard-old overlap — inspect that the hard-old event is not inserted into the verification buffer and the later cycle adapts only buffered entries.

### Risks

- Reordering can change when the model is mutated. Use spies and a one-window
  deterministic fixture before running a full stream. Keep `VerificationResult`
  and buffer cleanup stable during this phase.

## Phase 5: Replace scalar runtime state and checkpoint fields

### Goal

Resume a vector-based stream without losing point EWMA/prediction state or
   mutable verification state.

### Changes

#### 1. Runtime schema

- **File:** `src/engine/online_tta/runtime_state.py`
- **Symbol:** `OnlineRuntimeState`, `to_dict`, `from_dict`,
  `validate_resume_state`, `restore_online_runtime_state`
- **Change:** Replace `previous_ewma_score` with serializable absolute-index
  point EWMA/prediction state and the current window indices. Keep verification
  entries, verification history, and hard-old intervals. Remove persistent
  `signature_history`; do not add a list of finalized points. Increment
  `runtime_schema_version` and reject incompatible payloads before restore.
- **Reason:** The sliding-window rule does not need explicit finalization, but
  resume does need the active point state.

#### 2. Runtime synchronization

- **File:** `src/engine/online_tta/online_engine_shared.py`
- **Symbol:** `_sync_online_runtime_state`
- **Change:** Write vector EWMA/prediction state, current cursor, buffer entries,
  and hard-old intervals. Do not write signatures from windows that never
  entered the buffer.
- **Reason:** Checkpoints must reproduce the next window's decisions.

#### 3. Sequence restore and checkpoint export

- **Files:** `src/engine/online_tta/online_engine_run.py`,
  `scripts/benchmarks/run_thesis_online_benchmark.py`
- **Symbols:** `_run_online_sequence`, `_finalize_online_execution`,
  `_load_runtime_state_snapshot`
- **Change:** Restore point state instead of resetting
  `previous_ewma_score=None`. Export the new state keys in both structured and
  legacy checkpoint wrappers. Keep legacy loading explicit and reject a legacy
  scalar payload when the requested runtime schema is vector-based.
- **Reason:** Continuous and resumed runs must have the same next-window result.

### Verification

#### Automated

- [ ] Extend `tests/online/test_online_runtime_state.py` and
  `tests/online/test_online_state_roundtrip.py` — compare vector predictions
  from an uninterrupted run with a save/restore run and verify buffer/guard
  identity.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_online_runtime_state.py tests/online/test_online_state_roundtrip.py` — schema mismatch fails before mutation.

#### Manual

- [ ] Stop an A2 smoke run at a fixed cursor, resume it, and compare the next
  record and `window_point_predictions` with the uninterrupted run.

### Risks

- Old checkpoints cannot safely provide missing per-point state. Keep a clear
  schema-version error and retain old artifacts for rollback; do not silently
  fill missing vector values with zeros.

## Phase 6: Update event records, metrics, retention, and live demo

### Goal

Expose the same vector contract to reports and the UI without inventing a
second prediction state.

### Changes

#### 1. Event record and metrics

- **Files:** `src/engine/online_tta/online_engine_step.py`,
  `src/engine/online_tta/online_engine_window_metrics.py`,
  `src/engine/online_tta/online_engine_window_core.py`
- **Symbols:** `_build_step_record`, `_build_online_window_outputs`,
  `_build_event_window_outputs`, `_finalize_online_execution`
- **Change:** Emit `entity_id`, interval, `window_point_scores`,
  `current_window_ewma_point_scores`, `window_point_predictions`,
  `online_point_ewma_threshold`, `online_variant`, `triage_region`,
  `did_update`, and `online_total_loss`. Keep `absolute_indices` under
  `causal_window` unless the ontology explicitly adds it to the record.
  Separate record construction from model update.
- **Reason:** A scalar endpoint record cannot represent overlapping point
  results.

#### 2. Benchmark normalization and retention

- **File:** `scripts/benchmarks/run_thesis_online_benchmark.py`
- **Symbols:** `_normalize_online_records`, `_export_online_retention_bundle`,
  `_load_runtime_state_snapshot`
- **Change:** Preserve explicit `did_update=False`; never default missing
  non-A0 records to true. Persist vector records, metrics, artifact, and new
  runtime state. Keep scalar baseline compatibility only in a clearly marked
  normalization branch.
- **Reason:** The current default can report an update that never occurred.

#### 3. Live UI callback

- **Files:** `src/engine/online_tta/online_engine_run.py`,
  `src/engine/online_tta/online_engine_window_core.py`, and the existing demo
  entrypoints
- **Symbol:** proposed read-only `online_event_callback`
- **Change:** Add an optional callback invoked after
  `window_point_predictions` and event fields are ready. The callback receives
  one immutable/copy-safe event and cannot mutate model, buffer, or runtime
  state. Keep file replay as a separate compatibility reader.
- **Reason:** The chosen behavior is direct display during the loop.
- **Dependencies:** Phase 6 record schema must be stable before UI wiring.

#### 4. Demo consumers

- **Files:** `demo/online_replay.py`, `demo/demo_state.py`, `demo/plotting.py`,
  `tests/demo/test_demo_state.py`, `tests/demo/test_live_online_replay.py`
- **Symbol:** current scalar-to-array conversion and plotting state
- **Change:** Build the time axis from absolute indices and vector fields for
  THESIS records. Keep a separate scalar adapter for baseline records. Do not
  repeat one endpoint value across every point in a window.
- **Reason:** The current demo fabricates a vector from one scalar per window.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/benchmarks/test_thesis_online_benchmark_wrapper.py tests/demo/test_demo_state.py tests/demo/test_live_online_replay.py` — vector retention, explicit update flags, and absolute-index plotting work.

#### Manual

- [ ] Run with the UI callback enabled and inspect one overlapping pair of
  windows — the display updates shared points when the second event arrives.

### Risks

- Callback consumers could mutate event data. Pass a copy or immutable view and
  test that callback mutation cannot change runtime state.

## Phase 7: Align protocol status and specifications

### Goal

Keep the source, ontology, pseudocode, and reports consistent after the runtime
change.

### Changes

- **Files:** `src/engine/online_tta/online_engine_run.py`,
  `documents/spec/online_tta_terminology_ontology.md`,
  `documents/spec/full-spec-v3.md`,
  `documents/notes/online_runtime_flow_debug.md`
- **Symbols/sections:** completion report status, vector record/state schemas,
  threshold-artifact contract, and the desired online pseudocode
- **Change:** Set `runtime_protocol_status` to `full_spec_v3` after all runtime
  checks pass. Document the approved artifact-path field, cycle-local
  `recurrent_signature_set`, direct callback, A0 no-projector rule, first-seen
  EWMA rule, and the fact that `pnn_verified` is internal compatibility control
  data rather than a `triage_region`.
- **Reason:** The documents are the repository source of truth for names and
  lifecycle contracts.
- **Dependencies:** Update only after the code and tests establish the final
  schema; do not document unsupported behavior early.

### Verification

#### Automated

- [ ] `rg -n "verification_outcome|signature_history|previous_ewma_score" src scripts tests documents/spec documents/notes` — no obsolete canonical field remains in the active vector contract; any legacy reference is explicitly labeled.
- [ ] `.venv/bin/python -m pytest -q tests/online/test_full_spec_online_contract.py tests/online/test_full_spec_online_losses.py` — code and spec-level contracts agree.

#### Manual

- [ ] Read the final pseudocode once from top to bottom — every object name maps to one ontology entry and every step has one owner.

### Risks

- A documentation-only rename can hide a runtime mismatch. Update status only
  after the end-to-end smoke check in Phase 8.

## Testing strategy

Test in dependency order:

1. Artifact schema and checkpoint identity.
2. Batch indices and vector EWMA/prediction helpers.
3. A0/A1/A2 branches, loss parameters, triage and verification ordering.
4. Runtime-state round trip and checkpoint resume.
5. Record normalization, retention, demo, and callback isolation.
6. One concrete online benchmark smoke combination before any matrix run.

The focused baseline command from the research report is:

```text
.venv/bin/python -m pytest -q \
  tests/online/test_online_ewma_threshold.py \
  tests/online/test_online_tta_triage.py \
  tests/online/test_online_tta_variants.py \
  tests/online/test_online_verification_buffer.py \
  tests/online/test_verification_cycle.py \
  tests/online/test_online_runtime_state.py \
  tests/online/test_online_stream.py \
  tests/online/test_threshold_artifact.py
```

After focused tests pass, run one existing smoke configuration through
`scripts/run_thesis_online_benchmark.py` with an explicit `--experiment-config`,
the existing protocol config, and one explicit `--online-variant`. Inspect the
artifact, checkpoint, runtime state, vector records, and UI event before any
broader benchmark.

## Migration and rollback

- Increase `runtime_schema_version` for the vector state. Reject incompatible
  checkpoints before model, buffer, or stream mutation.
- Keep old scalar checkpoints and artifacts untouched for rollback. Do not
  overwrite them with vector records.
- Keep the legacy checkpoint-key reader only as an explicitly labeled
  compatibility path. It must not silently create a vector state from a scalar.
- If the new artifact is missing or its identity does not match the Stage B
  checkpoint, stop before processing the first window.
- Rollback means selecting the previous code and its matching scalar artifact;
  no data migration is required for old retained bundles.

## Documentation

- Update the online ontology and full specification after the implementation
  contract is verified.
- Keep the desired-flow pseudocode aligned with actual call order.
- Add the new vector record/state fields and callback behavior to the debug
  guide in plain language.
- Mark legacy scalar fields and `pnn_verified` as implementation compatibility
  details, not canonical ontology objects.

## Final verification

- [ ] Online startup reads the offline `threshold_artifact` and
  `stage_b_best_checkpoint` without recalibration.
- [ ] A0 creates no `online_mlp_projector`, optimizer, verification entry, or
  verification cycle.
- [ ] Two overlapping windows update the same absolute-index point's EWMA and
  prediction; a point outside later windows keeps its latest prediction.
- [ ] Triage precedes PNN creation; accepted A2 hard-old update precedes
  `verification_cycle`; rejected hard-old intervals remain hard-old.
- [ ] A resumed run produces the same next vector record as an uninterrupted
  run.
- [ ] Retention files and the live callback expose vectors, not fabricated
  endpoint copies.
- [ ] Focused tests and one smoke combination pass before the report is marked
  `full_spec_v3`.

## Assumptions and non-blocking uncertainties

- The canonical configuration field for locating the offline
  `threshold_artifact` must be confirmed in the ontology/config contract before
  Phase 1 code changes. The plan intentionally does not invent that name.
- The existing demo entrypoint remains the UI owner; the callback is optional
  and read-only. If the demo cannot accept a callback without a broader UI
  change, keep file replay as a fallback and record that limitation.
- `absolute_indices` remains owned by `causal_window`; adding it directly to
  `online_event_record` requires an ontology update first.
