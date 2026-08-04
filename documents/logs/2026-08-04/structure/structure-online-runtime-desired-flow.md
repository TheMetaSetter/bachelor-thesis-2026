---
date: 2026-08-04 17:35:00 +07:00
topic: "Structure for implementing the desired online runtime flow"
status: proposed
revision: 4be64456d6aa652457a0702154bae0d9b742a803
related_documents:
  - documents/logs/2026-08-03/research/research-online-runtime-desired-flow-change-surface.md
  - documents/logs/2026-08-04/plan/plan-online-runtime-desired-flow.md
  - prompts/3_structure_prompt.md
---

# Implementation Structure: Desired online runtime flow

## Summary

The current online runtime does not yet follow the desired order. It recalibrates
thresholds at startup, keeps endpoint scalars instead of window vectors, creates
verification data before triage, and runs verification before the current A2
action.

The proposed delivery path has seven phases. Each phase produces one observable
result and leaves a clear boundary for the next phase. The stages below describe
responsibilities and outcomes only; they do not prescribe individual functions
or code edits.

## Request

Turn the approved implementation plan into higher-level stages for every phase.
Keep the canonical online ontology names. Keep the solution simple: no explicit
prediction-finalisation mechanism, no separate `verification_outcome`, no
`online_mlp_projector` for A0, and no persistent `recurrent_signature_set`.

## Confirmed context

- The desired order is documented in the research report and the online runtime
  flow note: score the current `causal_window`, update point-level state, then
  classify `triage_region`, perform the current action, run `verification_cycle`,
  and emit the event.
- The ontology makes `stage_b_best_checkpoint` and `threshold_artifact` offline
  outputs that the online phase reads.
- `window_point_scores`, `current_window_ewma_point_scores`, and
  `window_point_predictions` are vectors indexed by point identity. They are
  not aliases of endpoint scalar fields.
- A point without a previous EWMA value uses its current score directly. A
  point that remains in later sliding windows is updated again; a point that no
  longer appears keeps its latest prediction.
- `VerificationResult` remains the per-entry verification result, and
  `was_adapted` remains buffer bookkeeping. The design does not add a common
  verification result field.

## Scope

### In scope

- Offline-to-online threshold and checkpoint handoff.
- A0/A1/A2 model and update boundaries.
- Absolute point identity and vector score state.
- Triage, hard-old guard, update, verification, and cycle-local signatures.
- Runtime state, checkpoint resume, event records, metrics, retention, and demo.
- Direct read-only UI event delivery and end-to-end verification.

### Out of scope

- Replacing the four-region `classify_online_window` rule.
- Changing the formulas inside the existing hard-old and masked PNN loss
  functions.
- Removing `VerificationResult`, `was_adapted`, or the verification buffer.
- Running a full benchmark matrix before one concrete smoke path passes.

## Proposed phases

### Phase 1: Establish the offline calibration handoff

**Result:** The online phase can load one verified `threshold_artifact` and the
matching `stage_b_best_checkpoint` without recalibrating thresholds.

**Stages:**

1. **Canonical artifact contract** — confirm the ontology field used to locate
   the offline artifact and define the required thresholds, identity fields, and
   checkpoint provenance.
2. **Offline artifact production** — produce the four online thresholds using
   the same point-level score rules that the online phase will use.
3. **Online resolution gate** — resolve the artifact and checkpoint from the
   same offline run, then validate their identity before the first window.
4. **Startup handoff** — remove online startup calibration from the THESIS path
   and make the loaded artifact part of the initial runtime context.

**Depends on:** The offline-first ontology and the existing checkpoint/artifact
identity rules.

**Verification:**

- Automated: artifact schema, identity, and no-recalibration tests pass.
- Manual: one startup report shows matching artifact and checkpoint provenance.

**Risks:** Old artifacts may not have the new fields. Reject them or route them
through an explicitly labeled legacy path; never reinterpret them silently.

**Complete when:** A valid artifact reaches the online loop, and an invalid or
mismatched artifact stops the run before stream state changes.

### Phase 2: Separate inference-only and adaptation variants

**Result:** A0 runs inference only, while A1/A2 own the projector and update
resources they need.

**Stages:**

1. **Variant construction boundary** — make model construction aware of
   `online_variant` before creating mutable online modules.
2. **A0 inference path** — use `source_hidden`, create no
   `online_mlp_projector`, create no optimizer, and finish the current step
   after prediction state is ready.
3. **A1/A2 adaptation path** — preserve the projector-only trainable boundary
   and create update resources only for these variants.
4. **Loss configuration path** — carry `input_window_threshold` separately from
   `online_point_ewma_threshold` and obtain `lambda_online_contrastive` from the
   model configuration.

**Depends on:** Phase 1 startup context and the existing online variant
ontology.

**Verification:**

- Automated: variant, trainable-surface, and loss-branch tests pass.
- Manual: parameter summaries show no projector for A0 and only projector
  parameters trainable for A1/A2.

**Risks:** Generic callers may assume the projector always exists. Keep the
variant boundary explicit and fail clearly for invalid accesses.

**Complete when:** A0 cannot enter an adaptation or verification path, and A1/A2
receive the correct thresholds and configured loss weight.

### Phase 3: Make point identity and vector state explicit

**Result:** Every online point has a stable absolute identity, and each window
produces score, EWMA, and prediction vectors before adaptation.

**Stages:**

1. **Absolute point identity** — carry increasing `absolute_indices` through the
   stream, batch collation, validation, and verification-entry reconstruction.
2. **Shared score representation** — preserve all `window_point_scores` instead
   of only the endpoint value.
3. **Per-point EWMA update** — merge overlapping windows by absolute index and
   apply the first-seen rule or the configured EWMA update rule for each point.
4. **Prediction state** — create `window_point_predictions` from the current
   EWMA vector and keep the latest value for points that leave later windows.
5. **Minimal end-to-end vector path** — pass the vectors from one window through
   the event boundary without requiring model adaptation.

**Depends on:** Phase 1 threshold names and the causal-window ontology.

**Verification:**

- Automated: batch-contract and EWMA tests cover first-seen and overlapping
  points; an integration test covers prediction persistence.
- Manual: two overlapping windows show two updates for shared absolute indices.

**Risks:** A state map can grow unnecessarily. Keep only active point state and
the latest values required by records and the UI; do not add finalization state.

**Complete when:** A vector-only A0 stream produces correct values for the first
window, an overlapping window, and a point that no longer appears.

### Phase 4: Enforce the desired per-window order

**Result:** The runtime classifies `triage_region` before verification work and
executes the current window's action before `verification_cycle`.

**Stages:**

1. **Triage stage** — classify the four-region `triage_region` from the current
   window scores without changing it based on guard history.
2. **Current action stage** — let A2 handle accepted
   `hard_old_normality`, let `gray_zone` enter the buffer, and let normal and
   strong-anomaly windows do nothing.
3. **Guard stage** — apply `hard_old_interval_guard` only when admitting a
   hard-old update. A rejection keeps the region and sets `did_update=false`.
4. **Verification stage** — run `verification_cycle` only on buffered
   `verification_entry` objects and use its `pnn_mask` for verified updates.
5. **Cycle-local signature stage** — build and discard
   `recurrent_signature_set` inside one cycle; remove the global
   `signature_history` route.

**Depends on:** Phase 2 variant boundaries and Phase 3 vector event data.

**Verification:**

- Automated: spy-based order checks and buffer/guard tests prove the call order.
- Manual: one hard-old overlap and one gray-zone entry show separate behavior.

**Risks:** Reordering changes when the model mutates. Use deterministic one-window
fixtures before running a longer stream.

**Complete when:** The runtime order is
`score -> EWMA/prediction -> triage -> current action -> verification_cycle ->
record`, and no current hard-old window becomes `gray_zone` because of the guard.

### Phase 5: Persist and restore the vector runtime state

**Result:** A resumed stream produces the same next-window result as an
uninterrupted stream.

**Stages:**

1. **Vector runtime schema** — replace scalar previous-EWMA state with the
   absolute-index point state needed for the next window.
2. **Mutable verification state** — persist buffered entries, verification
   history, and hard-old intervals, but not a long-lived signature set.
3. **Checkpoint version gate** — increment the runtime schema version and reject
   incompatible scalar payloads before restoring mutable objects.
4. **Resume equivalence** — restore point state and verify the next vector event
   against an uninterrupted run.

**Depends on:** Phase 3 point state and Phase 4 buffer/guard ownership.

**Verification:**

- Automated: runtime-state round-trip and resume-equivalence tests pass.
- Manual: stop and resume one smoke run at a fixed cursor and compare the next
  event.

**Risks:** Old checkpoints cannot supply missing vector values. Keep them for
rollback and fail with a schema message instead of filling zeros.

**Complete when:** The new state restores all required vector and buffer data,
and old incompatible state is rejected before mutation.

### Phase 6: Publish vector results to records and the live demo

**Result:** Reports, retention artifacts, plots, and the live UI use the same
vector event contract.

**Stages:**

1. **Event record boundary** — expose point vectors, thresholds, triage region,
   update flag, and loss summary in one `online_event_record`.
2. **Metrics and retention** — persist vectors and explicit update outcomes;
   never infer `did_update=true` from a missing field.
3. **Live event delivery** — send a copy-safe, read-only event to the UI after
   predictions are ready.
4. **Demo compatibility** — plot THESIS vectors by absolute index while keeping
   a clearly separate adapter for scalar baseline records.

**Depends on:** Phase 3 vector state, Phase 4 event order, and Phase 5 state
schema.

**Verification:**

- Automated: record, retention, demo, and callback-isolation tests pass.
- Manual: the UI updates a shared point when a later overlapping window arrives.

**Risks:** A UI callback must not mutate runtime data. Pass a copy or immutable
view and verify that callback changes do not affect the stream.

**Complete when:** One online event can drive the report, retention file, plot,
and live callback without converting one endpoint scalar into a fake vector.

### Phase 7: Validate the complete flow and align specifications

**Result:** The runtime, tests, reports, ontology, and pseudocode describe the
same `full_spec_v3` behavior.

**Stages:**

1. **Focused validation** — run contract, variant, EWMA, triage, verification,
   state, record, and demo tests in dependency order.
2. **One concrete smoke path** — run one existing online benchmark combination
   and inspect artifact, checkpoint, vector records, runtime state, and UI event.
3. **Protocol status** — mark the runtime as `full_spec_v3` only after the smoke
   path passes.
4. **Specification alignment** — update ontology, full specification, and debug
   pseudocode to record the final vector fields, cycle-local signatures, A0
   boundary, and callback behavior.

**Depends on:** All previous phases.

**Verification:**

- Automated: focused tests and one smoke run pass.
- Manual: read the final pseudocode from start to finish and check that every
  object has one canonical name and one owner.

**Risks:** Documentation can claim behavior that the runtime does not provide.
Update the status and specification only after end-to-end evidence exists.

**Complete when:** The full flow passes, the report carries `full_spec_v3`, and
the documents no longer describe endpoint-only or pre-triage verification as
the desired runtime.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1. Calibration handoff | Offline ontology and artifact identity rules | Verified online startup |
| 2. Variant boundaries | Phase 1 startup context | Safe A0/A1/A2 actions |
| 3. Point/vector state | Threshold names and causal-window contract | Vector event data |
| 4. Runtime order | Variant boundaries and vector event data | Correct triage/update/verification flow |
| 5. Runtime persistence | Point state and buffer ownership | Resume-equivalent execution |
| 6. Output and demo | Vector events and runtime state | Reports and live display |
| 7. Final validation | All previous phases | `full_spec_v3` status and aligned documents |

## Decisions confirmed

- A0 does not create `online_mlp_projector` or an optimizer.
- A first-seen point uses its current score as its EWMA value.
- Sliding-window updates do not need explicit prediction finalization.
- `recurrent_signature_set` is cycle-local.
- `VerificationResult` and `was_adapted` remain; no common
  `verification_outcome` is added.
- The UI receives events directly during the online loop.
- The final protocol status is `full_spec_v3`.

## Non-blocking uncertainties

- Phase 1 must confirm the canonical configuration field for locating the
  offline `threshold_artifact` before implementation starts. This is a contract
  gate, not a reason to change the later phase order.
- `absolute_indices` remains owned by `causal_window`; adding it directly to
  `online_event_record` requires a separate ontology decision.

## Feedback requested

- Anh xem giúp thứ tự bảy phase này đã đúng với cách anh muốn triển khai chưa.
- Có phase nào cần tách nhỏ hơn hoặc gộp lại trước khi chuyển sang bản detail
  theo `4_detail_prompt.md` không?
