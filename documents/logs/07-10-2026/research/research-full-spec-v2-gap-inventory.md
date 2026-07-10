---
date: 2026-07-10T17:12:14+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Codebase gaps against full-spec-v2.md"
tags: [research, full-spec-v2, online-tta, pnn, ttl, hard-old]
status: complete
last_updated: 2026-07-10
last_updated_by: Codex
---

# Research: Codebase gaps against `full-spec-v2.md`

## Research question

Which implemented code paths do not yet satisfy the locked behavior described
in [documents/spec/full-spec-v2.md](../../../spec/full-spec-v2.md), especially
recurrent-signature PNN masking, verification-cycle TTL, and hard-old loss?

## Scope and method

This note documents the repository as it exists at commit
`8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1`. It compares the specification with
the active online engine, model, buffers, configs, tests, and the smoke runs
already executed for O0/O1 and A0/A2. It records deviations only; it does not
propose implementation changes.

The design files named by the research prompt (`documents/design/idea.md` and
`documents/design/design_starter.md`) are not present in the current checkout,
so they could not be used as additional intent evidence.

## Current pipeline that is present

### Offline data and training

The active benchmark configs use SMD machine entities with window size 20,
normalization and split-specific loaders. The two-stage runner executes Stage A,
Stage B, and evaluation. O1 enables the point-score loss through the existing
model configuration fields. Memory initialization is called from the training
engine and records train-feature source labels in checkpoint extra state.

Relevant implementation surfaces:

- `src/models/thesis_multitask_state_mixin.py:277-327` initializes memories from
  the training loader and marks them initialized.
- `src/models/thesis_multitask_state_mixin.py:578-640` creates continuous and
  class-stratified discrete memories.
- `scripts/run_two_stage_offline_pretraining.py` orchestrates Stage A and Stage B.
- `scripts/generate_smd_benchmark_configs.py:108-120` enables O1 score-loss
  fields without adding a separate `lambda_score`.

### Online model and stream

The online model loads a frozen THESIS checkpoint, creates reference and online
encoder adapters, and exposes `online_mlp_projector` as the trainable group.
`src/engine/online_tta/online_optimizer.py:24-54` verifies that only projector
parameters require gradients. The stream uses stride-one windows and applies
the configured EWMA point score.

The current O0/O1 offline and A0/A2 online smoke runs completed and wrote their
reports, thresholds, scores, metrics, and checkpoints under
`outputs/benchmark_smoke/`.

## Detailed gaps against the specification

### 1. Online configuration contract is incomplete

`full-spec-v2.md:367-470` requires explicit fields for online EWMA, window
thresholds, A0/A1/A2 update switches, PNN buffer admission, TTL policy, hard-old
guard, online optimizer, and contrastive settings.

The active shared configs contain only the older projector/alignment surface:

- `configs/model/online_adaptation.yaml:1-11` has projector dimensions and
  alignment lambdas, but no online loss, threshold-band, hard-old, PNN, or
  anomalous-codeword settings.
- `configs/task/online_adaptation.yaml:1-12` has a reference checkpoint,
  projector target, stream limits, and view noise, but no A0/A1/A2 behavior
  switches, buffer size, TTL values, or optimizer reset policy.

The runner receives `online_variant` as a CLI argument in
`scripts/run_thesis_online_benchmark.py`, but the YAML itself does not encode
the locked variant contract.

### 2. Window threshold calibration is not implemented as specified

The specification requires three clean-validation thresholds:

- input-window `B_window` at q99;
- latent band `A_low` at q95;
- latent band `A_high` at q99.

The active engine builds all triage thresholds from one online EWMA point
threshold in `src/engine/online_tta/online_engine.py:311-317`. No input-window
or latent-window calibration artifact is produced.

`src/engine/online_tta/online_engine.py:396-418` uses the final point score as
`raw_point_score` and the model alignment loss as `latent_window_score`; these
are not the specified input-window MSE and latent-window memory score.

### 3. Triage predicates differ from the locked predicates

`full-spec-v2.md:871-944` defines:

- normal: input score `<= B_window`;
- hard-old: input `> B_window` and latent `<= A_low`;
- gray zone: input `> B_window` and `A_low < latent <= A_high`;
- strong anomaly: input `> B_window` and latent `> A_high`.

The current classifier in `src/engine/online_tta/triage.py:22-57` instead:

- marks strong anomaly when either score exceeds the same strong threshold;
- marks hard-old when both scores are below one shared threshold;
- marks PNN candidate using `input <= candidate` and `latent >= candidate`.

Therefore the current labels are not equivalent to the locked four-region
partition.

### 4. Recurrent-signature PNN mask is absent

`full-spec-v2.md:1076-1122` requires nearest discrete-codeword filtering,
ordered top-3 continuous-prototype signatures, recurrence across non-overlapping
windows, and a token-level `M_pnn` mask.

The current path in `src/engine/online_tta/online_engine.py:421-441` only stores
window start/end and a point score when the triage label is `pnn_candidate`.
There is no code for:

- anomalous discrete-codeword radius checks;
- continuous top-k prototype IDs;
- signature recurrence;
- token-level PNN masks;
- verification of a buffer after eight admitted windows.

`src/engine/online_tta/online_losses.py:26-31` accepts an optional generic
mask, but the online engine passes `batch.get("mask")`, not a computed PNN mask.

### 5. Verification buffer admission and verification-cycle state are incomplete

The specification requires non-overlapping gray-zone admission, an entry with
`ttl_remaining=2`, a verification trigger after buffer size eight plus a new
entry, and removal/retention decisions after each verification cycle.

`src/engine/online_tta/verification_buffer.py:15-30` provides an `admit()`
method, but `_update_online_window_buffers()` calls `add()` directly at
`src/engine/online_tta/online_engine.py:429-435`; the admission guard is not
used. Entries also do not carry status, TTL, or `was_adapted` fields.

The current `TTLBuffer` in `src/engine/online_tta/ttl_buffer.py:7-28` expires
items by stream step (`expires_at`), while the specification requires TTL to
decrement only after verification cycles. `online_engine.py:743-772` constructs
the TTL buffer with `protocol_config["window_size"]`, not the locked two-cycle
policy, and no verification-cycle update is invoked.

### 6. Hard-old loss is not the locked hinge objective

`full-spec-v2.md:951-1022` requires hard-old adaptation only in A2, a
non-overlap guard, and

```text
relu(s_input_window_online - B_window)^2
```

The current `compute_a2_hard_old_reconstruction_loss()` in
`src/engine/online_tta/online_losses.py:34-39` is the same masked mean-squared
error helper used for A1. It has no threshold hinge and returns a non-zero loss
even when reconstruction is already below `B_window`.

`src/engine/online_tta/online_engine.py:542-553` applies this reconstruction
loss to every non-strong A2 decision, including gray-zone and PNN candidates,
and does not consult a hard-old non-overlap guard.

### 7. A2 contrastive loss does not implement the required token-level objective

The specification requires projected token anchors, same-token frozen-source
positives, recurrent-signature positives, anomalous discrete codeword negatives,
optional known-anomaly negatives, and ignoring non-PNN/non-anomaly tokens.

The current `compute_a2_online_contrastive_loss()` in
`src/engine/online_tta/online_losses.py:42-55` pools each window, compares
projected and reference batch means, and uses a batch diagonal objective. It
does not receive codewords, token masks, signatures, or known-anomaly buffers.

### 8. Online optimizer contract is only partially explicit

The projector-only filter is present, but the specification also requires a
fresh AdamW optimizer per adaptation event, learning rate `1e-4`, weight decay
`1e-4`, one step, no scheduler, and gradient clipping at `0.5`.

The generated online configs currently set optimizer learning rate `0.001` and
weight decay `0.0` in `scripts/generate_online_benchmark_configs.py`.
The runtime builds one optimizer in `_build_runtime_online_context()` and reuses
it across the stream. The current tests verify projector ownership, but not the
fresh-optimizer/reset-per-event contract.

### 9. Required online logging fields are not all emitted

The spec lists input/latent window scores, buffer admission/rejection counts,
signature counts, PNN token counts, TTL removals, separate hard/PNN/contrastive
losses, and frozen-component gradient norms.

`src/engine/online_tta/online_engine.py:456-467` currently emits only step,
point scores, threshold, prediction, update flag, total loss, triage label, and
buffer lengths. The detailed diagnostic fields are absent.

### 10. Demo implementation is not mapped to the locked queue contract

The spec requires an offline replay and a queue-based online producer/consumer
demo. The repository has `demo/offline_replay.py`, `demo/online_replay.py`, and
`demo/app.py`, but the current research pass did not find the exact
`Queue`/producer/consumer flow or the full spec logging fields in the THESIS
online engine. Existing demo tests cover state builders and plotting helpers,
not the complete queue behavior described in `full-spec-v2.md:1427-1579`.

## Evaluation and safety findings

The active threshold artifact records clean-validation provenance and the smoke
reports mark test labels as metrics-only. The offline runner therefore has
evidence for the split-level threshold safety contract. However, because the
online input/latent window thresholds are not separately calibrated, the online
triage safety claims cannot yet be mapped one-to-one to sections 8 and 10 of
the specification.

The existing tests cover projector ownership, A0 no-update, basic A1/A2 update
dispatch, triage branches, and buffer primitives. They do not cover the
specification's missing recurrent-signature, verification-cycle, hinge-loss,
optimizer-reset, or token-level contrastive contracts.

## Code references

- `documents/spec/full-spec-v2.md:1024-1312` - PNN buffer, loss, optimizer, and TTL contracts.
- `src/engine/online_tta/triage.py:17-57` - current triage predicates.
- `src/engine/online_tta/online_engine.py:311-317` - current threshold derivation.
- `src/engine/online_tta/online_engine.py:396-441` - current score and buffer flow.
- `src/engine/online_tta/online_engine.py:528-559` - current A1/A2 update losses.
- `src/engine/online_tta/online_losses.py:26-55` - current online losses.
- `src/engine/online_tta/verification_buffer.py:7-39` - current verification buffer primitive.
- `src/engine/online_tta/ttl_buffer.py:7-34` - current stream-step TTL primitive.
- `src/models/online_adaptation.py:216-232` - frozen adapters and projector.
- `src/engine/online_tta/online_optimizer.py:24-54` - projector-only trainability assertion.
- `configs/model/online_adaptation.yaml:1-11` - current model config surface.
- `configs/task/online_adaptation.yaml:1-12` - current task config surface.

## Open questions for the planning stage

1. Should the missing `B_window`, `A_low`, and `A_high` artifacts be calibrated
   per entity or per experiment matrix run?
2. Should the recurrent signature map live in the online engine or in a separate
   buffer-verification module?
3. Should the existing `VerificationBuffer` be extended in place or replaced by
   a verification-cycle controller while preserving its public methods?
4. Should A1/A2 optimizer creation remain in the engine or move to a per-event
   optimizer factory so the reset policy is explicit?
5. Which existing demo entrypoint is the official owner of the queue-based
   producer/consumer contract?

## Follow-up decisions from the user

### Threshold scope

The user decided that `B_window`, `A_low`, and `A_high` must be calibrated
separately for each machine/entity. A threshold artifact therefore belongs to
one entity and must not be silently reused by another entity.

### Recommended ownership for the remaining implementation choices

To keep the code simple and educational, the following choices are recommended:

1. Put recurrent-signature calculations in a small pure helper module. The
   online engine should only call the helper and own the high-level sequence.
   This separates “calculate a signature” from “decide when to adapt”.
2. Extend `VerificationBuffer` in place with explicit verification-cycle
   methods. The same class keeps window entries, TTL, and admission state, so a
   reader does not need to follow a second controller class.
3. Add a small `build_online_optimizer()` helper in the existing optimizer
   module. The engine calls it once per adaptation event, making the reset rule
   visible without introducing an additional class hierarchy.

### Decision still required: demo queue owner

The user requested an explicit choice among these options:

- **Option A — `demo/online_replay.py` owns the queue.** This file creates the
  queue, producer, consumer, and replay state. It is the smallest design and
  keeps all online replay behavior in one place.
- **Option B — new `demo/stream_queue.py` owns the queue.** The queue logic is
  isolated into a small reusable module; `demo/online_replay.py` only connects
  it to the model. This is cleaner if future demos will reuse the same stream
  mechanism, but adds one more file for a beginner to read.
- **Option C — `demo/app.py` owns the queue.** The UI entrypoint controls
  production and consumption directly. This is easy to launch, but mixes UI
  code with data-flow and model-execution logic.

The user selected **Option B**. The planned ownership boundary is therefore:

- `demo/stream_queue.py` owns the standard-library queue, producer, consumer,
  empty-queue handling, and point-by-point stream timing.
- `demo/online_replay.py` connects the queue module to the online model and
  replay state.
- `demo/app.py` remains the UI entrypoint and does not own queue mechanics.
