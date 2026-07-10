---
date: 2026-07-10T17:49:18+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Structure outline for full-spec-v2 gap remediation"
tags: [structure, full-spec-v2, online-tta, pnn, ttl, hard-old, demo]
status: ready_for_detail
last_updated: 2026-07-10
last_updated_by: Codex
---

# Structure outline: detailed programming plan for `full-spec-v2`

## Overview

The implementation will preserve the already-working offline O0/O1 pipeline and
the public online benchmark entrypoints, then close the missing online-TTA and
demo contracts in dependency order. The central design is composition: pure
threshold/signature/loss helpers provide calculations, stateful buffers own
their lifecycle, and the online engine remains the visible coordinator.

The first implementation slice is intentionally small: entity-specific
threshold artifacts, exact four-region triage, projector-only update checks,
and compatibility tests. Prototype filtering, verification-cycle state,
token-level losses, detailed logging, and queue-based demo behavior are added
only after that slice passes.

## Implementation Phases

### 1. Contract freeze and entity-scoped calibration

**Purpose:** establish executable contracts before changing online behavior.

**Main boundaries:**

- `tests/contract/` owns public output, checkpoint, optimizer, and threshold
  assertions.
- `src/engine/online_tta/threshold_calibration.py` owns pure score and quantile
  calculations.
- `src/protocols/threshold_artifact.py` owns the entity-scoped serialized
  threshold schema.
- `src/engine/online_tta/online_engine.py` owns one-artifact-per-entity runtime
  selection.

**Contract flow:**

```text
clean validation for one entity
    -> point scores and EWMA scores
    -> B_window, A_low, A_high
    -> artifact with entity_id and provenance
```

**Engineering principles:** single responsibility for calibration versus
artifact serialization; factory/registry boundaries remain unchanged; no
validation labels or gradients enter calibration.

**Gate:** deterministic threshold tests, entity mismatch rejection, and the
existing O0-A0 smoke run pass.

### 2. Online scoring and exact triage partition

**Purpose:** replace the current shared-threshold approximation with the
specification's input/latent threshold band while preserving the public
`classify_online_window()` function.

**Main boundaries:**

- `src/engine/online_tta/triage.py` owns the four-region truth table.
- `src/engine/online_tta/online_engine.py` obtains the input-window MSE and
  latent-window memory score, then passes them to triage.
- `tests/unit/` owns boundary truth-table tests.

**Triage flow:**

```text
input <= B_window                         -> normal
input > B_window, latent <= A_low         -> hard_old_normality
input > B_window, A_low < latent <= A_high -> gray_zone
input > B_window, latent > A_high         -> strong_anomaly
```

**Engineering principles:** keep legacy key aliases at the facade only;
prefer explicit data objects or dictionaries with named threshold fields;
avoid embedding calibration formulas inside the classifier.

**Gate:** equality and near-boundary cases match the truth table exactly.

### 3. Prototype-aware verification and buffer lifecycle

**Purpose:** implement the missing recurrent-signature PNN mask and the
verification-cycle lifecycle without distributing state across the engine.

**Main boundaries:**

- `src/engine/online_tta/signature_verification.py` owns pure nearest-codeword,
  anomaly-radius, ordered top-k signature, recurrence, and token-mask helpers.
- `src/engine/online_tta/verification_buffer.py` remains the state owner and is
  extended with `try_admit()`, `should_verify()`, result marking, and cycle
  finalization.
- `src/engine/online_tta/ttl_buffer.py` remains compatible for older callers,
  while the specification PNN path uses verification-cycle TTL state.
- `src/engine/online_tta/online_engine.py` only sequences admission and cycle
  calls.

**Verification flow:**

```text
window scores
    -> gray-zone admission with non-overlap check
    -> nearest anomalous-codeword filter
    -> ordered continuous top-3 signatures
    -> recurrent signatures across windows
    -> M_pnn token mask
    -> verification cycle when size=8 and new data exists
    -> adapted removal / unresolved TTL decrement
```

**Engineering principles:** pure helper functions do not mutate memories;
composition is preferred over another controller hierarchy; the existing
buffer methods remain a compatibility facade.

**Gate:** tests prove initial TTL 2, no per-step decrement, overlap rejection,
cycle-trigger behavior, and exact unresolved/adapted retention.

### 4. A1/A2 adaptation strategies and projector safeguards

**Purpose:** align online objectives, optimizer lifecycle, and frozen-component
behavior with the locked A1/A2 contracts.

**Main boundaries:**

- `src/engine/online_tta/online_losses.py` owns masked PNN reconstruction,
  hard-old hinge, and token multi-positive InfoNCE primitives.
- `src/engine/online_tta/online_optimizer.py` owns
  `build_online_optimizer()` and projector-only parameter groups.
- `src/engine/online_tta/online_engine.py` selects A0/A1/A2 strategies and
  creates one optimizer per adaptation event.
- `src/models/online_adaptation.py` remains the frozen source/reference model
  and residual projector entrypoint.
- `configs/model/online_adaptation.yaml` and
  `configs/task/online_adaptation.yaml` expose the locked settings.

**Strategy flow:**

```text
A0 -> score and predict, no optimizer event
A1 -> verified PNN mask -> masked reconstruction -> projector step
A2 -> hard-old hinge OR PNN reconstruction
      + token contrastive regularizer where applicable
      -> fresh AdamW projector-only step
```

**Engineering principles:** Strategy pattern is represented by explicit
variant dispatch, not subclasses; the optimizer factory makes reset ownership
visible; detached keys and checksum tests protect frozen modules.

**Gate:** loss boundary tests, fresh optimizer-state tests, gradient clipping,
projector-only updates, and frozen source/memory/head checksums pass.

### 5. Engine integration, reporting, and experiment matrix

**Purpose:** integrate the validated primitives into a causal online loop and
prove all required O/A combinations remain runnable.

**Main boundaries:**

- `src/engine/online_tta/online_engine.py` owns event ordering only.
- `scripts/run_thesis_online_benchmark.py` remains the stable wrapper.
- `src/protocols/threshold_artifact.py` and report writers own additive output
  schemas.
- `scripts/preflight_full_benchmark_matrix.py` verifies matrix completeness and
  per-entity artifact references.
- `tests/integration/` owns causal stream and report tests.

**Engine flow:**

```text
score -> EWMA -> triage -> hard-old guard or buffer admission
      -> verification cycle if due
      -> event optimizer and projector update
      -> future-only prediction finalization
      -> structured metrics and artifacts
```

**Required matrix:** `O0-A0`, `O0-A2`, `O1-A0`, `O1-A2`; optional A1 follows
after the required variants pass.

**Engineering principles:** facade wrapper preserves CLI; report fields are
additive; the registry/factory remains the only model construction boundary;
no test labels or future stream windows enter adaptation.

**Gate:** entity-specific reports, threshold provenance, causal ordering,
required logs, and the active full matrix preflight pass.

### 6. Queue-based demo and final documentation

**Purpose:** implement the user-selected queue ownership and keep demo logic
separate from official evaluation.

**Main boundaries:**

- New `demo/stream_queue.py` owns `Queue`, producer, consumer, timeout,
  backpressure, speed, and stop signaling.
- `demo/online_replay.py` owns model calls and replay state.
- `demo/app.py` owns UI controls and plots only.
- `tests/unit/test_demo_stream_queue.py` owns queue behavior tests.

**Demo flow:**

```text
sequence -> stream_queue producer -> Queue -> consumer
         -> online_replay model step -> demo state -> plot/UI
```

The demo reads finished artifacts and never calibrates thresholds, changes
model parameters, or supplies labels to adaptation.

**Engineering principles:** separation of concerns between queue, replay, and
UI; dependency injection for model/state in tests; no coupling from official
evaluation back into demo code.

**Gate:** empty queue, backpressure, pause/resume, first complete window,
label-free mode, and demo isolation tests pass.

## Cross-phase test structure

```text
contract tests
    -> pure unit tests
        -> buffer/loss/optimizer integration tests
            -> online engine causal tests
                -> O0/O1/A0/A1/A2 matrix smoke
                    -> full repository acceptance
```

Every phase must run its focused tests before and after edits. A phase cannot
be considered complete if a public contract changes without an explicit
compatibility assertion.

## Structure risks requiring attention in the detailed plan

1. Existing legacy threshold aliases must not override entity-specific fields.
2. Prototype buffers may be registered model buffers; helper code must not
   re-register or mutate them accidentally.
3. Verification-cycle state must survive checkpoint/report serialization if the
   online run is resumed.
4. A2 token contrastive keys must be detached except for the current anchor.
5. Fresh optimizer creation must not accidentally reset the projector weights,
   only optimizer state.
6. The current repository has legacy tests/configs outside the active w20
   matrix; the detailed plan must distinguish migration work from active
   full-spec acceptance.

## Readiness for detail phase

This structure is ready for expansion with `4_detail_prompt.md`. The detail
phase should specify exact symbols, input/output shapes, mutation ownership,
test fixtures, migration order, and rollback boundaries for each sub-batch.
