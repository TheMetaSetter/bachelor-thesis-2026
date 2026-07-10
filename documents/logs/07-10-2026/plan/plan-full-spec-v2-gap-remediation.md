---
date: 2026-07-10T17:24:19+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary implementation plan for full-spec-v2 gap remediation"
tags: [plan, full-spec-v2, online-tta, pnn, ttl, hard-old, demo]
status: preliminary_ready
last_updated: 2026-07-10
last_updated_by: Codex
---

# Preliminary plan: remediate gaps against `full-spec-v2.md`

## 1. Objective

Bring the active THESIS online-TTA and demo paths into alignment with the
locked behavior in `documents/spec/full-spec-v2.md`, while preserving current
public entrypoints, checkpoint loading, registry names, report schemas, and the
already-working O0/O1/A0/A2 smoke matrix.

The implementation must remain readable to a high-school-level reader:
responsibilities are separated into small functions and modules, each helper
has one clear purpose, and orchestration remains visible in the engine.

The research note found that offline O0/O1 and the minimum online smoke matrix
already execute. This plan therefore starts with characterization tests and
online contract artifacts instead of replacing the offline training pipeline.

## 2. Locked decisions and constraints

### 2.1 User decisions

- Calibrate `B_window`, `A_low`, and `A_high` separately for every machine/entity.
- Put recurrent-signature calculations in a small pure helper module.
- Extend `VerificationBuffer` in place with explicit verification-cycle methods.
- Add a small optimizer factory that creates a fresh optimizer per adaptation event.
- Use a new `demo/stream_queue.py` for queue producer/consumer ownership.

### 2.2 Preserved public contracts

- `scripts/run_thesis_online_benchmark.py` remains the online benchmark CLI.
- `run_thesis_online_tta_experiment()` remains the runtime entrypoint.
- `execute_online_tta_step()` remains importable and keeps A0/A1/A2 arguments.
- `OnlineAdaptationModel` keeps its public constructor and
  `online_mlp_projector` parameter group.
- `VerificationBuffer.admit()`, `add()`, `clear()`, `items()`, and `__len__()`
  remain callable.
- Online report keys, threshold artifact provenance, registry names, and
  checkpoint/state-dict compatibility remain stable unless a new additive field
  is required by the specification.
- Test labels remain metrics-only. No validation/test labels may control
  thresholds or adaptation.

### 2.3 Missing design context

`documents/design/idea.md` and `documents/design/design_starter.md` are absent
in the current checkout. The implementation must therefore use
`full-spec-v2.md`, active configs, source code, and tests as the authoritative
design context until those documents are restored.

## 3. Target architecture

```text
clean validation per entity
    -> point threshold + B_window + latent A_low/A_high
    -> entity threshold artifact

online stream window
    -> frozen source encoder and memories
    -> input/latent scores
    -> four-region triage
    -> hard-old guard OR VerificationBuffer
    -> signature verifier
    -> A1/A2 loss selection
    -> fresh projector-only optimizer event
    -> future point predictions and structured logs

demo stream sequence
    -> demo/stream_queue.py
    -> demo/online_replay.py
    -> demo state and plots
```

The engine owns sequencing and state transitions. Pure scoring, signature,
triage, loss, and threshold calculations live in focused helpers.

## 4. Phase plan

### Phase 0 — Freeze current contracts before changing online behavior

Add or extend:

- `tests/contract/test_full_spec_online_contract.py`
- `tests/contract/test_entity_threshold_artifact.py`
- `tests/integration/test_online_tta_spec_runtime.py`

Capture the current public report and checkpoint keys, then add failing tests
for the locked requirements:

- threshold artifacts carry `entity_id` and separate point/input/latent values;
- A0 never steps an optimizer;
- A1/A2 update only projector parameters;
- all frozen component checksums remain unchanged;
- online optimizer settings are `AdamW`, `1e-4`, `1e-4`, one step, no scheduler;
- no test labels are read by calibration or adaptation.

Run the existing focused online tests before and after adding assertions.

Acceptance: the baseline behavior is recorded, and every later phase has a
specific failing contract test rather than a behavior guess.

### Phase 1 — Entity-scoped threshold calibration

Modify:

- `src/engine/online_tta/online_engine.py`
- `src/protocols/threshold_artifact.py`
- `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`

Add small helpers, preferably in a new
`src/engine/online_tta/threshold_calibration.py`:

- `compute_input_window_score(window, reconstruction) -> Tensor`
- `compute_latent_window_score(outputs) -> Tensor`
- `calibrate_entity_thresholds(clean_validation_sequence, model, protocol) -> dict`

The calibration helper must:

1. Receive exactly one entity's clean validation sequence.
2. Compute offline point q99 and online EWMA q99 as currently supported.
3. Compute `B_window` at q99, `A_low` at q95, and `A_high` at q99.
4. Return an artifact containing `entity_id`, `window_size`, quantiles,
   source split, score rules, and values.
5. Run under `model.eval()` and `torch.no_grad()`.

The runtime context stores one artifact per entity. It must reject using an
artifact whose `entity_id` differs from the current stream entity.

Tests:

- clean-validation-only calibration;
- entity artifact separation;
- q95/q99 values on deterministic tensors;
- no gradient and no label access;
- backward-compatible reading of existing point-threshold fields.

Acceptance: each machine receives its own threshold artifact and the existing
O0-A0 smoke report still completes.

### Phase 2 — Specification-aligned four-region triage

Modify:

- `src/engine/online_tta/triage.py`
- `src/engine/online_tta/online_engine.py`

Keep `classify_online_window()` as the public function, but make its preferred
threshold keys explicit:

```text
input_window_threshold = B_window
latent_low_threshold = A_low
latent_high_threshold = A_high
```

Implement the exact order:

1. `input <= B_window` -> `normal`;
2. `input > B_window and latent <= A_low` -> `hard_old_normality`;
3. `input > B_window and A_low < latent <= A_high` -> `gray_zone`;
4. `input > B_window and latent > A_high` -> `strong_anomaly`.

Retain legacy threshold aliases only as a compatibility path when the new
artifact fields are absent; new artifacts must always use the specification
fields.

Tests cover boundary equality, every region, and missing-field errors.

Acceptance: triage results match the truth table in `full-spec-v2.md` for all
boundary cases.

### Phase 3 — Recurrent-signature verification helper

Add:

- `src/engine/online_tta/signature_verification.py`

The module contains pure, stateless operations:

- `nearest_discrete_codeword(hidden, codebook) -> (index, distance)`;
- `filter_known_anomaly_tokens(hidden, codebook, anomaly_radii) -> BoolTensor`;
- `ordered_continuous_signature(hidden, prototype_bank, topk=3) -> tuple[int, ...]`;
- `find_recurrent_signatures(signature_records) -> set[tuple[int, ...]]`;
- `build_pnn_token_mask(records, recurrent_signatures) -> BoolTensor`.

The helper receives detached tensors and does not mutate model memories. It
must use the same cosine-distance convention as codebook queries and preserve
token order.

Tests:

- top-3 ordering is deterministic;
- anomalous codeword radius filtering excludes known-anomaly tokens;
- recurrence requires more than one non-overlapping window;
- non-PNN and known-anomaly masks are mutually exclusive;
- empty signature and empty codebook inputs fail clearly or return documented
  empty masks.

Acceptance: the verifier returns a token-level `M_pnn` mask without changing
source model parameters or memory buffers.

### Phase 4 — VerificationBuffer and verification-cycle TTL

Modify:

- `src/engine/online_tta/verification_buffer.py`
- `src/engine/online_tta/ttl_buffer.py`
- `src/engine/online_tta/online_engine.py`

Extend `VerificationBuffer` in place. Each entry stores:

```text
window, start, end, ttl_remaining=2, status, was_adapted
```

Add small methods:

- `try_admit(entry) -> bool` using non-overlap checks;
- `should_verify() -> bool` when size reaches 8 and a new entry exists;
- `mark_verification_result(entry_id, adapted) -> None`;
- `finish_verification_cycle() -> dict[str, int]`.

The buffer must not repeatedly verify unchanged entries. Admission is the
first chance; each later verification cycle decrements unresolved TTL. Adapted
entries are removed immediately; unresolved entries are removed only at zero.

Replace stream-step expiry for this PNN path with cycle-based TTL ownership.
Keep the existing `TTLBuffer` public methods for older callers, but do not use
`expires_at` for the specification PNN lifecycle.

Tests:

- overlap rejection;
- initial TTL equals 2;
- verification requires a new admission;
- adapted removal;
- unresolved retention for exactly two cycles;
- no decrement on ordinary stream steps.

Acceptance: A1/A2 buffer state follows sections 12 and 15 of the spec.

### Phase 5 — Correct A1/A2 losses and optimizer lifecycle

Modify:

- `src/engine/online_tta/online_losses.py`
- `src/engine/online_tta/online_optimizer.py`
- `src/engine/online_tta/online_engine.py`
- `configs/model/online_adaptation.yaml`
- `configs/task/online_adaptation.yaml`
- `scripts/generate_online_benchmark_configs.py`

Add pure loss functions:

- `compute_pnn_reconstruction_loss(reconstruction, target, pnn_mask)`;
- `compute_hard_old_hinge_loss(reconstruction, target, b_window)`;
- `compute_token_multi_positive_infonce(anchors, positives, negatives, temperature)`.

The hard-old function returns zero when the input-window reconstruction score is
at or below `B_window`. The PNN function uses only `M_pnn` tokens. The A2
contrastive function always includes anomalous discrete codewords as negatives,
optionally adds known-anomaly keys, and ignores unrelated tokens.

Add `build_online_optimizer(model, learning_rate=1e-4, weight_decay=1e-4)` to
`online_optimizer.py`. It must select only projector parameters, use AdamW, and
be called once per adaptation event. Apply one step and gradient clipping at
`0.5`; do not create a scheduler.

Update config fields to make A0/A1/A2 behavior explicit while keeping CLI
variant selection compatible. Preserve `target_param_group=projector_params`.

Tests:

- hinge zero/non-zero cases;
- masked PNN denominator and empty-mask behavior;
- anomalous codeword negative presence;
- optimizer class/hyperparameters and fresh state per event;
- projector changes while source encoder, memories, reconstruction heads, and
  classification heads remain bitwise unchanged.

Acceptance: A1 computes only masked PNN reconstruction; A2 selects hard-old or
PNN objectives according to triage and verification state.

### Phase 6 — Integrate engine sequencing and required logs

Modify:

- `src/engine/online_tta/online_engine.py`
- `scripts/run_thesis_online_benchmark.py`
- `src/protocols/threshold_artifact.py`

Refactor the stream loop into explicit steps:

```text
score window -> update EWMA -> triage -> guard/admit -> verify if due
-> create event optimizer -> adapt projector -> finalize future prediction
```

Add structured fields for:

- input/latent scores;
- triage counts;
- admitted/rejected windows;
- total/remaining/recurrent signatures;
- PNN and known-anomaly token counts;
- TTL kept/removed counts;
- hard-old, PNN, contrastive, and total losses;
- projector and frozen-component gradient norms.

Do not retroactively rewrite finalized point predictions. Keep report schema
additive and preserve existing keys.

Tests:

- one synthetic stream with known region transitions;
- causal order, no future window access;
- A0/A1/A2 event counts;
- report and threshold artifact schema;
- frozen checksum and log-field assertions.

Acceptance: the engine executes the same public wrapper for all six O/A
combinations and produces auditable per-entity reports.

### Phase 7 — Queue-based demo ownership

Add:

- `demo/stream_queue.py`

Modify:

- `demo/online_replay.py`
- `demo/app.py` only where wiring is required
- `tests/test_demo_stream_queue.py`
- existing `tests/test_demo_state.py` if interfaces change

`stream_queue.py` owns:

- `queue.Queue(maxsize=...)`;
- point-by-point producer;
- consumer timeout/empty handling;
- speed/delay control;
- clean stop signaling.

`online_replay.py` owns model calls and replay state updates. `app.py` owns UI
controls and plotting only.

Tests cover queue empty, queue full/backpressure, fewer-than-window points,
first complete window, pause/resume, speed changes, and label-free operation.

Acceptance: demo behavior cannot change official metric artifacts or threshold
calibration.

### Phase 8 — Matrix verification and documentation

Modify/add:

- `tests/integration/test_full_spec_matrix_smoke.py`
- `tests/contract/test_full_spec_config_contract.py`
- `scripts/preflight_full_benchmark_matrix.py`
- `documents/design/design_starter.md` if the missing file is restored
- this plan or a follow-up detail artifact with final paths

Validate all required combinations:

```text
O0-A0, O0-A2, O1-A0, O1-A2
```

Run optional A1 after A0/A2 passes. For every entity, assert an entity-matched
threshold artifact and exported score/prediction/metric/config files.

Run, in order:

```bash
.venv/bin/python -m pytest -q tests/contract/test_full_spec_online_contract.py
.venv/bin/python -m pytest -q tests/unit tests/contract tests/integration
.venv/bin/python scripts/preflight_full_benchmark_matrix.py
.venv/bin/python -m pytest -q
```

The full suite must be interpreted together with the active config inventory;
legacy tests referring to removed `w100` configs must either be migrated to
active `w20` configs or explicitly archived before claiming full acceptance.

## 5. Dependency order

1. Phase 0 must pass before any online behavior changes.
2. Phase 1 must produce entity threshold artifacts before Phase 2 triage.
3. Phase 3 must produce token masks before Phase 4 verification and Phase 5
   masked losses.
4. Phase 4 must define adaptation events before the optimizer factory in Phase 5.
5. Phase 5 must pass isolated loss/optimizer tests before Phase 6 engine wiring.
6. Phase 6 must be stable before demo wiring in Phase 7.
7. Phase 8 is the final acceptance gate.

## 6. Main risks and controls

| Risk | Control |
| --- | --- |
| Threshold leakage across machines | Artifact contains `entity_id`; runtime rejects mismatch. |
| PNN mask accidentally includes anomalies | Discrete radius filter runs before signature recurrence. |
| Buffer adapts overlapping windows | Admission is centralized in `try_admit()`. |
| TTL changes with stream speed | TTL changes only in `finish_verification_cycle()`. |
| Hard-old updates gray-zone windows | Engine dispatches hard-old loss only for hard-old triage. |
| Contrastive loss updates frozen components | Detached keys and projector-only optimizer assertion. |
| Optimizer state carries contamination | Fresh AdamW instance per adaptation event. |
| Demo changes official evaluation | Demo consumes artifacts and never calibrates thresholds. |
| Public contract drift | Existing tests and additive snapshot assertions run at every phase. |

## 7. Preliminary acceptance criteria

The plan is ready to expand with `3_structure_prompt.md` when:

- every user decision is represented by an explicit ownership boundary;
- every research gap maps to a source file and test owner;
- per-entity threshold scope is explicit;
- A1/A2 losses, buffer lifecycle, optimizer reset, logging, and demo queue are
  separated into independently testable phases;
- no implementation phase depends on a later phase's unverified behavior.
