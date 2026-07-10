---
date: 2026-07-10T17:59:36+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for full-spec-v2 gap remediation"
tags: [detail, full-spec-v2, online-tta, thresholds, pnn, ttl, hard-old, demo]
status: implementation_in_progress
last_updated: 2026-07-10
last_updated_by: Codex
---

# Detailed programming plan: `full-spec-v2` gap remediation

## 1. Implementation contract

This document is the implementation contract derived from
`structure-full-spec-v2-gap-remediation.md`. It closes the gaps recorded in
`research-full-spec-v2-gap-inventory.md` while preserving the current offline
O0/O1 path, the public online wrapper, checkpoint compatibility, and existing
report keys.

The implementation is ordered as characterization tests, pure calculations,
stateful buffer behavior, strategy dispatch, engine integration, and finally
demo wiring. No later batch may be used to justify an earlier untested change.

### 1.1 Immutable tensor and state contracts

| Object | Required contract |
| --- | --- |
| Window input | `batch["x"]`: `Tensor[B, L, C]`; active benchmark `L=20`, `C=38`. |
| Online metadata | `batch["meta"]`: one record per sample with `entity_id`, `start_index`, `end_index`, and `stream_step`. |
| Source hidden | `Tensor[B, L, H]`; encoder parameters are frozen online. |
| Projected hidden | `Tensor[B, L, H]`; only `online_mlp_projector` can receive gradients. |
| Point scores | `Tensor[B, L]`; online endpoint score uses the latest point of the current window. |
| Window input score | scalar MSE for one complete window, compared with `B_window`. |
| Window latent score | scalar memory/latent distance, compared with `A_low` and `A_high`. |
| PNN mask | `BoolTensor[B, L]`; known-anomaly tokens must be excluded. |
| Threshold artifact | JSON-compatible dictionary containing `entity_id`, window size, values, quantiles, score rules, source split, and provenance. |
| Online record | Existing keys remain; new diagnostics are additive. |

### 1.2 Ownership rules

- Calibration owns no persistent model mutation and runs under `eval()` and
  `torch.no_grad()`.
- Signature helpers receive detached tensors and do not mutate memories.
- `VerificationBuffer` owns admission, entry status, TTL, and verification-cycle
  mutation.
- The optimizer factory owns optimizer construction, not projector weights.
- `online_engine.py` owns event order only.
- `demo/stream_queue.py` owns queue lifecycle; demo code never calibrates
  thresholds or updates official artifacts.

### 1.3 Public compatibility rules

Keep these symbols and surfaces callable:

- `classify_online_window()`;
- `VerificationBuffer.admit()`, `add()`, `clear()`, `items()`, `__len__()`;
- `execute_online_tta_step()`;
- `run_thesis_online_tta_experiment()`;
- `OnlineAdaptationModel` constructor and `online_mlp_projector`;
- registry model names and current report/checkpoint keys.

Legacy threshold key aliases may be read only as fallback. New artifacts and
new generated configs must use the specification field names.

## 2. Batch 0 — Baseline and contract freeze

### 2.1 Batch 0.1: record baseline behavior

Run before edits:

```bash
.venv/bin/python -m pytest -q tests/test_online_tta_variants.py \
  tests/test_online_tta_triage.py \
  tests/test_online_tta_trainable_surface.py \
  tests/test_online_reference_checkpoint.py \
  tests/test_full_benchmark_matrix_preflight.py
```

Record the result in the implementation log. Do not use generated output files
as source-controlled fixtures.

### 2.2 Batch 0.2: add contract tests

Add:

- `tests/contract/test_full_spec_online_contract.py`;
- `tests/contract/test_entity_threshold_artifact.py`;
- `tests/integration/test_online_tta_spec_runtime.py`.

The tests must use a small synthetic model/checkpoint fixture, not the full SMD
sequence. Assert:

1. A0 leaves all projector tensors unchanged and calls no optimizer step.
2. A1 and A2 expose only projector parameters as trainable.
3. Reference encoder, source memory buffers, reconstruction head, and
   classification head checksums are unchanged after one event.
4. Existing report keys remain present.
5. New threshold artifacts require an entity identifier.
6. Test labels are not required by calibration or adaptation helpers.

Acceptance: these tests fail only for the known missing specification behavior,
and existing compatibility tests continue to pass.

Rollback boundary: only the three new test files.

## 3. Phase 1 — Entity-scoped threshold calibration

### 3.1 Batch 1.1: define threshold schema

Modify `src/protocols/threshold_artifact.py`.

Add typed helpers or a frozen dataclass with these logical fields:

```text
entity_id: str
window_size: int
offline_point: {value, quantile, source_split, score_rule}
online_ewma_point: {value, quantile, source_split, score_rule,
                     ewma_current_weight, ewma_previous_weight}
input_window: {value, quantile, source_split, score_rule}
latent_window_low: {value, quantile, source_split, score_rule}
latent_window_high: {value, quantile, source_split, score_rule}
artifact_version: int
provenance: dict[str, Any]
```

Do not remove current `thresholds.offline_point` or
`thresholds.online_ewma_point`. Add `input_window`, `latent_window_low`, and
`latent_window_high` alongside them.

Validation rules:

- `entity_id` is non-empty;
- q99 fields use quantile `0.99` and latent low uses `0.95`;
- `latent_window_low.value <= latent_window_high.value`;
- source split must be `clean_validation`;
- test labels are never accepted as calibration input.

Tests in `test_entity_threshold_artifact.py` cover roundtrip serialization,
missing entity, invalid quantile, invalid ordering, and legacy artifact reading.

### 3.2 Batch 1.2: add pure score helpers

Add `src/engine/online_tta/threshold_calibration.py`.

Functions:

```python
compute_input_window_score(
    reconstruction: Tensor,
    input_window: Tensor,
) -> Tensor

compute_latent_window_score(
    model_outputs: dict[str, Any],
) -> Tensor

aggregate_endpoint_ewma(
    current: float,
    previous: float | None,
    current_weight: float,
    previous_weight: float,
) -> float

quantile_threshold(
    scores: Tensor | Sequence[float],
    quantile: float,
) -> float
```

Shape rules:

- reconstruction and input are `[B, L, C]`;
- input-window score returns `[B]` after mean over `L,C`;
- latent score returns `[B]` or a scalar only after explicit reduction;
- no function reads `point_labels`.

The latent score must use the explicit memory/latent score field selected by the
online model adapter. It must not silently substitute classification loss or
alignment loss when the required field is absent; raise a clear error instead.

Unit tests use deterministic tensors with known means and quantiles.

### 3.3 Batch 1.3: calibrate one entity

Modify `src/engine/online_tta/online_engine.py` and add:

```python
calibrate_entity_thresholds(
    *,
    model: torch.nn.Module,
    clean_validation_sequence: dict[str, Any],
    entity_id: str,
    experiment_config: dict[str, Any],
    protocol_config: dict[str, Any],
    device: str,
) -> dict[str, Any]
```

Execution order:

1. Assert sequence metadata entity matches `entity_id`.
2. Create clean sliding/non-overlapping windows according to each score rule.
3. Run model in `eval()` and `no_grad()`.
4. Collect offline point, online endpoint EWMA, input-window, and latent-window
   scores.
5. Compute q99, q99, q99, q95, q99 respectively.
6. Serialize one entity artifact.

Mutable state ownership: the helper may allocate local score lists but must not
modify model parameters, memory buffers, optimizer state, or data labels.

### 3.4 Batch 1.4: integrate entity selection

The runtime context must store:

```python
threshold_artifacts: dict[str, dict[str, Any]]
```

Before processing a test entity, select exactly its artifact. If the artifact
is absent or has a different `entity_id`, fail before adaptation begins.

Modify threshold writing to keep the current single-entity artifact path for
one-entity runs and use `thresholds/{entity_id}/online_thresholds.json` for
multi-entity runs.

Focused command:

```bash
.venv/bin/python -m pytest -q tests/contract/test_entity_threshold_artifact.py
```

Acceptance: two synthetic entities receive different artifacts, and O0-A0
smoke still writes the existing point threshold keys plus new window fields.

## 4. Phase 2 — Exact triage partition

### 4.1 Batch 2.1: implement truth-table classifier

Modify `src/engine/online_tta/triage.py`.

Keep the public signature but prefer these fields:

```python
thresholds = {
    "input_window_threshold": B_window,
    "latent_window_low_threshold": A_low,
    "latent_window_high_threshold": A_high,
}
```

Use the exact ordered predicates:

```text
if input <= B_window: normal
elif latent <= A_low: hard_old_normality
elif latent <= A_high: gray_zone
else: strong_anomaly
```

The first comparison is input score, so a low-latent/high-input window cannot
be classified as normal. Keep legacy aliases only if all three new fields are
missing; mixed legacy/new threshold dictionaries should prefer new fields.

Tests:

- input exactly equal to `B_window` is normal;
- input just above `B_window` and latent equal `A_low` is hard-old;
- latent equal `A_high` is gray-zone;
- latent just above `A_high` is strong anomaly;
- malformed thresholds raise `ValueError`.

### 4.2 Batch 2.2: calculate real online window scores

Modify `_score_online_window()` in `online_engine.py`.

Return a named result, preferably a small immutable dataclass:

```text
batch_on_device
raw_endpoint_point_score
input_window_score
latent_window_score
ewma_point_score
```

Do not use alignment loss as a substitute for latent-window memory score. The
model adapter must expose the latent score used by prototype/codebook semantics.
The endpoint point score remains the value used for causal EWMA prediction.

Tests assert that changing the input reconstruction changes `input_window_score`
without changing the threshold artifact and that no future batch is inspected.

## 5. Phase 3 — Prototype-aware signature verification

### 5.1 Batch 3.1: nearest codeword and anomaly radius

Add `src/engine/online_tta/signature_verification.py`.

Functions and shapes:

```python
nearest_discrete_codeword(
    hidden: Tensor[B, L, H],
    codebook: Tensor[K, H],
) -> tuple[Tensor[B, L], Tensor[B, L]]

filter_known_anomaly_tokens(
    hidden: Tensor[B, L, H],
    codebook: Tensor[K, H],
    anomalous_codeword_mask: BoolTensor[K],
    anomaly_radii: Tensor[K],
) -> BoolTensor[B, L]
```

Normalize hidden/codebook using the same epsilon and cosine convention as the
offline model. `filter_known_anomaly_tokens` returns true only when the nearest
codeword is anomalous and the distance is within that codeword's radius.

No helper may update codebook values or anomaly radii.

### 5.2 Batch 3.2: continuous ordered signatures

Add:

```python
ordered_continuous_signature(
    hidden: Tensor[B, L, H],
    continuous_prototypes: Tensor[P, H],
    topk: int = 3,
) -> list[list[tuple[int, ...]]]
```

For every token, return prototype IDs sorted by distance, not by arbitrary
memory order. Validate `1 <= topk <= P`.

Add:

```python
find_recurrent_signatures(
    window_signatures: Sequence[SignatureWindow],
) -> set[tuple[int, ...]]
```

`SignatureWindow` carries entity ID, start, end, and token signatures. Only
non-overlapping windows count as separate recurrence evidence.

### 5.3 Batch 3.3: construct token PNN mask

Add:

```python
build_pnn_token_mask(
    signatures: list[list[tuple[int, ...]]],
    recurrent_signatures: set[tuple[int, ...]],
    known_anomaly_mask: BoolTensor[B, L],
) -> BoolTensor[B, L]
```

The result is recurrent-signature membership AND NOT known anomaly. Preserve
batch/window order. Empty recurrence produces an all-false mask.

Tests:

- deterministic top-3 ordering;
- anomalous radius exclusion;
- recurrence requires two non-overlapping windows;
- known anomaly always overrides PNN membership;
- no mutation of prototype/codebook tensors.

## 6. Phase 4 — VerificationBuffer and cycle TTL

### 6.1 Batch 4.1: extend entry schema

Modify `verification_buffer.py` while retaining existing public methods.

Canonical entry:

```python
{
    "entry_id": str,
    "window": dict[str, Any],
    "window_start": int,
    "window_end": int,
    "ttl_remaining": 2,
    "status": "unresolved",
    "was_adapted": False,
}
```

Use a private list as the single mutable owner. `items()` returns defensive
copies; callers cannot mutate internal state accidentally.

### 6.2 Batch 4.2: admission and verification methods

Add:

```python
try_admit(entry: dict[str, Any]) -> bool
should_verify() -> bool
mark_verification_result(entry_id: str, adapted: bool) -> None
finish_verification_cycle() -> dict[str, int]
```

Rules:

- admission rejects overlap with any unresolved/current entry;
- admission sets `ttl_remaining=2` and marks `new_since_last_cycle=True`;
- `should_verify()` is true only at size `>= 8` and new admission;
- adapted entries are removed during cycle finalization;
- unresolved entries decrement once per cycle;
- unresolved entries with zero TTL are removed;
- finalization resets the new-admission flag.

Keep `admit()` as a compatibility wrapper that delegates to the new overlap
logic where possible.

### 6.3 Batch 4.3: connect engine without per-step TTL

Modify `_update_online_window_buffers()` and `_run_online_sequence()`.

Only gray-zone windows may enter the verification buffer. The engine must call
`try_admit()`, not direct `add()`. It must not decrement TTL on each stream
window. At a verification cycle:

1. collect entries;
2. run signature filtering and PNN mask construction;
3. adapt if the selected variant permits it;
4. mark entries adapted/unresolved;
5. call `finish_verification_cycle()`;
6. log kept/removed/adapted counts.

The existing `TTLBuffer` remains available to legacy callers but is not the
owner of specification PNN TTL state.

Tests run with a synthetic stream of 10 windows and assert exact cycle counts.

## 7. Phase 5 — A1/A2 losses and optimizer factory

### 7.1 Batch 5.1: masked PNN reconstruction

Modify `online_losses.py` and add:

```python
compute_pnn_reconstruction_loss(
    reconstruction: Tensor[B, L, C],
    target: Tensor[B, L, C],
    pnn_mask: BoolTensor[B, L],
    eps: float = 1e-8,
) -> Tensor[()]
```

Expand mask to `[B,L,C]`. Compute only selected tokens. If no PNN token exists,
return a differentiable zero tensor on the reconstruction device and log a
skip event at the engine layer.

### 7.2 Batch 5.2: hard-old hinge loss

Add:

```python
compute_hard_old_hinge_loss(
    reconstruction: Tensor[B,L,C],
    input_window: Tensor[B,L,C],
    b_window: float,
) -> Tensor[()]
```

Compute window MSE, then `relu(mse - b_window).pow(2)`. The returned value must
be exactly zero when MSE is at or below the threshold. This function must not
use the PNN mask or labels.

### 7.3 Batch 5.3: token multi-positive InfoNCE

Add:

```python
compute_token_multi_positive_infonce(
    anchors: Tensor[N,H],
    positives: Sequence[Tensor[M_i,H]],
    negatives: Sequence[Tensor[K_i,H]],
    temperature: float,
) -> Tensor[()]
```

Normalize anchors and keys. Detach positive/negative key copies except when a
token is used as its own anchor. Require anomalous codeword negatives for A2.
If no recurrent positive exists, use same-token frozen source latent as the
single valid positive.

### 7.4 Batch 5.4: optimizer factory

Modify `online_optimizer.py`:

```python
build_online_optimizer(
    model: torch.nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer
```

The factory:

1. calls `collect_projector_parameters()`;
2. creates fresh AdamW state;
3. passes only projector parameters;
4. does not modify projector tensors;
5. returns no scheduler.

The engine creates this optimizer immediately before an adaptation event and
discards it afterward. Weight changes persist in the projector; optimizer
moments do not persist across events.

Apply `clip_grad_norm_(projector.parameters(), 0.5)` before `step()`.

### 7.5 Batch 5.5: strategy dispatch

Refactor `_run_online_variant_update()` into explicit small strategy helpers:

- `_run_a0_update()` returns no update;
- `_run_a1_pnn_update()` requires a verified PNN mask;
- `_run_a2_hard_old_update()` requires hard-old triage and guard acceptance;
- `_run_a2_pnn_update()` requires PNN verification;
- `_run_a2_contrastive_regularizer()` composes with either A2 branch.

Do not create subclasses for A0/A1/A2. The strategy is a mapping or explicit
dispatch table consumed by the engine.

## 8. Phase 6 — Engine integration and logging

### 8.1 Batch 6.1: explicit event pipeline

Refactor `_process_online_window()` into named steps without changing its
public caller:

```text
score_window()
update_ewma()
classify_online_window()
apply_hard_old_guard_or_try_admit()
maybe_run_verification_cycle()
maybe_adapt_projector()
build_future_only_record()
```

Each step receives explicit inputs and returns explicit outputs. Avoid shared
module-level mutable state.

### 8.2 Batch 6.2: hard-old non-overlap guard

Add a small guard type in `online_engine.py` or
`src/engine/online_tta/non_overlap_guard.py`:

```python
accept(interval: tuple[int, int]) -> bool
add(interval: tuple[int, int]) -> None
```

Use a deque with configured maximum size, default `1`. Add the interval only
after a successful A2 hard-old update. Reject overlapping intervals.

### 8.3 Batch 6.3: additive diagnostics

Extend per-step metric records with:

```text
online/input_window_score
online/latent_window_score
online/triage_counts
online/num_buffer_admitted_windows
online/num_buffer_rejected_overlap_windows
online/num_points_removed_by_discrete_anom_filter
online/num_points_remaining_for_signature
online/num_recurrent_signatures
online/num_pseudo_new_normality_points
online/loss_hard_recon
online/loss_pnn_recon
online/loss_contrastive
online/projector_grad_norm
online/source_encoder_grad_norm
online/source_memory_grad_norm
online/recon_head_grad_norm
online/classification_head_grad_norm
```

Use zero or `None` consistently when a branch is not active. Do not overwrite
existing keys or introduce label-derived fields into adaptation decisions.

### 8.4 Batch 6.4: serialization and resume

Include threshold artifact identity, buffer entries, hard-old guard intervals,
and online variant in checkpoint extra state if resume is supported by the
current wrapper. On load, validate entity ID and variant before restoring
state. Never restore optimizer moments because the policy is reset-per-event.

Integration tests:

- causal synthetic stream;
- no future access;
- A0/A1/A2 event decisions;
- checkpoint state roundtrip;
- additive report schema.

## 9. Phase 7 — Queue-based demo

### 9.1 Batch 7.1: add queue module

Add `demo/stream_queue.py` with small functions/classes:

```python
class StreamQueueController:
    def __init__(self, maxsize: int = 128, delay_seconds: float = 0.05): ...
    def start(self, sequence: Iterable[Any]) -> None: ...
    def get(self, timeout: float = 0.1) -> dict[str, Any] | None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
```

The queue controller owns only queue/thread/event state. It does not import the
online model or threshold calibrator. Producer entries contain `t` and `x`;
consumer reads one point at a time.

Use `queue.Queue(maxsize=...)`, `threading.Event`, and timeout-based `get()`.
Handle queue full through blocking `put()` with stop-event checks.

### 9.2 Batch 7.2: connect online replay

Modify `demo/online_replay.py` so it accepts an injected queue controller,
model callback, and state object. It must:

1. append incoming points;
2. wait until `L` points exist;
3. form one latest window;
4. call the online scoring callback;
5. update demo scores and provisional predictions;
6. mark finalized points without retroactive official metric changes.

Modify `demo/app.py` only to wire UI buttons and speed controls to the queue
controller. UI code must not own queue internals.

### 9.3 Batch 7.3: demo tests

Add `tests/unit/test_demo_stream_queue.py`:

- empty queue returns safely after timeout;
- producer emits one point per item;
- full queue applies backpressure;
- pause stops production and resume continues;
- stop terminates producer/consumer cleanly;
- consumer waits below `L` and processes at `L`;
- demo runs without labels;
- label overlay is not passed to model/adaptation.

## 10. Phase 8 — Matrix and final acceptance

### 10.1 Contract/config checks

Add or modify:

- `tests/contract/test_full_spec_config_contract.py`;
- `tests/integration/test_full_spec_matrix_smoke.py`;
- `scripts/preflight_full_benchmark_matrix.py`.

Assert every generated online config includes explicit variant settings,
entity-compatible Stage-B reference path, projector target, and optimizer
defaults.

### 10.2 Required command sequence

Before full suite:

```bash
.venv/bin/python -m pytest -q tests/contract/test_full_spec_online_contract.py
.venv/bin/python -m pytest -q tests/contract/test_entity_threshold_artifact.py
.venv/bin/python -m pytest -q tests/unit/test_demo_stream_queue.py
.venv/bin/python -m pytest -q tests/integration/test_online_tta_spec_runtime.py
```

Then:

```bash
.venv/bin/python scripts/preflight_full_benchmark_matrix.py
.venv/bin/python -m pytest -q tests/unit tests/contract tests/integration
.venv/bin/python -m pytest -q
```

Required smoke wrappers:

```bash
.venv/bin/python scripts/run_thesis_offline_benchmark.py \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__smoke.yaml \
  --skip-completed

.venv/bin/python scripts/run_thesis_offline_benchmark.py \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml \
  --skip-completed

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml \
  --online-variant A0

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A2__machine_1_6__w20__seed6__smoke.yaml \
  --online-variant A2

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O1_A0__machine_1_6__w20__seed6__smoke.yaml \
  --online-variant A0

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__smoke.yaml \
  --online-variant A2
```

### 10.3 Final measurable acceptance

The implementation is complete only when:

1. Every required entity has a separate threshold artifact.
2. Triage boundary tests pass for all four regions.
3. PNN masks exclude known anomalies and require recurrent signatures.
4. Buffer TTL starts at 2 and decrements only during verification cycles.
5. Hard-old hinge is zero below `B_window` and is used only for accepted A2
   hard-old events.
6. A1/A2 optimizer is fresh AdamW with `lr=1e-4`, `weight_decay=1e-4`, one
   step, no scheduler, and clipping norm `0.5`.
7. Only `online_mlp_projector` changes during online updates.
8. Required logs and report fields are present without label leakage.
9. Queue demo tests pass and demo does not write official evaluation artifacts.
10. O0-A0, O0-A2, O1-A0, and O1-A2 smoke runs complete.
11. Active contract/unit/integration tests pass; legacy `w100` tests are either
    migrated or explicitly separated before claiming repository-wide green.
12. `git diff --check` passes and all newly added source functions remain below
    the repository's hard readability limits.

## 11. Rollback boundaries

- Batch 1 rollback: threshold schema/calibration files only.
- Batch 2 rollback: triage/scoring files only.
- Batch 3 rollback: signature helper only.
- Batch 4 rollback: buffer/TTL/engine lifecycle changes only.
- Batch 5 rollback: losses/optimizer/config changes only.
- Batch 6 rollback: engine/report integration only.
- Batch 7 rollback: demo files/tests only.
- Batch 8 rollback: acceptance tests/preflight/documentation only.

If a compatibility test fails after a batch, stop at that boundary and record
the exact mismatch before continuing. Do not weaken tests to make the phase
green.

## 12. Detail-phase readiness

This document is ready for implementation after review. Any change to public
threshold keys, model output fields, buffer method semantics, or demo ownership
must be recorded as a new decision before coding.

## 13. Blocker-remediation supplement

The original batches above are retained as the historical implementation
contract. The current repository audit found that helper-level implementation
was ahead of runtime integration. The executable continuation is now defined in
`documents/logs/07-10-2026/detail/detail-full-spec-v2-blocker-remediation.md`.

The continuation must be executed in this order:

1. freeze one `OnlineRuntimeState` and current semantic test folders;
2. route runtime calibration through an entity-keyed artifact map;
3. connect read-only prototype access and bounded recurrent signature history to
   each causal event;
4. trigger verification cycles through `try_admit()` and tick TTL only after a
   completed cycle;
5. enforce projector-only updates, hard-old guard acceptance, fresh AdamW and
   diagnostic gradient checks;
6. restore entity, variant, artifact, buffer, signature and guard state without
   optimizer moments;
7. wire the live demo consumer and validate the four GPU smoke wrappers;
8. refresh the source-audit detail using final AST/test evidence.

The authoritative blocker evidence is in
`documents/logs/07-10-2026/research/research-full-spec-v2-blocker-audit.md`.
The previous test paths `tests/unit`, `tests/contract`, and
`tests/integration` are obsolete; acceptance commands must use the current
semantic folders under `tests/`.

## 14. Execution checklist

### A. Contract and state

- [x] Public model registry names remain unchanged.
- [x] Existing online report keys remain present.
- [x] `OnlineRuntimeState` validates non-empty entity identity.
- [x] `OnlineRuntimeState` validates `A0`, `A1`, and `A2` variants.
- [x] Runtime state has JSON-safe serialization.
- [x] Runtime state rejects mismatched threshold artifact entity.
- [x] Runtime state roundtrip test passes.

### B. Entity-specific calibration

- [x] A pure per-entity calibration entrypoint exists.
- [x] Runtime builds an artifact map keyed by entity.
- [x] Runtime writes `thresholds/{entity_id}/online_thresholds.json`.
- [x] Test sequence selects its own entity artifact before scoring.
- [ ] Multi-entity integration test proves two different threshold values are
  used in one execution context.
- [ ] Artifact schema contains `B_window`, `A_low`, and `A_high` values from
  their own score distributions rather than aliases of point thresholds.

### C. Exact triage and causal scoring

- [x] New triage keys are preferred over legacy aliases.
- [x] Boundary tests cover normal, hard-old, gray-zone, and strong anomaly.
- [x] Latent score no longer silently uses alignment loss.
- [x] Online endpoint score is causal.
- [x] Input-window score is computed from full-window reconstruction MSE in the
  engine instead of endpoint point score.
- [ ] Latent score is explicitly sourced from prototype/memory distance.

### D. Prototype-aware PNN

- [x] Nearest discrete codeword helper exists.
- [x] Anomaly-radius filtering helper exists.
- [x] Ordered top-k continuous signature helper exists.
- [x] Non-overlapping recurrence helper exists.
- [x] Known-anomaly exclusion helper exists.
- [ ] Engine obtains codebook, anomaly mask, radii, and continuous prototypes
  through a read-only adapter.
- [x] Engine stores recurrent signatures per entity stream.
- [x] Engine creates `batch["pnn_mask"]` before A1/A2 dispatch.
- [x] A1 refuses adaptation when no verified PNN mask exists.
- [x] PNN diagnostics report real counts for codebook/prototype filtering.

### E. Verification buffer and hard-old guard

- [x] Buffer entry has status, TTL, and adaptation fields.
- [x] `try_admit()` rejects overlaps.
- [x] Verification capacity is explicit.
- [x] TTL decrement is implemented in cycle finalization.
- [x] Hard-old guard primitive exists.
- [x] Engine uses `try_admit()` for PNN candidates.
- [ ] Engine runs a real verification callback at capacity eight.
- [ ] Engine marks entries as adapted or unresolved from verification results.
- [ ] Hard-old guard is consulted before every A2 hard-old update.
- [ ] Rejected-overlap diagnostics count actual rejections.

### F. Loss and optimizer semantics

- [x] Hard-old hinge helper exists.
- [x] Masked PNN reconstruction helper exists.
- [x] Fresh AdamW factory exists.
- [x] Projector gradient clipping uses norm `0.5`.
- [x] A2 runtime uses hard-old hinge score.
- [x] A2 runtime uses token multi-positive InfoNCE in the active branch.
- [ ] Optimizer moments are proven absent from every event checkpoint.
- [ ] Projector-only mutation is asserted by integration test for A0/A1/A2.

### G. Checkpoint and resume

- [x] Final checkpoint writes threshold artifact identity.
- [x] Final checkpoint writes buffer entries.
- [x] Final checkpoint writes hard-old guard intervals.
- [x] Resume identity validator exists.
- [ ] Runtime loader restores buffer entries into a live `VerificationBuffer`.
- [ ] Runtime loader restores signature history.
- [ ] Runtime loader restores hard-old guard intervals.
- [ ] Resume test proves next event equals uninterrupted execution.

### H. Demo and reporting

- [x] Queue producer/consumer controller exists.
- [x] Queue pause/resume/stop tests pass.
- [x] Live consumer waits until `L` points exist.
- [ ] `demo/app.py` wires the live consumer rather than only replaying reports.
- [ ] Demo callback receives no labels.
- [ ] Online diagnostics contain real PNN, buffer, loss, and gradient values.

### I. GPU/server acceptance

- [x] Matrix preflight reports ready on the current machine.
- [x] Offline O0 smoke completes locally.
- [x] Offline O1 smoke completes locally.
- [x] Online O0-A0 smoke completes locally.
- [x] Online O0-A2 smoke completes locally.
- [x] Online O1-A0 smoke completes locally.
- [x] Online O1-A2 smoke completes locally.
- [ ] The four online smoke commands complete on the target CUDA GPU server.
- [ ] Artifact paths and checkpoint paths are preserved after server resume.

### J. Readability and test collection

- [x] Tests are grouped into semantic directories under `tests/`.
- [x] Legacy tests are archived outside pytest collection.
- [x] Focused online/demo/compliance tests pass.
- [ ] AST scan reports zero `src/` files over 500 lines.
- [ ] AST scan reports zero `src/` callables over 50 lines.
- [ ] Full active `pytest -q` passes without archived legacy tests.

## 15. Unfinished-item execution plans

The following plans close every remaining unchecked item above. An item may be
marked `[x]` only after its implementation and focused test are both present.

### 15.1 Real anomaly mask and radius contract

- Add `PrototypeVerificationMetadata` to
  `src/engine/online_tta/signature_verification.py` with detached tensors for
  `codebook`, `anomalous_codeword_mask`, and `anomaly_radii`.
- Require the online model adapter to expose this metadata explicitly. If the
  checkpoint does not contain it, fail before the first adaptive event instead
  of silently using an all-false mask or infinite radii.
- Add `tests/online/test_online_prototype_metadata_contract.py` covering shape,
  dtype, missing metadata, and radius filtering.

### 15.2 Verification callback and cycle controller

- Add `VerificationCycleController` in
  `src/engine/online_tta/verification_cycle.py`.
- The controller owns the eight-entry trigger, calls a pure verification
  callback, marks each entry, and finalizes TTL exactly once.
- Inject the controller into `_process_online_window()`; remove direct cycle
  logic from the engine body.
- Add tests for capacity 7/8, adapted versus unresolved entries, and TTL
  decrement count.

### 15.3 Hard-old guard wiring

- Pass `NonOverlapGuard` through `_run_online_sequence()` and
  `_process_online_window()`.
- Before A2 hard-old dispatch, call `accept((window_start, window_end))`.
- Add the interval only after `did_update=True`; record rejection count in the
  event metrics.
- Add a test proving a failed update does not reserve an interval.

### 15.4 Complete resume entrypoint

- Add `resume_online_runtime()` under
  `src/engine/online_tta/runtime_state.py` or a focused companion module.
- Load checkpoint extra state, validate entity, variant and artifact identity,
  restore buffer/history/guard, and create a fresh optimizer.
- Add an uninterrupted-versus-resumed next-event integration test.

### 15.5 Live demo ownership

- Add `run_live_online_replay()` in `demo/online_replay.py` accepting an
  injected `StreamQueueController`, window size, and score callback.
- Modify `demo/app.py` only to construct the controller and callback; labels may
  be used by plotting but never passed to scoring/adaptation.
- Add a test that spies on the callback arguments and proves labels are absent.

### 15.6 GPU evidence and artifact safety

- Extend `scripts/preflight_full_benchmark_matrix.py` with a CUDA requirement
  switch, checkpoint identity check, artifact-map check, and deterministic seed
  check.
- Run the four online smoke wrappers on the target GPU server and save command
  output, CUDA device name, artifact paths, and checkpoint paths in a dated
  implementation log.
- Do not mark GPU checklist items from CPU-only local runs.

### 15.7 AST and full-suite closure

- Run the AST scanner after every source split; split every remaining `src/`
  file over 500 lines and every callable over 50 lines.
- Update imports only through public entrypoints and retain checkpoint keys.
- Run `pytest --collect-only -q`, focused semantic folders, then full
  `pytest -q`; archive only tests proven to target removed legacy flows.
- Mark the final checklist items only after zero AST violations and a green
  active suite are recorded.
