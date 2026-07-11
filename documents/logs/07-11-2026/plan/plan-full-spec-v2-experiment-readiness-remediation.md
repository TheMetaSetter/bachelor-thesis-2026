---
date: 2026-07-11T01:17:09+0700
planner: Codex
git_commit: 99e6e586150cf19363618d98d4462b19e19d93eb
branch: dev
repository: bachelor-thesis-2026
topic: "Programming plan for complete full-spec-v2 experiment readiness"
tags: [plan, full-spec-v2, offline, online-tta, benchmark, demo, reproducibility]
status: accepted_for_structure
source_research: documents/logs/07-10-2026/research/research-full-spec-v2-experiment-readiness.md
last_updated: 2026-07-11
last_updated_by: Codex
design_decisions_confirmed_at: 2026-07-11
design_decisions_confirmed_by: Khôi Nguyễn Anh
---

# Plan: complete `full-spec-v2` experiment readiness

## 0. Planning decision

This draft adopts a **contract-first vertical remediation** approach. The implementation must preserve the current public registry names, configuration hierarchy, checkpoint identity, result keys, and benchmark matrix while replacing every result-changing behavior that differs from `documents/spec/full-spec-v2.md`.

The work is not a new architecture implementation. It is a controlled convergence of the active runtime toward the already locked experiment protocol. A phase is complete only when its executable test proves the relevant mathematical, causal, state, and artifact contract. Structural preflight success alone is not experiment readiness.

The required implementation order is:

```text
freeze contracts
  -> one-window source forward and independent calibration
  -> exact triage and adaptation dispatch
  -> exact A2 contrastive sets
  -> complete-stream execution and resume
  -> demo and reporting parity
  -> readability refactor
  -> CPU preflight and CUDA smoke evidence
  -> complete benchmark launch
```

This order is the smallest safe vertical slice because each later component consumes outputs or state created by the earlier component.

## 1. Sources and authority

The implementation must use the following authority order:

1. `documents/spec/full-spec-v2.md` defines the mathematical experiment protocol and final acceptance criteria.
2. `documents/logs/07-09-2026/detail/detail-thesis-first-full-spec-v2-offline-online-benchmark-demo.md` defines the intended file-level implementation and experiment matrix.
3. `documents/logs/07-10-2026/research/research-full-spec-v2-experiment-readiness.md` defines the current verified gaps.
4. `documents/logs/07-10-2026/detail/detail-full-spec-v2-gap-remediation.md` records completed helper work and remaining acceptance debt.
5. Active source, configuration, and tests define current executable behavior but do not override the specification.

The design notes currently reside in `documents/abstract-design-notes/idea.md` and `documents/abstract-design-notes/design_starter.md`. Their stable contracts remain applicable: batch-first interfaces, `hidden: Tensor[B, L, H]`, composition over inheritance, registry-driven construction, one public model entrypoint, and a minimal number of runtime code paths.

## 2. Current state

### 2.1 Implemented surfaces to preserve

The repository already provides the following usable foundation:

- SMD entities `machine-1-6`, `machine-3-4`, and `machine-3-9`, seeds `6`, `8`, and `36`, and window length `20` are represented in the generated matrix.
- THESIS offline variants O0 and O1 use the locked two-stage budget of Stage A `25` epochs and Stage B `5` epochs.
- The matrix contains 18 THESIS offline main configurations, 54 THESIS online main configurations, 9 RedLamp configurations, 27 traditional offline configurations, and 81 online baseline configurations.
- Clean-validation threshold artifacts, entity-scoped online state, verification-buffer helpers, non-overlap guards, recurrent-signature helpers, queue components, checkpoint state helpers, and focused tests already exist.
- Local smoke artifacts exist for O0-A0, O0-A2, O1-A0, and O1-A2 on `machine-1-6`, seed `6`.
- The active test collection was green in the readiness research, but that fact does not prove runtime equivalence with the specification.

These surfaces must be strengthened in place. New public model names, alternative result schemas, or parallel runners are prohibited unless a later structure review demonstrates that the existing public entrypoint cannot satisfy the locked protocol.

### 2.2 Result-changing gaps

The active runtime differs from the specification in the following decisive ways:

1. `src/core/contracts.py::validate_online_batch()` requires `view_a` and `view_b`, while the locked online forward accepts one window `x`.
2. `src/models/online_adaptation.py::OnlineAdaptationModel.forward()` encodes two views, although the locked computation requires one frozen `Z_source`, `Z_online = Z_source`, and adaptation only through `g_psi`.
3. The same method publishes reconstruction `window_scores` as `aux["latent_window_score"]` instead of the computed prototype/memory distance.
4. `src/engine/online_tta/online_engine.py::_run_online_variant_update()` permits gray-zone updates before verification and omits A2 contrastive regularization for hard-old events.
5. `src/engine/online_tta/online_losses.py::compute_token_multi_positive_info_nce()` is same-position cross-entropy. It does not implement recurrent-signature positives, anomalous-codeword negatives, known-anomaly negatives, or ignored-token semantics.
6. `scripts/generate_online_benchmark_configs.py::_task_overrides()` writes `max_online_steps: 200` for main configurations, so main runs do not process the full test stream.
7. Offline non-overlapping threshold calibration and online stride-1 EWMA/window calibration are not independently owned artifacts.
8. `demo/app.py` and `demo/online_replay.py` do not yet provide the complete live control and plotting surface required by the specification.
9. The source organization violates the hard readability policy. The audited files include lifecycle mixins and multiple files above 500 lines; many callables exceed 50 lines.
10. CUDA smoke execution and resume-safe artifact-path evidence remain unverified.

## 3. Design options and selected approach

### 3.1 Option A: patch only the failing branches

This option would alter a few conditions in `online_engine.py`, replace the latent score assignment, and remove the 200-step limit. It has low initial edit volume but leaves the two-view batch contract, mixed calibration ownership, lifecycle mixins, and oversized event loop intact. It is rejected because later changes would continue to depend on ambiguous state and hidden coupling.

### 3.2 Option B: rewrite the online subsystem

This option would replace the active online model, engine, buffer, and runners with a new implementation. It could produce a clean local design but would introduce large compatibility risk for checkpoints, registries, configurations, result reports, baselines, and already tested helpers. It is rejected because it creates more code paths and weakens reproducibility.

### 3.3 Option C: contract-first vertical remediation

This option freezes public contracts, introduces small pure typed helpers, and rewires the active runtime phase by phase. It uses composition for the source-model adapter, strategy dispatch for A0/A1/A2, and existing registry/factory construction for models and datasets. It is selected because it preserves reproducibility while making each semantic change independently testable and reversible.

## 4. Locked low-level decisions

The following decisions are sufficiently specified and shall be treated as locked:

- Online input is one `batch["x"]` window of shape `[B, 20, D]`.
- `Z_source` is produced once by the frozen source encoder; `Z_online` is the same tensor and `Z_proj = g_psi(Z_source)`.
- A0 bypasses the projector for scoring and performs no optimizer step.
- A1 adapts only on verified PNN tokens using masked reconstruction.
- A2 adapts on accepted hard-old windows or verified PNN tokens, using reconstruction plus exact token contrastive regularization.
- Gray-zone admission never causes an ordinary adaptation step.
- Test labels are metrics-only and must not reach calibration, triage, verification, adaptation, or the demo scoring callback.
- Main online runs process the entire stream. Only smoke configurations may use a finite step cap.
- Stage-B checkpoint identity, entity identity, threshold artifact identity, variant, seed, window length, and runtime-state version must be validated before online execution or resume.

The repository owner confirmed the following two implementation decisions on 2026-07-11. They are part of the locked implementation contract:

1. **Anomalous-codeword metadata source:** compute anomalous codeword membership and radii from training/synthetic-training memory during offline memory initialization, serialize them in the Stage-B checkpoint, and copy their provenance into the threshold/runtime artifact. Do not infer them from test data.
2. **Verification semantics:** perform an independent, label-free frozen-source forward for each buffered entry during a verification cycle. Verification uses codeword radius filtering and recurrent continuous signatures; it never consults ground-truth labels.

These two choices align with sections 6, 12, 13, and 21 of `full-spec-v2`. They must be carried unchanged into the structure and detail artifacts. Any later change requires an explicit protocol decision, followed by revisions to the checkpoint/artifact schema and the tests in Phases 1 and 3 before implementation continues.

## 5. Stable runtime contracts

### 5.1 Batch contract

`src/core/contracts.py` shall enforce the shared batch shape:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

For the THESIS online path, `L == 20`. `view_a` and `view_b` are not required and shall not be synthesized. If older non-`full-spec-v2` experiments require two views, they must use a separately named legacy validation function and may not be selected by the active full-spec configurations.

Tests shall reject rank, window-length, channel, timestamp, and metadata mismatches. Tests shall also prove that online forward succeeds with only the shared batch keys.

### 5.2 Encoder and source-adapter contract

`src/models/online_adaptation.py::ThesisMultitaskEncoderAdapter` remains a composition-based adapter around `ThesisMultitaskModel`. Its public operations shall be:

```python
encode_source(batch) -> Tensor[B, L, H]
score_source(hidden, x) -> ModelOutputs
score_projected(projected_hidden, x) -> ModelOutputs
prototype_verification_metadata() -> PrototypeVerificationMetadata
```

The adapter must detach source hidden states and expose frozen memory access without allowing mutation. It must not own optimizer or stream state.

### 5.3 Model output contract

The model output remains:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H_or_flat]],
    "recon": Tensor[B, L, D],
    "logits": Optional[Tensor[B, C]],
    "point_scores": Tensor[B, L],
    "window_scores": Tensor[B],
    "aux": {
        "source_hidden": Tensor[B, L, H],
        "projected_hidden": Tensor[B, L, H],
        "latent_window_score": Tensor[B],
        "nearest_codeword_ids": Optional[Tensor[B, L]],
        "continuous_signature_ids": Optional[Tensor[B, L, K]],
    },
}
```

`window_scores` means full-window reconstruction MSE. `latent_window_score` means nearest normal continuous-memory distance. These meanings must never be aliased.

### 5.4 Threshold artifact contract

One entity artifact shall contain distinct calibration products:

```text
entity_id
window_size
offline_point_threshold_nonoverlap
online_point_threshold_ewma
B_window
A_low
A_high
calibration_split = validation_clean
offline_stride = 20
online_stride = 1
quantiles
source_checkpoint_sha256
config_sha256
seed
schema_version
```

The offline threshold uses clean validation non-overlapping windows. The online point and window thresholds use clean validation stride-1 simulation. Test labels and test scores are forbidden inputs.

### 5.5 Online event and state contract

Each event record shall contain entity, absolute interval, triage decision, scores, threshold values, admission/verification status, update type, losses, gradient norms, and artifact identity. `OnlineRuntimeState` owns buffer, recurrent signatures, hard-old intervals, EWMA accumulation, finalized point index, and step count. Model objects do not own stream position.

Resume shall restore causal state before reading the next point. It shall rebuild a fresh AdamW optimizer and shall not restore optimizer moments.

## 6. Phase 0 — Baseline freeze and executable contract map

### Goal

Record the current behavior and create failing tests for every known mismatch before modifying result-producing code.

### Files

- Modify `tests/online/test_online_adaptation_step.py`.
- Modify `tests/online/test_online_tta_variants.py`.
- Modify `tests/online/test_online_engine_max_steps.py`.
- Modify `tests/online/test_threshold_artifact.py`.
- Modify `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
- Add `tests/online/test_full_spec_online_contract.py`.
- Add `tests/benchmarks/test_full_spec_runtime_readiness.py`.

### Programming instructions

Add small tests that express one invariant each: one-window batch acceptance, one source encode call, exact latent-score source, no gray-zone update, hard-old contrastive execution, full-stream main config, independent threshold streams, and label isolation. Use deterministic tensors and spy modules; do not use SMD for unit tests.

Add an integration fixture for a length-`T` synthetic stream and assert `processed_windows == T - L + 1` when no smoke cap is configured.

### Gate

All pre-existing tests remain green. Each new mismatch test fails for the expected reason and its failure message names the violated contract.

## 7. Phase 1 — Offline metadata and threshold provenance

### Goal

Make Stage-B checkpoint and clean-validation artifacts sufficient for exact, label-free online execution.

### Files

- Modify `src/models/thesis_multitask.py` and replace lifecycle mixin-owned public checkpoint behavior through explicit composition.
- Modify the focused offline memory component that currently initializes continuous and discrete memories.
- Modify `src/engine/checkpoint.py`.
- Modify `src/engine/online_tta/signature_verification.py`.
- Modify `src/engine/online_tta/threshold_calibration.py`.
- Modify `scripts/train_two_stage.py` or the active two-stage orchestrator discovered during structure tracing.
- Modify `scripts/run_online_adaptation.py`.
- Modify `configs/protocol/benchmark.yaml` and the active THESIS offline/online shared configs.
- Modify `tests/online/test_online_prototype_metadata_contract.py`.
- Modify `tests/online/test_entity_threshold_runtime.py`.
- Add `tests/online/test_independent_threshold_calibration.py`.

### Programming instructions

1. Compute `anomalous_codeword_mask: BoolTensor[K]` and `anomaly_radii: FloatTensor[K]` from training/synthetic-training memory only. Store count, quantile rule, source split, and seed.
2. Export these tensors with the Stage-B checkpoint and validate shape, dtype, finite values, and non-negative radii on load.
3. Fail before the first adaptive event if metadata is absent or inconsistent. Do not use all-false masks or infinite radii.
4. Split threshold calibration into pure functions for non-overlap offline scores and stride-1 online simulation. Do not reuse one collected score array for both.
5. Write one entity-scoped artifact with distinct fields and checksums. Resolve it before test-stream creation.
6. Keep calibration inside `model.eval()` and `torch.no_grad()`.

### Tests

- Checkpoint roundtrip preserves exact metadata values and provenance.
- Missing mask, missing radius, shape mismatch, negative radius, and wrong entity fail closed.
- Offline calibration uses starts `0, 20, 40, ...`; online calibration uses starts `0, 1, 2, ...`.
- Mutating validation labels does not change any threshold.
- Mutating test labels or test values before evaluation cannot affect a previously created threshold artifact.

### Gate

One O0 and one O1 smoke checkpoint load into the online adapter with valid metadata; entity artifacts contain independent offline and online calibration fields.

## 8. Phase 2 — Exact one-window model forward and A0 path

### Goal

Implement the locked computation `x -> Z_source -> g_psi(Z_source)` with exactly one source encoding and exact A0 behavior.

### Files

- Modify `src/core/contracts.py`.
- Modify `src/models/online_adaptation.py`.
- Modify `src/engine/online_tta/threshold_calibration.py`.
- Modify `src/engine/online_tta/online_engine.py` through extracted focused helpers.
- Modify `src/core/console.py` to remove active dependence on view summaries.
- Modify active online model/task YAMLs and generator defaults.
- Modify `tests/online/test_online_adaptation_step.py`.
- Modify `tests/online/test_online_tta_trainable_surface.py`.
- Add `tests/models/test_online_one_window_forward.py`.

### Programming instructions

1. Replace `validate_online_batch()` with validation of the shared batch plus online metadata needed for absolute indices. Do not require views.
2. Call the frozen source encoder once under no-gradient semantics and detach `Z_source`.
3. For A1/A2, compute `Z_proj = online_mlp_projector(Z_source)` and score it through frozen memories and heads.
4. For A0, score `Z_source` directly. Do not pass through a near-identity projector.
5. Publish the actual memory-distance tensor as `aux["latent_window_score"]`.
6. Assert that only parameters under `online_mlp_projector` can require gradients for A1/A2 and that no parameters require gradients for A0.
7. Keep the current model registry name and checkpoint key prefixes. Add compatibility loading only where old key names are already persisted.

### Tests

- A spy encoder records exactly one call per window.
- A0 output equals direct source scoring and records zero updates.
- A1/A2 output shapes satisfy `[B, L, H]`, `[B, L, D]`, `[B, L]`, and `[B]`.
- Changing reconstruction values without changing hidden states changes `window_scores` but not `latent_window_score`.
- A backward pass produces projector gradients only.

### Gate

Threshold calibration, A0 evaluation, and A1/A2 scoring all consume the same one-window model interface without view-specific branches.

## 9. Phase 3 — Exact triage and verification-only gray zone

### Goal

Make event ordering match sections 10, 12, 15, and 16 of the specification.

### Files

- Modify `src/engine/online_tta/triage.py`.
- Modify `src/engine/online_tta/online_engine.py`.
- Modify `src/engine/online_tta/verification_buffer.py`.
- Modify `src/engine/online_tta/verification_cycle.py`.
- Modify `src/engine/online_tta/verification_adapter.py`.
- Modify `src/engine/online_tta/signature_verification.py`.
- Modify `src/engine/online_tta/non_overlap_guard.py` or the active guard module.
- Modify `tests/online/test_online_tta_triage.py`.
- Modify `tests/online/test_online_verification_buffer.py`.
- Modify `tests/online/test_verification_cycle.py`.
- Add `tests/online/test_online_event_order.py`.

### Programming instructions

Create one pure exhaustive triage function with the following disjoint partition:

```text
input <= B                         -> normal
input > B and latent <= A_low      -> hard_old
input > B and A_low < latent <= A_high -> gray_zone
input > B and latent > A_high      -> strong_anomaly
```

Remove `pnn_candidate` as a raw score-based triage class from the THESIS path. PNN is a verification result, not a fifth score region.

The event dispatcher shall use explicit strategy objects or small functions:

```text
A0Strategy: score only
A1Strategy: gray-zone admission; verified-PNN update only
A2Strategy: accepted hard-old update; gray-zone admission; verified-PNN update
```

For gray-zone events, call `try_admit()` and return without optimization. A verification cycle triggers only at capacity eight and after a new admission. It independently forwards all buffer entries, applies anomalous-codeword/radius filtering, computes ordered top-3 continuous signatures, constructs `M_pnn`, applies at most one adaptation decision per entry, marks status, and decrements each remaining TTL exactly once.

The hard-old guard shall reserve an interval only after a successful A2 update. A failed loss/gradient/update must not consume the interval.

### Tests

- Boundary values at `B_window`, `A_low`, and `A_high` match the specification.
- Gray-zone events produce no optimizer call before verification.
- Capacity seven does not verify; capacity eight plus new admission verifies; no new admission does not repeat verification.
- Admission initializes TTL at two; ordinary steps do not decrement TTL.
- Adapted entries leave immediately; unresolved entries remain for exactly the allowed cycles.
- Overlapping gray-zone and hard-old intervals are rejected independently.
- Labels are absent from every verification callback argument.

### Gate

A deterministic event trace matches the locked pseudocode event for event, including update counts, buffer states, TTL values, and guard intervals.

## 10. Phase 4 — Exact A1/A2 losses and optimizer lifecycle

### Goal

Implement the specified reconstruction and token contrastive objectives without hidden negative-set changes.

### Files

- Modify `src/engine/online_tta/online_losses.py`.
- Modify `src/engine/online_tta/online_optimizer.py`.
- Modify `src/engine/online_tta/online_engine.py` or its extracted strategy module.
- Modify `configs/model/online_adaptation.yaml`.
- Modify `configs/task/online_adaptation.yaml`.
- Modify `scripts/generate_online_benchmark_configs.py`.
- Modify `tests/online/test_online_tta_variants.py`.
- Modify `tests/online/test_online_tta_trainable_surface.py`.
- Add `tests/online/test_full_spec_online_losses.py`.

### Programming instructions

Implement three pure functions, each under 50 lines:

```python
compute_hard_old_hinge_loss(recon, x, B_window)
compute_masked_pnn_reconstruction_loss(recon, x, pnn_mask)
compute_token_multi_positive_info_nce(
    anchors,
    source_keys,
    recurrent_signature_ids,
    pnn_mask,
    anomalous_codeword_keys,
    known_anomaly_mask,
    temperature,
)
```

For hard-old A2, every projected token is an anchor; its same-token frozen source latent is positive; every anomalous codeword is negative. For PNN A2, each PNN anchor uses its same-token source key and detached projected keys with the same recurrent signature as positives. Negatives always include anomalous codewords and conditionally include projected/source known-anomaly keys. Non-PNN, non-known-anomaly tokens are ignored.

For A1, use only masked PNN reconstruction. For A2, use reconstruction plus `lambda_contrastive * contrastive`. Read the weight from explicit YAML; do not embed `0.1` in engine code.

Build a fresh AdamW optimizer for each accepted adaptation event with only projector parameters, `learning_rate = 1e-4`, `weight_decay = 1e-4`, and gradient clipping norm `0.5`. Do not use a scheduler. Record gradients for source encoder, memories, reconstruction head, classification head, and projector.

### Tests

- Hand-computed two-dimensional logits match the implementation.
- Removing a recurrent-signature positive increases or preserves the relevant loss.
- Removing anomalous codeword negatives changes the loss and is rejected by contract validation.
- Ignored tokens do not change the loss.
- A PNN token with only its source key still produces a finite loss.
- Empty PNN mask returns no update rather than NaN.
- Hard-old loss is zero at or below `B_window`.
- A1/A2 mutate projector parameters only; A0 mutates nothing.

### Gate

Finite-difference or autograd checks confirm the intended anchors receive gradients and all key copies, source modules, memories, and heads remain frozen.

## 11. Phase 5 — Complete-stream execution, checkpoint, and resume

### Goal

Ensure every main configuration evaluates the complete test stream and can resume without changing the next causal event.

### Files

- Modify `scripts/generate_online_benchmark_configs.py`.
- Modify generated THESIS online `__main.yaml` configurations.
- Modify `src/core/config_model_validation.py`.
- Modify `src/engine/online_tta/online_engine.py`.
- Modify `src/engine/online_tta/runtime_state.py`.
- Modify `scripts/run_online_adaptation.py`.
- Modify `scripts/run_thesis_online_benchmark.py` or the active wrapper.
- Modify `tests/online/test_online_benchmark_config_generation.py`.
- Modify `tests/online/test_online_engine_max_steps.py`.
- Modify `tests/online/test_online_runtime_state.py`.
- Add `tests/online/test_online_resume_runtime.py`.
- Modify `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`.

### Programming instructions

Main configs shall omit `max_online_steps` or set it to `null`; smoke configs shall retain `16`. Configuration validation must distinguish absent/null full-stream semantics from a positive smoke cap. Values `0` and negative values are invalid to avoid ambiguous meaning.

The engine shall count expected windows as `max(0, T - L + 1)` and record processed, skipped, and finalized point counts. A successful main report requires complete coverage, exactly one final decision per test point eligible under the protocol, and no silent early exit.

Checkpoint extra state shall include stream cursor, EWMA accumulators, provisional/finalized score state, buffer, verification trigger state, recurrent signatures, hard-old intervals, counts, variant, entity, source checkpoint identity, threshold artifact identity, and schema version. Resume validates identity before mutation, restores state, reconstructs a fresh optimizer, and consumes the next unseen point exactly once.

### Tests

- Length `T` produces `T-L+1` window forwards.
- Main generated configs have full-stream semantics; smoke configs remain capped at 16.
- Interrupted plus resumed execution produces the same next event and final report as uninterrupted execution.
- Entity, variant, seed, checkpoint, threshold artifact, and state-version mismatches fail before adaptation.
- Optimizer moments are absent from persisted online state.

### Gate

One complete CPU-safe synthetic stream and one SMD smoke stream finish with coverage status `complete`; resume produces byte-equivalent deterministic JSON after canonical key ordering, except permitted timing fields.

## 12. Phase 6 — Reporting, W&B, and artifact integrity

### Goal

Make every experiment auditable and prevent a semantically incomplete run from appearing successful.

### Files

- Modify `src/engine/artifact_sinks.py`.
- Modify `src/engine/logger.py`.
- Modify `scripts/summarize_benchmark_results.py`.
- Modify `scripts/preflight_full_benchmark_matrix.py`.
- Modify benchmark wrappers.
- Modify `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
- Modify `tests/benchmarks/test_summarize_benchmark_results.py`.
- Add `tests/benchmarks/test_full_spec_artifact_integrity.py`.

### Programming instructions

Separate `matrix_status`, `runtime_protocol_status`, `stream_coverage_status`, and `artifact_integrity_status`. The word `ready` may describe matrix structure only when qualified as `matrix_ready`; final experiment readiness requires all four statuses.

Every run shall persist resolved config, git commit, dirty-worktree flag, seed, entity, device, checkpoint hash, threshold artifact hash, processed counts, metric definitions, timing, and result checksum. W&B shall log the same identity fields and upload the resolved config, checkpoint reference, thresholds, metrics, and report as linked artifacts.

Reporting must define pointwise, eventwise, VUS, affiliation, and any adjusted metrics explicitly. Primary tables shall not silently substitute adjusted metrics for raw pointwise metrics. One-class slices shall record metric availability rather than converting warnings into protocol failure.

### Tests

- A truncated run cannot receive final success status.
- A missing checkpoint or mismatched artifact hash fails preflight.
- Two runs with the same identity cannot silently overwrite one another.
- Metric JSON includes definition/version and support counts.
- W&B-off mode writes complete local artifacts without changing computations.

### Gate

The summary script can reject deliberately corrupted, truncated, or identity-mismatched fixtures and can aggregate only specification-valid runs.

## 13. Phase 7 — Complete demo contract

### Goal

Provide the required offline replay and online live replay without creating a second scoring implementation.

### Files

- Modify `demo/app.py`.
- Modify `demo/online_replay.py`.
- Modify `demo/stream_queue.py`.
- Modify `demo/state.py` or the active demo-state module.
- Add a focused UI rendering module if needed to keep every file below 500 lines.
- Modify `tests/demo/test_demo_app.py`.
- Modify `tests/demo/test_live_online_replay.py`.
- Modify `tests/demo/test_demo_stream_queue.py`.
- Add `tests/demo/test_demo_visual_contract.py`.

### Programming instructions

The demo shall inject the same online scoring callback used by internal evaluation. It may control producer delay, play, pause, resume, stop, selected channel, entity, and visible time range. It shall display raw signal, anomaly score, threshold, exact point markers, current-window band, queue size, latest decision, TTA variant, buffer size, and adaptation counts.

The consumer waits until 20 points exist, processes one point at a time, and never passes labels to scoring or adaptation. Ground-truth labels may be an optional post-prediction overlay only. UI state does not own model state or experiment metrics.

### Tests

- Producer ordering and pause/resume/stop are deterministic.
- No forward occurs before 20 points.
- The scoring spy receives no labels.
- Selected-channel changes affect display only.
- Offline replay loads checkpoint and threshold identity before rendering.
- UI callback exceptions stop the stream and show a non-success state.

### Gate

Both demo modes run from persisted artifacts, use the shared scorer, expose all required controls, and pass label-isolation tests.

## 14. Phase 8 — Readability and lifecycle refactor

### Goal

Satisfy `codebase_preferences.md` without changing results, checkpoint keys, or public registries.

### Files

Priority targets include:

- `src/models/thesis_multitask.py` and `src/models/thesis_multitask_*_mixin.py`.
- `src/models/online_adaptation.py`.
- `src/engine/online_tta/online_engine.py`.
- Other files reported by `tests/codebase_compliance.py`.

### Programming instructions

Replace lifecycle mixins with explicit composition owned by the public `ThesisMultitaskModel` entrypoint. Small immutable configuration/value objects and reusable mathematical primitives may live in focused modules. Constructor, public forward/training API, phase routing, and checkpoint contract remain visible from `thesis_multitask.py`.

Split `online_engine.py` by single responsibility: orchestration, event strategy, score aggregation, runtime serialization, and report construction. Keep one obvious call chain and do not create parallel legacy/full-spec engines. Every function and method must be at most 50 lines; every code file must be at most 500 lines.

Before each extraction, add characterization tests for output keys, checkpoint keys, registry names, and deterministic tensor values. Use explicit dependency injection instead of inheritance.

### Tests

- `tests/codebase_compliance.py` reports zero file and callable violations under `src/`.
- Public import paths and registry names remain unchanged.
- Old Stage-B checkpoints load through the documented compatibility boundary.
- Deterministic before/after fixtures produce equal outputs for behavior not intentionally changed in Phases 1–7.

### Gate

Zero source-limit violations, zero lifecycle mixins that distribute public model behavior, green focused suites, and green full active collection.

## 15. Phase 9 — Benchmark preflight and GPU acceptance

### Goal

Prove the complete implementation on the target CUDA server before launching the full matrix.

### Files

- Modify `scripts/preflight_full_benchmark_matrix.py`.
- Modify or add safe smoke launch wrappers under `scripts/`.
- Add `tests/benchmarks/test_full_spec_gpu_preflight.py`.
- Write dated execution evidence under `documents/logs/MM-DD-YYYY/detail/` during implementation.

### Programming instructions

Add preflight checks for CUDA requirement, device name, deterministic seed state, dataset paths, Stage-B checkpoint hashes, entity threshold maps, full-stream main semantics, output writeability, disk-space estimate, resume identity, and collision-safe run directories. Preflight remains no-train.

Execute in this order on the server:

1. CPU/no-train matrix and protocol preflight.
2. One offline O0 smoke and its artifact validation.
3. One offline O1 smoke and its artifact validation.
4. O0-A0, O0-A2, O1-A0, and O1-A2 online smoke runs.
5. Interrupt and resume one A2 smoke run.
6. Validate artifact/checkpoint paths after resume.
7. Only then launch the full shard matrix.

Use `tmux`, one explicit GPU index per process, and `--skip-completed` only after the runner verifies result integrity. Save command, commit, device, environment lock, timestamps, hashes, and final statuses.

### Gate

All four online smoke combinations and the resume test complete on CUDA with correct identity and artifact paths. The active suite is green, preflight is fully valid, and no acceptance item remains unchecked.

## 16. Full benchmark launch plan

The complete launch must preserve the matrix defined by the specification:

- THESIS offline: `O0/O1 x 3 entities x 3 seeds = 18` main runs.
- THESIS online: `O0/O1 x A0/A1/A2 x 3 entities x 3 seeds = 54` main runs.
- RedLamp: 9 runs.
- Traditional offline baselines: 27 runs.
- Online baselines: 81 runs.

The runner shall first resolve dependencies from every online run to its exact Stage-B checkpoint and entity threshold artifact. A shard may start only when all dependencies exist and pass hash validation. Completion means valid artifacts and complete coverage, not process exit code zero.

Failures shall be resumable at run granularity. A failed run writes a non-success manifest with the last safe cursor and error type. The aggregator ignores incomplete runs and reports missing cells explicitly.

## 17. Risk and mitigation

### Protocol drift

Risk: tests or configs may preserve older two-view, `pnn_candidate`, or 100-epoch semantics. Mitigation: add a single full-spec protocol version, fail on incompatible keys, and lock the `25 + 5` budget and one-window contract in preflight tests.

### Calibration leakage or aliasing

Risk: offline and online thresholds may reuse the same stride-1 score collection, or test information may enter calibration. Mitigation: separate pure calibration functions, record split/stride provenance, and add label-mutation tests.

### Fusion or projector collapse

Risk: the projector may drift from the frozen source geometry or adaptation may favor one representation. Mitigation: near-identity initialization, exact contrastive regularization, projector-only gradients, drift logging, and stop-the-run checks for non-finite or excessive drift.

### Adaptation contamination

Risk: anomalous or unverified gray-zone windows may update the projector. Mitigation: fail-closed anomaly metadata, verification-only gray-zone updates, known-anomaly exclusions, hard-old guards, and explicit event-order tests.

### High-variance updates

Risk: one accepted window may cause an unstable step. Mitigation: fresh small AdamW, one step per accepted event, clip norm `0.5`, finite-gradient checks, and no optimizer-moment persistence.

### Resume divergence

Risk: restored execution may repeat or skip a point, lose TTL state, or resolve a different artifact. Mitigation: persist the full causal state and compare uninterrupted versus resumed traces.

### Metric inflation

Risk: adjusted or threshold-tuned metrics may appear as primary results. Mitigation: frozen clean-validation thresholds, explicit metric definitions, support counts, raw metrics as primary outputs, and separate protocol status.

### Readability refactor regression

Risk: splitting oversized files may change registry or checkpoint behavior. Mitigation: refactor only after semantic closure, use characterization tests, preserve public entrypoints, and create rollback boundaries per extraction.

### CUDA/environment mismatch

Risk: CPU success may conceal device, dependency, or artifact issues. Mitigation: CUDA-aware no-train preflight, four required GPU smokes, one resume smoke, and recorded environment/device evidence.

## 18. Validation sequence

Run the following gates after each relevant phase, using `.venv/bin/python`:

```bash
.venv/bin/python -m pytest -q tests/online/test_full_spec_online_contract.py
.venv/bin/python -m pytest -q tests/online
.venv/bin/python -m pytest -q tests/demo
.venv/bin/python -m pytest -q tests/benchmarks
.venv/bin/python tests/codebase_compliance.py
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q
.venv/bin/python scripts/preflight_full_benchmark_matrix.py
```

The structure/detail phase must replace provisional command names with the exact active CLI arguments after tracing each parser. No long benchmark may begin until the no-train preflight, focused tests, full suite, AST scan, and CUDA smoke gates pass.

## 19. Definition of done

The implementation is complete only when all conditions below are true:

1. O0/O1 Stage-A and Stage-B training, checkpointing, and clean-validation artifact export complete for all required entity/seed combinations.
2. A0/A1/A2 execute the exact one-window computation and adaptation rules.
3. Gray-zone windows adapt only after label-free PNN verification.
4. A2 hard-old and PNN losses contain the exact positive, negative, and ignored-token sets.
5. Main online configs process the complete test stream; smoke caps remain smoke-only.
6. Offline and online threshold calibration are independent and provenance-complete.
7. Resume is causally equivalent to uninterrupted execution.
8. Demo modes use the shared scorer, expose the required controls, and isolate labels.
9. Reports distinguish matrix, protocol, coverage, artifact, and metric status.
10. W&B and local artifacts reproduce config, code, data, checkpoint, thresholds, and results.
11. All active tests pass; source files are at most 500 lines; functions and methods are at most 50 lines; lifecycle mixins no longer distribute model behavior.
12. The required CUDA smokes and resume test pass with preserved artifact paths.
13. Every cell of the full matrix either has a specification-valid completed artifact or is explicitly reported missing/failed. No silent omission is allowed.

## 20. Accepted review gate before structure and coding

The repository owner accepted this programming plan's two previously proposed decisions on 2026-07-11:

1. anomalous-codeword mask and radii are generated from training/synthetic-training memory and stored in the Stage-B checkpoint;
2. buffer verification uses an independent label-free frozen-source forward.

The planning gate is therefore closed. The next authorized documentation stage is `prompts/3_structure_prompt.md`, which should assign exact helper boundaries and edit order without reopening these decisions. The later detail artifact should lock function signatures, line-level call flow, test fixtures, CLI invocations, and rollback points. Code changes begin only after the detail artifact is approved and the repository owner explicitly requests implementation.
