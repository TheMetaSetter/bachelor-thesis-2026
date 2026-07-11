---
date: 2026-07-11T01:30:18+0700
planner: Codex
git_commit: 99e6e586150cf19363618d98d4462b19e19d93eb
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for complete full-spec-v2 experiment readiness"
tags: [detail, full-spec-v2, offline, online-tta, benchmark, demo, reproducibility]
status: implementation_in_progress
source_plan: documents/logs/07-11-2026/plan/plan-full-spec-v2-experiment-readiness-remediation.md
source_structure: documents/logs/07-11-2026/structure/structure-full-spec-v2-experiment-readiness-remediation.md
last_updated: 2026-07-11
last_updated_by: Codex
---

# Detailed programming plan: complete `full-spec-v2` readiness

## 0. Implementation contract

This document is the implementation contract for making the active offline, online, reporting, and demo paths conform to `documents/spec/full-spec-v2.md`. It does not authorize coding by itself. Implementation begins only after the repository owner explicitly requests execution.

The work must preserve current public registries, model names, CLI entrypoints, O0/O1 Stage-B checkpoint compatibility, A0/A1/A2 config identity, existing report keys, and baseline behavior outside the active THESIS full-spec path. Every intentional result change must be locked by a failing test before source modification.

The accepted execution order contains eleven batches:

```text
0 contracts
  -> 1 checkpoint and calibration
  -> 2 one-window vertical slice
  -> 3 triage and strategies
  -> 4 verification and TTL
  -> 5 losses and update transaction
  -> 6 full stream and resume
  -> 7 artifacts, reporting, and W&B
  -> 8 demo
  -> 9 readability refactor
  -> 10 preflight, CUDA smoke, and full matrix
```

A later batch must not begin until the previous batch gate passes.

## 1. Locked scientific and runtime semantics

The following decisions are final for this implementation:

1. Online input is one window `x: FloatTensor[B, 20, D]`.
2. The frozen source encoder runs exactly once per window and produces `Z_source: FloatTensor[B, 20, H]`.
3. A0 scores `Z_source` directly. A1/A2 compute `Z_proj = g_psi(Z_source)` and score `Z_proj` through frozen prototype memories and heads.
4. `window_scores: FloatTensor[B]` means full-window reconstruction MSE. `latent_window_score: FloatTensor[B]` means nearest normal continuous-memory distance. They are never aliases.
5. Anomalous codeword membership and radii are computed from training/synthetic-training memory, serialized in the Stage-B checkpoint, and never derived from validation or test labels.
6. Gray-zone windows enter one verification buffer and cannot adapt before independent label-free verification.
7. A1 adapts verified PNN tokens with masked reconstruction only.
8. A2 adapts accepted hard-old windows or verified PNN tokens with the exact reconstruction and contrastive objectives.
9. Only `online_mlp_projector` parameters are trainable online.
10. Main runs process the complete stream. Finite `max_online_steps` is smoke-only.
11. Test labels are used only after predictions for metric computation or optional post-prediction demo overlay.

## 2. Stable cross-layer contracts

### 2.1 Dataset and batch contract

The existing registry in `src/core/registry.py` remains the only dataset/model factory. `register_dataset()`, `build_dataset()`, `register_model()`, and `build_model()` keep their public names.

The dataset path remains:

```text
raw entity sequence [T, D]
  -> train-only fitted scaler
  -> scaled sequence
  -> window construction
  -> batch [B, L, D]
```

The common batch remains:

```python
{
    "x": FloatTensor[B, L, D],
    "point_labels": BoolOrLongTensor[B, L] | None,
    "mask": FloatOrBoolTensor[B, L, D] | None,
    "timestamps": Tensor[B, L] | None,
    "meta": list[dict],
}
```

For active THESIS offline/online configs, `L == 20`. Online code may add `pnn_mask`, recurrent-signature metadata, or absolute indices after base validation, but it must not require `view_a` or `view_b`. Synthetic classification fields remain additive and may not replace the base keys.

### 2.2 Encoder adapter contract

`ThesisMultitaskEncoderAdapter` composes a frozen `ThesisMultitaskModel`. The final public adapter surface should be:

```python
encode_source(batch) -> FloatTensor[B, L, H]
score_source(source_hidden, x) -> ModelOutputs
score_projected(projected_hidden, x) -> ModelOutputs
get_verification_metadata() -> PrototypeVerificationMetadata
```

The adapter owns no optimizer, buffer, stream cursor, or report state. Returned source hidden states and verification tensors are detached. The source model remains a registry-built model rather than being constructed through a new subclass hierarchy.

### 2.3 Model output contract

Both scoring routes return:

```python
{
    "hidden": FloatTensor[B, L, H],
    "pooled": Tensor | None,
    "recon": FloatTensor[B, L, D],
    "logits": FloatTensor[B, C] | None,
    "point_scores": FloatTensor[B, L],
    "window_scores": FloatTensor[B],
    "aux": {
        "source_hidden": FloatTensor[B, L, H],
        "projected_hidden": FloatTensor[B, L, H] | None,
        "latent_window_score": FloatTensor[B],
        "nearest_codeword_ids": LongTensor[B, L] | None,
        "continuous_signature_ids": LongTensor[B, L, K] | None,
    },
}
```

`validate_model_outputs()` continues to enforce the stable top-level vocabulary. New checks should validate semantic tensor shapes without adding model-specific branches to evaluators.

### 2.4 Offline training and checkpoint contract

O0/O1 retain Stage A `25` epochs and Stage B `5` epochs for main runs. Stage B freezes encoder and prototype memories and fine-tunes only the accepted fusion/head surface. Smoke configs may retain `1 + 1` epochs.

Checkpoint transport remains `CheckpointManager`. `ThesisMultitaskModel` owns the extra state containing memory initialization identity, anomalous codeword mask, radii, and provenance. The manager must not calculate model metadata.

### 2.5 Threshold artifact contract

Each entity artifact must contain:

```text
schema_version
entity_id
window_size = 20
offline_point_threshold_nonoverlap
online_point_threshold_ewma
B_window
A_low
A_high
offline_stride = 20
online_stride = 1
calibration_split = clean_validation
quantile definitions
EWMA weights = 0.9 current, 0.1 previous
source checkpoint path and sha256
resolved config sha256
seed
created_at
```

### 2.6 Online runtime contract

`OnlineRuntimeState` owns entity, variant, stream cursor, EWMA state, provisional/finalized point state, verification entries, new-admission flag, recurrent signature history, hard-old intervals, update counts, threshold identity, and schema version. Optimizer moments are excluded.

### 2.7 Task strategy contract

The Strategy pattern is represented by explicit functions or immutable callable objects, not subclasses:

```text
A0: score only
A1: gray-zone admission; verified-PNN update
A2: accepted hard-old update; gray-zone admission; verified-PNN update
```

Baseline policies remain under `src/baselines/online/` and must not affect THESIS triage semantics.

## 3. Batch 0 — Baseline inventory and contract freeze

### 3.1 Summary

This batch protects reproducible offline/online behavior before semantic changes and creates executable evidence for every readiness gap.

### 3.2 Edit order

1. Add `tests/online/test_full_spec_online_contract.py`.
2. Extend `tests/compliance/fixtures/src_refactor_contracts.json` with public model, registry, checkpoint, and report symbols.
3. Add deterministic fixtures under the existing semantic test folders; do not create a new test hierarchy.
4. Record the baseline test collection and AST report in the implementation log created when coding begins.

### 3.3 Required test content

Add one test per invariant:

- a base online batch without views validates;
- a spy source encoder is called once;
- A0 never calls the projector;
- reconstruction-score changes do not alter latent score;
- a gray-zone event never constructs an optimizer;
- A2 hard-old includes contrastive loss;
- main configs have full-stream semantics;
- offline and online calibration consume different stride sequences;
- verification and scoring callbacks receive no labels;
- public registry names and checkpoint keys remain stable.

Golden fixtures must be small: `B=2`, `L=4` for pure unit tests, and a synthetic stream `T=8`, `L=4` for causal event tests. Specification-specific integration tests use `L=20`.

### 3.4 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_full_spec_online_contract.py
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python tests/codebase_compliance.py
```

### 3.5 Acceptance

- Existing active tests are green.
- Every new mismatch test fails at the intended assertion before implementation.
- Baseline public symbols and compliance counts are recorded.

### 3.6 Rollback

Only tests and fixtures are added. Revert the batch by removing those additions through a reviewed patch; no runtime artifact is affected.

## 4. Batch 1 — Stage-B metadata and independent calibration

### 4.1 Summary

This batch makes frozen offline memory the complete, provenance-safe source for online verification and separates calibration rules that currently share execution paths.

### 4.2 Metadata edits

Modify the current metadata flow in `src/models/thesis_multitask_state_mixin.py` first; do not add a parallel store.

1. Tighten the existing codeword mask/radius calculation to accept training or synthetic-training memory only.
2. Store `verification_metadata_source`, source split, class-to-codeword rule, radius statistic/quantile, number of contributing tokens, seed, and schema version.
3. Keep `anomalous_codeword_mask: BoolTensor[K]` and `anomaly_radii: FloatTensor[K]` as registered state/checkpoint-compatible tensors.
4. Reject missing, non-finite, negative, wrong-shape, or provenance-incomplete metadata at online startup.
5. Treat older checkpoints without the metadata as incompatible with A1/A2 full-spec execution. A0 may load only if it never requests verification metadata and the report marks the compatibility mode.

Modify `src/engine/online_tta/signature_verification.py::PrototypeVerificationMetadata` to carry immutable provenance and expose `from_model()` as a fail-closed adapter. Do not mutate model buffers.

### 4.3 Calibration edits

Keep pure tensor calculations in `threshold_calibration.py` and extract orchestration from `online_engine.py` into `src/engine/online_tta/online_calibration.py`.

Required functions for the detail implementation:

```python
collect_nonoverlap_offline_scores(...)
collect_stride1_online_scores(...)
calibrate_entity_threshold_artifact(...)
validate_threshold_artifact(...)
```

Each function must be at most 50 lines. Calibration runs under `eval()` and `no_grad()`. The offline collector uses non-overlapping/end-aligned windows; the online collector uses stride 1 and absolute-index EWMA. The two collectors may share point-score primitives but not their score timelines.

Update `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml` with explicit `offline_window_stride: 20`, `online_window_stride: 1`, `A_low_quantile: 0.95`, `A_high_quantile: 0.99`, and `B_window_quantile: 0.99`. Preserve existing key aliases only at config loading, then normalize to one internal name.

### 4.4 Tests

- Extend `tests/online/test_online_prototype_metadata_contract.py` for provenance, missing fields, non-finite radius, and checkpoint roundtrip.
- Add `tests/online/test_independent_threshold_calibration.py` for exact start indices and distinct artifacts.
- Extend `tests/online/test_entity_threshold_runtime.py` for entity/hash mismatch.
- Mutate validation labels and prove thresholds do not change.
- Mutate test labels and prove the already-persisted artifact does not change.

### 4.5 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_online_prototype_metadata_contract.py
.venv/bin/python -m pytest -q tests/online/test_independent_threshold_calibration.py tests/online/test_entity_threshold_runtime.py
.venv/bin/python -m pytest -q tests/runtime/test_checkpoint_roundtrip.py
```

### 4.6 Acceptance

- O0/O1 Stage-B smoke checkpoints contain valid metadata and provenance.
- Each entity receives independent offline/online thresholds with correct stride evidence.
- No validation/test label reaches calibration or metadata creation.

### 4.7 Rollback

Version the checkpoint and artifact schemas. Older artifacts remain readable only through explicit compatibility mode; newly generated A1/A2 artifacts must be regenerated if the schema is rolled back.

## 5. Batch 2 — Minimal one-window vertical slice

### 5.1 Summary

This batch establishes the smallest correct online path before PNN, A2 losses, resume, or demo complexity.

### 5.2 Batch contract edits

Modify `src/core/contracts.py::validate_online_batch()` to call `validate_batch()` and validate only online ordering metadata required by the active engine. Remove `view_a`/`view_b` requirements. If a supported legacy experiment still requires two views, create `validate_legacy_two_view_batch()` and keep it unreachable from active full-spec configs.

Update `src/core/console.py` so view summaries are optional legacy diagnostics rather than active contract evidence.

### 5.3 Adapter/model edits

Refactor `ThesisMultitaskEncoderAdapter` in `src/models/online_adaptation.py` around four explicit operations defined in Section 2.2. Replace `forward()` ambiguity with `encode_source()` in the new internal call chain while preserving `forward()` as a thin compatibility facade if current callers need it.

Refactor `OnlineAdaptationModel.forward()` into short steps:

```text
validate batch
  -> encode and detach source hidden once
  -> resolve variant scoring route
  -> source score or projector score
  -> attach semantically correct aux fields
  -> validate outputs
```

For A0, do not call `NearIdentityMLPProjector`. For A1/A2, project `Z_source`. Remove alignment losses that are not part of the locked online objective from the active adaptation decision; they may remain diagnostic-only if clearly named and disabled by config.

Publish the result of `_compute_latent_memory_score()` as `aux["latent_window_score"]`; never use `window_scores` there.

### 5.4 Engine slice

Extract score orchestration from `online_engine.py` into `src/engine/online_tta/online_scoring.py`. It should own window reconstruction score, latent score readback, EWMA updates, and point finalization helpers. It must not own triage or adaptation.

### 5.5 Tests

- Add `tests/models/test_online_one_window_forward.py` using spy adapters.
- Extend `tests/online/test_online_adaptation_step.py` for A0 direct scoring and A1/A2 projected scoring.
- Extend `tests/online/test_online_tta_trainable_surface.py` for exact parameter names.
- Extend `tests/online/test_online_stream.py` with one complete synthetic A0 stream.
- Retain loader tests for `[B,20,D]`, timestamps, masks, and metadata.

### 5.6 Validation

```bash
.venv/bin/python -m pytest -q tests/models/test_online_one_window_forward.py
.venv/bin/python -m pytest -q tests/online/test_online_adaptation_step.py tests/online/test_online_tta_trainable_surface.py tests/online/test_online_stream.py
.venv/bin/python -m pytest -q tests/data tests/models
```

### 5.7 Acceptance

- Exactly one frozen source encode occurs per window.
- A0 is source-only; A1/A2 are projector-only adaptation routes.
- Output shapes and meanings match Section 2.3.
- One short A0 stream completes deterministically.

### 5.8 Rollback

The boundary is the online batch/model interface. Preserve a compatibility validator and facade until all active callers migrate; do not maintain two active THESIS forward implementations.

## 6. Batch 3 — Exact triage and event strategies

### 6.1 Summary

This batch makes the online event path causal, exhaustive, and contamination-resistant.

### 6.2 Triage edits

Modify `triage.py::classify_online_window()` to implement only:

```text
input <= B_window                             -> normal
input > B_window and latent <= A_low          -> hard_old
input > B_window and A_low < latent <= A_high -> gray_zone
input > B_window and latent > A_high          -> strong_anomaly
```

Normalize legacy threshold aliases before calling the function. Remove `pnn_candidate` from THESIS result values; do not change baseline CANDI semantics.

### 6.3 Strategy module

Extract variant dispatch to `src/engine/online_tta/online_event_dispatch.py` with explicit callable functions such as `handle_a0_event`, `handle_a1_event`, and `handle_a2_event`. They receive scored event data and injected state owners. They do not load configs, write reports, or construct models.

Event order is fixed:

```text
score -> update EWMA -> triage
  -> A2 hard-old guard/update OR gray-zone admission
  -> verification cycle if due
  -> finalize future-only predictions
  -> record event
```

Gray-zone handlers may call only buffer admission and cycle triggering. Hard-old guard intervals are appended only after `did_update=True`.

### 6.4 Tests

- Exhaust all equality and epsilon boundaries in `test_online_tta_triage.py`.
- Add `tests/online/test_online_event_order.py` with spy buffer, guard, optimizer factory, and recorder.
- Extend `test_online_tta_variants.py` for allowed actions per variant.
- Prove no gray-zone optimizer call and no strong-anomaly adaptation.

### 6.5 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_online_tta_triage.py
.venv/bin/python -m pytest -q tests/online/test_online_event_order.py tests/online/test_online_tta_variants.py
```

### 6.6 Acceptance

The truth table is exhaustive and deterministic. Event traces match the specification, and state commits occur only after successful actions.

### 6.7 Rollback

Rollback the strategy module and engine wiring together. Do not restore `pnn_candidate` to the THESIS path merely to satisfy legacy tests; migrate only tests proven to encode obsolete semantics.

## 7. Batch 4 — Label-free verification and TTL

### 7.1 Summary

This batch verifies gray-zone windows independently and gives one object ownership of each piece of mutable state.

### 7.2 Verification batch edits

Modify `verification_adapter.py::build_entry_batch()` to return the shared one-window batch. Set `point_labels=None`; do not include view fields. Preserve absolute interval metadata.

`verify_buffer_entries()` should perform an independent frozen-source forward for every entry, then return immutable `VerificationResult` values. It must not mutate the buffer directly.

### 7.3 Tensor contracts

```text
hidden                    [N, L, H]
nearest_codeword_ids      [N, L]
nearest_codeword_distance [N, L]
known_anomaly_mask        [N, L]
continuous_signatures     [N, L, 3]
pnn_mask                  [N, L]
```

The codebook, radii, continuous prototypes, and source keys are detached. Signature order is meaningful. Recurrence requires the same ordered signature in more than one non-overlapping window.

### 7.4 State ownership edits

- `VerificationBuffer` owns entries, `ttl_remaining`, status, `was_adapted`, and new-admission state.
- `VerificationCycleController` owns `capacity=8`, trigger eligibility, callback invocation, and exactly-one cycle finalization.
- The engine owns neither TTL decrement nor buffer removal logic.

### 7.5 Tests

Extend the existing signature, buffer, and cycle tests to cover shapes, known-anomaly removal, ordered signatures, overlap, capacity 7/8, new-admission requirement, independent forward call count, labels absent, TTL 2 initialization, adapted removal, unresolved two-cycle retention, and no ordinary-step decrement.

### 7.6 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_online_signature_verification.py
.venv/bin/python -m pytest -q tests/online/test_online_verification_buffer.py tests/online/test_verification_cycle.py
```

### 7.7 Acceptance

One synthetic eight-entry cycle produces deterministic masks and statuses; every entry is finalized once per cycle; no labels or future windows enter verification.

### 7.8 Rollback

Version verification runtime state. If rolled back, mark checkpoints with the newer verification schema incompatible rather than partially restoring TTL values.

## 8. Batch 5 — Exact A1/A2 losses and atomic update

### 8.1 Summary

This batch implements the fine-grained adaptation objectives and prevents partial state updates when a gradient step fails.

### 8.2 Loss functions

Replace the current same-position matrix loss in `online_losses.py` with explicit functions under 50 lines:

```python
compute_hard_old_hinge_loss(reconstruction, target, b_window)
compute_masked_pnn_reconstruction_loss(reconstruction, target, pnn_mask)
compute_token_multi_positive_info_nce(
    projected_hidden,
    source_hidden,
    recurrent_signature_ids,
    pnn_mask,
    anomalous_codewords,
    known_anomaly_mask,
    temperature,
)
```

Hard-old A2 uses every projected token as anchor, same-token source latent as positive, and all anomalous codewords as negatives. PNN A2 uses PNN anchors, same-token source keys plus detached same-signature PNN projected keys as positives, and anomalous codewords plus available known-anomaly projected/source keys as negatives. Other tokens are ignored.

### 8.3 Config and optimizer

Add/normalize explicit task keys for `lambda_online_contrastive`, `online_learning_rate=1e-4`, `online_weight_decay=1e-4`, `projector_gradient_clip_norm=0.5`, `verification_buffer_size=8`, and `continuous_signature_topk=3`. Remove active engine constants.

Reuse `build_online_optimizer()` as the factory. Create a new AdamW instance per accepted event, but retain projector weights. Do not create a scheduler or persist optimizer state.

### 8.4 Atomic update helper

Add a focused transaction helper in `online_event_dispatch.py` or `online_update.py` if needed for the 500-line limit. It must assert the trainable surface, create/zero optimizer, forward, validate finite loss, backward, assert frozen gradients, clip, step, then return diagnostics. Buffer/guard commits happen only after the helper returns success.

### 8.5 Tests

- Add hand-computed tests in `tests/online/test_full_spec_online_losses.py`.
- Test empty PNN, source-only positive, multiple same-signature positives, anomalous-codeword negatives, known-anomaly negatives, ignored tokens, and temperature validation.
- Extend projector mutation/checksum tests for A0/A1/A2.
- Test a forced non-finite loss and forced optimizer exception; neither may commit buffer/guard state.

### 8.6 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_full_spec_online_losses.py
.venv/bin/python -m pytest -q tests/online/test_online_tta_trainable_surface.py tests/online/test_online_tta_variants.py
```

### 8.7 Acceptance

Hand-computed logits/losses match, intended anchors receive gradients, frozen components remain unchanged, clip norm is 0.5, and updates are atomic.

### 8.8 Rollback

Rollback loss functions, config keys, and dispatch together. Results generated with the incorrect or replaced loss version must retain a protocol-version marker and cannot be aggregated with corrected results.

## 9. Batch 6 — Full stream, checkpoint, and resume

### 9.1 Summary

This batch guarantees complete main-run coverage and causal equivalence after interruption.

### 9.2 Config generation

Modify `_task_overrides()` in `generate_online_benchmark_configs.py` so smoke writes `max_online_steps: 16` and main writes `null` or omits the key. Update `config_model_validation.py` so positive integers are smoke caps, `null` means complete stream, and zero/negative values fail.

Regenerate the THESIS online configs and review only expected mechanical changes. Main configurations must not retain `view_noise_std` as an active semantic input; remove obsolete view keys from generated full-spec configs after config compatibility tests are in place.

### 9.3 Execution split

Extract stream orchestration from `online_engine.py` to `online_execution.py`. The loop owns point arrival and calls injected scoring/dispatch components. It records:

```text
expected_windows = max(0, T - L + 1)
processed_windows
eligible_points
provisional_points
finalized_points
skipped_points
stream_coverage_status
```

Only smoke runs may report intentional truncation. A main report with incomplete coverage must be non-success.

### 9.4 Resume state

Extend `OnlineRuntimeState` with every causal field and a schema version. `resume_online_runtime()` must load checkpoint extra state, validate entity/variant/seed/window/checkpoint hash/threshold hash before mutation, restore buffer/signatures/guard/EWMA/cursor, rebuild a fresh optimizer only when an event later requires one, and continue at the next unseen point.

### 9.5 Tests

- Update generated-config tests from `200` to full-stream semantics.
- Assert `T-L+1` forwards for synthetic streams.
- Add `tests/online/test_online_resume_runtime.py` comparing event traces and canonical reports.
- Test every identity mismatch and absent optimizer moments.
- Extend wrapper tests for dry-run dependency resolution.

### 9.6 Validation

```bash
.venv/bin/python -m pytest -q tests/online/test_online_benchmark_config_generation.py tests/online/test_online_engine_max_steps.py
.venv/bin/python -m pytest -q tests/online/test_online_runtime_state.py tests/online/test_online_resume_runtime.py
.venv/bin/python -m pytest -q tests/benchmarks/test_thesis_online_benchmark_wrapper.py
```

### 9.7 Acceptance

All main configs are full-stream, smoke remains 16 steps, and resumed execution matches uninterrupted execution except timing fields.

### 9.8 Rollback

Rollback generator, generated YAMLs, validator, and runtime schema as one unit. Do not leave mixed `200`, `0`, and `null` meanings.

## 10. Batch 7 — Artifact, reporting, metrics, and W&B

### 10.1 Summary

This batch makes result validity explicit and reproducible independently of demo behavior.

### 10.2 Reporting module

Extract finalization from `online_engine.py` to `online_reporting.py`. Preserve existing report keys and add:

```text
matrix_status
runtime_protocol_status
stream_coverage_status
artifact_integrity_status
metric_availability_status
```

Only all-success statuses produce `experiment_status: complete`. `matrix_ready` must never be presented as full experiment readiness.

### 10.3 Artifact identity

Persist resolved config, git commit, dirty flag, seed, entity, device, dataset identity, source checkpoint path/hash, threshold path/hash, schema versions, processed counts, metric definitions, support counts, timing, and final report checksum. Use collision-safe run directories. A completion manifest is written last, after readback validation.

`src/engine/artifact_sinks.py` remains the output port. W&B and local sinks receive the same logical artifacts. Disabled W&B must not change computations or local completeness.

### 10.4 Metrics

Update the summary path to distinguish raw pointwise, eventwise, VUS, affiliation, and adjusted metrics. Raw metrics remain primary. One-class slices record unavailable metrics and support counts instead of inventing values or failing the protocol.

### 10.5 Tests

- Add `tests/benchmarks/test_full_spec_artifact_integrity.py`.
- Extend full-matrix preflight and summary tests with truncated, corrupted, hash-mismatched, duplicate, one-class, and W&B-disabled fixtures.
- Prove incomplete reports are excluded from aggregation and missing matrix cells are explicit.

### 10.6 Validation

```bash
.venv/bin/python -m pytest -q tests/benchmarks/test_full_spec_artifact_integrity.py
.venv/bin/python -m pytest -q tests/benchmarks/test_full_benchmark_matrix_preflight.py tests/benchmarks/test_summarize_benchmark_results.py
```

### 10.7 Acceptance

Only specification-valid, complete, hash-verified runs aggregate. Local and W&B metadata agree. Metric meanings and support are explicit.

### 10.8 Rollback

Version report/completion schemas. Older reports remain readable for historical display but cannot be silently promoted to corrected full-spec completion.

## 11. Batch 8 — Demo parity and label isolation

### 11.1 Summary

This batch builds the visual software on the validated scorer while keeping it outside official metric production.

### 11.2 Scorer boundary

Define one injected callback used by `run_live_online_replay()` that accepts only window values, absolute ordering metadata, runtime identity, and state. It returns point/window scores, decisions, and safe diagnostics. It never accepts labels.

### 11.3 Queue and UI edits

- `stream_queue.py` owns producer order, bounded queue, timeout, delay, pause, resume, and stop.
- `online_replay.py` owns accumulation until 20 points and calls the injected scorer.
- `demo_state.py` owns display state only.
- `plotting.py` owns raw channel, score, threshold, exact markers, and current-window band.
- `app.py` owns entity/channel selection, play/pause, speed, visible range, TTA mode, and status panels.

Offline replay loads checkpoint/threshold identity before drawing. Online replay shows queue size, window interval, latest score/decision, variant, buffer size, and update counts. Labels may be drawn afterward as an optional overlay.

### 11.4 Tests

Extend demo tests for queue order/backpressure, empty timeout, pause/resume/stop, no forward before 20 points, label-free spy callback, selected-channel display-only behavior, required state fields, and controlled failure status.

### 11.5 Validation

```bash
.venv/bin/python -m pytest -q tests/demo/test_demo_stream_queue.py
.venv/bin/python -m pytest -q tests/demo/test_live_online_replay.py tests/demo/test_demo_app.py tests/demo/test_demo_state.py
```

### 11.6 Acceptance

Both modes use persisted identities and the shared scorer; all controls work; labels cannot enter scoring/adaptation; demo outputs are never used as official metrics.

### 11.7 Rollback

Rollback the demo callback/UI boundary without touching official evaluation. Persisted experiment artifacts remain valid.

## 12. Batch 9 — Readability and lifecycle refactor

### 12.1 Summary

This batch enforces the hard readability contract only after corrected semantics are locked by tests.

### 12.2 Offline model composition

Map every method in the four `thesis_multitask_*_mixin.py` files. Replace lifecycle inheritance with composed, explicitly named collaborators while keeping `ThesisMultitaskModel` as the only public entrypoint. Suggested responsibilities are configuration parsing, memory state, forward routing, and objective calculation; they must not define a second public model or hide stage lifecycle.

Constructor, public forward/training/validation/test methods, stage transition, checkpoint extra-state hooks, and registry boundary must remain visible in `thesis_multitask.py`.

### 12.3 Online model and engine split

Keep `online_adaptation.py` as the public online model. Extract only focused source adapter/projector primitives when necessary.

Use the modules created earlier:

```text
online_engine.py          public facade and high-level call chain
online_calibration.py     entity calibration orchestration
online_scoring.py         scoring and EWMA
online_event_dispatch.py  strategies and atomic updates
online_execution.py       stream and coverage
online_reporting.py       checkpoint/report finalization
```

Dependencies point inward toward pure helpers and state types; no extracted module imports the public engine facade.

### 12.4 Compliance process

Before each extraction, run characterization tests. Apply one extraction at a time, update imports through public entrypoints, rerun focused tests, and then update `tests/compliance/fixtures/src_refactor_contracts.json`. Every method/function must be at most 50 lines and every `src/` code file at most 500 lines.

Update `documents/abstract-design-notes/design_starter.md` only if the repository tree materially changes.

### 12.5 Validation

```bash
.venv/bin/python tests/codebase_compliance.py
.venv/bin/python -m pytest -q tests/compliance
.venv/bin/python -m pytest --collect-only -q
.venv/bin/python -m pytest -q
```

### 12.6 Acceptance

Zero source/callable violations, zero lifecycle mixins distributing public model behavior, unchanged public registry/import/checkpoint contracts, and a green full active suite.

### 12.7 Rollback

Each file extraction is an independent rollback boundary. Use reviewed patches only; never use destructive Git reset/checkout commands. Preserve unrelated user changes.

## 13. Batch 10 — Preflight, CUDA smoke, and full matrix

### 13.1 Summary

This batch supplies real-environment evidence before expensive benchmark execution.

### 13.2 Preflight edits

Extend the existing `--require-cuda` path in `scripts/preflight_full_benchmark_matrix.py`. Validate matrix counts, full-stream main semantics, `25+5` epoch budgets, config loadability, dataset paths, exact Stage-B dependencies, entity threshold maps, checkpoint/artifact hashes, CUDA device identity, deterministic seeds, output writeability, disk estimate, resume identity, and collision-safe run directories.

Add `tests/benchmarks/test_full_spec_gpu_preflight.py` using simulated device outcomes; CPU test environments must not claim CUDA evidence.

### 13.3 Exact local commands

```bash
.venv/bin/python scripts/preflight_full_benchmark_matrix.py --json
.venv/bin/python -m pytest -q tests/online tests/benchmarks tests/demo tests/compliance
.venv/bin/python tests/codebase_compliance.py
.venv/bin/python -m pytest -q
```

### 13.4 Exact smoke entrypoints

Offline O0 smoke:

```bash
.venv/bin/python scripts/run_two_stage_offline_pretraining.py \
  --experiment-config configs/experiment/benchmark_smoke/thesis/smd__thesis_multitask__benchmark-two-stage-machine_1_6__w20__seed6__smoke.yaml
```

Offline O1 smoke:

```bash
.venv/bin/python scripts/run_two_stage_offline_pretraining.py \
  --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_1_6__w20__seed6__smoke.yaml
```

Online smoke commands use the existing wrapper:

```bash
.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A0__machine_1_6__w20__seed6__smoke.yaml \
  --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
  --online-variant A0

.venv/bin/python scripts/run_thesis_online_benchmark.py \
  --experiment-config configs/experiment/online_benchmark/thesis/smd__thesis__online__O0_A2__machine_1_6__w20__seed6__smoke.yaml \
  --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
  --online-variant A2
```

Repeat with resolved O1-A0 and O1-A2 configs. Run one A2 interruption/resume scenario after the official CLI exposes and tests an explicit resume argument; do not simulate resume by manually editing state files.

On the CUDA server, run:

```bash
.venv/bin/python scripts/preflight_full_benchmark_matrix.py --json --require-cuda
```

Use `tmux`, one explicit `CUDA_VISIBLE_DEVICES` assignment per process, and save command, git commit, device name, environment lock, timestamps, output/checkpoint/threshold paths, hashes, and status to a dated detail execution log.

### 13.5 Full matrix

Lock and validate:

- 18 THESIS offline main runs;
- 54 THESIS online main runs;
- 9 RedLamp main runs;
- 27 traditional offline main runs;
- 81 online baseline main runs.

Every online run resolves its exact Stage-B checkpoint and entity threshold artifact before launch. `--skip-completed` may skip only completion manifests that pass identity, coverage, and checksum readback. Failed runs write an explicit non-success manifest and remain visible to aggregation.

### 13.6 Acceptance

- CPU preflight, focused suites, compliance, and full suite pass.
- O0/O1 offline and O0/O1-A0/A2 online smokes pass on CUDA.
- One A2 resume smoke is causally equivalent.
- Artifact paths/hashes survive resume.
- Every full-matrix cell is complete, failed, or missing; none is silent.

### 13.7 Rollback

Preflight/launcher changes do not alter model mathematics. Stop the launch on any failed gate. Existing valid artifacts remain immutable; invalid or superseded artifacts are marked, not deleted.

## 14. Cross-cutting research risks and mitigation

### 14.1 Prototype redundancy

Log continuous/discrete usage, nearest-memory distances, codeword support, and branch outputs. Preserve config switches for continuous-only, discrete-only, and fused ablations. Do not change the main method during remediation; use later ablation results to assess redundancy.

### 14.2 Fusion collapse

Log reconstruction/classification fusion weights, branch norms, task losses, and gradient norms. Keep O0/O1 semantics fixed. A collapse diagnostic may fail an analysis gate but must not silently modify loss weights.

### 14.3 Adaptation contamination

Use training-derived anomaly metadata, four-region triage, gray-zone verification, known-anomaly exclusion, non-overlap guards, label-isolation tests, and atomic state commits.

### 14.4 Projector drift and high variance

Retain near-identity initialization, projector-only training, exact source/codeword contrastive keys, fresh conservative AdamW, clip norm 0.5, finite checks, and drift diagnostics. Stop rather than reset silently when drift is non-finite.

### 14.5 Metric inflation

Freeze clean-validation thresholds, isolate test labels, report raw metrics first, name adjusted metrics, include support/coverage, and reject incomplete or identity-mismatched runs.

## 15. Required regression ladder

After any shared contract change, run tests in this order:

```text
one failing contract test
  -> focused pure/unit file
  -> owning semantic folder
  -> checkpoint/config integration
  -> benchmark preflight
  -> compliance scan
  -> full active pytest
  -> CPU smoke
  -> CUDA smoke
```

No test may be archived merely to obtain a green suite. A test may move to legacy storage only when repository evidence proves it targets a removed, non-active contract and the active replacement test is present.

## 16. Final definition of done

Implementation is complete only when:

1. Dataset, batch, encoder, model, task, checkpoint, threshold, state, and report contracts are enforced by tests.
2. O0/O1 offline Stage A/Stage B outputs and metadata are reproducible.
3. A0/A1/A2 match the one-window and update equations exactly.
4. Gray-zone adaptation is verification-only and label-free.
5. A2 positive, negative, and ignored token sets are exact.
6. Main runs process the full stream and resume causally.
7. Reports distinguish structural readiness from protocol-valid completion.
8. W&B/local artifacts share identity and content.
9. Demo code uses the shared scorer and cannot affect official metrics.
10. All public compatibility surfaces remain documented and tested.
11. Every `src/` file is at most 500 lines and callable at most 50 lines.
12. Focused, full, preflight, CUDA smoke, resume, and artifact gates pass.
13. All 189 main matrix cells are explicitly accounted for.

## 17. Implementation review gate

This detail plan is ready for owner review. Coding should start at Batch 0 only after explicit approval. Once implementation begins, each batch must be completed, verified, and logged before the next batch starts. The implementation team must not redesign the two locked decisions or skip directly to the full benchmark.

## 18. Implementation progress — 2026-07-11

- [x] Batch 0 contract tests added; baseline was 390 passing tests.
- [x] Batch 1 metadata validation/provenance tightened and offline non-overlap calibration separated from stride-1 online calibration.
- [x] Batch 2 active THESIS forward uses one frozen source encoding and exposes the real latent memory score; A0 has a direct source-only route.
- [x] Batch 3 THESIS triage has exactly four regions; legacy baseline triage is isolated; gray-zone no longer updates.
- [x] Batch 4 verification batches are single-window and label-free; existing cycle/TTL ownership remains active.
- [x] Batch 5 hard-old and verified-PNN updates use anomalous codeword negatives and optional recurrent-signature positives.
- [x] Batch 6 generated main online configs use full-stream `null`; smoke configs retain 16 steps; existing causal resume tests pass.
- [ ] Batch 7 expanded artifact status/checksum readback is not yet implemented.
- [x] Batch 8 existing queue/demo label-isolation regression tests pass.
- [ ] Batch 9 is open: AST audit reports 12 files and 69 callables over the hard limits after checkpoint, evaluator, calibration, and online-window event extractions.
- [x] Batch 10 CPU matrix preflight reports structurally ready with counts 18/54/9/27/81.
- [ ] Batch 10 CUDA smokes, resume evidence, and full 189-run execution require the target GPU environment.

Current regression evidence: `397 passed, 23 warnings`. The warnings are existing one-class metric, joblib CPU-count, and STUMPY diagnostics rather than new failures.

## 19. Master implementation checklist

This checklist is the execution ledger for the complete remediation. An item is marked `[x]` only when its implementation and the corresponding verification evidence exist in the repository. An open item must not be reported as complete merely because the code path exists.

### 19.1 Batch 0 — shared contracts and characterization

- [x] Add focused tests for dataset, batch, encoder, model, task, checkpoint, threshold, state, and report contracts.
- [x] Establish the baseline regression count and preserve the active test suite.
- [x] Remove the active THESIS requirement for synthetic `view_a`/`view_b` fields while preserving an explicit legacy validator.
- [x] Record the locked design decisions before implementation.

### 19.2 Batch 1 — metadata, calibration, and non-overlap

- [x] Validate codebook and verification radii for finite values and compatible shapes.
- [x] Persist verification metadata schema, provenance, split, quantile, and label-source fields.
- [x] Separate offline non-overlap threshold collection from stride-1 online calibration.
- [x] Add regression coverage for metadata load/validation and threshold separation.

### 19.3 Batch 2 — source encoding and model scoring

- [x] Use one frozen source encoding in the active THESIS online forward path.
- [x] Expose source-only scoring for A0.
- [x] Expose the actual latent-memory distance as `latent_window_score`.
- [x] Preserve public model/checkpoint compatibility surfaces.
- [x] Test source/projected scoring and output shapes.

### 19.4 Batch 3 — triage and update gates

- [x] Implement the exact four-region THESIS truth table.
- [x] Isolate legacy baseline triage semantics from active THESIS semantics.
- [x] Make gray-zone decisions verification-only with no adaptation update.
- [x] Test all threshold regions and invalid/missing-threshold failures.

### 19.5 Batch 4 — verification lifecycle

- [x] Keep verification batches single-window and label-free.
- [x] Preserve verification cycle and TTL ownership.
- [x] Prevent test labels from entering scoring, verification, or adaptation.
- [x] Run verification-cycle and label-isolation tests.

### 19.6 Batch 5 — A1/A2 losses and atomic updates

- [x] Restrict A1 updates to verified PNN candidates.
- [x] Restrict A2 updates to verified PNN or hard-old normality.
- [x] Use anomalous codeword negatives and optional recurrent-signature positives.
- [x] Keep known anomalies out of positive/negative adaptation sets.
- [x] Preserve finite-gradient, clipping, and atomic state-commit guards.

### 19.7 Batch 6 — generated configs and resume identity

- [x] Generate main online configs with full-stream `max_online_steps: null`.
- [x] Keep smoke configs bounded and explicit.
- [x] Remove inactive view-noise/view-dropout overrides from active configs.
- [x] Validate O0/O1 and A0/A1/A2 config loadability.
- [x] Run existing causal resume tests.

### 19.8 Batch 7 — artifact/report integrity

- [x] Write and immediately checksum-readback an online-run manifest for checkpoint, threshold, metrics, and records; an integrity failure prevents `completed` status.
- [x] Write and immediately checksum-readback a separate online benchmark-report manifest.
- [ ] Add checksum generation for checkpoint, threshold, report, and manifest artifacts.
- [ ] Add checksum readback and identity verification before `--skip-completed`.
- [ ] Persist explicit non-success manifests for failed or incomplete runs.
- [ ] Reject aggregation of missing, incomplete, or identity-mismatched artifacts.
- [ ] Add focused artifact-integrity tests and report readback tests.

### 19.9 Batch 8 — demo boundary

- [x] Route demo scoring through the shared scorer.
- [x] Keep demo labels outside scoring and adaptation.
- [x] Preserve queue/state controls and persisted identities.
- [x] Verify demo outputs cannot become official metrics.
- [x] Run the focused demo regression suite.

### 19.10 Batch 9 — readability and lifecycle refactor

- [x] Add the compliance audit and deterministic violation report.
- [x] Refactor `CheckpointManager.save_checkpoint` into payload construction and artifact synchronization helpers without changing its public contract.
- [x] Extract evaluator window-payload validation, entity-accumulator initialization, and reconstructed-record construction helpers; evaluation semantics remain unchanged.
- [x] Extract clean-validation scoring, EWMA collection, and online-stream construction into `online_calibration.py`; the public online engine facade remains unchanged.
- [x] Split one-window online event handling into preparation, buffer/verification admission, update transaction, and result finalization helpers; A0/A1/A2 regressions remain green.
- [x] Extract Benjamini–Hochberg adjustment from anomaly-archive ranking; ranking and significance semantics remain unchanged.
- [ ] Refactor the remaining 11 oversized source files to at most 500 lines.
- [ ] Refactor the remaining oversized callables to at most 50 lines.
- [ ] Split `online_engine.py` into the planned calibration, scoring, event-dispatch, execution, and reporting seams.
- [ ] Keep `online_adaptation.py` as the public model entrypoint after extraction.
- [ ] Keep `ThesisMultitaskModel` as the only public offline model entrypoint.
- [ ] Ensure mixins no longer distribute hidden public lifecycle behavior.
- [ ] Update compliance fixtures only after each extraction is characterized and tested.
- [ ] Reach zero file and callable violations.

### 19.11 Batch 10 — preflight, CUDA smoke, and full matrix

- [x] Run CPU matrix preflight and verify structural counts `18/54/9/27/81`.
- [x] Run four local online dry-run wrappers for O0/O1-A0/A2 config resolution.
- [ ] Extend and test `--require-cuda` device and environment checks.
- [ ] Run O0 and O1 offline smokes on the rented CUDA server.
- [ ] Run O0/O1-A0/A2 online smokes on CUDA.
- [ ] Run one explicit A2 interruption/resume smoke on CUDA.
- [ ] Verify artifact hashes, resume identity, and collision-safe run directories.
- [ ] Execute and account for all 189 main matrix cells.
- [ ] Produce the dated remote execution log with command, commit, device, environment, timestamps, paths, hashes, and status.

### 19.12 Final gates

- [x] Focused online, loss, contract, verification, and demo tests pass.
- [x] Full local active suite passes: `397 passed, 23 warnings`.
- [ ] Compliance suite passes with zero readability violations.
- [ ] CPU preflight, compliance, and full suite are rerun after all Batch 9 changes.
- [ ] CUDA smoke, resume, artifact, and full-matrix gates pass.
- [ ] Final definition of done is checked item-by-item before renting the remote server.
