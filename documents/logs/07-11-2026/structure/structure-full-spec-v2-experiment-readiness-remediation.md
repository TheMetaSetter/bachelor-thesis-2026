---
date: 2026-07-11T01:30:18+0700
researcher: Codex
git_commit: 99e6e586150cf19363618d98d4462b19e19d93eb
branch: dev
repository: bachelor-thesis-2026
topic: "Structure outline for the detailed full-spec-v2 experiment-readiness programming plan"
tags: [structure, full-spec-v2, offline, online-tta, benchmark, demo, reproducibility]
status: accepted_for_detail
source_plan: documents/logs/07-11-2026/plan/plan-full-spec-v2-experiment-readiness-remediation.md
last_updated: 2026-07-11
last_updated_by: Codex
---

# Structure outline: detailed `full-spec-v2` programming plan

## 0. Overview

The detailed programming plan should organize the remediation as a sequence of small, dependency-ordered implementation batches. It must preserve the current registries, public model entrypoints, O0/O1 checkpoints, A0/A1/A2 configuration matrix, report keys, and already-valid helper implementations while replacing only behavior that conflicts with `documents/spec/full-spec-v2.md`.

The minimal vertical slice is the first executable target:

```text
one shared batch
  -> one frozen source encoding
  -> source or projected scoring
  -> independent clean-validation thresholds
  -> exact four-region triage
  -> one short causal stream
  -> complete, identity-checked report
```

Advanced PNN verification, multi-positive contrastive learning, resume, demo, readability refactoring, CUDA evidence, and the full benchmark matrix must build on that slice rather than introduce alternate runners or model paths.

## 1. Authority and non-negotiable decisions

The detailed plan should open by restating the authority order:

1. `documents/spec/full-spec-v2.md` owns mathematical and protocol semantics.
2. The accepted plan owns implementation phases and completion criteria.
3. The July 9 detail artifact supplies compatibility context for the current matrix and public surfaces.
4. The July 10 readiness research supplies verified current gaps.
5. Active code supplies the starting implementation, not permission to override the specification.

The following accepted decisions must appear as fixed premises, not open questions:

- Online THESIS input is one `batch["x"]` window with shape `[B, 20, D]`.
- `Z_source` is encoded once; A1/A2 project `Z_source`; A0 scores `Z_source` directly.
- Anomalous codeword membership and radii originate from training/synthetic-training memory and are stored in the Stage-B checkpoint with provenance.
- Buffer verification uses an independent, label-free frozen-source forward.
- Gray-zone admission does not perform adaptation.
- Main online runs process the full test stream; finite caps are smoke-only.
- Only the residual projector is trainable online.
- Test labels are metrics-only.

## 2. Current implementation surfaces to reuse

The detailed plan should explicitly recognize the following existing surfaces so implementation does not duplicate them:

- `src/models/thesis_multitask_state_mixin.py` already stores and restores `anomalous_codeword_mask` and `anomaly_radii` through checkpoint extra state.
- `src/engine/online_tta/signature_verification.py::PrototypeVerificationMetadata` already validates mask and radius shape, dtype, and non-negativity.
- `src/engine/online_tta/verification_buffer.py::VerificationBuffer` already owns admission and TTL-related methods.
- `src/engine/online_tta/verification_cycle.py::VerificationCycleController` already owns the capacity-trigger surface.
- `src/engine/online_tta/runtime_state.py` already provides state validation and resume helpers.
- `src/engine/online_tta/online_optimizer.py` already provides projector collection, optimizer construction, gradient clipping, and trainable-surface assertions.
- `demo/stream_queue.py`, `demo/online_replay.py`, and `demo/app.py` already provide a queue and replay skeleton.
- `scripts/preflight_full_benchmark_matrix.py` already verifies matrix counts and part of the protocol.

The detail artifact should classify edits to these surfaces as one of: preserve, tighten, rewire, split, or remove obsolete semantics. It should not propose parallel replacements without evidence that an existing public surface cannot satisfy its responsibility.

## 3. Target responsibility map

The detailed plan should use the following ownership structure.

### 3.1 Public model ownership

- `src/models/thesis_multitask.py::ThesisMultitaskModel` remains the only public offline THESIS model entrypoint.
- `src/models/online_adaptation.py::OnlineAdaptationModel` remains the only public THESIS online model entrypoint.
- `ThesisMultitaskEncoderAdapter` composes the source model and exposes read-only encoding, scoring, and verification metadata.
- `NearIdentityMLPProjector` owns only the residual transformation.
- Neither model owns stream cursor, buffer lifecycle, metric finalization, or report serialization.

### 3.2 Pure computation ownership

- `threshold_calibration.py` owns score aggregation and quantiles.
- `triage.py` owns the exhaustive four-region decision.
- `signature_verification.py` owns codeword filtering, ordered signatures, recurrence, and PNN masks.
- `online_losses.py` owns hard-old hinge, masked reconstruction, and exact token InfoNCE.
- Pure helpers accept tensors and typed values, do not load configs, do not write artifacts, and do not mutate model memory.

### 3.3 Stateful runtime ownership

- `VerificationBuffer` owns entries and per-entry TTL.
- `VerificationCycleController` owns trigger eligibility and one-cycle finalization.
- `NonOverlapGuard` owns successfully adapted hard-old intervals.
- `OnlineRuntimeState` owns entity/variant identity and all causal state required for resume.
- The online engine coordinates these objects but does not reimplement their rules.

### 3.4 Boundary ownership

- Configuration validation owns accepted key meanings.
- The checkpoint manager owns serialization transport; model/runtime objects own their own extra-state payloads.
- Benchmark scripts own dependency resolution and orchestration, not model mathematics.
- Demo code owns presentation and queue control, not scoring semantics.
- Report and artifact code owns status, provenance, hashes, and W&B mirroring.

## 4. Proposed detailed-plan sections and implementation batches

The later detail artifact should follow the sections below in this exact order.

## 5. Batch 0 — Baseline inventory and contract freeze

### Purpose

Freeze public behavior that must survive the remediation and introduce failing tests for every known result-changing mismatch.

### Required subsections in the detail plan

1. Record git commit, active configs, registry names, checkpoint keys, report keys, and current test collection.
2. Capture file/callable compliance violations before refactoring.
3. Add one-invariant-per-test contract cases for one-window input, one encoder call, latent-score meaning, gray-zone no-update, hard-old contrastive execution, full-stream main semantics, independent calibration, and label isolation.
4. Define golden deterministic fixtures for source scoring and runtime event traces.
5. State the rollback boundary: tests and fixtures only; no runtime behavior changes.

### Primary files

- `tests/online/test_full_spec_online_contract.py` as a new focused contract file.
- Existing tests under `tests/online/`, `tests/benchmarks/`, and `tests/demo/` for compatibility assertions.
- `tests/compliance/fixtures/src_refactor_contracts.json` for public refactor constraints.

### Gate

All existing tests remain green, while every new test fails only at the intended current mismatch.

### Engineering principles

Contract tests implement stable interfaces before internal refactoring. Characterization fixtures protect compatibility and allow composition-oriented extractions later.

## 6. Batch 1 — Checkpoint metadata and calibration provenance

### Purpose

Make the offline Stage-B checkpoint and clean-validation artifacts sufficient for exact, label-free online execution.

### Batch 1.1 — Training-memory verification metadata

The detail plan should trace and then tighten the existing flow in `thesis_multitask_state_mixin.py` rather than create a second metadata store:

```text
training/synthetic-training hidden tokens
  -> discrete codeword initialization
  -> anomalous class membership
  -> per-codeword anomaly radius
  -> registered model buffers
  -> Stage-B checkpoint extra state
```

It should specify provenance fields, fail-closed validation, compatibility behavior for older checkpoints, and focused tests. It must clarify which training labels are permitted for synthetic anomaly class memory and prove no validation/test label is consumed.

### Batch 1.2 — Independent offline and online calibration

The detail plan should separate:

- non-overlapping clean-validation point-score calibration for `offline_point_threshold_nonoverlap`;
- stride-1 clean-validation EWMA calibration for `online_point_threshold_ewma`;
- stride-1 clean-validation window calibration for `B_window`, `A_low`, and `A_high`.

It should assign pure calculations to `threshold_calibration.py` and entity artifact orchestration to extracted calibration orchestration rather than the 1,237-line `online_engine.py`.

### Primary files

- `src/models/thesis_multitask_state_mixin.py` initially, followed by the lifecycle refactor boundary defined in Batch 9.
- `src/engine/checkpoint.py`.
- `src/engine/online_tta/signature_verification.py`.
- `src/engine/online_tta/threshold_calibration.py`.
- Calibration portions currently inside `src/engine/online_tta/online_engine.py`.
- `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`.
- `tests/online/test_online_prototype_metadata_contract.py`.
- `tests/online/test_entity_threshold_runtime.py`.
- New `tests/online/test_independent_threshold_calibration.py`.

### Gate

O0 and O1 smoke checkpoints expose valid metadata with provenance, and one entity artifact contains distinct, reproducible offline and online thresholds.

### Engineering principles

Serialization transport is separated from domain-owned state. Pure calibration is separated from artifact orchestration. Existing adapter and checkpoint contracts are preserved.

## 7. Batch 2 — Minimal one-window online vertical slice

### Purpose

Complete the smallest end-to-end THESIS online flow before PNN or A2 complexity is enabled.

### Batch 2.1 — Shared batch contract

The detail plan should remove active THESIS dependence on `view_a` and `view_b` from `validate_online_batch()` while preserving a separately named compatibility boundary only if a currently supported non-full-spec experiment requires it.

### Batch 2.2 — One source encode and two scoring routes

The exact flow should be:

```text
batch["x"]
  -> frozen source encoder exactly once
  -> Z_source.detach()
      -> A0: frozen source memories and heads
      -> A1/A2: residual projector -> frozen source memories and heads
```

The detail plan should assign source encoding/scoring to `ThesisMultitaskEncoderAdapter`, public routing to `OnlineAdaptationModel`, and variant choice to the runtime strategy boundary. The adapter must expose the genuine memory-distance latent score.

### Batch 2.3 — Short causal A0 stream

Add an integration test that builds one short stream, forms all stride-1 windows, performs A0 source-only scoring, updates EWMA, finalizes point predictions, and writes an identity-complete report.

### Primary files

- `src/core/contracts.py`.
- `src/core/console.py`.
- `src/models/online_adaptation.py`.
- Scoring and stream portions of `src/engine/online_tta/online_engine.py`.
- `tests/models/test_online_one_window_forward.py` as a new file.
- `tests/online/test_online_adaptation_step.py`.
- `tests/online/test_online_tta_trainable_surface.py`.
- `tests/online/test_online_stream.py`.

### Gate

The source encoder is called exactly once per online window, A0 bypasses the projector, tensor/output meanings are correct, and the short A0 stream is complete and deterministic.

### Engineering principles

The adapter pattern isolates the offline source model. Composition replaces duplicated online encoders. The minimal vertical slice proves stable contracts before advanced adaptation.

## 8. Batch 3 — Triage, event strategies, and gray-zone ownership

### Purpose

Make each causal event follow one unambiguous strategy and prevent pre-verification contamination.

### Batch 3.1 — Exhaustive four-region triage

The detail plan should lock exact threshold equality behavior and remove `pnn_candidate` as a raw score region from the active THESIS path. PNN becomes a verification outcome.

### Batch 3.2 — Strategy dispatch without subclasses

The structure should use three explicit strategy functions or immutable strategy objects:

```text
A0 -> score only
A1 -> admit gray-zone; adapt verified PNN only
A2 -> adapt accepted hard-old; admit gray-zone; adapt verified PNN
```

The detailed plan should prefer explicit dispatch and dependency injection over an inheritance hierarchy. Baseline-specific meanings such as CANDI's `pnn_candidate` must remain isolated under baseline code.

### Batch 3.3 — Gray-zone and hard-old event order

The engine should sequence score, EWMA, triage, admission/guard decision, optional update, verification cycle, future-only prediction finalization, and event recording. Gray-zone dispatch must return without an optimizer event. Hard-old intervals are recorded only after successful adaptation.

### Primary files

- `src/engine/online_tta/triage.py`.
- `src/engine/online_tta/non_overlap_guard.py`.
- Event-dispatch portions of `src/engine/online_tta/online_engine.py`.
- `tests/online/test_online_tta_triage.py`.
- `tests/online/test_online_tta_variants.py`.
- New `tests/online/test_online_event_order.py`.

### Gate

An exhaustive boundary table and a deterministic event trace match the specification. A gray-zone event cannot reach optimizer construction.

### Engineering principles

Strategy semantics are explicit without proliferating subclasses. Triage remains pure; mutation remains in state owners; engine orchestration remains linear.

## 9. Batch 4 — Label-free verification and TTL lifecycle

### Purpose

Turn buffered gray-zone windows into verified PNN adaptation candidates using the accepted label-free design.

### Batch 4.1 — Independent verification forward

The detail plan should modify `verification_adapter.py::build_entry_batch()` to build the same one-window shared batch rather than synthesize two views. Each buffered entry receives an independent frozen-source forward with no labels.

### Batch 4.2 — Prototype filtering and recurrent signatures

Reuse the existing pure functions for nearest discrete codewords, anomaly-radius filtering, ordered top-3 continuous signatures, recurrence across non-overlapping windows, and PNN mask construction. The detailed plan should specify tensor shapes and exactly which tensors are detached.

### Batch 4.3 — One verification-cycle owner

`VerificationCycleController` should become the sole trigger coordinator. `VerificationBuffer` should remain the sole entry/TTL owner. The engine should call one controller method and consume a result; it should not duplicate capacity, new-admission, status, or TTL rules.

### Required lifecycle

```text
admission -> ttl_remaining = 2
capacity >= 8 and new admission -> verify all entries
adapted -> remove immediately
unresolved -> decrement once
ttl == 0 -> remove
ordinary stream step -> no TTL change
```

### Primary files

- `src/engine/online_tta/verification_adapter.py`.
- `src/engine/online_tta/signature_verification.py`.
- `src/engine/online_tta/verification_buffer.py`.
- `src/engine/online_tta/verification_cycle.py`.
- Verification portions of `src/engine/online_tta/online_engine.py`.
- `tests/online/test_online_signature_verification.py`.
- `tests/online/test_online_verification_buffer.py`.
- `tests/online/test_verification_cycle.py`.

### Gate

Capacity, new-admission, filter order, PNN mask, adapted/unresolved removal, and exactly-once TTL behavior pass without any label entering the callback.

### Engineering principles

State ownership is singular. Verification calculations are pure. The controller composes existing buffer and adapter objects instead of introducing new inheritance.

## 10. Batch 5 — Exact online losses and projector update transaction

### Purpose

Implement mathematically exact A1/A2 updates as one auditable transaction.

### Batch 5.1 — Loss primitives

The detail plan should lock exact function signatures and tensor contracts for:

- hard-old reconstruction hinge;
- masked PNN reconstruction;
- token multi-positive InfoNCE.

It should enumerate positive, negative, and ignored token sets separately for A2 hard-old and A2 PNN. Every anomalous codeword must be a negative. Projected keys for other anchors must be detached. The active anchor must retain gradient.

### Batch 5.2 — Variant loss composition

```text
A1 verified PNN -> L_pnn_recon
A2 hard-old -> L_hard_recon + lambda_contrastive * L_contrastive
A2 verified PNN -> L_pnn_recon + lambda_contrastive * L_contrastive
```

No loss weight should be hard-coded in the engine. Empty masks and invalid metadata should fail or skip according to explicit contract tests, never produce NaN.

### Batch 5.3 — Update transaction

One accepted event should perform:

```text
assert frozen surface
  -> create fresh AdamW over projector only
  -> zero gradients
  -> forward/loss
  -> finite-loss check
  -> backward
  -> frozen-gradient assertions
  -> clip projector norm to 0.5
  -> optimizer step
  -> commit guard/buffer state
  -> record diagnostics
```

If any step fails, no guard interval or adapted buffer state may be committed.

### Primary files

- `src/engine/online_tta/online_losses.py`.
- `src/engine/online_tta/online_optimizer.py`.
- Strategy/update portions of `src/engine/online_tta/online_engine.py`.
- `configs/model/online_adaptation.yaml`.
- `configs/task/online_adaptation.yaml`.
- New `tests/online/test_full_spec_online_losses.py`.
- `tests/online/test_online_tta_trainable_surface.py`.
- `tests/online/test_online_tta_variants.py`.

### Gate

Hand-computed loss fixtures and gradient checks pass. A0 changes nothing; A1/A2 change only projector parameters; state is committed only after success.

### Engineering principles

Losses are pure functions. Optimizer construction is a factory. The event update is transaction-like and keeps state mutation after successful computation.

## 11. Batch 6 — Full-stream execution and causal resume

### Purpose

Remove silent main-run truncation and make resumed execution identical to uninterrupted execution.

### Batch 6.1 — Configuration semantics

The detailed plan should define one meaning per value:

- main: missing or `null` cap means full stream;
- smoke: positive cap `16`;
- zero or negative values: invalid.

It should name the generator, validator, and generated-config tests that change together.

### Batch 6.2 — Coverage accounting

For a stream of length `T`, the report should prove that exactly `max(0, T-L+1)` windows were processed unless the run is explicitly smoke. It should track eligible points, provisional points, finalized points, skipped points, and the final coverage status.

### Batch 6.3 — Resume identity and causal state

The detailed plan should enumerate every serialized field, restore order, and mismatch check. It must state that optimizer moments are never stored or restored and that the next unseen point is processed exactly once.

### Primary files

- `scripts/generate_online_benchmark_configs.py`.
- `src/core/config_model_validation.py`.
- Stream/context/finalization portions of `src/engine/online_tta/online_engine.py`.
- `src/engine/online_tta/runtime_state.py`.
- `scripts/run_online_adaptation.py`.
- `scripts/run_thesis_online_benchmark.py`.
- `tests/online/test_online_benchmark_config_generation.py`.
- `tests/online/test_online_engine_max_steps.py`.
- `tests/online/test_online_runtime_state.py`.
- New `tests/online/test_online_resume_runtime.py`.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`.

### Gate

Main configs are full-stream, smoke configs remain capped, coverage is complete, and interrupted/resumed traces equal uninterrupted traces except timing fields.

### Engineering principles

Configuration meanings are single-purpose. Runtime state is explicit and serializable. Resume validation occurs before mutation.

## 12. Batch 7 — Artifact, report, and W&B integrity

### Purpose

Make successful execution reproducible and prevent incomplete or corrupted runs from being presented as valid benchmark results.

### Batch 7.1 — Status and artifact integrity

The detailed plan should separate matrix, runtime protocol, stream coverage, artifact integrity, and metric availability status. It should specify hashes, provenance, collision behavior, local/W&B parity, and rejection of corrupted or truncated reports.

### Batch 7.2 — Metric definitions

The plan should name primary raw pointwise/eventwise metrics, VUS and affiliation support, support counts, unavailable one-class cases, and the prohibition on silently substituting adjusted metrics.

### Primary files

- `src/engine/artifact_sinks.py`.
- `src/engine/logger.py`.
- Report construction extracted from `online_engine.py`.
- `scripts/summarize_benchmark_results.py`.
- `scripts/preflight_full_benchmark_matrix.py`.
- Relevant tests under `tests/benchmarks/`.

### Gate

Incomplete/corrupt fixtures are rejected and valid artifacts are reproducible locally and in W&B metadata.

### Engineering principles

Artifact sinks use a ports-and-adapters boundary. Report status is explicit rather than inferred.

## 13. Batch 8 — Demo parity and label isolation

### Purpose

Expose the validated causal scorer through the offline and online demo modes without duplicating official evaluation or adaptation logic.

### Batch 8.1 — Shared demo scoring callback

The demo should inject the validated online scorer. The queue and UI should not own thresholds, model state, or adaptation mathematics. Selected channel affects visualization only. Labels may appear only as a post-prediction overlay.

### Batch 8.2 — Queue and live controls

The producer and consumer should preserve point order, wait for a complete window, and support play, pause, resume, stop, speed, selected channel, and visible range controls.

### Primary files

- `demo/app.py`.
- `demo/online_replay.py`.
- `demo/stream_queue.py`.
- `demo/demo_state.py`.
- `demo/plotting.py`.
- Relevant tests under `tests/demo/`.

### Gate

Both demo modes use the shared label-free scorer, expose all required controls, and remain separate from official metric production.

### Engineering principles

Official evaluation and presentation remain separate. Dependency injection supplies the scorer and state. UI code cannot mutate protocol state directly.

## 14. Batch 9 — Readability and lifecycle refactor

### Purpose

Remove lifecycle mixins and hard size violations after semantic behavior is locked by tests.

### Batch 9.1 — Public THESIS model consolidation

The detailed plan should map every method currently distributed across:

- `thesis_multitask_setup_mixin.py`;
- `thesis_multitask_state_mixin.py`;
- `thesis_multitask_routing_mixin.py`;
- `thesis_multitask_loss_mixin.py`.

It should then group them into small composed responsibilities while keeping constructor, public inference/training methods, stage routing, and checkpoint contract visible from `thesis_multitask.py`. Helper modules may contain immutable configs and reusable primitives, but no second public model or hidden lifecycle.

### Batch 9.2 — Online model split

`online_adaptation.py` should remain the public entrypoint and expose the visible forward path. Checkpoint resolution, source adapter, residual projector, and diagnostics may be placed in focused modules only when the call chain remains explicit.

### Batch 9.3 — Online engine split

The detail plan should propose final files by responsibility, using existing modules where possible:

```text
online_engine.py          public orchestration facade
online_calibration.py     entity calibration orchestration
online_scoring.py         window and EWMA scoring orchestration
online_event_dispatch.py  A0/A1/A2 causal event strategies
online_execution.py       stream loop and coverage
online_reporting.py       report/checkpoint finalization
```

Names are provisional until the detail pass checks collisions and import direction. The structure requirement is more important than these exact filenames: one responsibility per module, no parallel engine, no circular imports.

### Batch 9.4 — Compliance closure

Refactor every remaining `src/` file above 500 lines and callable above 50 lines. Update `documents/abstract-design-notes/design_starter.md` only if the accepted project tree materially changes.

### Gate

Public imports, registry names, checkpoint keys, and deterministic outputs are preserved; zero source-size and callable-size violations remain.

### Engineering principles

Composition replaces inheritance. Single responsibility and dependency direction determine splits. Characterization tests precede every extraction.

## 15. Batch 10 — Preflight, CUDA smoke, and full matrix launch

### Purpose

Prove safety and correctness in the target environment before expensive execution.

### Batch 10.1 — No-train preflight

Extend the existing script rather than add a second preflight. Validate matrix counts, config semantics, dependencies, checkpoint/artifact hashes, CUDA requirement, device identity, seeds, data paths, output writeability, disk estimate, and run-directory collision behavior.

### Batch 10.2 — Ordered smoke evidence

```text
CPU no-train preflight
  -> O0 offline smoke
  -> O1 offline smoke
  -> O0-A0 GPU smoke
  -> O0-A2 GPU smoke
  -> O1-A0 GPU smoke
  -> O1-A2 GPU smoke
  -> interrupted/resumed A2 smoke
  -> artifact-path and checksum audit
```

The detailed plan should include exact commands only after inspecting every active CLI parser. It should specify tmux session naming, GPU-index assignment, evidence log location, and stop conditions.

### Batch 10.3 — Full matrix orchestration

The plan should lock the 18 THESIS offline, 54 THESIS online, 9 RedLamp, 27 traditional offline, and 81 online baseline main runs. Every online run must resolve its exact Stage-B checkpoint and entity artifact before launch. `--skip-completed` may skip only integrity-verified completion manifests.

### Primary files

- `scripts/preflight_full_benchmark_matrix.py`.
- Existing benchmark generators and wrappers.
- `tests/benchmarks/test_full_benchmark_matrix_preflight.py`.
- New `tests/benchmarks/test_full_spec_gpu_preflight.py`.
- Dated implementation evidence under `documents/logs/MM-DD-YYYY/detail/`.

### Gate

All CPU, test, compliance, GPU smoke, resume, and artifact gates pass before the first full shard starts. The aggregator reports every matrix cell as completed, failed, or missing; no silent omission is allowed.

### Engineering principles

Preflight is fail-fast and side-effect-free. Orchestration remains separate from model code. Completion is artifact-based, not exit-code-based.

## 16. Cross-batch dependency graph

The detail artifact should include this dependency graph and must not reorder batches across an unmet gate:

```text
Batch 0 contracts
  -> Batch 1 checkpoint + calibration
      -> Batch 2 one-window vertical slice
          -> Batch 3 triage + strategies
              -> Batch 4 verification + TTL
                  -> Batch 5 exact losses + update transaction
                      -> Batch 6 full stream + resume
                          -> Batch 7 reporting + W&B
                              -> Batch 8 demo
                                  -> Batch 9 readability refactor
                                      -> Batch 10 GPU + full matrix
```

Within a batch, tests should be added before behavior is modified. A failed gate stops later batches and produces a dated evidence note.

## 17. Cross-cutting test ladder

The detailed plan should assign every test to one rung:

```text
contract and shape tests
  -> pure calculation tests
      -> state lifecycle tests
          -> update/gradient integration tests
              -> causal stream and resume tests
                  -> artifact/demo tests
                      -> matrix preflight
                          -> CUDA smoke
                              -> full benchmark
```

Each batch should name focused `.venv/bin/python -m pytest` commands, followed by the cumulative semantic folder and full active suite when the change affects shared contracts.

## 18. Rollback boundaries

The detailed plan should define one rollback boundary per batch:

- Batch 0: test additions only.
- Batch 1: checkpoint schema version and calibration artifact version.
- Batch 2: one-window batch/model interface.
- Batch 3: triage and event dispatch.
- Batch 4: verification state schema.
- Batch 5: loss/optimizer transaction.
- Batch 6: configuration generation and runtime-state schema.
- Batch 7: report schema and W&B artifact boundary.
- Batch 8: demo callback and UI state boundary.
- Batch 9: one source-file extraction at a time.
- Batch 10: preflight and launch manifests; no model behavior change.

Rollback must preserve user changes and must not use destructive Git commands. If a schema migration has produced artifacts, the rollback instruction must state whether those artifacts remain readable, require regeneration, or must be marked incompatible.

## 19. Detailed-plan quality checklist

Before the future detail artifact is considered ready for implementation, it should answer all of the following:

1. What exact symbol changes in each edit?
2. What is the input/output tensor shape and semantic meaning?
3. Which object owns every mutable field?
4. What is detached, frozen, trainable, serialized, or recomputed?
5. What old config/checkpoint/report remains compatible?
6. What failing test is added before the change?
7. What focused and cumulative command verifies the change?
8. What evidence closes the batch gate?
9. What is the rollback boundary?
10. Does every function remain at most 50 lines and every source file at most 500 lines?
11. Can a high-school student follow the local and end-to-end flow from names, comments, and one small ASCII diagram?
12. Does any path accidentally use validation/test labels for calibration or adaptation?

## 20. Accepted structure review

The repository owner accepted eleven batches on 2026-07-11: contract freeze; checkpoint/calibration; minimal one-window slice; triage; verification; losses/update transaction; full stream/resume; artifact/report/W&B; demo; readability refactor; and GPU/full matrix.

The structure gate is closed and the document is accepted for expansion with `prompts/4_detail_prompt.md`. The two design decisions accepted in the source plan remain locked.
