---
date: 2026-07-15 16:32:50 +07
researcher: TheMetaSetter
git_commit: b9f98e329401455a97d981dff3a4eafe509f9d47
branch: dev
repository: bachelor-thesis-2026
topic: "Plan for checkpoint metadata and UQ persistence fixes"
tags: [plan, time-series, anomaly-detection, multi-class]
status: draft
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Plan: Checkpoint metadata and UQ persistence fixes

## Current State
- The repository already enforces the thesis-facing batch and model-output contracts.
- `src/core/config_model_validation.py` validates UQ configuration fields, but this is still input validation only.
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` already normalizes `verification_metadata_source` when memory is initialized.
- `src/engine/checkpoint.py` stores checkpoint metadata and checks config digests, but semantic non-empty validation is still limited.
- `src/core/contracts.py` validates the runtime `stochastic_query` and `uncertainty` schemas and tensor ranks.
- `src/engine/evaluator.py` constructs trace payloads that already contain `stochastic_query`, `uncertainty_history`, and `mc_sample_histories`, but the Stage-B benchmark outputs inspected on the remote host do not yet persist a dedicated trace artifact with the full payload.
- `src/protocols/threshold_artifact.py` validates threshold artifact provenance, but the online threshold artifact still needs a non-null checkpoint hash in the smoke output path.

## Design Options
- **Option A: Minimal vertical slice**
  - Add semantic validation helpers in the existing files.
  - Keep the full UQ trace export inside the evaluator and benchmark runner.
  - Preserve the current file layout and avoid introducing new abstractions unless a test gap forces it.
- **Option B: Dedicated validation helpers**
  - Add small validator classes or pure functions for checkpoint metadata, stochastic query payloads, uncertainty payloads, and threshold artifacts.
  - Keep the same runtime flow but centralize error messages and checks.
- **Option C: New trace artifact module**
  - Add a dedicated artifact writer for trace payloads and a manifest writer for those traces.
  - Use this only if the evaluator export path becomes too large to keep readable in one file.

## Risk and Mitigation
- **Risk: metadata exists but is semantically empty.**
  - Mitigation: fail closed when `verification_metadata_source` is placeholder-like or when mask/radii tensors are empty or invalid.
- **Risk: UQ fields exist in memory but are not persisted.**
  - Mitigation: export a dedicated trace artifact for Stage B and online evaluation, then checksum it in a manifest.
- **Risk: trace export becomes too large for one file.**
  - Mitigation: keep the first patch minimal, and only extract a helper module if the evaluator file grows beyond the readability gate.
- **Risk: online provenance remains incomplete.**
  - Mitigation: require `threshold_artifact.checkpoint_sha256` before accepting an online run as provenance-correct.
- **Risk: tests pass on schema but not on meaning.**
  - Mitigation: add semantic tests for non-empty metadata, finite tensors, sample-count consistency, and retention-policy consistency.

## Open Questions
- Should trace payloads be saved as JSON only, or split into JSON metadata plus compressed tensor payloads?
- Should Stage B and online use the same trace artifact schema, or should online keep a smaller stream-oriented subset?
- Should `verification_metadata_source` be rewritten only at save time, or also on load before any downstream verification logic runs?

## Implementation Plan

### Phase 1: Lock metadata semantics in checkpoint save/load
1. Extend `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` with explicit semantic checks before checkpoint serialization.
2. Keep the current provenance rewrite logic, but reject placeholder provenance when memory is initialized and verification tensors are present.
3. Add a small helper that checks the following conditions:
   - `memory_initialized == True`
   - `verification_metadata_source != "uninitialized"`
   - `anomalous_codeword_mask` is non-empty and has the expected shape
   - `anomaly_radii` is finite and non-negative
   - `verification_codeword_class_ids` and `verification_contributing_token_counts` match the codebook shape
4. Add a checkpoint round-trip test that proves the rewritten provenance survives save/load.

### Phase 2: Enforce runtime UQ semantics
1. Tighten `src/core/contracts.py` so that `stochastic_query` and `uncertainty` are not only schema-valid, but also semantically consistent.
2. Add checks for:
   - `stochastic_query["num_samples"] == model.monte_carlo_samples`
   - all sample tensors have the expected rank
   - variance tensors are finite
   - sample payloads are present whenever retention policy requires them
3. Add focused tests for `stochastic_query` and `uncertainty` validation.

### Phase 3: Persist full Stage-B trace payloads
1. Extend `src/engine/evaluator.py` so the trace payload is written to a dedicated artifact file instead of remaining only in memory or in summary records.
2. Persist at least:
   - `stochastic_query`
   - `uncertainty_history`
   - `mc_sample_histories`
   - `sample_retention_policy`
3. Add a checksum manifest for the new trace artifact so that later audits can detect partial or stale exports.
4. Keep the first version JSON-friendly and simple; only introduce compressed tensor storage if the JSON payload becomes too large.

### Phase 4: Fix threshold provenance in online output
1. Extend `src/protocols/threshold_artifact.py` validation so the online threshold artifact must carry a real checkpoint hash when the source checkpoint is known.
2. Update `src/engine/online_tta/online_engine_run.py` and `scripts/benchmarks/run_thesis_online_benchmark.py` so the online threshold artifact is built with a non-null `checkpoint_sha256`.
3. Add a manifest-level test that checks:
   - `checkpoint_sha256`
   - `resolved_config_sha256`
   - `threshold_artifact_sha256`
   - identity fields

### Phase 5: Validate end-to-end on one representative combination
1. Run one representative development-spec combination first.
2. Verify the Stage A checkpoint, Stage B checkpoint, Stage-B trace artifact, and online threshold artifact in one continuous pass.
3. Only after that pass should the full combination matrix be rerun.

## File Targets
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py`
- `src/core/contracts.py`
- `src/engine/checkpoint.py`
- `src/engine/evaluator.py`
- `src/protocols/threshold_artifact.py`
- `src/engine/online_tta/online_engine_run.py`
- `scripts/benchmarks/run_thesis_online_benchmark.py`
- `tests/runtime/test_checkpoint_roundtrip.py`
- `tests/online/test_online_prototype_metadata_contract.py`

## Validation Procedure
- Run checkpoint round-trip tests first.
- Run UQ contract tests second.
- Run a single remote smoke/full-flow representative combination third.
- Verify the Stage-B trace artifact on disk, not only the console output.
- Verify the online threshold artifact contains a real checkpoint hash.

## Draft Decision
- Keep the first patch minimal.
- Prefer semantic validation over new abstraction layers.
- Add a new trace artifact only if the evaluator export path cannot remain readable.
