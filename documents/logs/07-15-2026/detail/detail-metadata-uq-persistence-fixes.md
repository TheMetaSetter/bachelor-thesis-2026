---
date: 2026-07-15 16:32:50 +07
researcher: TheMetaSetter
git_commit: b9f98e329401455a97d981dff3a4eafe509f9d47
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed plan for checkpoint metadata and UQ persistence fixes"
tags: [detail, time-series, anomaly-detection, multi-class]
status: draft
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Detailed Plan: Checkpoint metadata and UQ persistence fixes

## Overview
This implementation plan strengthens the thesis pipeline at four concrete seams: checkpoint provenance, runtime UQ contract enforcement, Stage-B trace persistence, and online threshold provenance. The work should preserve the current repository structure, respect the one-model-one-file rule, and keep the first patch minimal enough to validate on one representative combination before any scale-out run.

## Phase 1: Semantic checkpoint provenance hardening

### Phase summary
This phase ensures that a checkpoint does not merely contain provenance keys, but contains provenance that is semantically meaningful when prototype memory is initialized. The objective is to prevent Stage B and downstream online verification from accepting placeholder metadata that cannot prove how the verification tensors were produced.

### File-level edits
- `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py`
  - Extend `get_checkpoint_extra_state()` with explicit semantic guards before serialization.
  - Keep `_normalize_verification_metadata_source()` as the canonical rewrite point for placeholder provenance.
  - Add a small validation helper for verification metadata consistency.
- `src/engine/checkpoint.py`
  - Keep checkpoint metadata construction and digest validation unchanged in structure.
  - Tighten the checkpoint metadata validation path so it fails closed on placeholder provenance that reaches the checkpoint boundary.
- `tests/runtime/test_checkpoint_roundtrip.py`
  - Add or extend round-trip tests for rewritten provenance and non-empty verification tensors.

### Interface and contract definitions
- The dataset contract remains unchanged: batches still expose `x: Tensor[B, L, D]`, optional labels, and metadata.
- The encoder contract remains unchanged: all thesis-facing encoders still expose `hidden: Tensor[B, L, H]`.
- The model output contract remains unchanged: the model still returns `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- The checkpoint contract is strengthened: when `memory_initialized = True`, the checkpoint metadata must also prove that `verification_metadata_source` is not a placeholder and that the verification tensors are materially present.

### Design pattern application
- Use composition over inheritance by keeping the validation logic inside the existing serialization mixin rather than splitting checkpoint policy into a new class hierarchy.
- Keep the checkpoint boundary as a stable interface and do not leak checkpoint-specific checks into the trainer loop.

### Risk mitigation
- Reject placeholder provenance values such as `""`, `"uninitialized"`, and `"disabled"` whenever memory is active.
- Reject empty or shape-incompatible verification tensors.
- Reject radii tensors that are non-finite or negative.

### Test plan
- Run a checkpoint save/load round-trip test.
- Verify that provenance is rewritten on save only when memory is initialized and verification tensors are present.
- Verify that the saved checkpoint can be loaded back without losing the rewritten provenance.

### Acceptance criteria
- A Stage B checkpoint with initialized memory must save with a non-placeholder `verification_metadata_source`.
- The checkpoint must fail closed if verification tensors are missing, empty, or inconsistent.
- The round-trip test must pass without relaxing the checkpoint contract.

## Phase 2: Runtime UQ contract enforcement

### Phase summary
This phase ensures that runtime UQ fields are not only schema-valid, but also semantically consistent with the model configuration and the Monte Carlo sampling policy. The objective is to make `stochastic_query` and `uncertainty` meaningful evidence, not merely present dictionaries.

### File-level edits
- `src/core/contracts.py`
  - Tighten `validate_stochastic_query_aux()` with semantic checks that go beyond field presence and tensor rank.
  - Tighten `validate_uncertainty_aux()` with finiteness and consistency checks.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`
  - Keep the trace payload construction stable.
  - Ensure the runtime UQ payload reflects the actual sample count and retention policy.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`
  - Keep the top-level `aux` composition stable and continue exposing `stochastic_query` and `uncertainty` in the same place.
- `tests/online/test_online_prototype_metadata_contract.py`
  - Add targeted runtime tests for stochastic query and uncertainty validation.

### Interface and contract definitions
- `stochastic_query` must continue to expose:
  - `schema_version`
  - `enabled`
  - `num_samples`
  - temperatures
  - sample tensors
  - retention metadata
- `uncertainty` must continue to expose variance summaries for point, window, reconstruction, and classification paths.
- The semantics must additionally enforce:
  - `num_samples == model.monte_carlo_samples`
  - sample tensors are present when the retention policy requires them
  - variance tensors are finite

### Design pattern application
- Use the strategy pattern implicitly through config-controlled validation rules rather than hard-coding a single runtime policy.
- Keep validation close to the output contract so the model remains the owner of its own semantics.

### Risk mitigation
- Prevent silent acceptance of missing Monte Carlo payloads.
- Prevent acceptance of sample-count mismatches between config and runtime.
- Prevent acceptance of variance tensors that are present but numerically meaningless.

### Test plan
- Validate one known-good `stochastic_query` payload.
- Validate one intentionally broken payload with wrong sample count.
- Validate one broken payload with non-finite or empty variance tensors.

### Acceptance criteria
- The runtime contract rejects semantically incomplete UQ payloads.
- The runtime contract still accepts valid smoke-run payloads without introducing false positives.
- The validation error messages identify the exact field that failed.

## Phase 3: Stage-B trace persistence

### Phase summary
This phase persists the full Stage-B trace payload to disk so that UQ evidence survives after the process exits. The objective is to move from summary-only output to durable trace artifacts that can be inspected later without rerunning the benchmark.

### File-level edits
- `src/engine/evaluator.py`
  - Extend the trace payload export path so it writes a dedicated artifact file.
  - Persist `stochastic_query`, `uncertainty_history`, `mc_sample_histories`, and `sample_retention_policy`.
- `scripts/benchmarks/run_thesis_offline_benchmark.py`
  - Add artifact writing or bundling hooks for Stage-B trace export if the evaluator does not already write them.
  - Keep summary metrics and trace payloads separate so the benchmark report remains readable.
- `src/core/artifact_integrity.py`
  - Reuse manifest logic for the new trace artifact if the trace file must be checksum-protected.
- `tests/runtime/test_checkpoint_roundtrip.py`
  - Add a test that confirms the trace artifact path is written and non-empty after Stage-B evaluation.

### Interface and contract definitions
- The evaluation record schema remains JSON-friendly and continues to summarize point scores and labels.
- The new trace artifact schema should preserve the runtime payload fields that the inventory lists as required UQ evidence.
- The benchmark report contract remains summary-oriented; the trace artifact becomes the durable payload layer beneath it.

### Design pattern application
- Use composition over inheritance by adding artifact export as a small extension to the evaluator and benchmark runner, not as a new framework.
- Keep the artifact writer simple and data-oriented, with no model-specific branching outside the existing output contract.

### Risk mitigation
- Start with JSON-friendly trace export so the first patch remains easy to inspect.
- Use a manifest checksum so stale or partial trace files are detectable.
- Avoid over-fragmenting the trace schema before the persistence path is proven.

### Test plan
- Run Stage-B evaluation on the smoke combination.
- Confirm that the trace artifact exists on disk.
- Confirm that the trace artifact contains the expected top-level UQ fields.
- Confirm that the artifact is included in a checksum manifest.

### Acceptance criteria
- Stage-B output must include a durable trace artifact, not only summary files.
- The trace artifact must contain the UQ fields listed in the inventory.
- The artifact must survive the end of the run and be auditable by checksum.

## Phase 4: Online threshold provenance repair

### Phase summary
This phase repairs the online path so the threshold artifact carries a real checkpoint hash and a complete provenance chain. The objective is to make online benchmark outputs provenance-complete enough for strict verification, not only runnable.

### File-level edits
- `src/protocols/threshold_artifact.py`
  - Tighten validation so the provenance block remains structurally complete.
  - Keep `checkpoint_sha256` and `resolved_config_sha256` as first-class provenance fields.
- `src/engine/online_tta/online_engine_run.py`
  - Ensure the threshold artifact passed into the online checkpoint has a real `checkpoint_sha256` when the source checkpoint is known.
  - Keep the online execution report and artifact manifest synchronized.
- `scripts/benchmarks/run_thesis_online_benchmark.py`
  - Make sure the online benchmark writer does not emit a threshold artifact with a null checkpoint hash when the checkpoint is available.
- `tests/online/test_online_prototype_metadata_contract.py`
  - Add a manifest-level provenance test for the online path.

### Interface and contract definitions
- The threshold artifact contract remains a JSON object with `thresholds`, provenance, and digest fields.
- The online checkpoint contract continues to store runtime state in `extra_state`, but the threshold artifact embedded there must be provenance-complete.
- The online artifact manifest must continue to map artifact names to paths and checksums.

### Design pattern application
- Use the registry/factory style already present in the benchmark path to keep threshold artifact construction centralized.
- Keep the online runtime projector-first and conservative; do not broaden adaptation scope as part of this fix.

### Risk mitigation
- Fail closed if the threshold artifact has a missing checkpoint hash where the checkpoint source is known.
- Keep the online provenance repair separate from any adaptation logic so debugging remains localized.

### Test plan
- Run one online smoke job after the Stage-B checkpoint is available.
- Inspect the threshold artifact JSON and confirm that the checkpoint hash is present and stable.
- Verify that the online artifact manifest checksum still matches the generated files.

### Acceptance criteria
- The online threshold artifact must contain a real checkpoint hash when the source checkpoint is available.
- The online artifact manifest must remain internally consistent.
- The online run must remain conservative and reproducible.

## Phase 5: End-to-end validation on one representative combination

### Phase summary
This phase validates the full flow on one representative development-spec combination before any batch-wide execution. The objective is to prevent a large-scale rerun from failing late because of a provenance, trace, or threshold artifact bug.

### File-level edits
- No additional production edits are required in this phase unless the earlier tests reveal a new contract gap.
- If a gap appears, fix only the smallest affected file first and re-run the same representative combination.

### Interface and contract definitions
- The dataset contract must remain stable.
- The encoder contract must remain stable.
- The model output contract must remain stable.
- The checkpoint provenance contract must be observable in the saved Stage A and Stage B checkpoints.
- The trace artifact contract must be observable in the Stage-B output tree.
- The online threshold provenance contract must be observable in the online output tree.

### Design pattern application
- Keep the validation sequence linear and observable.
- Use the same representative combination for every first-pass verification so the debugging surface remains small.

### Risk mitigation
- Validate exactly one combination first, as already established in the repository practice.
- Do not scale to all combinations until the one-combination path passes.
- Avoid broad cleanup or destructive changes during the verification pass.

### Test plan
- Run Stage A on one representative combination.
- Run Stage B on the same combination.
- Inspect the Stage-B checkpoint and trace artifact on disk.
- Run the online benchmark on the same combination.
- Inspect the online checkpoint, threshold artifact, and artifact manifest.

### Acceptance criteria
- The one representative combination must pass from Stage A through online output without provenance gaps.
- The Stage-B trace artifact must be present and non-empty.
- The online threshold artifact must include a non-null checkpoint hash.
- Only after this pass may the repository scale to the full combination matrix.

## Cross-phase engineering constraints
- Preserve the one-model-one-file rule.
- Prefer composition over inheritance.
- Keep the encoder adapter boundary explicit if any model wrapper needs to normalize outputs.
- Keep dataset and model registry behavior stable.
- Keep the first patch minimal enough to debug on a single remote smoke/full-flow pass.

## Final delivery order
1. Implement checkpoint semantic hardening.
2. Implement runtime UQ validation.
3. Persist the Stage-B trace artifact.
4. Repair online threshold provenance.
5. Run the one-combination validation pass.
6. Expand to the full benchmark matrix only after the first pass succeeds.
