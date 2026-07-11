---
date: 2026-07-11T23:02:58+07:00
planner: Codex
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for complete full-spec-v3 experiment readiness"
tags: [detail, full-spec-v3, offline, stochastic-retrieval, online-tta, benchmark, demo, reproducibility, post-experiment-eda]
status: draft
source_plan: documents/logs/07-11-2026/plan/plan-full-spec-v3-experiment-readiness.md
source_structure: documents/logs/07-11-2026/structure/structure-full-spec-v3-experiment-readiness.md
last_updated: 2026-07-11
last_updated_by: Codex
---

# Detailed programming plan: complete `full-spec-v3` experiment readiness

## 0. Implementation contract

This document is the implementation contract for bringing the active THESIS codebase into compliance with `documents/spec/full-spec-v3.md`. It follows the planning structure already accepted for this repository and expands it into an executable sequence of batches.

The work must preserve the repository’s fixed engineering rules:

- one public THESIS entrypoint per model;
- composition over inheritance;
- explicit batch and output contracts;
- registry-driven dataset and model construction;
- short functions, small files, and readable names;
- minimal codepaths with configuration-driven ablation;
- richer-than-minimal artifact retention for post-experiment EDA.

Implementation begins only after the repository owner explicitly requests coding. Every batch below is intended to be independently testable and reversible. A later batch must not begin until the current batch gate passes.

The accepted execution order contains eleven batches:

```text
0 contracts and provenance
  -> 1 stochastic query operators
  -> 2 Monte Carlo aggregation
  -> 3 deterministic geometry and evaluation
  -> 4 online adapter and causal runtime state
  -> 5 verification and projector safeguards
  -> 6 calibration and threshold artifacts
  -> 7 benchmark and demo export parity
  -> 8 post-experiment EDA retention
  -> 9 tests and validation hardening
  -> 10 readability and file-size remediation
```

This order is the smallest safe vertical slice because later stages consume sample tensors, variance summaries, threshold artifacts, verification metadata, and runtime state produced by earlier stages.

## 1. Locked scientific and runtime semantics

The following decisions are fixed for this implementation:

1. The public THESIS offline entrypoint remains `ThesisMultitaskModel`.
2. The public online adaptation entrypoint remains `OnlineAdaptationModel`.
3. Online input is one window `x: FloatTensor[B, 20, D]`.
4. Stochastic retrieval uses exactly ten inference samples in validation, calibration, test, and official online scoring.
5. Deterministic geometry must remain deterministic; stochastic samples must not be used for verification signatures or anomaly filtering.
6. Gray-zone admission must remain label-free at runtime.
7. Only the projector or residual adapter may mutate online.
8. The stable top-level model output contract remains `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
9. `aux` is the home for sample tensors, uncertainty summaries, deterministic geometry traces, and provenance metadata.
10. The repository should retain enough raw traces to support post-experiment EDA without rerunning the trial.

These semantics are not optional implementation preferences. They are acceptance-level constraints.

## 2. Stable cross-layer contracts

### 2.1 Dataset and batch contract

`src/core/contracts.py` remains the canonical guard for the shared batch shape. The active batch must remain:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The active online path must accept a single causal window only. Legacy two-view validation may remain as a named historical validator, but it must not be selected by any active v3 config.

Dataset-related implementation details should stay inside the data layer:

- `src/data/datasets/smd.py` for raw SMD parsing;
- `src/data/loaders.py` for train-only scaling and window assembly;
- `src/data/window.py` or the current windowing helper for stride handling;
- `src/data/collate.py` for canonical batch collation;
- `src/data/augment.py` for synthetic anomaly injection;
- `src/core/registry.py` for dataset construction.

### 2.2 Encoder and source-adapter contract

`src/models/online_adaptation.py::ThesisMultitaskEncoderAdapter` remains a composition-based adapter around `ThesisMultitaskModel`. Its public operations should stay narrow and explicit:

```python
encode_source(batch) -> FloatTensor[B, L, H]
score_source(source_hidden, x) -> ModelOutputs
score_projected(projected_hidden, x) -> ModelOutputs
prototype_verification_metadata() -> PrototypeVerificationMetadata
```

The adapter must not own optimizer state, stream state, or checkpoint orchestration. It may freeze the source model and expose detached tensors, but it must not hide phase-specific logic behind inheritance.

### 2.3 Model output contract

The public model output should remain:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "recon": Tensor[B, L, D],
    "logits": Tensor[B, C],
    "point_scores": Tensor[B, L],
    "window_scores": Tensor[B],
    "aux": dict,
}
```

For v3, `aux` must contain:

- stochastic sample tensors;
- uncertainty tensors;
- deterministic geometry tensors;
- provenance and schema metadata;
- optional retention artifacts for post-experiment EDA.

The contract must stay top-level stable. Stochastic outputs must not introduce a separate top-level Monte Carlo dimension.

### 2.4 Task strategy contract

The online task strategy should remain explicit and shallow:

```text
A0 -> score only
A1 -> verified-PNN update with masked reconstruction
A2 -> accepted hard-old update or verified-PNN update with exact reconstruction + contrastive regularization
```

This strategy should be represented by explicit callables or small immutable strategy objects, not by a new deep inheritance tree.

### 2.5 Threshold and checkpoint contract

`src/protocols/threshold_artifact.py` and `src/engine/checkpoint.py` must carry explicit provenance:

- schema version;
- entity identity;
- checkpoint SHA-256;
- resolved config SHA-256;
- sample count;
- correction mode;
- calibration split;
- stride settings;
- source provenance for anomalous-codeword metadata;
- runtime schema version.

Calibration must never silently mix offline non-overlap scores and online stride-1 scores.

## 3. Batch 0 — contracts, provenance, and configuration boundary

### 3.1 Summary

This batch locks the v3 schema before stochastic behavior is expanded. The goal is to make every relevant configuration and artifact field explicit so that later implementation steps cannot drift silently.

### 3.2 File-level edits

Modify these files first:

- `src/core/contracts.py`
- `src/protocols/threshold_artifact.py`
- `src/engine/checkpoint.py`
- `src/core/artifact_integrity.py`
- the active v3 config files under the repository’s config tree

### 3.3 Explicit edit content

1. Extend the model output and checkpoint validation logic so that v3 metadata fields are accepted and checked, not merely stored.
2. Add explicit config keys for:
   - `stochastic_inference`
   - `monte_carlo_samples`
   - `continuous_temperature`
   - `discrete_temperature`
   - `variance_correction`
   - `return_mc_samples`
   - `sample_retention_policy`
3. Ensure the threshold artifact records the Monte Carlo settings and calibration provenance.
4. Keep the stable batch keys unchanged.
5. Keep the current model registry and dataset registry names unchanged.

### 3.4 Interface and contract definitions

The contracts in this batch should define:

- a shared batch contract for datasets and engines;
- an online one-window batch contract;
- the model output contract with a stable `aux`;
- a threshold artifact schema with strict provenance fields;
- a checkpoint schema that can validate whether the stored memory metadata matches the runtime request.

### 3.5 Design pattern application

Use registry-driven configuration resolution for datasets, models, and artifacts. Keep validation in the contract layer and keep the model entrypoints free from artifact parsing logic. This preserves composition over inheritance and keeps the public model surface single and readable.

### 3.6 Risk mitigation

- Prototype redundancy risk: keep branch-specific config keys and separate logging for continuous versus discrete retrieval.
- Evaluation inflation risk: make score reduction and variance correction explicit in the artifact schema.
- Threshold misuse risk: reject mixed-protocol artifacts during calibration startup.

### 3.7 Test plan and validation

Add or extend tests for:

- config parsing and resolved schema fields;
- threshold artifact round-trip and provenance validation;
- checkpoint metadata round-trip;
- contract enforcement for batch and output keys.

### 3.8 Acceptance criteria

This batch is complete only when the repository can:

- load a v3 config with explicit Monte Carlo settings;
- save and reload threshold artifacts with schema-versioned provenance;
- fail closed on missing or inconsistent calibration metadata;
- preserve the existing batch and output vocabulary.

## 4. Batch 1 — vectorized stochastic query operators

### 4.1 Summary

This batch implements the stochastic retrieval core without duplicating encoder work. The goal is to make the Monte Carlo query path explicit, vectorized, and isolated from deterministic geometry.

### 4.2 File-level edits

Modify these files:

- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_components.py`
- `src/models/thesis_multitask_state_mixin.py`
- `src/models/thesis_multitask_setup_mixin.py`

### 4.3 Explicit edit content

1. Add a compact helper for vectorized stochastic retrieval over:
   - continuous prototypes;
   - discrete codewords.
2. Precompute similarity logits once and reuse them across the Monte Carlo sample dimension.
3. Keep continuous dense retrieval and discrete top-k retrieval separate in code and in naming.
4. Preserve existing `cosine_topk` and `gumbel_softmax` semantics, but make the stochastic path explicit in the forward flow.
5. Keep deterministic geometry helpers out of the stochastic helper.

### 4.4 Interface and contract definitions

The retrieval helper should expose a small interface such as:

```python
build_stochastic_queries(hidden) -> QueryBundle
sample_continuous_retrieval(query_bundle, num_samples) -> Tensor[B, M, L, H]
sample_discrete_retrieval(query_bundle, num_samples) -> Tensor[B, M, L, H]
sample_discrete_topk_ids(query_bundle, num_samples) -> LongTensor[B, M, L, 3]
```

The helper should not compute losses, thresholds, or runtime state.

### 4.5 Design pattern application

Use composition for the retrieval helper and keep the encoder contract isolated. The retrieval logic should behave like a strategy component selected by config, not as a hidden branch inside a large monolithic forward function.

### 4.6 Risk mitigation

- Prototype redundancy risk: log continuous and discrete sample tensors separately.
- Fusion collapse risk: keep both branch outputs visible before fusion, even if the top-level output only consumes the fused result.
- Numerical instability risk: test extreme pseudo-random inputs to confirm finite Gumbel samples and normalized weights.

### 4.7 Test plan and validation

Add tests for:

- retrieval tensor shapes `[B, M, L, H]`;
- top-k ids shape `[B, M, L, 3]`;
- normalized weights summing to one;
- finite Gumbel outputs under extreme sampled uniforms;
- encoder computation not repeated across samples.

### 4.8 Acceptance criteria

This batch is complete only when the retrieval path:

- performs one encoder pass per window;
- expands only over Monte Carlo samples after similarity precomputation;
- returns deterministic tensor shapes for both branches;
- remains independent from verification geometry.

## 5. Batch 2 — Monte Carlo aggregation and uncertainty schema

### 5.1 Summary

This batch makes the stochastic forward path scientifically usable. The goal is to return mean predictions and uncertainty summaries from exactly ten samples while keeping the stable top-level contract intact.

### 5.2 File-level edits

Modify these files:

- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_loss_mixin.py`
- `src/core/contracts.py`
- `src/engine/evaluator.py`

### 5.3 Explicit edit content

1. Aggregate `recon`, `logits`, `point_scores`, and `window_scores` as Monte Carlo means.
2. Compute unbiased variance for:
   - point anomaly score;
   - window anomaly score;
   - reconstruction;
   - continuous retrieval;
   - discrete retrieval;
   - classification probabilities.
3. Keep the sample tensors under `aux`.
4. Preserve the stable output vocabulary.
5. Avoid recomputing the mean score from the mean reconstruction; mean-of-samples and reconstruction-of-mean are not interchangeable.

### 5.4 Interface and contract definitions

`aux` should contain structured subfields such as:

```python
aux["stochastic_query"]
aux["uncertainty"]
aux["deterministic_geometry"]
aux["retention"]
```

The exact internal nesting may be adjusted for readability, but the semantics must remain separated.

### 5.5 Design pattern application

Keep aggregation as a pure post-processing stage after retrieval. This is a composition-friendly design because the model can reuse the same aggregation utilities for offline evaluation, calibration, and online scoring.

### 5.6 Risk mitigation

- Evaluation metric inflation risk: define the official mean score as the mean over sample-wise scores and make the correction mode explicit.
- Threshold contamination risk: never allow variance to alter the thresholding rule unless a separate ablation is created.
- Serialization risk: keep large sample tensors optional and versioned.

### 5.7 Test plan and validation

Add or extend tests for:

- exact sample count `M = 10` in evaluation mode;
- sample-mean correctness for reconstruction and point scores;
- unbiased variance with `correction=1` or equivalent;
- `log(mean_probability)` compatibility behavior for logits;
- `aux` schema round-trip.

### 5.8 Acceptance criteria

This batch is complete only when:

- the forward path returns means and variances from exactly ten samples;
- top-level output keys remain unchanged;
- stochastic sample tensors are preserved under `aux`;
- no public contract exposes an extra Monte Carlo leading dimension.

## 6. Batch 3 — deterministic geometry, offline training, and uncertainty-aware evaluation

### 6.1 Summary

This batch preserves the deterministic memory geometry and aligns offline training and evaluation with the stochastic v3 protocol. The goal is to keep the model file self-contained while letting the evaluator remain the place where metrics are computed and written.

### 6.2 File-level edits

Modify these files:

- `src/models/thesis_multitask_state_mixin.py`
- `src/engine/online_tta/signature_verification.py`
- `src/engine/evaluator.py`
- `scripts/run_thesis_offline_benchmark.py`

### 6.3 Explicit edit content

1. Keep nearest-codeword filtering deterministic.
2. Keep continuous signature extraction deterministic and ordered.
3. Store provenance for anomalous-codeword metadata in the checkpoint.
4. Align offline evaluation with the official ten-sample stochastic path.
5. Export uncertainty and geometry traces alongside metrics.
6. Preserve the Stage A / Stage B lifecycle and frozen memory semantics.

### 6.4 Interface and contract definitions

The verification metadata interface should include:

- anomalous codeword mask;
- codeword radii;
- codeword class ids;
- contributing token counts;
- source split;
- schema version;
- initialization seed.

The evaluator should distinguish between:

- metric computation;
- artifact export;
- trace retention.

### 6.5 Design pattern application

Use adapter-style boundaries between model memory metadata and verification logic. Keep the evaluator as a pure consumer of predictions and labels, not as a hidden computation owner.

### 6.6 Risk mitigation

- Adaptation contamination risk: ensure deterministic geometry uses frozen source latents, not mutable projected latents.
- Prototype redundancy risk: preserve separate logging of deterministic geometry and stochastic retrieval.
- Metric inflation risk: keep threshold setting and final metric computation separable.

### 6.7 Test plan and validation

Add tests for:

- round-trip of verification metadata;
- deterministic nearest codeword IDs;
- deterministic continuous signatures;
- metric exports containing uncertainty fields;
- Stage-B memory provenance rejection when metadata is missing or malformed.

### 6.8 Acceptance criteria

This batch is complete only when:

- deterministic geometry is still reproducible across runs;
- offline evaluation exports uncertainty and geometry traces;
- Stage-B metadata provenance is validated before online use.

## 7. Batch 4 — online adapter and causal runtime state

### 7.1 Summary

This batch implements the causal online path around the frozen source encoder and residual projector. The goal is to keep online execution label-free, resumable, and scientifically inspectable.

### 7.2 File-level edits

Modify these files:

- `src/models/online_adaptation.py`
- `src/engine/online_tta/runtime_state.py`
- `src/engine/online_tta/online_engine.py`
- `src/engine/online_tta/online_calibration.py`

### 7.3 Explicit edit content

1. Keep the source encoder frozen and run it once per window.
2. Make the projector near-identity and residual by default.
3. Keep `A0`, `A1`, and `A2` as explicit strategy routes.
4. Make the runtime state own cursor, EWMA state, verification history, recurrent signatures, and hard-old intervals.
5. Ensure online startup validates checkpoint, entity identity, threshold artifact, and runtime schema before reading the stream.

### 7.4 Interface and contract definitions

The adapter should expose:

```python
encode_source(batch) -> hidden
score_source(hidden, x) -> outputs
score_projected(projected_hidden, x) -> outputs
prototype_verification_metadata() -> metadata
```

The runtime state should expose small, explicit restore/save operations. It should not hide optimizer moments or future labels.

### 7.5 Design pattern application

Use composition over inheritance. The adapter composes the offline model, the online engine composes the adapter, and the runtime state composes the mutable online state. Strategy dispatch should remain explicit, with a small callable per variant.

### 7.6 Risk mitigation

- Projector drift risk: use residual initialization, warm-starting, and anchor regularization where needed.
- High-variance updates risk: keep online parameter updates conservative and projector-local.
- Adaptation contamination risk: keep runtime scoring and update decisions label-free.

### 7.7 Test plan and validation

Add tests for:

- exactly one source encode per window;
- A0 bypassing projector mutation;
- A1/A2 using projected hidden states;
- runtime state save/load round-trip;
- resume-safe entity-scoped state restoration.

### 7.8 Acceptance criteria

This batch is complete only when:

- online scoring is causal and one-window based;
- source encoding is executed once per window;
- runtime state can be resumed without hidden optimizer carryover.

## 8. Batch 5 — verification and projector safeguards

### 8.1 Summary

This batch makes verification independent, label-free, and contamination-resistant. The goal is to preserve the exact gray-zone verification semantics and the projector-only adaptation boundary.

### 8.2 File-level edits

Modify these files:

- `src/engine/online_tta/verification_buffer.py`
- `src/engine/online_tta/verification_adapter.py`
- `src/engine/online_tta/signature_verification.py`
- `src/engine/online_tta/online_losses.py`

### 8.3 Explicit edit content

1. Keep the verification buffer non-overlapping, TTL-based, and event-driven.
2. Build entry batches from one causal window only.
3. Verify using frozen source latents, not mutable projected latents.
4. Keep recurrent signatures deterministic and ordered.
5. Implement online adaptation losses in a way that preserves the exact anchor/negative semantics for hard-old and verified-PNN updates.

### 8.4 Interface and contract definitions

The verification buffer should own:

- entry list;
- TTL remaining;
- admission status;
- new-admission flags;
- removal state.

The verification cycle should own:

- trigger eligibility;
- batch reconstruction for verification;
- outcome finalization;
- buffer mutation only after success.

### 8.5 Design pattern application

Use explicit callable helpers for verification admission and verification cycle execution. This is a strategy-like boundary, but it should remain small and concrete rather than abstract-heavy.

### 8.6 Risk mitigation

- Adaptation contamination risk: reject any verification path that consults labels.
- Metric inflation risk: keep verification outputs separate from metric outputs.
- Deterministic geometry leakage risk: forbid stochastic retrieval ids from entering signature logic.

### 8.7 Test plan and validation

Add tests for:

- buffer capacity and TTL semantics;
- non-overlap admission;
- independent frozen-source verification forward call count;
- ordered signature recurrence;
- rejection of stochastic ids in signatures;
- label absence during verification.

### 8.8 Acceptance criteria

This batch is complete only when:

- gray-zone verification is independent and label-free;
- projector updates occur only after allowed verification or accepted hard-old conditions;
- buffer, TTL, and cycle semantics remain deterministic.

## 9. Batch 6 — calibration and threshold artifacts

### 9.1 Summary

This batch separates offline and online calibration ownership and makes the threshold artifact scientifically transparent. The goal is to prevent accidental score-timeline mixing.

### 9.2 File-level edits

Modify these files:

- `src/engine/online_tta/online_calibration.py`
- `src/engine/thresholding.py`
- `src/protocols/threshold_artifact.py`
- `src/core/artifact_integrity.py`

### 9.3 Explicit edit content

1. Keep offline calibration on non-overlapping windows.
2. Keep online calibration on stride-1 causal windows.
3. Persist Monte Carlo sample count, variance correction, and score reduction mode in the artifact.
4. Fail closed on missing provenance.
5. Keep artifact integrity checks SHA-256 based.

### 9.4 Interface and contract definitions

The calibration helpers should expose small pure functions such as:

```python
collect_nonoverlap_offline_scores(...)
collect_stride1_online_scores(...)
calibrate_entity_threshold_artifact(...)
validate_threshold_artifact(...)
```

Each function should remain short and testable.

### 9.5 Design pattern application

Use a single calibration boundary with pure collectors underneath it. This keeps the artifact schema stable and the orchestration readable.

### 9.6 Risk mitigation

- Evaluation metric inflation risk: do not reuse one-sample thresholds for ten-sample means.
- Calibration contamination risk: keep validation labels and test labels out of threshold creation.
- Artifact drift risk: include config hashes and checkpoint hashes in every artifact.

### 9.7 Test plan and validation

Add tests for:

- exact stride handling in offline and online score collection;
- artifact provenance validation;
- rejection of mismatched sample count or correction mode;
- checksum and identity validation.

### 9.8 Acceptance criteria

This batch is complete only when:

- offline and online threshold artifacts are independently valid;
- artifact provenance is complete and reproducible;
- calibration does not depend on test-time information.

## 10. Batch 7 — benchmark and demo export parity

### 10.1 Summary

This batch aligns the benchmark wrappers and demo surfaces with the same stochastic and deterministic contract. The goal is to make the demo tell the same scientific story as the benchmark.

### 10.2 File-level edits

Modify these files:

- `scripts/run_thesis_offline_benchmark.py`
- `scripts/run_thesis_online_benchmark.py`
- `demo/app.py`
- `demo/online_replay.py`
- `src/engine/evaluator.py`

### 10.3 Explicit edit content

1. Export sample-level traces, uncertainty traces, deterministic geometry traces, and threshold provenance.
2. Preserve the current benchmark matrix and result keys.
3. Ensure the demo can render the same core traces that the benchmark writes.
4. Keep report semantics aligned with the same official score definitions.
5. Add a clear distinction between final summary artifacts and inspection-ready retention artifacts.

### 10.4 Interface and contract definitions

The benchmark and demo layer should accept:

- a checkpoint path;
- a resolved config;
- an entity identifier;
- an export policy;
- a retention policy.

It should emit:

- metrics;
- histories;
- replayable trace bundles;
- provenance manifests.

### 10.5 Design pattern application

Use adapter-style wrappers around the core model and evaluator. Do not let the demo re-implement model logic; the demo should only visualize or replay artifacts already emitted by the engine.

### 10.6 Risk mitigation

- Evaluation inflation risk: keep report aggregation aligned with the same score contract as the model output.
- Post-experiment EDA risk: retain more trace data by default, but make the retention policy explicit and versioned.
- Visualization drift risk: make the demo consume the same exported artifacts as the benchmark.

### 10.7 Test plan and validation

Add tests for:

- offline benchmark export completeness;
- online benchmark export completeness;
- demo trace loading compatibility;
- retention policy toggles;
- report key stability.

### 10.8 Acceptance criteria

This batch is complete only when:

- benchmark and demo consume the same contract;
- the retained traces are sufficient for post-experiment EDA;
- the final summary still matches the official metric semantics.

## 11. Batch 8 — post-experiment EDA retention layer

### 11.1 Summary

This batch makes artifact retention a first-class requirement. The goal is to retain enough raw traces that later analysis can investigate score dynamics, uncertainty, and verification behavior without rerunning the experiment.

### 11.2 File-level edits

Modify these files:

- `src/core/artifact_integrity.py`
- `src/engine/checkpoint.py`
- `src/engine/evaluator.py`
- `scripts/run_thesis_offline_benchmark.py`
- `scripts/run_thesis_online_benchmark.py`

### 11.3 Explicit edit content

1. Store Monte Carlo sample tensors when enabled.
2. Store per-window score histories and uncertainty histories.
3. Store triage history, verification outcomes, and runtime state snapshots.
4. Store checkpoint and resolved config hashes with every retention bundle.
5. Separate inspection-ready retention artifacts from the final report artifact.

### 11.4 Interface and contract definitions

The retention layer should support:

- an explicit retention policy;
- optional compression;
- versioned artifact bundles;
- entity-scoped export directories.

The default should favor retention because the workflow requires post-experiment EDA.

### 11.5 Design pattern application

Use small artifact writers and explicit export policies. This keeps the retention layer compositional and easy to inspect.

### 11.6 Risk mitigation

- Storage blow-up risk: make retention switches explicit and versioned.
- Trace ambiguity risk: keep sample tensors, means, and variances clearly labeled.
- Resume confusion risk: separate runtime checkpoints from analysis exports.

### 11.7 Test plan and validation

Add tests for:

- export directory structure;
- presence of sample traces when retention is enabled;
- presence of summary-only outputs when retention is reduced;
- manifest integrity and reproducibility metadata.

### 11.8 Acceptance criteria

This batch is complete only when:

- the exported artifacts can support later EDA directly;
- the retention policy is reproducible and versioned;
- summary and inspection artifacts are clearly separated.

## 12. Batch 9 — tests and validation hardening

### 12.1 Summary

This batch proves the protocol with small focused tests. The goal is to prevent protocol drift before the codebase is considered experiment-ready.

### 12.2 File-level edits

Add or extend tests under:

- `tests/core/test_contracts.py`
- `tests/models/test_multitask_shapes.py`
- `tests/models/test_thesis_multitask_point_score_loss.py`
- `tests/online/test_full_spec_online_contract.py`
- `tests/online/test_online_prototype_metadata_contract.py`
- `tests/engine/test_checkpoint_roundtrip.py`
- `tests/engine/test_threshold_artifact.py`
- `tests/benchmarks/test_full_spec_runtime_readiness.py`

### 12.3 Explicit edit content

1. Verify the Monte Carlo sample dimension and aggregation rules.
2. Verify the one-window online contract.
3. Verify deterministic geometry and verification semantics.
4. Verify checkpoint and threshold artifact provenance.
5. Verify export retention paths and summary outputs.
6. Verify that the model and engine still satisfy the stable contracts after the new fields are added.

### 12.4 Interface and contract definitions

Tests should cover:

- data shapes;
- model forward contract;
- single training step;
- checkpoint save/load;
- online one-window causal flow;
- threshold artifact round-trip;
- export retention presence.

### 12.5 Design pattern application

Keep tests minimal and focused. Use registry-built fixtures and small contract-level assertions. Do not create a parallel testing framework for v3.

### 12.6 Risk mitigation

- Evaluation metric inflation risk: assert exact score definitions in tests.
- Adaptation contamination risk: assert that labels do not reach adaptation or verification calls.
- Projector drift risk: assert that frozen parameters remain unchanged where required.

### 12.7 Test plan and validation

Run focused pytest cases for each batch gate and keep smoke tests small enough to execute frequently.

### 12.8 Acceptance criteria

This batch is complete only when:

- the new protocol is covered by unit and integration tests;
- failing contract tests exist before implementation changes;
- all batch gates are measurable.

## 13. Batch 10 — readability and file-size remediation

### 13.1 Summary

This batch reduces complexity after the protocol is stable. The goal is to make the implementation readable to a high-school-level reader of the codebase without changing the scientific contract.

### 13.2 File-level edits

Refactor files that cross the repository’s readability gate, especially:

- `src/models/thesis_multitask_*.py`
- `src/engine/online_tta/*.py`
- any helper file larger than the file-size limit

### 13.3 Explicit edit content

1. Split files that become too large.
2. Keep methods and standalone functions short.
3. Replace ambiguous variable names with explicit names.
4. Keep comments explanatory and short.
5. Avoid deep inheritance or hidden phase-specific runtime behavior.

### 13.4 Interface and contract definitions

Refactoring must not change the public contracts. It may only reorganize internal helpers to keep the same semantics easier to read.

### 13.5 Design pattern application

Preserve composition, adapter boundaries, strategy dispatch, and registry-driven construction. The refactor should make those patterns more visible, not less visible.

### 13.6 Risk mitigation

- Readability regression risk: enforce explicit names and short functions.
- Hidden coupling risk: keep phase-specific logic inside the owning model or engine file.
- Scientific drift risk: refactor only after tests prove semantic stability.

### 13.7 Test plan and validation

Re-run the full focused test set after the refactor and confirm that no contract tests change behavior.

### 13.8 Acceptance criteria

This batch is complete only when:

- the codebase remains scientifically identical;
- files and functions are easier to read;
- the repository’s file-size and function-length preferences are restored or improved.

## 14. Implementation notes for later coding

The detail plan should be executed with the following discipline:

- keep each batch small enough to review;
- do not begin a later batch until earlier contract tests pass;
- log rich artifacts by default for post-experiment EDA;
- avoid introducing a second public THESIS model or a second online stack;
- preserve the source-of-truth role of `documents/spec/full-spec-v3.md`.

## 15. Final acceptance criteria

The whole plan is complete only when:

- the public THESIS offline and online entrypoints still match the stable contracts;
- stochastic retrieval is explicit, vectorized, and tested;
- Monte Carlo mean and variance use exactly ten samples in the official protocol;
- deterministic geometry and verification remain label-free and non-stochastic;
- calibration and checkpoint artifacts carry complete provenance;
- benchmark and demo outputs retain enough information for post-experiment EDA;
- the codebase continues to follow the repository rules on readability, composition, registry-driven construction, and single public entrypoint per model.

