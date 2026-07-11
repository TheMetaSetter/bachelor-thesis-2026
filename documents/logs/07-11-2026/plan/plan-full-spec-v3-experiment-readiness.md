---
date: 2026-07-11T23:02:58+07:00
planner: Codex
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Programming plan for complete full-spec-v3 experiment readiness"
tags: [plan, full-spec-v3, offline, stochastic-retrieval, online-tta, benchmark, demo, reproducibility, post-experiment-eda]
status: draft
source_research: documents/logs/07-11-2026/research/research-full-spec-v3-v2-code-map.md
last_updated: 2026-07-11
last_updated_by: Codex
design_decisions_confirmed_at: 2026-07-11
design_decisions_confirmed_by: Khôi Nguyễn Anh
---

# Plan: complete `full-spec-v3` experiment readiness

## 0. Planning decision

This draft adopts a contract-first vertical remediation approach on top of the current THESIS codebase. The implementation should not create a parallel model family or a second public THESIS entrypoint. Instead, it should extend the current public surfaces so that the repository can execute the v3 stochastic retrieval protocol, preserve the deterministic geometry contracts, and keep the current offline / online / benchmark / demo runners compatible.

The most important engineering rule for this round is the following: every experiment should retain more data than the immediate evaluation needs, so that post-experiment EDA can be performed without rerunning the full trial. That means the implementation should prefer richer artifacts, not slimmer ones, provided that the richer artifacts are explicitly versioned, reproducible, and bounded by clear config switches.

The required implementation order is:

```text
freeze public contracts
  -> implement vectorized stochastic retrieval
  -> aggregate Monte Carlo outputs and uncertainty
  -> preserve deterministic geometry and verification semantics
  -> update calibration / threshold artifacts
  -> extend offline and online benchmark exports
  -> add post-experiment EDA retention paths
  -> harden tests and readability
```

This order is the smallest safe vertical slice because later stages consume sample tensors, variance summaries, thresholds, and runtime state produced by earlier stages.

## 1. Sources and authority

The implementation must use the following authority order:

1. `documents/spec/full-spec-v3.md` defines the mathematical experiment protocol and final acceptance criteria.
2. `documents/logs/07-11-2026/research/research-full-spec-v3-v2-code-map.md` maps the current executable code to the v2 / v3 spec boundary.
3. `documents/abstract-design-notes/idea.md` defines the thesis-facing representation contract and the intended model topology.
4. `documents/abstract-design-notes/design_starter.md` defines the stable codebase skeleton and readability principles.
5. `codebase_preferences.md` defines repository-wide non-negotiable engineering rules.
6. Active source, configuration, tests, and existing artifact helpers define current executable behavior but do not override the specification.

## 2. Current state

### 2.1 Implemented surfaces to preserve

The repository already provides a usable foundation for the v3 work:

- `src/core/contracts.py` already enforces the stable batch and output shapes, including the one-window online contract.
- `src/models/thesis_multitask_components.py`, `src/models/thesis_multitask_setup_mixin.py`, `src/models/thesis_multitask_state_mixin.py`, `src/models/thesis_multitask_routing_mixin.py`, and `src/models/thesis_multitask_loss_mixin.py` already contain the offline THESIS lifecycle.
- `src/models/online_adaptation.py` and `src/engine/online_tta/` already contain the online adaptation path, verification helpers, triage, runtime state, and loss helpers.
- `src/engine/checkpoint.py`, `src/core/artifact_integrity.py`, and `src/protocols/threshold_artifact.py` already provide the core persistence and provenance boundary.
- `scripts/run_thesis_offline_benchmark.py` and `scripts/run_thesis_online_benchmark.py` already connect model execution to benchmark wrappers.
- Baseline model families already exist under `src/baselines/online/` and `src/baselines/traditional/`.

These surfaces must be strengthened in place. New public THESIS model names, parallel experiment runners, or incompatible output schemas are prohibited unless a later structure review proves the current public entrypoints cannot satisfy the locked protocol.

### 2.2 Result-changing gaps

The active runtime still needs explicit work for the v3 protocol:

1. Monte Carlo retrieval must be vectorized and made explicit in the public output contract.
2. Mean predictions and variance summaries must be aggregated from exactly ten stochastic samples in evaluation and online scoring paths.
3. The `aux` schema must retain sample tensors, uncertainty tensors, and deterministic geometry tensors in a versioned way.
4. Threshold calibration and artifact persistence must record stochastic settings and variance correction settings.
5. Offline and online exports must store richer raw traces for later EDA, not only final metrics.
6. The retention policy for experiment artifacts must be clarified so that sample-level traces, score histories, and verification history can be analyzed after the run.
7. Readability debt remains a planning constraint: large files and long callables should be reduced only after the protocol is stable.

## 3. Design options and selected approach

### 3.1 Option A: extend the current THESIS runtime with stochastic retrieval and richer artifacts

This option keeps the current public entrypoints and adds a vectorized Monte Carlo query path, uncertainty summaries, deterministic geometry exports, and artifact retention switches. It preserves checkpoint and benchmark compatibility while making the protocol explicit. This is the selected approach.

### 3.2 Option B: split v3 into a separate THESIS model family

This option would create a second public model or a second execution stack for stochastic retrieval. It would make the code harder to compare against v2, complicate checkpoints, and duplicate benchmark infrastructure. It is rejected because it violates the repository preference for a single public entrypoint per model.

### 3.3 Option C: keep the current deterministic outputs and log stochastic values only as side artifacts

This option would leave the public model contract mostly unchanged and hide stochastic retrieval in external wrappers. It is rejected because the Monte Carlo protocol would become fragile, hard to test, and easy to desynchronize from the model contract.

## 4. Locked low-level decisions

The following decisions shall be treated as fixed for the plan:

- `ThesisMultitaskModel` remains the only public THESIS offline entrypoint.
- `OnlineAdaptationModel` remains the only public online adaptation entrypoint.
- The main input contract stays batch-first and one-window online, with `x: Tensor[B, L, D]` and `L = 20`.
- The stable top-level output contract remains `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- Stochastic retrieval uses exactly ten inference samples in validation, calibration, test, and official online scoring.
- Deterministic geometry stays deterministic; it must not be replaced by stochastic samples.
- Gray-zone admission, verification, and projector-only adaptation must remain label-free at runtime.
- Post-experiment EDA retention is a first-class requirement, not a debugging convenience.

## 5. Stable runtime contracts

### 5.1 Batch contract

`src/core/contracts.py` shall remain the canonical guard for the shared batch shape:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

For the active v3 path, online inputs must remain one causal window only. Legacy two-view validation may remain as a named historical guard, but it may not be selected by the main v3 runtime.

### 5.2 Encoder and source-adapter contract

`src/models/online_adaptation.py::ThesisMultitaskEncoderAdapter` remains a composition-based adapter around `ThesisMultitaskModel`. Its public operations should stay narrow:

```python
encode_source(batch) -> Tensor[B, L, H]
score_source(hidden, x) -> ModelOutputs
score_projected(projected_hidden, x) -> ModelOutputs
prototype_verification_metadata() -> PrototypeVerificationMetadata
```

The adapter must keep source hidden states detached and frozen. It must not own optimizer state or stream state.

### 5.3 Model output contract

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

For v3, `aux` must carry sample tensors, uncertainty tensors, deterministic geometry tensors, and version metadata. The plan should not split these concepts into unrelated top-level fields.

## 6. Planning implications from the v3 research map

The code map shows that the main work is not building a new stack from zero. The main work is making the current stack explicit enough that the stochastic retrieval protocol, the deterministic geometry protocol, and the online protocol all remain inspectable.

The files that matter most for the implementation sequence are:

- `src/models/thesis_multitask_components.py`
- `src/models/thesis_multitask_setup_mixin.py`
- `src/models/thesis_multitask_state_mixin.py`
- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_loss_mixin.py`
- `src/models/online_adaptation.py`
- `src/engine/online_tta/online_engine.py`
- `src/engine/online_tta/online_calibration.py`
- `src/engine/online_tta/verification_adapter.py`
- `src/engine/online_tta/signature_verification.py`
- `src/engine/online_tta/online_losses.py`
- `src/engine/checkpoint.py`
- `src/core/artifact_integrity.py`
- `src/protocols/threshold_artifact.py`
- `scripts/run_thesis_offline_benchmark.py`
- `scripts/run_thesis_online_benchmark.py`

## 7. Proposed implementation batches

### Batch 1 — contract and config boundary

Goal: make the v3 schema explicit and prevent silent drift.

Files:

- `src/core/contracts.py`
- `src/protocols/threshold_artifact.py`
- `src/engine/checkpoint.py`
- `configs/*.yaml` or the repository’s active config location

Tasks:

- Add explicit config keys for `stochastic_inference`, `monte_carlo_samples`, `continuous_temperature`, `discrete_temperature`, `variance_correction`, and `return_mc_samples`.
- Ensure checkpoint metadata records stochastic settings, query schema version, entity identity, and config provenance.
- Keep the stable output contract unchanged while extending `aux`.

### Batch 2 — vectorized stochastic query helpers

Goal: implement the Monte Carlo retrieval path without duplicating encoder work.

Files:

- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_components.py`
- `src/models/thesis_multitask_state_mixin.py`

Tasks:

- Introduce a small helper for vectorized Gumbel-Softmax retrieval over continuous prototypes and discrete codewords.
- Precompute similarity logits once, then expand only across the Monte Carlo sample dimension.
- Keep deterministic geometry functions separate from stochastic sampling functions.
- Preserve the current `cosine_topk` and `gumbel_softmax` semantics already present in the codebase.

### Batch 3 — Monte Carlo aggregation in the model contract

Goal: make the public forward path return mean predictions plus uncertainty summaries.

Files:

- `src/models/thesis_multitask_routing_mixin.py`
- `src/models/thesis_multitask_loss_mixin.py`
- `src/core/contracts.py`

Tasks:

- Aggregate `recon`, `logits`, `point_scores`, and `window_scores` as Monte Carlo means.
- Compute and expose point-wise and window-wise variance for score, reconstruction, and retrieval traces.
- Preserve the stable top-level output keys and place all stochastic samples under `aux`.

### Batch 4 — deterministic geometry and verification semantics

Goal: keep the non-stochastic guard rails exact.

Files:

- `src/engine/online_tta/signature_verification.py`
- `src/engine/online_tta/verification_adapter.py`
- `src/engine/online_tta/verification_buffer.py`
- `src/models/online_adaptation.py`

Tasks:

- Keep nearest-codeword filtering and recurrent continuous signature extraction deterministic.
- Ensure stochastic retrieval IDs are never used as signatures.
- Ensure verification uses frozen source latents and provenance-checked radius metadata.

### Batch 5 — calibration and threshold artifact revision

Goal: calibrate exactly the right score streams and persist them cleanly.

Files:

- `src/engine/online_tta/online_calibration.py`
- `src/engine/thresholding.py`
- `src/protocols/threshold_artifact.py`
- `src/core/artifact_integrity.py`

Tasks:

- Record whether a threshold was computed from non-overlapping offline scores or stride-1 online scores.
- Persist the Monte Carlo sample count and correction mode in the artifact.
- Fail closed on missing or inconsistent provenance.

### Batch 6 — offline benchmark export enrichment

Goal: retain enough traces for EDA after the run.

Files:

- `scripts/run_thesis_offline_benchmark.py`
- `src/engine/evaluator.py`
- `src/engine/checkpoint.py`

Tasks:

- Export sample-level reconstructions, score traces, uncertainty traces, and deterministic geometry traces when enabled.
- Keep metric semantics unchanged, but write richer per-entity artifacts for later inspection.
- Preserve the current benchmark matrix and report keys.

### Batch 7 — online causal execution and state logging

Goal: keep the online protocol causal, resumable, and inspectable.

Files:

- `src/engine/online_tta/online_engine.py`
- `src/engine/online_tta/runtime_state.py`
- `src/engine/online_tta/online_optimizer.py`
- `src/engine/online_tta/triage.py`

Tasks:

- Keep source-once scoring and projector-only mutation.
- Log triage decisions, verification outcomes, update decisions, and update-free windows.
- Add richer event history so that post-experiment analysis can reconstruct the causal timeline.

### Batch 8 — post-experiment EDA retention layer

Goal: make the experiment artifacts useful after the benchmark finishes.

Files:

- `src/core/artifact_integrity.py`
- `src/engine/checkpoint.py`
- `src/engine/evaluator.py`
- `scripts/run_thesis_offline_benchmark.py`
- `scripts/run_thesis_online_benchmark.py`

Tasks:

- Store raw per-window samples, score histories, uncertainty histories, triage histories, and verification histories.
- Keep compression optional, explicit, and versioned.
- Add a clear export location for inspection-ready artifacts separate from the final metric summary.

### Batch 9 — benchmark and demo parity

Goal: keep scripts, reports, and demo behavior aligned with the same contract.

Files:

- `scripts/run_thesis_offline_benchmark.py`
- `scripts/run_thesis_online_benchmark.py`
- `demo/app.py`
- `demo/online_replay.py`

Tasks:

- Ensure the demo can display the same sample-rich traces that the benchmark exports.
- Keep the report layer aligned with the exact stochastic / deterministic split.
- Avoid introducing a second interpretation of the same score.

### Batch 10 — tests and validation

Goal: prove the protocol with small, focused tests before larger experiments.

Files:

- `tests/core/test_contracts.py`
- `tests/models/test_multitask_shapes.py`
- `tests/models/test_thesis_multitask_point_score_loss.py`
- `tests/online/test_full_spec_online_contract.py`
- `tests/online/test_online_prototype_metadata_contract.py`
- `tests/engine/test_checkpoint_roundtrip.py`
- `tests/engine/test_threshold_artifact.py`
- `tests/benchmarks/test_full_spec_runtime_readiness.py`

Tasks:

- Verify the Monte Carlo sample dimension, mean aggregation, and unbiased variance.
- Verify the one-window online contract and deterministic triage regions.
- Verify checkpoint round-trip, artifact integrity, and threshold provenance.
- Verify that post-experiment export paths are populated when enabled.

### Batch 11 — readability refactor and file-size gate

Goal: finish by reducing complexity, not by hiding it.

Files:

- `src/models/thesis_multitask_*.py`
- `src/engine/online_tta/*.py`
- any file that crosses the readability gate

Tasks:

- Split any file that grows too large.
- Keep methods and functions short.
- Prefer explicit names, explicit config keys, and short linear control flow.
- Keep comments explanatory and pedagogical.

## 8. Data retention and post-experiment EDA policy

This plan explicitly requires a richer-than-minimal artifact policy.

The following data should be retained by default whenever storage cost is acceptable:

- Monte Carlo sample tensors, not only means.
- Per-window uncertainty traces.
- Raw triage events and verification outcomes.
- Calibration traces for offline and online thresholds.
- Deterministic geometry traces such as nearest codeword IDs and continuous signatures.
- Checkpoint SHA-256, resolved config hash, and schema version.
- Entity-scoped runtime state needed for exact resume.
- Exported benchmark summaries and intermediate score timelines.

The implementation may offer a config switch to disable some heavy sample tensors, but the default should favor retention because the user’s workflow requires post-experiment EDA.

## 9. Risk and mitigation

### Risk: continuous and discrete branches become redundant

Mitigation: keep separate logging for branch-wise retrieval variance, separate config keys, and explicit ablations. Do not collapse both branches into one implicit abstraction.

### Risk: stochastic sampling changes the meaning of thresholds

Mitigation: make calibration artifacts explicit about sample count, reduction rule, and correction mode. Do not reuse a one-sample threshold for ten-sample means.

### Risk: deterministic geometry is polluted by stochastic IDs

Mitigation: keep signature extraction and anomaly filtering on the deterministic path only. Add tests that fail if stochastic IDs leak into verification metadata.

### Risk: post-experiment retention becomes too heavy

Mitigation: make retention explicit and versioned, then provide optional compression or sample-saving switches. Keep the default rich enough for EDA, but still bounded by config.

### Risk: readability regresses under feature pressure

Mitigation: split large helpers, keep one public entrypoint per model, and avoid hiding phase-specific runtime behavior in new inheritance layers.

## 10. Open questions

1. Should `return_mc_samples` default to true for all evaluation runs, or only for benchmark / demo runs?
2. Should the retained EDA artifact be stored as `.npz`, `.pt`, or a small structured directory per entity and split?
3. Should online replay keep all per-step sample tensors in memory, or stream them to disk as soon as they are produced?
4. Is the current config tree the final one, or should a small explicit `configs/experiment/` layer be introduced for v3-only runs?

## 11. Acceptance criteria

This plan is complete only when the following are true:

- The public THESIS offline and online entrypoints still match the stable contracts.
- The v3 stochastic retrieval path is explicit, vectorized, and test-covered.
- Monte Carlo means and variances are computed from exactly ten inference samples in the official evaluation path.
- Deterministic geometry and verification semantics remain label-free and non-stochastic.
- Threshold and checkpoint artifacts carry complete provenance.
- Offline and online benchmark outputs retain enough information for post-experiment EDA.
- The codebase still follows the repository rules on readability, composition, single public entrypoint per model, and small focused tests.

