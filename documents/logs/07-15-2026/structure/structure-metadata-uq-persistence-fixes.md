---
date: 2026-07-15 16:32:50 +07
researcher: TheMetaSetter
git_commit: b9f98e329401455a97d981dff3a4eafe509f9d47
branch: dev
repository: bachelor-thesis-2026
topic: "Structure for checkpoint metadata and UQ persistence fixes"
tags: [structure, time-series, anomaly-detection, multi-class]
status: draft
last_updated: 2026-07-15
last_updated_by: TheMetaSetter
---

# Structure: Checkpoint metadata and UQ persistence fixes

## Overview
The repository already has the core thesis contracts, but the current gap is semantic completeness: checkpoint metadata can still be placeholder-like, and the full UQ trace payload is computed at runtime but not yet persisted as a durable artifact. The implementation should therefore proceed in a minimal vertical slice: first lock checkpoint semantics, then enforce runtime UQ semantics, then persist trace artifacts, then repair online provenance, and only after that validate the full flow on one representative combination.

## Implementation Phases

### 1. Semantic checkpoint provenance hardening
This phase strengthens `src/models/thesis_multitask_impl/thesis_multitask_state_serialization_mixin.py` and related checkpoint logic so that metadata is not only present, but also meaningfully initialized when memory is active. It preserves the existing one-model-one-file layout and adds fail-closed checks for placeholder provenance, empty verification tensors, and shape or finite-value violations.

### 2. Runtime UQ contract enforcement
This phase tightens `src/core/contracts.py` and the model-side UQ construction paths so that `stochastic_query` and `uncertainty` are validated as meaningful runtime outputs rather than merely as schema-valid dictionaries. The phase preserves separation of concerns by keeping validation close to the model-output contract and avoiding new framework abstractions unless the tests expose a real gap.

### 3. Stage-B trace persistence
This phase extends `src/engine/evaluator.py` and the benchmark export path so that the full trace payload is written to a dedicated artifact instead of remaining only in memory or in summary metrics. The phase should preserve the minimal vertical slice principle by starting with a simple JSON-friendly trace export and adding compression only if the payload becomes too large for readable maintenance.

### 4. Online threshold provenance repair
This phase updates `src/protocols/threshold_artifact.py`, `src/engine/online_tta/online_engine_run.py`, and `scripts/benchmarks/run_thesis_online_benchmark.py` so the online threshold artifact contains a real checkpoint hash and a complete provenance record. This phase keeps the online path conservative and projector-oriented, and it preserves the existing runtime layering instead of introducing a separate online artifact framework.

### 5. End-to-end validation on one representative combination
This phase runs the full flow on exactly one development-spec combination first, then verifies Stage A provenance, Stage B provenance, Stage-B trace persistence, and online threshold provenance in one pass. Only after this representative pass succeeds should the repository scale out to the full combination matrix.

## Design Principle Check
- The batch contract remains fixed and unchanged.
- The encoder contract remains `hidden: Tensor[B, L, H]`.
- The model output contract keeps `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- Composition remains preferred over deep inheritance.
- The model file remains the home of its forward path, scoring path, and stage-specific logic.
- The first patch stays minimal and test-driven before any helper-module extraction.

## Feedback Questions
- Is this phase order acceptable, or should trace persistence move before checkpoint semantics?
- Should Stage B and online share the same trace artifact schema, or should online keep a smaller output subset?
- Should the online threshold artifact hash be treated as mandatory in the first patch, or only after the trace export is stable?
