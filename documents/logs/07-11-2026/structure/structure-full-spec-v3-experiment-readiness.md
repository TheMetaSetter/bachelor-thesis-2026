---
date: 2026-07-11T23:02:58+07:00
planner: Codex
git_commit: fbfd011ac85e94d559201fd2153161e5523ff8af
branch: dev
repository: bachelor-thesis-2026
topic: "Programming plan for complete full-spec-v3 experiment readiness"
tags: [structure, full-spec-v3, offline, stochastic-retrieval, online-tta, benchmark, demo, reproducibility, post-experiment-eda]
status: draft
source_plan: documents/logs/07-11-2026/plan/plan-full-spec-v3-experiment-readiness.md
last_updated: 2026-07-11
last_updated_by: Codex
---

# Structure: complete `full-spec-v3` experiment readiness

## Overview

The plan should follow a contract-first, minimal-vertical-slice progression that preserves the current THESIS public entrypoints while extending them to support vectorized stochastic retrieval, deterministic geometry, online projector-only adaptation, and rich artifact retention for post-experiment EDA. The sequencing must keep the batch contract, model output contract, and benchmark/report contracts stable while each deeper stage consumes only the outputs established by the preceding stage.

## Implementation Phases

### 1. Contract and configuration foundation

This phase should lock the v3 schema, threshold provenance, checkpoint metadata, and top-level output contract before any stochastic behavior is expanded. It should preserve separation of concerns by keeping validation in the contract layer, provenance in the artifact layer, and model behavior in the model entrypoints. The design pattern should remain registry- and config-driven, with one public entrypoint per model and no hidden runtime branching outside the owning model.

### 2. Vectorized stochastic retrieval and Monte Carlo aggregation

This phase should implement the continuous and discrete stochastic query operators, perform similarity precomputation once, and expand only across the Monte Carlo sample dimension. It should then aggregate means and unbiased variances for reconstruction, point scores, window scores, and retrieval traces under the stable `aux` schema. The design should use composition and small helper functions so that stochastic query logic stays isolated from deterministic geometry and from the encoder contract.

### 3. Deterministic geometry, offline training, and uncertainty-aware evaluation

This phase should preserve the deterministic nearest-codeword filtering, continuous signature extraction, Stage A / Stage B lifecycle, and calibration behavior while aligning offline evaluation with the official ten-sample stochastic protocol. It should keep the model file self-contained, keep losses modular, and keep the evaluator responsible for metric computation only. The implementation should remain ablation-friendly through explicit configuration switches rather than implicit codepath edits.

### 4. Online adaptation, verification, and causal runtime state

This phase should maintain the frozen source encoder, residual projector, triage regions, verification buffer, and projector-only updates as the causal online loop. It should preserve label-free runtime behavior and record enough event history for exact replay, resume, and later EDA. The design should use composition over inheritance, with runtime state, verification, and adaptation separated into narrow helpers that remain readable and testable.

### 5. Benchmark export, demo parity, and post-experiment EDA retention

This phase should extend the offline and online benchmark wrappers, the demo surfaces, and the checkpoint/evaluator exports so that the run retains sample-level traces, uncertainty traces, triage history, verification outcomes, and provenance data. It should make post-experiment EDA a first-class output requirement rather than an afterthought. The design should keep reporting and visualization aligned with the same contract so that the demo and benchmark tell the same scientific story.

### 6. Tests, readability, and file-size remediation

This phase should verify the Monte Carlo sample dimension, one-window online contract, deterministic geometry, checkpoint round-trip, threshold provenance, and export retention paths with small pytest cases. It should then reduce any readability debt that remains after the protocol is stable, including file-size gate violations and long functions. The design should keep codepaths minimal, names explicit, and modules pedagogical so that the implementation remains easy to audit and extend.

## Suggested ordering constraint

The phases should be implemented in the order above, because each later phase depends on contracts, outputs, or artifacts established earlier. The only acceptable deviation is a narrowly scoped prerequisite fix discovered by tests, and that fix should still preserve the same contract-first progression.

