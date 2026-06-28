---
date: 2026-06-27 21:20:00 +0700
planner: Codex
git_commit: 89a598f643cf0c20b0ab540b926e6b71f27e975f
branch: dev
repository: bachelor-thesis-2026
topic: "Structure outline for the unified dataset benchmark protocol"
tags: [structure, datasets, benchmark, loaders, evaluation, protocol]
status: draft
last_updated: 2026-06-27
last_updated_by: Codex
source_plan: documents/logs/06-27-2026/plan/plan-unified-dataset-benchmark-protocol.md
source_research: documents/logs/06-27-2026/research/research-unified-dataset-benchmark-protocol-spec-context.md
---

# Structure: Unified dataset benchmark protocol

## Overview

This structure turns the agreed benchmark protocol into a staged implementation outline that stays close to the current repository architecture. The main idea is to preserve the existing shared pipeline `parse -> clean -> scale -> windowize -> reconstruct -> evaluate`, remove benchmark-invalid split regimes, and add only the smallest protocol-validation and audit extensions needed for correctness across current and future dataset families.

## Implementation Phases

1. **Phase 1 - Freeze the benchmark contract and remove protocol-invalid split modes**

   The first phase should lock one simple benchmark meaning for all dataset families: `train` and `val` come from the past, `test` is a future labeled timeline segment, scaling is fit on `train` only before windowization, and evaluation is performed on the reconstructed pointwise `test` timeline. In this phase, the repository should remove `pre_vs_anomaly` and `pre_vs_post` from the benchmark runtime contract, narrow `anomaly_archive` to one benchmark-safe parsing path, and update config validation so benchmark-invalid split modes cannot silently re-enter the main path. This phase preserves software engineering quality by simplifying the public contract before adding more datasets or more audit logic.

2. **Phase 2 - Add one thin shared split-protocol validation layer**

   The second phase should introduce one small helper module that validates split semantics without creating a new framework. Its job is only to centralize a few rules such as temporal ordering, labeled-test requirements, and protocol-status description, while leaving dataset-specific parsing inside parser files and leaving scale/window logic inside `src/data/loaders.py`. This phase preserves design clarity because the repository gains one explicit place that defines benchmark correctness, but it does not add inheritance-heavy abstractions or policy objects.

3. **Phase 3 - Harden audit and evaluation metadata around coverage and comparability**

   The third phase should extend the current audit and evaluator outputs so every run states whether it is a full comparable benchmark evaluation or only a truncated smoke evaluation. The key additions in this phase are protocol-status fields, truncation flags, label-regime summaries, evaluated-vs-raw coverage counts, and metric-degeneracy diagnostics that explain when a score is undefined or non-comparable. This phase keeps the evaluator dataset-agnostic and preserves separation of concerns because it reports protocol conditions rather than trying to hide them.

4. **Phase 4 - Generalize parser onboarding for more dataset families without rewriting the loader architecture**

   The fourth phase should keep the current parser/builder structure, but make it easier to add dataset families such as `SWaT`, `IOPS`, `NASA`, and future datasets under the same benchmark contract. Each parser should remain responsible only for reading raw files, deriving split-local timelines, decoding labels, and emitting the already established raw split object shape. This phase preserves the repository's practical style because new datasets become parser additions plus focused tests, not a new loader framework.

5. **Phase 5 - Add suspicious, protocol-focused tests and finalize benchmark documentation**

   The final phase should lock the protocol with tests that target real failure modes: anomaly-only test slices, all-normal reconstructed labels, smoke truncation, wrong scaler fit scope, broken coverage reconstruction, and parser-specific label extraction mistakes. This phase should also refresh human-readable documentation in `documents/` so a reader can understand the benchmark rules, the meaning of truncation warnings, and the boundary between benchmark evaluation and exploratory analysis. This preserves long-term maintainability because future contributors can add datasets or experiments without rediscovering the same protocol mistakes.

## File-Level Rollout

### Stage A - Benchmark contract cleanup

- `src/core/config.py`
- `src/data/api.py`
- `src/data/datasets/anomaly_archive.py`

This stage removes obsolete benchmark-facing config keys and narrows the active split semantics.

### Stage B - Thin shared validation helper

- `src/data/split_protocol.py` or a similarly small helper file
- `src/data/loaders.py`
- `src/data/base.py` only if a minimal contract hook is truly needed

This stage adds explicit protocol checks while keeping the current builder path intact.

### Stage C - Audit and evaluator hardening

- `src/analysis/evaluation_protocol_audit.py`
- `src/engine/evaluator.py`
- `src/metrics/pointwise.py` only where metadata exposure or degeneracy wording needs to be tightened

This stage makes suspicious evaluation regimes visible instead of implicit.

### Stage D - Dataset-family onboarding path

- existing parser files under `src/data/datasets/`
- future parser files for `swat.py`, `iops.py`, `nasa.py`, and others
- `src/core/config.py` registry section

This stage expands dataset support using the same frozen contract.

### Stage E - Test and documentation lock

- `tests/`
- `documents/design/`
- `documents/logs/...`

This stage ensures the benchmark protocol is both executable and explainable.

## Structural Assessment

This phasing is intentionally conservative. It fixes the benchmark meaning first, then adds one thin shared validation layer, then strengthens audit and onboarding. That order is important because it avoids the common mistake of building more infrastructure before the repository agrees on what a valid `train/val/test` timeline actually is.

## Low-Level Decisions Still Open

1. A benchmark-valid `test` split should probably contain at least one anomalous point and at least one normal point on the reconstructed timeline. This matches the main goal of pointwise anomaly detection, but it should be confirmed before the detail step because some official datasets may still contain edge cases.
2. Smoke or truncated evaluations should probably remain runnable, but they should be marked clearly as non-comparable instead of being treated as benchmark results. This is the most practical option because it keeps fast debugging runs while protecting the benchmark.
3. When `UCR/AnomalyArchive` is brought back under the unified protocol, the safest default is probably to preserve the official time order and build one future `test` timeline that contains both normal and anomalous timesteps whenever the raw series allows it, rather than creating anomaly-only or post-anomaly-only slices.

## Recommended Next Step

Move to `prompts/4_detail_prompt.md` only after the open low-level decisions above are explicitly settled. The detail step should then decide exact files, exact tests to write first, and the smallest code edits needed to preserve the current architecture while fixing the benchmark protocol.
