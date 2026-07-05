---
date: 2026-07-05 14:04:14 +07
researcher: TheMetaSetter
git_commit: 757d9480d72ee0a1925b0b7194b05b599b3b2f0f
branch: dev
repository: bachelor-thesis-2026
topic: "Structure for eliminating stage/phase semantic drift"
tags:
  - structure
  - stage
  - phase
  - semantics
  - thesis-multitask
status: draft
last_updated: 2026-07-05
last_updated_by: TheMetaSetter
---

# Structure: Eliminate stage/phase semantic drift

## Overview

The active two-stage offline-pretraining path already executes correctly, but several public names still allow `phase` to be read as if it were the same thing as `stage`. The structure below keeps the runtime behavior stable while making the active contract single-meaning for a new reader.

## Implementation Phases

1. Terminology lock and compatibility boundary - Establish one explicit meaning for the active two-stage rerun. `offline pre-training` is the large phase, while Stage A and Stage B are sub-stages. Keep legacy three-stage names only inside compatibility boundaries and historical files. This phase preserves the minimal vertical slice principle because it changes wording before behavior.

2. Runner vocabulary cleanup - Rewrite the active two-stage runner so its internal variables and comments speak in stage-first terms. Keep the manifest and config schema stable for now, but make the generated outputs easier to read. This phase preserves composition and stable interfaces by changing only the orchestration vocabulary, not the execution contract.

3. Model runtime label cleanup - Clarify the stage labels exposed by the thesis model runtime state, including `semantic_stage_label` and related metadata. Keep the model behavior unchanged while making the active two-stage meanings obvious and separating them from Stage 3 compatibility wording. This phase preserves one-model-one-file and stable interfaces because all edits remain inside the owning model file family.

4. Config and compatibility isolation - Tighten config validation so the active two-stage contract is explicit and the legacy Stage 3 alias logic is visibly historical or compatibility-only. Where necessary, isolate legacy names behind narrow shims instead of letting them appear as the default public meaning. This phase preserves the registry/factory style and avoids introducing a new abstraction layer.

5. Docs, tests, and verification - Update the active documentation and test names so they teach the same meaning as the code. Add or adjust focused pytest coverage for runner plans, config loading, lifecycle labels, and compatibility behavior. This phase preserves the least-amount-of-codepaths principle because it validates the cleaned contract without broad architectural change.

## Phase Ordering

The order should stay as written:

1. terminology lock,
2. runner cleanup,
3. model runtime labels,
4. config and compatibility isolation,
5. docs and verification.

This order is minimalistic because it starts with the smallest semantic surface and only expands into compatibility and documentation after the active runtime meaning is stabilized.

## Pass-Level Validation

- Pass 1: repository grep should show one active meaning for the two-stage rerun and one clearly historical meaning for old three-stage material.
- Pass 2: two-stage runner tests should still pass unchanged.
- Pass 3: model lifecycle tests should still report the same behavior, but the labels should read stage-first.
- Pass 4: config-loading tests should still accept supported legacy inputs and reject conflicting inputs.
- Pass 5: the focused test bundle should still pass, and a fresh read of the active files should no longer force the reader to infer two meanings from one term.

## Notes on Design Principles

- Composition over inheritance remains unchanged because the current model split already isolates concerns without changing runtime behavior.
- Adapter pattern remains the right interpretation for the encoder-facing contract because the reader should see one stable hidden-state interface.
- Registry and factory usage remain unchanged so config-driven construction stays explicit.
- The codebase should not grow new codepaths just to rename concepts; the cleanup should collapse ambiguity, not add new layers.
