---
date: 2026-07-10T21:05:00+0700
researcher: Codex
git_commit: 52b18d95a0f4dd83efc25f5d99e41a20263ad591
branch: dev
repository: bachelor-thesis-2026
topic: "Structured outline for full-spec-v2 blocker remediation"
tags: [structure, full-spec-v2, online-tta, gpu]
status: ready_for_detail
last_updated: 2026-07-10
last_updated_by: Codex
---

# Structured outline

## Overview

The remediation is organized around one explicit online state object. The
implementation proceeds from contracts, to entity calibration, to signature and
buffer lifecycle, to resume safety, and finally to GPU smoke execution.

## Implementation phases

1. **Contract and state freeze**
   - Define `OnlineRuntimeState` and immutable event records.
   - Place tests in the current semantic folders.
   - Preserve adapter and registry surfaces.

2. **Entity threshold pipeline**
   - Calibrate clean validation independently per entity.
   - Persist `thresholds/{entity_id}/online_thresholds.json`.
   - Select artifact before the first test window.

3. **Prototype signature and PNN event flow**
   - Adapt model prototype tensors through a read-only adapter.
   - Maintain bounded, entity-scoped recurrent signatures.
   - Construct masks before A1/A2 dispatch.

4. **Verification cycle and update guards**
   - Admit only non-overlapping candidates.
   - Verify at capacity eight and tick TTL only once per cycle.
   - Accept hard-old intervals only after successful A2 update.

5. **Checkpoint and resume integrity**
   - Serialize state schema and artifact identity.
   - Validate entity and variant before restore.
   - Rebuild fresh AdamW after resume rather than restoring moments.

6. **GPU preflight, smoke, and reporting**
   - Validate config/checkpoint/device matrix.
   - Run four required smoke wrappers.
   - Record evidence in a final implementation log.

7. **Audit refresh and plan merge**
   - Re-run AST and test inventory.
   - Update source-audit detail with the final file/test layout.

## Design principles

- Composition owns runtime state; no new mixin lifecycle.
- Adapters expose prototype tensors without mutating model memory.
- Explicit strategy dispatch keeps A0/A1/A2 separate.
- Registry and checkpoint public surfaces remain stable.
- Every phase has a focused test gate before the next phase.
