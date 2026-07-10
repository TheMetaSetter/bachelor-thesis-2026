---
date: 2026-07-10T21:00:00+0700
researcher: Codex
git_commit: 52b18d95a0f4dd83efc25f5d99e41a20263ad591
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary plan to remove full-spec-v2 execution blockers"
tags: [plan, full-spec-v2, online-tta, gpu, integration]
status: preliminary_ready
last_updated: 2026-07-10
last_updated_by: Codex
---

# Preliminary plan: blocker remediation

## Objective

Make the current online pipeline deterministic and resumable on a real GPU
server while preserving public model constructors, registry names, report keys,
checkpoint compatibility, and the current test organization.

## Phases

1. **Freeze the current contracts.** Add tests under the current `tests/online`,
   `tests/runtime`, `tests/benchmarks`, and `tests/demo` layout. Define one
   `OnlineRuntimeState` schema and assert entity, variant, threshold, projector,
   buffer, and guard invariants.
2. **Make calibration entity-scoped.** Build and persist a map of artifacts by
   entity. Select and validate the artifact before processing each test entity;
   fail closed on missing or mismatched identity.
3. **Connect signature verification.** Add a read-only prototype adapter and a
   bounded recurrent-history owner. Construct PNN masks inside the event flow,
   excluding discrete anomalies and requiring non-overlapping recurrence.
4. **Connect verification cycles.** Replace direct buffer admission with
   `try_admit`, trigger verification at eight entries, and decrement TTL only
   after cycle completion. Add hard-old non-overlap acceptance after a successful
   update.
5. **Make checkpoint resume complete.** Serialize and restore runtime state with
   entity/variant/artifact validation; never restore optimizer moments.
6. **Validate GPU execution.** Add preflight checks for CUDA device, checkpoint
   identity, parameter mutation surface, deterministic seed, artifact paths and
   smoke commands O0-A0, O0-A2, O1-A0 and O1-A2.
7. **Refresh audit documentation.** After implementation, rerun AST and test
   inventories and merge the resulting detail plan into the existing full-spec
   detail document.

## Main risk

The central risk is state split across local variables and helper objects. The
plan therefore makes runtime state explicit before adding further behavior.

## Preliminary acceptance

- Two entities produce two artifacts and use the matching artifact at runtime.
- A causal synthetic stream reaches triage, PNN/buffer lifecycle and adaptation
  without reading future points or labels.
- Checkpoint roundtrip restores state but not optimizer moments.
- All current focused tests and four GPU smoke wrappers complete.
