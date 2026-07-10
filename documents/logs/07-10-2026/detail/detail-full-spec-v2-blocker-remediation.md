---
date: 2026-07-10T21:10:00+0700
researcher: Codex
git_commit: 52b18d95a0f4dd83efc25f5d99e41a20263ad591
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed plan for full-spec-v2 blocker remediation"
tags: [detail, full-spec-v2, online-tta, gpu, checkpoint, integration]
status: ready_for_merge
last_updated: 2026-07-10
last_updated_by: Codex
---

# Detailed blocker-remediation plan

This plan is an execution supplement to
`detail-full-spec-v2-gap-remediation.md`. It resolves the blockers found in
`research-full-spec-v2-blocker-audit.md` and uses the current test directories.

## Phase 0: state and test contract

### Edits

- Add `src/engine/online_tta/runtime_state.py` with typed records for entity
  identity, threshold artifact, recurrent history, verification buffer, guard
  intervals and online variant.
- Keep tensors out of serialized state except where the existing checkpoint
  contract already stores model state.
- Add tests in `tests/online/test_online_runtime_state.py` and
  `tests/runtime/test_online_checkpoint_contract.py`.
- Replace references to obsolete `tests/unit`, `tests/contract`, and
  `tests/integration` paths in the detail document with the current semantic
  folders.

### Acceptance

The state can be converted to JSON-safe dictionaries and back; malformed entity
or variant values fail before model mutation.

## Phase 1: entity-scoped calibration

### Edits

- Refactor `src/engine/online_tta/online_engine.py` so
  `_build_runtime_online_context()` calls
  `calibrate_entity_threshold_artifacts()` and stores an artifact map.
- Add `select_entity_threshold_artifact(artifacts, entity_id)` as a pure helper.
- Write a single-entity artifact at the existing path and multi-entity files at
  `thresholds/{entity_id}/online_thresholds.json`.
- Change `_run_online_execution_sequences()` to select the artifact and
  threshold value for the current sequence before entering its stream loop.
- Extend `tests/online/test_threshold_artifact.py` with two synthetic entities,
  missing-entity and mismatched-entity cases.

### Acceptance

No test sequence can begin with another entity's threshold. The report contains
the artifact map and the selected artifact identity for every processed entity.

## Phase 2: PNN adapter and recurrent history

### Edits

- Add `PrototypeReadOnlyAdapter` to
  `src/engine/online_tta/signature_verification.py` or a small companion file.
  It reads continuous prototypes, discrete codebook, anomaly mask and radii
  from the model without updating them.
- Add `RecurrentSignatureStore` with bounded per-entity history and explicit
  `append(window)` / `recurrent()` methods.
- In `_score_online_window()` or a named `build_online_signature_context()`:
  1. obtain detached hidden states;
  2. filter discrete anomaly tokens;
  3. calculate ordered signatures;
  4. update only the history store;
  5. construct `pnn_mask` as recurrent membership AND NOT known anomaly.
- Pass the mask and diagnostic counts into the batch used by A1/A2.
- Add tests in `tests/online/test_online_signature_verification.py` for entity
  isolation, bounded history, no model mutation, recurrence and causal order.

### Acceptance

PNN update decisions cannot use labels, future windows, or mutable prototype
state. A first occurrence produces no PNN mask; a second non-overlapping
occurrence can produce one.

## Phase 3: verification cycles and hard-old guard

### Edits

- Change `_update_online_window_buffers()` to use `try_admit()` only.
- Add a `run_verification_cycle()` helper that reads exactly the admitted
  entries, invokes the verification callback, marks results, and calls
  `finish_verification_cycle()` once.
- Store `verification_cycle_id`, admission/rejection counts and TTL summary in
  the event record.
- Pass `NonOverlapGuard` through `_process_online_window()` and add intervals
  only after a successful hard-old A2 update.
- Add tests in `tests/online/test_online_verification_buffer.py` and
  `tests/online/test_online_tta_variants.py` covering capacity eight, TTL=2,
  one decrement per cycle, overlap rejection and failed-update non-admission.

### Acceptance

No stream step decrements TTL. A buffer with fewer than eight new entries never
starts a verification cycle.

## Phase 4: loss, optimizer, and diagnostics contract

### Edits

- Split `_run_online_variant_update()` into explicit A0, A1-PNN, A2-hard-old
  and A2-PNN helpers in `online_engine.py`.
- Make A1 require a non-empty verified PNN mask.
- Make A2 hard-old require accepted guard interval and use
  `relu(window_score - B_window)^2`.
- Use `build_online_optimizer()` immediately before each update, clip projector
  gradients at 0.5, and discard optimizer state afterward.
- Add gradient norm helpers that prove source encoder, source memory,
  reconstruction head and classification head remain unchanged.
- Add integration assertions in `tests/online/test_online_tta_variants.py`.

### Acceptance

Each event has at most one optimizer step; projector-only mutation is proven by
parameter checksums; inactive loss fields are `None`, not stale values.

## Phase 5: checkpoint resume

### Edits

- Extend `src/engine/checkpoint.py` with a small validation callback for
  `entity_id`, `online_variant`, artifact hash and state version.
- Add `save_online_runtime_state()` and `load_online_runtime_state()` helpers
  under `src/engine/online_tta/`.
- Restore buffer entries, signature history and guard intervals before the next
  event; construct a fresh optimizer instead of loading optimizer moments.
- Add roundtrip tests in `tests/online/test_online_state_roundtrip.py`.

### Acceptance

Resume with a wrong entity or variant fails before adaptation. Resume with a
matching state produces the same next event decision as uninterrupted execution.

## Phase 6: demo and GPU matrix

### Edits

- Wire `demo/online_replay.py` to `StreamQueueController` and an injected score
  callback without exposing labels to the callback.
- Update `scripts/preflight_full_benchmark_matrix.py` to check artifact map,
  Stage-B checkpoint identity, CUDA device selection, and required smoke config
  existence.
- Add `tests/benchmarks/test_full_spec_matrix_smoke.py` using dry-run wrappers.
- Run, in order, O0-A0, O0-A2, O1-A0 and O1-A2 smoke commands from the original
  detail document, then the active `pytest -q` collection.
- Save command output and artifact paths under a dated implementation log.

### Acceptance

All four smoke wrappers complete on the target GPU server, each produces an
entity-specific threshold artifact and checkpoint, and no demo writes official
benchmark artifacts.

## Phase 7: documentation merge

- Append this blocker plan to `detail-full-spec-v2-gap-remediation.md` under a
  new “Blocker remediation supplement” section.
- Mark original batches as completed, partial, or pending based on test/smoke
  evidence only.
- Update `detail-src-code-audit-remediation.md` with final AST counts, current
  test folders, archived legacy tests and the final online state ownership.

## Rollback and GPU safety

Each phase is independently revertible. Do not delete checkpoints or artifacts
when a phase fails. Stop before GPU smoke if artifact identity, CUDA device,
or projector-only mutation checks fail.
