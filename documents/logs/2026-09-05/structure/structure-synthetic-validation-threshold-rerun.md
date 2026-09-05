---
date: 2026-09-05T17:09:13+07:00
topic: "Synthetic-validation normal-score q99 threshold rerun"
status: approved
revision: 5529a0c3f1ab9f4b7f013543aa7e61922b74b56d
related_documents:
  - /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/research/research-synthetic-validation-threshold-change-surface.md
  - /Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/2026-09-05/plan/plan-synthetic-validation-threshold-rerun.md
---

# Implementation Structure: Synthetic-validation normal-score q99 threshold rerun

## Summary

The delivery path has five dependent phases: establish an opt-in contract, implement synthetic-normal q99 selection, isolate and describe artifacts, prove compatibility with tests, and prepare the remote command. The user requested the detail document in the same instruction, so this structure is treated as approved for expansion.

## Request

Use covered normal samples from synthetic validation to set one q99 point threshold. Reuse it for synthetic/test evaluation. Cover all Stage-B best checkpoints on `cloud-gpu`. Write results into a new rerun directory. Preserve existing behavior.

## Confirmed context

- `val_synth` and covered-point reconstruction already exist in the evaluator.
- The runner currently selects the point threshold from clean validation.
- The active protocol and threshold artifact currently use clean validation as their default calibration contract.
- The CLI currently derives output paths from the experiment config.

## Scope

### In scope

- Opt-in synthetic-normal threshold source.
- One q99 point threshold reused within each evaluation run.
- Threshold-source and output-path provenance.
- Backward-compatible CLI/API behavior.
- Focused tests and the cloud command.

### Out of scope

- Training changes.
- Online threshold recalibration changes.
- Replacing the default clean-validation protocol.
- Remote execution during planning.

## Proposed phases

### Phase 1: The opt-in contract is explicit

**Result:** A new protocol/config can request `synthetic_validation_normal`; old configs continue to select clean validation.

**Stages:**

1. Define the optional source field and its allowed value.
2. Keep the legacy `offline_threshold_split` and top-level artifact calibration contract unchanged by default.
3. Define the optional output-directory override and fallback behavior.
4. Record the terminology mapping and compatibility rule.

**Depends on:** Research change surface and the current protocol/artifact validators.

**Verification:** Legacy protocol validation passes; the new protocol parses and rejects unsupported source values.

**Risks:** A new field could silently change the default. Mitigate with an explicit absent-field fallback and regression test.

**Complete when:** The contract has one canonical name for the new source and no legacy default changes.

### Phase 2: Synthetic-normal q99 drives offline point metrics

**Result:** Each opt-in run computes q99 from finite covered synthetic-normal point scores and uses that one threshold for synthetic and test point metrics.

**Stages:**

1. Run the existing synthetic validation path without a fixed point threshold.
2. Reconstruct the covered point payload and identify synthetic normal points.
3. Remove non-finite scores and fail clearly when no usable normal score remains.
4. Compute q99 from the remaining scores.
5. Re-run synthetic validation and evaluate test with the fixed point threshold.
6. Keep the window threshold calculation separate because it uses a different score unit.

**Depends on:** Phase 1 source selection contract.

**Verification:** A mixed synthetic fixture yields the q99 of only covered normal scores and never uses test labels.

**Risks:** Anomaly or uncovered points could contaminate calibration. Mitigate with explicit masks and a count assertion.

**Complete when:** The opt-in metrics show the selected threshold and source without changing the legacy branch.

### Phase 3: New rerun artifacts are isolated and auditable

**Result:** New runs write scores, metrics, threshold metadata, retention outputs, and reports below the requested rerun directory.

**Stages:**

1. Carry the effective output root through evaluation-only manifest creation.
2. Route artifact and retention writers to that root.
3. Record the synthetic-normal source on the offline point threshold record.
4. Preserve online clean-validation metadata and legacy artifact readability.

**Depends on:** Phase 2 selected threshold and Phase 1 compatibility contract.

**Verification:** A test proves the new root receives all expected artifacts and the configured legacy root is untouched.

**Risks:** Shared top-level artifact fields may mislabel online calibration. Mitigate by separating offline point source metadata from legacy calibration metadata.

**Complete when:** A rerun report and threshold artifact identify the new output root, checkpoint, score space, quantile, and source.

### Phase 4: Regression tests prove both paths

**Result:** Automated tests cover new threshold selection, artifact provenance, output isolation, and old behavior.

**Stages:**

1. Add unit coverage for normal/covered/finite filtering and q99.
2. Update benchmark fixtures for the extra synthetic score pass.
3. Add protocol and artifact compatibility assertions.
4. Add CLI/API output override assertions.
5. Run focused tests, compile checks, and diff checks.

**Depends on:** Phases 1–3.

**Verification:** `.venv/bin/python -m pytest` focused tests pass; `git diff --check` passes.

**Risks:** Test doubles may hide the second synthetic evaluation. Mitigate by asserting exact call order and threshold arguments.

**Complete when:** Both legacy and opt-in tests pass from a clean test fixture.

### Phase 5: The cloud rerun command is ready

**Result:** A readable command sequence targets the current 18 Stage-B best checkpoints and writes only to the new rerun tree.

**Stages:**

1. Re-read the current SSH endpoint and remote repository path.
2. Read the current Stage-B best-checkpoint inventory without changing remote state.
3. Build the command from the new protocol and output-root convention.
4. Run one concrete preflight combination in `tmux` after explicit authorization.
5. Verify its artifacts before expanding to the remaining 17 combinations.

**Depends on:** Phases 1–4 and a current remote inventory.

**Verification:** The command is dry-run inspectable, names all 18 checkpoints, and uses a new output root.

**Risks:** Stale inventory or accidental overwrite. Mitigate with read-only preflight and exact output-path checks.

**Complete when:** The command is documented and ready; execution remains a separate authorized action.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Current protocol/artifact contracts | Opt-in source and output semantics |
| 2 | Phase 1 | Correct synthetic-normal threshold and metrics |
| 3 | Phases 1–2 | Isolated, auditable rerun artifacts |
| 4 | Phases 1–3 | Evidence of compatibility |
| 5 | Phases 1–4 plus current remote inventory | Safe 18-checkpoint command |

## Decisions confirmed

- The default clean-validation path remains unchanged.
- The new source uses the explicit name `synthetic_validation_normal`.
- The point threshold is computed from covered, normal, finite synthetic scores.
- Existing window-threshold units remain separate.
- New output paths are opt-in and isolated.

## Non-blocking uncertainties

- The exact remote repository path and checkpoint inventory must be refreshed immediately before execution.
- The detailed plan must verify the artifact writer's optional source field against existing schema tests.

## Feedback requested

The user already requested the detailed document in the same turn, so no additional structure review is required before expansion.
