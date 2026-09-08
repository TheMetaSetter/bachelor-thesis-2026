---
date: 2026-09-08 Asia/Ho_Chi_Minh
topic: "Artifact naming and minimal retention code modifications"
status: proposed
revision: 040f3ce7d1cc23e8cb5735eabe69fe5b1530cdf2
related_documents:
  - documents/logs/09-08-2026/research/research-artifact-policy-code-modifications.md
  - documents/notes/artifact-naming-human-centered.md
---

# Implementation Structure: Artifact Policy Code Modifications

## Summary

The implementation will shorten W&B artifact and run names, preserve full identity in metadata, and retain only root data and small contracts by default.

The work has five phases.

Each phase produces a verifiable result before the next phase starts.

## Request

Create high-level phases, sequential stages inside each phase, and indivisible implementation steps for the code changes identified by the artifact naming and minimal-retention policy.

The structure preserves stable local filenames and avoids changes to model or metric semantics.

## Confirmed context

- `src/engine/logger.py:214-218` sends the supplied artifact name directly to W&B.
- `scripts/cli/train.py:344`, `scripts/cli/evaluate.py:397`, `scripts/experiments/run_online_adaptation.py:203`, and `scripts/experiments/run_ablation.py:149` use the full experiment name in W&B artifact names.
- `src/engine/artifact_sinks.py:50` and `src/engine/artifact_sinks.py:68` use only local path names for W&B artifacts.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:1012-1046` writes large derived outputs without a retention gate.
- `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` defaults to `retain_for_eda`.
- `scripts/benchmarks/run_thesis_online_benchmark.py:315-323` stores full online execution data in the report.
- Existing tests cover W&B logger behavior and summary-only thesis benchmark behavior.

## Scope

### In scope

- Shared short-name construction and validation.
- Direct W&B artifact producers and path-based artifact sinks.
- Explicit W&B run-name configuration that overrides runtime defaults.
- Default retention and export gates.
- Compact reports and manifests.
- Tests and one end-to-end smoke verification.

### Out of scope

- Renaming local files.
- Removing old output trees.
- Changing model, threshold, score, or online algorithms.
- Running the full benchmark matrix.

## Proposed phases

### Phase 1: Contracts and test seams are fixed

**Result:** The codebase has an agreed short-name contract, a root-versus-derived retention contract, and test cases that express the desired behavior.

**Scope:**

- Naming inputs and role vocabulary.
- Root artifacts, small contracts, and derived outputs.
- Compatibility with current local filenames and metadata.

**Depends on:**

- Research evidence in `documents/logs/09-08-2026/research/`.

**Stages:**

1. Define the structured identity fields and allowed artifact roles.
2. Define the retention policy behavior for `summary_only` and `retain_for_eda`.
3. Add focused failing tests at the naming, logger, sink, and export boundaries.

**Verification:**

- Automated: The new tests fail only because the desired behavior is not implemented.
- Manual: A reviewer confirms that local path contracts and full metadata identity remain preserved.

**Risks:**

- A missing identity field can cause two different runs to receive the same W&B name.
- Mitigation: Test different variants, entities, stages, seeds, and online variants.

**Complete when:**

- The naming schema and retention table are explicit.
- Each later implementation boundary has a test seam.

### Phase 2: The shared W&B naming boundary is implemented

**Result:** W&B receives short, validated, role-first names from one shared implementation.

**Scope:**

- Proposed `src/core/artifact_naming.py` helper.
- `ExperimentLogger` file and directory artifact boundaries.
- Checkpoint and output artifact sinks.

**Depends on:**

- Phase 1 naming contract and tests.

**Stages:**

1. Implement the short-name builder and length validation.
2. Route logger artifact calls through the shared boundary.
3. Route path-based sinks through structured identity metadata.

**Verification:**

- Automated: Logger and sink tests inspect generated names and reject names over 128 characters.
- Manual: A fake W&B run shows role, stage, variant, entity, and seed in the displayed name.

**Risks:**

- A central fallback may hide callers that still pass ambiguous names.
- Mitigation: Keep an explicit collision test and search all `artifact_name=` call sites.

**Complete when:**

- No W&B artifact can bypass name validation.
- Path stems are no longer the only identity in sink-generated names.

### Phase 3: All W&B producers use the shared identity

**Result:** Training, evaluation, online adaptation, ablation, two-stage execution, and benchmark configuration use short W&B names without losing full metadata identity.

**Scope:**

- Direct artifact calls.
- Runtime W&B run-name defaults.
- Config generators that explicitly set W&B run names.

**Depends on:**

- Phase 2 shared naming boundary.

**Stages:**

1. Migrate training and evaluation artifact producers.
2. Migrate online adaptation and ablation artifact producers.
3. Migrate two-stage and benchmark W&B run-name configuration.
4. Verify metadata preserves the full experiment name.

**Verification:**

- Automated: Direct call-site tests and W&B fake-run tests pass.
- Manual: One representative W&B-enabled configuration displays a short run name and short artifact names.

**Risks:**

- Explicit config values can override runtime defaults.
- Mitigation: Update generators and search all `wandb_run_name` assignments.

**Complete when:**

- Direct artifact names no longer concatenate the full experiment name.
- Explicit W&B run-name assignments use the same human identity policy.

### Phase 4: Minimal retention is enforced at export boundaries

**Result:** Summary-only runs retain roots and small contracts, while large derived data is written only for explicit EDA retention.

**Scope:**

- Training and evaluation history.
- Thesis offline and online benchmark exports.
- Generic offline and online benchmark outputs.
- Report and manifest payloads.

**Depends on:**

- Phase 1 retention contract.
- Phase 3 preserved identity and stable local path behavior.

**Stages:**

1. Change the default retention policy to `summary_only`.
2. Gate offline score, trace, UQ, and derived metrics writes.
3. Gate online metric and record writes while always retaining the threshold contract.
4. Remove full metric histories and records from reports.
5. Make raw JSONL histories opt-in and align the duplicate offline helper.

**Verification:**

- Automated: Summary-only tests assert root files exist and derived arrays or records do not exist.
- Manual: An EDA run still produces the selected diagnostic files.

**Risks:**

- Existing consumers may expect derived files in every run.
- Mitigation: Keep `retain_for_eda` as an explicit compatibility mode and preserve local basenames.

**Complete when:**

- A default run does not retain large derived outputs.
- A selected EDA run retains the existing diagnostic contract.
- Online summary-only bundles still contain the threshold contract.

### Phase 5: Verification and controlled rollout are complete

**Result:** Tests, static checks, and one real smoke combination demonstrate that naming and retention changes work together.

**Scope:**

- Unit tests.
- Wrapper tests.
- One end-to-end smoke combination.
- Documentation and rollout checks.

**Depends on:**

- Phases 2, 3, and 4.

**Stages:**

1. Run the focused automated test groups.
2. Run one repository-defined end-to-end smoke combination.
3. Perform static call-site and output-tree audits.
4. Update the design note and mark the implementation decision.

**Verification:**

- Automated: Focused pytest commands, smoke exit code 0, and `git diff --check` pass.
- Manual: W&B artifact names remain readable and all retained root files can identify their run.

**Risks:**

- A passing unit test may not catch a caller that overrides a name in a generated config.
- Mitigation: Run static searches and inspect one generated configuration end to end.

**Complete when:**

- The original W&B name failure path completes.
- Summary-only retention is verified.
- The full matrix remains unstarted until the smoke combination passes.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| Phase 1 | Research note and current code evidence | Stable naming and retention contracts. |
| Phase 2 | Phase 1 contracts and tests | Safe shared W&B naming boundary. |
| Phase 3 | Phase 2 helper and boundary | All active W&B producers using short identity. |
| Phase 4 | Phase 1 retention contract and Phase 3 metadata compatibility | Minimal persisted outputs. |
| Phase 5 | Phases 2-4 | Controlled smoke rollout and final acceptance. |

## Decisions confirmed

- Use role-first W&B names.
- Keep full experiment identity in metadata or resolved config.
- Keep stable local basenames.
- Use `summary_only` as the default.
- Keep the online threshold contract in summary-only retention.
- Treat scores, traces, records, UQ, full histories, and duplicated report payloads as derived data.

## Non-blocking uncertainties

- The exact compact token for `dataset_name` can be finalized from active configuration values during Phase 1.
- The internal duplicate offline helper has no active caller in the current search, so its update can follow the active public path.
- The exact final report summary fields depend on the thesis report requirements, but they do not change phase order.

## Feedback requested

- Confirm whether `summary_only` should become the default for generic benchmark scripts as well as thesis benchmark scripts.
- Confirm whether final report-ready metrics should remain as a compact summary artifact or be regenerated from roots when needed.
