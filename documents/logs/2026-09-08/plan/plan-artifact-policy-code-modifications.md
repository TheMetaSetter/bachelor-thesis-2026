# Artifact Policy Code Modifications Implementation Plan

> **For agentic workers:** Read the related research and structure documents before implementation. Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` when executing this plan task by task.

**Goal:** Make W&B artifact names short and human-readable, and make artifact retention keep only immutable roots and small runtime contracts by default.

**Architecture:** Add one shared naming helper that receives structured run identity and artifact role. Keep local filenames stable, keep full identity in metadata or resolved config, and gate large derived files with the existing retention-policy concept.

**Tech Stack:** Python, W&B, JSON, YAML, NumPy NPZ files, pytest, and the repository `.venv`.

**Spec:** `documents/notes/artifact-naming-human-centered.md`

## Global Constraints

- W&B artifact names must remain below the 128-character limit.
- Human-facing names keep only role, stage or online variant, model variant, entity, and seed.
- The full `experiment_name` remains in metadata or a retained root config.
- Stable local basenames such as `best.pt` and `evaluation_metrics.json` remain unchanged.
- Retain checkpoints, resolved behavior-affecting config, dataset reference, protocol, code revision, and threshold contract when needed downstream.
- Do not retain score arrays, raw traces, records, full metric histories, or duplicated report payloads by default.
- Use `summary_only` as the default retention policy unless a caller explicitly requests EDA data.
- Keep the threshold artifact in summary-only online bundles because the online runtime consumes it directly.
- Use `.venv/bin/python -m pytest` for project tests.
- Run one concrete end-to-end smoke combination before any benchmark matrix.

---

## Summary

The reported failure occurs because training, evaluation, online adaptation, and ablation scripts pass the full `experiment_name` into `wandb.Artifact`.

The current W&B boundary does not shorten or validate that value.

The current benchmark paths also persist derived data such as scores, traces, records, and metric history without one consistent default gate.

The implementation will preserve existing local paths and metadata while changing W&B naming and default persistence behavior.

## Request

Read `prompts/2_plan_prompt.md`, `prompts/3_structure_prompt.md`, and `prompts/4_detail_prompt.md`.

Create high-level phases, sequential stages inside each phase, and indivisible atomic steps for the code modifications identified in the artifact naming and minimal-retention research.

Do not modify source code while preparing the plan.

## Current state

The training path calls `ExperimentLogger.log_artifact_file()` with names such as `f"{experiment_name}-resolved-config"`.

The logger passes the value directly to `wandb.Artifact(name=...)`.

The same pattern appears in evaluation, online adaptation, and ablation paths.

Checkpoint and output sinks use only a local path stem or directory name.

The thesis offline exporter writes score NPZ files, traces, UQ summary, metrics, and resolved protocol files unconditionally.

The thesis online exporter defaults to `retain_for_eda` and stores the full online execution payload in its report.

Existing tests already cover summary-only behavior in parts of the thesis benchmark wrappers, so the plan extends those tests instead of replacing their contracts.

## Desired end state

Every W&B artifact uses a short role-first name such as `ckpt-best-stageA-O0-machine_1_6-s36`.

Every W&B artifact name is validated before W&B receives it.

The full experiment identity remains available in metadata and the resolved config.

Local filenames and path-based consumers remain compatible.

Summary-only runs retain only root files, small contracts, and compact report summaries.

EDA runs retain selected derived diagnostics only when explicitly configured.

Reports do not duplicate complete metric histories or online records.

## Scope

### In scope

- A shared short W&B artifact-name contract.
- Direct W&B artifact call sites in training, evaluation, online adaptation, and ablation.
- W&B checkpoint and output sinks.
- W&B run-name consistency where explicit configuration would override runtime defaults.
- Retention defaults and write gates in thesis and generic benchmark paths.
- Compact online and offline reports.
- Focused unit, wrapper, and one end-to-end smoke verification.

### Out of scope

- Renaming stable local output files.
- Deleting existing historical output trees.
- Changing model, metric, threshold, or online algorithm semantics.
- Changing the W&B project, entity, aliases, or artifact types.
- Running the full experiment matrix.
- Cleaning unrelated existing worktree changes.

## Evidence

- `src/engine/logger.py:190-220` — The logger sends caller-provided artifact names to W&B.
- `src/engine/artifact_sinks.py:48-72` — Checkpoint and directory sinks derive names only from local paths.
- `scripts/cli/train.py:342-362` — Training creates long artifact names from the full experiment name.
- `scripts/cli/evaluate.py:395-453` — Evaluation creates long names and uploads derived files.
- `scripts/experiments/run_online_adaptation.py:201-249` — Online adaptation creates long names and uploads histories.
- `scripts/experiments/run_ablation.py:147-174` — Ablation creates long names for summaries.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:957-1047` — Offline export writes derived files unconditionally.
- `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` — Online retention defaults to `retain_for_eda`.
- `scripts/benchmarks/run_thesis_online_benchmark.py:315-323` — Online reports include the complete execution payload.
- `scripts/benchmarks/run_online_streaming_benchmark.py:440-473` — Streaming benchmark writes and embeds complete online history.
- `src/engine/logger.py:142-171` — Raw metric histories are persisted to JSONL.
- `tests/runtime/test_logger_wandb.py:42-84` — Existing W&B logger test seam.
- `tests/benchmarks/test_thesis_offline_artifact_exports.py:396-493` — Existing offline summary-only test seam.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py:110-176` — Existing online summary-only test seam.

## Implementation approach

Use one small helper for names instead of duplicating string construction in each caller.

Use the existing `retention_policy` values rather than introducing another retention flag.

Keep the current local output contract and place the policy gate at export boundaries.

Keep full identity in metadata, while using structured fields for W&B names.

Change the smallest number of interfaces needed: add the naming helper, thread retention policy into offline export, and compact report payloads.

## Phase 1: Freeze contracts and test seams

Define the short-name fields, root-retention fields, compatibility rules, and failing tests before implementation.

## Phase 2: Implement the shared W&B naming boundary

Add and validate the shared builder, then make logger and artifact sinks use it.

## Phase 3: Migrate all W&B producers

Replace direct long-name construction and explicit long W&B run-name configuration across active producers.

## Phase 4: Enforce minimal retention

Make `summary_only` the default, keep root contracts, gate large derived outputs, and remove duplicated arrays from reports.

## Phase 5: Verify and roll out safely

Run focused tests, one end-to-end smoke combination, static searches, and documentation checks before any matrix run.

## Testing strategy

Add unit coverage for name format, identity fields, collision resistance, and the 128-character limit.

Extend logger and sink tests to inspect the actual W&B artifact names.

Extend offline, thesis online, generic online, generic offline, and direct online tests to verify summary-only output.

Keep existing tests for `retain_for_eda` so the explicit inspection mode remains supported.

Run the focused test files with `.venv/bin/python -m pytest`.

Run one repository-defined smoke command from the benchmark workflow after the focused tests pass.

## Migration and rollback

The migration changes W&B names but preserves local file basenames and the full experiment identity in metadata.

Existing W&B artifacts remain readable because aliases and artifact types do not change.

If a consumer still expects a derived file, set the explicit EDA retention policy for that run while migrating the consumer.

Rollback can restore the previous retention default and caller names without changing local checkpoint files.

Do not delete historical outputs as part of this change.

## Documentation

Update the artifact naming note if the final helper vocabulary or retention contract differs from the current proposal.

Add the selected default and the compatibility boundary to the implementation report after verification.

## Final verification

- [x] All direct `artifact_name` constructions no longer prefix the full `experiment_name`.
- [x] The common W&B boundary rejects or derives invalid names before `wandb.Artifact`.
- [x] The default retention path does not persist large derived files.
- [x] Summary-only online bundles retain the threshold contract.
- [x] Reports no longer duplicate complete online records or metric histories.
- [x] Focused tests pass.
- [x] One end-to-end smoke combination passes.
- [x] `git diff --check` passes.

## Assumptions and non-blocking uncertainties

- The existing `retention_policy` values remain the compatibility interface.
- The exact final short-name token for a dataset may use the current `dataset_name` or an existing compact dataset token after checking active configs.
- The generic benchmark scripts may be retained for backward compatibility, so their output gates should be implemented rather than removed.
- The internal duplicate offline helper has no active caller in the current search, but it should be aligned if a future entry point uses it.

## Implementation result

The plan was executed through all five phases.

The shared naming helper, direct W&B producers, retention gates, report compaction, focused tests, and smoke checks were implemented.

The final verification commands and their results are recorded in the detail log.
