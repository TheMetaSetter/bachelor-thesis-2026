---
date: 2026-09-08 Asia/Ho_Chi_Minh
topic: "Detailed implementation instructions for artifact naming and minimal retention"
status: ready
revision: 040f3ce7d1cc23e8cb5735eabe69fe5b1530cdf2
source_structure: documents/logs/09-08-2026/structure/structure-artifact-policy-code-modifications.md
related_documents:
  - documents/logs/09-08-2026/plan/plan-artifact-policy-code-modifications.md
  - documents/logs/09-08-2026/research/research-artifact-policy-code-modifications.md
  - documents/notes/artifact-naming-human-centered.md
---

# Detailed Implementation: Artifact Policy Code Modifications

## Summary

This document expands the proposed structure into implementation steps.

The implementation has two independent contracts.

The naming contract makes every W&B artifact name short, role-first, and human-readable.

The retention contract keeps immutable roots and small runtime contracts by default, and keeps large derived outputs only when explicitly requested.

The implementation preserves stable local filenames and the full experiment identity in metadata or resolved configuration.

## Source structure

The source structure defines five phases.

Phase 1 fixes contracts and test seams.

Phase 2 implements the shared W&B naming boundary.

Phase 3 migrates all W&B producers.

Phase 4 enforces minimal retention.

Phase 5 verifies one safe rollout path.

The user explicitly requested all three planning levels in one turn, so this detail document is based on the proposed structure without treating that structure as previously approved.

## Current state

`src/engine/logger.py:214-218` passes caller-provided artifact names directly to `wandb.Artifact`.

Training, evaluation, online adaptation, and ablation build those names from the full `experiment_name`.

`src/engine/artifact_sinks.py:50` and `src/engine/artifact_sinks.py:68` use only local path names for W&B artifacts.

The thesis offline exporter writes score arrays, traces, UQ summary, metrics, and protocol data at `scripts/benchmarks/run_thesis_offline_benchmark.py:1001-1046`.

The thesis online exporter defaults to `retain_for_eda` and places the complete online execution output in its report.

The generic online streaming benchmark writes and embeds complete records and metric history.

The logger persists raw metric histories to JSONL.

## Desired end state

The shared helper produces names such as `cfg-stageA-O0-machine_1_6-s36` and `ckpt-best-stageB-O0-machine_1_6-s36`.

The helper rejects empty identity fields, unsupported roles, and names longer than 128 characters.

W&B metadata still contains the full `experiment_name`.

Local basenames remain `best.pt`, `evaluation_metrics.json`, and other existing names.

The default retention mode is `summary_only`.

Summary-only output keeps the checkpoint, resolved behavior-affecting config, dataset and protocol references, code provenance, threshold contract when needed, and a compact final summary.

Score arrays, raw traces, per-step records, UQ details, and full metric histories are opt-in.

## Scope

### In scope

- Proposed `src/core/artifact_naming.py`.
- `ExperimentLogger`, `WandbArtifactSink`, and checkpoint metadata passed to sinks.
- Direct W&B artifact calls in training, evaluation, online adaptation, and ablation.
- W&B run-name defaults and explicit generator values.
- Thesis and generic benchmark retention gates.
- Report payload compaction.
- Focused tests and one smoke run.

### Out of scope

- Renaming local files.
- Deleting historical output directories.
- Changing model or metric calculations.
- Changing W&B projects, aliases, or artifact types.
- Running the full experiment matrix.

## Evidence

- `src/engine/logger.py:190-220` — Common file-artifact W&B boundary.
- `src/engine/logger.py:231-261` — Common directory-artifact W&B boundary.
- `src/engine/artifact_sinks.py:38-72` — Path-based W&B sink naming.
- `src/engine/checkpoint.py:217-238` — Checkpoint sink metadata boundary.
- `scripts/cli/train.py:342-362` — Direct training artifact names.
- `scripts/cli/evaluate.py:395-453` — Direct evaluation artifact names and uploads.
- `scripts/experiments/run_online_adaptation.py:201-249` — Direct online artifact names and uploads.
- `scripts/experiments/run_ablation.py:147-174` — Direct ablation artifact names.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:957-1047` — Unconditional offline derived writes.
- `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` — Online retention default.
- `scripts/benchmarks/run_thesis_online_benchmark.py:190-231` — Online retention bundle contents.
- `scripts/benchmarks/run_thesis_online_benchmark.py:315-323` — Full online output in report.
- `scripts/benchmarks/run_online_streaming_benchmark.py:440-473` — Generic online output and report duplication.
- `src/engine/logger.py:142-171` — Raw metric history persistence.

## Interface and data changes

### Proposed naming interface

**Proposed new file:** `src/core/artifact_naming.py`.

The file should expose one identity normalizer and one name builder.

```python
def build_artifact_identity(
    experiment_config: Mapping[str, Any],
    *,
    stage: str | None = None,
    online_variant: str | None = None,
) -> dict[str, str | int]: ...

def build_wandb_artifact_name(
    *,
    role: str,
    identity: Mapping[str, str | int],
) -> str: ...
```

`build_artifact_identity()` should return normalized keys: `dataset`, `variant`, `entity`, `seed`, and optional `stage`, `online_variant`, and `fpr_budget`.

`build_wandb_artifact_name()` should return `<role>-<stage-or-variant>-<entity>-s<seed>` with an online token only when needed.

The helper should map the current config fields without changing their meaning.

The helper should reject missing `dataset`, `variant`, `entity`, or `seed` for artifact roles that require run identity.

The helper should validate the final name length against 128 characters.

The helper should not silently truncate or hash the human-facing name.

### Proposed retention contract

Keep the existing values `summary_only` and `retain_for_eda`.

`summary_only` keeps root artifacts, small contracts, checksums, and compact final summaries.

`retain_for_eda` additionally keeps selected scores, traces, records, UQ details, and metric histories.

The threshold artifact is always retained for an online bundle because the online runtime consumes it.

Full online runtime state is retained only when resume or forensic inspection requires it.

## Phase 1: Contracts and test seams

### Goal

Make the intended naming and retention behavior executable as tests before changing production code.

### Stage 1.1: Define naming behavior in tests

#### Detailed changes

- **File:** `tests/core/test_artifact_naming.py`
- **Status:** Proposed new file.
- **Responsibility:** Specify normalized identity and short W&B name behavior.
- **Inputs:** Dataset `smd`, variant `O0`, entity `machine_1_6`, seed `36`, stage `stageA`, and online variant `A1`.
- **Outputs:** Short role-first names.
- **Errors:** Missing identity and names over 128 characters raise `ValueError`.
- **Compatibility:** No local filename behavior changes.

#### Atomic steps

- [ ] Create `tests/core/test_artifact_naming.py`.
- [ ] Add a test that expects `cfg-stageA-O0-machine_1_6-s36` for a config role.
- [ ] Add a test that expects `ckpt-online-A1-O0-machine_1_6-s36` for an online checkpoint role.
- [ ] Add a test that distinguishes O0 from O1.
- [ ] Add a test that distinguishes `machine_1_6` from `machine_3_9`.
- [ ] Add a test that rejects a missing seed.
- [ ] Add a test that rejects a generated name longer than 128 characters.
- [ ] Run `.venv/bin/python -m pytest tests/core/test_artifact_naming.py -q`.
- [ ] Confirm that the new tests fail because the proposed helper does not yet exist.

#### Complete when

- The tests state the exact name vocabulary, identity fields, collision behavior, and length limit.

### Stage 1.2: Define retention behavior in existing wrapper tests

#### Detailed changes

- **File:** `tests/benchmarks/test_thesis_offline_artifact_exports.py:396-493`.
- **Symbol:** `test_thesis_offline_wrapper_supports_summary_only_retention`.
- **Current responsibility:** Verifies that some EDA files are omitted for explicit `summary_only`.
- **Change:** Add assertions for the new minimal root set and remove the expectation that `uq_summary.json` is always retained if the selected policy excludes it.

- **File:** `tests/benchmarks/test_thesis_online_benchmark_wrapper.py:110-176`.
- **Symbol:** `test_thesis_online_a0_wrapper_can_reduce_retention_to_summary_only`.
- **Current responsibility:** Verifies that online metrics and records are omitted for `summary_only`.
- **Change:** Assert that the threshold contract remains present and that complete online execution data is absent from the saved report.

#### Atomic steps

- [ ] Add an assertion that the offline summary-only bundle keeps the threshold artifact or its root reference.
- [ ] Add an assertion that the offline summary-only bundle omits score NPZ files.
- [ ] Add an assertion that the offline summary-only bundle omits trace JSON files.
- [ ] Add an assertion that the online summary-only bundle keeps `threshold_artifact.json`.
- [ ] Add an assertion that the online summary-only report does not contain `metric_history`.
- [ ] Add an assertion that the online summary-only report does not contain `records`.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py tests/benchmarks/test_thesis_online_benchmark_wrapper.py -q`.
- [ ] Record the failing assertions before production changes.

#### Complete when

- The tests distinguish root and contract files from large derived files.

### Stage 1.3: Fix the compatibility boundary before implementation

#### Detailed changes

- **File:** `documents/logs/09-08-2026/plan/plan-artifact-policy-code-modifications.md`.
- **Responsibility:** Preserve implementation constraints for later stages.
- **Change:** Treat local basenames, W&B artifact types, aliases, metadata identity, and explicit `retain_for_eda` as compatibility requirements.

#### Atomic steps

- [ ] Record `best.pt` as a stable local checkpoint basename.
- [ ] Record `evaluation_metrics.json` as a stable local evaluation basename.
- [ ] Record `experiment_name` metadata as retained provenance.
- [ ] Record `retain_for_eda` as the explicit compatibility mode.
- [ ] Record historical output deletion as out of scope.

#### Complete when

- A later implementation can change W&B names and retention without renaming or deleting existing local artifacts.

## Phase 2: Shared W&B naming boundary

### Goal

Create one validated name implementation and make common W&B boundaries enforce it.

### Stage 2.1: Implement the naming helper

#### Detailed changes

- **File:** `src/core/artifact_naming.py`.
- **Status:** Proposed new file.
- **Symbol:** `build_artifact_identity()`.
- **Current responsibility:** No shared implementation exists.
- **Change:** Extract dataset, variant, entity, seed, stage, online variant, and optional FPR identity from the existing config shapes.
- **Inputs:** A mapping containing the current experiment fields.
- **Outputs:** A normalized identity mapping.
- **Errors:** Raise `ValueError` when a required identity field is absent or empty.
- **Dependencies:** Existing config keys in `scripts/cli/train.py`, `scripts/cli/evaluate.py`, and benchmark configs.

- **Symbol:** `build_wandb_artifact_name()`.
- **Change:** Compose a short role-first name and validate its length.
- **Inputs:** A role and normalized identity.
- **Outputs:** A string no longer than 128 characters.
- **Errors:** Raise `ValueError` for an unsupported role, missing required identity, or excessive length.

#### Atomic steps

- [ ] Add the proposed helper file.
- [ ] Define the allowed role tokens used by current producers.
- [ ] Implement identity extraction for dataset and entity fields.
- [ ] Implement identity extraction for variant, seed, and stage fields.
- [ ] Implement optional online-variant and FPR tokens.
- [ ] Implement role-first name composition.
- [ ] Implement the 128-character validation.
- [ ] Run `.venv/bin/python -m pytest tests/core/test_artifact_naming.py -q`.
- [ ] Confirm that all naming tests pass.

#### Complete when

- One helper creates short names for config, metrics, checkpoint, evaluation, online, threshold, report, and bundle roles.

### Stage 2.2: Validate names at the logger boundary

#### Detailed changes

- **File:** `src/engine/logger.py:190-220`.
- **Symbol:** `ExperimentLogger.log_artifact_file()`.
- **Current responsibility:** Sends the caller's name directly to `wandb.Artifact`.
- **Change:** Validate `artifact_name` immediately before line 214.
- **Inputs:** Existing `artifact_name`, artifact type, metadata, and file path.
- **Outputs:** The same artifact upload with a validated name.
- **Errors:** Raise a clear `ValueError` before W&B receives an invalid name.
- **Compatibility:** Keep the method's existing parameters and file basename.

- **File:** `src/engine/logger.py:231-261`.
- **Symbol:** `ExperimentLogger.log_artifact_directory()`.
- **Current responsibility:** Sends directory artifact names directly to W&B.
- **Change:** Apply the same validation before line 255.

#### Atomic steps

- [ ] Import the validation function into `src/engine/logger.py`.
- [ ] Validate file artifact names before constructing `wandb.Artifact`.
- [ ] Validate directory artifact names before constructing `wandb.Artifact`.
- [ ] Add a fake-W&B test for a valid short name.
- [ ] Add a fake-W&B test for a name over 128 characters.
- [ ] Run `.venv/bin/python -m pytest tests/runtime/test_logger_wandb.py -q`.
- [ ] Confirm invalid names fail before the fake artifact is recorded.

#### Complete when

- Both W&B boundaries enforce the same length and format contract.

### Stage 2.3: Route checkpoint and output sinks through run identity

#### Detailed changes

- **File:** `src/engine/artifact_sinks.py:34-72`.
- **Symbol:** `WandbArtifactSink`.
- **Current responsibility:** Uses `path_obj.stem` or `path_obj.name` as the W&B artifact name.
- **Change:** Accept normalized run identity and build a role-first name for checkpoint and output artifacts.
- **Inputs:** Sink artifact type, path, and identity metadata.
- **Outputs:** A unique short W&B artifact name.
- **Errors:** Reject missing identity when W&B sink is active.
- **Dependencies:** `ExperimentLogger.build_artifact_sinks()` and `CheckpointManager._sync_artifacts()`.
- **Compatibility:** Keep local path names and Kaggle upload behavior unchanged.

- **File:** `src/engine/checkpoint.py:217-238`.
- **Symbol:** `CheckpointManager._sync_artifacts()`.
- **Current responsibility:** Passes epoch, checkpoint name, and full experiment name to sinks.
- **Change:** Pass normalized identity fields needed by the W&B sink.
- **Outputs:** The checkpoint remains uploaded with the same aliases.

#### Atomic steps

- [ ] Add the normalized identity field to the W&B sink construction path.
- [ ] Replace `path_obj.stem` with a checkpoint-role name.
- [ ] Replace `path_obj.name` with an output-role name.
- [ ] Preserve `experiment_name` in sink metadata.
- [ ] Keep Kaggle sink calls unchanged.
- [ ] Extend `tests/runtime/test_artifact_sink_selection.py` with a name-capture fake logger.
- [ ] Run `.venv/bin/python -m pytest tests/runtime/test_artifact_sink_selection.py -q`.
- [ ] Confirm checkpoint names differ for two variants or entities.

#### Complete when

- Sink-generated W&B names contain role and run identity without using the local path as the only identity.

## Phase 3: Migrate all W&B producers

### Goal

Remove direct long-name construction while preserving full metadata identity.

### Stage 3.1: Migrate training and evaluation

#### Detailed changes

- **File:** `scripts/cli/train.py:342-362`.
- **Symbol:** `run_training_experiment()` artifact calls.
- **Change:** Replace the three `f"{experiment_name}-..."` expressions with `build_wandb_artifact_name()` calls for config, metrics, and best checkpoint.
- **Inputs:** The existing experiment config and stage identity.
- **Outputs:** Short W&B names.
- **Compatibility:** Keep aliases `latest` and `best`; keep metadata `experiment_name`.

- **File:** `scripts/cli/evaluate.py:395-453`.
- **Symbols:** Evaluation artifact calls.
- **Change:** Use short names for resolved config, evaluation metrics, records, curves, traces, and protocol audit.
- **Compatibility:** Preserve local output paths and artifact type `evaluation`.

#### Atomic steps

- [ ] Import the shared name builder into `scripts/cli/train.py`.
- [ ] Replace the training config artifact name.
- [ ] Replace the training metrics artifact name.
- [ ] Replace the training checkpoint artifact name.
- [ ] Import the shared name builder into `scripts/cli/evaluate.py`.
- [ ] Replace the six evaluation artifact names.
- [ ] Preserve full `experiment_name` in every metadata mapping.
- [ ] Run `.venv/bin/python -m pytest tests/runtime/test_logger_wandb.py tests/runtime/test_artifact_sink_selection.py -q`.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py -q`.

#### Complete when

- Training and evaluation no longer construct W&B artifact names from the full experiment name.

### Stage 3.2: Migrate online adaptation and ablation

#### Detailed changes

- **File:** `scripts/experiments/run_online_adaptation.py:201-249`.
- **Symbol:** `run_online_adaptation_experiment()` artifact calls.
- **Change:** Use `cfg`, `met`, `on-met`, `on-records`, and `ckpt-online` roles.
- **Compatibility:** Keep the existing aliases and full metadata identity.

- **File:** `scripts/experiments/run_ablation.py:147-174`.
- **Symbol:** `run_ablation_suite()` artifact calls.
- **Change:** Use `cfg`, `met`, and `ablation` roles for JSON and CSV summaries.
- **Compatibility:** Preserve the suite job type and summary files.

#### Atomic steps

- [ ] Import the shared name builder into `scripts/experiments/run_online_adaptation.py`.
- [ ] Replace the five online adaptation artifact names.
- [ ] Import the shared name builder into `scripts/experiments/run_ablation.py`.
- [ ] Replace the four ablation artifact names.
- [ ] Preserve the full experiment name in metadata.
- [ ] Run `.venv/bin/python -m pytest tests/online/test_online_entrypoint.py -q`.
- [ ] Run the relevant ablation test file if one exists for the changed runner.

#### Complete when

- Online adaptation and ablation use the shared role-first naming contract.

### Stage 3.3: Migrate W&B run-name producers

#### Detailed changes

- **Files:** `scripts/cli/train.py:262`, `scripts/cli/evaluate.py:303-305`, `scripts/experiments/run_online_adaptation.py:128-132`, `scripts/experiments/run_ablation.py:95-96`, `scripts/experiments/run_two_stage_offline_pretraining.py:155-158`, `scripts/benchmarks/run_thesis_online_benchmark.py:282-289`, and `scripts/benchmarks/run_online_streaming_benchmark.py:271-278`.
- **Current responsibility:** Default W&B run names from full experiment or benchmark names.
- **Change:** Use the compact run identity helper.

- **Files:** `scripts/benchmarks/generate_benchmark_smoke_configs.py:69`, `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:102`, `scripts/benchmarks/generate_pro_reconstruction_vus_budget_matrix.py:118`, `scripts/benchmarks/generate_smd_benchmark_configs.py:100`, `scripts/benchmarks/generate_online_benchmark_configs.py:224`, and `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:226`.
- **Current responsibility:** Write explicit W&B run names into generated configs.
- **Change:** Generate compact run names so they do not override the runtime helper.

#### Atomic steps

- [ ] Replace the training W&B run-name default.
- [ ] Replace the evaluation W&B run-name default.
- [ ] Replace the online adaptation W&B run-name default.
- [ ] Replace the ablation and two-stage W&B run-name defaults.
- [ ] Replace the thesis and generic benchmark W&B run-name defaults.
- [ ] Update each explicit config generator value.
- [ ] Generate one representative config and inspect its run name.
- [ ] Search with `rg -n --glob '*.py' "wandb_run_name" scripts src`.
- [ ] Confirm no active runtime assignment uses the full experiment name as the default.

#### Complete when

- Explicit configuration and runtime defaults agree on short human-facing W&B names.

## Phase 4: Enforce minimal retention

### Goal

Make summary-only the default and prevent large derived outputs from being persisted or duplicated.

### Stage 4.1: Change retention defaults

#### Detailed changes

- **Files:** `scripts/benchmarks/run_thesis_offline_benchmark.py:213-215`, `scripts/benchmarks/run_thesis_online_benchmark.py:115-117`, and `src/core/artifact_integrity.py:47-55`.
- **Current responsibility:** Default retention to `retain_for_eda`.
- **Change:** Default to `summary_only` or require callers to pass an explicit policy.
- **Compatibility:** Keep `retain_for_eda` as an explicit value.

- **Files:** `scripts/benchmarks/generate_online_benchmark_configs.py:217`, `scripts/benchmarks/generate_smd_benchmark_configs.py:180`, and generated configs with `retention_policy: retain_for_eda`.
- **Change:** Change the default-generated configs to `summary_only` when minimal retention is the desired project default.

#### Atomic steps

- [ ] Change the thesis offline resolver default to `summary_only`.
- [ ] Change the thesis online resolver default to `summary_only`.
- [ ] Align the integrity helper default or require an explicit policy at each call.
- [ ] Change active config generators that explicitly set `retain_for_eda`.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py tests/benchmarks/test_thesis_online_benchmark_wrapper.py -q`.
- [ ] Confirm explicit `retain_for_eda` tests still select EDA output.

#### Complete when

- An omitted retention setting selects `summary_only`.

### Stage 4.2: Gate thesis offline derived writes

#### Detailed changes

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py:957-1047`.
- **Symbol:** `_export_offline_artifacts()`.
- **Current responsibility:** Writes threshold, UQ summary, score arrays, traces, metrics, and resolved protocol.
- **Change:** Add `retention_policy` to the function input.
- **Inputs:** Existing artifact inputs, protocol, manifest, config paths, and selected retention policy.
- **Outputs:** Always write threshold and protocol roots; write large derived outputs only for `retain_for_eda`.
- **Errors:** Reject unsupported retention values using the existing validation rule.
- **Compatibility:** Keep return keys stable for retained files or return only paths that actually exist and update callers together.

- **File:** `scripts/benchmarks/run_thesis_offline_benchmark.py:1141-1169`.
- **Symbol:** `run_thesis_offline_benchmark()`.
- **Change:** Pass `retention_policy` into `_export_offline_artifacts()`.

#### Atomic steps

- [ ] Add the retention-policy parameter to `_export_offline_artifacts()`.
- [ ] Keep threshold artifact writing outside the EDA-only branch.
- [ ] Keep resolved protocol writing with the retained root configuration.
- [ ] Move score NPZ writes inside the `retain_for_eda` branch.
- [ ] Move trace JSON writes inside the `retain_for_eda` branch.
- [ ] Move UQ summary writing inside the selected derived-summary branch.
- [ ] Move offline metrics writing inside the selected report-retention branch.
- [ ] Pass the resolved policy from `run_thesis_offline_benchmark()`.
- [ ] Update `tests/benchmarks/test_thesis_offline_artifact_exports.py` to assert actual file existence.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py -q`.

#### Complete when

- Summary-only offline execution does not write score arrays or full traces.
- EDA retention still writes the selected files.

### Stage 4.3: Gate thesis online retention

#### Detailed changes

- **File:** `scripts/benchmarks/run_thesis_online_benchmark.py:172-253`.
- **Symbol:** `_export_online_retention_bundle()`.
- **Current responsibility:** Writes summary always and metrics, records, threshold artifact, and runtime state only for EDA.
- **Change:** Always write the small threshold contract; keep metrics and records EDA-only; keep runtime state only when explicitly required.
- **Outputs:** Summary-only bundle contains compact summary, threshold contract, and manifest.
- **Compatibility:** Keep entity and online-variant directory layout.

- **File:** `scripts/benchmarks/run_thesis_online_benchmark.py:315-323`.
- **Symbol:** `run_thesis_online_benchmark()` report creation.
- **Current responsibility:** Stores complete `online_outputs` in the report.
- **Change:** Replace it with compact counts, final metrics, threshold identity, checkpoint identity, and retained paths.

#### Atomic steps

- [ ] Move threshold artifact writing outside the EDA-only condition.
- [ ] Keep online metrics writing inside the EDA-only condition.
- [ ] Keep online records writing inside the EDA-only condition.
- [ ] Keep runtime state writing conditional on resume or forensic retention.
- [ ] Replace `report["online_execution"]` with a compact summary.
- [ ] Remove `metric_history` from the persisted report.
- [ ] Remove `records` from the persisted report.
- [ ] Update `tests/benchmarks/test_thesis_online_benchmark_wrapper.py` for threshold retention.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_online_benchmark_wrapper.py -q`.

#### Complete when

- Summary-only online bundles retain the threshold contract without retaining complete online history.

### Stage 4.4: Gate generic benchmark outputs

#### Detailed changes

- **File:** `scripts/benchmarks/run_offline_benchmark.py:452-471`.
- **Current responsibility:** Always writes thresholds, three score NPZ files, and offline metrics.
- **Change:** Apply the same retention policy boundary while preserving the generic benchmark's current metric semantics.

- **File:** `scripts/benchmarks/run_online_streaming_benchmark.py:440-473`.
- **Current responsibility:** Always writes online metrics and records and embeds both in the report.
- **Change:** Gate file writes and remove complete arrays from `report["online_execution"]`.

- **File:** `scripts/experiments/run_online_adaptation.py:182-239`.
- **Current responsibility:** Always writes and uploads online metrics and records.
- **Change:** Use the configured retention policy to keep only a compact summary and checkpoint by default.

#### Atomic steps

- [ ] Add the existing retention-policy read to the generic offline benchmark path.
- [ ] Gate generic offline score NPZ writes.
- [ ] Gate generic offline metric file writes when summary-only does not require them.
- [ ] Add the existing retention-policy read to the generic online streaming path.
- [ ] Gate generic online metric writes.
- [ ] Gate generic online record writes.
- [ ] Remove generic online metric history from the persisted report.
- [ ] Remove generic online records from the persisted report.
- [ ] Gate direct online adaptation metric and record writes.
- [ ] Gate direct W&B uploads for those derived files.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_run_offline_benchmark.py tests/online/test_online_streaming_benchmark_wrapper.py tests/online/test_online_entrypoint.py -q`.

#### Complete when

- Generic benchmark and direct online paths follow the same root-versus-derived rule.

### Stage 4.5: Reduce raw logging and align duplicate helpers

#### Detailed changes

- **File:** `src/engine/logger.py:142-171`.
- **Symbols:** `log_metrics()` and `log_focused_metrics()`.
- **Current responsibility:** Persist every metric event to JSONL when called.
- **Change:** Make raw JSONL history opt-in and keep compact final summaries through `log_summary()`.
- **Compatibility:** Preserve W&B scalar logging when W&B is enabled and preserve explicit raw-history mode.

- **File:** `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:189-191` and `542-590`.
- **Current responsibility:** Contains a duplicate retention implementation with the same EDA default and unconditional compact summary writes.
- **Change:** Align the default and retention decisions if a future caller uses this helper.
- **Active status:** The current public runner imports its retention exporter, but the duplicate internal artifact exporter has no active caller in the current search.

#### Atomic steps

- [ ] Add an explicit raw-history setting to the existing logging configuration contract.
- [ ] Set its default to disabled for summary-only runs.
- [ ] Guard JSONL writes in `log_metrics()`.
- [ ] Guard focused JSONL writes in `log_focused_metrics()`.
- [ ] Preserve W&B scalar logging for the compact summary path.
- [ ] Align the internal helper's retention default.
- [ ] Align the internal helper's UQ and derived-file conditions.
- [ ] Run `.venv/bin/python -m pytest tests/runtime/test_logger_wandb.py tests/core/test_uq_summary.py -q`.
- [ ] Search for every raw history writer with `rg -n --glob '*.py' "metrics_path|focused_metrics_path|metric_history|online_records" src scripts`.

#### Complete when

- Default execution no longer creates full raw histories unless the caller explicitly requests them.

## Phase 5: Verification and controlled rollout

### Goal

Prove the naming and retention contracts independently, then prove one complete flow.

### Stage 5.1: Run focused automated tests

#### Detailed changes

- **Files:** `tests/core/test_artifact_naming.py`, `tests/runtime/test_logger_wandb.py`, `tests/runtime/test_artifact_sink_selection.py`, `tests/benchmarks/test_thesis_offline_artifact_exports.py`, `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`, `tests/benchmarks/test_run_offline_benchmark.py`, `tests/online/test_online_streaming_benchmark_wrapper.py`, and `tests/online/test_online_entrypoint.py`.
- **Responsibility:** Verify helper, W&B boundaries, retention gates, and compatibility.

#### Atomic steps

- [ ] Run `.venv/bin/python -m pytest tests/core/test_artifact_naming.py -q`.
- [ ] Run `.venv/bin/python -m pytest tests/runtime/test_logger_wandb.py tests/runtime/test_artifact_sink_selection.py -q`.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_thesis_offline_artifact_exports.py tests/benchmarks/test_thesis_online_benchmark_wrapper.py -q`.
- [ ] Run `.venv/bin/python -m pytest tests/benchmarks/test_run_offline_benchmark.py tests/online/test_online_streaming_benchmark_wrapper.py tests/online/test_online_entrypoint.py -q`.
- [ ] Record any failing test node before changing another component.

#### Complete when

- All focused tests pass with no unexamined failure.

### Stage 5.2: Run one end-to-end smoke combination

#### Detailed changes

- **Entry point:** The repository's existing thesis benchmark smoke command that invokes one concrete configuration.
- **Responsibility:** Verify training, artifact naming, retention, and report generation together.
- **Constraint:** Do not run the full matrix before this combination passes.

#### Atomic steps

- [ ] Select one existing smoke configuration with W&B disabled for the first local functional check.
- [ ] Run the existing smoke command using `.venv/bin/python`.
- [ ] Inspect the resulting output directory.
- [ ] Confirm the selected checkpoint and resolved config exist.
- [ ] Confirm summary-only output does not contain unrequested score, trace, or record files.
- [ ] Run one W&B fake-boundary or offline-mode check with the same identity.
- [ ] Confirm every generated W&B artifact name is at most 128 characters.

#### Complete when

- One real flow completes and its output tree matches the root-versus-derived policy.

### Stage 5.3: Run static call-site and artifact audits

#### Atomic steps

- [ ] Search with `rg -n --glob '*.py' "artifact_name=f" scripts src`.
- [ ] Confirm no active result uses the full `experiment_name` as an artifact-name prefix.
- [ ] Search with `rg -n --glob '*.py' "wandb_run_name" scripts src`.
- [ ] Inspect every remaining explicit W&B run-name assignment.
- [ ] Search with `rg -n --glob '*.py' "retain_for_eda|summary_only" scripts src configs`.
- [ ] Confirm every `retain_for_eda` value is explicit or intentionally preserved for compatibility.
- [ ] Search with `rg -n --glob '*.py' "metric_history|online_records|evaluation_traces|point_scores\\.npz" scripts src`.
- [ ] Confirm each writer has a retention decision.
- [ ] Run `git diff --check`.

#### Complete when

- Static searches show no unreviewed name or retention bypass.

### Stage 5.4: Update documentation and decision status

#### Detailed changes

- **File:** `documents/notes/artifact-naming-human-centered.md`.
- **Change:** Update the decision record with the final helper vocabulary, default retention mode, and any intentional exceptions.

- **Files:** The plan, structure, detail, and research documents in `documents/logs/09-08-2026/`.
- **Change:** Record final test commands, smoke result, unresolved non-blocking uncertainties, and the implementation revision.

#### Atomic steps

- [ ] Update the naming note's decision record.
- [ ] Update the retention note with the final root set.
- [ ] Record the focused test commands and results.
- [ ] Record the smoke configuration and result.
- [ ] Record any retained compatibility exception.
- [ ] Run `git diff --check` again.

#### Complete when

- The documents describe the implemented behavior rather than only the proposal.

## Testing strategy

Unit tests cover name normalization, allowed roles, collision resistance, and length validation.

Logger tests cover the W&B boundary and invalid names.

Sink tests cover role-plus-run identity for checkpoint and directory artifacts.

Wrapper tests cover summary-only and EDA retention outputs.

The smoke run covers the complete path from configuration to retained output.

No test should require the W&B network when a fake W&B module or offline mode can prove the behavior.

## Deployment and rollout

Apply the naming helper and tests before changing generated configurations.

Run one local smoke combination before enabling any W&B online run.

Update generated configs only after runtime defaults and direct callers use the shared helper.

Keep `retain_for_eda` available for existing consumers during migration.

Do not delete old output files or old W&B artifacts.

After one smoke combination passes, run a single W&B-enabled representative combination if the environment allows it.

Only then consider a larger benchmark matrix.

## Risks and recovery

### Name collision

**Cause:** Omitting variant, entity, stage, or seed from a W&B name.

**Impact:** Humans may confuse artifacts from different runs.

**Mitigation:** Require those fields in the helper and test pairwise differences.

**Verification:** Run the collision tests and inspect one generated name per identity dimension.

**Recovery:** Restore the previous caller name temporarily while fixing the helper identity extraction.

### Lost derived diagnostics

**Cause:** Making `summary_only` the default for a consumer that expects score or trace files.

**Impact:** An existing analysis command may fail to find optional files.

**Mitigation:** Preserve `retain_for_eda` and update consumers to request it explicitly.

**Verification:** Run the existing EDA tests and one explicit `retain_for_eda` smoke.

**Recovery:** Set the affected configuration to `retain_for_eda` without deleting root artifacts.

### Incomplete report after compaction

**Cause:** Removing full execution arrays without retaining the fields needed for the thesis report.

**Impact:** The final report may lack a required summary metric.

**Mitigation:** Define compact report fields before removing arrays and test their presence.

**Verification:** Inspect the final report schema in wrapper tests.

**Recovery:** Re-run evaluation from retained roots with explicit EDA retention.

## Final verification

- [ ] The direct training traceback no longer reaches W&B with a long artifact name.
- [ ] All W&B artifact names are short and role-first.
- [ ] Full experiment identity remains in metadata or resolved config.
- [ ] Local filenames remain compatible.
- [ ] Default retention is `summary_only`.
- [ ] Threshold contracts remain available for online execution.
- [ ] Large derived files are opt-in.
- [ ] Reports do not contain complete records or metric histories.
- [ ] Focused tests pass.
- [ ] One smoke combination passes.
- [ ] `git diff --check` passes.

## Assumptions and non-blocking uncertainties

- The current `retention_policy` values remain the only policy values.
- The final compact dataset token will be selected from active config fields during implementation.
- The exact set of report-ready scalar metrics may depend on thesis reporting needs.
- The internal duplicate offline helper may be removed or aligned after a separate caller audit; it is not the first active path to change.
