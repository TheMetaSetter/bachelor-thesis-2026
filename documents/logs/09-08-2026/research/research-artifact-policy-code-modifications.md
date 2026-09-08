---
date: 2026-09-08 Asia/Ho_Chi_Minh
researcher: OpenAI Codex
topic: "Detect code lines that need modification for the artifact naming and minimal-retention policy"
status: complete
revision: 040f3ce7d1cc23e8cb5735eabe69fe5b1530cdf2
branch: dev
---

# Research: Artifact Policy Code Modifications

## Summary

The W&B failure comes from 17 call sites that build `artifact_name` by prefixing the full `experiment_name`.

The central W&B sink also accepts the name without validating or deriving a short name.

The minimal-retention policy has a separate issue: several benchmark and evaluation paths persist derived scores, traces, records, UQ summaries, and full execution payloads by default.

The first implementation should add one shared short-name builder, route every W&B artifact through it, and keep the full experiment identity only in metadata or a root config.

The second implementation should make root artifacts and small contracts unconditional, and make large derived outputs conditional on an explicit retention policy.

No source code was changed during this research.

## Research question

Which lines of code must change to follow the naming and minimal-retention policy in `documents/notes/artifact-naming-human-centered.md`?

## System context

Training, evaluation, online adaptation, and benchmark scripts create local files and optionally send them to W&B.

`ExperimentLogger.log_artifact_file()` and `ExperimentLogger.log_artifact_directory()` are the common W&B boundaries.

Some scripts call these methods directly with a composed artifact name.

Checkpoint and output sinks derive W&B names from local path stems or directory names.

The benchmark scripts also write derived output files before or independently of retention-bundle selection.

## Confirmed execution path for the reported failure

The benchmark runner launches `scripts.train` with a generated Stage-A config.

The training CLI creates an `ExperimentLogger` and reaches the artifact calls in `scripts/cli/train.py`.

The call at `scripts/cli/train.py:342-348` passes a name formed from the full `experiment_name`.

`ExperimentLogger.log_artifact_file()` passes that string unchanged to `wandb.Artifact(name=...)` at `src/engine/logger.py:214-218`.

W&B validates the name and raises `ValueError` before the file is uploaded.

This confirms that the error is caused by the artifact-name construction, not by the checkpoint file or the training computation.

## Findings: mandatory naming changes

### 1. Add one shared short-name builder

There is no shared builder for human-facing W&B artifact names.

Add one small helper, preferably under `src/core/`, that receives structured fields and an artifact role.

The helper should build names such as `ckpt-best-stageA-O0-machine_1_6-s36`.

It should use only the human-important fields: role, stage or online variant, model variant, entity, and seed.

It should validate the final length before the W&B call.

It should not truncate the full `experiment_name`.

The full `experiment_name` should remain in metadata and the resolved config.

### 2. Validate at the common W&B boundary

Modify these lines so the common boundary either receives a validated short name or derives one from structured metadata.

| File and lines | Current behavior | Required modification |
|---|---|---|
| `src/engine/logger.py:190-220` | Accepts `artifact_name` and sends it directly to `wandb.Artifact`. | Call the shared builder or validate the already-built short name before line 214. |
| `src/engine/logger.py:231-261` | Does the same for directory artifacts. | Apply the same short-name validation or builder. |
| `src/engine/artifact_sinks.py:48-54` | Uses only `path_obj.stem`, which is short but can be ambiguous across runs. | Build a role-plus-run name from metadata instead of using only the local stem. |
| `src/engine/artifact_sinks.py:66-72` | Uses only `path_obj.name` for a directory artifact. | Build a role-plus-run name from metadata instead of using only the directory name. |

The two sink changes are needed for uniqueness and human identification even when the current path name is below 128 characters.

### 3. Replace direct long-name construction

Every listed `artifact_name=f"{experiment_name}-..."` call is a direct modification point.

| File and lines | Artifact roles affected | Required modification |
|---|---|---|
| `scripts/cli/train.py:342-362` | Resolved config, metrics, best checkpoint. | Use `cfg`, `met`, and `ckpt-best` roles with the shared builder. |
| `scripts/cli/evaluate.py:395-453` | Resolved config, metrics, records, curves, traces, protocol audit. | Use short role-first names such as `eval-met`, `eval-records`, and `audit-eval`. |
| `scripts/experiments/run_online_adaptation.py:201-249` | Resolved config, metrics, online metrics, online records, final checkpoint. | Use `cfg`, `met`, `on-met`, `on-records`, and `ckpt-online` roles. |
| `scripts/experiments/run_ablation.py:147-174` | Resolved config, metrics, JSON summary, CSV summary. | Use `cfg`, `met`, and `ablation` roles with the suite identity. |

The metadata dictionaries at these call sites should continue to carry the full `experiment_name`.

That metadata is not the source of the W&B length failure.

### 4. Shorten W&B run names for consistency

These lines do not cause the reported `wandb.Artifact` exception, but they can still place long experiment names in the W&B run list.

They should use the same short identity builder or an equivalent run-name helper.

| File and lines | Evidence |
|---|---|
| `scripts/cli/train.py:262` | Defaults `wandb_run_name` to the full experiment name. |
| `scripts/cli/evaluate.py:303-305` | Appends `-evaluate` to the full experiment name. |
| `scripts/experiments/run_online_adaptation.py:128-132` | Defaults the online run name to the full experiment name. |
| `scripts/experiments/run_ablation.py:95-96` | Builds the ablation run name from the full experiment name. |
| `scripts/experiments/run_two_stage_offline_pretraining.py:155-158` | Copies the full stage experiment name into the W&B run name. |
| `scripts/benchmarks/run_thesis_online_benchmark.py:282-289` | Defaults the online benchmark run name to the full experiment name. |
| `scripts/benchmarks/run_online_streaming_benchmark.py:271-278` | Defaults the run name to the full benchmark name. |
| `scripts/run_direct_branch_routing_evaluation.py:42-54` | Creates an evaluation run name from the full experiment name. |
| `scripts/run_direct_branch_routing_full.py:113-115` | Copies the full experiment name into the run config. |
| `scripts/ops/prepare_raw_mse_offline_rerun.py:40` | Copies the full experiment name into the run config. |

The following config generators also provide explicit run names and can override runtime defaults.

| File and line | Required follow-up |
|---|---|
| `scripts/benchmarks/generate_benchmark_smoke_configs.py:69` | Generate a short run name when W&B is enabled. |
| `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:102` | Generate a short run name. |
| `scripts/benchmarks/generate_pro_reconstruction_vus_budget_matrix.py:118` | Generate a short run name. |
| `scripts/benchmarks/generate_smd_benchmark_configs.py:100` | Generate a short run name. |
| `scripts/benchmarks/generate_online_benchmark_configs.py:224` | Generate a short run name. |
| `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:226` | Generate a short run name. |

The disabled-W&B smoke assignment at `scripts/run_direct_branch_routing_smoke.py:62-69` is not required to fix the current error.

## Findings: mandatory minimal-retention changes

### 5. Do not write all offline derived files unconditionally

The active public offline benchmark calls `_export_offline_artifacts()` at `scripts/benchmarks/run_thesis_offline_benchmark.py:1149-1157`.

That function writes all of the following at `scripts/benchmarks/run_thesis_offline_benchmark.py:1001-1046`:

- `uq_summary.json`;
- three score NPZ files;
- three trace JSON files;
- `offline_metrics.json`;
- `resolved_protocol.json`.

Only the threshold artifact and the protocol/config roots should be unconditional under the minimal policy.

The score arrays, traces, UQ summary, and evaluation metrics should be written only for an explicit inspection or report-retention mode.

Required modification points:

| File and lines | Required modification |
|---|---|
| `scripts/benchmarks/run_thesis_offline_benchmark.py:957-1047` | Add `retention_policy` to the export decision and gate large derived writes. |
| `scripts/benchmarks/run_thesis_offline_benchmark.py:1097-1099` | Keep the resolved policy as the controlling value. |
| `scripts/benchmarks/run_thesis_offline_benchmark.py:1141-1169` | Pass the policy into `_export_offline_artifacts()` and keep only root or contract outputs by default. |

The in-memory `artifact_inputs` calculation may remain because it is temporary runtime state.

Temporary computation is not retained data.

### 6. Change the default retention policy

Both thesis benchmark entry points default to `retain_for_eda`.

That default conflicts with the minimal-retention policy.

Required modification points:

| File and lines | Current default | Required default |
|---|---|---|
| `scripts/benchmarks/run_thesis_offline_benchmark.py:213-215` | `retain_for_eda` | `summary_only`, unless the user explicitly requests EDA data. |
| `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` | `retain_for_eda` | `summary_only`, unless the user explicitly requests EDA data. |
| `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:189-191` | `retain_for_eda` | Match the public benchmark policy if this duplicate remains. |
| `src/core/artifact_integrity.py:47-55` | `retain_for_eda` for the function default. | Match the global default or require callers to pass the policy explicitly. |

The configuration generators explicitly set `retain_for_eda` in at least `scripts/benchmarks/generate_online_benchmark_configs.py:217` and `scripts/benchmarks/generate_smd_benchmark_configs.py:180`.

Those explicit values must also change if the minimal policy is intended to be the benchmark default.

### 7. Retain the online threshold contract independently of EDA data

The online retention exporter writes the threshold artifact only inside the `retain_for_eda` branch at `scripts/benchmarks/run_thesis_online_benchmark.py:214-226`.

The threshold contract is a small downstream input for online execution.

It should remain in `summary_only` bundles.

Move the threshold write outside the EDA-only branch.

Keep online metrics and online records inside the EDA-only branch.

Keep runtime state only when resume or forensic debugging requires it.

### 8. Do not embed full online execution data in the report

The thesis online benchmark stores the complete `online_outputs` under `report["online_execution"]` at `scripts/benchmarks/run_thesis_online_benchmark.py:315-323`.

That object can contain all online records and metric history.

The report should contain only compact counts, final metrics, threshold identity, checkpoint identity, and paths to retained root or contract files.

The same problem exists in the streaming benchmark at `scripts/benchmarks/run_online_streaming_benchmark.py:451-467`.

It embeds `metric_history` and `records` into `report["online_execution"]` after already writing them to separate files at lines 440-444.

The duplicate arrays should be removed from the report.

### 9. Make direct online outputs conditional

The direct online adaptation path always writes full metric history and records at `scripts/experiments/run_online_adaptation.py:182-192`.

It also uploads both files to W&B at `scripts/experiments/run_online_adaptation.py:221-239`.

Add a retention decision so the default keeps only the online checkpoint, threshold contract, and compact summary.

The streaming benchmark has the same unconditional writes at `scripts/benchmarks/run_online_streaming_benchmark.py:440-444`.

The metrics and records should be written only when the configured policy requests inspection data.

### 10. Make the evaluation CLI respect retention policy

The evaluation CLI always writes records, metrics, curves, traces, protocol audit files, and a resolved config at `scripts/cli/evaluate.py:316-386`.

It then uploads the resolved config and every evaluation output at `scripts/cli/evaluate.py:395-453`.

The config and protocol roots should remain available.

Records, curves, traces, and derived metrics should be written and uploaded only when the evaluation retention policy requests them.

The local stable basenames should not be renamed until all path consumers are migrated.

### 11. Reduce raw training and online logging

`ExperimentLogger.log_metrics()` appends every metric record to `metrics.jsonl` at `src/engine/logger.py:142-145`.

`ExperimentLogger.log_focused_metrics()` can also append a repeated stream at `src/engine/logger.py:158-171`.

These are derived histories, not root inputs.

The default policy should keep a compact final or best-step summary and make full JSONL history opt-in.

The online benchmark callback logs scalar values for every online record at `scripts/benchmarks/run_thesis_online_benchmark.py:130-145`.

That W&B history is also derived data and should be summary-only unless inspection retention is explicitly enabled.

## Findings: duplicated derived summaries

The retention summaries are small, but they currently copy derived or duplicated data.

These lines should be reviewed after the primary retention gates are added.

| File and lines | Duplicated data | Minimal form |
|---|---|---|
| `scripts/benchmarks/run_thesis_online_benchmark.py:190-208` | Full experiment name, paths, counts, and runtime flags. | Keep compact identity, checksums, contract identity, and counts only. |
| `scripts/benchmarks/run_thesis_online_benchmark.py:232-246` | Full experiment name in manifest identity. | Keep structured identity fields and references to root artifacts. |
| `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:498-520` | Artifact paths, offline metrics, and full two-stage execution report. | Keep root references, checksums, policy, and compact status. |
| `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:522-540` | UQ payload is built from raw score and trace inputs. | Keep only when explicitly requested or replace with a compact final summary. |
| `scripts/benchmarks/run_thesis_offline_benchmark.py:1170-1183` | The final report embeds protocol, manifest, and execution objects. | Keep report-ready summary plus root references, not repeated full payloads. |
| `scripts/benchmarks/run_thesis_online_benchmark.py:339-346` | Report integrity identity uses the full experiment name. | Replace with structured identity while retaining the full name in metadata if needed. |

The full experiment name is acceptable as a small metadata field when it is not copied into large repeated records.

## Active versus inactive code paths

The public thesis offline runner calls its local `_export_offline_artifacts()` at `scripts/benchmarks/run_thesis_offline_benchmark.py:1149`.

It imports `_export_offline_retention_bundle()` from `scripts/benchmarks/_internal/run_thesis_offline_benchmark_helpers.py:1050-1052`.

The internal helper also defines its own `collect_offline_artifact_inputs()` and `_export_offline_artifacts()` at lines 194 and 394, but the current repository search found no active caller for those two duplicate definitions.

Those duplicate functions should not be the first modification target.

If they are retained for a future entry point, they must follow the same retention policy before that entry point is used.

## Lines that do not need modification for this policy

Keep stable local basenames such as `best.pt`, `evaluation_metrics.json`, and `two_stage_manifest.json` when callers depend on them.

Keep `metadata["experiment_name"]` when it is a small provenance field.

Keep the resolved config as a root artifact, but consider storing only behavior-affecting fields if the config schema can provide deterministic defaults.

Keep checkpoint files required for later evaluation or resume.

Do not change `wandb.Artifact` types merely to solve the name-length error.

Do not solve the error by truncating the full name, because truncation can produce collisions and loses the human-important identity structure.

## Recommended modification order

1. Add the shared short-name builder and its length validation.

2. Replace the 17 direct artifact-name constructions.

3. Update the checkpoint and output artifact sinks.

4. Add focused tests for role, identity fields, collision resistance, and the 128-character limit.

5. Change benchmark defaults to `summary_only`.

6. Gate score, trace, record, UQ, and full-history writes.

7. Remove duplicate large payloads from reports.

8. Re-run one end-to-end benchmark combination before expanding the matrix.

## Evidence

- `prompts/1_research_prompt.md:1-10` — Defines this work as codebase research based on current implementation evidence.
- `prompts/1_research_prompt.md:57-78` — Requires tracing the real execution path and separating implementation from inference.
- `src/engine/logger.py:190-220` — Sends the supplied artifact name to W&B.
- `src/engine/logger.py:231-261` — Sends directory artifact names to W&B.
- `scripts/cli/train.py:342-362` — Builds the failing long names for training artifacts.
- `scripts/cli/evaluate.py:395-453` — Builds long names and uploads derived evaluation artifacts.
- `scripts/experiments/run_online_adaptation.py:201-249` — Builds long names for online artifacts.
- `scripts/experiments/run_ablation.py:147-174` — Builds long names for ablation artifacts.
- `src/engine/artifact_sinks.py:48-72` — Derives W&B names only from local paths.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:957-1047` — Writes offline derived files unconditionally.
- `scripts/benchmarks/run_thesis_offline_benchmark.py:1099-1169` — Resolves retention policy but does not use it to gate the primary export.
- `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` — Defaults online retention to `retain_for_eda`.
- `scripts/benchmarks/run_thesis_online_benchmark.py:190-231` — Writes summary, metrics, records, threshold artifact, and runtime state.
- `scripts/benchmarks/run_thesis_online_benchmark.py:315-323` — Places full online execution output in the report.
- `scripts/benchmarks/run_online_streaming_benchmark.py:440-473` — Writes and embeds full online records and history.
- `scripts/cli/evaluate.py:316-386` — Writes all evaluation outputs without a retention gate.
- `src/engine/logger.py:142-171` — Persists raw metric histories.

## Configuration observed

| Setting | Active value or default | Evidence | Scope |
|---|---|---|---|
| W&B artifact name | Caller-provided string | `src/engine/logger.py:214-218` | All direct W&B file artifacts. |
| W&B run name | Often full `experiment_name` | `scripts/cli/train.py:262`, `scripts/cli/evaluate.py:303-305` | Training and evaluation runs. |
| Offline retention policy | `retain_for_eda` when absent | `scripts/benchmarks/run_thesis_offline_benchmark.py:213-215` | Thesis offline benchmark. |
| Online retention policy | `retain_for_eda` when absent | `scripts/benchmarks/run_thesis_online_benchmark.py:115-117` | Thesis online benchmark. |
| Retention bundle choices | `retain_for_eda` or `summary_only` | `src/core/artifact_integrity.py:57-60` | Retention bundle validation. |
| Local output basenames | Stable names such as `best.pt` and `evaluation_metrics.json` | `scripts/cli/evaluate.py:318-324`, `scripts/experiments/run_two_stage_offline_pretraining.py:160-164` | Path-based consumers. |

## Uncertainty and limits

The repository does not currently show one global retention-policy gate shared by training, evaluation, and benchmark code.

The exact deletion or cleanup mechanism is therefore not implemented in the inspected paths.

The recommendation to change W&B run names is a consistency improvement, not the confirmed cause of the artifact exception.

The recommendation to change local output persistence is based on the stated minimal-retention policy and the observed unconditional writes.

It should be implemented only after deciding which final summaries are required for the thesis report.
