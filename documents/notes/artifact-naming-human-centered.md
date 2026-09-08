# Human-Centered Artifact Naming Notes

Date: 2026-09-08

Status: implemented policy and verification note.

## Main conclusion

Artifact names should carry only the information that a human needs to identify the artifact quickly.

The full experiment identity should remain in the resolved config, artifact metadata, manifests, checkpoint metadata, and the directory hierarchy.

Short names solve the W&B limit, but they do not by themselves minimize retained data.

The retention rule should be: keep immutable root inputs, then regenerate derived outputs when needed.

The implementation uses `summary_only` as the default retention policy.

The explicit `retain_for_eda` policy keeps derived records and diagnostics for analysis.

## Minimal-retention principle

An artifact or field is a root when the repository cannot calculate it from other retained fields.

A field is derived when the repository can calculate it deterministically from root fields and the recorded code version.

Do not keep a large derived file only because it is convenient to inspect.

Keep it only when recalculation is impossible, too expensive, or required for an external audit.

The word "deterministically" requires an immutable dataset reference, a fixed protocol, the code revision, and any required random seed.

If a dataset reference points to mutable files, retain a dataset snapshot or a verified dataset checksum.

## Root-field dependency graph

The graph below describes the smallest useful information flow for one experiment run.

Fields marked `[ROOT]` are retained inputs or small contracts.

Other fields are calculated outputs.

```text
[ROOT] dataset_ref + dataset_version + entity_id
                         |
                         v
                  data rows and labels
                         |
[ROOT] checkpoint_state + model_config + seed
                         |
                         v
                     point scores
                         |
          +--------------+--------------+
          |                             |
          v                             v
  point predictions                window scores
          |                             |
          v                             v
  point metrics                  window metrics
          |                             |
          +--------------+--------------+
                         v
                  curves and VUS
                         |
                         v
                  evaluation report

[ROOT] split_definition + threshold_rule + calibration_data_ref
                         |
                         v
                   threshold contract
                         |
                         +----------------------+
                         |                      |
                         v                      v
              online predictions         online triage decisions
                         |                      |
                         v                      v
                  online records       online metrics and report

[ROOT] code_revision + protocol_version + environment_lock
                         |
                         v
                 provenance and integrity
```

The graph has two important boundaries.

The checkpoint is a root because model weights cannot be calculated from metrics or from the experiment name.

The threshold contract is a small root for online evaluation because the online runtime consumes it directly.

If the calibration data and threshold rule are immutable and cheap to access, the threshold contract can be regenerated instead of retained.

## Minimal root fields to retain

Retain the following fields once per run or once per shared artifact.

| Root field | Why it is needed | Can it be omitted? |
|---|---|---|
| `dataset_ref` | Identifies the source data. | Only if the parent run already fixes it. |
| `dataset_version` or dataset checksum | Makes recalculation reproducible. | No, unless the dataset is immutable by contract. |
| `entity_id` | Selects the SMD machine or other data entity. | No for multi-entity runs. |
| `variant` | Selects O0, O1, or another model variant. | No when variants share a directory. |
| `seed` | Reproduces seeded training or sampling. | No when the checkpoint is the only retained result and retraining is not required. |
| `stage` | Selects Stage A, Stage B, or online evaluation. | No when one artifact directory contains multiple stages. |
| `resolved_config` | Stores execution parameters that affect behavior. | No, unless an immutable config file is retained elsewhere. |
| `checkpoint_state` | Stores the trained model and required optimizer or extra state. | No for any result that must be rerun. |
| `split_definition` | Defines train, validation, and test boundaries. | No for evaluation reproducibility. |
| `protocol_version` | Selects the metric, threshold, and online rules. | No when protocol code is not otherwise pinned. |
| `threshold_contract` | Supplies the threshold consumed by online evaluation. | Omit only when it can be regenerated from immutable calibration inputs. |
| `code_revision` | Identifies the source code used to calculate outputs. | No for reproducible research. |
| `environment_lock` | Identifies package and runtime versions. | Optional for a lightweight exploratory run. |

The run identity can be generated from `dataset_ref`, `entity_id`, `variant`, `seed`, `stage`, and the relevant online variant.

The human-facing artifact name can also be generated from that identity and the artifact role.

Therefore, neither the full `experiment_name` nor a second copied identity string is a root field.

## Derived fields and files

The following outputs should normally be generated on demand and not treated as retained roots.

| Derived output | Root inputs |
|---|---|
| `run_id` | Dataset, entity, variant, seed, stage, and online variant. |
| Short W&B artifact name | Artifact role plus run identity. |
| Point scores | Dataset rows plus checkpoint, model config, and inference protocol. |
| Predictions | Point scores plus threshold contract or evaluation rule. |
| Window scores | Point scores plus window definition. |
| `evaluation_metrics.json` | Labels, predictions, scores, and metric protocol. |
| `evaluation_curves.json` | Scores, labels, and curve protocol. |
| `evaluation_records.json` | Per-point inputs, scores, predictions, and protocol. |
| `evaluation_traces.json` | Evaluation records plus selected diagnostic rules. |
| `uq_summary.json` | Scores or traces plus UQ aggregation rules. |
| `online_records.json` | Test stream, checkpoint, threshold contract, and online config. |
| `online_metrics.json` | Online records plus metric aggregation rules. |
| Benchmark reports and tables | Metrics, protocol, and report formatting rules. |
| Plot files and CSV exports | Metrics, curves, records, or report data. |
| Checksum manifests | The retained files and their paths. |
| `retention_summary.json` | Selected retained files plus summary rules. |

The exact code may use some of these files as intermediate inputs for a later command.

That usage does not make them roots if the same files can be regenerated from the retained root set.

## Retention tiers

Use three simple tiers so the repository does not need a separate policy for every artifact type.

| Tier | Retain | Examples |
|---|---|---|
| Root | Always retain because later outputs depend on it. | Resolved config, checkpoint, dataset reference, split definition, protocol version, code revision. |
| Small contract | Retain when another runtime consumes it directly. | Threshold artifact, compact provenance record, selected checkpoint metadata. |
| Derived | Recalculate and remove by default. | Records, traces, curves, UQ summaries, reports, plots, CSV exports, checksum manifests. |

The threshold artifact belongs to the small-contract tier for online runs.

It should contain only the threshold values and the minimum identity needed to verify compatibility.

It should not copy the full experiment config, full score arrays, or full validation records.

Raw forward-pass outputs, complete training logs, and full online records should not be retained by default.

Keep a selected diagnostic sample only when it supports a known audit or debugging question.

## Artifact-type retention decisions

| Artifact type | Default decision | Minimal retained content |
|---|---|---|
| Generated config | Recreate from the generator when possible. | Generator inputs and generator revision. |
| Resolved config | Retain. | Only behavior-affecting resolved fields and schema version. |
| Training metrics and logs | Summarize, then remove raw logs. | Final or best-step summary and failure information if needed. |
| Model checkpoints | Retain selected checkpoints. | Stage-A best, Stage-B best, and online final only when used downstream. |
| Offline evaluation | Recalculate. | No raw records by default. |
| Evaluation diagnostics | Recalculate or sample. | Protocol audit only when required for review. |
| Online evaluation | Recalculate. | Compact online summary and threshold contract. |
| Threshold and calibration | Retain as a small contract or regenerate. | Threshold values, source rule, calibration split reference, and identity. |
| Scores, traces, and UQ | Recalculate or retain selected samples. | No full arrays by default. |
| Retention bundle | Keep only if it is the delivery unit. | Root files plus a small manifest of included files. |
| Provenance and integrity | Regenerate. | Code revision, dataset checksum, and checkpoint checksum. |
| Benchmark reports | Regenerate from summaries. | Final report only for submission or external sharing. |
| Plots and analysis exports | Regenerate. | Final figures used in the thesis. |
| External run bundle | Keep only for an external handoff. | The root set and small contracts, not every derived output. |
| Ablation summaries | Retain the comparison summary. | One row per condition and links to root runs. |

The Stage-A and Stage-B best checkpoints are the default retention choice because later evaluation normally starts from them.

Retain `final.pt` only when it is used for a specific comparison, resume operation, or audit.

## Practical consequence for naming

Names should identify the retained root or contract, not every derived calculation.

Use names such as `cfg-stageA-O0-machine_1_6-s36`, `ckpt-best-stageB-O0-machine_1_6-s36`, and `thr-O0-machine_1_6-s36`.

Generate names such as `eval-metrics-O0-machine_1_6-s36` only when the derived evaluation is actually retained or uploaded.

The full dependency information belongs in a compact manifest or metadata object.

That manifest should reference root artifacts and protocol versions instead of embedding full copies of their contents.

## Limits of the root-field approach

Root fields minimize stored data only when their dependencies remain available and unchanged.

If the dataset, source code, threshold rule, or environment changes, old derived results may no longer be reproducible.

For thesis results, retain the root set and at least the final report-ready summary.

For exploratory runs, retain the root set and regenerate all other outputs when needed.

The recommended W&B pattern is:

```text
<role>-<stage-or-variant>-<entity>-s<seed>[-<online-variant>]
```

Examples:

```text
cfg-stageA-O0-machine_1_6-s36
ckpt-best-stageA-O0-machine_1_6-s36
eval-metrics-O0-machine_1_6-s36
online-metrics-A1-O0-machine_1_6-s36
```

These names are short, readable, and much shorter than the current `experiment_name` strings.

## Important distinction

A local filename, a W&B artifact name, and an artifact directory are different naming surfaces.

The current failure occurs at the W&B surface because `wandb.Artifact(name=...)` rejects names longer than 128 characters.

The local output hierarchy can keep stable filenames such as `best.pt`, `evaluation_metrics.json`, and `two_stage_manifest.json` when existing code and tests depend on them.

The safest design is to shorten the W&B name separately instead of renaming every local file and breaking path-based consumers.

The implementation uses `src/core/artifact_naming.py` as the shared naming boundary.

The helper builds role-first names from the smallest human-useful identity: role, stage or online variant, model variant, entity, seed, and an optional budget.

The helper rejects invalid names longer than 128 characters instead of silently truncating them.

The common logger and artifact sink validate every name before constructing `wandb.Artifact`.

## Evidence from the codebase

The experiment logger writes `metrics.jsonl` and `resolved_experiment_config.json`, and it creates W&B file or directory artifacts when W&B logging is enabled.

The training CLI logs resolved config, metrics, and the best checkpoint as W&B artifacts.

The evaluation CLI logs resolved config, evaluation metrics, records, curves, traces, and the protocol audit as W&B artifacts.

The online runtime writes online metrics, online records, `online_final.pt`, and `online_artifact_manifest.json`.

The checkpoint manager writes `initial.pt`, `best.pt`, `final.pt`, and other explicitly requested checkpoint names, including `stage_b_init.pt` and `online_final.pt`.

The two-stage runner writes `two_stage_manifest.json` and `two_stage_execution_report.json`.

The retention bundle may contain summaries, traces, point-score NPZ files, offline metrics, resolved protocol data, and `retention_bundle_manifest.json`.

The code also has W&B artifact types named `config`, `metrics`, `checkpoint`, `evaluation`, `online-evaluation`, `ablation-summary`, and `run-output`.

Evidence files:

- [Experiment logger](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/logger.py>)
- [Training CLI](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/cli/train.py>)
- [Evaluation CLI](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/cli/evaluate.py>)
- [Online runtime](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_run.py>)
- [Artifact sinks](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/artifact_sinks.py>)
- [Artifact integrity](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/artifact_integrity.py>)

The verified smoke training run uses `configs/experiment/raw_mse_offline_cpu_smoke.yaml` and exits successfully with W&B disabled.

The fresh summary-only evaluation smoke writes only the resolved protocol, threshold contract, retention manifest, and retention summary.

The existing output tree was not cleaned, so old derived files may still exist in previously used directories.

The focused verification passed 50 tests.

The full suite passed 568 tests and skipped one test, with six failures in existing benchmark, compliance, model, and checkpoint-contract tests outside this change.

## Artifact inventory

| Artifact type | Current examples | Human purpose |
|---|---|---|
| Generated experiment config | `*.yaml` under `generated_configs/` | Shows the exact launch configuration. |
| Resolved config | `resolved_experiment_config.json` | Reproduces one run after config composition. |
| Training metrics and logs | `metrics.jsonl`, `focused_metrics.jsonl`, W&B run files | Shows training progress and resource information. |
| Model checkpoints | `initial.pt`, `best.pt`, `final.pt`, `stage_b_init.pt`, `online_final.pt` | Restores a model or identifies the selected model state. |
| Offline evaluation | `evaluation_metrics.json`, `evaluation_records.json`, `evaluation_curves.json` | Stores final offline results and supporting records. |
| Evaluation diagnostics | `evaluation_traces.json`, protocol audit JSON, protocol audit Markdown | Explains how evaluation was performed. |
| Online evaluation | `online_metrics.json`, `online_records.json` | Stores causal online results and step records. |
| Threshold and calibration | `thresholds.json`, V4 threshold files, recalibration audits | Defines the score thresholds and their provenance. |
| Scores, traces, and UQ | `*.npz`, `*_traces.json`, `uq_summary.json` | Supports uncertainty analysis and selected diagnostics. |
| Retention bundle | `retention_summary.json`, retention files, `retention_bundle_manifest.json` | Groups selected artifacts for later inspection. |
| Provenance and integrity | `two_stage_manifest.json`, execution reports, integrity manifests | Connects files to a run and verifies checksums. |
| Benchmark reports | benchmark reports, report data, tables, preflight summaries | Gives human-readable benchmark conclusions. |
| Plots and analysis exports | PNG, SVG, CSV, Markdown, and analysis JSON files | Supports visual inspection and report construction. |
| External run bundle | W&B `run-output`, W&B file artifacts, Kaggle mirror | Transfers or stores a complete run outside the local tree. |
| Ablation summaries | `ablation_summary.json`, `ablation_summary.csv` | Compares ablation settings. |

## Cloud GPU findings

The latest read-only cloud check on 2026-09-07 confirmed the remote repository at `/root/bachelor-thesis-2026`, revision `cc42b4e3`, branch `dev`, with CUDA available on an NVIDIA GeForce RTX 4070.

That check modified no remote files, jobs, or outputs.

The current remote checkout does not yet contain the newer normalized-input protocol, 54-config generator, or new smoke and wet-run scripts.

A previous read-only remote artifact inventory verified 18 THESIS two-stage combinations across O0/O1, three SMD entities, and seeds 6, 8, and 36.

Each verified two-stage run contained five core checkpoint files: `stage_b_init.pt`, Stage-A `best.pt`, Stage-A `final.pt`, Stage-B `best.pt`, and Stage-B `final.pt`.

The five checkpoint files imply 90 core checkpoint files across the 18 verified runs.

The remote inventory proves the checkpoint files, but it does not prove that every threshold artifact or W&B server artifact is valid.

The cloud GPU therefore needs the same local artifact taxonomy, plus the W&B run directory and external W&B artifact records when `use_wandb` is enabled.

Evidence files:

- [Remote checkpoint inventory](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/inventories/detail-remote-gpu-checkpoints-inventory.md>)
- [Latest cloud GPU tree findings](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/09-07-2026/remote/cloud-gpu-repo-tree-findings.md>)

## Naming vocabulary

Use full words when they remain short and clear.

The following abbreviations are acceptable because they are common in this project:

| Token | Meaning |
|---|---|
| `cfg` | Configuration. |
| `met` | Metrics. |
| `ckpt` | Checkpoint. |
| `eval` | Evaluation. |
| `thr` | Threshold or calibration. |
| `on` | Online. |
| `uq` | Uncertainty quantification. |
| `tr` | Trace. |
| `man` | Manifest. |
| `rpt` | Report. |
| `stageA`, `stageB` | Stage A or Stage B. |
| `machine_1_6` | SMD entity identity. |
| `s36` | Seed 36. |

Avoid opaque abbreviations such as `x`, `z`, or a bare hash in the human-facing name.

## Three naming ways for each artifact type

The examples below use `O0`, `machine_1_6`, seed 36, and online variant `A1` where relevant.

The first option is the recommended human-facing W&B name.

| Artifact type | Way 1: role first | Way 2: run first | Way 3: local role only |
|---|---|---|---|
| Generated config | `cfg-stageA-O0-machine_1_6-s36` | `O0-machine_1_6-s36-cfgA` | `config.yaml` |
| Resolved config | `cfg-resolved-O0-machine_1_6-s36` | `O0-machine_1_6-s36-cfg` | `resolved_config.json` |
| Training metrics | `met-stageA-O0-machine_1_6-s36` | `O0-machine_1_6-s36-metA` | `metrics.jsonl` |
| Focused metrics | `met-focus-stageA-O0-machine_1_6-s36` | `O0-machine_1_6-s36-met-focus` | `focused_metrics.jsonl` |
| Best checkpoint | `ckpt-best-stageB-O0-machine_1_6-s36` | `O0-machine_1_6-s36-ckptB-best` | `best.pt` |
| Initial checkpoint | `ckpt-init-stageA-O0-machine_1_6-s36` | `O0-machine_1_6-s36-ckptA-init` | `initial.pt` |
| Final checkpoint | `ckpt-final-stageB-O0-machine_1_6-s36` | `O0-machine_1_6-s36-ckptB-final` | `final.pt` |
| Stage-B initialization | `ckpt-stageB-init-O0-machine_1_6-s36` | `O0-machine_1_6-s36-ckptB-init` | `stage_b_init.pt` |
| Online checkpoint | `ckpt-online-A1-O0-machine_1_6-s36` | `A1-O0-machine_1_6-s36-ckpt-online` | `online_final.pt` |
| Offline metrics | `eval-met-O0-machine_1_6-s36` | `O0-machine_1_6-s36-eval-met` | `evaluation_metrics.json` |
| Offline records | `eval-records-O0-machine_1_6-s36` | `O0-machine_1_6-s36-eval-records` | `evaluation_records.json` |
| Evaluation curves | `eval-curves-O0-machine_1_6-s36` | `O0-machine_1_6-s36-eval-curves` | `evaluation_curves.json` |
| Evaluation traces | `eval-traces-O0-machine_1_6-s36` | `O0-machine_1_6-s36-eval-traces` | `evaluation_traces.json` |
| Protocol audit | `audit-eval-O0-machine_1_6-s36` | `O0-machine_1_6-s36-audit-eval` | `evaluation_protocol_audit.json` |
| Online metrics | `on-met-A1-O0-machine_1_6-s36` | `A1-O0-machine_1_6-s36-on-met` | `online_metrics.json` |
| Online records | `on-records-A1-O0-machine_1_6-s36` | `A1-O0-machine_1_6-s36-on-records` | `online_records.json` |
| Threshold artifact | `thr-O0-machine_1_6-s36` | `O0-machine_1_6-s36-thr` | `thresholds.json` |
| Threshold audit | `audit-thr-O0-machine_1_6-s36` | `O0-machine_1_6-s36-audit-thr` | `threshold_audit.json` |
| UQ summary | `uq-O0-machine_1_6-s36` | `O0-machine_1_6-s36-uq` | `uq_summary.json` |
| Point scores | `scores-O0-machine_1_6-s36` | `O0-machine_1_6-s36-scores` | `point_scores.npz` |
| Diagnostic traces | `traces-O0-machine_1_6-s36` | `O0-machine_1_6-s36-traces` | `traces.json` |
| Retention summary | `retention-O0-machine_1_6-s36` | `O0-machine_1_6-s36-retention` | `retention_summary.json` |
| Retention bundle | `bundle-O0-machine_1_6-s36` | `O0-machine_1_6-s36-bundle` | `retention/` |
| Two-stage manifest | `man-two-stage-O0-machine_1_6-s36` | `O0-machine_1_6-s36-man-two-stage` | `two_stage_manifest.json` |
| Execution report | `rpt-exec-O0-machine_1_6-s36` | `O0-machine_1_6-s36-rpt-exec` | `two_stage_execution_report.json` |
| Integrity manifest | `man-integrity-O0-machine_1_6-s36` | `O0-machine_1_6-s36-man-integrity` | `integrity_manifest.json` |
| Offline benchmark report | `rpt-offline-O0-machine_1_6-s36` | `O0-machine_1_6-s36-rpt-offline` | `benchmark_report.json` |
| Online benchmark report | `rpt-online-A1-O0-machine_1_6-s36` | `A1-O0-machine_1_6-s36-rpt-online` | `online_report.json` |
| Plot or visual diagnostic | `plot-scores-machine_1_6` | `machine_1_6-plot-scores` | `scores.png` |
| CSV analysis export | `data-scores-machine_1_6` | `machine_1_6-data-scores` | `scores.csv` |
| Ablation summary | `ablation-O0-machine_1_6-s36` | `O0-machine_1_6-s36-ablation` | `ablation_summary.json` |
| Complete run output | `run-O0-machine_1_6-s36` | `O0-machine_1_6-s36-run` | `run-output/` |

## Comparison of the three ways

Way 1 is the clearest choice for W&B because the artifact role appears first when names are sorted or displayed in a list.

Way 2 is useful when a human usually starts from one experiment run and wants all its artifacts grouped by the same prefix.

Way 3 is the clearest local filesystem choice when the parent directory already identifies dataset, variant, entity, seed, phase, and stage.

Way 3 should not be used as the W&B name unless the W&B project and metadata always provide the missing identity.

## Recommended policy

Use Way 1 for W&B artifact names.

Use Way 3 for stable local filenames inside the canonical run hierarchy.

Keep the full `experiment_name` in metadata and the resolved config.

Keep `dataset`, `variant`, `entity`, `seed`, `stage`, `online_variant`, `window_size`, `FPR budget`, protocol version, checkpoint path, and checksums in metadata or manifests rather than forcing them into the name.

Include the FPR budget in names only for artifacts whose contents change with that budget, such as budget-specific evaluation or report artifacts.

For budget-specific names, append a readable token such as `fpr0p1`, `fpr0p5`, or `fpr1`.

Do not put timestamps in the primary name unless two artifacts can otherwise have the same identity and the timestamp is genuinely needed by a human.

Do not put the full protocol description, loss weights, retrieval settings, or configuration filename into a W&B artifact name.

Do not use the full `experiment_name` as a W&B artifact-name prefix.

Do not use a hash as the primary human-facing name.

A hash may remain in metadata for integrity and exact provenance.

## Suggested W&B artifact names for the failing run

For the failing Stage-A run, the long current name can be replaced with these names:

```text
cfg-stageA-O0-machine_1_6-s36
met-stageA-O0-machine_1_6-s36
ckpt-best-stageA-O0-machine_1_6-s36
run-O0-machine_1_6-s36
```

The metadata should still contain the full experiment name, generated config path, output directory, stage name, seed, protocol settings, and checkpoint information.

## Migration caution

Existing scripts and tests resolve important files by stable basenames such as `best.pt`, `evaluation_metrics.json`, and `two_stage_manifest.json`.

Changing those local basenames would create a larger migration than the W&B error requires.

The minimal change is to add a compact W&B-name builder and use it only for `artifact_name` values.

That builder should receive structured fields from the config instead of truncating an already-composed string.

The builder should validate its output length before calling `wandb.Artifact`.

The full name should remain in `metadata["experiment_name"]` and in the uploaded config artifact.

The direct traceback is fixed at the artifact-construction boundary when the active caller uses the shared helper.

W&B upload activation remains unchanged because enabling new external uploads is outside this naming and retention change.

## Decision record

Selected default: Way 1 for W&B, Way 3 for local files, and metadata or manifests for all non-essential identity fields.

Reason: this keeps names readable for humans, stays far below the W&B 128-character limit, preserves the current local path contract, and keeps full reproducibility information available.
