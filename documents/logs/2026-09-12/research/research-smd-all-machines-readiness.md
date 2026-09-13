---
date: 2026-09-12 20:08:00 +07:00
researcher: OpenAI Codex
topic: "Detect code modifications required before running the SMD all-machines experiment matrix"
status: complete
revision: b8fff201cdfdb27aa9cec377d607487c4c2755cb
branch: dev
---

# Research: Detect code modifications required before running the SMD all-machines experiment matrix

## Summary

Code modifications are required before running the target matrix.

The local SMD data has 28 matching train, test, and test-label entities, but the current remaining-SMD generator selects only 25 entities and supports only O0/O1.

The current execution path also has unresolved direct-branch, O2, A0-config, threshold-artifact, W&B, six-GPU, and preflight mismatches.

No source code was modified during this research.

## Research question

Read `prompts/1_research_prompt.md` and detect whether lines of code in the current codebase need modification before running `documents/notes/smd-all-machines-experiment-matrix.md`.

## System context

The target matrix is a planning baseline for 28 SMD entities, seeds 6/8/36, THESIS offline variants O0/O1/O2, THESIS online variants A0/A1/A2, baseline methods, and six GPU workers.

The intended cloud entrypoint is `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh`, which generates a manifest and delegates execution to `scripts/benchmarks/run_remaining_smd_matrix.sh`.

## Execution path

The cloud launcher performs a generator dry-run, writes the manifest, starts offline workers, waits for offline completion, validates dependencies, and then starts online workers.

The THESIS offline worker uses `scripts/benchmarks/run_thesis_offline_benchmark.py`, which runs the two-stage runner and exports threshold artifacts.

The online worker uses the matching Stage-B checkpoint and threshold artifact through `scripts/benchmarks/run_thesis_online_benchmark.py`.

The traditional offline worker uses `scripts/benchmarks/run_offline_benchmark.py`.

## Detailed findings

### Entity coverage and run counts

Implemented: the dataset contains 28 files in each of `train`, `test`, and `test_label`.

Implemented: `generate_remaining_smd_benchmark_configs.py:33` defines three excluded entities, and `:68` applies that exclusion.

Implemented: the generator dry-run returned 25 entities and 1,275 current logical runs.

The target matrix requires 28 entities, O0/O1/O2, and 1,764 logical runs.

Required modification: `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:46-87` needs an explicit all-entities path that preserves the existing remaining-entity behavior for other workflows.

The matrix is internally inconsistent: its logical-unit table requires 588 offline and 1,176 online units at `documents/notes/smd-all-machines-experiment-matrix.md:44-55`, while its wet execution stages state 504 offline and 924 online units at `:384-394`.

### O2 and THESIS offline configuration

Implemented: `generate_remaining_smd_benchmark_configs.py:35`, `:256`, and `:287` enumerate only O0/O1.

Implemented: `generate_smd_benchmark_configs.py:31-33` also enumerates only the three historical entities and O0/O1.

Implemented: requesting `variant="O2"` from the current offline builder silently receives the O1 point-score configuration because `generate_smd_benchmark_configs.py:120-130` treats every non-O0 variant as O1.

Required modification: the O2 configuration contract, model fields, generator, online cross-product, checkpoint dependency, and related preflight/inventory code are not complete.

The available model validation keys include two-view contrastive settings and score-loss settings at `src/core/config_model_validation.py:129-164`, but no distinct point-level contrastive configuration name was found.

The available evidence does not establish that the matrix term “point-level contrastive loss” is semantically identical to the existing `enable_two_view_contrastive` loss.

### Direct-branch routing and two-stage lifecycle

Configured: the standard THESIS model config uses `fusion_mode: task_specific_concat_projection` at `configs/model/thesis_multitask_two_stage_window20.yaml:77-84`.

Implemented: the remaining-SMD THESIS builder does not override that fusion mode at `generate_remaining_smd_benchmark_configs.py:380-412`.

Implemented: direct routing is supported by the model, but the standalone bridge requires a Stage-B-only config and rejects a `two_stage` section at `scripts/experiments/run_direct_branch_routing_bridge.py:43-52`.

Implemented: the standard two-stage runner changes stage metadata but does not establish the matrix-wide direct-routing contract at `scripts/experiments/run_two_stage_offline_pretraining.py:151-176`.

Required modification: the direct-branch routing lifecycle must be reconciled with the matrix's O0/O1/O2 Stage-A → Stage-B → evaluation contract before those runs can represent the requested matrix.

### Online A1/A2 behavior

Documented: the matrix assigns online-to-source contrastive loss and hard-old-normality adaptation to A1, with additional pseudo-new-normality adaptation in A2, at `documents/notes/smd-all-machines-experiment-matrix.md:87-114`.

Implemented: the current runtime makes A1 PNN-only at `src/engine/online_tta/online_engine_step.py:129-144`.

Implemented: the current runtime assigns hard-old adaptation and online-to-source contrastive loss to A2 at `src/engine/online_tta/online_engine_step.py:145-179`.

Required modification: the runtime or the matrix contract must be reconciled before interpreting A1/A2 results as the target variants.

### A0 scoring-config path

Implemented: `_add_run` creates online run IDs with `phase[:3]`, producing `onl-thesis-...`, at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:183-188`.

Implemented: the A0 resolver and the cloud manifest validator look for `on-thesis-...` at `scripts/ops/threshold_artifact_v4_online_scoring.py:54-70` and `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh:185-192`.

Observed: a temporary manifest generated from the current code created `onl-thesis-O0-A0-...yaml`, while the expected `on-thesis-O0-A0-...yaml` file did not exist.

Implemented: the active THESIS offline wrapper validates this A0 path before training at `scripts/benchmarks/run_thesis_offline_benchmark.py:126-150` and `:1136-1138`.

Required modification: the generated A0 filename contract must be made identical across the generator, resolver, launcher, and static fallback.

### Threshold artifact contract

Configured: the matrix requires the matching V4 threshold artifact for every online THESIS run at `documents/notes/smd-all-machines-experiment-matrix.md:316-334`.

Implemented: the remaining-SMD generator points online runs to `thresholds/thresholds.json` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:520-523`.

Implemented: the THESIS offline exporter writes `thresholds/thresholds.json` at `scripts/benchmarks/run_thesis_offline_benchmark.py:1006-1016`.

Implemented: the separate recalibration inventory supports only O0/O1, three historical entities, and seeds 6/8/36 at `scripts/ops/recalibrate_thesis_threshold_artifacts_v4.py:61-68`.

Required modification: the all-machine online dependency must produce and validate the exact threshold artifact required by the matrix, including O2.

### W&B logging and identity

Configured: the traditional offline generated config enables W&B at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:476-511`.

Implemented: `run_offline_benchmark.py` has no `ExperimentLogger` construction or W&B logging call, while the training and evaluation entrypoints do so at `scripts/cli/train.py:264-277` and `scripts/cli/evaluate.py:306-322`.

Required modification: traditional offline runs need an active W&B lifecycle if the matrix target of 2,352 W&B runs is required.

The matrix itself records this gap and estimates 2,100 runs when traditional offline logging remains absent at `documents/notes/smd-all-machines-experiment-matrix.md:404-418`.

The current generator also builds run names directly from `run_id` at `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:350-363`, and no all-matrix uniqueness preflight was found.

### Six-GPU launcher

Implemented: the cloud launcher defaults to four GPUs and rejects any count other than four at `scripts/benchmarks/run_remaining_smd_cloud_tmux.sh:6-7` and `:46-52`.

Implemented: GPU masks, GPU availability checks, worker loops, wait limits, and dry-run output are hard-coded for four GPUs at `:20`, `:108-115`, `:211-215`, `:298-299`, `:314-318`, and `:383-384`.

Required modification: the launcher must support six GPU queues and derive non-overlapping resource masks from the actual available CPU set as required by the matrix.

### Static preflight inventory

Observed: the focused test command produced 25 passed and 1 failed test.

The failure is `tests/benchmarks/test_full_benchmark_matrix_preflight.py`, where `scripts/ops/preflight_full_benchmark_matrix.py:58-64` expects 18 THESIS offline configs but the current glob finds 19 because extra configuration files are included.

Required modification: the preflight must select the exact target inventory instead of counting all files matching `*__main.yaml`, and it must be extended for 28 entities and O2 if it is used for this matrix.

## Evidence

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/prompts/1_research_prompt.md:1-11` — defines the research-only workflow and evidence requirements.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/notes/smd-all-machines-experiment-matrix.md:9-19` — defines the 28-entity, O0/O1/O2, and direct-routing target.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:33-87` — confirms the current three-entity exclusion behavior.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_remaining_smd_cloud_tmux.sh:6-20` — confirms the four-GPU defaults and masks.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/benchmarks/run_offline_benchmark.py:14-43` — confirms the traditional offline runner imports no experiment logger.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_step.py:119-183` — confirms the current A1/A2 update split.
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/benchmarks/test_full_benchmark_matrix_preflight.py:8-23` — identifies the failing preflight readiness assertion.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| SMD split intersection | 28 entities per split | `data/ServerMachineDataset/{train,test,test_label}` inventory | Local dataset |
| Remaining generator entities | 25 entities | `generate_remaining_smd_benchmark_configs.py:33,68` | Current remaining-SMD path |
| THESIS offline variants | O0, O1 | `generate_remaining_smd_benchmark_configs.py:35` | Current remaining-SMD path |
| Window and stride | 20 and 1 | `generate_remaining_smd_benchmark_configs.py:39-40` | Current remaining-SMD path |
| Cloud GPU count | Exactly 4 | `run_remaining_smd_cloud_tmux.sh:6,50-52` | Current cloud launcher |
| Traditional offline W&B | Configured true, not instantiated | `generate_remaining_smd_benchmark_configs.py:494-510`; `run_offline_benchmark.py:14-43` | Traditional offline path |

## Conflicts and uncertainties

The working tree is dirty, including changes to the generator, launcher, runner, and tests.

This report evaluates the current working-tree files, not only revision `b8fff201cdfdb27aa9cec377d607487c4c2755cb`.

The matrix's “V4 threshold artifact” wording is not fully aligned with the current artifact module, which distinguishes schema versions 4, 5, and 6 from the recalibrated filename.

The six-GPU hardware state, W&B connectivity, free disk, and remote server process state were not checked because this task was limited to non-destructive codebase research.

## Open questions

The available code does not establish whether the matrix's point-level contrastive loss is the existing two-view contrastive loss or a new loss contract.

The available code does not establish whether the matrix's direct-routing rule applies to Stage A, Stage B, or both stages of the two-stage offline lifecycle.

