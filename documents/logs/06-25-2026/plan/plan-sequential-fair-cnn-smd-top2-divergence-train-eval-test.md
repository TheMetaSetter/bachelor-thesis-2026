---
date: 2026-06-25 16:51:40 +0700
researcher: TheMetaSetter
git_commit: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary implementation plan for fair top-divergence SMD experiments with thesis three-stage CNN and RedLamp baseline single-stage CNN pipelines"
tags: [plan, smd, cnn, thesis_multitask, redlamp_mlp_baseline, evaluation, metrics, tmux]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Plan: Preliminary implementation plan for fair top-divergence SMD experiments with thesis three-stage CNN and RedLamp baseline single-stage CNN pipelines

## Planning Goal

The goal of this plan is to define a safe, additive, and thesis-facing programming direction for running a **fair comparative experiment matrix** in which:

- `src/models/thesis_multitask.py` runs on the **three-stage offline pre-training path** with the `cnn_simple` 1D-CNN encoder family;
- `src/models/redlamp_mlp_baseline.py` runs on the **basic single-stage offline path** with the `cnn_simple` 1D-CNN encoder family;
- both methods are evaluated on the **two SMD entities selected by `KL(test || train)` divergence**;
- both methods use the same three seeds `6`, `36`, `68`;
- and each method receives its **own exact `300`-epoch budget per run**.

The plan is constrained by four principles. First, the repository should preserve the current registry-driven `train.py` and `evaluate.py` paths wherever possible. Second, fairness-critical settings between the two methods should be synchronized tightly. Third, method-specific structure should remain intact, meaning the baseline remains prototype-free while the thesis model retains its own prototype and fusion behavior. Fourth, the server execution path should prioritize **safe full-run completion on one RTX 3090** over aggressive but brittle parallelism.

This is a preliminary implementation plan. It intentionally stops before line-by-line programming detail and instead identifies the minimum safe codebase changes and experiment-preparation steps needed before implementation.

## Scope Assumption

This plan assumes a **mixed but explicit runtime split**:

1. the thesis method trains and validates through `scripts/run_three_stage_offline_pretraining.py`;
2. the baseline trains and validates through `scripts/train.py`;
3. both methods are tested afterward through `scripts/evaluate.py` from an explicit resolved checkpoint.

This plan therefore explicitly **does assume** the special three-stage offline pre-training orchestrator for the thesis side, and explicitly **does not collapse** the thesis method back into the generic single-stage path merely for convenience.

## Current State

- The repository already supports both requested models in the generic offline runtime:
  - `scripts/train.py` registers `thesis_multitask` and `redlamp_mlp_baseline`.
  - `scripts/evaluate.py` registers the same two models.
- The repository also already contains a runnable thesis-specific three-stage orchestrator:
  - `scripts/run_three_stage_offline_pretraining.py`
  - `configs/model/thesis_multitask_three_stage_window20.yaml`
  - `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- The data path already supports SMD entity filtering through `entity_ids`, and `WindowDataset` already preserves the metadata needed for timeline reconstruction.
- The evaluation path already reconstructs overlapped test windows back onto the original entity timeline using:
  - score reduction by overlap-aware mean,
  - label reduction by overlap-aware max.
- The evaluation runtime already computes `VUS-PR`, because `src/metrics/pointwise.py` implements `compute_vus_pr_exact_naive(...)` and `scripts/evaluate.py` passes `vus_max_buffer_size` and `vus_num_thresholds` into `Evaluator`.
- The active runtime does **not** yet expose `Affiliation-F1` or `VUS-ROC`.
- The active multi-seed launcher `scripts/run_multiseed_experiments.py` only launches training commands. It does not chain training into evaluation.
- The repository already has 1D-CNN support in both model families:
  - `configs/model/redlamp_cnn_baseline.yaml`
  - `configs/model/thesis_multitask_redlamp_multiclass_cnn_simple.yaml`
  - both model files already accept `encoder_family: cnn_simple`.
- The repository already has a dedicated SMD `machine-3-4`, `window_size=20`, `stride=1` data config:
  - `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`
- The thesis three-stage model config is already compatible with the requested fairness-critical thesis-side settings:
  - `encoder_family: cnn_simple`
  - `use_label_refurbishment: true`
  - `lambda_recon: 0.9`
  - `lambda_cls: 0.1`
  - exact three-stage budget fields summing to `300`
- The current top-divergence target entities are still **not yet locked in a durable artifact**, but the ranking criterion is now fixed at **`KL(test || train)`**. The only remaining micro-ambiguity is how to reduce channelwise KL values into one entity-level score.

## Established Contracts That Must Be Preserved

### Batch Contract

The current batch contract from `src/data/loaders.py` should remain unchanged:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

This is already consumed consistently by both models, the trainer, and the evaluator. The proposed work should not redesign batch structure.

### Encoder Contract

The thesis-facing encoder contract should remain:

`hidden: Tensor[B, L, H]`

The baseline already preserves this contract internally through its timestep encoder, and the thesis model already exposes this contract through `MultitaskWindowEncoder`. The new fair-comparison work should only switch configs and orchestration around this contract, not change the contract itself.

### Evaluation Reconstruction Contract

The current evaluator reconstructs timeline-level test signals from overlapped windows through entity-aware aggregation. That behavior is already aligned with the research note and should be preserved exactly:

- reconstruct first per entity,
- average scores over overlaps,
- merge labels by `max`,
- then concatenate entity-level records for global metric computation.

The new metrics should be implemented on top of this reconstructed timeline-level contract, not on raw per-window outputs.

### Mixed Runtime Contract

The current runtime split should be preserved and made explicit:

- thesis three-stage training remains in `scripts/run_three_stage_offline_pretraining.py`,
- baseline single-stage training remains in `scripts/train.py`,
- post-training test evaluation remains in `scripts/evaluate.py` for both.

The orchestration work should therefore be **additive**, by chaining these existing entrypoints and resolving checkpoints explicitly, not by merging them into one monolithic script.

## Existing Modules That Should Be Preserved

The following files already own the correct responsibilities and should be preserved rather than broadly refactored:

- `src/data/datasets/smd.py`
- `src/data/loaders.py`
- `src/data/collate.py`
- `src/engine/evaluator.py`
- `src/engine/trainer.py`
- `src/metrics/pointwise.py`
- `scripts/run_three_stage_offline_pretraining.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

The codebase already has the correct structural split for this task. The main missing pieces are metric coverage, fair config alignment, target-entity selection, and orchestration.

## Design Options

### Option A: Additive Mixed Orchestration Layer on Top of the Existing Thesis Three-Stage Path and Baseline Single-Stage Path

Under this option, the repository would:

1. keep `scripts/run_three_stage_offline_pretraining.py` unchanged as the authoritative thesis training entrypoint;
2. keep `scripts/train.py` unchanged as the authoritative baseline training entrypoint;
3. keep `scripts/evaluate.py` unchanged as the authoritative test entrypoint for both;
4. add the missing evaluation metrics into the existing evaluation stack;
5. add a new lightweight runner that dispatches either:
   - thesis `three-stage train + val -> test`, or
   - baseline `single-stage train + val -> test`;
6. add a `tmux` launcher on top of that runner for remote execution;
7. add new fair-aligned CNN experiment configs for both methods and all requested seeds.

This option minimizes runtime drift, keeps the write surface small, and uses the code paths already audited in the 06-25 research note.

### Option B: Collapse the Thesis Side Back Into the Generic Single-Stage Runtime for Easier Comparison

Under this option, the thesis method would stop using the three-stage orchestrator and would instead be forced back into the ordinary `scripts/train.py -> scripts/evaluate.py` path so the comparison runner becomes more uniform.

This option simplifies orchestration superficially, but it damages methodological fidelity because the user explicitly wants the thesis method to remain a three-stage method.

### Option C: Redesign Both Training Paths Into One New Unified Training Script

Under this option, the repository would try to centralize thesis three-stage training logic and baseline single-stage training logic into one newly invented meta-training script.

This is the least desirable option. It creates the largest write surface, duplicates already-working orchestration, and increases the chance of silently drifting away from the already-audited code paths.

## Recommended Approach

The recommended approach is **Option A**.

The repository already has the correct runtime pieces:

- a thesis-specific three-stage path,
- a baseline single-stage path,
- and a shared evaluator.

The safest path is therefore:

- preserve that split,
- add the missing metrics inside the shared evaluator path,
- add a new small runner whose only job is mixed orchestration and checkpoint resolution,
- add explicit fair-aligned configs for the exact experiment matrix.

This approach is the most compatible with the current design philosophy in `documents/design/design_starter.md`, which emphasizes a small number of stable runtime contracts and composition over large refactors.

## Recommended Experimental Policy

### Shared Fairness-Critical Settings

The following settings should be synchronized tightly between the thesis model and the baseline wherever the two pipelines are structurally comparable:

- dataset family: `smd`
- entity selection logic based on `KL(test || train)`
- `window_size = 20`
- `stride = 1`
- same data split rule
- same scaler fit policy on cleaned training split only
- same `batch_size`
- same `num_workers`
- same random seeds: `6`, `36`, `68`
- same encoder family: `cnn_simple`
- same checkpoint monitor metric
- same validation evaluator settings
- same synthetic anomaly family set
- same `classification_label_mode = redlamp_multiclass`
- same `train_balance_classes = true`
- same `use_label_refurbishment = true`
- same `lambda_recon = 0.9`
- same `lambda_cls = 0.1`
- same `num_classes = 12`
- same test metric definitions, all following the pseudo-code contract previously attached by the user:
  - `VUS-PR`
  - `VUS-ROC`
  - `Affiliation-F1`
- same exact training budget: `300` epochs for the thesis run and `300` epochs for the baseline run

### Method-Specific Structure That Should Remain Different

The following differences should remain intact so that each method preserves its identity:

- `redlamp_mlp_baseline.py` remains a baseline without continuous memory, discrete memory, or fusion logic.
- `thesis_multitask.py` retains its own prototype branches, fusion path, and method-specific auxiliary toggles.
- the thesis method remains a **three-stage** method with its own stage budget split that sums to `300`;
- the baseline remains a **single-stage** method with `epochs: 300`.
- The thesis model should not be collapsed into the baseline just to enforce superficial symmetry.

Fairness here should therefore mean **equal data path, equal encoder family, equal label contract, equal metric contract, equal seed policy, equal epoch budget, and equal reporting discipline**, not identical internal training topology.

## Preliminary Programming Scope

### Scope Included

- lock a truthful SMD top-divergence ranking procedure and produce a durable artifact for entity selection;
- add `Affiliation-F1` and `VUS-ROC` to the test metric runtime;
- keep `VUS-PR` on the same reconstructed timeline-level evaluation path;
- add a new mixed orchestration runner for:
  - thesis `three-stage train + val -> test`
  - baseline `single-stage train + val -> test`;
- add new CNN-aligned, fairness-oriented configs for both models and all requested seeds;
- add remote `tmux` launch support for the new sequential runner;
- add tests for the new metrics, config resolution, and orchestration preflight.

### Scope Excluded

- redesign of dataset parsing,
- redesign of evaluator reconstruction semantics,
- redesign of model output contracts,
- broad refactoring of the thesis or baseline model files,
- multi-GPU logic,
- online adaptation work,
- SWaT support in this cycle.

## File-Level Implementation Map

### Primary New Files

- `scripts/rank_smd_train_test_divergence.py`
  - new utility to rank SMD entities by a finalized train-vs-test divergence measure;
  - should emit a machine-readable artifact and a human-readable summary.

- `scripts/run_comparative_smd_experiments.py`
  - new orchestration script that:
    - accepts one or more experiment config paths,
    - dispatches thesis runs to `scripts/run_three_stage_offline_pretraining.py`,
    - dispatches baseline runs to `scripts/train.py`,
    - resolves the correct checkpoint to evaluate for each path,
    - runs `scripts/evaluate.py`,
    - writes a run manifest and failure summary.

- `scripts/launch_tmux_comparative_smd_experiment.sh`
  - new shell wrapper for resilient remote execution on the GPU server.

- `tests/test_smd_divergence_ranking.py`
  - focused tests for the new divergence-ranking helper logic.

- `tests/test_sequential_train_eval_runner.py`
  - tests dry-run and preflight behavior of the new orchestration layer.

- `tests/test_evaluation_range_metrics.py`
  - focused tests for `Affiliation-F1`, `VUS-ROC`, and reconstructed-timeline metric plumbing.

### Primary Modified Files

- `src/metrics/pointwise.py`
  - extend the active metric runtime with:
    - `VUS-ROC`
    - `Affiliation-F1`
  - keep `VUS-PR` in the same active metric surface to avoid duplication.

- `src/engine/evaluator.py`
  - only if necessary to propagate extra metric outputs or additional serialized curve fields.
  - avoid changing reconstruction logic unless a bug is discovered.

- `scripts/evaluate.py`
  - extend artifact writing so the new metrics are saved and logged consistently.

- `scripts/run_three_stage_offline_pretraining.py`
  - modify only if needed so the final thesis checkpoint path for post-training evaluation is surfaced unambiguously in its manifest or execution report.

- `src/core/config.py`
  - only if new config fields are required for metric toggles, divergence ranking settings, or orchestration metadata.
  - avoid broad config-schema churn.

- `src/models/redlamp_mlp_baseline.py`
  - ideally no behavioral refactor.
  - only modify if a fairness-critical config field is currently unsupported in the CNN-aligned experiment path.

- `src/models/thesis_multitask.py`
  - ideally no algorithmic refactor for this plan.
  - only modify if a fairness-critical config field is currently unsupported or incorrectly defaulted for the three-stage CNN-aligned path.

### Primary New Config Files

- two new data configs under `configs/data/` for the two selected SMD entities, both using:
  - `window_size: 20`
  - `stride: 1`
  - a fixed safe `num_workers`
  - a fixed shared `batch_size`

- one new baseline model config under `configs/model/`, derived from `redlamp_cnn_baseline.yaml`, but locked to the shared fair-comparison settings.

- one new thesis model config under `configs/model/`, derived from `thesis_multitask_three_stage_window20.yaml`, but locked to the shared fair-comparison settings and with any unnecessary diagnostic overhead disabled.

- one new task config under `configs/task/` only if the current balanced task config is still semantically ambiguous. Otherwise reuse:
  - `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`

- twelve experiment configs under a new dedicated experiment subdirectory, corresponding to:
  - 2 entities
  - 2 models
  - 3 seeds

### Secondary Modified Files

- `tests/test_config_loading.py`
- `tests/test_redlamp_aligned_configs.py`
- `tests/test_evaluator_thresholding.py`

These should be modified only as needed to cover the newly added aligned configs and metric outputs.

## Planned Programming Sequence

### Phase 1: Lock the `KL(test || train)` Ranking Procedure and Target Entities

This phase should happen before any main experiment config is created.

The repository currently does not contain a truthful final answer to the question: “Which two SMD entities have the largest `KL(test || train)` divergence?” Therefore the first implementation work should:

1. add a deterministic divergence-ranking script over raw SMD train/test entity pairs;
2. output a durable artifact under `documents/logs/06-25-2026/research/` or a new same-day research note;
3. lock the exact top two entities before fairness configs are finalized.

This phase is critical because entity choice changes the full experiment matrix and output directory naming.

### Phase 2: Implement the Missing Test Metrics on Reconstructed Timelines

This phase should add `Affiliation-F1` and `VUS-ROC` into the active evaluation runtime, while re-checking that `VUS-PR` remains semantically aligned with the user-provided pseudo-code.

The implementation principle should be:

- metrics operate on reconstructed timeline-level `point_scores` and `point_labels`,
- not on raw per-window scores,
- not in a notebook-only post-processing path,
- and follow the pseudo-code contract previously supplied by the user as closely as the current evaluator contract allows.

This keeps metric semantics consistent with the already-implemented VUS-PR path and with the 06-25 research note.

### Phase 3: Build the Additive Mixed Train-Then-Evaluate Runner

This phase should add a new small orchestration script rather than modifying `scripts/train.py` or trying to absorb the thesis three-stage runner.

The new runner should:

1. validate all provided experiment configs with `load_experiment_config(...)`;
2. verify dataset roots and artifact path uniqueness;
3. dispatch thesis runs to `scripts/run_three_stage_offline_pretraining.py`;
4. dispatch baseline runs to `scripts/train.py`;
5. run training sequentially on the single GPU;
6. locate the evaluation checkpoint for each path, preferably the `best.pt` selected by `val_realistic_vus_pr`;
7. run evaluation immediately after training;
8. write a manifest that records config path, exit codes, checkpoint path, and evaluation artifact paths.

This runner should support `--dry-run` and `--preflight-only` modes before main remote execution.

### Phase 4: Add Fair CNN-Aligned Experiment Configs

This phase should create a dedicated family of experiment configs rather than overloading older configs that were written for `machine-2-1` or mixed historical experiments.

The new configs should explicitly lock:

- `encoder_family: cnn_simple` in both model families,
- same task config,
- same validation evaluator settings,
- same wandb logging policy,
- exact `epochs: 300` for baseline configs,
- exact three-stage epoch split summing to `300` for thesis configs,
- exact seed values `6`, `36`, `68`,
- output directory names that encode model, entity, and seed clearly.

This phase should also remove fairness-confusing leftovers from the thesis three-stage config and the baseline CNN config, especially any diagnostic-only settings that create unnecessary runtime overhead or confuse user interpretation.

### Phase 5: Add Remote `tmux` Execution Support

This phase should add a small shell launcher for the new mixed runner.

The launcher should:

- create or replace a named `tmux` session,
- print the exact attach command,
- tee logs into a durable file under `outputs/tmux_logs/`,
- run a preflight-only stage first if requested,
- then run the main sequential matrix.

Because only one RTX 3090 is available, the main execution mode should stay **sequential**, not parallel.

### Phase 6: Verification Before Full Server Execution

Before the first long run, the repository should pass:

- config load tests for all new YAMLs,
- unit tests for the new metrics,
- dry-run / preflight tests for the new mixed runner,
- one smoke three-stage thesis experiment on one selected entity,
- one smoke single-stage baseline experiment on one selected entity,
- one smoke evaluation run that confirms the new metrics appear in output artifacts.

Only after this phase should the full `2 entities x 2 models x 3 seeds` matrix be launched on the server.

## Server Resource Strategy

### GPU Strategy

The RTX 3090 provides 24 GB VRAM and the experiment matrix should run on a single GPU. Therefore:

- main execution should be sequential;
- there should be no concurrent training processes on the GPU;
- fairness-critical batch size should be the **largest batch size that both models can sustain safely**, not the largest batch size that only one model can sustain.

The safest initial lock is to reuse the already-proven SMD window-20 batch scale of `256`, then consider promotion only after smoke verification. If a throughput probe is later desired, it should be an explicit preflight experiment and not an implicit assumption.

### CPU Worker Strategy

The loader helper currently supports `num_workers: auto`, but that path resolves to the full visible CPU count. For a 64-core EPYC machine, that behavior is unnecessarily risky for a single-GPU thesis run.

Therefore the plan should **not** use `num_workers: auto` for this experiment family.

The recommended policy is:

- use an explicit worker count;
- begin from a proven safe value close to the existing `machine-3-4` config, namely `16`;
- if throughput inspection later shows clear host-side idling, test promotion to `20` or `24` in smoke mode before locking the final config;
- do not exceed the user’s stated safety envelope.

This is the most conservative way to use the available CPU without creating avoidable loader instability.

### Memory and Logging Strategy

The server has 60 GB RAM, which is sufficient for:

- one training process,
- one evaluation process,
- persistent workers,
- wandb online logging,
- JSON artifact writing.

WandB should remain enabled in the new experiment configs, because both `train.py` and `evaluate.py` already support logging and artifact persistence. The main goal here is consistency of experiment records, not novelty.

## Risk and Mitigation

### Risk: The `KL(test || train)` Ranking Is Still Ambiguous at the Channel-Aggregation Level

If the repository does not lock how channelwise KL scores are reduced into one entity-level score, it may still prepare the wrong top-two entity pair even though the divergence family has already been chosen.

Mitigation:

- formalize the divergence definition first, including the channel aggregation rule,
- generate a durable ranking artifact,
- only then create the final experiment matrix.

### Risk: Fairness Drift Between the Two Model Families Because the Training Topology Is Intentionally Different

The thesis and baseline paths intentionally use different training topology. Even small leftover differences in task config, label policy, diagnostic toggles, or evaluation settings could contaminate comparison fairness beyond the intended method difference.

Mitigation:

- add a dedicated fair-comparison config family,
- isolate it from older historical configs,
- add config-loading tests that assert shared fairness-critical keys match exactly,
- and document explicitly which asymmetries are method-defining rather than accidental.

### Risk: Metric Drift Between Validation and Test

If `Affiliation-F1` and `VUS-ROC` are only added in a notebook or ad hoc post-processing script, the test path will drift away from the repository’s official evaluator.

Mitigation:

- implement the new metrics in the shared evaluation runtime,
- serialize them through `evaluation_metrics.json`,
- and cover them with tests.

### Risk: Single-GPU Run Fails Midway Because Loader Settings Are Too Aggressive

If `num_workers` or `batch_size` are pushed directly to a high value without smoke verification, the full run may fail after hours.

Mitigation:

- keep the main runner sequential,
- use explicit worker counts,
- run smoke configs first,
- only promote batch size or workers after the smoke path passes.

### Risk: Orchestration Logic Hides Failures

If the new mixed runner does not record exit codes and checkpoint paths explicitly, it will be hard to audit which runs actually completed and which metric artifacts belong to which training job.

Mitigation:

- write a run manifest,
- fail fast on non-zero return codes,
- and save evaluation outputs immediately after each run.

## Validation Plan

The minimum validation set for this implementation cycle should include:

1. config-load checks for all new data/model/task/experiment YAMLs;
2. a unit test that verifies `Affiliation-F1` on a toy reconstructed timeline;
3. a unit test that verifies `VUS-ROC` on a toy reconstructed timeline;
4. a regression test that ensures `VUS-PR` still works after the metric extension;
5. a dry-run test for the mixed runner;
6. a preflight-only test for the mixed runner;
7. one smoke three-stage thesis run on one selected entity;
8. one smoke single-stage baseline run on one selected entity;
9. one smoke evaluation run per model family that confirms:
   - `VUS-PR`
   - `VUS-ROC`
   - `Affiliation-F1`
   appear in serialized evaluation artifacts.

## Resolved Decisions

The following experiment-level decisions are now locked:

1. The top-two SMD entities must be selected by **`KL(test || train)`** rather than by Jensen-Shannon divergence or `KL(train || test)`.
2. The thesis method must remain on the **three-stage** path.
3. The baseline must remain on the **basic single-stage** path.
4. All three test metrics must follow the user-provided pseudo-code contract as closely as possible:
   - `VUS-PR`
   - `VUS-ROC`
   - `Affiliation-F1`
5. The exact epoch budget is locked at **`300` per method per run**.

## Remaining Micro-Clarification

One micro-clarification still remains before entity selection can be considered fully locked:

- when computing one entity-level `KL(test || train)` score from the 38 SMD channels, should the repository use the **mean** channelwise KL, the **median**, the **sum**, or another explicit reducer?

## Recommended Next Step

The next step should be to lock that final channel-aggregation detail, then write a **detail-level implementation note** that expands this preliminary plan into exact file edits, test commands, config naming, and rollout order before implementation begins.
