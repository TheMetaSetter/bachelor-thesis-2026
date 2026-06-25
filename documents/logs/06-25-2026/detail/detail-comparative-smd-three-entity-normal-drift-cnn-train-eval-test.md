---
date: 2026-06-25 18:04:00 +0700
researcher: TheMetaSetter
git_commit: c66927b06d3b94f3505792cd3aaf66c0fc6b1064
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for comparative SMD experiments on three normal-drift entities with thesis three-stage CNN and RedLamp baseline single-stage CNN pipelines"
tags: [detail, smd, cnn, three-stage, redlamp, evaluation, metrics, tmux]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Detailed Plan: Comparative SMD Experiments on Three Normal-Drift Entities with Thesis Three-Stage CNN and RedLamp Baseline Single-Stage CNN Pipelines

## Objective

The objective of this implementation cycle is to prepare the repository for a fair, reproducible, and server-safe comparative experiment matrix over the three official SMD entities:

- `machine-3-9`
- `machine-3-1`
- `machine-1-6`

These entities are locked because they show the strongest **drift in normality** under the chosen criterion, namely `KL(test_normal_only || train)`.

The comparative matrix shall preserve the method identity of both pipelines:

- `src/models/thesis_multitask.py` shall run through the existing **three-stage offline pre-training** path with the `cnn_simple` encoder family and an exact `300`-epoch budget split across the three-stage schedule.
- `src/models/redlamp_mlp_baseline.py` shall run through the existing **single-stage** offline path with the `cnn_simple` encoder family and an exact `300`-epoch budget in one stage.

The shared test metrics shall be:

- `VUS-PR`
- `VUS-ROC`
- `Affiliation-F1`

All three metrics shall be computed from the reconstructed timeline-level anomaly scores and labels, following the user-provided pseudo-code contract as closely as possible without changing the repository’s current overlap-reconstruction semantics.

## Input Artifacts

This detailed plan is grounded in the following artifacts:

- [plan-sequential-fair-cnn-smd-top2-divergence-train-eval-test.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-25-2026/plan/plan-sequential-fair-cnn-smd-top2-divergence-train-eval-test.md)
- [research-sequential-train-eval-smd-3-4-and-timeline-reconstruction-metrics.md](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HO%CC%A3C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/%C4%90H%20KHOA%20HO%CC%A3C%20TU%CC%9B%CC%A3%20NHIE%CC%82N/Khoa%CC%81%20lua%CC%A3%CC%82n%20to%CC%82%CC%81t%20nghie%CC%A3%CC%82p/bachelor-thesis-2026/documents/logs/06-25-2026/research/research-sequential-train-eval-smd-3-4-and-timeline-reconstruction-metrics.md)

## Repository Compatibility and `codebase_preferences.md` Constraints

All implementation that follows this plan shall satisfy the repository preferences document, especially the following constraints:

1. **Readability-first and least-codepath discipline**
   - new logic shall prefer explicit names, short helpers, and obvious control flow;
   - comparative logic shall be added as a thin layer around current entrypoints rather than hidden behind deep abstractions.

2. **`1 model - 1 file` preservation**
   - no model-related comparative logic shall be moved out of `src/models/redlamp_mlp_baseline.py` or `src/models/thesis_multitask.py` unless it is clearly not model logic;
   - this cycle shall avoid broad model refactors and preserve self-contained model files.

3. **TSLib-style loader alignment**
   - loader and dataset changes shall preserve the current parse -> clean -> scale -> window -> collate flow rather than introducing a second data-loading stack;
   - the comparative work shall stay compatible with future dataset parsers that emit the same raw-sequence contract.

4. **Future dataset compatibility**
   - SMD-specific selection logic shall remain isolated to SMD-specific scripts, data YAMLs, and experiment YAMLs;
   - generic metric logic, evaluator logic, and orchestration logic shall continue to operate over dataset/model contracts rather than over hard-coded SMD assumptions;
   - any new runner shall dispatch by resolved config semantics and existing registry ownership, not by string-matching file paths alone.

5. **Ablation friendliness**
   - fairness-critical changes shall be surfaced through explicit config files;
   - disabling or replacing one comparative component later should not require rewriting core engine code.

## Final Audit Findings Before Programming

This section records the last codebase audit that must constrain the implementation.

### Audit Finding 1: The Requested Scaling Semantics Already Exist if Each Run Targets Exactly One Entity

The user requested that a standard scaler be fit on the train part of each selected sequence and then applied to the corresponding test part. The current repository already satisfies this semantics **provided that each experiment config selects exactly one SMD entity**.

The current implementation path is:

1. `SMDDatasetParser.parse()` splits each selected raw training sequence into train and validation segments and keeps the raw test sequence intact in `src/data/datasets/smd.py`.
2. `_build_dataset_bundle_from_sequences(...)` creates one `SequenceStandardScaler`, fits it on `cleaned_sequences["train"]`, and applies it to `train`, `val`, and `test` in `src/data/loaders.py`.
3. `SequenceStandardScaler.fit(...)` concatenates all training points from the provided train sequences and computes feature-wise mean and standard deviation in `src/data/scalers.py`.

This means:

- if a config contains one entity only, the scaler is fit on that entity’s train split only and then applied to its val/test splits;
- if a config contains multiple entities, the scaler becomes multi-entity and would violate the requested semantics.

Therefore the comparative experiment family must be implemented as **one entity per run**, not as multi-entity joint training.

### Audit Finding 2: The Thesis Three-Stage Runner Already Executes Evaluation Internally

The preliminary plan assumed that both methods would always require an explicit external `evaluate.py` call after training. The final audit shows that this is true for the baseline, but **not** for the thesis three-stage runner.

The existing thesis three-stage path:

- materializes an evaluation reference config and checkpoint path in `materialize_three_stage_run_manifest(...)`;
- builds an evaluation command in `build_three_stage_execution_commands(...)`;
- executes that evaluation command at the end of `execute_three_stage_plan(...)`.

Therefore:

- the comparative orchestration layer shall **not** call `scripts/evaluate.py` a second time for thesis runs;
- the comparative orchestration layer shall treat the three-stage runner as an end-to-end `train + val -> test` command;
- only the baseline path shall need an external `train.py` followed by `evaluate.py`.

### Audit Finding 3: Timeline Reconstruction Semantics Are Already Fixed and Must Be Preserved

The evaluator currently reconstructs test scores and labels from overlapped windows with these hard-coded semantics:

- score aggregation across overlaps: `mean`
- label aggregation across overlaps: `max`
- entity-first reconstruction, then global concatenation across entities

These semantics are already implemented in `src/engine/evaluator.py` and should remain unchanged. The new metrics must plug into this existing reconstructed timeline contract rather than introducing a second reconstruction path.

### Audit Finding 4: Metric Coverage Is Still Incomplete

The active runtime already supports `VUS-PR` in `src/metrics/pointwise.py`, but the codebase still lacks active implementations for:

- `VUS-ROC`
- `Affiliation-F1`

This remains the primary algorithmic gap on the evaluation side.

### Audit Finding 5: Explicit Worker Counts Are Safer Than `num_workers: auto`

The loader helper `_resolve_data_loader_num_workers(...)` supports `num_workers: auto`, but the current implementation resolves that value to the full visible CPU count with a floor. On the target AMD EPYC server, this is unnecessarily risky for a single-GPU long-running thesis experiment. The comparative config family should therefore use explicit worker counts.

### Audit Finding 6: Future Dataset Compatibility Is Preserved Best by Keeping SMD-Specific Logic at the Edge

The current codebase is already organized around generic contracts:

- raw sequence parsing inside dataset-specific parsers;
- common scaling and windowing in `src/data/`;
- registry-driven dataset/model construction;
- shared evaluator logic operating on batch and model-output contracts.

Therefore the comparative implementation should:

- keep SMD-specific drift ranking inside an SMD-specific script;
- keep SMD entity selection inside data and experiment YAMLs;
- avoid hard-coding SMD assumptions into generic evaluator or trainer modules unless the logic is truly generic.

## Locked Experimental Matrix

### Official Entities

The official SMD entities are:

- `machine-3-9`
- `machine-3-1`
- `machine-1-6`

### Method Topology

- Thesis method: `three-stage`
- Baseline method: `single-stage`

### Seeds

- `6`
- `36`
- `68`

### Budget

- Thesis: exact `300` epochs per run, preserving the three-stage split
- Baseline: exact `300` epochs per run

### Comparative Granularity

The implementation shall create a matrix of:

- `3 entities`
- `2 methods`
- `3 seeds`

for a total of **18 main runs**, plus a small smoke subset.

## Contract Definitions

### Dataset Contract

The comparative dataset contract shall remain the current repository contract:

```python
{
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict[str, object]],
}
```

The dataset-level implementation rule for this experiment family is:

- exactly one SMD entity per run;
- `window_size = 20`;
- `stride = 1`;
- `validation_split_ratio = 0.2`;
- feature-wise standardization fit on the train split of that entity only;
- the same fitted scaler applied to val and test of that entity.

This contract preserves future compatibility because another dataset can reuse the same scaler and windowing path as long as it emits the current raw-sequence contract.

### Encoder Contract

The shared encoder-family contract shall remain:

- input: `x in R^{B x L x D}`
- output: `hidden in R^{B x L x H}`

Both methods shall use the already-supported `cnn_simple` family. No new encoder adapter shall be introduced. The comparative work shall only reuse the existing encoder-family configuration surface.

### Model Contract

The public model-output contract shall remain unchanged. The implementation cycle shall not add new public tensor keys to the model outputs unless strictly required by metric plumbing.

Method-specific internal contracts shall remain:

- baseline: reconstruction head + classification head, no prototype memory
- thesis: prototype branches + fusion + task heads + three-stage phase control

### Task Contract

The comparative task contract shall be shared across both methods:

- `classification_label_mode = redlamp_multiclass`
- `train_balance_classes = true`
- `use_synthetic_augmentation = true`
- `use_synthetic_validation = true`
- `val_realistic = true`
- `val_realistic_source = test_same_scope`

The shared task YAML shall be treated as the strategy object for task semantics. The comparative implementation shall therefore use **strategy through configuration**, not hard-coded special cases in the model files.

### Training Engine Contract

The training-engine contract shall remain registry-driven:

- datasets are built through `build_dataset(...)`
- models are built through `build_model(...)`
- training is driven by existing entrypoints

The comparative layer shall compose existing entrypoints rather than replace them:

- thesis path: `scripts/run_three_stage_offline_pretraining.py`
- baseline path: `scripts/train.py` followed by `scripts/evaluate.py`

This contract is intentionally future-friendly: another dataset family should be able to reuse the same comparative orchestration style if its experiment configs resolve through the same registry and entrypoint surfaces.

## Design Pattern Application

### Composition Over Inheritance

The comparative orchestration layer shall be a thin composition wrapper over existing scripts. It shall not introduce subclass hierarchies for trainers, evaluators, or models. This keeps the comparative work additive and avoids duplicating method-specific training semantics.

### Adapter Pattern for Encoders

No new adapter implementation is required. The codebase already exposes the required encoder variability through the existing `encoder_family: cnn_simple` surface. The detail plan therefore preserves the current implicit adapter behavior by configuration rather than by new code.

### Strategy Pattern for Tasks

Task behavior shall remain selected through the task config rather than by branching in the comparative runner. The comparative experiment family shall therefore use one dedicated task YAML as the shared task-strategy definition for both methods.

### Registry / Factory Pattern for Datasets and Models

The comparative runner shall rely on the existing dataset and model factory flow exposed through:

- `load_experiment_config(...)`
- `build_dataset(...)`
- `build_model(...)`

No new registry layer shall be created.

## Phase 1: Lock the Official Three-Entity Comparative Scope and Data Semantics

### Phase Summary

This phase converts the already-approved experimental scope into durable repository artifacts. The purpose is to prevent later ambiguity about which SMD entities are official, how the scaler semantics are enforced, and whether the comparative matrix is one-entity-per-run or multi-entity-per-run.

This phase must keep all SMD-specific selection logic at the script and config layer so the common loader stack remains reusable for future datasets.

### File-Level Edits

**Create**

- `documents/logs/06-25-2026/research/research-smd-normality-drift-three-official-entities.md`
- `scripts/rank_smd_train_test_divergence.py`
- `configs/data/smd_rtx3090_machine_1_6_20_stride1.yaml`
- `configs/data/smd_rtx3090_machine_3_1_20_stride1.yaml`
- `configs/data/smd_rtx3090_machine_3_9_20_stride1.yaml`

**Modify**

- `documents/logs/06-25-2026/plan/plan-sequential-fair-cnn-smd-top2-divergence-train-eval-test.md`
  - only if a short correction note is desired so the plan reflects the final official three-entity scope instead of the earlier two-entity wording.

### Explicit Edit Content

1. The new research note shall record:
   - the full-test `KL(test || train)` ranking,
   - the normal-only `KL(test_normal_only || train)` ranking,
   - the reducer values `mean`, `max`, and `top-5 mean`,
   - the user-approved final official set:
     - `machine-3-9`
     - `machine-3-1`
     - `machine-1-6`

2. `scripts/rank_smd_train_test_divergence.py` shall:
   - read raw SMD train, test, and test_label files;
   - support a flag for normal-only test filtering;
   - compute histogram-based `KL(test || train)` or `KL(test_normal_only || train)`;
   - report per-entity reducer outputs:
     - `mean`
     - `max`
     - `top5_mean`
   - write machine-readable outputs such as JSON or CSV under `outputs/`.

3. Each new data config shall:
   - contain exactly one `entity_id`;
   - use `window_size: 20`;
   - use `stride: 1`;
   - use the same explicit `batch_size`;
   - use the same explicit `num_workers`;
   - keep `validation_split_ratio: 0.2`.

4. No code change shall be made to `src/data/scalers.py` or `src/data/loaders.py` in this phase. The requested scaler semantics will be satisfied by config scoping, not by refactoring the loader stack.

5. `scripts/rank_smd_train_test_divergence.py` shall be clearly marked as SMD-specific and shall not become a hidden dependency of generic training or evaluation code.

### Risk Mitigation

- **Evaluation metric inflation:** the entity-ranking script must be marked as dataset-selection analysis only, not as performance reporting.
- **Projector drift / adaptation contamination:** explicitly out of scope; no online adaptation files shall be touched.

### Test Plan

**Create**

- `tests/test_smd_entity_selection_configs.py`
- `tests/test_smd_divergence_ranking.py`

**Validation**

- verify each new data config loads through `load_experiment_config(...)` once referenced by an experiment config;
- verify the ranking script returns one row per SMD entity;
- verify all three official data configs contain exactly one entity each.

### Acceptance Criteria

- The repository contains one durable research artifact that records the official entity choice.
- The repository contains three data configs, one per official entity, each with a single `entity_id`.
- No comparative config created later in the cycle may contain more than one entity.

## Phase 2: Add the Missing Timeline-Level Test Metrics Without Altering Reconstruction Semantics

### Phase Summary

This phase extends the evaluation runtime so that the requested test metrics become first-class outputs of the repository rather than notebook-only post-processing. The core thesis objective here is metric faithfulness: the same reconstructed timeline should support all three metrics for both methods.

Because this phase touches generic evaluation code, every new helper must be written in a dataset-agnostic way and must avoid SMD-only branches.

### File-Level Edits

**Modify**

- `src/metrics/pointwise.py`
- `src/engine/evaluator.py`
- `scripts/evaluate.py`
- `tests/test_evaluator_thresholding.py`

**Create**

- `tests/test_pointwise_range_metrics.py`

### Explicit Edit Content

1. `src/metrics/pointwise.py` shall gain:
   - one explicit implementation for `VUS-ROC` on reconstructed pointwise scores and labels;
   - one explicit implementation for `Affiliation-F1`;
   - helper functions that keep the existing `VUS-PR` path readable rather than overloading `compute_pointwise_metrics(...)` with one large monolithic block.

2. `compute_pointwise_metrics(...)` shall return:
   - existing scalar metrics,
   - `vus_pr`,
   - `vus_roc`,
   - `affiliation_f1`,
   - and `threshold`.

   The new metric keys shall be generic pointwise-evaluation outputs, not SMD-only metric names.

3. `src/engine/evaluator.py` shall remain responsible for:
   - overlap-aware reconstruction,
   - threshold selection,
   - metric dispatch,
   - curve payload generation.

   It shall **not** introduce a second reconstruction strategy.

4. `scripts/evaluate.py` shall continue writing:
   - `evaluation_records.json`
   - `evaluation_metrics.json`
   - `evaluation_curves.json`

   and shall ensure that the new metric keys are included in the serialized metrics and WandB logging payload.

### Interface and Contract Definitions

- The input to the new metric helpers shall be reconstructed timeline arrays, not window arrays.
- Threshold-based metrics shall use the repository’s current threshold selection path.
- `Affiliation-F1` and `VUS-ROC` shall follow the user-provided pseudo-code contract as closely as possible, and any unavoidable implementation approximation shall be documented inline and in tests.

### Risk Mitigation

- **Evaluation metric inflation:** prohibit point-adjusted shortcuts and prohibit window-level metric computation for these three headline metrics.
- **Prototype redundancy / fusion collapse:** metrics must remain architecture-agnostic and operate only on final reconstructed anomaly scores.
- **Adaptation contamination / projector drift:** still out of scope; no online adaptation metric hooks shall be introduced.

### Test Plan

**Create**

- `tests/test_pointwise_range_metrics.py`

The new test file shall include:

- one deterministic toy test for `VUS-ROC`;
- one deterministic toy test for `Affiliation-F1`;
- one regression test that `VUS-PR` remains present and finite where expected;
- one test that the metric helpers reject shape mismatches cleanly.

**Modify**

- `tests/test_evaluator_thresholding.py`
  - extend the evaluator-facing assertions so the reconstructed evaluation path exposes all new metric keys.

### Acceptance Criteria

- `evaluation_metrics.json` contains `vus_pr`, `vus_roc`, and `affiliation_f1` for supported test runs.
- The evaluator continues to reconstruct scores with overlap mean and labels with overlap max.
- Unit tests prove that the new metrics run on reconstructed timeline arrays rather than window arrays.

## Phase 3: Build the Mixed Comparative Orchestration Layer

### Phase Summary

This phase adds the orchestration layer needed to run the full comparative matrix safely on one RTX 3090 without duplicating the already-working training logic of either method. The thesis objective here is methodological rigor with minimal codepath drift.

The orchestration implementation must remain readable and future-compatible. Although the first target is SMD, the runner should still dispatch by config semantics and existing registry ownership so that the same pattern can be reused later for other datasets or entity-based experiment families.

### File-Level Edits

**Create**

- `scripts/run_comparative_smd_experiments.py`
- `tests/test_comparative_runner.py`

**Modify**

- `scripts/run_three_stage_offline_pretraining.py` only if necessary

### Explicit Edit Content

1. `scripts/run_comparative_smd_experiments.py` shall accept a list of experiment configs and dispatch per-method behavior:
   - if `experiment.stage_family == "thesis_three_stage"` or equivalent tag, run `scripts/run_three_stage_offline_pretraining.py`;
   - if `experiment.stage_family == "baseline_single_stage"` or equivalent tag, run:
     1. `scripts/train.py`
     2. `scripts/evaluate.py`

   The dispatch decision should be derived from explicit resolved config fields or a clearly documented comparative run-family marker, not from brittle filename heuristics.

2. The new comparative runner shall:
   - validate config resolution before launching subprocesses;
   - validate unique artifact paths;
   - fail fast on the first non-zero return code;
   - write a comparative manifest and comparative execution report.

3. The thesis dispatch path shall **not** append a second external evaluation call if the three-stage runner already succeeded.

4. `scripts/run_three_stage_offline_pretraining.py` shall only be modified if the external runner needs one additional stable field, such as:
   - a clearly named final evaluation metrics path,
   - or a stable success summary field.

   If its current manifest and execution report are already sufficient, no edit shall be made.

5. The comparative runner shall not become a second training engine. It shall remain a readable subprocess coordinator with minimal responsibilities.

### Interface and Contract Definitions

- The comparative runner shall be a **composition wrapper** over existing entrypoints.
- It shall not import or call model internals directly.
- It shall treat experiment configs as immutable runtime contracts.

### Risk Mitigation

- **Orchestration ambiguity:** write one comparative manifest row per run with:
  - method
  - entity
  - seed
  - config path
  - output dir
  - final status
  - evaluation artifact paths
- **Evaluation metric inflation:** comparative summary shall read metrics from official evaluation outputs only, not recompute them separately.

### Test Plan

**Create**

- `tests/test_comparative_runner.py`

The test file shall include:

- dry-run test for mixed dispatch planning;
- preflight-only test for config and dataset-root validation;
- thesis path test ensuring no duplicate external evaluation command is scheduled;
- baseline path test ensuring evaluation is scheduled after training.

### Acceptance Criteria

- The comparative runner can describe all `18` planned main runs in dry-run mode.
- Thesis runs are dispatched through the three-stage runner exactly once.
- Baseline runs are dispatched through `train.py` then `evaluate.py`.
- The comparative execution report identifies the failed run unambiguously if any subprocess fails.

## Phase 4: Add Dedicated Comparative Config Families for the 18 Main Runs and Smoke Runs

### Phase Summary

This phase creates a clean, fairness-oriented config family that isolates the comparative matrix from older historical experiments. The purpose is to eliminate user-facing ambiguity and make every run auditable from config alone.

This phase is also where future compatibility is protected most effectively: dataset-specific and experiment-family-specific differences should stay in YAMLs rather than leaking into common Python modules.

### File-Level Edits

**Create**

- `configs/model/redlamp_cnn_baseline_comparative_smd.yaml`
- `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`
- `configs/task/multitask_tsad_redlamp_multiclass_window20_comparative.yaml`
- `configs/experiment/comparative/` with:
  - `9` baseline main configs
  - `9` thesis main configs
  - `1` baseline smoke config
  - `1` thesis smoke config

### Explicit Edit Content

1. The baseline comparative model config shall inherit the semantics of `configs/model/redlamp_cnn_baseline.yaml` and lock:
   - `encoder_family: cnn_simple`
   - `lambda_recon: 0.9`
   - `lambda_cls: 0.1`
   - `use_label_refurbishment: true`
   - `num_classes: 12`

2. The thesis comparative model config shall inherit the semantics of `configs/model/thesis_multitask_three_stage_window20.yaml` and lock:
   - `encoder_family: cnn_simple`
   - `lambda_recon: 0.9`
   - `lambda_cls: 0.1`
   - `use_label_refurbishment: true`
   - existing three-stage memory and fusion settings
   - existing exact stage-budget semantics

3. The shared comparative task config shall lock:
   - balanced class semantics
   - realistic validation semantics
   - shared anomaly family list
   - explanatory comments clarifying that `anomaly_probability` is not the class-balancing mechanism when `train_balance_classes = true`

4. Each main experiment config shall encode:
   - method
   - entity
   - seed
   - output directory
   - checkpoint directory
   - logging and WandB identifiers

5. Baseline main configs shall set:
   - `epochs: 300`
   - explicit data config path per entity
   - model config path to the comparative baseline model YAML
   - task config path to the comparative shared task YAML

6. Thesis main configs shall set:
   - `epochs: 300`
   - `three_stage.expected_total_training_epochs: 300`
   - the exact stage split summing to `300`
   - explicit data config path per entity
   - comparative thesis model config path
   - shared comparative task config path

7. Smoke configs shall reduce runtime conservatively through config-level controls only. They shall not invent alternate algorithmic semantics.

8. The comparative YAML family shall be structured so that another dataset family can later add its own entity-specific configs without changing the comparative runner’s core logic.

### Interface and Contract Definitions

- Configs are the public source of truth for comparative fairness.
- Fairness-critical fields shall be shared across methods when structurally comparable.
- Method-defining asymmetries shall be preserved and documented in comments or the detail note.

### Risk Mitigation

- **Fairness drift:** create isolated comparative configs instead of editing historical configs in place.
- **Prototype redundancy / fusion collapse:** do not alter thesis internal regularization defaults in this cycle unless smoke verification reveals a runtime bug.
- **User-facing ambiguity:** use explicit naming that includes method, entity, seed, and stage family.

### Test Plan

**Create**

- `tests/test_comparative_config_loading.py`

**Modify**

- `tests/test_redlamp_aligned_configs.py`

The config tests shall assert:

- `18` main configs resolve cleanly;
- each baseline config uses `epochs: 300`;
- each thesis config uses exact three-stage budget sum `300`;
- shared fairness-critical fields match across methods where intended;
- each experiment config references exactly one entity-specific data YAML.

### Acceptance Criteria

- The repository contains an isolated comparative config family for all main and smoke runs.
- Every main config resolves through `load_experiment_config(...)`.
- No comparative main config mixes more than one entity.
- WandB logging remains enabled and named distinctly per run.

## Phase 5: Add Remote `tmux` Launch Support and Final Verification

### Phase Summary

This phase prepares the repository for robust remote execution on the target server. The goal is to minimize operational risk while keeping the full comparative matrix reproducible and auditable.

The `tmux` wrapper should remain operational glue only. It shall not embed dataset-specific training logic that belongs in experiment configs or Python orchestration.

### File-Level Edits

**Create**

- `scripts/launch_tmux_comparative_smd_experiment.sh`

**Modify**

- no existing shell launcher unless a shared helper is clearly reusable without expanding the write surface unnecessarily.

### Explicit Edit Content

1. The `tmux` launcher shall:
   - create or replace a named session;
   - print the attach command;
   - write a durable log file under `outputs/tmux_logs/`;
   - support a preflight-only mode;
   - launch the mixed comparative runner in sequential mode.

2. The launcher shall default to sequential execution because:
   - one RTX 3090 is available;
   - thesis three-stage and baseline runs are both GPU-bound;
   - fairness is easier to audit when the run order is explicit.

3. The launcher shall not use `num_workers: auto`. Comparative data configs shall carry the explicit worker count.

### Test Plan

**Validation Commands**

- config-load smoke for all new YAMLs
- dry-run for the mixed runner
- preflight-only comparative run
- one thesis smoke run
- one baseline smoke run

**Integration Validation**

- confirm that thesis smoke produces three-stage manifest, execution report, and evaluation outputs;
- confirm that baseline smoke produces train artifacts and evaluation outputs;
- confirm that all three requested metrics appear in the final evaluation metrics files.

### Acceptance Criteria

- The launcher prints:
  - session name
  - log path
  - attach command
- The mixed runner can be started from `tmux` without requiring the local client to remain connected.
- One thesis smoke run and one baseline smoke run complete end-to-end before the main `18`-run matrix is launched.

## Cross-Phase Risk Mitigation

### Prototype Redundancy

This cycle does not redesign the thesis architecture. The mitigation is therefore procedural:

- preserve the current thesis prototype architecture;
- do not introduce new prototype branches;
- isolate the comparative work to configs, metrics, and orchestration.

This mitigation is also consistent with `codebase_preferences.md`, because it avoids spreading model logic across new files.

### Fusion Collapse

This cycle does not modify the current fusion mechanism. The mitigation is:

- preserve current thesis fusion semantics;
- use shared validation monitoring through `val_realistic_vus_pr`;
- avoid confounding fusion changes with metric and orchestration changes.

### Adaptation Contamination

Online adaptation is excluded from scope. No comparative file in this cycle shall import or activate online adaptation code paths.

### Projector Drift

Online projector logic is excluded from scope. The detail plan explicitly forbids touching online projector modules in this cycle.

### Evaluation Metric Inflation

All three headline metrics shall be computed:

- from the same reconstructed timeline,
- with the same threshold selection path,
- inside the official evaluation runtime,
- and serialized through the same evaluation artifacts.

No notebook-only or post hoc metric path shall be treated as authoritative for the comparative matrix.

## Full Validation Matrix

### Unit Tests

- `tests/test_smd_divergence_ranking.py`
- `tests/test_smd_entity_selection_configs.py`
- `tests/test_pointwise_range_metrics.py`
- `tests/test_comparative_runner.py`
- `tests/test_comparative_config_loading.py`

### Regression Tests to Reuse or Extend

- `tests/test_evaluator_thresholding.py`
- `tests/test_redlamp_aligned_configs.py`
- `tests/test_loader_worker_resolution.py`

### Smoke and Integration Checks

1. One thesis smoke run on one official entity.
2. One baseline smoke run on one official entity.
3. One dry-run mixed comparative orchestration pass.
4. One preflight-only mixed comparative orchestration pass.
5. Verification that:
   - `evaluation_records.json` exists,
   - `evaluation_metrics.json` exists,
   - `evaluation_curves.json` exists,
   - all three headline metrics are present.

## Final Acceptance Criteria

The implementation guided by this plan will be considered complete only if all conditions below are satisfied:

1. The official entity set is durably recorded as:
   - `machine-3-9`
   - `machine-3-1`
   - `machine-1-6`
2. Comparative runs are one-entity-per-run, thereby preserving the requested per-entity standard-scaler semantics without refactoring the loader foundation.
3. The baseline path exposes an end-to-end `train + val -> test` flow through `train.py` then `evaluate.py`.
4. The thesis path exposes an end-to-end `train + val -> test` flow through `run_three_stage_offline_pretraining.py` without duplicate external evaluation.
5. The test runtime returns `VUS-PR`, `VUS-ROC`, and `Affiliation-F1` from reconstructed timeline-level signals.
6. The repository contains dedicated comparative configs for all `18` main runs.
7. The exact `300`-epoch budget is preserved for every run:
   - baseline single-stage `300`
   - thesis three-stage total `300`
8. The comparative launcher supports safe sequential server execution through `tmux`.
9. The smoke subset passes before the main matrix is launched.
10. New comparative logic remains readable, isolated, and compatible with future dataset families by keeping SMD-specific behavior at the script/YAML edge rather than in generic core modules.

## Recommended Execution Order After This Detail Plan

After this detail plan is approved, the recommended programming order is:

1. Phase 1
2. Phase 2
3. Phase 4
4. Phase 3
5. Phase 5

This order intentionally locks scope and metric correctness before orchestration polish. The comparative runner should be written only after the repository already knows:

- which entities are official,
- how metrics are computed,
- and which configs it must dispatch.
