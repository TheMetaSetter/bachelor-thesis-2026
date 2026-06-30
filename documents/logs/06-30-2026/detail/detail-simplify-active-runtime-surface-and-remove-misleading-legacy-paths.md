---
date: 2026-06-30 15:05:00 +0700
planner: Codex
git_commit: ddd20afb2f45c83a17fa93d54624789b783ca29d
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed cleanup plan for the active runtime surface before benchmark execution"
tags: [detail, cleanup, runtime, configs, tests, benchmark, simplicity]
status: draft
last_updated: 2026-06-30
last_updated_by: Codex
source_structure: documents/logs/06-30-2026/structure/structure-simplify-active-runtime-surface-and-remove-misleading-legacy-paths.md
source_plan: documents/logs/06-30-2026/plan/plan-simplify-active-runtime-surface-and-remove-misleading-legacy-paths.md
source_research: documents/logs/06-30-2026/research/research-code-paths-not-yet-simplified-and-easy-to-misunderstand.md
---

# Detail: Simplify the active runtime surface and remove misleading legacy paths

## Goal

Implement one clean, benchmark-safe runtime surface for the thesis codebase before large benchmark execution. The active surface must use one canonical baseline identity, must remove `val_realistic` and all test-derived anomaly-prior semantics from the active benchmark path, must keep the current benchmark-safe loader and evaluator behavior intact, and must remain simple enough that future interaction with the repository is easier rather than harder.

## Locked Decisions

The following decisions are already fixed and must be treated as implementation constraints:

1. `redlamp_baseline` is the only canonical active baseline model identity.
2. `redlamp_mlp_baseline` and `redlamp_cnn_baseline` should no longer remain first-class active runtime identities.
3. The active benchmark surface must not use `val_realistic`, `val_realistic_source`, or `val_anomaly_rate_override`.
4. The active benchmark checkpoint-selection path must use `val_synth_*`, especially `val_synth_vus_pr`.
5. The active benchmark protocol must not derive a synthetic validation anomaly ratio from the real test split.
6. The benchmark-safe loader and evaluator contracts already implemented for split safety, coverage reporting, and mixed-label test validation must be preserved.
7. The cleanup should prefer deletion, renaming, and boundary simplification over adding new abstraction layers.

## Current Code Reality That Drives This Plan

The codebase already contains the seeds of the final simpler design:

- The new benchmark configs under `configs/experiment/benchmark/` already use `checkpoint_monitor_metric: val_synth_vus_pr`.
- The trainer already contains a valid `val_synth` path and threshold handling.
- The benchmark task config `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml` already sets `val_realistic: false`.

However, the repository still contains three large confusion sources:

1. The old baseline naming still exists in `src/core/config.py`, `scripts/*.py`, many experiment YAML filenames, and several tests.
2. `val_realistic` still exists as a generic-seeming runtime concept in `src/engine/trainer.py`, `src/models/thesis_multitask.py`, `src/models/redlamp_baseline.py`, `configs/task/*.yaml`, and a large test surface.
3. Some helper texts, launcher scripts, and tests still teach the old public contract even though the benchmark-safe contract has already moved on.

This detailed plan therefore focuses on cleanup by removal and alignment, not on feature creation.

## Preserved Core Contracts

The cleanup must preserve the following contracts exactly:

### Dataset contract

Each split remains a bundle of windows or raw sequences centered on:

- `x`
- `point_labels`
- `mask`
- `timestamps`
- `meta`

The loader path remains:

1. parse raw split sequences,
2. clean raw values,
3. fit the scaler on train only,
4. transform train, val, and test,
5. windowize with split-specific stride,
6. evaluate only on covered points.

### Encoder and model contract

The active baseline remains encoder-agnostic through `encoder_family`, while the thesis model remains sequence-first with hidden states shaped `[B, L, H]`.

The active model step surface after cleanup should be:

- `training_step`
- `validation_step`
- `synthetic_validation_step`
- `test_step`

The active benchmark path should not depend on `realistic_validation_step`.

### Task contract

The active multitask task surface should keep only the synthetic-validation controls that are still meaningful:

- `use_synthetic_augmentation`
- `use_synthetic_validation`
- `synthetic_train_seed`
- `synthetic_validation_seed`
- `classification_label_mode`
- `freeze_fusion_for_epochs`
- `warmup_alpha_value`
- `warmup_beta_value`
- `anomaly_probability`
- `train_balance_classes`
- `min_segment_fraction`
- `max_segment_fraction`
- `spike_scale`
- `anomaly_visibility_boost`
- `anomaly_families`

The active task contract should no longer include:

- `val_realistic`
- `val_realistic_source`
- `val_anomaly_rate_override`

### Training-engine contract

The trainer remains a small orchestration engine. It should still:

- move batches to device,
- call model-owned step methods,
- aggregate logs,
- compute reconstructed pointwise validation metrics,
- step the scheduler,
- save checkpoints.

After cleanup, its auxiliary validation behavior should be simpler:

- always run `val`,
- run `val_synth` only when the model exposes `synthetic_validation_step` and the task enables synthetic validation,
- never run a separate `val_realistic` branch in the active benchmark path.

## Design Principles

The implementation should keep the current codebase simple through the following restrained design choices:

- **Composition over inheritance**: keep the current model files and trainer orchestration model. Do not introduce a validation-stage class hierarchy.
- **Adapter pattern**: continue using `encoder_family` and dataset parser files as explicit adapters rather than creating polymorphic plugin systems.
- **Strategy pattern**: keep stage behavior model-owned through step methods, but reduce the active strategy set to `train`, `val`, `val_synth`, and `test`.
- **Registry or factory pattern**: keep the thin explicit model and dataset registration in entry scripts, but register only the canonical baseline identity for active runtime use.

## Phase 1 - Canonicalize the baseline identity and remove public alias drift

### Phase summary

This phase removes the largest naming confusion first. The thesis objective is straightforward reproducibility: the same baseline should not appear to be three different models depending on which file a reader opens.

### Files to modify

- `src/core/config.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`
- `scripts/run_comparative_smd_experiments.py`
- `scripts/launch_tmux_comparative_smd_experiment.sh`
- `src/core/config_help.py`
- `src/models/redlamp_mlp_baseline.py`
- `configs/model/redlamp_mlp_baseline.yaml`
- `configs/model/redlamp_cnn_baseline.yaml`
- active experiment configs and tests that still reference `redlamp_mlp_baseline` or `redlamp_cnn_baseline`

### Planned edits

1. In `src/core/config.py`:
   - remove `redlamp_mlp_baseline` from `supported_model_names`,
   - remove `"redlamp_mlp_baseline": redlamp_baseline_model_keys` from `allowed_model_keys_by_model_name`,
   - update any model-name guards that still check `{"redlamp_baseline", "redlamp_mlp_baseline"}` so they check `{"redlamp_baseline"}` only.

2. In `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py`:
   - remove `register_model("redlamp_mlp_baseline", RedLampBaseline)`,
   - update any `model_name in {...}` guards so they only include `redlamp_baseline` for the baseline model.

3. In `scripts/run_comparative_smd_experiments.py`:
   - change `SUPPORTED_BASELINE_MODEL_NAMES` from
     - `redlamp_baseline`
     - `redlamp_mlp_baseline`
     - `redlamp_cnn_baseline`
     to
     - `redlamp_baseline`
   - keep encoder variation represented by model config paths, not by runtime model names.

4. In `src/core/config_help.py`:
   - replace example commands so they point at canonical baseline experiment files only.

5. In model preset files:
   - rename `configs/model/redlamp_mlp_baseline.yaml` to `configs/model/redlamp_baseline_mlp.yaml`,
   - rename `configs/model/redlamp_cnn_baseline.yaml` to `configs/model/redlamp_baseline_cnn_simple.yaml`,
   - ensure both files resolve to `model_name: redlamp_baseline`.

6. In source files:
   - delete `src/models/redlamp_mlp_baseline.py` entirely if no active runtime path still imports it after test migration,
   - otherwise keep it only for a very short compatibility tail and delete it in the last cleanup subphase of this same implementation.

7. In active experiment YAML filenames:
   - rename all active baseline files that still contain `redlamp_mlp_baseline` or `redlamp_cnn_baseline` in the filename to `redlamp_baseline`,
   - keep encoder identity in the experiment slug only when it describes the encoder preset, for example `mlp` or `cnn-simple`,
   - update every path reference in tests and launcher scripts accordingly.

### Test-first edits

Write or update failing tests before changing code:

- `tests/test_redlamp_baseline_active_config_paths.py`
- `tests/test_config_loading.py`
- `tests/test_comparative_runner.py`
- `tests/test_comparative_preflight.py`

The new failing assertions should enforce:

- no active benchmark or comparative config resolves to `model_name: redlamp_mlp_baseline`,
- canonical baseline config paths use the renamed model preset files,
- comparative runner accepts only `redlamp_baseline` for baseline runs.

### Acceptance criteria

- `load_experiment_config(...)` fails on `model_name: redlamp_mlp_baseline`.
- Active runtime scripts register exactly one baseline model name: `redlamp_baseline`.
- All active benchmark, comparative, baseline, and smoke baseline configs load through renamed canonical paths.
- No active CLI help or launcher example teaches `redlamp_mlp_baseline` or `redlamp_cnn_baseline` as a public runtime identity.

## Phase 2 - Remove `val_realistic` and test-derived validation-prior semantics from the active runtime

### Phase summary

This phase removes the most misleading semantic layer in the current codebase. The thesis objective is to eliminate benchmark contamination risk and to ensure that validation behavior is easy to explain, reproduce, and defend.

### Files to modify

- `src/engine/trainer.py`
- `src/models/redlamp_baseline.py`
- `src/models/thesis_multitask.py`
- `src/core/config.py`
- `configs/task/*.yaml`
- tests currently asserting `val_realistic` behavior

### Planned edits

1. In `src/engine/trainer.py`:
   - delete `_resolve_realistic_validation_anomaly_rate(...)`,
   - remove the import `compute_smd_test_window_anomaly_rate`,
   - simplify `_resolve_checkpoint_threshold_metric_name(...)` so it recognizes:
     - `val_synth_*`
     - `val_*`
     and no longer recognizes `val_realistic_*`,
   - remove `val_realistic_*` entries from checkpoint-monitor mode resolution,
   - remove the branch that runs `realistic_validation_step`,
   - keep only the branch that runs `synthetic_validation_step` as the auxiliary validation path.

2. In `src/models/redlamp_baseline.py`:
   - delete `prepare_realistic_validation_epoch(...)`,
   - remove `val_realistic` from any stage-name sets such as `_prepare_batch(...)`,
   - delete `realistic_validation_step(...)`,
   - keep `synthetic_validation_step(...)` as the only auxiliary synthetic-validation method.

3. In `src/models/thesis_multitask.py`:
   - remove `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` from `SyntheticAnomalyConfig`,
   - remove their validation logic from `SyntheticAnomalyConfig.__post_init__`,
   - remove their names from `from_flat_kwargs(...)`-related field mappings,
   - remove `self.val_realistic`, `self.val_realistic_source`, and `self.val_anomaly_rate_override`,
   - delete `prepare_realistic_validation_epoch(...)`,
   - delete `realistic_validation_step(...)`,
   - remove `val_realistic` from stage-name branches, keeping only `val_synth`.

4. In `src/core/config.py`:
   - remove `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` from task boolean and validation logic,
   - remove accepted scheduler and checkpoint monitor names:
     - `val_realistic_loss`
     - `val_realistic_roc_auc`
     - `val_realistic_pr_auc`
     - `val_realistic_vus_pr`
   - keep or strengthen support for:
     - `val_synth_loss`
     - `val_synth_roc_auc`
     - `val_synth_pr_auc`
     - `val_synth_vus_pr`
     - and the current benchmark-safe `val_*` metrics where still needed.

5. In task YAML files such as:
   - `configs/task/multitask_tsad.yaml`
   - `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`
   - `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`
   - `configs/task/multitask_tsad_redlamp_multiclass_window20_comparative.yaml`
   - `configs/task/multitask_tsad_redlamp_multiclass_window20_redlamp_aligned.yaml`
   - `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`
   - `configs/task/multitask_tsad_window10_binary.yaml`
   remove:
   - `val_realistic`
   - `val_realistic_source`
   - `val_anomaly_rate_override`

### Test-first edits

Before implementation, rewrite failing tests so they assert the new contract:

- `tests/test_multitask_validation_alignment.py`
- `tests/test_learning_rate_scheduler.py`
- `tests/test_redlamp_baseline_config_surface.py`
- `tests/test_cnn_encoder_config_loading.py`

Concrete test shifts:

1. Replace assertions expecting `val_realistic_*` keys with `val_synth_*`.
2. Replace scheduler-monitor tests that target `val_realistic_*` with `val_synth_*`.
3. Delete or rewrite tests that exist only to validate test-derived anomaly-rate estimation.

Files likely to delete or radically rewrite:

- `tests/test_smd_realistic_rate.py`
- `tests/test_redlamp_realistic_validation_alignment.py`

### Risk mitigation

- **Data leakage prevention**: deleting test-derived anomaly-prior estimation removes the remaining path that could reuse test statistics during validation.
- **Metric inflation prevention**: checkpoint selection now depends only on synthetic validation behavior under fixed seeds, not on ratios inferred from the test split.
- **Benchmark comparability**: `val_synth` remains reproducible and explicitly synthetic, which is easier to explain than the current hybrid “realistic” naming.

### Acceptance criteria

- No active runtime file references `val_realistic`.
- No active task YAML contains `val_realistic`, `val_realistic_source`, or `val_anomaly_rate_override`.
- `Trainer.train(...)` can produce `val_synth_*` metrics but never `val_realistic_*`.
- The codebase contains no active runtime dependency on `compute_smd_test_window_anomaly_rate`.

## Phase 3 - Align the active config and benchmark surface to the new simpler contract

### Phase summary

This phase turns the simplified runtime into a readable benchmark surface. The thesis objective is that a reader can open the benchmark configs and understand the experiment protocol without tracing legacy branches.

### Files to modify

- `configs/experiment/benchmark/baseline/*.yaml`
- `configs/experiment/benchmark/thesis/*.yaml`
- `configs/experiment/comparative/baseline/*.yaml`
- `configs/experiment/comparative/thesis/*.yaml`
- active smoke configs used for pipeline verification
- `tests/test_config_loading.py`
- `tests/test_redlamp_baseline_active_benchmark_config.py`
- `tests/test_comparative_config_loading.py`

### Planned edits

1. For the six active baseline benchmark configs:
   - rename filenames from `smd__redlamp_mlp_baseline__benchmark-...` to `smd__redlamp_baseline__benchmark-...`,
   - keep `checkpoint_monitor_metric: val_synth_vus_pr`,
   - keep `epochs: 100`,
   - keep split-specific stride settings and fixed synthetic seeds.

2. For the six active thesis benchmark configs:
   - preserve the current 100-epoch three-stage schedule,
   - preserve `checkpoint_monitor_metric: val_synth_vus_pr`,
   - preserve the benchmark-safe data paths and task config path.

3. For comparative baseline configs and smoke configs:
   - rename filenames to canonical baseline naming,
   - update any old model config paths to the renamed canonical baseline preset files,
   - ensure `checkpoint_monitor_metric` and scheduler monitor, when present, target `val_synth_vus_pr`.

4. For legacy scale or smoke configs that are still used in tests:
   - either migrate them to canonical naming and `val_synth_*`,
   - or explicitly move them out of the active test surface if they are no longer part of the benchmark-critical path.

5. Optional but recommended simplification:
   - rename `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`
     to
     `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark.yaml`
   because “fixed synth” is no longer a special case; it is the benchmark-default contract.

### Test-first edits

Add or update config-load assertions to require:

- active benchmark configs use `checkpoint_monitor_metric: val_synth_vus_pr`,
- active benchmark task configs do not contain removed keys,
- renamed canonical file paths load and validate,
- comparative configs no longer point to old baseline model names.

### Acceptance criteria

- Every active benchmark config for baseline and thesis loads successfully.
- Every active benchmark config uses `val_synth_vus_pr`.
- No active benchmark config filename contains `redlamp_mlp_baseline`.
- No active benchmark task config still carries removed `val_realistic` keys.

## Phase 4 - Reduce test-suite mental-model duplication and delete no-longer-needed legacy tests

### Phase summary

This phase ensures the test suite now teaches the same mental model as the runtime. The thesis objective is long-term maintainability: future debugging should not require remembering which tests still describe an older protocol.

### Files to modify or rename

- `tests/test_redlamp_cnn_baseline_shapes.py`
- `tests/test_redlamp_cnn_rerun_configs.py`
- `tests/test_cnn_encoder_config_loading.py`
- `tests/test_one_redlamp_mlp_train_step.py`
- `tests/test_redlamp_baseline_with_gradient_profiling_step.py`
- `tests/test_anomaly_archive_dataset_loader.py`
- `tests/test_config_loading.py`
- `tests/test_config_stress_cases.py`
- `tests/test_comparative_runner.py`
- `tests/test_comparative_preflight.py`
- `tests/test_multitask_metrics_runtime.py`
- `tests/test_multitask_validation_alignment.py`
- `tests/test_learning_rate_scheduler.py`

### Planned edits

1. Replace imports from `src.models.redlamp_mlp_baseline` with imports from `src.models.redlamp_baseline`.

2. Replace all test stub configs that still use `model_name="redlamp_mlp_baseline"` with `model_name="redlamp_baseline"`.

3. Replace all assertions expecting:
   - `val_realistic_loss`
   - `val_realistic_pr_auc`
   - `val_realistic_vus_pr`
   - `val_realistic_threshold`
   with their `val_synth_*` equivalents where the test is still meaningful.

4. Delete tests whose only purpose is validating the removed semantics:
   - `tests/test_smd_realistic_rate.py`
   - any purely `realistic_validation`-specific tests after functionality is removed

5. Rename misleading test files if they remain active:
   - `tests/test_one_redlamp_mlp_train_step.py` to `tests/test_one_redlamp_baseline_train_step.py`
   - any other test filenames that preserve the old model identity in the public active surface

6. Keep only benchmark-relevant canonical behavior plus small, direct smoke-style coverage. Do not keep a special compatibility test layer for removed aliases unless a remaining external dependency forces it.

### Test strategy

The rewritten suite should emphasize:

- config-loading correctness,
- one forward and backward pass,
- checkpoint monitor selection,
- synthetic validation namespace correctness,
- benchmark config integrity,
- comparative launcher path integrity.

This is more useful than retaining many tests that preserve legacy words after the code has already moved on.

### Acceptance criteria

- Canonical baseline tests no longer import or assert the old baseline identity.
- No active test requires `val_realistic` behavior.
- The remaining test suite covers the cleaned benchmark path more directly than before.

## Phase 5 - Verification and smoke validation before benchmark launch

### Phase summary

This phase is the final operational gate. The thesis objective is to ensure that the cleaned runtime not only looks simpler but also actually runs under the exact benchmark assumptions now locked in.

### Verification commands

The exact command list may be adjusted to the renamed paths, but the verification intent should be:

1. Config-load validation for all active benchmark configs:
   - baseline benchmark configs,
   - thesis benchmark configs,
   - comparative configs,
   - smoke configs still retained for runtime checking.

2. Focused pytest bundles:
   - config-loading and benchmark-config tests,
   - baseline forward/backward tests,
   - multitask validation and scheduler tests rewritten to `val_synth`,
   - comparative runner and preflight tests.

3. Smoke training runs:
   - one canonical baseline smoke config,
   - one canonical thesis smoke config.

4. Optional smoke evaluation:
   - one evaluation command on a smoke-produced checkpoint, if time allows and if the checkpoint path is immediately available.

### Risk mitigation during verification

- **Prototype redundancy and fusion collapse**: this cleanup does not alter the accepted three-stage schedule or memory-freezing contract, so these risks are controlled by preserving existing benchmark-safe thesis configs.
- **Adaptation contamination**: online adaptation runtime should only receive canonical naming updates; no behavioral change should be introduced there in this cleanup.
- **Projector drift**: online adaptation evaluation is outside the current cleanup scope and should not be refactored now.
- **Evaluation metric inflation**: because test-derived validation prior logic is removed, benchmark checkpoint selection can no longer be influenced by test statistics.

### Acceptance criteria

- The renamed active benchmark configs load and validate.
- Targeted pytest bundles pass.
- At least one canonical baseline smoke run completes.
- At least one canonical thesis smoke run completes.
- No smoke or benchmark log output mentions `val_realistic` as an active stage.

## Recommended Implementation Order

The implementation should be executed in this exact order:

1. Rewrite tests so they fail on the old semantics.
2. Remove `val_realistic` semantics from trainer and model files.
3. Remove old baseline model names from config validation and script registration.
4. Rename model config files and experiment YAML filenames.
5. Update launcher scripts, help text, and remaining tests.
6. Run the full verification bundle.

This order is important because it keeps failures local and interpretable. If filenames are renamed before semantic cleanup is enforced by tests, the repository can become harder to debug rather than easier.

## Hard No List

The implementation should explicitly avoid the following:

- no new validation-stage abstraction framework,
- no dataset-plugin system,
- no broad refactor of loader architecture,
- no redesign of evaluator metric formulas,
- no onboarding of `SWaT`, `IOPS`, `NASA`, or `iccad` in this cleanup task,
- no preservation of removed legacy semantics merely for nostalgia or naming continuity.

## Final Success Condition

The cleanup is successful only if all of the following are true at the same time:

1. A new reader sees one baseline identity, not several.
2. The active benchmark path has no `val_realistic` semantics left.
3. Synthetic validation is the only auxiliary validation path in the active benchmark runtime.
4. Active benchmark configs and launcher scripts point only at canonical names and canonical paths.
5. The cleaned codebase still passes focused tests and smoke runs needed before large benchmark execution.
