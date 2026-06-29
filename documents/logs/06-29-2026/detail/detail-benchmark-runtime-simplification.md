---
date: 2026-06-29 23:50:22 +07 +0700
researcher: Codex
git_commit: ad75d65538ac169b6253b757bdeef7a80f3bdfeb
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for benchmark runtime simplification"
tags: [detail, time-series, anomaly-detection, benchmark, simplification]
status: draft
last_updated: 2026-06-29
last_updated_by: Codex
---

# Detail: Benchmark runtime simplification

**Date**: 2026-06-29 23:50:22 +07 +0700  
**Researcher**: Codex  
**Git Commit**: `ad75d65538ac169b6253b757bdeef7a80f3bdfeb`  
**Branch**: `dev`

## Objective

The implementation will simplify the active benchmark runtime so that the codebase has one clear offline evaluation path:

- `train`
- `val`
- `val_synth`
- `test`

The implementation will remove the obsolete `val_realistic` runtime branch, remove the obsolete balancing aliases, rename active orchestration from `comparative` to `benchmark`, move the baseline naming surface toward an encoder-agnostic form, and keep the repository operational through targeted tests and smoke verification.

The implementation will preserve:

- the current batch contract
- the current encoder hidden-state contract
- the current model output contract
- the current registry-based dataset and model construction approach
- the current dataset builder architecture
- the current benchmark stride and epoch-budget decisions

The implementation will not introduce a new framework layer. It will simplify by deletion, migration, and validation.

## Fixed Design Decisions

The following implementation decisions are now fixed.

1. `val_realistic` will be removed from the active runtime, not merely deprecated.
2. The active orchestration surface will be renamed from `comparative` to `benchmark`.
3. `balance_binary_classes_within_batch` and `balance_classes_within_batch` will be removed immediately because no active benchmark configs depend on them.
4. The canonical balancing field will remain `train_balance_classes` for now.
5. Historical logs under `documents/logs/` will be preserved. They are documentation artifacts, not active runtime surfaces.
6. Active code, active tests, and active experiment configs must be made consistent with the simplified runtime. They must not remain in a half-migrated state.
7. The active RedLamp baseline should no longer expose `MLP` in its canonical public name because the implementation already supports multiple encoder families through `encoder_family`.
8. The simplest recommended canonical rename is:
   - class name: `RedLampBaseline`
   - registry name: `redlamp_baseline`
   - model module path: `src/models/redlamp_baseline.py`

## Contract Preservation

### Batch Contract

The active batch contract remains:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict[str, object]],
}
```

Files that enforce or consume this contract:

- `src/core/contracts.py`
- `src/data/collate.py`
- `src/data/loaders.py`
- `src/engine/trainer.py`
- `src/engine/evaluator.py`

No phase in this plan changes this contract.

### Encoder Contract

The thesis-facing encoder contract remains:

```python
hidden: Tensor[B, L, H]
```

Files that rely on this contract:

- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`
- tests that validate forward shapes

No phase in this plan changes hidden-state shape or public semantics.

### Model Output Contract

The active runtime will continue to rely on:

- `recon`
- `logits`
- `point_scores`
- `window_scores`
- `aux`

This preserves composition between model files and the generic trainer and evaluator.

### Design Pattern Preservation

This work preserves the existing repository design principles instead of replacing them.

- Composition over inheritance remains unchanged. The trainer, evaluator, logger, checkpoint manager, dataset builders, and model files stay separate.
- The registry or factory role remains unchanged in `src/core/registry.py`.
- The dataset-specific parser pattern remains unchanged under `src/data/datasets/`.
- The self-contained model-file rule remains unchanged. All stage logic for each model stays in its model file.
- No additional adapter or strategy layer will be introduced for this simplification task.

## High-Risk Simplification Points

The implementation must be suspicious about the following points.

1. Removing `val_realistic` globally will break many tests and configs unless they are migrated or deleted in the same pass.
2. Removing balancing aliases without also tightening constructor behavior can leave a false sense of cleanup, especially in `RedLampMLPBaseline`, which currently accepts `**unused_kwargs`.
3. Renaming orchestration files without updating test coverage and shell references atomically can break the server-launch path.
4. A broad config migration from `val_realistic_*` to `val_synth_*` must remain mechanical and auditable. It must not silently change tensor contracts or model objectives.
5. Renaming `RedLampMLPBaseline` touches imports, registry keys, model config names, experiment YAMLs, test names, and output naming. If this rename is done loosely, the codebase can look cleaner while still carrying stale active references.

## Implementation Order

The implementation order is mandatory.

1. Write or update the protection tests first.
2. Remove `val_realistic` from config validation and trainer logic.
3. Remove `val_realistic` from the two offline models.
4. Remove balancing aliases and tighten baseline constructor behavior.
5. Migrate surviving config families from `val_realistic_*` to `val_synth_*`.
6. Rename the active baseline naming surface to an encoder-agnostic canonical form.
7. Delete or rename the `comparative` orchestration and test surface.
8. Reduce logging noise and refresh help text.
9. Run focused tests.
10. Run smoke validation for the active benchmark pipeline.

The implementation must not start with launcher renaming or model renaming. It must first stabilize the active validation namespace.

## Phase 1: Lock the Active Contract in Tests

### Phase Summary

This phase protects the simplified runtime before destructive edits begin. The objective is to make the benchmark path explicit in tests so that later deletions become safe and measurable.

### File-Level Edits

Modify:

- `tests/test_config_loading.py`
- `tests/test_multitask_validation_alignment.py`
- `tests/test_learning_rate_scheduler.py`
- `tests/test_redlamp_mlp_baseline.py`
- `tests/test_thesis_multitask_config_refactor.py`
- `tests/test_multitask_metrics_runtime.py`
- `tests/test_config_stress_cases.py`
- `tests/test_cnn_encoder_config_loading.py`
- `tests/test_redlamp_realistic_validation_alignment.py`

### Explicit Edit Content

In `tests/test_config_loading.py`:

- Replace acceptance tests for `val_realistic_vus_pr`, `val_realistic_pr_auc`, and `val_realistic_loss` with rejection tests.
- Keep and strengthen benchmark-path assertions for:
  - `checkpoint_monitor_metric = val_synth_vus_pr`
  - `task.val_realistic` no longer existing after migration
  - `train_stride = 10`, `val_stride = 1`, `test_stride = 1`
- Add explicit rejection tests for:
  - `task.val_realistic`
  - `task.val_realistic_source`
  - `task.val_anomaly_rate_override`
  - `checkpoint_monitor_metric = val_realistic_*`
  - `optimizer.scheduler.monitor_metric = val_realistic_*`

In `tests/test_multitask_validation_alignment.py`:

- Remove the branch that treats `val_realistic` as the auxiliary validation namespace.
- Keep one clean validation test for `val`.
- Keep one auxiliary synthetic validation test for `val_synth`.
- Ensure the epoch metrics contain `val_synth_*` and do not contain `val_realistic_*`.

In `tests/test_learning_rate_scheduler.py`:

- Replace dummy metrics and logs from `val_realistic_*` to `val_synth_*`.

In `tests/test_redlamp_mlp_baseline.py`:

- Delete tests that verify compatibility of `balance_classes_within_batch` and `balance_binary_classes_within_batch`.
- Add tests that verify:
  - the canonical field is `train_balance_classes`
  - passing removed alias names fails loudly once constructor tightening is complete

In `tests/test_thesis_multitask_config_refactor.py`:

- Remove expectations that `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` survive in the synthetic config structure.
- Replace them with expectations tied only to `use_synthetic_validation`, seeds, anomaly fractions, balancing, and classification label mode.

In `tests/test_multitask_metrics_runtime.py`:

- Replace `stage_name="val_realistic"` with `stage_name="val_synth"` in metric aggregation tests.

In `tests/test_config_stress_cases.py`:

- Replace any invalid-metric sentinel based on `val_realistic_precision` with a `val_synth_*` invalid-metric sentinel.

In `tests/test_cnn_encoder_config_loading.py`:

- Replace `val_realistic_vus_pr` scheduler and checkpoint expectations with `val_synth_vus_pr`.

In `tests/test_redlamp_realistic_validation_alignment.py`:

- Either rename the file to `tests/test_redlamp_synthetic_validation_alignment.py` and rewrite it to the `val_synth` path, or delete it and merge essential coverage into `tests/test_multitask_validation_alignment.py`.

Recommended choice:

- Delete the file and merge the minimum useful coverage into the active alignment tests. This reduces duplicate test surfaces.

### Risk Mitigation

- This phase does not change model internals yet, so failures will localize to expectations rather than implementation.
- Prototype redundancy, fusion collapse, adaptation contamination, and projector drift are not modified here. Their risk is controlled by keeping forward contracts untouched.

### Acceptance Criteria

- The test suite clearly encodes one active auxiliary validation namespace: `val_synth`.
- No active test expects `val_realistic_*`.
- No active test expects balancing aliases.

## Phase 2: Remove `val_realistic` from the Active Runtime

### Phase Summary

This phase deletes the obsolete runtime branch from config validation, trainer flow, and offline model stage APIs. This is the core behavior change of the simplification task.

### File-Level Edits

Modify:

- `src/core/config.py`
- `src/core/config_help.py`
- `src/engine/trainer.py`
- `scripts/train.py`
- `src/models/redlamp_mlp_baseline.py`
- `src/models/thesis_multitask.py`

### Explicit Edit Content

In `src/core/config.py`:

- Remove these task keys from `allowed_task_keys_by_task_name["multitask_tsad"]`:
  - `val_realistic`
  - `val_realistic_source`
  - `val_anomaly_rate_override`
- Remove boolean validation for `val_realistic`.
- Remove validation logic for:
  - `val_realistic_source`
  - `val_anomaly_rate_override`
- Update the allowed checkpoint monitor metrics to:
  - `val_loss`
  - `val_synth_loss`
  - `val_synth_roc_auc`
  - `val_synth_pr_auc`
  - `val_synth_vus_pr`
  - `val_vus_pr`
- Update allowed scheduler monitor metrics to the same reduced set, minus `val_vus_pr` if the runtime still does not support it there.
- Update `logging.diagnostics_stages_for_classification` so supported stages become:
  - `train`
  - `val`
  - `val_synth`
  - `test`

In `src/core/config_help.py`:

- Add a short explicit hint that the active synthetic multitask benchmark uses `val_synth`.
- Add a short explicit hint that `train_balance_classes=true` means balanced synthetic classes and that `anomaly_probability` only governs the unbalanced branch.

In `src/engine/trainer.py`:

- Remove `_resolve_realistic_validation_anomaly_rate`.
- Remove any read of:
  - `task.val_realistic`
  - `task.val_realistic_source`
  - `task.val_anomaly_rate_override`
- In `_resolve_checkpoint_threshold_metric_name`, remove the `val_realistic_` branch.
- In `_aggregate_reconstruction_diagnostics`, update the stage loop from `("train", "val", "val_realistic", "test")` to `("train", "val", "val_synth", "test")`.
- In `_resolve_best_checkpoint_monitor`, remove `val_realistic_*`.
- In the epoch loop, collapse the auxiliary validation logic so that it always uses:
  - `prepare_synthetic_validation_epoch`
  - `synthetic_validation_step`
  - `stage_name="val_synth"`

In `scripts/train.py`:

- Remove scheduler mode entries for `val_realistic_*`.
- Keep only `val_synth_*` and `val_loss` as the active auxiliary monitor surfaces.

In `src/models/redlamp_mlp_baseline.py`:

- Delete `prepare_realistic_validation_epoch`.
- Delete `realistic_validation_step`.
- Restrict `_prepare_batch` synthetic-validation stage matching to `val_synth`.
- Ensure only `validation_step`, `synthetic_validation_step`, and `test_step` remain as active public stage methods.

In `src/models/thesis_multitask.py`:

- Remove `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override` from `SyntheticAnomalyConfig`.
- Remove the same fields from `from_flat_kwargs`.
- Remove the corresponding instance attributes.
- Delete `prepare_realistic_validation_epoch`.
- Replace the current wrapper relationship so that `prepare_synthetic_validation_epoch` directly resets the synthetic validation injector.
- Restrict stage handling and contrastive pair staging to the active set that includes `val_synth` but not `val_realistic`.
- Delete `realistic_validation_step`.

### Risk Mitigation

- Keep all forward tensor logic unchanged.
- Keep all prototype, memory, and contrastive internals unchanged except for stage-name routing.
- Add a narrow regression check that `val_synth` still emits the same metric families and pointwise payload structure previously emitted by synthetic or realistic validation.
- This phase does not alter prototype redundancy mitigation, fusion collapse controls, or projector logic. It only simplifies stage selection.

### Acceptance Criteria

- No active runtime code path mentions `val_realistic`.
- Both offline models expose `val_synth` as the only auxiliary synthetic validation stage.
- Benchmark configs and generic multitask task configs validate without any `val_realistic` keys.

## Phase 3: Remove Legacy Balancing Aliases and Tighten Constructor Strictness

### Phase Summary

This phase removes obsolete public API surface around synthetic class balancing and ensures that removed names fail loudly instead of being silently ignored.

### File-Level Edits

Modify:

- `src/data/augment.py`
- `src/models/redlamp_mlp_baseline.py`
- `tests/test_redlamp_mlp_baseline.py`
- `tests/test_synthetic_anomaly_injection.py`
- any remaining active direct-constructor tests for the baseline

### Explicit Edit Content

In `src/data/augment.py`:

- Remove `balance_binary_classes_within_batch` from `SyntheticAnomalyInjector.__init__`.
- Remove the fallback branch that maps `balance_binary_classes_within_batch` into `train_balance_classes`.
- Keep only `train_balance_classes`.
- Update comments so they say explicitly that the field controls balanced synthetic class quotas for the active taxonomy and is reused by both train and synthetic-validation injectors.

In `src/models/redlamp_mlp_baseline.py`:

- Remove these constructor parameters:
  - `balance_classes_within_batch`
  - `balance_binary_classes_within_batch`
- Remove the compatibility resolution block that computes `effective_balance_classes_within_batch`.
- Pass `train_balance_classes` directly into both injectors.
- Remove `**unused_kwargs`.

The removal of `**unused_kwargs` is required. Without this change, deleted alias names can still be passed into the constructor silently, which defeats the purpose of API simplification.

In tests:

- Replace alias-compatibility tests with strictness tests.
- Add a test that direct baseline construction with an unknown keyword now fails.

### Risk Mitigation

- The thesis model already rejects unknown flat kwargs. This phase aligns the baseline with that stricter behavior.
- This phase does not alter how balanced quotas are computed. It only removes obsolete naming branches.
- Evaluation metric inflation risk remains unaffected because no scoring logic changes here.

### Acceptance Criteria

- No active code surface mentions `balance_binary_classes_within_batch` or `balance_classes_within_batch`.
- `RedLampMLPBaseline` no longer swallows unknown constructor kwargs.
- All active constructor tests pass using only canonical argument names.

## Phase 4: Migrate Surviving Config Families to the Simplified Runtime

### Phase Summary

This phase makes the remaining config tree consistent with the new runtime. The goal is to avoid a half-clean codebase in which the runtime rejects most experiment YAMLs.

### File-Level Edits

Modify:

- `configs/task/multitask_tsad.yaml`
- `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`
- `configs/task/multitask_tsad_redlamp_multiclass_window20_balanced.yaml`
- `configs/task/multitask_tsad_redlamp_multiclass_window20_redlamp_aligned.yaml`
- `configs/task/multitask_tsad_window10_binary.yaml`
- `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`
- surviving experiment configs under:
  - `configs/experiment/baseline/`
  - `configs/experiment/scale/`
  - `configs/experiment/thesis/`
  - `configs/experiment/ablation/`
  - `configs/experiment/smoke/`
  - `configs/experiment/benchmark/`

Delete:

- `configs/task/multitask_tsad_redlamp_multiclass_window20_comparative.yaml`

Recommended delete or migrate:

- `configs/experiment/comparative/`
- `configs/experiment/comparative_stress_smoke/`

### Explicit Edit Content

In all surviving task YAMLs:

- Remove:
  - `val_realistic`
  - `val_realistic_source`
  - `val_anomaly_rate_override`

In `configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml`:

- Remove dormant `val_realistic_source` and `val_anomaly_rate_override`.
- Keep:
  - `synthetic_train_seed`
  - `synthetic_validation_seed`
  - `train_balance_classes`
  - `anomaly_probability`
  - synthetic anomaly family list

In all surviving experiment YAMLs:

- Replace:
  - `checkpoint_monitor_metric: val_realistic_*`
  with
  - `checkpoint_monitor_metric: val_synth_*`
- Replace scheduler `monitor_metric: val_realistic_*` with `val_synth_*`.
- Replace any diagnostics stage list containing `val_realistic` with `val_synth`.
- Replace any focused metrics or logging lists containing `val_realistic_*` with `val_synth_*`.
- Update comment tags that mark runs as `val_realistic` if those tags are meant to describe the active runtime.

Recommended treatment for `configs/experiment/comparative/` and `configs/experiment/comparative_stress_smoke/`:

- Delete them from the active working tree.
- Do not migrate them, because the active benchmark family already supersedes them and the historical record is preserved on GitHub.

### Risk Mitigation

- This migration is mechanical and must be reviewed with a repository search after edits.
- Use grep-based validation to ensure no active config still mentions `val_realistic` except historical notes under `documents/logs/`.
- This phase does not alter model architecture, prototype memory logic, fusion, or adaptation behavior.

### Acceptance Criteria

- Active task YAMLs contain no `val_realistic*` keys.
- Active experiment configs contain no `val_realistic_*` monitor metrics.
- Active config-loading tests pass after the migration.

## Phase 5: Rename the Active Baseline to an Encoder-Agnostic Public Name

### Phase Summary

This phase removes the misleading `MLP` qualifier from the active baseline public surface.

The current implementation already supports multiple encoder families through `encoder_family`, so the name `RedLampMLPBaseline` is no longer truthful enough for the active codebase direction. The public surface should describe one baseline family whose encoder is chosen by configuration.

### File-Level Edits

Rename and modify:

- `src/models/redlamp_mlp_baseline.py`
  to
  `src/models/redlamp_baseline.py`
- `tests/test_redlamp_mlp_baseline.py`
  to
  `tests/test_redlamp_baseline.py`
- `tests/test_one_redlamp_mlp_train_step.py`
  to
  `tests/test_one_redlamp_baseline_train_step.py`

Modify:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`
- `src/core/config.py`
- active experiment YAMLs under `configs/experiment/benchmark/`
- any remaining active experiment YAMLs that still use `redlamp_mlp_baseline`

Optional rename for clarity:

- `configs/model/redlamp_mlp_baseline.yaml`
  to
  `configs/model/redlamp_baseline.yaml`
- `configs/model/redlamp_mlp_baseline_comparative_smd.yaml`
  to
  `configs/model/redlamp_baseline_benchmark_smd.yaml`
- `configs/model/redlamp_mlp_baseline_redlamp_aligned.yaml`
  to
  `configs/model/redlamp_baseline_redlamp_aligned.yaml`
- `configs/model/redlamp_cnn_baseline.yaml`
  should be deleted or merged into the same canonical baseline surface

Recommended choice:

- Use `RedLampBaseline` as the only canonical class name for the active baseline surface.
- Use `redlamp_baseline` as the only canonical registry name for the active baseline surface.
- Remove `configs/model/redlamp_cnn_baseline.yaml` if it is only a second public alias for the same baseline family. One baseline family should be exposed through one canonical model name plus `encoder_family`, not through separate misleading public names.

### Explicit Edit Content

In `src/models/redlamp_baseline.py` after the rename:

- rename `class RedLampMLPBaseline` to `class RedLampBaseline`
- keep all tensor contracts unchanged
- keep `encoder_family` as the switch for `mlp`, `cnn_simple`, and future encoder variants
- update docstrings and comments so they describe a general RedLamp-style baseline, not an MLP-only baseline

In `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py`:

- update imports to `from src.models.redlamp_baseline import RedLampBaseline`
- update registry registration from:
  - `register_model("redlamp_mlp_baseline", RedLampMLPBaseline)`
  to
  - `register_model("redlamp_baseline", RedLampBaseline)`
- remove the old registry name from the active benchmark path once all active configs are migrated

In active model YAMLs:

- replace `model_name: redlamp_mlp_baseline` with `model_name: redlamp_baseline`
- keep `encoder_family` explicit in the config
- rename the canonical model-config file names so they no longer imply MLP-only behavior

In active experiment YAMLs:

- update experiment names, output directories, checkpoint directories, and `wandb_run_name` strings if they are part of the active benchmark surface and still hard-code `redlamp_mlp_baseline`
- keep the edits mechanical and grep-verifiable

In active baseline tests:

- update imports to the renamed module and class
- update test function names so they no longer say `mlp_baseline` when the test is about the encoder-agnostic baseline surface
- preserve explicit coverage for `encoder_family="mlp"` and `encoder_family="cnn_simple"`

### Risk Mitigation

- Keep this rename mechanical. Do not combine it with loss-function changes, tensor-shape changes, or validation-semantics changes.
- Run a repository-wide grep after the rename to confirm the active runtime no longer depends on `RedLampMLPBaseline` or `redlamp_mlp_baseline`.
- Keep historical `documents/logs/` artifacts untouched.

### Acceptance Criteria

- The active baseline public surface is encoder-agnostic:
  - class name `RedLampBaseline`
  - registry name `redlamp_baseline`
  - canonical config names without the misleading `mlp` qualifier
- Active benchmark configs choose the encoder through `encoder_family`, not through misleading baseline-family names.
- Active baseline tests pass after the rename.

## Phase 6: Rename Active Orchestration from `comparative` to `benchmark`

### Phase Summary

This phase makes the public benchmark launch path match the semantics it actually executes. The active run family is benchmark-oriented, not comparative-oriented.

### File-Level Edits

Rename and modify:

- `scripts/run_comparative_smd_experiments.py`
  to
  `scripts/run_benchmark_smd_experiments.py`
- `scripts/preflight_comparative_smd_server.py`
  to
  `scripts/preflight_benchmark_smd_server.py`
- `scripts/launch_tmux_comparative_smd_experiment.sh`
  to
  `scripts/launch_tmux_benchmark_smd_experiment.sh`

Rename and modify tests:

- `tests/test_comparative_runner.py`
  to
  `tests/test_benchmark_runner.py`
- `tests/test_comparative_preflight.py`
  to
  `tests/test_benchmark_preflight.py`
- `tests/test_comparative_tmux_launcher.py`
  to
  `tests/test_benchmark_tmux_launcher.py`

Delete and absorb:

- `tests/test_comparative_config_loading.py`

Its benchmark-relevant coverage should be absorbed into:

- `tests/test_config_loading.py`
- `tests/test_benchmark_runner.py`

Optional rename for clarity:

- `configs/model/redlamp_baseline_benchmark_smd.yaml`
  to
  keep this canonical benchmark-facing name after the orchestration rename
- `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`
  to
  `configs/model/thesis_multitask_three_stage_benchmark_smd.yaml`

Recommended choice:

- Keep the already-renamed baseline config files from Phase 5 and make the orchestration layer reference those canonical names directly.
- The files used by active benchmark configs should match their actual benchmark role and should not reintroduce the old `mlp`-specific public naming.

### Explicit Edit Content

In the renamed runner and preflight scripts:

- Replace runtime-facing text from `comparative` to `benchmark`.
- Rename report artifacts to:
  - `benchmark_manifest.json`
  - `benchmark_execution_report.json`
  - `benchmark_server_preflight_summary.json`
- Rename readiness strings from comparative-specific wording to benchmark wording.
- Keep the same underlying run-planning logic unless simplification by deletion is clearly possible.

In the renamed launcher shell script:

- Update script help text and session naming to benchmark vocabulary.
- Replace smoke config arrays so they point only to benchmark-family smoke configs.
- Remove any dependency on deleted comparative config paths.

If benchmark-family smoke configs do not yet exist:

- Rename the current comparative smoke configs into benchmark smoke configs before updating the launcher.

### Risk Mitigation

- File renames must be atomic with test updates.
- The launcher dry-run test must be updated before claiming the rename is complete.
- End-to-end launch semantics remain the same, only names and config families change.

### Acceptance Criteria

- No active script path or test file in `scripts/` or `tests/` uses `comparative` in its active benchmark role.
- The launcher dry-run test passes using the renamed benchmark scripts.
- The preflight summary test passes using benchmark artifact names.

## Phase 7: Remove Residual Logging Noise and Refresh Active Help Text

### Phase Summary

This phase improves readability and reduces runtime noise without changing algorithmic behavior.

### File-Level Edits

Modify:

- `src/data/collate.py`
- `src/core/config_help.py`
- `documents/design/experiment_config_organization_guideline.md`

### Explicit Edit Content

In `src/data/collate.py`:

- Remove unconditional per-batch `console_print` calls from `collate_windows`.
- Keep validation through `validate_window` and `validate_batch`.
- If needed, replace unconditional logging with a future debug-gated hook, but do not introduce a new configuration system in this phase.

In `src/core/config_help.py`:

- Update examples so active monitor metrics use `val_synth`.
- Add a compact explanation that balanced synthetic validation is the active auxiliary validation regime.

In `documents/design/experiment_config_organization_guideline.md`:

- Replace active examples that still present `val_realistic_*` as the standard namespace.

### Risk Mitigation

- This phase must not change training outputs or model behavior.
- Verification should focus on logs and config-help rendering, not numerical metrics.

### Acceptance Criteria

- Batch collation no longer floods logs in normal benchmark execution.
- Active help text and active design guidance refer to `val_synth`, not `val_realistic`.

## Phase 8: Verification Before Completion

### Phase Summary

This phase proves that the simplified codebase still works. It is mandatory and must be completed before any server launch command is prepared.

### Focused Test Passes

Run at least these focused suites first:

```bash
.venv/bin/python -m pytest -q \
  tests/test_config_loading.py \
  tests/test_multitask_validation_alignment.py \
  tests/test_learning_rate_scheduler.py \
  tests/test_redlamp_baseline.py \
  tests/test_thesis_multitask_config_refactor.py \
  tests/test_multitask_metrics_runtime.py \
  tests/test_trainer_checkpoint_fallback.py
```

Then run the renamed orchestration suites:

```bash
.venv/bin/python -m pytest -q \
  tests/test_benchmark_runner.py \
  tests/test_benchmark_preflight.py \
  tests/test_benchmark_tmux_launcher.py
```

Then run the one-step model smoke suites:

```bash
.venv/bin/python -m pytest -q \
  tests/test_one_redlamp_baseline_train_step.py \
  tests/test_one_multitask_train_step.py \
  tests/test_model_shapes.py \
  tests/test_multitask_shapes.py
```

Finally, run a broader regression slice if time permits:

```bash
.venv/bin/python -m pytest -q
```

### Launcher and Pipeline Smoke Validation

Run the active benchmark launcher in dry-run mode after renaming:

```bash
bash scripts/launch_tmux_benchmark_smd_experiment.sh --dry-run
```

Run the benchmark preflight script in its non-launch validation mode.

Run the benchmark runner in preflight-only or dry-run mode using the active benchmark config list.

### Manual Search-Based Sanity Checks

Run repository searches after edits:

```bash
rg -n "val_realistic" src scripts tests configs
rg -n "balance_binary_classes_within_batch|balance_classes_within_batch" src scripts tests configs
rg -n "RedLampMLPBaseline|redlamp_mlp_baseline" src scripts tests configs
rg -n "comparative" scripts tests configs/experiment configs/model configs/task
```

Expected result:

- any remaining hits should be only in historical `documents/logs/` notes, not in active runtime files

### Risk Mitigation

- Evaluation metric inflation risk is controlled by ensuring the active benchmark path still uses the same evaluator and thresholding logic, only with a cleaned namespace.
- Prototype redundancy and fusion collapse risks are controlled by not touching prototype or fusion calculations.
- Adaptation contamination and projector drift risks are controlled by not touching online adaptation logic in this simplification task.

### Acceptance Criteria

- Focused tests pass.
- Active benchmark launcher dry-run passes.
- Active benchmark preflight passes.
- No active runtime file still depends on `val_realistic`.
- No active runtime file still depends on balancing aliases.
- No active runtime file still depends on `RedLampMLPBaseline` or `redlamp_mlp_baseline` in the benchmark path.
- No active active-orchestration file still uses `comparative` vocabulary.

## Final Acceptance Criteria

The implementation is complete only if all conditions below hold.

1. The active offline runtime has exactly one auxiliary synthetic validation namespace: `val_synth`.
2. The active config schema rejects `val_realistic`, `val_realistic_source`, and `val_anomaly_rate_override`.
3. `RedLampBaseline` and `ThesisMultitaskModel` no longer expose `realistic_validation_step`.
4. `RedLampBaseline` no longer swallows unknown keyword arguments through `**unused_kwargs`.
5. The active baseline public surface is encoder-agnostic and uses the canonical naming:
   - `RedLampBaseline`
   - `redlamp_baseline`
6. The active benchmark configs, launcher, runner, and preflight scripts use `benchmark` vocabulary.
7. Active runtime files no longer mention `balance_binary_classes_within_batch` or `balance_classes_within_batch`.
8. Active tests pass and active smoke validation succeeds.

## Implementation Notes for the Coding Pass

The coding pass should be performed with a suspicious mindset.

- Delete obsolete branches instead of leaving compatibility stubs.
- Prefer moving existing benchmark tests over adding many new parallel files.
- Keep the dataset registry and builder contracts unchanged.
- Keep each model self-contained in its own file.
- Use apply-patch-driven manual edits.
- After each phase, stop and run the focused verification subset for that phase before proceeding.
