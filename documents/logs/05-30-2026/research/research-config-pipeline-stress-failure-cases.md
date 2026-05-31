# Research: Config Pipeline Stress Failure Cases

Date: 2026-05-30
Scope: `configs/**`, `src/core/config.py`, CLI entrypoints `scripts/train.py` and `scripts/evaluate.py`.

## Objective
Build adversarial test cases that intentionally break config loading/validation, then extract concrete operating lessons for train/validation and test CLI runs.

## Failure Case Matrix

### Case 1: Duplicate YAML key silently overrides previous block
- Surface: YAML parser stage (`load_yaml_config`).
- Example pattern:
  - `model_overrides:` appears twice in one file.
- Risk:
  - Earlier override block can be dropped silently.
  - Experiment semantics drift from filename/intent without obvious error.
- Action:
  - Enforce unique mapping keys at YAML load time.
  - Test: reject duplicate root key with explicit `ValueError`.

### Case 2: Scheduler field family mismatch
- Surface: config validation.
- Example pattern:
  - `scheduler_name: cosine` but includes plateau-only key like `factor`.
- Risk:
  - Hidden semantic contradiction, hard to debug LR behavior.
- Action:
  - Keep strict incompatibility checks.
  - Test: expect validation failure with precise error message.

### Case 3: Multiclass label mode contradicts class count
- Surface: task/model semantic coupling.
- Example pattern:
  - `classification_label_mode: redlamp_multiclass` + `num_classes != 12`.
- Risk:
  - Classification head/targets mismatch, invalid training objective.
- Action:
  - Keep hard validation failure.
  - Test: inject contradiction and assert fail.

### Case 4: Invalid worker spec (`data.num_workers`)
- Surface: data runtime config.
- Example pattern:
  - non-supported string (`"many"`) instead of integer or `"auto"`.
- Risk:
  - Runtime crashes late or inconsistent dataloader behavior.
- Action:
  - Keep strict type/value check.
  - Test: mutate resolved config and assert fail.

### Case 5: Out-of-range synthetic anomaly probability
- Surface: synthetic generation semantics.
- Example pattern:
  - `anomaly_probability > 1`.
- Risk:
  - Invalid sampling interpretation.
- Action:
  - Keep bounds check `[0,1]`.
  - Test: mutate and assert fail.

## Repo-Wide Guardrails Added
1. YAML duplicate-key rejection in `src/core/config.py` loader path.
2. Repo scan test to ensure every committed config under `configs/` has no duplicate root keys.
3. Stress validation tests for semantic contradictions above.
4. Hard-fail policy for wandb contradictions:
   - `logging.use_wandb: false` requires `logging.wandb_mode: disabled`.
   - `logging.use_wandb: true` forbids `logging.wandb_mode: disabled`.
5. Hard-fail policy for monitor mismatch under `reduce_on_plateau`:
   - `optimizer.scheduler.monitor_metric` must equal `checkpoint_monitor_metric`.
6. Strict unknown-key rejection for top-level, data/model/task sections.
7. Strict unknown-key rejection for optimizer/logging sections and scheduler sub-sections.

## Practical Lessons for CLI Reliability
1. Treat config loading as compile-time, not runtime convenience.
2. Fail fast before model/data objects are built.
3. Keep one semantic owner per concern:
   - scheduler behavior keys must match `scheduler_name` family,
   - class mode must match `num_classes`,
   - synthetic controls must remain physically valid.
4. Preserve resolved-config artifact in outputs and compare when debugging surprising run behavior.
5. Unknown keys are not harmless in research configs; they usually indicate typo, stale knobs, or dead ablation codepaths.
6. Prefer explicit fail messages that include direct English fix instructions to reduce user friction.

## Suggested Next Stress Extensions
1. Add resolved-config diff regression tests for key experiment families to catch accidental drift.
2. Add dedicated strict-schema tests for `optimizer` and `logging` unknown keys.
