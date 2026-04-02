---
date: 2026-04-01 13:21:39 +0700
planner: Artificial Intelligence Agent
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Pre-Phase-4 ablation-readiness checklist for the offline prototype-fusion codebase"
tags: [detail, ablation, pre-phase-4, fusion, multitask]
status: complete
last_updated: 2026-04-01
last_updated_by: Artificial Intelligence Agent
source_research: documents/logs/04-01-2026/research/research-pre-phase-4-ablation-readiness-and-branch-collapse-controls.md
---

# Detail: Pre-Phase-4 ablation-readiness checklist for the offline prototype-fusion codebase

## Objective

Before Phase 4 online adaptation begins, the offline codebase should support extensive ablations without introducing new model splits or hidden training paths. The current repository already contains the key model-level ingredients for such ablations. This checklist defined the remaining engineering work needed to make those ingredients easy to sweep, compare, and reproduce. Repository update on 2026-04-02: the listed checklist items are now implemented through the multitask experiment family, trainer-driven schedule controls, the compact ablation runner, richer JSONL metrics, and the canonical offline-to-online checkpoint handoff.

## Current baseline that should remain fixed

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: `src/models/thesis_multitask.py` now uses gate-entropy regularization directly while retaining the legacy margin field only for backward checkpoint compatibility.

- Keep one model per file.
- Keep all active multitask logic inside `src/models/thesis_multitask.py`.
- Keep the active data path registry-driven in `scripts/train.py` and `scripts/evaluate.py`.
- Keep synthetic anomaly visualization as a maintained script-level inspection surface.
- Keep all real task supervision on `H_rec` and `H_cls`, not on branch-local prediction heads.

## Checklist

### 1. Expose exact ablation modes at the configuration level

- Add a dedicated experiment configuration family for:
  - continuous-only
  - discrete-only
  - fused-learned
  - no-diversity-loss
  - no-variance-loss
  - no-covariance-loss
  - no-usage-loss
  - no-gate-entropy-regularization
  - no-augmentation
- Keep these as explicit YAML files rather than implicit command-line overrides.
- Make each ablation file differ only in the smallest necessary set of fields.

### 2. Add fusion-control scheduling without introducing a second training path

- Add configuration fields for:
  - `freeze_fusion_for_epochs`
  - `warmup_alpha_value`
  - `warmup_beta_value`
  - `temperature_start`
  - `temperature_end`
  - `temperature_anneal_fraction`
- Keep the active training path inside `ThesisMultitaskModel.training_step`.
- Implement scheduler state as explicit trainer-driven parameter updates, not as a second model class.
- Ensure the warm-up stage can hold `alpha = beta = 0.5` and `lambda_gate = 0` before later enabling gate entropy regularization if needed.

### 3. Make the central branch ablations exact limiting cases of one model

- Ensure `alpha = beta = 0` yields a continuous-only fused path.
- Ensure `alpha = beta = 1` yields a discrete-only fused path.
- Keep fused training as the same model with learnable `alpha` and `beta`.
- Avoid separate branch-only model files.

### 4. Promote loss terms into an explicit ablation surface

- Keep `lambda_cls`, `lambda_div`, `lambda_var`, `lambda_cov`, `lambda_use`, and `lambda_gate` as the only active objective switches.
- Ensure each loss helper returns zero cleanly when disabled by config.
- Log each loss term every epoch for train and validation.
- Keep naming stable between code, configs, and documents.

### 5. Add observability for branch-collapse monitoring

- Persist epoch-level logs for:
  - `alpha`
  - `beta`
  - continuous-branch norm
  - discrete-branch norm
  - discrete code usage summary
  - diversity loss
  - variance loss
  - covariance loss
  - gate entropy regularization
- Add at least one derived summary for discrete usage concentration so dead-code behavior can be inspected directly.
- Keep these summaries in the same JSONL metrics stream rather than a separate opaque artifact format.

### 6. Add ablation-oriented tests

- Add a test that verifies continuous-only fusion when `alpha = beta = 0`.
- Add a test that verifies discrete-only fusion when `alpha = beta = 1`.
- Add a test that verifies disabled loss weights produce zero-valued loss terms without shape regressions.
- Add a test that verifies warm-up fusion freezing leaves `alpha` and `beta` fixed for the configured number of epochs.
- Add a test that verifies temperature scheduling updates `gumbel_temperature` monotonically along the configured schedule.

### 7. Add experiment reporting surfaces

- Add an ablation runner script or a thin orchestration script that executes multiple experiment configs and writes a compact summary table.
- Keep it configuration-driven and readable.
- Store outputs per experiment under stable names so comparisons are reproducible.
- Add a summary artifact that compares:
  - reconstruction metrics
  - classification metrics
  - threshold values
  - final fusion scalars
  - active loss weights

### 8. Tighten repository reproducibility around ablations

- Add Weights & Biases logging if the repository is going to accumulate many ablation runs.
- Add `dvc.yaml` once augmented or derived datasets become materialized artifacts rather than only in-memory batch transforms.
- Record the exact experiment config with every checkpoint and evaluation output.

## File-level implementation targets

The following files are the natural homes for the ablation-readiness work:

```text
configs/experiment/
configs/model/thesis_multitask.yaml
configs/task/multitask_tsad.yaml
src/models/thesis_multitask.py
src/engine/trainer.py
src/engine/logger.py
scripts/train.py
scripts/evaluate.py
scripts/run_ablation.py
tests/test_multitask_shapes.py
tests/test_one_multitask_train_step.py
tests/test_registry.py
tests/test_fusion_ablation_modes.py
tests/test_temperature_schedule.py
```

## Pre-Phase-4 Gate

The original gate condition was that Phase 4 should remain blocked until the following are true. Repository update on 2026-04-02: the first conservative Phase 4 slice now exists because these conditions have been implemented for the accepted projector-first path.

- The offline multitask path can run continuous-only, discrete-only, and fused modes from config alone.
- Fusion warm-up and temperature annealing are first-class configuration options.
- Branch-collapse diagnostics are logged and inspectable.
- Ablation results can be compared from saved artifacts without ad hoc notebook code.
- The detail documents and design documents describe the same active codepath that the repository runs.

## Completion Standard

This checklist is complete when the repository can support repeated offline ablations by changing experiment YAML only, while preserving the one-model-one-file design and without reintroducing `tasks/`, `losses/`, or model-specific helper-file splits.
