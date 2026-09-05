---
date: 2026-09-05
planner: OpenAI Codex
topic: "O1 reconstruction/classification weights 0.75/0.25 for machine-3-4 seed36"
status: ready
revision: 8c295b05
---

# Implementation Plan: O1 reconstruction/classification weights 0.75/0.25

## Goal

Create an isolated O1 weight-ablation configuration, then run one full offline
benchmark for SMD `machine-3-4` with random seed `36`.

The model must use normalized-input reconstruction MSE, `lambda_recon: 0.75`,
and `lambda_cls: 0.25`. O1 keeps its existing Balanced Point-Score Loss (BPSL).

## Confirmed behavior

- `lambda_cls` weights the O1 classification-side branch. In Stage A, the
  runtime combines classification loss and BPSL as
  `0.5 * (classification_loss + score_loss)` before applying `lambda_cls`.
- With the requested weights, Stage A uses
  `0.75 * L_recon + 0.25 * 0.5 * (L_cls + L_BPSL) + 0.3 * L_contrastive`.
- BPSL is active only in Stage A. Stage B retains `lambda_cls: 0.25` but does
  not apply BPSL.
- The model default reconstruction-loss space is `normalized_input`; it squares
  the difference between reconstruction and normalized batch input.

## Scope

### In scope

- One new model YAML config.
- One new O1 experiment YAML config for `machine-3-4`, seed `36`.
- One two-stage full offline benchmark command.

### Out of scope

- Changes to Python source, tests, loss formulas, or BPSL weighting logic.
- Raw-input MSE.
- Online TTA, other entities, other seeds, and the full benchmark matrix.
- Overwriting the established O1 output tree.

## Phase 1: Code editing

**Result:** an isolated O1 weight-ablation configuration exists and resolves to
the requested loss weights without changing the standard O1 configuration.

**Technology:** YAML and the existing experiment-config loader.

### Stage 1.1: Create the model configuration

1. Copy `configs/model/thesis_multitask_two_stage_point_score_window20.yaml` to
   the proposed new file
   `configs/model/thesis_multitask_two_stage_point_score_window20_recon075_cls025.yaml`.
2. Set `lambda_recon: 0.75`.
3. Set `lambda_cls: 0.25`.
4. Preserve `enable_score_loss: true`, all point-score settings,
   `reconstruction_normal_only: true`, and the absence of
   `reconstruction_loss_space: raw_input`.

### Stage 1.2: Bind the new model configuration to one O1 run

1. Copy
   `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1__machine_3_4__w20__seed36__main.yaml`
   to the proposed new file
   `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__main.yaml`.
2. Set `model_config_path` to the proposed model config from Stage 1.1.
3. Rename `experiment_name` and `logging.wandb_run_name` to include
   `O1_recon075_cls025`.
4. Add `o1-recon075-cls025` to `logging.wandb_tags`.
5. Set `output_dir` to
   `outputs/benchmark/smd/thesis/O1_recon075_cls025/machine_3_4/seed36`.
6. Set `checkpoint_dir` below that new output directory.
7. Preserve the O1 score-loss override, SMD data and task config, CUDA device,
   seed `36`, monitor metric, and Stage-A/Stage-B epoch budget `25 + 5`.

**Complete when:** loading the new experiment config resolves
`model.lambda_recon == 0.75`, `model.lambda_cls == 0.25`, and
`model.enable_score_loss is True`.

## Phase 2: CLI command

**Result:** one CUDA full offline run completes Stage A, Stage-B initialization,
Stage B, evaluation, clean-validation calibration, and artifact export.

**Technology:** `.venv/bin/python`, `pytest`, CUDA, and
`scripts.benchmarks.run_thesis_offline_benchmark`.

### Stage 2.1: Preflight

1. Run the focused loss and configuration tests:

   ```bash
   .venv/bin/python -m pytest -q tests/models/test_thesis_multitask_point_score_loss.py tests/test_config_loading.py
   ```

2. Materialize and validate the two-stage plan without training:

   ```bash
   .venv/bin/python -m scripts.benchmarks.run_thesis_offline_benchmark \
     --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__main.yaml \
     --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml \
     --dry-run
   ```

### Stage 2.2: Run the full offline benchmark

1. Run the single requested combination on a CUDA-capable environment. Do not
   pass `--skip-completed`, because this is a new output identity.

   ```bash
   .venv/bin/python -m scripts.benchmarks.run_thesis_offline_benchmark \
     --experiment-config configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O1_recon075_cls025__machine_3_4__w20__seed36__main.yaml \
     --protocol-config configs/protocol/smd_window20_cleanval_q99_ewma09.yaml
   ```

### Stage 2.3: Verify run artifacts

1. Confirm the Stage-A `best.pt`, `stage_b_init.pt`, and Stage-B `best.pt` exist.
2. Confirm `thresholds/thresholds.json`, `metrics/offline_metrics.json`,
   `metrics/uq_summary.json`, `protocol/resolved_protocol.json`, and the
   benchmark report exist.
3. Confirm Stage-A logs include BPSL for eligible batches and Stage-B logs do
   not apply BPSL.
4. Confirm the threshold artifact identifies `machine_3_4`, seed `36`, and the
   Stage-B checkpoint SHA-256.

## Risks and recovery

- **Risk:** writing under the established `O1` output root can replace or mix
  standard O1 evidence. **Mitigation:** use the new `O1_recon075_cls025` output
  identity. **Recovery:** stop before training if the resolved output path is
  not the new root.
- **Risk:** CUDA is unavailable in the execution environment. **Mitigation:**
  run the dry-run before the full command and execute the full run only on a
  CUDA-capable environment. **Recovery:** do not substitute a CPU run for the
  planned benchmark.
