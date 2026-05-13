---
date: 2026-05-13 22:14:17 +0700 +07
researcher: Codex
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation notes for RedLamp timestep encoder baseline"
tags: [detail, implementation, redlamp, baseline, mlp, timestep-encoder]
status: complete
source_detail: documents/logs/05-13-2026/detail/detail-redlamp-timestep-encoder-baseline.md
---

# Detail: RedLamp Timestep Encoder Baseline Implementation Notes

## Implemented Scope

- Changed RedLamp baseline encoder from flattened-window input to timestep input.
- Changed RedLamp baseline decoder from flattened-window output to timestep output.
- Kept classifier window-level by mean-pooling timestep hidden states.
- Preserved the existing batch, output, training, validation, testing, and evaluation contracts.
- Kept the baseline free of continuous prototypes, discrete codebook, fusion gates, memory/bootstrap logic, online adaptation state, and projector modules.

## Scientific Note

The original CANDI MLP reference uses flattened-window encoding. This repository intentionally changes the RedLamp baseline to timestep encoding so the comparison against the thesis model isolates prototype memory, discrete codebook, fusion, update gate, and memory/bootstrap logic. Old RedLamp checkpoints trained with flattened-window geometry are not compatible with the new timestep baseline architecture because encoder and decoder parameter shapes changed.

## Final Verification

- `./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py` -> `2 passed in 1.59s`.
- `./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py tests/test_config_loading.py tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py` -> `38 passed, 4 warnings in 3.83s`. The warnings are expected sklearn warnings for the existing single-class metric safety test.
- `./.venv/bin/python scripts/run_multiseed_experiments.py --config-paths configs/experiment/smd_redlamp_mlp_baseline_window20.yaml --preflight-only` -> completed preflight validation with `total_configs=1`, `main_configs=1`, `smoke_configs=0`, `preflight_only=True`.
