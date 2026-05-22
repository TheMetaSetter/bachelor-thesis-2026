---
date: 2026-05-22T19:10:00+07:00
author: Artificial Intelligence Agent
status: completed
related_detail: documents/logs/05-22-2026/detail/detail-reconstruction-oscillation-ablation-and-classification-diagnostics.md
---

# Phase 0 Baseline Checklist

- Baseline experiment config fixed at `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_recon_diag_quick_100ep.yaml`.
- Canonical train command fixed at:
  - `python scripts/train.py --experiment-config configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20_recon_diag_quick_100ep.yaml`
- Pre-change expectation captured:
  - Classification scalar metrics are present in epoch logs for `train` and `val_synth`.
  - Classification diagnostics artifacts (hard-ratio and confusion-matrix JSON outputs) are not present yet.
- Scope boundary confirmed:
  - No change to batch contract, encoder contract, registry/factory contracts.
