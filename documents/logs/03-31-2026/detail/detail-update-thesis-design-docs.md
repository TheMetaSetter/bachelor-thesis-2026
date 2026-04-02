---
date: 2026-03-31 17:20:00 +0700
planner: Artificial Intelligence Agent
git_commit: 87c0e9b2a092b3e3b5a5b6f6ea5b54b4b948555d
branch: dev
repository: bachelor-thesis-2026
topic: "Update thesis design documents with fused-head objective and conservative online adaptation defaults"
tags: [detail, documentation, design, objective, fusion, online-adaptation]
status: complete
last_updated: 2026-03-31
last_updated_by: Artificial Intelligence Agent
---

# Detail: Update thesis design documents with fused-head objective and conservative online adaptation defaults

## Overview

Terminology normalized on 2026-04-02. Current design target: gate entropy regularization. Current implementation status: `src/models/thesis_multitask.py` now uses gate-entropy regularization directly while retaining the legacy margin field only for backward checkpoint compatibility.

This documentation update aligns the thesis design notes with the current consensus architecture and training strategy. The main goal was to make the design documents say one consistent thing about the fixed hidden-state contract, the offline multitask objective, the placement of the real prediction heads, and the scope of online adaptation.

## Updated design decisions

- Keep the thesis-facing encoder contract fixed at `H in R^{B x L x d_h}`.
- Keep the real reconstruction and anomaly-type classification heads only on the fused task-specialized states `H_rec` and `H_cls`.
- Record the modular offline objective surface explicitly, with default baseline `L_base = L_recon + lambda_cls L_cls` and optional extensions through `lambda_div L_div`, `lambda_var L_var`, `lambda_cov L_cov`, `lambda_use L_use`, and `lambda_gate L_gate`, where `L_gate` denotes gate entropy regularization in the design target.
- Document a minimal two-phase training recipe with a short warm-up, frozen fusion scalars at 0.5, and gradual Gumbel temperature annealing.
- Keep the main ablations as exact limiting cases of the same model: continuous-only, discrete-only, and fused.
- Treat online adaptation conservatively: update only the projector or another very small adapter first, use a near-identity residual projector with offline warm-start, and reserve NGD-style preconditioning for that small adapted subset rather than for the whole model.

## Files updated

- `documents/design/idea.md`
- `documents/design/design_starter.md`

## Notes

The documentation intentionally keeps the thin-waist codebase story unchanged. The stream stack remains River plus custom stream wrappers and drift injectors, and the one-model-one-file rule from `codebase_preferences.md` is preserved.
