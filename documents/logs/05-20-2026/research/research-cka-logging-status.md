---
date: 2026-05-20 22:03:32 +0700
researcher: TheMetaSetter
git_commit: 97ae0393eda8326b3e4bff9c7f19678014c9d25f
branch: dev
repository: bachelor-thesis-2026
topic: "Codebase status of CKA computation logging"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-05-20
last_updated_by: TheMetaSetter
---

# Research: Codebase status of CKA computation logging

**Date**: 2026-05-20 22:03:32 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: 97ae0393eda8326b3e4bff9c7f19678014c9d25f  
**Branch**: dev

## Research Question
Hiện tại codebase có log lại kết quả tính toán CKA chưa?

## Summary
The current codebase computes per-sample linear CKA values for fusion gating in `ThesisMultitaskModel`, but it does not explicitly log the CKA values themselves as training or validation metrics. The training logs include derived gate statistics (`alpha`, `beta`, `alpha_std`, `beta_std`) and contrastive loss, not raw CKA outputs.

## Detailed Findings

### Data Preparation
- Not applicable for this focused query. No data pipeline component logs CKA values.

### Modeling and Training
- CKA is computed inside fusion logic when `enable_cka_gated_fusion` is enabled.
- Two CKA vectors are built:
  - `cka_reconstruction = _compute_batch_linear_cka_scores(base_hidden, continuous_hidden)`
  - `cka_classification = _compute_batch_linear_cka_scores(paired_hidden, discrete_hidden)`
- These are stacked into `cka_features` and immediately passed to gate MLPs to produce `alpha` and `beta`.
- The forward output auxiliary dictionary logs gate summaries (`alpha`, `beta`, `alpha_std`, `beta_std`) but does not expose `cka_reconstruction`, `cka_classification`, or `cka_features`.

### Evaluation
- Stage logs include `*_contrastive_loss`, `*_alpha`, `*_beta`, `*_alpha_std`, and `*_beta_std`.
- Stage logs do not include `*_cka_*` metrics.

## Code References
- `src/models/thesis_multitask.py:1348` - computes `cka_reconstruction`
- `src/models/thesis_multitask.py:1352` - computes `cka_classification`
- `src/models/thesis_multitask.py:1356` - stacks into `cka_features`
- `src/models/thesis_multitask.py:1390` - fusion aux block includes gate summaries only
- `src/models/thesis_multitask.py:2012` - stage log dictionary definition
- `src/models/thesis_multitask.py:2028` - logs contrastive loss
- `src/models/thesis_multitask.py:2031` - logs `alpha` and `beta`
- `src/models/thesis_multitask.py:2033` - logs `alpha_std`
- `src/models/thesis_multitask.py:2036` - logs `beta_std`

## Pipeline Documentation
CKA currently acts as an internal feature transform for fusion gating and is not part of the exported metric surface.

## Historical Context (from documents/)
No explicit design or prior research note in `documents/` was required to determine this implementation fact. Evidence is directly present in model code.

## Open Questions
- Should raw CKA statistics (for example, mean and standard deviation per stage) be added to stage logs for experiment diagnostics?
