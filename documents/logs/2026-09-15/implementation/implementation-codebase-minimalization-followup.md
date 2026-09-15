# Codebase Minimalization Follow-up

## Main Point

The follow-up cleanup kept the required SMD and THESIS benchmark paths working.
The full test suite now passes.

## Changes

- Replaced deleted experiment paths in the training, evaluation, ablation,
  online adaptation, visualization, and configuration-help defaults.
- Allowed normalized-input checkpoints to omit scaler state.
- Kept a clear error when raw-input reconstruction lacks scaler state.
- Updated model and metric contract fixtures for the current encoder and metric
  surfaces.
- Updated the runtime-readiness fixture to provide the artifacts and metadata
  required by the current export contract.
- Removed five unreferenced stress, smoke5, and CUDA smoke experiment presets.

## Verification

The focused contract tests passed with 20 passed and 1 skipped.

The full suite passed with 611 passed and 2 skipped.

The suite emitted 32 existing metric warnings from single-class test fixtures
and STUMPY constant-channel handling.

## Remaining Scope Decisions

No additional feature family was removed because the repository still contains
references or unclear ownership for CANDI, M2N2, traditional baselines,
anomaly-archive tooling, generic online baselines, notebooks, and historical
documents.
