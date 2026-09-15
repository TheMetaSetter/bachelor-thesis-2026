---
date: 2026-09-14
topic: Minimalize the current codebase
status: partial-low-risk-tranche-complete
---

# Implementation: Codebase Minimalization

## Result

The low-risk cleanup tranche is complete. The active runtime now uses one
experiment configuration tree instead of two.

## Changes

- Removed tracked `.DS_Store` files.
- Added `.DS_Store` and LaTeX build outputs to `.gitignore`.
- Removed the unused online threshold-calibration module.
- Removed the unused RedLamp MLP compatibility module.
- Updated the RedLamp profiling notebook to import the canonical model name.
- Removed the duplicate top-level recalibration script.
- Removed two byte-identical RedLamp configuration aliases.
- Moved three debug CPU presets to `configs/debug/offline_benchmark/thesis/`.
- Moved active references from `scripts/configs/experiment/` to `configs/experiment/`.
- Removed the stale `scripts/configs/experiment/` tree.
- Restricted full-matrix THESIS preflight to the declared O0 and O1 variants.

## Verification

- Focused baseline before changes: `111 passed, 1 skipped`.
- Focused verification after changes: `122 passed, 1 skipped`.
- Python compilation passed for `src/`, `scripts/`, and `tests/`.
- No tracked generated metadata remains for the removed artifact classes.

## Residual failures

The full suite still has five unrelated pre-existing failures involving model
state-key snapshots, metric snapshots, a checkpoint fixture, an encoder
introspection contract, and an offline artifact fixture. These failures were
not changed in this tranche.

The large configuration validators, trainer, evaluator, and THESIS mixin tree
remain. They require separate parity-tested refactors and should not be
collapsed in the same change.
