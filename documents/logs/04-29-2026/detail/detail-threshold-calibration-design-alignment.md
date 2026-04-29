---
date: 2026-04-29
researcher: Codex
repository: bachelor-thesis-2026
topic: "Threshold calibration design alignment and unresolved evaluator leakage"
tags: [detail, anomaly-detection, evaluation, thresholding, streaming]
status: complete
---

# Detail: Threshold calibration design alignment and unresolved evaluator leakage

## Context

The current evaluator was checked against the intended thesis protocol for anomaly-score thresholding.

The implementation computes reconstruction-based anomaly scores correctly:

- reconstruction models expose `point_scores` as MSE between reconstruction and input;
- `window_scores` are derived by averaging point scores within each window;
- the evaluator merges overlapping window point scores back to the entity timeline.

The unresolved problem is threshold calibration.

## Current implementation behavior

`src/engine/evaluator.py` computes a 95th-quantile threshold from the concatenated point scores of whichever loader is passed to `Evaluator.evaluate()`.

`scripts/evaluate.py` passes `data_bundle["loaders"]["test"]`.

Therefore the active CLI evaluation path calibrates the threshold on the full test timeline.

## Why this is a problem

Full-test quantile thresholding assumes that all test windows are available before the first anomaly decision is made. This is incompatible with the streaming setting, where future windows have not arrived yet.

It is also weak for ordinary held-out test reporting because the threshold has been selected from the same score distribution being evaluated. The resulting thresholded metrics should be considered provisional or oracle-style unless explicitly labeled that way.

## Required design rule

The default protocol must separate scoring from threshold calibration:

1. train the model on the training split;
2. calibrate the static threshold on train or validation scores;
3. save the threshold as a checkpoint or evaluation artifact;
4. evaluate validation, test, or stream data using that pre-calibrated threshold;
5. if using adaptive thresholding, update the threshold only from causal past/current stream state.

## Documents updated

- `documents/design/design_starter.md`
- `documents/design/stream_design.md`
- `documents/design/idea.md`
- `documents/design/long_term_codebase_roadmap.md`

## Remaining implementation work

- Add a threshold calibration API, for example `calibrate_threshold(model, calibration_loader, quantile)`.
- Change `Evaluator.evaluate()` so test evaluation receives a threshold instead of fitting one from test scores by default.
- Persist the calibrated threshold in checkpoint metadata or evaluation artifacts.
- Add tests proving that test evaluation does not compute its threshold from test scores.
- Add streaming tests proving that adaptive threshold updates never use future windows.
