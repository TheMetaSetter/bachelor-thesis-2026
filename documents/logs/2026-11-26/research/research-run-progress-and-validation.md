# Research: Run progress and MUSTARD validation

Status: Source inspection complete.
Source: `simplifying` at `a6e2af78`, with local changes present.
Requirements: [User decisions](../design/run-progress-decisions.md).

## Current paths

| Source | Finding |
| --- | --- |
| `src/cli.py:179`, `src/experiments.py:241`, `src/experiments.py:431` | The CLI already edits both remote flags. New experiments default to off. Smoke runs force both off. W&B uses per-run metrics. Kaggle backs up checked files for the experiment. Resume uses flags from the saved attempt. |
| `src/experiments.py:489`, `src/tui.py:106` | The experiment executes one run at a time. Textual pipes CLI output. A terminal-only stderr bar leaves that pipe and stdout free of redraws. |
| `src/methods/mustard/model.py:341`, `src/methods/mustard/model.py:402` | The training loop keeps total loss and checkpoint history. It restores the lowest validation-loss checkpoint at each stage. Stage B initializes the bank from training data after stage A. Other staged methods already publish epoch history through checkpoint callbacks. |
| `src/prepared_data.py:179`, `src/offline.py:81` | The prepared loader yields original indices and respects gaps, strides, and tail rules. Final calibration uses the selected checkpoint. During fitting, synthetic masks belong to individual windows. |
| `src/scoring.py:160`, `src/config.py:230`, `src/online.py:203` | The q95 profiles share an offline point threshold with online. Current-score weight 1 and previous-score weight 0 give the latest unsmoothed observation through the same reducer and identity checks. The standard q95 profiles use weights 0.25 and 0.75. |
| `src/methods/mustard/model.py:255`, `src/evaluation_metrics.py:211` | MUSTARD estimates variance across passes. The evaluator already computes a finite-value mean and returns an undefined result if the field is absent. Existing aggregation counts each window-point occurrence. |
| `src/artifacts.py:511` | Resume checks the saved source and dependency identity. Runs interrupted before a source change require the original environment to resume. |

## Implications

- Reuse the prepared data and split the original validation series at runtime before window creation. The model's fit loss still uses the existing validation path.
- Score stage A in raw-input space for both paired profiles. Stage B uses the selected profile score space.
- Reuse threshold fitting, window generation, augmentation, metric evaluation, and checkpoint commits. Add an epoch monitor only where the profile selects it.
- Keep the epoch score arrays temporary. Keep existing report fields and store the small new summaries with their source identities.
- Both paired q95 profiles already support the final EWMA threshold. Expose the latest-score arm through existing weights.

## Limits

This is source evidence. No experiment ran during planning.
The preference file names `src/models/`. The current constructors use `src/methods/`.
