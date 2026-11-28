# Run progress decisions

Status: Design discussion.

## Confirmed

- Use the existing W&B and Kaggle on/off settings.
- Resume uses the saved W&B and Kaggle settings.
- Use `tqdm` to show progress for the current run in the direct CLI.
- Review Textual support in a later discussion.
- Show train and validation losses after each completed epoch.
- Show available loss terms and total loss.
- Show validation metrics after each completed epoch.
- During training, use original and synthetic validation windows.
- Split the original validation series 50/50 in time order before making windows.
- Use the first half to fit the epoch threshold at the 95th percentile.
- Use the held-out half to measure raw FPR and to generate synthetic windows for epoch metrics.
- For the paired MUSTARD q95 protocol, assign a normal pseudo-label (`0`) to every point in the original validation series. Use these labels in both epoch subsets and in full-series final validation.
- Keep the pseudo-labels in memory. Preserve prepared labels, synthetic injection masks, and test labels.
- Interpret held-out raw FPR as the fraction of covered window-point scores above the epoch threshold under the all-normal assumption. Unknown source labels do not make this rate undefined.
- Use unsmoothed scores for epoch calibration, held-out validation, and synthetic validation.
- Monitor MUSTARD offline training and select the best checkpoint by held-out raw FPR.
- Keep the earlier checkpoint when FPR ties.
- Skip epochs with undefined FPR. Fail the stage if no epoch has a defined FPR.
- For latent-score runs, use raw-input scores in stage A and latent scores in stage B.
- Fit a separate epoch threshold in the score space of each stage.
- Show the score space in the progress bar.
- Measure VUS-PR, Affiliation F1, and raw FPR on synthetic windows with the same epoch threshold where required.
- Use VUS-PR, Affiliation F1, and raw FPR as the three core metrics.
- Treat VUS-ROC and recall as secondary metrics.
- Show test metrics when test evaluation finishes.
- Use point EWMA as an ablation option for final calibration and test scoring.
- Fit a separate 95th-percentile test threshold on the full original validation series after training.
- For the EWMA option, apply point EWMA to final calibration and to scoring in both test phases.
- Use a nominal 5% anomaly contamination assumption for the full original validation series.
- Use the saved test threshold for both offline test and online test.
- Use the three core metrics for both test phases.
- Show one configured uncertainty summary per set during final evaluation.
- Use the mean of point-score variances across stochastic passes.
- Show `—` if no variance estimate is available.
- Use the selected checkpoint for original and synthetic validation. Use scores from the actual run for each test phase.
- Show progress for all methods and phases.
- Use epochs, windows, or milestones as applicable.
- Apply the new validation protocol to the paired MUSTARD profiles first.
- Keep the original baseline evaluation methods.

## Tentative

- For baselines without synthetic validation, show progress and the available train and validation losses.
- Show their detection metrics when evaluation finishes.

## Alignment

- The CLI specification and experiment runner use the two existing remote settings.
- The current resume path uses the saved settings.
- The current MUSTARD fit loop selects the best checkpoint by validation total loss.
- The current offline runner calibrates detection after it restores the selected checkpoint.
- The current metric functions require a threshold for Affiliation F1 and raw FPR.
- Synthetic masks belong to individual windows. Overlapping windows can give different labels to the same source point.
- The paired MUSTARD q95 profiles already share the offline threshold with online test.
- Other profiles can use synthetic calibration, phase-specific calibration, or different score reduction.
- The configured latent score is unavailable before the prototype bank exists.
