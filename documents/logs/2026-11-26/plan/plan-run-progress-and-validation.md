# Run progress and MUSTARD validation implementation plan

Status: Implementation and smoke check complete. See the [execution record](../detail/run-progress-and-validation-execution.md) for the six full-suite baseline failures.
Sources: [User decisions](../design/run-progress-decisions.md), [protocol rules](../../../specs/protocol-validation-rules.md#agreed-mustard-monitoring-and-calibration), [research](../research/research-run-progress-and-validation.md), [phase structure](../structure/structure-run-progress-and-validation.md).

**Goal.** Show progress and available statistics for each direct CLI run. Select and evaluate MUSTARD checkpoints with the agreed validation protocol.

**Design.** Use the existing experiment controls, prepared data, epoch callbacks, checkpoint commits, scoring functions, and result files. Give the CLI one display function. Give the paired MUSTARD profiles one epoch monitor.

**Tools.** Python, PyTorch, NumPy, PyYAML, tqdm, pytest, and Git.

## Scope and plan choices

- Apply the protocol to the paired MUSTARD q95 profiles. Preserve other profiles and the authors' baseline evaluation methods.
- Use the existing W&B and Kaggle flags. Both default to off. Smoke runs force both off. Resume uses the saved values.
- Keep Textual on its current subprocess path. Use terminal stderr for progress. Keep stdout available for command results.
- For an odd-length series, use `midpoint = length // 2`. Give the extra point to the held-out half.
- In epoch metrics, count each unsmoothed window-point occurrence once. Calculate the range metrics in separate synthetic windows. Do not join their events. For the paired MUSTARD q95 protocol, use a normal pseudo-label (`0`) for every point in the original validation series. Use a separate label array in memory. Use this array for epoch calibration, held-out FPR, and full-series final calibration and evaluation. Use the generated masks for synthetic metrics and the recorded test labels for test metrics.
- For the no-EWMA arm, use the most recent unsmoothed observation per point in final calibration and test. Use weights `(0.0, 1.0)` with the current reducer. Use `(0.25, 0.75)` in the existing EWMA arm.
- Store original and synthetic validation uncertainty as separate summary fields. Reuse the existing reports and the run result files. Show one uncertainty value per set. Keep the other configured report fields available.
- For baselines without a synthetic validation pass, show the losses that the method already gives. Show detection metrics at final evaluation. This is the proposed choice for Q8.
- Keep this work in `bachelor-thesis-2026`. No runtime change is planned for `tsad-lib`.
- The preference file names `src/models/`. The current code uses `src/methods/`. Use the current paths.

## File map

| Owner | Files | Work |
| --- | --- | --- |
| Run settings | `src/config.py`, `configs/evaluation_profiles/`, `src/experiments.py` | Select the paired protocol. Reuse stored remote flags. |
| Epoch evaluation | Proposed `src/mustard_validation.py`, `src/offline.py`, `src/methods/mustard/model.py` | Split and score the validation data. Track loss terms. Select checkpoints. |
| Final evaluation | `src/offline.py`, `src/online.py`, `src/evaluation_metrics.py` | Reuse calibration and scoring. Save per-set summaries. |
| Display | Proposed `src/progress.py`, `src/cli.py`, `src/experiments.py`, `src/offline.py`, `src/online.py`, `pyproject.toml` | Render run events with one tqdm bar. |
| Checks | `tests/test_cli.py`, `tests/test_integrations.py`, `tests/test_mustard.py`, `tests/test_offline.py`, `tests/test_online.py`, `tests/test_config.py`, proposed `tests/test_progress.py` | Check the new behavior and the current callers. |

## Phase 1. Check the current controls

### Stage 1.1. Record a baseline

- [x] **Git.** Record the branch, revision, and existing edits.
- [x] **File reader.** Read `python_semantics.md` before Python work.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_cli.py tests/test_experiments.py tests/test_integrations.py`. Record any existing failures.

### Stage 1.2. Verify remote controls

- [x] **pytest.** Check both `true` and `false` through `tsad experiment --name NAME set FIELD VALUE`. Cover `wandb_logging` and `kaggle_backup`.
- [x] **pytest.** Extend the existing tests if needed. Assert that disabled or smoke runs make zero provider calls. Change the YAML after interruption and assert that resume still uses the saved flags.
- [x] **pytest.** Repeat the focused command. Check that no real provider was contacted.

Phase check: Both switches work through the current CLI and saved attempt. Use the current code if it passes.

## Phase 2. Select MUSTARD checkpoints by held-out FPR

### Stage 2.1. Select the protocol

- [x] **pytest.** In `tests/test_config.py`, check that the paired profiles select `offline_monitor: held_out_fpr`. Check that retained profiles and baseline methods keep their current behavior.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_config.py` and confirm that the new assertion fails for the expected reason.
- [x] **Python.** Add optional `offline_monitor` validation in `src/config.py`. Permit `held_out_fpr` for paired MUSTARD q95 settings with original validation calibration and the identity score transform. Absence preserves the current monitor.
- [x] **YAML.** Set `offline_monitor: held_out_fpr` in the existing raw-input and latent MUSTARD q95 profiles.
- [x] **pytest.** Repeat the config tests.

### Stage 2.2. Compute the epoch data and statistics

- [x] **pytest.** In `tests/test_offline.py`, add fixtures for an odd-length split, a time gap, and overlapping windows. Assert that no calibration window or synthetic source window crosses the split. Assert q95 uses only the first half. Assert an injected label stays with its own window.
- [x] **pytest.** In `tests/test_mustard.py`, check that a loss term present in 32 of 40 windows has coverage 32. Check that its mean uses 32 as the denominator and that the original total loss is unchanged.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_offline.py tests/test_mustard.py`. Confirm the new checks fail for the missing behavior.
- [x] **Python.** In proposed `src/mustard_validation.py`, add `split_validation(split: dict) -> tuple[dict, dict]`. Slice values, labels, and timestamps at the midpoint. Rebuild segment bounds for each half. Keep the prepared split identity. Reuse `iter_batches` after the split.
- [x] **Python.** In `src/methods/mustard/model.py`, add keyword-only `score_space=None`, `generator=None`, and `uncertainty_fields=None` to `Mustard.score`. Preserve existing defaults. Use a separate evaluation generator for the monitor. Preserve training mode and training random state across monitoring.
- [x] **Python.** In the same file, keep a weighted sum and a window count per available loss term in `_epoch`. Add the term means and counts to the epoch result. Leave gradients and the existing total-loss formula in place.
- [x] **Python.** In `src/mustard_validation.py`, add `evaluate_epoch(method, validation: dict, config: dict, *, stage: str) -> dict`. Return `score_space`, `threshold`, `held_out_fpr`, and `synthetic_metrics`. In stage A, use raw-input MSE. In stage B, use the profile's score space. Score all groups before overlap smoothing. The initial implementation used known normal covered points for held-out FPR. Stage 2.4 replaces this label rule. Generate synthetic windows only from the held-out half with a separate repeatable seed. Pass synthetic segments and masks to the existing metric evaluator. If no eligible calibration scores or held-out normal scores exist, set the monitor value to `None` with a reason.
- [x] **pytest.** Repeat the focused epoch-data and loss tests.

### Stage 2.3. Select and restore checkpoints

- [x] **pytest.** In `tests/test_mustard.py`, use held-out FPR values `[None, 0.04, 0.04, 0.02]`. Assert epoch 4 wins and epoch 2 stays selected through epoch 3. Assert that an all-undefined stage fails. Assert raw-input scoring in stage A and latent scoring in stage B for the latent profile.
- [x] **pytest.** In `tests/test_offline.py`, interrupt after a saved epoch. Compare the resumed checkpoint selection, scores, and epoch history with an uninterrupted run.
- [x] **pytest.** Run the new checkpoint cases. Confirm that the old selection rule causes the expected failure.
- [x] **Python.** Add optional keyword-only `monitor_epoch=None` to `Mustard.fit`. The callback accepts the current method and stage and returns the Stage 2.2 record. Bind it in `src/offline.py` only when the evaluation profile selects held-out FPR.
- [x] **Python.** Update MUSTARD's stage checkpoint comparison to use a strict decrease in defined FPR for opted-in runs. Retain the initial checkpoint and the selected checkpoint. Save the monitor value, score space, and epoch result in the existing checkpoint state. Keep the established schema fields for other profiles.
- [x] **pytest.** Repeat `.venv/bin/python -m pytest -q tests/test_mustard.py tests/test_offline.py tests/test_artifacts.py`.

### Stage 2.4. Apply the all-normal validation rule

- [x] **pytest.** In `tests/test_offline.py`, add a validation fixture with null labels and a positive source label. Check that the paired protocol counts every covered held-out window-point occurrence in the denominator. Check that final calibration uses the complete original validation series, subject to the window schedule.
- [x] **pytest.** Add assertions that the prepared labels, generated masks, and test labels keep their values. Check that an epoch with no eligible calibration scores or no covered held-out scores still gives undefined FPR.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_offline.py` and confirm that the former label rule causes the expected failure.
- [x] **Python.** In `src/mustard_validation.py`, make a shallow copy of the original validation split with a separate all-zero label array.
- [x] **Python.** Use this validation split in the paired q95 epoch monitor and in the offline final calibration and validation evaluation. Keep the test scoring and evaluation paths as they are.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_mustard.py tests/test_offline.py tests/test_online.py tests/test_artifacts.py`. Check that the resumed fit and the uninterrupted fit still agree.
- [x] **pytest.** Repeat the final calibration, reporting, and display checks from Phases 3 and 4.

Continue at Stage 5.1 after these checks pass. Start a new smoke attempt for the updated code.

Phase check: Stage B starts from the selected stage A checkpoint. A restart uses the recorded winner and restores the same training state.

## Phase 3. Check final calibration and add uncertainty

### Stage 3.1. Verify both test arms

- [x] **pytest.** In `tests/test_online.py`, add a paired test with two scores at one point. Check that weights `0/1` give the latest score and `0.25/0.75` give the EWMA score. Check offline and online use the same saved q95 threshold. Assert that a parent with a different weight or score space is rejected.
- [x] **YAML.** Add raw-input and latent q95 companion profiles with weights `0.0` and `1.0`. Use names ending in `_latest`. Preserve validation eligibility, identity transform, and the offline-parent threshold setting.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_config.py tests/test_scoring.py tests/test_offline.py tests/test_online.py`. Check that test labels are used only by the evaluator and that each online score is recorded before adaptation.
- [x] **Python.** If a new test finds a gap, correct the owning scorer or validator. Keep one reducer for both weight choices.

### Stage 3.2. Save final summaries

- [x] **pytest.** Check that per-point pass scores `[1, 3, 5]` and `[2, 2, 2]` yield mean variance `2.0`. Check that one pass with correction 1 yields an undefined value. Check that offline test, online test, original validation, and synthetic validation summaries use records from their own scoring passes.
- [x] **Python.** For final original validation and test, reuse `point_anomaly_score_variance_mean` from `compute_metrics`. Use the selected checkpoint on validation and the actual causal score records on online test.
- [x] **Python.** In `src/offline.py`, score the held-out synthetic validation windows with the selected checkpoint. Reuse the Stage 2.2 window rule. Save a named synthetic uncertainty summary with the validation result. Reuse the existing finite-value mean. Keep only the new summary and its count, with no new saved score or variance arrays.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_offline.py tests/test_online.py tests/test_reporting.py`.

Phase check: Each arm shares one test threshold across phases. Its final results contain VUS-PR, Affiliation F1, and raw FPR. Show VUS-ROC and recall as secondary when configured. Preserve report compatibility.

## Phase 4. Show progress in the direct CLI

### Stage 4.1. Own one display per run

- [x] **pytest.** In proposed `tests/test_progress.py`, check one bar per run, stable stdout, no bar for piped stderr, `—` for missing values, and closure on success, error, and interrupt.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_progress.py`. Confirm the expected failure.
- [x] **TOML.** Add `tqdm` to the runtime dependencies in `pyproject.toml`. Use the repository's dependency pin format.
- [x] **Python.** In proposed `src/progress.py`, add `run_progress(config: dict)` as a context manager that yields a callback. The callback accepts keyword-only `stage`, `unit`, `completed`, `total`, and `stats`. Reuse one tqdm instance for the run. Update its label and counters as units change. Use stderr terminal detection and close in the context manager. Use `None` for unavailable totals or values.
- [x] **Python.** Connect the display to direct CLI run, standalone, and resume execution through an optional callback or context factory. Keep the callback outside the serialized run config. Preserve the existing no-display path for programmatic callers.
- [x] **pytest.** Repeat the progress tests.

### Stage 4.2. Publish progress from the current loops

- [x] **pytest.** Check that offline progress reads completed fitter epochs from the existing callback. Check that a method without epochs shows fit and evaluate milestones. Check that online progress starts at the saved cursor and counts each new window once.
- [x] **Python.** In `src/offline.py`, publish committed epochs from the existing fitter callback. Include available train and validation losses. Include stage, score space, held-out FPR, and the core synthetic metrics for the paired MUSTARD profiles. Use fit and evaluate milestones when a method has no epochs.
- [x] **Python.** In `src/online.py`, publish the processed-window count from the stream loop. Initialize the count from the saved cursor. Reuse the window schedule to obtain a total if available without reading or scoring the data again.
- [x] **Python.** At the end of evaluation, show the compact set of core test metrics and the agreed uncertainty summary. Use the source result and do not recalculate scores for display.
- [x] **pytest.** Run `.venv/bin/python -m pytest -q tests/test_progress.py tests/test_cli.py tests/test_tui.py tests/test_offline.py tests/test_online.py`.

Phase check: One direct CLI run has one bar. The bar displays the correct units and values. Piped and Textual output keeps its current form.

## Phase 5. Verify the complete path

### Stage 5.1. Run one prepared-data combination

- [x] **File editor.** In an isolated project directory, create one experiment named `progress_acceptance` using prepared SMD, one entity, seed 6, MUSTARD O0 offline and its A0 online pair, and the raw-input q95 EWMA profile. Turn both remote flags off. Use the bounded smoke path.
- [x] **CLI.** Run `.venv/bin/tsad --project-root "$verification_root" validate experiment --name progress_acceptance`.
- [x] **CLI.** Run `.venv/bin/tsad --project-root "$verification_root" run experiment --name progress_acceptance --smoke` in a terminal.
- [x] **Artifacts.** Check stage A and B checkpoint roles, held-out FPR, the final calibration identity, and the exact threshold reused online.
- [x] **CLI.** Run `.venv/bin/tsad --project-root "$verification_root" report experiment --name progress_acceptance --smoke` and inspect the reports.

### Stage 5.2. Review the result

- [x] **pytest.** Run `.venv/bin/python -m pytest -q`. Compare any failures with the baseline.
- [x] **File editor.** Update `README.md` with the existing flag commands, bar behavior, profile selection, and the epoch and test metric sources.
- [x] **Git.** Run `git diff --check`.
- [x] **Git.** Review the changed files against this plan and identify any duplicate score, progress, or calibration path that can be removed within scope.
- [x] **Markdown.** Record the observed checks and remaining limits in the execution record.


## Execution progress

1. **Phase 1.** The baseline had 26 passes and three failures. The repeated check had 31 passes and the same three failures. The failures use an SMD test definition that selects SWaT. The remote control tests used test doubles.
2. **Phase 2.** The checkpoint, offline run, and artifact checks had 29 passes. The resumed fit gave the same epoch history and test scores as the uninterrupted fit.
3. **Phase 3.** The score reduction checks had 42 passes. The uncertainty and reporting checks had 52 passes.
4. **Phase 4.** The display and runner checks had 92 passes.
5. **Phase 5.** The CLI validated two planned runs for SMD `machine-1-1`. The offline run stopped at stage A with `no defined held-out FPR`. The online run had no successful parent. The prepared SMD version has 28 entities. Each entity has zero points with a known normal label in its held-out validation half. The user approved all-normal pseudo-labels for original validation. Stage 2.4 passed. The new smoke attempt completed for both phases. The execution record gives the artifact and full-suite results.

## Recovery and compatibility

Saved attempts retain their original settings. Resume already checks source and dependency identity. Finish an earlier interrupted run with its original checkout and environment. Use a new run ID under the new code. Existing result files stay readable. If rollback is necessary, restore the earlier checkout and dependency environment and keep completed results.
