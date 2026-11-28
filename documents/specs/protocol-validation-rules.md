# Protocol validation rules

Status: proposed contract for the next version of `bachelor-thesis-2026`.

A protocol groups input and output validation rules for one context. An input rule checks data before the protocol uses it. An output rule checks a result before it is accepted or passed on. The lists below are proposed checks; the current code does not implement them as four shared protocol objects.

## Confirmed boundary: 2026-11-16

Apply the [runtime replacement decisions](../logs/2026-11-16/plan/decisions-runtime-replacement.md). All checks below remain proposed; this update does not change the current-code inventory.

## 1. Run and evaluation

### Input validation rules

1. The run identifies its dataset, entity, method, seed, phase, protocol, and resolved configuration.
2. The selected dataset, entity, method, phase, and variant are supported together.
3. An online run that needs offline state names the matching offline run and provides its required artifacts.
4. Evaluation inputs include labels for the same evaluated points as the produced scores.

### Output validation rules

1. The result record preserves the resolved run identity and status.
2. Scores, labels, and predictions cover the same evaluated points.
3. Reported metrics are finite and are computed from the recorded scores, labels, and declared evaluation settings.
4. A run is marked successful only when its required result artifacts are present and readable.

## 2. Data and stream

### Input validation rules

1. The selected entity exists and provides numeric time-series features with a consistent feature count.
2. Data splits are non-empty, chronological, and do not overlap.
3. Prepared data records a scaler fit on final training data. Runtime loading verifies its saved identity and reuses prepared values. It must not split, fill, or scale them again.
4. Window size and stride are positive, and any requested stream range is valid for the selected entity.

### Output validation rules

1. Each split has the expected feature shape and compatible labels.
2. Each window has the configured length and feature count.
3. Window labels and metadata point to the correct time positions in the source series.
4. Stream windows follow time order and remain within the requested range. Predictions use only observations available at their declared scoring time; no future-window contribution may change a committed prediction.

## 3. Scoring and calibration

### Input validation rules

1. The score space and score transform are supported by the selected method.
2. Score inputs and model outputs have compatible shapes and contain only finite values.
3. Calibration uses the source that the evaluation profile approves.
   For the retained normalized-input SMD q99 profile, synthetic calibration windows come from saved validation.
   Use fixed checkpoint parameters and check source-label eligibility separately from the generated mask.
   Test values do not fit score transforms or thresholds.
   RedLamp class selection and score scaling are fixed from validation before test scoring.
4. Quantiles are in `(0, 1)`, and any EWMA weights are valid and sum to `1`.
5. A loaded threshold artifact matches the run's method, variant, entity, seed, window size, score space, and source checkpoint. It also matches the sample reduction, the overlap reduction, and any EWMA weights.
6. MUSTARD latent MSE requires continuous prototype retrieval. The normalized query and the normalized reconstruction have the same latent width and point positions.
7. The MUSTARD q95 online comparison requires a point threshold from the selected offline parent. Its score space and point EWMA settings match those of the online run.

### Agreed MUSTARD monitoring and calibration

Use raw FPR on [held-out validation data](definitions.md#core-terms) to monitor offline training and select the best checkpoint.
Fit each epoch threshold at the 95th percentile of scores from the original calibration subset.
Use unsmoothed scores for epoch calibration, held-out validation, and synthetic validation.
After training, use the selected checkpoint to fit a separate 95th-percentile threshold from the full original validation series with the profile's eligibility rules.
For the point EWMA ablation, apply point EWMA to final calibration and scoring in both test phases.
Use this saved threshold for offline and online test.

The 95th percentile uses a nominal 5% anomaly contamination assumption for the full original validation series.
This is a threshold design assumption. A quantile does not establish the actual anomaly proportion or guarantee a 5% FPR.
If the profile excludes known anomaly windows, the quantile is computed on the eligible scores.

The current MUSTARD fit loop selects checkpoints by validation total loss. The raw FPR rule is an agreed change.
For latent-score runs, use raw-input scores to monitor stage A and latent scores to monitor stage B.
Fit each stage's epoch thresholds in that stage's score space.
Keep the earlier checkpoint when FPR ties.
Skip epochs with undefined FPR. Fail the stage if no epoch has a defined FPR.

At final evaluation, report the mean point-score variance across stochastic passes for original validation, synthetic validation, and test.
Show `—` if no variance estimate is available.

### Output validation rules

1. Produced scores are finite, correctly shaped, and aligned to their windows or points.
2. Score direction is declared and consistent; larger scores mean more anomalous when using the shared evaluator.
3. Threshold values are finite and are tied to the score definition and calibration settings that produced them.
   The retained SMD profile records the validation source and generation settings with its point and window q99 thresholds.
4. The threshold artifact records enough run and checkpoint identity to reject a mismatched artifact later.
5. Predictions use the declared threshold comparison rule.

## 4. Method updates

### Input validation rules

1. The selected method declares whether it supports online updates; an update is rejected when support is absent.
2. Update settings are valid for the selected method and variant. `O0`–`O2` and `A0`–`A2` are MUSTARD-specific labels.
3. An enabled update has the required optimizer, finite input tensors, and correctly shaped masks or verification results.
4. Update decisions use arrived, unlabeled observations and already recorded scores with the declared calibration rules. Test labels are unavailable to methods and adaptation diagnostics. Reject future observations, complete-test statistics, and test-based tuning.

### Output validation rules

1. Each update attempt records its stream position, decision, and outcome or skip reason.
2. Only parameters allowed by the selected method and variant can change.
3. Updated parameters and reported update losses are finite.
4. The score recorded for a window is unchanged by an update from that same window; an update may affect later windows.
5. The output state preserves enough information to resume at the next valid stream position.

## Evidence checked

1. `documents/specs/definitions.md` — agreed experiment terms and the four protocol contexts.
2. `documents/spec/online_tta_terminology_ontology.md` — offline artifacts, online stream objects, score records, and update state.
3. `documents/notes/experiment-flows-and-equations.md` — current reference split, scoring, calibration, evaluation, and update flows.
