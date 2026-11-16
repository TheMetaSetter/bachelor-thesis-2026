# Protocol validation rules

Status: proposed contract for the next version of `bachelor-thesis-2026`.

A protocol groups input and output validation rules for one context. An input rule checks data before the protocol uses it. An output rule checks a result before it is accepted or passed on. The lists below are proposed checks; the current code does not implement them as four shared protocol objects.

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
3. Any fitted scaler is fit on training data only; validation and test data use that fitted scaler.
4. Window size and stride are positive, and any requested stream range is valid for the selected entity.

### Output validation rules

1. Each split has the expected feature shape and compatible labels.
2. Each window has the configured length and feature count.
3. Window labels and metadata point to the correct time positions in the source series.
4. Stream windows follow time order and remain within the requested range.

## 3. Scoring and calibration

### Input validation rules

1. The score space and score transform are supported by the selected method.
2. Score inputs and model outputs have compatible shapes and contain only finite values.
3. Calibration uses the declared validation data; test labels do not select score transforms or thresholds.
4. Quantiles are in `(0, 1)`, and any EWMA weights are valid and sum to `1`.
5. A loaded threshold artifact matches the run's method, variant, entity, seed, window size, score space, and source checkpoint.

### Output validation rules

1. Produced scores are finite, correctly shaped, and aligned to their windows or points.
2. Score direction is declared and consistent; larger scores mean more anomalous when using the shared evaluator.
3. Threshold values are finite and are tied to the score definition and calibration settings that produced them.
4. The threshold artifact records enough run and checkpoint identity to reject a mismatched artifact later.
5. Predictions use the declared threshold comparison rule.

## 4. Method updates

### Input validation rules

1. The selected method declares whether it supports online updates; an update is rejected when support is absent.
2. Update settings are valid for the selected method and variant. `O0`–`O2` and `A0`–`A2` are MUSTARD-specific labels.
3. An enabled update has the required optimizer, finite input tensors, and correctly shaped masks or verification results.
4. Update decisions use the current window's scores and thresholds from the declared scoring and calibration rules.

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
