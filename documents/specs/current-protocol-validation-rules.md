# Current protocol validation rules

Status: static inventory of explicit checks in the current code, reviewed 2026-11-06.

The four headings below group checks by context. The current code does not implement four shared protocol objects; checks are spread across config, data, score, evaluator, and online-update modules. “Input” and “output” mean the side of the relevant code boundary being checked. A behavior that happens without a validator is not listed as a validation rule.

## 1. Run and evaluation

### Input validation rules

1. YAML must have a mapping at its root and may not repeat a key. Experiment config must include required sections and reject unknown top-level keys.
2. Config validation checks supported data and model settings, the allowed MUSTARD variants (`O0`/`O1` and `A0`/`A1`/`A2`), valid reconstruction loss space, and optimizer and logging fields.
3. An online task seed, when provided, must match the run seed.
4. `validate_protocol_config()` requires test labels to be metrics-only, disallows test-based threshold selection, and requires point adjustment to be off. The data loader also requires the test split to contain both normal and anomalous points.

### Output validation rules

1. Model outputs must include the required output fields. The main tensors must have the expected ranks; auxiliary stochastic, uncertainty, and geometry fields have type, rank, and other field checks.
2. Calibrated point scores must be finite and within `[0, 1]`.
3. Evaluator window scores and labels must both have shape `[B, L]` and match. Metadata row count must match `B`.
4. Raw and normalized point scores, when supplied, must have the score shape and must be supplied together.
5. An evaluation record must include an entity, point scores, labels, and point count. Score and label arrays must be one-dimensional; score length and optional coverage/score arrays must match the point count. Optional raw and normalized scores must be finite.

## 2. Data and stream

### Input validation rules

1. Data config accepts only known keys and supported dataset names. The archive dataset requires a file path; the validation split ratio must be in `(0, 1)`.
2. Configured strides may not exceed window size. Worker counts, entity IDs, boolean options, and window limits are checked for accepted types and ranges.
3. A raw sequence must contain `x`, labels, mask, timestamps, and metadata. `x` must be rank 2; labels must be rank 1 and match sequence length; a mask must match `x`. Required metadata keys must exist.
4. An online range must provide both endpoints or neither. When provided, it must satisfy `0 <= start < end <= sequence_length`.
5. Online streaming accepts only `clean_stream_only=True` and the modes `sliding_stride_1` or `nonoverlap_tail`.
6. A scaler cannot fit an empty sequence list, must be fit before transform, and requires the transformed feature count to match its fitted feature count.

### Output validation rules

1. A window must contain the expected fields, rank-2 `x`, and required time and entity metadata.
2. A batch must contain the expected fields. `x` must be rank 3; optional labels, masks, and timestamps must have their required ranks; metadata must be a list.
3. An online batch must include rank-2 integer absolute indices that align with the batch and increase strictly within each window.
4. These validators do not check every relation: for example, `validate_window` does not check that label, mask, timestamp, and `x` lengths all match.

## 3. Scoring and calibration

### Input validation rules

1. Reconstruction scoring accepts `input_scaled` with shape `[B, L, D]`; reconstruction must be `[B, L, D]` or `[B, M, L, D]` with matching batch, window, and feature dimensions. Both inputs must be finite.
2. Point-score calibration input must be non-empty and finite. Calibration center must be finite; scale must be finite and positive; transform, estimator, and MAD normalizer must match supported values.
3. `validate_protocol_config()` restricts score space and transform, split and threshold sources, window and stream stride, quantile ranges, point adjustment, test-label use, and EWMA weights. Normalized-input scoring requires synthetic-normal validation thresholds.
4. A threshold artifact must use a supported schema and score identity, include required identity and threshold fields, and have valid types, ranges, source splits, strides, and provenance matches. Point calibration parameters are checked for finiteness and positive scale.

### Output validation rules

1. Reconstruction scores must be finite. Window-score aggregation requires each score vector to match the window size; it returns point scores with a coverage mask.
2. EWMA scoring checks that its two weights sum to `1`.
3. Threshold artifacts are validated when written or loaded. Checks include required threshold names and fields, quantile range, allowed score/source definitions, artifact identity, and ordered latent low/high thresholds.
4. The code does not generally check that every aggregated point score is finite or covered; uncovered points are represented as `NaN` and a coverage mask.

## 4. Method updates

### Input validation rules

1. The online runner accepts only `A0`, `A1`, or `A2`.
2. The online optimizer accepts only `projector_params` as its target group and supports `adam` or `adamw`.
3. Before optimizer creation, A0 must have no trainable online parameters. For other variants, only projector parameters may be trainable; projector parameters must be trainable and the model must have at least one trainable parameter.
4. Each online batch must contain exactly one causal window and one metadata row. The online batch contract also checks absolute indices.
5. Full-spec triage requires input and latent thresholds, and the latent low threshold may not exceed the high threshold.
6. Update-loss helpers check expected tensor shapes, mask shape, and positive temperature. Verification checks include positive capacity, required result entries, selected tensor shapes, and finite codebook values/radii.

### Output validation rules

1. No general post-update validator was found for finite updated parameters or losses, allowed parameter changes, or the full update-record schema. The trainable-parameter check runs before optimizer creation; it does not validate the resulting parameter state.
2. The step returns a record, a `did_update` flag, and an optional loss, but the returned values are not passed through a general output validator.

The step records scores before updating. It skips updates for A0, a missing optimizer, strong anomalies, or an update path with no loss. These are execution rules, not validation checks.

## Code checked

1. Config and shared contracts: `src/core/config.py`, `src/core/config_experiment_validation.py`, `src/core/contracts.py`.
2. Data and stream: `src/data/api.py`, `src/data/cleaning.py`, `src/data/scalers.py`, `src/data/split_protocol.py`, `src/data/stream.py`, `src/protocols/online_stream_range.py`.
3. Scoring and calibration: `validate_protocol_config()`, `src/engine/evaluator.py`, `src/engine/thresholding.py`, `src/protocols/point_scores.py`, `src/protocols/point_score_calibration.py`, `src/protocols/reconstruction_scores.py`, `src/protocols/threshold_artifact.py`.
4. Updates: `src/engine/online_tta/online_engine_run.py`, `src/engine/online_tta/online_engine_step.py`, `src/engine/online_tta/online_optimizer.py`, `src/engine/online_tta/online_losses.py`, `src/engine/online_tta/triage.py`, `src/engine/online_tta/verification_cycle.py`, and `src/engine/online_tta/signature_verification.py`.
