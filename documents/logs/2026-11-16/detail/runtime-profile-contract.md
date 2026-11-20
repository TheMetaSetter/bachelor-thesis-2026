# Runtime profile contract

Date: 2026-11-17.
Status: Approved for implementation on 2026-11-18. Sections 13 and 14 control score storage and epoch counts. See the implementation checklist for runtime progress.

Authority: [confirmed decisions](../plan/decisions-runtime-replacement.md). Follow the [implementation checklist](../plan/plan-implement-runtime-signatures.md) in order.

## 1. Shared rules

1. Each field has one owner. Reject unknown fields at every supported mapping level.
2. A required field has no hidden default. Supplied profiles must state its value.
3. An integer field rejects booleans. Numeric fields must be finite. Counts and lengths must be positive unless zero has an explicit meaning.
4. Profile references are relative to `configs/`. Reject absolute profile references and references that escape that folder, including through a symbolic link.
5. A prepared-data reference selects an existing version and manifest. It is not a raw-source reference.
6. Resolution records selected profile paths and effective values without changing inputs or writing outputs.
7. A cell cannot override arbitrary profile fields. An intentional alternative uses a named profile. Reporting precision has the already agreed narrow override.
8. An experiment uses one data profile per dataset. A named method window length inside that profile does not create another data profile.

## 2. Data profile fields

| Field | Type | Required or default | Rule |
| --- | --- | --- | --- |
| `dataset` | string | Required | One of `iccad`, `smd`, `iops`, `nasa`, `swat`. |
| `prepared_root` | path string | Required | Existing selected version directory. Built-in profiles use `data/preprocessed/<dataset>/<version_id>/`. Resolve a relative path from the project root. |
| `window_size` | positive integer | Required | Dataset default window length. |
| `method_window_sizes` | mapping of method names to positive integers | Default `{moment: 512}` | Known names only. An unlisted method uses `window_size`. A supplied mapping must state its intended overrides explicitly. |
| `train_stride` | positive integer | Required | Training-window step. |
| `validation_stride` | positive integer | Required | Offline validation-window step. |
| `test_stride` | positive integer | Required | Offline test-window step. |
| `online_stride` | positive integer | Required | Online window step. |
| `method_strides` | mapping of known method names to stride mappings | Default empty mapping | Each child permits only `train`, `validation`, `test`, and `online`, with positive integer values. An absent child uses the dataset stride. |
| `tail_policy` | mapping | Required | Exactly `train`, `validation`, `test`, and `online`; each value is `drop` or `end_align`. |
| `gap_policy` | string | Required | `break_at_time_gaps` for timestamped sources with known cadence; `preserve_source_rows` for timestamp-free sources. |
| `expected_interval_seconds` | positive finite number or null | Required | Five minutes is 300; SWaT uses 1. Timestamp-free profiles use null. Do not infer it from test data. |
| `batch_size` | positive integer | Required | Offline batch size. Online execution uses one arrived window at a time. |
| `shuffle_train` | boolean | Required | Training order only. Never shuffle online windows. |
| `synthetic_profile` | profile path or null | Default null | Training augmentation only. It never changes prepared observations. |

The effective window length is `method_window_sizes.get(method_name, window_size)`. Record it in run, checkpoint, calibration, and resume identities. Do not pad MOMENT solely to retain another method's shorter window length.

All window construction settings belong to the data profile. Evaluation profiles must not repeat length, stride, tail, or gap settings. Numeric stride values and tail/gap selection are bound in Stage 1.2 step 4 of the decision record. A required field has no implicit fallback. Run-specific stream ranges remain in execution `limits`, not in a reusable profile.

A synthetic profile permits exactly `min_segment_fraction`, `max_segment_fraction`, `spike_scale`, `anomaly_visibility_boost`, and `family_intensity`. The first two are required fractions with `0 < min <= max <= 1`; the next two are required positive numbers. `family_intensity` defaults to an empty mapping. Its only family keys are `spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, and `mixture`. The retained family fields are listed in section 8. Do not repeat window length in a synthetic profile.

## 3. Evaluation profile fields

Scoring policy belongs here. Its phase applicability follows the reviewed method capabilities; offline-only methods do not require unused online calibration fields.

| Field | Type | Required or default | Rule |
| --- | --- | --- | --- |
| `score_space` | string | Required | `raw_input`, `normalized_input`, `method_native`, or `latent`, subject to method support. `latent` requires MUSTARD with active continuous prototype retrieval. |
| `point_score_transform` | string | Required | `identity`, `validation_median_iqr`, or `validation_mad_logistic`. Record the fitted source and transform parameters; test each retained formula before use. |
| `offline_threshold_split`, `online_threshold_split` | string | Required for each selected phase | `train` or `validation`; offline also permits `synthetic_validation` when the profile declares generation from the saved validation split. Calibration replays its approved source with fixed method parameters. Online uses this source for locally fitted thresholds, including triage when applicable. |
| `offline_point_threshold_source`, `offline_window_threshold_source` | string | Default to `offline_threshold_split` | Also permit `synthetic_validation_normal` for the retained explicit synthetic-normal profiles. Record generation and source eligibility separately. |
| `offline_threshold_quantile`, `online_threshold_quantile` | number | Required for each phase that fits its point threshold | Strictly between zero and one. Omit the online quantile when the online point threshold comes from the offline parent. |
| `offline_overlap_reduction` | string | Default `mean` | Use `last_ewma_estimate` for the MUSTARD q95 comparison. |
| `online_point_threshold_source` | string | Default `phase_calibration` | Use `offline_parent` to reuse the selected parent's saved point threshold. This setting does not select the source for triage calibration. |
| `adaptation_threshold_split` | string or null | Required for M2N2; otherwise null | M2N2 uses `train` for its native predicted-normal adaptation threshold, with the explicit online quantile. This threshold is separate from reported evaluation thresholds. No test source is permitted. |
| `triage` | mapping or null | Default null | When required, exactly `window_quantile`, `latent_low_quantile`, `latent_high_quantile`; each is in `(0, 1)`, with low < high. |
| `metrics` | nonempty list of distinct metric names | Required | Names resolve through shared metric definitions and supported computations. |
| `vus_max_buffer_size` | nonnegative integer | Required when a VUS metric is requested | Explicit metric parameter. |
| `vus_num_thresholds` | positive integer | Default 200 when VUS is requested | Record the resolved value. |
| `fpr_budgets` | distinct numbers in `(0, 1]` | Default `[0.001, 0.005, 0.01]` for budgeted VUS | Do not use these values to tune on test labels. |
| `uncertainty_fields` | list of distinct names | Default empty list | Only retained uncertainty fields in section 9 are permitted. |
| `uncertainty_statistics` | list of distinct names | Required when uncertainty is requested | Select from `mean`, `std`, `min`, `p50`, `p95`, `max`. |
| `test_label_usage` | string | Required, fixed value `metrics_only` | Test labels do not reach calibration or adaptation. |
| `point_adjustment` | boolean | Required, fixed value false | Do not alter ordinary predictions with labels. |
| `calibration_label_policy` | string | Required, value `exclude_known_anomalies` for normal-validation calibration | Exclude a whole window if any point is a known anomaly. Permit unknown labels under the recorded assumption. |
| `ewma_current_weight` | number | Default 0.75 | In `[0, 1]`. |
| `ewma_previous_weight` | number | Default 0.25 | In `[0, 1]`; the pair sums to one. |
| `online_reporting_score` | string | Required, fixed value `last_ewma_estimate` for the selected smoothing policy | Record the final estimate's availability time. |

1. A supplied EWMA override must supply both weights. Do not combine one supplied weight with an incompatible default.
2. Apply EWMA to repeated scores for the same source point index across windows. The first estimate equals its raw score.
3. Preserve each window's pre-update event. Pending point estimates may change until their final contributing window; finalized scores cannot change.
4. Calibration records the selected, excluded, and unknown-label window counts. Stop when no eligible windows remain.
5. Never delete labeled anomaly points and join the remaining points into a window. Select from original indexed windows.
6. Unknown labels are ineligible for label-dependent metrics. Calibration eligibility and metric eligibility have separate meanings.
7. Explicit old 0.1/0.9 profiles retain those values. A new default profile must use a name that does not claim a different pair.

### MUSTARD validation q95 comparison

1. Test two point scores. Use mean squared error in raw input space for the first score and mean squared error in latent space for the second score.
2. For raw input MSE, inverse-transform the observation and reconstruction with the saved training scaler. Take the mean of the squared differences across input features at each point.
3. Offline, replay eligible validation windows with the selected best checkpoint and fixed parameters. Take the 0.95 quantile of the final point EWMA estimates for the selected score. Save the threshold with its calibration identity. Apply the existing validation label policy.
4. Offline test and online test both use that score definition and compare the final point estimates with the saved threshold. Apply EWMA only to estimates of the same point from overlapping windows. Initialize each point with its first score. Use weights 0.25 for the previous estimate and 0.75 for the newest score. Use the same offline window policy for validation and test. Online retains its causal window policy.
5. For latent MSE, compare the L2-normalized query to the reconstruction from the continuous prototype bank at the same time point. Take the mean of the squared differences across latent features. Use the continuous retrieval output before fusion and the input reconstruction head.
6. For stochastic retrieval, L2-normalize each retrieved prototype mixture. Compute its MSE against the same normalized query, then take the arithmetic mean of these MSE values at each point within the window.
7. Use `evaluation_profiles/mustard_raw_input_q95_ewma075.yaml` or `evaluation_profiles/mustard_latent_q95_ewma075.yaml` for both phases of a paired comparison. Both use the `identity` score transform. Offline overlap averaging remains the default for other profiles.
8. The online run loads the point threshold from its selected compatible offline parent and keeps it fixed under input distribution shift. Reject a missing or incompatible threshold. Match the parent checkpoint, score space, sample and overlap reductions, EWMA weights, and calibration provenance. Preserve the offline source and schedule in that provenance. Online update triage has separate thresholds and keeps its existing rules.
9. This comparison takes the place of the proposed translation of the ambiguous legacy SMD q95 profile. The other retained profiles keep their settings.

## 4. Method profile fields

A profile selects one method variant. Several supported phases can share its settings. This prevents copied baseline profiles that differ only by phase.

| Field | Type | Required or default | Rule |
| --- | --- | --- | --- |
| `method` | string | Required | One retained method name. |
| `variant` | string | Required | MUSTARD phase variant, `default` for a baseline without variants, or `zero_shot`/`linear_probing` for MOMENT. |
| `phases` | nonempty list of distinct strings | Required | `offline` and/or `online`, restricted by capabilities. |
| `parameters` | method-specific mapping | Required | Exactly the fields listed below for the selected method and variant. |
| `training` | mapping or null | Default null | Required for a profile that fits state offline. Forbidden for a zero-shot profile. |
| `update` | mapping or null | Default null | Required only for an update-capable variant. |
| `required_offline_variant` | string or null | Default null | Required when the selected online profile depends on an offline variant. |

### Training and update fields

1. `training` contains exactly `stages`. It is a nonempty list of stage mappings. Each stage contains `name` (`stage_a`, `stage_b`, or `fit`), `epochs`, `optimizer`, `monitor`, `gradient_clip_norm`, and optional `scheduler` (default null). MUSTARD offline uses `stage_a` then `stage_b`; other fitted methods use `fit`.
2. `epochs` is a positive integer. `monitor` is `validation_total_loss` or `validation_reconstruction_loss`, as supported by that method. `gradient_clip_norm` is null or positive. Stage names and monitor semantics are bound by method review, not supplied as executable code.
3. `scheduler`, when present, contains exactly `name: cosine`, `warmup_epochs` (nonnegative integer), and `minimum_learning_rate` (nonnegative finite number), and `interval` (`batch` or `epoch`). Warmup must be shorter than the stage.
4. `update` contains exactly `steps` (positive integer), `optimizer`, and `gradient_clip_norm` (null or positive). Scientific masks, eligible parameters, and update order remain method-owned rules.
5. An `optimizer` contains `name`, `learning_rate`, and `weight_decay`. Supported names are `adam`, `adamw`, and `sgd`. Learning rate is positive; weight decay is nonnegative. Adam and AdamW additionally require `betas` (two values in `[0, 1)`) and positive `epsilon`. SGD additionally requires `momentum` in `[0, 1)`, nonnegative `dampening`, and boolean `nesterov`. Reject fields for another optimizer. Nesterov requires positive momentum and zero dampening.
6. All applicable method parameters below are required. They have no parser defaults. The default YAML profile states their numeric values explicitly. Later Phase 1 method bindings must select those values before the implementation gate passes.
7. A known method with an empty or incomplete scientific profile is not ready. Do not create dummy fit or update behavior.

### Closed method parameter fields

| Method or phase | Allowed parameter fields and types |
| --- | --- |
| MUSTARD offline | `classifier_hidden_size`: positive integer; `head_dropout`: `[0,1)`; `continuous`: mapping; `discrete`: mapping; `memory`: mapping; `loss_weights`: mapping; `label_correction`: mapping; `contrastive_temperature`: positive; `stochastic_inference`: boolean; `monte_carlo_samples`: positive integer; `variance_correction`: 0 or 1. |
| MUSTARD online | `projector_hidden_size`: positive integer; `projector_dropout`: `[0,1)`; `verification`: mapping or null; `contrastive_weight`: nonnegative; `contrastive_temperature`: positive. A0 forbids update settings and uses null verification. |
| RedLamp | `classifier_hidden_size`: positive integer; `classifier_dropout`: `[0,1)`; `label_correction`: mapping; `classification_weight`: `[0,1]`; `class_selection_probability`: `[0,1]`; `zero_range_epsilon`: positive. |
| STUMPY channel AB | `normalize`: boolean; `p`: finite number >= 1; `channel_aggregation`: fixed string `max`; `calibration_epsilon`: positive. |
| KMeansAD | `n_clusters`: positive integer; `n_init`: positive integer; `normalize_windows`: boolean; `normalization_epsilon`: positive. |
| Isolation Forest | `n_estimators`: positive integer; `max_samples`: `auto`, positive integer, or fraction in `(0,1]`; `max_features`: fraction in `(0,1]`; `contamination`: `auto` or fraction in `(0,0.5]`; `normalize_windows`: boolean; `normalization_epsilon`: positive. |
| CANDI | `latent_pooling`: fixed `mean_time`; `latent_normalization_epsilon`: positive; `covariance_rcond`: nonnegative; `similarity_quantile`: `(0,1)`; `minimum_samples`: positive integer; `sana`: mapping. |
| M2N2 | `trend_decay`: `[0,1)`; Its threshold source is owned by the evaluation profile, not repeated here. The zero-initialized trend and its later fitted state follow the reviewed method lifecycle; they are not a second input scaler. |
| MOMENT | `checkpoint`: nonempty local path or model identifier; `revision`: nonempty immutable revision; `channel_aggregation`: fixed `mean`; `patch_length`: positive integer. `zero_shot` has no training; `linear_probing` has an explicit training stage. |

### Closed nested fields

1. `continuous`: `enabled` (boolean), `num_prototypes` (positive integer), `temperature` (positive).
2. `discrete`: `enabled` (boolean), `codebook_size` (positive integer), `top_k` (positive integer, at most codebook size), `temperature` (positive). The target uses the reviewed sparse lookup; do not add an unreviewed retrieval branch through a string value.
3. `memory`: `normalization_epsilon` (positive), `initialization_batches` (positive integer), `freeze_after_initialization` (boolean).
4. `loss_weights`: exactly `reconstruction`, `classification`, `contrastive`, `diversity`, `variance`, `covariance`, `usage`, and `gate`, each nonnegative. O0/O1/O2 determine applicable terms; positive weights for a forbidden term are rejected.
5. `label_correction`: exactly `alpha` and `beta`, each nonnegative and satisfying the reviewed target-probability constraints for 12 classes. The correction formula must be checked separately; these bounds alone do not prove it correct.
6. `verification`: exactly `buffer_size` and `cycle_size` (positive integers, cycle at most buffer), `non_overlap_gap` (nonnegative integer), and `signature_top_k` (positive integer, at most continuous prototype count). The reference defines recurrence and anomaly-radius rules; do not invent additional controls.
7. `sana`: exactly `hidden_size`, `attention_heads`, `feedforward_size`, `layers` (positive integers), `dropout` in `[0,1)`, and finite `gate_initialization`. Hidden size must be divisible by attention heads. Use the reviewed temporal-convolution and attention mechanism; no alternate adapter hierarchy is added.
8. Shared CNN fields do not belong in a method profile. The experiment or standalone encoder mapping contains exactly `hidden_channels`, `output_dim`, `kernel_size`, `num_layers` (positive integers; layers at least two), and `dropout` in `[0,1)`. Input channel count comes from the prepared manifest. Effective window length comes from the data profile. Select the common numeric settings in Stage 1.2 step 3.
9. CANDI uses its confirmed time mean before latent normalization. MOMENT uses its confirmed channel mean of point reconstruction errors. Reconstruction itself retains its full time dimension.

## 5. Reporting profile fields

| Field | Type | Required or default | Rule |
| --- | --- | --- | --- |
| `phase` | string | Required | Exactly one of `offline` or `online`. |
| `splits` | nonempty list of distinct strings | Required | Selected from `train`, `validation`, and `test`. |
| `metrics` | nonempty list of distinct metric names | Required | Every name must exist in the shared metric definitions. |
| `precision` | mapping from metric names to integers | Default empty mapping | Rare display override. Each name must be selected by this profile. Digits must be in `[0, 15]`. |

1. The profile path is its identity. Do not repeat its filename in a second name field.
2. Generate one table per configured dataset. Keep exactly one phase in that table. Multiple splits are permitted.
3. Use rows for method variants. Use columns for split, entity, and metric. The profile changes the selected results, not the recorded run metrics.
4. Always write detailed and mean-only CSV files. Mean, sample standard deviation, and valid/expected count follow the confirmed reporting rules. Do not expose alternate estimators through this profile.
5. Shared metric definitions use one root mapping: each key is a supported metric name; its value is a mapping with exactly `precision`, an integer in `[0, 15]`. The file states every precision explicitly. Unknown metric names or fields are errors.
6. Source metric definitions from `configs/metrics.yaml`. Resolve a reporting profile from `configs/reporting_profiles/`. Record the profile reference and any effective precision override in report provenance.
7. Do not change thresholds, model state, or metric calculation through a reporting profile. Layout remains the agreed row and column arrangement; no layout engine is added.

## 6. Profile default sources

1. Dataset discovery uses one default file per dataset under `configs/data_profiles/`.
2. Method discovery uses the profile references below and the capability table from Stage 1.2 step 2. It is not a plugin system. Known scientific variants do not prove implementation readiness.
3. Experiment selection supplies shared CNN settings. Methods receive separate instances with the same architecture settings. The encoder mapping is held in the experiment or standalone definition, outside individual method profiles. Its exact fields are listed in section 4; numeric bindings belong to Stage 1.2 step 3.
4. Default choices must be visible in `show`. A missing required profile is an error, not a reason to guess another profile.
5. This field contract does not approve unresolved scientific values. The complete Phase 1 gate and its user review remain required.

## Terminology changes

No domain term changes. `method_window_sizes` is a new data-profile field. It selects method-specific windows from the same prepared observations. It does not create new datasets, new data profiles, or another input scaler.

## Contamination evidence

1. Unknown source labels remain unknown in training, calibration, and reports.
2. Record label counts and the normal-data assumption separately. A selected calibration window is eligible under a rule; it is not verified normal merely because it has no known anomaly.
3. Do not promise a test false-positive rate from a calibration quantile.
4. Keep the original-training-split exception. No new training filter or input transformation is authorized here.

## 8. Synthetic family fields

1. `spike`: optional positive `spike_scale_multiplier` (default 1) and optional positive integer `max_spikes` (default 1).
2. `flip` and `average`: optional `min_segment_fraction` and `max_segment_fraction`, supplied together, with the same fraction bounds as the parent profile. Omission uses parent values.
3. `speedup` and `scale`: required nonempty `factors` when that family has an override. Factors are positive finite numbers; reject a factor of one when it would make the selected transformation a no-op.
4. `cutoff`: optional `force_mode`, fixed `hold_or_zero` for the retained override.
5. `wander`: optional positive `drift_multiplier`, default 1.
6. `contextual`: optional positive `offset_multiplier`, default 1.
7. `upsidedown`: optional positive `visibility_multiplier`, default 1.
8. `mixture`: optional `keep_legacy_behavior`, fixed true for the retained profile. Review the source recipe before changing its meaning.
9. `noise` permits no override fields. Unknown family fields fail. Actual sampling and label masks require the scientific checks in Phase 4; this field list is not proof of augmentation parity.

## 9. Retained uncertainty field names

1. `point_anomaly_score_variance`.
2. `window_anomaly_score_variance`.
3. `continuous_retrieval_variance_point` and `continuous_retrieval_variance_window`.
4. `discrete_retrieval_variance_point` and `discrete_retrieval_variance_window`.
5. `reconstruction_variance_point`, `reconstruction_variance_window`, and `reconstruction_variance_full`.
6. `classification_probability_variance` and `classification_variance_mean`.

Method and metric review must define the sample axis, reductions, and missing-value behavior for each retained field before implementation. Do not substitute another statistic under an existing name. Do not retain raw Monte Carlo tensors by default.

## 10. Scope of this field contract

1. Field names, types, missing-field behavior, and narrow defaults are specified here. Required scientific parameter values must be explicit in supplied YAML, not hidden parser defaults.
2. Later Stage 1.2 steps bind capabilities, common heads, window behavior, metric semantics, recovery, and provider behavior. Required scientific values and method-specific applicability must be bound there before the Phase 1 gate passes.
3. If source review proves that a required input is missing, record and review the smallest schema change before implementation. Do not accept extra fields silently.
4. Refer to the named mathematics notes and current source in the execution review for evidence. This document changes no runtime files.

## 11. Exact profile references

All paths below are relative to `configs/`. These are proposed files, not claims that the files already exist.

1. Data defaults: `data_profiles/iccad.yaml`, `data_profiles/smd.yaml`, `data_profiles/iops.yaml`, `data_profiles/nasa.yaml`, and `data_profiles/swat.yaml`. Each selects an explicit prepared version. Never select a version by the lexical order of its hash.
2. Synthetic choices: `data_profiles/synthetic_profiles/redlamp_legacy.yaml` and `data_profiles/synthetic_profiles/redlamp_visible.yaml`.
3. MUSTARD offline: `method_profiles/mustard_o0_offline.yaml`, `method_profiles/mustard_o1_offline.yaml`, and `method_profiles/mustard_o2_offline.yaml`.
4. MUSTARD online: `method_profiles/mustard_a0_online.yaml`, `method_profiles/mustard_a1_online.yaml`, and `method_profiles/mustard_a2_online.yaml`.
5. Baselines: `method_profiles/baselines/redlamp.yaml`, `method_profiles/baselines/stumpy_channel_ab.yaml`, `method_profiles/baselines/kmeans_ad.yaml`, `method_profiles/baselines/iforest.yaml`, `method_profiles/baselines/candi.yaml`, and `method_profiles/baselines/m2n2.yaml`.
6. MOMENT: `method_profiles/baselines/moment.yaml` for zero-shot and `method_profiles/baselines/moment_linear_probing.yaml` for linear probing. Two explicit profiles avoid a second variant schema inside one profile.
7. New evaluation defaults: `evaluation_profiles/raw_input_q99_ewma075.yaml` for MUSTARD and `evaluation_profiles/native_q99_ewma075.yaml` for native baseline scores. Each supplies the required phase quantiles and explicit metric settings. An incompatible method/score-space pair fails validation.
8. The two MUSTARD q95 profiles are listed in the [scoring comparison](#mustard-validation-q95-comparison).
9. Retained explicit SMD evaluation choices: `evaluation_profiles/smd/cleanval_q99_ewma09.yaml`, `evaluation_profiles/smd/cleanval_q995_ewma09.yaml`, `evaluation_profiles/smd/synthnormal_q99_ewma09.yaml`, and `evaluation_profiles/smd/synthnormal_q99_normalized_input_mse_ewma09.yaml`. Their explicit 0.1/0.9 weights remain unchanged. The names are historical profile identifiers, not proof of clean data. For `synthnormal_q99_normalized_input_mse_ewma09.yaml`, use the saved validation split to generate synthetic calibration windows. Keep the selected checkpoint fixed. Record the generation settings and source split. Keep source-label eligibility separate from the generated anomaly mask. Use scores from unmasked points for the point q99 and scores from windows with no masked points for the window q99.
10. Reporting defaults: `reporting_profiles/offline_detection.yaml`, `reporting_profiles/online_detection.yaml`, and `reporting_profiles/uncertainty.yaml`. Each selects one phase. Add a separately named profile when another phase or selection is needed.
11. Shared display definitions: `metrics.yaml`. Experiment definitions live under `experiments/`; standalone definitions live under `standalones/`. They reference profiles rather than copy their scientific values.

The new `phases` list replaces the older singular method-profile `phase` field. Run `phase` remains a single value. This is a change in profile cardinality, not a rename of the offline or online phases. Record it in the signature migration review before implementation. The one-variant-per-profile rule remains unchanged.

## Confirmed gap policy

1. For timestamped sources with known cadence, use `gap_policy: break_at_time_gaps`. No window crosses a known gap. ICCAD's known interval is five minutes; SWaT's is one second.
2. Retain timestamp-free source order without a claim of verified elapsed-time continuity. Do not invent time steps to apply the gap rule.
3. Record the expected interval, gap locations, usable-window counts, and coverage. Reduced coverage is a limitation; it is not a reason to fill values or label unscored points as normal.
4. Tail and stride bindings are recorded in Stage 1.2 step 4. Online tails permit only `drop`. This field update changes no runtime code or prepared outputs.

## Metric-source review amendments

1. The retained synthetic-normal evaluation profiles need explicit point/window calibration sources. Added those fields rather than silently dropping their distinct behavior. Source labels remain separate from generated augmentation labels. An unmodified synthetic view does not prove that its source window was normal.
2. Retained the reviewed validation MAD-logistic score-transform kind alongside identity and median/IQR. Its exact zero-scale and fitting behavior requires known-value tests before readiness. No fallback transform is permitted.
3. Metric and uncertainty meanings, numerical tolerances, and the confirmed discontinuous-range choice are recorded in Stage 1.2 step 5 of the decision file.

## Segment summaries

1. The user selected length-weighted means of defined segment range metrics. Add `segment_mean_vus_pr`, `segment_mean_vus_roc`, `segment_mean_affiliation_f1`, and the corresponding prefixed budgeted VUS names to shared metric definitions.
2. Evaluation and reporting profiles select these exact names. Do not make a canonical metric name silently mean its segment average.
3. Segment and eligibility records belong in the metric result envelope. Scalar report values remain under its `metrics` mapping.

## 12. Explicit default profile values for review

These values are proposed implementation inputs. They are not tuned results or proof of paper parity. Named alternative profiles can change permitted fields. Do not add parser fallbacks for these required values.

1. Data values follow Stage 1.2 step 4: length 20, training stride 1, offline validation/test stride 20, online stride 1, batch size 256, and no training shuffle. Preserve the stated native-method, M2N2, and MOMENT stride overrides. Synthetic training uses `redlamp_legacy.yaml` by default when the method requires synthetic classes. Other methods use null.
2. Common Adam settings are betas `[0.9, 0.999]` and epsilon `1e-8`. A training stage has no scheduler and no gradient clipping unless the method row states them. Every applicable YAML states these values explicitly.
3. MUSTARD offline: classifier width 32, dropout 0.1, continuous retrieval enabled with 32 prototypes and temperature 0.9, discrete retrieval enabled with 60 codes, top-k 3, and temperature 0.1. Memory epsilon is `1e-6`, initialization uses 16 batches, and memories then freeze. Label correction uses alpha 0.1 and beta 0.01. Stochastic inference is true, with 10 samples and variance correction 1. Contrastive temperature is 0.1.
4. MUSTARD reconstruction and classification weights are 0.5 each. O2 adds contrastive weight 0.3; O0/O1 use zero contrastive weight. All five optional regularizer weights are zero. Reject a positive inapplicable gate weight under direct routing. For O1 in Stage A, use the mean of the classification loss and the balanced point-score loss when both point groups are present. If a group is missing, use the full classification loss for that batch. Mark the point-score loss as unavailable. See the [decision and limitation](../plan/decisions-runtime-replacement.md#mustard-o1-batches-with-a-missing-point-group). All three variants retain the same enabled retrieval branches.
5. MUSTARD uses 25 epochs in `stage_a` then five epochs in `stage_b`, for 30 total, with Adam learning rate `1e-3`, weight decay `1e-4`, and validation-total-loss monitoring. Preserve the documented stage freeze rules. These are explicit project defaults, not a universal training schedule.
6. MUSTARD online: projector width 64, dropout zero, contrastive temperature 0.1, and contrastive weight zero for A0/A1 or 0.1 for A2. A0 has null verification and update. A1/A2 verification uses buffer size 64, cycle size 8, non-overlap gap zero, and signature top-k 3. Retain the reviewed two-cycle expiry rule as fixed behavior. Updates use one Adam step, learning rate `1e-4`, weight decay zero, and clipping norm 1. The required offline variant is explicitly O0 by default.
7. RedLamp: classifier width 32, dropout 0.1, label correction alpha 0.1/beta 0.01, classification weight 0.1, class-selection probability 0.05, and zero-range epsilon `1e-8`. Fit for 30 epochs with Adam learning rate `1e-3`, weight decay zero, and validation-total-loss monitoring. Fit class selection and min/max scaling on eligible validation predictions only. Floor a zero denominator by epsilon; do not refit or clip test scores to a test-derived range. Shared point EWMA supplies causal smoothing.
8. KMeansAD: 20 clusters, 10 initializations, window normalization true, epsilon `1e-6`. Isolation Forest: 100 estimators, `max_samples: auto`, `max_features: 1.0`, `contamination: auto`, window normalization true, epsilon `1e-6`. STUMPY: normalization true, p 2, channel aggregation max, calibration epsilon `1e-8`. These native estimators use their fit capability without a neural epoch or optimizer stage; `training` is null for them.
9. CANDI: time-mean pooling, latent normalization epsilon `1e-12`, pseudoinverse relative tolerance `1e-6`, similarity quantile 0.05, and minimum samples 16. SANA uses the values in Stage 1.2 step 3. Offline fitting uses 30 epochs, Adam learning rate `1e-4`, weight decay `1e-4`, clipping 0.5, and validation-reconstruction-loss monitoring. Use an epoch cosine scheduler with five warmup epochs and minimum learning rate zero. Online uses one SGD step at `1e-4`, weight decay `1e-4`, momentum 0.9, dampening zero, Nesterov true, and clipping 0.5. These optimizer settings follow the inspected author configuration.
10. CANDI hard validation references are the highest `floor(n * (1 - q))` eligible validation window scores, where q is the selected window threshold quantile. Zero references means an empty set, not all windows. Use stable source-index tie order. Moderate references have scores strictly between validation Q1 and Q3, with linear percentiles. Preserve the reviewed matching and update masks; do not replace them with threshold-only adaptation.
11. M2N2: trend decay 0.99. Fit for 30 epochs with AdamW learning rate `1e-3`, weight decay zero, and validation-reconstruction-loss monitoring. Online uses one SGD step at `1e-3`, with weight decay, momentum, and dampening zero, Nesterov false, and no clipping. These values follow its author default YAML. Keep the reviewed zero-initialized causal trend and predicted-normal mask. Its adaptation threshold uses training calibration; this method-specific source rule does not permit test fitting.
12. MOMENT: use the immutable base revision in Stage 1.2 step 3, patch length 8, and mean channel error. Zero-shot has null training/update. Linear probing fits only its reconstruction head for 30 epochs with Adam learning rate `5e-5`, weight decay zero, and validation-reconstruction-loss monitoring. The earlier ten-epoch default is withdrawn. Both variants have null online update.
13. Neural method profiles that fit state must provide their training stages. Native estimator profiles and zero-shot MOMENT must use null training. This narrows the earlier blanket training requirement to methods with actual epoch-based fitting.
14. New evaluation defaults use identity score transform, threshold quantile 0.99 for each phase, normal-validation eligibility, EWMA 0.25/0.75, VUS buffer 20 and 200 thresholds, and budgets `[0.001, 0.005, 0.01]`. Pre-test calibration normally uses validation. M2N2's native adaptation threshold uses its explicit training-source exception, separate from any reported evaluation threshold. MUSTARD triage uses window 0.99 and latent low/high quantiles 0.95/0.99.
15. Detection defaults request `segment_mean_vus_pr`, `segment_mean_vus_roc`, `segment_mean_affiliation_f1`, and `fpr`. Here `fpr` is the unadjusted false-positive rate; no metric field is renamed. The uncertainty profile selects the retained supported uncertainty fields and statistics `mean`, `std`, `min`, `p50`, `p95`, `max`, with validation and test splits in one offline table. Unavailable method statistics remain N/A. Shared display precision is four decimal places for all supplied metrics, including uncertainty summaries; it does not round stored values.
16. The retained explicit SMD profiles keep their original quantiles, score spaces, transforms, sources, and 0.1/0.9 EWMA pair. Translate their settings field by field; reject a source setting that has no reviewed equivalent. Do not inherit the new generic defaults over an explicit old value.

Evidence: current MUSTARD and RedLamp YAML; the inspected CANDI `config.py`; M2N2 `cfgs/train_defaults.yaml` and `cfgs/test_defaults.yaml`; the method mathematics notes; and the ordered source review. New project defaults above must be visible in the final contract review. Method implementation still requires the planned scientific tests.

## 13. User amendments after the walkthrough

1. Save one nullable score column in selected split order. Save alignment and provenance once in shared metadata. Do not persist per-point identities, indices, timestamps, segment IDs, or availability/finality columns. Runtime arrays still provide alignment.
2. Preserve null positions. Recover point positions, source rows, timestamps, gaps, and window membership from checked prepared data and the recorded schedule. Recovery saves ordered continuation values and sufficient cursor state.
3. Withdraw the unequal method epoch counts in section 12. The confirmed common total is 30: MUSTARD uses 25 + 5; RedLamp, CANDI, M2N2, and MOMENT linear probing each use 30. Those earlier unequal counts are not active defaults. Epoch rules apply only to epoch-based offline fitting.
4. These amendments supersede conflicting earlier storage and epoch statements. Prepared data and metric definitions remain unchanged.

## 14. Confirmed epoch budget

The 30-epoch comparison budget supersedes author-specific epoch defaults. Source attribution in section 12 applies to the retained optimizer and method settings, not to the replaced epoch totals.

1. The user selected 30 total offline epochs per fitted neural method on 2026-11-17. Count all stages together. MUSTARD uses 25 + 5; RedLamp, CANDI, M2N2, and MOMENT linear probing each use one 30-epoch stage.
2. Keep `training.stages[].epochs` in each method profile. Do not add a second epoch field to experiment definitions. Run/evaluation validation checks the sum across stages and equal totals across selected fitted neural methods. Supplied normal profiles explicitly total 30.
3. Native estimators and zero-shot MOMENT have no epoch-based stage. Online adaptation steps remain method-specific. A common epoch count does not mean equal compute or equal pretraining exposure.
4. Smoke execution retains its explicit reduced limits and separate output root. It is not the fair comparison. Resume continues the recorded budget and completed-epoch cursor; it does not add another 30 epochs.
5. This resolves the epoch total, stage allocation, and field ownership. The user authorized execution of this updated contract on 2026-11-18.
