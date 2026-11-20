# Runtime replacement decisions

Date: 2026-11-16
Status: confirmed requirements for the installed thesis runtime. See the [execution record](../detail/runtime-replacement-review.md) for verification and limits.

## 1. Data and test access

1. Reuse saved prepared Parquet files, scaler states, and manifests. Runtime loading checks their identity and creates windows. It must not read raw sources, split observations again, fill missing values, or fit another input scaler.
2. Offline and online runs use the same selected prepared version and scaler identity. Keep the self-contained preprocessing script separate from runtime loading.
3. Fit models and input transforms on training data. Use approved training or validation sources for calibration and validation for state selection.
4. Test values may be used for scoring and causal, unlabeled adaptation. Score and record each arriving window before an update. Changes can affect later scores only.
5. Do not use future test observations, complete-test statistics, test-based tuning, or test labels outside metric calculation. Methods and adaptation diagnostics must not receive test labels.
6. A window may use observations available at its scoring time. A time-preserving convolution within that arrived window does not permit a prediction to use observations after that time. The 2026-11-17 decision below defines last-estimate reporting and its declared delay. Remaining tail and gap rules still need review.

## 2. Encoder and method behavior

1. Use one shared time-preserving CNN implementation for MUSTARD, RedLamp, CANDI, and M2N2. Its sequence output has shape `[batch, time, hidden]`. Add method-specific pooling where needed.
2. Each method owns a separate encoder instance, learned weights, heads, and update state. Share architecture and settings within an experiment, not a mutable model or trained weights. Equal initialization is not required by this decision.
3. Keep method-specific losses, sample selection, heads, and update rules. CANDI's internal SANA network remains method-specific.
4. Stay as close as possible to reviewed papers and author code within these requirements. Do not preserve a thesis adaptation unchanged when it conflicts with this target. Review each method before replacement and ask about unresolved scientific conflicts.
5. The common CNN changes the inspected author architectures. Document each departure and test the retained calculations. Structural compatibility does not establish numerical parity or equal detection performance.
6. RedLamp uses validation predictions to fix class selection and score scaling before test scoring. Use causal smoothing. Preserve reviewed multiclass training. Exact fitting, zero-range handling, and smoothing settings require method review.
7. MOMENT retains its pretrained encoder and zero-shot and linear-probing profiles. Neither profile performs online parameter adaptation. Traditional methods need no neural encoder.

## 3. Replacement and commands

1. Replace the full runtime in stages. Remove each old implementation only after its replacement passes the required behavior and scientific checks.
2. Use new commands and checkpoints. Do not add old-command wrappers or require new runtime readers for old checkpoints. Preserve historical results as evidence; do not delete them under this decision.
3. Keep workflow logic in reusable functions and learned state in method classes. Keep functions within 50 lines and code files within 500 lines. Avoid duplicated runner, shell, and CLI workflows.
4. Measure source, scripts, tests, and configurations separately. Do not count removed scientific scope or minified code as simplification.
5. Preserve `scripts/preprocess.py` and `scripts/cleandocs_hook.py`. The CLI forwards preprocessing arguments to the existing implementation.
6. Validate supported method–dataset–phase combinations before execution. Stop and list unsupported combinations; do not silently skip them.

## 4. Smoke execution and results

1. Smoke execution selects the first configured dataset, entity, applicable variant, and seed for each selected method and supported phase. Include required offline dependencies.
2. Apply explicit work limits without changing prepared data. Exclude smoke results from normal reports. Exact limits and output isolation still need specification.
3. Keep the agreed hierarchy: `<experiment_id>/<phase>/<dataset>/<entity>/<method>/<variant>/<seed>/<split>/<run_id>/`.
4. Omit single-value dimensions as specified in the configuration plan. Read recorded dimensions; do not infer meaning from folder position alone. Keep split and execution identity.
5. This hierarchy supersedes the older six-level maximum and earlier method/seed/phase folder order for the next thesis version. Do not migrate existing results in this documentation update.

## 5. Evidence and remaining work

1. [Runtime signatures](../detail/proposed-runtime-signatures.md), [configuration plan](../../2026-11-09/plan/plan-config-signatures-and-fields.md), and [desired CLI](../../../specs/desired-cli-20261109.md) define the related design boundaries.
2. CANDI and M2N2 author MLP paths share their encoder/decoder structure: [CANDI](https://github.com/kimanki/CANDI/blob/main/models/mlp/modeling_mlp.py), [M2N2](https://github.com/carrtesy/M2N2/blob/master/models/MLP.py). The selected common CNN is a documented architecture change.
3. [RedLamp scoring](https://github.com/KoheiObata/RedLamp/blob/main/main.py) uses complete-sequence statistics. The validation-fixed, causal replacement is a project adaptation.
4. [CANDI adaptation](https://github.com/kimanki/CANDI/blob/main/tta/candi/adapter_candi.py) receives test labels for diagnostic counts. Remove that access from the next thesis implementation.
5. The online reference sources are mutable. Pin source revisions during method review. These notes record inspection, not runtime parity.
6. Complete method schemas, pooling, point alignment, window gap/tail rules, metric inputs, smoke limits, artifact filenames, and recovery details remain open. Do not infer them from an interface sketch.

## 6. Terminology changes

No domain terms changed. Shared encoder means a common implementation, architecture, and settings with separate method-owned state. Protocol retains its agreed meaning: related input and output validation rules. These decisions apply to the next thesis version; they do not impose a common encoder on all tsad-lib methods.

## Online point smoothing: 2026-11-17

Status: user confirmed point-wise smoothing and last-estimate reporting. Use the suggested weights as configurable next-version defaults.

1. Calculate an anomaly score for every point in an arriving window before adaptation.
2. Apply EWMA separately to each point across its scores from overlapping windows.
3. A point with no previous score keeps its raw score. Thus, the last point of a newly advanced window has the same raw and smoothed score when it first appears.
4. Do not average scores from different point indices through this rule.

The current `src/engine/online_tta/point_ewma.py` implements this recurrence for point `i` at window step `k`:

```text
E[k, i] = S[k, i]                                  if no earlier score exists
E[k, i] = previous_weight * E[k-1, i]
          + current_weight * S[k, i]               otherwise
```

Here, `S` is the current raw score and `E` is the smoothed score. The weights sum to one. The default `previous_weight` is `0.25` and the default `current_weight` is `0.75`. An evaluation profile may select another valid pair. Record the resolved pair in calibration and result identity. Explicit historical `0.1`/`0.9` profiles keep their stated weights; do not rename their meaning silently. This is smoothing across window appearances of the same point.

The earlier A/B question was incomplete: overlapping windows can supply all point scores. Do not treat the reply as approval of full blocks without overlap or endpoint-only method output.

### Final reporting score

1. Use a point's last EWMA estimate from the last processed window that contains that point.
2. Record the source point index and the window endpoint at which that estimate became available. This is a delayed score. Do not describe it as a prediction available at the point's original arrival time.
3. Earlier window records retain their pre-update scores. A later window can revise a pending point estimate, but cannot change an earlier window event or its adaptation decision.
4. Finalize a point only after no later selected window can contain it. At stream completion, finalize remaining pending points using their last available estimates. Record whether completion was caused by the natural stream end or an explicit work limit.
5. Keep finalized scores fixed. Resume restores pending EWMA values, cursor, method state, and the score writer state together.
6. The final last point has no earlier score when first observed, so its first EWMA estimate equals its raw score. If it appears in a later window, the same per-point recurrence applies.
7. These rules apply to methods that return scores for those point indices. Do not broadcast a native endpoint score to other points to create smoothing contributions.

This reply closes the first score-reporting question. Window tails, gaps, complete method schemas, and other open Phase 1 choices remain separate work. See the [execution review](../detail/runtime-replacement-review.md).

## Data-profile decision: MOMENT window length

Status: option A confirmed by the user. Stage 1.2, step 1 continues.

1. Current SMD profiles use 20 observations per window. The reviewed MOMENT anomaly experiment uses 512 observations. This is a reference setup, not a claim that all MOMENT code rejects shorter input.
2. The existing `src/adapters/moment.py` pads shorter windows and supplies an observation mask. It returns embeddings. It is not evidence of a verified replacement anomaly detector.
3. The configuration plan gives window length to the data profile and requires one data profile per dataset. The replacement must state whether that profile can supply a different window length for MOMENT.
4. Option A: keep the dataset default and permit an explicit MOMENT window length of 512 in the same data profile. This keeps prepared data shared and changes only window creation. It adds a method-specific field and makes context lengths differ.
5. Option B: use 512 observed points for all methods. This uses one length but changes the current shorter context and increases work for other methods.
6. Option C: keep one dataset window length and pad MOMENT input to 512. This needs masks and a separate scoring check. It departs from the reviewed 512-observation setup.

The user selected option A. Keep `window_size` as the dataset default. Add `method_window_sizes`, a mapping from known method names to positive integer lengths, in the same data profile. The default mapping is `{moment: 512}`. Unlisted methods use `window_size`. A cell cannot override either field. Record the effective length in run, calibration, and checkpoint identity. Reject unknown method names. Window creation changes; prepared observations and their scaler do not change.

Evidence: the current SMD data YAML files, `src/adapters/moment.py`, and the library [MOMENT source review](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/moment.md>).

## Method review rule: scientific intent

Status: confirmed by the user on 2026-11-17. Neither the paper nor author code wins automatically.

1. Identify the scientific aim and the quantities that the method must preserve.
2. Compare the paper equations, surrounding explanation, author code, configuration, and available corrections.
3. Check whether a difference changes the method or only its parameter form. Use small numerical checks where applicable.
4. Record the chosen behavior, supporting evidence, and remaining uncertainty. Keep project-specific departures explicit.
5. Do not infer author intent from a name or one line alone. Ask the user only when the evidence leaves a consequential scientific choice unresolved.
6. Confirmed project rules still apply. Do not copy test-label access, future-input use, or complete-test fitting from a reference.

### CANDI output correction

1. The [paper, equations 12–18](https://arxiv.org/html/2604.01845v1#S3.SS3) describes learned input and output corrections, with only adaptation modules trained. The pretrained backbone remains frozen.
2. Equation 17 adds the gated output correction. The [author code](https://github.com/kimanki/CANDI/blob/main/tta/candi/adapter_candi.py) subtracts it. The gate is unconstrained and passes through `tanh`.
3. Algebraic inference: replacing `g_paper` with `-g_code` gives the same output correction. This follows from `tanh(-g) = -tanh(g)`. The two signs do not alone establish a different method.
4. Keep the author's subtract form for implementation. Preserve the signed gate and selective adaptation. Do not add a sign-selection field or a second implementation path.
5. Small scalar and PyTorch checks matched outputs and mapped gradients at gates `0`, `0.3`, and `-0.7`, using float64 and absolute tolerance `1e-12`. These checks do not prove complete training, optimizer, or checkpoint parity. Existing checkpoints cannot be assumed compatible without the gate mapping and full state checks.
6. The source-priority question is closed by this scientific-intent rule. Other differences, including reduction and calibration conventions, still need their own evidence checks.

The online code branch is mutable. Pin its revision before scientific implementation and reference comparison, as required by the plan.

## Calibration decision: unknown validation labels

Status: option A confirmed by the user. This settles calibration-label eligibility in Stage 1.2, step 1.

The following read-only check used one entity per dataset, not every entity. It counted stored validation labels with PyArrow. Missing labels were not changed.

| Dataset | Entity | Validation rows | Unknown | Normal | Anomalous |
| --- | --- | ---: | ---: | ---: | ---: |
| SMD | `machine-1-1` | 5,695 | 5,695 | 0 | 0 |
| IOPS | `KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa` | 29,251 | 0 | 28,781 | 470 |
| NASA | `A-1` | 576 | 576 | 0 | 0 |
| ICCAD | `console` | 28,375 | 0 | 27,506 | 869 |
| SWaT | `swat` | 99,000 | 0 | 99,000 | 0 |

The SWaT check used version `83957f47be5f7b142ae0bec77a64aa43c306e6e258ef7bfa176f2e46e0daee06`. It does not select that version as the runtime default. Other version identities are recorded in the execution review.

1. Option A: exclude calibration windows with a known anomaly. Permit unknown labels under an explicit normal-data assumption. Keep unknown labels unchanged. Record selected-window counts and the assumption. This permits calibration on the inspected SMD and NASA examples, but does not prove they are anomaly-free.
2. Option B: require every calibration point to have a known normal label. Stop if no eligible windows remain. This is stricter, but blocks the inspected SMD and NASA examples.

The user selected option A. For calibration that needs normal validation data, reject a whole window if any source label in it is a known anomaly. Allow unknown labels under the recorded normal-data assumption. Do not remove individual points to join the remaining observations into a new window. Save eligible, excluded, and unknown-label window counts in the calibration record. Stop if no eligible windows remain. Do not change prepared splits or fit another input scaler. Test labels never affect calibration. Metric eligibility remains separate: an unknown label is never a normal label for metric calculation.

## Profile contract work

The [working profile contract](../detail/runtime-profile-contract.md) records settled field names, types, defaults, and reference rules for Stage 1.2, step 1. Its remaining sections are explicitly incomplete. It is not an approved runtime schema.

## Method-profile decisions: new reductions

Status: both mean reductions confirmed by the user. These are new project adaptations, not paper/code precedence questions. They affect method fields in Stage 1.2, step 1.

1. MOMENT: choose how channel reconstruction errors become one point score. A mean matches shared reconstruction scoring but can dilute a single-channel anomaly. A maximum retains that signal but can be dominated by one noisy channel. Both agree for one channel. The method source review explicitly leaves this multivariate choice open.
2. CANDI: choose how `[time, hidden]` CNN output becomes a window latent vector for selection. A time mean keeps `hidden` dimensions and a smaller covariance matrix. Flattening keeps `time * hidden` dimensions and position detail, with a larger covariance matrix. The original MLP produces one window vector; replacing it with the common CNN creates this choice.

The user selected a mean for each reduction. MOMENT averages channel reconstruction errors at each point. CANDI averages CNN features over time, then applies the reviewed latent normalization before reference statistics and sample selection. Keep reconstruction output at its full time resolution. CANDI's latent reduction must not change the reconstruction target shape.

The [working profile contract](../detail/runtime-profile-contract.md) now also states the reporting-profile fields and shared precision schema. All method fields are still incomplete. No execution checkbox is newly complete.

## Data contamination: explicit limitation

Status: confirmed user concern, recorded on 2026-11-17. This does not authorize another preprocessing pass or another anomaly filter.

1. Unlabeled training and validation observations can contain anomalies. Unknown labels do not establish normality.
2. Excluding known anomalies does not prove that the remaining observations are normal. State the assumption when a method needs normal training or calibration data.
3. Preserve the confirmed preprocessing rules and the original-training-split exception. Runtime loading does not remove more training rows or change saved splits.
4. Keep source labels unchanged. Record known-normal, known-anomaly, and unknown-label counts separately. Record the calibration-window selection and its counts.
5. A calibration quantile describes its calibration scores. It does not guarantee the same false-positive rate on test data, especially under contamination or distribution shift.
6. Report possible effects on fitted weights, validation checkpoint selection, thresholds, and CANDI reference statistics. Do not call an unknown-label set verified clean data.

## Stage 1.2 step 1: profile field contract

1. The [profile field contract](../detail/runtime-profile-contract.md) specifies the four profile types, exact method parameter fields, required values, parser defaults, reference paths, and narrow overrides.
2. Required method numeric values have no parser default. Default YAML files must state them explicitly. Later Phase 1 steps select applicable values and scientific policies before the implementation gate.
3. Data profiles own all window construction fields. Evaluation profiles do not duplicate length, stride, tail, or gap fields. Limits remain run-specific.
4. One baseline profile can declare supported `phases`, avoiding copies that differ only by phase. A run still has one phase. Each profile still selects one variant. Two MOMENT profiles retain both requested variants.
5. Add two explicitly named 0.75-weight evaluation defaults. Retain the old explicit 0.9-weight choices. Do not change an old profile's meaning under its existing name.
6. These are implementation design bindings, subject to the completed Phase 1 contract review. They do not claim current runtime support or authorize bypassing later scientific checks.

## Stage 1.2 step 2: method capabilities

These are target capabilities, not passed readiness checks. Every row applies to ICCAD, SMD, IOPS, NASA, and SWaT when the per-entity input checks pass. Method accuracy outside an author's evaluated datasets is not established by this table.

| Method | Variants | Offline behavior | Online behavior | Required online source |
| --- | --- | --- | --- | --- |
| MUSTARD | Offline O0, O1, O2; online A0, A1, A2 | Fit the two reviewed stages, select state, and calibrate | A0 scores with fixed weights; A1/A2 permit reviewed projector updates | Selected successful compatible offline O variant and its artifacts |
| RedLamp | `default` | Fit reviewed multiclass training; fix class selection and score scaling from validation | Frozen model with causal scoring | Matching offline `default` state and calibration |
| STUMPY channel AB | `default` | Save training reference and fit validation channel/score calibration | Frozen training reference; no parameter updates | Matching offline `default` reference and calibration |
| KMeansAD | `default` | Fit training clusters and approved calibration | Frozen clusters; no parameter updates | Matching offline `default` clusters and calibration |
| Isolation Forest | `default` | Fit training estimator and approved calibration | Frozen estimator; no parameter updates | Matching offline `default` estimator and calibration |
| CANDI | `default` | Fit the common-CNN reconstruction backbone and validation reference statistics | Freeze backbone; update only selected SANA modules under the reviewed rules | Matching offline `default` backbone, covariance, references, and calibration |
| M2N2 | `default` | Fit the common-CNN reconstruction model and method threshold/state | Causal detrending and predicted-normal parameter updates | Matching offline `default` model and method calibration/state |
| MOMENT | `zero_shot`, `linear_probing` | Zero-shot loads immutable pretrained state without fitting; linear probing fits only its reconstruction head | Both score with fixed learned parameters; neither adapts online | Zero-shot loads pinned pretrained state and calibrates on the approved pre-test source; probing loads its matching offline head and calibration |

1. Default discovery retains all documented MUSTARD O/A variants and both MOMENT variants. Other methods use `default`. Smoke still selects only the first configured applicable variant per method and phase.
2. A default MUSTARD online profile explicitly requests offline O0. The user can select another required offline variant in run configuration. O/A labels are not matched by digit. Every O0/O1/O2 parent and A0/A1/A2 child combination is a target only when the requested checkpoint and reviewed update contract are compatible.
3. Online MUSTARD display identity includes both choices, such as `O0_A1`. Store `offline_variant` and `online_variant` separately in metadata. Never average O0_A1 with O1_A1 as the same reporting variant.
4. Require unchanged feature order and the same prepared version and scaler across an offline parent and its child. Effective window length and method architecture must also match. A method-specific length remains inside the dataset's single data profile.
5. Require at least one usable training and validation window for a fitted/calibrated method, at least one test window for detection, and at least two eligible validation latent vectors for CANDI covariance. Check other actual numerical prerequisites explicitly. A zero-shot method does not require a dummy training stage.
6. MOMENT requires 512 observed points per selected window under the confirmed default. Its reviewed patch length is 8. A selected immutable pretrained checkpoint and its required model dependency must be available before it is ready.
7. Unknown calibration labels are permitted only under the recorded assumption. Original-training contamination follows the confirmed preprocessing exception. No method receives test labels or uses them to establish readiness.
8. Empty CANDI reference sets cannot produce an eligible match. Record that condition; do not invent zero-distance matches. Singular covariance uses the reviewed pseudoinverse with explicit tolerance. No selected samples means no adaptation, not a fabricated successful update.
9. Native window-level traditional scores retain endpoint indices. Do not broadcast them across a window to create point scores. Their coverage differs from a method that supplies per-point reconstruction errors. Record coverage; metric eligibility must not hide that difference.
10. Stop validation for unsupported methods, unknown variants, incompatible phase/profile selections, missing artifacts, too-short entities, or missing required numerical inputs. List the failures. Do not silently skip a configured combination.
11. A capability declaration is not evidence of paper parity, sufficient anomaly coverage, statistical power, or successful execution. Prove each method through the later method and smoke gates.

Evidence: the eight method mathematics notes, the inspected thesis baseline and MUSTARD sources, the confirmed replacement decisions, and the 2026-11-17 source/shape reviews. Exact source revisions and complete numerical checks remain required by Phase 5.

## Stage 1.2 step 3: shared CNN and method heads

These are next-version design bindings. They require the later shape, gradient, state, and scientific checks. They are not numerical parity claims.

### Common CNN

1. Default settings: `hidden_channels: 64`, `output_dim: 64`, `kernel_size: 3`, `num_layers: 3`, `dropout: 0.1`. These match the inspected simple-CNN configuration. The experiment states the mapping explicitly; a standalone states the same mapping when applicable.
2. Input is `[B, L, C]`; output is `[B, L, H]`, with default `H = 64`. Each layer preserves `L`. Use zero padding, Conv1d, and ReLU; dropout follows each hidden layer, not the final output layer. The inspected `SimpleWindowCnnEncoder` supplies this construction.
3. Use separate instances and weights in MUSTARD, RedLamp, CANDI, and M2N2. No method profile can override these shared settings. Reset and record run random state; equal learned weights are not required.
4. Method tensor operations use float32 by default. Loading preserves prepared arrays and scaler bytes; conversion for model computation is a separate, recorded precision choice.
5. The arrived input window can use its available observations. This architecture does not make an earlier point prediction available before that window arrived. Record delayed score availability under the selected reporting rule.

### Head and representation shapes

| Method | Representation and head |
| --- | --- |
| MUSTARD | Keep point features `[B,L,H]`. Direct routing sends continuous retrieved features to reconstruction and discrete retrieved features to classification. Reconstruction uses a point-wise three-linear-layer MLP `H -> H -> H -> C`. Classification flattens the window and uses `L*H -> 32 -> 32 -> 12`. Hidden layers use ReLU and configured dropout; output layers are linear. |
| RedLamp | Mean over the encoder time axis gives `[B,H]`, preserving a single window vector for the author-style classifier. Use `H -> 32 -> 12` with hidden BatchNorm, ReLU, and dropout. Produce logits for the reviewed corrected cross-entropy objective; probabilities are a scoring output. Reconstruction uses a point-wise three-linear-layer MLP `H -> H -> H -> C`. |
| CANDI | The confirmed time mean gives `[B,H]`. Use that window bottleneck for reconstruction and normalize it for FPM statistics and selection. Preserve the reference decoder shape `H -> (L*C)//4 -> (L*C)//2 -> L*C`, then reshape to `[B,L,C]`; use ReLU only on hidden layers. Require `L*C >= 4`. |
| M2N2 | Use the same CNN, time-mean bottleneck, and window decoder structure as CANDI, with separate weights and method-owned detrending state. Keep point reconstruction errors for the normal mask. |
| MOMENT | Keep the pinned model's patch embedding, encoder, and reconstruction head. Input/output time length is 512 under the confirmed default; patch length is 8. Preserve channel-independent processing and average point errors across channels. Linear probing changes only the reconstruction-head parameters. |
| Traditional methods | No neural encoder or neural head is created. Keep the reviewed native input representation and score orientation. |

1. The mean for RedLamp replaces its author's learned compressed window representation after the confirmed CNN change. It is a project architecture adaptation, not a claim that mean pooling equals the original encoder. CANDI and M2N2 keep their original window-decoder width pattern after the changed encoder.
2. RedLamp's author loop skips a singleton training batch because of classifier BatchNorm. Preserve that boundary explicitly and record the skipped-window count. Do not remove its underlying prepared observations. Evaluation uses fixed BatchNorm statistics and can score one window.
3. MUSTARD retains the inspected window-flattened classification head. Its direct-routing input dimension is `L*H`; it does not retain an unused fusion projection. Use `classifier_hidden_size` for that head width. Reconstruction hidden width follows common `H`.
4. CANDI SANA stays separate: default hidden size 512, eight attention heads, feed-forward size 512, one layer, zero dropout, and zero-initialized signed gates. Keep the reviewed temporal convolution, inter-channel attention, LayerNorm, GELU, and per-channel output projection. Backbone freezing does not freeze SANA.
5. Keep MUSTARD stochastic-inference enablement explicit alongside Monte Carlo sample count and variance correction. Source review found that omitting this input from the initial field list would hide a scientific choice. Add the boolean `stochastic_inference` to its offline parameters before implementation. Review stochastic retrieval and uncertainty calculations in their owning scientific checks.
6. Proposed MOMENT default family: `AutonLab/MOMENT-1-base`, using immutable revision `b0ae5751d8ef43d72ad48fb5128e2ddc93c94b53`, whose published configuration states length 512, patch length 8, and hidden width 768. Both variants must start from the same selected pretrained revision. This does not establish equivalence to every original-paper experiment. Weight availability, file hashes, model compatibility, and licensing remain mandatory readiness checks; no weights were downloaded here.
7. The replacement uses raw logits for the corrected cross-entropy formula. RedLamp author code applies CrossEntropyLoss after a classifier softmax. The paper's probability-based cross-entropy objective supports computing log-softmax once on logits. Record this scientific-intent correction and test the stated loss; do not claim identical author-code gradients.

Evidence: `src/models/neural_blocks.py`, the current simple-CNN and MUSTARD head construction, the reviewed CANDI/M2N2 MLP shape, RedLamp author classifier, and [MOMENT's immutable configuration](https://huggingface.co/AutonLab/MOMENT-1-base/blob/b0ae5751d8ef43d72ad48fb5128e2ddc93c94b53/config.json). Further departure review remains required in Phase 5.

## Stage 1.2 step 4: window-policy review

Status: resumed on 2026-11-17. Gap policy is confirmed. The complete window bindings below finish this step.

Read-only checks of the saved timestamp columns found the following. A jump is the difference between consecutive observations; it is not a new observation. These counts do not alter the prepared data.

| Dataset and split | Expected interval | Larger jumps | Largest jump |
| --- | --- | ---: | --- |
| ICCAD train | 5 minutes | 2 | 10 minutes |
| ICCAD validation | 5 minutes | 10 | 15 minutes |
| ICCAD test | 5 minutes | 1 | 10 minutes |
| SWaT train | 1 second | 0 | 1 second |
| SWaT validation | 1 second | 0 | 1 second |
| SWaT test | 1 second | 1 | 82 seconds |

1. ICCAD checks used its sole saved Console entity. SWaT checks used the explicitly checked `67e1e175...` version. No backward timestamps or duplicate timestamps were found in these six split files. This is not an audit of every dataset's timestamp behavior.
2. Option A: split the window schedule at a known time gap. Keep saved observations unchanged. This avoids a window joining nonadjacent time steps, but reduces available windows near a gap.
3. Option B: create windows from consecutive saved rows even across gaps. Keep time metadata and record the limitation. This gives more windows but does not preserve regular elapsed time within every window.
4. The user confirmed option A: break windows at known time gaps in timestamped sources with known cadence. Do not fill the gaps or join observations across them. Sources without timestamps retain source order; their elapsed-time continuity remains unverified.
5. The user later requested a recap and continued implementation. That request resumes execution. The following bindings complete this checklist step; they do not pass the whole Phase 1 gate.

### Real-world challenge: time gaps

1. Real-world recordings can have jumps in their timestamps. Data ordered by time is not necessarily sampled at regular intervals.
2. A window that crosses a known gap can treat observations separated by missing time steps as adjacent regular samples. The selected policy prevents that window construction.
3. Breaking windows does not recover missing observations. It can reduce training windows, calibration windows, and score coverage. A segment shorter than the required window length supplies no full window.
4. Record gap locations, the declared expected interval, usable-window counts, and score coverage. Do not report an unscored point as normal or give it a valid zero score.
5. Preserve original timestamps, source-row mapping, and saved observations. Do not invent timestamps, interpolate values, or refit the saved input scaler.
6. The observed jumps do not identify their cause. Do not describe them as a verified sensor failure or recording outage without source evidence.
7. Gap handling does not prove that the remaining data is normal or fully regular. Keep the contamination assumption and timestamp-quality checks separate.

### Complete window bindings

1. Dataset default length is 20. The confirmed MOMENT override is 512. Training stride is 1, offline validation/test stride is 20, and online stride is 1 unless the named method override below applies. Batch size is 256 and default training shuffle is false, matching the inspected thesis data configuration. No evaluation or online shuffle is permitted.
2. STUMPY, KMeansAD, and Isolation Forest use validation/test/online stride 1 for native endpoint scores. KMeansAD and Isolation Forest use training stride 20, retaining their reference training-window representation. STUMPY keeps its training sequence as the fixed join reference. This changes the old non-overlap evaluation schedule: record that departure rather than copying a window score onto every point.
3. M2N2 online stride equals its effective window length, preserving the original non-overlap adaptation setup. Default length/stride is 20. Reject an overlapping M2N2 online profile until a separately reviewed scientific contract allows it.
4. MOMENT uses stride 512 in all four contexts. Do not downsample or pad prepared observations to reproduce another source's preprocessing. Those would change the confirmed prepared-data boundary.
5. Train tails use `drop`; offline validation/test tails use `end_align`; online tails use `drop`. For an offline segment `[a,b)` of length at least `L`, regular starts are `a + k*stride` while the whole window fits. Under `end_align`, append `b-L` only if it is not already a start. No partial window is padded. The online end-aligned option is not supported in this version because it would require another commit policy.
6. A timestamp jump larger than the declared expected interval starts a new window segment. Use five minutes for ICCAD and one second for SWaT. Add `expected_interval_seconds` to the data profile: a positive finite value for these timestamped inputs, and null for timestamp-free inputs. Do not estimate cadence from complete-test values or invent timestamps. Keep duplicate observations and report duplicate timestamps separately; absence of a large gap alone does not prove full regularity.
7. Generate windows separately inside each prepared split and each gap-separated segment. A segment shorter than `L` supplies no window. Retain its observations and record uncovered points. Stop a run if a required split has no usable windows; do not silently skip a configured entity.
8. Stream range fields `limits.stream_start` and `limits.stream_end` are paired integer bounds into the entity's complete saved test split, using `[start,end)`. They are not original CSV row numbers and not a concatenation of train, validation, and test. The inspected old online caller also applies its absolute range to the test sequence. After range selection, batch indices retain their positions in the complete saved split.
9. Keep `source_file` and `source_row` mapping separate from these ordered split indices. They can differ after source sorting. A test point's identity is dataset, prepared version, entity, split, and split index.
10. Offline per-point reconstruction scores use the arithmetic mean over aligned windows covering that point. Native endpoint scores remain indexed only at their endpoints. Do not fill unscored points. Online scores use the confirmed per-point EWMA and last-estimate rule.
11. In ordinary online processing, record a window's score before its parameter update. A window can revise pending estimates for its own point indices; it cannot change an earlier window event. Keep `scored_at_index` as the latest observation used for that estimate.
12. Finalize a pending point when the next regular window start exceeds its index, or when the segment or selected stream ends. Record `finalized_at_index` and `finalization_event` separately from `scored_at_index`. At a detected gap, finalization can be known only when the next observation arrives. At EOF, use an explicit end event rather than inventing a timestamp.
13. An explicit work limit ends a bounded run and records `work_limit`. Flush its remaining pending estimates with their last available values. An interruption does not finalize them; keep them pending for resume. Never present a delayed final score as an immediate prediction at the source point's arrival.
14. Reset overlapping-point EWMA storage when no future window can cover those points. A gap does not silently reset learned weights, the method optimizer, or adaptation buffers. They remain one entity run. This preserves method lifecycle rather than inventing a new adaptation reset.
15. Calibration selects eligible indexed windows under the approved label rule, then replays their scores without learned-parameter adaptation. Excluded windows contribute no calibration scores and are not joined into a synthetic sequence. Use the same score definition and smoothing policy as the target phase. Keep validation replay state separate from test state.

These policies are explicit next-version bindings. Their tests are still pending. The shared default schedule is a comparison choice; preserving a method's equations does not establish parity after a different encoder, stride, or coverage policy.

## Stage 1.2 step 5: metric and uncertainty review

Status: complete after the user selected the segment-summary rule below. Reference implementation and numerical tests remain pending.

### Point metrics and missing values

1. Align labels, finite scores, coverage, source segment, and original split indices before computing metrics. Metric eligibility requires a known label, a score, and coverage. Keep unknown labels and uncovered points out of label-dependent calculations without relabeling them.
2. Preserve source segment boundaries. Point metrics can pool eligible observations because their confusion counts do not require temporal adjacency. Do not remove missing points and then treat the compressed result as a continuous series for a range metric.
3. Use strict `score > threshold` for threshold-dependent predictions. Compute FPR as `FP/(FP+TN)`, precision as `TP/(TP+FP)`, recall as `TP/(TP+FN)`, and F1 as `2*TP/(2*TP+FP+FN)`. A zero denominator is missing, not a forced zero. Legitimate zero numerators with positive denominators give zero.
4. ROC-AUC requires both label classes. Average precision requires at least one positive label. Constant finite scores remain valid; a constant predictor with both classes has ROC-AUC 0.5. Use average precision, not trapezoidal PR area, for the retained `pr_auc` field. Record that meaning explicitly.
5. Requested unknown metric names or malformed inputs fail validation. A requested unavailable implementation/dependency is an error, not permission to substitute another metric. Undefined values due to class or coverage limits return null with a reason and counts. Do not use broad exception handling to hide implementation errors as undefined results.
6. Return unrounded numbers. Convert unavailable or non-finite metric results to null with a recorded reason. The reporting layer displays `N/A`; it does not change stored metric precision.
7. Keep `vus_pr`, `vus_roc`, `affiliation_f1`, `roc_auc`, `pr_auc`, `precision`, `recall`, `f1`, and `fpr`. Retain budgeted VUS at 0.001, 0.005, and 0.01. Use flat budget names such as `vus_pr_at_fpr_0.001`, preserving the budget in provenance. Do not silently substitute an author's unrestricted area for a project budgeted area.
8. Use the reviewed author VUS computation and author affiliation precision/recall where applicable. Derive affiliation F1 from their returned values; an empty prediction set with known positive events has F1 zero by the declared wrapper convention. A segment lacking required class support remains undefined. Do not call a reference convenience function that chooses a different classification threshold or fits on test labels.
9. Default VUS buffer size is 20 observed steps and threshold count is 200. These are explicit comparison settings, not universal metric constants. Parameter identity includes their values. Version and test reference calculations before using new dependencies.

### Uncertainty meaning

1. Preserve the source meanings of the eleven retained fields in the profile contract. Compute variance across actual Monte Carlo samples, not across unrelated time points. Default correction is 1. If `sample_count <= correction`, the variance estimate is missing. Do not turn one sample into proof of zero uncertainty.
2. For reconstructions `[B,M,L,C]`, take variance over `M` for `reconstruction_variance_full`; mean over `C` gives point variance; mean over `L` gives window variance.
3. For retrieval values `[B,M,L,H]`, take variance over `M`, then mean over `H` for point variance and over `L` for window variance. Apply this separately to continuous and discrete retrieval.
4. `point_anomaly_score_variance` is variance over the `M` raw point-error scores `[B,M,L]`. `window_anomaly_score_variance` is variance over per-sample window means `[B,M]`. These are not the same as variance of a point mean or a mean of variances.
5. For class probabilities `[B,M,K]`, variance over `M` gives class-probability variance; its mean over `K` is `classification_variance_mean`.
6. These are raw method-pass statistics. Do not label a smoothed variance as the variance of the final EWMA score: repeated-window covariance is not defined by the current source. Preserve raw-statistic provenance and sample count.
7. Align point uncertainty to the same point-selection rule as its score contribution: offline alignment averages its raw window statistics; online uses the last contributing window. This is a summary of raw statistics, not a derived variance of the aligned final estimator. Window summaries use window units. Keep these meanings explicit in metric metadata.
8. Summary `mean`, `std`, `min`, `p50`, `p95`, and `max` apply to the selected finite raw-statistic values. Here `std` uses population correction 0, matching the inspected source summary. Across-seed report standard deviation still uses correction 1. Quantiles use the linear method.
9. A method without a defined uncertainty output reports null and `not_provided_by_method`. A deterministic branch can have zero observed Monte Carlo variance without proving general predictive certainty. Do not save full Monte Carlo tensors by default; retain summaries and identities.

### Numerical checks

1. Select tolerances before comparison. For deterministic float64 formula checks, use absolute tolerance `1e-10` and relative tolerance `1e-8`. For float32 neural values, gradients, and same-device resume checks, use absolute tolerance `1e-6` and relative tolerance `1e-5`.
2. Identities, source indices, labels, masks, counts, state roles, and allowed update sets require exact equality. A floating tolerance cannot excuse an index, method, or parameter-group mismatch.
3. Metric comparisons must use the same arrays, domain, algorithm revision, and parameters. Do not claim equal metrics merely because both lie in `[0,1]` or report a perfect toy result.
4. Do not widen tolerances after a failure without finding and recording the cause. Passing CPU checks does not establish cross-device parity.

### Open choice: range metrics on separate segments

A small example shows the issue: `[0,1] | gap | [1,0]` contains two separate anomaly events. Removing the gap and concatenating the labels produces `[0,1,1,0]`, which appears to contain one event. The scalar index check confirmed counts of two and one, respectively. This is not a metric implementation test.

1. Option A: compute each eligible continuous segment separately and report a length-weighted mean of defined segment values. Name the aggregate as a segment summary. Report total, contributing, and undefined segments and their observation counts. Do not claim the mean equals a canonical whole-series range metric. In particular, normal-only segments can be undefined and absent from that mean; pooled pointwise FPR must still include their eligible observations.
2. Option B: compute segment values, but leave the whole-entity range metric null when the eligible domain is discontinuous. This keeps the original per-segment metric meaning and gives fewer complete reporting cells.

The user selected option A. Apply the rule to known time gaps, unknown labels, and uncovered intervals without inventing normal labels or continuous time.

Sources checked: [author VUS interface](https://github.com/TheDatumOrg/VUS/blob/main/vus/metrics.py), [author affiliation implementation](https://github.com/ahstat/affiliation-metrics-py/blob/main/affiliation/metrics.py), the thesis metric-flow note and tests, and `thesis_multitask_routing_geometry_helpers.py`. Public source revisions still require pinning before implementation.

### Confirmed segment-summary output

1. Split the eligible metric domain at source time gaps, unknown-label intervals, and uncovered score intervals. Keep original split indices. Never join across a break.
2. Compute reference range metrics separately on each segment. For each metric, let `v_i` be a defined segment value and `n_i` its eligible observation count. The segment summary is `sum(n_i * v_i) / sum(n_i)` over defined values only. If the denominator is zero, return null.
3. Store canonical per-segment metric names in the segment records. Use the distinct whole-entity names `segment_mean_vus_pr`, `segment_mean_vus_roc`, and `segment_mean_affiliation_f1`. Use the same prefix for budgeted VUS names. Do not label these means as canonical whole-series range metrics.
4. A canonical whole-domain range value is available only when its eligible domain is one continuous segment with required class support. Otherwise that canonical value is null. The segment summary remains available under the declared aggregation policy when any segment contributes.
5. Save every segment's index bounds, observation count, metric values, and missing-value reasons. For each summary save total/contributing/undefined segment counts, total/contributing/omitted eligible observations, and coverage. A normal-only segment may contribute no range value. Keep its observations in pooled pointwise FPR and other applicable point metrics.
6. Default detection reports select the clearly named segment summaries and raw FPR. A single contributing segment does not conceal omitted segments; still report counts and coverage.
7. `compute_metrics` must return a record containing `metrics`, `eligibility`, `segments`, and `metric_metadata`, rather than only a flat numeric dictionary. The run/result `metrics` field remains the flat scalar mapping used by the reporter. This additional output is required by the user's segment-reporting decision; record the signature change before implementation.
8. Prepared split records need original segment bounds. Batch and score records retain the relevant segment identity. Point-score records retain segment membership alongside coverage. A missing-label interval can split a metric segment without changing the prepared source segment or rewriting its observations.
9. Scores for the original point are evaluated against that point's label, with the score-availability delay retained. A segment summary is not proof of equal latency across methods. Keep latency and coverage visible in result metadata.

Ruling: use distinct segment-summary names and explicit counts. This preserves the requested summary without changing the meaning of a canonical VUS or affiliation value. If names are conflated, reports can make a false whole-series claim.

## Stage 1.2 step 6: smoke limits and isolation

1. Normal experiment runs use `outputs/experiments/<experiment_id>/`. Smoke runs use `outputs/smoke/<experiment_id>/`. Both apply the agreed grouping hierarchy below that experiment root. Record `execution_mode` as `normal` or `smoke`; do not infer it only from a path.
2. Keep the confirmed selection rule: first configured dataset, entity, applicable variant, and seed for each selected method and supported phase. Add required offline dependencies in stable order. Do not select an entity based on test labels or a convenient metric result.
3. Default smoke limits are one epoch per training stage, 64 training windows, 64 validation windows, 64 offline test windows, and 32 online windows. Add positive integer `max_epochs_per_stage` to `limits`. Apply it as an execution cap without editing the method profile.
4. MOMENT smoke limits are one epoch and four windows in each of train, validation, offline test, and online execution. This bounds pretrained-model work. Record these method-specific effective limits in the resolved run, including any dependency runs.
5. Limits cap selected indexed windows. They do not change prepared splits or silently lower configured prototype counts, class counts, cluster counts, covariance requirements, or selected model size. If a scientific prerequisite cannot be satisfied, report the failure.
6. Use the same execution functions, validation, checkpoint, and result formats for smoke and normal runs. A completed smoke run can be successful as a bounded execution; it is not a full-data accuracy result.
7. Parent lookup requires the same execution mode and experiment. Normal runs cannot select smoke parents. Normal reports exclude smoke records even if their other identities match.
8. Do not add a new smoke-report command. The planned complete-flow check confirms that normal report lookup excludes smoke outputs. Tests can inspect an explicit smoke-record list through the pure aggregation function.
9. Verification experiment definitions disable real W&B logging and Kaggle backup. Provider tests use fakes. The implementation plan authorizes no live verification upload.
10. A resumed smoke run keeps its recorded limits and mode. Changing limits or switching to a normal run creates another execution ID.

Ruling: use separate roots plus recorded mode. This keeps the workflow reusable and makes accidental smoke inclusion detectable. Smaller MOMENT limits reduce local work; they do not stand in for its real checkpoint and linear-probe checks.

## Stage 1.2 step 7: identity, status, and artifact files

1. Use schema version 1 for next-version run, result, calibration, checkpoint, and resume envelopes. This is separate from an existing preprocessing-manifest format. Do not require a new version field in protected prepared files.
2. Generate UUIDv7 run IDs at execution time. The checked environment is Python 3.12.12 and has no standard `uuid7`. Use a small private helper with standard-library time, secrets, and UUID according to RFC 9562 section 5.7: a 48-bit Unix millisecond timestamp, version 7, RFC variant bits, and 74 random bits. Do not use a model RNG for IDs. Test the layout before use. No generator state or within-millisecond monotonicity claim is needed.
3. Atomically reserve every new ID at `outputs/run_ids/<run_id>.json` using exclusive creation. This small project registry prevents two next-version allocations from claiming an ID even when their output roots differ. It records the actual state path. Before accepting a user ID, check discoverable existing project run identities as well. No old checkpoint reader is needed for an identity check.
4. A user ID is allowed only for a standalone run. It must be a safe single path component: 1–128 ASCII letters, digits, dots, underscores, or hyphens, beginning with a letter or digit. Reject an existing ID; never replace it. For a generated collision, retry generation at most five times, then fail clearly. Keep reservations after failure; do not recycle execution IDs.
5. Use UTC RFC 3339 timestamps with microseconds and `Z`. Preserve the original `started_at` on resume. Terminal records have `completed_at`, meaning the end of the recorded attempt; status determines success. Keep a small status history and resume count so an interruption is not erased.
6. Allowed status transitions are allocation to `running` or `blocked`; `running` to `successful`, `failed`, `interrupted`, or `blocked`; and an explicit eligible resume from `interrupted`, `blocked`, or `failed` to `running`. `successful` is final. A valid saved continuation or an unstarted blocked dependency is required for resume. Do not restart completed runs.
7. An experiment's result group is `<experiment_id>/<phase>/<dataset?>/<entity?>/<method?>/<variant?>/<seed?>/`, using the agreed omission rules. Store each existing split result at `<group>/<split>/<run_id>/result.json`. Store `point_scores.parquet` as one nullable score column in selected split order. Save alignment settings once in shared metadata. Do not save per-point identity, index, timestamp, segment ID, or availability/finality columns. Reconstruct these from the checked prepared data and recorded window schedule. Do not duplicate features or source labels.
8. Whole-run state lives beside the split branches at `<group>/state/<run_id>/`. This is a state branch, not another split or scientific grouping dimension. It contains `run.json`, `calibration.json`, `resume.json`, resume payloads, and flat stage checkpoint names such as `stage_a.initial.pt`, `stage_a.best.pt`, `stage_b.initial.pt`, and `stage_b.best.pt`. Native single-stage methods use `fit.initial.pt` and `fit.best.pt`; immutable pretrained state uses its explicit pretrained role and reference.
9. Each checkpoint has a small JSON descriptor with role, schema, codec, identities, and SHA-256. Resume uses a commit descriptor as specified in step 9. Do not save every epoch checkpoint by default. Keep each required initialization and best role; identical content may share a recorded immutable reference rather than duplicate bytes.
10. A standalone root is its explicit `output_dir`, or `outputs/standalones/<run_id>/`. That root contains run state and split folders without repeating experiment grouping levels. Reject an already owned or nonempty output directory for a new execution. Only explicit resume can reuse its state.
11. Write payloads to temporary files on the same filesystem, flush them, and atomically replace the target. Write JSON descriptors only after payload hashes are known. Mark the whole run successful only after all requested split results and required artifacts are readable and checked. A split result alone does not establish whole-run success.
12. Discovery follows result/state metadata through the recorded hierarchy, not positional guesses. Keep the grouping/omission snapshot so edits to an experiment cannot make earlier paths ambiguous. The ID registry is for allocation and location, not a substitute for result identity checks.
13. Parent ordering is descending `(started_at, completed_at, run_id)`. Reporting uses the same tie order after its own metric-eligibility filter. Compare parsed UTC times; the final ID order is lexical. Do not confuse a later completion with the primary latest-start rule.

Ruling: one visible state branch keeps whole-run status out of a false train/test split. A small exclusive ID claim is necessary for project-wide allocation across optional standalone roots. The selected UUID layout follows [RFC 9562](https://www.rfc-editor.org/rfc/rfc9562.html#section-5.7); correctness still requires implementation tests.

## Stage 1.2 step 8: parent and artifact compatibility

1. Find successful offline candidates only in the same experiment and execution mode. Match dataset, entity, seed, method, and explicitly required offline variant. A standalone that needs offline state uses the user-supplied parent directory and validates that run's recorded identity.
2. Also match the prepared version, manifest SHA-256, scaler SHA-256, feature-order hash, input feature count, effective window length, and scientific data-profile settings. Match the common encoder settings, head architecture, and expected source method-profile content. Profile paths alone are insufficient.
3. Match runtime scientific code identity and relevant dependency versions. Compute runtime code identity from maintained Python source that implements the run; do not include document edits, caches, generated results, or the report-only hook. Keep the original preprocessing code hash as prepared provenance, not a demand to rerun preprocessing.
4. Compare compatible scientific subsets rather than demanding equality of an entire offline and online config. Their phases, variants, and update controls necessarily differ. The online configuration names the required source profile/variant; source fitting choices remain in the selected parent's snapshot.
5. Calibration identity includes its selected checkpoint hash/role, method and variant, prepared identity, features, window/stride/tail/gap rules, score space, channel/point reduction, score transform and fitted parameters, calibration source and eligibility counts, quantiles, EWMA pair and reporting policy, execution mode, and any work limits. Do not use a threshold with another score definition.
6. An offline parent must be successful and record its required best artifact role. Filter for compatible recorded identities first, then use the agreed start/completion/ID ordering. Validate the selected artifact's existence, checksum, role, codec, schema, and shape. If selected required files are missing or damaged, stop; do not silently choose an older candidate.
7. Record the selected parent ID and all used artifact hashes before online scoring. Resume restores that selection. It never looks for a newer parent to replace it.
8. Zero-shot MOMENT has no fitted offline-parent requirement. Its immutable checkpoint, model configuration, and pre-test calibration have their own identities. Linear-probing MOMENT must match the chosen offline trained reconstruction head and the same pretrained revision.
9. Reporting compatibility uses the recorded per-run scientific settings, metric definition/version, domain/segment policy, mode, and limits. Do not invalidate a run solely because an unrelated method was added to an experiment. Do not combine a changed profile or metric definition silently.
10. Record incompatible candidate reasons. No candidate means a blocked dependency. A selected corrupt or missing artifact is a failed integrity check. Neither case is a successful run with an invented empty state.

Ruling: compatibility follows quantities that affect the scientific result. A path name or a shared dataset name cannot establish compatible weights, calibration, or preprocessing.

## Stage 1.2 step 9: exact recovery boundaries

1. Offline neural fitting commits after each complete epoch and each stage transition. Resume starts the next uncommitted epoch from the last committed state. An interrupted partial epoch is recomputed; no partial epoch is advertised as committed work. Native non-iterative estimators commit before and after `fit`; an interrupted fit is recomputed from its recorded initial seed and inputs.
2. Add optional `resume: Record | None = None` and `on_checkpoint: Callable[[Record], None] | None = None` to fitted method `fit`. The method supplies its complete epoch/stage snapshot to the callback. The owning workflow adds run identity, RNG state, scoring/evaluation cursor, and writer state, then calls `save_resume`. This avoids hiding workflow recovery inside a model helper.
3. Offline calibration and evaluation commit at completed split boundaries. Save the selected best state and the RNG state before the next split. Recompute only an uncommitted split after interruption. Do not select a new best checkpoint from test metrics.
4. Online recovery commits at window boundaries, after the score event, allowed update, and pending/finalized-score state agree. Default checkpoint interval is eight windows, with an initial snapshot and a final snapshot. A graceful interruption commits the latest complete window. A hard stop resumes the last durable snapshot and recomputes its uncommitted suffix.
5. A snapshot contains the complete resolved config, scientific code/dependency identities, source parent and artifact hashes, next cursor, method state, optimizer/scheduler state when applicable, method RNGs, Python/NumPy/PyTorch CPU and used-device RNGs, EWMA/pending scores, finalized-score accumulator, segment cursor, and any unfinished method buffers. Restore RNGs after constructing/loading objects so construction does not consume the resumed stream.
6. Resume payloads use two working slots, `resume-0.pt` and `resume-1.pt`. Write the inactive payload and its hash first, then atomically replace `resume.json` as the commit pointer. The pointer includes all selected role references and writer offsets. A reader sees one complete generation. Corrupt committed state is an error; do not choose an older scientific parent or invent missing optimizer state.
7. Best checkpoint candidates use immutable generation/hash filenames, not an overwrite of a payload still referenced by the current commit. Publish role references with the resume commit. After a new commit, remove only unreferenced working candidates; retain required stage initialization and selected best roles. This refines the flat filenames in step 7 without adding another grouping level.
8. Final score tables and split JSON are atomically published. Partial tables are not valid completed results. Keep unfinished score accumulators in the committed resume payload, rather than persisting a raw per-window tensor trace. On successful completion, keep required final method state and compact results; close the resume descriptor and remove obsolete working slots.
9. Use a per-run operating-system file lock during execution/resume on supported macOS/Linux filesystems. Reject a second active writer. A run left `running` after a process death can be resumed only after the lock is available and its committed state validates. Record the interrupted attempt in status history.
10. `resume_experiment` preserves successful runs. It resumes eligible interrupted/failed snapshots and can recheck an unstarted blocked dependency using that run's saved settings. An already selected parent remains pinned. A run with neither valid continuation nor a recorded unstarted boundary is not silently restarted.
11. Resume does not apply edited profiles, new defaults, changed limits, or a new model revision. Use the saved resolved snapshot. Reject incompatible scientific code or dependencies. A new execution is required for a changed scientific configuration.
12. Use tensor/primitive checkpoint envelopes for safe structured loading. Encode NumPy arrays without changing precision. Native estimators carry their versioned local estimator representation. Only registered project-generated payloads are load candidates after role/hash/version checks; historical checkpoint readers are not added. Immutable pretrained weights and frozen parent weights can be referenced by identity instead of duplicated in every child state.

Ruling: exact continuation means equality from a declared durable boundary, not a claim that uncommitted work survived a crash. Immutable best-role generations and one commit pointer are required to avoid a snapshot referring to an overwritten best state.

## Stage 1.2 step 10: optional provider operations

1. `wandb_logging` and `kaggle_backup` are independent booleans, false by default. Disabled providers need no credentials, imports, or network calls. Verification uses fakes only.
2. W&B receives committed summaries and run identity. Use the run ID as provider ID, the experiment ID as group, and project `bachelor-thesis-2026` unless explicitly configured. Resume uses the same ID. Do not send credentials, prepared features, source labels, or raw Monte Carlo tensors. Provider history is not the recovery authority.
3. Kaggle backup runs after experiment completion, including a recorded failed experiment, and after a graceful interruption has committed local state. A hard process stop cannot guarantee a backup. Resume must not depend on cloud availability.
4. Require an existing private Kaggle dataset named by `kaggle_dataset` in provider settings. Validate its privacy before upload. Do not create a dataset, choose a license, or make data public. Credentials remain in the provider's external credential store.
5. Select only this experiment's checked result files, configuration snapshots, manifests, required initialization/best/final checkpoints, and committed recovery state. Exclude prepared datasets, credentials, temporary payloads, uncommitted checkpoints, caches, and unrelated experiments.
6. Use a temporary flat upload folder on the output filesystem. Hard-link immutable selected files under content-hash names. Write a small manifest from original relative paths to uploaded names and checksums. Do not make ZIP/TAR archives or copy checkpoint bytes. An unsupported hard link is a clear backup error. Remove the temporary links after the attempt.
7. Preserve file formats on Kaggle. Upload flat files with directory handling `skip`. The manifest preserves the local hierarchy. No download or restore command is added under this plan.
8. Make one application-level provider attempt per event; do not loop on upload failures. Record the attempted manifest hash and returned remote version when available. An ambiguous upload outcome is reported as unknown, not automatically retried. A local confirmed receipt avoids a duplicate upload of the same manifest. The implementation must configure or document any retries inside the selected SDK.
9. Provider failures do not change a successful scientific result into a failed scientific run. Record provider status separately and return a nonzero command status when an explicitly requested provider operation fails. Never retrain a successful run to retry an upload. Emit a short sanitized error without credential values.
10. Complete local state first on interruption. Attempt the optional backup once. A second interrupt may stop the provider operation. Local continuation remains valid.

Evidence: the [Kaggle dataset commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) document dataset versions, file-format retention, and directory modes. [W&B settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py) document explicit run IDs and resume modes. These sources establish API options, not a tested integration.

Ruling: use an existing private dataset and a temporary hard-link view. This avoids an invented publication license and duplicate local archives. The cost is that users must configure the dataset before enabling backup.

## Stage 1.2 step 11: package and command ownership

1. The thesis currently has no `pyproject.toml` or installed `.venv/bin/tsad`. The library distribution is `tsad-lib`; its checked metadata has no console-script registration. Do not edit the library package for this thesis delivery.
2. Register thesis distribution `thesis-tsad` with console script `tsad = thesis_tsad.cli:main`. Map package `thesis_tsad` to physical `src/`, and `thesis_scripts` to physical `scripts/`. List only the new maintained packages explicitly. Do not discover or import old runtime packages automatically.
3. Install into the thesis `.venv` only. Before installation, inspect that environment's console entrypoints. Stop if another distribution owns `tsad`; do not overwrite it silently. Do not install globally or in the library environment. Document activation and `.venv/bin/tsad` usage.
4. Keep preprocessing self-contained. The CLI calls `thesis_scripts.preprocess.main` without making preprocessing import runtime modules. Package initializers remain empty or contain documentation only.
5. New prepared-data and metric modules are `src/prepared_data.py` and `src/evaluation_metrics.py`. The old `src/data/` and `src/metrics/` packages still exist. Different module names avoid import ambiguity while the plan retains old code until Phase 11. Public function names do not change.
6. The project root defaults to the current directory, validated by the maintained project metadata and config structure. An optional global `--project-root` selects another root explicitly. Do not search unrelated parent folders or silently fall back to a different checkout.
7. Register core runtime dependencies and separate optional provider/MOMENT dependencies in the thesis metadata. Resolve actual tested versions during their owning implementation stages. Missing requested functionality fails clearly; installation of the command alone does not prove all methods are ready.

Ruling: isolate the distribution and import namespace, and rename two colliding modules. This preserves the simple file tree and the removal order. Users must use the intended environment when another project also supplies a command named `tsad`.

## Stage 1.2 step 12: user-decision review

1. The user confirmed point-index EWMA, last-estimate reporting, and default weights 0.25/0.75. Availability and finalization times remain explicit.
2. The user confirmed a method-specific MOMENT length of 512 inside the single data profile per dataset.
3. The user required scientific intent to guide paper/code conflicts. The CANDI signed-gate reconciliation is recorded separately from unproved training parity.
4. The user confirmed calibration that excludes windows with known anomalies and permits unknown labels under a recorded assumption. The user also required care because real training data can contain anomalies.
5. The user confirmed MOMENT's channel mean and CANDI's time mean for its latent vector.
6. After a pause and a repeated question, the user confirmed breaking windows at known gaps and recording the real-world limitation. The later instruction resumed work.
7. The user confirmed per-segment range metrics and length-weighted means of defined segment values, with omitted segments and coverage. Use the distinct segment-summary metric names.
8. These replies settle their named questions. They do not approve the entire implementation contract. Stage 1.2 step 14 remains the explicit review gate. Do not request the same scientific choices again.

## User changes after the walkthrough: 2026-11-17

### Decision 1: reconstruct point identity

1. Confirmed: point indices and identity arrays are temporary runtime values. Do not save an identity record for each point.
2. Save the prepared-data reference/hash, entity, split, selected half-open range, window/stride/tail/gap settings, effective work limits, schedule code identity, and run end reason once in shared metadata. Preserve prepared data for reconstruction.
3. Save one nullable `score` column in selected split order. Retain null rows for positions without scores. Array position j corresponds to split index `stream_start + j`; a full split starts at zero. Do not compress the selected range.
4. Reconstruct source rows, timestamps, gaps, segment membership, window membership, and score availability. Do not persist those per-point columns. Aggregate segment bounds and coverage remain permitted.
5. Runtime batches still use indices for overlap alignment and EWMA. A requirement to preserve pre-update events does not require saving every window trace.
6. Recovery saves cursor, method/RNG/scoring state, completed scores, and ordered pending EWMA values. Reconstruct their indices from the same schedule. Do not serialize per-point identity maps. Test gaps, tails, work limits, and interruption boundaries.
7. This supersedes earlier per-point identity and availability storage requirements. Scientific alignment and delayed-score semantics remain unchanged.

### Decision 2: fair epoch counts

1. Confirmed: epoch counts must be fair across methods. Withdraw the unequal 30/100/30/10/10 comparison defaults.
2. Confirmed after clarification: use 30 total offline epochs per fitted neural method, counting all stages together. Do not give a two-stage method twice the budget.
3. The user selected 30 total epochs. MUSTARD uses 25 + 5; RedLamp, CANDI, M2N2, and MOMENT linear probing each use 30. The ten-epoch option was not selected.
4. Zero-shot MOMENT and native STUMPY/KMeansAD/Isolation Forest have no epoch-based training. Online update steps are separate from offline epochs. Do not invent epochs for these cases.
5. Equal epochs do not establish equal optimizer steps, compute time, or pretraining exposure. Preserve method-specific optimizers, losses, and permitted parameters.
6. Keep stage epoch counts in method profiles. Validate equal stage totals across selected fitted neural methods; supplied normal profiles total 30. Do not add a duplicate experiment epoch field. Smoke caps remain separate. Resume continues the recorded budget. This resolves the epoch clarification; full-contract approval remains pending. No runtime implementation is changed here.

## MUSTARD scoring comparison

1. Compare point-wise mean squared error (MSE) in raw input space and latent space. For raw input MSE, inverse-transform the observation and reconstruction with the saved training scaler. For latent MSE, compare the L2-normalized query with the L2-normalized continuous prototype reconstruction before fusion.
2. Fit a separate point threshold for each score. Use the 0.95 quantile of final point EWMA estimates from eligible validation windows. Use 0.25 for the previous estimate and 0.75 for the newest score. Keep the current validation label policy and fit with fixed model parameters.
3. For stochastic latent retrieval, compute MSE for each sample and average the MSE values at each point within the window. In both phases, use EWMA across overlapping-window estimates of the same time point. Keep arithmetic overlap averaging for ordinary offline profiles.
4. Offline saves the validation q95 threshold for the selected score. Offline test and the matching online run use that threshold. Online keeps the threshold fixed during adaptation under input distribution shift. Raw pairs with raw and latent pairs with latent.
5. The [runtime profile contract](../detail/runtime-profile-contract.md#mustard-validation-q95-comparison) owns the configuration and compatibility rules. The comparison supersedes the proposed translation of the ambiguous legacy SMD q95 profile.

## MUSTARD O1 batches with a missing point group

1. In Stage A, if a batch has both normal and anomalous points in its synthetic mask, the classification branch uses the mean of the classification loss and the balanced point-score loss.
2. If either group is absent, use the full classification loss for the branch and keep the reconstruction term. Mark the point-score loss as unavailable for that batch. Exclude it from point-score loss averages. This follows the reference implementation.
3. **Limitation.** The classification loss has twice its usual O1 coefficient on a batch that uses the fallback. The effective balance between the two losses can therefore vary with batch composition.
4. **Future analysis.** Report the number and fraction of fallback batches by seed and epoch. Keep training and validation counts separate. Check whether changes in batch composition alter these counts, checkpoint selection, or detection metrics.

## SMD synthetic calibration choice — 2026-11-19

1. **Confirmed Option A.** For `synthnormal_q99_normalized_input_mse_ewma09.yaml`, generate synthetic calibration windows from the saved validation split.
   Score these windows with the selected offline checkpoint and keep its parameters fixed.
   Use eligible scores to set the point and window q99 thresholds.
2. Keep the existing SMD profile name and calibration source.
   Use test data only for evaluation after threshold selection.
3. **Limitation for future analysis.** If the upper quantile of training scores is lower than that of validation scores, a threshold set on training scores can flag more normal test points.
   This can occur without a large distribution shift.
   Compare the quantiles for raw and latent mean squared error separately.
   A decrease in mean training error does not establish a decrease in the upper quantile.
4. The [MUSTARD q95 comparison](#mustard-scoring-comparison) uses its own validation calibration rules.
