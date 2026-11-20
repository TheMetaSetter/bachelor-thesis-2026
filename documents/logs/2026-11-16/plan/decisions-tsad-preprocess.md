# Decisions: tsad preprocess

Date: 2026-11-16
Status: user-confirmed decisions. All required data is processed. The installed `tsad preprocess` command uses the standalone script.
Project: `bachelor-thesis-2026`.

## Goal

Provide `tsad preprocess` to prepare local data for later runs.

## Completion status

1. On 2026-11-16, the user confirmed that all required data is processed. Saved manifests cover ICCAD (1 entity), SMD (28), IOPS (29), NASA (80), and SWaT (1).
2. NASA excludes P-2 and T-10. One complete NASA version remains after removal of two incomplete versions.
3. The five original source folders were removed after preprocessing and saved-output checks. SWaT removal released 844,392,461 bytes. The script itself does not delete sources.
4. Two SWaT output versions remain. This document update does not remove either version.
5. This update checked manifest presence and entity counts. It did not repeat all output integrity checks or prove publisher provenance.

## Confirmed decisions

1. **Dataset scope: option 1C.** The first version must support these five datasets:
   1. ICCAD
   2. SMD
   3. IOPS
   4. SWaT
   5. NASA
2. **Output: option 2B.** Produce training, validation, and test data with scaling. Define the default split rules and scaler state during this design work. Fit the scaler on training data only. Apply that fitted scaler to validation and test data.
3. **Command input: option 3A.** The user selects the dataset and data profile directly. An experiment is not required before preprocessing. The data profile supplies the preprocessing settings.

The earlier proposal included only ICCAD and SMD. The confirmed five-dataset scope replaces that proposal.

## Data rules: interview decisions 1–5

1. **Preserve source splits (1A).** Preserve verified original train/test splits. Create validation data from the original training data. For sources without a verified original split, apply decision 6 below.
2. **Scaler scope (2A).** Fit and save one scaler per entity. Use only that entity's final training portion for fitting. Apply the saved scaler to its validation and test data.
3. **Missing values (3A).** Stop with an error if missing values occur and the data profile has no explicit handling rule. Report the affected entity and columns. Do not apply an implicit repair rule.
4. **Validation size (4A).** Use the last 20% of the original training sequence for validation by default. Preserve time order. Fit the scaler on the remaining training portion only. A data profile can override the ratio. Reserve `floor(validation_ratio * n)` observations. Stop if any split is empty. Later data loading checks usable window lengths.
5. **Scaler type (5B).** Use robust scaling by default: subtract the training median and divide by the training interquartile range. A data profile can override the scaler. Apply decision 7 when the range is zero.

## Data rules: interview decisions 6–7

6. **No original split (6B, with later clarifications).** Use chronological splits with a target of 60% training, 20% validation, and 20% testing. Newly created training splits must contain no confirmed labeled anomalies, as specified in decision 11. This requirement takes priority over the target ratios. A data profile can supply explicit split boundaries instead. These percentages apply directly to the full sequence; do not apply the 20% validation rule from decision 4 again. Apply the boundary rule below. Use `floor(0.6 * n)` and `floor(0.8 * n)` before boundary adjustment. Later data loading checks usable window lengths.
7. **Zero interquartile range (7A).** Use a divisor of 1 for an affected feature. Subtract its training median and keep the feature. Record the affected feature and effective divisor in the saved scaler state. Apply the same saved transform to validation and test data.

## Data rules: interview decisions 8–9

8. **Time order (8B, with later clarifications).** Every data profile must explicitly enable timestamp sorting by default. Use a stable sort within each entity and each verified source split. Preserve source order between rows with equal timestamps. If a profile disables sorting, stop when timestamps go backward. For a source without an original split, establish time order before creating chronological splits. Where timestamps are absent, preserve source row order and record that elapsed-time continuity is unverified.
9. **Duplicate timestamps (9A).** Keep rows with equal timestamps by default. Preserve their source order. Make duplicate handling configurable in the data profile. Record the applied rule in output provenance. Other duplicate-handling modes still need specification.

## Data rules: interview decision 10 and training constraint

10. **Time gaps (10A).** Preserve gaps by default. Do not insert timestamps or observations. Irregularly sampled time series may be future research objects. Rules for later window creation across gaps remain to be specified.
11. **Training anomaly constraint (revised).** For datasets without a verified original train/test split, select boundaries so the newly created training split contains no confirmed labeled anomalies. Training data may still contain unknown, unlabeled anomalies.

**Original-split exception:** if a verified original training split contains labeled anomalies, those observations may remain in the resulting training portion. Preserve the source train/test assignment and apply the agreed validation holdout. Do not remove labeled anomalies solely to enforce the new-split rule in this case.

Fit the scaler on the final training portion under the applicable rule, including any labeled anomalies allowed by the original-split exception. Record split provenance, effective boundaries, and known training anomaly counts. Establish label meanings and annotation mapping before applying the rule. Do not relabel observations to satisfy it.

### Superseded proposals

The boundary-based rule and original-split exception replace automatic training anomaly removal. The proposed 20% maximum exclusion fraction is no longer applicable. The proposed 100-observation minimum was not confirmed.

### Confirmed boundary rule

1. For a new split, start with the 60/20/20 boundaries in time order.
2. If the proposed training portion contains labeled anomalies, end training before its first labeled anomaly.
3. Keep the test start at the original 80% boundary. Assign all observations between training and test to validation. Validation can grow and training can shrink. Preserve all observations.
4. The last validation observation must be immediately before the first test observation in the ordered sequence. The splits must not overlap or omit observations at this boundary. For irregular sampling, adjacency refers to observation order; do not insert timestamps to remove a source time gap.
5. Preserve verified original train/test assignments under the original-split exception. Check the validation/test timeline against source evidence. Report a source inconsistency rather than inventing temporal continuity.
6. Stop if any split is empty. Later window checks must verify usable window lengths.

## Output decisions 12–15

12. **Format.** Save each split as Parquet. Save scaler state and metadata as JSON.
13. **Location.** Default to `data/preprocessed/<dataset>/<version_id>/` (updated 2026-11-20). Allow an explicit output root. Save `manifest.json` at the version root. Each entity folder contains `train.parquet`, `validation.parquet`, `test.parquet`, and `scaler.json`.
14. **Version identity.** Derive `version_id` from selected source-file contents, resolved preprocessing settings, and preprocessing code version. Hash source-file contents. The script hashes canonical JSON containing source hashes, resolved settings, and the preprocessing code hash with SHA-256.
15. **Repeat execution.** Reuse a complete matching version after checking its recorded files. Stop if that version is incomplete or damaged. Do not overwrite it automatically. An explicit recovery action remains to be specified.

## Dataset-specific missing-value decisions

Confirmed on 2026-11-16. Option letters refer to the [missing-data options](../research/research-missing-data-options.md). These choices are implemented in the self-contained script.

1. **ICCAD: option A, count features only.** Use the 5XX count-feature group confirmed in the later implementation discussion. Fill missing counts with zero under the explicit assumption that absence means no recorded events. This assumption does not prove that collection failures did not occur. Exclude response-time statistics. Treat the Console as one entity.
2. **SMD: option A, check and stop.** Keep feature values unchanged during missing-value handling. Stop if the selected source contains missing values. Do not fill them. The relationship between the two local SMD representations remains unverified.
3. **IOPS: option A, check and stop.** Keep feature values unchanged during missing-value handling. Stop if missing values occur. Do not fill them. Confirm value and label columns before preprocessing.
4. **SWaT: option B, explicit feature subset.** Exclude `MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, and `MV303` from every split. Keep the other 44 numeric features after checks. Keep all rows. Do not fill the excluded features. Stop if retained features contain unresolved missing values. Record the reduced feature scope in the data profile and output metadata. Source assembly and validation/test timeline checks remain required.
5. **NASA: option A, check and stop.** Keep feature values unchanged during missing-value handling. Stop if a selected channel array contains missing values. Do not fill them.

These choices preserve the confirmed sorting, split, label, scaling, output, and version rules. Apply robust scaling after missing-value handling. Keep the zero-interquartile-range divisor-1 rule for retained features. Do not enable median filling, forward filling, interpolation, or missingness indicators under these choices.

## SWaT source split repair

Confirmed by the user on 2026-11-16. Implemented and checked in `scripts/preprocess.py`.

1. Use the trimmed early normal-operation version as the original training source. It contains 495,000 observations from 2015-12-22 16:30:00 through 2015-12-28 09:59:59. In the checked local normal.csv, these are zero-based source rows `[892098, 1387098)`. Do not combine it with the full version. The user confirmed the trimmed version on 2026-11-16.
2. Use the complete later period as the original test source. Include both its 395,298 Normal-labeled rows and its 54,621 Attack-labeled rows, for 449,919 observations. Preserve their labels and time order.
3. Do not treat all of normal.csv as original training or attack.csv alone as the full test source. These files separate observations by label across recording periods.
4. Apply the existing validation holdout and training-only scaler rules to the selected early training source. Keep the approved seven-feature exclusion and timeline checks.
5. This source-selection repair replaces keeping both copies of the early recording and using later Normal rows for training. It does not authorize general duplicate removal or timestamp changes.
6. See the [local checks, source evidence, and figures](../research/research-swat-source-timeline.md). The exact source archive and assembly script remain unverified. The [official iTrust description](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/) states that normal version 1 removes the first 30 minutes of tank drainage. Local checks confirm that the remaining rows match the full version across all columns.
7. Remove the original SWaT folder only after preprocessing and saved-output verification succeed.
8. The reader finds the three blocks from backward timestamp jumps in source row order. It checks that the trimmed block equals the full block after its first 30 minutes across all 44 retained features and timestamps. Unexpected layouts or labels stop with an error. Original source file names and row positions remain in saved observations.
9. Verification: all 46 temporary contract tests pass. The final real SWaT run saves 396,000 training rows, 99,000 validation rows, and 449,919 test rows. Testing contains 395,298 Normal and 54,621 Attack observations. Repeat verification, source mapping, validation/test adjacency, and independent training-only scaling checks on three sensors pass. Source files remained unchanged during those checks. They were later removed after saved-output verification.
10. Checked version at its current location: `data/preprocessed/swat/67e1e1753e66f44858d0b31ec38ccff60bb86e805b50a889d576fb02ca39e391/`.

## NASA P-2 exclusion

Confirmed by the user on 2026-11-16. Implemented in `scripts/preprocess.py`.

1. Exclude NASA channel `P-2` from preprocessing for now. Do not write training, validation, or test outputs for this channel.
2. The two published P-2 label rows describe separate recordings from different times, as the [dataset author explains in question 5](https://github.com/khundman/telemanom/issues/8). Both specify 8,209 observations. The local file length cannot identify the correct label row.
3. Do not combine the intervals or choose a label row without a verified recording match. After all selected NASA channels complete preprocessing and saved-output verification, remove the original NASA folder, including P-2 arrays and source label rows. This later user decision replaces the earlier instruction to retain P-2 source files. If preprocessing or verification fails, keep the source folder.
4. This exclusion follows the selection in [OmniAnomaly preprocessing](https://github.com/NetManAIOps/OmniAnomaly/blob/master/data_preprocess.py#L38). Reconsider it when a verified file-to-label mapping is available.
5. See the [source review and recording diagram](../research/research-nasa-p2-annotations.md). This decision changes channel selection only; the other NASA preprocessing rules stay the same.
6. Default NASA selection skips P-2. An explicit P-2 request through CLI arguments or a profile stops with a clear exclusion error. The script does not change or delete source files.
7. Earlier verification: 40 temporary contract tests passed. That NASA attempt stopped at T-10. The later T-10 exclusion resolved this blocker. NASA processing is complete; the failed versions and original source folder were removed after checks.

## NASA T-10 exclusion

Confirmed by the user on 2026-11-16. Implemented in `scripts/preprocess.py`.

1. Exclude NASA channel `T-10` from preprocessing. Do not write training, validation, or test outputs for this channel.
2. The local data file exists, but the source label file has no matching annotation row. Do not invent labels. The [dataset author states in question 5](https://github.com/khundman/telemanom/issues/8) that T-10 was intentionally removed because it had too few values.
3. NASA preprocessing must exclude both P-2 and T-10. Keep the other NASA preprocessing rules unchanged.
4. Remove the original NASA folder, including excluded channels, only after preprocessing and saved-output verification succeed. Keep the original files if either check fails.
5. Two earlier failed NASA versions were removed. One complete version remains. Checks for incomplete or damaged versions stay in place.
6. Default selection excludes P-2 and T-10. Explicit requests for either channel through CLI arguments or a profile stop with an exclusion error. All 42 temporary contract tests pass. Local source selection returns 80 channels and excludes both channels. Lint, syntax, and size checks pass. A later complete NASA version passed saved-output checks. All required NASA data is processed.

## Implementation decisions: one self-contained script

Confirmed on 2026-11-16 in the implementation discussion.

1. Deliver one self-contained `scripts/preprocess.py`. Keep the source readers, built-in profiles, shared flow, storage checks, and CLI in this file. No helper modules or required external profiles.
2. Allow an optional YAML profile to override explicit settings. Keep input root, output root, and entity selection available as command arguments.
3. Select ICCAD 5XX count features. Treat the Console as one entity. Label a five-minute observation when its interval overlaps an anomaly interval in UTC.
4. For sources without timestamps, preserve source row order and save source row positions. Do not invent timestamps or claim verified elapsed-time continuity.
5. Stop if SWaT source timestamps conflict with the confirmed source split rules. The 44-feature choice does not waive timeline checks.
6. Reserve `floor(validation_ratio * n)` original training observations for validation. Default to 0.2. For new splits, use `floor(0.6 * n)` and `floor(0.8 * n)` before the confirmed anomaly-boundary adjustment. Stop on empty splits.
7. Keep unknown labels unknown. Use NASA's inclusive annotation endpoints. Do not create windows.
8. Use linear percentiles for robust scaling. Support standard scaling only as an explicit profile override. Fit either scaler on final training data only.

The script's module documentation is its usage and interface specification. Tests may use temporary files; they are not additional delivered scripts. This single-file requirement replaces the earlier proposal for separate reader modules and external default profile files.

## Command example

```bash
tsad preprocess --dataset iccad --profile <data_profile>
```

The installed command calls the same script as `python scripts/preprocess.py --dataset iccad`. It uses built-in defaults and accepts optional `--profile`, `--input-root`, `--output-root`, and `--entities` arguments.

## Decisions still required

1. Verify publisher provenance for the local sources. The script preserves supplied train/test roles; filenames alone do not prove publisher provenance. The two local SMD representations have not been proved equivalent. The script uses the separate machine files.
2. Define usable window lengths during later data loading. The script resolves split rounding but does not create windows.
3. Investigate source-specific missing markers and ICCAD collection failures. The script implements the accepted missing-value rules; it cannot prove source completeness.
4. The approved SWaT source-selection repair resolves the local timeline conflict. Publisher provenance remains unverified. Preserve its wall-clock timestamps without inventing a timezone.
5. Define an explicit recovery action for incomplete or damaged versions if required later. The script currently stops without overwriting them.

## Implementation checks

1. [x] Record the accepted single-file interface and data rules. Tools: Markdown and module documentation.
2. [x] Implement built-in profiles and five source readers. Tools: Python, NumPy, pandas, and PyArrow.
3. [x] Implement shared splitting, missing-value checks, scaling, and versioned output. Tools: Python, JSON, and SHA-256.
4. [x] Implement CLI arguments and embedded usage instructions. Tool: argparse.
5. [x] Run 37 contract tests from a temporary directory. Tool: pytest. No additional test script is delivered in the project.
6. [x] Review the script and fix shared train/test source paths and damaged-manifest reuse checks. Tools: independent review and regression tests.
7. [x] Run SMD `machine-1-1` through the final script. Tool: local Python environment. Saved rows: 22,784 training, 5,695 validation, and 28,479 testing.
8. [x] Reload Parquet and JSON outputs and compare them with the raw training-only transform. Tools: pandas, NumPy, and JSON. Source row positions, feature order, and unknown training labels are preserved.
9. [x] Verify repeat execution does not rewrite the complete matching version. Tools: hash checks and file modification times.
10. [x] Check syntax, lint, whitespace, and size limits. Tools: py_compile, Ruff, Git, and Python AST. After the SWaT repair, the delivered script has 500 lines and no function exceeds 50 lines.

SMD machine-1-1 and the repaired SWaT source received complete real-data executions checked in this work. All five readers have small-fixture tests; the latest suite has 46 tests. All required data for the five datasets is now processed, as confirmed by the user and supported by saved manifests. This does not complete every earlier exploration check or establish publisher provenance.

## Specification updates

1. Add the accepted command behavior to `documents/specs/desired-cli-20261109.md` when its details are settled.
2. The later single-file requirement uses the script's module documentation for its interface and usage specification. No separate `documents/specs/data-preprocessing.md` was created.
3. Keep these decisions consistent with [the execution plan](plan-tsad-preprocess.md).

## Terminology changes

No existing terms were renamed. `tsad preprocess` is a new proposed CLI command. Data profiles retain their existing meaning: reusable settings for data preparation.
