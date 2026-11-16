# Missing data: options for preprocessing

Date: 2026-11-16
Status: local inspection and proposed options. No option is approved or implemented here.
Scope: ICCAD, SMD, IOPS, SWaT, and NASA.

## 1. Rules that every option must follow

The [confirmed decisions](../plan/decisions-tsad-preprocess.md) take priority.

1. Put each missing-value rule in the data profile. Without a rule, stop and report affected entities and columns.
2. Preserve observations, timestamps, labels, duplicate rows, and source time gaps. Do not delete rows or insert observations.
3. Apply stable timestamp sorting as agreed. Establish final splits before fitting fill values or feature filters.
4. Preserve verified source train/test assignments. Apply the confirmed training anomaly constraint and its original-split exception.
5. Keep the final validation observation immediately before the first test observation. Filling must not change split boundaries.
6. Learn fill values and any data-based feature selection from final training data only, per entity. Freeze them for validation and test.
7. Fit the agreed robust scaler on final training data after missing-value handling. Keep the divisor-1 rule for zero interquartile range. Missing-value handling does not replace this rule.
8. Save rules, selected features, fitted fill values, counts, and source identity in the existing JSON metadata. Keep the agreed Parquet outputs and version rules.
9. Do not fill labels or timestamps with feature fill rules. Resolve their meaning or stop. Infinite values require an explicit error or handling rule too.

## 2. What was checked

These counts come from fresh local reads. They do not prove publisher provenance or complete time coverage.

| Dataset | Check | Result |
| --- | --- | --- |
| ICCAD | All pivoted Parquet footer null counts; six decoded columns | 39,365 rows and 117,448 metric columns. Only 224 metric columns have no stored nulls. Half have at least 99.89% nulls. No column is entirely null over the full file. |
| SMD | Three NumPy arrays and 84 separate entity text files | No NaN or infinite values. The two representations are still not proved equivalent. |
| IOPS | All 58 local two-column files | No NaN or infinite values. Named missing markers and publisher label semantics remain unverified. |
| SWaT | All three CSV files | Seven columns each have 991,800 nulls in normal and merged. Attack has no parsed nulls. All numeric columns have no infinite values. |
| NASA | All 164 train/test NumPy arrays | No NaN or infinite values. This does not prove complete original telemetry. |

ICCAD footer counts are not a full decoded scan. The six selected columns matched their stored null counts and had no observed infinite values. The previous unpivoted-table scan used footer statistics only. A long table with no stored nulls can still omit time/feature combinations.

The SWaT columns are `MV101`, `AIT201`, `MV201`, `P201`, `P202`, `P204`, and `MV303`. Each null block occupies source rows `[395298, 1387098)` in normal and merged. These are zero-based positions, before timestamp sorting. At that boundary, normal jumps from 2016-01-02 back to 2015-12-22. The cause is not established. This is evidence to check source assembly, not proof of its cause.

## 3. ICCAD

The paper links missing cells to inactivity or low activity. Some response-time statistics are undefined with too few observations. Its example uses 2,406 server-error count features and replaces their nulls with zero. This does not justify zero for every statistic. [Dataset paper, Section IV-B1 and Appendix B-A](https://arxiv.org/html/2411.09047)

| Option | Rule | Pros | Cons |
| --- | --- | --- | --- |
| A. Count features only | Select an explicit count-feature group. Fill null counts with zero only where absence means no events. Stop for unexplained missing counts. | Small rule. Clear units. Close to the paper's example. | Removes response-time information. Zero is wrong when collection failed. |
| B. Rules by metric type | Apply A to counts. Fill selected continuous statistics with their observed training median. Stop if a required training median cannot be fitted. | Retains response-time features. Uses fixed, simple state. | Filled values can hide changes. Extremely sparse features have little observed evidence. |
| C. B plus missingness indicators | Apply B. Add a fixed binary indicator for each selected feature: 1 means missing before filling. | Preserves which values were absent. | More input features. Models may respond to collection failures. It does not recover unobserved measurements. |

**Proposed first choice:** A for an explicit count-based profile. Use B or C only for a separate profile that needs response-time features. Do not fill all 117,448 metrics with zero. Confirm entity and annotation mapping first.

## 4. SMD

| Option | Rule | Pros | Cons |
| --- | --- | --- | --- |
| A. Check and stop | Keep values unchanged. Stop if a selected source contains missing values. | No invented values. Matches the scanned files. | A damaged or different source blocks preprocessing. |
| B. Training median | Fill missing feature cells with the observed training median for that entity and feature. Stop for an empty training feature. | Simple, fixed state. No future values. | Hides gaps and can reduce anomaly size. |
| C. Short forward fill | Carry the last observed value within a split, up to an explicit gap limit. Use B for remaining missing cells. | Uses recent state for short gaps. | Can preserve a stale value. Requires a gap limit and a boundary rule. |

**Proposed first choice:** A. B and C are optional profiles for a source that actually needs repair. Do not combine the aggregate and separate entity representations without proving their relationship.

## 5. IOPS

| Option | Rule | Pros | Cons |
| --- | --- | --- | --- |
| A. Check and stop | Preserve feature values. Stop on missing values. | Smallest rule for the scanned files. | Does not repair a new source with gaps. |
| B. Short forward fill | Fill short feature gaps from the last observed value. Use the observed training median for remaining gaps. | Simple for a single KPI value sequence. | Can flatten a spike or a level change. No local timestamps are established for a duration limit. |
| C. B plus one missingness indicator | Keep an indicator beside the filled KPI value. | Retains the distinction between measured and filled values. | Changes the method input. Missing collection may become a detection signal. |

**Proposed first choice:** A. Confirm which column contains values and which contains labels. Never fill the label column. Row order alone does not establish that no original time steps are absent.

## 6. SWaT

The missing block affects about 71.5% of normal rows in each affected column. Short-gap filling cannot reconstruct it.

| Option | Rule | Pros | Cons |
| --- | --- | --- | --- |
| A. Check the source, then stop if unresolved | Verify source assembly and column mapping. Restore values only from a verified source. Preserve time records and record the source version. | Can correct an assembly error without inventing measurements. | May require source access. Does not guarantee recovery. |
| B. Use an explicit feature subset | Exclude the seven named features from every split. Keep the other 44 numeric features after checks. Keep all rows. | No invented values. Same feature schema in every split. | Loses seven signals. Results must state the reduced feature scope. |
| C. Fill by verified feature type and add indicators | For continuous features, use observed training medians. For discrete states, use an observed training mode with a fixed tie rule. Add missingness indicators. | Keeps the feature slots and marks filled cells. | Most values in these seven columns remain invented. Long blocks can distort learning. State meanings need verification. |

**Proposed first choice:** A before publishing results. If recovery is not possible, B is the simpler proposed profile. B is an explicit feature choice, not the removal of zero-IQR features. Keep the confirmed zero-IQR rule for retained features. C is a weaker fallback and needs separate approval.

Do not forward-fill across the entire block. Do not merge normal and attack files by position or assume their timestamps form consecutive source splits. Check the agreed validation/test timeline separately.

## 7. NASA

| Option | Rule | Pros | Cons |
| --- | --- | --- | --- |
| A. Check and stop | Preserve each channel array. Stop on missing values. | No invented telemetry. Matches the scanned arrays. | A new incomplete source needs a different profile. |
| B. Training median for continuous features | Use an observed training median per entity and feature. Stop on unresolved discrete-feature semantics. | Simple and repeatable. | Can reduce short anomaly peaks. A median can be invalid for a discrete state. |
| C. Short forward fill by feature type | Carry prior observations through short gaps. Use an explicitly defined fallback for unresolved cells. | Can retain recent telemetry or state. | Stale values can hide changes. Feature types, gap limits, and boundary rules need definition. |

**Proposed first choice:** A. Use B or C only if a selected future source has missing values. Do not infer original completeness from finite processed arrays.

## 8. Shared limits for optional filling

1. A training median uses observed training cells only. Stop when none exist. Do not use validation or test values to fit it.
2. A missingness indicator records absence before filling. Define the same indicator columns for all splits. [scikit-learn guidance](https://scikit-learn.org/stable/modules/impute.html)
3. Forward fill uses the last valid observation. Its limit controls consecutive missing cells. A row-count limit is not an elapsed-time limit. [pandas guidance](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html)
4. Proposed conservative boundary rule: reset forward-fill state at each split. Use the fixed training fallback for leading missing cells. Cross-split state carry is not approved here.
5. For timestamped data, a forward-fill rule should also limit elapsed time. Preserve source time gaps. Do not choose a numeric limit without source evidence.
6. Do not use backward fill or two-sided interpolation for the shared offline/online profile. Those operations can use observations that were not available at the prediction time.
7. Feature filtering must not inspect test outcomes to select features. Save a fixed selected schema. Do not remove rows to obtain complete cases.

## 9. Conclusion and open decisions

1. Proposed defaults: check and stop for SMD, IOPS, and NASA; an explicit count-based rule for ICCAD; source verification before an explicit reduced-feature SWaT profile.
2. These proposals do not change the confirmed decisions. Profile rules, selected features, and optional fill limits still require approval.
3. Missing markers, entity mapping, publisher provenance, label mapping, and SWaT source assembly still need source-specific checks.
4. This is a missing-data study. It does not complete the broader dataset exploration plan or prove CLI behavior.

## 10. Check record

1. Read the confirmed preprocessing decisions and earlier dataset findings.
2. Scanned the listed numeric files with NumPy and pandas. Read ICCAD metadata and six selected columns with PyArrow.
3. Compared each proposal with split, sorting, scaling, output, and version decisions.
4. Changed no dataset, code, configuration, or confirmed decision file.

No existing terms were renamed.
