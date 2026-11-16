# Local dataset exploration

Date: 2026-11-10

This continues the 2026-11-09 scan. The report remains incomplete.

**Coverage:** Local files and ZIP members were inventoried. CSV, text, and NumPy counts below come from full reads unless marked otherwise. ICCAD Parquet row and null counts come from footer metadata. The scan is incomplete for timestamps, original split provenance, several label semantics, MAT values, and graph data; see limitations. Existing data was not intentionally modified.

Detailed source and member counts, column types, and null counts: [dataset details](../../2026-11-09/research/local-dataset-details.csv). ICCAD’s 117,449 pivoted column names, types, and footer null counts: [ICCAD columns](../../2026-11-09/research/iccad-column-details.csv).

| Source | Observed entities or files | Time points / rows | Labels and splits |
| --- | --- | --- | --- |
| `AnomalyArchive` | 250 files/members | 19353766 | 250 series; interval indices in filenames; original split provenance unknown |
| `IOPS` | 58 files/members | 5922913 | 29 train/test filename pairs; unnamed label semantics unconfirmed |
| `NASA` | 165 files/members | 82 | 82 train and 82 test arrays; 82 annotation rows; local split provenance unknown |
| `SMD` | 3 files/members | See array shapes in details | three arrays; one test label array; entity mapping unknown |
| `ServerMachineDataset` | 84 files/members | 2125245 | 28 train/test/test_label filename triples; provenance unknown |
| `SWaT` | 3 files/members | 2883438 | normal, attack, merged; merged overlaps the other files |
| `TSB-AD-U.zip` | 870 files/members | 33768217 | 870 archived CSVs; Label values 0/1; publisher split provenance unknown |
| `TSB-AD-M.zip` | 200 files/members | 21552097 | 200 archived CSVs; Label values 0/1; publisher split provenance unknown |
| ICCAD | Two Parquet tables; 25 anomaly intervals | Pivoted 39,365 rows; unpivoted 413,241,248 rows | Interval labels; point mapping not established |
| Extra industrial | Two MAT files and one CSV | CSV 3,811,152 rows; MAT shapes in details | Labels and entity semantics unconfirmed |
| T-Finance | Binary file and ZIP | Unknown | Graph/serialized content not deserialized |
| Financial graph folder and ZIP | Documents and images inspected by inventory | Not established | Dataset and labels not established |

## Findings

1. **AnomalyArchive:** 250 text files; 19,353,766 time points; no parser-detected null cells. The local parser identifies train and anomaly indices from filenames. A strict `train_end < anomaly_start < anomaly_end <= length` check holds for 247 of 250 files; three have equal anomaly start and end, which the current parser rejects. Nine whitespace-delimited single-line files were counted by values rather than lines.
2. **IOPS:** 58 files, 5,922,913 rows; no parser-detected null cells. The two columns are unnamed; label meaning is unconfirmed.
3. **NASA:** 82 annotation rows and 82 test arrays. All annotation rows matched a test array, declared `num_values`, and the checked interval bounds. Training arrays exist for 82 channels. Array shapes and numeric non-finite counts are in the details; annotations are intervals, not automatically verified point labels.
4. **SMD:** `SMD_train.npy` is 708,405 × 38; `SMD_test.npy` is 708,420 × 38; test labels have length 708,420, with 678,976 zeros and 29,444 ones. Array entity boundaries and label semantics require source verification. The separate `ServerMachineDataset/` has 28 files each under train, test, and test_label; its equivalence to these arrays was not established.
5. **SWaT:** `normal.csv` has 1,387,098 rows, `attack.csv` 54,621, and `merged.csv` 1,441,719. In both normal and merged, seven named columns each have 991,800 parsed nulls. The attack file has none reported by the parser. The merged label counts equal the normal and attack row counts, but row-by-row identity was not checked; do not sum these as independent data.
6. **TSB archives:** U has 870 CSV members and 33,768,217 rows; M has 200 CSV members and 21,552,097 rows. No parser-detected null cells in these members. Label counts are U: 32,854,956 zeros and 913,261 ones; M: 20,617,686 zeros and 934,411 ones. The full source/member inventory is in the detail CSV. Label meanings and original split provenance need independent evidence.
7. **ICCAD:** Parquet footer statistics report 4,210,099,272 null cells across the pivoted table (39,365 rows × 117,449 columns); the nine unpivoted columns report zero nulls across 413,241,248 rows. These are stored footer statistics, not a full decoded-value scan. `anomaly_windows.csv` contains 25 interval rows; no point-label mapping was established.
8. **Extra industrial:** the tuna CSV has a three-line metadata preamble, 3,811,152 parsed rows, seven float columns, and no parser-detected nulls. MAT metadata reports `DNA_C1` shape 1 × 100,000,000 and `Human_Chromosome_0ne_Length_is_225934570` shape 1 × 225,934,570, both double. MAT values and missingness were not scanned.
9. **T-Finance and financial graph materials:** the `tfinance` binary and its archive were inventoried but not deserialized, avoiding unsafe object loading. The financial-graph folder and archive contain documents and images in the observed listings; their data and labels were not established.

## Additional findings on 2026-11-10

1. **Inventory:** [source and archive-member inventory](source-inventory.csv) lists 590 local files and 1,074 ZIP members, including hidden files. File type in this inventory is based on the name extension, not a content check. No ZIP listing failed. Classification and coverage of each member still need reconciliation.
2. **SWaT timestamps:** all timestamps parsed using `day/month/year hour:minute:second AM/PM`; timezone was not specified. In file order, `normal.csv` has two backward steps among 1,387,097 adjacent intervals, and `merged.csv` also has two among 1,441,718 intervals; `attack.csv` has none among 54,620. There are no adjacent equal timestamps. This does not rule out non-adjacent duplicates. The most common interval is one second in all three files. Normal ranges from 2015-12-22 16:00:00 to 2016-01-02 14:59:59; attack ranges from 2015-12-28 10:29:14 to 2016-01-02 13:41:11. Longer intervals occur; no nominal sampling frequency is assumed.
3. **IOPS:** the 58 files form 29 matching train/test filename pairs. The second column contains 5,788,799 zeros and 134,114 ones across all files. The local `scripts/analysis/summarize_anomaly_span_lengths.py` treats the second column of test files as labels, with nonzero values as anomalous. This establishes the local code's interpretation, not original publisher semantics. [Per-file column counts](iops-column-details.csv) give the split-specific numbers.
4. **ICCAD:** the entire pivoted timestamp column has 39,365 values, no nulls, duplicate values, or backward steps. Of 39,364 adjacent intervals, 39,351 are 300 seconds, 11 are 600 seconds, and two are 900 seconds. The local `scripts/plot_iccad_a8.py` interprets `interval_start` as Unix seconds in UTC; under that interpretation, the range is 2024-01-22 19:25:00 to 2024-06-07 13:00:00 UTC. All 25 annotation windows overlap that overall range and none has a start after its end; this does not establish a per-series or point-level label mapping. The unpivoted Parquet footer has matching minimum and maximum timestamp values across 3,363 row groups; its rows were not decoded in full.
5. **SMD representations:** the combined array has 708,405 training rows and 708,420 test rows; the 28 separate files total exactly the same numbers for each corresponding split. A search for exact matching contiguous entity blocks found no match for any of the 28 feature files in either array. Float representation or preprocessing differences have not been ruled out, so these counts alone do not prove equivalence or non-equivalence. The label comparison was interrupted and is incomplete.
6. **Data integrity:** compared with the previous scan's inventory, all 590 source files have the same recorded sizes and modification times; no path was added or removed. This is not a byte-level comparison.

## Limitations and remaining work

1. This is a partial execution of the exploration plan: timestamp ranges, ordering, sampling intervals outside the SWaT and ICCAD checks above, split overlap checks, and full source-specific validation are not complete.
2. ICCAD statistics rely on metadata; decoded values, entity names and per-series anomaly alignment were not fully scanned. The pivoted timestamp column was decoded in full as described above. MAT arrays were inspected only by schema. The T-Finance graph object was not loaded.
3. Archive members were read without extraction. A few files are documentation, images, notebook, binary, or metadata and were inventoried rather than treated as time-series rows. Hidden cache content was excluded.
4. `NaN` and empty CSV values are counted by pandas defaults; named markers and absent observations were not comprehensively classified. Counts are not evidence of original publisher-provided splits or evaluation readiness.
5. Data file size and modification times were compared before and after; this does not prove byte-level equality.

Tools: Python in `.venv`, NumPy 2.4.4, pandas 2.3.3, SciPy 1.16.3, PyArrow 23.0.1. Repository revision: `3041e44b`, branch `simplifying`.
