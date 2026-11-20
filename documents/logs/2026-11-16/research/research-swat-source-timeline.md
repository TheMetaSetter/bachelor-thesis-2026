# SWaT source timeline

Date: 2026-11-16.
Status: Local inspection and web source review complete. Local file relationships are verified. The exact download source and assembly script remain unverified. The later approved reconstruction is implemented and checked. All required SWaT data is processed, and the original source folder was removed after saved-output verification.

## Main finding

The inspected source files separated rows by label, not by an earlier training period and a later test period.
Every row in `normal.csv` has label Normal. Every row in `attack.csv` has label Attack.
Their time ranges overlap, although they have no identical timestamps across files.
Timestamp sorting cannot turn these files into the chronological original splits required by the [decisions](../plan/decisions-tsad-preprocess.md).

## Measured ranges

Source at inspection: `data/SWaT/normal.csv` and `data/SWaT/attack.csv`. The original folder has since been removed; the measurements and figures are retained evidence.
Times are parsed with `%d/%m/%Y %I:%M:%S %p`. The source does not specify a timezone.
Source row positions start at zero. Range endpoints are inclusive.

| File | Rows | Earliest time | Latest time | Backward jumps | Repeated timestamps after the first occurrence |
|---|---:|---|---|---:|---:|
| normal.csv | 1,387,098 | 2015-12-22 16:00:00 | 2016-01-02 14:59:59 | 2 | 495,000 |
| attack.csv | 54,621 | 2015-12-28 10:29:14 | 2016-01-02 13:41:11 | 0 | 0 |

1. There are 388,816 normal-file rows within the attack file's minimum-to-maximum time range.
2. There are zero shared timestamps between the two files. Overlapping ranges do not mean identical timestamps.
3. The earlier script treated normal.csv as original training and attack.csv as original test. That selection failed the timeline check. The approved reconstruction below replaces it.

![Source time ranges and source row order](swat-source-timeline.png)

## Blocks inside normal.csv

The two backward jumps separate three source-order blocks.

| Block | Source rows | Count | Earliest time | Latest time |
|---|---|---:|---|---|
| 1 | 0–395297 | 395,298 | 2015-12-28 10:00:00 | 2016-01-02 14:59:59 |
| 2 | 395298–892097 | 496,800 | 2015-12-22 16:00:00 | 2015-12-28 09:59:59 |
| 3 | 892098–1387097 | 495,000 | 2015-12-22 16:30:00 | 2015-12-28 09:59:59 |

1. At row 395298, source time jumps from 2016-01-02 back to 2015-12-22. The missing MV101 block also starts there.
2. At row 892098, source time jumps from 2015-12-28 back to 2015-12-22 again.
3. Blocks 2 and 3 have identical FIT101, LIT101, and LIT301 values at all 495,000 shared timestamps. Other feature values were not compared for equality in this inspection. This is not proof that every complete row is identical.
4. MV101 has 991,800 missing values. This agrees with the earlier [missing-data inspection](research-missing-data-options.md). The approved seven-feature exclusion remains necessary, but it does not solve source assembly.

## Series near the first attack

The figure uses retained sensors FIT101, LIT101, and LIT301 in their source units.
Blue shows block 1 of normal.csv; red shows attack.csv.
The window is ten minutes before to twenty minutes after the first Attack-labeled observation.
The vertical line marks 2015-12-28 10:29:14.
Values are not scaled or filled. Lines break at gaps longer than two seconds. Dots show every tenth observation.
The supplied labels change from Normal to Attack and back to Normal within the same time period.
The plotted signal shapes do not independently prove whether those labels are correct.

![Three source series near the first attack](swat-series-first-attack.png)

## Approved reconstruction and completion

1. Use block 3, the trimmed early normal recording, as the original training source. Reserve its last 20% for validation.
2. Combine block 1 with attack.csv and sort by timestamp to reconstruct the complete later test source. Keep both Normal and Attack labels.
3. Saved counts are 396,000 training, 99,000 validation, and 449,919 test observations. Test data contains 395,298 Normal and 54,621 Attack observations.
4. Keep the seven approved feature exclusions and all timeline checks. Do not change timestamps or add general duplicate removal.
5. Processing and saved-output verification are complete. The original SWaT folder was removed, releasing 844,392,461 bytes. Two prepared output versions remain.
6. See [the confirmed decisions](../plan/decisions-tsad-preprocess.md) for implementation checks and source-selection rules.

## Checks and limits

1. Counts and ranges were calculated from all rows of both files. Only timestamp, labels, three plotted retained sensors, and MV101 were loaded.
2. Both figures were opened and visually checked. Plot labels match the calculated dates and source row boundaries.
3. The first inspection did not read merged.csv. The later review below checks its relationship to the other files.

## Web source review and full-row checks

Reviewed on 2026-11-16 with the default web search tool.

1. The [official iTrust description](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/) defines separate normal-operation and attack-operation periods. Its 2018 update states that normal version 1 is normal version 0 with the first 30 minutes removed. These are versions of the same normal recording, not independent observations to combine.
2. The [SWaT-Curated uploader](https://huggingface.co/datasets/minervar/SWaT-Curated/blob/main/README.md) distributes files with these three names and describes merged.csv as containing both normal and attack phases. This is a repackaged dataset. Matching filenames do not establish that the local files came from that upload. Its published file sizes differ from the local sizes, and no matching remote checksums were established.
3. Local full-row check: filtering merged.csv by label reproduces normal.csv and attack.csv, in their original row order, across all 53 columns. Text values match after removing surrounding whitespace. Numeric values match with relative and absolute tolerance 1e-12; some decimal representations differ. Counts are 1,387,098 Normal rows and 54,621 Attack rows. This establishes file relationships, not which file was created first.
4. Local full-row check: normal block 2 after its first 1,800 rows is identical to all of block 3 across all 53 columns after removing surrounding whitespace. This strengthens the earlier three-sensor check.
5. Inference: the local collection is consistent with combining a later attack-operation recording, normal version 0, and normal version 1, then separating rows by label. The exact original files and assembly script were not located. Do not describe this inferred history as verified publisher provenance.

```text
Likely recording content                 Local file relationship (verified)
Later period: 395,298 Normal rows  ─┐
Later period:  54,621 Attack rows  ─┼── merged.csv ── label Normal ── normal.csv
Early normal: 496,800 rows         ─┤             └─ label Attack ── attack.csv
Early normal: 495,000 rows         ─┘
             = previous early block without its first 30 minutes
```

The later period totals 449,919 observations. Its Normal rows must remain part of evaluation if this recording is reconstructed as a test source. Using those rows as original training and the Attack rows as test would split one time period by label.
Keeping both early normal versions would repeat 495,000 observations.
These findings supported the source-selection repair that the user later approved. The reconstruction is now implemented and checked.

## Remaining verification

1. Establish an exact source archive, source code, or publisher checksum match before claiming the inferred original-file names as confirmed.
2. The user selected reconstruction from the local files. Obtaining verified publisher files remains optional future provenance work; it does not block the completed local processing.
3. The cause of the seven missing feature columns remains unverified. Preserve the approved feature exclusions until a separate decision changes them.
