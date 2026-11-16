# Plan: explore all local datasets

Date: 2026-11-09  
Status: plan only; exploration has not been executed.

## Goal

Describe every dataset or potential dataset currently in `data/`, including archived and unlabeled sources. Record entity names, time-point counts, column names and types, missing values, labels, and original splits. Distinguish facts visible in local files from inferred meanings and unknowns.

## Scope and output

Inspect every dataset or potential dataset under `data/`, including archives and unlabeled sources. Inventory hidden paths and distinguish datasets from caches and operating-system metadata.

Write one report: `documents/logs/<execution-date>/research/research-local-datasets.md`. Use the execution date in `yyyy-mm-dd` format. Inspect an existing report before overwriting it.

## Findings format

Use one Markdown report with:

1. **Overview:** one table with dataset, source, number of entities, label availability, original train/test split status, and scan status.
2. **One section per dataset:** a short table or list showing entity names and time-point counts, columns and types, missing values, timestamps, labels, and split evidence. Use ranges or compact tables for large collections while preserving access to complete entity details.
3. **Limitations:** unreadable or sampled sources, unknown label meanings, and uncertain original splits.

For very large entity lists, write a separate CSV only when needed for complete details, and link it from the relevant dataset section. Label sampled findings clearly. Distinguish unknown values from measured zeroes, and cite source paths or archive members.

## Current inventory to reconcile

1. `AnomalyArchive/`, `IOPS/`, `NASA/`, `SMD/`, `SWaT/`, and `ServerMachineDataset/`.
2. `ibm-cloud-console-anomaly-dataset-iccad/` and `extra-industrial-data/`.
3. `TSB-AD-U.zip`, `TSB-AD-M.zip`, `tfinance`, and `tfinance.zip`.
4. `Graph Based Fraud Detection Dataset for Financial/` and its ZIP archive.
5. Hidden cache and metadata such as `.cache/` and `.DS_Store` require classification. A fresh inventory may discover more sources; add each to the checklist and account for it.

These are filesystem observations, not assertions that every item is a labeled time-series dataset. The earlier dataset tree sampled selected sources only; it is not a full audit.

## Rules

1. Read data without altering, renaming, extracting into, or deleting from `data/`. Stream archive members when possible. If a reader requires extraction, use a unique temporary directory outside `data/` and remove only that directory after the check.
2. Inspect schemas and aggregate values; do not copy raw records, credentials, or entire datasets into reports. Record counts and short sanitized examples only when needed to explain structure.
3. Prefer exact full-file counts. Process large files in chunks or stream rows. Mark sampling, failures, unavailable counts, denominator, and reason explicitly. Do not label a sample-based result as a complete result.
4. Keep reported source splits separate from proposed splits. Do not construct a new split or fit a scaler during exploration. Do not treat file names alone as proof of original publisher-provided splits.
5. Determine label meaning from local schema, metadata, documentation, or loader code. Record unknown semantics and weak interval labels explicitly. Do not assume labels are point-level or that numerical `0` and `1` have a known meaning.
6. Avoid loading untrusted serialized objects with pickle-based deserialization. Inspect format and metadata first; if safe inspection is unavailable, record the limitation rather than executing object code.
7. Do not install dependencies or download data. If a format cannot be read with available tools, record the missing reader and continue with other sources.

## Phase 1: inventory and classify

Tools: `tree` or `find`, `rg --files`, `file`, `unzip -l`, shell, Python standard library.

1. [ ] Resolve the absolute `data/` path and record the local date and repository revision when available.
2. [ ] Inventory all paths, including hidden paths, without opening or changing file contents.
3. [ ] Record each file's relative path, size, and detected format.
4. [ ] List member paths and sizes for each ZIP without extracting it.
5. [ ] Group source files into candidate datasets using local directory names, metadata, and observed structure.
6. [ ] Classify each remaining file as dataset, annotation, documentation, archive, duplicate candidate, cache, metadata, or unknown.
7. [ ] Add every discovered dataset or candidate to the coverage checklist in the research report.
8. [ ] Record inaccessible or corrupt paths and continue with other paths.
9. [ ] Verify every dataset-bearing file and archive member is covered or has a recorded reason for exclusion or failure. Store a separate inventory only if it is needed to make coverage verifiable.

## Phase 2: inspect structure and entities

Tools: Python standard library, available NumPy, pandas, SciPy, Parquet reader, and local project loaders or documentation as read-only references.

1. [ ] Record the available reader and version for each encountered format.
2. [ ] Inspect each dataset's local documentation, headers, metadata, and loader code when present.
3. [ ] Record the source location of each dataset's data and annotations.
4. [ ] Record each table's column names, stored types when available, and observed value types.
5. [ ] Record each array's shape and dtype; identify dimension meanings only when supported by evidence.
6. [ ] Identify each dataset's entity naming rule and cite its evidence.
7. [ ] List every identifiable entity and its source files or archive members.
8. [ ] Record each entity's feature count or state that feature dimensions are unknown.
9. [ ] Count the time points for each entity and each available split; keep duplicate timestamps and rows visible in separate counts.
10. [ ] For graph or other non-series sources, record node, edge, snapshot, timestamp, and feature structure where available; mark unsupported concepts as not applicable or unknown.
11. [ ] Match label files and interval annotations to entities only when local evidence supports the mapping.
12. [ ] Record sources that lack a reliable entity or time-point interpretation as unresolved rather than forcing them into the table.

## Phase 3: quality, time, labels, and splits

Tools: streaming Python readers and the available format-specific libraries from Phase 2.

1. [ ] Count missing values by column and by entity/split when feasible; record both count and denominator.
2. [ ] Distinguish null, empty string, recognized missing markers, non-finite numbers, parse failures, and absent observations where the format permits.
3. [ ] Record constant columns and columns with inconsistent types or unexpected values.
4. [ ] Identify timestamp columns or implicit row-order indexing for each source, with evidence.
5. [ ] Record timestamp range, timezone when known, ordering, duplicates, and observed interval distribution; mark irregular sampling and gaps without inventing a nominal frequency.
6. [ ] Record label location, representation, allowed values, and known semantics.
7. [ ] Count labeled normal and anomalous points when labels are confirmed point-level; otherwise record interval counts or leave point counts unknown.
8. [ ] Record anomaly fraction with its denominator and split where calculable.
9. [ ] Record original train, validation, and test assignments only when source evidence establishes them; separately note local file organization and unknown provenance.
10. [ ] Check entity and label lengths or interval bounds when the format allows direct comparison.
11. [ ] Identify potential overlap or duplication across splits and local dataset representations; report evidence without assuming two datasets are equivalent.
12. [ ] Record dataset-specific limitations, especially weak interval labels, absent timestamps, incomplete files, graph data, and ambiguous label mappings.

## Phase 4: source-specific checks

Tools: the same readers, with ZIP streaming and chunked processing when supported.

1. [ ] Inspect `AnomalyArchive/` filename-derived split and interval metadata against the local parser and actual series lengths.
2. [ ] Inspect `IOPS/` paired files and establish the unnamed columns' meanings only from supporting local evidence.
3. [ ] Inspect `NASA/` channel files and `labeled_anomalies.csv`; report entity coverage and interval-to-series alignment.
4. [ ] Inspect both SMD representations separately; compare entity mapping, shapes, labels, and splits before claiming equivalence.
5. [ ] Inspect `SWaT/` headers, label values, timestamps, and overlap among normal, attack, and merged files.
6. [ ] Inspect ICCAD Parquet schemas, series identifiers, time ranges, and anomaly intervals; distinguish event-level labels from point-level labels.
7. [ ] Inspect every data member in both TSB archives without extracting into `data/`; record per-member label columns, entities, rows, and split evidence.
8. [ ] Inspect each file in `extra-industrial-data/` and report its actual format, structure, and label availability.
9. [ ] Inspect `tfinance` and its archive format without unsafe deserialization; record whether graph, temporal, feature, and label structure can be established safely.
10. [ ] Inspect the financial graph folder and ZIP to distinguish data from documents or images; report any safely readable data and labels.
11. [ ] Inspect all newly discovered dataset candidates using the same phases; record any source that cannot be opened and why.

## Phase 5: review and deliver

Tools: Python, CSV reader, Markdown review, `git diff --check`.

1. [ ] Reconcile the file inventory with the dataset coverage checklist.
2. [ ] Confirm each entity count points to a specific source and split or is explicitly unknown.
3. [ ] Confirm each missing-value rate includes its denominator and scan scope.
4. [ ] Confirm each split statement distinguishes observed local layout from evidence of the original split.
5. [ ] Confirm graph, interval, and unclassified sources are not described as ordinary labeled time series without evidence.
6. [ ] Check aggregate totals against source-level counts and any optional detail CSV.
7. [ ] Run `git diff --check` for documentation whitespace.
8. [ ] Verify the source data and archives were not modified.
9. [ ] Summarize datasets covered, unreadable sources, sampled-only findings, and unresolved semantics.

## Risks and handling

1. Large archives and Parquet files may require substantial time or memory. Stream or chunk, record progress and failures, and avoid loading all data at once.
2. Missing optional readers may prevent exact counts. Report the limitation and continue; do not install dependencies without authorization.
3. Graph structures and interval labels may not fit a simple table. Describe their actual structure instead of inventing rows.
4. A local folder named `train` or `test` does not alone establish that the publisher supplied the split. Record provenance separately.
5. Source data stays read-only. If a generated report is rejected, remove only the new report artifacts after identifying them; do not touch source data.
