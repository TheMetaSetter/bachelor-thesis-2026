# Next-version cleanup

Date: 2026-11-16
Status: cleanup complete for confirmed unnecessary paths. Method migration remains pending.
Scope: bachelor-thesis-2026 only.

## Later status: 2026-11-16

1. All required data for the five selected datasets is processed. Prepared outputs replace the original source folders for later method execution.
2. The original SMD, IOPS, NASA, ICCAD, and SWaT source folders were removed after preprocessing and saved-output checks. The results below describe the earlier cleanup stage.
3. NASA output contains 80 selected channels; P-2 and T-10 are excluded. The earlier inventory of 82 source channels remains a historical count.
4. The approved SWaT reconstruction is implemented and checked. The local timeline blocker is resolved; exact publisher provenance remains unverified.

## Rules

1. Preserve all specifications and detailed mathematics files, including thesis sources and supporting assets.
2. Preserve the self-contained preprocessing script and prepared outputs for the five selected datasets. Original inputs were retained until preprocessing and saved-output checks passed.
3. Keep current method code, configs, tests, dependencies, and reference implementations needed until their proposed replacements exist.
4. Remove unnecessary paths in place. Do not create local archives or recovery copies.
5. Git does not cover all ignored local data. Record every deletion and its reason.

## Phase 1: inventory

### Stage 1.1: establish the boundary

1. [x] Read the new tree and project instructions. Tool: file reader.
2. [x] Check the current Git state. Tool: Git.
3. [x] Read the earlier cleanup lessons. Tool: file reader.
4. [x] Identify protected files and dependencies. Tools: Python and text search.
5. [x] Save the removal list with reasons and sizes. Tool: CSV.

## Phase 2: remove unnecessary paths

### Stage 2.1: validate the list

1. [x] Check source readers against selected data paths. Tools: Python and script inspection.
2. [x] Check protected local document links. Tool: Python.
3. [x] Check redundant archive contents. Tools: zipfile and streamed checksums.

### Stage 2.2: apply removal

1. [x] Remove only listed files. Tool: Python.
2. [x] Remove empty folders outside tool state and environments. Tool: Python.

## Phase 3: verify

### Stage 3.1: check retained work

1. [x] Verify protected file hashes. Tool: hashlib.
2. [x] Verify the five dataset source groups. Tool: preprocessing script.
3. [x] Verify the existing SMD output can be reused. Tool: preprocessing script.
4. [x] Check source and test collection. Tools: Python and pytest.
5. [x] Review Git deletions and document links. Tools: Git and Python.
6. [x] Record removed paths, space released, and retained migration needs. Tool: Markdown.

## Results at the cleanup stage

1. Removed 1,059 unique file paths and 75 empty folders. Removed files totaled 11,050,959,722 logical bytes, approximately 11.05 GB.
2. Created no local archive or recovery copy. Removed the obsolete compressed reporting archive.
3. Preserved all identified specifications and mathematics files. Hash checks passed for 390 protected or required inputs before this work log was updated.
4. The second review restored seven notes from Git because their content was a specification or mathematical reference. They are excluded from the final removal list.
5. Kept all five preprocessing source groups: one ICCAD entity, 28 SMD entities, 29 IOPS entities, one SWaT entity, and 82 NASA channels.
6. Kept two small AnomalyArchive files required by current tests or tools. Kept SWaT's merged source for the current anomaly-span audit. These are migration dependencies, not additional preprocessing datasets.
7. Kept current method code, configs, tests, demos, needed reference implementations, Git history, tool settings, and the working Python environment. The proposed replacements do not exist yet.
8. Removed unreferenced reference code outside the retained method scope. Protected mathematics documents and required licenses remain in those reference folders.
9. Removed old per-window result traces. Kept the 99-run compact online metrics, offline metrics, tables, and thesis mathematics. Updated the reporting README to explain the retained summary files.
10. Verified the extracted ICCAD files against the archive's member sizes and CRC32 values before removing the redundant archive. The selected reader still has its pivoted table and annotations.

## Checks at the cleanup stage

1. Project suite: 513 passed, 2 skipped, 33 warnings in 46.80 seconds. No project test files were deleted.
2. Separate preprocessing suite: 37 passed.
3. SMD `machine-1-1`: the complete matching version passed the script's reuse checks.
4. All listed input paths for the five readers remain present.
5. No new broken file links were found in retained project documents.
6. No files under `src/`, `tests/`, `configs/`, or `demo/` were deleted.
7. Removed test-generated project caches again after verification.

The [file-level removal list](../detail/cleanup-removals.csv) records each deleted path, its size, Git tracking status, and reason. It is an audit record, not a backup.

## Remaining work

1. Implement and verify the proposed next-version modules before retiring current method implementations and their required callers.
2. Verify remaining source provenance before making publisher-provenance claims. The local SWaT timeline conflict is resolved.
3. Retained source arrays and Git history were not compacted or rewritten. No claim is made that every future module in the proposed tree is implemented.

