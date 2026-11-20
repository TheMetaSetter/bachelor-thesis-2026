# Prepared data location

Date: 2026-11-20

1. Preprocessing now saves versions in `data/preprocessed/<dataset>/<version_id>/`. Use `--output-root` to select a different folder.
2. All five data profiles use the new location and keep their version IDs. SWaT keeps version `67e1e175…`. Restored the required SMD window size of 20. Updated the CLI specification, preprocessing decisions, and directory trees.
3. An SMD sample verified the default location, repeat use, and the override. Both SMD smoke runs passed with the moved data. All six saved manifest checksums matched the execution record.
4. All 25 focused tests passed. The full suite had 165 passes and two failures. Both failures refer to missing legacy experiment configuration files.
