# Baseline restoration

The simplification wrongly removed offline and online scientific baselines. Its source reduction claims are withdrawn.

Restored all tracked files under src/, scripts/, configs/, tests/, and demo/ from revision 753ad03ed60b2d41b078e53bf421429bb0bd6c0f, together with README and architecture notes. Git confirms exact tracked-file equality. This rolls back the core simplification attempt rather than mixing old baseline code with changed contracts.

Restored paths include traditional IForest, KMeans-AD and STUMPY; MOMENT adapter; online frozen, adaptive, pre-TTA, CANDI and M2N2; RedLamp baseline code; their supporting runtime, configurations and tests.

The first full test run stopped during collection because demo/ was missing; demo/ was then restored. The next run had 507 passed, 2 skipped and one missing local data-config failure. Recovered four local data configurations from the backup. All 7 comparative-runner tests then passed. Final full current suite: 508 passed, 2 skipped, 33 warnings in 48.77 seconds.

Current files were backed up before restoration. The backup location is recorded in baseline-restoration-backup.txt. Ambiguous flattened experiment configuration copies are retained in that backup. Existing user edits to AGENTS.md and .gitignore were preserved. No full real-data experiment was run.
