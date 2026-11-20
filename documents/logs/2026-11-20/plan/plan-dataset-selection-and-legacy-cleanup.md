# Dataset selection and legacy cleanup

Follow the [decision record](../detail/dataset-selection-and-cleanup-decisions.md)
and Phase 11 of the [runtime implementation plan](../../2026-11-16/plan/plan-implement-runtime-signatures.md).

## Phase 1 — Record the starting state

### Baseline and scope

1. [x] Record Git status. Tool: Git.
2. [x] Count source files and folders. Tool: Python.
3. [x] Record the six prepared manifest hashes. Tool: SHA-256.
4. [x] Run the full test suite. Tool: pytest.
5. [x] Run the focused numerical tests. Tool: pytest.
6. [x] Trace retained callers and legacy runtime paths. Tool: rg.

The branch is `simplifying` at `034833af`. The checkout has 136 Python
files in 15 source subfolders. It has 1,220 deletions, 17 modifications,
and 53 untracked status entries before this task. The baseline has 165
passes and two failures caused by absent legacy YAML. The focused
numerical suite has 64 passes. Work in this checkout because the new
runtime includes changes that are absent from HEAD.

## Phase 2 — Add version selection

### Discovery

1. [x] Check default priority, invalid manifests, and tied times. Tool: pytest.
2. [x] Read candidate metadata and order valid versions. Tool: Python.
3. [x] Check the real SWaT default. Tool: Python.

### Screen

1. [x] Pin the Textual dependency. Tool: setuptools.
2. [x] Check keyboard selection, Save, and Cancel at two terminal sizes. Tool: Textual Pilot.
3. [x] Implement one version selector per dataset. Tool: Textual.

### Save

1. [x] Check CLI success, cancellation, missing data, corruption, retry, and write failure. Tool: pytest.
2. [x] Resolve profiles with the existing planning rules. Tool: Python.
3. [x] Add the interactive and noninteractive paths. Tool: Python.
4. [x] Verify each selected version before writing. Tool: Python.
5. [x] Save profile paths and experiment dataset names under the project edit lock. Tool: Python.
6. [x] Run the CLI tests. Tool: pytest.

## Phase 3 — Retire the old runtime

### Move retained code

1. [x] Move baseline modules into methods. Tool: Python.
2. [x] Combine the four protocol contexts. Tool: Python.
3. [x] Move live MUSTARD, CANDI, online triage, augmentation, and budgeted VUS code. Tool: Python.
4. [x] Update package mappings. Tool: setuptools.
5. [x] Run the focused tests. Tool: pytest.

### Remove cleared paths

1. [x] Check each candidate for retained callers. Tool: rg.
2. [x] Record the cleared paths. Tool: CSV.
3. [x] Remove the cleared legacy paths. Tool: file editor.
4. [x] Remove folders that became empty. Tool: rmdir.
5. [x] Verify the source tree and line limits. Tool: Python AST.

## Phase 4 — Verify the installed flow

### Installed package and real data

1. [x] Install a wheel in an isolated test environment. Tools: setuptools and pip.
2. [x] Check the installed command from outside the checkout. Tool: tsad.
3. [x] Save a dataset selection with keyboard input. Tool: tsad.
4. [x] Show a one-entity SMD MUSTARD plan. Tool: tsad.
5. [x] Validate the plan. Tool: tsad.
6. [x] Run O0 and A0 with smoke limits. Tool: tsad.
7. [x] Verify the matching offline parent. Tool: Python.
8. [x] Verify smoke exclusion in the report. Tool: tsad.
9. [x] Compare prepared manifest hashes. Tool: SHA-256.

### Final review

1. [x] Run the full retained suite. Tool: pytest.
2. [x] Check lint and whitespace. Tools: Ruff and Git.
3. [x] Update the CLI guide, tree, and decision record. Tool: Markdown.
4. [x] Review the change against the starting state. Tool: Git.
5. [x] Record the final evidence. Tool: Markdown.

## Results

1. The source tree changed from 136 Python files in 15 subfolders to
   37 Python files in two subfolders. All source size checks passed.
2. The cleanup map records 206 retired paths. The five active dataset
   profiles, the shared metric settings, and all prepared versions remain.
3. All 143 retained tests passed. The checks included keyboard interaction,
   profile permission bits, rollback, and saved attempt settings.
4. Seeded augmentation matched the earlier output, labels, masks, metadata,
   and RNG state in two checks. The retained method and metric tests passed.
5. The isolated wheel ran from outside the checkout. It saved a dataset
   selection with Tab and Enter. The one-entity SMD MUSTARD O0 and A0 smoke
   runs passed. The online run used the matching offline checkpoint.
   The normal report excluded the smoke results.
6. All six prepared manifest hashes matched the baseline.
7. Ruff passed for the runtime, preprocessing, and tests. Git whitespace
   and local document link checks passed. The retained documentation hook
   has an earlier import-style finding.
8. The existing environment still reports five missing VUS dependencies
   in pip check. The budgeted VUS and other exercised metric tests passed.
9. Source paths contribute to run identity. New runs can require fresh
   offline checkpoints. No active run required a migration.

[Removal list](../detail/runtime-cleanup-paths.csv) ·
[User flow and diagram](../detail/dataset-selection-and-cleanup-decisions.md)
