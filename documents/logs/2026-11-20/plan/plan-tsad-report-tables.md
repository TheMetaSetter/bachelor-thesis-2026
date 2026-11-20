# Implementation plan for `/report` tables

Date 2026-11-20. Source [report table design](../detail/tsad-report-table-design.md) and [rendered preview](../detail/report-preview/report-table-preview.pdf).

**Goal** — Produce grouped HTML tables from completed runs in the active experiment. Support normal and smoke reports through the existing command screen.

**Approach** — Keep result selection and aggregation in `src/reporting.py`. Use small functions that render the aggregate as standalone HTML. The CLI forwards the report mode. The terminal user interface (TUI) uses the existing slash resolver and command screen.

**Tools** — Python 3.12 or later, pytest, Textual Pilot, HTML and CSS, a browser, and Git. Use the Python standard library for HTML rendering.

Complete the checklist in order. Record the result of each check before the next phase. Compare failures with the baseline from Phase 1. See the [execution record](../detail/tsad-report-tables-execution.md).

## File map

| File | Work |
| --- | --- |
| [src/reporting.py](../../../../src/reporting.py) | Select runs, aggregate saved results, render HTML, and write reports. |
| [src/cli.py](../../../../src/cli.py) | Add the report mode option and display paths or the empty report message. |
| [src/tui.py](../../../../src/tui.py) | Update help. Reuse slash routing and the command screen. |
| [tests/test_reporting.py](../../../../tests/test_reporting.py) | Check discovery, stored results, coverage, and HTML. |
| [tests/test_cli.py](../../../../tests/test_cli.py), [tests/test_tui.py](../../../../tests/test_tui.py) | Check command arguments, output, and exit status. |
| [README.md](../../../../README.md) | Document normal and smoke report commands and output paths. |

## Phase 1 — Record the baseline

### Stage 1 — Check the current contract

1. [x] Read the design and the [current report sequence](../detail/tsad-report-sequence.md). Tool: Markdown.
2. [x] Check the requirements for normal and smoke isolation in the [CLI design](../../../specs/desired-cli-20261109.md). Tool: Markdown.
3. [x] Trace `report_experiment` through `find_runs`, `_split_record`, and `aggregate_table`. Check the existing aggregate keys `columns`, `rows`, and `cells`. Tool: source inspection.
4. [x] Record the working tree status for the files in the file map. Include untracked files in the record. Tool: Git.
5. [x] Run `.venv/bin/python -m pytest -q`. Record the result before edits. Tool: pytest.

## Phase 2 — Select and aggregate saved results

### Stage 2 — Add fixture checks

1. [x] Extend `test_report_formats_shared_precision_and_excludes_smoke` in `tests/test_reporting.py`. Use seeds `[1, 2]` and distinct normal and smoke values. Assert normal FPR `0.1235` with coverage `1/2`, smoke FPR `0.9877` with coverage `1/1`, and separate output directories. Keep the precision override check. Tool: pytest.
2. [x] Add `test_report_requires_successful_runs`. Use failed or running records and a successful record for a phase absent from the report profiles. Assert an empty list and no report files. Add a successful matching run without a split file. Assert that its report has `N/A`. Tool: pytest.
3. [x] Add `test_report_rejects_invalid_saved_result`. Check a split identity mismatch and a score file hash mismatch through `report_experiment`. Assert the validation errors from `_split_record`. Tool: pytest.
4. [x] Run `.venv/bin/python -m pytest -q tests/test_reporting.py`. Record which new assertions fail. Tool: pytest.

### Stage 3 — Select the mode and keep the current aggregate

1. [x] Add `smoke: bool = False` as a keyword-only parameter to `report_experiment(path: Path, *, project_root: Path, smoke: bool = False) -> list[Path]`. Tool: Python.
2. [x] Select the normal root `outputs/experiments/NAME/` or the smoke root `outputs/smoke/NAME/`. Pass the corresponding `execution_mode` and experiment ID to `find_runs`. Tool: Python.
3. [x] Keep successful run records with resolved settings. Select records by profile phase and dataset. Write reports only for matching groups. Return an empty list when there is no group. Tool: Python.
4. [x] Use the definition seed list for normal reports and its first entry for smoke reports. Keep the existing default `[6]` when the definition has no seed list. Tool: Python.
5. [x] Reuse `_split_record` for each selected split. Keep its identity and hash checks and its empty metrics for a missing split file. Tool: Python.
6. [x] Reuse `aggregate_table(runs, reporting: dict, *, expected_seeds) -> dict` for each profile and dataset. Keep its finite-value check, latest valid attempt per seed, and incompatible configuration error. Keep the CSV formats and JSON schema. Tool: Python.
7. [x] Run `.venv/bin/python -m pytest -q tests/test_reporting.py`. Expect all reporting tests to pass. Tool: pytest.

## Phase 3 — Produce the HTML tables

### Stage 4 — Check the output contract

1. [x] Add `test_detection_html` in `tests/test_reporting.py`. Cover both phases, multiple entities, a single entity, and a custom metric selection with two splits. Assert metric order, one row per method and reporting variant, equal column definitions, matching header spans, and the position or absence of Average. Tool: pytest and `html.parser`.
2. [x] Add `test_html_coverage_and_ranks`. Assert `N/A` for missing values, a count such as `(1/2)` for partial coverage, and Average `0.3000` for complete entity means `0.2000` and `0.4000`. Assert `N/A` for Average if one entity has partial coverage. Check exact ties, distinct values that round to the same string, lower FPR ranks, and absence of rank marks on partial values. Tool: pytest.
3. [x] Add `test_uncertainty_html`. Use two metrics and two entities. Assert one table per metric in profile order. Check that test detection values and both uncertainty splits match the same method, reporting variant, and entity. Check that Average uses the first entity grid, missing context shows `N/A`, and the CSV headers contain only the selected profile metrics. Tool: pytest.
4. [x] Extend the normal and smoke fixture to check the HTML path, all four output files, and the smoke title. Include a label with HTML markup and assert that it appears as text. Tool: pytest.
5. [x] Run `.venv/bin/python -m pytest -q tests/test_reporting.py`. Expect the new HTML checks to fail before the renderer is connected. Tool: pytest.

### Stage 5 — Render and write each report

1. [x] Implement `_entity_average(cells: Sequence[dict]) -> float | None`. Return the mean of the entity means when all cells have full seed coverage. Return `None` otherwise. Tool: Python.
2. [x] Implement `_metric_ranks(values: Sequence[float | None], *, higher_is_better: bool) -> list[int | None]`. Return rank 1 and rank 2 for the first two distinct values. Preserve ties. Use `None` for all other values. Pass only full-coverage means to the ranking function. Tool: Python.
3. [x] Build the test detection aggregate for the built-in offline `uncertainty.yaml` profile. Reuse the same records and seeds. Read the test split once if it was not already selected. Use `segment_mean_vus_pr` and `segment_mean_affiliation_f1` with their shared precision. Keep these columns in HTML only. Tool: Python.
4. [x] Implement `_render_report_html(table: dict, profile: dict, precision: dict[str, int], *, experiment_id: str, dataset: str, smoke: bool, detection_context: dict | None = None) -> str`. Use escaped text, inline CSS, and the title `Smoke run report` for smoke mode. Join cells by their keys and rows by method and reporting variant. Tool: Python and HTML.
5. [x] Render one detection table per selected split. Order `segment_mean_vus_pr`, `segment_mean_affiliation_f1`, `segment_mean_vus_roc`, and `fpr` as VUS-PR, Aff. F1, VUS-ROC, and FPR if selected. Put other metrics after them in profile order. Put Average on the right when there is more than one entity. Apply rank marks before formatting values. Tool: Python and HTML.
6. [x] Render one uncertainty table per configured uncertainty metric. Use four columns per entity in this order: test VUS-PR, test Aff. F1, validation uncertainty, and test uncertainty. Add the note about the test split. Put Average below the entity rows on the same grid as the first entity. Hide it for one entity. Use plain values. Tool: Python and HTML.
7. [x] Set equal widths for metric columns with `colgroup` and a fixed table layout. Use a header span equal to the number of metric columns. Keep metric labels on one line and allow horizontal scroll. Use the same formatting and coverage rules in all tables. Tool: HTML and CSS.
8. [x] Build the HTML before writing the current profile and dataset group. Write `PROFILE.html` with `_write_atomic` beside `PROFILE_detail.csv`, `PROFILE_mean.csv`, and `PROFILE.json`. Return the two CSV paths and the HTML path for the group. Tool: Python.
9. [x] Run `.venv/bin/python -m pytest -q tests/test_reporting.py`. Expect all reporting tests to pass. Tool: pytest.

## Phase 4 — Connect the command

### Stage 6 — Check command behavior

1. [x] Add CLI tests in `tests/test_cli.py` for the default mode and `--smoke`. Assert the flag passed to `report_experiment`, the printed CSV and HTML paths, and exit status `0`. For an empty list, assert `No completed runs to report in normal mode.` or `No completed runs to report in smoke mode.`. For a saved result error, assert the error on standard error and exit status `1`. Tool: pytest.
2. [x] Add the exact `/report --smoke` argument tuple to `tests/test_tui.py`. Check that `/report` without an active experiment requests selection. Check that `/help` shows `/report [--smoke]`. Tool: pytest and Textual Pilot.
3. [x] Run `.venv/bin/python -m pytest -q tests/test_cli.py tests/test_tui.py`. Expect the existing routing tests to pass. Record the new failures. Tool: pytest.

### Stage 7 — Update dispatch and help

1. [x] Add `--smoke` to the `report experiment` parser in `src/cli.py`. Keep the existing `run` option and help. Tool: Python.
2. [x] Pass `smoke=command.smoke` to the reporter. Print each returned path. Print the mode-specific message when the path list is empty. Use the existing error handler. Tool: Python.
3. [x] Change the report entry in `src/tui.py` help to `/report [--smoke]`. Tool: Python.
4. [x] Run `.venv/bin/python -m pytest -q tests/test_reporting.py tests/test_cli.py tests/test_tui.py`. Expect all focused tests to pass. Tool: pytest.

## Phase 5 — Verify the installed command and layout

### Stage 8 — Check the complete flow

1. [x] Prepare a temporary project with the same experiment ID in both modes. Use saved artifacts made by the existing test fixture functions. Include two entities in a normal offline and online report, two uncertainty metrics, and one smoke entity. Tool: Python and pytest fixtures.
2. [x] Run the installed `tsad report experiment --name NAME` in the temporary project. Check that it returns status `0` and writes only under the normal report root. Tool: shell.
3. [x] Open the installed TUI in the temporary project. Select the experiment with `/experiment NAME`. Tool: terminal.
4. [x] Enter `/report --smoke`. Check the command screen paths, status `0`, and smoke report title. Check that **Back** or Esc returns the status to the experiment history. Tool: terminal.
5. [x] Open the generated HTML files. Compare the three table types with the preview. Measure column and header bounds in the browser. A four-column group must span exactly four equal metric columns, within browser pixel rounding. Check that the uncertainty Average column edges match those of the first entity group. Check narrow-window scrolling and the single-entity layout. Tool: browser.

### Stage 9 — Document and review

1. [x] Add normal and smoke report examples and output paths to `README.md`. State that these commands use saved results. Tool: Markdown.
2. [x] Run `.venv/bin/python -m pytest -q`. Compare with the Phase 1 result and report any new failures. Tool: pytest.
3. [x] Run `git diff --check`. Check the changed untracked files for whitespace separately. Tool: Git and source inspection.
4. [x] Review the changes against the design and the Phase 1 working tree record. Check that all changes are in scope. Tool: Git.
5. [x] Record the test results and paths of the HTML files used for visual review. Tool: Markdown.

## Recovery and limits

Each report file uses the existing atomic writer. If a report stops during a write, run it again to refresh the report files. There is no migration of saved runs. Report verification uses saved fixtures and does not start an experiment. Normal and smoke files use separate directories.
