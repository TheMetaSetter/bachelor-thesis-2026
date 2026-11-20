# Execution record for `/report` tables

Date 2026-11-20. [Implementation plan](../plan/plan-tsad-report-tables.md) and [table design](tsad-report-table-design.md).

| Stage | Result |
| --- | --- |
| 1 | Baseline suite had 180 passes and six failures. The affected runtime and test files were already untracked. The README already had edits. |
| 2 | Two new mode and discovery checks failed. The two result validation cases passed with the existing checks. |
| 3 | Six reporting tests passed after the mode and discovery changes. |
| 4 | Six HTML cases failed before the renderer was connected. |
| 5 | Eleven reporting tests passed with HTML output. |
| 6 | Seven CLI and help cases failed before dispatch and help were updated. Slash routing already forwarded the option. |
| 7 | All 71 focused reporting, CLI, and TUI cases passed. |
| 8 | The installed CLI and TUI generated reports from saved fixtures in distinct roots. Chrome measured the table bounds. |
| 9 | The full suite had 197 passes and the same six baseline failures after review fixes. Whitespace checks passed. |

Three baseline failures could not resolve Hugging Face to fetch the MOMENT checkpoint. Three experiment tests expected SMD settings that differ from the current SWaT experiment definition.

## Installed command and browser evidence

The installed normal CLI returned status 0. It wrote 12 files for three profiles. The installed TUI displayed smoke report paths and status 0. Esc returned the command status to the experiment history.

Chrome checked all three table types at 1500 and 480 pixels. Each group header spans four equal metric columns. The uncertainty Average aligns with the first entity group. Wide tables scroll horizontally in narrow windows and single-entity reports omit Average. The first browser pass showed that the initial minimum width caused unnecessary scrolling. The final width fits the two-entity fixtures at 1500 pixels.

The following HTML files are temporary verification artifacts with fixture values.

1. [Offline detection](/tmp/tsad-report-work.gAVa3U/project/outputs/experiments/try/reports/smd/offline_detection.html)
2. [Online detection](/tmp/tsad-report-work.gAVa3U/project/outputs/experiments/try/reports/smd/online_detection.html)
3. [Offline uncertainty](/tmp/tsad-report-work.gAVa3U/project/outputs/experiments/try/reports/smd/uncertainty.html)
4. [Smoke offline detection](/tmp/tsad-report-work.gAVa3U/project/outputs/smoke/try/reports/smd/offline_detection.html)
5. [Smoke online detection](/tmp/tsad-report-work.gAVa3U/project/outputs/smoke/try/reports/smd/online_detection.html)
6. [Smoke offline uncertainty](/tmp/tsad-report-work.gAVa3U/project/outputs/smoke/try/reports/smd/uncertainty.html)


## Final review

The review found two cases that needed regression tests.

1. A damaged split outside the report selection could stop the report. Report discovery now requests run records only. Other callers of `find_runs` keep the existing default.
2. An uncertainty profile could select a metric also used for detection context. Each HTML column now uses its own precision rule.

Both regression tests failed before the fixes. All 80 focused artifact, reporting, CLI, and TUI tests passed after the fixes. The installed commands produced the same six HTML files that Chrome measured.

## Verification after the clipping fix

1. Reviewed all 47 completed steps in order. The original record above contains the results of the tests that ran before implementation.
2. Ran the installed normal CLI and smoke TUI against separate copies of saved run data in one temporary project. Each mode wrote 12 report files in its own directory. Esc returned status 0 to the experiment history.
3. Checked both table edges in Chrome. All three normal report types fit in a 1452-pixel report area. In a 432-pixel area, scrolling showed the last column of each normal report. The single-entity smoke tables fit in that area. The 4:1 header ratio and uncertainty Average alignment passed.
4. Used the current design rule for long labels in place of the original one-line rule in Stage 5. Long labels can wrap to help prevent clipping.
5. Ran 80 focused tests. All passed. The full suite had 197 passes and the same six failures as the baseline. Whitespace checks passed.

HTML from the rerun: [normal reports](/private/tmp/tsad-report-rerun.gujCjX/project/outputs/experiments/try/reports/smd) and [smoke reports](/private/tmp/tsad-report-rerun.gujCjX/project/outputs/smoke/try/reports/smd).

Chrome captures: [offline detection](/private/tmp/tsad-report-rerun.gujCjX/offline_detection.png), [online detection](/private/tmp/tsad-report-rerun.gujCjX/online_detection.png), and [offline uncertainty](/private/tmp/tsad-report-rerun.gujCjX/uncertainty.png).
