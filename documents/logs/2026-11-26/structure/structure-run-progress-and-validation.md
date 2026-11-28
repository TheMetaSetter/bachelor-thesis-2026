# Implementation structure: Run progress and MUSTARD validation

Status: Proposed for review.
Sources: [User decisions](../design/run-progress-decisions.md), [research](../research/research-run-progress-and-validation.md), [ordered plan](../plan/plan-run-progress-and-validation.md).

| Phase | Stages in order | Phase check |
| --- | --- | --- |
| 1. Preserve current controls | Capture the baseline. Check both existing remote flags. | Disabled providers receive no calls. Resume uses the saved flags. |
| 2. Monitor MUSTARD offline | Add profile selection. Split and score epoch data. Commit metrics and the best checkpoint. | No crossing windows. Both score spaces work at the correct stage. Resume keeps the same selection. |
| 3. Calibrate and evaluate | Check the two reduction choices. Add final uncertainty and test summaries. | The paired phases share one threshold. Core metrics and uncertainty have an explicit source. |
| 4. Show progress | Add the terminal renderer. Connect offline and online events. | One bar per run. Correct initial counts on resume. Plain piped output. |
| 5. Verify delivery | Check the complete path for one real-data combination. Run the suite. Review and document the result. | The outputs and terminal trace agree with the decisions. |

The saved-state and metric paths come before the display. The display uses their results and has no write authority over run state.
