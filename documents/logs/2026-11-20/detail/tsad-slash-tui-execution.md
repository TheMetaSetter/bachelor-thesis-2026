# TSAD slash-command TUI execution

Date 2026-11-20. This record follows the [implementation plan](../plan/plan-tsad-slash-tui.md) and [approved design](tsad-slash-tui-design.md).

## Results in plan order

| Phase | Check | Result |
| --- | --- | --- |
| 1 | Record the checkout and run the baseline. | Saved the Git status, diff, and copies of existing files in scope. The focused suite passed 13 tests. The full suite passed 140 tests and failed 3 tests. |
| 2 | Check slash command mapping. | The tests failed before the resolver was available. All 26 mapping and input checks passed with the resolver. |
| 3 | Check the command screen and child process. | The tests failed before the screen was available. All 5 checks passed with the screen. The tests checked live output, Back, Ctrl+C, and start errors. |
| 4 | Check the session view. | The session tests failed before the app was available. All 34 TUI tests passed with the app. |
| 5 | Check the entry point and wheel. | The new test failed before interactive dispatch. Then all 48 focused tests passed. The wheel loaded from the test environment. Both entry points showed the same help. Redirected bare `tsad` gave the existing parser error and status 2. |
| 6 | Check the installed app in a terminal. | Verified the project prompt, experiment list, session history, picker Save and Cancel, method edit, parser error, and Ctrl+C. The one-cell SMD MUSTARD smoke run passed. |

## Terminal and smoke evidence

1. Ran the installed wheel in a disposable project with a space in its path. Cancel preserved all YAML bytes. Save selected the second SMD and SWaT versions and updated both profiles and the experiment. All four prepared manifest hashes matched.
2. Ctrl+C stopped the controlled command with status 130. The current picker handled Ctrl+C as Cancel and returned status 0. Both child processes exited. The app stayed open.
3. Ran `/run --smoke` for SMD entity `machine-1-1` and MUSTARD O0 in a disposable project. The TUI showed status 0. The run record showed success. The four checkpoint payload hashes matched their descriptors.

## Final checks

1. The full suite passed 183 tests after review. The same three experiment tests failed before and after the change.
   - `test_planning_has_no_filesystem_side_effects`
   - `test_shared_profile_updates_two_new_plans`
   - `test_smoke_adds_pinned_offline_dependency_for_selected_variant`
2. Ruff, the Git whitespace check, and the code size checks passed. The README gives the terminal steps.
3. The test compared changes with copies of the files from the start of execution. The code was implemented in the existing `simplifying` checkout because the current runtime was untracked there.
4. The default uv cache failed under the sandbox. The build passed with a temporary cache. The isolated wheel environment used the existing installed dependency directory.
5. The existing code identity check includes Python source paths and contents. The new source can affect resume and checkpoint reuse for runs made before the change.

## Independent review

1. The reviewer found that an explicit name flag could select a different experiment from the active session. The resolver now rejects the name flag and its abbreviations for these slash commands.
2. The reviewer found that Ctrl+C or end of input at the final terminal prompt could replace the completed command status. The screen now preserves the command status.
3. All eight regression cases failed before the fixes and passed after them. The rebuilt wheel passed the terminal checks and smoke run. After Save, Ctrl+D returned to the command screen with status 0. The smoke checkpoint hashes matched.
