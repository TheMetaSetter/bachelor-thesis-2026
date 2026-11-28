# Run progress and validation execution record

Status: The implementation and bounded SMD smoke run are complete. The full test suite has six baseline failures.

Source: [Ordered implementation plan](../plan/plan-run-progress-and-validation.md).

## Checks

1. The remote switch checks used test doubles. Disabled and smoke runs made zero provider calls. Resume used the saved settings.
2. Stage 2.4 reproduced two failures with the former label rule. The checkpoint, resume, and runner checks then had 47 passes. The calibration, reporting, and display checks had 144 passes.
3. The new `progress_acceptance` smoke attempt completed for SMD `machine-1-1`, seed 6, and the raw-input q95 EWMA profile. Both remote switches were off. The terminal showed the offline epochs and all 32 online windows.
4. The stage A and stage B held-out raw false-positive rates were 0.01640625 and 0.0015625. The initial and best checkpoints for both stages had the expected roles and matching file hashes.
5. Offline calibration used stage B's best checkpoint and recorded the all-normal validation assumption. Both phases used the same calibration file contents and the exact same point threshold, `0.013487492408603429`.
6. The offline and online reports contained the expected MUSTARD variants and test false-positive rates. The report values were 0.0070 and 0.0000. The covered test segments lacked one label class, so the reports showed VUS-PR and Affiliation F1 as unavailable. CSV, HTML, and JSON files were present.
7. The full suite had 223 passes and six failures. Three experiment tests match the Phase 1 failures because the SMD fixture selects SWaT. Three MOMENT tests require a model snapshot that is absent from the local cache. The same three MOMENT tests failed in a clean copy of revision `a6e2af78` with offline mode enabled.
8. The README gives the remote switch commands, the display rules, and the evaluation profile options. The diff passed the Git whitespace check. The review found one terminal display path, one final point reduction path, and one saved point threshold for the pair. The epoch threshold uses its separate unsmoothed calibration subset. The terminal display reads saved results. It does not score the data again.

## Run references

1. Offline run `01a2183d-3698-77ac-95a8-669e8c58a12a`.
2. Online run `01a2183d-3699-705d-aec2-c261fb895433`.
3. The run files and reports are in `/private/tmp/tsad-progress-acceptance.DbvwOeed/outputs/smoke/progress_acceptance`.
