---
date: 2026-09-15 15:30:00 +07
topic: "Apply one fixed THESIS smoke contract across datasets"
status: implemented
source_structure: documents/logs/2026-09-15/structure/structure-thesis-smoke-contract.md
---

# Detail: One THESIS Smoke Contract

## Fixed contract

For `model="thesis"` and `experiment_type="benchmark_smoke"` only:

```text
stage_a_epochs = 3
stage_b_epochs = 2
online_window_count = 4096
required_points = 4115 when window_size=20 and online_stride=1
```

The contract is internal. Users keep calling `run()` exactly as they do today.

## Phase 1: resolve the fixed smoke values

### Stage 1.1: write the failing unit tests

1. Add one parametrized test in `tsad-lib/tests/test_thesis_runtime.py` for the four `(model, experiment_type)` branches.
2. Assert that only `("thesis", "benchmark_smoke")` returns `stage_a_epochs=3`, `stage_b_epochs=2`, and `online_window_count=4096`.
3. Run that test with `tsad-lib/.venv/bin/python -m pytest` and confirm failure because the helper does not exist.

### Stage 1.2: add the smallest resolver

1. Add `THESIS_SMOKE_STAGE_A_EPOCHS = 3` in `tsad-lib/src/tsad/experiment.py`.
2. Add `THESIS_SMOKE_STAGE_B_EPOCHS = 2` in `tsad-lib/src/tsad/experiment.py`.
3. Add `THESIS_SMOKE_ONLINE_WINDOWS = 4096` in `tsad-lib/src/tsad/experiment.py`.
4. Add `_thesis_smoke_values(model, experiment_type)` in `experiment.py`.
5. Return the three values only for the THESIS smoke leaf.
6. Return an empty mapping for every other leaf.
7. Run the focused unit test and confirm pass.

### Stage 1.3: connect the existing configuration

1. Add a test that captures the resolved THESIS smoke configuration.
2. Run it and confirm the current configuration contains 1/1 epochs.
3. Merge `_thesis_smoke_values()` into the configuration values in `ExperimentRunner.run_request()` before `resolve_config()`.
4. Run the focused test and confirm that its configuration contains 3/2 epochs.
5. Run the non-THESIS branch test and confirm it still uses generic values.

## Phase 2: select exact causal work

### Stage 2.1: write point-count tests first

1. Add a unit test for `_thesis_smoke_point_count(20, 1)` in `tests/test_thesis_runtime.py`.
2. Assert that the returned value is `4115`.
3. Run it and confirm failure because the helper does not exist.

### Stage 2.2: add the point-count helper

1. Add `_thesis_smoke_point_count(window_size, online_stride)` in `experiment.py`.
2. Calculate `window_size + (THESIS_SMOKE_ONLINE_WINDOWS - 1) * online_stride`.
3. Run the focused point-count test and confirm pass.

### Stage 2.3: enforce the THESIS smoke range

1. Add a unit test that passes a short label array to the THESIS smoke range path.
2. Assert that it raises a clear `ValueError` naming the required 4,115 points.
3. Run it and confirm failure because the current branch falls back to the whole test series.
4. In the THESIS smoke leaf, require labels, at least one anomaly label, and the derived point count.
5. Call `select_online_range(labels, length=required_points)`.
6. Keep windows fully inside that selected range.
7. Raise `ValueError` when the retained count differs from 4,096.
8. Run the short-label test and confirm pass.

### Stage 2.4: record the evidence

1. Add a matrix test for a smoke artifact containing 4,115 points and 4,096 windows.
2. Run it and confirm failure because the current smoke gate accepts 2,048 points only.
3. Add `window_count` to `online_range.json` in `experiment.py`.
4. Require `point_count == 4115` and `window_count == 4096` in `tsad-lib/src/tsad/matrix.py`.
5. Update the passing matrix fixture with both new values.
6. Add an old 2,048-point fixture assertion that the gate returns `blocking`.
7. Run `tests/test_matrix.py` and confirm pass.

## Phase 3: prove the public path

### Stage 3.1: add a fast integration test

1. Add a `write_thesis_smoke_fixture()` helper in `tests/test_public_runtime.py` with 4,115 test points and a multi-point anomaly label.
2. Add a `RecordingOnlineRunner` fake with a `run(data)` method that records `len(data.test)` and returns a valid `ThesisOnlineState` plus 4,096 small records.
3. Add one public `run()` test that replaces only `tsad.experiment.ThesisOnlineRunner` with `RecordingOnlineRunner`.
4. Assert the report is completed.
5. Assert the fake received 4,096 windows.
6. Assert `resolved_config.yaml` contains `stage_a_epochs: 3` and `stage_b_epochs: 2`.
7. Assert `online_range.json` contains `point_count: 4115` and `window_count: 4096`.
8. Run the test and confirm failure before the implementation exists.
9. Run the same test after Phase 2 and confirm pass.

### Stage 3.2: run the real stress check

1. Choose `ServerMachineDataset`, one entity with a labelled 4,115-point range, O2-A0, and seed 6.
2. Choose a new non-existing output root under `tsad-lib/outputs/benchmark_smoke`.
3. Run public `run()` with the actual THESIS runner on CPU or MPS.
4. Read `resolved_config.yaml` and confirm 3/2 epochs.
5. Read `online_records.json` and confirm 4,096 records.
6. Read `online_range.json` and confirm 4,115 points and 4,096 windows.
7. Hash `stage_b_best.pt` and confirm it equals `thresholds.json["checkpoint_hash"]`.
8. Run the focused test group: `tests/test_thesis_runtime.py`, `tests/test_matrix.py`, and `tests/test_public_runtime.py`.

## Phase 4: update the student story

### Stage 4.1: write the SSOT rule

1. Add the THESIS-only smoke rule to `documents/spec/tsad-lib-development-spec.md`.
2. State that non-THESIS methods do not inherit Stage A, Stage B, or online TTA counts.
3. State that the wet matrix keeps its 2,048-point policy in `documents/notes/smd-all-machines-experiment-matrix.md`.
4. Replace the old 2,048-point smoke evidence only after the new real artifact exists.

### Stage 4.2: update the notebook inputs

1. Add a notebook test that requires `4115` in each ICCAD smoke notebook.
2. Run it and confirm failure because each notebook uses 80 test points.
3. Change the full Apple notebook to choose a real labelled 4,115-point test segment.
4. Change the practice Apple notebook to leave its existing four learning gaps while choosing the same segment size.
5. Change the full Docker notebook to choose the same segment size.
6. Change the practice Docker notebook to leave its existing four learning gaps while choosing the same segment size.
7. Add assertions for 4,115 test points and one or more labelled anomaly points.
8. Run `tests/test_tutorial_notebooks.py` and confirm pass.

## Final verification

Run:

```bash
cd '/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/tsad-lib'
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_thesis_runtime.py tests/test_matrix.py tests/test_public_runtime.py tests/test_tutorial_notebooks.py
```

The full suite is an extra check after the focused suite. It must not be described as passing until its current failures are resolved or reported.
