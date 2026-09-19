---
date: 09-15-2026
topic: "Reduce the repository pytest suite to 28 collected contract tests"
status: implemented_with_legacy_inventory_retained
revision: 61cb84bb450624031f6a27fde86262fd180565e9
branch: dev
source_structure: documents/logs/09-15-2026/structure/structure-pytest-suite-cap-30.md
related_documents:
  - documents/logs/09-15-2026/research/research-pytest-suite-cap-30.md
  - documents/logs/09-15-2026/plan/plan-pytest-suite-cap-30.md
---

# Detailed Implementation: Pytest Suite Cap

## Summary

Implement the approved-by-request structure as a test-only reorganization. Keep 28 collected pytest nodes as the target and 30 as the hard ceiling.

## Atomic execution steps

### Phase 1: Establish the replacement map

#### Stage 1.1: Record the baseline

1. Run pytest collection from the repository root.
2. Record the collected-node count as the baseline.
3. Record the acceptance rule as no more than 30 collected nodes.
4. Record 28 as the target count.

#### Stage 1.2: Assign boundaries

1. List every current test file under `tests/`.
2. Assign each test to one boundary group.
3. Mark repeated examples that share one contract.
4. Mark unique behaviors that require separate failure boundaries.
5. Assign the six boundary budgets: 5, 5, 4, 4, 5, and 5 nodes.
6. Do not delete any test during mapping.

### Phase 2: Build replacement contracts

#### Stage 2.1: Configuration and data

1. Create one contract for configuration loading and validation.
2. Create one contract for loader batch size, tensor shape, and orientation.
3. Create one contract for data and window metadata.
4. Create one contract for synthetic anomaly injection and label preservation.
5. Create one contract for the visualization artifact.
6. Loop over equivalent dataset or anomaly members inside the relevant contract.
7. Include the member name in every folded assertion message.
8. Run this group before deleting any data tests.

#### Stage 2.2: Models and baselines

1. Create one contract for model output shapes.
2. Create one contract for one forward and backward pass.
3. Create one contract for baseline and registry selection.
4. Create one contract for reference-model construction and scoring.
5. Create one contract for shared model-family behavior.
6. Loop over equivalent ready models inside the shared contract.
7. Run this group before deleting model or baseline tests.

#### Stage 2.3: Evaluation and engine

1. Create one contract for score transformations and metric outputs.
2. Create one contract for threshold calibration and identity.
3. Create one contract for training-loop behavior and optional controls.
4. Create one contract for checkpoint round trips, fallback behavior, and provenance.
5. Run this group before deleting evaluation or engine tests.

#### Stage 2.4: Online runtime

1. Create one contract for causal stream windows and absolute indices.
2. Create one contract for adaptation parameter isolation.
3. Create one contract for verification-buffer admission and filtering.
4. Create one contract for online state round trips.
5. Create one contract for online threshold and artifact identity.
6. Keep A0, A1, and A2 separate when their update rules differ.
7. Run this group before deleting online tests.

#### Stage 2.5: Public and release

1. Create one contract for script or CLI routing.
2. Create one contract for smoke execution.
3. Create one contract for output files and artifact structure.
4. Create one contract for documentation and compliance requirements.
5. Create one contract for release preflight.
6. Run this group before deleting public, operations, or compliance tests.

### Phase 3: Remove proven duplicates

#### Stage 3.1: Remove group duplicates

1. Select one completed boundary group.
2. Confirm its replacement contracts pass.
3. Compare each candidate old test with the replacement assertions.
4. Delete only tests whose behavior is fully represented.
5. Recollect pytest after each group.

#### Stage 3.2: Recheck coverage

1. Compare the old test inventory with the replacement map.
2. Identify any unique behavior that lacks an assertion.
3. Add the smallest missing assertion to its owning contract.
4. Recollect pytest.
5. Stop adding cases if collection reaches 30.

### Phase 4: Prove and record the cap

#### Stage 4.1: Collect and run

1. Run the project’s collection command.
2. Confirm the result is 28 target nodes.
3. Confirm the result is no more than 30 nodes.
4. Run the complete collected suite.
5. Confirm every node passes.

#### Stage 4.2: Record evidence

1. Record the final collection command.
2. Record the final collected-node count.
3. Record the full-suite pass result.
4. State that folded tests can check multiple inputs without creating multiple pytest nodes.

## Risks and recovery

- A folded loop may hide the failing member. Include member names in assertion messages.
- Folding unrelated contracts may make failures ambiguous. Split only that boundary, and keep the total at 30 or fewer.
- Deleting too early may lose coverage. Write and pass the replacement before deletion.
- A count-only reduction may remove important evidence. Compare the replacement map with the old inventory before final verification.

## Final acceptance criteria

- The repository’s normal pytest collection reports 28 target nodes.
- The collection never exceeds 30 nodes.
- The full suite passes.
- Required test categories remain represented.
- Runtime source files remain unchanged.

## Execution evidence

- Baseline collection: `./.venv/bin/python -m pytest --collect-only -q` reported 615 nodes.
- Replacement collection: the same command reports 28 nodes.
- Final execution: `./.venv/bin/python -m pytest -q` reports 28 passed.
- The six contract files are under `tests/contracts/`.
- `pytest.ini` now sets `testpaths = tests/contracts`.

## Deliberate limitation

The original tests remain under `tests/` and were not deleted. The current checkout does not provide enough evidence to label all 587 excluded nodes redundant one by one. A later cleanup can remove them only after each replacement contract has been compared with its source tests.
