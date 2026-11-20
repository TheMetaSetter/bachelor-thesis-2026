# Runtime signatures implementation plan

Date: 2026-11-16
Status: completed on 2026-11-19. See the [completion report](../detail/runtime-replacement-review.md#completion-report--2026-11-19).
Project: `bachelor-thesis-2026`.

> **For agentic workers:** Use `superpowers:executing-plans` for approved execution. Complete each checklist item in order. Do not start the next phase before the current phase passes its gate.

**Goal:** Replace the current runtime with a small, reusable CLI implementation that follows the proposed signatures.

**Architecture:** Use functions for workflows and method classes for learned state. Reuse prepared data. Keep one preprocessing script, four profile types, and four protocol contexts.

**Tools and technologies:** Python, NumPy, PyTorch, pandas, PyArrow, PyYAML, pytest, Ruff, Git, `rg`, JSON, CSV, and Markdown. Use reviewed method and metric dependencies where required. Resolve package versions before implementation.

**Specification documents**

1. [Runtime signatures](../detail/proposed-runtime-signatures.md)
2. [Runtime profile contract](../detail/runtime-profile-contract.md)
3. [Confirmed replacement decisions](decisions-runtime-replacement.md)

## 1. Scope and order

1. Implement the next thesis runtime. Do not replace the `tsad-lib` runtime under this plan.
2. Keep all eight methods and their approved variants. Method readiness must be proved separately.
3. Keep the prepared Parquet files, scaler states, manifests, historical results, specifications, detailed mathematics, and required licenses.
4. Keep `scripts/preprocess.py` self-contained. Keep the active `scripts/cleandocs_hook.py`.
5. Use new commands and checkpoints. Do not add old-command wrappers or mandatory old-checkpoint readers.
6. Remove old code only after its replacement passes the stated checks. Do not create local archives.
7. Keep each function or method within 50 lines and each code file within 500 lines. Do not reduce counts by removing scientific scope or compressing code.
8. Keep runtime loading read-only. Do not split, fill, or scale prepared observations again.
9. Share the CNN architecture and settings across MUSTARD, RedLamp, CANDI, and M2N2. Construct separate instances and weights.
10. Keep test labels with metric calculation. Permit arrived test values for causal scoring and adaptation after the score is recorded.
11. Stop at an unresolved scientific decision. Record the question and affected steps. Obtain a decision before affected implementation.
12. The older path-depth limit, raw runtime loaders, compatibility wrappers, and restart-only online recovery do not define this thesis target. Exact continuation follows the newer signature proposal after its contract is approved.

## 2. Evidence and approval status

The following 15 documents supplied the recent updates. A proposal is not proof of runtime behavior. The confirmed replacement record takes precedence for the decisions it states. Library-wide rules remain scoped to `tsad-lib`.

1. [Thesis replacement decisions](decisions-runtime-replacement.md) — confirmed requirements.
2. [Runtime signatures](../detail/proposed-runtime-signatures.md) — proposed interfaces.
3. [Configuration signatures and fields](../../2026-11-09/plan/plan-config-signatures-and-fields.md) — earlier proposal with confirmed decisions and superseded sections.
4. [Desired CLI](../../../specs/desired-cli-20261109.md) — desired command surface.
5. [Current tree proposal](../structure/proposed-next-version-tree.md) — proposed file owners.
6. [Updated earlier tree](../../2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md) — aligned proposal and migration notes.
7. [Protocol validation rules](../../../specs/protocol-validation-rules.md) — proposed input and output checks.
8. [Thesis definitions](../../../specs/definitions.md) — agreed terms.
9. [Library definitions](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/definitions.md>) — agreed terms with thesis-specific scope.
10. [Library simplification design](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/2026-11-01-experiment-codebase-simplification.md>) — discussion draft with scoped thesis updates.
11. [Library interface design](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/2026-11-01-simplified-interfaces.md>) — design draft; its larger framework is not the thesis file plan.
12. [MUSTARD mathematics](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/mustard.md>) — method evidence with pending checks.
13. [RedLamp mathematics](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/redlamp.md>) — paper behavior and explicit thesis adaptation.
14. [CANDI mathematics](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/candi.md>) — paper behavior and explicit thesis adaptation.
15. [M2N2 mathematics](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib/documents/specs/method-mathematics/m2n2.md>) — paper behavior and explicit thesis adaptation.

Read the current project instructions before execution. The thesis has no `documents/specs/README.md` index. Use the named sources and their stated status. The library index does not approve every linked draft.

## 3. File ownership

Use the exact public signatures in the signature proposal. Change a signature only when a reviewed contract proves that an input or output is missing. Record that change before implementation.

| Files | Responsibility | Tests |
| --- | --- | --- |
| `pyproject.toml`, `src/__init__.py`, `scripts/__init__.py` | Package installation and installed command | `tests/test_cli.py` |
| `src/config.py`, four profile folders, `configs/metrics.yaml`, experiment and standalone YAML | Parsing, defaults, references, and method construction | `tests/test_config.py` |
| `src/prepared_data.py` | Prepared data, windows, batches, and indices | `tests/test_data.py` |
| `src/protocols/{run_evaluation,data_stream,scoring_calibration,method_updates}.py` | Four sets of input and output checks | `tests/test_protocols.py` |
| `src/artifacts.py` | Run IDs, paths, status, parent discovery, checkpoint storage, and resume storage | `tests/test_artifacts.py` |
| `src/scoring.py`, `src/evaluation_metrics.py` | Calibration, scoring state, alignment, and metrics | `tests/test_scoring.py`, `tests/test_metrics.py` |
| `src/encoders.py`, `src/methods/augmentation.py` | Shared CNN construction and reviewed training transformations | `tests/test_encoders.py`, owning method tests |
| `src/methods/mustard/{model,retrieval,losses,state,online_update}.py` | MUSTARD lifecycle and focused computations | `tests/test_mustard.py` |
| `src/methods/baselines/{redlamp,stumpy_channel_ab,kmeans_ad,iforest,candi,m2n2,moment}.py` | Seven independent method entrypoints | `tests/test_baselines.py` |
| `src/offline.py`, `src/online.py` | Phase execution | `tests/test_offline.py`, `tests/test_online.py` |
| `src/experiments.py` | Definition editing, planning, execution order, and recovery | `tests/test_experiments.py` |
| `src/reporting.py`, `src/integrations.py` | Tables, W&B logging, and Kaggle backup | `tests/test_reporting.py`, `tests/test_integrations.py` |
| `src/cli.py` | Argument parsing, completion, and workflow calls | `tests/test_cli.py` |
| `tests/conftest.py`, package `__init__.py` files | Small fixtures and package boundaries | Owning test files |
| `README.md`, `MAIN_FOLDERS.md`, maintained design documents | Commands, structure, decisions, and evidence | Link checks and diff review |

Brace groups list individual files from the tree; they are not new folders. Do not add a plugin system, shared base class, validator router, service layer, or `types.py`. Keep permanent preprocessing tests outside this delivery unless the user changes the current temporary-test decision.

## 4. Check cycle

For each implementation stage, use this order:

1. Write the named behavior tests.
2. Run the tests against the incomplete replacement.
3. Confirm that failures concern the missing behavior, not a broken test environment.
4. Implement the named owner with its proposed signatures.
5. Run the same tests.
6. Review the stage diff.

These actions appear as separate checklist steps below. Use small deterministic fixtures. Select numerical tolerances before comparing scientific outputs.

## 5. Review focus

1. Prepared values must remain unchanged when raw sources are absent.
2. Online scores must not depend on later observations or updates from the same window.
3. Resume must restore all state and must not select a newer offline parent.
4. Omitted result folders must not merge different runs or seeds.
5. Architecture compatibility must not be reported as numerical parity.

## Phase 1 — Fix the implementation contract

### Stage 1.1 — Record the starting state

1. [x] Read the current project instruction files. **Tool:** file reader.
2. [x] Read `python_semantics.md`. **Tool:** file reader.
3. [x] Record the current Git revision and working-tree status in `../detail/runtime-replacement-review.md`. **Tool:** Git.
4. [x] Inventory current source, scripts, tests, configurations, callers, and installed commands in that record. **Tool:** `rg --files`.
5. [x] Record current line and file counts separately for source, scripts, tests, and YAML, with one repeatable counting rule. **Tool:** Python.
6. [x] Record the protected prepared versions and manifest hashes. **Tool:** Python and SHA-256.
7. [x] Map each retained old path to its new owner and required removal check. **Tool:** Markdown.

### Stage 1.2 — Close the open decisions

Record these decisions in `decisions-runtime-replacement.md`. Use current code as evidence, not as an automatic implementation requirement.

1. [x] Specify exact profile fields, defaults, reference paths, and allowed overrides. **Tool:** Markdown and YAML.
2. [x] Specify method capabilities and supported dataset–phase–variant combinations for all eight methods. **Tool:** method sources and Markdown.
3. [x] Specify the common CNN settings and each method's pooling and head shapes. **Tool:** method sources and Markdown.
4. [x] Specify window tails, source gaps, stream ranges, score alignment, and online commit times. **Tool:** prepared manifests and Markdown.
5. [x] Specify metric inputs, missing-value rules, uncertainty inputs, and numerical comparison tolerances. **Tool:** metric sources and Markdown.
6. [x] Specify numeric smoke limits and smoke output isolation. **Tool:** Markdown.
7. [x] Specify run status transitions, filenames, schema versions, timestamp format, ID collision handling, and final tie order. **Tool:** Markdown.
8. [x] Specify parent compatibility fields and required checkpoint and calibration identities. **Tool:** Markdown.
9. [x] Specify offline and online resume boundaries, including how `fit` exposes continuation state through the method interface. **Tool:** signature review and Markdown.
10. [x] Specify backup timing, retries, artifact selection, interruption handling, and failure policy for this thesis target. **Tool:** integration sources and Markdown.
11. [x] Specify package registration and command ownership so a thesis installation does not silently replace the library command. **Tool:** package metadata inspection and Markdown.
12. [x] Record user decisions for unresolved scientific or workflow choices. **Tool:** user clarification and Markdown.
13. [x] Update affected signatures and schemas to match those decisions. **Tool:** `apply_patch`.
14. [x] Obtain approval for the completed implementation contract. **Tool:** user review.

**Phase gate:** All implementation inputs are explicit. Do not treat this plan as a decision on an open scientific detail. Stop here if a required decision remains open.

## Phase 2 — Implement profiles and prepared-data loading

### Stage 2.1 — Implement configuration resolution

Files: `src/config.py`, `tests/test_config.py`, and the exact YAML files listed in the tree.

1. [x] Write tests for duplicate YAML keys, invalid roots, invalid references, unknown fields, caller-data immutability, and equal total offline epoch budgets across fitted neural methods. **Tool:** pytest.
2. [x] Run the configuration tests to confirm the missing behavior. **Tool:** pytest. Collection stopped because `src.config` did not exist.
3. [x] Write the approved data, method, evaluation, and reporting profiles. Check the two MUSTARD q95 profiles against the [runtime profile contract](../detail/runtime-profile-contract.md#mustard-validation-q95-comparison). **Tool:** YAML. All 32 profiles parse, and the two comparison profiles differ only in score space.
4. [x] Write `configs/metrics.yaml` with approved metric names and display precision. **Tool:** YAML.
5. [x] Write the example experiment and standalone definitions. **Tool:** YAML.
6. [x] Implement `load_yaml`, `list_names`, and `resolve_run` with read-only resolution. **Tool:** Python and PyYAML.
7. [x] Run the configuration tests. Twelve tests passed. **Tool:** pytest.
8. [x] Review the configuration diff. **Tool:** `git diff`.

Defer `create_method` until the method entrypoints exist. Do not create dummy ready methods. Use one runtime data profile per dataset. Select an explicit prepared version, including one of the two retained SWaT versions; do not delete the other version.

### Stage 2.2 — Implement prepared-data loading and windows

Files: `src/prepared_data.py`, `tests/test_data.py`.

1. [x] Write tests for hash failures, feature order, unknown labels, optional timestamps, and preserved source mapping. **Tool:** pytest and Parquet fixtures.
2. [x] Write tests for approved tail and gap rules, split isolation, batch shapes, lazy windows, and causal indices. **Tool:** pytest and NumPy.
3. [x] Write a test that loads prepared data with raw source paths unavailable and returns unchanged feature values. **Tool:** pytest.
4. [x] Run the data tests to confirm the missing behavior. **Tool:** pytest.
5. [x] Implement `load_entity` and `iter_batches` with the approved data contract. **Tool:** Python, PyArrow, and NumPy.
6. [x] Run the data tests. **Tool:** pytest.
7. [x] Review the data diff. **Tool:** `git diff`.

### Stage 2.3 — Implement the four validation contexts

Files: the four `src/protocols/` modules and `tests/test_protocols.py`.

1. [x] Write input and output rejection tests for every approved record kind in the four contexts. **Tool:** pytest.
2. [x] Run the protocol tests to confirm the missing checks. **Tool:** pytest.
3. [x] Implement the eight proposed `validate_input` and `validate_output` functions. **Tool:** Python.
4. [x] Connect available input checks to configuration and data boundaries. **Tool:** Python.
5. [x] Run the configuration, data, and protocol tests. All 24 tests passed. **Tool:** pytest.
6. [x] Review the protocol diff. **Tool:** `git diff`.

**Phase gate:** Loading uses prepared values without new preprocessing. Resolution writes no outputs and allocates no run ID. Invalid inputs fail with file, field, or run context.

## Phase 3 — Implement storage, scores, and metrics

### Stage 3.1 — Implement run and checkpoint storage

Files: `src/artifacts.py`, `tests/test_artifacts.py`.

1. [x] Write tests for UUIDv7 allocation, standalone ID rejection, repeated execution, and allocation collisions. **Tool:** pytest.
2. [x] Write tests for omitted dimensions, exact metadata lookup, parent ordering, and missing selected-parent artifacts. **Tool:** pytest.
3. [x] Write tests for atomic status and checkpoint writes, checksum failures, role mismatch, and incomplete resume state. **Tool:** pytest.
4. [x] Run the artifact tests to confirm the missing behavior. **Tool:** pytest.
5. [x] Implement `allocate_run`, `result_directory`, `write_run_status`, and `write_result`. **Tool:** Python and JSON.
6. [x] Implement `find_runs` and `select_offline_parent`. **Tool:** Python.
7. [x] Implement `save_checkpoint`, `load_checkpoint`, `save_resume`, and `load_resume`. **Tool:** Python and reviewed serialization tools.
8. [x] Run the artifact tests. **Tool:** pytest.
9. [x] Review the artifact diff. **Tool:** `git diff`.

Use the agreed hierarchy:
`<experiment_id>/<phase>/<dataset>/<entity>/<method>/<variant>/<seed>/<split>/<run_id>/`.
Keep phase, split, and execution identity. Apply the approved single-value omission rules through recorded metadata. Never fall back silently when the selected parent lacks a required artifact.

### Stage 3.2 — Implement shared scoring

Files: `src/scoring.py`, `tests/test_scoring.py`.

1. [x] Write known-value tests for reconstruction score spaces, approved calibration, thresholds, and smoothing state. **Tool:** pytest and NumPy.
2. [x] Write tests for explicit point indices, window reduction, uncovered points, and immutable committed online scores. **Tool:** pytest.
3. [x] Run the scoring tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `reconstruction_scores`, `calibrate`, `score_batch`, and `aggregate_points`. **Tool:** Python, NumPy, and PyTorch.
5. [x] Run the scoring tests. **Tool:** pytest.
6. [x] Review the scoring diff. **Tool:** `git diff`.

Do not refit an input scaler to compute raw-input scores. Use saved scaler parameters when inverse conversion is required. Do not broadcast endpoint scores across a window. Reset or isolate validation scoring state before test execution.

### Stage 3.3 — Implement metrics

Files: `src/evaluation_metrics.py`, `tests/test_metrics.py`.

1. [x] Write known-value tests for each approved detection and uncertainty metric. **Tool:** pytest and reviewed reference outputs.
2. [x] Write tests for unknown labels, uncovered points, constant scores, absent classes, non-finite values, and unavailable dependencies. **Tool:** pytest.
3. [x] Run the metric tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `compute_metrics` through the reviewed metric dependencies. **Tool:** Python and NumPy.
5. [x] Run the metric tests. **Tool:** pytest.
6. [x] Review the metric diff. **Tool:** `git diff`.

**Phase gate:** Results retain identity. Scoring preserves alignment and causality. Metrics retain missing values and return unrounded numbers. Checkpoint storage rejects invalid artifacts.

## Phase 4 — Implement shared neural computations

### Stage 4.1 — Implement the common encoder

Files: `src/encoders.py`, `tests/test_encoders.py`.

1. [x] Write tests for `[B,L,C] -> [B,L,H]`, backward gradients, and independent encoder parameters. **Tool:** pytest and PyTorch.
2. [x] Run the encoder tests to confirm the missing behavior. **Tool:** pytest.
3. [x] Implement `create_encoder` with the approved common CNN settings. **Tool:** PyTorch.
4. [x] Run the encoder tests. **Tool:** pytest.
5. [x] Review the encoder diff. **Tool:** `git diff`.

### Stage 4.2 — Implement training augmentation

Files: `src/methods/augmentation.py` and owning method tests.

1. [x] Write tests for each retained transformation, masks, class labels, explicit RNG use, and unchanged source batches. **Tool:** pytest and PyTorch.
2. [x] Run the augmentation tests to confirm the missing behavior. **Tool:** pytest.
3. [x] Implement `augment` with only the reviewed transformation families. **Tool:** Python and PyTorch.
4. [x] Run the augmentation tests. **Tool:** pytest.
5. [x] Save one small visual check of synthetic changes and their masks. **Tool:** plotting library.
6. [x] Review the augmentation diff. **Tool:** `git diff`.

**Phase gate:** Shared computation does not share mutable learned state. Augmentation preserves the approved method-specific formulas and provenance.

## Phase 5 — Implement the eight method entrypoints

For each stage, first record the pinned source revision, license, equations, accepted fields, capabilities, checkpoint state, and explicit departures in `../detail/runtime-replacement-review.md`. Do not claim parity with a changed author architecture. Check retained computations separately.

### Stage 5.1 — Implement MUSTARD

Files: the five MUSTARD files, `tests/test_mustard.py`.

1. [x] Record the reviewed O0–O2 and A0–A2 contracts, including any unresolved O2 evidence. **Tool:** source reader and Markdown.
2. [x] Write known-input tests for retrieval, memory initialization, direct routing, loss reductions, and allowed updates. Test O1 with both point groups and with either group missing. **Tool:** pytest and PyTorch.
3. [x] Write tests for stage transitions, frozen parameters, empty masks, verification buffers, and full state restoration. **Tool:** pytest.
4. [x] Run the MUSTARD tests to confirm the missing behavior. **Tool:** pytest.
5. [x] Implement `retrieve` and `compute_loss`. **Tool:** PyTorch.
6. [x] Implement `update_window` with the approved triage and verification rules. **Tool:** PyTorch.
7. [x] Implement `pack_state` and `restore_state`. **Tool:** Python and PyTorch.
8. [x] Implement the `Mustard` public lifecycle using those helpers. **Tool:** Python and PyTorch.
9. [x] Run the MUSTARD tests. **Tool:** pytest.
10. [x] Review the MUSTARD diff against the approved method contract. **Tool:** Git and source reader.

No retrieval, dense retrieval, and sparse retrieval keep their distinct meanings. Do not treat the separate retrieval study as a change to the approved O/A variants.

### Stage 5.2 — Implement RedLamp

Files: `src/methods/baselines/redlamp.py`, `tests/test_baselines.py`.

1. [x] Record the reviewed label correction, multiclass losses, pooling, validation-fixed class selection, score scaling, and causal smoothing. **Tool:** source reader and Markdown.
2. [x] Write tests for corrected targets, masked loss, zero-range scaling, score alignment, and unchanged calibration after test input changes. **Tool:** pytest.
3. [x] Run the RedLamp tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `RedLamp` public lifecycle. **Tool:** Python and PyTorch.
5. [x] Run the RedLamp tests. **Tool:** pytest.
6. [x] Review the RedLamp diff against its approved project adaptation. **Tool:** Git and source reader.

### Stage 5.3 — Implement STUMPY channel AB

Files: `src/methods/baselines/stumpy_channel_ab.py`, `tests/test_baselines.py`.

1. [x] Record the approved reference join, channel reduction, score direction, and alignment. **Tool:** source reader and Markdown.
2. [x] Write known-value tests for the join, retained state, and supported input boundaries. **Tool:** pytest and reviewed STUMPY outputs.
3. [x] Run the STUMPY tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `StumpyChannelAB` public lifecycle. **Tool:** Python and STUMPY.
5. [x] Run the STUMPY tests. **Tool:** pytest.
6. [x] Review the STUMPY diff. **Tool:** `git diff`.

### Stage 5.4 — Implement KMeansAD

Files: `src/methods/baselines/kmeans_ad.py`, `tests/test_baselines.py`.

1. [x] Record the approved window representation, distance, initialization, and scoring alignment. **Tool:** source reader and Markdown.
2. [x] Write known-value tests for distances, deterministic fitting, and saved cluster state. **Tool:** pytest and NumPy.
3. [x] Run the KMeansAD tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `KMeansAD` public lifecycle. **Tool:** Python and the reviewed clustering dependency.
5. [x] Run the KMeansAD tests. **Tool:** pytest.
6. [x] Review the KMeansAD diff. **Tool:** `git diff`.

### Stage 5.5 — Implement Isolation Forest

Files: `src/methods/baselines/iforest.py`, `tests/test_baselines.py`.

1. [x] Record the approved input representation, fitting source, score orientation, and alignment. **Tool:** source reader and Markdown.
2. [x] Write tests for score direction, deterministic fitting, and complete estimator restoration. **Tool:** pytest.
3. [x] Run the Isolation Forest tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `IsolationForestAD` public lifecycle. **Tool:** Python and the reviewed estimator dependency.
5. [x] Run the Isolation Forest tests. **Tool:** pytest.
6. [x] Review the Isolation Forest diff. **Tool:** `git diff`.

### Stage 5.6 — Implement CANDI

Files: `src/methods/baselines/candi.py`, `tests/test_baselines.py`.

1. [x] Record the approved backbone fitting, latent pooling, reference sets, covariance, selection thresholds, SANA, and optimizer rules. **Tool:** source reader and Markdown.
2. [x] Write tests for singular covariance, empty reference sets, boundary equality, frozen backbone, and pending-sample restoration. **Tool:** pytest and PyTorch.
3. [x] Write a test that changes test labels without changing method scores, updates, or diagnostics. **Tool:** pytest.
4. [x] Run the CANDI tests to confirm the missing behavior. **Tool:** pytest.
5. [x] Implement the `CANDI` public lifecycle with method-owned SANA state. **Tool:** Python and PyTorch.
6. [x] Run the CANDI tests. **Tool:** pytest.
7. [x] Review the CANDI diff against the approved equations. **Tool:** Git and source reader.

### Stage 5.7 — Implement M2N2

Files: `src/methods/baselines/m2n2.py`, `tests/test_baselines.py`.

1. [x] Record the approved trend initialization, update order, window overlap, normal mask, loss reduction, and threshold source. **Tool:** source reader and Markdown.
2. [x] Write known-value tests for trend state, threshold equality, empty normal masks, parameter updates, and resumed output. **Tool:** pytest and PyTorch.
3. [x] Run the M2N2 tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `M2N2` public lifecycle. **Tool:** Python and PyTorch.
5. [x] Run the M2N2 tests. **Tool:** pytest.
6. [x] Review the M2N2 diff against the approved equations. **Tool:** Git and source reader.

Method-internal detrending must not refit the prepared input scaler.

### Stage 5.8 — Implement MOMENT

Files: `src/methods/baselines/moment.py`, `tests/test_baselines.py`.

1. [x] Record the approved pretrained checkpoint, input limits, channel handling, masks, scoring, and probing target. **Tool:** official method sources and Markdown.
2. [x] Write tests for zero-shot behavior, linear probing, frozen encoder state, and absence of online parameter updates. **Tool:** pytest.
3. [x] Run the MOMENT tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement the `Moment` public lifecycle with its pretrained encoder. **Tool:** Python and the official MOMENT dependency.
5. [x] Run the MOMENT tests with the selected checkpoint. **Tool:** pytest.
6. [x] Review the MOMENT diff. **Tool:** `git diff`.

An unavailable checkpoint is a failed readiness check, not a passed mock test.

### Stage 5.9 — Connect method construction and shared state checks

Files: `src/config.py`, method package files, and owning tests.

1. [x] Write parameterized constructor and checkpoint round-trip tests for all approved method profiles. **Tool:** pytest.
2. [x] Run the shared method tests to confirm missing connections. **Tool:** pytest.
3. [x] Implement `create_method` with a small explicit constructor table and lazy method imports. **Tool:** Python.
4. [x] Run the shared method tests. **Tool:** pytest.
5. [x] Check required methods against the public signature checklist. **Tool:** Markdown and `rg`.
6. [x] Review the method construction diff. **Tool:** `git diff`.

**Phase gate:** All eight methods pass their required scientific, shape, gradient, capability, and state checks. List each deliberate departure. Do not remove a method because it is harder to implement.

## Phase 6 — Implement phase execution and exact recovery

### Stage 6.1 — Implement offline execution

Files: `src/offline.py`, `tests/test_offline.py`.

**Calibration decision:** For `synthnormal_q99_normalized_input_mse_ewma09.yaml`, generate synthetic windows from saved validation and score them with the selected offline checkpoint.
Use eligible scores to set separate point and window q99 thresholds.
Record the source split and generation settings.
The MUSTARD raw and latent profiles use their own validation q95 rules.
See the [confirmed decision](decisions-runtime-replacement.md#smd-synthetic-calibration-choice--2026-11-19).

1. [x] Write tests for fit selection, best-state loading, approved calibration, required stage checkpoints, and no test-based selection. **Tool:** pytest.
   Check that the SMD profile uses saved validation windows and fixed checkpoint parameters.
   Check that the generated mask and source-label eligibility are separate, that both thresholds use q99, and that the result records its source.
   Check that the MUSTARD q95 profiles use eligible validation windows and save a threshold for each score space.
2. [x] Write an interrupted-training test against uninterrupted execution at the approved resume boundary. **Tool:** pytest.
3. [x] Run the offline tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `run_offline` with the approved resume contract. **Tool:** Python.
5. [x] Run the offline tests. **Tool:** pytest.
6. [x] Review the offline diff. **Tool:** `git diff`.

### Stage 6.2 — Implement online execution

Files: `src/online.py`, `tests/test_online.py`.

1. [x] Write an event-order test requiring score, record, decision, then optional update. **Tool:** pytest.
2. [x] Write a future-input test requiring identical committed prefix scores when only later observations change. **Tool:** pytest.
3. [x] Write interruption tests for cursor, RNG, optimizer, buffers, scoring state, and the recorded offline parent. **Tool:** pytest.
4. [x] Write tests for forbidden parameter changes, A0, fixed methods, failure status, and missing dependencies. **Tool:** pytest.
5. [x] Run the online tests to confirm the missing behavior. **Tool:** pytest.
6. [x] Implement `run_online` with label-free method batches and approved causal commit rules. **Tool:** Python.
7. [x] Connect output validation at score, update, and result boundaries. **Tool:** Python.
8. [x] Run the online tests. **Tool:** pytest.
9. [x] Review the online diff. **Tool:** `git diff`.

**Phase gate:** Offline and online use the same prepared identity and scaler. Resume keeps the run ID, configuration, and parent. Resumed outputs match uninterrupted outputs within the approved tolerances.

## Phase 7 — Implement experiment workflows

### Stage 7.1 — Implement definition editing and read-only planning

Files: `src/experiments.py`, `tests/test_experiments.py`.

1. [x] Write tests for duplicate names, approved edit fields, stable combination order, defaults, and one profile per dataset. **Tool:** pytest.
2. [x] Write tests proving that planning allocates no IDs and creates no result files. **Tool:** pytest.
3. [x] Write tests for unsupported combinations and exact smoke selection with required offline dependencies. **Tool:** pytest.
4. [x] Run the planning tests to confirm the missing behavior. **Tool:** pytest.
5. [x] Implement `list_experiments`, `create_experiment`, and `edit_experiment`. **Tool:** Python and PyYAML.
6. [x] Implement `plan_experiment`. **Tool:** Python.
7. [x] Run the planning tests. **Tool:** pytest.
8. [x] Review the planning diff. **Tool:** `git diff`.

### Stage 7.2 — Implement execution and experiment recovery

1. [x] Write tests for dependency order, independent-run failures, blocked children, interrupted runs, and completed-run preservation. **Tool:** pytest.
2. [x] Run the experiment execution tests to confirm the missing behavior. **Tool:** pytest.
3. [x] Implement `run_experiment` using the two phase functions. **Tool:** Python.
4. [x] Implement `resume_experiment` from recorded execution state. **Tool:** Python.
5. [x] Run the experiment execution tests. **Tool:** pytest.
6. [x] Review the experiment execution diff. **Tool:** `git diff`.

**Phase gate:** Unsupported combinations stop validation. A run failure cannot become experiment success. Recovery does not apply changed defaults to saved runs.

## Phase 8 — Implement reports and optional integrations

### Stage 8.1 — Implement reporting

Files: `src/reporting.py`, `tests/test_reporting.py`.

1. [x] Write table tests for repeated executions, missing seeds, one seed, no valid seeds, multiple splits, and incompatible configurations. **Tool:** pytest.
2. [x] Write tests for smoke exclusion, shared precision, rare overrides, and method-profile traceability for `default`. **Tool:** pytest.
3. [x] Run the reporting tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `aggregate_table` as a pure calculation. **Tool:** Python and NumPy.
5. [x] Implement `report_experiment` through `find_runs`. **Tool:** Python and CSV.
6. [x] Run the reporting tests. **Tool:** pytest.
7. [x] Review the reporting diff. **Tool:** `git diff`.

Select the latest successful finite metric per seed before calculating the mean. Report sample standard deviation and valid/expected count. One table covers one dataset and one phase. Keep split identity. Write both detailed and mean-only CSV files.

### Stage 8.2 — Implement independent logging and backup

Files: `src/integrations.py`, `tests/test_integrations.py`.

1. [x] Write tests for disabled integrations, provider failure, session cleanup, approved backup timing, and exact experiment-root selection. **Tool:** pytest and provider fakes.
2. [x] Write tests that reject credentials and unrelated experiment files from backup selection. **Tool:** pytest.
3. [x] Run the integration tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `log_wandb` with the provider's native session object. **Tool:** Python and W&B.
5. [x] Implement `backup_kaggle` with the approved artifact selection. **Tool:** Python and the reviewed Kaggle client.
6. [x] Connect approved logging and backup lifecycle calls to the owning workflows. **Tool:** Python.
7. [x] Run the integration tests. **Tool:** pytest.
8. [x] Review the integration diff. **Tool:** `git diff`.

**Phase gate:** Reports preserve seed identity and missing values. Remote failures follow the approved policy. Test execution uses fake providers and sends no real uploads.

## Phase 9 — Implement the installed CLI

### Stage 9.1 — Implement command routing and packaging

Files: `src/cli.py`, `pyproject.toml`, package files, and `tests/test_cli.py`.

1. [x] Write command tests for discovery, editing, show, validate, smoke, run, resume, report, completion, and error exit codes. **Tool:** pytest.
2. [x] Write tests proving that preprocessing forwards supported arguments to the existing script entrypoint. **Tool:** pytest.
3. [x] Run the CLI tests to confirm the missing behavior. **Tool:** pytest.
4. [x] Implement `main` and `complete_names` with the approved parser. **Tool:** Python.
5. [x] Register the installed `tsad` entrypoint under the approved package design. **Tool:** `pyproject.toml`.
6. [x] Install the package in the project environment. **Tool:** project Python package installer.
7. [x] Run the CLI tests through the installed entrypoint. **Tool:** pytest.
8. [x] Check preprocessing defaults with `tsad preprocess --dataset smd --show-profile`. **Tool:** installed CLI.
9. [x] Review the CLI and packaging diff. **Tool:** `git diff`.

Do not add dataset downloading in this plan. It remains a future feature. Do not rerun source preprocessing against deleted source folders. Standalone calls use the shared resolver and phase functions; do not invent an extra runner class.

**Phase gate:** The installed command resolves to this project in its environment. Help and errors use short, direct English. CLI commands contain no second workflow implementation.

## Phase 10 — Check the complete flow

### Stage 10.1 — Check one real prepared-data combination

1. [x] Select one SMD entity and one approved MUSTARD offline/online variant pair in a bounded experiment definition. **Tool:** YAML.
2. [x] Validate the definition with the installed CLI. **Tool:** `tsad validate experiment`.
3. [x] Run its offline and online dependency chain with smoke limits. **Tool:** `tsad run experiment --smoke`.
4. [x] Reload the saved results and checkpoints. **Tool:** Python.
5. [x] Compare recorded scores, metric inputs, parent identity, and prepared-data identity with the approved contract. **Tool:** Python and NumPy.
6. [x] Produce the experiment tables under the approved smoke report policy. **Tool:** reporting tests and CSV reader.
7. [x] Recheck protected prepared manifest hashes. **Tool:** SHA-256.

Normal reports must exclude smoke outputs. The smoke report check uses the approved explicit inspection route or confirms normal-report exclusion; it must not silently mix smoke results into a normal table.

### Stage 10.2 — Check retained coverage

1. [x] Load one prepared entity from each of ICCAD, SMD, IOPS, NASA, and SWaT. **Tool:** project Python.
2. [x] Run bounded smoke execution for every retained method and supported phase. **Tool:** installed CLI.
3. [x] Compare interrupted and uninterrupted real execution for one approved update-capable method. **Tool:** pytest and saved artifacts.
4. [x] Record scientific comparisons and their explicit tolerances. **Tool:** Markdown.
5. [x] Run the complete replacement test suite. **Tool:** pytest.
6. [x] Record unavailable checkpoints, dependencies, and failed checks. **Tool:** Markdown.

**Phase gate:** One real complete command flow and retained-method smoke coverage pass. An unavailable required method check blocks removal of its old implementation. Do not start a full experiment set under this plan.

## Phase 11 — Remove replaced code and finish

### Stage 11.1 — Remove only proved replacements

Use the path map from Stage 1.1. Review paths against the current working tree; do not treat earlier user deletions as new work.

1. [x] Mark each old path whose replacement checks passed in the path map. **Tool:** Markdown.
2. [x] Search for remaining callers of those exact paths. **Tool:** `rg`.
3. [x] Remove approved replaced source files from the checked path list. **Tool:** file removal.
4. [x] Remove approved replaced runner, generator, collector, and shell scripts from the checked path list. **Tool:** file removal.
5. [x] Remove approved replaced YAML files from the checked path list. **Tool:** file removal.
6. [x] Remove obsolete tests whose retained behavior is covered by replacement tests. **Tool:** file removal.
7. [x] Remove newly empty directories within the changed paths. **Tool:** empty-directory inspection and `rmdir`.
8. [x] Run the replacement test suite after removal. **Tool:** pytest.
9. [x] Search maintained commands and imports for removed paths. **Tool:** `rg`.

Keep `preprocess.py`, `cleandocs_hook.py`, prepared data, historical results, protected documents, mathematics, reference evidence, and licenses. Do not delete unrelated raw data or old outputs under this runtime plan.

### Stage 11.2 — Review size, documents, and final evidence

1. [x] Check function and code-file limits using the counting rule recorded in Phase 1. **Tool:** Python AST and line counts.
2. [x] Compare final source, script, test, and configuration counts with the starting counts. **Tool:** Python.
3. [x] Remove repeated workflow code identified by that review. **Tool:** `apply_patch`.
4. [x] Run tests affected by those simplifications. **Tool:** pytest.
5. [x] Check code formatting and static rules. **Tool:** Ruff.
6. [x] Verify that `.codex/` remains ignored. **Tool:** `git check-ignore`.
7. [x] Update `README.md` with installed commands and current limitations. **Tool:** `apply_patch`.
8. [x] Update `MAIN_FOLDERS.md` and the maintained tree documents to match the actual files. **Tool:** `apply_patch`.
9. [x] Update signature and decision statuses from proposed to verified only where evidence exists. **Tool:** `apply_patch`.
10. [x] Check local links in changed Markdown files. **Tool:** Python.
11. [x] Review the complete task diff against the recorded starting state. **Tool:** `git diff`.
12. [x] Check whitespace errors. **Tool:** `git diff --check`.
13. [x] Write the completion report with checks, code counts, deliberate method changes, and remaining limits. **Tool:** Markdown.

**Final gate:** The new runtime is small, readable, reusable, and checked. Required scientific scope is retained. No deletion relies only on a smaller line count or a successful import.

## Terminology changes

No domain terms change in this plan. `src/core/config.py` to `src/config.py` remains the previously proposed path rename. Preprocessing and runtime loading remain separate owners. Protocol means a contextual set of input and output validation rules in this thesis project.

## Completion record

User amendments on 2026-11-17 require compact score persistence and fair epoch counts. Earlier checked contract steps record historical progress. The common epoch value and its schema are now resolved; full-contract approval is still required. Storage/recovery tests must reconstruct positions without persisted point identity columns.

Leave all execution boxes unchecked until their actions are complete. For a blocked step, record its ID, reason, evidence, and required decision in `../detail/runtime-replacement-review.md`. Keep the distinction between a document review, a passed interface test, and verified numerical behavior.
