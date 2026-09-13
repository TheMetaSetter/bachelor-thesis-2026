---
date: 2026-09-12 20:20:39 +07:00
planner: OpenAI Codex
topic: Minimal standalone codebase for the all-machine SMD matrix
status: high-level-draft-pending-scientific-contract
revision: b8fff201cdfdb27aa9cec377d607487c4c2755cb
branch: dev
related_research: ../research/research-minimal-smd-runtime.md
---

# Plan: Minimal SMD codebase

## Summary and scope

Write a standalone, SMD-focused codebase with direct function calls, one public model per file, one experiment manifest, and one worker command.
Keep scientific computations and required experiment evidence; remove historical compatibility and unused framework features.
This is the requested high-level plan, not an implementation-ready specification while the decisions in Phase 1 remain unresolved.

Proposed future location: `../smd-minimal/`, a sibling of `bachelor-thesis-2026`, outside the existing repository.
The directory has not been created.
Research and planning artifacts stay under the current repository's `documents/` according to the named prompts.
The new project will own its own `documents/` contract, environment, Git history, checkpoints, and outputs.
Use an explicit external dataset path without copying SMD or importing the old repository at runtime.

Include THESIS O0/O1/O2, revised A0/A1/A2, RedLamp, CANDI, M2N2, and all three traditional baselines.
Exclude other datasets, UI dashboards, historical checkpoint compatibility, plugin registries, and unused fusion/backbone modes.
Retain library implementations for numerical primitives and baseline algorithms; from scratch means a new runtime, not reimplementing PyTorch or scikit-learn.

## Approach

| Option | Benefit | Cost | Decision |
| --- | --- | --- | --- |
| Small standalone modules with direct calls | Visible computation, independent execution, focused tests | Requires numerical comparison and explicit contracts | Recommended |
| One large script | Very few files | Mixes model, online state, scheduling, and metrics; exceeds local size rules | Reject |
| New wrappers importing the old repository | Fastest initial reuse | Retains old dependency graph and compatibility paths | Reject for the final runtime |

Use Python, PyTorch, NumPy, scikit-learn, STUMPY, PyYAML, pytest, W&B, and the existing metric dependencies as required by the selected algorithms.
Choose version bounds from the current working environment and test them; do not guess new versions.
Use standard-library subprocess only at the worker boundary, JSON for resolved manifests/results, and YAML for human settings.
Use no new design pattern beyond direct composition and small method adapters.
Keep functions at most 50 lines and code files at most 500 lines; split reusable numerical primitives without distributing model lifecycle through mixins.

Proposed components, not a finalized directory tree:

| Component | Responsibility |
| --- | --- |
| Data and augmentation | Train/validation split, scaler, windows, synthetic masks |
| THESIS model | Constructor, forward, stage loss, trainable parameters, checkpoint contract |
| Memory primitives | Clustering, retrieval, verification geometry |
| Offline runner | Stage A → memory initialization → Stage B → calibration → evaluation |
| Online runner | Causal loop, EWMA, triage, verification, projector update |
| Baseline model adapters | One model file per method; preserve method-specific training and inference |
| Metrics and artifacts | Common coverage, required metrics, checkpoint identity, compact outputs, W&B |
| Matrix runner | One resolved manifest, resource queues, offline barrier, completion checks |

Finalize file layout only in the later `3_structure` step; expand atomic code changes in `4_detail`.

## Phase 1: Freeze the computational contract

Tools: `rg`, `.venv/bin/python`, existing source/tests, Markdown.

Stage 1.1 — Record decisions in proposed `smd-minimal/documents/runtime-contract.md`.
1. Use PDF equations (3.25)/(3.28) as the point-level contrastive formula; specify complete sets, empty-set behavior, weight/temperature, and whether O2 replaces or adds to the current two-view term.
2. Record revised A1 = hard-old + online contrastive; A2 adds verified PNN; A0 remains inference only.
3. Specify direct routing in Stage B and memory bypass in Stage A; preserve the stage names.
4. Select discrete-memory token eligibility and missing-class behavior explicitly against the ontology conflict.

Stage 1.2 — Lock the protocol and experiment identity.
1. Record training stride separately from offline-evaluation stride and online stride; recommended starting values are 1, 20 with end alignment, and 1, subject to resolving the matrix wording.
2. Specify raw-input identity scoring, reconstruction-loss units, all threshold units, and Monte Carlo reduction order; preserve each loss's defined units rather than assuming every loss is raw.
3. Replace ambiguous “V4” wording with explicit schema, score-space, scaler, checkpoint, and calibration fields.
4. Fix 28 entities, seeds 6/8/36, length 20, shared 2048-point ranges, 588/1176 logical counts, and 2352 W&B runs.

Acceptance: every unresolved scientific decision has a formula or precise rule with a source or recorded human decision.
Do not implement O2, memory selection changes, or conflicting protocol behavior before this acceptance condition.
The remaining phases specify stable outcomes and can be refined after these decisions; this document does not claim they are already resolved.

PDF source: `../T826_KL_KHMT06_BaoCao.pdf` relative to the old repository root, printed pages 30–31 and 48.
It defines the point-level objective but assigns it to O0, whereas the new matrix assigns that component to O2.
The current normal-token diagonal cross-entropy implementation is not equivalent merely because it is also called contrastive.
A formula test must verify that an eligible injected negative affects the loss, that moving a positive closer lowers loss, and that the selected empty-set policy produces no invalid reduction.
Record the new policy version so historical O0 results cannot be mislabeled as the new O2 experiment.

## Phase 2: Establish one reproducible data path

Tools: Python, NumPy, PyTorch DataLoader, PyYAML, pytest; DVC for the required dataset/augmentation provenance.

Stage 2.1 — Create the standalone environment and input contract.
1. Create the sibling project and its own environment.
2. Load one experiment settings object with explicit defaults and reject unknown variant names.
3. Discover and validate the exact train/test/test-label entity intersection.
4. Split training chronologically, fit its scaler, and expose windows with absolute indices.

Stage 2.2 — Make augmentation and benchmark selection reproducible.
1. Generate synthetic batches with deterministic seed control and separate class labels and injection masks.
2. Record dataset hashes, augmentation parameters, code revision, and random state; define the DVC reproduction stage without persisting every injected batch.
3. Select and save one 2048-point range per entity before running methods.
4. Keep range-selection labels outside model/adaptation inputs; record this label-informed sampling in the protocol.

Acceptance: focused tests verify shapes, split isolation, training-only scaler fit, mask correctness, reproducibility, and range coverage.
All methods receive identical entity ranges; test labels cannot reach update functions.

## Phase 3: Complete one THESIS offline-to-A0 path

Tools: PyTorch, NumPy, pytest, JSON, W&B.

Stage 3.1 — Implement the selected model computation.
1. Write the CNN encoder and task heads behind one THESIS public model.
2. Implement Stage A objectives, including explicit O0/O1/O2 activation from Phase 1.
3. Build memory and verification metadata from Stage A best using the accepted token policy.
4. Implement direct deterministic and stochastic retrieval without unused fusion modules.

Stage 3.2 — Connect training and evaluation with direct calls.
1. Run Stage A, save best, initialize memory, then run Stage B with explicit frozen parameters.
2. Save stage initialization and best checkpoints with scaler, model settings, RNG state, and identity.
3. Replay clean validation through offline scoring and causal A0 scoring to obtain their distinct thresholds.
4. Evaluate test timelines and return explicit checkpoint/artifact paths to the caller.

Acceptance: one-batch forward/backward, memory provenance, frozen-parameter checks, checkpoint reload parity, score-space tests, and tail-coverage tests pass.
Compare unchanged computations against the old implementation with fixed inputs and controlled RNG; test changed policies against Phase 1 rules, not old outputs.
Keep Monte Carlo samples in memory only as needed for reduction; retain report-ready statistics.
Use separate W&B runs for Stage A, Stage B, and evaluation, even when they execute in one process.

## Phase 4: Add online adaptation and method baselines

Tools: PyTorch, NumPy, scikit-learn, STUMPY, existing reference adapters, pytest.

Stage 4.1 — Implement the causal THESIS loop.
1. Load the matching Stage B model and thresholds; freeze the source.
2. Compute current scores, vector EWMA, and predictions before an optional update.
3. Gate hard-old adaptation with non-overlap and the revised variant policy.
4. Verify gray-zone entries using the buffer, anomaly metadata, recurrence, and a non-empty PNN mask.
5. Update only the projector with the accepted event loss and fresh optimizer; retain only required active state.

Stage 4.2 — Add baseline flows and common metrics.
1. Implement RedLamp training and encoder checkpoint export.
2. Port CANDI and M2N2 method rules into their model files using the matching RedLamp checkpoint.
3. Implement fit/calibrate/score for frozen Stumpy, KMeansAD, and Isolation Forest.
4. Feed aligned scores, predictions, and coverage to one metric interface without forcing identical native scoring semantics.

Acceptance: event-sequence tests cover no-update, hard-old overlap rejection, empty/non-empty PNN, TTL/capacity, and frozen source invariants.
Compare unchanged baseline calculations with reference adapters.
Test budget boundaries, ties, missing classes, incomplete coverage, and undefined metrics; never substitute zero for an undefined result.

## Phase 5: Make one manifest executable

Tools: Python itertools/subprocess, JSON, W&B, pytest; Linux CUDA visibility and CPU affinity for deployment.

Stage 5.1 — Generate identity and dependencies once.
1. Expand the 28 × 3 entity/seed combinations into exactly 1764 logical records.
2. Assign each record its resolved configuration, output directory, W&B names, and explicit dependency paths.
3. Validate unique identities and paths; reject any online record with the wrong offline variant/entity/seed/protocol.
4. Generate a plan-only manifest without model execution or W&B run creation.

Stage 5.2 — Execute simple resource queues.
1. Allocate one subprocess worker per GPU and separate CPU workers from the actual allowed CPU set.
2. Assign each logical record to exactly one worker; pass its manifest record rather than generating more YAML files.
3. Finish and validate offline dependencies before starting online queues.
4. Mark a record complete only after required outputs and identities pass checks; restart failed logical records from their inputs.

Acceptance: dry-run gives 588 offline, 1176 online, and 2352 planned W&B stage runs; names and output paths do not collide.
Mocked process failures cannot unlock online work or create false completion markers.
Prefer record-level restart over a resumable online-state framework for 2048-point runs.
Keep the canonical output hierarchy; resolve variant identity once into its method directory token and preserve the explicit variant fields in the manifest.
Reject incompatible schema/config hashes rather than silently resuming old results.

## Phase 6: Verify one combination, then run all machines

Tools: new project's Python/pytest, JSON, W&B; `nvidia-smi`, `taskset`, and optional tmux on the selected server.

Stage 6.1 — Run a complete development combination.
1. Use `machine-1-6`, seed 6, in a separate smoke root; explicit reduced epochs/limits must be recorded.
2. Run its required THESIS variants, offline-to-online dependencies, RedLamp, and baseline paths.
3. Verify metrics, initialization/best checkpoints, thresholds, W&B lifecycle, and record-level restart.
4. Inspect synthetic examples and actual six-GPU/CPU resource availability before wet execution.

Stage 6.2 — Execute and audit the full matrix.
1. Run 588 offline records into a new wet root.
2. Validate dependencies and run 1176 online records on the saved ranges.
3. Collect only the requested result metrics plus necessary provenance and training diagnostics.
4. Report missing/failed records explicitly and retry only their exact records.

Acceptance: 1764 logical results and 2352 successful logical W&B stage identities, with no missing dependencies or duplicate results.
Retries may create additional W&B attempts; distinguish attempt count from successful logical stage count.
Do not interpret a reduced smoke pass as research-quality performance evidence.

## Terminology and migration

| Existing object | Proposed treatment | Status and owner |
| --- | --- | --- |
| Stage A, memory initialization, Stage B, offline evaluation | Preserve distinct operations and checkpoint roles | Unchanged; offline runner |
| Continuous prototype bank, discrete codebook | Preserve names; resolve token selection separately | Names unchanged; THESIS model |
| Frozen source model, online MLP projector | Preserve immutable/mutable ownership | Unchanged; online model |
| VerificationBuffer, NonOverlapGuard | Preserve separate state and purpose | Unchanged; online loop |
| A1 | Hard-old + contrastive instead of legacy PNN-only | Changed semantics; versioned online policy |
| PDF O0 point-level objective → matrix O2 | Formula from (3.25)/(3.28); relationship to the existing two-view term requires a decision | New variant assignment, not an exact alias; THESIS loss owner |
| Generated YAML paths and two-stage process manifest | One resolved experiment manifest with explicit dependencies | Merged orchestration responsibility; matrix runner |
| Historical sigmoid/V4/raw score names | Explicit score space and schema | Not interchangeable; calibration owner |

The new checkpoint schema is independent; no automatic loading of historical checkpoints is promised.
Old source and results remain the comparison reference, and rollback means continuing to use that project.
No historical output trees need migration or deletion.
Document future reuse of reference code with its provenance and applicable license.

## Verification and handoff

Research evidence: `../research/research-minimal-smd-runtime.md`; readiness context: `../research/research-smd-all-machines-readiness.md`.
Current focused source tests: 22 passed; the new project does not yet exist and has no passing implementation tests.
During implementation, use the established `.venv/bin/python -m pytest -q` pattern inside the new project; exact new CLI commands belong in `4_detail` after their parsers exist.
Human review is needed for the scientific decisions in Phase 1 and synthetic-injection examples; hardware/W&B checks require the real execution environment.
Continue in order: `1_research` → `2_plan` decision closure → `3_structure` → `4_detail` → implementation.
