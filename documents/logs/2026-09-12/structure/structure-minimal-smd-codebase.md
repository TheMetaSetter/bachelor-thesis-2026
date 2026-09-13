---
date: 2026-09-12 21:23:46 +07:00
topic: Minimal standalone multi-dataset codebase
status: approved
revision: b8fff201cdfdb27aa9cec377d607487c4c2755cb
related_documents:
  - ../plan/plan-minimal-smd-codebase.md
  - ../research/research-minimal-smd-runtime.md
  - ../research/research-smd-all-machines-readiness.md
  - ../../../notes/smd-all-machines-experiment-matrix.md
  - ../../../spec/offline_pretraining_terminology_ontology.md
  - ../../../spec/online_tta_terminology_ontology.md
---

# Structure: Minimal Multi-Dataset Codebase

## The story

We will build a small multi-dataset codebase in the sibling directory `dual-stochastic-operators`.
It will use external paths under `data/` and will not import the old repository at runtime.
It will support every dataset family currently present under `data/`.
The SMD matrix remains the first benchmark and the first complete vertical slice.
The multi-dataset notebook and CLI API are defined in `proposal-dual-stochastic-operators.md`.

The work begins with one scientific contract.
That contract becomes one data path.
The data path feeds THESIS offline training and A0.
The same outputs then feed online adaptation and baselines.
Finally, one manifest runs the matrix after one complete smoke combination passes.

The code will use direct function calls.
Each model will have one public entrypoint.
The matrix will have one manifest and one worker entrypoint.
Model lifecycle logic will stay in the model file.
Functions will stay under 50 lines and code files under 500 lines.
`subprocess` will be used only at the worker boundary.

The new directory does not exist yet.
No source code has been changed.

## The scientific contract

Stage A has three variants.
O0 uses reconstruction and classification.
O1 adds `point_score_loss` to O0.
O2 adds `two_view_contrastive_loss` to O0.
These are component choices, not equal loss weights.
The current formulas and weights remain unchanged.

The names `point-level contrastive loss` and `two_view_contrastive_loss` mean the same offline computation.
The implementation keeps `_compute_two_view_contrastive_loss` as it is.
It filters positions whose masks are zero in both views, flattens the full batch, normalizes the tokens, builds a `K × K` matrix, uses the same position as the positive, uses the other augmented tokens as negatives, and computes clean-to-augmented cross-entropy.
It returns zero when `K = 0`.
It does not replace this computation with another reading of PDF Equation (3.28).

The online contract also stays clear.
A0 performs inference only.
A1 uses hard-old plus online contrastive adaptation.
A2 adds verified PNN.
Stage A does not use memory.
Stage B uses `direct_branch_routing`.

The current memory, stride, and scoring policies are kept.
Continuous memory uses normal positions in normal-class windows.
Discrete memory uses all tokens for the window class and the current fallback for a missing class.
Training uses stride `1`.
Offline evaluation uses stride `20` with `end_align`.
Online evaluation uses stride `1`.
Scoring uses `raw_input` and `identity`.
Thresholds use clean validation.
EWMA uses `0.9 current + 0.1 previous`.
Point adjustment is disabled.

These policies must still be checked against the two ontologies, the protocol, the data config, and the current memory initialization code.

## Phase 1 — We freeze the contract

The first phase removes scientific ambiguity.
Its result is a short contract that an implementer can follow without guessing.
The tools are Markdown, `rg`, Python, the current source, configs, and tests.

### Stage 1.1 — We name the variants

First, create `dual-stochastic-operators/documents/runtime-contract.md`.
Next, write the O0/O1/O2 table and the exact alias between `point-level contrastive loss` and `two_view_contrastive_loss`.
Then, record each loss activation and online policy.
Finally, run a Python check that rejects unknown variants and loss names.

The contract owns `variant`, `losses`, and `online_policy`.
It accepts the confirmed user decisions and returns a versioned contract.
It must reject a missing activation.
The check must confirm that O0 has no contrastive loss, O1 has `point_score_loss`, and O2 has `two_view_contrastive_loss`.

### Stage 1.2 — We name the data rules

First, write separate keys for training, offline, and online stride.
Next, record memory eligibility, fallback behavior, score units, threshold units, and Monte Carlo reduction order.
Then, record the `K = 0`, diagonal-positive, and cross-window-negative rules.
Finally, add a small fixture with expected shapes and window alignment.

The fixture belongs in the proposed `tests/test_runtime_contract.py`.
It must reject a missing stride, unit, or reduction rule.
It must verify end alignment and the empty-set loss rule.

### Stage 1.3 — We define handoffs

First, define identity fields for checkpoints, scalers, thresholds, protocols, and outputs.
Next, give offline calibration and causal A0 calibration different names.
Then, define the handoff from Stage A to memory initialization, Stage B, and evaluation.
Finally, make the schema reject a different contract, entity, seed, variant, or protocol.

The proposed owner is `src/artifacts/schema.py`.
Its key fields are `contract_hash`, `entity`, `seed`, `variant`, and `protocol_hash`.
The tests use one valid artifact and four mismatch cases.

### Stage 1.4 — We record evidence

First, map every retained computation to its old source symbol.
Next, record provenance and license for reused references.
Then, define fixed-input parity tests for loss, score, memory, and metrics.
Finally, scan the new codebase for imports from the old repository.

The proposed evidence file is `documents/provenance.md`.
The stage stops if a required symbol has no definition or use.
It passes when every provenance path exists and the import scan is empty.

## Phase 2 — We build one data path

Now the new project can read SMD without depending on the old project.
Every method will later receive the same batches and the same 2048-point ranges.
The tools are Python, NumPy, PyTorch DataLoader, PyYAML, pytest, and DVC.

### Stage 2.1 — We create the environment

First, create `dual-stochastic-operators` with its own Git history.
Next, add only the dependency bounds required by the new project.
Then, create `src/config.py` with one settings loader and one default location.
Finally, reject unknown config keys and invalid variants before loading a model.

The proposed test is `tests/test_config.py`.
The loader turns YAML into one resolved settings object.
It reports the exact invalid key or path.
The tests load one valid config and reject an invalid variant and a missing dataset path.

### Stage 2.2 — We load and split SMD

First, create `src/data/smd.py` for one entity's train, test, and label files.
Next, check the train/test/label entity intersection.
Then, split training data in time order.
Finally, fit the scaler only on training data and save its metadata.

The main functions are `load_entity`, `split_train`, and `fit_scaler`.
They return scaled arrays, labels, and split metadata.
They reject missing files and length or shape mismatches.
The tests cover all 28 entities, temporal isolation, and train-only scaler fitting.

### Stage 2.3 — We create windows and views

First, create `src/data/windows.py` with window values, absolute indices, and entity identity.
Next, create `src/data/augmentation.py` with seeded view generation.
Then, return class labels and `synthetic_anomaly_mask` as separate fields.
Finally, test clean and injected points in one batch.

The main functions are `make_windows` and `augment_view`.
They reject a window that exceeds the tail or an invalid seed.
The tests cover shapes, tail coverage, same-seed equality, and different-seed inequality.

### Stage 2.4 — We freeze the benchmark input

First, create `src/data/ranges.py` for one 2048-point range per entity.
Next, save range boundaries, selection rule, label-use note, and data hashes as JSON.
Then, add a DVC stage for range and provenance reproduction.
Finally, make every method read this range artifact instead of selecting its own range.

The proposed artifacts are `data/ranges.json` and `dvc.yaml`.
They must reject incomplete coverage and hash mismatch.
Reproducing the stage twice must produce the same metadata.

## Phase 3 — We run THESIS offline through A0

The data path now feeds the first model path.
The phase must run one entity and one seed from data to A0 metrics.
The tools are PyTorch, NumPy, pytest, JSON, and W&B.

### Stage 3.1 — We implement Stage A

First, create `src/models/thesis.py` with one encoder and two task heads.
Next, implement `_compute_two_view_contrastive_loss` with the confirmed computation.
Then, implement one Stage A loss function with O0/O1/O2 activation.
Finally, run forward and backward once for each variant.

The public symbols are `THESISModel`, `stage_a_loss`, and `_compute_two_view_contrastive_loss`.
They receive two views, labels, and masks and return total loss plus named components.
They reject incompatible shapes and return a scalar zero on the correct device and dtype when `K = 0`.
Tests cover gradients, `K = 0`, `K = 1`, diagonal targets, cross-window negatives, and the variant call matrix.

### Stage 3.2 — We save checkpoints and memory

First, create `src/offline/checkpoints.py` for the Stage A best checkpoint.
Next, create `src/memory/init.py` for continuous and discrete pools.
Then, save verification metadata and the Stage B initialization artifact.
Finally, reload every artifact and compare tensors and metadata.

The main functions are `save_checkpoint`, `load_checkpoint`, and `initialize_memory`.
They reject missing class policy, contract mismatch, and tensor shape mismatch.
Tests cover reload parity, pool eligibility, and missing-class fallback.

### Stage 3.3 — We implement Stage B

First, create `src/offline/stage_b.py` with direct retrieval and `direct_branch_routing` calls.
Next, mark frozen parameters before creating the optimizer.
Then, train only the allowed parameters.
Finally, save initialization and best checkpoints with scaler, settings, RNG state, and identity.

The public function is `run_stage_b`.
It receives the Stage A checkpoint and memory and returns Stage B checkpoints plus a training summary.
It rejects an optimizer containing frozen parameters or a wrong branch route.
Tests compare frozen parameter hashes and reload the checkpoint.

### Stage 3.4 — We implement scoring and A0

First, create `src/scoring.py` for `raw_input` and `identity` scoring.
Next, create `src/offline/calibration.py` for two clean-validation thresholds.
Then, create `src/offline/evaluate.py` for absolute-index timeline reconstruction.
Finally, save compact metrics, provenance, checkpoints, and thresholds in one evaluation record.

The public functions are `score_window`, `calibrate_thresholds`, and `evaluate_a0`.
They reject score-unit mismatch, missing tail coverage, and non-clean calibration data.
Tests cover score units, thresholds, Monte Carlo reduction, timeline coverage, and W&B lifecycle.

## Phase 4 — We add adaptation and baselines

The offline path now provides a stable source model and score contract.
This phase adds online behavior and baseline methods without forcing them to share native algorithms.
The tools are PyTorch, NumPy, scikit-learn, STUMPY, current metric dependencies, pytest, and W&B.

### Stage 4.1 — We implement A1

First, create `src/online/a1.py` and load the matching Stage B checkpoint and thresholds.
Next, compute score, vector EWMA, and prediction before any update.
Then, apply `NonOverlapGuard` and the hard-old policy.
Finally, create a fresh optimizer for the online projector only and save the event decision.

The public symbols are `run_a1`, `NonOverlapGuard`, and `OnlineProjector`.
The source model must remain immutable.
Tests cover predict-before-update, no-update, overlap rejection, and frozen-source behavior.

### Stage 4.2 — We implement A2

First, create `src/online/a2.py` with gray-zone admission and capacity checks.
Next, run source verification before creating the PNN mask.
Then, skip the update when the PNN mask is empty.
Finally, apply TTL and capacity rules and save event state.

The public symbols are `run_a2`, `VerificationBuffer`, and `pn_mask`.
They reject invalid TTL, capacity overflow, and merged buffer/guard state.
Tests cover empty and non-empty PNN, TTL expiry, capacity, recurrence, and source immutability.

### Stage 4.3 — We implement neural baselines

First, create one model file for RedLamp with train and evaluate entrypoints.
Next, save its encoder checkpoint with the new schema.
Then, create the CANDI model file and load the RedLamp checkpoint.
Finally, create the M2N2 model file and validate its output shape.

The proposed files are `src/models/redlamp.py`, `candi.py`, and `m2n2.py`.
They receive shared SMD ranges and return method-specific checkpoints, scores, and coverage.
They reject incompatible source checkpoints and missing settings.
Tests compare fixed-input results with the references and run one smoke test per model.

### Stage 4.4 — We implement traditional baselines and metrics

First, create `src/models/traditional.py` for Stumpy, KMeansAD, and Isolation Forest.
Next, create `src/metrics.py` for aligned scores, predictions, and coverage.
Then, define budget boundaries, ties, missing-class behavior, and incomplete coverage.
Finally, keep undefined metrics undefined instead of replacing them with zero.

The main functions are `fit_traditional`, `score_traditional`, and `compute_metrics`.
They reject invalid budgets and report undefined metrics explicitly.
Tests cover fixed-input parity, budget boundaries, ties, missing classes, and coverage.

## Phase 5 — We make one manifest executable

The model paths are now independent and testable.
This phase connects them with one resolved manifest.
The tools are Python `itertools`/`subprocess`, JSON, pytest, W&B, CUDA visibility, and CPU affinity.

### Stage 5.1 — We build the manifest

First, create `src/matrix/manifest.py` for 28 entities and 3 seeds.
Next, add method, variant, phase, stage, output path, and W&B identity to each record.
Then, add the exact offline dependency to each online record.
Finally, write one resolved JSON manifest and check the expected counts.

The main functions are `build_manifest` and `validate_manifest`.
They reject duplicate identities, duplicate paths, and missing dependencies.
The checks must find 588 offline records, 1176 online records, and 2352 W&B stage identities.

### Stage 5.2 — We run preflight

First, create `src/matrix/preflight.py` that reads the manifest without creating models.
Next, check contract hash, config hash, entity, seed, variant, protocol, and paths.
Then, check that each online record points to the correct offline record.
Finally, write a report and return failure on any mismatch.

The public function is `run_preflight`.
It fails closed and never edits the manifest or reuses old artifacts.
Tests cover valid input, collisions, mismatches, and a stale output root.

### Stage 5.3 — We run worker queues

First, create `src/worker.py` that accepts exactly one manifest record.
Next, create `src/matrix/runner.py` and use `subprocess` only at this boundary.
Then, block online workers until all offline dependencies pass preflight.
Finally, assign resources from actual GPU visibility and CPU affinity and log the assignment.

The worker returns stage artifacts and an exit status.
It must not create a completion marker after failure.
Tests cover process failure, resource assignment, and the offline barrier.

### Stage 5.4 — We complete and retry records

First, create `src/matrix/completion.py` to check artifacts and identity before completion.
Next, build retry input from failed logical records only.
Then, use a new output root for a new contract hash and reject stale schemas.
Finally, confirm that retry does not add resumable online state.

The public functions are `validate_completion` and `make_retry_manifest`.
They treat incomplete output, wrong hash, and duplicate result as failures.
Tests cover missing artifacts, false markers, exact retry, and path collisions.

## Phase 6 — We verify one combination and run the matrix

The final phase protects the research run from an untested workflow.
We first check the environment, then run one complete development combination, then audit it, and only then run all records.
The tools are the new project's Python and pytest, JSON, W&B, `nvidia-smi`, `taskset`, and optional tmux.

### Stage 6.1 — We check the environment

First, check Python, dependencies, dataset path, and project revision without writes.
Next, run `nvidia-smi` and `taskset` to record GPU and CPU visibility.
Then, check W&B authentication and the output root with a small preflight.
Finally, save the environment report in the smoke root.

The proposed script is `scripts/environment_preflight.py`.
It returns a read-only report with revision, package versions, devices, and paths.
It stops before workers when a dependency, data, resource, or W&B check fails.

### Stage 6.2 — We run one development combination

First, create a separate smoke manifest for `machine-1-6` and seed `6`.
Next, record every reduced epoch or limit in that manifest.
Then, run THESIS, RedLamp, CANDI, M2N2, and all traditional baselines.
Finally, run online records only after their offline dependencies pass.

The proposed script is `scripts/run_smoke.py`.
It produces a complete smoke artifact set.
Any failed stage blocks smoke acceptance.
The checks cover one full path, checkpoint lineage, thresholds, metrics, and W&B lifecycle.

### Stage 6.3 — We audit the smoke run

First, audit smoke artifacts and completion markers.
Next, compare checkpoint identity, thresholds, metrics, and W&B names with the manifest.
Then, inspect one injected example and its mask pair.
Finally, mark the project matrix-ready only after a failed-record restart passes.

The proposed script is `scripts/audit_smoke.py`.
It must distinguish logical identity from attempt identity.
It must fail when smoke is treated as research evidence.

### Stage 6.4 — We run and audit the full matrix

First, run 588 offline records in a new wet output root.
Next, audit dependencies before running 1176 online records on the shared ranges.
Then, save required metrics and provenance without saving every forward output.
Finally, retry only failed logical records and report missing, failed, and duplicate records.

The proposed scripts are `scripts/run_full_matrix.py` and `scripts/audit_matrix.py`.
The expected result is 1764 logical results and 2352 successful W&B stage identities.
The final check reconciles the manifest, artifacts, and W&B records and confirms six-GPU/CPU readiness.

## Dependencies and terminology

The phase order is:

`Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6`.

Stage 3.4 is the first small end-to-end flow.
Phase 6 runs one complete combination before the full 28-entity matrix.

The following names remain explicit.

| Object | Treatment |
| --- | --- |
| Point-level contrastive → `two_view_contrastive_loss` | Unchanged exact alias |
| Historical O0/O1 → new O0/O1 | Same names, changed semantics; do not relabel old checkpoints |
| No old O2 runtime → O2 | New variant using contrastive only |
| Historical A1 → target A1 | Same name, changed semantics |
| Stage A / memory initialization / Stage B | Separate offline operations |
| `VerificationBuffer` / `NonOverlapGuard` | Separate online objects |
| Generated configs + process manifest | One resolved experiment manifest |

Historical checkpoints will not load automatically into the new project.
Old source and results remain comparison references.
The new codebase has not been created and no benchmark has been run.
