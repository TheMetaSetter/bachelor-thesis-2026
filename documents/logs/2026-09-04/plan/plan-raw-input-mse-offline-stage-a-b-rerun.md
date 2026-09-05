---
date: 2026-09-04T00:00:00+07:00
planner: OpenAI Codex
topic: "Re-run offline Stage A and Stage B with raw-input-space MSE"
status: local_implementation_verified_cloud_execution_handed_to_user
revision: 0fe7e6cae63ea114b8cc028ce9116ccffef1d1e1
branch: dev
related_research:
  - prompts/1_research_prompt.md
  - documents/logs/2026-09-04/research/research-raw-input-space-mse-score-change-surface.md
  - documents/spec/full-spec-v4.md
  - documents/spec/two-stage-offline-pretraining-spec.md
---

# Raw-input-space MSE Offline Rerun Implementation Plan

> **For agentic workers:** Use this plan only after the two scope gates are approved. Execute the phases sequentially and keep the remote GPU work behind the one-combination preflight.

**Goal:** Re-train offline Stage A and Stage B on the three requested SMD
entities, select each Stage B `best.pt`, and measure VUS-PR, Affiliation F1,
and Stage-B VUS-ROC with raw-input-space MSE.

**Architecture:** Keep scaled tensors as the model input. Use the fitted
training scaler to inverse-transform input and reconstruction inside the
reconstruction-loss boundary and the evaluation-score boundary. Keep
classification, contrastive, and latent regularization objectives unchanged
unless a new experiment specification explicitly replaces them.

**Tech stack:** PyTorch, the existing `scripts.train` and
`scripts.benchmarks.run_thesis_offline_benchmark` entry points, YAML configs,
pytest, Weights & Biases, SSH, and tmux on the GPU host described by
`cloud-gpu.txt`.

**Primary files:** `src/models/thesis_multitask_impl/`, `src/data/scalers.py`,
`src/engine/trainer.py`, `src/engine/evaluator.py`,
`scripts/experiments/run_two_stage_offline_pretraining.py`,
`scripts/benchmarks/run_thesis_offline_benchmark.py`, and the relevant YAML
files under `scripts/configs/experiment/offline_benchmark/thesis/`.

## Global constraints

- The model continues to receive standardized input; raw MSE is computed only after inverse transformation.
- Raw point MSE is the mean over features, raw window MSE is the mean over time, and MC scoring averages per-sample MSE before the MC reduction.
- Synthetic validation labels are metrics-only; clean validation alone fits thresholds.
- A normal point/window has label `0`; an anomalous point/window has label `1`; predictions are separate strict `score > threshold` results.
- Stage A and Stage B use the existing `25 + 5` main epoch budget, window size `20`, MC sample count `10`, and train-fitted scaler.
- Every stage retains its initialization checkpoint, best checkpoint, configuration, metric history, provenance, and checksums.
- The first remote run is one complete smoke/preflight combination. The full matrix starts only after that combination passes.
- Remote cleanup, if needed, targets only exact run paths or process IDs created by this work.
- Historical sigmoid artifacts and previous benchmark outputs are not overwritten.

## Scope decision gate

The current model has several distinct objectives. The code shows:

| Objective | Current form | Plan interpretation |
| --- | --- | --- |
| `reconstruction_loss` | MSE, optionally masked to non-injected positions | Change to raw-input-space MSE while preserving the mask and target semantics |
| O1 `score_loss` | Balanced BCE over reconstruction-derived point scores | Recompute its reconstruction error in raw input space; do not silently change BCE into MSE |
| `classification_loss` | Cross-entropy | Keep unchanged |
| `contrastive_loss` | Existing two-view contrastive objective | Keep unchanged |
| diversity/variance/covariance/usage/gate terms | Existing latent-space regularizers | Keep unchanged |

This is the smallest interpretation that makes all reconstruction-based losses
and anomaly scores use raw input units without removing the objectives that give
O0 and O1 their defined meanings. A literal replacement of every objective by
raw-input MSE would remove classification, contrastive, and latent objectives;
that would be a new experiment rather than the current O0/O1 rerun. Execution
must stop at this gate if that literal replacement is intended.

## Experiment matrix gate

The repository's locked offline benchmark matrix is O0/O1 ×
`machine_1_6`/`machine_3_4`/`machine_3_9` × seeds `6/8/36`, giving 18 main
cells. The existing main YAML files encode this matrix, with O0 as the base
two-stage variant and O1 as the point-score-supervised variant. The plan uses
this matrix unless the requested rerun is intentionally one seed or one
variant; that narrower choice must be recorded before remote launch.

## Sequential phases and stages

### Phase 0 — Reconfirm the contract and repository state

**Purpose:** Make the new training-loss meaning explicit before changing code or
starting a GPU job.

**Stages:**

1. Read `prompts/1_research_prompt.md`, `cloud-gpu.txt`,
   `codebase_preferences.md`, the raw-score research report, `full-spec-v4.md`,
   and the two-stage pretraining specification.
2. Compare the v4 raw-score contract with the earlier v1–v3 score terminology.
   Create a new experiment-specific specification version if changing the
   training objective would make v4's statement “training loss unchanged”
   false.
3. Trace the active call chain:
   `scripts.benchmarks.run_thesis_offline_benchmark` →
   `run_two_stage_offline_pretraining` → `scripts.train` → `Trainer.train` →
   model `_shared_step` → loss functions → Stage A/Stage B checkpoints →
   evaluator metrics.
4. Record the selected matrix, seeds, entity IDs, epoch budgets, protocol
   values, checkpoint monitor, metric buffer size, and metric threshold count.

**Tools:** `rg`, `sed`, `realpath`, git metadata, YAML parser, and the local
documentation tree. Do not connect to the remote host in this phase.

**Exit evidence:** A dated contract note identifies the raw loss formula, the
scope decision above, the matrix, and every source/config owner.

### Phase 1 — Implement raw-input reconstruction losses

**Purpose:** Make training and evaluation use the same raw-input reconstruction
error definition while preserving the model's scaled-input interface.

**Stages:**

1. Extend `SequenceStandardScaler` with a differentiable tensor inverse
   transform that preserves device, dtype, active-feature masking, and the
   existing `epsilon` rule.
2. Extend the model loss context so `ThesisMultitaskModel` receives the fitted
   scaler state or an equivalent immutable raw-space transform without fitting
   on validation or test data.
3. Change `_compute_reconstruction_loss` to inverse-transform both the model
   reconstruction and the actual post-injection input before MSE. Preserve
   `reconstruction_normal_only` and its clean-position mask.
4. Change `_compute_reconstruction_diagnostics` to report the raw loss as the
   primary training reconstruction loss and retain normalized MSE under an
   explicit diagnostic name.
5. Change O1 point-score supervision so its reconstruction-derived point error
   is raw-input-space MSE before the existing balanced reduction.
6. Verify Stage A and Stage B both receive the same raw-loss configuration;
   Stage B still freezes the encoder and memories according to the existing
   two-stage contract.

**Tools:** PyTorch autograd, `SequenceStandardScaler`, model loss mixins, and
the existing checkpoint scaler state. Do not add raw tensors to the model batch
unless the traced implementation proves that the loss boundary cannot use the
scaler state directly.

**Exit evidence:** A one-batch calculation shows the expected raw MSE and a
backward pass produces finite gradients. A synthetic anomaly remains excluded
from the masked reconstruction loss but remains included in the operational
anomaly score.

### Phase 2 — Extend configuration, checkpoint, and metric contracts

**Purpose:** Ensure a run cannot mix normalized training, raw evaluation, and
legacy sigmoid metadata without being detected.

**Stages:**

1. Add an explicit training reconstruction-loss-space field only if the current
   config loader requires it; resolve it to `raw_input` for this experiment.
2. Keep the evaluation protocol fields `score_space: raw_input` and
   `point_score_transform: identity` explicit.
3. Make the trainer's validation metric path use the raw scorer and the fitted
   scaler before it computes `val_synth_vus_pr`, because that metric selects
   `best.pt`.
4. Preserve `val_synth_vus_pr` as the Stage B best-checkpoint monitor with
   mode `max`, and record the monitor value in checkpoint metadata.
5. Confirm `compute_pointwise_metrics` receives the raw point-score timeline;
   it must emit `vus_pr`, `affiliation_f1`, and `vus_roc` from that timeline.
6. Keep VUS-ROC as a required Stage-B report metric even if VUS-PR remains the
   checkpoint-selection metric.

**Tools:** YAML validation, `src/core/config.py`, `Trainer`, `Evaluator`,
`src/metrics/pointwise.py`, threshold-artifact validation, and checkpoint
metadata inspection.

**Exit evidence:** A resolved config and a saved checkpoint both identify raw
loss/score space, identity transform, clean-validation threshold source, and
the exact Stage B monitor.

### Phase 3 — Add focused tests and local CPU preflight

**Purpose:** Catch mathematical, gradient, checkpoint, and leakage errors
before spending GPU time.

**Stages:**

1. Add scaler round-trip tests for active and inactive features, including
   dtype/device behavior and unfitted-scaler rejection.
2. Add hand-computed raw reconstruction-loss tests for clean and synthetic
   batches, including the normal-position mask.
3. Add MC tests proving the implementation computes mean(per-sample MSE), not
   MSE(mean reconstruction).
4. Add O0 and O1 tests for one Stage A forward/backward step and one Stage B
   forward/backward step; assert finite raw losses and the existing Stage B
   freeze contract.
5. Add checkpoint save/load tests proving scaler state, raw-loss identity, and
   Stage B memory state survive a round trip.
6. Add evaluator tests for raw thresholding, covered-point overlap aggregation,
   synthetic labels, window labels, and all three requested metrics.
7. Run the focused pytest set with `.venv/bin/python`; then run the relevant
   existing benchmark, model, evaluator, and checkpoint tests.
8. Run one local CPU end-to-end smoke using the smallest existing debug config.
   Label it as a functional preflight, not CUDA or benchmark evidence.

**Tools:** `.venv/bin/python -m pytest`, existing debug CPU YAML, temporary
pytest fixtures, and the repository's normal benchmark entry point.

**Exit evidence:** Focused tests pass; any unrelated baseline failures are
   recorded separately with their exact test names and are not attributed to
   this change.

### Phase 4 — Prepare exact remote GPU runs

**Purpose:** Reproduce the local source and configuration on the cloud GPU
   without exposing credentials or mixing run trees.

**Stages:**

1. Re-read `cloud-gpu.txt` immediately before connecting and resolve its host,
   port, user, remote repository root, and authentication method. Do not copy
   the password or private credential into logs, configs, W&B metadata, or this
   plan.
2. Perform a read-only SSH check for repository revision, Python launcher,
   CUDA availability, GPU identity, disk space, and existing processes.
3. Create one exact remote experiment root for this raw-loss rerun and sync the
   selected source files, configs, tests, and specification snapshot.
4. Record local and remote git revision, resolved-config hashes, Python/package
   versions, CUDA/PyTorch versions, and dataset availability.
5. Create one tmux session for the preflight run and one separately named
   session per full matrix group. Store stdout/stderr in each exact run's
   canonical stage directory.

**Tools:** SSH using the endpoint from `cloud-gpu.txt`, `tmux`, `rsync` or an
   equivalent scoped transfer, `.venv/bin/python`, `nvidia-smi`, and SHA256
   hashing. The remote write is authorized by the requested GPU rerun; cleanup
   remains exact-path only.

**Exit evidence:** The remote environment can import the changed code, sees the
   expected CUDA device, and has no unrelated process or artifact mutation.

### Phase 5 — Run one complete GPU preflight

**Purpose:** Verify the full Stage A → memory initialization → Stage B → best
   checkpoint → raw evaluation flow on one concrete combination.

**Stages:**

1. Use one selected main combination, preferably O0, `machine_1_6`, seed 6,
   with the main `25 + 5` budget unless the development specification defines
   a separate reduced CUDA preflight.
2. Run Stage A and verify raw train/validation/synthetic-validation losses,
   memory initialization, checkpoint creation, and W&B logging.
3. Build the Stage B initialization checkpoint from Stage A `best.pt` using
   train data only.
4. Run Stage B and verify frozen encoder/memory gradients, raw losses, and
   creation of Stage B `best.pt` and `final.pt`.
5. Load Stage B `best.pt`, not `final.pt`, and run clean validation, synthetic
   validation, and test inference with raw MSE.
6. Verify thresholds come only from clean validation and that the evaluation
   report contains VUS-PR, Affiliation F1, and VUS-ROC.
7. Stop and diagnose if the preflight has missing classes, missing metrics,
   non-finite scores, absent scaler state, wrong checkpoint role, or any
   normalized/sigmoid operational field.

**Tools:** Existing two-stage runner, benchmark wrapper, W&B, tmux log tail,
and the raw-score evaluator.

**Exit evidence:** One complete run has a valid Stage A checkpoint, Stage B
   initialization checkpoint, Stage B `best.pt`, raw threshold artifact,
   metrics report, and provenance manifest.

### Phase 6 — Run the requested offline matrix

**Purpose:** Produce comparable raw-MSE offline results for all selected
   entities and variants.

**Stages:**

1. Launch the full locked matrix only after Phase 5 succeeds.
2. Run each cell through the existing two-stage orchestrator in the order
   `machine_1_6`, `machine_3_4`, `machine_3_9`; keep each seed and variant in a
   separate canonical output directory.
3. For every cell, retain Stage A initialization/best checkpoints, Stage B
   initialization/best checkpoints, metric history, scaler state, and hashes.
4. Evaluate only Stage B `best.pt` for the final offline report. Do not replace
   a missing Stage B checkpoint with a RedLamp checkpoint, a debug CPU
   checkpoint, or `final.pt`.
5. Compute point-level raw MSE on covered timeline points and window-level raw
   MSE on windows. Keep normal/anomalous labels separate from predictions.
6. Compute VUS-PR and Affiliation F1 for the requested offline report and
   compute VUS-ROC explicitly for the Stage-B result. Preserve metric support,
   label regime, threshold, VUS buffer size, and threshold count.
7. Log every run and summary statistic to W&B, then perform exact artifact
   readback before moving to the next group.

**Tools:** `scripts/benchmarks/run_thesis_offline_benchmark.py`,
`scripts/evaluate.py`, `.venv/bin/python`, W&B, and canonical output paths:
`outputs/benchmark/smd/<entity>/<seed>/thesis/<phase>/<stage>/`.

**Exit evidence:** Every selected cell either has a complete, checksum-verified
   report or a precise failure record. No cell is silently omitted.

### Phase 7 — Audit and report the results

**Purpose:** Make the rerun reproducible and prevent score-space or checkpoint
   confusion in later analysis.

**Stages:**

1. Read every result manifest and verify entity, seed, variant, window size,
   stride, scaler provenance, raw-loss identity, checkpoint role, and hashes.
2. Build one report table with entity, seed, variant, Stage A best checkpoint,
   Stage B best checkpoint, raw reconstruction losses, VUS-PR, Affiliation F1,
   VUS-ROC, support counts, and runtime.
3. Mark missing or one-class metrics as unavailable with the reason; never
   replace them with zero.
4. State that raw-MSE threshold magnitudes are not directly rankable across
   entities because sensor units and channel scales differ.
5. Keep old sigmoid artifacts and historical results unchanged, and label all
   new artifacts as the raw-training/raw-score protocol.
6. Run the final focused test suite and `git diff --check`; report unrelated
   baseline failures separately.

**Tools:** Existing offline metric-table renderer, JSON/NPZ readers, SHA256,
pytest, and Markdown report files under `documents/logs/2026-09-04/`.

**Exit evidence:** A report-ready table and provenance package identifies the
   exact Stage B `best.pt` used for every metric and demonstrates that no
   calibrated sigmoid score entered thresholding or prediction.

## Implementation review gate

Before any code or remote run, confirm two points:

1. “All losses use raw-input-space MSE” means the scope decision above:
   reconstruction-based MSE terms move to raw units, while cross-entropy,
   contrastive, and latent regularizers keep their defined formulas.
2. The requested rerun uses the locked 18-cell O0/O1 × three entities × three
   seeds matrix, rather than a narrower variant/seed selection.

After those points are approved, execute Phases 1–3 locally, then Phases 4–7
on the cloud GPU.

## Execution update — 2026-09-05

The user approved both scope gates and requested edits in the current worktree.
The user then requested manual cloud commands. No remote connection or GPU run
was performed by the agent.

Local changes use the existing model, trainer, checkpoint loader, and two-stage
runner. The manual CLI calls the two-stage runner directly: it already selects
Stage B `best.pt` and runs `scripts.evaluate`. This avoids adding another runner
or requiring online-threshold/UQ exports for the requested offline metrics.
Raw loss identity is an explicit experiment config field because historical
normalized-loss checkpoints must remain distinguishable.

One reduced CPU flow completed both stages and raw test evaluation. Its data,
model size, epochs, and VUS settings are reduced, so its metrics are functional
evidence only. Stage B encoder and memory tensors remained unchanged.
The cloud 25+5 preflight and full matrix remain for the user to run.

See `documents/spec/raw-input-mse-training-addendum.md` for the contract and
`documents/logs/2026-09-05/detail/raw-mse-offline-cloud-cli.md` for commands.
Independent subagent review was unavailable because the reviewer hit its usage
limit; local tests, source review, and CPU artifact checks were performed.
