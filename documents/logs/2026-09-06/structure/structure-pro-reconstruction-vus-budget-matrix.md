---
date: 2026-09-06 Asia/Ho_Chi_Minh
topic: "Pro-reconstruction offline matrix with per-run VUS-PR FPR-budget checkpoint monitoring"
status: approved-by-request
revision: cc42b4e3faa433d9d81e33719dfab0cbbe1cee66
related_documents:
  - documents/notes/pro-reconstruction-offline-fairness-decisions.md
  - documents/logs/2026-09-06/research/research-pro-reconstruction-code-modification-surfaces.md
  - documents/logs/2026-09-06/plan/plan-pro-reconstruction-vus-budget-matrix.md
---

# Implementation Structure: Pro-reconstruction VUS-budget matrix

## Summary

The delivery path first establishes one precise score and threshold contract, then makes it operational, then binds one budget to every main-method training run, then evaluates baselines, and finally validates one complete smoke flow before remote rollout.

## Request

The implementation must run 54 main-method combinations and use one VUS-PR FPR budget per combination to monitor both training stages.

## Confirmed context

- `src/engine/trainer.py:901-903` requires one float monitor value for checkpoint selection.
- `src/metrics/pointwise.py:749-778` already returns all three budget mappings.
- `src/engine/evaluator.py:588-722` currently treats raw-input MSE as operational reconstruction score.
- `scripts/benchmarks/generate_full_direct_recon075_cls025_matrix.py:26-28` currently creates 18 base combinations.

## Scope

### In scope

- Offline protocol, evaluator, trainer, artifacts, main matrix, RedLamp evaluation, traditional baseline evaluation, tests, smoke run, and remote-launch preparation.

### Out of scope

- Model architecture changes, baseline retraining, online evaluation, M2N2, CANDI, and remote execution during implementation.

## Proposed phases

### Phase 1: Establish an isolated normalized synthetic-normal protocol

**Result:** The repository can validate a new protocol with normalized-input MSE and two synthetic-validation-normal q99 thresholds while old raw-input protocol files remain valid.

**Stages:**

1. Define the protocol keys and artifact identity for normalized-input point and window MSE.
2. Add point/window synthetic-normal threshold selectors and their input validation.
3. Protect the new and historical contracts with focused tests.

**Depends on:** The agreed score, threshold, and three-budget decisions.

**Verification:** Automated threshold and artifact tests pass, and protocol loading rejects missing or contradictory fields.

**Risks:** Artifact compatibility risk is contained by a new schema version rather than reinterpretation of raw artifacts.

**Complete when:** A protocol config and threshold artifact can unambiguously state normalized MSE plus both synthetic threshold sources.

### Phase 2: Make normalized MSE and dual thresholds drive reconstruction evaluation

**Result:** THESIS and RedLamp evaluation select normalized MSE as score, calibrate both threshold levels from synthetic validation, and make predictions with the matched score type.

**Stages:**

1. Extend evaluator score-space selection and record reconstruction point/window scores with a shared identity.
2. Route the THESIS offline benchmark through synthetic point and window calibration before test inference.
3. Route RedLamp checkpoint evaluation through the same protocol without changing checkpoint weights or model hyperparameters.
4. Verify point and window predictions, labels, thresholds, and serialized outputs end to end.

**Depends on:** Phase 1 protocol and threshold selectors.

**Verification:** Evaluator and checkpoint-evaluation tests prove normalized MSE is the metric input and that test labels never influence threshold selection.

**Risks:** Raw and normalized arrays can be mixed accidentally, so all outputs must carry score-space and score-definition provenance.

**Complete when:** A reconstruction checkpoint can produce synthetic-calibrated normalized-MSE point/window predictions and all three budgeted VUS outputs.

### Phase 3: Bind one VUS-PR budget to each main-method training run

**Result:** The trainer receives an unambiguous scalar monitor metric, and the matrix expands to 54 isolated budget-specific two-stage combinations.

**Stages:**

1. Compute normalized-input MSE synthetic-validation scores during trainer validation and flatten the selected budget to a scalar epoch metric.
2. Allow the three new monitor names in checkpoint-monitor validation and preserve the existing single-float save path.
3. Generate `fpr001`, `fpr005`, and `fpr01` configurations for every O0/O1/entity/seed base cell.
4. Propagate the budget identity to Stage A, Stage B, W&B, checkpoints, manifests, and output paths.
5. Preflight all 54 configurations and assert exactly 108 stage run identities.

**Depends on:** Phase 2 score identity and all three VUS maps.

**Verification:** Trainer tests prove one configured budget selects `best.pt`, and generator tests prove unique config and output identities.

**Risks:** A nested metric cannot be cast to `float`, and an omitted budget path component can overwrite a sibling run.

**Complete when:** Every materialized Stage A and Stage B config monitors the budget stated in its run identity.

### Phase 4: Apply the threshold contract to baselines without altering methods

**Result:** RedLamp, iForest, KMeans-AD, and StumPy evaluate under the agreed calibration rule while preserving existing model weights, fit scope, and native score semantics.

**Stages:**

1. Expose native baseline point and window scores without changing the historical calibrated-score API.
2. Score synthetic validation before filtering normal point/window values and derive two q99 thresholds.
3. Persist new baseline artifacts and metrics in an isolated root.
4. Verify each baseline keeps its train-only learned state and its stated score direction.

**Depends on:** Phase 1 threshold artifacts and Phase 2 evaluation-output conventions.

**Verification:** Traditional-baseline contract and runner tests prove native scores, synthetic threshold sources, and both prediction levels.

**Risks:** Baseline score transformations can silently violate fairness, so tests assert the raw method-specific score before any threshold selection.

**Complete when:** Every in-scope baseline evaluation writes the same calibration provenance fields as the main method, except for its method-specific native score definition.

### Phase 5: Validate a complete smoke flow and prepare remote rollout

**Result:** One budget-specific main-method flow and baseline checks provide local evidence before the 54-combination remote run is allowed.

**Stages:**

1. Run focused unit and integration suites for contracts, evaluation, monitoring, and baselines.
2. Run one Stage A → Stage-B initialization → Stage B → evaluation smoke combination for one explicit budget.
3. Inspect generated checkpoint, manifest, thresholds, metrics, and W&B identity.
4. Re-read `ssh-gpu.txt`, verify remote revision and disk location, then prepare the non-destructive remote command.

**Depends on:** Phases 1 through 4.

**Verification:** The smoke run selects one budget-specific best checkpoint and writes both VUS maps with all three budgets.

**Risks:** GPU time is consumed before a contract failure appears, so Phase 5 blocks the matrix launch until the smoke result is accepted.

**Complete when:** The smoke result satisfies all provenance and metric checks and the remote command targets a new output root.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Agreed contract | Valid score, threshold, and artifact definitions |
| 2 | Phase 1 | Correct reconstruction and RedLamp evaluation |
| 3 | Phase 2 | 54 main-method budget-specific combinations |
| 4 | Phases 1 and 2 | Fair baseline evaluation |
| 5 | Phases 1 through 4 | Safe smoke gate and remote rollout preparation |

## Decisions confirmed

- One base O0/O1/entity/seed cell is repeated for each FPR budget, which creates 54 main-method combinations.
- Each budget-specific combination selects checkpoints with one VUS-PR budget metric.
- Each combination still runs Stage A and Stage B, which creates 108 W&B training runs.
- Evaluation calculates and persists VUS-PR and VUS-ROC for all three budgets in one pass.
- Baseline evaluation does not multiply baseline training or checkpoint evaluation by FPR budget.

## Non-blocking uncertainties

- The implementation will use a proposed isolated output root and include the budget in its leaf run identity; the final remote location remains an operational decision after reading `ssh-gpu.txt`.
- The active code has no literal `pro-reconstruction` runtime label; the generated experiment-name suffix will carry the budget while preserving the existing O0/O1 terminology.

## Feedback requested

The user explicitly requested the complete phase, stage, and atomic-step sequence in one turn, so this structure is marked approved-by-request and is expanded in the accompanying detail document.
