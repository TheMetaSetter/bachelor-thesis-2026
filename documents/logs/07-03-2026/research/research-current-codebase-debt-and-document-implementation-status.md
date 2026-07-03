---
date: 2026-07-03 15:31:54 +07
researcher: Codex
git_commit: 0fd6abfb39ed704bf41a83845ec826b577ebdd94
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase technical debt and document implementation status"
tags: [research, codebase, technical-debt, documents, thesis_multitask, current-state]
status: complete
last_updated: 2026-07-03
last_updated_by: Codex
---

# Research: Current codebase technical debt and document implementation status

**Date**: 2026-07-03 15:31:54 +07  
**Researcher**: Codex  
**Git Commit**: `0fd6abfb39ed704bf41a83845ec826b577ebdd94`  
**Branch**: `dev`

## Research Question

Use the repository-grounded research workflow to scan the current codebase, identify technical debt and low-quality engineering surfaces, then classify design documents by their relationship to the implementation that exists today: outdated, not implemented, partially implemented, or no longer relevant as an active implementation contract.

## Summary

The current repository is not a broken codebase. It is a **transition-heavy** codebase. The main runtime for offline SMD training, evaluation, three-stage orchestration, and the first online adaptation slice is real and tested. The main debt is that several files still carry **multiple generations of semantics at once**.

The biggest technical-debt surface is `src/models/thesis_multitask.py`. It is now the single owner of the thesis model, but it still contains both older and newer prototype-query and fusion semantics in one file. The second biggest debt is `src/core/config.py`, which has become a large centralized semantic choke point. The third debt is doc/runtime drift: several design documents still describe earlier intended semantics more cleanly than the code now exposes, while the code intentionally preserves compatibility and ablation toggles.

The document situation is mixed. A few documents remain useful and broadly aligned. Several are only partially implemented. Several are now historically useful but should not be treated as the active runtime contract without checking code first.

## Detailed Findings

### 1. High-confidence technical debt in the current code

#### A. `src/models/thesis_multitask.py` is a mini-subsystem, not just one readable model file

- The file is `3255` lines long and is doing too many jobs at once under the `1 model - 1 file` rule.
- It still contains:
  - legacy-compatible Stage 3 naming (`src/models/thesis_multitask.py:37-38`)
  - both `gumbel_softmax` and `cosine_topk` discrete-query modes (`src/models/thesis_multitask.py:308-312`, `1934-1968`)
  - EMA-updated discrete memory state (`src/models/thesis_multitask.py:951-966`, `1740-1804`)
  - both scalar fusion and concat-projection fusion (`src/models/thesis_multitask.py:2006-2125`)
  - optional CKA-gated forward fusion (`src/models/thesis_multitask.py:2056-2078`)
- This is ablation-friendly, but it also means the file mixes:
  - active runtime behavior
  - compatibility behavior
  - experiment-family toggles
  - transitional historical semantics

#### B. `src/core/config.py` is a monolithic semantic choke point

- The file is `1434` lines long.
- `validate_experiment_config(...)` starts at `src/core/config.py:199` and then owns:
  - supported dataset/model/task surfaces (`300-318`)
  - allowed key surfaces for each model and task (`320-496`)
  - type validation (`498-662`)
  - optimizer and scheduler semantics (`663-760`)
  - three-stage normalization and validation (`90-186`, reused before runtime build)
- This centralization is good for fail-fast behavior, but bad for skimmability. A reader has to hold many unrelated semantic layers in one pass.

#### C. Runtime wiring is duplicated across entry scripts

- `scripts/train.py:44-52`
- `scripts/evaluate.py` has a parallel registration surface
- `scripts/run_online_adaptation.py:43-50`
- Each script separately registers datasets and models and separately rebuilds model kwargs from resolved config. This is not a correctness bug, but it is a real drift surface.

#### D. There are still stale comments and low-signal TODOs in active metric code

- `src/metrics/pointwise.py:552-553` still says TODO for not using point-adjusted metrics and for adding `VUS-PR`.
- But the same file already computes `affiliation_f1`, `vus_pr`, and `vus_roc` in the active path (`src/metrics/pointwise.py:576-587` and later lines).
- This is a small line-level quality issue, but it matters because metric semantics are thesis-critical.

#### E. Active config tree still mixes old `w100` families with newer `w20` / three-stage families

- The repo now clearly contains newer `window20`, `three_stage`, and benchmark/comparative config families.
- But older `w100` experiment families still remain active-looking, for example:
  - `configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml:1-13`
  - `configs/experiment/ablation/smd__thesis_multitask__multitask-continuous-only__w100__seed7__default.yaml:1-17`
- This is not necessarily wrong if they are preserved intentionally, but it increases the risk that a reader treats an older thesis surface as the current default path.

### 2. Current implementation surfaces that are actually real and verified

These are the parts that should be treated as implemented today, not just planned:

- Offline SMD data path with train-only scaling before windowization:
  - `src/data/loaders.py:142-178`
- Synthetic anomaly injection at window level:
  - `src/data/augment.py:1-260`
- Offline training entrypoint:
  - `scripts/train.py:1-220`
- Overlap-aware evaluator reconstructing pointwise timelines:
  - `src/engine/evaluator.py:68-167`
- Three-stage offline orchestration:
  - `scripts/run_three_stage_offline_pretraining.py:126-299`, `619-737`
- First online adaptation slice:
  - `src/data/stream.py:37-244`
  - `src/models/online_adaptation.py:1-260`
  - `scripts/run_online_adaptation.py:1-260`

### 3. Document status against the codebase as it exists today

#### A. Broadly aligned and still relevant

1. `documents/design/long_term_codebase_roadmap.md`
- Reason: it already starts from the claim that offline thesis path and conservative online adaptation slice both exist (`documents/design/long_term_codebase_roadmap.md:9-17`, `100-117`).
- Status: relevant high-level roadmap, not low-level contract.

2. `documents/design/sequence_dataset_loader_strategy_design.md`
- Reason: the shared sequence pipeline it describes is already close to the current data loader architecture (`documents/design/sequence_dataset_loader_strategy_design.md:41-52`, `77-109`), and the current repo does have `src/data/base.py`, `loaders.py`, `window.py`, parser modules, and a stable batch contract.
- Status: mostly relevant design direction, but still only partially realized for broader datasets.

#### B. Partially implemented

1. `documents/design/design_starter.md`
- It proposes a folder and contract structure that is only partly realized.
- Implemented parts:
  - `src/data/api.py`, `cleaning.py`, `download.py`, `window.py`, `scalers.py`, `collate.py`, `loaders.py`
  - `src/models/base_model.py`, `reconstruction_mlp_ae.py`, `thesis_multitask.py`, `online_adaptation.py`
  - `src/adapters/moment.py`
  - `src/engine/checkpoint.py`, `logger.py`, `artifact_sinks.py`
- Not implemented as written:
  - `configs/data/msl.yaml`, `custom.yaml`
  - `src/data/datasets/msl.py`, `custom_csv.py`
  - `src/metrics/eventwise.py`, `uncertainty.py`
  - `src/core/paths.py`
  - `src/utils/`
- Evidence: proposed tree at `documents/design/design_starter.md:84-169`.

2. `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- Implemented parts:
  - `enable_two_view_contrastive`
  - `enable_cka_gated_fusion`
  - `window_size = 20`
  - batch-level synthetic mask handling
- Evidence in code:
  - `src/models/thesis_multitask.py:284-312`
  - `src/models/thesis_multitask.py:2056-2078`
  - `src/models/thesis_multitask.py:2268-2283`
- Not fully matching as an active global contract anymore:
  - it describes the `Exp2` design family cleanly, but current runtime also has a later three-stage family.
- Status: implemented for one experiment family, no longer sufficient as the single active thesis contract.

3. `documents/design/stream_design.md`
- Implemented parts:
  - stream wrapper exists
  - online batcher exists
  - causal sequential cursor/state exists
  - online loop exists
- Evidence:
  - `src/data/stream.py:21-244`
  - `scripts/run_online_adaptation.py:166-207`
- Not implemented as written:
  - River, tsaug, TSGM, MOA stack is not present in the runtime
  - generic multi-dataset stream wrappers are not present
- Status: partly implemented conceptually, but the concrete chosen tooling stack in the note is not the current runtime stack.

4. `documents/design/experiment_config_organization_guideline.md`
- Implemented parts:
  - grouped config tree exists (`baseline`, `ablation`, `scale`, `smoke`, `thesis`, `comparative`, `benchmark`, `archive`)
  - metadata comment headers exist in active configs
- Evidence:
  - sample active config `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml:1-13`
- Not fully aligned:
  - the note says monitor-metric naming should align with `val_realistic_*` (`documents/design/experiment_config_organization_guideline.md:79-92`)
  - the active config/runtime surface now heavily uses `val_synth_*` and `val_vus_pr` (`src/core/config.py:690-706`)
- Status: partially implemented, partially outdated naming guidance.

#### C. Outdated or drifted relative to current code

1. `documents/design/idea.md`
- It says the current main discrete-query design is top-k codebook query and that Gumbel-Softmax is historical only (`documents/design/idea.md:60-103`).
- Current code still keeps both query modes in the owning model and still defaults the runtime dataclass to `gumbel_softmax` unless overridden (`src/models/thesis_multitask.py:308-312`, `1934-1968`).
- It also says the codebook is frozen by default after initialization (`documents/design/idea.md:94-103`), while the model still supports EMA-updated memory and conditional memory updates (`src/models/thesis_multitask.py:1740-1804`, `2315-2335`).
- Status: important terminology source, but outdated as a literal description of the implementation.

2. `documents/design/offline_pretraining_three_stage_first_implementation_spec.md`
- This spec says the old active path semantics must be removed or bypassed:
  - remove active Gumbel dependence
  - remove EMA-updated memory behavior
  - keep CKA diagnostic-only
  - use concat-projection fusion
  - use stage-aware behavior
  - `documents/design/offline_pretraining_three_stage_first_implementation_spec.md:202-216`
- Current code has indeed added stage-aware behavior and concat-projection support, but it still keeps:
  - Gumbel query mode
  - EMA state
  - optional forward-path CKA gating
- Evidence:
  - `src/models/thesis_multitask.py:1740-1804`
  - `src/models/thesis_multitask.py:1934-1968`
  - `src/models/thesis_multitask.py:2056-2078`
- Status: partially implemented, but the spec is no longer a clean description of the actual model file today.

3. `documents/design/stream_design.md`
- The note still frames the stack around River/tsaug/TSGM/MOA (`documents/design/stream_design.md:9-18`, `41-127`).
- The actual repo runtime is a custom `SMDOnlineStream` + `OnlineWindowBatcher` + `OnlineLoop` stack with no River dependency.
- Status: outdated as a concrete implementation guide, though still useful as older design rationale.

#### D. Implemented but superseded as active thesis contract

1. `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- This is not “wrong,” but it now describes the older `Exp2` family more than the full current thesis runtime.
- The three-stage orchestration and benchmark/comparative config families have moved the active surface forward.
- Status: implemented and still useful for historical `Exp2`, but superseded as the single active contract.

#### E. No longer relevant as an active implementation contract

1. `documents/design/design-contrastive-loss-21-jun-2026.md`
- This file is a raw ChatGPT export rather than a repository-native spec (`documents/design/design-contrastive-loss-21-jun-2026.md:1-18`).
- It is useful as historical discussion context, but it should not drive current implementation decisions directly.
- Status: historical conversation artifact, not active SSOT.

## Verification Performed

Ran:

```bash
pytest -q tests/test_config_loading.py tests/test_registry.py tests/test_online_entrypoint.py tests/test_three_stage_phase_runtime.py tests/test_evaluation_protocol_audit.py
```

Result:

- `131 passed in 5.08s`

This verifies that the key config, registry, online entrypoint, three-stage phase, and evaluation-audit surfaces described above are currently healthy.

## Code References

- `src/models/thesis_multitask.py:37-38` - legacy Stage 3 naming still preserved
- `src/models/thesis_multitask.py:308-312` - dual query/fusion runtime mode surface
- `src/models/thesis_multitask.py:937-966` - discrete assignment + EMA buffers
- `src/models/thesis_multitask.py:1740-1804` - EMA discrete codebook update path
- `src/models/thesis_multitask.py:1934-1968` - `cosine_topk` and `gumbel_softmax` both remain active
- `src/models/thesis_multitask.py:2006-2125` - concat projection and scalar/CKA fusion both remain active
- `src/core/config.py:199-760` - centralized semantic validation surface
- `src/metrics/pointwise.py:552-553` - stale TODO comments in active metrics file
- `src/data/loaders.py:142-178` - train-only scaling and bundle build
- `src/engine/evaluator.py:68-167` - overlap-aware pointwise reconstruction
- `src/data/stream.py:37-244` - actual current online stream implementation
- `scripts/run_three_stage_offline_pretraining.py:126-299` - three-stage manifest and config materialization
- `scripts/run_three_stage_offline_pretraining.py:619-737` - actual stage execution path

## Open Questions

1. Does the repository still want `thesis_multitask.py` to preserve both the older `Exp2` semantics and the newer three-stage semantics in one owner file, or should one of those families be downgraded to purely historical configs?
2. Which document should now be treated as the lowest-level active thesis contract: the three-stage spec, the benchmark/comparative config families, or the code in `src/models/thesis_multitask.py` itself?
3. Should older `w100` thesis experiment families remain visibly active, or should they be moved deeper into `archive/` to reduce misreads?

## Follow-up: Recommended Debt-Cleanup Priority

This follow-up prioritizes cleanup work by impact on thesis correctness, runtime readability, and risk of future misunderstanding.

### Priority 0: Clarify the single active thesis runtime contract

This should happen first because the current confusion is not mainly caused by broken code. It is caused by multiple active-looking semantics coexisting.

- Decide whether the active thesis contract is:
  - the newer three-stage family,
  - the older `Exp2` two-view + CKA family,
  - or both, with one explicitly marked as historical.
- Then reflect that decision consistently in:
  - `documents/design/idea.md`
  - `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
  - `documents/design/offline_pretraining_three_stage_first_implementation_spec.md`
  - `configs/experiment/thesis/`
  - `src/models/thesis_multitask.py`

Reason: until this is explicit, every later cleanup risks preserving the wrong “active” semantics.

### Priority 1: Reduce mixed active-path semantics inside `src/models/thesis_multitask.py`

This is the highest-impact code cleanup.

Recommended order:

1. Separate active-path semantics from compatibility-only semantics inside the file.
2. Make one query mode clearly default and one explicitly legacy or ablation-only.
3. Make one fusion mode clearly default and one explicitly legacy or ablation-only.
4. Decide whether EMA-updated memory is still part of the accepted thesis method or only an old branch.

Files most affected:

- `src/models/thesis_multitask.py`
- `configs/model/thesis_multitask*.yaml`
- `configs/experiment/thesis/`
- `configs/experiment/ablation/`
- targeted tests around query/fusion/runtime modes

Reason: this one file currently creates the biggest gap between “what the thesis says” and “what the runtime can still do.”

### Priority 2: Clean the experiment-config surface so old families stop looking active by accident

Recommended actions:

1. Review all `w100` thesis configs and decide which are:
  - still intentionally runnable,
  - historical but useful,
  - or effectively superseded.
2. Move clearly superseded configs deeper into `configs/experiment/archive/`.
3. Keep only one obvious active thesis path per concern:
  - benchmark
  - comparative
  - three-stage thesis
  - ablation

Strong candidates for review:

- `configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml`
- `configs/experiment/ablation/smd__thesis_multitask__multitask-continuous-only__w100__seed7__default.yaml`
- `configs/experiment/scale/smd__thesis_multitask__multitask-rtx3090-*.yaml`

Reason: right now old `w100` and newer `w20` / three-stage families coexist with similar visibility.

### Priority 3: Split semantic burden in `src/core/config.py`

This is important, but it comes after thesis-model contract cleanup.

Recommended direction:

1. Isolate model-specific key validation from task-specific validation.
2. Isolate three-stage normalization logic from generic config checks.
3. Isolate evaluation/checkpoint-monitor validation from optimizer validation.

Files most affected:

- `src/core/config.py`
- `tests/test_config_loading.py`
- tests for three-stage config semantics

Reason: the file is stable and tested, so this is more of a maintainability cleanup than an urgent correctness fix.

### Priority 4: Unify entry-script runtime registration and model-build wiring

Recommended actions:

1. Extract shared dataset/model registration from:
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/run_online_adaptation.py`
2. Extract shared “build model from resolved experiment config” logic where possible without hiding runtime ownership.

Reason: this reduces drift between entrypoints, but it is lower priority than clarifying the thesis method itself.

### Priority 5: Fix stale comments and document-only drift

These are cheap fixes and should be bundled once the active contract is settled.

Examples:

- update stale TODO comments in `src/metrics/pointwise.py`
- mark `documents/design/design-contrastive-loss-21-jun-2026.md` as historical context only
- revise `documents/design/stream_design.md` so it no longer reads like the current implementation stack
- revise `documents/design/experiment_config_organization_guideline.md` to reflect current metric namespaces

Reason: these issues are real, but they are mostly interpretation debt, not runtime risk.

### Practical recommendation

If the goal is to improve the repo with the least wasted effort, the best order is:

1. lock one active thesis contract
2. simplify `thesis_multitask.py` around that contract
3. archive or relabel superseded thesis configs and design docs
4. only then refactor shared config/runtime plumbing

This order gives the highest reduction in misunderstanding per unit effort.

## Follow-up: Concrete Execution Checklist

This checklist is intentionally operational. It is not a redesign plan. It is the shortest practical path to clean the repo without losing the working runtime.

### Checklist A: Lock one active thesis contract first

Goal:

- stop the repo from simultaneously implying that both `Exp2` and the later three-stage family are the main thesis method

Files to review first:

- `documents/design/idea.md`
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- `documents/design/offline_pretraining_three_stage_first_implementation_spec.md`
- `configs/experiment/thesis/exp2/`
- `configs/experiment/thesis/exp3/`
- `configs/experiment/thesis/exp4/`
- `configs/experiment/benchmark/thesis/`
- `configs/experiment/comparative/thesis/`

Concrete action:

1. Pick one label for the active thesis method.
2. Mark the other family as either:
   - `historical`
   - `ablation-only`
   - or `superseded`
3. Update document wording so “current active” appears in only one place.

Tests to keep green:

- `tests/test_config_loading.py`
- `tests/test_three_stage_phase_runtime.py`
- `tests/test_comparative_config_loading.py`

### Checklist B: Simplify `thesis_multitask.py` around the chosen contract

Goal:

- make the file readable without silently breaking old artifacts

Files to touch:

- `src/models/thesis_multitask.py`
- `configs/model/thesis_multitask*.yaml`
- `tests/test_thesis_multitask_config_refactor.py`
- `tests/test_multitask_shapes.py`
- `tests/test_one_multitask_train_step.py`
- `tests/test_fusion_ablation_modes.py`
- `tests/test_multitask_memory_updates.py`
- `tests/test_multitask_memory_initialization.py`

Concrete action:

1. Add one clearly named comment block or section boundary for:
   - active runtime path
   - compatibility path
   - ablation-only path
2. Make the default query mode explicit in config and code.
3. Make the default fusion mode explicit in config and code.
4. If EMA memory remains, label it clearly as active or legacy.

Do not do yet:

- large structural file split
- changing batch/output contracts

### Checklist C: Clean the config tree so older thesis presets stop pretending to be current

Goal:

- reduce accidental misreads from old `w100` presets

Files and folders to inspect:

- `configs/experiment/baseline/`
- `configs/experiment/ablation/`
- `configs/experiment/scale/`
- `configs/experiment/archive/`

Concrete action:

1. For each old `w100` thesis config, assign one status:
   - keep active
   - archive
   - rename comment metadata only
2. If a config is superseded, move it under `configs/experiment/archive/`.
3. If kept, change comment metadata so its role is obvious.

Tests to keep green:

- `tests/test_config_loading.py`
- `tests/test_ablation_runner.py`
- `tests/test_comparative_config_loading.py`

### Checklist D: Fix document-only drift after contract cleanup

Goal:

- make `documents/` stop disagreeing with code on the most visible surfaces

Files to update:

- `documents/design/idea.md`
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- `documents/design/offline_pretraining_three_stage_first_implementation_spec.md`
- `documents/design/stream_design.md`
- `documents/design/experiment_config_organization_guideline.md`

Concrete action:

1. Remove or relabel any sentence that says a legacy path is the current main path.
2. Keep historical notes, but mark them as historical explicitly.
3. Align metric namespace wording with current runtime names.

No code change required unless docs reveal a real contract bug.

### Checklist E: Clean cheap low-risk debt

Goal:

- remove small misleading artifacts after the big semantic decisions are done

Files to update:

- `src/metrics/pointwise.py`
- `src/core/config_help.py`
- helper shell launchers that still teach legacy paths

Concrete action:

1. Delete or rewrite stale TODO comments already contradicted by implementation.
2. Update example commands to point at the active config family.
3. Check launcher comments and example paths for legacy naming drift.

Tests to keep green:

- `tests/test_evaluation_metrics_audit.py`
- `tests/test_evaluation_protocol_audit.py`

### Checklist F: Refactor shared runtime plumbing last

Goal:

- reduce duplication across entrypoints only after semantics are stable

Files to inspect:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`

Concrete action:

1. Compare the three registration surfaces side by side.
2. Extract only the clearly duplicated pieces.
3. Do not hide model-specific runtime behavior just for DRY.

Tests to keep green:

- `tests/test_registry.py`
- `tests/test_online_entrypoint.py`
- `tests/test_config_loading.py`

### Minimal safest execution order

If the cleanup has to be done with minimal risk, the safest exact order is:

1. Checklist A
2. Checklist B
3. Checklist C
4. Checklist D
5. Checklist E
6. Checklist F

This order protects thesis semantics first, then readability, then tooling cleanup.

## Follow-up: Checklist A Decision Proposal

This section gives one concrete recommendation for the current thesis experiment families, based on the config tree, the owning model/config surfaces, and the regression coverage that exists now.

### Proposed family labels

#### 1. `configs/experiment/thesis/exp4/`

Recommended label:

- `active_thesis_method`

Reason:

- It is the cleanest thesis-owned three-stage family.
- It already points at the dedicated three-stage model/config surface:
  - `configs/model/thesis_multitask_three_stage_window20.yaml`
  - `configs/data/smd_rtx3090_machine_3_4_20_stride1.yaml`
- It encodes the intended three-stage budget and stage semantics directly.
- It is covered by targeted tests for config loading, orchestration smoke, server preflight, and launcher behavior.

Evidence:

- `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-three-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml:1-47`
- `configs/model/thesis_multitask_three_stage_window20.yaml:1-56`
- `tests/test_smd_machine_3_4_three_stage_config_loading.py:24-58`
- `tests/test_three_stage_orchestration_smoke.py`
- `tests/test_three_stage_server_preflight.py`

#### 2. `configs/experiment/benchmark/thesis/`

Recommended label:

- `active_benchmark_family`

Reason:

- This family is not just a side experiment. It is the active benchmarkized thesis surface for the three-stage method.
- It uses the comparative-style three-stage model config and benchmark-specific data/task configs.
- It is heavily referenced by config-loading tests and benchmark-audit work.

Evidence:

- benchmark configs use `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`
- benchmark configs lock `expected_total_training_epochs: 100`
- `tests/test_config_loading.py:1855-1865`

Interpretation:

- `exp4/` should be treated as the active thesis-owned development and method-definition family.
- `benchmark/thesis/` should be treated as the active benchmark evaluation family derived from that method.

#### 3. `configs/experiment/comparative/thesis/`

Recommended label:

- `active_comparative_family`

Reason:

- This family is clearly current and operational.
- It uses stride-1 `window20` entity-specific configs, three-stage contracts, and dedicated comparative task/model presets.
- It is validated by focused comparative config tests.

Evidence:

- `configs/experiment/comparative/thesis/*.yaml`
- `configs/model/thesis_multitask_three_stage_comparative_smd.yaml:1-56`
- `tests/test_comparative_config_loading.py:1-166`

Interpretation:

- This should remain active, but it should be documented as a comparative evaluation family, not as the core thesis-method-definition family.

#### 4. `configs/experiment/thesis/exp2/`

Recommended label:

- `historical_exp2_family`

Reason:

- It is still meaningful and still tested.
- It preserves the older two-view + CKA-gated thesis runtime family on `machine-2-1`.
- But it does not match the newer three-stage thesis contract.
- It should not keep the generic comment status `active` if the repo now wants one single active thesis method.

Evidence:

- `configs/experiment/thesis/exp2/...:1-49`
- `configs/model/thesis_multitask_redlamp_multiclass.yaml:1-49`
- `tests/test_one_multitask_train_step.py:66-153`
- `tests/test_multitask_memory_updates.py:111`

Interpretation:

- Keep it runnable.
- Keep tests that intentionally preserve old semantics.
- But rename its role in comments/docs to `historical_exp2_family` or equivalent wording.

#### 5. `configs/experiment/thesis/exp3/`

Recommended label:

- `historical_exp2_variant_family`

Reason:

- Despite the folder name `exp3`, most configs in this folder still point to the same old thesis model surface:
  - `configs/model/thesis_multitask_redlamp_multiclass.yaml`
  - `configs/data/smd_rtx3090_machine_2_1_20.yaml`
- They look more like tuned variants of the older window20 multitask family than a distinct new thesis-method contract.
- The exception is that some files here are alignment or diagnostic variants, not a new core method.

Evidence:

- `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml:1-31`
- `configs/experiment/thesis/exp3/*redlamp-aligned*.yaml`
- `tests/test_redlamp_aligned_configs.py:40-116`

Interpretation:

- This folder should not stay globally labeled `active` unless the repo explicitly wants to keep the older family as a co-equal active thesis line.
- Practically, it should be split mentally into:
  - historical mainline variants of the old family
  - aligned/diagnostic variants

#### 6. `configs/experiment/thesis/exp1/`

Recommended label:

- `historical_ablation_seed_family`

Reason:

- It is an older recon-focused experiment family tied to the same old window20 `machine-2-1` thesis surface.
- It is useful as history and maybe as an ablation reference, but not as the current thesis-method contract.

Evidence:

- `configs/experiment/thesis/exp1/...:1-31`
- same old `machine-2-1` data/model/task surface

### Recommended final policy

If the repository wants one single active thesis method, the cleanest policy is:

- `exp4/` = active thesis method definition
- `benchmark/thesis/` = active benchmark thesis family
- `comparative/thesis/` = active comparative thesis family
- `exp2/` = historical but runnable
- `exp3/` = historical variants and aligned diagnostics
- `exp1/` = historical ablation/reference family

### Exact metadata cleanup to do later

When editing config headers later, the lowest-friction cleanup would be:

1. Change header comments in `exp1/`, `exp2/`, and most of `exp3/` from `# status: active` to one of:
   - `# status: historical`
   - `# status: legacy`
   - `# status: ablation`
2. Keep `exp4/`, `benchmark/thesis/`, and `comparative/thesis/` as the only thesis families still labeled `active`.
3. In docs, refer to:
   - `Exp2/Exp3` as the older prototype-fusion family
   - `Exp4 + benchmark/comparative` as the current three-stage family

### Verification performed for this decision

Ran:

```bash
pytest -q tests/test_smd_machine_3_4_three_stage_config_loading.py tests/test_comparative_config_loading.py tests/test_three_stage_orchestration_smoke.py tests/test_one_multitask_train_step.py tests/test_multitask_memory_updates.py tests/test_redlamp_aligned_configs.py
```

Result:

- `32 passed in 3.05s`

This confirms that both surfaces are still live:

- the newer three-stage family
- the older `Exp2`-style thesis runtime semantics

That is exactly why a naming/status cleanup is now necessary.
