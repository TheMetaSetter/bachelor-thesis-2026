---
date: 2026-07-05 14:04:14 +07
researcher: TheMetaSetter
git_commit: 757d9480d72ee0a1925b0b7194b05b599b3b2f0f
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed plan to eliminate stage/phase semantic drift"
tags:
  - detail
  - stage
  - phase
  - semantics
  - thesis-multitask
status: draft
last_updated: 2026-07-05
last_updated_by: TheMetaSetter
---

# Detail: Eliminate stage/phase semantic drift

## Overview

The repository already executes the active two-stage offline-pretraining rerun correctly, but the reader-facing vocabulary is still not single-meaning. The implementation work below keeps behavior stable and narrows the public meaning so that `offline pre-training` is read as the phase and Stage A / Stage B are read as the only sub-stages in that phase.

## Phase 1: Lock the active meaning and isolate legacy terminology

The first pass should establish a clear naming boundary before any broader refactor. This phase exists to prevent future edits from reintroducing mixed `phase` and `stage` usage in the active two-stage path.

### File-level edits

- `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`
  - Add a short terminology note near the top that states:
    - `offline pre-training` is the large phase.
    - Stage A and Stage B are the only stages in that phase for the active rerun.
    - Historical three-stage names remain historical only.
  - Keep all behavioral contracts unchanged.
- `documents/logs/07-05-2026/research/research-stage-phase-semantic-drift.md`
  - Leave the research evidence intact.
  - Add a brief follow-up note only if later edits need a pointer for the chosen terminology boundary.

### Interface and contract definition

- The active semantic contract is:
  - one phase: offline pre-training,
  - two stages: Stage A and Stage B.
- The compatibility contract is:
  - legacy three-stage wording may still exist in older files and compatibility shims,
  - legacy wording must not be presented as the default active meaning.

### Design pattern application

- Composition over inheritance remains unchanged.
- No new abstraction layer should be introduced in this phase.
- The only design action is to narrow terminology.

### Risk mitigation

- Risk: old notes and logs still dominate grep output.
  - Mitigation: explicitly label historical files as historical and keep the active SSOT file authoritative.

### Test plan and validation

- Run a repository grep focused on active files and ensure the two-stage path reads stage-first.
- Confirm that no active design file implies that one `phase` contains multiple unrelated meanings.

### Acceptance criteria

- A new reader can open the active SSOT and understand the phase/stage boundary without consulting historical notes.
- No active contract document uses `phase` and `stage` interchangeably for the same runtime concept.

## Phase 2: Clean the two-stage runner without changing its output contract

The second pass should make the orchestration file read as stage-first internally while keeping the manifest schema and execution behavior stable.

### File-level edits

- `scripts/run_two_stage_offline_pretraining.py`
  - Keep the execution logic intact.
  - Keep the manifest keys `phase_name`, `two_stage_phase`, and `training_stages` stable for compatibility.
  - Change internal variable names and comments so they read stage-first:
    - `phase_name` -> `stage_name` in local variables,
    - `phase_record` -> `stage_record`,
    - `phase_index` -> `stage_index`,
    - `phase_epochs` -> `stage_epochs`.
  - Rewrite the TODO comment so it states the chosen meaning precisely:
    - `offline pre-training` is the phase.
    - Stage A and Stage B are the stages inside it.
  - Keep output directory behavior unchanged.

### Interface and contract definition

- Manifest contract stays unchanged:
  - `training_stages` remains a list of stage records,
  - each record still serializes the existing compatibility keys.
- Internal readability contract changes:
  - local variables and comments must use stage-first wording.

### Design pattern application

- Preserve stable interfaces.
- Use composition by keeping the runner as orchestration only.
- Do not introduce a new orchestration class hierarchy.

### Risk mitigation

- Risk: changing serialized keys now could break downstream consumers.
  - Mitigation: keep schema stable in this pass.
- Risk: the runner remains easy to misread if only comments change.
  - Mitigation: rename the internal variables as well as the comments.

### Test plan and validation

- Re-run:
  - `tests/test_offline_pretraining_two_stage_runner.py`
  - `tests/test_offline_pretraining_two_stage_config_loading.py`
- Confirm that the generated manifest and dry-run behavior remain unchanged.

### Acceptance criteria

- The runner still passes the same tests.
- The runner source reads stage-first internally.
- The manifest schema remains backward compatible.

## Phase 3: Make the model runtime labels single-meaning

The third pass should clean the thesis model runtime metadata so the active two-stage meaning is obvious, while historical Stage 3 compatibility remains explicitly marked as legacy.

### File-level edits

- `src/models/thesis_multitask_state_mixin.py`
  - Rewrite comments and label strings where they describe the active two-stage path.
  - Ensure `semantic_stage_label` reads clearly as a stage label, not as a general phase label.
  - Keep the behavior of `_should_bypass_memory_for_stage`, `_should_update_memory`, `_memory_initialization_substep_active`, and `_fusion_warmup_substep_active` unchanged.
  - Keep historical Stage 3 handling, but label it as compatibility-only in comments.
- `src/models/thesis_multitask_components.py`
  - Clarify the docstring around runtime configuration.
  - Keep accepted runtime values unchanged.
  - Mark legacy Stage 3 naming as historical.
- `src/models/thesis_multitask_setup_mixin.py`
  - Review comments around `_phase_uses_prototype_path`, `_phase_uses_contrastive_objective`, `_phase_freezes_encoder`, and `_configure_trainable_parameters_for_phase`.
  - Add or rewrite comments so they do not suggest that the active two-stage path is a three-stage contract.

### Interface and contract definition

- The public runtime state contract remains:
  - `semantic_stage_label`,
  - `memory_initialization_substep`,
  - `fusion_warmup_substep`,
  - `trainable_module_names`,
  - the memory lifecycle fields already emitted by `get_memory_lifecycle_state()`.
- The meaning of these fields becomes stage-first for the active two-stage path.

### Design pattern application

- Continue using composition over inheritance through mixins.
- Keep the model file self-contained.
- Avoid introducing new task abstractions in this pass.

### Risk mitigation

- Risk: the model runtime has historical names that could be mistaken for active semantics.
  - Mitigation: explicitly label the historical branches as compatibility-only.
- Risk: changing comments without changing labels may be too weak.
  - Mitigation: adjust the labels shown to the reader wherever the active meaning is exposed.

### Test plan and validation

- Re-run the focused two-stage runner tests.
- Re-run any lifecycle or trainable-surface tests already covering:
  - phase switching,
  - memory lifecycle state,
  - encoder freezing in Stage B.

### Acceptance criteria

- A new reader can inspect the runtime state and understand Stage A / Stage B without decoding the historical Stage 3 language first.
- Runtime behavior remains unchanged.

## Phase 4: Isolate legacy three-stage compatibility in config validation

The fourth pass should keep the current compatibility support but make the active two-stage contract easier to distinguish from the historical three-stage contract.

### File-level edits

- `src/core/config.py`
  - Keep the two-stage validation logic intact.
  - Keep legacy Stage 3 alias normalization intact if it is still required.
  - Rewrite comments and error messages so the active two-stage contract is clearly separate from the historical three-stage compatibility surface.
- `src/core/config_experiment_validation.py`
  - Review any wording that still makes the historical three-stage path sound active by default.
- `src/core/config_model_validation.py`
  - Review validation messages involving `training_phase` so they do not imply that `phase` and `stage` are interchangeable for the active two-stage rerun.

### Interface and contract definition

- Active two-stage config contract remains:
  - `expected_total_training_epochs`,
  - `stage_a_multitask_epochs`,
  - `stage_b_fusion_finetuning_epochs`,
  - `discrete_memory_label_source`,
  - `freeze_encoder_and_memories_in_stage_b`.
- Historical three-stage compatibility remains separate and should not be presented as the active contract.

### Design pattern application

- Keep the registry/factory construction unchanged.
- Keep config-driven composition unchanged.
- Do not add extra config layers to rename concepts.

### Risk mitigation

- Risk: removing too much legacy compatibility could break older experiment YAMLs.
  - Mitigation: keep compatibility input handling until a separate migration is explicitly approved.
- Risk: keeping too much legacy wording can continue to confuse readers.
  - Mitigation: restrict legacy wording to narrow compatibility sections and test names.

### Test plan and validation

- Re-run:
  - two-stage config-loading tests,
  - three-stage compatibility config-loading tests,
  - general config validation tests that exercise `training_phase`.

### Acceptance criteria

- Supported legacy inputs still load.
- Conflicting legacy inputs still fail.
- Active two-stage validation reads as the default contract.

## Phase 5: Update docs, tests, and verification so they teach one meaning only

The final pass should align the visible documentation and the focused tests with the cleaned terminology.

### File-level edits

- `documents/logs/07-05-2026/plan/plan-stage-phase-semantic-drift.md`
  - Keep this plan as the source for the implementation ordering.
- `documents/logs/07-05-2026/structure/structure-stage-phase-semantic-drift.md`
  - Keep the structure as the intermediate outline.
- `tests/test_offline_pretraining_two_stage_runner.py`
  - Update test names or assertions only if the active vocabulary changes in the code.
- `tests/test_offline_pretraining_two_stage_config_loading.py`
  - Keep it as the contract check for the active two-stage config.
- `tests/test_three_stage_phase_runtime.py`
  - Keep it as explicit legacy compatibility coverage.
- Any active docs that a new reader will encounter first:
  - align wording to the current two-stage meaning,
  - label historical three-stage notes as historical when they are referenced.

### Interface and contract definition

- Tests should distinguish:
  - active two-stage meaning,
  - legacy three-stage compatibility.
- The repository should not rely on the reader inferring meaning from context.

### Design pattern application

- Preserve stable interfaces and explicit configuration.
- Preserve the minimal vertical slice principle by keeping verification focused.
- Do not add broad framework abstractions just to host renamed terms.

### Risk mitigation

- Risk: docs can drift away from the cleaned code again.
  - Mitigation: keep active docs close to the active SSOT and keep historical logs labeled as historical.
- Risk: tests can accidentally preserve the old mental model.
  - Mitigation: rename tests so they describe the active contract rather than the legacy wording.

### Test plan and validation

- Run the focused two-stage pytest bundle again.
- Run the legacy three-stage compatibility tests.
- Run a repository grep to ensure the active surfaces no longer force the reader to interpret one term as two meanings.

### Acceptance criteria

- The active docs and active tests teach the same contract as the code.
- Historical notes remain available but are clearly historical.
- A new reader can identify the active contract without decoding legacy terminology first.

## Cross-Phase Verification Checklist

- The active two-stage rerun still behaves the same.
- No public schema change is introduced unless a later migration explicitly approves it.
- Historical three-stage support remains isolated.
- Grep from the repository root shows the active code path is stage-first, while legacy material is explicitly historical.

## Final Acceptance Criteria

- The active contract is single-meaning.
- The repository remains minimalistic and readable.
- The codebase owner can point a reader to one active interpretation of `phase` and `stage` without ambiguity.
- All focused tests relevant to the touched surfaces continue to pass.
