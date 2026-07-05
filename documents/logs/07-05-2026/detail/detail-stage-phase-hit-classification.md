# Stage/Phase Semantic Rename Detailed Plan

**Objective:** Eliminate semantic drift between `phase`, `stage`, and `stage_name` so that each term denotes exactly one runtime concept, while preserving the active two-stage offline pre-training contract and the historical three-stage compatibility surface.

**Scope:** This plan only renames, re-comments, and fences terminology. It does not change the active two-stage behavior, does not alter model math, and does not change public schema keys unless a later migration task explicitly isolates that change.

**Core Thesis Constraint:** `offline pre-training` is the large phase. Stage A and Stage B are the stages inside that phase. Legacy three-stage codepaths remain historical or compatibility-only. Runtime `stage_name` in the trainer and model step methods continues to mean `train`, `val`, `val_synth`, or `test`.

---

## Context From Research

The research note at [`documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/ĐH%20KHOA%20HO%CC%A1C%20TỰ%20NHIÊN/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md) separates grep hits into three buckets:

1. Active two-stage offline pre-training.
2. Legacy three-stage offline pre-training.
3. Runtime `stage_name` naming for execution splits.

This detail plan converts that classification into a sequence of small implementation passes. Each pass is designed to preserve the minimal vertical slice principle: change the smallest readable surface first, verify it, and only then continue to the next bucket.

---

## Design Contracts to Preserve

### Batch contract
The batch structure passed into the trainers must remain unchanged. Data loaders, collators, and trainer loops should continue to receive the same tensor shapes and keys. Any rename in this plan must not require changing batch parsing logic.

### Encoder contract
The active encoder contract must remain stable. Encoders continue to expose the same input-output tensor expectations, and any adapter or helper introduced by the codebase must preserve composition over inheritance.

### Model output contract
The multitask model output format must remain stable. In particular, the semantics of reconstruction, classification, prototype usage, and memory initialization must remain readable and consistent with the current active two-stage design.

### Compatibility contract
Legacy three-stage support remains available. Compatibility aliases may remain in the codebase, but the code must clearly label them as historical rather than active.

### Runtime naming contract
Ordinary runtime `stage_name` values such as `train`, `val`, `val_synth`, and `test` must not be renamed for symmetry. Their meaning is already single-purpose and should remain so.

---

## Repository Interfaces and Pattern Boundaries

### Dataset interface and registry boundary
The dataset layer must remain registry-driven or factory-driven, so that active two-stage configs and legacy three-stage configs resolve through stable dataset names. This plan does not change dataset registration; it only ensures that terminology in docs and comments does not obscure the dataset contract. Any loader or factory helper referenced during the rename passes must continue to return the same batch structure and tensor keys.

### Encoder interface and adapter boundary
Encoders must remain accessible through a clear adapter boundary. The active encoder contract is still the same: the model expects the same tensor shapes and the same encoder output structure. This plan does not add a new encoder architecture; it preserves composition over inheritance and keeps adapter names readable so that future encoder swaps remain explicit.

### Task interface and strategy boundary
Task behavior must remain strategy-based. The repository already treats training objectives, evaluation behavior, and phase-specific behavior as distinct strategies or configurable branches. This plan only renames terminology around those branches when it improves single-meaning readability. It does not collapse task strategies into a single codepath.

### Model and engine interface boundary
The training engine, model forward path, and model output contract must remain stable. Trainer and evaluator code should continue to consume the same batch structure and produce the same outputs, while the naming around phase and stage becomes less ambiguous. If a helper is renamed, the helper should continue to serve one responsibility only and should remain short enough to satisfy the repository readability rules.

### Design pattern application
- **Composition over inheritance:** keep the refactor focused on localized helpers, comments, and wrappers rather than adding new inheritance layers.
- **Adapter pattern for encoders:** preserve the current encoder adapter boundary and only clarify terminology around it.
- **Strategy pattern for tasks:** keep active two-stage, legacy three-stage, and runtime step behavior as distinct strategies or branches.
- **Registry or factory for datasets and models:** preserve existing resolution points and avoid renaming those surfaces in a way that would hide the registry contract.

---

## Detailed Implementation Phases

### Phase 1: Terminology lock in the SSOT design note

**Phase summary:** Establish one canonical glossary before touching code. This phase makes the design document itself enforce the naming rule so that later code changes do not reintroduce ambiguity.

**Files to modify:**
- [`documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐA%CC%A3I%20HO%CC%A1C%20QUO%CC%82%CC%81C%20GIA%20TPHCM/ĐH%20KHOA%20HO%CC%A1C%20TỰ%20NHIÊN/Khoa%CC%81%20luận%20tốt%20nghiệp/bachelor-thesis-2026/documents/design/offline_pretraining_two_stage_kmeans_memory_design.md)
- [`documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/ĐẠI%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md) only if one clarification sentence is needed in the classification table

**Explicit edit content:**
- Add a compact `Terminology` section near the top of the active SSOT doc.
- State that `offline pre-training` is the phase and Stage A/B are the stages inside it.
- State that the older three-stage material is historical or compatibility-only.
- State that runtime `stage_name` means execution-step naming and must not be conflated with offline pre-training taxonomy.
- Avoid any wording that implies behavior change, migration completion, or public key renaming in this phase.

**Design pattern principles preserved:**
- Single responsibility: the SSOT note holds one glossary and one contract boundary.
- Stable interface: no schema or code interface is changed here.

**Risk mitigation:**
- Prevents later rename passes from reintroducing mixed terminology.
- Keeps active and legacy concepts separated before code changes start.

**Test and validation plan:**
- Manual review of the updated terminology section.
- Confirm that the note still reflects the current active contract without claiming a migration that has not happened.

**Acceptance criteria:**
- The SSOT doc contains a clear glossary for `phase`, `stage`, and `stage_name`.
- A reader can distinguish active two-stage, legacy three-stage, and runtime naming from the document alone.
- No runtime behavior is changed.

---

### Phase 2: Internal stage-first naming in the active two-stage runner

**Phase summary:** Make the active two-stage runner read in the same way the contract behaves. The runner should describe Stage A and Stage B directly in internal variables and comments, while leaving public manifest keys intact for compatibility.

**Files to modify:**
- [`scripts/run_two_stage_offline_pretraining.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_two_stage_offline_pretraining.py)

**Explicit edit content:**
- Rename local variables that encode stage semantics:
  - `phase_name` to `stage_name` in local control flow.
  - `phase_record` to `stage_record`.
  - `training_stages` to `stage_records`.
- Rewrite comments so they say “Stage A/B inside offline pre-training phase” instead of using `phase` as the local unit of work.
- Keep public keys such as `phase_name`, `two_stage_phase`, and `training_phase` unchanged in this phase because downstream consumers may already depend on them.
- Ensure the `build_two_stage_training_plan()` helper remains short, linear, and readable.

**Interface and contract definitions:**
- The manifest contract remains stable.
- The runner continues to emit the same record keys and epoch spans.
- The batch contract is unaffected because only vocabulary and comments change.

**Design pattern principles preserved:**
- Composition over inheritance is preserved because the runner continues to orchestrate existing configuration and helpers instead of introducing a new wrapper hierarchy.
- Single responsibility is improved because internal naming now matches the runner’s actual job.

**Risk mitigation:**
- Avoid changing manifest keys in the same pass as internal renames.
- Avoid renaming `phase_name` in the serialized output until a dedicated migration is planned.

**Test and validation plan:**
- Run `pytest -q tests/test_offline_pretraining_two_stage_runner.py`.
- Confirm the produced manifest keys remain identical to the pre-change state.
- Confirm that comments and log text now reflect stage-first internal meaning.

**Acceptance criteria:**
- The file reads as stage-first internally.
- Output schema remains unchanged.
- The runner still passes the existing two-stage runner tests.

---

### Phase 3: Clarify active two-stage semantics in model helpers

**Phase summary:** Make the model helper files explain the active two-stage contract directly. This pass improves readability of Stage A/B lifecycle logic without changing model outputs, serialization, or training behavior.

**Files to modify:**
- [`src/models/thesis_multitask_state_mixin.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_state_mixin.py)
- [`src/models/thesis_multitask_setup_mixin.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_setup_mixin.py)
- [`src/models/thesis_multitask_components.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_components.py) only if a helper or dataclass name can be clarified without changing the public interface

**Explicit edit content:**
- Replace phase-oriented helper comments with stage-oriented comments when the helper specifically concerns Stage A or Stage B.
- Make the lifecycle state helper state plainly that it is stage-facing for active two-stage runs.
- Keep the Stage B freeze-point comment explicit so readers understand why encoder parameters are frozen at that point.
- Preserve any public dataclass field names that are already part of the compatibility contract.
- Do not split the model across files in this pass; keep the model owner readable as a self-contained file boundary, consistent with repository guidance.

**Interface and contract definitions:**
- The active two-stage semantic labels continue to resolve exactly as before.
- The model output contract remains unchanged.
- Serialization and lifecycle-state shapes remain stable.

**Design pattern principles preserved:**
- Composition is preserved by keeping helper functions narrow and reusable.
- Separation of concerns is improved because state, setup, and semantic label logic are easier to scan independently.
- Single-meaning is reinforced because the helper names now match the semantic role they serve.

**Risk mitigation:**
- Avoid renaming helper functions that would require wide import updates unless the rename clearly improves meaning.
- Avoid altering tensor math, checkpoint state, or label serialization in the same pass.

**Test and validation plan:**
- Run `pytest -q tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py`.
- Confirm Stage A and Stage B labels still resolve exactly as before.
- Confirm checkpoint and lifecycle-state behavior does not drift.

**Acceptance criteria:**
- The active two-stage helper files read stage-first when discussing Stage A/B.
- No public interface changes are introduced.
- Existing model tests continue to pass.

---

### Phase 4: Fence off legacy three-stage compatibility

**Phase summary:** Make the historical three-stage path visibly historical. This pass does not remove compatibility; it clarifies that the legacy path is not the same contract as the active two-stage rerun.

**Files to modify:**
- [`scripts/run_three_stage_offline_pretraining.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/run_three_stage_offline_pretraining.py)
- [`src/core/config.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py)
- [`src/models/thesis_multitask_components.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_components.py)
- [`tests/test_three_stage_phase_runtime.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_three_stage_phase_runtime.py)
- [`tests/test_three_stage_orchestration_smoke.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_three_stage_orchestration_smoke.py) if the test names still suggest active semantics

**Explicit edit content:**
- Add “legacy three-stage” wording in comments and docstrings wherever the code currently reads as if three-stage were active.
- Keep compatibility aliases in `src/core/config.py` where old configs still require them.
- Make the distinction between historical three-stage and active two-stage explicit in the runtime config and test names.
- Preserve current legacy loading behavior so that historical artifacts still load.

**Interface and contract definitions:**
- Legacy config loading remains stable.
- The compatibility surface stays readable as compatibility, not as the active path.
- Public field aliases remain available only where necessary.

**Design pattern principles preserved:**
- Registry/factory behavior already present in config resolution remains unchanged.
- The compatibility boundary is explicit, which improves maintainability without changing inheritance structure.

**Risk mitigation:**
- Keep alias translation behavior intact while editing terminology.
- Do not collapse three-stage and two-stage into one naming scheme.

**Test and validation plan:**
- Run `pytest -q tests/test_three_stage_phase_runtime.py tests/test_three_stage_orchestration_smoke.py tests/test_three_stage_server_preflight.py`.
- Confirm legacy configs still load.
- Confirm new comments clearly label the path as legacy.

**Acceptance criteria:**
- The legacy three-stage path is unmistakably marked as historical or compatibility-only.
- Tests continue to validate the old path.
- No active two-stage code becomes dependent on legacy wording.

---

### Phase 5: Preserve runtime `stage_name` semantics

**Phase summary:** Confirm that ordinary runtime `stage_name` usage is already single-meaning and should remain unchanged. This pass protects the codebase from unnecessary symmetry-driven renames.

**Files to review without changing behavior unless a true semantic mismatch is found:**
- [`src/engine/trainer.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/trainer.py)
- [`src/models/reconstruction_mlp_ae.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/reconstruction_mlp_ae.py)
- [`src/models/thesis_multitask_loss_mixin.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask_loss_mixin.py)
- [`scripts/visualize_classification_diagnostics.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/scripts/visualize_classification_diagnostics.py)
- [`src/models/online_adaptation.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/online_adaptation.py)

**Explicit edit content:**
- Review each `stage_name` occurrence and confirm that it refers to a runtime split, not offline pre-training taxonomy.
- Leave these names unchanged when they already mean `train`, `val`, `val_synth`, or `test`.
- Only add or adjust comments if the surrounding text incorrectly suggests a phase/stage taxonomy meaning.

**Interface and contract definitions:**
- Runtime step semantics remain stable.
- Trainer and model step behavior remains unchanged.
- Visualization and online-adaptation logging continue to use their existing step names.

**Design pattern principles preserved:**
- Stable interfaces are preserved by not forcing unnecessary rename symmetry.
- Single-meaning is preserved because this runtime bucket is intentionally separate from offline pre-training terminology.

**Risk mitigation:**
- Prevent “cleanup” renames that would actually reduce comprehension.
- Keep runtime step naming aligned with existing logs and metrics.

**Test and validation plan:**
- Grep for `stage_name` after all rename passes.
- Confirm the remaining hits belong to runtime execution splits.
- If a file still mixes runtime `stage_name` with offline phase taxonomy, annotate it explicitly instead of renaming blindly.

**Acceptance criteria:**
- Runtime `stage_name` continues to mean runtime execution splitting only.
- No unnecessary renames are introduced for symmetry.
- Grep results remain readable and classifiable.

---

### Phase 6: Align tests and docs with the same semantic buckets

**Phase summary:** Make the test titles, helper names, and research notes teach the same terminology as the code. This is the final readability pass that makes the rename effort visible to readers outside the implementation files.

**Files to modify:**
- [`tests/test_offline_pretraining_two_stage_runner.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_offline_pretraining_two_stage_runner.py)
- [`tests/test_offline_pretraining_two_stage_config_loading.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_offline_pretraining_two_stage_config_loading.py)
- [`tests/test_three_stage_phase_runtime.py`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_three_stage_phase_runtime.py)
- [`documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md`](/Users/conquerormikrokosmos/Downloads/LAPTOP%20MAC/MYUNIVERSITY/%C4%90A%CC%A3I%20HỌC%20QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/documents/logs/07-05-2026/research/research-stage-phase-hit-classification.md) only if the classification table needs a final wording pass

**Explicit edit content:**
- Rename test titles so active two-stage tests say `stage` when they mean Stage A/B.
- Rename legacy three-stage test helpers or titles so they say `legacy` or `compatibility` where appropriate.
- Keep runtime `stage_name` test names intact when they are already correct.
- If the research note still contains wording that could be read as a mixed contract, clarify it there rather than in code.

**Interface and contract definitions:**
- The test suite remains a contract for behavior and terminology.
- Docs remain the SSOT for semantic interpretation.
- The final grep classification should still divide hits into the three research buckets.

**Design pattern principles preserved:**
- The repository remains easy to comprehend because tests and docs use the same vocabulary as the code.
- Stable interfaces are not altered by terminology cleanup alone.

**Risk mitigation:**
- Avoid renaming tests in a way that obscures what behavior they actually verify.
- Avoid editing historical docs so aggressively that they lose traceability.

**Test and validation plan:**
- Run `pytest -q tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py tests/test_three_stage_phase_runtime.py tests/test_config_loading.py`.
- Run a final grep classification pass:
  - `rg -n -w "phase_name|two_stage_phase|training_phase|stage_name|phase|stage" scripts src tests documents/design`
- Review the hits in the three buckets:
  - active two-stage
  - legacy three-stage
  - runtime `stage_name`

**Acceptance criteria:**
- Test names teach the correct meaning.
- Historical terminology is explicitly marked as legacy.
- The grep hits are classifiable without ambiguity.

---

## Global Risk and Mitigation

- **Risk: continuous and discrete prototype branches are described as if they were interchangeable.**
  - **Mitigation:** preserve distinct terminology for the prototype branches and keep any prototype-related comments tied to their actual branch semantics. Do not rename them into generic “phase” language.

- **Risk: fusion can be described in a way that hides branch imbalance or collapse.**
  - **Mitigation:** keep fusion terminology explicit in comments and test names so that branch-specific meaning remains visible. If a future implementation pass touches fusion, require a separate validation for branch contribution balance.

- **Risk: adaptation contamination can be confused with ordinary runtime stage naming.**
  - **Mitigation:** keep online adaptation comments and runtime `stage_name` terminology separate. Mark contamination guards as adaptation-specific, not generic stage logic.

- **Risk: projector drift and poor initialization become harder to trace when terminology is mixed.**
  - **Mitigation:** keep projector-related comments stage-specific only when they refer to the online adaptation contract, and preserve any residual or warm-start terminology in place.

- **Risk: evaluation metric inflation is masked by naming drift.**
  - **Mitigation:** keep metric names and test titles aligned with the exact metric definition, and do not rename them for aesthetic symmetry.

- **Risk: public keys such as `phase_name` or `two_stage_phase` get renamed too early.**
  - **Mitigation:** keep public schema stable in the first pass and isolate any migration into a separate task.

- **Risk: active two-stage and legacy three-stage are conflated.**
  - **Mitigation:** keep a dedicated terminology section in the SSOT design doc and use `legacy three-stage` wording consistently for historical paths.

- **Risk: runtime `stage_name` is renamed unnecessarily for symmetry.**
  - **Mitigation:** treat runtime `stage_name` as a separate semantic bucket and leave it untouched unless a true ambiguity is found.

- **Risk: comments change but the reader still cannot infer the contract.**
  - **Mitigation:** align code comments, test titles, and research notes with the same terminology.

- **Risk: rename passes accidentally disturb behavior.**
  - **Mitigation:** run the smallest relevant pytest subset after each phase and preserve the batch, encoder, and model output contracts.

---

## Validation Matrix

The detailed plan must be validated with a small, explicit set of checks that match the repository preferences:

1. **Config loading validation**
   - Load active two-stage YAML configs.
   - Load legacy three-stage YAML configs.
   - Confirm that terminology changes did not alter config parsing or alias resolution.

2. **Batch shape validation**
   - Verify that trainer batches still present the expected tensor keys and shapes.
   - Keep loader batch size checks explicit so that rename work does not hide data-contract regressions.

3. **Single-step integration validation**
   - Run one forward pass and one backward pass on one batch.
   - Use the active two-stage path as the primary smoke target.

4. **Checkpoint validation**
   - Confirm that checkpoint save and load still round-trip correctly.
   - This remains mandatory because checkpoint names and lifecycle state are part of the contract.

5. **Synthetic anomaly injection validation**
   - Preserve the test path that injects anomalies into one batch and visualizes the result.
   - Do not change injection terminology while the rename pass is still active unless the rename clarifies the contract.

6. **Grep classification validation**
   - Re-run a root-level grep for `phase`, `stage`, `stage_name`, `phase_name`, `two_stage_phase`, and `training_phase`.
   - Classify the hits again into active two-stage, legacy three-stage, and runtime `stage_name`.

---

## Global Acceptance Criteria

- Active two-stage files read stage-first internally where Stage A/B is the real meaning.
- Legacy three-stage codepaths are clearly labeled as historical or compatibility-only.
- Runtime `stage_name` remains intact where it means ordinary execution splitting.
- Public schema keys remain stable unless a separate migration task is approved.
- The repository can be read with one meaning per term.
- The remaining grep hits can be sorted cleanly into the three buckets from the research note.
- Any newly extracted helper remains short, readable, and within the repository limits for methods and file size.
