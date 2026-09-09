---
date: 2026-09-09 21:35:18 +07:00
topic: "Apply terminology ontology naming conventions to smoke W&B run names and identity fields"
status: implemented
revision: ed10fbb4db70f9ed17f98330c18f6e456ba5f2f5
source_structure: documents/logs/09-09-2026/structure/structure-terminology-ontology-naming-code-modifications.md
related_documents:
  - documents/logs/09-09-2026/research/research-terminology-ontology-naming-code-modifications.md
  - documents/logs/09-09-2026/plan/plan-terminology-ontology-naming-code-modifications.md
  - documents/notes/terminology_ontology_naming_conventions.md
  - documents/spec/offline_pretraining_terminology_ontology.md
  - documents/spec/online_tta_terminology_ontology.md
---

# Detailed Implementation: Apply terminology ontology naming conventions to smoke W&B run names and identity fields

## Summary

The implementation will change only the smoke W&B display-name surface and the explicit identity fields needed to construct it safely.

It will not rename local files, checkpoints, output directories, metrics, threshold fields, or intentional online compatibility fields.

The exact selected format is `smk-<phase>-<method_or_variant>-<entity>-s<seed>`.

## Execution record

The implementation is complete for the source helpers, generators, runtime callers, derived smoke YAML, and focused tests.

All tracked smoke configurations retain `use_wandb: true`, `wandb_mode: online`, and a validated `smk-` display name.

The focused naming, logger, and smoke-inventory tests pass with `19 passed`.

The source tree compiles successfully and `git diff --check` reports no whitespace errors.

The full repository suite still has seven unrelated baseline failures involving matrix counts, artifact export setup, model snapshots, metric snapshots, checkpoint loading, and multitask model structure.

An actual online W&B smoke training run was not started in this execution because it requires the project runtime, checkpoint source, GPU, and W&B credentials.

## Source structure

The approved structure contains four phases.

Phase 1 defines canonical identity and the display-name contract.

Phase 2 routes generators and runtime callers through the contract.

Phase 3 regenerates derived smoke configurations and preserves provenance.

Phase 4 verifies compatibility and closes the change.

## Current state

`src/core/artifact_naming.py:204-216` builds a generic W&B name from artifact-style identity.

`src/engine/logger.py:104-116` passes the final name to `wandb.init` and falls back to `experiment_name`.

The main generators use legacy or long names at `scripts/benchmarks/generate_smd_benchmark_configs.py:98`, `scripts/benchmarks/generate_benchmark_smoke_configs.py:69-70`, `scripts/benchmarks/generate_online_benchmark_configs.py:224`, `scripts/benchmarks/generate_offline_benchmark_configs.py:117`, and `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:226`.

Runtime callers overwrite names at `scripts/cli/train.py:268`, `scripts/cli/evaluate.py:310-312`, `scripts/experiments/run_online_adaptation.py:144-146`, `scripts/benchmarks/run_thesis_online_benchmark.py:315-319`, and `scripts/experiments/run_two_stage_offline_pretraining.py:159-161`.

## Desired end state

Every smoke YAML has a valid short W&B display name.

The display name uses `off` or `on` for the phase token.

The display identity uses canonical variant values or an explicitly selected method display token.

The entity token uses the existing display form such as `e1_6`, `e3_9`, or the project-selected equivalent recorded by the generator.

The seed token has the exact `s<seed>` form.

Runtime callers preserve a valid configured name instead of replacing it with the generic artifact-style name.

## Scope

### In scope

- `src/core/config.py` optional canonical identity fields.
- `src/core/artifact_naming.py` smoke display-name helper.
- Active smoke config generators.
- Runtime name-preservation and fallback behavior.
- Derived smoke YAML and focused tests.

### Out of scope

- Renaming `experiment_name` because it remains a full experiment identity.
- Renaming `experiment_variant` because it remains protocol detail.
- Renaming scalar online compatibility fields.
- Changing `stage_b_best_checkpoint`, `threshold_artifact`, or output path contracts.

## Evidence

- `documents/spec/offline_pretraining_terminology_ontology.md:80-85` — canonical offline phase and stage objects.
- `documents/spec/offline_pretraining_terminology_ontology.md:109` — `experiment_variant` does not replace `offline_variant`.
- `documents/spec/online_tta_terminology_ontology.md:58-60` — canonical online phase and variant.
- `documents/spec/online_tta_terminology_ontology.md:88-90` — combined labels are not new online variants.
- `documents/notes/terminology_ontology_naming_conventions.md:343-369` — selected W&B smoke display convention.
- `src/core/artifact_naming.py:73-163` — current identity extraction and fallbacks.
- `src/core/artifact_naming.py:204-216` — current generic run-name construction.
- `src/engine/logger.py:104-116` — final W&B initialization boundary.
- `tests/benchmarks/test_smoke_wandb_logging.py:8-27` — existing smoke W&B test boundary.

## Phase 1: Establish canonical identity and the smoke display-name contract

### Goal

Make the selected naming rule executable from explicit canonical inputs.

### Dependencies

- Both ontology files have been read in full.
- The naming note is the selected project convention.
- `KA` is the only explicitly selected method display token.

### Detailed changes

#### 1. Add optional canonical identity fields to config validation

- **File:** `src/core/config.py`
- **Symbol:** `_validate_experiment_top_level_structure` around lines 245-263 and the experiment validation path.
- **Current responsibility:** Reject unknown top-level experiment fields while allowing `experiment_variant` and stage fields.
- **Change:** Allow optional top-level `offline_variant` and `online_variant` fields and validate their domains when present.
- **Reason:** The ontology distinguishes canonical experiment dimensions from protocol detail and filename inference.
- **Inputs:** `offline_variant` must be `O0` or `O1`; `online_variant` must be `A0`, `A1`, or `A2`.
- **Outputs:** Valid resolved configs expose canonical identity without replacing the existing `task` fields used by current runtime code.
- **Errors:** Reject unsupported variant values with a field-specific `ValueError`.
- **Dependencies:** Generator outputs and artifact identity extraction.
- **Compatibility:** Keep fields optional for legacy baseline configurations that do not use the THESIS variant dimensions.

Atomic steps:

- [ ] Add `offline_variant` and `online_variant` to the allowed top-level key set.
- [ ] Add validation for each field only when the field is present.
- [ ] Keep `experiment_variant` validation and semantics unchanged.
- [ ] Add tests for valid `O0`, `O1`, `A0`, `A1`, and `A2` values.
- [ ] Add tests for invalid values and missing optional fields.

#### 2. Add an explicit smoke display-name helper

- **File:** `src/core/artifact_naming.py`
- **Symbol:** Add a focused helper beside `build_wandb_run_name`.
- **Current responsibility:** `build_wandb_run_name` creates an artifact-style `run-...` name from inferred identity.
- **Change:** Add a helper that accepts explicit `phase_token`, `identity_tokens`, `entity_token`, and `seed` inputs and returns the validated `smk-...` string.
- **Reason:** Display labels and artifact identity have different contracts and must not be silently conflated.
- **Inputs:** `phase_token` is `off` or `on`; identity tokens are non-empty validated display tokens; entity and seed are required.
- **Outputs:** A name matching `smk-[A-Za-z0-9_.-]+-[A-Za-z0-9_.-]+-e...-s...` with no trailing mode or stage suffix.
- **Errors:** Reject unsupported phase tokens, empty token lists, empty entity, invalid seed, or unsupported characters.
- **Dependencies:** Generator and runtime fallback callers.
- **Compatibility:** Leave `build_wandb_artifact_name` and non-smoke artifact names unchanged.

Atomic steps:

- [ ] Define the phase-token validation set `off` and `on`.
- [ ] Define the exact token-joining order.
- [ ] Preserve the existing W&B character and length validation.
- [ ] Add the three user-selected example assertions.
- [ ] Add a negative test for an omitted seed.
- [ ] Add a negative test for a token containing unsupported characters.

### Tests

#### Smoke display-name composition

- **Location:** `tests/core/test_artifact_naming.py`
- **Level:** Unit.
- **Setup:** Pass explicit phase, token sequence, entity token, and seed values.
- **Action:** Call the new smoke display-name helper.
- **Expected result:** The helper returns `smk-off-O0-e1_6-s8`, `smk-on-KA-A2-e3_9-s36`, and `smk-off-KA-e3_4-s6` for the selected examples.
- **Edge cases:** Empty tokens, unsupported phase, missing seed, and overlong output.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/core/test_artifact_naming.py` — all naming tests pass.

#### Manual

- [ ] Compare the helper contract with `documents/notes/terminology_ontology_naming_conventions.md:343-369` — all three examples and canonical-identity rules match.

### Risks and recovery

- **Risk:** A helper could infer `offline_variant` from `experiment_variant` and produce a plausible but wrong name.
- **Mitigation:** Accept canonical identity explicitly and fail when it is unavailable.
- **Verification:** Unit tests provide protocol values that differ from `O0` and `O1` and assert they are not used as variant tokens.
- **Recovery:** Revert only the new helper and tests; no generated config or output path changes occur in this phase.

### Complete when

- The helper passes exact-example and negative tests.
- The helper does not change existing artifact-name outputs.

## Phase 2: Route generators and runtime callers through the contract

### Goal

Ensure every active smoke path emits or preserves the selected display name.

### Dependencies

- Phase 1 helper and config fields exist.

### Detailed changes

#### 1. Update THESIS offline generation

- **Files:** `scripts/benchmarks/generate_smd_benchmark_configs.py:81-103` and `scripts/benchmarks/generate_benchmark_smoke_configs.py:47-80`.
- **Symbol:** `_variant_logging` and `build_benchmark_smoke_config`.
- **Current responsibility:** Build W&B logging with `off-...-smoke` or an inherited long name.
- **Change:** Write root `offline_variant` and use the smoke helper only when `smoke` is true.
- **Reason:** THESIS offline smoke runs use `O0/O1` as canonical variant values.
- **Inputs:** `variant`, entity ID, seed, and smoke flag.
- **Outputs:** `smk-off-O0-e...-s...` or `smk-off-O1-e...-s...` for smoke configs.
- **Errors:** Propagate helper validation errors.
- **Dependencies:** Generated YAML and tests.
- **Compatibility:** Preserve non-smoke display names unless the same helper is intentionally selected for them later.

Atomic steps:

- [ ] Add `offline_variant: variant` to the generated THESIS offline root config.
- [ ] Convert the entity ID to the selected display token format.
- [ ] Call the smoke helper for smoke mode.
- [ ] Keep stage, window, protocol, and experiment identity in tags or config.
- [ ] Update the benchmark-smoke wrapper to use the same helper.

#### 2. Update THESIS online generation

- **File:** `scripts/benchmarks/generate_online_benchmark_configs.py:160-240`.
- **Symbol:** `build_online_benchmark_config` and its logging dictionary.
- **Current responsibility:** Store `offline_variant` in task overrides and generate `on-O0-A2-...-smoke`.
- **Change:** Store root `offline_variant` and `online_variant`, retain task fields for existing callers, and use `smk-on-O0-A2-e...-s...` for smoke.
- **Reason:** The online ontology treats `O0_A2` as a combined label, not a new `online_variant` value.
- **Inputs:** `offline_variant`, `online_variant`, entity ID, seed, and smoke flag.
- **Outputs:** A display name with the combined identity sequence and canonical root fields.
- **Errors:** Reject values outside the ontology domains.
- **Dependencies:** Config validation, online runner, and artifact identity.
- **Compatibility:** Keep `task.offline_variant` and task checkpoint fields for the current online engine.

Atomic steps:

- [ ] Add root `offline_variant` to the config dictionary.
- [ ] Add root `online_variant` to the config dictionary.
- [ ] Keep task override fields needed by the current runtime.
- [ ] Compose the smoke display name as `smk-on-<offline_variant>-<online_variant>-<entity>-s<seed>`.
- [ ] Keep `wandb_job_type` and tags for online benchmark, variant, entity, seed, window, and smoke.

#### 3. Update baseline generators

- **Files:** `scripts/benchmarks/generate_offline_benchmark_configs.py:91-126` and `scripts/benchmarks/generate_online_streaming_benchmark_configs.py:189-237`.
- **Symbol:** `build_offline_benchmark_config` and `build_online_streaming_benchmark_config`.
- **Current responsibility:** Use the full benchmark name or a legacy `on-...-smoke` string as the W&B name.
- **Change:** Use `KA` for `kmeans_ad` and preserve existing method tokens for methods without a selected abbreviation.
- **Reason:** Baseline smoke runs use method identity rather than THESIS variant identity.
- **Inputs:** Baseline method, online variant when present, entity ID, seed, and smoke flag.
- **Outputs:** `smk-off-KA-e...-s...` for offline `kmeans_ad` and `smk-on-KA-<online_variant>-e...-s...` when the baseline config exposes an online variant.
- **Errors:** Fail when a required method or seed is missing.
- **Dependencies:** Shared helper and baseline tests.
- **Compatibility:** Keep `baseline_name` and `online_variant` as canonical runtime fields; do not rename them without a separate mapping.

Atomic steps:

- [ ] Add a method-display-token map containing the explicit `kmeans_ad -> KA` mapping.
- [ ] Leave unconfirmed method tokens unchanged.
- [ ] Add the smoke prefix and remove the trailing `-smoke` suffix.
- [ ] Preserve full method name in tags and config.
- [ ] Add one exact `kmeans_ad` offline test and one online baseline test.

#### 4. Update the remaining-SMD matrix generator

- **File:** `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:160-202` and `325-336`.
- **Symbol:** `_add_run` and `_common_logging` call sites.
- **Current responsibility:** Use `run_id` as both full matrix identity and W&B name.
- **Change:** Keep `run_id` for full identity and construct a separate smoke display name from phase, method or variant, entity, and seed.
- **Reason:** The ontology requires the short display label and full reproducibility identity to remain separate.
- **Inputs:** `phase`, `method`, `variant`, `entity_id`, `seed`, and mode.
- **Outputs:** A short W&B name plus unchanged `run_id` in config and provenance.
- **Errors:** Reject an online run whose variant cannot be separated into offline and online components.
- **Dependencies:** Phase 1 helper and matrix tests.
- **Compatibility:** Keep output paths and report names unchanged.

Atomic steps:

- [ ] Add a display-token selection step after `_add_run` resolves method and variant.
- [ ] Call the helper only for smoke mode.
- [ ] Keep `run_id` in `experiment_name` or `benchmark_name`.
- [ ] Keep full run identity in tags and resolved config.

#### 5. Preserve configured names at runtime

- **Files:** `scripts/cli/train.py:264-269`, `scripts/cli/evaluate.py:306-312`, `scripts/experiments/run_online_adaptation.py:140-152`, `scripts/benchmarks/run_thesis_online_benchmark.py:310-324`, `scripts/experiments/run_two_stage_offline_pretraining.py:157-162`, and `scripts/benchmarks/run_online_streaming_benchmark.py:328-341`.
- **Symbol:** Logging configuration preparation before `ExperimentLogger` construction.
- **Current responsibility:** Rebuild or default W&B names before logger initialization.
- **Change:** Preserve a valid configured smoke name and use the new helper only when a smoke config has no name.
- **Reason:** Runtime overwrites currently bypass source-generator output.
- **Inputs:** Resolved logging config, experiment identity, and smoke mode.
- **Outputs:** The exact configured smoke name reaches `ExperimentLogger` and `wandb.init`.
- **Errors:** Fail when a smoke run has W&B enabled but has no enough identity to build a name.
- **Dependencies:** The helper and config generators.
- **Compatibility:** Keep the old helper for non-smoke artifact and run paths until separately migrated.

Atomic steps:

- [ ] Add a valid-name predicate for the selected smoke grammar.
- [ ] Check for an explicit configured name before invoking the generic helper.
- [ ] Preserve the configured name when the predicate passes.
- [ ] Use the smoke helper only when the config is smoke and the name is absent.
- [ ] Leave non-smoke fallback behavior unchanged.
- [ ] Add a fake-W&B test that captures the final `name` argument.

#### 6. Update the direct-branch-routing smoke entry point

- **File:** `scripts/run_direct_branch_routing_smoke.py:62-71`.
- **Symbol:** `build_smoke_experiment_config`.
- **Current responsibility:** Hard-code `off-O0-machine_1_6-s6-direct-smoke`.
- **Change:** Use `smk-off-O0-e1_6-s6` and retain `direct-branch-routing` in tags and experiment metadata.
- **Reason:** The current name adds unapproved suffixes and omits the `smk-` prefix.
- **Inputs:** Existing fixed variant, entity, and seed.
- **Outputs:** A compliant smoke name with unchanged output path and execution behavior.
- **Errors:** None beyond shared helper validation.
- **Dependencies:** Phase 1 helper.
- **Compatibility:** Keep direct branch routing in `experiment_variant` and tags.

Atomic steps:

- [ ] Replace the hard-coded W&B name with the helper call.
- [ ] Keep the direct-branch-routing tag.
- [ ] Add an exact assertion in `tests/benchmarks/test_direct_branch_routing_smoke_runner.py`.

### Tests

#### Generator output

- **Location:** Existing generator-specific test files under `tests/benchmarks/` and `tests/online/`.
- **Level:** Unit and integration-style generator tests.
- **Setup:** Call each existing config-generator function with smoke mode.
- **Action:** Read the generated logging and canonical identity fields.
- **Expected result:** The display name matches the selected grammar and the full identity remains present.
- **Edge cases:** THESIS O0, THESIS O1, online A0/A1/A2, `kmeans_ad`, and the remaining baseline method tokens.

#### Runtime name preservation

- **Location:** `tests/runtime/test_logger_wandb.py` and the relevant runner tests.
- **Level:** Runtime boundary test.
- **Setup:** Inject a fake `wandb` module and a smoke config with an explicit `smk-` name.
- **Action:** Construct the logger through the runner preparation path.
- **Expected result:** Fake `wandb.init` receives the explicit `smk-` name unchanged.
- **Edge cases:** Missing name, invalid name, and non-smoke config.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/core/test_artifact_naming.py tests/runtime/test_logger_wandb.py` — helper and logger boundary pass.
- [ ] `.venv/bin/python -m pytest -q tests/benchmarks/test_benchmark_config_generation.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py` — generator contracts pass.

#### Manual

- [ ] Inspect one THESIS offline, one THESIS online, one offline `kmeans_ad`, and one online baseline YAML after regeneration.

### Risks and recovery

- **Risk:** A runtime caller reintroduces a generic name after generation.
- **Mitigation:** Test the final `wandb.init` argument and preserve explicit config names.
- **Verification:** Fake-W&B runtime tests.
- **Recovery:** Revert only the caller change and use the previous configured-name behavior while investigating the failing path.

### Complete when

- All active smoke sources use the shared helper or preserve a valid configured name.
- No active smoke source appends `-smoke` or uses the long experiment name as the W&B display name.

## Phase 3: Regenerate derived smoke configurations and preserve provenance

### Goal

Regenerate all tracked smoke YAML from source generators and verify canonical identity retention.

### Dependencies

- Phase 2 generator and runtime changes.

### Detailed changes

#### 1. Regenerate source-owned YAML

- **Files:** `configs/experiment/**` and `scripts/configs/experiment/**` smoke YAML generated by the active benchmark scripts.
- **Symbol:** Existing generator entry points.
- **Current responsibility:** Store derived experiment configurations.
- **Change:** Recreate the derived files from updated sources.
- **Reason:** Direct YAML edits would not persist when generators run again.
- **Inputs:** Existing repository generators and current matrix constants.
- **Outputs:** Derived YAML with compliant display names and canonical fields.
- **Errors:** Generator or config-loader failures stop regeneration.
- **Dependencies:** All Phase 2 sources.
- **Compatibility:** Keep filenames, output paths, checkpoint paths, and full experiment names unchanged unless a generator already owns them.

Atomic steps:

- [ ] Run the THESIS offline generator.
- [ ] Run the benchmark-smoke generator.
- [ ] Run the THESIS online generator.
- [ ] Run the offline baseline generator.
- [ ] Run the online streaming generator.
- [ ] Run the remaining-SMD matrix generator only for its configured smoke output root.
- [ ] Inspect the generated diff before any test run.

#### 2. Preserve metadata omitted from display names

- **Files:** Generator logging dictionaries and resolved-config serialization.
- **Symbol:** `wandb_tags`, `wandb_job_type`, `experiment_name`, `experiment_variant`, stage fields, and artifact references.
- **Current responsibility:** Store detailed run identity beside the W&B name.
- **Change:** Keep detailed identity fields and add no stage or protocol suffix to the selected display name.
- **Reason:** The ontology requires compact labels without losing reproducibility.
- **Inputs:** Existing generator metadata.
- **Outputs:** W&B metadata and resolved config retain the omitted details.
- **Errors:** Fail tests if a required canonical field disappears.
- **Dependencies:** Phase 2 generators.
- **Compatibility:** No W&B artifact filename or output-directory change.

Atomic steps:

- [ ] Check canonical phase or phase-token metadata.
- [ ] Check offline and online variant metadata.
- [ ] Check method identity metadata for baseline runs.
- [ ] Check entity and seed metadata.
- [ ] Check stage, window, protocol, checkpoint, and threshold references.

### Tests

#### Full smoke inventory

- **Location:** `tests/benchmarks/test_smoke_wandb_logging.py`.
- **Level:** Repository inventory test.
- **Setup:** Enumerate YAML under `configs/experiment` and `scripts/configs/experiment` using the existing smoke predicate.
- **Action:** Load each YAML and inspect `logging` and identity fields.
- **Expected result:** Every smoke file enables online W&B and has a valid `smk-` name.
- **Edge cases:** `smoke5`, `smoke_cuda`, benchmark-smoke directories, and baseline configurations with different schemas.

### Verification

#### Automated

- [ ] `.venv/bin/python -m pytest -q tests/benchmarks/test_smoke_wandb_logging.py tests/benchmarks/test_benchmark_config_generation.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py` — smoke grammar, generation, and representative loading contracts pass.
- [ ] Confirm that the generator tests load the representative generated configurations without configuration-loading exceptions.

#### Manual

- [ ] Compare a short W&B name with its resolved config and output path.
- [ ] Confirm the stage and checkpoint identity remain outside the display name.

### Risks and recovery

- **Risk:** Regeneration changes unrelated generated content.
- **Mitigation:** Inspect `git diff` and retain only changes caused by naming and canonical identity fields.
- **Verification:** `git diff --check` and focused generator tests.
- **Recovery:** Re-run the generator at the previous revision or revert only generated files after confirming exact targets.

### Complete when

- The full inventory contains only compliant smoke display names.
- The source generators reproduce the checked-in files.

## Phase 4: Verify compatibility and finalize the change

### Goal

Prove the change is limited to W&B display names and explicit identity fields.

### Dependencies

- Phases 1 through 3.

### Detailed changes

#### 1. Extend tests without changing compatibility contracts

- **Files:** `tests/core/test_artifact_naming.py`, `tests/runtime/test_logger_wandb.py`, `tests/benchmarks/test_smoke_wandb_logging.py`, and generator-specific tests.
- **Symbol:** Existing test functions and new focused test functions.
- **Current responsibility:** Verify artifact naming, logger behavior, W&B enablement, and generator output.
- **Change:** Add exact smoke-name assertions, identity-preservation assertions, and final `wandb.init` assertions.
- **Reason:** Existing tests pass while leaving the selected naming rule untested.
- **Inputs:** Representative configs and generated files.
- **Outputs:** Failing tests for legacy names and passing tests after regeneration.
- **Errors:** Tests must identify the exact config path or generated field that fails.
- **Dependencies:** All earlier phases.
- **Compatibility:** Keep existing artifact-name tests unless their intended contract is explicitly changed.

Atomic steps:

- [ ] Add exact assertions for the three selected examples.
- [ ] Add a grammar assertion for every smoke config.
- [ ] Add a canonical identity assertion for THESIS variants.
- [ ] Add a method-token assertion for `kmeans_ad -> KA`.
- [ ] Add a runtime preservation assertion at fake `wandb.init`.
- [ ] Add a test that non-smoke behavior is not changed accidentally.

#### 2. Run final verification

- **File:** No source file change.
- **Symbol:** Repository verification commands.
- **Current responsibility:** Establish evidence for the completed modification.
- **Change:** Run focused tests, config loading, and whitespace validation.
- **Reason:** The naming change spans generators, runtime callers, and derived YAML.
- **Inputs:** Updated source and generated files.
- **Outputs:** Test results and a clean diff check.
- **Errors:** Stop on any failed test or malformed YAML.
- **Dependencies:** All earlier phases.
- **Compatibility:** No remote W&B run is required for this read-only planning task.

Atomic steps:

- [ ] Run the naming helper tests.
- [ ] Run the logger boundary tests.
- [ ] Run generator tests.
- [ ] Run the smoke inventory test.
- [ ] Load every changed YAML with the existing config loader.
- [ ] Run `git diff --check`.
- [ ] Inspect the final diff for accidental output-path or checkpoint changes.

### Tests

#### Final focused suite

- **Location:** Existing test files named above.
- **Level:** Unit, generator, and runtime boundary tests.
- **Setup:** Current repository virtual environment and regenerated YAML.
- **Action:** Run the focused commands from the verification section.
- **Expected result:** All tests pass and every smoke name matches the selected grammar.
- **Edge cases:** Missing identity, invalid token, baseline config, online combined variant, and legacy compatibility fields.

### Verification

#### Automated

- [x] `.venv/bin/python -m pytest -q tests/core/test_artifact_naming.py tests/runtime/test_logger_wandb.py tests/benchmarks/test_smoke_wandb_logging.py` — `19 passed`.
- [x] `.venv/bin/python -m pytest -q tests/benchmarks/test_benchmark_config_generation.py tests/benchmarks/test_offline_benchmark_config_generation.py tests/online/test_online_benchmark_config_generation.py tests/online/test_online_streaming_benchmark_config_generation.py` — representative generator tests pass.
- [x] `git diff --check` — no whitespace errors.

#### Manual

- [ ] Confirm no local filename, checkpoint path, output path, metric key, threshold key, or online compatibility field was renamed.

### Risks and recovery

- **Risk:** A naming refactor could alter historical artifact lookup or report parsing.
- **Mitigation:** Keep local and artifact names unchanged and restrict changes to W&B display names and explicit identity fields.
- **Verification:** Inspect the final diff and run existing artifact and report tests.
- **Recovery:** Revert the W&B display-name changes while retaining any independently approved canonical identity migration.

### Complete when

- All focused tests pass.
- All smoke configs use the selected `smk-` format.
- Full canonical identity remains available outside the display name.
- The final diff contains no unrelated terminology changes.

## Interface and data changes

The optional root config fields `offline_variant` and `online_variant` become explicit canonical identity fields.

Existing task-level fields remain for current runtime consumers until a separate migration changes their ownership.

The W&B display name becomes a short presentation field and is not used as the source of checkpoint, artifact, or protocol identity.

No serialized threshold, runtime state, checkpoint, or metric schema changes are part of this plan.

## Deployment and rollout

Update source generators before regenerating derived YAML.

Run focused tests before any smoke experiment.

Run one concrete smoke flow after the tests pass, following the repository rule for one end-to-end combination before a larger batch.

Do not rename or mutate historical W&B runs.

## Documentation changes

Keep `documents/notes/terminology_ontology_naming_conventions.md` as the naming authority.

Add a method-token mapping to the note only when anh explicitly selects additional abbreviations.

Keep the research, plan, structure, and detail logs linked through their front matter.

## Final verification

- [x] Every smoke configuration has W&B enabled and online mode.
- [x] Every smoke W&B name begins with `smk-` and has the selected token order.
- [x] `offline_variant` and `online_variant` are not silently replaced by `experiment_variant` or filename inference.
- [x] Runtime callers preserve valid configured names.
- [x] Full identity remains available in config, tags, metadata, or provenance.
- [x] Focused tests pass and `git diff --check` is clean.

## Assumptions and non-blocking uncertainties

`KA` is the only method abbreviation selected by the user.

Other method tokens remain unchanged until anh selects a new mapping.

The online state variable `previous_ewma_point_scores` remains outside this plan because it is a separate runtime contract migration.
