---
date: 2026-07-13T00:00:00+07:00
researcher: Codex
planner: Codex
git_commit: unknown
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for thesis online benchmark checkpoint metadata resolver"
tags: [detail, thesis, online-benchmark, checkpoint-resolver, config-schema, validation, wrapper, preflight, legacy-compatibility]
status: draft
source_plan: user-provided plan outline in chat
source_structure: prompts/4_detail_prompt.md
last_updated: 2026-07-13
last_updated_by: Codex
---

# Detailed programming plan: thesis online benchmark checkpoint metadata resolver

## 1. Implementation contract

This document is the implementation contract for the thesis online benchmark checkpoint metadata refactor. It follows the selected design option and expands it into an executable sequence of phases. The target change is narrow and specific: online benchmark YAML files must no longer store a full `reference_checkpoint_path`. Instead, they must store metadata that identifies the offline run, and the wrapper must resolve the correct Stage B checkpoint before the online runtime starts.

The implementation must preserve the repository constraints:

- one public model entrypoint per model file;
- composition over inheritance;
- short functions and small files;
- explicit configuration and validation boundaries;
- deterministic, fail-fast resolution;
- backward compatibility for legacy config and artifact readers;
- minimal codepaths with readable names.

The implementation order is intentionally strict. Schema changes must land before resolver logic, resolver logic must land before runtime integration, and runtime integration must land before regeneration of benchmark YAML files. Tests must cover every step.

## 2. Stable cross-layer contracts

### 2.1 Online benchmark config contract

The online thesis benchmark config must shift from path-based reference to metadata-based identity.

Required metadata fields:

- `offline_variant`
- `entity_id`
- `seed`
- `benchmark_mode`
- `stage_name`

The config may still contain other benchmark fields, but the resolver must only depend on the explicit metadata above plus the existing benchmark naming context. The config must not depend on an inferred directory layout that is hidden from the schema.

### 2.2 Checkpoint resolution contract

The wrapper must resolve exactly one Stage B checkpoint from metadata before calling the online runtime. Resolution must be deterministic and must not guess.

Resolution rules:

- if zero candidate checkpoints exist, fail immediately with a clear error;
- if more than one candidate checkpoint matches, fail immediately with a clear error;
- if a legacy config already contains a direct path, the model may still read it through the legacy fallback;
- the new wrapper path is the primary path for new configs.

### 2.3 Model and runtime contract

The online adaptation model must keep its public entrypoint stable. The model must accept a resolved checkpoint path from the wrapper and must keep the legacy fallback only as a secondary path.

The runtime contract is:

```text
config metadata -> wrapper resolve -> resolved checkpoint path -> online runtime
```

The model must not become the place where path search logic grows. It may own fallback loading for old artifacts, but it must not become a second resolver.

### 2.4 Validation contract

Validation must accept the metadata-based config and must reject incomplete or ambiguous metadata. The validator should be strict enough that a broken config fails before the benchmark matrix starts.

### 2.5 Preflight contract

Preflight must use the same resolution rules as the wrapper. It must verify that every benchmark config in the matrix resolves to a real Stage B checkpoint before the benchmark run begins.

## 3. Phase 1 - Define the new online config schema

### 3.1 Phase summary

This phase freezes the schema boundary. The objective is to replace `reference_checkpoint_path` as the primary online config mechanism with a metadata-only contract that is small, explicit, and easy to validate.

### 3.2 File-level edits

Modify these files first:

- `src/core/config_model_validation.py`
- `configs/experiment/online_benchmark/thesis/` config templates if they encode schema assumptions
- any shared config schema helper that the validation layer already uses for online benchmark configs

### 3.3 Explicit edit content

1. Remove the assumption that online config validity depends on storing a full checkpoint path.
2. Add validation for the required metadata fields listed above.
3. Define strict missing-field errors so that invalid configs fail with explicit messages.
4. Keep a compatibility path for legacy configs that still contain `reference_checkpoint_path`.
5. Ensure schema validation remains readable and linear, with no hidden inference chain.

### 3.4 Interface and contract definitions

The validation layer must define:

- a metadata-based online benchmark config shape;
- a legacy config shape with direct checkpoint path support;
- a clear error contract for missing metadata;
- a clear error contract for ambiguous metadata.

The validator should not know how checkpoint search works internally. It should only know what a complete config looks like and when to reject it.

### 3.5 Design pattern application

Use composition over inheritance for the validation helpers. If legacy and new schemas share logic, factor that into small pure functions. If multiple benchmark modes map to different metadata rules, use a strategy-style dispatch table or registry instead of branching deeply inside one validator.

### 3.6 Risk mitigation

- Metadata may be incomplete. Mitigation: require all mandatory fields and fail closed.
- Metadata may be too broad. Mitigation: enforce uniqueness-sensitive validation rules.
- Legacy support may hide schema drift. Mitigation: keep legacy logic explicitly separated from new logic.

### 3.7 Test plan and validation

Add or update tests to verify:

- missing metadata is rejected;
- valid metadata passes;
- legacy path-based configs still load;
- invalid combinations produce deterministic errors.

Suggested test targets:

- `tests/online/test_online_benchmark_config_generation.py`
- `tests/online/test_online_reference_checkpoint.py`
- `tests/runtime/test_kaggle_config_validation.py` if shared validation helpers are reused there

### 3.8 Acceptance criteria

This phase is complete only when:

- metadata-based online config validates successfully;
- missing metadata fails fast;
- legacy path-based config still validates through the fallback route;
- no new config contract depends on a hard-coded checkpoint path.

## 4. Phase 2 - Update the config generator

### 4.1 Phase summary

This phase changes the YAML generation step so that generated online configs encode offline-run identity, not a full Stage B checkpoint path.

### 4.2 File-level edits

Modify:

- `scripts/benchmarks/generate_online_benchmark_configs.py`
- any helper used only by that generator for filename or metadata construction
- benchmark config fixtures under `configs/experiment/online_benchmark/thesis/`

### 4.3 Explicit edit content

1. Replace direct checkpoint path serialization with metadata serialization.
2. Ensure generated configs contain the new required fields.
3. Keep the generated YAML concise and stable across reruns.
4. Avoid embedding filesystem-specific knowledge inside generation logic.
5. Keep filename generation separate from checkpoint resolution.

### 4.4 Interface and contract definitions

The generator must define:

- input benchmark descriptors;
- output config records with the required metadata;
- stable naming rules for generated YAML files;
- no runtime checkpoint discovery.

The generator should only emit data needed to resolve the checkpoint later.

### 4.5 Design pattern application

Use a small registry or lookup table to map benchmark variants to the metadata payload. Avoid adding inheritance layers for format variations. If multiple benchmark families are supported, keep them as explicit generation strategies.

### 4.6 Risk mitigation

- Generated metadata may not be sufficient for resolution. Mitigation: reuse the same required fields that the validator expects.
- Generated configs may silently diverge from validation rules. Mitigation: run validation immediately after generation in tests.

### 4.7 Test plan and validation

Update generation tests to assert:

- no generated config contains `reference_checkpoint_path` as the main path contract;
- the new metadata fields are present;
- generated config remains valid under the updated validator.

Suggested test targets:

- `tests/benchmarks/test_benchmark_config_generation.py`
- `tests/online/test_online_benchmark_config_generation.py`

### 4.8 Acceptance criteria

This phase is complete only when generated thesis online configs:

- store metadata instead of a full checkpoint path;
- validate under the new schema;
- remain readable and stable.

## 5. Phase 3 - Implement the wrapper checkpoint resolver

### 5.1 Phase summary

This phase adds the primary runtime resolver in the benchmark wrapper. The wrapper becomes responsible for translating config metadata into a concrete Stage B checkpoint path before the online runtime starts.

### 5.2 File-level edits

Modify:

- `scripts/run_thesis_online_benchmark.py`
- a small new helper module if the resolver logic would otherwise become too large
- any shared utility used by benchmark wrappers for path joining or artifact lookup

### 5.3 Explicit edit content

1. Read the new metadata-based config.
2. Resolve the exact Stage B checkpoint path from the offline run identity.
3. Fail fast if the checkpoint is missing.
4. Fail fast if multiple checkpoints match the same metadata.
5. Pass only the resolved path into the online runtime.
6. Keep the resolution helper small enough to test directly.

### 5.4 Interface and contract definitions

The wrapper should expose a narrow internal contract:

```text
resolve_stage_b_checkpoint(config) -> pathlib.Path
```

The wrapper contract must guarantee:

- deterministic path resolution;
- no silent fallback to a guessed file;
- clear error messages for missing or ambiguous candidates;
- a single resolved path for the runtime.

### 5.5 Design pattern application

Apply composition over inheritance by keeping the resolver as a helper component rather than a base class. If benchmark mode changes the resolution rule, use a strategy mapping keyed by `benchmark_mode`. If metadata names map to artifact groups, a small registry is appropriate.

### 5.6 Risk mitigation

- Multiple similar runs may exist. Mitigation: require exact metadata matching and reject ambiguity.
- Path construction may be correct syntactically but wrong semantically. Mitigation: verify file existence before runtime launch.
- Wrapper logic may grow too large. Mitigation: split resolve, validate, and launch into separate helpers.

### 5.7 Test plan and validation

Update wrapper tests to confirm:

- the wrapper resolves the correct Stage B path;
- the wrapper fails on missing artifacts;
- the wrapper fails on ambiguous matches;
- the wrapper passes the resolved path into the runtime call.

Suggested test targets:

- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`
- `tests/online/test_online_streaming_benchmark_wrapper.py` if the wrapper shares resolver behavior

### 5.8 Acceptance criteria

This phase is complete only when:

- the wrapper resolves the checkpoint on its own;
- the runtime receives a concrete path;
- the wrapper never guesses when metadata is insufficient.

## 6. Phase 4 - Keep legacy fallback in the model

### 6.1 Phase summary

This phase preserves compatibility with older artifacts and config files while making the new wrapper path the primary path for all new runs.

### 6.2 File-level edits

Modify:

- `src/models/online_impl/online_adaptation.py`
- any small helper used only for legacy config or artifact loading

### 6.3 Explicit edit content

1. Prefer the resolved checkpoint supplied by the wrapper.
2. Keep the existing legacy fallback path reader for older artifacts.
3. Separate the primary load path from the fallback path with explicit helper names.
4. Avoid path discovery logic inside the model.
5. Keep the public model entrypoint stable.

### 6.4 Interface and contract definitions

The model must define a clear load contract:

```text
load_resolved_checkpoint(path) -> model_state
load_legacy_artifact(config_or_path) -> model_state
```

The resolved path is the main route. The legacy route only exists to avoid breaking older experiments.

### 6.5 Design pattern application

Use an adapter for legacy artifacts if needed. Do not build a second inheritance tree for checkpoint loading. Keep the logic explicit and readable so the primary path is obvious to a reader.

### 6.6 Risk mitigation

- Legacy fallback may hide new schema failures. Mitigation: ensure the wrapper is the default path for new configs.
- Model path logic may become duplicated. Mitigation: centralize the load helpers.

### 6.7 Test plan and validation

Update tests so they verify:

- the new resolved checkpoint path is preferred;
- the legacy fallback still works for old artifacts;
- the model does not invent checkpoint paths on its own.

Suggested test targets:

- `tests/online/test_online_reference_checkpoint.py`
- `tests/runtime/test_trainer_checkpoint_fallback.py`
- `tests/runtime/test_checkpoint_roundtrip.py` if the checkpoint load path is shared

### 6.8 Acceptance criteria

This phase is complete only when:

- the model loads from the wrapper-resolved path by default;
- old artifacts still load through legacy fallback;
- no new code path in the model guesses a checkpoint location.

## 7. Phase 5 - Extend preflight to the new resolver contract

### 7.1 Phase summary

This phase ensures the benchmark matrix is checked before launch. Preflight must use the same resolution logic as the wrapper so that mismatches do not appear only at runtime.

### 7.2 File-level edits

Modify:

- `scripts/ops/preflight_full_benchmark_matrix.py`
- the helper used by preflight to enumerate benchmark configs

### 7.3 Explicit edit content

1. Load each online benchmark config.
2. Resolve the Stage B checkpoint using the same rule as the wrapper.
3. Verify the file exists.
4. Fail immediately on missing or ambiguous artifacts.
5. Emit concise per-config and aggregate diagnostics.

### 7.4 Interface and contract definitions

Preflight must expose a clear contract:

```text
preflight_benchmark_config(config) -> resolved_path | error
```

The preflight path must not diverge from wrapper semantics. If the wrapper can resolve it, preflight must resolve it the same way.

### 7.5 Design pattern application

If benchmark modes vary, use a strategy dispatch table. Keep the shared resolver component in one place so both wrapper and preflight consume the same behavior.

### 7.6 Risk mitigation

- Preflight may drift from wrapper if logic is copied. Mitigation: share the core resolver helper.
- Preflight may pass too much silently. Mitigation: print explicit failures for each broken config.

### 7.7 Test plan and validation

Update preflight tests to assert:

- valid configs resolve cleanly;
- missing checkpoint files fail;
- ambiguous metadata fails;
- preflight and wrapper agree on the same config set.

Suggested test targets:

- `tests/benchmarks/test_full_benchmark_matrix_preflight.py`
- `tests/benchmarks/test_comparative_preflight.py` if it shares config scanning logic

### 7.8 Acceptance criteria

This phase is complete only when preflight can block a bad benchmark matrix before the online run begins.

## 8. Phase 6 - Regenerate configs and close regression coverage

### 8.1 Phase summary

This phase synchronizes the checked-in YAML files with the new schema and seals the regression surface with tests.

### 8.2 File-level edits

Modify:

- `configs/experiment/online_benchmark/thesis/`
- any checked-in benchmark YAML that the generator rewrites
- related test fixtures or snapshots if the repository uses them

### 8.3 Explicit edit content

1. Regenerate all thesis online benchmark configs.
2. Ensure the regenerated YAML uses metadata-only identity.
3. Remove lingering direct checkpoint path usage from checked-in generated files.
4. Update snapshot tests if file contents changed intentionally.
5. Check that regenerated configs still validate and preflight correctly.

### 8.4 Interface and contract definitions

The regenerated config set must satisfy:

- metadata-based identity is present in every new config;
- resolved path is not stored as the main contract;
- legacy configs remain readable only through fallback.

### 8.5 Design pattern application

No new patterns are needed here. This phase is about contract freezing and regression closure.

### 8.6 Risk mitigation

- Some files may still contain the old path field. Mitigation: scan the entire config directory after regeneration.
- Regenerated files may not match tests. Mitigation: update fixtures and tests in the same change set.

### 8.7 Test plan and validation

Run or update tests covering:

- config generation;
- wrapper resolution;
- preflight resolution;
- legacy checkpoint fallback;
- round-trip load behavior where applicable.

Suggested test targets:

- `tests/benchmarks/test_benchmark_config_generation.py`
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`
- `tests/benchmarks/test_full_benchmark_matrix_preflight.py`
- `tests/online/test_online_reference_checkpoint.py`

### 8.8 Acceptance criteria

This phase is complete only when:

- all regenerated thesis online configs use metadata-based identity;
- no checked-in new config relies on a hard-coded checkpoint path;
- the validation, wrapper, preflight, and legacy fallback tests all pass together.

## 9. Final verification checklist

Before implementation is considered complete, the repository must satisfy all of the following:

- online config schema accepts metadata-based identity;
- generator produces metadata-based YAML;
- wrapper resolves the correct Stage B checkpoint;
- model still supports legacy fallback for old artifacts;
- validation rejects missing or ambiguous metadata;
- preflight uses the same resolution rule as the wrapper;
- benchmark YAML has been regenerated;
- regression tests cover config generation, wrapper resolution, preflight, and fallback behavior.

## 10. Recommended execution order

The safest implementation order is:

```text
schema and validation
  -> generator update
  -> wrapper resolver
  -> model legacy fallback cleanup
  -> preflight update
  -> tests and regeneration
```

This order keeps the contract narrow at each step and prevents the runtime from getting ahead of the schema.
