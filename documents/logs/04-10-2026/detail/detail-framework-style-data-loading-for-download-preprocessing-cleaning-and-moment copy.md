---
date: 2026-04-10 18:18:59 +0700
planner: TheMetaSetter
git_commit: 33f0e4ef21ad9862ee5d979ae9143084497736e4
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for backward-compatible framework-style data loading, notebook usage, baseline handoff, and Kaggle checkpoint persistence"
tags: [detail, data-framework, smd, moment, backward-compatibility, kaggle]
status: complete
last_updated: 2026-04-10
last_updated_by: TheMetaSetter
source_research: documents/logs/04-10-2026/research/research-framework-style-data-loading-for-download-preprocessing-cleaning-and-moment.md
---

# Detail: Detailed implementation plan for backward-compatible framework-style data loading, notebook usage, baseline handoff, and Kaggle checkpoint persistence

## Overview

This detailed plan is derived directly from `documents/logs/04-10-2026/research/research-framework-style-data-loading-for-download-preprocessing-cleaning-and-moment.md`. The prompt under `prompts/4_detail_prompt.md` normally expects a preceding structure document. No same-topic `structure` document exists for April 10, 2026, and the user explicitly requested that the detail pass be invoked on the research note. Accordingly, this document uses the research note as the source artifact and translates it into a programming-level execution sequence.

The governing constraints for this implementation pass are:

- backward compatibility is the primary engineering rule;
- the existing YAML-driven script path must continue to function during the migration;
- the existing batch contract and model output contract must remain stable;
- readability and least-amount-of-codepaths must remain the dominant codebase values;
- one-model-one-file remains non-negotiable for active model logic;
- Kaggle persistence, if introduced, must be additive and must not weaken existing local checkpointing or Weights & Biases logging.

The objective of this detail pass is not to redesign the thesis pipeline. It is to expose the already-existing SMD data path and batch contract through a more framework-like public surface while preserving current training, evaluation, ablation, and online-adaptation behavior.

## Global Contracts And Backward-Compatibility Rules

### Runtime layers

The active runtime shall remain organized around four layers:

1. Configuration
2. Data
3. Model
4. Engine

No phase in this document shall collapse those layers into a notebook-only or script-only path. New notebook-friendly helpers must wrap the existing layers rather than bypass them.

### Dataset, window, and batch contracts

The current contracts must remain the active contracts throughout the migration:

```python
raw_sequence = {
    "x": Tensor[T, D],
    "point_labels": Optional[Tensor[T]],
    "mask": Optional[Tensor[T, D]],
    "timestamps": Optional[Tensor[T]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "num_channels": int,
        "sequence_length": int,
    },
}
```

```python
window = {
    "x": Tensor[L, D],
    "point_labels": Optional[Tensor[L]],
    "mask": Optional[Tensor[L, D]],
    "timestamps": Optional[Tensor[L]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "start_index": int,
        "end_index": int,
        "window_size": int,
    },
}
```

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The active default SMD window parameters remain:

- `window_size = 100`
- `stride = 10`

### Model and engine contracts

Every active model must continue to expose:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor],
    "point_scores": Optional[Tensor[B, L]],
    "window_scores": Optional[Tensor[B]],
    "aux": dict,
}
```

Every active stage method must continue to expose:

```python
step_output = {
    "loss": Tensor,
    "log": dict[str, float],
    "outputs": dict[str, Any],
}
```

This detail pass must not change the trainer so that existing offline models stop working. Classical-baseline support, if added later in the migration, must be introduced through an additive adapter or second execution path rather than by mutating the PyTorch model contract in place.

### Compatibility policy

The following rules are mandatory across all phases:

- Existing imports such as `build_smd_dataset_bundle(...)` and `build_smd_dataloaders(...)` must remain available until a complete deprecation cycle is documented.
- Existing YAML keys such as `data.root_dir`, `checkpoint_dir`, `output_dir`, and the current `logging.*` keys must remain valid.
- Existing scripts must keep their current command-line arguments and defaults.
- Existing tests must continue to pass before any new public wrapper is treated as active.
- New notebook-friendly helpers must be wrappers around current internal implementations, not parallel reimplementations.

## Phase 1 - Establish a Backward-Compatible Public Data API Layer

### Phase summary

This phase introduces the framework-facing public surface without disturbing the current script-facing surface. The thesis objective is to expose the existing SMD data path in a way that notebook users can call in a few lines while preserving the current parser, scaler, loader, collate, and registry behavior exactly.

### File-level edits

The following files should be modified or added in this phase:

```text
src/data/loaders.py
src/data/base.py
src/data/__init__.py
src/data/api.py
src/data/public_types.py
src/core/contracts.py
tests/test_public_data_api.py
tests/test_registry.py
tests/test_smoke_loader_limits.py
notebooks/time_series_loading_template.ipynb
notebooks/smd_colab_window_preprocessing_template.ipynb
```

### Explicit edit content

`src/data/api.py`

- Add a public wrapper function `load_smd_data(...)` that accepts explicit keyword arguments such as:
  - `root`
  - `window_size`
  - `stride`
  - `batch_size`
  - `validation_split_ratio`
  - `num_workers`
  - `shuffle_train`
  - `download`
  - `max_train_windows`
  - `max_val_windows`
  - `max_test_windows`
- The function must internally translate those explicit arguments into the same config dictionary expected by `build_smd_dataset_bundle(...)`.
- The function must return the same bundle shape already returned by `build_smd_dataset_bundle(...)`.
- The function must not duplicate parser, scaler, or dataset-building logic.

`src/data/public_types.py`

- Add small typed containers or dataclasses that describe the public return object. The public object may still carry the existing dictionary data, but a named type should make notebook usage more legible.
- The type must describe:
  - `parser`
  - `scaler`
  - `raw_sequences`
  - `scaled_sequences`
  - `datasets`
  - `loaders`
- The type definition must remain additive. Existing dictionary-based downstream code must still work unchanged.

`src/data/loaders.py`

- Keep `build_smd_dataset_bundle(...)` and `build_smd_dataloaders(...)` intact.
- Refactor any repeated config-default resolution into private helpers so that the new public API and the existing registry path share one implementation.
- Do not remove or rename the existing entrypoints.

`src/data/__init__.py`

- Export the new notebook-facing public helpers from one obvious import surface.

`src/core/contracts.py`

- Add validation helpers for the public bundle wrapper only if the helper does not create a second incompatible bundle vocabulary.
- Do not rename or relax the existing raw sequence, window, batch, or model-output contracts.

`notebooks/smd_colab_window_preprocessing_template.ipynb`

- Replace notebook-local calls that reconstruct logic already present in `src/data/` with calls into the new public wrapper where possible.
- Keep the notebook readable by showing the public wrapper call first, then optionally exposing internal stages for inspection.
- Preserve the current educational cells that inspect shapes and metadata.

### Interface and contract definitions

- Dataset interface:
  - `load_smd_data(...)` is the notebook-facing wrapper.
  - `build_smd_dataset_bundle(data_config)` remains the registry-facing constructor.
- Encoder interface:
  - unchanged in this phase.
- Model interface:
  - unchanged in this phase.
- Task interface:
  - unchanged in this phase.
- Training engine:
  - unchanged in this phase.

### Design pattern application

- Composition over inheritance:
  - the public API must compose the existing parser, scaler, and dataset builder.
- Adapter pattern for encoders:
  - not activated yet; reserved for Phase 3.
- Strategy pattern for tasks:
  - no new task strategy is introduced in this phase.
- Registry or factory:
  - the current dataset registry must remain the canonical script-facing construction path.

### Risk mitigation

- Prototype redundancy:
  - no model logic changes are permitted in this phase.
- Fusion collapse:
  - no multitask fusion code is permitted to change in this phase.
- Adaptation contamination:
  - online stream behavior remains untouched.
- Projector drift:
  - online adaptation code remains untouched.
- Evaluation metric inflation:
  - evaluator behavior remains untouched; no changes to threshold logic are permitted.
- Backward-compatibility breakage:
  - new public wrapper must round-trip to the exact same `WindowDataset`, `DataLoader`, and bundle contents as the current builder.

### Test plan and validation steps

- Add a unit test that `load_smd_data(...)` returns loaders whose first batch matches the shape and keys of the current builder.
- Add a parity test that `load_smd_data(...)` and `build_smd_dataset_bundle(...)` produce equal dataset lengths for the same input arguments.
- Extend the smoke-limit tests so the public API respects `max_train_windows`, `max_val_windows`, and `max_test_windows`.
- Re-run the existing loader tests and config-loading tests.

### Acceptance criteria

- Existing scripts continue to pass without import or behavior changes.
- Existing tests continue to pass unchanged.
- The new public wrapper can reproduce the existing SMD bundle with no second preprocessing codepath.
- A notebook can call the public wrapper in one cell and immediately access `data.loaders["train"]` or equivalent without manual config assembly.

## Phase 2 - Add a Named Download and Cleaning Surface Without Breaking the Existing Parser Path

### Phase summary

This phase makes download and cleaning visible parts of the public framework while preserving the existing parser semantics. The thesis objective is to expose the full `download -> parse -> clean/validate -> preprocess -> window/batch` path as a readable, explicit, reproducible sequence.

### File-level edits

The following files should be modified or added in this phase:

```text
src/data/download.py
src/data/cleaning.py
src/data/scalers.py
src/data/api.py
src/core/config.py
configs/data/smd.yaml
configs/data/smd_smoke.yaml
configs/data/smd_128.yaml
tests/test_smd_download_metadata.py
tests/test_data_cleaning_pipeline.py
tests/test_config_loading.py
```

### Explicit edit content

`src/data/download.py`

- Move the reusable SMD GitHub download logic out of notebook-only code into a packaged module.
- Expose a function such as `download_smd_dataset(...)` that:
  - checks for required directories;
  - optionally skips download if the dataset is already present;
  - downloads the same canonical SMD tree used by the current notebook templates.
- The implementation must preserve the current SMD directory assumptions used by `SMDDatasetParser`.

`src/data/cleaning.py`

- Introduce a named cleaning abstraction such as `SequenceCleaningPipeline`.
- The initial version must be conservative and limited to operations that already exist implicitly:
  - contract validation;
  - split-integrity checks;
  - optional metadata annotation of preprocessing state.
- Do not add speculative or dataset-specific cleaning heuristics in this phase.

`src/data/scalers.py`

- Add `fit_transform_sequences(...)` only if it wraps the current `fit(...)` plus `transform_sequences(...)` behavior without changing stored scaler state semantics.
- Preserve `state_dict()` and `load_state_dict()` exactly.

`src/data/api.py`

- Extend the public wrapper to accept `download=True` and optional cleaning controls.
- Ensure the public wrapper still delegates to the canonical builder path.

`src/core/config.py`

- Add optional data-config keys for:
  - `download`
  - `skip_existing_download`
  - cleaning toggles
  - cleaning metadata flags
- All new keys must be optional and default to the current behavior.

### Interface and contract definitions

- Dataset interface:
  - `download_smd_dataset(...)` becomes the public raw-materialization helper.
- Encoder interface:
  - unchanged.
- Model interface:
  - unchanged.
- Task interface:
  - unchanged.
- Training engine:
  - unchanged.

### Design pattern application

- Composition over inheritance:
  - `SequenceCleaningPipeline` must compose validators and optional metadata annotators.
- Adapter pattern for encoders:
  - not yet active.
- Strategy pattern for tasks:
  - unchanged.
- Registry or factory:
  - no new builder registry is introduced. The public API remains a wrapper over the existing builder.

### Risk mitigation

- Prototype redundancy and fusion collapse:
  - no model-path edits in this phase.
- Adaptation contamination and projector drift:
  - no online-model edits in this phase.
- Evaluation metric inflation:
  - do not change labels, score construction, or evaluator semantics in the cleaning layer.
- Download-path drift:
  - the downloader must materialize the exact directory layout expected by `SMDDatasetParser`.
- Codepath duplication:
  - the notebook downloader logic must be refactored into the packaged helper rather than copied.

### Test plan and validation steps

- Add a metadata-only downloader test that verifies required path planning without performing a full network download in unit tests.
- Add a cleaning-pipeline test that verifies no tensor shape, label alignment, or metadata keys are broken.
- Extend config-loading tests so new data-config keys are optional and default-safe.

### Acceptance criteria

- `download=True` is optional and does not change behavior when omitted.
- The packaged downloader yields the same directory layout expected by the current parser.
- The cleaning surface is visible in the codebase as a named module but does not change current parser outputs when left at default settings.
- Existing local datasets continue to parse without requiring any config migration.

## Phase 3 - Add Backward-Compatible Model Adapters and Baseline Handoff Surfaces

### Phase summary

This phase creates a formal adapter layer for pretrained backbones and a clearer handoff surface for classical baselines, while preserving the existing PyTorch training engine. The thesis objective is to let notebook and research users reuse the common data contract across MOMENT, the current thesis models, and baseline feature-based workflows without disturbing current train and evaluate scripts.

### File-level edits

The following files should be modified or added in this phase:

```text
src/adapters/moment.py
src/adapters/base.py
src/adapters/__init__.py
src/data/api.py
src/core/config.py
notebooks/smd_colab_window_preprocessing_template.ipynb
tests/test_moment_adapter_shapes.py
tests/test_public_baseline_handoff.py
```

### Explicit edit content

`src/adapters/base.py`

- Add a small adapter protocol for external encoder or foundation-model handoffs.
- The adapter interface must remain batch-contract-centered and should expose methods such as:
  - `prepare_batch(batch) -> dict`
  - `forward_prepared(prepared_batch) -> Any`
  - `postprocess_outputs(model_outputs) -> dict[str, Any]`

`src/adapters/moment.py`

- Move the notebook MOMENT preparation logic into a reusable module.
- The module must:
  - accept `[B, L, D]` input batches from the current repository contract;
  - transpose and pad as needed for MOMENT;
  - preserve metadata and labels in the returned bundle;
  - expose an embedding helper that returns embeddings plus metadata.
- The adapter must not replace the thesis model contract. It is a sidecar adapter layer.

`src/data/api.py`

- Add helper functions for baseline handoff, such as:
  - `point_labels_to_window_labels(...)`
  - `flatten_windows_for_baseline(...)`
- These helpers must preserve the current notebook semantics but move them into an importable module.

`src/core/config.py`

- Add optional adapter-related experiment settings only if they default to inactive and do not affect current training experiments.

### Interface and contract definitions

- Dataset interface:
  - unchanged from Phase 2.
- Encoder interface:
  - external encoders are integrated only through adapters.
- Model interface:
  - current `BaseModel` remains the engine-facing PyTorch contract.
- Task interface:
  - unchanged.
- Training engine:
  - unchanged in this phase; classical baseline support remains notebook-facing rather than engine-facing.

### Design pattern application

- Composition over inheritance:
  - the MOMENT adapter composes the current batch contract and the external model surface.
- Adapter pattern for encoders:
  - this phase formally introduces it.
- Strategy pattern for tasks:
  - still unchanged for the current engine.
- Registry or factory:
  - adapters may use a small local registry only if it does not create another competing model registry.

### Risk mitigation

- Prototype redundancy:
  - the adapter must not alter thesis-model internal prototype logic.
- Fusion collapse:
  - the adapter must not change multitask fusion coefficients or schedules.
- Adaptation contamination:
  - the adapter must not alter online stream or online batch semantics.
- Projector drift:
  - the adapter must not change `online_adaptation` checkpoint semantics.
- Evaluation metric inflation:
  - classical-baseline helper utilities must not silently relabel windows beyond the explicit current notebook rule.
- Backward-compatibility breakage:
  - notebook-local helpers may be preserved as thin wrappers during the transition so old notebooks still run.

### Test plan and validation steps

- Add a shape test for MOMENT input preparation from `[B, L, D]` to the expected padded format.
- Add a test for embedding-bundle metadata length parity.
- Add a baseline-handoff test for `flatten_windows_for_baseline(...)` and `point_labels_to_window_labels(...)`.
- Re-run the existing model-shape, one-train-step, and checkpoint-roundtrip tests to ensure no engine regression.

### Acceptance criteria

- The MOMENT adapter is importable from `src/` and reproduces the current notebook transformation logic.
- The notebook can import the adapter instead of redefining it locally.
- The current train and evaluate scripts remain unchanged and fully operational.
- Classical-baseline handoff utilities are available from `src/` without forcing classical baselines into the current PyTorch trainer.

## Phase 4 - Add Optional Kaggle Checkpoint Mirroring as a Post-Save Sink

### Phase summary

This phase introduces optional Kaggle-backed remote persistence for checkpoints and run artifacts without replacing the current local checkpoint flow. The thesis objective is reproducibility and weight survivability for long-running experiments, while ensuring that the repository remains safe for users who do not configure Kaggle credentials.

### File-level edits

The following files should be modified or added in this phase:

```text
src/engine/checkpoint.py
src/engine/logger.py
src/engine/artifact_sinks.py
src/core/config.py
configs/experiment/baseline/smd__thesis_multitask__multitask__w100__seed7__default.yaml
configs/experiment/baseline/smd__online_adaptation__online-adaptation__w100__seed7__default.yaml
requirements.txt
environment.yml
tests/test_kaggle_config_validation.py
tests/test_artifact_sink_selection.py
tests/test_checkpoint_roundtrip.py
```

### Explicit edit content

`src/engine/artifact_sinks.py`

- Add a small artifact-sink layer that is explicitly post-save and additive.
- Define one local invariant:
  - the checkpoint must always be written successfully to local disk before any remote mirroring begins.
- Define sink interfaces such as:
  - `save_file(path, metadata) -> None`
  - `save_directory(path, metadata) -> None`
- Implement:
  - a no-op sink for default local-only runs;
  - a Weights & Biases sink that delegates to the current `ExperimentLogger`;
  - a Kaggle sink that uses the officially documented Kaggle interfaces.

`src/engine/checkpoint.py`

- Keep the local `save_checkpoint(...)` path unchanged.
- After a successful local save, optionally invoke configured artifact sinks.
- The checkpoint manager must not depend on remote persistence for success.

`src/engine/logger.py`

- Preserve current Weights & Biases behavior.
- If a shared sink abstraction is introduced, adapt the logger so the current W&B logic becomes one sink implementation rather than a hidden special case.

`src/core/config.py`

- Add optional logging or persistence keys such as:
  - `artifact_backend`
  - `kaggle_dataset_handle`
  - `kaggle_version_notes_template`
  - `mirror_best_checkpoint_to_kaggle`
  - `mirror_output_dir_to_kaggle`
- All new keys must default to disabled and must preserve current behavior when omitted.
- Validate that Kaggle-specific fields are present only when the selected backend requires them.

`requirements.txt` and `environment.yml`

- Add Kaggle dependencies only after selecting one official interface:
  - `kaggle`
  - or `kagglehub`
- The selection must be documented in comments or environment notes so users can understand the credential expectation.

### Interface and contract definitions

- Dataset interface:
  - unchanged.
- Encoder interface:
  - unchanged.
- Model interface:
  - unchanged.
- Task interface:
  - unchanged.
- Training engine:
  - engine still consumes model stage methods exactly as before.
  - checkpoint persistence becomes `local-save-first, optional-remote-mirror-second`.

### Design pattern application

- Composition over inheritance:
  - artifact sinks compose around the existing checkpoint manager.
- Adapter pattern for encoders:
  - unchanged in this phase.
- Strategy pattern for tasks:
  - unchanged in this phase.
- Registry or factory:
  - a small sink factory may be introduced for artifact backends if it does not collide with model or dataset registries.

### Risk mitigation

- Prototype redundancy and fusion collapse:
  - no model-path edits are allowed in this phase.
- Adaptation contamination and projector drift:
  - no online-model semantics are changed.
- Evaluation metric inflation:
  - evaluation code remains untouched.
- Remote-save fragility:
  - a Kaggle upload failure must not invalidate a successful local checkpoint save.
- Credential risk:
  - Kaggle upload must remain fully opt-in and must fail with a clear configuration error when credentials are absent.
- Artifact duplication:
  - the best-checkpoint mirror path must be deterministic and must not silently upload every intermediate checkpoint unless explicitly configured.

### Test plan and validation steps

- Add config-validation tests for disabled and enabled Kaggle modes.
- Add sink-selection tests that verify:
  - default local-only mode;
  - W&B-only mode;
  - Kaggle-only mode;
  - combined mirroring policies if supported.
- Extend checkpoint-roundtrip tests to ensure local checkpoint saving still works when remote sinks are disabled.
- Add isolated tests for remote-sink invocation with mocks so no live Kaggle upload is required in unit tests.

### Acceptance criteria

- Existing local checkpoint saving still works without any Kaggle configuration.
- Existing W&B logging still works without behavior change.
- Kaggle mirroring is disabled by default.
- When enabled and configured, Kaggle mirroring occurs only after a successful local save.
- A failed Kaggle upload does not delete or invalidate the local best checkpoint.

## Phase 5 - Documentation, Migration Wrappers, and End-to-End Validation

### Phase summary

This phase closes the migration by documenting the new public surfaces, preserving compatibility wrappers, and validating that notebook users, script users, and long-run experiment users all have one readable, reproducible path. The thesis objective is to make the framework easier to use without requiring hidden engineering knowledge.

### File-level edits

The following files should be modified or added in this phase:

```text
documents/design/design_starter.md
documents/design/long_term_codebase_roadmap.md
notebooks/smd_colab_window_preprocessing_template.ipynb
notebooks/time_series_loading_template.ipynb
tests/test_public_data_api.py
tests/test_moment_adapter_shapes.py
tests/test_checkpoint_roundtrip.py
tests/test_registry.py
tests/test_config_loading.py
scripts/train.py
scripts/evaluate.py
scripts/run_ablation.py
scripts/run_online_adaptation.py
```

### Explicit edit content

`documents/design/design_starter.md`

- Update the documented tree only if new stable modules such as `src/data/api.py`, `src/data/download.py`, `src/data/cleaning.py`, or `src/adapters/moment.py` are accepted into the permanent structure.
- Document the backward-compatible public surfaces explicitly.

`notebooks/*.ipynb`

- Present the notebook-friendly public imports first.
- Keep thin compatibility wrappers or explanatory cells for users who still want to inspect the lower layers.

`scripts/*.py`

- Keep current command-line behavior unchanged.
- Only update inline comments if they need to mention the new public layer or optional Kaggle mirroring.

### Interface and contract definitions

- Dataset interface:
  - public wrapper and registry-facing builder must both be documented.
- Encoder interface:
  - adapter entrypoints must be documented.
- Model interface:
  - the PyTorch `BaseModel` contract remains the active engine-facing contract.
- Task interface:
  - no breaking change.
- Training engine:
  - document local-save-first semantics for checkpoints and optional remote sinks.

### Design pattern application

- Composition over inheritance:
  - document the final accepted wrapper structure.
- Adapter pattern for encoders:
  - document MOMENT integration as the canonical example.
- Strategy pattern for tasks:
  - document that current task behavior remains model-owned in active paths.
- Registry or factory:
  - document the continued script-facing registry role and the non-competing role of any helper factories.

### Risk mitigation

- Prototype redundancy:
  - verify multitask ablation configs still run unchanged.
- Fusion collapse:
  - verify multitask schedule comments and config fields remain intact.
- Adaptation contamination:
  - verify online adaptation still consumes the same scaled test stream.
- Projector drift:
  - verify online checkpointing remains independent of public data wrappers.
- Evaluation metric inflation:
  - verify evaluation reports and thresholds remain unchanged after wrapper introduction.
- Documentation drift:
  - ensure notebooks and design docs refer to the same accepted public entrypoints.

### Test plan and validation steps

- Run the existing focused loader, model-shape, one-train-step, checkpoint-roundtrip, and registry tests.
- Add one notebook-equivalent integration test that calls the new public API, takes one batch, and passes it through either:
  - the reconstruction baseline;
  - or the MOMENT adapter preparation path.
- Add one integration test for optional artifact-sink selection with Kaggle disabled by default.

### Acceptance criteria

- The framework exposes a readable notebook-facing import path without breaking current script execution.
- Existing YAML-driven experiments continue to run with no required config migration.
- Existing model files remain self-contained and unchanged in contract semantics.
- Public documentation, notebooks, and tests all describe the same active public surfaces.
- Backward compatibility is preserved by keeping old builder names and script entrypoints valid through the end of the migration.

## Final Implementation Order

The implementation order must be:

1. Phase 1: public API wrapper over the current SMD bundle
2. Phase 2: packaged downloader and named cleaning surface
3. Phase 3: MOMENT adapter and baseline handoff helpers
4. Phase 4: optional Kaggle checkpoint mirroring as a post-save sink
5. Phase 5: documentation cleanup, compatibility wrappers, and end-to-end validation

This order is required because it maximizes backward compatibility. It introduces new framework conveniences as additive layers around the current repository rather than by replacing the current repository’s working internals.
