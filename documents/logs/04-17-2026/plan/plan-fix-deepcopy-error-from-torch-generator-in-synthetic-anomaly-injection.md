---
date: 2026-04-17 16:44:37 +0700
planner: TheMetaSetter
git_commit: ce9c92c9c052f39818c1186016886d1c9d0b12dd
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan to fix the deepcopy error caused by torch.Generator inside synthetic anomaly injection"
tags: [plan, time-series, anomaly-detection, online-adaptation, deepcopy, synthetic-anomaly-injection]
status: complete
last_updated: 2026-04-17
last_updated_by: TheMetaSetter
source_research: documents/logs/04-17-2026/research/research-different-window-sizes-between-pretraining-and-online-adaptation.md
---

# Plan: Implementation plan to fix the deepcopy error caused by `torch.Generator` inside synthetic anomaly injection

## Current State

- The active offline thesis model remains `src/models/thesis_multitask.py`. This file owns the encoder, continuous and discrete prototype branches, fusion, synthetic anomaly injection, and stage-specific logic. This ownership boundary is consistent with the repository rule that one model should remain readable inside one file.
- The active online adaptation path remains `src/models/online_adaptation.py`, `src/data/stream.py`, and `src/engine/online_loop.py`. The online model constructs a frozen reference encoder and an online encoder by wrapping a loaded `ThesisMultitaskModel`.
- The immediate blocking failure occurs in `src/models/online_adaptation.py` when `ThesisMultitaskEncoderAdapter` executes `copy.deepcopy(thesis_model)`. That call traverses the full offline model object graph.
- The root cause is located in `src/data/augment.py`. `SyntheticAnomalyInjector` stores a live `torch.Generator` in `self._rng` after `reset_rng()`. The deterministic validation injector created in `src/models/thesis_multitask.py` therefore makes the offline model instance non-pickleable under Python deep copy semantics.
- The batch, encoder, and model-output contracts are already stable and should not be altered by this fix. The relevant contracts remain:
  - offline and online batches use `x`, `point_labels`, `mask`, `timestamps`, and `meta`, with online batches adding `view_a` and `view_b`;
  - the encoder contract remains thesis-facing `hidden: Tensor[B, L, H]` with optional pooled output;
  - the model-output contract remains `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux`.
- The failure is already visible in the existing online regression surface. `tests/test_online_adaptation_step.py` currently fails before the first online step because `OnlineAdaptationModel(...)` cannot be constructed from a multitask checkpoint.

## Design Options

### Option A: Make `SyntheticAnomalyInjector` deepcopy-safe while preserving the current model structure

This option keeps the current ownership boundaries intact and makes the augmentation object safe to deep-copy. The injector would continue to store configuration fields explicitly, but the live `torch.Generator` would be excluded from object-state serialization and rebuilt deterministically after copy.

This option preserves the present design:

- `src/models/thesis_multitask.py` remains the single owning file for offline augmentation logic;
- `src/models/online_adaptation.py` continues to reuse the offline thesis model through a small adapter boundary;
- no public batch or output contracts need to change;
- no experiment YAML files need to change.

This is the smallest and safest vertical slice.

### Option B: Sanitize or rebuild the copied thesis model inside `ThesisMultitaskEncoderAdapter`

This option keeps `SyntheticAnomalyInjector` unchanged and instead modifies `src/models/online_adaptation.py` so the adapter does not deep-copy live injector state. For example, the adapter could deep-copy only selected submodules, or it could null out and reconstruct injectors before cloning.

This option can solve the immediate error, but it is less desirable because it spreads knowledge of offline augmentation internals into the online model file. That weakens separation of concerns and makes the online file responsible for implementation details that belong to the offline model and its augmentation helper.

### Option C: Narrow the online adapter to copy only encoder and scoring submodules

This option redesigns `ThesisMultitaskEncoderAdapter` to avoid holding an entire copied `ThesisMultitaskModel`. It would instead reconstruct only the encoder, prototype banks, fusion parameters, reconstruction head, and classification head needed for online scoring.

This option is architecturally clean in the long term, but it is not the correct first response to this bug. It is a wider refactor that increases risk, expands the write surface, and adds more opportunities for silent divergence between offline and online scoring behavior.

## Selected Approach

The recommended approach is **Option A**.

The repository should fix the failure at the true ownership boundary by making `SyntheticAnomalyInjector` deepcopy-safe. This aligns best with the current codebase principles:

- single responsibility: the injector owns its own random-state behavior;
- stable interfaces: the online adapter should not need to know how synthetic augmentation stores its internal RNG;
- composition over inheritance: the online model continues to compose around the offline thesis model rather than partially re-implementing it;
- minimal vertical slice: restore the existing online runtime without broadening the refactor unnecessarily.

## Risk And Mitigation

- Risk: the fix may remove deterministic validation behavior by reinitializing the RNG incorrectly.
  Mitigation: store only the deterministic seed as durable state and rebuild `self._rng` through `reset_rng()` after copy, deserialization, or explicit state restoration.
- Risk: the fix may introduce hidden differences between an original injector and a copied injector.
  Mitigation: add targeted tests that compare deterministic draws before and after `copy.deepcopy(...)` when `deterministic_seed` is set, and confirm that non-deterministic injectors remain valid and callable.
- Risk: the fix may solve deep copy but break checkpoint loading or future pickling flows.
  Mitigation: implement the state-handling method at the helper level using explicit object-state hooks rather than relying on accidental Python behavior.
- Risk: changing the online adapter instead of the injector would leak offline augmentation knowledge into the online file.
  Mitigation: keep `src/models/online_adaptation.py` unchanged unless the helper-level fix proves insufficient.
- Risk: a larger adapter refactor could accidentally change online scoring outputs.
  Mitigation: defer any narrowing or architectural cleanup until after the minimal regression fix has restored the accepted online path and tests are green.

## Open Questions

- Should the injector preserve exact generator progression across deep copies, or is it sufficient to preserve deterministic reproducibility by re-seeding from `deterministic_seed`? For this repository, the correct answer is the second one. The repository already treats deterministic validation as seed-driven rather than generator-snapshot-driven behavior.
- Should `SyntheticAnomalyInjector` expose explicit serialization helpers such as `state_dict()` and `load_state_dict()`? This is not required to fix the current failure, but the helper-level state contract may become useful later if online or offline experiment state needs to materialize augmentation state more explicitly.
- Should the online adapter later be narrowed to hold only the necessary scoring components instead of a full copied thesis model? This is a valid later cleanup task, but it should remain outside the scope of the present bug fix.

## Implementation Plan

### 1. Codify the failure as a targeted regression surface

Before changing the implementation, preserve the failure mode explicitly in tests.

Modify or extend the following tests:

- `tests/test_online_adaptation_step.py`
- `tests/test_synthetic_anomaly_injection.py`

Add one focused regression test in `tests/test_synthetic_anomaly_injection.py`:

- `test_synthetic_anomaly_injector_supports_deepcopy_with_deterministic_seed()`

This test should:

- construct a `SyntheticAnomalyInjector(deterministic_seed=7)`;
- call `copy.deepcopy(injector)`;
- assert that the copy succeeds;
- assert that the copied injector retains `deterministic_seed`;
- call augmentation or the internal random helpers on both instances after `reset_rng()` and verify equivalent deterministic behavior.

Add one online-construction regression test, either by extending `tests/test_online_adaptation_step.py` or by adding a new test function near the existing checkpoint construction path:

- `test_online_adaptation_model_constructs_from_multitask_checkpoint()`

This test should:

- build the same multitask checkpoint fixture currently used in the online-step test;
- instantiate `OnlineAdaptationModel(...)`;
- assert that `reference_encoder`, `online_encoder`, and `projector` are created successfully.

The purpose is to isolate model construction from later optimization logic so the failure surface remains obvious if it regresses.

### 2. Make `SyntheticAnomalyInjector` responsible for its own copy-safe runtime state

Modify `src/data/augment.py`.

Keep the current public constructor unchanged:

```python
SyntheticAnomalyInjector(
    anomaly_probability: float = 0.5,
    min_segment_fraction: float = 0.1,
    max_segment_fraction: float = 0.2,
    spike_scale: float = 3.0,
    anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
    deterministic_seed: int | None = None,
)
```

Add one explicit object-state hook using standard Python copy and pickle semantics. The simplest acceptable form is:

- `__getstate__(self) -> dict[str, Any]`
- `__setstate__(self, state: dict[str, Any]) -> None`

The implementation should follow these rules:

- `__getstate__` must exclude the live `torch.Generator` object from serialized state;
- `__setstate__` must restore ordinary configuration fields and then call `reset_rng()` so deterministic injectors recreate a valid CPU generator;
- the helper must continue to treat `deterministic_seed is None` as the non-deterministic path with `self._rng = None`;
- the taxonomy registry and `anomaly_families` must remain valid after restore.

The important design decision is that the live generator is runtime-only state, not durable configuration state.

### 3. Preserve explicit randomness semantics after the helper-level change

Within `src/data/augment.py`, preserve the current semantics of:

- deterministic validation via `deterministic_seed`;
- runtime regeneration through `reset_rng()`;
- random tensor helper methods `_rand`, `_randn`, `_randint`, and `_randperm`.

Do not change:

- anomaly taxonomy names;
- augmentation metadata schema;
- batch augmentation contract;
- family dispatch mechanism.

This keeps the fix narrow and prevents unnecessary changes to the offline multitask path.

### 4. Keep `src/models/thesis_multitask.py` unchanged unless a contract issue appears

The current injector ownership in `src/models/thesis_multitask.py` is correct. The model file should continue to instantiate:

- `self.synthetic_anomaly_injector`
- `self.synthetic_validation_injector`

No constructor signature changes are required in the thesis model for this bug fix.

Only change this file if the final helper-level fix requires a tiny readability comment explaining why the injector now excludes live RNG state from deep-copy serialization. If no such comment is necessary, leave the file untouched.

### 5. Keep `src/models/online_adaptation.py` unchanged for the first fix pass

Do not begin by rewriting `ThesisMultitaskEncoderAdapter`.

The current online adapter should remain:

- responsible for loading the multitask checkpoint;
- responsible for constructing frozen reference and online encoder copies;
- unaware of augmentation-helper internals.

After the helper-level fix in `src/data/augment.py`, rerun the online construction test and the existing online step test. If both pass, keep `src/models/online_adaptation.py` unchanged.

Only if the deep-copy error persists after the helper-level fix should a second pass be considered. That second pass would be a separate design decision, not part of the initial implementation.

### 6. Preserve the stable contracts explicitly

This fix must not alter:

- `src/core/contracts.py`
- the offline batch contract
- the online batch contract
- the thesis-facing hidden-state contract
- the model output contract

The reason is simple: this bug is about model object copyability, not tensor schema design.

State this explicitly in code review and verification notes so later readers do not infer that the bug fix changed any training or evaluation interfaces.

### 7. Validation procedure

Run the smallest meaningful validation stack in this order:

1. `pytest -q tests/test_synthetic_anomaly_injection.py`
2. `pytest -q tests/test_online_reference_checkpoint.py tests/test_online_adaptation_step.py`
3. If those pass, run a broader nearby regression set:
   `pytest -q tests/test_online_state_roundtrip.py tests/test_online_stream.py tests/test_multitask_validation_alignment.py`

Validation success criteria:

- the new deepcopy regression test passes;
- `OnlineAdaptationModel(...)` can be constructed from a multitask checkpoint;
- the online-step test reaches at least one forward and backward pass;
- no existing synthetic anomaly injection metadata behavior regresses.

### 8. Optional follow-on cleanup after the bug fix

If the minimal helper-level fix succeeds, record but defer the following cleanup ideas:

- add an explicit `state_dict()` and `load_state_dict()` surface to `SyntheticAnomalyInjector`;
- add a tiny comment in `src/models/online_adaptation.py` noting that the adapter depends on the thesis model being deep-copy-safe;
- consider a later architectural cleanup in which the online adapter narrows its copied surface to only the components required for scoring and alignment.

These are not part of the required fix. They should not delay the accepted regression repair.

## File-Level Change Summary

### Required modifications

- `src/data/augment.py`
  - add explicit copy-safe object-state handling for `SyntheticAnomalyInjector`
- `tests/test_synthetic_anomaly_injection.py`
  - add deepcopy regression coverage
- `tests/test_online_adaptation_step.py`
  - add or refine construction-level regression coverage

### Files expected to remain unchanged in the first fix pass

- `src/models/thesis_multitask.py`
- `src/models/online_adaptation.py`
- `src/core/contracts.py`
- `src/core/config.py`
- YAML experiment files under `configs/`

## Recommended Execution Order

1. Add the new injector deepcopy regression test.
2. Add the online model construction regression test.
3. Implement copy-safe state handling in `src/data/augment.py`.
4. Run the targeted test stack.
5. Only if the failure remains, open a second plan for adapter-level changes.

## Conclusion

The correct first implementation is a helper-level repair, not an online-model refactor. The repository should make `SyntheticAnomalyInjector` safe for deep copy by treating the live `torch.Generator` as runtime-only state and rebuilding it from `deterministic_seed` after restore. This is the smallest fix that respects current module ownership, preserves stable contracts, and unblocks the accepted online adaptation path without creating additional codepaths.
