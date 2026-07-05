# Two-Stage KMeans Memory Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the current `thesis_multitask` codepath with the approved two-stage k-means memory design while preserving the existing runner, config, and evaluation contracts.

**Architecture:** Keep the current model-centric structure intact and change the smallest set of files that control stage semantics, memory bootstrap, and discrete-query runtime behavior. The implementation should preserve the current registry and trainer surfaces, move memory initialization from heuristic selection to deterministic k-means, and keep Stage A / Stage B orchestration in the dedicated two-stage runner.

**Tech Stack:** Python, PyTorch, PyYAML, `pytest`.

---

## Current State

- The active SSOT is `documents/design/offline_pretraining_two_stage_kmeans_memory_design.md`.
- The codebase already has a two-stage orchestration path in `scripts/run_two_stage_offline_pretraining.py`.
- The model already supports stage-aware behavior in `src/models/thesis_multitask.py`, `src/models/thesis_multitask_setup_mixin.py`, and `src/models/thesis_multitask_state_mixin.py`.
- Memory initialization currently collects token pools from the training split, but the centroid selection still uses a covering-vector heuristic rather than k-means.
- The current discrete runtime path still carries legacy Gumbel-style machinery for compatibility, even though the approved design prefers `cosine_topk`.

## Design Options

- Option A: Implement k-means bootstrap inside `src/models/thesis_multitask_state_mixin.py` and keep the runner unchanged except for the existing Stage B initialization checkpoint flow. This is the recommended path because it minimizes surface area and keeps the model file as the owner of memory semantics.
- Option B: Extract a reusable k-means helper module and call it from the state mixin. This is cleaner if the clustering helper grows, but it adds another module and another interface to maintain.
- Option C: Refactor the discrete runtime path and memory bootstrap together in one pass. This is broader, but it increases risk because it mixes the approved memory redesign with legacy runtime cleanup.

Recommended choice: Option A first, then a narrow cleanup pass for Option C only where the SSOT requires it.

## Risk and Mitigation

- Risk: continuous and discrete prototype branches stay redundant after the redesign. Mitigation: keep explicit branch-level tests and log the finalized memory source labels.
- Risk: fusion collapses to one branch. Mitigation: preserve the existing fusion metrics and add stage-specific assertions for trainable surfaces.
- Risk: adaptation or bootstrap logic leaks into Stage B. Mitigation: keep the Stage B freeze contract in the setup mixin and validate checkpoint metadata.
- Risk: projector or discrete runtime cleanup changes legacy checkpoints. Mitigation: preserve backward-compatible loading paths while removing unused runtime state only when tests confirm no breakage.
- Risk: evaluation metrics look improved because the wrong checkpoint metadata is carried forward. Mitigation: keep the two-stage runner’s manifest and initialization checkpoint assertions under test.

## Open Questions

- Should the k-means routine be implemented as a private helper inside `src/models/thesis_multitask_state_mixin.py`, or extracted into a small local helper module once the first pass is stable?
- Do we want to remove the legacy Gumbel-only discrete path in the same change set, or keep it behind the current compatibility guard until after the k-means bootstrap is validated?

## Plan

### Task 1: Lock the new bootstrap contract with failing tests

**Files:**
- Modify: `tests/test_multitask_memory_initialization.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_kmeans_bootstrap_produces_exact_prototype_counts() -> None:
    model = _build_initialization_model()
    # Build a token pool with two obvious clusters per branch.
    # Assert that initialization fills:
    # - continuous_prototype_bank with exactly continuous_num_prototypes rows
    # - discrete_codebook with exactly discrete_codebook_size rows
    # and that the buffers are updated from clustered centroids, not raw tokens.

def test_two_stage_runner_still_writes_stage_b_init_checkpoint() -> None:
    # Assert that the two-stage runner still materializes stage_b_init.pt
    # and that the checkpoint extra_state keeps memory_initialized=True.
```

- [ ] **Step 2: Run the targeted tests and confirm they fail for the right reason**

Run:
`pytest tests/test_multitask_memory_initialization.py tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py -v`

Expected: the new assertions fail because the current heuristic bootstrap does not yet use k-means.

- [ ] **Step 3: Keep the assertions narrowly scoped**

Use small synthetic token pools with obvious cluster centers so the test proves the bootstrap mechanism, not an incidental numeric tolerance.

---

### Task 2: Replace heuristic memory bootstrap with deterministic k-means

**Files:**
- Modify: `src/models/thesis_multitask_state_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py` if any stage-gating or runtime-state fields need to be simplified for the new design
- Modify: `src/models/thesis_multitask_components.py` only if a small shared helper or config constant is needed

- [ ] **Step 1: Write the failing implementation-level test or keep the Task 1 failures as the driver**

Focus the test target on the private bootstrap helpers:

```python
def test_initialize_memory_buffers_from_token_pool_uses_kmeans_centroids() -> None:
    # Verify that the selected centroids come from a clustering pass over
    # normalized tokens, not from the old covering-vector heuristic.
```

- [ ] **Step 2: Run the focused test and confirm the current helper still fails**

Run:
`pytest tests/test_multitask_memory_initialization.py -v`

Expected: failure or mismatch around centroid selection and cluster assignment.

- [ ] **Step 3: Implement the minimal k-means bootstrap path**

Implement the clustering logic inside `src/models/thesis_multitask_state_mixin.py` so the model still owns memory semantics. The helper should:

```python
def _run_kmeans(
    self,
    tokens: torch.Tensor,
    k: int,
    *,
    num_iterations: int,
) -> torch.Tensor:
    # 1. normalize input tokens
    # 2. seed centers deterministically
    # 3. assign tokens to nearest centers
    # 4. recompute centers
    # 5. normalize final centroids
```

Then update `_initialize_memory_buffers_from_token_pool(...)` so:

```python
continuous_centroids = self._run_kmeans(
    continuous_hidden_tokens,
    self.continuous_num_prototypes,
    num_iterations=...
)

for class_index, class_tokens in discrete_hidden_tokens_by_class.items():
    class_centroids = self._run_kmeans(class_tokens, 5, num_iterations=...)
```

Keep the existing buffer copies and checkpoint state contract unchanged.

- [ ] **Step 4: Run the targeted tests and confirm the bootstrap now passes**

Run:
`pytest tests/test_multitask_memory_initialization.py -v`

Expected: the initialization tests pass and the buffers are filled with the new k-means centroids.

- [ ] **Step 5: Commit the memory-bootstrap change**

Use a small commit that only covers the bootstrap logic and its tests.

---

### Task 3: Simplify the discrete runtime surface for `cosine_topk`

**Files:**
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `src/models/thesis_multitask_routing_mixin.py`
- Modify: `src/models/thesis_multitask_state_mixin.py` if the checkpoint state should stop reporting unused Gumbel-only fields
- Modify: `tests/test_thesis_multitask_classification_path_toggle.py`
- Modify: `tests/test_multitask_memory_updates.py` if memory update assertions need to reflect the new runtime path

- [ ] **Step 1: Write the failing test for the intended cosine-topk surface**

```python
def test_cosine_topk_stage_does_not_require_gumbel_assignment() -> None:
    model = ThesisMultitaskModel(
        ...
        discrete_query_mode="cosine_topk",
        training_phase="stage_b_fusion_finetuning",
    )
    assert model.discrete_assignment is None or not any(
        p.requires_grad for p in model.discrete_assignment.parameters()
    )
```

- [ ] **Step 2: Run the focused test and confirm the current surface still exposes legacy assignment machinery**

Run:
`pytest tests/test_thesis_multitask_classification_path_toggle.py -v`

Expected: the test fails until the cosine-topk branch no longer depends on the legacy assignment path.

- [ ] **Step 3: Narrow the runtime branch to the approved path**

Update the discrete branch so `cosine_topk` is the first-class runtime path and Gumbel-only state is not treated as required for the approved rerun. Keep backward-compatible loading only if the existing checkpoint tests require it.

- [ ] **Step 4: Re-run the discrete runtime tests**

Run:
`pytest tests/test_thesis_multitask_classification_path_toggle.py tests/test_multitask_memory_updates.py -v`

Expected: the cosine-topk path behaves correctly in Stage B and the stage-freeze assertions still pass.

---

### Task 4: Verify the two-stage runner and checkpoint handoff end to end

**Files:**
- Modify: `scripts/run_two_stage_offline_pretraining.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`

- [ ] **Step 1: Add or tighten the runner assertions**

Verify that the runner still:

```python
manifest["training_stages"] == [
    {"stage_name": "stage_a_multitask_pretraining", ...},
    {"stage_name": "stage_b_fusion_finetuning", ...},
]
```

and that `stage_b_init.pt` exists after Stage A finishes.

- [ ] **Step 2: Run the runner tests**

Run:
`pytest tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py -v`

Expected: the manifest, generated configs, and initialization checkpoint all match the approved two-stage budget.

- [ ] **Step 3: Confirm Stage A to Stage B handoff still loads the initialized memory state**

Check that `memory_initialized` remains true in the Stage B initialization payload and that the stage labels remain stable.

---

### Task 5: Final validation on the narrow smoke set

**Files:**
- No new files expected unless a test-only fixture needs a small adjustment

- [ ] **Step 1: Run the focused pytest slice**

Run:
`pytest tests/test_multitask_memory_initialization.py tests/test_thesis_multitask_classification_path_toggle.py tests/test_offline_pretraining_two_stage_runner.py tests/test_offline_pretraining_two_stage_config_loading.py -v`

Expected: all targeted tests pass.

- [ ] **Step 2: Run the existing two-stage smoke if available**

Run the smallest existing smoke command for the two-stage exp4 path.

Expected: the orchestration completes and still writes the manifest and execution report.

- [ ] **Step 3: Review the final diff for scope creep**

Confirm that the change set stayed inside the current thesis surfaces and did not pull in unrelated refactors.

## Self-Review

- Spec coverage: the plan covers the active two-stage contracts, the current heuristic bootstrap gap, the discrete runtime cleanup, and the two-stage runner handoff.
- Placeholder scan: the plan avoids undefined tasks and keeps the file paths explicit.
- Type consistency: the task names, file paths, and test names are aligned across the plan.

