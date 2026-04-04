# Long-Term Codebase Roadmap

## Summary

The long-term direction of this repository should still follow the same high-value principle from the older roadmap:

**first make the pipeline real, then make it reusable, then make it advanced.**

That logic remains correct, but the current repository is no longer at the beginning of that journey. The codebase already has:

- a fixed contract surface for offline batches, online batches, model outputs, and evaluation records
- an SMD-first offline vertical slice with a reconstruction baseline
- an offline thesis model with continuous and discrete branches, task-specific fusion, and RedLamp-default synthetic anomaly injection
- a conservative projector-first online adaptation slice
- YAML-driven experiments, checkpointing, JSONL metrics, evaluation artifacts, and Weights & Biases integration

So this document is not a reset to "Phase 0". It is a **translated long-term roadmap** that starts from the current repository state and sets the order for future work without reviving outdated structure plans or adding unnecessary codepaths.

This roadmap must also obey `codebase_preferences.md`:

- readability first
- one model per file
- least amount of codepaths
- explicit YAML control for ablations
- Weights & Biases as the experiment-history surface
- DVC for versioned derived synthetic-data artifacts when those artifacts are materialized
- RedLamp as the active default synthetic anomaly taxonomy, with CARLA retained as a mechanism reference

## Current starting point

The present repository should be treated as having the following foundations already in place.

### Contracts are already frozen enough to build on

The current repository already has stable contracts for:

- raw sequences
- offline windows
- online batches
- model outputs
- evaluation records

The active thesis-facing model-output contract is:

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

The active offline batch contract is:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The active online batch contract extends that with:

```python
"view_a"
"view_b"
```

These contracts should remain the thin waist of the framework. New datasets, drifts, and models should adapt to them rather than silently redefining them.

### The first real vertical slice is already present

The repository already supports:

```text
SMD files
-> parser / loader
-> sliding windows
-> reconstruction baseline
-> anomaly score
-> evaluation
```

That means the codebase has passed the point where speculative abstractions should dominate. Future design should remain anchored to working experiment paths.

### The thesis model and first online slice already exist

The offline thesis path is already more advanced than the older roadmap assumed:

- continuous branch
- discrete branch
- task-specific fusion
- reconstruction and classification heads
- RedLamp-default synthetic anomaly injection

The online path also exists already, but in its intentionally conservative first form:

- clean stream only
- frozen reference encoder
- projector-first adaptation
- explicit checkpoint state for stream progress and projector anchor

So the long-term roadmap below must not pretend these pieces are absent. Instead, it must define how to stabilize and extend them in the right order.

## Governing principles

### 1. Keep one active path per concern

When a path becomes accepted, it should be obvious which file, config family, and script own it.

Examples:

- one accepted baseline model file
- one accepted offline thesis model file
- one accepted online adaptation model file
- one accepted SMD data path

Alternative ideas should usually appear as:

- optional helper methods inside the owning file
- explicit YAML ablations
- deferred design notes

not as parallel hidden codepaths.

### 2. Prefer semantic translation over structural churn

Older plans often proposed many new folders such as dedicated `streams/`, `drift/`, `heads/`, or `adaptation/` trees. The current repository already has a simpler fixed architecture:

```text
configuration -> data -> model -> engine
```

That architecture should be preserved unless a strong readability reason appears. New capabilities should be integrated into the existing tree in a way that keeps ownership obvious.

### 3. Keep the thesis objective small by default

The default offline thesis starting objective should remain:

$$
\mathcal{L}_{\text{base}} =
\mathcal{L}_{\text{recon}} +
\lambda_{\text{cls}} \mathcal{L}_{\text{cls}}.
$$

Optional regularizers belong to the modular objective surface, but they should stay disabled by default until observed failure modes justify them. The correct place to turn them on first is ablation configs, not the default experiment family.

### 4. Generalize only after one stable end-to-end result

The repository should keep one strong paper-style path stable before expanding aggressively:

- one real dataset
- one accepted offline thesis result
- one accepted drifted streaming baseline
- one accepted conservative online adaptation result

This is still the correct anti-fragmentation rule for a bachelor thesis codebase.

## Long-term milestones

## Milestone A - Keep the foundation stable and inspectable

This milestone is mostly closed already, but it remains a permanent maintenance rule.

The repository must preserve:

- SMD-first executability
- readable baseline path
- fixed batch and output contracts
- full train and evaluate loops
- checkpoint round-tripping
- evaluation visualizations

The main work here is not new implementation. It is preventing regression while later capabilities are added.

**Acceptance rule**

If the thesis model, online adaptation, or drift modules are temporarily removed, the repository must still run a clean offline SMD baseline end to end.

## Milestone B - Consolidate the reusable data and stream architecture

This is the first major long-term area that is still incomplete.

The current repository already has:

- SMD parser
- offline window construction
- SMD-specific online stream

What is still missing is a clean, reusable streaming architecture across datasets and future drift scenarios. The target structure should remain semantic rather than folder-heavy:

- dataset-specific parsers stay under `src/data/datasets/`
- window construction remains centralized
- stream access becomes a first-class reusable interface
- drift injection and drift scenario definition become explicit data-layer modules

The desired mental model is:

```text
data source
-> stream wrapper
-> optional drift transformation
-> window construction
-> model
```

not:

```text
dataset-specific special cases inside model or engine code
```

### Near-term outcome for this milestone

- generalize the stream abstraction beyond the current SMD-only online slice
- add explicit scenario objects that define start, end, affected channels, and severity schedules
- keep drift logic outside model files
- preserve the current window contract while supporting both offline and online use

**Acceptance rule**

The same high-level streaming/evaluation script should be able to switch datasets and drift scenarios through config rather than code edits.

## Milestone C - Finish the offline thesis model as the main research path

The thesis model already exists, so this milestone is now about stabilization and evidence, not first implementation.

The long-term target is:

- clear branch behavior
- stable fusion behavior
- ablation-ready objective surface
- explicit diagnostics for branch collapse or dead-code usage
- synthetic anomaly injection that is visible, configurable, and reproducible

### What should remain true

- the owning model file stays `src/models/thesis_multitask.py`
- forward path, scoring path, and training logic stay in that same file
- regularizers remain explicitly config-controlled
- the default experiment path keeps the small objective

### What should improve over time

- stronger ablation evidence for continuous-only, discrete-only, and fused behavior
- better interpretation of when the discrete branch helps or collapses
- more reliable evaluation thresholds and score diagnostics
- clearer mapping from synthetic anomaly families to downstream classification usefulness

**Acceptance rule**

The repository should be able to answer, from YAML-driven experiments alone:

- does continuous help?
- does discrete help?
- does fusion help?
- when do optional regularizers help, and why?

## Milestone D - Add controlled drift as a first-class research surface

This is still one of the largest open items.

Synthetic anomaly injection already exists for offline auxiliary training, but that is not the same as controlled drift for stream research.

The drift system should eventually support a small, explicit initial family such as:

- mean drift
- variance drift
- trend drift
- sensor dropout
- correlation drift

The first drift system should stay deterministic and inspectable. Randomness can exist, but the design goal should be controlled reproducibility before maximal variety.

Each drift operator should make these questions explicit:

- when does drift start?
- when does it end?
- which channels are affected?
- how does severity evolve?

### Important separation

- synthetic anomaly injection is an offline training augmentation surface
- drift injection is a streaming-distribution-shift surface

They are related conceptually, but they should not collapse into one ambiguous mechanism.

**Acceptance rule**

A user should be able to render before-drift and after-drift visualizations and clearly see that the generated disturbance matches the configured scenario.

## Milestone E - Build a proper non-adaptive online baseline under drift

The current online slice already does projector-first adaptation on a clean stream. The next long-term step is not broader adaptation yet. It is to add a stronger **non-adaptive streaming baseline under drift**.

Why this matters:

- without that baseline, online adaptation gains are hard to interpret
- offline evaluation and online evaluation are different experimental objects
- drift robustness needs a fixed-model comparison point

The desired progression is:

1. clean streaming inference
2. drifted streaming inference with no adaptation
3. conservative online adaptation under the same scenarios

This ordering is more scientifically useful than immediately expanding adaptation mechanisms.

**Acceptance rule**

For at least one drift family on SMD, the repository should support direct comparison between:

- offline evaluation
- online clean-stream inference
- online drifted non-adaptive inference
- online conservative adaptation

## Milestone F - Extend online adaptation only conservatively

The current online direction is already the correct starting point:

- frozen reference encoder
- warm-started projector
- small trainable parameter group
- explicit stream state and checkpoint state

Long-term expansion should remain conservative:

1. stabilize projector-only adaptation
2. add better monitoring and reset logic
3. optionally unfreeze a small online encoder subset
4. only later test broader or geometry-aware methods such as NGD-style updates

This should remain explicitly separate from "full-network online finetuning", which is a later experimental option rather than a default target.

**Acceptance rule**

Every expansion of online adaptation should be justified against a smaller accepted slice that already works.

## Milestone G - Generalize benchmark coverage only after the SMD path is strong

The older roadmap was right that dataset expansion should come after one working benchmark path.

The long-term dataset order should still be:

- SMD first
- then other industrial or sensor benchmarks such as MSL, SMAP, SWaT
- then broader archives such as UCR only when the streaming and evaluation assumptions are compatible

The main requirement is not identical parser internals. It is identical external behavior:

- same contract surface
- same experiment script family
- same evaluation protocol class

**Acceptance rule**

A dataset change should mostly be a config change plus a dataset adapter file, not a trainer or model rewrite.

## Milestone H - Make reproducibility strong enough for thesis use

This is a permanent cross-cutting milestone rather than one isolated implementation phase.

The repository should treat reproducibility as a first-class runtime concern:

- resolved configs must be saved
- metric histories must be saved
- checkpoints must be versioned as artifacts
- evaluation outputs must be persisted
- W&B should log the run history and artifacts
- DVC should version materialized derived synthetic-data artifacts when such artifacts are written to disk

The current repository is already stronger here than the older roadmap assumed because W&B support exists now. The remaining long-term requirement is to keep that support complete and disciplined rather than optional in practice.

**Acceptance rule**

Two months later, a user should be able to reconstruct:

- which config was run
- which checkpoint was used
- which metrics were obtained
- which artifact set belongs to that run
- which synthetic-data or drifted-data version was used

## Milestone I - Keep tests focused on silent-failure risks

The long-term test surface should stay small but high value.

The most important risks to cover are:

- shape consistency
- checkpoint save/load
- config loading correctness
- synthetic anomaly metadata correctness
- stream order preservation
- overlap-aware evaluation correctness
- thresholding correctness
- no future leakage in online processing
- drift-scenario correctness once drift injection is added

The current repository already covers much of the offline and early online surface. Future tests should emphasize the new streaming and drift risks rather than only expanding generic unit coverage.

**Acceptance rule**

When new streaming or drift features are added, the first tests should target leakage, ordering, and checkpoint recovery before performance claims.

## Recommended implementation order from the current repository state

This is the preferred long-term order now, starting from the current codebase rather than from an empty framework.

1. Keep the SMD baseline and current thesis contracts stable.
2. Strengthen offline thesis evidence with simple-loss-default runs and clearer ablations.
3. Complete a reusable stream abstraction beyond the current SMD-only online slice.
4. Implement explicit drift scenarios and a small deterministic drift family.
5. Add non-adaptive online evaluation under drift.
6. Compare conservative projector-first adaptation against that baseline.
7. Expand benchmark coverage only after one strong SMD paper-style result.
8. Add broader online adaptation variants only if the smaller slice justifies them.

## Consistency checks for the long-term roadmap

The repository is moving in the right direction when these statements remain true:

- If the drift layer is removed, clean offline and clean streaming paths still run.
- If the thesis model is removed, the baseline reconstruction model still runs end to end.
- If online adaptation is removed, streaming inference and evaluation still run.
- If optional regularizers are turned off, the offline thesis path still runs with only reconstruction plus classification loss.
- If a new dataset is added, the high-level experiment scripts do not need to be rewritten.

If these conditions keep holding, the codebase is becoming more reusable without losing readability.

## Final conclusion

The valuable part of the older roadmap was never the literal folder plan or the exact phase numbering. The valuable part was the sequencing logic:

- make one thing real
- keep contracts stable
- modularize what already works
- add controlled drift
- add online evaluation
- expand adaptation conservatively
- generalize only after one strong result

That logic should remain the long-term planning rule for this repository.
