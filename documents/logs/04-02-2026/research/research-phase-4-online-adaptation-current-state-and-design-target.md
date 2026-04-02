---
date: 2026-04-02 00:34:05 +0700
researcher: TheMetaSetter
git_commit: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
branch: dev
repository: bachelor-thesis-2026
topic: "Phase 4 online adaptation: current repository state and design target"
tags: [research, time-series, anomaly-detection, online-adaptation, phase-4]
status: complete
last_updated: 2026-04-02
last_updated_by: TheMetaSetter
---

# Research: Phase 4 online adaptation: current repository state and design target

**Date**: 2026-04-02 00:34:05 +0700
**Researcher**: TheMetaSetter
**Git Commit**: 597dc3a4a4a01f933e133424b78c67fdd51a75f0
**Branch**: dev

## Research Question

Apply the repository research workflow to Phase 4 of `documents/logs/03-31-2026/detail/detail-smd-loader-windowing.md` and determine two things clearly and separately: what the current repository actually implements for Phase 4 today, and what Phase 4 is intended to become according to the design documents under `documents/design/`.

## Summary

The current repository now exposes an active but conservative Phase 4 implementation in `src`, `configs`, `scripts`, and `tests`. The codebase now contains `src/data/stream.py`, `src/models/online_adaptation.py`, `src/engine/online_loop.py`, `scripts/run_online_adaptation.py`, `configs/model/online_adaptation.yaml`, `configs/task/online_adaptation.yaml`, `configs/experiment/smd_online_adaptation.yaml`, and focused online tests. The active runtime is therefore no longer offline-only. It now consists of an ablation-ready offline multitask path plus a first projector-first online adaptation slice.

Phase 4 nevertheless still needs to be described carefully. The design sources converge on the same conservative online-adaptation story: a frozen reference encoder, a partially trainable online encoder, a lightweight near-identity residual projector, offline warm-start of that projector before true streaming updates, projector-first adaptation scope, and a narrow online optimization boundary that keeps parameter roles explicit. The same document set also states that broader online work should remain blocked until the pre-Phase-4 offline gate is closed. The current repository now meets that gate for the first conservative slice, but it still defers drift injection, NGD-style optimization, and encoder unfreezing.

## Detailed Findings

### Data Preparation

- **Current repository state.** The configuration inventory now includes both the offline multitask family and the first online family: `configs/experiment/smd_multitask.yaml`, `configs/experiment/smd_multitask_smoke.yaml`, the explicit ablation variants under `configs/experiment/`, and `configs/experiment/smd_online_adaptation.yaml` together with `configs/model/online_adaptation.yaml` and `configs/task/online_adaptation.yaml`.
- **Current repository state.** `src/data/stream.py` now provides the sequential online stream and batcher used by the first online slice. The code still respects the design warning that this stream surface should stay narrow for the first accepted runtime: clean stream only, no drift injector in the active path.
- **Design-defined future contract.** The Phase 4 design documents describe a future streaming stack in which benchmark datasets are wrapped as sequential streams, optional drift injectors operate on those streams, and a windowized online path feeds the adaptation stage. This is documented as Phase 4 and later design context rather than active implementation.

### Modeling and Training

- **Current repository state.** `src/models/thesis_multitask.py` now owns the ablation-ready offline multitask path, including fusion warm-up, temperature annealing, gate-entropy regularization, branch-usage diagnostics, and the optional-loss surface driven by `lambda_*` weights. `scripts/run_ablation.py` turns those config-level variants into a compact summary artifact without creating a second training codepath.
- **Current repository state.** The repository now exposes an active Phase 4 runtime API: `src/models/online_adaptation.py` defines the projector-first online model, `src/engine/online_loop.py` defines the checkpointable online engine, and `scripts/run_online_adaptation.py` builds the sequential stream and runs the first accepted online slice. The online checkpoint boundary is explicit and the default reference checkpoint now points to the canonical offline multitask run at `outputs/smd_multitask/checkpoints/best.pt`.
- **Design-defined future contract.** `documents/design/idea.md` defines Phase 4 as an online adaptation stage that creates two semantic views for each incoming sample, passes one view through a frozen reference encoder and the other through a partially trainable online encoder, and then maps the online representation into the reference space with a lightweight projector. The same document specifies a contrastive alignment objective, a prototype-alignment objective, and a near-identity residual form for the projector, with offline warm-start before real streaming updates.
- **Design-defined future contract.** `documents/design/design_starter.md` makes the one-model-one-file implication explicit by naming a future `online_adaptation.py` model file that should own the frozen reference encoder, online encoder, residual projector, optional NGD-ready preconditioning surface, and online alignment losses in one readable unit.
- **Design-defined future contract.** `documents/design/stream_design.md` narrows the intended optimization boundary for future online adaptation. It defines separate parameter roles for `reference_params`, `online_encoder_params`, `projector_params`, and optional adapter parameters, and it states that the projector is the first and safest parameter group for geometry-aware or NGD-style adaptation. The same document also defines an explicit optimizer boundary so that the future online loop chooses both optimizer family and target parameter group through configuration rather than by branching into a second architecture.

### Evaluation

- **Current repository state.** The repository now contains focused online evaluation and monitoring surfaces in code. The online loop logs alignment loss, update norm, projector drift, and window-score summaries; online checkpoints persist stream state, projector anchor state, target parameter-group selection, online metric history, and reset-policy state; and focused tests cover stream batches, update boundaries, checkpoint round-tripping, invalid reference-checkpoint rejection, and top-level entrypoint wiring.
- **Design-defined future contract.** The future online checkpoint contract in `documents/design/stream_design.md` includes model parameters, optimizer state, scheduler state, a projector anchor copy or initialization reference if used, stream cursor or stream-loop progress state, and reset-policy state if present. The same document also defines a monitoring contract for alignment loss, anomaly score stability, update norm, projector drift from initialization or anchor state, and reset-trigger signals.
- **Pre-Phase-4 gate status.** The current detail and checklist documents agree that this evaluation and reporting surface must not become active before the offline ablation-readiness work is completed. The repository therefore documents the future Phase 4 evaluation surface, but it does not yet expose it as running code.

## Code References

- `documents/logs/03-31-2026/detail/detail-smd-loader-windowing.md:172` - pre-Phase-4 offline priorities that must close before online adaptation begins
- `documents/logs/03-31-2026/detail/detail-smd-loader-windowing.md:217` - `src/data/stream.py` is no longer treated as part of the active pre-Phase-4 path
- `documents/logs/03-31-2026/detail/detail-smd-loader-windowing.md:231` - Phase 4 remains blocked until the pre-Phase-4 ablation-readiness work is addressed
- `documents/logs/04-01-2026/detail/detail-pre-phase-4-ablation-readiness-checklist.md:19` - offline ablation readiness is the formal gate before Phase 4
- `documents/logs/04-01-2026/detail/detail-pre-phase-4-ablation-readiness-checklist.md:139` - explicit blocking conditions for Phase 4
- `documents/design/idea.md:65` - design definition of the online adaptation stage
- `documents/design/idea.md:95` - contrastive alignment, prototype alignment, near-identity projector, and offline warm-start design
- `documents/design/idea.md:107` - explicit pre-Phase-4 gate in the design narrative
- `documents/design/design_starter.md:336` - one-model-one-file rule applied to online adaptation
- `documents/design/design_starter.md:346` - future `online_adaptation.py` ownership boundary
- `documents/design/stream_design.md:261` - Phase 4 and later status plus pre-Phase-4 gate
- `documents/design/stream_design.md:275` - future online adaptation parameter-group boundary
- `documents/design/stream_design.md:314` - future online checkpoint and state contract
- `documents/design/stream_design.md:327` - future online monitoring contract
- `src/models/thesis_multitask.py:41` - current active offline multitask model surface
- `src/engine/trainer.py:12` - current trainer surface remains an offline training loop
- `scripts/train.py:23` - current runtime registration covers the offline dataset and model set
- `scripts/train.py:58` - current training script builds the active dataset through the offline registry path

## Pipeline Documentation

The current implemented repository pipeline remains:

```text
raw SMD files
-> offline data configuration
-> build_dataset(...)
-> offline window batches
-> ThesisMultitaskModel or ReconstructionMLPAutoencoder
-> offline Trainer / Evaluator
-> checkpoints, metrics, and evaluation outputs
```

The repository now implements the first Phase 4 extension on top of that pipeline. In particular, the current codebase exposes an online stream wrapper, a dual-view online batch path, a frozen-reference-versus-online-encoder split, a residual projector stage, and an online adaptation loop. What it still does not expose is the broader later-slice streaming surface from the design documents, such as drift injection and NGD-style optimization.

The design-defined future Phase 4 pipeline is documented as:

```text
benchmark dataset stream
-> optional drift injector
-> sliding-window construction
-> two semantic views per online sample
-> frozen reference encoder and partially trainable online encoder
-> near-identity residual projector
-> alignment and prototype-consistency objectives
-> online monitoring and checkpointed adaptation state
```

This second pipeline is presently a documented design contract only. It should not be described as active code in the current repository snapshot.

## Historical Context (from documents/)

The target detail note reframes the repository after the Phase 1 to Phase 3 closure work and makes pre-Phase-4 ablation readiness the remaining gate before online adaptation. The separate ablation-readiness checklist strengthens that gate by requiring exact config-level ablations, fusion-control scheduling, branch-collapse observability, ablation-oriented tests, and reproducible reporting before Phase 4 begins.

The design documents then describe what Phase 4 is supposed to be once that gate is passed. `documents/design/idea.md` defines the semantic structure of online adaptation, `documents/design/design_starter.md` defines the self-contained model-file boundary, and `documents/design/stream_design.md` defines the future stream stack, optimizer boundary, checkpoint contract, and monitoring contract. Taken together, the repository documentation is now consistent with the implementation direction: the first conservative Phase 4 slice exists in code, and broader online expansion remains intentionally deferred.

## Open Questions

- The current online path implements only the clean-stream projector-first slice. The design documents still describe deferred expansions such as drift injection, broader parameter-group adaptation, and NGD-style optimization.
- The reset-policy schema now exists in the online checkpoint and config surface, but only the conservative disabled or threshold policy scaffold is active.
- The online runtime now requires a `thesis_multitask` checkpoint explicitly. The remaining open implementation question is how later slices should warm-start or widen beyond `projector_params` without broadening the accepted first codepath too early.
