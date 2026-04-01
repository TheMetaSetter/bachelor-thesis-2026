---
date: 2026-03-31 16:35:00 +0700
planner: Artificial Intelligence Agent
git_commit: 7779c876c7da79c961ec7ac18f710620d5172533
branch: dev
repository: bachelor-thesis-2026
topic: "Structure outline for the minimum runnable SMD vertical slice"
tags: [structure, smd, loader, stream, windowing, baseline, evaluation]
status: complete
last_updated: 2026-03-31
last_updated_by: Artificial Intelligence Agent
source_plan: documents/logs/03-31-2026/plans/plan-smd-loader-windowing.md
source_research: documents/logs/03-31-2026/researches/research-smd-loader-windowing.md
---

# Structure: Outline for the minimum runnable SMD vertical slice

## Overview

This structure organizes the first executable thesis milestone as a minimal, contract-preserving SMD pipeline that remains narrow enough to validate the repository foundations before advanced thesis modules are introduced. The design preserves the fixed batch and model-output contracts, the readability-first codebase preferences, and the minimal vertical slice principle so that later prototype, fusion, augmentation, and online adaptation modules can be added without structural rewrite.

## Implementation Phases

1. **Phase 1 - Establish the minimum runnable SMD vertical slice**

   The first phase should implement only the essential path `SMD parser -> scaler -> stream/windowizer -> dataloader -> reconstruction baseline -> trainer/evaluator`. This phase freezes the thesis-facing encoder contract `hidden: [B, L, H]`, preserves the documented batch schema `x: [B, L, D]`, and adopts the directory skeleton from `documents/design/design_starter.md` only for the modules required by the vertical slice. Software engineering discipline is preserved by enforcing explicit contracts, keeping responsibilities separated across the configuration, data, model, task, and engine layers, and following the one-model-per-file rule to maximize readability.

2. **Phase 2 - Reserve extension points for continuous and discrete prototype branches**

   The second phase should introduce only the structural boundaries required for later continuous and discrete prototype modules, without yet implementing their full thesis logic. The baseline model and downstream task code should already consume standardized named outputs rather than model internals, so prototype branches can later enrich `aux`, `hidden`, and task-specific representations without modifying data loaders or the engine. Design pattern principles are preserved here through composition instead of deep inheritance, explicit interface freezing, and minimal registries that reduce coupling while avoiding premature framework complexity.

3. **Phase 3 - Add task-specific fusion and synthetic anomaly augmentation for classification**

   The third phase should extend the validated vertical slice into a multitask setting by adding the classification path, task-specific fusion, and CARLA-inspired synthetic anomaly injection while retaining the stable SMD-native preprocessing pipeline. Reconstruction and classification should remain separate task concerns built on the same encoder-facing contracts, and anomaly augmentation should be introduced as an adapter-style data or task component rather than as implicit behavior inside the model. Engineering principles are preserved by isolating augmentation logic, maintaining explicit configuration boundaries, and ensuring that the added multitask behavior remains testable through focused, minimal `pytest` coverage.

4. **Phase 4 - Introduce the online adaptation stage with a residual projector**

   The fourth phase should add the online adaptation path only after the offline SMD baseline, multitask interfaces, and evaluation procedures are stable. This phase should preserve the frozen reference representation contract while introducing a lightweight projector and adaptation-specific task logic as separate modules, so online behavior does not leak into the offline trainer or baseline data abstractions. The design remains aligned with sound engineering practice by enforcing separation of concerns, minimizing stateful codepaths, and constraining the projector and adaptation utilities behind explicit interfaces that can be checkpointed and tested.

5. **Phase 5 - Consolidate evaluation, ablations, and reproducible reporting**

   The final phase should formalize evaluation, ablation studies, checkpoint round-tripping, and experiment reporting once all prior interfaces are stable and empirically validated. The evaluator should continue to consume standardized outputs and serialize results through a schema that is compatible with Weights and Biases logging, while ablations should compare the minimal vertical slice, prototype extensions, augmentation strategies, and online adaptation safeguards without changing the core repository contracts. Software engineering and design pattern principles are preserved by treating reproducibility, configuration validation, and serialization compatibility as first-class architectural requirements rather than as afterthoughts.

## Structural Assessment

The proposed phasing is coherent for this topic because it protects the minimal SMD vertical slice from being blocked by higher-risk thesis modules while still reserving the correct interfaces for later expansion. The only adjustment from the generic prompt is that Phase 1 is intentionally narrower than the full thesis architecture: continuous and discrete prototypes, task-specific fusion, synthetic anomaly injection, and online adaptation are acknowledged structurally, but they remain deferred until the first offline SMD path is executable, testable, and checkpointable.
