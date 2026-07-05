# Two-Stage KMeans Memory Redesign Structure

## Overview
The implementation should begin from the smallest stable vertical slice: preserve the current batch, encoder, and model-output contracts, then align Stage A and Stage B to the approved two-stage memory redesign. Only after the offline contract is stable should the codebase absorb the more advanced prototype and online-adaptation surfaces, so that each additional module reuses a known interface instead of introducing a second one.

## Implementation Phases

1. **Phase 1: Stabilize the core contracts and the minimal vertical slice** - This phase preserves the current fixed-length window pipeline, the `[B, L, D]` batch contract, and the thesis-facing hidden-state contract. It keeps the existing registry, trainer, and self-contained model-file organization intact, which satisfies separation of concerns and composition over inheritance before any prototype logic is modified.

2. **Phase 2: Replace memory initialization with k-means-based prototype bootstrapping** - This phase changes only the memory-bootstrap surface so that continuous memory is initialized from clean latent tokens and discrete memory is initialized from class-stratified synthetic anomaly tokens. The code should keep the model as the owner of memory semantics, which preserves single responsibility and avoids scattering clustering logic across the runner or trainer.

3. **Phase 3: Align the task-specific fusion and synthetic anomaly classification path** - This phase keeps the continuous and discrete branches separate up to fusion, then verifies that reconstruction and classification consume the fused task-specific representations. Synthetic anomaly injection remains the source of classification supervision, and the objective stays modular so that optional losses can remain configuration-driven rather than hard-coded.

4. **Phase 4: Formalize the online adaptation stage with a residual projector** - This phase is downstream of the offline slice and should only be activated once the offline encoder and memory contracts are stable. It adds a lightweight projector that aligns a trainable online encoder to a frozen reference encoder, while residual initialization, warm-starting, and gated updates preserve the adapter pattern and reduce the risk of uncontrolled drift.

5. **Phase 5: Validate the full pipeline through ablations, evaluation, and reporting** - This phase compares the minimal baseline against the prototype-enabled and adaptation-enabled variants, then records the metrics, checkpoint artifacts, and qualitative diagnostics in the repository’s logging structure. The evaluation plan should keep metric definitions explicit so that offline gains, branch usage, and online adaptation behavior remain interpretable rather than conflated.

## Structure Check

The order is intentional. The codebase should first lock the minimal offline contract, then add prototype bootstrapping, then confirm fusion and classification behavior, and only after that expand into online adaptation. This sequencing preserves the minimal vertical slice principle and reduces the chance that the residual projector or prototype logic is built on top of an unstable interface.

Does this phase order and granularity look right before I expand it into a fuller execution outline?

