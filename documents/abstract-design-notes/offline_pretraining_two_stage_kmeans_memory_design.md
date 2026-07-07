---
title: "Offline Pre-Training Two-Stage KMeans Memory Design"
date: 2026-07-04
status: approved
owners:
  - TheMetaSetter
  - Codex
tags:
  - design
  - offline-pretraining
  - two-stage
  - thesis-multitask
  - kmeans
  - memory-initialization
---

# Offline Pre-Training Two-Stage KMeans Memory Design

## Purpose

This document records the currently approved design for the next rerun of
`src/models/thesis_multitask.py`.

It supersedes the older three-stage training intent for this rerun. The goal is
to keep the active thesis experiment simple:

1. train one multitask encoder from scratch,
2. initialize both memory banks from the resulting latent space,
3. freeze the encoder and both memories together,
4. train only the fusion and prediction layers in the final stage.

This document is the new SSOT for the approved high-level training contract
before code and config changes.

## Terminology

For the active rerun described in this document:

- `offline pre-training` is the large phase.
- `Stage A` and `Stage B` are the only stages inside that phase.
- The older three-stage material remains historical context only and does not
  define the active meaning of `phase` for this rerun.

## Scope

This design applies to:

- `src/models/thesis_multitask.py`
- the benchmark-style thesis experiment configs that will be derived from the
  current active SMD benchmark configs

This design does **not** apply to:

- `src/models/redlamp_baseline.py`
- the older three-stage rerun plan
- the old separate-task pretraining workflow

## Confirmed Design Choices

### 1. Model Scope

Only `thesis_multitask.py` will be rerun under this redesign.

`redlamp_baseline.py` is out of scope for this rerun.

### 2. Training Topology

The approved workflow is now **two-stage**, not three-stage.

```text
Stage A
  -> multitask encoder training from scratch
  -> end-of-stage memory initialization

Stage B
  -> freeze encoder + both memories
  -> train only fusion heads + prediction heads
```

The old middle stage is removed entirely.

### 3. Epoch Budget

The approved epoch split is:

- `Stage A = 80 epochs`
- `Stage B = 20 epochs`

Total:

- `100 epochs`

### 4. Stage A Objective

Stage A trains the shared multitask encoder from scratch using exactly these
losses:

- `reconstruction loss`
- `classification loss`
- `contrastive loss`

No separate task-specific pretraining is used.

### 5. Stage B Trainable Surface

Stage B trains only:

- `reconstruction fusion head`
- `classification fusion head`
- `reconstruction prediction head`
- `classification prediction head`

Stage B freezes:

- shared encoder
- continuous memory bank
- discrete memory bank

The reason is explicit: the encoder and both memory banks should remain aligned
in the same latent space. The design does not allow freezing one side while
continuing to drift the other.

## Stage Contract

### Stage A

```text
input
  -> shared encoder                         [trainable]
  -> reconstruction head                    [trainable]
  -> classification head                    [trainable]
  -> contrastive objective                  [active]
  -> memory banks                           [not yet used as frozen final banks]
```

At the end of Stage A:

- collect latent pools from the training split only
- initialize continuous memory bank
- initialize discrete memory bank

### Stage B

```text
input
  -> shared encoder                         [frozen]
  -> continuous memory bank                 [frozen]
  -> discrete memory bank                   [frozen]
  -> reconstruction fusion head             [trainable]
  -> classification fusion head             [trainable]
  -> reconstruction prediction head         [trainable]
  -> classification prediction head         [trainable]
```

## Memory Design

### 1. Continuous Memory Size

The approved size is:

- `continuous_num_prototypes = 32`

This replaces the earlier active value `16` for the new rerun.

### 2. Discrete Memory Size

The approved choice is:

- keep the current discrete codebook size unchanged

In the active benchmark-style thesis setup this remains:

- `discrete_codebook_size = 60`

### 3. Discrete Query Mode Direction

The approved direction is:

- keep `cosine_topk` as the intended discrete query mode
- avoid keeping Gumbel-only machinery when it is not needed

Design intent:

- if the active discrete query path is `cosine_topk`, the implementation should
  avoid initializing or persisting unnecessary Gumbel-only components such as
  `self.discrete_assignment` when they are no longer required by the new design
  path
- the checkpoint should be kept lighter and the runtime surface should become
  simpler

This is a design constraint, not yet an implementation claim.

## Memory Initialization Contract

Memory initialization happens **at the end of Stage A**.

All memory initialization pools must come from the training split only.

### 1. Continuous Memory Initialization

The approved contract is:

```text
input:
  clean-token latent vectors from the clean train split

method:
  k-means

K:
  continuous_num_prototypes = 32

output:
  32 centroids
  -> continuous_prototype_bank
```

This means continuous memory is initialized from **clean latent tokens** only.

### 2. Discrete Memory Initialization

The approved contract is:

```text
input:
  latent vectors grouped by class from the synthetic train split

method:
  k-means per class

K:
  5 centroids for each class

output:
  12 groups of centroids
  -> concatenate
  -> 60 codewords
  -> discrete_codebook
```

This means the discrete memory is initialized from **synthetic anomaly latent
tokens**, grouped by class.

## Token-Pool Construction Contract

The memory initialization token pools must follow these rules.

### Continuous Pool

Source:

- clean train split

Kept tokens:

- clean / normal latent tokens only

Therefore:

- continuous memory represents clean latent structure

### Discrete Pool

Source:

- synthetic train split

Kept tokens:

- only anomaly tokens
- not all tokens from anomalous windows

Grouping:

- group anomaly latent tokens by `classification_labels`

Therefore:

- discrete memory represents anomaly-class latent structure, not broad window
  structure

## Approved Pool-Building Semantics

The intended high-level behavior is:

```text
TRAIN LOADER
   |
   v
take first memory_initialization_batches
   |
   v
for each batch
   |
   +--> move batch to device
   |
   +--> clean_batch
   |      |
   |      v
   |   encoder(clean_batch)["hidden"]
   |
   +--> if synthetic memory init is OFF
   |      |
   |      +--> CONTINUOUS POOL:
   |      |       add clean hidden tokens
   |      |
   |      +--> DISCRETE POOL[class 0]:
   |              add clean hidden tokens
   |
   +--> if synthetic memory init is ON
          |
          +--> synthetic_batch = augment_batch(...)
          |
          +--> synthetic_hidden = encoder(synthetic_batch)["hidden"]
          |
          +--> classification_labels
          +--> synthetic_anomaly_mask
          |
          +--> CONTINUOUS POOL:
          |      keep only clean / normal tokens
          |
          +--> DISCRETE POOL:
                 keep only anomaly tokens
                 then group them by classification class
```

## Approved KMeans Initialization Semantics

### Continuous

```text
continuous_hidden_tokens
   |
   v
normalize vectors
   |
   v
k-means with K = continuous_num_prototypes
   |
   v
continuous centroids
   |
   v
copy into continuous_prototype_bank
```

### Discrete

```text
discrete_hidden_tokens_by_class
   |
   v
for each class
   |
   +--> take anomaly tokens of that class only
   |
   +--> normalize vectors
   |
   +--> k-means with K = 5
   |
   v
class centroids
   |
   v
concat all class centroids
   |
   v
discrete_codebook
```

## Design Implications for the Current Codebase

Compared with the current code, the approved rerun direction implies:

1. remove dependence on the old separate-task pretraining workflow
2. replace the current `_select_covering_vectors(...)` initialization heuristic
   with k-means-based centroid extraction
3. restrict discrete initialization tokens to anomaly tokens only, instead of
   broad anomalous-window token collection
4. increase continuous prototype count from `16` to `32`
5. simplify the discrete runtime path around `cosine_topk` so unnecessary
   Gumbel-only state is not carried forward without need

## Non-Negotiable Rules

1. Only `thesis_multitask.py` is in scope for this rerun.
2. Training is strictly two-stage.
3. Stage A uses exactly:
   - reconstruction loss
   - classification loss
   - contrastive loss
4. Stage B trains only:
   - two fusion heads
   - two prediction heads
5. Encoder and both memory banks freeze together in Stage B.
6. Continuous memory initialization uses clean train latent tokens only.
7. Discrete memory initialization uses synthetic anomaly tokens only.
8. Continuous memory uses k-means with `K = 32`.
9. Discrete memory uses per-class k-means with `K = 5`.
10. No test-derived statistics or test-derived latent pools may enter memory
    initialization.

## Status

This design is approved at the concept level and is ready to be translated into
an implementation plan and then code changes.
