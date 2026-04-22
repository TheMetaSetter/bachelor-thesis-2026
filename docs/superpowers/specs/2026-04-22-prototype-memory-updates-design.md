# Prototype Memory Updates Design

## Goal

Add explicit train-time memory updates to the offline `thesis_multitask` model while keeping the current simple objective surface. The model should keep one continuous memory bank and one discrete codebook, preserve the current query-reconstruction read mechanism, update memory only during training and outside the gradient graph, save all memory state in checkpoints, and enforce hard magnitude control across encoder and memory representations.

## Scope

This design covers only the offline multitask model path centered on `src/models/thesis_multitask.py` and its direct engine, checkpoint, config, and test surfaces. It does not change the online adaptation design except where checkpoint loading must continue to work with the enriched offline memory state. It does not add new loss terms beyond the current simple default objective.

## Current Baseline

The active repository already has:

- one continuous prototype branch with token-to-prototype attention for reconstruction
- one discrete codebook branch with Gumbel-Softmax assignments for reconstruction
- a simple default objective of reconstruction plus classification
- stage-specific step methods for train, validation, synthetic validation, and test
- checkpoint save and load for model, optimizer, scheduler, scaler, and config

What it does not yet have is explicit memory state with train-only write behavior. The current continuous bank and discrete codebook are ordinary learnable parameters updated through optimizer steps rather than explicit memory writes.

## Design Principles

The implementation should follow these principles:

- keep one continuous bank and one discrete codebook only
- keep query reconstruction exactly in the current direction: each timestep attends to all memory entries, then reconstructs its query from the resulting softmax weights
- separate read and write semantics clearly
- allow memory content to update only through explicit memory-write rules inside training forward passes
- keep memory writes outside the gradient graph
- allow the continuous gate network to be learnable by gradient while keeping the actual continuous bank non-differentiable
- keep the discrete branch EMA-only for memory updates
- preserve the current simple objective and avoid adding new loss complexity
- store enough memory state in checkpoints to resume training exactly
- add explicit tests for train-updates versus val/test freeze behavior
- enforce hard magnitude control so encoder outputs and memory vectors remain on compatible scales

## High-Level Architecture

The offline multitask model will operate in two phases.

### Phase A: Bootstrap Encoder

For the first ten epochs, the model runs in bootstrap mode.

- Memory is bypassed entirely.
- The encoder and current task heads train with the existing simple objective.
- Synthetic anomaly augmentation remains active exactly as in the current code.
- No memory read occurs.
- No memory write occurs.
- This phase exists only to make encoder hidden states less random before memory initialization.

### Phase B: Memory-Backed Training

After bootstrap finishes, the model performs one explicit data-driven initialization pass to create:

- the continuous memory bank
- the discrete codebook
- the EMA running statistics needed by the discrete codebook

From that point onward, each training forward pass follows this order:

1. Encode the batch into hidden states.
2. Update the continuous memory bank outside the gradient graph.
3. Update the discrete codebook with EMA outside the gradient graph.
4. Reconstruct queries from the updated memories using the current read mechanism.
5. Fuse branch outputs and compute reconstruction and classification exactly as today.

Validation and test never update memory. They only read from the current memory snapshot.

## Continuous Branch Design

The continuous branch will have two separate mechanisms: a read path and a write path.

### Continuous Read Path

The read path remains conceptually unchanged from the current implementation.

- Each hidden state at each timestep attends to all entries in the continuous memory bank.
- A softmax over bank entries produces weights for that timestep.
- The reconstructed continuous query representation is the weighted linear combination of the bank entries.

This preserves the current timestep-to-memory reconstruction behavior and keeps the downstream reconstruction and fusion surfaces readable.

### Continuous Write Path

The write path changes to follow the H-PAD direction:

- each prototype attends to all hidden states in the current batch
- each prototype forms a weighted summary of the hidden states most relevant to it
- a learned gate determines how much of the old prototype to keep and how much of the new summary to write
- the actual write into the continuous memory bank happens outside the gradient graph

The gate network is learnable. It may receive gradients through the normal task losses because its output affects the updated memory used later in the same training forward pass. However:

- the continuous memory bank itself is not a trainable parameter
- the final prototype write is not part of the optimizer-driven update path

This creates a clean hybrid:

- learnable update controller
- non-differentiable memory content

## Discrete Branch Design

The discrete branch also keeps separate read and write semantics.

### Discrete Read Path

The read path remains conceptually unchanged from the current implementation.

- Each hidden state produces assignment logits over the discrete codebook.
- The model converts these logits into soft assignment probabilities.
- Each timestep reconstructs its discrete representation as the weighted linear combination of codebook entries.

This preserves the current query-reconstruction mechanism and the current fusion interface.

### Discrete Write Path

The discrete write path becomes EMA-only.

For each codebook vector, the model computes:

- soft usage counts from assignment probabilities
- soft weighted sums of hidden states assigned to that code
- EMA updates for running counts and running sums
- normalized code vectors from those running EMA statistics

The discrete codebook is therefore memory, not a trainable parameter.

No gradient should update:

- discrete codebook entries
- discrete EMA counts
- discrete EMA sums

Only the discrete assignment network remains learnable through gradient.

## Data-Driven Initialization

Memory initialization is a one-time transition from bootstrap mode into memory-backed training.

### Initialization Timing

The initialization pass runs after epoch ten, before the first memory-backed training epoch begins.

It should happen exactly once unless a checkpoint is loaded whose memory state indicates that initialization has not yet happened.

### Initialization Data Pool

Initialization uses a clean-dominant, anomaly-aware pool of encoder hidden states.

The pool should include:

- hidden states from clean training windows
- hidden states from synthetically augmented training windows, but only at timesteps whose anomaly mask indicates normal behavior

The pool must exclude:

- every timestep explicitly marked as synthetic anomaly

This enforces the user’s principle that memory should contain normal patterns only.

### Continuous Initialization

The continuous bank should be initialized from normal hidden states using a coverage-oriented centroid seeding strategy. The purpose is to place the initial prototypes into representative regions of the normal hidden space rather than sampling randomly.

This design should prefer:

- representative coverage of normal hidden-space regions
- stable initial norms
- readable implementation over algorithmic sophistication

### Discrete Initialization

The discrete codebook should also be initialized from normal hidden states, but with stronger emphasis on broad coverage so the EMA-only update path does not begin from a degenerate configuration where only a small subset of codes become active.

The initial discrete EMA running statistics should be seeded consistently with the initialized codebook so that the first EMA updates are numerically stable.

## Magnitude Control

Hard magnitude control is part of the architecture.

### Objective

Ensure that:

- encoder hidden states
- continuous prototypes
- discrete code vectors
- weighted hidden summaries used for writes
- memory-backed reconstructed hidden representations

all live on compatible scales.

### Required Control Points

The design should apply explicit normalization or rescaling at these points:

- hidden states before continuous write
- hidden states before continuous read
- hidden states before discrete assignment
- hidden states before discrete EMA aggregation
- continuous weighted summaries before mixing with old prototypes
- continuous memory bank immediately after initialization and after each write
- discrete codebook immediately after initialization and after each EMA update

This should not add a new loss term. It is a forward-path and post-update control surface.

### Rationale

Without hard scale control, the system risks:

- unstable attention or assignment logits
- gate behavior dominated by norm drift rather than semantic relevance
- discrete codebook collapse due to norm imbalance
- degraded gradient quality in the learnable parts of the network

## Model State Design

The model needs explicit state to represent memory lifecycle and memory content.

### Continuous State

- continuous memory bank
- boolean flag indicating continuous memory initialization

### Discrete State

- discrete codebook
- discrete EMA counts
- discrete EMA sums
- boolean flag indicating discrete memory initialization

### Lifecycle State

- bootstrap epoch count
- boolean or mode flag for whether memory-backed training is active
- boolean or mode flag indicating whether one-time data-driven initialization has already run

These should live as model-owned state that is saved with the model. They should not be optimizer-owned trainable parameters.

## Train, Validation, and Test Contracts

### Train Contract

During bootstrap epochs:

- memory is bypassed
- memory state remains unchanged

During memory-backed training:

- memory updates are allowed
- updates happen before query reconstruction
- continuous and discrete writes are outside the gradient graph

### Validation Contract

- no memory update is allowed
- the model reads from fixed memory state only

### Test Contract

- no memory update is allowed
- the model reads from fixed memory state only

This train-only update contract must be explicit in code, not only implicit in `model.train()` versus `model.eval()`.

## Checkpoint Design

Checkpoint payloads must contain all information required to resume training or evaluate with exactly the same memory state.

In addition to the current payload, checkpoints must preserve:

- continuous memory bank
- continuous memory initialization flag
- discrete codebook
- discrete EMA counts
- discrete EMA sums
- discrete memory initialization flag
- memory lifecycle mode or equivalent bootstrap-versus-memory status

This requirement exists because memory state evolves outside the optimizer and cannot be reconstructed from trainable parameters alone.

## Testing Requirements

The test suite should be extended to lock in the new behavior.

### Train-Update Tests

- training step updates continuous memory bank after initialization
- training step updates discrete codebook after initialization
- training step updates discrete EMA counts and sums after initialization

### Freeze Tests

- validation step does not change continuous memory
- validation step does not change discrete memory or EMA state
- test step does not change continuous memory
- test step does not change discrete memory or EMA state

### Bootstrap Tests

- bootstrap epochs bypass memory completely
- bootstrap epochs do not mutate memory state
- the model still computes the existing simple objective during bootstrap

### Initialization Tests

- data-driven initialization runs once at the bootstrap transition
- anomaly-masked timesteps are excluded from the initialization pool
- initialized memory vectors respect magnitude-control constraints

### Checkpoint Tests

- save/load roundtrip preserves continuous memory state
- save/load roundtrip preserves discrete codebook and EMA statistics
- save/load roundtrip preserves lifecycle mode and initialization flags

## Expected File-Level Changes

The design implies targeted changes in these areas:

- `src/models/thesis_multitask.py`
  - bootstrap mode
  - memory lifecycle state
  - one-time data-driven initialization
  - continuous write path
  - discrete EMA write path
  - hard magnitude control
  - train-only update gating
- `src/engine/checkpoint.py`
  - checkpoint payload support for explicit memory state
- `tests/`
  - train-update, freeze, bootstrap, initialization, and checkpoint tests
- config files under `configs/model/` and possibly `configs/task/`
  - bootstrap epoch count
  - memory initialization controls
  - EMA hyperparameters
  - magnitude-control knobs if explicitly configurable

## Non-Goals

This design intentionally does not include:

- multiple continuous banks
- multiple discrete codebooks
- patch-versus-period memory splits
- new multitask loss terms
- online adaptation redesign
- gradient-based prototype optimization
- anomaly patterns written into memory during initialization

## Open Implementation Notes

These are implementation notes, not unresolved product requirements.

- The exact centroid-seeding procedure for initialization should favor readability and deterministic testability.
- The exact normalization operator can follow the codebase’s readability-first style as long as scale control is explicit and consistent.
- The train-only update gate should be implemented in a way that is easy to audit in tests and logs.

## Acceptance Criteria

The design is complete when the repository supports all of the following:

- ten bootstrap epochs that bypass memory while preserving the simple current objective
- one-time clean-dominant, anomaly-aware memory initialization from normal hidden states
- continuous memory updates during training only, with learned gate and non-differentiable memory writes
- discrete EMA-only codebook updates during training only
- current query-reconstruction read direction preserved for both branches
- hard magnitude control across encoder and memory interfaces
- validation and test guaranteed not to mutate memory state
- checkpoint save/load roundtrip for all memory state
- tests that explicitly lock the train-update versus val/test-freeze contract
