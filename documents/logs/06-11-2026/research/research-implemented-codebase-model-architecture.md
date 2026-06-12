---
date: 2026-06-11 13:40:58 +0700
researcher: Artificial Intelligence Agent
git_commit: 4a4e23939b0b8961fa27919282c1622e44840d66
branch: dev
repository: bachelor-thesis-2026
topic: "Implemented model architecture currently present in the codebase"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-11
last_updated_by: Artificial Intelligence Agent
---

# Research: Implemented model architecture currently present in the codebase

**Date**: 2026-06-11 13:40:58 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `4a4e23939b0b8961fa27919282c1622e44840d66`  
**Branch**: `dev`

## Research Question
Write the version of the model architecture that is already implemented in the codebase.

## Summary
The main thesis-facing model that is actually implemented in the repository is `ThesisMultitaskModel` in `src/models/thesis_multitask.py`. This model is a self-contained offline multitask architecture built around one encoder, one continuous prototype branch, one discrete prototype branch, a fusion block that produces separate hidden states for reconstruction and classification, a reconstruction head, a window-level classification head, and a synthetic anomaly injection path that prepares RedLamp-style multi-class supervision during training and synthetic validation.

The active architecture is not DMTRL-LAF. It does not factorize convolution kernels into shared basis kernels and task coefficients. Instead, it uses a standard encoder family abstraction with either `mlp` or `cnn_simple`, then applies prototype retrieval and fusion on the encoder hidden representation. Optional two-view contrastive learning and optional CKA-gated fusion are already wired into the same model file and are activated only through configuration flags.

## Detailed Findings

### Data Preparation
- The model expects a standardized batch dictionary and validates it before forward propagation.
- Input windows follow the repository contract `X in R^{B x L x D}`.
- Synthetic anomaly injection is owned by the model path itself, not by a separate offline preprocessing artifact.
- During train:
  - if synthetic augmentation is enabled, the batch is passed through `SyntheticAnomalyInjector`,
  - the augmented batch receives `classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`, and `augmentation_metadata`.
- During `val_synth` and `val_realistic`, the model can also build synthetic validation batches through a deterministic validation injector.
- If a clean batch is used, the model materializes default zero anomaly labels and a zero synthetic anomaly mask so that the same downstream codepath still works.

### Modeling and Training

#### 1. Encoder block
- The encoder wrapper is `MultitaskWindowEncoder`.
- Two encoder families are supported:
  - `mlp`
  - `cnn_simple`
- The simple CNN path is implemented by `SimpleWindowCnnEncoder`.
- The encoder preserves the window axis and returns:
  - `hidden in R^{B x L x H}`
  - `pooled = mean(hidden, dim=1)`

#### 2. Prototype branches
- The continuous branch uses a prototype bank stored in `continuous_prototype_bank`.
- The continuous lookup is soft:
  - normalize token vectors,
  - compute token-to-prototype similarity,
  - apply softmax over prototypes,
  - reconstruct a continuous prototype context as a weighted mixture.
- The discrete branch uses:
  - `discrete_assignment` linear projection,
  - `discrete_codebook`,
  - Gumbel-Softmax assignment probabilities,
  - a quantized hidden representation as a soft mixture over codebook entries.
- The discrete codebook is maintained with EMA buffers:
  - `discrete_ema_counts`
  - `discrete_ema_sums`

#### 3. Memory lifecycle
- Prototype memories are not always active from epoch 0.
- The model contains an explicit bootstrap phase controlled by `bootstrap_encoder_epochs`.
- Before memories are initialized, prototype lookup falls back to bypass mode.
- After the bootstrap phase, the model can initialize memories from collected hidden tokens and then switch to train-time memory updates.
- During train, memory updates are stage-aware:
  - the continuous memory can use normal-token information,
  - the discrete memory can use anomaly-token information when a synthetic anomaly mask exists.

#### 4. Fusion block
- The model does not attach default prediction heads directly to the branch-local outputs.
- Instead, it fuses the continuous branch output and the discrete branch output into:
  - `hidden_reconstruction`
  - `hidden_classification`
- In the default scalar-gated mode:
  - `alpha` controls how much the classification path uses the discrete branch,
  - `beta` controls how much the reconstruction path uses the discrete branch.
- In optional CKA-gated mode:
  - the model computes per-sample linear CKA scores,
  - builds a two-dimensional CKA feature vector,
  - feeds that vector through two small gating MLPs,
  - predicts per-sample `alpha` and `beta`.

#### 5. Task heads
- The reconstruction head is an MLP applied tokenwise to `hidden_reconstruction`.
- The classification head is an MLP applied to the flattened window representation:

  `hidden_classification.reshape(B, L * H)`

- Therefore, classification is window-level, not token-level.

#### 6. Scores and outputs
- Reconstruction produces `recon`.
- Point anomaly scores are computed as tokenwise mean squared reconstruction error:

  `point_scores = mean((recon - x)^2, dim=-1)`

- Window anomaly scores are then:

  `window_scores = mean(point_scores, dim=1)`

- The forward output contract contains:
  - `hidden`
  - `pooled`
  - `recon`
  - `logits`
  - `point_scores`
  - `window_scores`
  - `aux`

#### 7. Contrastive path
- If `enable_two_view_contrastive` is off, the model stays on the standard one-view path.
- If it is on:
  - the model prepares a clean batch and an augmented batch,
  - the clean batch is encoded first,
  - the augmented batch uses the clean hidden state as `paired_hidden_for_fusion`,
  - contrastive loss is computed over normal tokens only,
  - anomalous tokens contribute as negatives only through the augmented view geometry.

#### 8. Loss assembly
- The actual training objective is assembled in `_shared_step`.
- The main components are:
  - reconstruction loss,
  - classification loss,
  - optional regularizers,
  - optional contrastive loss.
- The total loss is:

  `L_total = L_recon + w_cls * L_cls + optional_losses + lambda_contrastive * L_contrastive`

- Optional regularizers include:
  - diversity loss,
  - variance loss,
  - covariance loss,
  - usage loss,
  - gate loss.

### Evaluation
- The model itself emits scores and logits, but metric computation is external to the model file.
- The implemented architecture is therefore responsible for:
  - producing reconstruction outputs,
  - producing window-level classification logits,
  - producing point-level and window-level anomaly scores,
  - logging branch, fusion, memory, and CKA diagnostics.

## Code References
- `src/models/thesis_multitask.py:3` - self-contained model scope and intended reading order
- `src/models/thesis_multitask.py:95` - simple CNN encoder implementation
- `src/models/thesis_multitask.py:153` - encoder wrapper that returns thesis-facing hidden states
- `src/models/thesis_multitask.py:203` - structured config groups for architecture, prototypes, schedule, objective, memory, and synthetic augmentation
- `src/models/thesis_multitask.py:475` - main `ThesisMultitaskModel`
- `src/models/thesis_multitask.py:626` - continuous and discrete prototype memory construction
- `src/models/thesis_multitask.py:676` - fusion parameter and gate construction
- `src/models/thesis_multitask.py:728` - synthetic anomaly injectors
- `src/models/thesis_multitask.py:1361` - continuous prototype lookup
- `src/models/thesis_multitask.py:1404` - discrete prototype lookup
- `src/models/thesis_multitask.py:1459` - fusion outputs, including optional CKA-gated fusion
- `src/models/thesis_multitask.py:1555` - linear CKA helper
- `src/models/thesis_multitask.py:1588` - two-view contrastive loss
- `src/models/thesis_multitask.py:1711` - main forward pass
- `src/models/thesis_multitask.py:2332` - shared step where the total objective is assembled
- `src/core/config.py:237` - config allowlist for prototype, contrastive, fusion, and memory controls

## Pipeline Documentation
The implemented architecture follows this computational story:

`batch -> optional synthetic anomaly injection -> encoder -> continuous branch + discrete branch -> fusion -> reconstruction head + classification head -> anomaly scores + classification logits -> multitask loss`

If two-view contrastive is enabled, the path becomes:

`clean batch + augmented batch -> clean hidden + augmented hidden -> contrastive loss + fusion-conditioned multitask loss`

So, hiểu nôm na thì, model hiện tại là một prototype-fusion multitask model có thể dùng `simple CNN` làm encoder, chứ chưa phải một CNN factorization model kiểu DMTRL-LAF.

## Historical Context (from documents/)
- `documents/design/design_starter.md` describes the intended decomposition `encoder -> prototype modules -> fusion -> task heads`, and the implemented model follows that decomposition closely.
- `documents/design/idea.md` states that real task supervision should remain on fused task-specific states, and the implemented model does exactly that by attaching default prediction heads only after fusion.
- `documents/logs/06-09-2026/research/research-exp2-readiness-status-for-dmtrl-laf-cnn-simple.md` already noted that the codebase is runnable for `cnn_simple + two-view contrastive + CKA-gated fusion`, but not for true DMTRL-LAF.

## Open Questions
- The implemented architecture already supports `cnn_simple`, prototype branches, CKA-gated fusion, and two-view contrastive learning, but it does not expose a staged single-task-to-multitask orchestration surface as a first-class experiment protocol.
- The implemented architecture does not include a consensus loss between reconstruction and classification heads.
- The implemented architecture does not include DMTRL-LAF kernel factorization or SVD-based initialization.
