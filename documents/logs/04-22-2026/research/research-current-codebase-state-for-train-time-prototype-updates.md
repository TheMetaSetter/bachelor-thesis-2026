---
date: 2026-04-22 15:34:26 +07+0700
researcher: TheMetaSetter
git_commit: 3f87d9596f475fe811dd7b50097f0de39993a520
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase state for adding train-time continuous and discrete prototype updates while keeping the existing simple loss surface"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-04-22
last_updated_by: TheMetaSetter
---

# Research: Current codebase state for adding train-time continuous and discrete prototype updates while keeping the existing simple loss surface

**Date**: 2026-04-22 15:34:26 +07+0700
**Researcher**: TheMetaSetter
**Git Commit**: `3f87d9596f475fe811dd7b50097f0de39993a520`
**Branch**: `dev`

## Research Question
Research the current repository state in preparation for adding a training-time prototype update mechanism, based on the attached discrete-memory mathematics note and the repository H-PAD memory-gate note, while preserving the current simple offline loss surface. The requested focus is the update mechanism for continuous and discrete prototypes only.

## Summary
The current repository already has the exact high-level pipeline that a train-time prototype-write mechanism would need: SMD windows of length one hundred are parsed, standardized, batched, optionally augmented with synthetic anomalies, encoded into hidden states, read through a continuous prototype bank and a discrete codebook branch, fused into reconstruction and classification paths, and then evaluated with pointwise anomaly scores. However, the present prototype banks are not memory modules in the H-PAD sense. They are ordinary learnable `nn.Parameter` tensors that are read inside `forward()` and updated only indirectly through backpropagation and the optimizer step.

This distinction is the most important current-state finding. The continuous branch already computes token-to-prototype attention weights for prototype retrieval, but it does not compute prototype-specific weighted summaries over the current batch and it does not write updated prototype values back into the bank during training. The discrete branch already computes relaxed assignment probabilities and reads a quantized hidden representation from a learned codebook, but it does not maintain running counts, running feature sums, or any explicit exponential-moving-average or learned-gate write rule. The current simple objective also already matches the user’s constraint: the active default is reconstruction loss plus classification loss, with all extra regularizers disabled by default.

## Detailed Findings

### Data Preparation
- The active dataset path is SMD only. The parser expects the standard `train/`, `test/`, and `test_label/` folders and, when no entity filter is supplied, expects all twenty-eight machine files per split. Training and validation labels are initialized as zeros, while test labels are loaded from the SMD label files.
- The default data configuration fixes `window_size: 100` and `stride: 10`, with `batch_size: 32`. This matches the thesis-facing fixed-length window contract described in the design documents.
- Standardization is fit on the cleaned training split only, using feature-wise mean and standard deviation, and the fitted scaler is then applied to train, validation, and test sequences.
- Windowing is overlap-aware. Each entity sequence is sliced into `[L, D]` windows, then collated into a batch contract with `x`, `point_labels`, `mask`, `timestamps`, and `meta`.
- Synthetic anomaly injection is not part of the dataset builder. It is model-owned. During `training_step`, the multitask model may replace a clean batch with an augmented batch that includes `classification_labels`, `synthetic_anomaly_mask`, and `augmentation_metadata`. Validation may also run a deterministic synthetic branch through `val_synth`.

### Modeling and Training
- The offline thesis model remains fully self-contained in one file, as required by repository policy. It contains the encoder, continuous branch, discrete branch, fusion logic, task heads, optional losses, and stage-specific step methods.
- The encoder exposes the expected thesis-facing hidden representation with shape `[B, L, H]`, and the forward pass then branches into continuous prototype lookup and discrete codebook lookup before task-specific fusion.
- The continuous branch currently performs a read-only attention lookup. It computes logits with `torch.einsum("blh,kh->blk", hidden, prototype_bank)` and applies `softmax(..., dim=-1)` across prototypes for each token. The resulting `prototype_context` is a weighted mixture of prototype vectors for each token.
- The discrete branch currently performs a read-only relaxed quantization. It maps each token to assignment logits through a linear layer, applies `torch.nn.functional.gumbel_softmax`, and reads a quantized token representation by multiplying those probabilities by the codebook vectors.
- Both prototype stores are ordinary parameters. `continuous_prototype_bank` and `discrete_codebook` are created as `nn.Parameter(torch.randn(...))`. There is no separate memory buffer, no running state, and no explicit mutation of these tensors inside `training_step`, `validation_step`, or `test_step`.
- The current simple loss surface is already conservative. The default model YAML sets `lambda_cls: 1.0` and leaves `lambda_div`, `lambda_var`, `lambda_cov`, `lambda_use`, and `lambda_gate` at zero. The training step explicitly assembles `total_loss = reconstruction_loss + classification_weight * classification_loss + optional_terms`, and the regression suite contains a test that confirms the total equals reconstruction plus classification when optional losses are disabled.
- The trainer already exposes the two natural train-only gating surfaces that a future write mechanism would need. First, `Trainer.train()` places the model in training mode before calling `training_step`. Second, the model’s `_shared_step()` receives a `stage_name` of `train`, `val`, `val_synth`, or `test`, so training-only behavior can be isolated without adding a second engine path.
- The current tests cover shapes, one-step backward execution, fusion limiting cases, temperature scheduling, and usage-loss scheduling. Repository search did not find a test that asserts prototype state mutation during training or prototype immutability during evaluation, which means the present suite does not yet encode a memory-write contract.

### Current State Relative to the Requested Update Mechanisms
- The repository H-PAD note defines continuous-style prototype writing as a training-time update that first computes prototype-to-query weights over the current batch, then forms a weighted query summary for each prototype, then mixes old and new content through a learned gate, and finally disables that write path at test time.
- The attached discrete-memory note defines the discrete write path differently from the current code. It starts from the same kind of assignment probabilities already present in the repository, but then aggregates soft usage counts and weighted token sums per code, updates running statistics with exponential moving averages, and normalizes those running statistics into refreshed discrete prototypes. The note also describes an optional learned gate interpretation.
- The current continuous branch is structurally closer to a read equation than a write equation. It uses token-to-prototype softmax over the prototype axis, which is appropriate for reconstructing hidden tokens from stored prototypes. The H-PAD write equations instead require, for each prototype, a normalization over tokens in the current batch so each prototype can summarize the query tokens most relevant to it.
- The current discrete branch already computes one of the critical ingredients for a discrete write path: `assignment_probabilities`. The repository also already computes a simple aggregate of those probabilities for usage regularization. What is missing from the current codebase state is the second aggregate required by the attached note: a weighted token sum per code, together with any persistent running statistics that survive across batches.
- Because the continuous and discrete banks live inside the standard model state dict today, checkpointing and online adaptation already preserve and reuse the learned prototype values. The online model loads a saved `thesis_multitask` checkpoint, deep-copies it into frozen reference and online encoders, and continues to use the same continuous and discrete read functions. In other words, the repository already has the “frozen memory at later stages” pattern, but only after offline learning has finished and only through ordinary checkpoint serialization.

### Evaluation
- Evaluation is pointwise and overlap-aware. The evaluator calls `model.test_step()` on window batches, accumulates window-level point scores back onto each entity timeline, averages overlaps, and computes a single threshold from the 0.95 quantile of positive scores when available.
- The metric payload includes pointwise ROC AUC, PR AUC, precision, recall, F1, and false positive rate. Evaluation also stores ROC and precision-recall curve payloads.
- The evaluation script writes `evaluation_records.json`, `evaluation_metrics.json`, `evaluation_curves.json`, and `resolved_experiment_config.json` into the experiment output directory. The training logger separately writes `metrics.jsonl` and the resolved configuration for training runs.

## Code References
- [configs/data/smd.yaml#L1-L9](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/configs/data/smd.yaml#L1-L9) - active SMD data configuration with window length 100 and stride 10
- [src/data/datasets/smd.py#L14-L179](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/data/datasets/smd.py#L14-L179) - SMD parser, split creation, and label handling
- [src/data/loaders.py#L118-L226](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/data/loaders.py#L118-L226) - scaling, window datasets, and loader construction
- [src/data/scalers.py#L10-L62](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/data/scalers.py#L10-L62) - training-split standardization
- [src/data/collate.py#L11-L37](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/data/collate.py#L11-L37) - batch contract assembly
- [src/core/contracts.py#L91-L139](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/core/contracts.py#L91-L139) - batch and model-output contracts
- [src/models/thesis_multitask.py#L223-L318](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/thesis_multitask.py#L223-L318) - prototype-bank parameters and default objective controls
- [src/models/thesis_multitask.py#L448-L515](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/thesis_multitask.py#L448-L515) - current continuous and discrete prototype read paths
- [src/models/thesis_multitask.py#L643-L740](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/thesis_multitask.py#L643-L740) - end-to-end forward path from hidden states to fused outputs and anomaly scores
- [src/models/thesis_multitask.py#L755-L956](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/thesis_multitask.py#L755-L956) - reconstruction/classification losses and optional regularizers
- [src/models/thesis_multitask.py#L1057-L1158](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/thesis_multitask.py#L1057-L1158) - shared step and stage-specific entrypoints
- [configs/model/thesis_multitask.yaml#L1-L36](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/configs/model/thesis_multitask.yaml#L1-L36) - model defaults showing the current simple loss setup
- [configs/task/multitask_tsad.yaml#L1-L23](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/configs/task/multitask_tsad.yaml#L1-L23) - synthetic anomaly task controls and anomaly families
- [src/engine/trainer.py#L203-L300](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/engine/trainer.py#L203-L300) - train/eval mode switching and epoch loop
- [src/engine/checkpoint.py#L28-L105](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/engine/checkpoint.py#L28-L105) - checkpoint payload contents
- [src/models/online_adaptation.py#L29-L82](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/online_adaptation.py#L29-L82) - online adapter reuse of offline continuous/discrete branches
- [src/models/online_adaptation.py#L163-L229](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/models/online_adaptation.py#L163-L229) - checkpoint loading and freezing of offline prototype geometry
- [src/engine/evaluator.py#L48-L190](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/engine/evaluator.py#L48-L190) - overlap-aware evaluation and thresholding
- [src/metrics/pointwise.py#L68-L117](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/src/metrics/pointwise.py#L68-L117) - classification and pointwise anomaly metrics
- [tests/test_one_multitask_train_step.py#L63-L120](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/tests/test_one_multitask_train_step.py#L63-L120) - regression proof that the baseline loss is reconstruction plus classification when optional losses are disabled
- [tests/test_fusion_ablation_modes.py#L18-L61](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/tests/test_fusion_ablation_modes.py#L18-L61) - continuous-only and discrete-only fusion limiting cases
- [bsc-thesis-ref-codebases/h-pad/h-pad-memory-gate.md#L1-L260](https://github.com/TheMetaSetter/bachelor-thesis-2026/blob/3f87d9596f475fe811dd7b50097f0de39993a520/bsc-thesis-ref-codebases/h-pad/h-pad-memory-gate.md#L1-L260) - repository note describing H-PAD gated prototype updates

## Pipeline Documentation
The current offline pipeline is:

1. Parse SMD entity files into train, validation, and test sequences.
2. Fit a feature-wise standard scaler on training sequences only.
3. Slice each sequence into overlapping windows of length one hundred and stride ten.
4. Collate windows into the stable batch contract `[B, L, D]`.
5. During training, optionally inject synthetic anomalies and attach binary classification labels and a pointwise anomaly mask.
6. Encode the batch once into hidden states.
7. Read a continuous prototype context from the continuous bank and a quantized discrete hidden state from the discrete codebook.
8. Fuse those two branch outputs into reconstruction and classification hidden states.
9. Decode reconstruction, pool classification hidden states, compute logits, and derive pointwise anomaly scores from reconstruction error.
10. Optimize the current small default objective or the explicitly configured weighted extension.
11. During evaluation, merge overlapping window scores back onto entity timelines and compute pointwise anomaly-detection metrics.

Within this pipeline, the present prototype behavior is “read during forward, learn through gradient descent, serialize in checkpoints.” It is not yet “read and write memory during training, then freeze memory at test time.”

## Historical Context (from documents/)
The design documents already define the repository direction as a readability-first offline multitask model with one continuous prototype branch, one discrete prototype branch, task-specific fusion, and a small default objective. They also explicitly state that optional regularizers should remain off until concrete failure modes justify them. This matches the current implementation and also matches the user’s instruction to keep the loss function simple while focusing only on prototype updating.

The most relevant recent repository note is the 2026-04-13 detail log on discrete-usage stabilization and validation alignment. That note documents a conservative step that preserved the existing binary classifier and main reconstruction path while only refining temperature scheduling, usage-loss scheduling, and validation semantics. It is important context because it confirms that recent work in this codebase has intentionally protected the simple objective and has not converted the prototype branches into explicit memory-write modules yet.

The repository also already contains a prose H-PAD memory-gate note whose training-versus-testing interpretation matches the user’s requested behavior: prototypes should be updated during training and treated as frozen memory during inference. The attached discrete-memory mathematics note is compatible with that high-level distinction, but it introduces a different low-level write rule for the discrete bank based on soft counts, weighted token sums, and exponential-moving-average state.

## Open Questions
1. The current repository has one token-level continuous bank and one token-level discrete codebook, while the H-PAD note describes patch prototypes and period prototypes. The present codebase therefore has a representational mapping to define before any exact equation-level transplant can be claimed.
2. The current continuous branch softmax is over prototypes for each token, which serves the read path. The H-PAD update rule instead needs a per-prototype normalization over the current token set, which is a different tensor normalization pattern than the one currently implemented.
3. The attached discrete-memory note requires persistent running statistics for each discrete prototype. The current checkpoint payload contains the model state dict, optimizer state, scaler state, config, epoch, and metric history, but no dedicated non-parameter prototype-running-state surface exists today.
4. Repository search did not find a test that checks “prototype state changes after a training step but does not change during validation or test.” If train-only memory writes are added later, that behavioral contract is not currently pinned by the test suite.
