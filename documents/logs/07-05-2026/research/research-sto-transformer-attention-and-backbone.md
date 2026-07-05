---
date: 2026-07-05 21:20:29 +0700
researcher: Artificial Intelligence Agent
git_commit: c0ef2451ab524914cbb2343a031c8455f1737a5c
branch: TheMetaSetter
repository: sto-transformer-main
topic: "Does stochastic attention run during training from scratch, and what is the backbone of the main model?"
tags: [research, transformer, stochastic-attention, backbone]
status: complete
last_updated: 2026-07-05
last_updated_by: Artificial Intelligence Agent
---

# Research: Does stochastic attention run during training from scratch, and what is the backbone of the main model?

**Date**: 2026-07-05 21:20:29 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `c0ef2451ab524914cbb2343a031c8455f1737a5c`  
**Branch**: `TheMetaSetter`

## Research Question
In the original `sto-transformer-main` codebase, is stochastic attention used during training from scratch, and what is the backbone of the main model?

## Summary
The codebase uses stochastic attention directly inside the model forward pass when `model_type == "tf-sto"` in IMDB, and when `--sto_transformer True` in CoLA. There is no separate training-phase gate that disables stochastic attention at initialization or enables it only later. The main model backbone is a Transformer encoder: token embedding plus positional embedding, followed by stacked transformer blocks, then a linear classification head. In the stochastic variant, the standard attention module is replaced by `StoSelfAttention` or `StoSelfDualAttention`, depending on the `direction` flag.

## Detailed Findings

### Data Preparation
- IMDB and CoLA are separate experiments with their own data loaders and training scripts.
- This note only documents the model path, not dataset preprocessing.

### Modeling and Training
- IMDB chooses the stochastic model implementation when `args.model_type == "tf-sto"` and imports `IMDB.Model_sto_transformers` in that case.
- The IMDB training path instantiates `IMDB(...)` and immediately trains it; there is no warm-up or delayed switch that turns stochastic attention on only after some initial deterministic phase.
- In `IMDB/Model_sto_transformers.py`, the `IMDB` class always constructs `StoTransformerEncoder`, not the deterministic encoder.
- In `common/transformer.py`, `StoTransformerBlock` selects `StoSelfAttention` for `direction == 1` or `StoSelfDualAttention` for `direction == 2`.
- Both stochastic attention classes call `F.gumbel_softmax(...)` directly inside `forward`, with no `if self.training` condition. That means the stochastic sampling path is part of the forward computation itself.
- CoLA follows the same pattern: `train.py` creates `Model_S` when `--sto_transformer True`, and `Model_S` is built from `DualStoSelfAttention` or `StoSelfAttention` depending on `dual`.

### Backbone
- The backbone of the main model is a Transformer encoder stack.
- Concrete components are:
  - token embedding
  - positional embedding
  - `StoTransformerBlock` repeated for `num_layers`
  - LayerNorm and feed-forward sublayers
  - linear classification head
- For the stochastic IMDB variant, the encoder backbone is `StoTransformerEncoder`, which wraps `StoEncoderNetwork`.

## Code References
- `bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Run.py:16-19` - selects the stochastic IMDB model implementation
- `bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Run.py:27-40` - initializes the model and starts training
- `bsc-thesis-ref-codebases/sto-transformer-main/code/IMDB/Model_sto_transformers.py:12-36` - IMDB model uses `StoTransformerEncoder`
- `bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py:61-153` - stochastic attention implementations
- `bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py:421-587` - stochastic transformer block and encoder stack
- `bsc-thesis-ref-codebases/sto-transformer-main/code/CoLA/train.py:28-71` - CoLA model selection and training
- `bsc-thesis-ref-codebases/sto-transformer-main/code/CoLA/Sto_Transformer.py:273-361` - CoLA stochastic model wiring

## Pipeline Documentation
When the stochastic model path is selected, stochastic attention is not a special evaluation-only mode. It is the actual attention mechanism inside the model from the first training step. The model backbone remains Transformer-based rather than LSTM-based or BERT-based.

## Historical Context (from documents/)
This repo contains separate IMDB and CoLA experiments. The stochastic variant is exposed as a model type or flag, while the deterministic Transformer remains available as a separate baseline.

## Open Questions
- None for this question. The code path is explicit enough to answer the training-time behavior and backbone choice.
