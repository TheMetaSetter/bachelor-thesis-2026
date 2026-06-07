---
date: 2026-06-04 15:25:41 +0700 +07
researcher: TheMetaSetter
git_commit: 32417993875f677a86743ab3a770d0ccc67b32fe
branch: dev
repository: bachelor-thesis-2026
topic: "Current change points required to replace the active MLP encoders with a simple RedLamp-style CNN backbone in redlamp_mlp_baseline.py and thesis_multitask.py"
tags: [research, time-series, anomaly-detection, redlamp, cnn, backbone]
status: complete
last_updated: 2026-06-04
last_updated_by: TheMetaSetter
---

# Research: Current change points required to replace the active MLP encoders with a simple RedLamp-style CNN backbone in redlamp_mlp_baseline.py and thesis_multitask.py

**Date**: 2026-06-04 15:25:41 +0700 +07
**Researcher**: TheMetaSetter
**Git Commit**: 32417993875f677a86743ab3a770d0ccc67b32fe
**Branch**: dev

## Research Question

Use `prompts/1_research_prompt.md` to determine which code positions must change if the active MLP encoders are replaced by a simple convolutional neural network backbone similar to the reference RedLamp encoder in `bsc-thesis-ref-codebases/RedLamp/`, specifically inside `src/models/redlamp_mlp_baseline.py` and `src/models/thesis_multitask.py`.

## Summary

The active repository does not currently implement a RedLamp-style convolutional encoder in either the baseline model or the thesis multitask model. The active baseline is an MLP timestep encoder defined directly inside `src/models/redlamp_mlp_baseline.py`, while the active thesis model uses `MultitaskWindowEncoder`, another MLP timestep encoder defined inside `src/models/thesis_multitask.py`. The RedLamp reference encoder instead uses `nn.Conv1d`, transposes input windows from `[B, L, D]` to `[B, D, L]`, applies stacked convolutional blocks, performs global temporal max pooling, and then applies a `1x1` convolution.

The repository-level model contract only requires that model outputs expose `hidden` with rank 3, plus the standard reconstruction, classification, and scoring fields. Therefore, the engine, evaluator, logger, and checkpoint code do not currently force an MLP encoder. The main implementation burden is inside the two owning model files, plus any local helper logic that assumes the encoder is a list of `nn.Linear` layers. The strongest hidden dependency of that kind appears in the baseline gradient-conflict profiling helper, which enumerates encoder linear layers by direct `isinstance(layer_module, nn.Linear)` checks.

## Detailed Findings

### Data Preparation

- Offline batches remain standardized as `batch["x"]: Tensor[B, L, D]` under `validate_batch(...)` in `src/core/contracts.py`.
- Synthetic anomaly augmentation remains independent of the encoder family. `src/data/augment.py` appends `classification_labels`, `classification_class_names`, `synthetic_anomaly_mask`, and `augmentation_metadata`, but it does not assume MLP or convolutional tensors.
- Because augmentation leaves the base batch shape unchanged, a convolutional backbone can be introduced only inside the model files as long as the model internally handles any required transpose between `[B, L, D]` and `[B, D, L]`.

### Modeling and Training

#### RedLamp reference implementation

The reference encoder in `bsc-thesis-ref-codebases/RedLamp/models/cnn.py` defines the target style being discussed:

- `ConvEncoder` stacks `ConvBlock` modules and uses `nn.Conv1d` throughout.
- Its `forward(...)` transposes from time-major repository shape to channel-first convolution shape, then performs temporal pooling and a `1x1` convolution.
- The reference path therefore changes the internal tensor convention, even though the thesis repository contract externally stays in `[B, L, D]`.

#### Active baseline model

The active baseline in `src/models/redlamp_mlp_baseline.py` is explicitly an MLP baseline.

The exact encoder-related change points in the current file are:

1. Constructor parameter surface:
   - The constructor currently exposes MLP-oriented arguments such as `latent_dim`, `mlp_num_linear_layers`, and `classifier_dim`.
   - There are no convolution-oriented arguments such as filter widths, channel stacks, stride, padding, or normalization.

2. Encoder construction block:
   - `self.encoder = build_multilayer_perceptron(...)` is the active backbone construction.
   - This is the primary location where a CNN backbone would be instantiated instead of an MLP encoder.

3. Forward encoder call:
   - `hidden = self.encoder(x_tensor)` currently assumes that the encoder accepts `[B, L, D]` directly and returns `[B, L, latent_dim]`.
   - If a RedLamp-style convolutional encoder is introduced, this forward path becomes the place where input transposition and post-convolution reshaping or broadcasting back into `[B, L, H]` must be handled.

4. Gradient-conflict profiling helper:
   - `_get_encoder_profiled_parameters(...)` enumerates only `nn.Linear` layers inside `self.encoder`.
   - `_resolve_focus_layer_parameter_name(...)` also assumes the focus layer concept `encoder_last_linear`.
   - These helpers are not generic over backbone type and will break or silently become meaningless if the encoder becomes convolutional without updating the profiling logic.

5. Decoder compatibility surface:
   - The baseline currently uses `self.decoder = build_multilayer_perceptron(...)` and applies `recon = self.decoder(hidden)`.
   - If only the encoder backbone changes but `hidden` remains `[B, L, latent_dim]`, the current decoder path can remain structurally compatible.
   - If the intent is to mirror RedLamp more closely as a convolutional autoencoder, then the decoder block becomes an additional change point because the reference RedLamp path uses `ConvDecoder`, not an MLP decoder.

6. Output contract exposure:
   - `outputs["hidden"]`, `outputs["pooled"]`, `outputs["recon"]`, `outputs["logits"]`, `outputs["point_scores"]`, and `outputs["window_scores"]` are already aligned with repository contracts.
   - These locations do not require semantic changes if the CNN backbone still returns `hidden: Tensor[B, L, H]`.

#### Active thesis multitask model

The active thesis model in `src/models/thesis_multitask.py` centralizes all encoder, prototype, fusion, and loss logic in one file. The exact encoder-related change points are concentrated and clearer than in the baseline.

The exact change points in the current file are:

1. Shared encoder helper class:
   - `MultitaskWindowEncoder` is the active encoder implementation.
   - It currently builds `self.network = build_multilayer_perceptron(...)`.
   - Its `forward(...)` currently returns `hidden = self.network(batch["x"])`, with `hidden` shaped as `[B, L, hidden_dim]`.
   - This class is the primary thesis-side replacement point for introducing a CNN backbone while preserving the output contract used by the rest of the file.

2. Architecture configuration surface:
   - `MultitaskArchitectureConfig` currently stores `input_dim`, `window_size`, `encoder_dim`, `hidden_dim`, `mlp_num_linear_layers`, `num_classes`, and `dropout`.
   - There is currently no config surface for convolutional hyperparameters such as channel schedule, kernel size, stride, padding, normalization, or encoder type selection.
   - If the repository wants the CNN backbone to be config-driven rather than hard-coded, this dataclass is the current configuration boundary that would need to expand.

3. Flat kwargs resolution:
   - `ThesisMultitaskModelConfig.from_flat_kwargs(...)` whitelists architecture keys and rejects unknown values.
   - Any new encoder-family or convolution-specific architecture fields would need to be admitted here, otherwise construction will fail with unknown-key errors.

4. Encoder build block:
   - `_build_encoder(...)` currently instantiates `MultitaskWindowEncoder(...)`.
   - This is the exact owning construction point where the thesis model selects its backbone.
   - Because the rest of the file only consumes `self.encoder(batch)["hidden"]`, this method is the narrowest and cleanest switch point.

5. Forward path entry:
   - In `forward(...)`, the line `encoder_outputs = self.encoder(batch)` is the only place where the thesis model obtains hidden states before the prototype branches.
   - If the new encoder preserves `hidden: Tensor[B, L, hidden_dim]`, the remaining prototype, fusion, reconstruction, classification, and scoring logic can remain shape-compatible.

6. Hidden-state consumers that depend on shape but not on encoder family:
   - `_update_continuous_memory_bank(...)`
   - `_update_discrete_codebook_memory(...)`
   - `_continuous_prototype_lookup(...)`
   - `_discrete_prototype_lookup(...)`
   - `_compute_fusion_outputs(...)`
   - `_normalize_branch_tokens(...)`
   - `_compute_contrastive_loss(...)`
   - These functions operate on `hidden_dim` tokens and do not directly assume `nn.Linear`, `MLP`, or time-independent encoding. Their dependency is on hidden tensor shape, not backbone family.

7. Reconstruction and classification heads:
   - `self.reconstruction_head` expects tokenwise input of shape `[B, L, hidden_dim]`.
   - `self.classification_head` expects flattened window-level input of shape `[B, window_size * hidden_dim]`.
   - These blocks do not need to change if the encoder still returns sequence-shaped hidden states with the same final hidden width.

### Evaluation

- `validate_model_outputs(...)` in `src/core/contracts.py` only checks ranks and required keys, not encoder implementation.
- Offline scoring logic in both active models computes pointwise anomaly scores from `recon - batch["x"]`, so this surface only depends on reconstruction output shape.
- No evaluator code currently hard-codes MLP assumptions for these two models, provided that the model continues returning:
  - `hidden: rank-3`
  - `recon: rank-3`
  - `point_scores: rank-2`
  - `window_scores: rank-1`

## Code References

- `bsc-thesis-ref-codebases/RedLamp/models/cnn.py:75` - RedLamp reference `ConvEncoder`
- `bsc-thesis-ref-codebases/RedLamp/models/cnn.py:108` - RedLamp encoder forward with transpose and pooling
- `bsc-thesis-ref-codebases/RedLamp/models/meta.py:135` - RedLamp model wiring uses `ConvEncoder`
- `src/core/contracts.py:83` - offline batch contract validator
- `src/core/contracts.py:123` - model output contract validator
- `src/data/augment.py:14` - RedLamp anomaly taxonomy constants
- `src/data/augment.py:33` - synthetic injector constructor remains encoder-agnostic
- `src/models/redlamp_mlp_baseline.py:3` - baseline declared as MLP baseline
- `src/models/redlamp_mlp_baseline.py:114` - active baseline encoder construction
- `src/models/redlamp_mlp_baseline.py:219` - active baseline forward path
- `src/models/redlamp_mlp_baseline.py:297` - baseline gradient-profiling helper assumes encoder `nn.Linear` layers
- `src/models/redlamp_mlp_baseline.py:387` - baseline gradient-conflict profiling over encoder parameters
- `src/models/thesis_multitask.py:36` - shared MLP builder used by encoder and heads
- `src/models/thesis_multitask.py:95` - `MultitaskWindowEncoder`
- `src/models/thesis_multitask.py:127` - architecture config surface
- `src/models/thesis_multitask.py:241` - flat config resolution boundary
- `src/models/thesis_multitask.py:509` - thesis encoder construction point
- `src/models/thesis_multitask.py:521` - prototype-memory blocks that consume `hidden_dim`
- `src/models/thesis_multitask.py:584` - thesis task heads
- `src/models/thesis_multitask.py:1605` - thesis forward path entry through encoder

## Pipeline Documentation

The runtime path relevant to this encoder substitution remains:

`batch["x"] [B, L, D]` -> model-local encoder -> `hidden [B, L, H]` -> reconstruction and classification paths -> output contract validators.

For the thesis model, the sequence then continues:

`hidden [B, L, H]` -> continuous prototype branch and discrete prototype branch -> task-specific fusion -> reconstruction head and flattened classifier head -> anomaly scores.

The current repository design isolates the backbone change well because the trainer and evaluator do not consume encoder internals directly. The real implementation coupling is local to each owning model file.

## Historical Context (from documents/)

- `documents/design/idea.md` defines the stable thesis-facing encoder contract as `H in R^{B x L x d_h}` and explicitly recommends freezing the encoder interface first.
- `documents/design/design_starter.md` states that the encoder block should be replaceable without rewriting the trainer, provided the shared batch and model-output contracts stay fixed.
- `documents/logs/05-13-2026/detail/detail-redlamp-timestep-encoder-baseline-implementation.md` records that the baseline was intentionally moved away from the older flattened-window geometry to a timestep MLP baseline for a more controlled comparison with the thesis model.
- `documents/logs/05-14-2026/detail/detail-flatten-classifier-latent-redlamp-thesis-implementation.md` records that both baseline and thesis models now follow the RedLamp-style flatten-before-classifier pattern, while keeping reconstruction tokenwise.
- `documents/logs/05-27-2026/research/research-current-thesis-multitask-computation-flows.md` already records that the active thesis encoder is MLP-based and explicitly notes that a future REDLAMP CNN substitution would need a prior decision on preserving the current output and shape contract of `pooled`.

## Open Questions

- Whether the intended substitution means only a convolutional encoder backbone, or a fuller RedLamp-style convolutional autoencoder path with both convolutional encoder and convolutional decoder in the baseline.
- Whether the thesis model should hard-code one CNN encoder replacement or expose an explicit encoder-family selector for ablation and reproducibility.
- Whether baseline gradient-conflict profiling should remain focused on a specific named layer when the encoder is no longer an ordered stack of linear layers.
