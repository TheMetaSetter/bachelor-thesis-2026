---
date: 2026-01-18 20:52:43 +0700
researcher: Artificial Intelligence Agent
git_commit: 9c4bf62559eb982f4cb6b586ba4e4d9883e2f2b7
branch: dev
repository: bachelor-thesis-2026
topic: "Research and Compatibility Report for Stochastic Attention, Memory-guided Fusion, and Label Refurbishment"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-01-18
last_updated_by: Artificial Intelligence Agent
---

# Research: Research and Compatibility Report for Stochastic Attention, Memory-guided Fusion, and Label Refurbishment

**Date**: 2026-01-18 20:52:43 +0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 9c4bf62559eb982f4cb6b586ba4e4d9883e2f2b7  
**Branch**: dev

## Research Question
Conduct a rigorous technical analysis and compatibility assessment of the stochastic attention, memory-guided fusion, and label refurbishment components across the referenced codebases, with emphasis on modular boundaries for ablation studies and alignment with the drafted architecture.

## Summary
The repository provides three distinct but partially complementary components: a stochastic attention implementation with hierarchical centroids in `bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`, a multi-frequency and multi-scale encoder with sinusoidal memory guidance in `bsc-thesis-ref-codebases/MtsCID/model/Transformer.py` and `bsc-thesis-ref-codebases/MtsCID/model/embedding.py`, and a multi-task reconstruction-and-classification framework with label refurbishment and anomaly scoring in `bsc-thesis-ref-codebases/RedLamp/models/meta.py` and `bsc-thesis-ref-codebases/RedLamp/main.py`. The loader interface required by the objective is already realized in `loaders/augmented_loader.py`, which provides injected anomaly windows, reconstruction targets, and anomaly masks that align with RedLamp-style losses. The primary compatibility risks are shape conventions, embedding expectations, and loss interface mismatches between probabilistic labels and logits.

## Detailed Findings

### Data Preparation and Loader Interface
- **Augmented loader contract**: `AugmentedLoader` and `Loader_aug_batch` produce dictionaries with `Y` (injected window), `Z` (clean window), `anomaly_mask`, and `label` as one-hot vectors (`loaders/augmented_loader.py`). These tensors are shaped as `(batch, n_features, window_size)` and require transposition for models that expect `(batch, window_size, n_features)`.
- **Entity-level dataset structure**: `DataEntity` and `Dataset` standardize `Y`, optional `X`, labels, and masks, and enforce consistent feature dimensions across entities (`loaders/dataset.py`).
- **Heterogeneous dataset selection**: Dataset selection is centralized in `loaders/load.py`, which supports `smd`, `msl`, `smap`, `anomaly_archive`, and `iops` with entity lists and validation splits, enabling explicit in-domain and out-of-domain evaluation via different dataset and entity selections (`loaders/load.py`).

### Stochastic Attention Component (Sto-Transformer)
- **Core attention modules**: `SelfAttention`, `StoSelfAttention`, and `StoSelfDualAttention` define deterministic, Gumbel-softmax stochastic, and centroid-quantized stochastic attention respectively (`bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`).
- **StoSelfDualAttention specifics**: A learnable centroid matrix `centroid` is used to quantize key vectors before stochastic attention is applied; temperatures `tau1` and `tau2` control the Gumbel-softmax sampling of centroids and attention weights (`bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`).
- **Transformer block integration**: `StoTransformerBlock` swaps `SelfAttention` for stochastic attention based on `direction`, then applies LayerNorm and feed-forward sublayers (`bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`).

### Memory-guided Multi-Frequency and Multi-Scale Encoder (MtsCID)
- **Two-branch encoder**: `TransformerVar` defines two parallel embeddings (`encoder_branch1`, `encoder_branch2`) with configurable networks such as `intra_fc_transformer`, `inter_fc_transformer`, and `multiscale_ts_attention`, providing a structural analog to intra-variate and inter-variate processing (`bsc-thesis-ref-codebases/MtsCID/model/Transformer.py`, `bsc-thesis-ref-codebases/MtsCID/model/embedding.py`).
- **Frequency-domain processing**: `TokenEmbedding` applies real FFT and inverse FFT around transformer layers depending on branch configuration, implementing automated multi-frequency processing (`bsc-thesis-ref-codebases/MtsCID/model/embedding.py`).
- **Multi-scale attention**: `Inception_Attention_Block` aggregates attention over multiple patch sizes, while `Inception_Block` performs multi-kernel convolution; these modules implement scale diversity in time and frequency domains (`bsc-thesis-ref-codebases/MtsCID/model/multi_attention_blocks.py`, `bsc-thesis-ref-codebases/MtsCID/model/Conv_Blocks.py`).
- **Memory-guided attention**: `TransformerVar` constructs `mem_R` and `mem_I` via `create_memory_matrix`, then derives an attention distribution over `mem_R` using dot products with branch-2 queries (`bsc-thesis-ref-codebases/MtsCID/model/Transformer.py`). The imaginary component `mem_I` is constructed but unused in the forward path.

### Label Refurbishment and Multi-task Training (RedLamp)
- **Meta architecture**: `MetaAEC` defines an encoder-decoder-classifier pipeline with joint reconstruction and classification losses, with optional anomaly masking and label refurbishment (`bsc-thesis-ref-codebases/RedLamp/models/meta.py`).
- **Label refurbishment**: The label smoothing mechanism redistributes probability mass using parameters `alpha` and `beta`, increasing normal-class probability and injecting mass into non-target anomalies (`bsc-thesis-ref-codebases/RedLamp/models/meta.py`).
- **Loss decomposition**: Reconstruction uses mean-squared error and classification uses cross entropy; the total loss is a convex combination controlled by `c_loss_ratio` (`bsc-thesis-ref-codebases/RedLamp/models/meta.py`).
- **Anomaly scoring**: The final anomaly score averages a smoothed reconstruction error and a smoothed classification-based score computed from predicted class distributions (`bsc-thesis-ref-codebases/RedLamp/main.py`).

## Dependency Mapping and Compatibility Assessment
- **Input shape convention**: Loaders yield `(batch, features, window)`, while RedLamp and MtsCID models process `(batch, window, features)`; integration requires systematic transposition to avoid silent misalignment (`loaders/augmented_loader.py`, `bsc-thesis-ref-codebases/RedLamp/main.py`, `bsc-thesis-ref-codebases/MtsCID/model/Transformer.py`).
- **Embedding expectations**: Sto-transformer encoders assume token indices and positional embeddings, whereas MtsCID encoders accept continuous tensors and operate in the frequency domain; only the attention blocks are directly interoperable (`bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`, `bsc-thesis-ref-codebases/MtsCID/model/embedding.py`).
- **Loss interface mismatch**: RedLamp classifiers output probabilities via `Softmax`, yet `CrossEntropyLoss` is applied to those outputs; this implies non-standard gradient dynamics if logits are expected (`bsc-thesis-ref-codebases/RedLamp/models/classifier.py`, `bsc-thesis-ref-codebases/RedLamp/models/meta.py`).
- **Memory guidance scope**: MtsCID uses `mem_R` only for attention scoring and detaches it from gradient flow, implying fixed sinusoidal memory rather than trainable memory, which may limit integration with learnable stochastic attention modules (`bsc-thesis-ref-codebases/MtsCID/model/Transformer.py`).
- **Scoring alignment**: RedLamp anomaly scoring assumes window-wise predictions and outputs a scalar anomaly score per window, which is consistent with the required sequence-level score given `window_size = 100` (`bsc-thesis-ref-codebases/RedLamp/main.py`).

## Ablation Readiness Assessment
- **Stochastic attention boundary**: The stochastic attention mechanism is localized to `StoSelfDualAttention` and `StoTransformerBlock`, enabling a clean toggle between deterministic and stochastic attention by swapping attention modules (`bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py`).
- **Memory-guided fusion boundary**: Memory-guided components are confined to `create_memory_matrix` and the attention computation using `mem_R` in `TransformerVar`, which can be isolated as an optional guidance pathway without altering the branch embeddings (`bsc-thesis-ref-codebases/MtsCID/model/Transformer.py`).
- **Label refurbishment boundary**: Label refurbishment is encapsulated within `MetaAEC.calculate_loss`, enabling ablation by controlling the smoothing parameters and the `label_smoothing` flag without affecting architecture (`bsc-thesis-ref-codebases/RedLamp/models/meta.py`).

## Architectural Alignment with MtsCID Multi-Frequency and Multi-Scale Logic
- **Fourier-transform pathway**: The architecture draft’s Fourier and inverse Fourier transforms align with MtsCID’s repeated `rfft` and `irfft` operations within `TokenEmbedding`, which implement multi-frequency encoding on a per-layer basis (`bsc-thesis-ref-codebases/MtsCID/model/embedding.py`).
- **Intra-variate encoder**: Drafted intra-variate blocks correspond to `branch1_networks` that include `intra_fc_transformer` and `multiscale_ts_attention`, both operating on per-variable spectral representations (`bsc-thesis-ref-codebases/MtsCID/model/embedding.py`, `bsc-thesis-ref-codebases/MtsCID/model/multi_attention_blocks.py`).
- **Inter-variate encoder**: Drafted inter-variate blocks correspond to `branch2_networks` that include `multiscale_conv1d` and `inter_fc_transformer`, which couple variables via convolution and attention across frequency and time (`bsc-thesis-ref-codebases/MtsCID/model/embedding.py`, `bsc-thesis-ref-codebases/MtsCID/model/Conv_Blocks.py`).
- **Multi-scale attention**: The “MIS Attention” in the draft is structurally consistent with `Inception_Attention_Block`, which aggregates attention across multiple patch sizes to provide multi-scale context (`bsc-thesis-ref-codebases/MtsCID/model/multi_attention_blocks.py`).
- **Decoder and classification heads**: MtsCID provides a decoder for reconstruction but does not include a classifier; RedLamp provides a classifier but not stochastic attention or memory guidance, indicating that alignment requires explicit cross-component assembly rather than direct reuse.

## Mathematical Foundations

### Stochastic Dual Attention (StoSelfDualAttention)
\[
Q = X W_Q,\quad K = X W_K,\quad V = X W_V.
\]
**Explanation**: \(X\) is the input sequence embedding, \(W_Q, W_K, W_V\) are learned projection matrices per head, and \(Q, K, V\) are the query, key, and value tensors.

\[
K_c = K C,\quad P = \mathrm{GumbelSoftmax}(K_c / \tau_1),\quad K_{\mathrm{sto}} = P C^\top.
\]
**Explanation**: \(C \in \mathbb{R}^{d_h \times k}\) is the learnable centroid matrix, \(K_c\) are centroid scores, \(P\) is a stochastic assignment over centroids with temperature \(\tau_1\), and \(K_{\mathrm{sto}}\) is the centroid-quantized key tensor.

\[
A = Q K_{\mathrm{sto}}^\top,\quad S = \mathrm{GumbelSoftmax}(A / \tau_2),\quad O = S V.
\]
**Explanation**: \(A\) are attention logits, \(S\) is the stochastic attention distribution with temperature \(\tau_2\), and \(O\) is the attention output prior to linear projection.

### Memory-guided Attention in MtsCID
\[
A_{b,f,j} = \sum_{\ell=1}^{L} q_{b,\ell,f} \; r_{j,\ell},\quad \alpha_{b,f,j} = \mathrm{softmax}(A_{b,f,j} / T).
\]
**Explanation**: \(q\) is the branch-2 query tensor, \(r\) is the sinusoidal memory matrix \(\mathrm{mem\_R}\), \(L\) is window length, \(T\) is the temperature parameter, and \(\alpha\) is the memory-guided attention distribution.

\[
\mathrm{mem\_R}_{n,\ell} = \cos\left(\frac{2\pi}{L} n\ell\right),\quad \mathrm{mem\_I}_{n,\ell} = \sin\left(\frac{2\pi}{L} n\ell\right).
\]
**Explanation**: \(n\) indexes the memory basis, \(\ell\) indexes time, and the real and imaginary components form a fixed sinusoidal memory when `mem_type` is `sinusoid`.

### Label Refurbishment (RedLamp)
\[
\tilde{y} = y \cdot (1 - \alpha - \beta C + \beta) + (1 - y) \cdot \beta.
\]
**Explanation**: \(y\) is the original one-hot label, \(C\) is the number of classes, \(\alpha\) is the mass assigned to the normal class, and \(\beta\) is the mass redistributed to non-target classes.

\[
\tilde{y}_0 \leftarrow \tilde{y}_0 + \alpha.
\]
**Explanation**: \(\tilde{y}_0\) is the normal-class component after smoothing; adding \(\alpha\) increases the probability of normality as specified in the refurbishment mechanism.

### Masked Reconstruction and Multi-task Objective
\[
L_{\mathrm{rec}} = \| (X \odot M) - (\hat{X} \odot M) \|_2^2.
\]
**Explanation**: \(X\) is the input window, \(\hat{X}\) is its reconstruction, \(M\) is the anomaly mask, and \(\odot\) is elementwise multiplication.

\[
L = (1 - \lambda) L_{\mathrm{rec}} + \lambda L_{\mathrm{cls}}.
\]
**Explanation**: \(\lambda\) is `c_loss_ratio`, and \(L_{\mathrm{cls}}\) is the cross-entropy classification loss averaged over the batch.

### Anomaly Scoring (RedLamp)
\[
\tilde{s}_{\mathrm{mse}} = \mathrm{MinMax}(\mathrm{Conv}(s_{\mathrm{mse}})),\quad
\tilde{s}_{\mathrm{ce}} = \mathrm{MinMax}(\mathrm{Conv}(s_{\mathrm{ce}})).
\]
**Explanation**: \(s_{\mathrm{mse}}\) is the per-window mean-squared error score, \(s_{\mathrm{ce}}\) is a classification-derived score computed from predicted class distributions, and the convolution applies a smoothing window before min-max normalization.

\[
\mathrm{score} = \frac{1}{2} (\tilde{s}_{\mathrm{mse}} + \tilde{s}_{\mathrm{ce}}).
\]
**Explanation**: The final anomaly score is the average of normalized reconstruction and classification scores for each window.

## Code References
- `loaders/augmented_loader.py` - Augmented window creation, anomaly injection, and loader output structure.
- `loaders/dataset.py` - Dataset and entity abstractions with labels and masks.
- `bsc-thesis-ref-codebases/sto-transformer-main/code/common/transformer.py` - Stochastic attention modules and transformer blocks.
- `bsc-thesis-ref-codebases/MtsCID/model/Transformer.py` - Memory-guided multi-branch transformer.
- `bsc-thesis-ref-codebases/MtsCID/model/embedding.py` - Frequency-domain embedding and intra/inter attention layers.
- `bsc-thesis-ref-codebases/MtsCID/model/multi_attention_blocks.py` - Multi-scale attention via patch aggregation.
- `bsc-thesis-ref-codebases/RedLamp/models/meta.py` - Label refurbishment and multi-task loss.
- `bsc-thesis-ref-codebases/RedLamp/main.py` - Anomaly scoring definition.

## Historical Context (from documents/)
The directive in `documents/01-18-2026/teaching-integration-prompt.md` aligns with the current analysis scope, emphasizing modular boundaries for ablation, the loader interface, and adherence to the RedLamp scoring procedure.

## Open Questions
- The current memory-guided pathway in MtsCID constructs `mem_I` but does not use it; its role in the intended architecture is unspecified.
- The stochastic attention blocks operate on continuous embeddings but are originally embedded in token-based transformer scaffolding; the precise embedding interface for time-series adaptation is not defined in these sources.
- RedLamp uses softmax outputs with `CrossEntropyLoss`; whether this non-standard usage is intentional or incidental is not specified in the repository.
