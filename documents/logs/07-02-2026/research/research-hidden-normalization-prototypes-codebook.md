---
date: 2026-07-02
researcher: Codex
git_commit: unknown
branch: unknown
repository: bachelor-thesis-2026
topic: "Hidden normalization, prototype normalization, and codebook normalization"
tags:
  - research
  - normalization
  - prototypes
  - codebook
  - redlamp
  - thesis_multitask
  - online_adaptation
status: complete
last_updated: 2026-07-02
last_updated_by: Codex
---

# Research: Hidden normalization, prototype normalization, and codebook normalization

**Date**: 2026-07-02
**Researcher**: Codex
**Repository**: `bachelor-thesis-2026`

## Research Question

Check whether the original RedLamp codebase and the Shen et al. H-PAD paper normalize hidden vectors before prototype query, and whether the stored continuous prototypes and discrete codebook vectors are unit-norm vectors. Also verify the current thesis implementation and online adaptation code paths.

## Summary

The current codebase does not use one single normalization rule everywhere.

In the original RedLamp baseline, there is no prototype-query mechanism at all, and the hidden tensor is not L2-normalized before it is consumed by the decoder or classifier. Any normalization that appears in the file is ordinary module normalization such as batch normalization or layer normalization inside submodules, not explicit unit-norm projection.

In `src/models/thesis_multitask.py`, the encoder output `hidden` is also not normalized at the point of creation, but it is normalized before both prototype lookups. The continuous prototype bank and discrete codebook are also normalized to unit norm when they are initialized or updated. So in the current thesis model, the memory states are unit-norm vectors, and the query vectors are normalized before similarity lookup.

In `src/models/online_adaptation.py`, `score_from_hidden()` does not normalize `hidden` itself, but it passes `hidden` into the thesis model lookup helpers, and those helpers normalize internally before scoring. So the online score path uses normalized lookup even though the adapter function does not visibly perform normalization.

In the H-PAD paper by Shen et al., the prototype query and reconstruction equations are written with inner product similarity and softmax weights. I did not find an explicit statement that query vectors or prototype vectors are projected to unit norm before lookup. The paper also defines the prototype update rule as a gated weighted sum, which does not preserve unit norm in general. Based on the formulas shown in the PDF, the safe conclusion is that unit-norm normalization is not specified explicitly in the paper.

## Detailed Findings

### 1. Original RedLamp baseline

The baseline model in [`src/models/redlamp_baseline.py`](../../../../../../src/models/redlamp_baseline.py) is a simple encoder-decoder-classifier model. The `hidden` tensor is produced by the encoder and then passed directly to the decoder and classifier:

```text
x -> encoder -> hidden -> decoder
                     \-> classifier
```

Relevant code:

- `hidden = self.encoder(x_tensor)` at [`src/models/redlamp_baseline.py:352`](../../../../../../src/models/redlamp_baseline.py#L352)
- `flattened_classification_hidden = hidden.reshape(...)` at [`src/models/redlamp_baseline.py:358-361`](../../../../../../src/models/redlamp_baseline.py#L358-L361)
- `recon = self.decoder(hidden)` at [`src/models/redlamp_baseline.py:362`](../../../../../../src/models/redlamp_baseline.py#L362)
- `logits = self.classification_head(flattened_classification_hidden)` at [`src/models/redlamp_baseline.py:363`](../../../../../../src/models/redlamp_baseline.py#L363)

What is present:

- The file defines `SimpleWindowCnnEncoder`, but that encoder only uses `Conv1d`, `ReLU`, and `Dropout`.
- The file also uses `BatchNorm1d` or `LayerNorm` indirectly inside some helper modules, but those are not the same as explicit L2 normalization.

What is absent:

- No `F.normalize(hidden, dim=-1, ...)` at the model output.
- No `torch.linalg.vector_norm(hidden)` used to divide the hidden vector.
- No prototype bank.
- No codebook.
- No continuous or discrete query path.

Conclusion for RedLamp baseline:

- `hidden` is **not** explicitly normalized to unit norm.
- There is **no prototype query mechanism** in this file, so normalization before prototype query does not apply here.

### 2. Current thesis model: `src/models/thesis_multitask.py`

The thesis model follows a different contract. The encoder output `hidden` is created first and then normalized inside the prototype lookup helpers.

Relevant code:

- `hidden = self.network(batch["x"])` at [`src/models/thesis_multitask.py:193`](../../../../../../src/models/thesis_multitask.py#L193)
- `_continuous_prototype_lookup()` begins with `normalized_hidden = self._normalize_hidden_for_memory(hidden)` at [`src/models/thesis_multitask.py:1877`](../../../../../../src/models/thesis_multitask.py#L1877)
- `_discrete_prototype_lookup()` begins with `normalized_hidden = self._normalize_hidden_for_memory(hidden)` at [`src/models/thesis_multitask.py:1922`](../../../../../../src/models/thesis_multitask.py#L1922)

So the hidden tensor itself is not changed at encoder output, but the lookup path uses normalized hidden vectors.

ASCII flow:

```text
hidden from encoder
   |
   +--> normalize(hidden) -> continuous lookup
   |
   +--> normalize(hidden) -> discrete lookup
```

#### 2.1 Continuous prototype bank

The continuous prototype memory is normalized.

Relevant code:

- `_normalize_memory_vectors()` returns `F.normalize(vectors, dim=-1, eps=self.memory_norm_epsilon)` at [`src/models/thesis_multitask.py:1462-1463`](../../../../../../src/models/thesis_multitask.py#L1462-L1463)
- `_select_covering_vectors()` normalizes candidate vectors before selecting seeds at [`src/models/thesis_multitask.py:1476-1485`](../../../../../../src/models/thesis_multitask.py#L1476-L1485)
- During initialization, `continuous_seed_vectors` are selected from normalized vectors and copied into `continuous_prototype_bank` at [`src/models/thesis_multitask.py:1620-1626`](../../../../../../src/models/thesis_multitask.py#L1620-L1626)
- During update, `weighted_hidden_summary` is normalized, the gated interpolation result is normalized again, and then copied back into the bank at [`src/models/thesis_multitask.py:1713-1736`](../../../../../../src/models/thesis_multitask.py#L1713-L1736)

Conclusion:

- Continuous prototype vectors are **unit-norm normalized** in the stored bank.

#### 2.2 Discrete codebook

The discrete codebook is also normalized.

Relevant code:

- During initialization, codebook seeds are chosen via `_select_covering_vectors()` and copied into `self.discrete_codebook` at [`src/models/thesis_multitask.py:1627-1669`](../../../../../../src/models/thesis_multitask.py#L1627-L1669)
- During update, the EMA codebook estimate is normalized before being copied back into `self.discrete_codebook` at [`src/models/thesis_multitask.py:1769-1803`](../../../../../../src/models/thesis_multitask.py#L1769-L1803)
- The update path explicitly calls `self._normalize_memory_vectors(normalized_codebook)` before storing the codebook at [`src/models/thesis_multitask.py:1791-1798`](../../../../../../src/models/thesis_multitask.py#L1791-L1798)

Conclusion:

- Discrete codebook vectors are **unit-norm normalized** in the stored state.

#### 2.3 Query mechanics in the thesis model

The continuous branch uses cosine-like inner product on normalized vectors:

- `attention_logits = torch.einsum("blh,kh->blk", normalized_hidden, memory_bank_for_read) / math.sqrt(self.hidden_dim)` at [`src/models/thesis_multitask.py:1885-1890`](../../../../../../src/models/thesis_multitask.py#L1885-L1890)

The discrete branch uses normalized hidden vectors and normalized codebook vectors in both query modes:

- `cosine_topk` path at [`src/models/thesis_multitask.py:1934-1958`](../../../../../../src/models/thesis_multitask.py#L1934-L1958)
- `gumbel_softmax` path at [`src/models/thesis_multitask.py:1959-1968`](../../../../../../src/models/thesis_multitask.py#L1959-L1968)

Conclusion:

- In the thesis model, the query vectors are normalized before both continuous and discrete lookup.
- The stored continuous prototypes and discrete codebook are also normalized to unit norm.

### 3. Online adaptation: `src/models/online_adaptation.py`

The online adapter does not normalize `hidden` inside `score_from_hidden()`. It passes the tensor directly to the thesis model lookup helpers:

- `reference_encoder.score_from_hidden(projected_hidden, batch["x"])` is called in the forward path.
- Inside `score_from_hidden()`, the model calls `_continuous_prototype_lookup(hidden, stage_name="test")` and `_discrete_prototype_lookup(hidden, stage_name="test")` at [`src/models/online_adaptation.py:44-58`](../../../../../../src/models/online_adaptation.py#L44-L58)

Because those lookup helpers normalize internally, the online scoring path uses normalized vectors before scoring even though `score_from_hidden()` itself does not visibly normalize `hidden`.

Conclusion:

- `hidden` is **not normalized in the adapter function itself**.
- `hidden` **is normalized before prototype scoring** through the thesis model helper functions.

### 4. H-PAD paper by Shen et al.

The PDF text shows prototype and query equations using inner product similarity and softmax weighting.

Relevant extracted lines from the PDF:

- Patch prototype query weights use `exp(<b_i^z, q_j^z>/tau)` and softmax at lines 252-271 of the extracted text.
- Query reconstruction uses weighted sums at lines 282-314.
- Period prototype query weights use `exp(<b^p, q_j^p>/tau)` and softmax at lines 340-380.
- The loss section defines reconstruction loss, entropy sparsity loss, and period loss, but does not add an explicit L2 unit-norm constraint on vectors at lines 395-424.

Relevant extracted text file:

- [`/private/tmp/hpad-paper-XXXXXX.txt`](file:///private/tmp/hpad-paper-XXXXXX.txt) was generated from the PDF for inspection during this research session.

Conclusion from the paper:

- The paper **does not explicitly state** that query vectors are L2-normalized before prototype lookup.
- The paper **does not explicitly state** that prototypes are constrained to unit norm.
- The update rules are gated weighted sums, so prototype norms can change during training.

## Final Conclusion

Current state of the codebase:

1. **RedLamp baseline**:
   - hidden is **not** L2-normalized
   - no prototype or codebook query exists

2. **`thesis_multitask.py`**:
   - hidden is **normalized before prototype lookup**
   - continuous prototypes are **unit-norm normalized**
   - discrete codebook vectors are **unit-norm normalized**

3. **`online_adaptation.py`**:
   - the adapter does **not** normalize hidden directly
   - scoring still uses normalized lookup through the thesis model

4. **H-PAD paper**:
   - uses inner-product similarity and softmax weights
   - does **not explicitly specify** unit-norm normalization for queries or prototypes

## Open Questions

- The PDF text extracted in this session does not show an explicit unit-norm constraint, but I did not inspect any supplementary appendix figures or source code.
- If the thesis note needs an even stricter statement about the paper, the next step would be to inspect the appendix or any official implementation if available.
