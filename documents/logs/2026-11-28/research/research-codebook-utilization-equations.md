# Research: Equations for Discrete Codebook Utilization in Premier ML Venues

## 1. Purpose and Canonical Notation

This document records the standard equations for discrete codebook utilization from premier machine-learning venues and official reference codebases.

Let:
- $K \in \mathbb{N}^+$ denote the total codebook size (number of discrete entries).
- $N \in \mathbb{N}^+$ denote the total number of quantized vectors or tokens in the evaluation scope (one mini-batch, one epoch, or a full evaluation split).
- $z_i \in \{1, \dots, K\}$ denote the discrete codebook index assigned to token $i \in \{1, \dots, N\}$.
- $c_k = \sum_{i=1}^{N} \mathbb{I}(z_i = k)$ denote the selection count of codebook entry $k \in \{1, \dots, K\}$, where $\mathbb{I}(\cdot)$ is the indicator function.
- $p_k = c_k / N$ denote the empirical usage probability of codebook entry $k$, with $\sum_{k=1}^{K} p_k = 1$.

---

## 2. Primary Equations from Premier Venues and Official Codebases

### 2.1 Codebook Perplexity ($\text{PPL}$)

Codebook perplexity measures the effective number of uniformly used codebook entries. It is the standard diagnostic for codebook collapse in VQ-VAE, VQ-VAE-2, SoundStream, and EnCodec.

$$H(p) = -\sum_{k=1}^{K} p_k \log p_k$$

$$\text{PPL} = \exp(H(p)) = \exp\left(-\sum_{k=1}^{K} p_k \log p_k\right)$$

**Bounds and interpretation:**
- Minimum value: $\text{PPL} = 1$ when all tokens map to one codebook entry ($p_{k^*} = 1$, complete collapse).
- Maximum value: $\text{PPL} = K$ when all $K$ entries have equal probability ($p_k = 1/K$, uniform usage).
- Logarithm base: natural logarithm ($\ln$), matching $\exp(\cdot)$.

**Official reference implementations:**
- DeepMind `sonnet` (`sonnet/src/nets/vqvae.py`, for VQ-VAE by van den Oord et al., NeurIPS 2017, [arXiv:1711.00937](https://arxiv.org/abs/1711.00937), and VQ-VAE-2 by Razavi et al., NeurIPS 2019):
  ```python
  avg_probs = tf.reduce_mean(encodings, 0)
  perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
  ```
  Here `encodings` is the $N \times K$ one-hot assignment matrix, `avg_probs` is $p \in \mathbb{R}^K$, and `1e-10` prevents $\log(0)$.
- Standard PyTorch translation (e.g., `vector-quantize-pytorch`):
  ```python
  perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
  ```
- Neural audio codecs: SoundStream (Zeghidour et al., IEEE/ACM TASLP 2021) and EnCodec (Défossez et al., TMLR 2023) monitor per-codebook perplexity across Residual Vector Quantization (RVQ) stages to verify that each residual codebook remains active.

---

### 2.2 Active Codebook Usage / Utilization Rate ($U_{\text{active}}$)

Active codebook usage measures the fraction (or percentage) of codebook entries selected at least once over the evaluation scope $N$.

$$U_{\text{active}} = \frac{\left|\{k \in \{1, \dots, K\} : c_k > 0\}\right|}{K} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}(c_k > 0)$$

$$U_{\text{active}}(\%) = 100 \times U_{\text{active}}$$

**Bounds and evaluation scope:**
- Range: $U_{\text{active}} \in [1/K, 1]$ (or $[100/K\%, 100\%]$).
- Scope constraint: On a single mini-batch of size $N < K$, at most $N$ entries can be active, so $U_{\text{active}} \le N/K$. Premier papers therefore compute $U_{\text{active}}$ over an epoch or a full validation set (or report the exact batch/eval token count).

**Primary sources:**
- ViT-VQGAN (Yu et al., ICLR 2022, [arXiv:2110.04627](https://arxiv.org/abs/2110.04627)): reports codebook usage over $K = 8192$ entries using factorized codes and $\ell_2$-normalized lookup.
- Descript Audio Codec / DAC (Kumar et al., NeurIPS 2023): reports codebook utilization rate across RVQ codebooks.
- Finite Scalar Quantization / FSQ (Mentzer et al., ICLR 2024): defines the implicit codebook size as the Cartesian product of per-dimension scalar levels $L_i$,
  $$V = \prod_{i=1}^{d} L_i$$
  and reports near-100% active codebook usage across all tested configurations.
- SimVQ (Zhu et al., ICCV 2025, [arXiv:2411.02038](https://arxiv.org/abs/2411.02038)): evaluates codebook utilization rate up to $K = 262{,}144$ entries and achieves $100\%$ utilization via linear reparameterization $C W$.

---

### 2.3 Normalized Entropy ($H_{\text{norm}}$) and Perplexity Utilization Ratio ($\text{PPL}/K$)

Normalized entropy divides the empirical Shannon entropy by the maximum possible entropy $\log K$ so that utilization is comparable across different codebook sizes $K$.

$$H_{\text{norm}} = \frac{H(p)}{\log K} = -\frac{1}{\log K} \sum_{k=1}^{K} p_k \log p_k \in [0, 1]$$

A closely related normalized metric is the Perplexity Utilization Ratio (also called Codebook Volume Utilization, CVU):

$$U_{\text{PPL}} = \frac{\text{PPL}}{K} = \frac{\exp(H(p))}{K} \in \left[\frac{1}{K}, 1\right]$$

**Key distinction from $U_{\text{active}}$:**
- $U_{\text{active}}$ only checks whether $c_k > 0$. It equals $1.0$ even if $99\%$ of tokens use one entry and the remaining $K-1$ entries each appear once.
- $H_{\text{norm}}$ and $\text{PPL}/K$ penalize skewed distributions and equal $1.0$ only when usage is uniform ($p_k = 1/K$).

---

### 2.4 Dead-Code Ratio ($R_{\text{dead}}$) and EMA Dead-Code Threshold

The dead-code ratio is the complement of active codebook usage:

$$R_{\text{dead}} = 1 - U_{\text{active}} = \frac{\left|\{k \in \{1, \dots, K\} : c_k = 0\}\right|}{K}$$

In Meta's `audiocraft` (`audiocraft/quantization/core_vq.py`, `EuclideanCodebook`), dead codes during training are detected using an Exponential Moving Average (EMA) of cluster sizes rather than hard zero counts. Entries whose EMA cluster size falls below `threshold_ema_dead_code` are replaced with randomly sampled latent vectors from the current batch (used in SoundStream and EnCodec).

---

### 2.5 Entropy Regularization Objective in LFQ / MAGVIT-v2

In MAGVIT-v2 / Lookup-Free Quantization (Yu et al., ICLR 2024), the authors encourage high codebook utilization over large implicit binary codebooks ($K = 2^d$, e.g., $2^{18}$) using an explicit entropy loss:

$$\mathcal{L}_{\text{entropy}} = H(\mathbb{E}[q(z)]) - \mathbb{E}[H(q(z))]$$

where $q(z)$ is the per-token probability distribution over codebook entries. Maximizing $H(\mathbb{E}[q(z)])$ pushes the marginal batch distribution toward uniform usage, while minimizing $\mathbb{E}[H(q(z))]$ keeps each individual token assignment sharp.

---

### 2.6 Soft and Top-$k$ Weighted Utilization

For architectures with soft assignments, Gumbel-Softmax relaxations, Mixture-of-Experts (MoE) routing, or top-$k$ attention over memory slots (where each token $i$ assigns weight $w_{i,k} \ge 0$ with $\sum_{k=1}^{K} w_{i,k} = 1$):

1. **Mass-weighted marginal probability:**
   $$\bar{w}_k = \frac{1}{N} \sum_{i=1}^{N} w_{i,k}$$
   $$\text{PPL}_{\text{soft}} = \exp\left(-\sum_{k=1}^{K} \bar{w}_k \log \bar{w}_k\right), \qquad H_{\text{norm, soft}} = -\frac{1}{\log K} \sum_{k=1}^{K} \bar{w}_k \log \bar{w}_k$$

2. **Top-$k$ hard selection frequency** (for top-$k$ routing with $k_{\text{top}}$ selected slots per token):
   $$p_k^{\text{top-}k} = \frac{1}{N \cdot k_{\text{top}}} \sum_{i=1}^{N} \mathbb{I}(k \in \text{TopK}_i), \qquad U_{\text{active}}^{\text{top-}k} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{I}\left(\sum_{i=1}^{N} \mathbb{I}(k \in \text{TopK}_i) > 0\right)$$

Reporting both $\text{PPL}_{\text{soft}}$ (or $H_{\text{norm, soft}}$) and $U_{\text{active}}$ separates slot reachability from attention-weight concentration.

---

## 3. Local Reference Codebase Check (`bsc-thesis-ref-codebases/DALL-E-master`)

Inspection of `bsc-thesis-ref-codebases/DALL-E-master` (Ramesh et al., ICML 2021, [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)):
- `dall_e/encoder.py` defines `vocab_size: int = attr.ib(default=8192, ...)` (`dall_e/encoder.py#L60`) and projects the final feature map to `self.vocab_size` channels (`dall_e/encoder.py#L177`).
- `dall_e/decoder.py` defines `vocab_size: int = attr.ib(default=8192, ...)` (`dall_e/decoder.py#L61`), builds the input convolution over `self.vocab_size` channels (`dall_e/decoder.py#L87`), and validates `x.shape[1] != self.vocab_size` (`dall_e/decoder.py#L208-L210`).
- **Finding:** `DALL-E-master` is an inference-only discrete VAE (dVAE) release. It contains no training loop, no codebook perplexity calculation, and no codebook utilization logging.

---

## 4. Summary Comparison

| Metric | Equation | Range | Sensitive to Skew? | Primary Sources |
| :--- | :--- | :--- | :--- | :--- |
| Codebook Perplexity ($\text{PPL}$) | $\exp(-\sum_{k=1}^K p_k \log p_k)$ | $[1, K]$ | Yes | VQ-VAE (NeurIPS 2017), `sonnet` `vqvae.py`, VQ-VAE-2 (NeurIPS 2019), SoundStream (2021), EnCodec (TMLR 2023) |
| Active Codebook Usage ($U_{\text{active}}$) | $\frac{1}{K}\sum_{k=1}^K \mathbb{I}(c_k > 0)$ | $[1/K, 1]$ | No | ViT-VQGAN (ICLR 2022), DAC (NeurIPS 2023), FSQ (ICLR 2024), SimVQ (ICCV 2025) |
| Normalized Entropy ($H_{\text{norm}}$) | $-\frac{1}{\log K}\sum_{k=1}^K p_k \log p_k$ | $[0, 1]$ | Yes | Standard information-theoretic normalization across varying $K$ |
| Perplexity Ratio ($U_{\text{PPL}}$) | $\text{PPL} / K$ | $[1/K, 1]$ | Yes | Direct normalized effective-vocabulary ratio |
| Dead-Code Ratio ($R_{\text{dead}}$) | $1 - U_{\text{active}}$ | $[0, 1 - 1/K]$ | No | `audiocraft` `core_vq.py` (`threshold_ema_dead_code`), SoundStream, EnCodec |
| Soft / Top-$k$ Perplexity ($\text{PPL}_{\text{soft}}$) | $\exp(-\sum_{k=1}^K \bar{w}_k \log \bar{w}_k)$ | $[1, K]$ | Yes | Soft VQ, Gumbel-Softmax, MoE / top-$k$ attention routing |
