**Summary:**
Your thesis idea is coherent and modular enough to start implementation now. The main design choices are now fixed more tightly: keep the thesis-facing hidden-state contract $H \in \mathbb{R}^{B \times L \times d_h}$, keep the real prediction heads only on the two fused task-specialized representations, use an objective-modular offline design with a small default objective and optional failure-mode-triggered regularizers, and treat NGD-style adaptation as an optional geometry-aware method mainly for the small online projector or adapter rather than for the whole model.

## SSOT synchronization note

This file is synchronized with the active design source of truth under `documents/design/`.

For the offline pre-training phase two-view contrastive design, use this companion specification as authoritative:

- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- In that document, treat these sections as the active implementation contract:
  - `CKA-Gated Per-Sample Fusion`
  - `Experiment Protocol v2`

Contrastive learning now appears in both phases but with different roles:

- offline pre-training phase: two-view InfoNCE between normal and injected views.
- online adaptation phase: contrastive alignment for adaptation with frozen-reference semantics.

## Detailed description of the thesis idea

You want to build a **multi-task time series anomaly detection** system for multivariate windows of length

$$
L = 20
$$

using a **TSLib-style** input and data-loading pipeline, starting with **SMD** as the first benchmark. TSLib is a reasonable base because it already provides a unified time-series code structure and includes anomaly-detection workflows. ([GitHub][1])

The core offline input-output contract we settled on is:

$$
X \in \mathbb{R}^{B \times L \times D}
$$

where (B) is batch size, (L=20), and (D) is the number of variates.
Your encoder should expose a thesis-facing hidden representation

$$
H = f_\theta(X) \in \mathbb{R}^{B \times L \times d_h},
$$

and optionally a pooled representation

$$
z = \mathrm{Pool}(H) \in \mathbb{R}^{B \times d_h}.
$$

That representation will be used in two prototype modules.

First, the **continuous prototype module**. The hidden representation queries a bank of continuous prototypes in an attention-like way. Conceptually,

$$
a_{\ell} = \mathrm{softmax}(q_\ell K^\top), \qquad
\hat h_\ell^{(c)} = \sum_{m=1}^{M_c} a_{\ell,m} p_m^{(c)},
$$

where $(p_m^{(c)})$ are continuous prototypes. This branch is meant to preserve smooth semantic structure and support reconstruction.

Second, the **discrete prototype module**. The current main design is a **distance-based top-$k$ codebook query** rather than a learned assignment-logit head. For one latent token $h_\ell$, let the discrete codebook be

$$
E = \{e_k\}_{k=1}^{M_d},
\qquad
e_k \in \mathbb{R}^{d_h}.
$$

Compute distances:

$$
d_{\ell,k} = \|h_\ell - e_k\|_2^2,
$$

select the nearest-codeword index set:

$$
S_k(h_\ell) = \operatorname{TopK}_k(-d_{\ell,\cdot}),
$$

then aggregate only over the selected codewords:

$$
\alpha_{\ell,k}
=
\frac{\exp(-d_{\ell,k}/\tau)}
{\sum_{j \in S_k(h_\ell)} \exp(-d_{\ell,j}/\tau)}
\qquad \text{for } k \in S_k(h_\ell),
$$

$$
\hat h_\ell^{(d)} = \sum_{k \in S_k(h_\ell)} \alpha_{\ell,k} e_k.
$$

The intended first settings are sparse query variants such as $k \in \{1, 3\}$. The discrete codebook is **frozen by default** in the main method after initialization; if an update variant is ever revisited, it should be treated only as a later ablation.

Historical note: an older draft design used a learned assignment-logit head with Gumbel-Softmax relaxation,

$$
\pi_\ell = \mathrm{softmax}\left(\frac{s_\ell + g}{\tau}\right), \qquad
\hat h_\ell^{(d)} = \sum_{k=1}^{M_d} \pi_{\ell,k} p_k^{(d)}.
$$

That older formulation is kept here only for design history. It is **not** the current main discrete-query design.

You then want to fuse these two prototype-derived representations into **task-specific task representations**, so that reconstruction and classification do not have to use the exact same representation:

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
$$

$$
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}.
$$

The reconstruction branch should make predictions only from $H_{\text{rec}}$, while the classification branch should make predictions only from $H_{\text{cls}}$. In the default thesis design, you do not move the real prediction paths onto branch-local heads attached directly to $\hat H^{(c)}$ or $\hat H^{(d)}$. You plan to inject artificial anomalies during training and perform anomaly-type classification, with the active repository default now using the 11 anomaly types from RedLamp while retaining CARLA as a mechanism reference for subsequence-oriented corruption. SMD remains the first experimental dataset. This overall architecture and motivation are consistent with your proposal draft, including the emphasis on continuous and discrete prototypes, task-specialized fusion, uncertainty, and online adaptation.

Then comes the **online adaptation phase**. For each online mini-batch of (k) windows,

$$
{x_1, \dots, x_k},
$$

you create two semantic augmentations for each sample:

$$
x_i^{A}, \quad x_i^{B}.
$$

View A is passed through a **frozen reference encoder**:

$$
r_i = f_{\text{ref}}(x_i^{A}),
$$

and view B is passed through a **partially trainable online encoder**:

$$
u_i = f_{\text{online}}(x_i^{B}).
$$

Then a lightweight projector maps the online representation into the reference space:

$$
\tilde u_i = g(u_i).
$$

You want a contrastive alignment loss that pulls $\tilde u_i$ toward $r_i$ for the same sample and pushes it away from other samples’ reference or mapped representations. In addition, you want a prototype-alignment objective so that the mapped online representations remain close to the frozen prototype spaces learned offline. The projector should be treated as a near-identity residual adapter, ideally $g(u)=u+F(u)$ with the last layer initialized near zero and warm-started offline before real streaming updates begin. If you later use NGD-style preconditioning, the intended scope is this small adapted subset such as the projector or a very light adapter, not the whole online model. This is also aligned with the proposal text and adaptation figure you uploaded.

On the codebase side, we agreed the safest engineering principle is: **freeze the encoder interface first, then build a minimal vertical slice before implementing the full model**. So the first practical milestone is not “full thesis architecture,” but:

$$
\text{SMD loader} \rightarrow \text{encoder adapter} \rightarrow \text{simple head} \rightarrow \text{train/eval loop}.
$$

That gives you a stable base for later prototype modules and online adaptation.

For this repository specifically, that vertical slice should also obey the strict rule in `codebase_preferences.md` that one model stays in one file. So the reconstruction baseline, the offline multitask thesis model, and the online adaptation model should each keep their forward path, scoring path, and stage-specific losses in the same model file rather than splitting them across separate task or loss files.

There should also be an explicit pre-Phase-4 gate in the implementation plan. Before attempting the online adaptation phase, phases 1 to 3 need to close the earlier debt around registry-only script construction, explicit RedLamp-default synthetic anomaly injection with CARLA-informed subsequence mechanics, and user-visible inspection of injected anomalies.

---

## Current consensus objective and training recipe

The current offline thesis core should be documented with an explicit objective surface, not only a high-level architecture sketch.

Let the encoder expose

$$
H = f_\theta(X) \in \mathbb{R}^{B \times L \times d_h}.
$$

The continuous prototype branch produces $\hat H^{(c)}$, and the discrete prototype branch produces $\hat H^{(d)}$ through the current **distance-based top-$k$ codebook query**. The two fused task representations remain

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
\qquad
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}.
$$

The default prediction paths are:

$$
\hat X = D_\phi(H_{\text{rec}}),
\qquad
\hat y = C_\psi(\mathrm{MeanPool}(H_{\text{cls}})).
$$

So the default thesis model keeps all task supervision on the fused paths. The branch outputs are still observable before fusion, but they are not separate default decoder or classifier inputs.

The formal design rule for the offline loss should now be:

- keep the objective **modular**
- keep the default objective **small**
- add extra regularizers only when a concrete failure mode is observed
- activate or deactivate every loss term through explicit configuration
- keep every active loss helper inside the same model file as the owning model

This is the most readable and ablation-friendly interpretation of `codebase_preferences.md`. It keeps the number of codepaths low while still allowing the repository to grow into a richer objective later.

The default offline baseline objective should therefore be

$$
\mathcal{L}_{\text{base}}
=
\mathcal{L}_{\text{recon}} +
\lambda_{\text{cls}} \mathcal{L}_{\text{cls}}.
$$

At the beginning of experimentation, this simple objective should be the active
default. Additional regularizers should remain disabled until concrete observed
failure modes justify turning them on.

The broader weighted-sum objective remains part of the design surface, but it is not the default starting point. It is the superset from which extra terms are enabled only when diagnostics justify them:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{recon}} +
\lambda_{\text{cls}} \mathcal{L}_{\text{cls}} +
\lambda_{\text{div}} \mathcal{L}_{\text{div}} +
\lambda_{\text{var}} \mathcal{L}_{\text{var}} +
\lambda_{\text{cov}} \mathcal{L}_{\text{cov}} +
\lambda_{\text{use}} \mathcal{L}_{\text{use}} +
\lambda_{\text{gate}} \mathcal{L}_{\text{gate}}.
$$

To make later implementation direct and readable, the full form of every optional component can still be stated with explicit tensor shapes below. The important design change is that these terms are now treated as a modular objective surface rather than as a mandatory default stack.

Let

$$
X \in \mathbb{R}^{B \times L \times D},
\qquad
H = f_\theta(X) \in \mathbb{R}^{B \times L \times d_h}.
$$

For the continuous prototype branch, let the learnable prototype bank be

$$
P^{(c)} = \left\{ p_k^{(c)} \right\}_{k=1}^{K_c},
\qquad
p_k^{(c)} \in \mathbb{R}^{d_h}.
$$

For each token index $(b,\ell)$, define

$$
s^{(c)}_{b\ell k}
=
\frac{\langle h_{b,\ell}, p_k^{(c)} \rangle}{\sqrt{d_h}}
\in \mathbb{R},
\qquad
a^{(c)}_{b\ell k}
=
\frac{\exp(s^{(c)}_{b\ell k})}{\sum_{j=1}^{K_c} \exp(s^{(c)}_{b\ell j})}.
$$

Then the continuous branch output is

$$
\hat h^{(c)}_{b,\ell}
=
\sum_{k=1}^{K_c} a^{(c)}_{b\ell k} p_k^{(c)}
\in \mathbb{R}^{d_h},
\qquad
\hat H^{(c)} \in \mathbb{R}^{B \times L \times d_h}.
$$

For the discrete branch, let the learnable codebook be

$$
E^{(d)} = \left\{ e_k^{(d)} \right\}_{k=1}^{K_d},
\qquad
e_k^{(d)} \in \mathbb{R}^{d_h},
$$

The current main-query interpretation is distance-based rather than assignment-logit-based. For one token $h_{b,\ell}$, define

$$
d_{b\ell k} = \|h_{b,\ell} - e_k^{(d)}\|_2^2.
$$

Let the selected nearest-codeword set be

$$
S_k(h_{b,\ell}) = \operatorname{TopK}_k(-d_{b\ell\cdot}).
$$

Using a temperature $\tau > 0$, compute sparse weights only over the selected set:

$$
\alpha_{b\ell k}
=
\frac{\exp(-d_{b\ell k}/\tau)}
{\sum_{j \in S_k(h_{b,\ell})}\exp(-d_{b\ell j}/\tau)}
\qquad \text{for } k \in S_k(h_{b,\ell}).
$$

Then

$$
\hat h^{(d)}_{b,\ell}
=
\sum_{k \in S_k(h_{b,\ell})} \alpha_{b\ell k} e_k^{(d)}
\in \mathbb{R}^{d_h},
\qquad
\hat H^{(d)} \in \mathbb{R}^{B \times L \times d_h}.
$$

The discrete codebook is initialized from class-balanced train-derived windows and is frozen by default in the main method.

Historical note: an older design variant instead used branch assignment logits

$$
q_{b,\ell} = W_d h_{b,\ell} + b_d \in \mathbb{R}^{K_d}.
$$

with Gumbel-Softmax relaxation

$$
\pi_{b\ell k}
=
\frac{\exp\left((q_{b\ell k} + g_{b\ell k}) / \tau\right)}
{\sum_{j=1}^{K_d} \exp\left((q_{b\ell j} + g_{b\ell j}) / \tau\right)},
\qquad
\pi_{b,\ell} \in \mathbb{R}^{K_d},
$$

where $g_{b\ell k}$ is i.i.d. Gumbel noise. Then

$$
\hat h^{(d)}_{b,\ell}
=
\sum_{k=1}^{K_d} \pi_{b\ell k} e_k^{(d)}
\in \mathbb{R}^{d_h},
\qquad
\hat H^{(d)} \in \mathbb{R}^{B \times L \times d_h}.

That assignment-logit formulation is retained only as a historical note and is **not** the current main design target.
$$

The fused task-specific representations are

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)} \in \mathbb{R}^{B \times L \times d_h},
\qquad
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)} \in \mathbb{R}^{B \times L \times d_h},
$$

with

$$
\alpha = \sigma(a) \in (0,1),
\qquad
\beta = \sigma(b) \in (0,1),
\qquad
a,b \in \mathbb{R}.
$$

The decoder and classifier heads are

$$
\hat X = D_\phi(H_{\text{rec}}) \in \mathbb{R}^{B \times L \times D},
$$

$$
c_b = \frac{1}{L} \sum_{\ell=1}^{L} H_{\text{cls},\,b,\ell} \in \mathbb{R}^{d_h},
\qquad
\hat y_b = C_\psi(c_b) \in \Delta^{C-1},
$$

where $y_b \in \{0,1\}^{C}$ is the one-hot anomaly-type pseudo-label for window $b$.

The reconstruction loss is

$$
\mathcal{L}_{\text{recon}}
=
\frac{1}{B L D} \left\| X - \hat X \right\|_F^2.
$$

The cross-entropy classification loss is

$$
\mathcal{L}_{\text{cls}}
=
-\frac{1}{B} \sum_{b=1}^{B} \sum_{c=1}^{C} y_{b,c} \log \hat y_{b,c}.
$$

For the pre-fusion regularizers, define

$$
N = B L,
$$

and flatten token axes after LayerNorm:

$$
Z^{(c)} = \operatorname{reshape}\!\left(\operatorname{LN}\!\left(\hat H^{(c)}\right)\right) \in \mathbb{R}^{N \times d_h},
\qquad
Z^{(d)} = \operatorname{reshape}\!\left(\operatorname{LN}\!\left(\hat H^{(d)}\right)\right) \in \mathbb{R}^{N \times d_h}.
$$

For each branch $m \in \{c,d\}$ and each feature dimension $j \in \{1,\dots,d_h\}$, define

$$
\mu_j^{(m)} = \frac{1}{N} \sum_{n=1}^{N} Z_{n,j}^{(m)},
\qquad
\sigma_j^{(m)} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} \left(Z_{n,j}^{(m)} - \mu_j^{(m)}\right)^2},
$$

$$
\tilde Z_{:,j}^{(m)}
=
\frac{Z_{:,j}^{(m)} - \mu_j^{(m)}}{\sigma_j^{(m)} + \varepsilon}
\in \mathbb{R}^{N}.
$$

The cross-branch decorrelation loss is

$$
C^{(cd)} = \frac{1}{N} \left(\tilde Z^{(c)}\right)^\top \tilde Z^{(d)} \in \mathbb{R}^{d_h \times d_h},
$$

$$
\mathcal{L}_{\text{div}}
=
\frac{1}{d_h^2} \left\| C^{(cd)} \right\|_F^2.
$$

For each branch $m \in \{c,d\}$, the variance-floor regularizer is

$$
\mathcal{L}_{\text{var}}^{(m)}
=
\frac{1}{d_h}
\sum_{j=1}^{d_h}
\left[
\max\left(0, \gamma - \operatorname{Std}\!\left(Z_{:,j}^{(m)}\right)\right)
\right]^2,
$$

and the total variance loss is

$$
\mathcal{L}_{\text{var}}
=
\mathcal{L}_{\text{var}}^{(c)} + \mathcal{L}_{\text{var}}^{(d)}.
$$

For each branch, define the within-branch correlation matrix

$$
C^{(m)} = \frac{1}{N} \left(\tilde Z^{(m)}\right)^\top \tilde Z^{(m)} \in \mathbb{R}^{d_h \times d_h}.
$$

Then the covariance reduction term is

$$
\mathcal{L}_{\text{cov}}^{(m)}
=
\frac{1}{d_h(d_h - 1)}
\sum_{i \neq j} \left(C^{(m)}_{ij}\right)^2,
$$

and the total covariance loss is

$$
\mathcal{L}_{\text{cov}}
=
\mathcal{L}_{\text{cov}}^{(c)} + \mathcal{L}_{\text{cov}}^{(d)}.
$$

For discrete code usage balancing under the current sparse-query design, use codeword-selection frequency or selected-weight statistics across all tokens. Historical note: the dense relaxed-assignment formula below belongs to the older Gumbel-Softmax variant and is not the main design target:

$$
\bar{\pi}_k
=
\frac{1}{B L}
\sum_{b=1}^{B} \sum_{\ell=1}^{L} \pi_{b\ell k},
\qquad
\bar{\pi} \in \mathbb{R}^{K_d}.
$$

Then

$$
\mathcal{L}_{\text{use}}
=
\sum_{k=1}^{K_d}
\left(
\bar{\pi}_k - \frac{1}{K_d}
\right)^2.
$$

To prevent early saturation of the fusion scalars, use gate entropy regularization

$$
\mathcal{L}_{\text{gate}}
=
\alpha \log \alpha + (1-\alpha) \log (1-\alpha)
+
\beta \log \beta + (1-\beta) \log (1-\beta),
$$

so that the weighted objective contributes

$$
\lambda_{\text{gate}} \mathcal{L}_{\text{gate}}
=
\lambda_{\text{gate}}
\Big[
\alpha \log \alpha + (1-\alpha) \log (1-\alpha)
+
\beta \log \beta + (1-\beta) \log (1-\beta)
\Big].
$$

Current design target: gate entropy regularization.
Current implementation status: the code still uses a barrier-style gate term and should be updated separately.

So the design-facing implementation target is: task supervision is applied only through $H_{\text{rec}}$ and $H_{\text{cls}}$, while $\mathcal{L}_{\text{div}}$, $\mathcal{L}_{\text{var}}$, $\mathcal{L}_{\text{cov}}$, $\mathcal{L}_{\text{use}}$, and $\mathcal{L}_{\text{gate}}$ remain optional pre-fusion regularizers acting on branch outputs, sparse discrete-query statistics, and fusion coefficients. If any regularizer depends on dense assignment probabilities, that dependency belongs only to the deprecated historical variant, not the current main design.

The activation policy for those optional terms should be fixed clearly.

Stage A is the default baseline. Train first with only

$$
\mathcal{L}_{\text{recon}} +
\lambda_{\text{cls}} \mathcal{L}_{\text{cls}}.
$$

This first stage answers the simplest scientific question: can the fused dual-branch model learn useful structure at all without auxiliary regularization?

Stage B adds the first anti-collapse extensions only if collapse is actually observed. The first-choice additions are

$$
\lambda_{\text{var}} \mathcal{L}_{\text{var}}
\qquad \text{and} \qquad
\lambda_{\text{cov}} \mathcal{L}_{\text{cov}},
$$

because they are the most standard anti-collapse ingredients in the current design.

Stage C adds failure-mode-specific regularizers only when the corresponding failure mode is observed:

- add $\lambda_{\text{div}} \mathcal{L}_{\text{div}}$ only if the continuous and discrete branches become too similar
- add $\lambda_{\text{use}} \mathcal{L}_{\text{use}}$ only if the discrete branch under-uses its codebook
- add $\lambda_{\text{gate}} \mathcal{L}_{\text{gate}}$ only if $\alpha$ and $\beta$ saturate too early, and treat it as an early-training stabilizer rather than a default permanent term

This means the repository should be designed for **objective modularity** or, equivalently, an **ablation-friendly objective surface**. Every loss term should have:

1. a dedicated helper in the model file
2. a clear diagnostic
3. a clear activation condition
4. a clear YAML-level configuration switch
5. a matching ablation when it is introduced

The central ablations should still be exact limiting cases of the same model:

$$
\text{continuous only: } \alpha=\beta=0,
\qquad
\text{discrete only: } \alpha=\beta=1,
\qquad
\text{fused: learn } \alpha,\beta.
$$

That is still the cleanest way to test whether the fused dual-branch thesis story actually adds value.

## Risks and problems discussed so far

### 1. Continuous and discrete prototype branches may become redundant

Even though the architecture contains both a continuous prototype path and a discrete prototype path, the two branches may end up learning nearly the same function.

If that happens, the dual-prototype idea adds complexity without real benefit. The architecture would still run, but the thesis claim that the two prototype types are complementary would be weak.

### 2. Task-specific fusion may collapse onto one branch

Even if you define

$$
H_{\text{rec}} \quad \text{and} \quad H_{\text{cls}}
$$

separately, a bad choice of fusion weights or losses may cause both tasks to rely mostly on one prototype branch. Then the second branch becomes decorative.

This is especially likely if one branch is much easier to optimize early in training.

### 3. Online adaptation may adapt to anomalies, not just drift

Because your online adaptation is label-free, the model cannot know whether the incoming mini-batch reflects genuine distribution shift or contains many anomalous windows.

So the adaptation signal may be contaminated. In the worst case, the model gradually aligns itself to anomalous structure and degrades future detection.

### 4. Using two encoder paths online is computationally expensive

Your online phase uses a frozen reference encoder and a partially trainable online encoder. Even if one is frozen, this still means two forward pathways plus projection plus contrastive loss, and potentially prototype-query losses too.

This may become expensive in latency and memory, especially once you move beyond first experiments.

### 5. A single online mini-batch gives weak and high-variance gradients

This was one of your strongest concerns, and it is valid.

A single online mini-batch is a small and noisy estimate of the current stream distribution. If you update even lightweight parameters from such a batch, the update may be high-variance and biased. Over time, those noisy updates can accumulate.

### 6. The projector initially “knows nothing”

At the beginning of online adaptation, the projector

$$
g(\cdot)
$$

has not yet learned how to map online representations into reference space.

You correctly noticed that this is not a trivial issue. If (g) is randomly initialized and trained online only, then the early updates are spent learning the map itself rather than adapting to drift.

### 7. The projector may drift over time and accumulate bias

Even after initialization, continual updates to the projector may gradually move it away from its original alignment role. That can happen because of temporary corrupt mini-batches, anomalies, or persistent noisy updates.

So the concern is not only how to initialize the projector, but also how to prevent it from slowly becoming a bad adapter.

### 8. Hard resetting the projector periodically may help or hurt

You raised the idea that projector parameters might need to be reset after a while.

That is partly right, but not universally right. A blind periodic hard reset may help if the stream contains temporary contamination, but it may hurt if the distribution shift is persistent and the projector has learned something useful. So reset itself becomes an experimental design choice.

### 9. Evaluation may be misleading if metrics are chosen carelessly

We briefly discussed that some anomaly-detection reporting conventions, especially point-adjusted metrics, can artificially inflate results. So the thesis should not rely only on such metrics.

The evaluation protocol must be clean enough that later reviewers cannot dismiss your numbers as metric artifacts.

### 9a. Full-test quantile thresholding leaks test-time information

A separate evaluation risk is threshold leakage. If the threshold is selected as a high quantile of all anomaly scores on the test sequence, then the evaluation has already seen future test windows before making the earliest anomaly decisions.

That protocol is not valid for streaming and is also weak for ordinary held-out testing, because the decision rule is calibrated on the same distribution it is supposed to evaluate.

The cleaner thesis protocol is:

1. learn the model on train;
2. calibrate the static threshold on train or validation reconstruction scores;
3. freeze and save that threshold;
4. evaluate validation, test, or stream windows with that fixed threshold unless the experiment is explicitly an adaptive-threshold experiment;
5. if the threshold adapts online, update it only from causal state available up to the current stream position.

Current codebase mismatch: the active offline evaluator computes the 95th-quantile threshold from the loader being evaluated. When that loader is the test loader, the reported threshold is not train/validation-calibrated. This remains an unresolved protocol problem and should be fixed before treating thresholded test metrics as final thesis evidence.

### 10. SMD handling must be done carefully

SMD is not just “one dataset” in the naive sense. It contains multiple machine subsets, and code organization, checkpoint naming, and metric aggregation should respect that structure.

This matters from the start because it affects your data loader and experiment organization.

### 11. Pretrained spectral-temporal encoders with open weights are surprisingly sparse

You wanted models that combine temporal and frequency analysis and still release reusable weights.

The main problem is that many good spectral-temporal papers release **code**, but not a clearly reusable, pretrained encoder checkpoint. From the public repos we checked, strong spectral-temporal candidates like **CATCH**, **TFMAE**, **TimesNet**, **TimeMixer**, and **FITS** are very useful, but they are more naturally treated as models you may need to train yourself rather than just plug in as pretrained encoders. CATCH is a public frequency-patching TSAD repo from the Decision Intelligence Lab, TFMAE is a public temporal-frequency masked autoencoder TSAD repo, and TimeMixer publicly mentions DFT-based decomposition, but these repos are not as clearly packaged as reusable pretrained encoder families in the way MOMENT or Timer are. ([GitHub][2])

### 12. MOMENT is practical, but patch-based and not perfectly matched to your notation

Among open-weight options, MOMENT is still one of the strongest practical backbones. Its public repo describes it as a family of open-source foundation models for general-purpose time-series analysis, and the Hugging Face checkpoint family is openly available. ([GitHub][3])

But the problem is that MOMENT is patch-based, not naturally time-step-based in the way your thesis notation wants. Also, the published config for a public checkpoint uses a fixed patch length rather than 1-point patches. So if you use MOMENT, you should not force it into a fundamentally different patching scheme unless you are ready to lose the benefit of pretrained compatibility. The public modeling/config information shows patch-based handling with default patch length 8 in the referenced model artifacts. ([Hugging Face][4])

### 13. MOMENT can handle many variates, but cross-variate interaction is limited in spirit

We discussed whether MOMENT can handle something like 150 variates. In practice, yes, that is feasible as an input scale.

But conceptually, MOMENT’s multivariate handling is not the same as a strongly cross-channel interaction backbone. So if you use it, some of the richer cross-variate relational modeling may need to happen later in your own prototype modules or adaptation blocks rather than being fully supplied by the backbone itself.

### 14. There is a tension between “use an open-weight backbone now” and “match the thesis idea exactly”

The more practical pretrained families with clear open weights are things like **MOMENT**, **Timer/OpenLTM**, and **TTM**. MOMENT is an open foundation model family; Timer’s official THUML repo explicitly says it provides official code, datasets, and checkpoints; TTM openly provides research-use model weights and markets itself as a tiny pre-trained family. ([GitHub][3])

But the models that match your **spectral-temporal TSAD** preference more closely are often the ones you may need to train yourself.

---

## Solutions and design decisions we discussed

### A. Freeze the thesis-facing representation contract first

This was the most important engineering decision.

Every backbone, pretrained or self-trained, should be wrapped so that it always outputs

$$
H \in \mathbb{R}^{B \times L \times d_h}.
$$

That way, your prototype modules, task heads, and online adaptation logic do not depend on the internal quirks of the encoder.

For patch-based or spectral encoders, you can define an adapter

$$
H = U(\tilde H),
$$

where (\tilde H) is the native hidden representation and (U) maps it to your thesis notation.

### B. Build a minimal vertical slice before the full architecture

The first code milestone should be:

$$
\text{SMD loader} \rightarrow \text{encoder adapter} \rightarrow \text{simple head} \rightarrow \text{train/eval loop}.
$$

Not full prototypes, not full online adaptation, not full uncertainty logic.

This keeps the codebase testable and reduces debugging chaos.

### B1. Translate the earlier generic roadmap into the current repository phases

The older generic `Phase 0-10` discussion is still useful, but it should now be
read through the repository's actual current state rather than copied as a
literal implementation sequence.

- Old Phase 0 and Phase 1 are already closed in the current repository through
  the frozen batch and encoder contracts, the SMD-first runnable vertical slice,
  and the registry-driven offline path.
- Old Phase 2 and Phase 3 are partly closed. The current repository already has
  a stable `configuration -> data -> model -> engine` structure and a readable
  one-model-one-file offline path, but broad multi-dataset generalization
  remains deferred.
- Old Phase 4 and Phase 6 now map to later streaming expansion rather than to
  missing foundations. The main deferred items are deterministic drift
  injection, a non-adaptive online baseline under drift, and broader streaming
  evaluation policies.
- Old Phase 5 is partly closed through explicit YAML experiments, resolved-config
  persistence, JSONL logging, and optional Weights & Biases support. The
  remaining reproducibility debt is DVC-backed versioning when synthetic or
  derived dataset artifacts become materialized outputs.
- Old Phase 7 is already realized in the current offline thesis model boundary:
  encoder, continuous branch, discrete branch, task-specific fusion, and
  reconstruction/classification heads live inside the active multitask model.
- Old Phase 8 is already realized as the accepted first online slice:
  projector-first, clean-stream-only, frozen-reference online adaptation.
- Old Phase 9 is mostly closed for the currently accepted offline and online
  slices through shape tests, one-step train tests, checkpoint round-trips, and
  stream-state tests. Future drift-specific leakage checks remain later-slice
  work.
- Old Phase 10 remains a documentation policy: generalize only after one stable
  paper-style result is secured on the current accepted path.

This translation is intentionally semantic rather than structural. The current
repository already has its fixed runtime layering and should not be rewritten to
imitate the older generic folder sketch.

### C. Use strong ablations to test branch specialization

To deal with possible redundancy between continuous and discrete prototype branches, the model should be tested in at least these variants:

$$
\text{continuous only},
\quad
\text{discrete only},
\quad
\text{fusion}.
$$

If fusion does not beat the single-branch variants, then the thesis claim about complementarity is weak and needs redesign.

### C1. Keep the default supervision only on the fused representations

The current consensus is that the real reconstruction and anomaly-type classification heads should stay on $H_{\text{rec}}$ and $H_{\text{cls}}$. The continuous and discrete branch outputs may still be exposed for regularization, monitoring, and ablations, but branch-local decoder or classifier heads should not be the default architecture.

### D. Use conservative online adaptation at first

Instead of updating many online parameters immediately, start conservatively.

A good first version is to update only the projector or only a very small subset of parameters. Then later you can test unfreezing higher layers. If you experiment with natural-gradient-style methods, treat them as geometry-aware tools mainly for that small adapted subset rather than as a default optimizer for the whole model.

This directly addresses the risk of one-batch high-variance updates.

### E. Treat the projector as a near-identity residual adapter

The best conceptual solution to the projector-initialization problem was:

$$
g(u) = u + F(u),
$$

where (F) is a small bottleneck MLP and its final layer is initialized to zero, so initially

$$
g(u) \approx u.
$$

That way, the projector starts as an identity map and learns only small corrections.

This fits your setup because the frozen reference encoder and the online encoder initially come from the same pretrained encoder, so the correct initial mapping is approximately identity anyway.

### F. Warm-start the projector offline before real online adaptation

We also discussed an offline calibration stage:

$$
\mathcal L_{\text{warm}} =
1 - \cos\big(g(f_{\text{online}}(x^B)), f_{\text{ref}}(x^A)\big).
$$

This teaches the projector the reference space before it sees true stream drift.

So the first online mini-batches are no longer responsible for teaching the projector basic alignment from scratch.

### G. Do not use blind periodic hard resets as the default

For the projector-drift problem, the better first strategy was not hard reset every (T) batches.

A safer plan is to keep an anchor copy of the original projector parameters

$$
\theta_g^{(0)}
$$

and use either soft restoration or anchor regularization:

$$
\mathcal L_{\text{anchor}} = \gamma |\theta_g - \theta_g^{(0)}|_2^2.
$$

Then use a **trigger-based** hard reset only if monitoring signals collapse, such as sharply worsening alignment loss or unstable anomaly scores.

### H. Compare reset strategies as an ablation

Since reset may help in some regimes and hurt in others, the clean thesis move is to compare:

$$
\text{no reset},
\quad
\text{soft restoration},
\quad
\text{trigger-based hard reset},
\quad
\text{periodic hard reset}.
$$

That turns your concern into an experiment rather than a guess.

### I. Handle compute cost with staged alternatives

For the two-encoder online cost, we discussed three possible regimes:

$$
\text{full dual-encoder baseline},
\quad
\text{shared encoder with stop-gradient reference branch},
\quad
\text{EMA teacher + one trainable student}.
$$

The full dual-encoder version is the clean conceptual baseline. The others are compute-relief variants you can compare later.

### J. Guard online updates against contamination

To reduce the chance that anomalous mini-batches corrupt the online adapter, you can later gate updates using only more reliable samples or mini-batches, for example those with lower uncertainty or lower anomaly scores.

We did not force a single method yet, but the principle is: **not every mini-batch deserves the same update strength**.

### J1. Keep the future online optimizer boundary explicit

If you later introduce Natural Gradient Descent, the clean design is not to let it touch the whole online stack by default. The preferred boundary is:

* frozen reference encoder parameters are never updated
* the projector or another very small adapter is the first NGD-eligible parameter group
* any partially trainable online-encoder subset remains a separate explicit decision

So the future online design should stay optimizer-agnostic at the loop level and optimizer-specific only at the parameter-group level. In other words, the repository should be prepared to swap `adamw` and `ngd` on the same adaptation boundary rather than treating NGD as a second adaptation architecture.

### K. Keep the evaluation protocol honest

Use ordinary point-wise and event-wise metrics clearly, and do not rely only on adjusted metrics that can exaggerate performance.

### L. For pretrained backbones, prefer practicality over perfect ideological match

The open-weight options we settled on were:

* **MOMENT** as the strongest practical pretrained backbone. ([GitHub][3])
* **TTM** as a lightweight backup for faster experimentation. ([Hugging Face][5])
* **Timer/OpenLTM** as a THUML-aligned backup with official checkpoints. ([GitHub][6])

For spectral-temporal models that better match your taste but are less clearly pretrained-and-reusable, we settled on these as train-yourself candidates:

* **CATCH** for frequency-patching TSAD. ([GitHub][2])
* **TFMAE** for temporal-frequency masked autoencoding in TSAD. ([GitHub][7])
* **TimesNet** as a strong TSLib-native spectral-temporal baseline. TSLib itself highlights broader interest in large time-series models and TimeMixer-like ecosystems. ([GitHub][1])
* **TimeMixer / TimeMixer++** as efficient multi-scale spectral-leaning candidates with DFT-based decomposition in the public repo. ([GitHub][8])
* **FITS** as an ultra-lightweight frequency-domain model that is realistic for a student to train from scratch. ([GitHub][9])

### M. If using MOMENT, do not force patch size 1

For public MOMENT checkpoints, patch-based handling is part of the pretrained interface, so changing to one time point per patch is not a clean reuse of the published weights. The safer solution is:

$$
X \xrightarrow{\text{MOMENT}} \tilde H \xrightarrow{U} H
$$

where (U) is your own adapter that converts patch-level output to your thesis-facing sequence representation.

### N. If spectral pretrained weights are too sparse, train a moderate model yourself and keep only the encoder

This became one of the most practical conclusions.

If open-weight spectral-temporal backbones remain too sparse, then a realistic student-friendly plan is:

1. Choose a modest spectral-temporal model such as **TFMAE**, **CATCH**, **TimesNet**, **TimeMixer**, or **FITS**.
2. Train it yourself on SMD or another source dataset.
3. Discard the original task head.
4. Reuse only the encoder in your thesis codebase.

This is especially attractive for **FITS** if you need cheap experiments, and for **TFMAE/CATCH** if you want a stronger conceptual match to your thesis.

---

## Current recommended plan

The cleanest current plan is:

1. Use **TSLib-style structure** and build the codebase around a stable encoder contract. ([GitHub][1])
2. Start with **SMD** and a minimal vertical slice.
3. Use **MOMENT** first if you want a practical open-weight starting point. ([GitHub][3])
4. Wrap MOMENT so the output still matches your thesis notation:

$$
H \in \mathbb{R}^{B \times L \times d_h}.
$$

5. Keep the real offline prediction heads only on the fused task-specialized states $H_{\text{rec}}$ and $H_{\text{cls}}$, and use the modular objective surface above with $\mathcal{L}_{\text{recon}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}}$ as the default thesis starting point.
6. Then add, in order: continuous prototypes, discrete prototypes, task-specific fusion, and only then online adaptation.
7. Start online adaptation conservatively by updating only the projector or another very small adapter, with near-identity initialization, offline warm-start, and optional NGD-style preconditioning only on that small subset.
8. In parallel, consider training **TFMAE**, **CATCH**, **TimesNet**, **TimeMixer**, or **FITS** yourself and extracting only the encoder if you decide that frequency-aware latent structure matters more than immediate open-weight reuse. ([GitHub][2])

---

## Carry-over brief for a new conversation

You can paste this into a new chat:

I am building a bachelor-thesis codebase for multivariate time-series anomaly detection on SMD with window length 20. The stable contract is $X \in \mathbb{R}^{B \times L \times D}$ and every encoder must expose $H \in \mathbb{R}^{B \times L \times d_h}$. The intended offline model has a continuous prototype branch with soft retrieval and a discrete prototype branch with distance-based top-$k$ codebook query. Their outputs are fused into two task-specialized states $H_{\text{rec}}$ and $H_{\text{cls}}$, and the real prediction heads stay only on those fused states. The discrete codebook is initialized from class-balanced train-derived windows and frozen by default in the main method. The offline objective is designed as a modular weighted-sum surface, but the default starting point is only $\mathcal{L}_{\text{recon}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}}$. Additional terms such as $\mathcal{L}_{\text{var}}$, $\mathcal{L}_{\text{cov}}$, $\mathcal{L}_{\text{div}}$, $\mathcal{L}_{\text{use}}$, and $\mathcal{L}_{\text{gate}}$ are added only when diagnostics reveal concrete failure modes, and each extra term must be justified by ablation. The main ablations are continuous-only, discrete-only, and fused. Later, an online adaptation phase uses two augmentations per incoming sample, a frozen reference encoder, a partially trainable online encoder, and a lightweight near-identity projector that is warm-started offline and aligned to the frozen reference and prototype geometry. NGD-style preconditioning is attractive mainly for that small adapted subset, not for the whole model. Current codebase decisions: freeze the encoder output contract first, build a minimal vertical slice first, keep one model per file, and keep the loss design ablation-friendly with explicit YAML-controlled objective modularity.

## Check

A good consistency check is this: if you remove the online adaptation block entirely, the remaining offline model is still a valid thesis core. If you remove one prototype branch, the model is still trainable. If you swap the encoder, the rest of the architecture should still work as long as the encoder wrapper preserves

$$
H \in \mathbb{R}^{B \times L \times d_h}.
$$

That modularity is a strong sign that the codebase plan is sound.

**Confidence:** High.
Main assumptions: you will keep the thesis-facing hidden-state contract fixed, and you are willing to treat pretrained-backbone choice as a practical engineering decision rather than as part of the thesis identity itself.

[1]: https://github.com/thuml/Time-Series-Library?utm_source=chatgpt.com "Time Series Library (TSLib)"
[2]: https://github.com/decisionintelligence/CATCH?utm_source=chatgpt.com "[ICLR 2025] CATCH: Channel-Aware Multivariate Time ..."
[3]: https://github.com/moment-timeseries-foundation-model/moment?utm_source=chatgpt.com "MOMENT: A Family of Open Time-series Foundation Models"
[4]: https://huggingface.co/AutonLab/MOMENT-1-large?utm_source=chatgpt.com "AutonLab/MOMENT-1-large"
[5]: https://huggingface.co/ibm-research/ttm-research-r2?utm_source=chatgpt.com "ibm-research/ttm-research-r2"
[6]: https://github.com/thuml/Large-Time-Series-Model?utm_source=chatgpt.com "thuml/Large-Time-Series-Model: Official code, datasets ..."
[7]: https://github.com/LMissher/TFMAE?utm_source=chatgpt.com "LMissher/TFMAE: [ICDE'2024] Temporal-Frequency ..."
[8]: https://github.com/kwuking/TimeMixer?utm_source=chatgpt.com "(ICLR'24) TimeMixer: Decomposable Multiscale Mixing for ..."
[9]: https://github.com/VEWOXIC/FITS?utm_source=chatgpt.com "VEWOXIC/FITS: FITS: Frequency Interpolation Time Series ..."

## Streaming simulation and drift generation design

To support the online adaptation part of the thesis, I will not rely on a single all-in-one simulator. Instead, I will use a **hybrid streaming framework** composed of a real-dataset stream wrapper, a drift injection layer, and an optional synthetic multivariate time-series generator.

This choice is more suitable for my datasets: **SMD, MSL, SMAP, SWaT, and UCR Anomaly Archive**. These datasets are naturally offline benchmark datasets, so the most practical solution is to convert them into sequential streams and then inject controlled drift or non-stationarity on top of them.

### Final choice

I choose the following stack as the main solution:

1. **River** as the primary Python streaming backbone.
2. **Custom dataset stream wrappers** for SMD, MSL, SMAP, SWaT, and UCR.
3. **Custom drift injection operators** as the main mechanism for generating non-stationarity.
4. **tsaug** as a helper library for simple time-series augmentation primitives.
5. **TSGM** as the optional synthetic multivariate time-series generator.
6. **MOA** as an optional benchmark-oriented concept drift engine, not as the main codebase dependency.

I do **not** choose scikit-multiflow as the main dependency because its project states that it is merging into River, so River is the cleaner long-term choice.

### Why this is the best fit for my case

My thesis does not only need synthetic streams. It needs a framework that can:

- evaluate models on realistic benchmark datasets in an online manner,
- simulate multiple kinds of drift on top of those datasets,
- remain modular enough to support different encoders, anomaly detectors, and online adaptation strategies,
- stay Python-first and easy to integrate into a reusable thesis codebase.

A pure synthetic generator would not be enough, because the final evaluation should still be performed on realistic benchmark streams derived from SMD, MSL, SMAP, SWaT, and UCR. Therefore, the best design is:

```text
offline dataset
-> sequential stream wrapper
-> sliding window construction
-> drift injection / non-stationarity injection
-> online model update and evaluation

## Current implementation note

The active offline thesis model now treats multilayer perceptron depth as an explicit symmetric contract across the encoder, reconstruction head, and classification head. The intended default is a three-linear-layer encoder and three-linear-layer heads inside the same self-contained model file, rather than a hard-coded two-layer design.
