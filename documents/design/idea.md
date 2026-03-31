**Summary:**
Your thesis idea is coherent and modular enough to start implementation now. The main bottlenecks we identified are not “the idea makes no sense,” but rather engineering and statistical risks around online adaptation, projector stability, branch specialization, and the scarcity of reusable spectral-temporal pretrained encoders.

## Detailed description of the thesis idea

You want to build a **multi-task time series anomaly detection** system for multivariate windows of length

[
L = 100
]

using a **TSLib-style** input and data-loading pipeline, starting with **SMD** as the first benchmark. TSLib is a reasonable base because it already provides a unified time-series code structure and includes anomaly-detection workflows. ([GitHub][1])

The core offline input-output contract we settled on is:

[
X \in \mathbb{R}^{B \times L \times D}
]

where (B) is batch size, (L=100), and (D) is the number of variates.
Your encoder should expose a thesis-facing hidden representation

[
H = f_\theta(X) \in \mathbb{R}^{B \times L \times d_h},
]

and optionally a pooled representation

[
z = \mathrm{Pool}(H) \in \mathbb{R}^{B \times d_h}.
]

That representation will be used in two prototype modules.

First, the **continuous prototype module**. The hidden representation queries a bank of continuous prototypes in an attention-like way. Conceptually,

[
a_{\ell} = \mathrm{softmax}(q_\ell K^\top), \qquad
\hat h_\ell^{(c)} = \sum_{m=1}^{M_c} a_{\ell,m} p_m^{(c)},
]

where (p_m^{(c)}) are continuous prototypes. This branch is meant to preserve smooth semantic structure and support reconstruction.

Second, the **discrete prototype module**. The hidden representation also queries a discrete codebook using something like a Gumbel-Softmax relaxation:

[
\pi_\ell = \mathrm{softmax}!\left(\frac{s_\ell + g}{\tau}\right), \qquad
\hat h_\ell^{(d)} = \sum_{k=1}^{M_d} \pi_{\ell,k} p_k^{(d)}.
]

This branch is meant to encourage quantized, more categorical structure.

You then want to fuse these two prototype-derived representations into **task-specific task representations**, so that reconstruction and classification do not have to use the exact same representation:

[
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
]

[
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}.
]

The reconstruction branch tries to reconstruct the input window, while the classification branch is trained using **synthetic anomaly injection**. You plan to inject artificial anomalies during training and perform anomaly-type classification, initially inspired by the anomaly taxonomy used in CARLA-style TSAD work, while keeping SMD as the first experimental dataset. This overall architecture and motivation are consistent with your proposal draft, including the emphasis on continuous and discrete prototypes, task-specialized fusion, uncertainty, and online adaptation. 

Then comes the **online adaptation phase**. For each online mini-batch of (k) windows,

[
{x_1, \dots, x_k},
]

you create two semantic augmentations for each sample:

[
x_i^{A}, \quad x_i^{B}.
]

View A is passed through a **frozen reference encoder**:

[
r_i = f_{\text{ref}}(x_i^{A}),
]

and view B is passed through a **partially trainable online encoder**:

[
u_i = f_{\text{online}}(x_i^{B}).
]

Then a lightweight projector maps the online representation into the reference space:

[
\tilde u_i = g(u_i).
]

You want a contrastive alignment loss that pulls (\tilde u_i) toward (r_i) for the same sample and pushes it away from other samples’ reference or mapped representations. In addition, you want a prototype-alignment objective so that the mapped online representations remain close to the frozen prototype spaces learned offline. This is also aligned with the proposal text and adaptation figure you uploaded. 

On the codebase side, we agreed the safest engineering principle is: **freeze the encoder interface first, then build a minimal vertical slice before implementing the full model**. So the first practical milestone is not “full thesis architecture,” but:

[
\text{SMD loader} \rightarrow \text{encoder adapter} \rightarrow \text{simple head} \rightarrow \text{train/eval loop}.
]

That gives you a stable base for later prototype modules and online adaptation.

---

## Risks and problems discussed so far

### 1. Continuous and discrete prototype branches may become redundant

Even though the architecture contains both a continuous prototype path and a discrete prototype path, the two branches may end up learning nearly the same function.

If that happens, the dual-prototype idea adds complexity without real benefit. The architecture would still run, but the thesis claim that the two prototype types are complementary would be weak.

### 2. Task-specific fusion may collapse onto one branch

Even if you define

[
H_{\text{rec}} \quad \text{and} \quad H_{\text{cls}}
]

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

[
g(\cdot)
]

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

[
H \in \mathbb{R}^{B \times L \times d_h}.
]

That way, your prototype modules, task heads, and online adaptation logic do not depend on the internal quirks of the encoder.

For patch-based or spectral encoders, you can define an adapter

[
H = U(\tilde H),
]

where (\tilde H) is the native hidden representation and (U) maps it to your thesis notation.

### B. Build a minimal vertical slice before the full architecture

The first code milestone should be:

[
\text{SMD loader} \rightarrow \text{encoder adapter} \rightarrow \text{simple head} \rightarrow \text{train/eval loop}.
]

Not full prototypes, not full online adaptation, not full uncertainty logic.

This keeps the codebase testable and reduces debugging chaos.

### C. Use strong ablations to test branch specialization

To deal with possible redundancy between continuous and discrete prototype branches, the model should be tested in at least these variants:

[
\text{continuous only},
\quad
\text{discrete only},
\quad
\text{fusion}.
]

If fusion does not beat the single-branch variants, then the thesis claim about complementarity is weak and needs redesign.

### D. Use conservative online adaptation at first

Instead of updating many online parameters immediately, start conservatively.

A good first version is to update only the projector or only a very small subset of parameters. Then later you can test unfreezing higher layers.

This directly addresses the risk of one-batch high-variance updates.

### E. Treat the projector as a near-identity residual adapter

The best conceptual solution to the projector-initialization problem was:

[
g(u) = u + F(u),
]

where (F) is a small bottleneck MLP and its final layer is initialized to zero, so initially

[
g(u) \approx u.
]

That way, the projector starts as an identity map and learns only small corrections.

This fits your setup because the frozen reference encoder and the online encoder initially come from the same pretrained encoder, so the correct initial mapping is approximately identity anyway.

### F. Warm-start the projector offline before real online adaptation

We also discussed an offline calibration stage:

[
\mathcal L_{\text{warm}} =
1 - \cos!\big(g(f_{\text{online}}(x^B)), f_{\text{ref}}(x^A)\big).
]

This teaches the projector the reference space before it sees true stream drift.

So the first online mini-batches are no longer responsible for teaching the projector basic alignment from scratch.

### G. Do not use blind periodic hard resets as the default

For the projector-drift problem, the better first strategy was not hard reset every (T) batches.

A safer plan is to keep an anchor copy of the original projector parameters

[
\theta_g^{(0)}
]

and use either soft restoration or anchor regularization:

[
\mathcal L_{\text{anchor}} = \gamma |\theta_g - \theta_g^{(0)}|_2^2.
]

Then use a **trigger-based** hard reset only if monitoring signals collapse, such as sharply worsening alignment loss or unstable anomaly scores.

### H. Compare reset strategies as an ablation

Since reset may help in some regimes and hurt in others, the clean thesis move is to compare:

[
\text{no reset},
\quad
\text{soft restoration},
\quad
\text{trigger-based hard reset},
\quad
\text{periodic hard reset}.
]

That turns your concern into an experiment rather than a guess.

### I. Handle compute cost with staged alternatives

For the two-encoder online cost, we discussed three possible regimes:

[
\text{full dual-encoder baseline},
\quad
\text{shared encoder with stop-gradient reference branch},
\quad
\text{EMA teacher + one trainable student}.
]

The full dual-encoder version is the clean conceptual baseline. The others are compute-relief variants you can compare later.

### J. Guard online updates against contamination

To reduce the chance that anomalous mini-batches corrupt the online adapter, you can later gate updates using only more reliable samples or mini-batches, for example those with lower uncertainty or lower anomaly scores.

We did not force a single method yet, but the principle is: **not every mini-batch deserves the same update strength**.

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

[
X \xrightarrow{\text{MOMENT}} \tilde H \xrightarrow{U} H
]

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

[
H \in \mathbb{R}^{B \times L \times d_h}.
]

5. Then add, in order: continuous prototypes, discrete prototypes, task-specific fusion, and only then online adaptation.
6. In parallel, consider training **TFMAE**, **CATCH**, **TimesNet**, **TimeMixer**, or **FITS** yourself and extracting only the encoder if you decide that frequency-aware latent structure matters more than immediate open-weight reuse. ([GitHub][2])

---

## Carry-over brief for a new conversation

You can paste this into a new chat:

I am building a bachelor-thesis codebase for multivariate time-series anomaly detection on SMD with window length 100. The intended model has an encoder producing (H \in \mathbb{R}^{B \times L \times d_h}), then two prototype modules: a continuous prototype branch with attention-like soft retrieval, and a discrete prototype branch with Gumbel-Softmax-style codebook assignment. Their outputs are fused into task-specific latent spaces for reconstruction and anomaly-type classification using synthetic anomaly injection. Later, an online adaptation stage uses two augmentations per incoming sample, a frozen reference encoder, a partially trainable online encoder, and a lightweight projector that maps online representations into the reference space with contrastive and prototype-alignment losses. Main risks already identified: branch redundancy, one-branch-dominant fusion, anomaly contamination during online adaptation, dual-encoder online cost, noisy one-mini-batch gradients, projector initialization and drift, and lack of clearly reusable spectral-temporal pretrained encoders. Current design decisions: freeze the encoder output contract first, build a minimal vertical slice first, use a near-identity residual projector with offline warm-start, prefer soft restoration or trigger-based reset over blind periodic reset, and compare open-weight practical backbones like MOMENT/TTM/Timer against self-trained spectral-temporal encoders like TFMAE/CATCH/TimesNet/TimeMixer/FITS.

## Check

A good consistency check is this: if you remove the online adaptation block entirely, the remaining offline model is still a valid thesis core. If you remove one prototype branch, the model is still trainable. If you swap the encoder, the rest of the architecture should still work as long as the encoder wrapper preserves

[
H \in \mathbb{R}^{B \times L \times d_h}.
]

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
