---
date: 2026-06-17 14:05:37 +07
researcher: TheMetaSetter
git_commit: 3787f5b3ae94c599d99a64dc9aa8707c76161a6f
branch: dev
repository: bachelor-thesis-2026
topic: "Continuation context for offline pre-training three-stage discussion"
tags: [detail, offline-pretraining, multitask, contrastive, prototypes, zipping]
status: in_progress
last_updated: 2026-06-21
last_updated_by: Codex
---

# Detail: Offline Pre-Training Three-Stage Discussion Context

## Purpose

This note records the latest discussion about the newest intended offline pre-training method so the conversation can continue cleanly in later chats without restating the same context.

It is a continuation note, not yet a finalized design contract.

## Continuation Update on 2026-06-17

The user later answered a first group of clarification questions. This note is updated so the latest intended method is no longer mixed with the older ambiguous interpretation.

The user then added a second clarification block on the same day covering:

- the exact negative-pool source for normal anchors,
- the overlap consequences of window size 20 with stride 1,
- a correction to the classification-task label space,
- and the exact masking rule for Stage 1 reconstruction loss.

The user then added a third clarification block on the same day covering:

- the corrected class taxonomy: **12 total classes = 1 normal class + 11 anomaly classes**,
- the decision to use **both** aligned-view positives and same-source-timestep positives,
- the intended sequential layer-wise zipping interpretation for the 1D-CNN encoder,
- the preference to try full channel sharing first during zipping,
- the decision to initialize task-specific heads from the corresponding Stage 1 heads after zipping,
- the ordering `zip encoder -> initialize prototypes -> start multitask training`,
- the exact 300-epoch offline pre-training budget,
- and the decision to freeze the encoder during Stage 3 prototype warm-up so the prototypical branches and CKA-gated fusion heads can stabilize.

## Continuation Update on 2026-06-21

The user later saved a separate design note focused specifically on the contrastive-loss design refinement:

- `documents/design/design-contrastive-loss-21-jun-2026.md`

That later note materially sharpens the contrastive part of this continuation context.

The most important new clarifications are:

- the core contrastive objective should stay **low-compute** and should not introduce unnecessary tuning burden,
- the first practical weighting preference is no longer "free Gaussian with tunable \(\sigma\)",
- the first practical weighting preference is either:
  - a **uniform theoretical receptive-field clean ratio** \(\rho^{RF}\), or
  - a **fixed architecture-derived Gaussian receptive-field prior** with \(\sigma_{arch}\) computed once from the encoder config,
- if a Gaussian weighting is used, it must be described as a **fixed receptive-field prior / surrogate**, not as the true trained effective receptive field,
- \(\sigma_{arch}\) should be computed **once before training**, not re-estimated every epoch,
- Jacobian-based ERF estimation should **not** be part of the training loop,
- contrastive loss placement across stages is now sharper:
  - Stage 1 classification: yes,
  - Stage 1 reconstruction: yes,
  - Stage 2 zipping: no,
  - Stage 2 recovery: no in the first implementation, optional later as an ablation,
  - Stage 3 prototype initialization: no,
  - Stage 3 prototype warm-up: no in the first implementation,
  - main multitask pre-training: yes, but only when the multi-positive metadata contract is implemented cleanly,
- the naming should avoid calling this **standard SupCon**, because positives are defined by metadata and masks rather than by simple class-label equivalence,
- and the injected-position case is now narrower than before:
  - the first implementation can let injected aligned tokens act as negatives in the denominator,
  - while a separate explicit repulsion term remains an optional later design choice rather than a first-pass requirement.

## Immediate Context

Before this discussion, the active repository design context for the thesis model was still centered on:

- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- `documents/design/idea.md`
- `src/models/thesis_multitask.py`

That earlier design and code currently describe an offline phase with:

- two-view contrastive learning,
- continuous and discrete prototype branches,
- CKA-gated per-sample fusion,
- a multitask objective combining reconstruction and classification.

During the present discussion, a newer intended method was described that restructures the offline pre-training phase into three training stages before the final multitask training proceeds.

## Newest Intended Method Discussed

The latest method described in the discussion is:

### Stage 1: Train Two Task-Specific Models Separately

Train one model for the classification task:

- objective uses `classification loss + contrastive loss`
- only the classification head is active
- no prototype branches are used in this stage

Train one model for the reconstruction task:

- objective uses `reconstruction loss + contrastive loss`
- only the reconstruction head is active
- no prototype branches are used in this stage

This point is now clarified more strongly than before:

- Stage 1 trains **two separate encoder networks**
- Stage 1 does **not** yet use the continuous prototype branch
- Stage 1 does **not** yet use the discrete prototype branch
- Stage 1 is therefore a prototype-free warm-start stage for the later zipping procedure

### Stage 2: Multi-Task Zipping

Apply the idea from:

- `papers/He et al. - 2019 - NeurIPS-18 - Multi-Task Zipping via Layer-wise Neuron Sharing.pdf`

The purpose of zipping, as stated in the discussion, is:

- zip the two task-specific encoders from Stage 1 into one single encoder
- use the zipped encoder as the parameter initialization for later multitask training

The intended role of zipping is therefore initialization, not the final multitask training objective itself.

The stated motivation is:

- help the shared 1D-CNN encoder start from a stronger task-informed initialization
- support better generalization of the deep representation before multitask training

### Stage 3: Prototype Initialization

Initialize discrete prototypes using windows drawn uniformly across classes:

- 1 normal class
- 11 anomaly classes

Initialize continuous prototypes using only windows from the normal class.

An important nuance mentioned in the discussion:

- the "normal" class here may consist of original windows that are not guaranteed to be truly clean if one does not assume perfect normality

So the continuous prototype bank is intended to be initialized from nominally normal data, but contamination risk is acknowledged explicitly.

After prototype initialization, the current preferred warm-up behavior is:

- freeze the zipped shared encoder,
- train or stabilize the continuous prototype branch,
- train or stabilize the discrete prototype branch,
- train or stabilize the CKA-gated fusion heads,
- keep this warm-up short relative to the full 300-epoch budget.

The reason for freezing the encoder in this warm-up is to avoid moving the latent space while the newly initialized prototype memories and fusion gates are still adapting to it.

## Working Interpretation from the Discussion

The working interpretation written during the chat was:

1. Stage 1 learns two specialized representations with two separate encoders:
   - one representation optimized for classification
   - one representation optimized for reconstruction
   - both regularized by a contrastive objective
   - neither representation uses prototype branches yet

2. Stage 2 compresses or merges those two specialized encoders into one shared encoder through multi-task zipping.

3. Stage 3 seeds prototype memories after encoder zipping:
   - discrete prototypes receive class-balanced initialization
   - continuous prototypes receive normal-only initialization

4. The multitask training that follows uses the zipped encoder as initialization rather than learning the shared encoder fully from scratch.

5. The final intended order is now:
   - train two task-specific Stage 1 models,
   - zip their encoders,
   - optionally run a short post-zipping recovery phase without prototypes,
   - initialize prototypes,
   - run a short prototype warm-up with the encoder frozen,
   - then run the main multitask pre-training with prototype branches.

## Mathematical Sketch Used in the Discussion

The discussion used the following base tensor contract:

\[
X \in \mathbb{R}^{B \times L \times D}
\]

where:

- \(B\) is batch size
- \(L\) is window length
- \(D\) is number of input channels

The task-specific encoders in Stage 1 were interpreted as:

\[
f_{\theta^{cls}}: X \mapsto H^{cls} \in \mathbb{R}^{B \times L \times d_h}
\]

\[
f_{\theta^{rec}}: X \mapsto H^{rec} \in \mathbb{R}^{B \times L \times d_h}
\]

with classification logits:

\[
\hat{y} = C_{\phi}(\mathrm{Pool}(H^{cls})) \in \mathbb{R}^{B \times C}
\]

and reconstruction output:

\[
\hat{X} = R_{\psi}(H^{rec}) \in \mathbb{R}^{B \times L \times D}
\]

The Stage 1 losses were interpreted as:

\[
\mathcal{L}^{cls}_{total}
=
\lambda_{task}
\left(
\alpha_{cls}\mathcal{L}_{cls}
\right)
+
\lambda_{ctr}\mathcal{L}^{cls}_{ctr}
\]

\[
\mathcal{L}^{rec}_{total}
=
\lambda_{task}
\left(
\alpha_{rec}\mathcal{L}_{rec}
\right)
+
\lambda_{ctr}\mathcal{L}^{rec}_{ctr}
\]

with the latest stated weighting preference:

\[
\lambda_{task} = 1.0
\]

and with the intended task-importance bias in later multitask use leaning toward reconstruction, for example:

\[
\alpha_{rec}=0.9,
\qquad
\alpha_{cls}=0.1
\]

However, at the present discussion point, the above `0.9 / 0.1` ratio is conceptually attached to the task-loss balance and is not yet fully translated into a final Stage 1 and Stage 3 implementation contract, especially once optional losses are added.

Stage 2 was summarized conceptually as:

\[
\theta^{zip} = \mathrm{Zip}(\theta^{cls}, \theta^{rec})
\]

where the returned encoder parameters initialize the subsequent multitask encoder.

Stage 3 was interpreted as initializing:

\[
P^{(d)} \in \mathbb{R}^{K_d \times d_h}
\]

from class-balanced windows and:

\[
P^{(c)} \in \mathbb{R}^{K_c \times d_h}
\]

from normal-only windows.

## Latest Consolidated Stage Schedule Under the 300-Epoch Budget

The user clarified that the total offline pre-training epoch budget is exactly:

\[
300 \text{ epochs}
\]

The current recommended allocation discussed in the chat is:

| Phase | Epochs | Training status | Purpose |
|---|---:|---|---|
| Stage 1A: classification encoder | 50 | train `E_cls` and classification head | learn multiclass synthetic-anomaly decision boundaries and contrastive structure |
| Stage 1B: reconstruction encoder | 70 | train `E_rec` and reconstruction head | prioritize normal reconstruction geometry for TSAD |
| Stage 2: MTZ zipping | 0 | parameter transformation, not epoch training | merge the two Stage 1 encoders into one shared encoder initialization |
| Stage 2 recovery | 20 | train zipped encoder plus reused heads, no prototypes | recover task performance after zipping |
| Stage 3 prototype initialization | 0 | statistical initialization, not epoch training | seed continuous and discrete prototype memories from latent tokens |
| Stage 3 prototype warm-up | 20 | freeze encoder; train/stabilize prototype branches and CKA-gated fusion heads | let new prototype branches and fusion heads adapt to fixed latent geometry |
| Main multitask pre-training with prototypes | 140 | train full multitask model | jointly train encoder, heads, prototypes, fusion, and optional losses |
| **Total** | **300** |  |  |

This allocation treats Stage 1 classification and Stage 1 reconstruction as separate training runs and therefore counts both against the 300-epoch budget.

The rationale is:

- Stage 1 should be long enough to produce meaningful task-specialized encoders for zipping.
- Stage 1 should not consume most of the budget because the final deployed representation is the multitask prototype model.
- Stage 2 zipping itself is not epoch training.
- Stage 2 recovery is needed because zipping may perturb both task-specific functions.
- Stage 3 prototype initialization is a deterministic/statistical step, not a training phase.
- Stage 3 prototype warm-up should freeze the encoder so the prototype branches and CKA-gated fusion heads adapt to a stable latent space.
- The main multitask pre-training receives the largest share because it is the phase where the final encoder, heads, prototype memories, fusion gates, and optional losses become consistent with each other.

The preferred logging contract is:

- `global_epoch`: runs from 1 to 300 across the whole offline pre-training phase,
- `phase_name`: one of `stage1_classification`, `stage1_reconstruction`, `stage2_recovery`, `stage3_prototype_warmup`, `multitask_pretraining`,
- `phase_epoch`: local epoch index inside the active phase,
- `encoder_frozen`: especially important during `stage3_prototype_warmup`,
- `prototypes_initialized`: whether continuous and discrete prototype memories have been seeded,
- `memory_mode`: whether prototype branches are bypassed, initialized-only, warm-up, or fully trainable.

Scheduler note:

- A single cosine schedule across all 300 epochs is not ideal because the active objective and active parameter set change by phase.
- The safer design is one scheduler per training phase, with explicit phase-local warm-up and decay behavior.
- Stage 2 zipping and Stage 3 prototype initialization should not advance an optimizer scheduler because they are not training epochs.

## Clarified Contrastive Construction from the User's Latest Answers

The user later clarified the intended contrastive logic in more detail.

### Base Paired Views

For each original normal window:

\[
x \in \mathbb{R}^{L \times D}
\]

construct one anomalous augmented view:

\[
x' \in \mathbb{R}^{L \times D}
\]

with a binary injection mask:

\[
M \in \{0,1\}^{L}
\]

where:

\[
M_t = 1
\]

means the timestep at index \(t\) is injected or corrupted in the anomalous augmented view.

The paired latent tensors from one task-specific encoder are:

\[
z = E(x) \in \mathbb{R}^{L \times d_h},
\qquad
z' = E(x') \in \mathbb{R}^{L \times d_h}
\]

and, at batch level:

\[
Z, Z' \in \mathbb{R}^{B \times L \times d_h}
\]

### Clarified Intended Semantics for Normal Timestep Anchors

The current intended interpretation is:

- choose anchor points only from timesteps in the original normal window whose corresponding positions are **not** injected in the anomalous view
- therefore, for one sample window, the number of anchors is generally smaller than \(L=20\)

Define the eligible anchor index set:

\[
\mathcal{I}_{normal}
=
\{(b,t)\mid M_{b,t}=0\}
\]

For each such anchor:

\[
q_{b,t} = \operatorname{norm}(Z_{b,t})
\]

The intended positive is the corresponding timestep in the anomalous augmented view:

\[
k^+_{b,t} = \operatorname{norm}(Z'_{b,t})
\]

The user's later clarification strengthens this point:

- the negative pool for a normal anchor should come from both the original clean-view windows and the corresponding anomalous augmented-view windows inside the same batch

So the intended contrastive pool is not restricted to augmented-view tokens only.

The current best-faith interpretation is:

\[
\mathcal{N}_{b,t}
=
\mathcal{C}_{clean,aug}
\setminus
\{(b,t)\}
\]

where \(\mathcal{C}_{clean,aug}\) is the union of timestep embeddings from:

- all clean-view windows in the batch,
- and all anomalous augmented-view windows in the batch.

At minimum, the current understanding is:

- the positive is the aligned non-injected timestep across the two views
- negatives are all other timestep embeddings in the batch under a within-batch InfoNCE strategy
- the batch-level pool includes both views rather than one view only

### Additional Clarification from Sliding-Window Overlap

The user explicitly tied the contrastive design to the dataset windowing mechanism:

- window size is \(L=20\),
- stride is \(1\),
- therefore one original timestep from the long source sequence may appear in several overlapping windows that happen to coexist in the same batch

This means the same underlying source-sequence timestep may contribute multiple latent tokens across different windows in the same batch.

The user then refined the intended positive semantics further:

- for an anchor point in one normal window,
- whose aligned position in the augmented counterpart is anomalous or otherwise contrastively relevant,
- the intended positive may be a **normal point corresponding to the same absolute position in the original long sequence**
- but located in a different overlapping window in the same batch

This is a much stronger and more subtle requirement than simple within-window aligned pairing.

It introduces an additional notion of identity:

- not only `(batch index, local timestep index inside the window)`,
- but also `(absolute timestep identity in the original long sequence)`

This distinction is critical for implementation because standard batch tensors do not automatically preserve that identity unless explicit metadata is carried through the loader and batch collation path.

### Immediate Consequence for Positive Construction

Under this clarification, the intended design is now a **multi-positive construction**.

For a normal anchor, there are two positive sources:

1. **Aligned-view positive**
   - the clean token at `(b,t)` is paired with the token at the same local position `(b,t)` in the augmented counterpart of the same window

2. **Same-source-timestep positive across windows**
   - the clean token at `(b,t)` is paired with another clean token from a different overlapping window in the same batch
   - both tokens correspond to the same absolute timestep in the original source sequence

The user explicitly selected **both 1 and 2** as positives for each eligible normal anchor point.

This is a major conceptual constraint because it means the contrastive objective needs access to:

- absolute source-timestep indices,
- overlap-aware batch bookkeeping,
- and multiple positives per anchor rather than exactly one positive.

At implementation time, a standard one-positive InfoNCE loss is therefore insufficient unless it is extended to a supervised-contrastive or multi-positive InfoNCE form.

### Can One Source Timestep Appear in Only One Window in a Batch?

Yes, that can happen.

Even though stride \(=1\) means one timestep can appear in multiple overlapping windows in the full dataset, the number of times it appears **inside one batch** depends on batch composition.

If batching is not explicitly constructed to preserve contiguous overlapping windows from the same entity, then a source timestep may:

- appear many times in a batch,
- appear only a few times,
- appear exactly once,
- or not appear at all in that particular batch

More concretely:

- in the full sliding-window dataset, an interior timestep can belong to as many as 20 windows when `window_size = 20` and `stride = 1`
- near sequence boundaries, the multiplicity is lower
- inside a training batch, multiplicity depends on sampler behavior, window ordering, batch size, and whether the batch gathers contiguous windows from the same long sequence

Therefore, if the positive definition requires "another window in the same batch that shares the same original timestep", then there is no guarantee that such a second occurrence always exists unless the batching strategy is deliberately designed to guarantee it.

### Clarified Intended Semantics for Injected Timestep Anchors

The user also described a second idea:

- consider a timestep in the original normal window whose corresponding position **will become anomalous** in the augmented view
- use the anomalous version of that same aligned timestep in the augmented latent tensor as a negative relation rather than a positive relation

Define the injected-position index set:

\[
\mathcal{I}_{anom}
=
\{(b,t)\mid M_{b,t}=1\}
\]

For such a position, the intended relation is conceptually:

\[
q^{anom}_{b,t} = \operatorname{norm}(Z_{b,t})
\]

and the aligned anomalous counterpart in the augmented view:

\[
k^{anom}_{b,t} = \operatorname{norm}(Z'_{b,t})
\]

should be treated as a representation that must be pushed away from the clean anchor, not pulled toward it.

The user explicitly noted that this anomalous-anchor case seems to have:

- an anchor,
- a designated negative counterpart,
- but no obvious positive counterpart.

This is an extremely important clarification because it means the intended objective is no longer a plain standard InfoNCE formulation if this second case is included directly.

## Current Best Interpretation of the Two-Part Contrastive Intent

The latest discussion suggests the user is aiming at two distinct semantic effects:

1. **Normal-position alignment**
   - if a timestep stays normal across views, its latent embedding should remain close across the clean and anomalous views

2. **Injected-position separation**
   - if a timestep is turned anomalous in the augmented view, the clean latent and anomalous latent at that aligned position should become different or repulsive

This can be summarized informally as:

- preserve similarity for non-injected aligned timesteps
- encourage dissimilarity for injected aligned timesteps

That high-level intention is now clear.

What was **not** yet mathematically closed in the earlier discussion was whether both effects should be implemented:

- inside one single InfoNCE loss,
- or as one InfoNCE term plus one separate repulsion term,
- or by using injected positions only as negatives for the normal-anchor InfoNCE without making them anchors themselves.

The latest overlap-based clarification no longer leaves the positive construction as an either-or choice.

For a normal anchor, the intended design is now:

- use the aligned token in the paired augmented view as one positive,
- use same-source-timestep tokens from other overlapping windows in the batch as additional positives when available,
- therefore implement the contrastive term as a multi-positive objective rather than a one-positive objective.

The later 2026-06-21 design note narrows the first implementation further:

- keep the multi-positive construction for normal anchors,
- let injected aligned tokens default to **negative-only roles in the denominator**,
- do **not** add a separate explicit repulsion term in the first implementation,
- and keep any later repulsion-term variant as an ablation or second-step extension.

So the current best first-pass interpretation is:

1. normal anchors use a multi-positive InfoNCE-style objective,
2. injected aligned tokens are available as negatives,
3. explicit repulsion for injected aligned pairs remains optional rather than mandatory.

## Clarified Receptive-Field Weighting for Weak Positives

The later 2026-06-21 design note also sharpened how weak positives should be weighted.

The intent remains:

- if a positive token is fully clean in the local receptive-field sense, it should pull more strongly,
- if a positive token is center-clean but its surrounding receptive field is partially contaminated, it can still be used,
- but it should exert a weaker attractive force.

The clean/anomalous binary mask is:

\[
C = 1 - M
\]

where:

- \(M_{b,t}=1\) means the timestep is injected/anomalous in the augmented view,
- \(C_{b,t}=1\) means the timestep is clean.

### First Low-Compute Preference: Uniform Theoretical RF Clean Ratio

The current most conservative low-compute choice is:

\[
\rho^{RF}_{b,t}
=
\frac{1}{2R+1}
\sum_{r=-R}^{R} C_{b,t+r}
\]

where:

- \(R\) is the theoretical receptive-field radius implied by the 1D-CNN architecture,
- \(r\in[-R,R]\) is an offset around the central timestep \(t\),
- and \(\rho^{RF}_{b,t}\) measures how much of the token's theoretical receptive field remains clean.

Under this first preference, the positive weight is:

\[
w_{i,p} = \rho^{RF}_{p}
\]

rather than introducing an extra exponent \(\gamma\) in the first implementation.

The practical rationale is:

- no extra tuning for \(\sigma\),
- no extra tuning for \(\gamma\),
- no Jacobian estimation,
- and a more defensible low-compute method for the thesis setting.

### Optional Smoother Alternative: Fixed Architecture-Derived Gaussian RF Prior

The later note does **not** forbid Gaussian weighting entirely. It only restricts how it should be interpreted.

If a Gaussian smoothing prior is used, it should be defined through an architecture-derived scale:

\[
g_r=
\frac{
\exp\left(-\frac{r^2}{2\sigma^2_{arch}}\right)
}{
\sum_{q=-R}^{R}\exp\left(-\frac{q^2}{2\sigma^2_{arch}}\right)
}
\]

and:

\[
\rho^{arch}_{b,t}
=
\sum_{r=-R}^{R} g_r C_{b,t+r}
\]

with:

\[
w_{i,p} = \rho^{arch}_{p}
\]

The key interpretive rule is now explicit:

- \(\sigma_{arch}\) is **not** the true trained ERF variance,
- it is a **fixed Gaussian receptive-field prior** derived from architecture,
- and it should be computed once from the model config before training, not re-estimated during training.

For a simple 1D-CNN with odd kernel size \(K\), stride \(1\), dilation \(1\), and \(L_{conv}\) convolutional layers, the later note records the approximate initialization prior:

\[
\sigma^2_{arch}
=
L_{conv}\frac{K^2-1}{12}
\]

and:

\[
R
=
L_{conv}\frac{K-1}{2}
\]

For heterogeneous layers, the note records the more general stride/dilation-aware idea:

\[
\sigma^2_{arch}
=
\sum_{l=1}^{L_{conv}}
\left(
d_l\prod_{m<l}s_m
\right)^2
\frac{K_l^2-1}{12}
\]

and:

\[
R
=
\sum_{l=1}^{L_{conv}}
d_l\prod_{m<l}s_m
\frac{K_l-1}{2}
\]

The role of this formula is intentionally narrow:

- it gives a fixed prior scale from architecture,
- it avoids extra hyperparameter search,
- but it must **not** be described as measuring the trained ERF exactly.

### What Should Not Be Done in the Main Training Loop

The later note is explicit that the following should **not** be part of the first implementation:

- re-computing \(\sigma\) every epoch,
- estimating ERF dynamically inside training,
- or using Jacobian-based influence estimation as part of the loss computation.

If Jacobian-based ERF diagnostics are ever used, they should be treated as:

- analysis-only,
- optional after-training diagnostics,
- not as part of the core loss path.

### Loss Naming and Terminology

The current safest naming is now:

- `RF-mask weighted multi-positive InfoNCE`, or
- `receptive-field-mask weighted multi-positive contrastive loss`.

If the Gaussian prior version is used instead, the naming should stay careful, for example:

- `architecture-derived RF-mask weighted multi-positive contrastive loss`, or
- `fixed Gaussian RF-prior weighting`.

The note explicitly advises against calling the method:

- `standard SupCon`,
- `learned ERF weighting`,
- or `true ERF weighting`.

## Clarified Classification Task Statement

The user later explicitly corrected the intended classification task:

- the classification task has **12 total classes**
- the taxonomy is **1 normal class + 11 anomaly classes**

This closes the earlier arithmetic tension.

This materially fixes:

- Stage 1 classification-head output dimension,
- Stage 3 class-balanced discrete prototype initialization,
- confusion-matrix semantics,
- and compatibility with the currently implemented RedLamp multiclass surfaces in the repository

The intended label space is now:

\[
C=12
\]

with:

\[
\mathcal{Y}
=
\{\text{normal}\}
\cup
\{\text{11 synthetic anomaly families}\}
\]

This matches the existing RedLamp-style taxonomy in the codebase:

- `REDLAMP_MULTICLASS_CLASS_NAMES = ("normal", *REDLAMP_ANOMALY_FAMILIES)`
- `REDLAMP_ANOMALY_FAMILIES` contains 11 anomaly families

## Clarified Stage 1 Reconstruction-Head Objective

The user later clarified the reconstruction behavior in Stage 1 as follows:

- the reconstruction head reconstructs the **entire window**
- but the reconstruction loss is computed **only on normal positions**
- positions injected as anomalous in the augmented view do **not** contribute to reconstruction loss

This yields the following intended masked reconstruction semantics.

For one window pair:

\[
x \in \mathbb{R}^{L \times D},
\qquad
x' \in \mathbb{R}^{L \times D},
\qquad
M \in \{0,1\}^{L}
\]

let the reconstruction branch produce:

\[
\hat{x} = R(E_{rec}(x)),
\qquad
\hat{x}' = R(E_{rec}(x'))
\]

with:

\[
\hat{x}, \hat{x}' \in \mathbb{R}^{L \times D}
\]

The user-intended reconstruction objective is now best interpreted as:

\[
\mathcal{L}_{rec}
=
\frac{
\sum_{t=1}^{L}(1-M_t)\,\ell(\hat{x}_t, x_t)
}{
\sum_{t=1}^{L}(1-M_t)
}
+
\frac{
\sum_{t=1}^{L}(1-M_t)\,\ell(\hat{x}'_t, x_t)
}{
\sum_{t=1}^{L}(1-M_t)
}
\]

where:

- the target is still the clean normal signal \(x\),
- anomalous injected positions are masked out of the loss,
- and the reconstruction path is therefore encouraged to model normal timesteps well while not being directly penalized on injected anomalous positions

### Design Intention Behind This Reconstruction Masking

The user also clarified the final rationale:

- in the final model with prototype branches attached,
- the reconstruction pathway should reconstruct normal timesteps well,
- and reconstruct anomalous timesteps worse

So the Stage 1 reconstruction design is intended to support that later behavior by:

- rewarding accurate modeling on normal positions,
- while avoiding a training signal that would force the model to reconstruct injected anomalous positions well.

## Clarified Multi-Task Zipping Interpretation

The user currently expects Stage 2 zipping to run sequentially from shallow to deep layers:

\[
l = 1,2,\ldots,L
\]

For the current 1D-CNN encoder in the codebase, the unit of zipping should be interpreted as:

- a `Conv1d` output channel,
- equivalently a convolutional kernel/filter,
- not a timestep token.

For a convolutional layer:

\[
W_l \in \mathbb{R}^{C_{out} \times C_{in} \times k}
\]

the `i`-th channel/kernel has incoming weights:

\[
W_{l,i} \in \mathbb{R}^{C_{in} \times k}
\]

which can be flattened into:

\[
\tilde{w}_{l,i} \in \mathbb{R}^{C_{in}k}
\]

to compute functional difference.

The MTZ paper's main functional-difference metric is Hessian-based rather than a simple cosine or Euclidean distance:

\[
d(\tilde{w}^{A}_{l,i}, \tilde{w}^{B}_{l,j})
=
\delta E_l^{opt}
\]

where:

\[
\delta E_l^{opt}
=
\frac{1}{2}
(\tilde{w}^{A}_{l,i}-\tilde{w}^{B}_{l,j})^\top
\left[
(\tilde{H}^{A}_{l,i})^{-1}
+
(\tilde{H}^{B}_{l,j})^{-1}
\right]^{-1}
(\tilde{w}^{A}_{l,i}-\tilde{w}^{B}_{l,j})
\]

The corresponding incoming-weight update is:

\[
f(\tilde{w}^{A}_{l,i}, \tilde{w}^{B}_{l,j})
=
\tilde{w}^{A}_{l,i}+\delta\tilde{w}^{A,opt}_{l,i}
=
\tilde{w}^{B}_{l,j}+\delta\tilde{w}^{B,opt}_{l,j}
\]

The current experimental priority is to try **full channel sharing** first:

\[
\tilde{N}_l = \min(C^A_{out,l}, C^B_{out,l})
\]

for each corresponding convolutional layer.

However, a critical implementation detail remains:

- full sharing should not mean blindly merging channel `i` in `E_cls` with channel `i` in `E_rec`,
- because CNN channels are permutation-equivalent,
- so a full-sharing implementation should still compute pairwise functional differences and use matching, for example Hungarian/min-cost matching, before merging.

The practical first implementation can use one of two zipping metrics:

1. **Faithful MTZ metric**
   - Hessian-based functional difference as in the paper.
   - More faithful but more complex and numerically delicate.

2. **Engineering baseline metric**
   - Activation-based or weight-based distance for channel matching.
   - Less faithful to the paper but easier to implement and debug first.

The note does not yet finalize which metric must be used in code, but records that the paper's metric is Hessian-based and that any simplified metric should be named as an approximation, not as the exact MTZ metric.

## Task-Specific Head Initialization After Zipping

After zipping, the shared encoder is:

\[
E_{zip}
\]

The current best design choice is to initialize the multitask heads from the Stage 1 heads:

\[
C_{multi}^{(0)} \leftarrow C_{stage1}
\]

\[
R_{multi}^{(0)} \leftarrow R_{stage1}
\]

This is preferred over random initialization because:

- the classification head already knows how to read the classification-specialized latent space,
- the reconstruction head already knows how to decode the reconstruction-specialized latent space,
- zipping is meant to preserve and merge task knowledge, not discard the learned task-specific heads.

This assumes the zipped encoder preserves the output hidden dimension:

\[
E_{zip}(X) \in \mathbb{R}^{B \times L \times d_h}
\]

If zipping changes the hidden dimension, then either:

- a projection layer is needed before the heads,
- or the zipping implementation must be constrained to preserve `hidden_dim`.

The preferred first experiment is to preserve `hidden_dim` and reuse the two heads directly.

## Prototype Initialization Methods Discussed

The user asked how the following prototype sampling strategies differ:

- random,
- centroid-based,
- covering-based,
- clustering-based.

The current interpretation is:

- **Random sampling** chooses prototype seed tokens uniformly or pseudo-randomly from eligible tokens. It is simple but can pick redundant nearby tokens.
- **Centroid-based initialization** uses the mean vector of a group, such as a class mean. It is stable but may collapse diversity into one central prototype.
- **Covering-based initialization** selects tokens that spread across the latent space, for example farthest-first selection. It keeps diversity without running a full clustering algorithm.
- **Clustering-based initialization** runs an algorithm such as KMeans and uses cluster centers. It is stronger but adds implementation cost, runtime cost, and another source of randomness.

For the first version of the new method, the current recommendation is:

- discrete prototypes: class-balanced plus covering-based selection inside each class,
- continuous prototypes: normal-only plus covering-based selection.

This aligns with the current codebase's existing `_select_covering_vectors` style while extending it to respect class balance for the discrete codebook.

## Clarified Discrete Codebook Query Proposal

The later discussion added a more concrete preference for how the discrete branch should query the codebook.

The current best interpretation of that preference is:

- for each latent token \(z_{b,t}\), query the discrete codebook by nearest-neighbor distance in latent space,
- do not treat the codebook query primarily as a learned logits head over code indices,
- allow sparse multi-codeword aggregation with top-\(k\) nearest codewords, with \(k=2\) and \(k=3\) as the first planned experiments,
- keep \(k=1\) as the hard nearest-codeword baseline.

This means the discrete branch is currently being conceptualized as a **metric-based retrieval module** over a codebook of real-valued vectors, not merely as a categorical assignment head.

### Mathematical Form of the Proposed Query

Let the encoder output:

\[
Z \in \mathbb{R}^{B \times L \times d_h}
\]

and let the discrete codebook be:

\[
E \in \mathbb{R}^{K_d \times d_h}
\]

where each codeword:

\[
e_i \in \mathbb{R}^{d_h}
\]

is a real-valued latent prototype vector.

For one token \(z_{b,t}\), compute its squared Euclidean distance to every codeword:

\[
d_{b,t,i} = \|z_{b,t} - e_i\|_2^2
\]

This yields a distance tensor:

\[
D \in \mathbb{R}^{B \times L \times K_d}
\]

The selected codeword-index set is:

\[
S_k(z_{b,t}) = \operatorname{TopK}_{i}(-d_{b,t,i})
\]

where \(S_k(z_{b,t})\) contains the indices of the \(k\) nearest codewords.

Soft weights are then computed only over the selected set:

\[
\alpha_{b,t,i}
=
\frac{\exp(-d_{b,t,i}/\tau)}
{\sum_{j \in S_k(z_{b,t})}\exp(-d_{b,t,j}/\tau)}
\qquad \text{for } i \in S_k(z_{b,t})
\]

The queried discrete prototype vector is:

\[
z^q_{b,t}
=
\sum_{i \in S_k(z_{b,t})}\alpha_{b,t,i} e_i
\]

At sequence level:

\[
Z^q \in \mathbb{R}^{B \times L \times d_h}
\]

The current preferred residualized branch output is:

\[
H^{(d)}
=
\operatorname{LayerNorm}(Z + \lambda Z^q)
\]

where \(\lambda\) controls how strongly the queried codebook representation perturbs the raw latent token.

This is the currently preferred conceptual query. It is not the same as the current repository implementation.

### Role of \(W_d\) and \(b_d\) Under the Two Competing Interpretations

The discussion earlier also referenced:

\[
z^{logit}_{b,t} = W_d \tilde{h}_{b,t} + b_d \in \mathbb{R}^{K_d}
\]

Under a **learned-assignment interpretation**, \(W_d \in \mathbb{R}^{K_d \times d_h}\) and \(b_d \in \mathbb{R}^{K_d}\) are the parameters of a linear scoring map from latent space into a \(K_d\)-dimensional assignment-logit space. In that interpretation:

- each output coordinate corresponds to one codeword slot,
- the output entries are affinity-like scores or logits,
- and the codebook query is mediated by a learned projection rather than by direct distance to the codebook vectors themselves.

Under the **currently preferred nearest-neighbor interpretation**, those two parameters are no longer the core query operator. The decisive object becomes the discrete codebook \(E\) itself, and assignment is produced directly from distances between \(z_{b,t}\) and the codewords \(e_i\).

So the current best semantic distinction is:

- \(E\) is the set of real-valued codeword vectors in latent space,
- \(W_d, b_d\) are only needed if the branch is implemented as a learned logits head,
- if the branch is implemented as top-\(k\) nearest-neighbor retrieval, \(W_d, b_d\) are optional at most and are not the primary mathematical object.

### Clarified Backpropagation Semantics for the Proposed Top-\(k\) Query

The later discussion also narrowed the intended backprop interpretation.

The current best interpretation is:

- in one forward pass, compute distances to all codewords,
- choose the top-\(k\) nearest codewords,
- treat that selected index set as fixed during that pass,
- and backpropagate through the differentiable computations inside that fixed selection: distance values, softmax weights, codeword aggregation, residual addition, and layer normalization.

This means the top-\(k\) operator is not treated as fully differentiable with respect to ranking changes.

Let:

\[
s_{b,t,i} = -d_{b,t,i}/\tau
\]

for \(i \in S_k(z_{b,t})\), and let:

\[
g_{b,t} = \frac{\partial \mathcal{L}}{\partial z^q_{b,t}} \in \mathbb{R}^{d_h}
\]

denote the gradient that reaches the queried discrete vector after the downstream losses and residual path.

Because:

\[
z^q_{b,t}
=
\sum_{i \in S_k(z_{b,t})}\alpha_{b,t,i} e_i
\]

the gradient with respect to one selected score \(s_{b,t,j}\) takes the standard softmax-mixture form:

\[
\frac{\partial \mathcal{L}}{\partial s_{b,t,j}}
=
\alpha_{b,t,j}\,
g_{b,t}^{\top}(e_j - z^q_{b,t})
\qquad \text{for } j \in S_k(z_{b,t})
\]

and the score derivatives are:

\[
\frac{\partial s_{b,t,j}}{\partial z_{b,t}}
=
-\frac{2}{\tau}(z_{b,t} - e_j)
\]

\[
\frac{\partial s_{b,t,j}}{\partial e_j}
=
\frac{2}{\tau}(z_{b,t} - e_j)
\]

Therefore the query-path gradient into the latent token is:

\[
\frac{\partial \mathcal{L}}{\partial z_{b,t}}\Big|_{query}
=
\sum_{j \in S_k(z_{b,t})}
\frac{\partial \mathcal{L}}{\partial s_{b,t,j}}
\left[-\frac{2}{\tau}(z_{b,t} - e_j)\right]
\]

and the full gradient into \(z_{b,t}\) also includes the direct residual path from:

\[
H^{(d)} = \operatorname{LayerNorm}(Z + \lambda Z^q)
\]

so the latent token receives:

- a direct identity-like residual contribution through \(Z\),
- plus the indirect codebook-query contribution through \(Z^q\).

For one selected codeword \(e_j\), the gradient has two components:

\[
\frac{\partial \mathcal{L}}{\partial e_j}
=
\alpha_{b,t,j} g_{b,t}
+
\frac{\partial \mathcal{L}}{\partial s_{b,t,j}}
\frac{2}{\tau}(z_{b,t} - e_j)
\qquad \text{for } j \in S_k(z_{b,t})
\]

The first term is the direct contribution because \(e_j\) appears explicitly in the weighted sum \(z^q_{b,t}\). The second term is the indirect contribution because the soft weight \(\alpha_{b,t,j}\) depends on the distance, and the distance depends on \(e_j\).

For any codeword not selected in the top-\(k\) set of that token:

\[
\frac{\partial \mathcal{L}}{\partial e_i} = 0
\qquad \text{for } i \notin S_k(z_{b,t})
\]

for that particular token and forward pass.

So the intended differentiability picture is:

- piecewise differentiable inside a region where the top-\(k\) identity set does not change,
- discontinuous when ranking changes cause different codewords to enter or leave the selected set,
- but still practical to optimize in PyTorch because gather/index operations propagate gradients to the selected tensors.

### Special Cases: \(k=1\), \(k=2\), and \(k=3\)

The current intended experiments are:

- \(k=1\): hard nearest-codeword assignment baseline,
- \(k=2\): sparse two-codeword aggregation,
- \(k=3\): sparse three-codeword aggregation.

For \(k=1\), the forward pass becomes:

\[
z^q_{b,t} = e_{i^\star},
\qquad
i^\star = \arg\min_i d_{b,t,i}
\]

if the implementation uses a truly hard nearest-neighbor selection without soft weighting.

That is simple, but the routing decision is maximally discontinuous.

For \(k=2\) and \(k=3\), the routing remains sparse while still allowing a differentiable soft combination among the selected codewords. The current discussion favors these settings as the more stable first experiments for the discrete branch.

### Programming-Level Consequences That Still Need Explicit Closure

The semantic idea is now much clearer than before, but several implementation details remain open and should be closed explicitly before code changes:

- **Distance metric**: whether to use squared Euclidean distance exactly as currently preferred, cosine distance, or squared Euclidean on normalized vectors. This matters because the current code normalizes hidden states and codebook vectors before lookup.
- **Codebook update policy**: whether the discrete codebook \(E\) should be updated by ordinary backprop as trainable parameters, by EMA-style memory updates, or by a hybrid rule. The current repository uses EMA-style update buffers for the discrete codebook, which is not the same as pure gradient-trained nearest-neighbor codewords.
- **Status of `discrete_assignment`**: whether the current linear head should be removed entirely, kept only for ablation, or reused as an auxiliary scorer or regularizer. Under the new nearest-neighbor interpretation it is no longer the primary query path.
- **Exact tensor implementation**: whether distances are computed with `torch.cdist`, manual broadcasting, or a normalized inner-product equivalent. This affects memory cost because the full distance tensor has shape \([B, L, K_d]\).
- **Top-\(k\) gradient handling**: whether the implementation relies on standard gather-based autograd only, or introduces a straight-through estimator for the hard \(k=1\) case.
- **Residual design**: whether the discrete branch should use \(H^{(d)} = \operatorname{LayerNorm}(Z + \lambda Z^q)\) exactly, another normalization rule, or no residual mixing during some warm-up phases.
- **Initialization-to-query consistency**: whether Stage 3 discrete prototype initialization seeds codewords in the same geometry and normalization regime later used by top-\(k\) querying.
- **Class-to-codeword allocation**: if discrete initialization is class-balanced but \(K_d\) is not divisible by 12, the slot-allocation rule needs to be stated exactly.
- **Collapse prevention**: sparse top-\(k\) routing can still collapse onto a few codewords, so it must be decided whether to retain a usage regularizer, entropy regularizer, or codebook-balancing loss.
- **Diagnostics**: logs should include at least top-1 code index histogram, top-\(k\) usage frequency, selected-weight entropy, number of inactive codewords, and gradient norms for encoder and codebook.

### Immediate Difference from the Current Repository Implementation

At the time of this note update, the current code in `src/models/thesis_multitask.py` still implements the discrete branch in a different way:

- a linear map `self.discrete_assignment = nn.Linear(hidden_dim, discrete_codebook_size)` produces assignment logits,
- `F.gumbel_softmax(..., hard=False)` converts those logits into soft assignment probabilities,
- the discrete hidden is then the weighted sum of the codebook via `torch.einsum("blk,kh->blh", assignment_probabilities, normalized_codebook)`,
- and the codebook memory is updated by EMA-style statistics in `_update_discrete_codebook_memory`.

So the current codebase should still be described as:

- **learned-assignment + Gumbel-Softmax + EMA-updated codebook**

not yet as:

- **distance-based top-\(k\) nearest-neighbor discrete query**

If the new proposal is adopted, the most decisive implementation touchpoints are:

- `_discrete_prototype_lookup`,
- `_update_discrete_codebook_memory`,
- the parameter definition around `self.discrete_assignment`,
- and any loss terms that currently assume dense assignment probabilities over all \(K_d\) codewords.

## Contrastive Learning Placement Across Stages

The user proposed that contrastive learning may happen in all 3 stages.

The sharp counter-argument recorded in the discussion is:

- Stage 2 zipping is primarily a parameter matching and merging procedure, not an ordinary representation-learning epoch loop.
- Adding contrastive training inside Stage 2 would confound the effect of zipping with the effect of extra contrastive recovery.
- Stage 3 prototype initialization should ideally be deterministic or at least stable, because the purpose is to seed prototype memories from a known latent geometry.
- Training contrastively while initializing prototypes can move the latent geometry at the same time that prototypes are being seeded, making analysis harder.
- With stride 1 sliding windows, naive "all other timesteps are negatives" creates false-negative risk because the same source timestep or semantically equivalent normal tokens can appear in several windows.
- Therefore, contrastive learning should be treated as a controlled objective in Stage 1 and possibly in the main multitask pre-training phase, rather than as an automatic objective inside every stage.

The current recommended contrastive placement is:

- Stage 1 classification: contrastive enabled,
- Stage 1 reconstruction: contrastive enabled,
- Stage 2 zipping: contrastive disabled because this is not a training epoch phase,
- Stage 2 recovery: contrastive disabled in the first implementation; optional later as an ablation choice,
- Stage 3 prototype initialization: contrastive disabled,
- Stage 3 prototype warm-up: contrastive disabled in the first implementation because the encoder is frozen and the warm-up target should stay narrow,
- main multitask pre-training: contrastive can be enabled if the multi-positive batch metadata contract is implemented cleanly.

The later 2026-06-21 note makes the Stage 3 warm-up recommendation stronger than before:

- not merely "very low weight by default",
- but preferably **off** in the first implementation,
- so prototype branches and CKA-gated fusion heads adapt to a fixed latent geometry without extra contrastive complexity.

## Updated Ambiguities and Current Resolution Status

The following list supersedes older ambiguity notes in this file. It separates points that are now closed from points that still require implementation-level design.

### 1. Nature of Stage 1 Models

Status: **closed**.

Stage 1 uses two separate training runs:

- one classification-specific encoder and classification head,
- one reconstruction-specific encoder and reconstruction head.

Stage 1 does not use continuous prototypes, discrete prototypes, or CKA-gated fusion.

### 2. Classification Label Semantics

Status: **closed**.

The intended label space is:

\[
C = 12 = 1 + 11
\]

with:

- 1 normal class,
- 11 synthetic anomaly classes.

This matches the current RedLamp-style multiclass taxonomy in the repository.

### 3. Stage 1 Reconstruction Target

Status: **mostly closed**.

Both raw view and synthetic anomalous view are passed through the reconstruction encoder and reconstruction head:

\[
\hat{x} = R(E_{rec}(x)),
\qquad
\hat{x}' = R(E_{rec}(x'))
\]

Both are compared against the clean target \(x\), but loss is computed only on normal positions:

\[
M_t = 0
\]

Injected anomalous positions do not contribute to the MSE loss.

Residual implementation choice:

- whether the clean-view reconstruction term and anomalous-view denoising term have equal weights,
- or whether they receive separate coefficients.

### 4. Positive Construction for Normal Anchors

Status: **closed at the semantic level, open at the implementation level**.

For each eligible normal anchor, the intended positive set includes both:

- the aligned token at the same local timestep in the paired augmented view,
- and same-source-timestep tokens from other overlapping windows in the batch.

This means the contrastive loss must support multiple positives per anchor.

Implementation requirement:

- standard one-positive InfoNCE is not enough unless extended,
- a supervised-contrastive or multi-positive InfoNCE form is the more natural match,
- and the first implementation should treat this as an **InfoNCE-like metadata-defined multi-positive objective**, not as standard class-label SupCon.

The later 2026-06-21 note adds another now-closed semantic point:

- positives may be **weakly weighted** according to local receptive-field cleanliness,
- so a positive can remain valid even if its receptive field is partially contaminated,
- but it should pull less strongly than a cleaner positive.

What remains open here is not whether weighting exists, but which first implementation is used:

- uniform theoretical-RF weighting \(\rho^{RF}\),
- or fixed architecture-derived Gaussian-RF weighting \(\rho^{arch}\).

### 5. Negative Pool for Normal Anchors

Status: **mostly closed**.

The negative pool should come from both:

- raw/clean-view timestep tokens,
- synthetic anomalous-view timestep tokens.

The negative pool is within-batch.

The later 2026-06-21 note also narrows one practical first-pass decision:

- injected aligned tokens can safely act as negatives in the denominator,
- even if a separate explicit repulsion term is not added yet.

Residual implementation choices:

- exclude all positives from the denominator,
- decide whether same-source-timestep duplicates that are not selected as positives should also be excluded to avoid false negatives,
- decide whether same-class anomaly tokens should be excluded when the contrastive loss is class-aware.

### 6. Role of Injected Positions in Contrastive Learning

Status: **narrowed, but not fully closed**.

The desired geometry is clear:

- normal latent vectors should move away from anomalous latent vectors,
- class-A anomaly latent vectors should move away from class-B anomaly latent vectors when \(A \neq B\).

The earlier unresolved mathematical point was how to handle a raw clean token whose paired augmented token is injected/anomalous.

The later 2026-06-21 note now gives the following first-pass preference:

- in the first implementation, injected aligned tokens should default to **negative-only roles in denominators**,
- no explicit repulsion term is required in the first pass,
- and a more aggressive repulsion-term variant can be deferred to a later ablation if needed.

So the remaining open point is narrower than before:

- whether to keep the first-pass negative-only treatment permanently,
- or later add a dedicated repulsion term for injected aligned pairs.

### 7. Batch Metadata Required for Overlap-Aware Positives

Status: **closed as a requirement**.

Because positives can come from different overlapping windows that correspond to the same source timestep, the batch contract must expose enough metadata to recover absolute timestep identity.

Required metadata should include at least:

- `entity_id`,
- `series_id` when that identity is not already implied by `entity_id`,
- `window_start`,
- `source_timestep_index` or enough information to compute it for every local timestep,
- local timestep index \(t\) inside the window.

Without this metadata, same-source-timestep positives cannot be implemented reliably.

### 8. Whether Batching Must Guarantee Overlap-Based Positives

Status: **open**.

Stride 1 means one source timestep can appear in multiple windows in the full dataset, but not necessarily inside the same batch.

Two implementation options remain:

1. Opportunistic batching:
   - use same-source-timestep positives when they appear naturally in a batch,
   - fall back to aligned-view positives when no overlap positive exists.

2. Custom overlap-aware batching:
   - deliberately place nearby overlapping windows from the same entity into the same batch,
   - increases positive availability,
   - but changes sampling behavior and may reduce stochastic diversity.

The first implementation can start with opportunistic batching if it is simpler, but the metric logs should report how many overlap positives were actually found.

The later 2026-06-21 note is slightly more opinionated here:

- the first implementation can still start with opportunistic batching,
- but only if the logs explicitly report overlap-positive availability and false-negative filtering behavior.

### 9. Exact Zipping Metric

Status: **open, but bounded**.

The faithful MTZ functional difference is Hessian-based.

The implementation still needs to decide whether to use:

- faithful Hessian-based functional difference,
- activation-distance approximation,
- weight-distance approximation,
- cosine-distance approximation,
- or a staged implementation where a simple approximation is implemented first and the Hessian metric is added later.

Any approximation should be named explicitly as an approximation rather than presented as exact MTZ.

### 10. Zipping Scope and Sharing Ratio

Status: **mostly closed**.

Scope:

- zip encoders only,
- do not zip classification and reconstruction heads.

Preferred first experiment:

- full channel sharing for corresponding 1D-CNN layers.

Critical requirement:

- full sharing should still perform channel matching before merging,
- because channel index equality across independently trained CNNs is not semantically reliable.

### 11. Task-Specific Head Initialization After Zipping

Status: **closed for the first experiment**.

The multitask model should initialize:

- classification head from the Stage 1 classification head,
- reconstruction head from the Stage 1 reconstruction head.

This assumes the zipped encoder preserves the same hidden dimension as the Stage 1 encoders.

### 12. Source of Prototype Initialization Samples

Status: **mostly closed**.

Prototype initialization uses the training part of the selected series/entity.

Discrete prototype initialization:

- class-balanced across 12 classes,
- equal number of windows or tokens per class,
- recommended first method: covering-based selection within each class.

Continuous prototype initialization:

- uses only normal windows/tokens from the training part,
- "normal" currently means likely-normal training data or synthetic-normal positions, but this still requires one final implementation definition.

### 13. Definition of "Normal" for Continuous Prototypes

Status: **partly open**.

The user noted that "normal timesteps" may mean timesteps/windows with synthetic normal labels, but this is not fully settled.

Candidate definitions:

- raw training windows from the original training split, treated as likely normal,
- positions where `synthetic_anomaly_mask == 0`,
- intersection of likely-normal raw training data and synthetic-normal positions,
- heuristic-filtered normal tokens after excluding high reconstruction-error or high anomaly-score tokens.

The first implementation should choose a simple definition and log it explicitly.

### 14. Weak-Positive RF Weighting Semantics

Status: **closed at the conceptual level, open at the formula-selection level**.

The later 2026-06-21 note closes the need for weak-positive weighting conceptually:

- positive strength should depend on how clean the positive token's receptive field is,
- fully clean contexts should pull more strongly,
- partially contaminated-but-still-valid contexts should pull more weakly.

The first implementation should avoid unnecessary extra hyperparameters.

So the open formula choice is now bounded to:

- **uniform theoretical RF weighting**
  - \(\rho^{RF}_{b,t} = \frac{1}{2R+1}\sum_{r=-R}^{R} C_{b,t+r}\),
  - preferred for maximum simplicity,
- or **fixed architecture-derived Gaussian RF weighting**
  - \(\rho^{arch}_{b,t} = \sum_{r=-R}^{R} g_r C_{b,t+r}\),
  - acceptable if described strictly as a fixed prior, not as the true trained ERF.

The first implementation should not add an extra exponent \(\gamma\) unless there is a strong ablation reason.

### 15. Meaning and Lifecycle of \(\sigma_{arch}\)

Status: **closed at the semantic level, open only for exact config parsing in complex CNNs**.

If the Gaussian-RF variant is used:

- \(\sigma_{arch}\) is computed from architecture/config,
- \(\sigma_{arch}\) is computed once before training,
- \(\sigma_{arch}\) remains fixed throughout training,
- and \(\sigma_{arch}\) must be described as a fixed architectural prior rather than as the trained ERF variance.

What remains open is only the exact implementation detail for architectures with:

- mixed kernel sizes,
- non-unit stride,
- non-unit dilation,
- or other encoder details that make the config-to-\(\sigma_{arch}\) mapping less trivial.

### 16. Jacobian-Based ERF Estimation Inside Training

Status: **closed for the first implementation**.

The later 2026-06-21 note is explicit:

- Jacobian-based ERF estimation should not be used inside the main training loop,
- \(\sigma\) should not be recomputed every epoch,
- and any Jacobian-based study, if ever done, should be diagnostic-only rather than part of the core method.

### 17. Ordering of Zipping, Prototype Initialization, and Multitask Training

Status: **closed**.

The intended order is:

1. train Stage 1 classification model,
2. train Stage 1 reconstruction model,
3. zip the two encoders,
4. run short Stage 2 recovery without prototypes,
5. initialize prototypes,
6. run Stage 3 prototype warm-up with encoder frozen,
7. run main multitask pre-training with prototypes.

### 18. Stage 3 Prototype Warm-Up Freezing

Status: **closed**.

During Stage 3 prototype warm-up:

- freeze the zipped shared encoder,
- train/stabilize continuous prototype branch,
- train/stabilize discrete prototype branch,
- train/stabilize CKA-gated fusion heads,
- keep the phase short relative to the full 300-epoch budget.

The purpose is to let the two prototypical branches and CKA-gated fusion heads adapt to the latent geometry without the encoder moving underneath them.

The later 2026-06-21 note adds a stronger first-pass contrastive rule for this stage:

- keep the encoder frozen,
- and keep contrastive loss off in the first implementation.

### 19. Exact 300-Epoch Budget

Status: **closed as the current recommended allocation**.

The current recommended allocation is:

\[
50 + 70 + 20 + 20 + 140 = 300
\]

where:

- 50 epochs for Stage 1 classification,
- 70 epochs for Stage 1 reconstruction,
- 20 epochs for Stage 2 post-zipping recovery,
- 20 epochs for Stage 3 prototype warm-up with frozen encoder,
- 140 epochs for main multitask prototype pre-training.

Stage 2 zipping and Stage 3 prototype initialization themselves use 0 epochs because they are parameter/statistical procedures rather than optimizer training loops.

### 20. Exact Discrete Codebook Query Operator

Status: **mostly closed at the semantic level, still open at the implementation level**.

The current intended discrete query is:

- compute token-to-codeword distances directly in latent space,
- choose top-\(k\) nearest codewords,
- compute a soft weighting only within the selected set,
- aggregate those selected codewords into one queried discrete vector,
- inject it back into the token stream through a residualized branch output.

This is now semantically clearer than the older "learned logits over codebook slots" interpretation.

Residual implementation choices still open:

- exact distance metric,
- normalized versus unnormalized lookup,
- exact residual and normalization rule,
- and whether \(k=1\) should use hard routing only or a soft one-element weighting.

### 21. Backpropagation Through the Discrete Query

Status: **closed at the conceptual level, open at the autograd-policy level**.

The current intended backprop semantics are:

- do not differentiate through changes of the top-\(k\) index set itself,
- do differentiate through distances, softmax weights on the selected set, codeword aggregation, residual addition, and normalization,
- only selected codewords receive direct gradient from a given token in a given forward pass.

Residual implementation choices still open:

- whether \(k=1\) should use standard gather-based autograd only,
- whether a straight-through estimator is needed for hard routing experiments,
- and how this interacts with any EMA-based memory update policy.

### 22. Relationship Between the New Query Idea and the Current Code

Status: **closed as a discrepancy note, open as a migration plan**.

The current repository implementation is still:

- learned assignment logits from a linear head,
- Gumbel-Softmax over all codebook slots,
- dense soft aggregation over the full codebook,
- EMA-based codebook updates.

The newer preferred idea is:

- nearest-neighbor distance to codewords,
- sparse top-\(k\) selection,
- soft aggregation only over the selected neighbors,
- and a backprop path that follows the selected codewords only.

So this note should not be read as saying the codebase already implements the new nearest-neighbor top-\(k\) branch. It records the newer intended direction only.

## Relationship to Current Codebase

At the time of writing this note, the current implementation in:

- `src/models/thesis_multitask.py`

still reflects the earlier active repository design more than the newly discussed three-stage training pipeline.

In particular, the current code already contains:

- multitask forward computation,
- continuous prototype lookup,
- discrete prototype lookup,
- CKA-gated fusion,
- two-view contrastive loss,
- reconstruction loss,
- classification loss,
- memory initialization and memory update logic

but it does not yet encode the newly stated three-stage training process as a formal implementation contract.

It also does not yet encode the newly preferred discrete-branch query semantics recorded above. The current implementation remains assignment-logit-driven rather than distance-top-\(k\)-driven.

## Recommended Next Continuation Point

When resuming this topic in a later chat, the next useful step is:

1. choose the first weak-positive weighting implementation:
   - uniform \(\rho^{RF}\),
   - or fixed architecture-derived Gaussian \(\rho^{arch}\),
2. if the Gaussian variant is used, specify the exact config-to-\(\sigma_{arch}\) mapping for the actual 1D-CNN encoder,
3. define boundary handling for \(t+r\) when the receptive-field window crosses the local window boundary,
4. define the exact batch metadata contract for overlap-aware multi-positive contrastive learning,
5. decide whether Stage 1 batching is opportunistic or custom overlap-aware,
6. decide whether the injected-position case remains negative-only permanently or later gains an explicit repulsion ablation,
7. formalize the exact definition of "normal" for continuous prototype initialization,
8. choose the first zipping metric implementation: faithful Hessian MTZ or an explicitly named approximation,
9. choose the discrete query metric and normalization regime,
10. choose whether the discrete codebook update rule is gradient-based, EMA-based, or hybrid,
11. decide whether `discrete_assignment` is removed, retained for ablation, or reused only as an auxiliary scorer,
12. define the hard-routing policy for \(k=1\): pure gather autograd or straight-through estimator,
13. restate the complete three-stage computational specification with the 300-epoch allocation and updated contrastive placement,
14. decide which parts replace the older `offline_pretraining_phase_two_view_contrastive_design.md` contract and which parts extend it,
15. only then map the design into config surfaces and code changes.

## Primary References Mentioned in the Discussion

- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`
- `documents/design/idea.md`
- `src/models/thesis_multitask.py`
- `papers/He et al. - 2019 - NeurIPS-18 - Multi-Task Zipping via Layer-wise Neuron Sharing.pdf`
