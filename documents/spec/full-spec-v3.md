# Development Specification v3: THESIS Offline, Stochastic Prototype Retrieval, Online TTA, Benchmark, and Demo

**Status:** normative implementation specification  
**Date:** 2026-07-11  
**Supersedes:** `documents/spec/full-spec-v2.md` where this document differs
**Codebase reference:** branch `dev`, commit `fbfd011ac85e94d559201fd2153161e5523ff8af`  
**Primary model:** `ThesisMultitaskModel`  
**Window length:** `L = 20`

---

## 0. Purpose and normative language

This document specifies the code changes required to:

1. preserve the existing two-stage THESIS offline pipeline;
2. add stochastic Gumbel-Softmax retrieval to both memory branches;
3. estimate uncertainty from exactly ten Monte Carlo retrieval samples;
4. preserve a deterministic path for anomalous-cluster filtering and PNN signatures;
5. support causal A0/A1/A2 online execution with projector-only adaptation;
6. minimize additional scientific runs and runtime complexity;
7. keep the existing trainer, evaluator, checkpoint, reporting, and demo contracts compatible.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. Sections marked `OPEN` are not implementation defaults and MUST NOT silently receive guessed values.

Scientific wording:

> The primary uncertainty statistic is the Monte Carlo point-wise anomaly-score variance induced by stochastic prototype retrieval.

The implementation MUST NOT claim that this statistic identifies all epistemic uncertainty of the model or cleanly separates epistemic and aleatoric uncertainty.

---

## 1. Locked scientific design

### 1.1 Data and model constants

```yaml
window_size: 20
input_dim: 38                 # entity-specific validation still required
hidden_dim: 32
num_classes: 12               # class 0 normal + 11 RedLamp anomaly classes

continuous_num_prototypes: 32
discrete_codebook_size: 60
discrete_codewords_per_class: 5
discrete_topk: 3

lambda_recon: 0.5
lambda_cls: 0.5
lambda_contrastive: 0.3
```

The eleven synthetic anomaly types remain:

```text
spike, flip, speedup, noise, cutoff, average,
scale, wander, contextual, upsidedown, mixture
```

### 1.2 Lifecycle

```text
Stage A multitask training
    -> initialize continuous bank and discrete codebook from train latents
    -> compute discrete anomalous-cluster metadata from train/synthetic-train only
    -> freeze encoder, continuous bank, and discrete codebook
    -> Stage B trains only task-specific fusion and prediction heads
    -> calibrate thresholds on clean validation
    -> offline evaluation or causal online A0/A1/A2
```

The banks are initialized **after** Stage A and frozen immediately after initialization. “Frozen after Stage A” means this boundary; it does not mean that initialized banks participated in Stage A.

### 1.3 Stochastic and deterministic boundaries

Stochastic Gumbel-Softmax MUST be used for prediction retrieval in both branches:

- continuous branch: dense soft retrieval over all 32 prototypes;
- discrete branch: stochastic top-3 retrieval over 60 codewords.

Gumbel noise MUST NOT be used for:

- nearest-discrete-codeword anomaly filtering;
- stored anomalous-cluster covering-radius checks;
- ordered continuous top-3 signature extraction;
- memory initialization or radius estimation;
- threshold quantile computation after scores have been produced.

### 1.4 Inference default

Clean validation, synthetic validation, offline test, and online A0/A1/A2 use stochastic inference by default:

```yaml
stochastic_inference: true
monte_carlo_samples: 10
monte_carlo_score_reduction: mean
sample_variance_correction: unbiased
numeric_precision: fp32
```

Exactly ten stochastic retrieval samples are required for the main uncertainty-enabled configuration.

---

## 2. Existing codebase seams and required ownership

The implementation MUST extend the current codebase instead of introducing a parallel training framework.

| Concern | Existing owner | Required v3 change |
|---|---|---|
| SMD parsing | `src/data/datasets/smd.py` | No semantic change |
| Scaling/windowing | `src/data/loaders.py` | Preserve train-only scaler fit; explicit stride per protocol |
| Batch collation | `src/data/collate.py` | Preserve canonical fields |
| Synthetic injection | `src/data/augment.py` | Preserve labels and point anomaly mask |
| Public offline model | `src/models/thesis_multitask.py` | Remain the only public THESIS model entrypoint |
| Forward routing | `src/models/thesis_multitask_routing_mixin.py` | Add deterministic logits and vectorized stochastic retrieval helpers |
| Loss assembly | `src/models/thesis_multitask_loss_mixin.py` | Consume mean predictions during stochastic Stage-B/evaluation paths |
| Output validation | `src/core/contracts.py` | Keep stable top-level keys; uncertainty remains under `aux` |
| Offline trainer | `src/engine/trainer.py` | No Monte Carlo-specific branching outside model contract |
| Evaluator | `src/engine/evaluator.py` | Export uncertainty timelines without changing score semantics |
| Online entrypoint | `src/models/online_adaptation.py` | One-window, source-once stochastic scorer; projector-only mutation |
| Online loop | `src/engine/online_loop.py` | Replace unconditional update loop with triage/event dispatcher |
| Online stream | `src/data/stream.py` | Active full-spec path MUST NOT require `view_a/view_b` |
| Checkpoints | `src/engine/checkpoint.py` | Persist memory provenance, query schema, and deterministic radius metadata |

Recommended logical additions, subject to the repository file-size gate:

```text
src/models/components/stochastic_prototype_query.py
src/models/components/uncertainty_aggregation.py
src/models/components/deterministic_memory_geometry.py
src/engine/thresholds/stochastic_calibration.py
src/engine/online/verification_buffer.py
src/engine/online/verification_cycle.py
src/engine/online/runtime_state.py
src/engine/online/event_dispatch.py
```

Helpers MAY be placed elsewhere to match repository conventions, but MUST NOT create a second public THESIS model.

---

## 3. Canonical data contracts

### 3.1 Raw sequence

```python
{
    "x": FloatTensor[T, D],
    "point_labels": LongTensor[T] | None,
    "mask": Tensor | None,
    "timestamps": Tensor[T] | None,
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "series_id": str,
        "num_channels": int,
        "sequence_length": int,
    },
}
```

`SequenceStandardScaler` MUST be fitted on the entity train split only. Validation, test, and online observations MUST NOT influence scaler parameters.

### 3.2 Offline batch

```python
batch["x"]: FloatTensor[B, 20, D]
batch["point_labels"]: LongTensor[B, 20] | None
batch["mask"]: Tensor[B, 20, D] | None
batch["timestamps"]: Tensor[B, 20] | None
batch["meta"]: list[dict]
```

Synthetic training adds:

```python
batch["classification_labels"]: LongTensor[B]
batch["synthetic_anomaly_mask"]: BoolTensor[B, 20]
```

### 3.3 Online batch

The active online input is exactly one causal window:

```python
online_batch["x"]: FloatTensor[B, 20, D]
online_batch["point_labels"]: None
online_batch["absolute_indices"]: LongTensor[B, 20]
online_batch["timestamps"]: Tensor[B, 20] | None
online_batch["meta"]: list[dict]
```

Labels MAY remain outside the scorer for final metric computation but MUST be absent from scoring, triage, verification, and adaptation calls. Legacy `view_a/view_b` creation MUST be bypassed for v3 configurations.

### 3.4 Stable model output

The public top-level contract remains:

```python
{
    "hidden": FloatTensor[B, 20, H],
    "pooled": FloatTensor[B, H_pool],
    "recon": FloatTensor[B, 20, D],
    "logits": FloatTensor[B, 12],
    "point_scores": FloatTensor[B, 20],
    "window_scores": FloatTensor[B],
    "aux": dict,
}
```

For stochastic inference, `recon`, `point_scores`, and `window_scores` are Monte Carlo means. `logits` is a compatibility representation defined from the mean class probabilities as `log(mean_probability.clamp_min(eps))`; therefore `softmax(logits)` recovers the official mean probability distribution. Per-sample tensors and variances live only under `aux`; they MUST NOT add a leading Monte Carlo dimension to stable top-level fields.

---

## 4. Query configuration and schema

```yaml
query:
  schema_version: 3
  similarity: cosine
  l2_normalize_queries: true
  l2_normalize_memories: true

  continuous:
    mode: gumbel_softmax_dense
    num_prototypes: 32
    hard: false
    topk: null
    temperature: 0.9
    annealing: false

  discrete:
    mode: gumbel_softmax_topk
    codebook_size: 60
    topk: 3
    hard: false
    temperature: 0.9
    annealing: false

  monte_carlo:
    train_samples: 1
    inference_samples: 10
    vectorized: true
    unbiased_variance: true

  deterministic_geometry:
    enabled: true
    nearest_discrete_topk: 1
    continuous_signature_topk: 3
    continuous_signature_ordered: true
```

Separate temperature keys are mandatory even while both defaults equal `0.9`. A later change to one branch MUST NOT implicitly change the other.

No temperature sweep belongs to the main v3 matrix. A pilot MAY report a no-retraining sensitivity analysis, but it MUST be labeled exploratory and MUST NOT select test-optimal temperature.

---

## 5. Vectorized stochastic query operators

Let:

```text
Z       [B,L,H]       latent queries
P_c     [Kc,H]        continuous prototypes, Kc=32
E_d     [Kd,H]        discrete codewords, Kd=60
M       10            inference samples
```

### 5.1 Similarity precomputation

Normalize once and compute logits once:

\[
\ell^c = \bar Z\bar P_c^\top\in\mathbb R^{B\times L\times K_c},
\qquad
\ell^d = \bar Z\bar E_d^\top\in\mathbb R^{B\times L\times K_d}.
\]

The implementation MUST NOT recompute these matrix products inside the Monte Carlo sample dimension.

### 5.2 Gumbel sampling

For uniform random values clipped to the open interval `(eps, 1-eps)`:

\[
G=-\log[-\log(U)].
\]

Required shapes:

```text
G_c [B,M,L,Kc]
G_d [B,M,L,Kd]
```

`eps` MUST use a dtype-safe constant. Tests MUST confirm finite values for extreme pseudo-random inputs.

### 5.3 Continuous dense soft retrieval

\[
\alpha^c_{bmtk}
=\operatorname{softmax}_k
\left(\frac{\ell^c_{btk}+G^c_{bmtk}}{\tau_c}\right),
\]

\[
Z^{c,(m)}_{bt:}=\sum_{k=1}^{K_c}\alpha^c_{bmtk}P_{c,k:}.
\]

Contract:

```text
continuous_weights  [B,M,L,32]
continuous_retrieved[B,M,L,H]
weights > 0 subject to floating-point underflow
weights.sum(-1) ~= 1
```

No `argmax`, hard one-hot, straight-through estimator, or continuous retrieval top-k is permitted in the main configuration.

### 5.4 Discrete stochastic top-3 retrieval

Perturbed logits control both selection and weighting:

\[
\tilde\ell^d=\ell^d+G^d,
\qquad
I^{(m)}=\operatorname{TopK}(\tilde\ell^d,3),
\]

\[
\alpha^{d,(m)}_I
=\operatorname{softmax}
\left(\frac{\tilde\ell^{d,(m)}_I}{\tau_d}\right),
\qquad
Z^{d,(m)}=\sum_{k\in I^{(m)}}\alpha^{d,(m)}_kE_{d,k}.
\]

Contract:

```text
discrete_topk_ids    [B,M,L,3]
discrete_topk_weights[B,M,L,3]
discrete_retrieved   [B,M,L,H]
topk_weights.sum(-1) ~= 1
```

It is invalid to choose `I` from perturbed logits and then weight selected entries with original logits.

### 5.5 Sample-dimension downstream computation

The two retrieved tensors MUST be fused sample-wise. Recommended flattening:

```python
z_cont = z_cont.reshape(B * M, L, H)
z_disc = z_disc.reshape(B * M, L, H)
z_cat = torch.cat([z_cont, z_disc], dim=-1)
recon_m, logits_m = frozen_or_trainable_heads(z_cat)
recon_m = recon_m.reshape(B, M, L, D)
logits_m = logits_m.reshape(B, M, 12)
```

The model MUST NOT duplicate encoder computation across `M`. During online A1/A2, the projector is also evaluated once before similarity computation; only memory retrieval and downstream heads expand across samples.

### 5.6 Training sample count

Stage B training uses `M_train=1` stochastic sample per forward to avoid a tenfold training expansion. Validation, threshold calibration, synthetic validation, test, and official online scoring use `M_eval=10`.

This asymmetry MUST be recorded in the resolved config and checkpoint metadata.

---

## 6. Deterministic memory geometry

### 6.1 Known anomalous-cluster filter

Use unperturbed cosine logits and frozen source latent tokens:

\[
k^*_{bt}=\arg\max_k\ell^d_{btk},
\qquad
d_{bt}=1-\ell^d_{bt,k^*}.
\]

\[
M^{known}_{bt}
=\mathbf 1[
anomalous\_codeword\_mask[k^*]
\land d_{bt}\le R_{k^*}
].
\]

Online filtering MUST use `Z_source`, not a mutable projected latent, so projector updates cannot weaken the guard.

### 6.2 Continuous PNN signature

For tokens not marked known anomaly, compute ordered top-3 prototype IDs using unperturbed continuous cosine similarity:

```python
signature_ids = continuous_logits.topk(k=3, dim=-1).indices
# [N,20,3], sorted by descending similarity
```

A signature is recurrent only if it occurs in more than one non-overlapping admitted window. Stochastic retrieval IDs MUST NOT be used as signatures.

### 6.3 Radius metadata

For each anomalous codeword, persist:

```text
anomaly_radii[Kd]
anomalous_codeword_mask[Kd]
codeword_class_ids[Kd]
distance_family=cosine
radius_quantile=0.99
contributing_token_count per codeword
source split and synthetic class
initialization seed
schema version
```

A1/A2 fail closed on missing, negative, non-finite, shape-incompatible, or provenance-incomplete metadata.

---

## 7. Offline two-stage training

### 7.1 Stage A

Stage A operates before memory initialization. It retains the current multitask path and losses:

\[
\mathcal L_A
=\lambda_{recon}\mathcal L_{recon}
+\lambda_{cls}\mathcal L_{cls}
+\lambda_{contrastive}\mathcal L_{contrastive}
+\mathbf 1[O1]\mathcal L_{point\text{-}score}.
\]

O0 disables point-wise balanced reconstruction-score loss. O1 enables it only in Stage A. O1 MUST NOT introduce a new main `lambda_score`; retain the balanced formulation already specified by v2.

Stage A does not use continuous or discrete memory retrieval because the final banks do not yet exist.

### 7.2 Memory construction boundary

Run the Stage-A encoder in `eval()` and `no_grad()` over allowed train tokens only.

Continuous bank:

- collect normal/clean train latent tokens;
- exclude injected anomaly positions;
- construct 32 covering prototypes using the configured k-means/covering procedure.

Discrete codebook:

- class 0 uses normal train tokens;
- classes 1–11 use injected anomaly tokens from the matching synthetic class;
- construct exactly five codewords per class.

Validation, test, and future online data are forbidden.

### 7.3 Stage B

Frozen:

```text
encoder
continuous prototype bank
discrete codebook
deterministic radius metadata
```

Trainable:

```text
reconstruction fusion head
classification fusion head
reconstruction prediction head/decoder
classification prediction head
```

Stage-B forward uses stochastic retrieval with `M_train=1`. Stage-B loss is:

\[
\mathcal L_B=\mathcal L_{recon}+\mathcal L_{cls}.
\]

Do not use Stage-A contrastive or point-score BCE in Stage B by default.

### 7.4 Epoch budgets

Main budget:

```yaml
stage_a_epochs: 25
stage_b_epochs: 5
```

Smoke-only budget:

```yaml
stage_a_epochs: 1
stage_b_epochs: 1
```

Smoke runs MUST be marked non-scientific and excluded from aggregation.

---

## 8. Monte Carlo outputs and uncertainty statistics

Let `M=10`.

### 8.1 Mean prediction contract

```text
recon_samples       [B,M,L,D]
classification_probs[B,M,12]
point_score_samples [B,M,L]
window_score_samples[B,M]
```

Top-level reconstruction and scores are means over `M`:

\[
\bar X=\frac1M\sum_m\hat X^{(m)},\quad
\bar p=\frac1M\sum_m p^{(m)},\quad
\bar s_t=\frac1M\sum_m s_t^{(m)},\quad
\bar S=\frac1M\sum_m S^{(m)}.
\]

Classification prediction is `argmax(mean_probability)`, not majority vote over sample-wise class IDs.

For compatibility with code that expects top-level logits:

```python
mean_probability = classification_probs.mean(dim=1)
outputs["logits"] = mean_probability.clamp_min(eps).log()
```

The official mean point score is the mean of the ten per-sample MSE scores. It MUST NOT be recomputed as the MSE between `x` and the mean reconstruction because, in general,

\[
\frac1M\sum_m\lVert x-\hat x^{(m)}\rVert^2
\ne
\left\lVert x-\frac1M\sum_m\hat x^{(m)}\right\rVert^2.
\]

### 8.2 Score construction

Point-wise MSE for sample `m`:

\[
s^{(m)}_{bt}=\frac1D\sum_d(x_{btd}-\hat x^{(m)}_{btd})^2.
\]

Window MSE for sample `m`:

\[
S^{(m)}_b=\frac1L\sum_t s^{(m)}_{bt}.
\]

`window_score_variance` MUST be variance across `S^(m)`. It MUST NOT be replaced by the mean of point variances because temporal covariance would be lost.

### 8.3 Unbiased sample variance

For every reported random tensor `v`:

\[
\widehat{Var}_m[v]
=\frac1{M-1}\sum_{m=1}^{M}(v^{(m)}-\bar v)^2.
\]

PyTorch implementation MUST use `correction=1` or an explicitly equivalent formula.

### 8.4 Primary uncertainty

Primary statistic:

```text
point_anomaly_score_variance [B,L]
```

\[
u^{score}_{bt}=\widehat{Var}_m[s^{(m)}_{bt}].
\]

Also export:

```text
window_anomaly_score_variance [B]
```

### 8.5 Retrieval uncertainty

Branch-wise point statistics:

\[
u^c_{bt}=\frac1H\sum_h\widehat{Var}_m[Z^{c,(m)}_{bth}],
\quad
u^d_{bt}=\frac1H\sum_h\widehat{Var}_m[Z^{d,(m)}_{bth}].
\]

Window summaries are arithmetic means over valid time-points. Log continuous and discrete values separately. An optional fused retrieval variance MAY be logged, but cannot replace branch-wise values.

### 8.6 Reconstruction uncertainty

Per-channel reconstruction variance:

```text
reconstruction_variance_full [B,L,D]
```

Scalar point summary:

\[
u^{recon}_{bt}=\frac1D\sum_d\widehat{Var}_m[\hat x^{(m)}_{btd}].
\]

Window summary is the mean across valid time-points. Full `[B,L,D]` tensors SHOULD be exported to artifacts but SHOULD NOT be emitted as one W&B scalar per element.

### 8.7 Classification uncertainty

For:

\[
p^{(m)}=softmax(a^{(m)})\in\mathbb R^{12},
\]

compute class-wise probability variance:

\[
u^{cls}_{bc}=\widehat{Var}_m[p^{(m)}_{bc}].
\]

Scalar summary:

\[
u^{cls}_b=\frac1{12}\sum_{c=1}^{12}u^{cls}_{bc}.
\]

Do not compute variance of integer class IDs. Classification uncertainty is window/sample-level because the current classification head outputs one distribution per window. It MUST NOT be mislabeled point-level unless the architecture is later changed to produce token-level class logits.

### 8.8 Role in v3 decisions

All uncertainty values are diagnostic except that Monte Carlo mean scores are the official prediction scores. Variance MUST NOT alter:

- anomaly thresholds;
- four-region triage;
- verification-buffer admission;
- TTL;
- PNN recurrence;
- adaptation loss weights;
- optimizer steps.

Adding any such mechanism requires a new explicit uncertainty-aware ablation.

---

## 9. `aux` output schema

```python
outputs["aux"]["stochastic_query"] = {
    "schema_version": 3,
    "enabled": bool,
    "num_samples": int,
    "continuous_temperature": float,
    "discrete_temperature": float,
    "continuous_retrieved_samples": Tensor[B,M,L,H] | None,
    "discrete_retrieved_samples": Tensor[B,M,L,H] | None,
    "discrete_topk_ids": LongTensor[B,M,L,3] | None,
    "reconstruction_samples": Tensor[B,M,L,D] | None,
    "classification_probability_samples": Tensor[B,M,12] | None,
    "point_score_samples": Tensor[B,M,L] | None,
    "window_score_samples": Tensor[B,M] | None,
}

outputs["aux"]["uncertainty"] = {
    "point_anomaly_score_variance": Tensor[B,L] | None,
    "window_anomaly_score_variance": Tensor[B] | None,
    "continuous_retrieval_variance_point": Tensor[B,L] | None,
    "continuous_retrieval_variance_window": Tensor[B] | None,
    "discrete_retrieval_variance_point": Tensor[B,L] | None,
    "discrete_retrieval_variance_window": Tensor[B] | None,
    "reconstruction_variance_full": Tensor[B,L,D] | None,
    "reconstruction_variance_point": Tensor[B,L] | None,
    "reconstruction_variance_window": Tensor[B] | None,
    "classification_probability_variance": Tensor[B,12] | None,
    "classification_variance_mean": Tensor[B] | None,
}

outputs["aux"]["deterministic_geometry"] = {
    "nearest_codeword_ids": LongTensor[B,L] | None,
    "nearest_codeword_distances": FloatTensor[B,L] | None,
    "known_anomaly_mask": BoolTensor[B,L] | None,
    "continuous_signature_ids": LongTensor[B,L,3] | None,
    "latent_window_score": FloatTensor[B] | None,
}
```

Memory-heavy sample tensors MAY be omitted after aggregation when `return_mc_samples=false`. Required means and uncertainty summaries MUST remain available. Training defaults SHOULD avoid retaining unnecessary sample tensors after loss computation.

---

## 10. Threshold calibration

### 10.1 Entity artifact

Each entity owns an independent threshold artifact:

```json
{
  "schema_version": 3,
  "entity_id": "machine-3-9",
  "window_size": 20,
  "calibration_split": "clean_validation",
  "stochastic_inference": true,
  "monte_carlo_samples": 10,
  "continuous_temperature": 0.9,
  "discrete_temperature": 0.9,
  "score_reduction": "mean",
  "variance_correction": 1,
  "numeric_precision": "fp32",
  "offline_point_threshold_nonoverlap": 0.0,
  "online_point_threshold_ewma": 0.0,
  "B_window": 0.0,
  "A_low": 0.0,
  "A_high": 0.0,
  "offline_stride": 20,
  "online_stride": 1,
  "ewma_current_weight": 0.9,
  "ewma_previous_weight": 0.1,
  "checkpoint_sha256": "...",
  "resolved_config_sha256": "...",
  "seed": 0,
  "created_at": "..."
}
```

### 10.2 Calibration identity

Calibration runs in `eval()` and `no_grad()` using the exact official stochastic inference contract. A threshold calibrated from one-pass scores MUST NOT be used with ten-sample mean scores.

Clean validation alone calibrates anomaly thresholds. Synthetic validation MAY report classification and uncertainty diagnostics but MUST NOT set anomaly thresholds.

### 10.3 Offline and online timelines

- offline calibration/evaluation: non-overlapping windows, stride `20`, end-aligned handling explicitly recorded;
- online calibration: sliding windows, stride `1`, absolute-index point aggregation, EWMA `0.9 current + 0.1 previous`;
- the two score timelines MUST NOT share one threshold value by assumption.

---

## 11. Offline evaluation

Offline test labels remain outside model inference and are joined only after fixed scores exist. Required exports per entity:

```text
absolute_index
timestamp optional
mean_point_score
point_score_variance
mean_reconstruction
reconstruction_variance_point
continuous_retrieval_variance_point
discrete_retrieval_variance_point
ground_truth_label added only by evaluator
predicted_label
```

Window export additionally includes classification mean probabilities, class-wise probability variance, mean classification variance, mean window score, and window score variance.

Primary metrics remain raw point-level metrics. VUS-PR and Affiliation-F1 are public report metrics; VUS-ROC may be retained internally. One-class slices record unavailable metrics and support rather than fabricated values.

---

## 12. Online forward and causal prediction

### 12.1 Source-once contract

For each latest causal window:

```text
Z_source = frozen_source_encoder(x)             # exactly once
A0: Z_query = Z_source
A1/A2: Z_query = online_mlp_projector(Z_source)
similarity logits computed once from Z_query
10 stochastic retrieval samples vectorized
frozen Stage-B memories/heads produce 10 predictions
means and variances aggregated
```

The source encoder MUST NOT be repeated for each Monte Carlo sample.
If a gray-zone causal window is admitted and verified in the same event, the
verification cycle MUST reuse that event's `reference_hidden` for the new
entry. It may encode older buffered entries independently. The cached tensor is
event-local and MUST NOT be serialized into `verification_buffer` or runtime
state.

### 12.2 Point prediction

The official point score is the Monte Carlo mean score. Each causal window
stores `absolute_indices [L]`, `window_point_scores [L]`,
`current_window_ewma_point_scores [L]`, and `window_point_predictions [L]`.
For a newly seen point, EWMA equals the current score. For a point that appears
again in an overlapping window, EWMA uses the previous value for that same
absolute index. Runtime state keeps only the active point map needed for the
next causal window.

The uncertainty timeline MUST use the same absolute indices; its aggregation
rule MUST be explicit. Default:

```yaml
online_uncertainty_overlap_aggregation: same_ewma_as_score
```

The record of one event is immutable after emission. A later overlapping causal
window may refresh its own prediction for a shared absolute index. Adaptation
only affects future causal windows.

### 12.3 Four-region triage

Use Monte Carlo mean window reconstruction score `s_input` and deterministic latent-memory score `s_latent`:

```text
s_input <= B_window                                normal
s_input > B_window and s_latent <= A_low           hard_old
s_input > B_window and A_low < s_latent <= A_high  gray_zone
s_input > B_window and s_latent > A_high           strong_anomaly
```

Uncertainty variance does not participate in this truth table.

Required event order:

```text
score -> uncertainty aggregation -> EWMA -> triage
      -> permitted update/admission -> verification if due
      -> event record -> runtime checkpoint
```

Online reads `task.threshold_artifact_path`. That file MUST be schema version 4
and MUST match the Stage B checkpoint hash, entity, offline variant, seed,
window size, and EWMA weights. Online MUST NOT calibrate thresholds from its
test stream. A new vector runtime MUST reject older scalar artifact and runtime
state schemas.

---

## 13. Verification buffer and PNN

### 13.1 Admission

Only gray-zone windows may be admitted. Entries MUST be non-overlapping by absolute interval. Adjacent intervals are allowed.

Entry schema:

```python
{
    "entry_id": str,
    "entity_id": str,
    "start_index": int,
    "end_index": int,
    "x": FloatTensor[20,D],
    "status": "unresolved" | "adapted",
    "ttl_remaining": 2,
    "admitted_at_cursor": int,
}
```

### 13.2 Verification trigger

A cycle starts only when:

- buffer capacity is at least eight entries;
- at least one new entry has been admitted since the preceding cycle;
- no cycle is already active.

### 13.3 Verification computation

Verification performs independent frozen-source encoding for stored windows, with labels absent. It computes deterministic geometry:

```text
hidden                      [N,20,H]
nearest_codeword_ids        [N,20]
nearest_codeword_distances  [N,20]
known_anomaly_mask          [N,20]
continuous_signature_ids    [N,20,3]
pnn_mask                    [N,20]
```

Tokens inside anomalous codeword radii are excluded. A remaining signature becomes recurrent only after appearing in more than one non-overlapping window.

### 13.4 TTL

Adapted entries are removed after a successful atomic update. Unresolved entries lose exactly one TTL only when a verification cycle completes; remove them at zero. A failed adaptation MUST NOT mark an entry adapted or consume an adaptation success.

`ttl_remaining` belongs to a `verification_entry` inside
`verification_buffer`. It is not the TTL of an endpoint, point, absolute index,
or causal window outside the buffer.

---

## 14. Online adaptation

### 14.1 Update surface

Only parameters whose names start with `online_mlp_projector` may have `requires_grad=true` during an accepted event.

Frozen online:

```text
source encoder
continuous bank
discrete codebook
both query modules' parameters/state
fusion heads
reconstruction head
classification head
radius metadata
```

Random Gumbel samples are runtime inputs, not trainable state.

### 14.2 A0, A1, A2

- A0: inference only; projector is not called and no optimizer exists.
- A1: verified non-empty PNN masked reconstruction only.
- A2: guarded hard-old or verified non-empty PNN reconstruction plus source-consistency contrastive regularization.

### 14.3 Losses

\[
L_{hard}=relu(S_{online}-B_{window})^2.
\]

\[
L_{pnn}=\frac{\sum ((\hat X-X)^2\odot M_{pnn})}
{\max(1,\sum M_{pnn}\cdot D)}.
\]

```text
A1 PNN: L_total = L_pnn
A2 hard: L_total = L_hard + lambda_contrastive * L_online_contrastive
A2 PNN:  L_total = L_pnn  + lambda_contrastive * L_online_contrastive
```

For adaptation loss, stochastic retrieval uses the configured training sample count `M_train=1` unless a separate ablation explicitly changes it. The pre-update official prediction still uses `M_eval=10`.

### 14.4 Atomic event

Each accepted event:

1. asserts the trainable-parameter allowlist;
2. creates a fresh AdamW with `lr=1e-4`, `weight_decay=1e-4`;
3. zeroes gradients;
4. computes one finite loss;
5. backpropagates;
6. asserts all frozen gradients are absent or zero;
7. clips projector gradient norm at `0.5`;
8. performs exactly one optimizer step;
9. commits buffer/guard/update state only after success.

No online scheduler and no serialized optimizer moments.

---

## 15. Runtime optimizations

Only two optimizations are in scope:

### 15.1 Precompute logits once

Cosine similarity matrix multiplication occurs once for each branch and input latent. Broadcast/expand the result across `M`; do not materialize copies when an expanded view is sufficient.

### 15.2 Vectorize ten samples

All ten Gumbel samples are represented by a tensor sample dimension and evaluated without a Python loop in the main query path.

Out of scope for v3:

```text
mixed precision
single-point CNN feature caching
CUDA Graphs
custom Triton kernels
manual kernel fusion
architecture changes for causal convolution caching
```

All official computation remains FP32.

---

## 16. Minimal experiment design

Do not construct a Cartesian product of offline loss, online TTA, and uncertainty choices.

### 16.1 THESIS configurations

| ID | Point-score loss | Online evaluation | Stochastic query | Purpose |
|---|---:|---|---:|---|
| T0 | Off | A0 | On, M=10 | Point-score-loss ablation |
| T1 | On | A0 and A2 from reset states | On, M=10 | Main THESIS and online-TTA ablation |
| T2 | On | A2 | Off | Stochastic-query uncertainty ablation |

T2 replaces both stochastic query operators by deterministic counterparts while preserving banks, heads, thresholds calibrated for that counterpart, data splits, seed policy, and A2 logic. T2 does not log Monte Carlo variance.

A1 remains supported for diagnostic execution but is not required in the minimum main run plan.

### 16.2 Run meaning

A “run” is one W&B run. One offline checkpoint MAY execute multiple named evaluation phases within the same run only if:

- each phase starts from an explicitly restored immutable checkpoint;
- online runtime state is reset;
- phase metrics use distinct namespaces;
- artifacts record phase identity and full config;
- failures are recorded per phase.

Scientific random seeds multiply the above configurations only when the final reporting policy requires repeated seeds. Smoke, debug, and interrupted runs are never aggregated.

### 16.3 Baselines

Deep-learning and traditional-ML baselines retain their own native prediction protocols. They MUST NOT inherit:

- THESIS Gumbel sampling;
- ten-pass inference;
- THESIS four-region triage;
- PNN verification;
- projector updates;
- THESIS uncertainty ablations.

This avoids multiplying baseline runs merely to mirror a THESIS-specific uncertainty mechanism.

---

## 17. Logging and artifacts

### 17.1 Scalar logs

```text
query/continuous_temperature
query/discrete_temperature
query/num_samples_train
query/num_samples_eval
query/continuous_weight_entropy_mean
query/discrete_topk_weight_entropy_mean

uncertainty/point_score_variance_mean
uncertainty/point_score_variance_p95
uncertainty/window_score_variance_mean
uncertainty/continuous_retrieval_variance_point_mean
uncertainty/continuous_retrieval_variance_window_mean
uncertainty/discrete_retrieval_variance_point_mean
uncertainty/discrete_retrieval_variance_window_mean
uncertainty/reconstruction_variance_point_mean
uncertainty/reconstruction_variance_window_mean
uncertainty/classification_variance_mean
```

Retain existing offline loss, memory, frozen-gradient, online triage, buffer, adaptation, coverage, timing, and metric logs from v2.

### 17.2 Array artifacts

Per-entity compressed artifact SHOULD contain:

```text
absolute indices
mean point scores
point score variances
point reconstruction variances
continuous/discrete retrieval variances
mean class probabilities per window
class probability variances per window
window score means/variances
predictions
labels joined after prediction
```

Store full Monte Carlo samples only when `export_mc_samples=true`; default false for the main matrix. Summary statistics are mandatory.

### 17.3 Identity and integrity

Every artifact records:

```text
schema, entity, variant, phase, seed, git commit/dirty flag
checkpoint/config/threshold hashes
query temperatures and sample counts
processed counts and expected counts
metric definitions and supports
file checksum and completion status
```

Completion manifests are written last and checksum-read back. Incomplete or identity-mismatched artifacts MUST NOT aggregate.

---

## 18. Checkpoint and resume

Stage-B checkpoint must serialize:

```text
model state
continuous/discrete bank tensors
bank initialization provenance
codeword class IDs and anomalous mask
anomaly radii and radius provenance
query schema and temperatures
M_train and M_eval
frozen parameter manifest
source config and checkpoint hashes
```

Online runtime state owns cursor, EWMA state, provisional/final predictions, verification entries, cycle state, signature history, hard-old intervals, update count, threshold identity, RNG state required for reproducible continuation, and schema version.

Resume validates entity, variant, seed, window size, checkpoint hash, threshold hash, query schema, temperatures, Monte Carlo count, and runtime-state schema. It resumes at the next unseen point. Uninterrupted and resumed event traces MUST match except timing fields when deterministic execution settings are enabled.

---

## 19. Demo contract

Two modes remain:

1. offline test replay from exported fixed predictions;
2. online queue replay using the same causal scorer and runtime state.

The queue producer emits points in strict entity order. The consumer waits for exactly 20 points before the first score. Displayed diagnostics MAY include:

```text
mean anomaly score and threshold
point anomaly-score variance
continuous/discrete retrieval variance
reconstruction variance
classification mean probabilities and uncertainty
triage region
buffer/adaptation events
```

Labels are optional post-prediction overlays. The demo MUST NOT tune thresholds, mutate official artifacts, select update events using labels, or produce official metrics.

---

## 20. Tests

### 20.1 Query mathematics

- `[B,L,H] x [K,H]` logits have exact expected shapes.
- similarities are computed once per branch, verified by spy/counter.
- continuous weights sum to one and remain soft/dense.
- discrete IDs have shape `[B,M,L,3]`, contain no duplicates within top-3, and weights sum to one.
- discrete selection and weighting both consume the same perturbed logits.
- vectorized outputs match a seeded reference loop within FP32 tolerance.
- Gumbel generation is finite.
- `M=10` is enforced in official inference configs.

### 20.2 Uncertainty

- all ten identical samples yield zero variance.
- a hand-computed tensor matches unbiased variance with denominator nine.
- window-score variance is computed from per-sample window MSE.
- class-wise probability variance has shape `[B,12]`.
- classification scalar equals the mean of 12 class variances.
- no variance of class indices exists.
- top-level predictions equal sample means.
- `softmax(top_level_logits)` equals the official mean class probabilities.
- mean point scores are means of per-sample MSE values, not MSE of mean reconstruction.
- `return_mc_samples=false` preserves all required summaries.

### 17.2 Online vector records and state

THESIS `online_event_record` stores `absolute_indices` only under
`causal_window`. It stores the three point vectors with the same length. Legacy
endpoint fields may remain only as compatibility fields for scalar readers.

`online_runtime_state` stores `active_ewma_point_scores`, verification entries,
verification history, hard-old intervals, and a `stream_cursor` equal to the
number of processed causal windows. It does not store a cross-cycle
`recurrent_signature_set`.

### 20.3 Deterministic geometry

- changing Gumbel RNG does not change nearest codeword, known-anomaly mask, radii decision, or continuous signatures.
- changing projector parameters does not change source-latent known-anomaly filtering for the same source window.
- anomalous-radius boundary is inclusive.
- invalid metadata fails A1/A2 before stream mutation.

### 20.4 Offline lifecycle

- Stage A completes before bank initialization.
- banks have shapes `[32,H]` and `[60,H]`.
- discrete codebook has five codewords per class.
- validation/test tokens never enter memories or radii.
- Stage B trainable allowlist contains only fusion/prediction heads.
- frozen encoder/bank gradients are zero.
- O1 point-score loss is Stage-A-only.

### 20.5 Calibration/evaluation

- calibration and evaluation use identical query mode, `M`, temperatures, precision, and score reduction.
- clean validation is the only anomaly-threshold source.
- offline and online threshold artifacts cannot be interchanged.
- labels are absent until metrics join.
- absolute-index aggregation preserves entity length and ordering.
- online point vectors preserve their causal-window ordering.

### 20.6 Online

- source encoder is called exactly once per causal window.
- A0 never calls projector or optimizer.
- only projector changes in A1/A2.
- uncertainty variance does not affect triage or admission.
- non-overlap buffer guard works for adjacent and overlapping intervals.
- recurrence requires more than one non-overlapping window.
- one cycle decrements TTL exactly once.
- failed atomic update commits no state.
- predictions are not rewritten retroactively.
- an overlapping later causal window may emit a newer value for its shared point.
- expected causal forwards equal `max(0,T-20+1)`.

### 20.7 Artifacts/resume/demo

- hash mismatch rejects resume.
- resumed trace matches uninterrupted trace under deterministic settings.
- incomplete artifacts do not aggregate.
- demo cannot pass labels into scorer.
- queue pause/resume/stop retains causal order.

### 20.8 Repository gates

Maintain repository readability gates from v2:

```text
one public offline model entrypoint
one public online entrypoint
src callable <= 50 lines
src Python file <= 500 lines
full pytest suite passes
```

---

## 21. Data-leakage and fairness checklist

Forbidden:

- fitting scaler on validation/test;
- using validation/test latents in memories or radii;
- calibrating anomaly thresholds on synthetic validation or test labels;
- selecting temperature from test results;
- using future online windows;
- using labels in scoring, triage, verification, TTL, PNN, or adaptation;
- giving baselines THESIS-specific extra data or test-informed tuning;
- comparing stochastic test scores with thresholds calibrated from a different inference protocol.

Required:

- split and entity identity in every artifact;
- same preprocessing for comparable methods where method assumptions permit;
- explicit method-native deviations for baselines;
- complete stream coverage for main online results;
- raw point metrics reported before adjusted metrics;
- failed and unavailable cells reported rather than omitted.

---

## 22. Implementation sequence

1. Add v3 config schema and fail-fast validation.
2. Implement standalone vectorized continuous and discrete query helpers.
3. Add deterministic geometry helper and prove RNG independence.
4. Integrate sample-wise heads while preserving top-level model output.
5. Add uncertainty aggregation and `aux` schema.
6. Extend Stage-B checkpoint provenance and freeze assertions.
7. Update clean/synthetic validation and evaluator export.
8. Recalibrate entity threshold artifact using `M=10`.
9. Migrate online input from legacy two-view path to one causal window.
10. Integrate source-once stochastic scorer with deterministic filtering.
11. Integrate buffer/cycle/atomic projector-only adaptation.
12. Add W&B namespaces and compressed uncertainty artifacts.
13. Add T0/T1/T2 launch configs without Cartesian expansion.
14. Update demo only after scorer and artifacts pass acceptance tests.
15. Run CPU tests/preflight, then CUDA smoke gates, then main runs.

---

## 23. Acceptance criteria

V3 is implementation-complete only when:

- both query branches use their specified Gumbel-Softmax operators;
- prediction uses ten vectorized samples and logits are computed once;
- continuous retrieval is dense soft mode;
- discrete top-3 selection/weighting is mathematically coupled;
- deterministic anomaly filtering and signatures are RNG-independent;
- all four uncertainty families are exported with correct shapes;
- point anomaly-score variance is clearly marked primary;
- classification uncertainty uses probability variance;
- calibration and evaluation inference identities match;
- Stage B and online frozen-surface assertions pass;
- A0/A2 complete full causal streams;
- T0/T1/T2 configurations are reproducible and artifact-complete;
- baselines remain isolated from THESIS-specific uncertainty mechanics;
- all tests, integrity checks, readability gates, and CUDA smoke gates pass.

---

## 24. Final locked decisions

```text
[LOCKED] Window length is 20.
[LOCKED] Continuous bank size is 32; discrete codebook size is 60.
[LOCKED] Both prediction query operators use Gumbel-Softmax.
[LOCKED] Continuous query is dense soft mode, not hard/top-k.
[LOCKED] Discrete query uses stochastic top-3 with perturbed-logit weighting.
[LOCKED] Initial continuous and discrete temperatures are separately configured as 0.9.
[LOCKED] No temperature annealing or main temperature sweep.
[LOCKED] Official stochastic inference uses exactly 10 samples.
[LOCKED] Similarity logits are computed once before Monte Carlo expansion.
[LOCKED] Samples are vectorized; no Python sample loop in the main query path.
[LOCKED] Official numeric precision is FP32.
[LOCKED] Deterministic anomalous-radius filtering uses frozen source geometry.
[LOCKED] Deterministic PNN signatures use ordered continuous top-3 IDs.
[LOCKED] Point-wise anomaly-score variance is the primary uncertainty statistic.
[LOCKED] Retrieval, reconstruction, window-score, and classification uncertainty are logged diagnostics.
[LOCKED] Classification uncertainty is variance of class probabilities, never class IDs.
[LOCKED] Variance does not affect thresholding, triage, buffers, PNN, or adaptation in v3.
[LOCKED] Clean validation alone calibrates anomaly thresholds using the same M=10 protocol.
[LOCKED] Stage B freezes encoder and both memories; trains only fusion/prediction heads.
[LOCKED] Online A1/A2 update only the light-weight MLP projector.
[LOCKED] No two augmented views are required online.
[LOCKED] No mixed precision, point-feature cache, CUDA Graph, or Triton optimization.
[LOCKED] Minimum uncertainty study uses T0/T1/T2, not a full Cartesian product.
[LOCKED] Baselines do not inherit THESIS-specific stochastic inference or uncertainty ablations.
```

---

## 25. Open values that must remain explicit

The following are not newly decided by stochastic uncertainty and MUST retain their validated v2/entity-specific definitions:

```text
exact quantiles or estimators for A_low and A_high
exact B_window quantile if not already locked by entity config
final entity list and seed count for the published benchmark
whether A1 appears in the final paper beyond diagnostic reporting
final W&B grouping/project naming convention
```

No implementation may infer these from test labels or silently substitute defaults.
