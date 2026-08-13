# Development Specification: Two-Phase THESIS Pipeline

> **Notation authority:** Khi đối chiếu anomaly score mức điểm, tài liệu lịch sử này dùng mapping trong [Thiết kế anomaly score mức điểm và bộ ký hiệu chuẩn](anomaly-score-designs-and-notation.md). Tên runtime và ngữ nghĩa lịch sử trong thân tài liệu được giữ nguyên.


## 0. Status

This document specifies the current two-phase design of **THESIS**:

1. **Offline pre-training phase**
2. **Online test-time adaptation phase**

The current design direction is:

```text
Final anomaly decision: point-level
Offline training: two-stage source model training
Source memories: initialized from train split only
Online TTA: sliding-window stream
Online adaptation trigger: window-level triage + latent time-point verification
Verification buffer: non-overlapping windows only
Online adaptation target: small online branch only
Source model and source memories: frozen
```

Unresolved design choices are explicitly marked as:

```text
Status: Undecided
```

---

# Part I. Core Concepts

## 1. Tensor Notation

Input window:

```text
X ∈ R[B, L, C]
```

Current working configuration:

```text
L = 20
C = 38 for SMD-like multivariate data
```

Encoder output:

```text
Z = Eθ(X)
Z ∈ R[B, L, d_model]
```

Current working configuration:

```text
d_model = 32
```

A latent time-point is:

```text
z[i,t] ∈ R[d_model]
```

Continuous prototype bank:

```text
P_c ∈ R[K_c, d_model]
K_c = 32
```

Discrete codebook:

```text
E_d ∈ R[K_d, d_model]
K_d = 60
```

The discrete codebook is class-stratified:

```text
12 classes = 1 normal class + 11 synthetic anomaly classes
5 codewords per class
12 × 5 = 60 codewords
```

---

## 2. Data Split Protocol

## 2.1 Training Split

The training split may be used for:

```text
train encoder
train reconstruction head
train classification head
train contrastive objective
generate synthetic anomaly windows
initialize continuous prototype bank
initialize discrete codebook
compute discrete anomalous cluster radii
```

The training split must not use:

```text
validation labels
test labels
validation windows for k-means
test windows for k-means
validation/test statistics for train-time loss normalization
```

## 2.2 Clean Validation Set

The clean validation set is used for threshold calibration only.

Allowed:

```text
calibrate point-level anomaly threshold
calibrate window-level reconstruction threshold
calibrate latent triage thresholds
simulate online sliding-window + EWMA for online threshold calibration
```

Forbidden:

```text
backpropagation
memory initialization
prototype fitting
codebook fitting
online adaptation training
```

## 2.3 Synthetic Validation Set

Synthetic validation set is used for diagnostic analysis.

It may be used to inspect:

```text
point-score separation
window-score separation
classification behavior
```

It should not be used as the main operational threshold source unless explicitly reported as an ablation.

## 2.4 Offline Test Set

The offline test set is cut into non-overlapping windows.

It is used only for final evaluation.

## 2.5 Online Test-Time Stream

Online TTA uses sliding windows:

```text
Wτ = x[τ-L+1 : τ]
```

Since sliding windows overlap, one absolute time-point may receive multiple point-level anomaly scores.

These scores are aggregated using EWMA.

---

# Part II. Offline Pre-training Phase

## 3. Offline Phase Objective

The offline phase produces the **source model** and **source memories**.

The source model should learn:

```text
normal time-points → low reconstruction MSE
synthetic anomalous time-points → high reconstruction MSE
normal/anomaly synthetic classes → separable latent geometry
```

The source memories should represent:

```text
continuous prototype bank:
    clean / normal latent structure

discrete codebook:
    class-stratified synthetic pattern structure
```

The source model and source memories will later remain frozen during online TTA.

---

## 4. Offline Phase Overview

```text
Stage A:
    train encoder + heads using multitask losses

End of Stage A:
    initialize continuous prototype bank by k-means
    initialize discrete codebook by class-wise k-means
    compute anomalous discrete cluster radii

Stage B:
    freeze encoder and memories
    train task-specific fusion heads and prediction heads
```

---

## 5. Stage A: Multitask Source Representation Learning

## 5.1 Stage A Inputs

Each training batch should provide:

```python
batch = {
    "x_clean": Tensor[B, L, C],
    "x_input": Tensor[B, L, C],
    "class_labels": LongTensor[B],
    "synthetic_anomaly_mask": BoolTensor[B, L] or BoolTensor[B, L, C],
    "is_synthetic": BoolTensor[B],
}
```

Meaning:

```text
x_clean:
    original clean window

x_input:
    actual model input
    can be clean or synthetic anomalous

class_labels:
    0 = normal
    1..11 = synthetic anomaly class

synthetic_anomaly_mask:
    marks injected anomaly positions

is_synthetic:
    true if sample is synthetic anomalous
```

Important:

```text
Do not infer point-level anomaly labels from window-level class labels.
Only injected positions are anomalous time-points.
```

---

## 5.2 Stage A Forward Pass

```text
x_input: [B, L, C]
        │
        ▼
shared encoder Eθ
        │
        ▼
Z: [B, L, d_model]
```

From `Z`, compute:

```text
reconstruction output
classification logits
contrastive representations
optional point-wise reconstruction-score loss
```

---

## 5.3 Reconstruction Loss

The reconstruction branch should learn normal structure.

For clean windows:

```text
target = x_clean = x_input
```

For synthetic anomalous windows:

```text
input  = x_input
target = x_clean
```

If `reconstruction_normal_only = true`, reconstruction loss is computed only on clean / non-injected positions.

Let:

```text
M[i,t] = 1 if time-point t is injected anomaly
M[i,t] = 0 otherwise
```

Clean mask:

```text
C_mask[i,t] = 1 - M[i,t]
```

Reconstruction loss:

```text
L_recon =
    sum_{i,t,c} C_mask[i,t] * (x_hat[i,t,c] - x_clean[i,t,c])²
    /
    (sum_{i,t,c} C_mask[i,t] + eps)
```

Important distinction:

```text
training reconstruction loss:
    compare x_hat with x_clean
    use clean-position mask

validation/test anomaly score:
    compare x_hat with x_input
    no clean-position mask
```

---

## 5.4 Classification Loss

Current implemented classification objective:

```text
window-level 12-class classification
```

```text
logits_window ∈ R[B, 12]
```

Loss:

```text
L_cls = CrossEntropy(logits_window, class_labels)
```

Current role:

```text
distinguish:
    class 0 = normal
    class 1..11 = synthetic anomaly classes
```

---

## 5.5 Updated Classification Interpretation

The intended semantic meaning of classification is:

```text
If a latent window contains latent points belonging to anomaly classes:
    classify the window as the anomaly class with the largest number of anomaly points.

If a latent window contains only normal latent points:
    classify the window as normal.
```

This suggests a future point-level classification design:

```text
input:
    latent tensor from discrete branch or fused branch

output:
    point-level logits

shape:
    logits_point ∈ R[B, L, 12]
```

Then:

```text
point_class[i,t] = argmax(logits_point[i,t])

window_class[i] =
    majority anomaly class among point_class[i, :]
    if at least one point is anomalous

    normal class
    if all points are normal
```

Status:

```text
Undecided
```

Do not implement this point-level classification loss in the main pipeline yet.

The current main offline training still uses window-level classification loss.

---

## 5.6 Point-wise Balanced Reconstruction-Score Loss

This loss is enabled in the point-score-supervised variant.

Point-wise reconstruction score:

```text
r[i,t] = mean_c (x_hat[i,t,c] - x_input[i,t,c])²
```

Shape:

```text
r ∈ R[B, L]
```

Point-wise anomaly label:

```text
a[i,t] = 1 if time-point t is injected anomaly
a[i,t] = 0 otherwise
```

If `synthetic_anomaly_mask` has shape `[B, L, C]`:

```python
a = synthetic_anomaly_mask.any(dim=-1)
```

If `synthetic_anomaly_mask` has shape `[B, L]`:

```python
a = synthetic_anomaly_mask
```

Normal-token set:

```text
N = {(i,t): a[i,t] = 0}
```

Anomaly-token set:

```text
A = {(i,t): a[i,t] = 1}
```

Compute normal-token score statistics from the current training batch only:

```text
mu_N  = mean_{(i,t) in N} r[i,t]
std_N = std_{(i,t) in N} r[i,t]
```

Detach:

```text
mu_N  = stopgrad(mu_N)
std_N = stopgrad(std_N)
```

Normalize:

```text
z_score[i,t] = (r[i,t] - mu_N) / (std_N + eps)
```

Per-token BCEWithLogits:

```text
loss_score[i,t] = BCEWithLogits(z_score[i,t], a[i,t])
```

Balanced reduction:

```text
L_score_point =
    0.5 * mean_{a=0} BCEWithLogits(z_score, 0)
  + 0.5 * mean_{a=1} BCEWithLogits(z_score, 1)
```

If either normal-token group or anomaly-token group is empty:

```text
skip L_score_point for this batch
log warning once per epoch
```

---

## 5.7 Contrastive Loss

The contrastive branch uses training data only.

```text
L_contrastive
```

Required rule:

```text
No validation/test window may enter contrastive positives.
No validation/test window may enter contrastive negatives.
No validation/test queue is allowed.
```

The current implementation may reuse the existing two-view contrastive objective.

---

## 5.8 Stage A Total Loss

Base two-stage run:

```text
L_StageA =
    λ_recon * L_recon
  + λ_cls   * L_cls
  + λ_con   * L_contrastive
```

Point-score-supervised two-stage run:

```text
L_cls_score = (L_cls + L_score_point) / 2
```

```text
L_StageA =
    λ_recon * L_recon
  + λ_cls   * L_cls_score
  + λ_con   * L_contrastive
```

Current default weights:

```text
λ_recon = 0.5
λ_cls   = 0.5
λ_con   = 0.3
```

No new `lambda_score` should be added in the main version.

Reason:

```text
avoid adding another hyperparameter
```

---

# Part III. End-of-Stage-A Memory Initialization

## 6. General Rule

Memory initialization is performed after Stage A.

Before memory initialization:

```python
model.eval()
torch.no_grad()
```

Allowed data source:

```text
training split only
```

Forbidden:

```text
validation windows
test windows
validation statistics
test statistics
future online stream
```

---

## 7. Continuous Prototype Bank Initialization

The continuous prototype bank represents clean / normal latent structure.

Collect latent tokens from:

```text
clean original train windows
clean positions in synthetic train windows
```

Forbidden:

```text
injected anomaly tokens
validation tokens
test tokens
```

Normalize latent tokens:

```text
z_norm = z / (||z||_2 + eps)
```

Run k-means:

```text
K_c = 32
```

Output:

```text
P_c ∈ R[32, d_model]
```

Store in:

```python
model.continuous_prototype_bank
```

Freeze after initialization.

---

## 8. Discrete Codebook Initialization

The discrete codebook represents class-stratified synthetic pattern structure.

For each class:

```text
class 0:
    use clean / normal latent tokens

class 1..11:
    use injected anomaly latent tokens from that synthetic class only
```

Run k-means per class:

```text
K_per_class = 5
```

Concatenate centroids:

```text
E_d = [C_0; C_1; ...; C_11]
```

Shape:

```text
E_d ∈ R[60, d_model]
```

Index layout:

```text
class 0:  codewords 0..4
class 1:  codewords 5..9
class 2:  codewords 10..14
...
class 11: codewords 55..59
```

Freeze after initialization.

---

## 9. Discrete Anomalous Cluster Radius

For each anomalous discrete codeword:

```text
e_k, where class_id(e_k) ∈ {1, ..., 11}
```

compute a cluster radius from eligible train tokens assigned to that codeword.

Top-1 assignment:

```text
k*(z) = argmin_k d(z, e_k)
```

For each anomalous codeword `e_k`, collect:

```text
A_k = {
    z_i from training split:
        class_id(e_k) is anomalous
        k*(z_i) = k
}
```

Radius:

```text
R_anom[k] = Quantile_q { d(z_i, e_k) : z_i ∈ A_k }
```

Recommended default:

```text
q = 0.99
```

Status:

```text
The exact radius quantile is a design choice.
Prefer q = 0.99 for conservative coverage.
```

Distance function should match the normalized latent geometry.

Recommended:

```text
cosine distance or angular distance
```

If latent vectors and codewords are L2-normalized:

```text
d_cos(z, e) = 1 - zᵀe
```

or:

```text
d_ang(z, e) = arccos(clip(zᵀe, -1, 1))
```

Store:

```python
discrete_codeword_class_id[k]
discrete_anom_radius[k]
```

Important:

```text
R_anom must be computed from train split only.
Do not use validation/test tokens.
Do not use online stream tokens.
```

---

# Part IV. Stage B: Frozen Encoder and Frozen Memories

## 10. Stage B Objective

Stage B trains task-specific heads after source memories are initialized.

Frozen components:

```text
shared encoder
continuous prototype bank
discrete codebook
```

Trainable components:

```text
reconstruction fusion head
classification fusion head
reconstruction prediction head
classification prediction head
```

---

## 11. Stage B Forward Pass

```text
X: [B, L, C]
    │
    ▼
frozen encoder
    │
    ▼
Z: [B, L, d_model]
    │
    ├── query continuous prototype bank
    │       ▼
    │    Z_cont: [B, L, d_model]
    │
    └── query discrete codebook
            ▼
         Z_disc: [B, L, d_model]
```

Concatenate:

```text
Z_cat = concat(Z_cont, Z_disc, dim=-1)
Z_cat ∈ R[B, L, 2 * d_model]
```

Reconstruction path:

```text
Z_cat
    │
    ▼
reconstruction fusion head
    │
    ▼
Z_recon: [B, L, d_model]
    │
    ▼
reconstruction prediction head
    │
    ▼
x_hat: [B, L, C]
```

Classification path:

```text
Z_cat
    │
    ▼
classification fusion head
    │
    ▼
classification prediction head
    │
    ▼
logits_window: [B, 12]
```

---

## 12. Stage B Losses

Main Stage B losses:

```text
L_recon
L_cls
```

Do not use point-wise score loss in Stage B by default.

Reason:

```text
point-wise score loss mainly shapes encoder/reconstruction score geometry in Stage A.
Stage B should only train heads over frozen source latent/memory representations.
```

Status:

```text
Stage B point-wise score loss is possible future ablation only.
```

---

# Part V. Validation and Threshold Calibration

## 13. Offline Validation and Test Windowing

The following sets must use non-overlapping windows:

```text
clean validation set
synthetic validation set
offline test set
```

Reason:

```text
avoid overlap-induced score inflation
avoid false recurrence caused by overlapping windows
keep validation/test evaluation deterministic
```

---

## 14. Offline Point-Level Threshold

Point-level anomaly score:

```text
s_point(t) = mean_c (x[t,c] - x_hat[t,c])²
```

Offline point threshold:

```text
T_point_nonoverlap =
    Q_0.99 { s_point(t) on clean validation non-overlapping windows }
```

Use this threshold for:

```text
synthetic validation point-level decision
offline test point-level decision
```

Prediction:

```text
y_pred(t) = 1 if s_point(t) > T_point_nonoverlap
```

---

## 15. Online EWMA Threshold Calibration

Online TTA uses sliding windows, so the clean validation set should also be simulated with sliding windows for online threshold calibration.

Procedure:

```text
1. Take clean validation sequence.
2. Generate sliding windows.
3. Forward each sliding window.
4. Compute point reconstruction scores.
5. Aggregate repeated absolute time-point scores by EWMA.
6. Take q99 over final EWMA point scores.
```

EWMA:

```text
S_new(t) = 0.9 * s_current(t) + 0.1 * S_old(t)
```

Use explicit config names:

```yaml
online_score_current_weight: 0.9
online_score_previous_weight: 0.1
```

Online threshold:

```text
T_point_EWMA =
    Q_0.99 { S_clean_val(t) after sliding-window + EWMA simulation }
```

Use this threshold for:

```text
online point-level anomaly decision
```

---

## 16. Window-Level Thresholds

Window input reconstruction score:

```text
s_input_window(W) =
    mean_{t,c} (x[t,c] - x_hat[t,c])²
```

Threshold:

```text
B_window =
    Q_p { s_input_window(W) on clean validation }
```

Recommended:

```text
p = 0.99
```

Latent window score:

```text
s_latent_window(W) = MSE(Z_W, Z_W_cont)
```

Canonical notation: the historical runtime quantity `s_latent_window(W)` maps to \(S_t^{(\mathrm{latent})}\). It is a window-level triage score and is not automatically identical to the proposed point-level score \(\overline{\ell}^{(c)}_{t,i}\).

where:

```text
Z_W:
    latent tensor before continuous prototype retrieval

Z_W_cont:
    latent tensor after continuous prototype retrieval
```

Threshold band:

```text
A_low  = Q_p1 { s_latent_window(W) on clean validation }
A_high = Q_p2 { s_latent_window(W) on clean validation }
```

Recommended candidate:

```text
p1 = 0.95
p2 = 0.99
```

Status:

```text
Exact p, p1, p2 remain configurable.
```

---

# Part VI. Online Test-Time Adaptation Phase

## 17. Online TTA Objective

Online TTA aims to adapt a very small online branch to recurrent pseudo-new-normality patterns.

It must not update:

```text
source encoder
source continuous prototype bank
source discrete codebook
source memory banks
source model parameters
```

Allowed online update targets:

```text
light-weight MLP projector
small part of reconstruction fusion head
small part of reconstruction prediction head
```

Possible but undecided:

```text
classification fusion head
classification prediction head
online contrastive projector update
```

---

## 18. Online Model Components

## 18.1 Light-weight MLP Projector

The online model contains:

```text
g_ψ: R[d_model] → R[d_model]
```

Applied token-wise:

```text
Z_proj = g_ψ(Z_online)
```

Shape:

```text
Z_online ∈ R[B, L, d_model]
Z_proj   ∈ R[B, L, d_model]
```

Initialization:

```text
g_ψ ≈ identity function
```

Purpose:

```text
map online latent distribution back toward source latent distribution
on the unit-hypersphere manifold
```

Assumption:

```text
There exists at least one useful mapping from online latent distribution
to source latent distribution.
```

Status:

```text
This is an assumption, not a guaranteed theorem.
```

---

## 19. Online Point-Level Score Aggregation

At online step `τ`, receive:

```text
W_τ = x[τ-L+1 : τ]
```

Forward pass gives local point scores:

```text
s_point^{(τ)}(t)
```

If absolute time-point `t` is seen for the first time:

```text
S(t) = s_point^{(τ)}(t)
```

If `t` already has an old score:

```text
S_new(t) = 0.9 * s_point^{(τ)}(t) + 0.1 * S_old(t)
```

A point is predicted anomalous if:

```text
S(t) > T_point_EWMA
```

Important:

```text
This point-level prediction is the final anomaly decision.
The verification buffer only controls whether online adaptation is allowed.
```

---

## 20. Online Window-Level Triage

For each sliding window `W`, compute:

```text
s_input_window(W)
s_latent_window(W)
```

Case 1:

```text
s_input_window(W) <= B_window
```

Action:

```text
do not add to verification buffer
continue point-level detection only
```

Case 2:

```text
s_input_window(W) > B_window
s_latent_window(W) <= A_low
```

Interpretation:

```text
hard old-normality candidate
```

Action:

```text
log only by default
do not add to verification buffer
```

Status:

```text
Immediate adaptation for this case is undecided.
```

Case 3:

```text
s_input_window(W) > B_window
A_low < s_latent_window(W) <= A_high
```

Interpretation:

```text
gray-zone latent deviation
```

Action:

```text
try to add W to verification buffer
only if W does not overlap with current buffer windows
```

Case 4:

```text
s_input_window(W) > B_window
s_latent_window(W) > A_high
```

Interpretation:

```text
strong anomaly candidate
```

Action:

```text
do not add to verification buffer
do not use for reconstruction adaptation
log only
```

---

## 21. Verification Buffer Admission

A window is admitted into the verification buffer if and only if:

```text
s_input_window(W) > B_window
```

and:

```text
A_low < s_latent_window(W) <= A_high
```

and:

```text
W is non-overlapping with all windows currently in the buffer
```

Two windows are non-overlapping if:

```text
end_i < start_j
or
end_j < start_i
```

Adjacent windows are allowed.

Example allowed:

```text
[0, 19], [20, 39]
```

Example forbidden:

```text
[0, 19], [1, 20]
```

Current buffer size:

```text
N_buf = 8 or 16
```

Status:

```text
Earlier spec used 16.
Recent discussion used 8.
Final value should be explicitly set in config.
```

Recommended config:

```yaml
verification_buffer_size: 8
verification_buffer_non_overlap: true
```

---

# Part VII. Verification Buffer Processing

## 22. Buffer Trigger

When the buffer reaches `N_buf` windows:

```text
verification is triggered
```

If:

```text
N_buf = 8
L = 20
```

then the buffer contains:

```text
8 × 20 = 160 latent time-points
```

---

## 23. Step 1: Collect Latent Points

For each buffered window:

```text
W_i
```

obtain:

```text
Z_i ∈ R[L, d_model]
```

Stack:

```text
Z_buffer ∈ R[N_buf, L, d_model]
```

Each latent point is:

```text
z[i,t] ∈ R[d_model]
```

---

## 24. Step 2: Remove Latent Points Inside Anomalous Discrete Clusters

For every latent point `z[i,t]`, find the nearest discrete codeword:

```text
k* = argmin_k d(z[i,t], e_k)
```

Check whether:

```text
class_id(e_k*) ∈ {1, ..., 11}
```

and:

```text
d(z[i,t], e_k*) <= R_anom[k*]
```

If both are true:

```text
discard z[i,t]
```

Interpretation:

```text
this latent point lies inside a known abstract anomalous pattern cluster
```

If false:

```text
keep z[i,t] for continuous-signature verification
```

Important:

```text
Do not use T_point as a token filter in buffer verification.
The buffer verification first removes known anomaly-like latent points
using discrete anomalous cluster membership.
```

---

## 25. Step 3: Compute Continuous Top-3 Signature

For each remaining latent point:

```text
z[i,t]
```

compute top-3 nearest continuous prototypes:

```text
top3_cont(z[i,t]) = (p_a, p_b, p_c)
```

Define the signature:

```text
σ(z[i,t]) = (id(p_a), id(p_b), id(p_c))
```

Distance should match continuous prototype geometry.

Recommended:

```text
cosine distance or angular distance
```

If vectors are normalized:

```text
d_cos(z, p) = 1 - zᵀp
```

The signature is ordered by distance unless explicitly configured otherwise.

Recommended:

```yaml
continuous_signature_ordered: true
continuous_signature_topk: 3
```

Note:

```text
continuous_signature_topk = 3 is a design choice.
It is different from discrete_query_topk.
```

---

## 26. Step 4: Detect Recurrent Pseudo-New-Normality Signatures

Group latent points by signature:

```python
signature_to_points[signature].append((window_id, local_t, z))
```

Also collect window ids:

```python
signature_to_windows[signature].add(window_id)
```

A signature is recurrent if:

```text
len(signature_to_windows[signature]) > 1
```

Meaning:

```text
the same abstract latent pattern appears in more than one non-overlapping window
```

For recurrent signatures:

```text
all latent points carrying that signature are pseudo-new-normality latent points
```

Definition:

```text
Z_pnn = {
    z[i,t]:
        σ(z[i,t]) is recurrent
}
```

Terminology:

```text
pseudo-new-normality latent point
```

Do not call these:

```text
confirmed normal points
confirmed new normality
```

---

## 27. Step 5: Build Windows To Be Adapted

For each recurrent signature:

```text
σ
```

collect all windows that contain at least one latent point with that signature.

```python
windows_to_be_adapted = set()

for sig in recurrent_signatures:
    windows_to_be_adapted.update(signature_to_windows[sig])
```

Also build a token mask:

```text
M_pnn ∈ {0,1}^{B_adapt, L}
```

where:

```text
M_pnn[i,t] = 1
    if z[i,t] has a recurrent pseudo-new-normality signature

M_pnn[i,t] = 0
    otherwise
```

Verification output:

```python
verification_output = {
    "recurrent_signatures": recurrent_signatures,
    "pseudo_new_normality_points": Z_pnn,
    "windows_to_be_adapted": windows_to_be_adapted,
    "token_mask_pnn": M_pnn,
}
```

If there is no recurrent signature:

```text
do not adapt
clear or refresh buffer according to implementation policy
```

---

# Part VIII. Online Adaptation

## 28. Online Adaptation Input

Collect windows to be adapted:

```text
X_adapt ∈ R[B_adapt, L, C]
```

The corresponding mask:

```text
M_pnn ∈ {0,1}^{B_adapt, L}
```

Only positions with:

```text
M_pnn[i,t] = 1
```

are used in masked reconstruction adaptation.

---

## 29. Online Adaptation Forward Pass

Forward source/online encoder:

```text
X_adapt
    │
    ▼
Z_online ∈ R[B_adapt, L, d_model]
```

Apply light-weight MLP projector:

```text
Z_proj = g_ψ(Z_online)
```

Shape:

```text
Z_proj ∈ R[B_adapt, L, d_model]
```

Query frozen source memories:

```text
Z_proj
    ├── continuous prototype bank
    │       ▼
    │    Z_cont ∈ R[B_adapt, L, d_model]
    │
    └── discrete codebook
            ▼
         Z_disc ∈ R[B_adapt, L, d_model]
```

---

## 30. Reconstruction Adaptation Path

Concatenate:

```text
Z_recon_cat = concat(Z_cont, Z_disc, dim=-1)
```

Shape:

```text
Z_recon_cat ∈ R[B_adapt, L, 2 * d_model]
```

Forward:

```text
Z_recon_cat
    │
    ▼
reconstruction fusion head
    │
    ▼
Z_recon ∈ R[B_adapt, L, d_model]
    │
    ▼
reconstruction prediction head
    │
    ▼
X_hat_adapt ∈ R[B_adapt, L, C]
```

Masked online reconstruction loss:

```text
L_online_recon =
    sum_{i,t,c} M_pnn[i,t] * (X_hat_adapt[i,t,c] - X_adapt[i,t,c])²
    /
    (sum_{i,t,c} M_pnn[i,t] + eps)
```

Equivalent implementation:

```python
m = token_mask_pnn.float()[:, :, None]  # [B_adapt, L, 1]

loss = ((x_hat_adapt - x_adapt) ** 2 * m).sum()
loss = loss / (m.sum() * C + eps)
```

Important:

```text
Positions not verified as pseudo-new-normality must be masked out completely.
```

---

## 31. Online Update Targets

Main online update targets:

```text
light-weight MLP projector
small subset of reconstruction fusion head
small subset of reconstruction prediction head
```

Recommended conservative default:

```text
update full light-weight MLP projector
update only final affine layer or small adapter inside reconstruction fusion head
update only final affine layer or small adapter inside reconstruction prediction head
```

Frozen:

```text
source encoder
source continuous prototype bank
source discrete codebook
source memories
```

Do not update:

```text
source memory
source codebook
source encoder
full source model
```

---

## 32. Classification Online Adaptation

Status:

```text
Undecided
```

Current rule:

```text
Do not update classification path in the main online TTA method.
```

Reason:

```text
The loss for point-level classification and majority-window classification is not finalized.
Updating classification head online may amplify pseudo-label errors.
```

Future design may include:

```text
point-level logits: [B, L, 12]
majority anomaly class rule
normal if no point is classified as anomaly
```

But this is not part of the current main implementation.

---

## 33. Contrastive Online Projector Update

Status:

```text
Undecided
```

Potential goal:

```text
align projected online latent points with source latent distribution
```

Potential anchors:

```text
projected online pseudo-new-normality latent points
```

Potential positives:

```text
source latent of same window
nearest continuous prototype
centroid of recurrent signature group
```

Potential negatives:

```text
other recurrent signatures
anomalous discrete codewords
far continuous prototypes
```

Open decisions:

```text
anchor definition
positive definition
negative definition
temperature
whether to reuse offline contrastive temperature
whether to use only M_pnn positions
whether to combine with L_online_recon
```

Current main method:

```text
Do not include online contrastive loss until the design is finalized.
```

---

## 34. Online Adaptation Schedule

Recommended conservative update:

```text
one or very few gradient steps per full verification buffer
```

Status:

```text
Exact number of online steps is undecided.
```

Suggested config:

```yaml
online_adaptation_steps: 1
online_adaptation_lr: null
online_adaptation_weight_decay: null
```

The learning rate must be selected without using test labels.

---

## 35. Buffer Reset Policy

After adaptation:

```text
clear verification buffer
```

Recommended first implementation:

```python
verification_buffer.clear()
```

Alternative future policies:

```text
remove only adapted windows
keep unresolved rare windows
use rolling buffer
```

Status:

```text
Future ablation only.
```

---

# Part IX. Full Online TTA Pseudocode

```python
for tau in online_stream:

    # ------------------------------------------------------------
    # 1. Build sliding window
    # ------------------------------------------------------------
    W = get_sliding_window_ending_at(tau)

    # ------------------------------------------------------------
    # 2. Forward pass
    # ------------------------------------------------------------
    outputs = model.forward(W)

    point_scores = outputs["point_reconstruction_mse"]      # [L]
    input_w_score = outputs["input_window_mse"]             # scalar
    latent_w_score = outputs["latent_window_mse"]           # scalar

    # ------------------------------------------------------------
    # 3. EWMA point-level score update
    # ------------------------------------------------------------
    for abs_t, local_t in W.absolute_to_local.items():

        cur = point_scores[local_t]

        if abs_t not in point_score_state:
            point_score_state[abs_t] = cur
        else:
            old = point_score_state[abs_t]
            point_score_state[abs_t] = (
                0.9 * cur + 0.1 * old
            )

    # ------------------------------------------------------------
    # 4. Finalize expired point-level decisions
    # ------------------------------------------------------------
    finalized_points = get_points_that_can_no_longer_overlap(tau, L)

    for abs_t in finalized_points:
        y_pred[abs_t] = point_score_state[abs_t] > T_point_EWMA

    # ------------------------------------------------------------
    # 5. Window-level triage
    # ------------------------------------------------------------
    if input_w_score <= B_window:
        continue

    if latent_w_score <= A_low:
        log_hard_old_normality_candidate(W)
        continue

    if latent_w_score > A_high:
        log_strong_anomaly_candidate(W)
        continue

    # ------------------------------------------------------------
    # 6. Gray-zone candidate
    # ------------------------------------------------------------
    if A_low < latent_w_score <= A_high:
        if not overlaps_any(W, verification_buffer):
            verification_buffer.add(W)

    # ------------------------------------------------------------
    # 7. Trigger verification
    # ------------------------------------------------------------
    if len(verification_buffer) >= N_buf:

        verification_output = verify_buffer(
            buffer=verification_buffer,
            discrete_codebook=source_discrete_codebook,
            discrete_codeword_class_id=discrete_codeword_class_id,
            discrete_anom_radius=discrete_anom_radius,
            continuous_prototypes=source_continuous_prototype_bank,
            continuous_signature_topk=3,
        )

        if len(verification_output["windows_to_be_adapted"]) > 0:
            online_adapt(
                windows=verification_output["windows_to_be_adapted"],
                token_mask_pnn=verification_output["token_mask_pnn"],
            )

        verification_buffer.clear()
```

---

# Part X. Verification Buffer Pseudocode

```python
def verify_buffer(
    buffer,
    discrete_codebook,
    discrete_codeword_class_id,
    discrete_anom_radius,
    continuous_prototypes,
    continuous_signature_topk=3,
):
    signature_to_points = defaultdict(list)
    signature_to_windows = defaultdict(set)

    for window_id, W in enumerate(buffer):

        Z = encode_window(W)  # [L, d_model]

        for local_t, z in enumerate(Z):

            # ----------------------------------------------------
            # 1. Remove points inside anomalous discrete clusters
            # ----------------------------------------------------
            k_star = nearest_codeword(z, discrete_codebook)

            is_anom_codeword = (
                discrete_codeword_class_id[k_star] != 0
            )

            inside_anom_radius = (
                distance(z, discrete_codebook[k_star])
                <= discrete_anom_radius[k_star]
            )

            if is_anom_codeword and inside_anom_radius:
                continue

            # ----------------------------------------------------
            # 2. Compute continuous top-3 signature
            # ----------------------------------------------------
            signature = topk_continuous_prototype_ids(
                z,
                continuous_prototypes,
                k=continuous_signature_topk,
            )

            signature_to_points[signature].append(
                (window_id, local_t, z)
            )

            signature_to_windows[signature].add(window_id)

    recurrent_signatures = set()
    pseudo_new_normality_points = []
    windows_to_be_adapted = set()

    for sig, window_ids in signature_to_windows.items():

        if len(window_ids) > 1:
            recurrent_signatures.add(sig)
            windows_to_be_adapted.update(window_ids)
            pseudo_new_normality_points.extend(
                signature_to_points[sig]
            )

    token_mask_pnn = build_token_mask(
        buffer=buffer,
        pseudo_new_normality_points=pseudo_new_normality_points,
    )

    return {
        "recurrent_signatures": recurrent_signatures,
        "pseudo_new_normality_points": pseudo_new_normality_points,
        "windows_to_be_adapted": windows_to_be_adapted,
        "token_mask_pnn": token_mask_pnn,
    }
```

---

# Part XI. Configuration Draft

```yaml
# ============================================================
# THESIS two-phase pipeline
# ============================================================

model_name: thesis_multitask

# ------------------------------------------------------------
# Windowing
# ------------------------------------------------------------
window_size: 20
offline_eval_window_mode: non_overlapping
validation_window_mode: non_overlapping
test_window_mode: non_overlapping
online_window_mode: sliding

# ------------------------------------------------------------
# Model dimensions
# ------------------------------------------------------------
input_dim: 38
hidden_dim: 32
num_classes: 12

# ------------------------------------------------------------
# Stage A
# ------------------------------------------------------------
stage_a_enabled: true
stage_a_epochs: 80

lambda_recon: 0.5
lambda_cls: 0.5
lambda_contrastive: 0.3

enable_classification_path: true
enable_two_view_contrastive: true
contrastive_temperature: 0.1

enable_score_loss: true
score_loss_type: pointwise_balanced_bce_logits
score_loss_granularity: point
score_loss_target: synthetic_anomaly_mask
score_loss_reduction: pointwise_binary_balanced_mean
score_loss_normalization: train_batch_normal_tokens_detached_mean_std

reconstruction_normal_only: true

# ------------------------------------------------------------
# Memory initialization
# ------------------------------------------------------------
memory_initialization_after_stage_a: true
memory_initialization_source: train_only
memory_initialization_with_synthetic_windows: true
freeze_memories_after_initialization: true

continuous_enabled: true
continuous_num_prototypes: 32

discrete_enabled: true
discrete_codebook_size: 60
discrete_num_classes: 12
discrete_codewords_per_class: 5
discrete_query_mode: cosine_topk
discrete_topk: 3
discrete_query_temperature: 0.1

discrete_anom_radius_enabled: true
discrete_anom_radius_source: train_only
discrete_anom_radius_quantile: 0.99
discrete_anom_radius_distance: cosine

# ------------------------------------------------------------
# Stage B
# ------------------------------------------------------------
stage_b_enabled: true
stage_b_epochs: 20

stage_b_freeze_encoder: true
stage_b_freeze_continuous_memory: true
stage_b_freeze_discrete_codebook: true

stage_b_trainable_modules:
  - reconstruction_fusion_head
  - classification_fusion_head
  - reconstruction_prediction_head
  - classification_prediction_head

# ------------------------------------------------------------
# Threshold calibration
# ------------------------------------------------------------
threshold_source: clean_validation

offline_point_threshold_enabled: true
offline_point_threshold_quantile: 0.99
offline_point_threshold_window_mode: non_overlapping

online_point_threshold_enabled: true
online_point_threshold_quantile: 0.99
online_point_threshold_calibration_mode: sliding_window_ewma

window_input_threshold_quantile: 0.99
window_latent_low_quantile: 0.95
window_latent_high_quantile: 0.99

# ------------------------------------------------------------
# Online score aggregation
# ------------------------------------------------------------
online_point_score_aggregation: ewma
online_score_current_weight: 0.9
online_score_previous_weight: 0.1

# ------------------------------------------------------------
# Online verification buffer
# ------------------------------------------------------------
online_tta_enabled: true
verification_buffer_enabled: true
verification_buffer_size: 8
verification_buffer_non_overlap: true
buffer_admission_rule: input_window_high_and_latent_gray_zone

# ------------------------------------------------------------
# Buffer verification
# ------------------------------------------------------------
buffer_verification_level: latent_time_point

discrete_anomaly_filter_enabled: true
discrete_anomaly_filter_rule: nearest_anom_codeword_inside_radius

continuous_signature_enabled: true
continuous_signature_mode: topk_continuous_prototype_ids
continuous_signature_topk: 3
continuous_signature_ordered: true

recurrent_signature_rule: appears_in_more_than_one_non_overlapping_window

verification_outputs:
  - recurrent_signatures
  - pseudo_new_normality_points
  - windows_to_be_adapted
  - token_mask_pnn

# ------------------------------------------------------------
# Online adaptation
# ------------------------------------------------------------
online_update_enabled: true
source_model_frozen: true
source_memory_frozen: true

online_projector_enabled: true
online_projector_init: approximate_identity

online_update_targets:
  - online_mlp_projector
  - reconstruction_fusion_head_small_subset
  - reconstruction_prediction_head_small_subset

online_reconstruction_adaptation_enabled: true
online_reconstruction_loss_mask: pseudo_new_normality_tokens_only

online_classification_adaptation_enabled: false
online_contrastive_adaptation_enabled: false

online_adaptation_steps: 1
online_adaptation_lr: null
online_adaptation_weight_decay: null

buffer_reset_after_adaptation: true
```

---

# Part XII. Logging Requirements

## 36. Offline Logs

Stage A:

```text
train/loss_total
train/loss_recon
train/loss_cls
train/loss_contrastive
train/loss_score_point
train/loss_cls_score
train/score_loss_skipped_batches
train/point_score_normal_mean
train/point_score_anomaly_mean
train/point_score_gap_mean
```

Memory initialization:

```text
memory/continuous_pool_size
memory/discrete_pool_size_class_0
memory/discrete_pool_size_class_1
...
memory/discrete_pool_size_class_11
memory/continuous_shape
memory/discrete_shape
memory/discrete_anom_radius_mean
memory/discrete_anom_radius_min
memory/discrete_anom_radius_max
```

Stage B:

```text
stage_b/loss_total
stage_b/loss_recon
stage_b/loss_cls
stage_b/encoder_grad_norm
stage_b/continuous_memory_grad_norm
stage_b/discrete_memory_grad_norm
```

Expected:

```text
encoder_grad_norm = 0
continuous_memory_grad_norm = 0
discrete_memory_grad_norm = 0
```

## 37. Online Logs

Online point score:

```text
online/point_score_current
online/point_score_ewma
online/num_finalized_points
online/num_predicted_anomaly_points
```

Window triage:

```text
online/input_window_score
online/latent_window_score
online/num_hard_old_normality_windows
online/num_gray_zone_windows
online/num_strong_anomaly_windows
online/num_buffer_admitted_windows
online/num_buffer_rejected_overlap_windows
```

Buffer verification:

```text
online/buffer_size
online/num_latent_points_total
online/num_points_removed_by_discrete_anom_filter
online/num_points_remaining_for_signature
online/num_unique_signatures
online/num_recurrent_signatures
online/num_pseudo_new_normality_points
online/num_windows_to_be_adapted
```

Online adaptation:

```text
online/loss_recon_masked
online/adaptation_steps
online/projector_grad_norm
online/recon_fusion_grad_norm
online/recon_head_grad_norm
online/source_encoder_grad_norm
online/source_memory_grad_norm
```

Expected:

```text
source_encoder_grad_norm = 0
source_memory_grad_norm = 0
```

---

# Part XIII. Unit Tests

## 38. Offline Unit Tests

Required tests:

```text
[ ] point-wise score shape is [B, L]
[ ] point-wise anomaly target shape is [B, L]
[ ] synthetic anomalous window does not mark all positions as anomalous
[ ] L_score_point uses balanced normal/anomaly reduction
[ ] L_score_point uses train-batch normal-token mean/std only
[ ] L_score_point detaches mean/std
[ ] L_score_point skips safely if one group is empty
[ ] continuous prototype bank shape is [32, d_model]
[ ] discrete codebook shape is [60, d_model]
[ ] discrete codebook has 5 codewords per class
[ ] anomalous discrete radii are computed from train split only
[ ] Stage B freezes encoder and memory banks
```

## 39. Online Unit Tests

Required tests:

```text
[ ] EWMA uses 0.9 * current + 0.1 * old
[ ] online TTA threshold uses sliding-window + EWMA clean validation simulation
[ ] buffer admits only gray-zone windows
[ ] buffer rejects overlapping windows
[ ] anomalous discrete cluster filter removes correct latent points
[ ] continuous signature uses top-3 continuous prototypes, not discrete top-k
[ ] recurrent signature is counted across non-overlapping windows
[ ] pseudo-new-normality token mask has shape [B_adapt, L]
[ ] masked reconstruction loss ignores non-pseudo-new-normality positions
[ ] online adaptation does not update source encoder
[ ] online adaptation does not update source memories
```

---

# Part XIV. Data Leakage Safety Checklist

The implementation is invalid if any of the following occurs:

```text
[ ] validation/test windows enter Stage A training loss
[ ] validation/test windows enter k-means memory initialization
[ ] validation/test windows enter anomalous radius fitting
[ ] validation/test labels are used during online TTA
[ ] future online windows are used in causal online adaptation
[ ] validation/test statistics are backpropagated
[ ] source memories are updated during online TTA
[ ] source encoder is updated during online TTA
[ ] test labels are used to tune thresholds
[ ] test labels are used to tune online adaptation hyperparameters
```

Allowed:

```text
[ ] clean validation used for threshold calibration
[ ] validation used for no_grad checkpoint evaluation
[ ] test used for final evaluation only
[ ] online test stream used for unsupervised test-time adaptation
```

---

# Part XV. Acceptance Criteria

## 40. Functional Criteria

```text
[ ] Stage A runs successfully.
[ ] End-of-Stage-A memory initialization runs successfully.
[ ] Continuous prototype bank is initialized from train-normal tokens only.
[ ] Discrete codebook is initialized class-wise from train tokens only.
[ ] Anomalous discrete cluster radii are computed from train tokens only.
[ ] Stage B runs with frozen encoder and frozen memories.
[ ] Clean validation thresholds are calibrated.
[ ] Online EWMA threshold is calibrated by simulating sliding windows on clean validation.
[ ] Online stream produces point-level predictions.
[ ] Verification buffer stores only non-overlapping gray-zone windows.
[ ] Buffer verification returns recurrent signatures, pseudo-new-normality points, windows_to_be_adapted, and token_mask_pnn.
[ ] Online adaptation runs only on masked pseudo-new-normality positions.
```

## 41. Shape Criteria

```text
[ ] X input shape is [B, 20, C].
[ ] Z latent shape is [B, 20, d_model].
[ ] point score shape is [B, 20].
[ ] continuous prototype bank shape is [32, d_model].
[ ] discrete codebook shape is [60, d_model].
[ ] Z_cont shape is [B, 20, d_model].
[ ] Z_disc shape is [B, 20, d_model].
[ ] concatenated fusion input shape is [B, 20, 2 * d_model].
[ ] online token_mask_pnn shape is [B_adapt, 20].
[ ] reconstructed output shape is [B_adapt, 20, C].
```

## 42. Main Research Claim Boundaries

The method may claim:

```text
THESIS performs conservative online adaptation using recurrent pseudo-new-normality latent points.
```

The method may claim:

```text
Discrete codebook filters latent points that resemble known synthetic anomalous pattern clusters.
```

The method may claim:

```text
Continuous prototype signatures identify recurrent latent patterns across non-overlapping windows.
```

The method must not claim:

```text
confirmed discovery of true new normality
```

The safer term is:

```text
pseudo-new-normality latent point
```

or:

```text
weak emerging new-normality candidate
```

---

# Part XVI. Current Undecided Items

The following items are not finalized:

```text
[ ] final verification_buffer_size: 8 or 16
[ ] exact quantile for B_window
[ ] exact quantiles for A_low and A_high
[ ] exact quantile for discrete anomalous cluster radius
[ ] whether continuous top-3 signature should be ordered or unordered
[ ] whether hard old-normality candidate should trigger immediate adaptation
[ ] online adaptation learning rate
[ ] number of online adaptation steps
[ ] whether to update classification path online
[ ] point-level classification loss design
[ ] contrastive projector update design
[ ] buffer reset policy beyond simple clear()
```

Recommended first implementation:

```text
verification_buffer_size = 8
B_window = q99 clean validation
A_low = q95 clean validation
A_high = q99 clean validation
discrete_anom_radius = q99 train-assigned anomaly tokens
continuous signature = ordered top-3 continuous prototype ids
hard old-normality = log only
classification online update = disabled
contrastive online update = disabled
buffer reset = clear after verification/adaptation
```

---

# Final Summary

The current THESIS pipeline has two cleanly separated phases.

Offline phase:

```text
train source representation
initialize source continuous and discrete memories from train split only
freeze source model and memories after training
calibrate validation thresholds without gradient
```

Online phase:

```text
detect anomaly at point level using EWMA score
admit only gray-zone non-overlapping windows into verification buffer
remove latent points inside known anomalous discrete clusters
detect recurrent top-3 continuous-prototype signatures
treat recurrent latent points as pseudo-new-normality
adapt only a small online reconstruction path on masked pseudo-new-normality positions
```

The strongest safety rule remains:

```text
The buffer controls adaptation.
The point-level threshold controls anomaly decision.
The source model and source memories stay frozen.
```
