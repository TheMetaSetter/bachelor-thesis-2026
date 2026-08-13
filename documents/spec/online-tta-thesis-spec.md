# Development Specification

> **Notation authority:** Khi đối chiếu anomaly score mức điểm, tài liệu lịch sử này dùng mapping trong [Thiết kế anomaly score mức điểm và bộ ký hiệu chuẩn](anomaly-score-designs-and-notation.md). Tên runtime và ngữ nghĩa lịch sử trong thân tài liệu được giữ nguyên.


# Point-Level THESIS with Window-Level Online Verification Buffer

## 0. Status

This document specifies the latest design decisions for the THESIS online test-time adaptation mechanism.

Current design direction:

```text
Final anomaly decision: point-level
Online candidate selection: window-level
Verification: latent time-point level
Buffer admission: non-overlapping windows only
Calibration source: clean validation set
```

The design intentionally minimizes additional hyperparameters. Any unresolved decision is explicitly marked as **Undecided**.

---

## 1. Core Objective

THESIS aims to perform time-series anomaly detection with point-level anomaly decisions.

The model should learn to reconstruct normal time-points well and reconstruct anomalous time-points poorly. The raw input-space point score is

\[
\overline{s}_{t,i}
=
\frac{1}{M}\sum_{m=1}^{M}
\frac{1}{C}
\left\|\mathbf{x}_{t,i}-\widehat{\mathbf{x}}^{(m)}_{t,i}\right\|_2^2.
\]

When the run uses calibrated input-space scoring, the score sent to the online timeline is \(s^{(\mathrm{cal})}_{t,i}\), as defined by the notation authority. A point is predicted anomalous if

\[
\widehat{a}_n=\mathbb{I}\left(\widetilde{s}_n>T_{\mathrm{point}}\right).
\]

The threshold \(T_{\mathrm{point}}\) is the main operational anomaly threshold. Online test-time adaptation is not triggered directly by every anomalous point. Instead, THESIS uses window-level signals to decide whether a window should be placed into a verification buffer.

---

## 2. Design Principles

The online TTA design follows five principles.

First, final anomaly decisions are made at the point level, because the evaluation metrics are ultimately computed on point-wise timeline scores.

Second, window-level scores are used only for online buffer admission and adaptation triage, not as final anomaly labels.

Third, all thresholds are calibrated from the clean validation set. Synthetic validation may be used for diagnostic separation checks, but it is not the default source for operational thresholds.

Fourth, verification buffer windows must be non-overlapping. They may be adjacent in time or far apart, but their time intervals must not overlap.

Fifth, source model and source memories must remain frozen. Online adaptation may update only the online branch or a small subset of online parameters.

---

## 3. Data Protocol

### 3.1 Training Split

The training split is used for offline pre-training only.

Allowed uses:

```text
train encoder
train reconstruction head
train classification head
train contrastive objective
initialize continuous prototype bank
initialize discrete codebook
generate synthetic anomaly windows
compute training losses
```

Forbidden uses:

```text
use validation/test labels for training
initialize memories from validation/test
backpropagate through validation/test statistics
```

### 3.2 Validation Set

The validation set is cut into non-overlapping windows.

The clean validation set is used to calibrate:

[
T_{\text{point}},
]

[
B_{\text{window}},
]

[
A_{\text{low}}, A_{\text{high}}.
]

All these thresholds are constants after calibration.

### 3.3 Offline Test Set

The offline test set is also cut into non-overlapping windows.

Point-level anomaly scores are produced and mapped back to the original timeline for evaluation.

### 3.4 Online Test-Time Adaptation Stream

Online TTA uses sliding windows.

At online time (\tau), the model receives:

[
W_{\tau} = x_{\tau-L+1:\tau}.
]

The online sliding-window mechanism may generate multiple scores for the same absolute time-point. The current working design allows score refinement using EWMA.

---

## 4. Thresholds

THESIS uses three threshold objects.

### 4.1 Point-Level Anomaly Threshold

\[
T_{\mathrm{point}}
=
Q_p\left(\widetilde{s}^{\mathrm{clean-val}}_n\right).
\]

The clean-validation timeline must use the same selected score design and EWMA path as online inference. The score may be \(\overline{s}_{t,i}\), \(s^{(\mathrm{cal})}_{t,i}\), \(\overline{\ell}^{(c)}_{t,i}\), or \(s^{(\mathrm{latent})}_{t,i}\), but one run MUST use one explicitly named score space.

A point is predicted anomalous only when \(\widetilde{s}_n>T_{\mathrm{point}}\). The exact quantile \(p\) remains a protocol choice; candidates are \(0.95\) and \(0.99\), with the conservative historical default \(0.99\).


### 4.2 Window-Level Input Reconstruction Threshold

Let the Monte Carlo mean input-window reconstruction score be

\[
S_t^{(\mathrm{input})}
=
\frac{1}{T}\sum_{i=1}^{T}\overline{s}_{t,i}.
\]

The window enters online triage only when

\[
S_t^{(\mathrm{input})}>B_{\mathrm{window}},
\qquad
B_{\mathrm{window}}
=
Q_p\left(S_t^{(\mathrm{input,clean-val})}\right).
\]

This threshold is used for window triage, not for the final point-level anomaly decision. The exact quantile \(p\) remains an explicit protocol value; historical candidates are \(0.95\) and \(0.99\).

### 4.3 Window-Level Latent Threshold Band

THESIS uses the window-level latent score \(S_t^{(\mathrm{latent})}\) with a threshold band:

\[
A_{\mathrm{low}}
=
Q_{p_1}\left(S_t^{(\mathrm{latent,clean-val})}\right),
\qquad
A_{\mathrm{high}}
=
Q_{p_2}\left(S_t^{(\mathrm{latent,clean-val})}\right).
\]

The runtime field for this quantity remains **latent_window_score**. It is a window-level triage score and is not automatically identical to the proposed point-level prototype displacement \(\overline{\ell}^{(c)}_{t,i}\). Historical candidate quantiles are \(p_1=0.95\) and \(p_2=0.99\).

Interpretation:

    S_t^(latent) <= A_low                    close to old normal memory
    A_low < S_t^(latent) <= A_high           gray-zone latent deviation
    S_t^(latent) > A_high                    far from old normal memory

---

## 5. Online Point-Level Score Aggregation

Online TTA uses sliding windows, so one absolute time-point may appear in several windows. Let \(s^{(r)}_n\) be the score produced for absolute point \(n\) when processing causal window \(r\). The active timeline uses

\[
\widetilde{s}^{(r)}_n
=
\rho s^{(r)}_n+(1-\rho)\widetilde{s}^{(r-1)}_n,
\qquad \rho=0.9.
\]

For a newly seen point, initialize \(\widetilde{s}^{(r)}_n=s^{(r)}_n\). Runtime keeps only the active absolute-index map required by the next causal window. A separate point-finalization mechanism is unnecessary: a point stops changing naturally when later windows no longer contain it.

The implementation must keep the weight direction explicit:

    online_score_aggregation: ewma
    online_score_current_weight: 0.9
    online_score_previous_weight: 0.1

Do not name the current weight **ema_decay**, because that name commonly means the weight on the previous value.

---

## 6. Online Window-Level Triage

At each online step, THESIS receives a sliding window (W).

First compute:

[
s_{\text{input-window}}(W).
]

If:

[
s_{\text{input-window}}(W)\le B_{\text{window}},
]

then the window is not considered for verification buffer admission.

The model still produces point-level anomaly decisions using:

[
S_{\text{point}}(t)>T_{\text{point}}.
]

If:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

then compute:

[
s_{\text{latent-window}}(W).
]

Then classify the window into one of three cases.

### Case 1: Hard Old-Normality Candidate

Condition:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
s_{\text{latent-window}}(W)\le A_{\text{low}}.
]

Interpretation:

```text
The reconstruction branch says the window is abnormal,
but the latent representation is still close to old normal memory.
```

This may indicate a normal pattern already covered by the old normal latent space, but not reconstructed well by the encoder-decoder.

Current action:

```text
do not add to verification buffer
optionally log as hard-old-normality candidate
```

**Undecided:** whether this case should trigger immediate online reconstruction adaptation. If used, it must update only the online branch and must not update the source encoder or source memories.

### Case 2: Verification Buffer Candidate

Condition:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
A_{\text{low}}
<
s_{\text{latent-window}}(W)
\le
A_{\text{high}}.
]

Interpretation:

```text
The reconstruction branch says the window is abnormal,
and the latent representation is neither close enough to old normal memory
nor far enough to be called a clear anomaly.
```

This is the main uncertain region.

Current action:

```text
try to add W into verification buffer
```

The window is admitted only if it does not overlap with any existing buffer window.

### Case 3: Strong Anomaly Candidate

Condition:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
s_{\text{latent-window}}(W)>A_{\text{high}}.
]

Interpretation:

```text
The reconstruction branch says the window is abnormal,
and the latent representation is far from old normal memory.
```

Current action:

```text
do not use for reconstruction adaptation
do not add to verification buffer
optionally log as strong anomaly candidate
```

Future work may consider using this case for discrete codebook or classification-head adaptation, but this is not part of the main method.

---

## 7. Verification Buffer

### 7.1 Buffer Purpose

The verification buffer stores uncertain windows for later inspection.

It is not a normal training buffer.

It is not an anomaly replay buffer.

Its role is to collect candidate windows that may contain either:

```text
new normality patterns
or
new anomaly patterns
```

### 7.2 Buffer Size

Current design:

[
|\mathcal{B}_{verify}|=16.
]

The buffer triggers verification when it contains 16 windows.

### 7.3 Non-Overlap Rule

Each window is represented by its absolute interval:

[
W_i=[s_i,e_i].
]

Two windows are non-overlapping if:

[
e_i < s_j
\quad \text{or} \quad
e_j < s_i.
]

Adjacent windows are allowed.

Example allowed:

[
[0,19],\quad [20,39].
]

Example forbidden:

[
[0,19],\quad [1,20].
]

Admission rule:

```python
def is_non_overlapping(candidate, buffer):
    s, e = candidate.start, candidate.end

    for w in buffer:
        if not (e < w.start or w.end < s):
            return False

    return True
```

### 7.4 Buffer Admission Rule

A window (W) is admitted into the verification buffer if:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
A_{\text{low}}
<
s_{\text{latent-window}}(W)
\le
A_{\text{high}},
]

and:

[
W \text{ does not overlap with any existing buffer window}.
]

Pseudocode:

```python
def should_add_to_verification_buffer(W, buffer):
    if overlaps_any(W, buffer):
        return False

    input_w = compute_window_input_mse(W)

    if input_w <= B_window:
        return False

    latent_w = compute_window_latent_mse(W)

    if A_low < latent_w <= A_high:
        return True

    return False
```

This design uses no additional hyperparameters beyond existing thresholds.

---

## 8. Buffer Verification (MOSTLY UNDECIDED - description in this section is just suggestions)

Once the verification buffer reaches 16 non-overlapping windows, trigger verification.

Verification is performed at the latent time-point level.

For each buffered window (W), obtain latent tokens:

[
Z_W = {z_t}_{t\in W}.
]

The goal is to identify whether certain latent time-point patterns repeat across non-overlapping windows.

High-level interpretation:

```text
recurrent pattern
    -> weak new-normality candidate

rare pattern
    -> suspicious anomaly candidate

ambiguous pattern
    -> unresolved candidate
```

### 8.1 Signature-Based Verification (UNDECIDED TO USE OR NOT - signature-based checking is too rigid, not flexible enough)

To minimize new hyperparameters, the recommended first implementation should avoid distance thresholds or clustering radii.

Instead, define a discrete signature for each latent time-point using quantities the model already computes.

Candidate signature components:

```text
nearest continuous prototype id
top-1 discrete codeword id
predicted synthetic class
```

Example:

```python
signature_t = (
    nearest_continuous_prototype_id,
    top1_discrete_codeword_id,
    predicted_class_id,
)
```

Then count repeated signatures inside the buffer.

```python
from collections import Counter

counts = Counter(signatures)
```

A signature that appears more than once is treated as recurrent.

A signature that appears only once is treated as rare.

This rule avoids introducing a new frequency threshold.

### 8.2 Verification Output (UNDECIDED)

Current simple rule:

```text
count(signature) > 1
    -> weak recurrent candidate

count(signature) == 1
    -> rare suspicious candidate
```

The recurrent candidate must not be called confirmed new normality.

Use the safer term:

```text
weak new-normality candidate
```

The rare candidate should be logged as:

```text
suspicious anomaly candidate
```

**Undecided:** whether middle-frequency states should exist. If the design wants no new hyperparameters, then the first implementation should avoid a middle state and use only repeated vs singleton signatures.

**Undecided:** exact signature definition.

Candidate signatures:

```text
Option A:
    nearest continuous prototype id

Option B:
    nearest continuous prototype id
    + top-1 discrete codeword id

Option C:
    nearest continuous prototype id
    + top-1 discrete codeword id
    + predicted synthetic class

Option D:
    top-k discrete codeword tuple
    + predicted synthetic class
```

Recommended first implementation:

```text
Option B
```

because it uses both old-normal memory and discrete anomaly-pattern memory, while remaining simple.

---

## 9. Online Adaptation Policy

### 9.1 Source Model

The source model must remain frozen.

Frozen components:

```text
source encoder
source reconstruction head
source continuous prototype bank
source discrete codebook
source classifier head
```

### 9.2 Online Model

The online branch may be adapted.

Allowed update targets in the main design:

```text
MLP projector / adapter
possibly a very small subset of 1D-CNN encoder parameters
```

Avoid updating in the main method:

```text
source memory
source encoder
source codebook
full encoder-decoder (both online and source)
classification head
discrete codebook
```

### 9.3 Adaptation on Hard Old-Normality Candidate

Condition:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
s_{\text{latent-window}}(W)\le A_{\text{low}}.
]

Possible interpretation:

```text
normal-like latent structure, poor reconstruction
```

Possible adaptation:

```text
small online reconstruction update
or
online-to-source contrastive alignment
```

**Undecided:** whether to perform this update in the main method.

Caution:

```text
Do not update from a single hard-old-normality window too aggressively.
A subtle anomaly may still have low latent deviation if the encoder fails to separate it.
```

### 9.4 Adaptation on Verification Buffer Output

For recurrent signatures:

```text
allow very light online representation alignment
```

Potential loss:

[
\mathcal{L}_{O\to S}
]

using online projected token as anchor:

[
q_{i,t}^{on}
============

\operatorname{Normalize}(g_\psi(z_{i,t}^{on})).
]

Positive source key:

[
k_{i,t}^{src}
=============

\operatorname{sg}
\left(
\operatorname{Normalize}(z_{i,t}^{src})
\right).
]

Contrastive objective:

[
\mathcal{L}_{O\to S}(i,t)
=========================

-\log
\frac{
\exp((q_{i,t}^{on})^\top k_{i,t}^{src}/\tau)
}{
\exp((q_{i,t}^{on})^\top k_{i,t}^{src}/\tau)
+
\sum_{n\in N(i,t)}
\exp((q_{i,t}^{on})^\top n/\tau)
}.
]

Main rule:

```text
recurrent buffer pattern
    -> projector-only or adapter-only update

rare buffer pattern
    -> no update

strong anomaly candidate
    -> no reconstruction update
```

**Undecided:** whether recurrent buffer candidates should update the reconstruction branch. Current conservative recommendation: no, not in the main method.

---

## 10. Point-Level Final Decision

Regardless of online buffer logic, point-level anomaly prediction uses:

[
S_{\text{point}}(t)>T_{\text{point}}.
]

The buffer does not directly define final anomaly labels.

Buffer verification only controls whether online adaptation is allowed.

This separation is important:

```text
point-level threshold
    -> detection decision

window-level thresholds
    -> online adaptation candidate selection

buffer verification
    -> decide whether adaptation is allowed
```

---

## 11. Full Online TTA Pseudocode (just for reference only)

```python
for tau in online_stream:

    # 1. Build sliding window
    W = get_sliding_window_ending_at(tau)

    # 2. Forward pass
    outputs = model.forward(W)

    point_scores = outputs.point_reconstruction_mse
    input_window_score = outputs.window_reconstruction_mse
    latent_window_score = outputs.window_latent_mse

    # 3. Update point-level EWMA scores
    for abs_t in W.absolute_points:
        cur_score = point_scores[abs_t]

        if abs_t not in point_score_state:
            point_score_state[abs_t] = cur_score
        else:
            point_score_state[abs_t] = (
                0.9 * cur_score
                + 0.1 * point_score_state[abs_t]
            )

    # 4. Finalize expired point scores
    finalize_points_that_can_no_longer_overlap()

    # 5. Point-level anomaly decision
    for finalized_t in finalized_points:
        y_pred[finalized_t] = (
            point_score_state[finalized_t] > T_point
        )

    # 6. Window-level triage
    if input_window_score <= B_window:
        continue

    if latent_window_score <= A_low:
        log_hard_old_normality_candidate(W)
        # Optional adaptation: undecided
        continue

    if latent_window_score > A_high:
        log_strong_anomaly_candidate(W)
        continue

    # 7. Gray-zone window
    if A_low < latent_window_score <= A_high:
        if not overlaps_any(W, verification_buffer):
            verification_buffer.add(W)

    # 8. Trigger verification
    if len(verification_buffer) == 16:
        recurrent, rare = verify_buffer(verification_buffer)

        adapt_on_recurrent_candidates(recurrent)
        log_rare_candidates(rare)

        verification_buffer.clear()
```

---

## 12. Minimal Configuration Draft

```yaml
# ============================================================
# THESIS online TTA: point-level decision + window-level buffer
# ============================================================

online_tta_enabled: true

# Windowing
online_window_mode: sliding
offline_eval_window_mode: non_overlapping
validation_window_mode: non_overlapping
test_window_mode: non_overlapping

# Point-level decision threshold
point_threshold_source: clean_validation
point_threshold_quantile: null  # TODO: choose 0.95 or 0.99

# Window-level reconstruction gate
window_input_threshold_source: clean_validation
window_input_threshold_quantile: null  # TODO: choose 0.95 or 0.99

# Window-level latent triage band
window_latent_low_quantile: null   # TODO: candidate 0.95
window_latent_high_quantile: null  # TODO: candidate 0.99

# Online point-score aggregation
online_point_score_aggregation: ewma
online_score_current_weight: 0.9
online_score_previous_weight: 0.1

# Verification buffer
verification_buffer_enabled: true
verification_buffer_size: 16
verification_buffer_non_overlap: true

# Buffer admission
buffer_admission_rule: input_window_high_and_latent_gray_zone

# Buffer verification
buffer_verification_level: latent_time_point
buffer_signature_mode: null  # TODO: choose prototype_id, prototype+codeword, etc.

# Online update
source_model_frozen: true
source_memory_frozen: true
online_update_targets:
  - mlp_projector
  # TODO: optionally a small subset of 1D-CNN encoder

online_update_reconstruction_branch: false  # TODO: undecided for hard-old-normality
online_update_discrete_codebook: false
online_update_classification_head: false
```

---

## 13. Undecided Items

### 13.1 Exact Quantiles

Need to decide:

[
T_{\text{point}} = Q_p(\cdot)
]

[
B_{\text{window}} = Q_p(\cdot)
]

[
A_{\text{low}} = Q_{p_1}(\cdot)
]

[
A_{\text{high}} = Q_{p_2}(\cdot)
]

Candidate values:

```text
T_point: 0.99
B_window: 0.99
A_low: 0.95
A_high: 0.99
```

But these are not yet finalized.

### 13.2 Calibration Protocol for Online EMA

Validation and offline test are non-overlapping.

Online TTA is sliding-window based.

Undecided:

```text
Should online thresholds be calibrated with non-overlapping validation windows,
or by simulating sliding-window + EWMA on clean validation?
```

Conservative statistical answer:

```text
simulate online sliding-window + EWMA on clean validation
```

Simpler engineering answer:

```text
use non-overlapping clean validation thresholds
```

This should be decided before final experiments.

### 13.3 Hard Old-Normality Update

For:

[
s_{\text{input-window}}(W)>B_{\text{window}},
]

[
s_{\text{latent-window}}(W)\le A_{\text{low}},
]

the window may be a normal pattern that the encoder-decoder has not reconstructed well.

Undecided:

```text
Should THESIS update the online reconstruction branch on this case?
If yes, how many steps or epochs?
Which parameters?
What loss?
```

Current conservative default:

```text
log only, no main update
```

Alternative ablation:

```text
5-epoch online reconstruction update
on online branch only
with source regularization
```

### 13.4 Verification Signature

Undecided signature choices:

```text
A. nearest continuous prototype id
B. nearest continuous prototype id + top-1 discrete codeword id
C. nearest continuous prototype id + top-1 discrete codeword id + predicted class
D. top-k discrete codeword tuple + predicted class
```

Recommended first implementation:

```text
B. nearest continuous prototype id + top-1 discrete codeword id
```

### 13.5 Meaning of Recurrent Candidate

Current minimal rule:

```text
signature count > 1
    -> weak recurrent candidate
```

This avoids new hyperparameters.

However, it is weak. A repeated anomaly may also satisfy this rule.

Therefore, THESIS must not claim:

```text
confirmed new normality
```

It should claim only:

```text
weak new-normality candidate
```

### 13.6 Whether to Update Online Continuous Prototype Bank

Undecided.

Current main method:

```text
do not update online continuous prototype bank
```

Possible future ablation:

```text
update shadow online continuous prototype bank
using recurrent verification candidates only
```

Source continuous prototype bank must remain frozen.

### 13.7 Whether to Update Discrete Codebook and Classification Head

Not part of the main method.

Future work only:

```text
discrete codebook update
classification fusion head update
classification prediction head update
```

This may be useful for clear anomaly patterns, but it is outside the current minimal online TTA design.

---

## 14. Data Leakage Rules

The following rules are mandatory.

Validation statistics may be used only for threshold calibration.

No gradient may pass through validation statistics.

Test labels must never be used during online TTA.

Test future windows must not be used if the protocol claims causal online adaptation.

Source model and source memories must remain frozen.

Verification buffer contains test-stream windows, but its use must be described as test-time adaptation, not supervised training.

Synthetic validation should not be used as the main source of operational thresholds unless explicitly reported as an ablation.

---

## 15. Recommended First Implementation

The cleanest first implementation should use:

```text
1. Point-level final anomaly decision with T_point.

2. Window-level triage:
   input_window > B_window,
   then latent_window is classified into:
       <= A_low      -> hard-old-normality candidate
       (A_low,A_high] -> verification buffer candidate
       > A_high      -> strong anomaly candidate

3. Verification buffer:
   16 non-overlapping windows.

4. Buffer verification:
   latent time-point signature counting.

5. Online update:
   projector-only contrastive alignment for recurrent signatures,
   no source update,
   no reconstruction update in the main method.
```

This version is simple, conservative, and easy to defend.

---

## 16. Terminology Notes

The term **anomaly score** is standard in anomaly detection literature, including Chandola, Banerjee, and Kumar (2009).

The term **effective receptive field** is standard in convolutional neural network analysis, especially Luo et al. (2016), *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks*.

The term **EWMA** or **exponentially weighted moving average** is standard in statistical process control, classically associated with Roberts (1959).

The term **InfoNCE** is standard in contrastive representation learning from van den Oord, Li, and Vinyals (2018), *Representation Learning with Contrastive Predictive Coding*.

The term **supervised contrastive learning** is standard from Khosla et al. (2020), *Supervised Contrastive Learning*.

The term **test-time adaptation** is widely used in works such as Sun et al. (2020), *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*, and Wang et al. (2021), *Tent: Fully Test-Time Adaptation by Entropy Minimization*.

---

## 17. Final Check

The design is internally consistent if the following statements remain true:

```text
Final detection is point-level.
Buffer admission is window-level.
Buffer verification is latent time-point-level.
Validation/test offline windows are non-overlapping.
Online TTA uses sliding windows.
Verification buffer windows are non-overlapping.
All thresholds are derived from clean validation.
Synthetic validation is diagnostic, not the main calibration source.
Source model and source memories are frozen.
Online updates are conservative and parameter-limited.
```

Confidence: High.

Main risk:

```text
The verification buffer may still confuse recurrent anomalies with weak new-normality candidates.
```

Main mitigation:

```text
Do not update reconstruction branch aggressively.
Do not update source memory.
Do not claim confirmed new-normal discovery.
```
