# Development Specification: Two-Stage Offline Pre-training with Point-wise Balanced Reconstruction-Score Loss for `thesis_multitask.py`

> **Notation authority:** Khi đối chiếu anomaly score mức điểm, tài liệu lịch sử này dùng mapping trong [Thiết kế anomaly score mức điểm và bộ ký hiệu chuẩn](anomaly-score-designs-and-notation.md). Tên runtime và ngữ nghĩa lịch sử trong thân tài liệu được giữ nguyên.


## 1. Purpose

Tài liệu này đặc tả cách implement thí nghiệm **two-stage offline pre-training** cho mô hình `thesis_multitask.py`, với thay đổi chính là sử dụng **point-wise balanced reconstruction-score loss** thay cho bản balanced reconstruction-score loss ở mức window.

Mục tiêu kỹ thuật là giữ thí nghiệm đơn giản, kiểm soát được, không vi phạm data leakage, và làm cho reconstruction MSE có ý nghĩa trực tiếp hơn như **point-wise anomaly score**. Lý do chính: các metric cuối như `VUS-PR`, `VUS-ROC`, và `Affiliation F1` cuối cùng đều được tính từ chuỗi point-wise scores đã được sắp xếp lại theo timeline.

Bản spec này định nghĩa hai chế độ chạy:

1. **Base two-stage run**: giữ thiết kế gốc, Stage A dùng 3 loss:
   - reconstruction loss,
   - classification loss,
   - contrastive loss.

2. **Point-score-supervised two-stage run**: thêm point-wise balanced reconstruction-score loss vào Stage A:
   - reconstruction loss,
   - classification loss,
   - contrastive loss,
   - point-wise balanced reconstruction-score loss.

Để tránh nhầm với SSOT cũ, bản có point-wise score loss phải được đặt tên rõ trong config/log, ví dụ:

```yaml
experiment_variant: two_stage_point_score_supervised_v1
score_loss_granularity: point
```

Không được âm thầm thay thế base two-stage run mà không đổi tên variant.

---

## 2. Scope

### 2.1 In scope

Spec này áp dụng cho:

```text
src/models/thesis_multitask.py
```

và các training scripts/configs liên quan đến benchmark-style thesis experiment.

Các thành phần nằm trong scope:

```text
1. Stage A multitask pre-training
2. End-of-Stage-A k-means memory initialization
3. Stage B frozen-encoder frozen-memory head training
4. Point-wise balanced reconstruction-score loss
5. Timeline-compatible point-score logging
6. Validation, checkpointing, and threshold calibration
7. Leakage-safety checks
8. Unit tests and integration tests
```

### 2.2 Out of scope

Không áp dụng cho:

```text
src/models/redlamp_baseline.py
```

Không implement trong bản này:

```text
1. Online test-time adaptation
2. Online memory update
3. Shadow memory
4. COMET-style direct codebook update
5. Recurrence-based new-normal discovery
6. CKA-gated fusion, unless explicitly enabled in a later experiment
7. Direct differentiable optimization of VUS-PR, VUS-ROC, or Affiliation F1
8. Event-level or segment-level loss
```

---

## 3. Terminology

### 3.1 Window

Input window:

$$
x_i \in \mathbb{R}^{L \times C}
$$

Với config hiện tại:

$$
L=20,\qquad C=38
$$

Nên:

$$
x_i \in \mathbb{R}^{20 \times 38}
$$

Batch input:

$$
X \in \mathbb{R}^{B \times 20 \times 38}
$$

### 3.2 Time-point / token

Một time-point trong window là:

$$
x_{i,t} \in \mathbb{R}^{C}
$$

với:

$$
t \in \{1,\dots,L\}
$$

Trong spec này, **token** nghĩa là một latent vector tương ứng với một time-point:

$$
z_{i,t} \in \mathbb{R}^{d_h}
$$

Không dùng từ token để chỉ channel riêng lẻ.

### 3.3 Latent tensor

Encoder output:

$$
Z = E_\theta(X)
$$

Shape kỳ vọng:

$$
Z \in \mathbb{R}^{B \times L \times d_h}
$$

Với config hiện tại:

$$
d_h=32
$$

Nên:

$$
Z \in \mathbb{R}^{B \times 20 \times 32}
$$

### 3.4 Continuous prototype bank

Continuous memory:

$$
P^{(c)} \in \mathbb{R}^{K_c \times d_h}
$$

Với:

$$
K_c=32,\qquad d_h=32
$$

Nên:

$$
P^{(c)} \in \mathbb{R}^{32 \times 32}
$$

Continuous memory đại diện cho **clean / normal latent structure**.

### 3.5 Discrete codebook

Discrete memory:

$$
E^{(d)} \in \mathbb{R}^{K_d \times d_h}
$$

Với:

$$
K_d=60,\qquad d_h=32
$$

Nên:

$$
E^{(d)} \in \mathbb{R}^{60 \times 32}
$$

Discrete codebook đại diện cho **class-stratified synthetic pattern structure**.

Thiết kế hiện tại có:

```text
12 classes = 1 normal class + 11 synthetic anomaly classes
```

Với 5 codewords mỗi class:

$$
12 \times 5 = 60
$$

### 3.6 Point-wise reconstruction score

Với reconstruction output:

$$
\hat{x}_i \in \mathbb{R}^{L \times C}
$$

point-wise reconstruction score tại time-point `t` là:

$$
r_{i,t}
=
\frac{1}{C}
\sum_{c=1}^{C}
\left(\hat{x}_{i,t,c}-x^{input}_{i,t,c}\right)^2
$$

Shape:

$$
R = [r_{i,t}] \in \mathbb{R}^{B \times L}
$$

Trong bản point-wise score-supervised run, `L_score` dùng `r_{i,t}`, không dùng window-level score `r_i`.

### 3.7 Window-level reconstruction score

Window-level score vẫn có thể dùng cho diagnostic hoặc backward compatibility:

$$
r_i
=
\frac{1}{L}
\sum_{t=1}^{L} r_{i,t}
$$

Tuy nhiên, đây không phải score chính của `point-wise balanced reconstruction-score loss`.

### 3.8 Point-wise binary score label

Từ synthetic anomaly mask, tạo point-wise binary score label:

$$
a_{i,t} \in \{0,1\}
$$

Trong đó:

$$
a_{i,t}=0 \quad \text{nếu time-point } t \text{ là normal / non-injected}
$$

$$
a_{i,t}=1 \quad \text{nếu time-point } t \text{ chứa synthetic injected anomaly}
$$

Nếu `synthetic_anomaly_mask` có shape `[B, L, C]`, quy về point-level bằng:

$$
a_{i,t} = \mathbf{1}\left[\sum_{c=1}^{C} M_{i,t,c} > 0\right]
$$

Nếu `synthetic_anomaly_mask` có shape `[B, L]`, dùng trực tiếp:

$$
a_{i,t}=M_{i,t}
$$

Quan trọng: không được gán toàn bộ synthetic anomalous window là anomalous tokens. Chỉ những vị trí thật sự bị injected mới có `a_{i,t}=1`.

---

## 4. High-Level Training Topology

Toàn bộ offline pre-training gồm hai stage.

```text
┌────────────────────────────────────────────┐
│                STAGE A                     │
│        multitask encoder training           │
│                                            │
│ trainable:                                 │
│   - shared encoder                          │
│   - reconstruction head                     │
│   - classification head                     │
│   - contrastive/projector components        │
│                                            │
│ losses:                                    │
│   - L_recon                                │
│   - L_cls                                  │
│   - L_contrastive                          │
│   - optional L_score_point                 │
│                                            │
│ memory banks:                              │
│   - not yet final frozen banks              │
└──────────────────────┬─────────────────────┘
                       │
                       v
┌────────────────────────────────────────────┐
│       END-OF-STAGE-A MEMORY INIT            │
│                                            │
│ source: training split only                 │
│ method: k-means                             │
│ output:                                     │
│   - continuous_prototype_bank               │
│   - discrete_codebook                       │
└──────────────────────┬─────────────────────┘
                       │
                       v
┌────────────────────────────────────────────┐
│                STAGE B                     │
│      frozen encoder + frozen memories       │
│                                            │
│ frozen:                                    │
│   - shared encoder                          │
│   - continuous_prototype_bank               │
│   - discrete_codebook                       │
│                                            │
│ trainable:                                 │
│   - reconstruction fusion head              │
│   - classification fusion head              │
│   - reconstruction prediction head          │
│   - classification prediction head          │
└────────────────────────────────────────────┘
```

Epoch budget:

```yaml
stage_a_epochs: 80
stage_b_epochs: 20
total_epochs: 100
```

---

## 5. Configuration Contract

### 5.1 Required existing config

The following config values are expected:

```yaml
model_name: thesis_multitask
enable_classification_path: true
input_dim: 38
window_size: 20
num_classes: 12

encoder_dim: 64
hidden_dim: 32
encoder_family: cnn_simple
cnn_num_layers: 3
cnn_kernel_size: 3
cnn_hidden_channels: 64
cnn_dropout: 0.1
dropout: 0.1

continuous_enabled: true
continuous_num_prototypes: 32

discrete_enabled: true
discrete_codebook_size: 60
discrete_query_mode: cosine_topk
discrete_topk: 3
discrete_query_temperature: 0.1

use_label_refurbishment: true
refurbishment_alpha: 0.1
refurbishment_beta: 0.01
reconstruction_normal_only: true

lambda_recon: 0.5
lambda_cls: 0.5
lambda_contrastive: 0.3

enable_two_view_contrastive: true
contrastive_temperature: 0.1

memory_initialization_batches: 16
memory_initialization_with_synthetic_windows: true
freeze_memories_after_initialization: true

fusion_mode: task_specific_concat_projection
```

### 5.2 Recommended naming update

Current field:

```yaml
training_phase: stage_a_multitask_pretraining
```

Recommended future name:

```yaml
training_phase: phase_multitask_pretraining
```

However, to avoid breaking existing scripts, implement backward compatibility:

```python
if training_phase == "stage_a_multitask_pretraining":
    training_phase = "phase_multitask_pretraining"
```

### 5.3 New config for point-score-supervised variant

Add:

```yaml
enable_score_loss: true
score_loss_type: pointwise_balanced_bce_logits
score_loss_granularity: point
score_loss_normalization: train_batch_normal_tokens_detached_mean_std
score_loss_reduction: pointwise_binary_balanced_mean
score_loss_target: synthetic_anomaly_mask
```

Do **not** add `lambda_score` in the main version.

Reason: the current design wants to avoid extra hyper-parameter tuning.

Instead, merge `L_score_point` into the classification branch:

```text
L_cls_score = (L_cls + L_score_point) / 2
```

Then use the existing `lambda_cls`.

If `enable_score_loss: false`, fallback exactly to base two-stage behavior.

### 5.4 Deprecated window-level score-loss config

Do not use the old window-level settings for the main thesis rerun:

```yaml
score_loss_type: binary_balanced_bce_logits
score_loss_granularity: window
score_loss_normalization: batch_detached_mean_std
```

These may remain only for backward compatibility or ablation.

---

## 6. Stage A: Forward Pass Contract

### 6.1 Inputs

A Stage A training batch must provide:

```python
batch = {
    "x_clean": Tensor[B, L, C],
    "x_input": Tensor[B, L, C],
    "class_labels": LongTensor[B],
    "synthetic_anomaly_mask": BoolTensor[B, L] or BoolTensor[B, L, C],
    "is_synthetic": BoolTensor[B],
}
```

Expected meaning:

```text
x_clean:
    clean original window

x_input:
    actual model input
    can be clean original or synthetic anomalous

class_labels:
    0 = normal / unchanged
    1..11 = synthetic anomaly classes

synthetic_anomaly_mask:
    marks injected anomaly positions
    all false for normal class
    this is the source of point-wise score labels

is_synthetic:
    true if the sample belongs to one of the 11 synthetic anomaly classes
```

If the existing dataloader does not provide both `x_clean` and `x_input`, then augmentation code must return both.

If the existing dataloader does not provide `synthetic_anomaly_mask`, the point-wise score-supervised variant must not run. Do not infer point labels from window labels alone.

### 6.2 Encoder output

```python
outputs = model(x_input)
```

Expected keys:

```python
outputs = {
    "hidden": Tensor[B, L, hidden_dim],
    "reconstruction": Tensor[B, L, input_dim],
    "classification_logits": Tensor[B, num_classes],
    ...
}
```

With current config:

```text
hidden: [B, 20, 32]
reconstruction: [B, 20, 38]
classification_logits: [B, 12]
```

---

## 7. Stage A Losses

Stage A has the following losses:

```text
1. reconstruction loss
2. classification loss
3. two-view contrastive loss
4. optional point-wise balanced reconstruction-score loss
```

The base run uses only the first three.

The point-score-supervised run uses all four.

---

## 8. Reconstruction Loss

### 8.1 Design intent

The reconstruction branch should learn to reconstruct normal structure.

Because:

```yaml
reconstruction_normal_only: true
```

the reconstruction loss must not reward the model for copying injected anomaly values.

### 8.2 Reconstruction target

For clean windows:

```text
target = x_clean = x_input
```

For synthetic anomalous windows:

```text
target = x_clean
```

The model input is synthetic:

```text
input = x_input
```

but the target is the original clean value:

```text
target = x_clean
```

### 8.3 Masking rule

If `reconstruction_normal_only: true`, then reconstruction loss only uses clean / non-injected positions.

Let:

$$
M_i \in \{0,1\}^{L}
$$

where:

```text
M_i[t] = 1 -> injected anomaly position
M_i[t] = 0 -> clean position
```

Clean mask:

$$
C_i[t] = 1 - M_i[t]
$$

Reconstruction loss:

$$
\mathcal{L}_{recon}
=
\frac{
\sum_{i,t,c}
C_i[t]
\left(\hat{x}_{i,t,c}-x^{clean}_{i,t,c}\right)^2
}{
\sum_{i,t,c} C_i[t] + \epsilon
}
$$

For normal class, `M_i[t]=0` for all `t`, so all positions are used.

### 8.4 Important distinction

Training reconstruction loss:

```text
masked clean-position MSE against x_clean
```

Point-wise score loss and validation/test anomaly score:

```text
point-wise MSE against x_input
```

This distinction is intentional.

The reconstruction loss prevents copying synthetic anomaly values.

The point-wise score loss makes the per-time-point reconstruction MSE useful as an anomaly score.

---

## 9. Classification Loss

Classification head predicts one of 12 classes:

$$
y_i^{cls} \in \{0,\dots,11\}
$$

Classification loss:

$$
\mathcal{L}_{cls}
=
\operatorname{CrossEntropy}\left(\operatorname{logits}_i, y_i^{cls}\right)
$$

This loss preserves class-level information.

Role:

```text
L_cls:
    distinguish 1 normal class and 11 synthetic anomaly classes
```

Important: this remains window-level classification. It should not be replaced by point-wise score loss.

---

## 10. Two-View Contrastive Loss

### 10.1 Design intent

Contrastive loss improves latent geometry.

It should encourage representation consistency between views of compatible samples and discourage collapse.

This spec does not redefine the existing contrastive implementation. It only requires the following contract.

### 10.2 Required behavior

If:

```yaml
enable_two_view_contrastive: true
```

then Stage A must compute:

$$
\mathcal{L}_{contrastive}
$$

using existing two-view contrastive logic.

Expected config:

```yaml
contrastive_temperature: 0.1
lambda_contrastive: 0.3
```

### 10.3 Safety rule

Contrastive pairs and positives must be constructed from training data only.

Do not use validation or test windows to build contrastive positives, negatives, queues, or memory banks.

---

## 11. Point-wise Balanced Reconstruction-Score Loss

### 11.1 Purpose

Point-wise balanced reconstruction-score loss teaches the model:

```text
normal time-points:
    point-wise reconstruction score should be low

synthetic anomalous time-points:
    point-wise reconstruction score should be high
```

This loss does not replace classification loss.

It only supervises the point-wise reconstruction score:

$$
r_{i,t}
$$

### 11.2 Point-wise score computation

For every sample `i` and time-point `t`, compute:

$$
r_{i,t}
=
\frac{1}{C}
\sum_{c=1}^{C}
\left(\hat{x}_{i,t,c}-x^{input}_{i,t,c}\right)^2
$$

Shape:

```text
r: [B, L]
```

Important:

```text
Use x_input, not x_clean, for score loss.
Use point-wise MSE, not window-level MSE.
Do not use clean-position reconstruction mask for score loss.
```

Reason:

```text
The score loss should match the anomaly-score definition used during validation/test.
The final metrics consume point-wise scores arranged on the original timeline.
```

### 11.3 Point-wise labels from synthetic anomaly mask

The score target is not derived from `class_labels > 0` at window level.

It must be derived from the injected anomaly mask.

If `synthetic_anomaly_mask` shape is `[B, L]`:

```python
a = synthetic_anomaly_mask.float()  # [B, L]
```

If `synthetic_anomaly_mask` shape is `[B, L, C]`:

```python
a = synthetic_anomaly_mask.any(dim=-1).float()  # [B, L]
```

Mathematically:

$$
a_{i,t}
=
\begin{cases}
0, & \text{normal / non-injected time-point} \\
1, & \text{synthetic injected anomalous time-point}
\end{cases}
$$

For normal class windows:

$$
a_{i,t}=0 \quad \forall t
$$

For synthetic anomalous windows, only injected positions have:

$$
a_{i,t}=1
$$

### 11.4 Batch score normalization

Normalize point-wise scores using **normal tokens in the current training batch only**.

Let:

$$
\mathcal{N}=\{(i,t):a_{i,t}=0\}
$$

$$
\mathcal{A}=\{(i,t):a_{i,t}=1\}
$$

Compute:

$$
\mu_N
=
\operatorname{mean}_{(i,t)\in\mathcal{N}} r_{i,t}
$$

$$
\sigma_N
=
\operatorname{std}_{(i,t)\in\mathcal{N}} r_{i,t}
$$

Detach both:

$$
\bar{\mu}_N=\operatorname{sg}(\mu_N)
$$

$$
\bar{\sigma}_N=\operatorname{sg}(\sigma_N)
$$

Then:

$$
z_{i,t}
=
\frac{r_{i,t}-\bar{\mu}_N}{\bar{\sigma}_N+\epsilon}
$$

where:

```text
epsilon = 1e-6
```

This epsilon is a numerical stability constant, not a tunable hyper-parameter.

Rationale: using normal-token statistics makes the logit represent deviation above the normal reconstruction-score baseline. Using all tokens for mean/std can be distorted if the batch contains many synthetic anomalous tokens.

### 11.5 Per-token BCEWithLogits

For every eligible time-point:

$$
\ell^{score}_{i,t}
=
\operatorname{BCEWithLogits}(z_{i,t},a_{i,t})
$$

Expanded form:

$$
\ell^{score}_{i,t}
=
-
\left[
 a_{i,t}\log\sigma(z_{i,t})
 +
 (1-a_{i,t})\log(1-\sigma(z_{i,t}))
\right]
$$

where:

$$
\sigma(z_{i,t})=\frac{1}{1+e^{-z_{i,t}}}
$$

Two explicit cases:

$$
a_{i,t}=0
\Rightarrow
\ell^{score}_{i,t}=\log(1+e^{z_{i,t}})
$$

$$
a_{i,t}=1
\Rightarrow
\ell^{score}_{i,t}=\log(1+e^{-z_{i,t}})
$$

Thus:

```text
normal token:
    loss decreases when z_{i,t} is low

synthetic anomalous token:
    loss decreases when z_{i,t} is high
```

### 11.6 Balanced point-wise reduction

Do not average over all `[B, L]` tokens directly.

Reason: synthetic anomalous tokens are usually sparse, while normal tokens dominate. A raw mean can make the anomaly-token term too weak.

Instead:

$$
\mathcal{L}_{normal}^{point}
=
\operatorname{mean}_{(i,t)\in\mathcal{N}}
\operatorname{BCEWithLogits}(z_{i,t},0)
$$

$$
\mathcal{L}_{anomaly}^{point}
=
\operatorname{mean}_{(i,t)\in\mathcal{A}}
\operatorname{BCEWithLogits}(z_{i,t},1)
$$

Final point-wise balanced score loss:

$$
\mathcal{L}_{score}^{point}
=
\frac{1}{2}\mathcal{L}_{normal}^{point}
+
\frac{1}{2}\mathcal{L}_{anomaly}^{point}
$$

Expanded:

$$
\mathcal{L}_{score}^{point}
=
\frac{1}{2}
\operatorname{mean}_{a_{i,t}=0}
\log(1+e^{z_{i,t}})
+
\frac{1}{2}
\operatorname{mean}_{a_{i,t}=1}
\log(1+e^{-z_{i,t}})
$$

### 11.7 Empty group handling

The preferred behavior is to construct training batches so that both groups exist:

```text
normal token count > 0
synthetic anomalous token count > 0
```

If one group is missing:

```python
if normal_count == 0 or anomaly_count == 0:
    skip L_score_point for this batch
    log warning once per epoch
```

Do not silently compute a biased score loss.

### 11.8 Optional receptive-field-safe normal mask

This is optional and should not be required for the main implementation.

Because `cnn_simple` may let an injected anomaly affect neighboring latent/reconstruction outputs through the convolutional receptive field, some non-injected positions may still be indirectly contaminated.

If an architecture-derived receptive field mask is already available, define a stricter normal set:

$$
\mathcal{N}_{RF-safe}
=
\{(i,t):a_{i,t}=0 \text{ and the receptive field of } t \text{ contains no injected anomaly}\}
$$

This does not require a learned hyper-parameter if the receptive field is computed from architecture only.

Main run default:

```yaml
score_loss_normal_token_policy: direct_non_injected
```

Optional ablation only:

```yaml
score_loss_normal_token_policy: receptive_field_safe
```

Do not introduce this ablation unless time allows.

### 11.9 Score loss pseudocode

```python
import torch
import torch.nn.functional as F


def point_mask_from_synthetic_mask(synthetic_anomaly_mask: torch.Tensor) -> torch.Tensor:
    """Return point-wise anomaly mask with shape [B, L]."""
    if synthetic_anomaly_mask.ndim == 2:
        return synthetic_anomaly_mask.bool()
    if synthetic_anomaly_mask.ndim == 3:
        return synthetic_anomaly_mask.bool().any(dim=-1)
    raise ValueError(
        "synthetic_anomaly_mask must have shape [B, L] or [B, L, C]."
    )


def compute_pointwise_balanced_score_loss(
    x_input: torch.Tensor,
    x_hat: torch.Tensor,
    synthetic_anomaly_mask: torch.Tensor,
    eps: float = 1e-6,
):
    """
    x_input: Tensor[B, L, C]
    x_hat: Tensor[B, L, C]
    synthetic_anomaly_mask: BoolTensor[B, L] or BoolTensor[B, L, C]

    Returns:
        loss_score_point: scalar tensor, or None if one group is missing
        diagnostics: dict[str, Tensor]
    """

    # [B, L]
    r = ((x_hat - x_input) ** 2).mean(dim=-1)

    # [B, L], bool
    anomaly_mask = point_mask_from_synthetic_mask(synthetic_anomaly_mask)
    normal_mask = ~anomaly_mask

    normal_count = normal_mask.sum()
    anomaly_count = anomaly_mask.sum()

    diagnostics = {
        "point_score_normal_count": normal_count.detach(),
        "point_score_anomaly_count": anomaly_count.detach(),
    }

    if normal_count == 0 or anomaly_count == 0:
        return None, diagnostics

    # Normal-token baseline, train batch only, detached.
    normal_scores = r[normal_mask]
    mu = normal_scores.mean().detach()
    std = normal_scores.std(unbiased=False).detach()

    # [B, L]
    z = (r - mu) / (std + eps)

    # [B, L], float target
    a = anomaly_mask.float()

    loss_per_token = F.binary_cross_entropy_with_logits(
        z,
        a,
        reduction="none",
    )

    loss_normal = loss_per_token[normal_mask].mean()
    loss_anomaly = loss_per_token[anomaly_mask].mean()
    loss_score = 0.5 * loss_normal + 0.5 * loss_anomaly

    with torch.no_grad():
        anomaly_scores = r[anomaly_mask]
        diagnostics.update(
            {
                "point_score_normal_mean": normal_scores.mean(),
                "point_score_normal_std": normal_scores.std(unbiased=False),
                "point_score_anomaly_mean": anomaly_scores.mean(),
                "point_score_anomaly_std": anomaly_scores.std(unbiased=False),
                "point_score_gap_mean": anomaly_scores.mean() - normal_scores.mean(),
                "point_score_gap_extreme": anomaly_scores.min() - normal_scores.max(),
            }
        )

    return loss_score, diagnostics
```

---

## 12. Stage A Total Loss

### 12.1 Base run

If:

```yaml
enable_score_loss: false
```

then:

$$
\mathcal{L}_{StageA}
=
\lambda_{recon}\mathcal{L}_{recon}
+
\lambda_{cls}\mathcal{L}_{cls}
+
\lambda_{contrastive}\mathcal{L}_{contrastive}
$$

### 12.2 Point-score-supervised run

If:

```yaml
enable_score_loss: true
score_loss_granularity: point
```

then:

$$
\mathcal{L}_{cls\_score}
=
\frac{
\mathcal{L}_{cls}
+
\mathcal{L}_{score}^{point}
}{2}
$$

Total:

$$
\mathcal{L}_{StageA}
=
\lambda_{recon}\mathcal{L}_{recon}
+
\lambda_{cls}\mathcal{L}_{cls\_score}
+
\lambda_{contrastive}\mathcal{L}_{contrastive}
$$

With current config:

$$
\lambda_{recon}=0.5
$$

$$
\lambda_{cls}=0.5
$$

$$
\lambda_{contrastive}=0.3
$$

So:

$$
\mathcal{L}_{StageA}
=
0.5\mathcal{L}_{recon}
+
0.5
\left(
\frac{\mathcal{L}_{cls}+\mathcal{L}_{score}^{point}}{2}
\right)
+
0.3\mathcal{L}_{contrastive}
$$

### 12.3 If point-wise score loss is skipped for a batch

If `L_score_point` returns `None` because one group is missing:

$$
\mathcal{L}_{cls\_score}=\mathcal{L}_{cls}
$$

Then:

$$
\mathcal{L}_{StageA}
=
\lambda_{recon}\mathcal{L}_{recon}
+
\lambda_{cls}\mathcal{L}_{cls}
+
\lambda_{contrastive}\mathcal{L}_{contrastive}
$$

Log:

```text
train/score_loss_skipped_batches += 1
```

### 12.4 Rationale

This avoids introducing a new hyper-parameter such as:

```yaml
lambda_score
```

If future ablation wants to tune score loss strength, then a new config may be added later:

```yaml
lambda_score: ...
```

But that is out of scope for the main implementation.

---

## 13. End-of-Stage-A Memory Initialization

Memory initialization happens after Stage A is completed.

Before collecting latent pools:

```python
model.eval()
# use torch.no_grad()
```

No gradient is allowed.

### 13.1 Source data rule

All memory initialization pools must come from training split only.

Forbidden:

```text
validation windows
test windows
validation-derived statistics
test-derived statistics
future online stream
```

### 13.2 Continuous memory pool

Continuous pool should represent clean / normal latent structure.

Collect latent tokens:

$$
Z_{i,t}
$$

from clean / normal positions only.

Allowed sources:

```text
1. clean original train windows
2. clean positions in synthetic train windows, if synthetic memory initialization is enabled
```

Main requirement:

```text
No injected anomaly token may enter the continuous pool.
```

Continuous pool construction:

```text
for each train batch:
    run encoder
    collect hidden tokens where token is clean / normal
```

Then normalize vectors:

$$
\tilde{z}=
\frac{z}{\lVert z\rVert_2+\epsilon}
$$

Run k-means:

$$
K=32
$$

Output:

$$
P^{(c)}\in\mathbb{R}^{32\times 32}
$$

Copy centroids into:

```python
model.continuous_prototype_bank
```

### 13.3 Discrete memory pool

Discrete pool should represent class-stratified synthetic pattern structure.

Required output:

$$
E^{(d)}\in\mathbb{R}^{60\times 32}
$$

Implementation rule:

```text
5 codewords per class
12 classes total
```

Class 0:

```text
use clean / normal latent tokens
```

Classes 1..11:

```text
use injected anomaly latent tokens from that synthetic class only
```

For each class `c`:

$$
\mathcal{P}_c
=
\{z_{i,t}:y_i^{cls}=c,\ z_{i,t}\text{ is eligible for class }c\}
$$

Run k-means:

$$
K=5
$$

for each class.

Then concatenate:

$$
E^{(d)}=[C_0;C_1;\dots;C_{11}]
$$

where:

$$
C_c\in\mathbb{R}^{5\times 32}
$$

Final shape:

$$
E^{(d)}\in\mathbb{R}^{60\times 32}
$$

### 13.4 Insufficient class tokens

If any class has fewer than 5 eligible tokens:

```python
raise RuntimeError(
    "Not enough tokens for class-level k-means memory initialization."
)
```

Do not silently reduce `K`.

Do not fill with validation/test tokens.

Do not duplicate tokens without logging.

If this error occurs, fix the train augmentation / memory initialization loader so that all 12 classes have enough tokens.

### 13.5 Memory normalization

After k-means:

```python
centroids = F.normalize(centroids, dim=-1, eps=memory_norm_epsilon)
```

Expected:

```yaml
memory_norm_epsilon: 1.0e-6
```

### 13.6 Memory initialization pseudocode

```python
def initialize_memories_after_stage_a(model, train_loader, config, device):
    model.eval()

    continuous_tokens = []
    discrete_tokens_by_class = {c: [] for c in range(config.num_classes)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= config.memory_initialization_batches:
                break

            batch = move_to_device(batch, device)

            x_input = batch["x_input"]
            class_labels = batch["class_labels"]
            anomaly_mask = point_mask_from_synthetic_mask(
                batch["synthetic_anomaly_mask"]
            )  # [B, L]

            outputs = model.forward_encoder_only(x_input)
            hidden = outputs["hidden"]  # [B, L, H]

            clean_token_mask = ~anomaly_mask  # [B, L]
            continuous_tokens.append(hidden[clean_token_mask])

            for c in range(config.num_classes):
                class_mask = class_labels == c  # [B]

                if c == 0:
                    eligible = class_mask[:, None] & clean_token_mask
                else:
                    eligible = class_mask[:, None] & anomaly_mask

                discrete_tokens_by_class[c].append(hidden[eligible])

    continuous_pool = torch.cat(continuous_tokens, dim=0)

    continuous_centroids = run_kmeans(
        F.normalize(continuous_pool, dim=-1),
        k=config.continuous_num_prototypes,
    )

    model.continuous_prototype_bank.copy_(
        F.normalize(continuous_centroids, dim=-1)
    )

    class_centroids = []

    for c in range(config.num_classes):
        pool_c = torch.cat(discrete_tokens_by_class[c], dim=0)

        if pool_c.shape[0] < 5:
            raise RuntimeError(f"Not enough tokens for class {c}")

        centroids_c = run_kmeans(
            F.normalize(pool_c, dim=-1),
            k=5,
        )

        class_centroids.append(centroids_c)

    discrete_codebook = torch.cat(class_centroids, dim=0)

    model.discrete_codebook.copy_(
        F.normalize(discrete_codebook, dim=-1)
    )
```

---

## 14. Stage B: Frozen Encoder and Frozen Memories

### 14.1 Trainable surface

In Stage B, train only:

```text
1. reconstruction fusion head
2. classification fusion head
3. reconstruction prediction head
4. classification prediction head
```

Freeze:

```text
1. shared encoder
2. continuous_prototype_bank
3. discrete_codebook
```

### 14.2 Trainable parameter setup

Pseudocode:

```python
def configure_stage_b_trainable_params(model):
    for p in model.parameters():
        p.requires_grad = False

    modules_to_train = [
        model.reconstruction_fusion_head,
        model.classification_fusion_head,
        model.reconstruction_prediction_head,
        model.classification_prediction_head,
    ]

    for module in modules_to_train:
        for p in module.parameters():
            p.requires_grad = True
```

### 14.3 Safety assertion

Before Stage B starts:

```python
assert not any(p.requires_grad for p in model.encoder.parameters())
assert not model.continuous_prototype_bank.requires_grad
assert not model.discrete_codebook.requires_grad
```

If memory banks are registered as `nn.Parameter`, set:

```python
requires_grad = False
```

If memory banks are buffers, verify they are not passed to optimizer.

### 14.4 Stage B losses

Stage B should use:

```text
1. reconstruction loss
2. classification loss
```

Do not use Stage A contrastive loss unless the frozen/trainable boundary has been explicitly reviewed.

Do not use point-wise score loss in Stage B by default.

Reason:

```text
Stage B trains heads after frozen memory retrieval.
The point-wise score loss mainly shapes encoder/reconstruction score geometry in Stage A.
```

If future ablation wants Stage B point-wise score loss, create a separate experiment variant.

---

## 15. Discrete Retrieval Runtime Contract

Current intended mode:

```yaml
discrete_query_mode: cosine_topk
discrete_topk: 3
discrete_query_temperature: 0.1
```

Implementation requirements:

```text
1. Use cosine similarity or normalized dot product.
2. Select top-k codewords.
3. Compute temperature-scaled soft weights.
4. Return weighted codeword representation.
```

If `discrete_query_mode: cosine_topk`, avoid initializing or storing unnecessary Gumbel-only components.

Specifically:

```text
Do not initialize self.discrete_assignment if it is only used by Gumbel mode.
```

This keeps checkpoint and runtime cleaner.

---

## 16. Validation, Checkpointing, and Timeline Scoring

### 16.1 Validation usage

Validation set may be used for:

```text
1. checkpoint selection
2. early stopping, if enabled
3. post-training threshold calibration
4. metric reporting during training
```

Validation set must not be used for:

```text
1. backpropagation
2. score loss mean/std computation for training
3. memory initialization
4. k-means prototype/codebook construction
5. contrastive memory/queue construction
```

### 16.2 Offline point-wise anomaly score

Validation/test point-wise anomaly score for a window:

For a deterministic forward pass, this historical formula is the \(M=1\) case of the canonical raw point score:

\[
s^{(1)}_{t,i}
=
\frac{1}{C}
\left\|\mathbf{x}_{t,i}-\widehat{\mathbf{x}}^{(1)}_{t,i}\right\|_2^2.
\]

For stochastic inference with \(M\) retrieval samples, validation and test use

\[
\overline{s}_{t,i}=\frac{1}{M}\sum_{m=1}^{M}s^{(m)}_{t,i}.
\]

Shape:

```text
point_scores_window: [B, L]
```

Do not use masked clean-only MSE as anomaly score during validation/test.

### 16.3 Merging overlapping windows back to timeline

If test windows overlap, point-wise scores must be merged back to the original timeline before computing `VUS-PR`, `VUS-ROC`, `Affiliation F1`, or any point/event metric.

Let window `i` start at original timeline index `start_i`. For every local time index `t`, add:

```text
timeline_index = start_i + t
```

For each original timeline index `u`, collect all scores that map to it:

$$
\mathcal{S}(u)=\{s_{i,t}:start_i+t=u\}
$$

Default aggregation:

$$
s_{timeline}(u)=\operatorname{mean}\mathcal{S}(u)
$$

Use the same aggregation for all methods and all experiments.

Do not change aggregation between runs unless the experiment variant name records it.

Recommended config:

```yaml
point_score_timeline_aggregation: mean
```

This field records a deterministic evaluation choice. It should not be tuned on test labels.

### 16.4 Optional threshold calibration

After training, clean validation set may be used to compute threshold `B` from timeline-level or window-level clean validation point scores.

Recommended point-wise threshold:

$$
B=Q_p(s_{timeline}^{clean-val})
$$

where:

$$
p\in\{0.95,0.99\}
$$

This is calibration only.

No gradient.

No parameter update.

For score-based metrics such as `VUS-PR` and `VUS-ROC`, threshold `B` is not required for computing the metric, but it may still be logged for offline decision analysis.

---

## 17. Data Leakage Rules

### 17.1 Absolute prohibitions

Never use validation/test data to:

```text
1. compute training loss
2. update encoder
3. update heads
4. initialize continuous memory
5. initialize discrete codebook
6. fit k-means
7. compute train-time score normalization statistics
8. build contrastive positives/negatives
9. tune timeline aggregation rule using test labels
```

### 17.2 Allowed validation/test usage

Allowed:

```text
1. no_grad validation forward pass
2. checkpoint metric computation
3. post-training threshold calibration
4. final evaluation
5. deterministic timeline score aggregation, if fixed before evaluation
```

### 17.3 Point-wise score loss leakage check

For point-wise score loss:

$$
\mu_N,\sigma_N
$$

must be computed from **normal tokens in the current training batch only**.

They must be detached:

```python
mu = normal_scores.mean().detach()
std = normal_scores.std(unbiased=False).detach()
```

No running mean/std from validation.

No global normalization fitted from validation/test.

No use of validation/test anomaly labels in loss.

---

## 18. Logging Requirements

Log the following per training epoch.

### 18.1 Stage A logs

```text
train/loss_total
train/loss_recon
train/loss_cls
train/loss_contrastive
train/loss_score_point
train/loss_cls_score
train/score_loss_skipped_batches

train/point_score_normal_count
train/point_score_anomaly_count
train/point_score_normal_mean
train/point_score_normal_std
train/point_score_anomaly_mean
train/point_score_anomaly_std
train/point_score_gap_mean
train/point_score_gap_extreme

train/window_score_normal_mean        # diagnostic only
train/window_score_anomaly_mean       # diagnostic only
train/window_score_gap_mean           # diagnostic only
```

Where:

$$
\text{point\_score\_gap\_mean}
=
\operatorname{mean}(r_{anomaly\ tokens})
-
\operatorname{mean}(r_{normal\ tokens})
$$

Optional diagnostic:

$$
\text{point\_score\_gap\_extreme}
=
\min(r_{anomaly\ tokens})
-
\max(r_{normal\ tokens})
$$

This diagnostic is not a loss.

### 18.2 Validation logs

```text
val/point_score_mean
val/point_score_std
val/point_score_p95
val/point_score_p99
val/window_score_mean
val/window_score_std
val/vus_roc
val/vus_pr
val/affiliation_f1
```

If `VUS-PR`, `VUS-ROC`, or `Affiliation F1` is not computed during every validation epoch due to runtime, log it at checkpoint evaluation only.

### 18.3 Memory initialization logs

```text
memory/continuous_pool_size
memory/discrete_pool_size_class_0
memory/discrete_pool_size_class_1
...
memory/discrete_pool_size_class_11

memory/continuous_centroid_norm_mean
memory/discrete_codeword_norm_mean
memory/continuous_shape
memory/discrete_shape
```

Expected shapes:

```text
continuous_shape = [32, 32]
discrete_shape = [60, 32]
```

### 18.4 Stage B logs

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

If any of these is non-zero, Stage B freeze is broken.

---

## 19. Unit Tests

### 19.1 Point-wise balanced score loss shape test

Create synthetic tensors:

```text
x_input: [B, L, C]
x_hat: [B, L, C]
synthetic_anomaly_mask: [B, L]
```

Verify:

```text
r shape is [B, L]
z shape is [B, L]
loss_score_point is scalar
```

### 19.2 Point-wise target test

Create a batch where:

```text
B = 2
L = 20
sample 0: normal class, no anomaly positions
sample 1: synthetic anomaly class, anomaly positions only at t = 5, 6
```

Expected:

```text
a[0, :] = 0

a[1, 5] = 1
a[1, 6] = 1
all other a[1, t] = 0
```

This test must fail if the implementation sets all tokens in sample 1 to anomaly.

### 19.3 Balanced reduction test

Create synthetic point labels with severe imbalance:

```text
normal tokens = 238
anomaly tokens = 2
```

Verify:

```text
L_score_point = 0.5 * L_normal_point + 0.5 * L_anomaly_point
```

not raw mean over all 240 tokens.

### 19.4 Normal-token normalization test

Verify that:

```text
mu and std are computed from normal tokens only
mu and std are detached
```

Expected implementation behavior:

```python
normal_scores = r[normal_mask]
mu = normal_scores.mean().detach()
std = normal_scores.std(unbiased=False).detach()
```

### 19.5 Empty group test

Case 1:

```text
all tokens normal
```

Expected:

```text
score loss returns None or is skipped
warning is logged
training does not crash
```

Case 2:

```text
all tokens anomaly
```

Expected same behavior.

### 19.6 No validation leakage test

Mock train and validation loaders with distinguishable IDs.

Verify:

```text
memory initialization only sees train IDs
score loss only sees train batch
timeline aggregation does not use validation/test labels to choose an aggregation rule
```

### 19.7 Timeline aggregation test

Create two overlapping windows with known start indices and known point scores.

Example:

```text
window 0 start = 0, scores = [1, 2, 3]
window 1 start = 1, scores = [4, 5, 6]
```

Expected timeline with mean aggregation:

```text
timeline[0] = 1
timeline[1] = mean(2, 4) = 3
timeline[2] = mean(3, 5) = 4
timeline[3] = 6
```

### 19.8 Stage B freeze test

After Stage B backward pass:

```python
assert all(
    p.grad is None or torch.all(p.grad == 0)
    for p in model.encoder.parameters()
)
```

Also verify memory banks do not receive gradients.

### 19.9 K-means shape test

After memory initialization:

```python
assert model.continuous_prototype_bank.shape == (32, 32)
assert model.discrete_codebook.shape == (60, 32)
```

### 19.10 K-means class coverage test

Verify each class contributes exactly 5 centroids to discrete codebook.

Expected index layout:

```text
class 0:  codewords 0..4
class 1:  codewords 5..9
class 2:  codewords 10..14
...
class 11: codewords 55..59
```

### 19.11 Cosine-topk mode test

If:

```yaml
discrete_query_mode: cosine_topk
```

then verify Gumbel-only states are not required in forward pass.

---

## 20. Integration Test

Run a small smoke experiment:

```yaml
stage_a_epochs: 2
stage_b_epochs: 1
memory_initialization_batches: 2
batch_size: small
enable_score_loss: true
score_loss_type: pointwise_balanced_bce_logits
score_loss_granularity: point
```

Expected:

```text
1. Stage A completes.
2. Point-wise score loss is finite if both token groups exist.
3. Point-wise score loss is skipped safely if one token group is missing.
4. Memory initialization completes.
5. Continuous memory shape is [32, 32].
6. Discrete codebook shape is [60, 32].
7. Stage B starts with frozen encoder and frozen memories.
8. Stage B completes.
9. Validation/test point scores can be merged back to timeline.
10. No validation/test data is used for training or memory initialization.
```

---

## 21. Acceptance Criteria

Implementation is accepted only if all criteria pass.

### 21.1 Functional criteria

```text
[ ] Stage A runs for 80 epochs.
[ ] Stage B runs for 20 epochs.
[ ] Stage A can run with enable_score_loss=false.
[ ] Stage A can run with enable_score_loss=true.
[ ] Point-wise score loss runs with score_loss_granularity=point.
[ ] End-of-Stage-A k-means memory initialization works.
[ ] Stage B freezes encoder and both memories.
```

### 21.2 Shape criteria

```text
[ ] hidden tensor shape is [B, 20, 32].
[ ] reconstruction output shape is [B, 20, 38].
[ ] classification logits shape is [B, 12].
[ ] point reconstruction score shape is [B, 20].
[ ] point anomaly label shape is [B, 20].
[ ] timeline score shape matches original evaluated sequence length after aggregation.
[ ] continuous_prototype_bank shape is [32, 32].
[ ] discrete_codebook shape is [60, 32].
```

### 21.3 Loss criteria

```text
[ ] L_recon is computed on clean positions only when reconstruction_normal_only=true.
[ ] L_cls uses 12-class window labels.
[ ] L_contrastive uses training data only.
[ ] L_score_point uses point-wise labels derived from synthetic_anomaly_mask.
[ ] L_score_point does not treat an entire synthetic window as anomalous.
[ ] L_score_point uses balanced normal-token/anomaly-token reduction.
[ ] L_score_point uses normal-token train-batch mean/std only.
[ ] L_score_point does not use validation/test statistics.
```

### 21.4 Leakage criteria

```text
[ ] No validation/test windows enter k-means.
[ ] No validation/test windows enter train loss.
[ ] No validation/test statistics are backpropagated.
[ ] Validation is no_grad only.
[ ] Test is evaluation only.
[ ] Timeline aggregation rule is fixed before test evaluation.
```

### 21.5 Logging criteria

```text
[ ] All Stage A losses are logged.
[ ] Point-wise score separation diagnostics are logged.
[ ] Window-level score diagnostics are logged only as diagnostics.
[ ] Validation point-score distribution is logged.
[ ] Memory pool sizes are logged.
[ ] Memory shapes are logged.
[ ] Stage B frozen-gradient diagnostics are logged.
```

---

## 22. Recommended Experiment Matrix

Minimum experiment matrix:

```text
E0: base two-stage
    losses:
        L_recon + L_cls + L_contrastive

E1: point-score-supervised two-stage
    losses:
        L_recon + (L_cls + L_score_point)/2 + L_contrastive
```

Optional ablation if time remains:

```text
E2: point-score-supervised without contrastive
    losses:
        L_recon + (L_cls + L_score_point)/2

E3: window-score-supervised two-stage
    losses:
        L_recon + (L_cls + L_score_window)/2 + L_contrastive
    purpose:
        verify whether point-wise supervision is better aligned with VUS/Affiliation metrics

E4: DevNet-style point-wise deviation score loss
    replace:
        L_score_point = pointwise balanced BCEWithLogits
    with:
        DevNet-style point-wise deviation loss
```

Do not run too many variants unless compute budget allows.

Priority:

```text
1. E0
2. E1
3. E3 only if you want to directly show point-wise > window-level
4. E4 only if E1 is stable and time remains
```

---

## 23. Notes on DevNet-Style Loss

DevNet-style deviation loss is academically attractive because it directly encourages anomaly scores to deviate from a reference distribution.

However, it introduces a margin or confidence parameter, commonly written as a z-score confidence parameter.

For the current implementation, point-wise balanced BCEWithLogits is preferred because:

```text
1. no new margin hyper-parameter
2. easy PyTorch implementation
3. clear point-wise normal/anomaly score supervision
4. better aligned with timeline-based score metrics
5. easier ablation against the base two-stage run
```

DevNet-style loss should be implemented only as a separate ablation.

---

## 24. Final Design Decision

The main implementation should support both:

```text
base two-stage:
    3 losses

point-score-supervised two-stage:
    3 original losses + point-wise balanced reconstruction-score loss
```

The default thesis rerun should report both E0 and E1 if compute budget allows.

If only one new experiment can be run, run:

```text
E1: point-score-supervised two-stage
```

because it directly addresses the desired behavior:

$$
r_{normal\ token}\downarrow
$$

$$
r_{synthetic\ anomalous\ token}\uparrow
$$

while preserving:

```text
1. train-only memory initialization
2. frozen encoder + frozen memories in Stage B
3. no validation/test leakage
4. no extra lambda_score hyper-parameter
5. point-wise score compatibility with timeline-based TSAD metrics
```

The old window-level balanced score loss should be kept only as an ablation or backward-compatible legacy mode, not as the main score-supervised run.
