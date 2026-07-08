# Development Specification: THESIS Offline/Online Experiments and Demo Software

## 0. Status

This specification describes the codebase changes needed to run the current THESIS experiment plan and to build a visual demo for offline and online time-series anomaly detection.

Current locked direction:

```text
Offline pre-training variants:
    O0 = two-stage base
    O1 = two-stage + Point-wise Balanced Reconstruction-Score Loss

Online variants:
    A0 = no online TTA
    A1 = PNN reconstruction-only
    A2 = full online TTA

Online TTA trainable parameters:
    only the light-weight MLP projector g_psi

Frozen during online TTA:
    source encoder
    continuous prototype bank
    discrete codebook
    reconstruction fusion head
    reconstruction prediction head
    classification path
```

The goal is to keep the implementation small, testable, and safe from data leakage.

---

## 1. Scope

### 1.1 Goals

The codebase must support:

```text
1. Run offline O0 and O1 training.
2. Initialize source memories from train split only.
3. Calibrate thresholds from clean validation only.
4. Run offline test prediction.
5. Run online stream prediction with optional online TTA.
6. Run online A0, A1, A2 experiment variants.
7. Save scores, predictions, logs, and metrics in reproducible files.
8. Provide a visual demo for offline replay and online streaming replay.
```

### 1.2 Non-goals

The current implementation should not attempt to:

```text
1. Update source encoder online.
2. Update reconstruction fusion head online.
3. Update reconstruction prediction head online.
4. Update source memories online.
5. Update classification path online.
6. Add a new lambda for online contrastive loss.
7. Add complex UI architecture such as a custom React frontend unless strictly necessary.
```

Recommended UI stack for the first demo:

```text
Streamlit + PyTorch + matplotlib/plotly
```

---

## 2. Experiment Matrix

### 2.1 Offline variants

#### O0: two-stage base

Stage A loss:

```text
L_StageA =
    lambda_recon       * L_recon
  + lambda_cls         * L_cls
  + lambda_contrastive * L_contrastive
```

Stage B loss:

```text
L_StageB = L_recon + L_cls
```

Stage B freezes:

```text
shared encoder
continuous prototype bank
discrete codebook
```

Stage B trains:

```text
reconstruction fusion head
classification fusion head
reconstruction prediction head
classification prediction head
```

#### O1: two-stage + Point-wise Balanced Reconstruction-Score Loss

Stage A still uses the three required losses:

```text
L_recon
L_cls
L_contrastive
```

In addition, Stage A enables:

```text
Point-wise Balanced Reconstruction-Score Loss
```

Config name:

```yaml
enable_score_loss: true
score_loss_type: pointwise_balanced_bce_logits
score_loss_granularity: point
score_loss_target: synthetic_anomaly_mask
score_loss_reduction: pointwise_binary_balanced_mean
score_loss_normalization: train_batch_normal_tokens_detached_mean_std
```

The loss is merged into the classification-side objective:

```text
L_cls_score = (L_cls + L_score_point) / 2
```

Then:

```text
L_StageA =
    lambda_recon       * L_recon
  + lambda_cls         * L_cls_score
  + lambda_contrastive * L_contrastive
```

Do not add `lambda_score` in the main version.

Stage B remains the same as O0. Do not use the point-score BCE loss in Stage B by default.

### 2.2 Online variants

#### A0: no online TTA

```text
No online parameter update.
Use source model only.
Still use sliding-window point-score EWMA for online prediction.
```

#### A1: PNN reconstruction-only

```text
Gray-zone windows enter verification buffer.
Recurrent pseudo-new-normality tokens are verified.
Only MLP projector g_psi is updated.
Loss = masked PNN reconstruction loss.
No hard-old adaptation.
No online contrastive loss.
```

#### A2: full online TTA

```text
Gray-zone PNN adaptation:
    masked reconstruction
    + online contrastive regularizer

Hard-old-normality adaptation:
    hinge-to-B_window reconstruction
    + online contrastive regularizer
    + non-overlap guard

Only MLP projector g_psi is updated.
Anomalous discrete codewords are always used as contrastive negatives.
Known-anomaly tokens from buffer are optional extra negatives if available.
TTL buffer keeps unresolved windows temporarily.
```

### 2.3 Minimum run plan under limited time

Required:

```text
O0-A0
O0-A2
O1-A0
O1-A2
```

Optional if time remains:

```text
O0-A1
O1-A1
```

This is enough to answer two main questions:

```text
1. Does the point-wise score loss improve the source model?
2. Does full online TTA improve online stream prediction over no TTA?
```

---

## 3. Codebase Organization Plan

Add or modify modules using the smallest possible surface area.

```text
configs/
    experiment/
        offline_o0_two_stage_base.yaml
        offline_o1_two_stage_point_score.yaml
        online_a0_no_tta.yaml
        online_a1_pnn_recon_only.yaml
        online_a2_full_tta.yaml
        demo_offline_replay.yaml
        demo_online_stream.yaml

training/
    stage_a_trainer.py
    stage_b_trainer.py
    losses/
        reconstruction_loss.py
        classification_loss.py
        contrastive_loss.py
        pointwise_score_loss.py

memory/
    initialize_continuous_prototypes.py
    initialize_discrete_codebook.py
    compute_discrete_anom_radius.py

thresholds/
    calibrate_offline_point_threshold.py
    calibrate_online_ewma_threshold.py
    calibrate_window_thresholds.py
    threshold_artifact.py

online_tta/
    online_engine.py
    online_projector.py
    triage.py
    verification_buffer.py
    ttl_buffer.py
    online_losses.py
    online_optimizer.py
    non_overlap_guard.py

evaluation/
    offline_predict.py
    online_stream_predict.py
    metrics_export.py
    score_export.py

demo/
    app_streamlit.py
    stream_simulator.py
    queue_runner.py
    demo_state.py
    plotting.py
```

Existing files can be reused. The names above are recommended logical boundaries, not mandatory exact filenames.

---

## 4. Configuration Contract

### 4.1 Shared config

```yaml
model_name: thesis_multitask
window_size: 20
input_dim: 38
hidden_dim: 32
num_classes: 12

lambda_recon: 0.5
lambda_cls: 0.5
lambda_contrastive: 0.3
contrastive_temperature: 0.1

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
discrete_anom_radius_quantile: 0.99
discrete_anom_radius_distance: cosine
```

### 4.2 Offline O0 config

```yaml
experiment_id: O0_two_stage_base
stage_a_enabled: true
stage_b_enabled: true

enable_score_loss: false

stage_a_losses:
  - reconstruction
  - classification
  - contrastive

stage_b_losses:
  - reconstruction
  - classification

memory_initialization_after_stage_a: true
memory_initialization_source: train_only
freeze_memories_after_initialization: true
```

### 4.3 Offline O1 config

```yaml
experiment_id: O1_two_stage_point_score
stage_a_enabled: true
stage_b_enabled: true

enable_score_loss: true
score_loss_type: pointwise_balanced_bce_logits
score_loss_granularity: point
score_loss_target: synthetic_anomaly_mask
score_loss_reduction: pointwise_binary_balanced_mean
score_loss_normalization: train_batch_normal_tokens_detached_mean_std

stage_a_losses:
  - reconstruction
  - classification
  - contrastive
  - pointwise_balanced_reconstruction_score

stage_b_losses:
  - reconstruction
  - classification
```

### 4.4 Online shared config

```yaml
online_window_mode: sliding
online_point_score_aggregation: ewma
online_score_current_weight: 0.9
online_score_previous_weight: 0.1

online_point_threshold_source: clean_validation_sliding_ewma
window_input_threshold_source: clean_validation
window_latent_threshold_source: clean_validation

window_input_threshold_quantile: 0.99
window_latent_low_quantile: 0.95
window_latent_high_quantile: 0.99

source_model_frozen: true
source_memory_frozen: true
online_update_targets:
  - online_mlp_projector

online_projector_enabled: true
online_projector_init: approximate_identity

online_adaptation_steps: 1
online_optimizer: adamw
online_adaptation_lr: 0.0001
online_adaptation_weight_decay: 0.0001
online_lr_scheduler: none
online_gradient_clip: true
online_gradient_clip_norm: 0.5
online_optimizer_state: reset_per_adaptation_event
```

### 4.5 Online A0 config

```yaml
online_variant: A0_no_online_tta
online_tta_enabled: false
online_update_enabled: false
```

### 4.6 Online A1 config

```yaml
online_variant: A1_pnn_reconstruction_only
online_tta_enabled: true
online_update_enabled: true

hard_old_adaptation_enabled: false
pnn_adaptation_enabled: true
online_reconstruction_adaptation_enabled: true
online_contrastive_adaptation_enabled: false

verification_buffer_enabled: true
verification_buffer_size: 8
verification_buffer_non_overlap: true
buffer_admission_rule: input_window_high_and_latent_gray_zone

buffer_reset_policy: keep_unresolved_with_ttl
buffer_entry_total_ttl: 3
buffer_entry_ttl_after_admission: 2
buffer_ttl_decrement_event: verification_cycle
```

### 4.7 Online A2 config

```yaml
online_variant: A2_full_online_tta
online_tta_enabled: true
online_update_enabled: true

hard_old_adaptation_enabled: true
hard_old_non_overlap_guard: true
hard_old_loss_type: hinge_to_B_window

pnn_adaptation_enabled: true
online_reconstruction_adaptation_enabled: true
online_contrastive_adaptation_enabled: true
online_contrastive_type: source_consistency_multi_positive_infonce
online_contrastive_temperature: 0.1
online_contrastive_weight: reuse_lambda_contrastive

online_contrastive_anchor: projected_online_token
online_contrastive_positive:
  - same_token_frozen_source_latent
  - same_recurrent_signature_projected_pnn_tokens
online_contrastive_negative:
  - anomalous_discrete_codewords
  - known_anomaly_projected_tokens_if_available
  - known_anomaly_source_latents_if_available
online_contrastive_ignore:
  - non_pnn_non_known_anomaly_tokens

verification_buffer_enabled: true
verification_buffer_size: 8
verification_buffer_non_overlap: true
buffer_admission_rule: input_window_high_and_latent_gray_zone

buffer_reset_policy: keep_unresolved_with_ttl
buffer_entry_total_ttl: 3
buffer_entry_ttl_after_admission: 2
buffer_ttl_decrement_event: verification_cycle
```

---

## 5. Offline Training Pipeline

### 5.1 Stage A inputs

Each batch should provide:

```python
batch = {
    "x_clean": Tensor[B, L, C],
    "x_input": Tensor[B, L, C],
    "class_labels": LongTensor[B],
    "synthetic_anomaly_mask": BoolTensor[B, L] or BoolTensor[B, L, C],
    "is_synthetic": BoolTensor[B],
}
```

Do not infer point-level anomaly labels from window-level labels. Only injected positions are anomalous point labels for the point-wise score loss.

### 5.2 Stage A forward pass

```python
outputs = model.forward_stage_a(batch["x_input"])

z = outputs["hidden"]                       # [B, L, d_model]
x_hat = outputs["reconstruction"]           # [B, L, C]
logits = outputs["classification_logits"]   # [B, 12]
contrastive = outputs.get("contrastive")
```

### 5.3 Reconstruction loss

For synthetic windows, reconstruct `x_clean`, not `x_input`.

```python
clean_mask = 1 - point_anomaly_mask  # [B, L]
m = clean_mask.float()[:, :, None]   # [B, L, 1]

loss_recon = ((x_hat - x_clean) ** 2 * m).sum()
loss_recon = loss_recon / (m.sum() * C + eps)
```

Validation/test anomaly scores are different:

```text
validation/test score = MSE(x_hat, x_input)
```

### 5.4 Classification loss

```python
loss_cls = cross_entropy(logits, class_labels)
```

The current main pipeline uses window-level 12-class classification.

### 5.5 Contrastive loss

Offline contrastive loss remains the existing training-only two-view contrastive objective.

Safety rule:

```text
No validation window, test window, or online stream window may enter offline contrastive positives or negatives.
```

### 5.6 Point-wise Balanced Reconstruction-Score Loss

Enabled only for O1.

Point reconstruction score:

```python
r = ((x_hat - x_input) ** 2).mean(dim=-1)  # [B, L]
```

Point anomaly labels:

```python
if synthetic_anomaly_mask.ndim == 3:
    a = synthetic_anomaly_mask.any(dim=-1)  # [B, L]
else:
    a = synthetic_anomaly_mask              # [B, L]
```

Batch-normal-token statistics:

```python
normal_scores = r[a == 0]
mu = normal_scores.mean().detach()
std = normal_scores.std(unbiased=False).detach()
z_score = (r - mu) / (std + eps)
```

Balanced BCEWithLogits:

```python
loss_normal = bce_with_logits(z_score[a == 0], zeros_like(...)).mean()
loss_anom = bce_with_logits(z_score[a == 1], ones_like(...)).mean()
loss_score_point = 0.5 * loss_normal + 0.5 * loss_anom
```

If either token group is empty:

```text
skip L_score_point for this batch
log train/score_loss_skipped_batches
```

### 5.7 Stage A total loss

O0:

```python
loss_total = (
    lambda_recon * loss_recon
    + lambda_cls * loss_cls
    + lambda_contrastive * loss_contrastive
)
```

O1:

```python
loss_cls_score = 0.5 * (loss_cls + loss_score_point)

loss_total = (
    lambda_recon * loss_recon
    + lambda_cls * loss_cls_score
    + lambda_contrastive * loss_contrastive
)
```

---

## 6. Memory Initialization

Memory initialization runs after Stage A.

Required mode:

```python
model.eval()
with torch.no_grad():
    collect_train_latents()
```

Allowed data:

```text
train split only
```

Forbidden data:

```text
validation windows
test windows
online stream windows
```

### 6.1 Continuous prototype bank

Collect:

```text
clean train latent tokens
clean positions from synthetic train windows
```

Exclude:

```text
injected anomaly tokens
```

Run k-means:

```text
K_c = 32
P_c shape = [32, d_model]
```

Freeze after initialization.

### 6.2 Discrete codebook

Per class:

```text
class 0:
    clean / normal train latent tokens

class 1..11:
    injected anomaly train latent tokens from that synthetic class
```

Run class-wise k-means:

```text
5 codewords per class
12 classes
E_d shape = [60, d_model]
```

Index layout:

```text
class 0:  0..4
class 1:  5..9
...
class 11: 55..59
```

Freeze after initialization.

### 6.3 Anomalous discrete cluster radius

For each anomalous codeword:

```text
R_anom[k] = q99 distance of train anomaly tokens assigned to codeword k
```

Use the same distance family as the discrete codebook query:

```text
cosine distance if embeddings/codewords are L2-normalized
```

---

## 7. Stage B Pipeline

Stage B freezes:

```text
encoder
continuous prototype bank
discrete codebook
```

Stage B trains:

```text
reconstruction fusion head
classification fusion head
reconstruction prediction head
classification prediction head
```

Forward:

```python
z = frozen_encoder(x_input)
z_cont = query_continuous_memory(z)
z_disc = query_discrete_codebook(z)
z_cat = torch.cat([z_cont, z_disc], dim=-1)

x_hat = reconstruction_path(z_cat)
logits = classification_path(z_cat)
```

Loss:

```python
loss_stage_b = loss_recon + loss_cls
```

No point-wise score BCE loss in Stage B by default.

---

## 8. Threshold Calibration

Threshold calibration is no-gradient only.

```python
model.eval()
with torch.no_grad():
    calibrate_thresholds()
```

### 8.1 Offline point threshold

Use clean validation non-overlapping windows.

```text
T_point_nonoverlap = Q_0.99(point_scores_on_clean_validation_nonoverlap)
```

Used for:

```text
offline test point-level prediction
```

### 8.2 Online EWMA point threshold

Simulate online sliding windows on clean validation.

```text
1. Generate sliding windows from clean validation sequence.
2. Forward each window.
3. Compute point reconstruction scores.
4. Aggregate repeated absolute time-point scores using EWMA.
5. Take q99 over final clean-validation EWMA point scores.
```

EWMA:

```python
S_new[t] = 0.9 * current_score[t] + 0.1 * S_old[t]
```

Threshold:

```text
T_point_EWMA = Q_0.99(clean_validation_EWMA_scores)
```

Used for:

```text
online stream point-level prediction
```

### 8.3 Window thresholds

Input-window reconstruction threshold:

```text
B_window = Q_0.99(s_input_window on clean validation)
```

Latent-window threshold band:

```text
A_low  = Q_0.95(s_latent_window on clean validation)
A_high = Q_0.99(s_latent_window on clean validation)
```

These thresholds are constants during online TTA. Do not backpropagate through them.

---

## 9. Online Forward Design

The online forward pass uses one input window, not two augmented views.

For a window `W`:

```text
W -> frozen source encoder -> Z_source
```

Source branch:

```text
Z_source is used as frozen reference latent.
No gradient flows into source encoder.
```

Online branch:

```text
Z_online = Z_source
Z_proj = g_psi(Z_online)
```

Only `g_psi` is trainable.

Then:

```text
Z_proj -> frozen continuous prototype query -> Z_cont
Z_proj -> frozen discrete codebook query      -> Z_disc
concat(Z_cont, Z_disc) -> frozen reconstruction path -> X_hat_online
```

Important consequence:

```text
Online adaptation changes only the projected latent input to frozen memories and frozen heads.
The source encoder and prediction heads do not change.
```

---

## 10. Online Window Triage

For every sliding window `W`, compute:

```text
s_input_window(W)
s_latent_window(W)
```

### 10.1 Normal/easy window

Condition:

```text
s_input_window(W) <= B_window
```

Action:

```text
No adaptation.
Point-level EWMA prediction still runs.
```

### 10.2 Hard-old-normality window

Condition:

```text
s_input_window(W) > B_window
s_latent_window(W) <= A_low
```

Interpretation:

```text
The latent representation is close to old-normal memory,
but reconstruction is still too high.
```

Action in A2:

```text
Adapt g_psi immediately if non-overlap guard accepts W.
```

Action in A0/A1:

```text
No hard-old adaptation.
```

### 10.3 Gray-zone window

Condition:

```text
s_input_window(W) > B_window
A_low < s_latent_window(W) <= A_high
```

Action:

```text
Try to add W to verification buffer if W does not overlap current buffer windows.
```

### 10.4 Strong anomaly candidate

Condition:

```text
s_input_window(W) > B_window
s_latent_window(W) > A_high
```

Action:

```text
No adaptation.
Log only.
```

---

## 11. Hard-Old Adaptation

Hard-old adaptation is enabled only in A2.

### 11.1 Non-overlap guard

Maintain a small list of recently adapted hard-old intervals.

```python
hard_old_guard_intervals = deque(maxlen=hard_old_guard_size)
```

Default:

```yaml
hard_old_guard_size: 1
```

Accept a hard-old window only if it does not overlap any interval in the guard list.

```python
def accept_hard_old(W):
    for old in hard_old_guard_intervals:
        if overlaps(W.interval, old):
            return False
    return True
```

After adaptation:

```python
hard_old_guard_intervals.append(W.interval)
```

### 11.2 Reconstruction loss

Goal:

```text
Make online model reconstruction MSE smaller than B_window.
```

Loss:

```text
L_hard_recon = [s_input_window_online(W) - B_window]_+^2
```

Pseudocode:

```python
score = ((x_hat_online - W.x) ** 2).mean()
loss_hard_recon = torch.relu(score - B_window).pow(2)
```

### 11.3 Contrastive regularizer for hard-old

For every token in the hard-old window:

```text
anchor = projected online token q_{t}
positive = same-token frozen source latent z_source_t
negatives = all anomalous discrete codewords
```

The source latent and codewords are detached.

```python
q = normalize(g_psi(z_source))            # [L, d]
k_pos = normalize(z_source).detach()      # [L, d]
k_neg = normalize(anom_codewords).detach() # [K_anom, d]
```

This online contrastive term is a regularizer. It allows reconstruction-driven adaptation, but discourages the MLP projector from moving the online latent representation too far from the frozen source geometry.

---

## 12. Verification Buffer for PNN

PNN adaptation is enabled in A1 and A2.

### 12.1 Buffer admission

A window enters the buffer if:

```text
s_input_window(W) > B_window
A_low < s_latent_window(W) <= A_high
W does not overlap any current buffer window
```

Each buffer entry stores:

```python
entry = {
    "window": W,
    "start": start,
    "end": end,
    "ttl_remaining": 2,
    "status": "unresolved",
    "was_adapted": False,
}
```

Rationale:

```text
Total TTL is 3 verification chances.
The admission itself counts as the first chance.
Therefore ttl_remaining is initialized as 2.
```

### 12.2 Verification trigger

Trigger verification when:

```text
buffer_size >= verification_buffer_size
and at least one new entry has been admitted since the last verification cycle
```

This avoids verifying the same unresolved buffer repeatedly without new information.

Default:

```yaml
verification_buffer_size: 8
```

### 12.3 Discrete anomalous cluster filter

For each latent point in the buffer:

```text
1. Find nearest discrete codeword.
2. If nearest codeword is anomalous and distance is within its anomalous radius, mark as known-anom.
3. Otherwise keep for continuous signature verification.
```

Output mask:

```text
M_known_anom[i,t] = 1 if token is inside a known anomalous discrete cluster
```

Known-anom tokens are not used as PNN tokens.

### 12.4 Continuous top-k signature

For each remaining token:

```text
signature = ordered top-3 nearest continuous prototype ids
```

Default:

```yaml
continuous_signature_topk: 3
continuous_signature_ordered: true
```

A signature is recurrent if it appears in more than one non-overlapping window.

```python
if len(signature_to_windows[signature]) > 1:
    recurrent_signatures.add(signature)
```

PNN token mask:

```text
M_pnn[i,t] = 1 if token has recurrent continuous signature
```

---

## 13. PNN Adaptation

### 13.1 Masked reconstruction loss

Used in A1 and A2.

```python
m = M_pnn.float()[:, :, None]  # [B_adapt, L, 1]
loss_pnn_recon = ((x_hat_online - x_adapt) ** 2 * m).sum()
loss_pnn_recon = loss_pnn_recon / (m.sum() * C + eps)
```

Only PNN tokens contribute to the reconstruction loss.

### 13.2 Contrastive regularizer for PNN

Used only in A2.

For each PNN token:

```text
anchor:
    projected online token

positives:
    same-token frozen source latent
    projected PNN tokens with the same recurrent signature

negatives:
    anomalous discrete codewords
    known-anom projected tokens if available
    known-anom source latents if available

ignored:
    non-PNN, non-known-anom tokens
```

Recommended implementation detail:

```text
When projected tokens are used as positives or negatives for another anchor,
use detached key copies.
Each projected token still receives gradient when it is used as its own anchor.
```

Pseudocode:

```python
q_all = normalize(g_psi(z_source))          # [B, L, d]
z_src_key = normalize(z_source).detach()    # [B, L, d]
code_neg = normalize(anom_codewords).detach()

loss_terms = []
for token in pnn_tokens:
    q = q_all[token]

    positives = []
    positives.append(z_src_key[token])

    for other in same_signature_pnn_tokens[token.signature]:
        if other != token:
            positives.append(q_all[other].detach())

    negatives = [code_neg]

    if known_anom_tokens_exist:
        negatives.append(q_all[M_known_anom].detach())
        negatives.append(z_src_key[M_known_anom])

    loss_terms.append(multi_positive_infonce(q, positives, negatives, tau))

loss_online_contrastive = mean(loss_terms)
```

If a PNN token has no same-signature projected positive besides itself, the same-token frozen source latent is still a valid positive.

---

## 14. Online Total Loss and Optimizer

### 14.1 A1 loss

```text
L_online = L_pnn_recon
```

### 14.2 A2 hard-old loss

```text
L_online = L_hard_recon + lambda_contrastive * L_online_contrastive
```

### 14.3 A2 PNN loss

```text
L_online = L_pnn_recon + lambda_contrastive * L_online_contrastive
```

### 14.4 Trainable parameter filter

Before every online update:

```python
for p in model.parameters():
    p.requires_grad_(False)

for p in model.online_mlp_projector.parameters():
    p.requires_grad_(True)
```

Assert:

```python
trainable = [name for name, p in model.named_parameters() if p.requires_grad]
assert all(name.startswith("online_mlp_projector") for name in trainable)
```

### 14.5 Optimizer

Default optimizer:

```yaml
online_optimizer: adamw
online_adaptation_lr: 0.0001
online_adaptation_weight_decay: 0.0001
online_adaptation_steps: 1
online_lr_scheduler: none
online_gradient_clip: true
online_gradient_clip_norm: 0.5
online_optimizer_state: reset_per_adaptation_event
```

Implementation rule:

```text
Create a fresh AdamW optimizer for each adaptation event.
Only pass MLP projector parameters.
Exclude bias and normalization parameters from weight decay if the helper already supports parameter groups.
```

Pseudocode:

```python
params = build_weight_decay_param_groups(
    model.online_mlp_projector,
    weight_decay=1e-4,
)

optimizer = torch.optim.AdamW(params, lr=1e-4)

optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.online_mlp_projector.parameters(), 0.5)
optimizer.step()
```

---

## 15. TTL Buffer Reset Policy

After every verification cycle:

```python
for entry in buffer.entries:
    if entry.was_adapted:
        buffer.remove(entry)
    else:
        entry.ttl_remaining -= 1
        if entry.ttl_remaining <= 0:
            buffer.remove(entry)
```

Meaning:

```text
Admission counts as chance 1.
If not adapted after first verification, ttl_remaining becomes 1.
If not adapted after second later verification, ttl_remaining becomes 0.
Then the window is removed permanently.
```

Do not decrement TTL on every stream step. Decrement only on verification cycles.

Do not keep adapted windows.

Do keep unresolved windows until TTL expires.

---

## 16. Online Stream Prediction Loop

### 16.1 Internal evaluation loop

Internal evaluation should be optimized for speed and reproducibility.

```python
for tau, x_t in enumerate(test_stream):
    stream_state.append(x_t)

    if stream_state.num_points < L:
        continue

    W = stream_state.latest_window()

    outputs = online_engine.forward_window(W)

    update_ewma_point_scores(
        point_scores=outputs.point_scores,
        absolute_indices=W.absolute_indices,
    )

    finalize_expired_points_if_needed(tau)

    if online_variant == "A0":
        continue

    triage = classify_window(outputs, thresholds)

    if triage == "hard_old" and config.hard_old_adaptation_enabled:
        if hard_old_guard.accept(W):
            online_engine.adapt_hard_old(W)
            hard_old_guard.add(W)

    elif triage == "gray_zone":
        ttl_buffer.try_add(W)

    if ttl_buffer.should_verify():
        verification = verify_buffer(ttl_buffer)

        if verification.has_pnn and config.pnn_adaptation_enabled:
            online_engine.adapt_pnn(verification)

        ttl_buffer.update_after_verification(verification)
```

### 16.2 Point-level prediction

For each absolute time point:

```python
y_pred[t] = 1 if ewma_score[t] > T_point_EWMA else 0
```

The final anomaly decision remains point-level.

The buffer controls adaptation only. It does not directly assign final anomaly labels.

### 16.3 Retroactive update rule

Default:

```text
Do not retroactively rewrite finalized predictions after online adaptation.
Online adaptation affects future windows only.
```

For demo visualization, provisional labels for recent points may visually update while they are still inside overlapping windows. Finalized labels should be visually distinguished if implemented.

---

## 17. Offline Prediction Export

After offline training and threshold calibration, export:

```text
outputs/{run_id}/offline_test_scores.npz
outputs/{run_id}/offline_test_predictions.csv
outputs/{run_id}/offline_test_metrics.json
outputs/{run_id}/thresholds.json
outputs/{run_id}/model_summary.json
```

Recommended `.npz` fields:

```python
{
    "point_scores": array[T],
    "point_predictions": array[T],
    "point_labels": array[T] or None,
    "threshold": float,
    "timestamps": array[T],
    "selected_channel_values": array[T],
}
```

Offline test evaluation should use non-overlapping windows unless explicitly running the online stream evaluation.

---

## 18. Demo Software

The demo is separate from internal evaluation.

```text
Internal evaluation:
    fast, deterministic, metric-focused

Demo:
    visual, friendly, stream-like, slower is acceptable
```

The demo must not be used as the official metric pipeline.

### 18.1 Demo modes

#### Mode 1: offline test-set replay

Purpose:

```text
Show what the trained source model predicts on a test sequence after offline training.
```

Inputs:

```text
checkpoint
thresholds.json
offline_test_scores.npz or raw test sequence
selected entity / machine id
selected channel id
```

UI elements:

```text
raw signal timeline for one selected channel
anomaly score timeline
threshold line
point-level anomaly markers
play / pause
speed slider
time index display
```

#### Mode 2: online stream replay

Purpose:

```text
Show a simulated real-time stream where points enter a queue,
the model reads them continuously,
forms sliding windows,
updates scores,
and marks anomalies on the timeline.
```

Architecture:

```text
Test sequence / CSV
    -> StreamProducer
    -> Queue
    -> OnlineModelConsumer
    -> DemoState
    -> LivePlot
```

### 18.2 Queue-based stream design

Use standard Python queue for first version:

```python
from queue import Queue, Empty

stream_queue = Queue(maxsize=queue_size)
```

Producer pseudocode:

```python
def producer(sequence, stream_queue, speed):
    for t, x_t in enumerate(sequence):
        stream_queue.put({"t": t, "x": x_t})
        sleep(speed)
```

Consumer pseudocode:

```python
def consumer(stream_queue, model, demo_state):
    while demo_state.running:
        try:
            item = stream_queue.get(timeout=0.1)
        except Empty:
            continue

        t = item["t"]
        x_t = item["x"]
        demo_state.append_point(t, x_t)

        if demo_state.num_points < L:
            demo_state.update_status("waiting_for_full_window")
            continue

        W = demo_state.latest_window()
        outputs = model.forward_window(W)

        demo_state.update_scores(outputs.point_scores, W.absolute_indices)
        demo_state.update_predictions(threshold=T_point_EWMA)

        if demo_state.tta_enabled:
            online_tta_step_if_needed(W, outputs)

        demo_state.update_status("processed")
```

### 18.3 Visualization design

For multivariate data, do not plot all channels by default.

Default display:

```text
1 selected raw channel
1 anomaly score timeline
1 threshold line
anomaly markers on exact point indices
current sliding window boundary
```

Optional side panel:

```text
current index
current queue size
current window start/end
latest score
threshold
latest decision: normal/anomaly
TTA mode: off/A1/A2
verification buffer size
number of hard-old adaptations
number of PNN adaptations
```

Recommended visual semantics:

```text
normal point: small neutral marker
anomaly point: highlighted marker on exact time index
current window: translucent vertical band
threshold: horizontal dashed line
```

For high-school-friendly demo text, avoid showing too many equations. Use simple labels:

```text
Raw data
Model anomaly score
Alarm threshold
Detected anomaly
Current window being processed
```

### 18.4 Demo safety boundary

Demo may show ground-truth labels only as an optional overlay after prediction, for explanation.

It must not use labels to:

```text
choose thresholds
change model parameters
select online update timing
tune demo model behavior
```

---

## 19. Logging Requirements

### 19.1 Offline logs

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

memory/continuous_pool_size
memory/discrete_pool_size_class_0
...
memory/discrete_pool_size_class_11
memory/continuous_shape
memory/discrete_shape
memory/discrete_anom_radius_mean
memory/discrete_anom_radius_min
memory/discrete_anom_radius_max

stage_b/loss_total
stage_b/loss_recon
stage_b/loss_cls
stage_b/encoder_grad_norm
stage_b/continuous_memory_grad_norm
stage_b/discrete_memory_grad_norm
```

Expected frozen gradients in Stage B:

```text
encoder_grad_norm = 0
continuous_memory_grad_norm = 0
discrete_memory_grad_norm = 0
```

### 19.2 Online logs

```text
online/point_score_current
online/point_score_ewma
online/num_finalized_points
online/num_predicted_anomaly_points

online/input_window_score
online/latent_window_score
online/num_hard_old_normality_windows
online/num_gray_zone_windows
online/num_strong_anomaly_windows
online/num_buffer_admitted_windows
online/num_buffer_rejected_overlap_windows

online/buffer_size
online/num_latent_points_total
online/num_points_removed_by_discrete_anom_filter
online/num_points_remaining_for_signature
online/num_unique_signatures
online/num_recurrent_signatures
online/num_pseudo_new_normality_points
online/num_windows_to_be_adapted
online/num_unresolved_windows_kept
online/num_unresolved_windows_removed_by_ttl

online/loss_hard_recon
online/loss_pnn_recon
online/loss_contrastive
online/loss_total
online/adaptation_steps
online/projector_grad_norm
online/source_encoder_grad_norm
online/source_memory_grad_norm
online/recon_head_grad_norm
online/classification_head_grad_norm
```

Expected online frozen gradients:

```text
source_encoder_grad_norm = 0
source_memory_grad_norm = 0
recon_head_grad_norm = 0
classification_head_grad_norm = 0
```

---

## 20. Unit Tests

### 20.1 Offline tests

```text
[ ] Stage A runs with enable_score_loss=false.
[ ] Stage A runs with enable_score_loss=true.
[ ] point-wise score shape is [B, L].
[ ] point-wise anomaly label shape is [B, L].
[ ] synthetic anomalous window does not mark all positions as anomalous.
[ ] L_score_point uses balanced normal/anomaly reduction.
[ ] L_score_point detaches normal-token mean/std.
[ ] L_score_point skips safely if one token group is empty.
[ ] continuous prototype bank shape is [32, d_model].
[ ] discrete codebook shape is [60, d_model].
[ ] discrete codebook has 5 codewords per class.
[ ] anomalous radii are computed from train split only.
[ ] Stage B freezes encoder and memories.
```

### 20.2 Online tests

```text
[ ] A0 performs no optimizer step.
[ ] A1 updates only online_mlp_projector.
[ ] A2 updates only online_mlp_projector.
[ ] Hard-old non-overlap guard rejects overlapping hard-old windows.
[ ] Hard-old loss is zero when score <= B_window.
[ ] EWMA uses 0.9 current + 0.1 old.
[ ] Buffer admits only gray-zone windows.
[ ] Buffer rejects overlapping gray-zone windows.
[ ] Buffer entry starts with ttl_remaining = 2.
[ ] TTL decrements only on verification cycle.
[ ] Adapted windows are removed.
[ ] Unresolved windows are kept until TTL reaches zero.
[ ] Anomalous codewords are always present in online contrastive negatives.
[ ] Contrastive loss still works when no known-anom buffer token exists.
[ ] Non-PNN non-known-anom tokens are ignored, not treated as negatives.
[ ] Online optimizer receives only MLP projector parameters.
[ ] Source encoder gradients remain zero during online TTA.
[ ] Source memory gradients remain zero during online TTA.
[ ] Reconstruction head gradients remain zero during online TTA.
```

### 20.3 Demo tests

```text
[ ] Offline replay loads checkpoint and thresholds.
[ ] Offline replay displays raw signal, score, threshold, and anomaly markers.
[ ] Stream producer pushes one point at a time into queue.
[ ] Consumer waits safely if queue is empty.
[ ] Consumer does not forward before L points are available.
[ ] Consumer forwards when a complete window exists.
[ ] Demo state updates point scores and predictions.
[ ] Pause/resume works.
[ ] Speed slider changes producer delay.
[ ] Demo can run without labels.
[ ] Labels, if shown, are overlay-only and not used by model.
```

---

## 21. Data Leakage Safety Checklist

Implementation is invalid if any of the following occurs:

```text
[ ] validation/test windows enter Stage A training loss
[ ] validation/test windows enter offline contrastive queue
[ ] validation/test windows enter memory initialization
[ ] validation/test windows enter anomalous radius fitting
[ ] validation/test labels tune thresholds
[ ] test labels tune online TTA hyperparameters
[ ] online future windows are used before they arrive
[ ] source encoder is updated online
[ ] source memories are updated online
[ ] reconstruction heads are updated online in the current simplified version
[ ] classification path is updated online
```

Allowed:

```text
[ ] clean validation calibrates thresholds with no gradient
[ ] validation selects checkpoints with no gradient
[ ] test is used for final evaluation only
[ ] online test stream is used for causal unsupervised TTA
[ ] demo uses labels only as optional overlay after predictions
```

---

## 22. Acceptance Criteria

### 22.1 Experiment acceptance

```text
[ ] O0 training completes.
[ ] O1 training completes.
[ ] O0 threshold calibration completes.
[ ] O1 threshold calibration completes.
[ ] O0-A0 online evaluation completes.
[ ] O0-A2 online evaluation completes.
[ ] O1-A0 online evaluation completes.
[ ] O1-A2 online evaluation completes.
[ ] Optional A1 runs if time remains.
[ ] All runs export scores, predictions, metrics, and config snapshots.
```

### 22.2 Online TTA acceptance

```text
[ ] Only online MLP projector changes after TTA.
[ ] Source encoder checksum is unchanged after TTA.
[ ] Source memory checksum is unchanged after TTA.
[ ] Reconstruction head checksum is unchanged after TTA.
[ ] Classification path checksum is unchanged after TTA.
[ ] Online optimizer is AdamW with lr=1e-4 and weight_decay=1e-4.
[ ] Online adaptation uses one step per event.
[ ] No online scheduler is used.
```

### 22.3 Demo acceptance

```text
[ ] Demo can load one finished run folder.
[ ] Demo can replay offline test predictions.
[ ] Demo can simulate online stream from queue.
[ ] Demo marks anomaly at exact point index.
[ ] Demo remains understandable with one selected channel and one score plot.
[ ] Demo does not affect official experiment outputs.
```

---

## 23. Recommended Implementation Order

### Phase 1: experiments first

```text
1. Add/verify O0 and O1 configs.
2. Verify point-wise score loss switch.
3. Verify memory initialization artifacts.
4. Add threshold artifact export.
5. Add A0 online evaluator.
6. Add online MLP projector-only update mechanism.
7. Add hard-old adaptation.
8. Add PNN buffer verification and TTL.
9. Add online contrastive regularizer.
10. Run O0-A0, O0-A2, O1-A0, O1-A2.
```

### Phase 2: demo after core outputs exist

```text
1. Implement offline replay from exported .npz.
2. Implement queue producer.
3. Implement online consumer using the same online engine.
4. Add live plotting.
5. Add pause/resume and speed slider.
6. Add optional labels overlay.
```

This order prevents the demo from delaying the thesis experiment pipeline.

---

## 24. Final Locked Decisions

```text
[LOCKED] Offline has exactly two main variants: O0 and O1.
[LOCKED] O1 uses Point-wise Balanced Reconstruction-Score Loss.
[LOCKED] Online has A0, A1, A2, but minimum required run is A0 and A2.
[LOCKED] Online TTA updates only light-weight MLP projector.
[LOCKED] No online update to encoder, memories, reconstruction heads, or classification path.
[LOCKED] No two augmented views are needed for online TTA.
[LOCKED] Online contrastive acts as source-consistency regularizer.
[LOCKED] Anomalous discrete codewords are always negatives.
[LOCKED] Known-anom buffer tokens are optional extra negatives.
[LOCKED] Non-PNN non-known-anom tokens are ignored, not negatives.
[LOCKED] Hard-old adaptation uses non-overlap guard.
[LOCKED] Hard-old objective pushes online reconstruction MSE below B_window.
[LOCKED] Online optimizer is AdamW, lr=1e-4, weight_decay=1e-4, one step, no scheduler.
[LOCKED] TTL buffer keeps unresolved windows for two verification cycles after admission.
[LOCKED] Demo is separate from official evaluation.
```
