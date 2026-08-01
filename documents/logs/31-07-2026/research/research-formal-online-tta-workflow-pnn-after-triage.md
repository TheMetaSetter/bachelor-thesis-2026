---
date: 2026-07-31 15:30:00 +07:00
researcher: OpenAI Codex
topic: "Đặc tả hình thức workflow online TTA theo ý tưởng triage trước PNN"
status: complete
revision: 52e518e0b175a1ce6891e27a501322f91c9b0978
branch: dev
---

# Research: Đặc tả hình thức workflow online TTA theo ý tưởng triage trước PNN

## Summary

Workflow anh mô tả có thể viết hình thức như sau:

text
score window
  -> triage window
  -> immediate hard-old projector update OR gray-zone admission
  -> verification when buffer trigger is satisfied
  -> point classification inside admitted windows
  -> PNN-only masked loss
  -> projector-only update

Tên canonical dùng trong workflow này là:

text
normal
hard_old_normality
gray_zone
strong_anomaly
pnn_verified
verification_buffer / VerificationBuffer
online_mlp_projector
pnn_mask

Code và hai spec không có trạng thái strong_normal; trạng thái hiện tại là strong_anomaly. Báo cáo dùng strong_anomaly để bám theo code, nhưng không tự kết luận rằng hai cụm từ này có cùng nghĩa.

## Research question

Viết lại formal workflow online TTA theo ý tưởng: triage window trước; hard-old adaptation ngay; gray-zone admission tiếp theo; khi verification buffer đủ điều kiện thì phân loại point-level, chỉ tạo PNN mask, tính loss trên PNN points và chỉ cập nhật MLP projector.

## System context and terminology

| Vai trò | Tên hiện tại |
| --- | --- |
| Window input score | input_window_score |
| Window latent score | latent_window_score |
| EWMA score | ewma_point_score |
| Window decision | triage_decision |
| Buffer | verification_buffer / VerificationBuffer |
| Signature state | signature_history |
| PNN selection | pnn_mask |
| Frozen source hidden | reference_hidden |
| Online trainable module | online_mlp_projector |
| Post-verification decision | pnn_verified |
| Hard-old guard | hard_old_guard |

| Concept | v2 | v3 | Current code | Canonical name |
| --- | --- | --- | --- | --- |
| Low-score region | normal/easy window | normal | normal | normal |
| Old-normal region | hard-old-normality | hard_old | hard_old_normality | hard_old_normality |
| Intermediate region | gray-zone window | gray_zone | gray_zone | gray_zone |
| High-anomaly region | strong anomaly candidate | strong_anomaly | strong_anomaly | strong_anomaly |
| Window store | verification buffer | verification buffer | VerificationBuffer | verification_buffer |
| Trainable object | MLP projector g_psi | online_mlp_projector | online_mlp_projector | online_mlp_projector |
| Verified PNN event | recurrent PNN tokens verified | PNN adaptation path | pnn_verified | pnn_verified |

Evidence: v2 documents/spec/full-spec-v2.md:879-959; v3 documents/spec/full-spec-v3.md:809-940; code src/engine/online_tta/triage.py:17-41.

## Formal workflow

### 1. Score one causal window

At cursor c, let W_c = batch["x"]. The scorer produces:

text
raw_point_score(W_c)
input_window_score(W_c)
latent_window_score(W_c)
ewma_point_score(W_c)

The current scorer computes these values in _score_online_window().

Evidence: src/engine/online_tta/online_engine_window_metrics.py:82-119.

### 2. Classify the window

Let B_window = input_window_threshold, A_low = latent_window_low_threshold, and A_high = latent_window_high_threshold. Define:

text
if input_window_score <= B_window:
    triage_decision = "normal"
elif latent_window_score <= A_low:
    triage_decision = "hard_old_normality"
elif latent_window_score <= A_high:
    triage_decision = "gray_zone"
else:
    triage_decision = "strong_anomaly"

This is the active four-region classifier. The requested target ordering requires this step to finish before PNN point classification.

Evidence: src/engine/online_tta/triage.py:17-41; spec documents/spec/full-spec-v3.md:809-828.

### 3. Branch actions

For normal:

text
no projector update
no verification_buffer admission
no PNN verification
finalize prediction and event record

Evidence: documents/spec/full-spec-v2.md:888-901.

For hard_old_normality in a variant that permits hard-old adaptation:

text
hard_old_guard.accept(W_c.interval)
    -> forward W_c
    -> compute L_hard
    -> add A2 contrastive regularizer if configured
    -> backward
    -> update online_mlp_projector only
    -> hard_old_guard.add(W_c.interval) after success

The hard-old loss is L_hard = ReLU(window_score_online(W_c) - B_window)^2. The current A2 branch uses this decision without requiring pnn_mask.

Evidence: src/engine/online_tta/online_engine_step.py:136-170; guard update src/engine/online_tta/online_engine_window_core.py:227-252; projector allowlist src/engine/online_tta/online_optimizer.py:1-66.

For strong_anomaly:

text
no projector update
no verification_buffer admission
no PNN verification
log/finalize only

Evidence: src/engine/online_tta/online_engine_step.py:136-149; v2 documents/spec/full-spec-v2.md:946-959.

### 4. Admit a gray-zone window

For triage_decision == "gray_zone", attempt:

text
verification_buffer.try_admit(entry(W_c))

The current entry contains entry_id, entity_id, window_start, window_end, window, scores and stream_step. Admission rejects overlapping absolute intervals; adjacent intervals are allowed by v3.

Evidence: src/engine/online_tta/online_engine_window_metrics.py:194-220; v3 documents/spec/full-spec-v3.md:832-851.

### 5. Trigger verification

Let N_buffer = len(verification_buffer). Start a cycle only when:

text
N_buffer >= verification_capacity
AND a new entry was admitted since the previous cycle
AND no cycle is active

The current controller checks capacity and should_verify().

Evidence: src/engine/online_tta/verification_cycle.py:12-36; v3 documents/spec/full-spec-v3.md:853-859.

### 6. Classify points inside admitted entries

Let E = verification_buffer.items(). For every entry, frozen-source encoding produces reference_hidden, nearest codeword IDs/distances and continuous_signature_ids.

Define the conceptual partition:

text
P_known_anomaly
P_pseudo_new_normality
P_other

text
P_known_anomaly
    = points inside an anomalous discrete-codeword radius
P_pseudo_new_normality
    = recurrent continuous-signature points across more than one
      non-overlapping admitted window, excluding P_known_anomaly
P_other
    = all remaining points

Evidence: documents/spec/full-spec-v3.md:861-874.

### 7. Materialize only the PNN mask

The target contract is:

text
pnn_mask[b,t] = True iff (b,t) belongs to P_pseudo_new_normality

No separate adaptation mask is needed for P_known_anomaly or P_other; both have pnn_mask == False.

The current code additionally materializes known_anomaly_mask and computes:

text
pnn_mask = recurrent_signature_mask AND NOT known_anomaly_mask

That is current implementation state, not the requested minimal mask contract. The current A2 contrastive loss also accepts known_anomaly_mask.

Evidence: src/engine/online_tta/signature_verification.py:254-277; src/engine/online_tta/verification_adapter.py:99-113; src/engine/online_tta/online_losses.py:105-168.

### 8. PNN-only forward, loss and update

For an entry with pnn_mask.sum() > 0, forward the window and compute:

text
L_pnn = sum(((recon - x)^2) * pnn_mask)
        / max(1, sum(pnn_mask) * D)

Known-anomaly and other points contribute no PNN reconstruction term.

text
A1: L_total = L_pnn
A2: L_total = L_pnn + lambda_contrastive * L_online_contrastive

Then run backward, gradient clipping and one optimizer step. Only online_mlp_projector may change.

Evidence: v3 documents/spec/full-spec-v3.md:903-940; loss src/engine/online_tta/online_losses.py:57-66; optimizer src/engine/online_tta/online_optimizer.py:1-66.

## Formal state machine

text
S0 receive W_c
  -> S1 compute scores
  -> S2 set triage_decision
S2(normal) -> finalize_without_update
S2(hard_old_normality)
  -> guard accepts: hard_old_projector_update
  -> otherwise: finalize_without_update
S2(strong_anomaly) -> finalize_without_update
S2(gray_zone)
  -> admission succeeds: verification_buffer
  -> otherwise: finalize_without_update
verification_buffer
  -> trigger satisfied: point_classification
point_classification
  -> pnn_mask.sum() == 0: unresolved_entry_lifecycle
  -> otherwise: pnn_verified
pnn_verified
  -> forward admitted windows
  -> PNN-only loss
  -> backward
  -> update online_mlp_projector only

## Current implementation versus requested workflow

| Stage | Requested workflow | Current code |
| --- | --- | --- |
| Score before triage | Yes | Yes |
| Triage before PNN point classification | Yes | No; preliminary PNN path runs in prepare_event before triage |
| Immediate hard-old update | Yes for permitted variant | A2 branch supports it |
| Gray-zone admission after triage | Yes | Yes |
| Point classification only after buffer trigger | Yes | Verification path does this, but preliminary path also runs earlier for A1/A2 |
| Only PNN mask | Yes | Final mask selects PNN, but current code also materializes known_anomaly_mask |
| PNN-only loss | Yes | Masked reconstruction uses pnn_mask; A2 may use known-anomaly values as contrastive negatives |
| Projector-only update | Yes | Enforced by optimizer allowlist |

## Conflicts and uncertainties

1. strong normal has no matching active code/spec name. The active term is strong_anomaly.
2. The requested PNN-only mask contract conflicts with the current known_anomaly_mask field used by verification and A2 contrastive loss.
3. The current implementation computes a preliminary PNN/signature path before triage. This report describes the requested target workflow and does not claim that the current code already follows it.
4. Current buffer fields (window_start, window_end, window) differ from v3 schema names (start_index, end_index, x). This boundary needs explicit terminology approval before implementation.

## Open questions

- Does strong normal mean strong_anomaly, or is it a new state?
- Should A2 contrastive negatives still include known-anomaly latents when only PNN points may be adaptation anchors?
- Should the canonical buffer entry use v3 fields start_index/end_index/x, or preserve current implementation fields?

This report documents the requested formal workflow and terminology conflicts. It does not modify source code, tests, configuration, or specifications.
