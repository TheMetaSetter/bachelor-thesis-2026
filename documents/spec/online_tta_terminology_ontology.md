---
title: "Online TTA Terminology Ontology"
status: authoritative-for-naming
scope: "THESIS online test-time adaptation"
applies_to: "specifications, pseudocode, runtime code, tests, configuration, checkpoints, metrics, and experiment artifacts"
root_ontology: "documents/spec/offline_pretraining_terminology_ontology.md"
evidence_revision: e58602f45ee5439a1e001f060e8ea640aeddde9c
---

# Online TTA Terminology Ontology

## 1. Mục đích và nguyên tắc offline-first

Tài liệu này chuẩn hóa tên object trong `online_tta_phase`. Nó kế thừa object identity từ [`offline_pretraining_terminology_ontology.md`](offline_pretraining_terminology_ontology.md), không tạo tên mới cho cùng offline object.

Nguyên tắc gốc:

```text
offline_pretraining_phase
    produces stage_b_best_checkpoint
    produces threshold_artifact

online_tta_phase
    loads stage_b_best_checkpoint as its reference checkpoint role
    loads the same threshold_artifact
    freezes all inherited offline model state
    may update only online_mlp_projector
```

Vì vậy:

- `reference_checkpoint_path` là field trỏ đến `stage_b_best_checkpoint`; nó không định nghĩa một checkpoint type mới.
- `frozen_source_model` là `offline_source_model` sau khi nạp Stage B checkpoint và freeze.
- `continuous_prototype_bank`, `discrete_codebook`, `anomaly_verification_metadata`, `reconstruction_head` và `classification_head` giữ nguyên tên từ offline ontology.
- `online_point_ewma_threshold` thuộc `threshold_artifact` do offline tạo. Online chỉ đọc, không calibrate lại từ test stream.

## 2. Evidence status và mapping rules

Các status được dùng trong tài liệu này:

- `inherited`: object đã được định nghĩa trong offline ontology;
- `implemented`: source online hiện hành có object hoặc behavior;
- `desired-contract`: object thuộc “Flow người dùng mong muốn” nhưng runtime hiện tại chưa có cùng contract;
- `documented-intent`: specification yêu cầu, cần đối chiếu runtime;
- `historical`: chỉ để đọc code hoặc docs cũ;
- `unknown`: chưa đủ bằng chứng.

Các mapping types giữ nguyên nghĩa từ offline ontology: `canonical`, `exact alias`, `contextual alias`, `historical name`, `not an alias`.

## 3. Ontology cấp cao

| Canonical name | Object type | Owner | Status |
| --- | --- | --- | --- |
| `online_tta_phase` | phase | online TTA engine | implemented |
| `offline_variant` | inherited experiment dimension | offline config/checkpoint identity | inherited |
| `online_variant` | online experiment dimension | online config | implemented |
| `causal_window` | online input object | online stream | implemented |
| `frozen_source_model` | inherited model state | `OnlineAdaptationModel` | implemented |
| `online_mlp_projector` | mutable online module | `OnlineAdaptationModel` | implemented |
| `threshold_artifact` | inherited calibration artifact | offline benchmark, read by online | inherited |
| `online_event` | one score/triage/adaptation lifecycle | online engine | implemented |
| `verification_buffer` | stateful container | `VerificationBuffer` | implemented |
| `online_runtime_state` | resumable stream state | runtime-state owner | implemented |

### 3.1 `online_tta_phase`

Pha xử lý causal stream sau offline pre-training. Mỗi prediction phải được finalise trước adaptation của event hiện tại; update chỉ ảnh hưởng future causal windows.

| Tên gặp trong repo | Mapping |
| --- | --- |
| `online TTA` | exact alias trong prose |
| `online phase` | contextual alias |
| `online adaptation phase` | contextual alias |
| `Phase 4` | historical project-planning name, không dùng làm runtime identifier |

### 3.2 `online_variant`

| Value | Canonical meaning |
| --- | --- |
| `A0` | Inference only; không gọi projector và không có optimizer |
| `A1` | Chỉ update bằng verified non-empty PNN reconstruction path |
| `A2` | Update bằng guarded hard-old path hoặc verified non-empty PNN path, kèm `online_contrastive_loss` |

`O0_A2` hoặc `O1_A2` là combined run label gồm `offline_variant` và `online_variant`. Nó không phải value mới của riêng `online_variant`.

## 4. Offline objects được kế thừa

### 4.1 `stage_b_best_checkpoint`

Checkpoint input chính thức của online source model.

| Online name | Mapping | Ghi chú |
| --- | --- | --- |
| `reference_checkpoint_path` | field trỏ đến canonical object | Field path, không phải checkpoint object |
| `reference checkpoint` | contextual role alias | Chỉ đúng khi path resolve Stage B `best.pt` |
| `offline checkpoint` | contextual alias | Quá chung cho contract mới |
| `stage_a_best_checkpoint` | not an alias | Chưa chứa final initialized/fine-tuned Stage B state |

### 4.2 `frozen_source_model`

`ThesisMultitaskModel` được khôi phục từ `stage_b_best_checkpoint`, đặt evaluation mode và freeze. Nó sở hữu:

```text
shared_encoder
continuous_prototype_bank
discrete_codebook
anomaly_verification_metadata
reconstruction_fusion_projection
classification_fusion_projection
reconstruction_head
classification_head
```

`reference_encoder.model` là runtime path đi đến object này. `online_encoder` là adapter thứ hai trên cùng inherited offline model; nó không phải một encoder được online optimizer cập nhật.

### 4.3 `threshold_artifact`

Entity-scoped artifact do offline phase tạo. Online đọc ít nhất:

```text
online_point_ewma_threshold
input_window_threshold
latent_window_low_threshold
latent_window_high_threshold
checkpoint identity
entity identity
window_size
EWMA weights
```

Artifact runtime có nested field names như `thresholds.online_ewma_point.value`. Field serialization có thể khác canonical name, nhưng object semantics không đổi.

## 5. Online input, score, and prediction objects

### 5.1 `causal_window`

Latest online window chỉ chứa observations đã xuất hiện đến current cursor:

```text
x: FloatTensor[B, L, D]
point_labels: None
absolute_indices: LongTensor[B, L]
timestamps: Tensor[B, L] | None
meta.entity_id: str
meta.start_index: int
meta.end_index: int
meta.stream_step: int
```

`W_t`, `online_batch` và `batch` là contextual aliases. `offline_window` không phải alias vì online window có causal lifecycle và stride khác.

### 5.2 `window_point_scores`

Inherited model output vector `[B,L]`. Mỗi value là point-wise reconstruction score trong current window.

`raw_point_scores` là exact alias trong desired-flow draft. Canonical name nói rõ container thuộc một window.

### 5.3 Desired vector prediction objects

Các object sau thuộc `desired-contract` hiện có trong “Flow người dùng mong muốn”:

| Canonical name | Shape | Meaning |
| --- | --- | --- |
| `previous_window_ewma_point_scores` | `[L]` | EWMA vector lưu từ online step trước |
| `current_window_ewma_point_scores` | `[L]` | EWMA vector của current causal window |
| `window_point_predictions` | `[L]` | Binary prediction vector sau threshold |

`point_level_binary_predictions` là exact alias của `window_point_predictions`. Không gọi vector này là `prediction` vì runtime record hiện dùng `prediction` cho một scalar endpoint prediction.

### 5.4 Implemented endpoint prediction objects

Runtime hiện hành lấy point cuối của `window_point_scores`:

| Canonical name | Runtime name | Shape |
| --- | --- | --- |
| `endpoint_point_score` | `raw_point_score` | scalar |
| `previous_endpoint_ewma_point_score` | `previous_ewma_score` | scalar hoặc absent |
| `current_endpoint_ewma_point_score` | `ewma_point_score` | scalar |
| `endpoint_point_prediction` | `prediction` | scalar binary |

Các scalar endpoint objects không phải aliases của desired vector objects. Đây là contract difference, không chỉ là naming difference.

### 5.5 `online_point_ewma_threshold`

Threshold point-level áp dụng lên online EWMA score.

| Tên gặp trong repo | Mapping |
| --- | --- |
| `online_ewma_point_threshold` | exact runtime/artifact alias |
| `B_point_high` | exact mathematical alias |
| `T_point_EWMA` | exact mathematical alias |
| `threshold_value` | contextual local-variable alias |
| `offline_point_threshold` | not an alias |
| `input_window_threshold` | not an alias |

## 6. Representation and model objects

### 6.1 `source_hidden`

Frozen source encoder output cho `causal_window`. Runtime field là `reference_hidden`. Trong A0, source hidden trực tiếp làm query representation.

### 6.2 `projected_hidden`

Output của `online_mlp_projector(source_hidden)`. A1/A2 dùng object này làm mutable query representation. `online_hidden` là contextual alias; không dùng vì dễ nhầm với frozen `online_encoder` adapter.

### 6.3 `online_mlp_projector`

Module mutable duy nhất trong accepted A1/A2 event. `projector` là runtime convenience alias. Không map `online_encoder` sang projector; encoder vẫn frozen.

### 6.4 `online_model_outputs`

Stable output contract kế thừa offline model:

```text
hidden
pooled
reconstruction
classification_logits
window_point_scores
window_anomaly_scores
aux.reference_hidden
aux.projected_hidden
aux.latent_window_score
```

Runtime top-level aliases là `recon`, `logits`, `point_scores` và `window_scores`.

## 7. Triage objects

### 7.1 `input_window_score`

Window-level reconstruction MSE trong input space. Đây là score dùng với `input_window_threshold` trong four-region triage.

`raw_point_score` và `endpoint_point_score` không phải aliases: chúng có level và reduction khác.

### 7.2 `latent_window_score`

Deterministic latent-memory score dùng với latent threshold band. Runtime có thể fallback sang `window_scores` nếu output không cung cấp field riêng; fallback không làm hai score types trở thành aliases trong ontology.

### 7.3 Triage thresholds

| Canonical name | Mathematical alias | Runtime artifact meaning |
| --- | --- | --- |
| `input_window_threshold` | `B_window` | High quantile của clean-validation input-window score |
| `latent_window_low_threshold` | `A_low` | Lower edge của latent threshold band |
| `latent_window_high_threshold` | `A_high` | Upper edge của latent threshold band |

`online_point_ewma_threshold` không phải alias của ba triage thresholds.

### 7.4 `triage_region`

Kết quả four-region classification trước admission và verification:

```text
normal
hard_old_normality
gray_zone
strong_anomaly
```

Truth table:

| Condition | `triage_region` |
| --- | --- |
| `input_window_score <= input_window_threshold` | `normal` |
| `input_window_score > input_window_threshold` và `latent_window_score <= latent_window_low_threshold` | `hard_old_normality` |
| `input_window_score > input_window_threshold` và `latent_window_low_threshold < latent_window_score <= latent_window_high_threshold` | `gray_zone` |
| `input_window_score > input_window_threshold` và `latent_window_score > latent_window_high_threshold` | `strong_anomaly` |

| Tên cũ | Mapping |
| --- | --- |
| `triage_decision` | exact alias khi value thuộc đúng bốn regions |
| `decision` | contextual alias, không dùng trong contract mới |
| `event_decision` | not an alias nếu object còn gộp verification/adaptation state |

### 7.5 `hard_old_normality`

Một `triage_region`, không phải buffer entry hay verification result. Chỉ A2 có thể update trên region này, và chỉ khi `hard_old_interval_guard` chấp nhận interval.

`hard_old` và `hard-old` là prose aliases. Dùng `hard_old_normality` cho identifier.

### 7.6 `hard_old_interval_guard`

State object ngăn accepted hard-old updates dùng overlapping intervals. Runtime class là `NonOverlapGuard`. Nó không phải `verification_buffer`.

## 8. Verification and PNN objects

### 8.1 `verification_buffer`

Instance của `VerificationBuffer`. Container này sở hữu admitted gray-zone entries, non-overlap admission, capacity, TTL và “new since cycle” state.

`VerificationBuffer` là class name. `TTLBuffer` không phải alias; endpoint TTL buffer lịch sử đã bị loại khỏi active flow.

### 8.2 `verification_entry`

Một admitted gray-zone causal window:

```text
entry_id: str
entity_id: str
start_index: int
end_index: int
x: FloatTensor[L,D]
status: "unresolved" | "adapted"
ttl_remaining: int
admitted_at_cursor: int
```

Runtime serialization mapping:

| Runtime field | Canonical field | Mapping |
| --- | --- | --- |
| `window_start` | `start_index` | exact alias |
| `window_end` | `end_index` | exact alias |
| `window` | `x` | exact alias |
| `stream_step` | `admitted_at_cursor` | contextual alias; counter equivalence phải được giữ |
| `point_score` | `endpoint_point_score` | contextual snapshot field |

`verification_buffer` không phải alias của `verification_entry`; một object là container, object kia là item.

### 8.3 `verification_cycle`

Một cycle bắt đầu khi buffer đạt capacity, có entry mới kể từ cycle trước và không có cycle khác đang chạy. Cycle encode stored entries bằng frozen source model, tính deterministic geometry, tạo results, commit adapted statuses rồi tick TTL cho unresolved entries.

### 8.4 Deterministic verification tensors

| Canonical name | Shape | Meaning |
| --- | --- | --- |
| `nearest_codeword_ids` | `[N,L]` | Nearest discrete codeword per token |
| `nearest_codeword_distances` | `[N,L]` | Cosine distance tới nearest codeword |
| `known_anomaly_mask` | `[N,L]` | Tokens nằm trong anomalous codeword radii |
| `continuous_signature_ids` | `[N,L,3]` | Ordered top-3 continuous prototype ids per token |
| `recurrent_signature_set` | set of signature tuples | Signatures xuất hiện trong hơn một non-overlapping window |
| `pnn_mask` | `[N,L]` | Pseudo-new-normal tokens còn lại sau known-anomaly filtering và recurrence check |

`recurrent_signatures` là exact runtime alias của `recurrent_signature_set`. `recurrent_signature_ids` không phải alias; runtime field này có thể chứa tensor aligned với selected tokens.

### 8.5 `verification_outcome`

Kết quả verification tách biệt với `triage_region`. `pnn_verified` là outcome cho entry có non-empty `pnn_mask` và được phép đi vào PNN update path.

`triage_region = pnn_verified` là overloaded runtime compatibility behavior. Nó không được dùng trong contract mới vì `pnn_verified` không phải một trong four triage regions.

## 9. Adaptation objects and losses

### 9.1 `hard_old_reconstruction_loss`

Hinge loss đẩy online window reconstruction score xuống dưới `input_window_threshold`:

```text
hard_old_reconstruction_loss
    = RELU(online_window_anomaly_score - input_window_threshold)^2
```

Runtime names gồm `reconstruction_loss` trong hard-old branch và `loss_hard_recon` trong metrics.

### 9.2 `pnn_reconstruction_loss`

Masked reconstruction loss chỉ trên positions có `pnn_mask = TRUE`.

`PNN loss`, `masked PNN reconstruction loss` và `loss_pnn_recon` là contextual aliases.

### 9.3 `online_contrastive_loss`

Source-consistency contrastive regularization được thêm vào accepted A2 hard-old hoặc PNN event.

| Tên gặp trong repo | Mapping |
| --- | --- |
| `L_online_contrastive` | exact mathematical alias |
| `SRC-ON loss` | exact prose alias |
| `contrastive_loss` | contextual runtime local-variable alias |

### 9.4 `online_total_loss`

| Event | Formula |
| --- | --- |
| A1 `pnn_verified` | `pnn_reconstruction_loss` |
| A2 `hard_old_normality` | `hard_old_reconstruction_loss + lambda_online_contrastive * online_contrastive_loss` |
| A2 `pnn_verified` | `pnn_reconstruction_loss + lambda_online_contrastive * online_contrastive_loss` |

`lambda_online_contrastive` là canonical config meaning. Runtime hiện hard-code multiplier `0.1`; không gọi value này là offline `lambda_contrastive` nếu config ownership khác.

### 9.5 `online_update_event`

Một atomic update gồm fresh optimizer, zero gradients, one finite loss, backward, frozen-gradient assertions, projector gradient clipping và đúng một optimizer step. Chỉ `online_mlp_projector` được mutation.

## 10. Record and state objects

### 10.1 `online_event_record`

Per-window immutable record sau scoring và optional update:

```text
entity_id
point_index
start_index
end_index
endpoint_point_score or window_point_scores
current endpoint/vector EWMA point score
online_point_ewma_threshold
point prediction
online_variant
triage_region
verification_outcome
did_update
online_total_loss
```

Desired vector flow và implemented scalar flow phải dùng field names khác nhau như Section 5 quy định.

### 10.2 `online_runtime_state`

Resumable state chứa entity identity, offline/online variant identity, threshold artifact identity, cursor, EWMA state, projector state, verification buffer state, signature history và hard-old guard state. Nó không chứa optimizer moments.

## 11. Quan hệ chuẩn giữa các object

| Subject | Relation | Object |
| --- | --- | --- |
| `online_tta_phase` | `loads` | `stage_b_best_checkpoint` |
| `stage_b_best_checkpoint` | `restores` | `frozen_source_model` |
| `online_tta_phase` | `reads` | `threshold_artifact` |
| `frozen_source_model.shared_encoder` | `produces` | `source_hidden` |
| `online_mlp_projector` | `maps` | `source_hidden` to `projected_hidden` |
| `frozen_source_model` | `produces` | `window_point_scores` |
| desired EWMA step | `maps` | `window_point_scores` to `current_window_ewma_point_scores` |
| implemented EWMA step | `maps` | `endpoint_point_score` to `current_endpoint_ewma_point_score` |
| `triage_region` | `depends on` | `input_window_score`, `latent_window_score`, and triage thresholds |
| `gray_zone` | `may create` | `verification_entry` |
| `verification_buffer` | `contains` | `verification_entry` |
| `verification_cycle` | `produces` | `verification_outcome` and `pnn_mask` |
| `pnn_mask` | `selects positions for` | `pnn_reconstruction_loss` |
| `hard_old_interval_guard` | `gates` | A2 hard-old update |
| `online_update_event` | `mutates only` | `online_mlp_projector` |

## 12. Desired contract versus implemented runtime

| Concern | Desired contract | Implemented runtime |
| --- | --- | --- |
| Point-score object | `window_point_scores [L]` | `endpoint_point_score` lấy point cuối |
| EWMA state | `previous_window_ewma_point_scores [L]` | `previous_endpoint_ewma_point_score` scalar |
| Prediction | `window_point_predictions [L]` | `endpoint_point_prediction` scalar |
| PNN order | Triage, gray-zone admission, rồi verification | Preliminary PNN computation trước triage, sau đó verification recompute |
| Verification outcome | Tách khỏi `triage_region` | Compatibility path truyền `pnn_verified` qua `triage_decision` argument |
| Signature history | Chỉ selected/admitted protocol windows | Preliminary path append current window trước triage |

Tài liệu pseudocode “Flow người dùng mong muốn” phải dùng desired-contract names. Phần “Flow code hiện tại” phải giữ implemented names. Không sửa một bên bằng tên của bên kia vì chúng là các object contracts khác nhau.

## 13. Terminology changes

| Old name | New canonical name | Status | Runtime owner | Migration boundary |
| --- | --- | --- | --- | --- |
| `raw_point_scores` | `window_point_scores` | renamed for container clarity | model output | Desired pseudocode |
| `previous_ewma_point_scores` | `previous_window_ewma_point_scores` | renamed for scope | desired online state | Desired pseudocode |
| `current_ewma_point_scores` | `current_window_ewma_point_scores` | renamed for scope | desired online state | Desired pseudocode |
| `point_level_binary_predictions` | `window_point_predictions` | renamed for container clarity | desired output | Desired pseudocode |
| `raw_point_score` | `endpoint_point_score` | renamed for scalar meaning | current online engine | Docs first; source migration separate |
| `previous_ewma_score` | `previous_endpoint_ewma_point_score` | renamed for scalar meaning | current online engine | Docs first |
| `ewma_point_score` | `current_endpoint_ewma_point_score` | renamed for scalar meaning | current online engine | Docs first |
| `B_point_high`, `T_point_EWMA` | `online_point_ewma_threshold` | exact alias normalization | threshold artifact | Pseudocode/docs |
| `B_window` | `input_window_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `A_low` | `latent_window_low_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `A_high` | `latent_window_high_threshold` | exact mathematical alias normalization | threshold artifact/triage | Pseudocode/docs |
| `triage_decision` | `triage_region` | renamed for object type | triage | New contracts |
| `decision = pnn_verified` | `verification_outcome = pnn_verified` | split | verification cycle | New contracts |
| `entries` | `verification_entries` | renamed for ownership | verification buffer | Pseudocode |
| `P_known_anomaly` | `known_anomaly_mask` | replaced set-like prose with tensor object | verification | Pseudocode |
| `P_pseudo_new_normality` | `pnn_mask` | replaced set-like prose with tensor object | verification | Pseudocode |
| `SRC-ON loss` | `online_contrastive_loss` | renamed for identifier clarity | online loss | Pseudocode |

## 14. Bắt buộc đối chiếu khi có specification mới

Mọi specification version mới phải đối chiếu cả offline và online ontologies. Với từng rename hoặc object mới, ghi:

```text
old name
new canonical name
mapping type
semantic equivalence or difference
runtime owner
schema and stored data
lifecycle and callers
offline-to-online lineage
checkpoint and artifact impact
migration boundary
```

Không được map tự động:

- `offline_point_threshold` với `online_point_ewma_threshold`;
- `input_window_threshold` với point thresholds;
- `triage_region` với `verification_outcome`;
- `verification_buffer` với `verification_entry`;
- `window_point_scores` vector với `endpoint_point_score` scalar;
- `frozen_source_model` với `online_mlp_projector`;
- `stage_a_best_checkpoint` với `stage_b_best_checkpoint`.
