---
title: "Offline Pre-training Terminology Ontology"
status: authoritative-for-naming
scope: "THESIS offline pre-training, memory initialization, offline evaluation, and the offline-to-online handoff"
applies_to: "specifications, pseudocode, runtime code, tests, configuration, checkpoints, metrics, and experiment artifacts"
evidence_revision: e58602f45ee5439a1e001f060e8ea640aeddde9c
---

# Offline Pre-training Terminology Ontology

> **Notation authority:** Ký hiệu anomaly score mức điểm trong tài liệu này tuân theo [Thiết kế anomaly score mức điểm và bộ ký hiệu chuẩn](anomaly-score-designs-and-notation.md). Tên runtime, config và artifact không bị đổi bởi việc chuẩn hóa ký hiệu.


## 1. Mục đích

Tài liệu này là nguồn chuẩn về **tên object** và **quan hệ giữa object** trong pha offline của THESIS. Nó giải quyết bốn câu hỏi cho mỗi tên:

1. Object đó là gì?
2. Object đó thuộc phase, stage hay operation nào?
3. Object đó nhận input gì và tạo output gì?
4. Tên nào trong code hoặc tài liệu cũ đang nói về cùng object?

Ontology chuẩn hóa tên nhưng không che giấu khác biệt giữa code và specification. Mỗi claim hành vi phải mang một trong các trạng thái:

- `implemented`: source hiện hành thực hiện behavior;
- `configured`: config hiện hành chọn behavior;
- `tested`: test kiểm tra behavior;
- `documented-intent`: specification yêu cầu nhưng source chưa chắc khớp;
- `historical`: chỉ thuộc thiết kế cũ;
- `unknown`: chưa đủ bằng chứng.

Tên chính thức dùng `snake_case` trong pseudocode, schema và prose kỹ thuật. Tên class Python giữ `PascalCase`. Tên artifact file giữ đúng tên file runtime.

## 2. Background: pha offline nằm ở đâu?

THESIS có hai pha lớn nối tiếp nhau:

```text
offline_pretraining_phase
    -> stage_b_best_checkpoint
    -> threshold_artifact
    -> online_tta_phase
```

`offline_pretraining_phase` học model từ dữ liệu train và cố định các model state cần cho deployment. `online_tta_phase` không học lại offline model. Pha online nạp `stage_b_best_checkpoint`, giữ source model frozen và chỉ cập nhật online projector khi protocol cho phép.

Trong phạm vi offline:

```text
offline_pretraining_phase
    contains stage_a_multitask_pretraining
    transitions through stage_b_memory_initialization
    contains stage_b_fusion_finetuning
    is followed by offline_evaluation
```

Hai từ dễ nhầm nhất là `phase` và `stage`:

- `offline_pretraining_phase` là toàn bộ vòng đời offline.
- `stage_a_multitask_pretraining` và `stage_b_fusion_finetuning` là hai training stages bên trong phase.
- `stage_b_memory_initialization` là transition operation giữa hai stages.
- `offline_evaluation` là post-training operation, không phải Stage C.

## 3. Quy tắc mapping

| Mapping | Nghĩa | Có được dùng cho contract mới? |
| --- | --- | --- |
| `canonical` | Tên chính thức duy nhất | Có |
| `exact alias` | Cùng object và cùng semantics | Không; migrate sang canonical name |
| `contextual alias` | Chỉ cùng object trong context được ghi rõ | Không |
| `historical name` | Object thuộc workflow cũ | Không |
| `not an alias` | Object khác, dù tên gần nhau | Không được map |

Tên giống nhau không chứng minh identity. Khi mapping một tên mới, phải so schema, owner, lifecycle, callers, state mutation, checkpoint contract và artifact contract.

## 4. Ontology cấp cao

| Canonical name | Object type | Owner | Lifecycle |
| --- | --- | --- | --- |
| `offline_pretraining_phase` | phase | offline benchmark runner | Bắt đầu khi nạp experiment config; kết thúc sau artifact export |
| `offline_variant` | experiment dimension | experiment config | Cố định cho một offline run; miền giá trị `O0`, `O1` |
| `stage_a_multitask_pretraining` | training stage | two-stage orchestrator + `ThesisMultitaskModel` | Chạy trước memory initialization |
| `stage_b_memory_initialization` | transition operation | two-stage orchestrator + model memory state | Chạy một lần giữa Stage A và Stage B |
| `stage_b_fusion_finetuning` | training stage | two-stage orchestrator + `ThesisMultitaskModel` | Chạy từ initialization checkpoint |
| `offline_evaluation` | post-training operation | benchmark wrapper + `Evaluator` | Chạy bằng Stage B best checkpoint |
| `online_tta_phase` | downstream phase | online TTA engine | Nhận checkpoint và threshold artifact từ offline |

### 4.1 `offline_pretraining_phase`

**Định nghĩa.** Toàn bộ pha tạo source model offline, gồm hai training stages, một memory-initialization transition và post-training evaluation/export.

| Tên gặp trong repo | Mapping | Ghi chú |
| --- | --- | --- |
| `offline pre-training` | exact alias trong prose | Cách viết có dấu gạch nối |
| `offline training` | contextual alias | Chỉ khi nói cả phase, không phải một batch update |
| `two-stage offline pre-training` | contextual alias | Nhấn mạnh topology hiện hành |
| `phase_multitask_pretraining` | not an alias | Tên đề xuất cũ cho một training stage, không phải toàn phase |
| `training_phase` | not an alias ở cấp phase | Runtime field hiện giữ `stage_name` |

### 4.2 `offline_variant`

**Định nghĩa.** Trục ablation quyết định Stage A có dùng `point_score_loss` hay không.

| Value | Canonical meaning |
| --- | --- |
| `O0` | `point_score_loss` tắt; Stage A dùng reconstruction, classification và two-view contrastive losses |
| `O1` | `point_score_loss` bật trong Stage A; không bật mặc định trong Stage B |

`experiment_variant` như `two_stage_base_v1` và `two_stage_point_score_supervised_v1` mô tả protocol chi tiết. Nó không thay thế object identity `offline_variant` trong cross-phase checkpoint và artifact matching.

### 4.3 `stage_a_multitask_pretraining`

**Định nghĩa.** Training stage thứ nhất. Model học encoder và hai task heads từ scratch theo multitask objective. Final continuous/discrete memory retrieval không tham gia forward path ở stage này.

| Runtime name | Mapping | Ghi chú |
| --- | --- | --- |
| `Stage A` | exact alias trong prose | Dùng canonical identifier trong pseudocode |
| `Stage A: Multitask Pretraining` | exact display label | Chỉ dùng UI/log |
| `TWO_STAGE_A_PHASE_NAME` | contextual alias | Constant bị đặt type-name là `PHASE`; value thực tế là stage identifier |
| `training_phase = stage_a_multitask_pretraining` | compatibility field | Semantics là `stage_name` |

### 4.4 `stage_b_memory_initialization`

**Định nghĩa.** Transition operation nạp `stage_a_best_checkpoint`, encode train batches trong `eval()` và `no_grad()`, khởi tạo memory banks cùng verification metadata, rồi lưu `stage_b_initialization_checkpoint`.

| Tên gặp trong repo | Mapping | Ghi chú |
| --- | --- | --- |
| `end-of-Stage-A memory initialization` | exact alias | Nhìn từ output của Stage A |
| `Stage B initialization` | exact alias | Nhìn từ input của Stage B |
| `memory initialization stage` | not an alias | Operation này không phải training stage độc lập |
| `bootstrap` | contextual alias | Chỉ memory bootstrap; không map sang encoder bootstrap epochs |

### 4.5 `stage_b_fusion_finetuning`

**Định nghĩa.** Training stage thứ hai. Stage này dùng frozen encoder và frozen memory banks để huấn luyện task-specific fusion projections và task heads.

| Runtime name | Mapping | Ghi chú |
| --- | --- | --- |
| `Stage B` | exact alias trong prose | Dùng canonical identifier trong pseudocode |
| `Stage B: Fusion Finetuning` | exact display label | Chỉ dùng UI/log |
| `TWO_STAGE_B_PHASE_NAME` | contextual alias | Constant chứa stage identifier |
| `training_phase = stage_b_fusion_finetuning` | compatibility field | Semantics là `stage_name` |
| `fusion warm-up` | not an alias | Có thể là substep hoặc historical term; không đại diện toàn Stage B |

### 4.6 `offline_evaluation`

**Định nghĩa.** Operation chạy Stage B model trên clean validation, synthetic validation và test; khôi phục point-score timelines; hiệu chỉnh threshold từ clean validation; tính metrics; export artifacts.

`evaluation` trong `two_stage_execution_report` là display step name. Nó không phải training stage và không làm thay đổi model parameters.

## 5. Data objects

### 5.1 `raw_sequence`

Một entity sequence trước windowing:

```text
x: FloatTensor[T, D]
point_labels: LongTensor[T] | None
mask: Tensor | None
timestamps: Tensor[T] | None
meta.entity_id: str
meta.split: str
meta.sequence_length: int
```

`train_sequence`, `clean_validation_sequence` và `test_sequence` là cùng object type nhưng thuộc split khác nhau.

### 5.2 `offline_window`

Một đoạn có độ dài `window_size` lấy từ `raw_sequence`:

```text
x: FloatTensor[L, D]
point_labels: LongTensor[L] | None
meta.start_index: int
meta.end_index: int
```

`window`, `input window` và `segment` chỉ là aliases khi schema trên được giữ nguyên. `causal_window` của online không phải alias vì lifecycle và stride khác.

### 5.3 `offline_batch`

Batch chuẩn mà trainer truyền cho model:

```text
x: FloatTensor[B, L, D]
point_labels: LongTensor[B, L] | None
mask: Tensor[B, L, D] | None
timestamps: Tensor[B, L] | None
meta: list[dict]
```

### 5.4 `synthetic_training_batch`

`offline_batch` sau synthetic anomaly injection:

```text
x: FloatTensor[B, L, D]
classification_labels: LongTensor[B]
synthetic_anomaly_mask: BoolTensor[B, L]
augmentation_metadata: list[dict]
```

Canonical field `x` là actual model input. Các tên thiết kế cũ `x_input` và `x_clean` không phải field runtime hiện hành. Chỉ dùng chúng nếu schema mới thực sự lưu cả hai tensors.

### 5.5 `classification_labels`

Window-level class labels. Với `redlamp_multiclass`, class `0` là normal và class `1..11` là 11 synthetic anomaly families.

`class_labels` là exact alias trong specification cũ; runtime field chính thức là `classification_labels`.

### 5.6 `synthetic_anomaly_mask`

Point-level binary mask đánh dấu vị trí thực sự bị injection. Nó không đồng nhất với `classification_labels` vì một anomalous-class window vẫn chứa nhiều clean positions.

### 5.7 `latent_tokens`

Encoder output có shape `[B, L, H]`. Runtime top-level field là `hidden`.

| Tên | Mapping |
| --- | --- |
| `hidden` | exact runtime field |
| `latent tensor` | exact alias trong specification |
| `features` | contextual alias; quá chung cho contract mới |
| `token` | một row `latent_tokens[b, t, :]`, không phải channel |

## 6. Model and state objects

### 6.1 `offline_source_model`

Model được huấn luyện và checkpoint trong offline phase. Runtime class là `ThesisMultitaskModel`. Khi cùng model được nạp vào online và freeze, online ontology gọi instance đó là `frozen_source_model`.

### 6.2 `shared_encoder`

Module biến `offline_batch.x` thành `latent_tokens`. Runtime attribute là `encoder`.

### 6.3 `continuous_memory_initialization_token_pool`

Tập latent tokens dùng để khởi tạo `continuous_prototype_bank`.

Normative meaning: chỉ clean/normal train tokens; không nhận validation, test hoặc future online tokens.

Implemented source hiện hành: normal positions thuộc normal-class synthetic batches. Injected positions không vào pool này.

### 6.4 `discrete_memory_initialization_token_pools_by_class`

Mapping từ class id sang latent-token pool dùng để khởi tạo `discrete_codebook`.

Normative meaning của full-spec-v3:

- class `0`: normal train tokens;
- class `1..11`: injected anomaly tokens của đúng synthetic class.

Implemented source hiện hành gom toàn bộ tokens của mỗi class window. Đây là conflict, không phải alias khác.

### 6.5 `continuous_prototype_bank`

Frozen memory bank biểu diễn normal latent structure. Shape hiện hành là `[32, H]`. Nó được tạo bằng k-means trên `continuous_memory_initialization_token_pool`.

`continuous memory`, `continuous bank` và `continuous prototypes` là contextual aliases. `prototype_context` không phải alias; đó là retrieval output.

### 6.6 `discrete_codebook`

Frozen memory bank chứa class-stratified codewords. Shape hiện hành là `[60, H]`, tương ứng 12 classes và 5 codewords mỗi class.

`discrete memory` và `discrete bank` là contextual aliases. `quantized_hidden` không phải alias; đó là retrieval output.

### 6.7 `anomaly_verification_metadata`

Metadata được tạo cùng `discrete_codebook` để online verification dùng deterministic source geometry:

```text
anomalous_codeword_mask
anomaly_radii
verification_codeword_class_ids
verification_contributing_token_counts
verification_metadata_source
```

Đây là offline-owned state nhưng online-consumed state. Không gọi cả object là `anomaly_radii`, vì radii chỉ là một field.

### 6.8 Stage B retrieval and heads

| Canonical object | Runtime field/module | Meaning |
| --- | --- | --- |
| `continuous_prototype_context` | `prototype_context` | Retrieval output từ continuous bank |
| `discrete_codeword_context` | `quantized_hidden` | Retrieval output từ discrete codebook |
| `reconstruction_fusion_projection` | `reconstruction_concat_projection` | Fuses base, continuous và discrete representations cho reconstruction |
| `classification_fusion_projection` | `classification_concat_projection` | Fuses representations cho classification |
| `reconstruction_fused_hidden` | `hidden_reconstruction` | Input latent của reconstruction head |
| `classification_fused_hidden` | `hidden_classification` | Input latent của classification head |
| `reconstruction_head` | `reconstruction_head` | Tạo `reconstruction` |
| `classification_head` | `classification_head` | Tạo `classification_logits` |

`fusion head` là contextual group name, không phải một module duy nhất trong active `task_specific_concat_projection` mode.

## 7. Prediction, loss, and score objects

### 7.1 Model outputs

| Canonical name | Runtime field | Shape | Meaning |
| --- | --- | --- | --- |
| `reconstruction` | `recon` | `[B,L,D]` | Reconstructed input window |
| `classification_logits` | `logits` | `[B,12]` | Window-class logits |
| `raw_point_mse` | `aux.point_score_samples` reduced over `M` | `[B,L]` | Monte Carlo mean channel-wise reconstruction MSE before score transformation |
| `window_point_scores` | `point_scores` | `[B,L]` | Point-level anomaly score after the shifted-and-scaled logistic sigmoid |
| `window_anomaly_scores` | `window_scores` | `[B]` | Per-window raw reconstruction MSE used by window-level triage |

Raw point MSE remains an intermediate value used to compute `window_point_scores`.
The score transformation parameters are estimated from clean-validation raw
point MSEs: \(\mu^{(\mathrm{input})}_{\mathrm{val}} = \operatorname{median}(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}})\) and \(\gamma^{(\mathrm{input})}_{\mathrm{val}} = \operatorname{MAD}(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}}) / 0.6745\). Here \(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}}\) is the clean-validation raw point-MSE timeline \(\overline{s}_{t,i}\), not the window-level MSE. The calibrated runtime field `point_scores` corresponds to \(s^{(\mathrm{cal})}_{t,i}\).

### 7.2 Stage A losses

| Canonical name | Runtime name | Active in O0 | Active in O1 |
| --- | --- | --- | --- |
| `reconstruction_loss` | `reconstruction_loss` | Có | Có |
| `classification_loss` | `classification_loss` | Có | Có |
| `two_view_contrastive_loss` | `contrastive_loss` | Có | Có |
| `point_score_loss` | `score_loss` | Không | Có, nếu batch có đủ groups |
| `stage_a_total_loss` | `total_loss` trong Stage A | Có | Có |

`L_recon`, `L_cls`, `L_contrastive`, `L_score_point` là mathematical aliases. Dùng canonical snake-case names trong pseudocode; dùng ký hiệu `L_*` trong công thức.

`point_score_loss` dùng `raw_point_mse` trong training. The shifted-and-scaled
logistic sigmoid is applied after raw MSE computation for inference, score
timeline construction, and threshold calibration; it is not an additional
training loss term.

### 7.3 Stage B losses

| Canonical name | Active |
| --- | --- |
| `reconstruction_loss` | Có |
| `classification_loss` | Có |
| `two_view_contrastive_loss` | Không mặc định |
| `point_score_loss` | Không mặc định |
| `stage_b_total_loss` | Có |

### 7.4 Timeline scores and thresholds

| Canonical name | Meaning |
| --- | --- |
| `clean_validation_point_score_timeline` | Transformed window point anomaly scores được đưa về absolute entity timeline trên clean validation |
| `synthetic_validation_point_score_timeline` | Timeline tương tự cho synthetic validation |
| `test_point_score_timeline` | Timeline tương tự cho test |
| `offline_point_threshold` | `Q_0.99` của transformed `clean_validation_point_score_timeline` trên timeline non-overlapping |
| `online_point_ewma_threshold` | `Q_0.99` của transformed clean-validation timeline sau sliding-window + absolute-index EWMA |

`offline_point_threshold_nonoverlap` là exact schema alias trong full-spec-v3. `online_ewma_point_threshold` là runtime artifact alias của `online_point_ewma_threshold`.

## 8. Checkpoint and artifact objects

### 8.1 `two_stage_run_manifest`

Manifest mô tả stage order, generated config paths, checkpoint paths, global epoch ranges và evaluation checkpoint.

### 8.2 `stage_a_best_checkpoint`

Best checkpoint do Stage A trainer chọn bằng configured monitor metric. Nó chưa chứa initialized final memory banks.

### 8.3 `stage_b_initialization_checkpoint`

Checkpoint được tạo từ `stage_a_best_checkpoint` sau `stage_b_memory_initialization`. Runtime filename là `initializations/stage_b_init.pt`.

`stage_a_checkpoint` không phải exact alias vì Stage A có thể có best và final checkpoints. Pseudocode phải ghi rõ `stage_a_best_checkpoint`.

### 8.4 `stage_b_best_checkpoint`

Best checkpoint của `stage_b_fusion_finetuning`. Đây là checkpoint chính thức cho offline evaluation và online source-model loading.

`evaluation_checkpoint` là contextual alias khi manifest trỏ đúng vào Stage B `best.pt`. `reference_checkpoint` là online-context alias.

### 8.5 `threshold_artifact`

Entity-scoped artifact chứa `offline_point_threshold`,
`online_point_ewma_threshold`, triage thresholds, point-score transform
parameters (`point_score_transform`, `point_score_c`, `point_score_tau`,
`point_score_tau_estimator`, `point_score_mad_normalizer`), calibration
identity, checkpoint identity, seed và protocol fields.

### 8.6 `offline_evaluation_record`

Per-entity record sau khi window scores được khôi phục về timeline:

```text
entity_id
point_scores
point_labels
covered_point_mask
raw_num_points
evaluated_num_points
```

### 8.7 `offline_metrics`

Metric mapping tính từ covered `test_point_score_timeline`, ground-truth labels và fixed threshold. VUS-PR, VUS-ROC và Affiliation F1 là metric names; chúng không phải score objects.

### 8.8 `offline_artifact_bundle`

Nhóm output của `offline_evaluation`, gồm score artifacts, `offline_metrics`, `threshold_artifact`, uncertainty summary, provenance, retention manifest và benchmark report. Đây là logical bundle; nó không bắt buộc là một file duy nhất.

## 9. Quan hệ chuẩn giữa các object

| Subject | Relation | Object |
| --- | --- | --- |
| `offline_pretraining_phase` | `contains` | `stage_a_multitask_pretraining` |
| `offline_pretraining_phase` | `contains` | `stage_b_fusion_finetuning` |
| `stage_a_multitask_pretraining` | `produces` | `stage_a_best_checkpoint` |
| `stage_b_memory_initialization` | `loads` | `stage_a_best_checkpoint` |
| `stage_b_memory_initialization` | `reads only` | train split |
| `stage_b_memory_initialization` | `constructs` | `continuous_prototype_bank` |
| `stage_b_memory_initialization` | `constructs` | `discrete_codebook` |
| `stage_b_memory_initialization` | `constructs` | `anomaly_verification_metadata` |
| `stage_b_memory_initialization` | `produces` | `stage_b_initialization_checkpoint` |
| `stage_b_fusion_finetuning` | `loads` | `stage_b_initialization_checkpoint` |
| `stage_b_fusion_finetuning` | `produces` | `stage_b_best_checkpoint` |
| `offline_evaluation` | `loads` | `stage_b_best_checkpoint` |
| `offline_evaluation` | `computes` | `raw_point_mse` |
| score transformation | `maps` | `raw_point_mse` to `window_point_scores` |
| `offline_evaluation` | `calibrates from` | `clean_validation_point_score_timeline` |
| `offline_evaluation` | `produces` | `threshold_artifact` |
| `offline_evaluation` | `produces` | `offline_artifact_bundle` |
| `online_tta_phase` | `inherits` | `stage_b_best_checkpoint` |
| `online_tta_phase` | `inherits` | `threshold_artifact` |

## 10. Terminology changes from older specifications

### 10.1 Raw-input-space MSE v4

Version 4 splits the old ambiguous point-score names. The operational raw
fields are `raw_input_point_mse` and `raw_input_window_mse`. The diagnostic
fields are `normalized_input_point_mse` and `normalized_input_window_mse`.
Raw input means original sensor units restored with the fitted train-only
scaler. The v4 raw protocol uses `score_space: raw_input` and
`point_score_transform: identity`; sigmoid calibration is outside this path.

`point_labels` and `window_labels` are ground truth categories. Predictions are
separate threshold outputs. A window is anomalous when any point label is
anomalous.

| Old name | Canonical name | Status | Runtime owner | Migration boundary |
| --- | --- | --- | --- | --- |
| `offline pre-training` | `offline_pretraining_phase` | unchanged, normalized identifier | benchmark runner | Docs/pseudocode identifiers |
| `Stage 1: Separate Task-Specific Training` | — | deprecated historical stage | historical runner | Không map vào active Stage A |
| `Stage 2: Zipping and Short Recovery` | — | deprecated historical stage | historical runner | Không map vào active Stage B |
| `Stage 3: Memory Initialization and Fusion Warm-Up` | `stage_b_memory_initialization` + `stage_b_fusion_finetuning` | split | active orchestrator/model | Operation và training stage tách riêng |
| `TWO_STAGE_A_PHASE_NAME` | `stage_a_multitask_pretraining` | contextual alias | orchestrator/model source | Rename source constant khi có approved migration |
| `TWO_STAGE_B_PHASE_NAME` | `stage_b_fusion_finetuning` | contextual alias | orchestrator/model source | Rename source constant khi có approved migration |
| `training_phase` | `stage_name` | renamed semantically, compatibility retained | generated config/model config parser | Không đổi source khi chưa có migration plan |
| `continuous memory` | `continuous_prototype_bank` | renamed for specificity | model state | Docs/pseudocode |
| `discrete memory` | `discrete_codebook` | renamed for specificity | model state | Docs/pseudocode |
| `point-wise reconstruction score` | `window_point_scores` | renamed/refined: canonical field now stores transformed anomaly score; raw MSE is a separate intermediate | model output | Field vẫn là `point_scores` |

## 11. Known semantic conflicts

### 11.1 Discrete token pool

`full-spec-v3` quy định class 1–11 chỉ dùng injected anomaly tokens. Source hiện hành reshape và giữ toàn bộ tokens của class windows. Ontology giữ canonical object `discrete_memory_initialization_token_pools_by_class`, nhưng composition của object đang có hai trạng thái:

- `documented-intent`: anomaly positions only cho class 1–11;
- `implemented`: all positions của mỗi class window.

Không được gọi hai semantics này là equivalent.

### 11.2 Missing-class handling

Specification yêu cầu fail khi không đủ eligible class tokens. Source hiện hành dùng combined fallback pool khi một class trống. Đây là behavior conflict, không phải naming conflict.

### 11.3 Offline variant artifact identity

Offline O1 config hiện có `experiment_variant` nhưng thiếu root `offline_variant`. Artifact collector đọc root `offline_variant` và fallback `O0`. Trước khi dùng threshold artifact làm cross-phase evidence, phải xác nhận `variant_name` khớp checkpoint thật.

## 12. Bắt buộc đối chiếu khi có specification mới

Specification mới phải có mục `Terminology changes`. Với mỗi object mới hoặc đổi tên, ghi:

```text
old name
new canonical name
mapping type
semantic equivalence or difference
runtime owner
schema and stored data
lifecycle and callers
checkpoint and artifact impact
migration boundary
```

Nếu evidence chưa đủ để chọn `exact alias`, ghi `not an alias` hoặc `unknown`. Không tự map `phase` với `stage`, `memory bank` với retrieval output, `window score` với point-score timeline, hoặc `checkpoint` với một checkpoint role cụ thể.
