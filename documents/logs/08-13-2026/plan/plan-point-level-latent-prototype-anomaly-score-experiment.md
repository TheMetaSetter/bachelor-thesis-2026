---
date: 2026-08-13T00:00:00+07:00
planner: OpenAI Codex
topic: "Thí nghiệm point-level latent prototype-displacement anomaly score"
status: ready
revision: 38bcf7027a470213475f3a2f7b7bd7c297338a0c
branch: dev
related_research: documents/logs/08-13-2026/research/research-point-level-latent-prototype-anomaly-score.md
---

# Plan: Thí nghiệm point-level latent prototype-displacement anomaly score

## Mục tiêu

Kiểm tra liệu point-level latent MSE giữa representation trước và sau continuous prototype retrieval có là anomaly score tốt hơn official reconstruction score hay không.

Thí nghiệm đầu tiên là paired post-hoc scoring: cùng data split, cùng Stage-B checkpoint, cùng prototype bank và cùng evaluator; chỉ thay score tensor. Không retrain model trong vòng đầu.

## Câu hỏi nghiên cứu

1. `S_latent-displacement` có tăng VUS-PR và Affiliation-F1 so với hai cách biểu diễn input-space score không?
2. Score mới có giữ false-positive burden trên test-normal points ở mức chấp nhận được không?
3. Kết quả có ổn định trên O0/O1, ba entity và ba seed không?
4. Input-space score và latent-space score phát hiện cùng anomaly spans hay bổ sung cho nhau?

## Score contract khóa cho primary experiment

Input của score:

```text
normalized_hidden         [B,L,H]
continuous_prototype_context [B,L,H]
```

Raw point-level latent displacement:

\[
d_{bt}=\operatorname{mean}_{h}
\left(\tilde z_{bth}-c_{bth}\right)^2.
\]

Primary path dùng deterministic soft `prototype_context`. Score được tính sau continuous retrieval và trước fusion/reconstruction head. Continuous bank phải frozen.

Calibration theo từng entity, chỉ từ clean validation:

\[
c_d=\operatorname{median}(d_{cv}),\qquad
\tau_d=\operatorname{MAD}(d_{cv})/0.6745,
\]

\[
q^{latent}_{bt}=\sigma((d_{bt}-c_d)/\tau_d).
\]

Primary threshold là q99 của clean-validation timeline để khớp default protocol `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml`. q95 chỉ là sensitivity analysis. Test labels không được dùng cho transform, threshold hoặc model selection.

## Thiết kế so sánh

| Nhánh | Score | Vai trò |
| --- | --- | --- |
| `S_input-raw` | Raw point-level reconstruction MSE trong input space | Thiết kế 1; control nguyên thủy |
| `S_input-calibrated` | Shifted-and-scaled sigmoid của `S_input-raw` | Thiết kế 2; official control |
| `S_latent-displacement` | Shifted-and-scaled sigmoid của point soft-retrieval latent MSE | Thiết kế 3; candidate chính |
| `S_latent-nearest` | Point cosine distance tới continuous prototype gần nhất | Diagnostic control cho runtime `latent_window_score` hiện tại |
| `S_latent-mc` | Mean của per-sample latent displacement qua `M=10` stochastic retrievals | Sensitivity ablation |

Không thử score fusion trong vòng đầu. Fusion thêm hyperparameter và không trả lời câu hỏi score latent tự thân có hữu ích hay không.

`S_input-raw` và `S_input-calibrated` là hai thiết kế score khác nhau về scale và calibration contract, nhưng không phải hai scientific baselines độc lập. Vì transform là đơn điệu tăng, plan phải có equivalence check: rank ordering, q99 hard predictions và rank-based metrics phải khớp khi threshold được calibrate nhất quán.

## Ma trận thí nghiệm

Primary paired matrix:

```text
offline variant: O0, O1
entity: machine-1-6, machine-3-4, machine-3-9
checkpoint seed: 6, 8, 36
window size: 20
score branch: S_input-raw, S_input-calibrated, S_latent-displacement
```

Tổng cộng có 18 checkpoint identities và 54 score evaluations từ cùng forward payload. Hai input-space branches tạo một equivalence pair; scientific paired comparison chính là `S_input-calibrated` với `S_latent-displacement`. `S_latent-nearest` và `S_latent-mc` chạy trên cùng cached latent tensors nếu Phase 1 cho thấy contract đúng.

Theo benchmark safety rule, chạy trước một combination:

```text
O1 / machine-1-6 / seed8 / window20
```

Chỉ mở rộng 18 identities sau khi combination này pass shape, calibration, artifact và metric checks.

## Phase 0: Khóa semantics và acceptance gates

### Việc cần làm

- Ghi terminology mapping vào spec kế tiếp: `latent_window_score` hiện tại, proposed `raw_point_latent_displacement`, proposed `latent_point_scores`.
- Xác nhận primary reduction là mean theo hidden dimension, không sum.
- Xác nhận pre-query vector là normalized latent mà continuous lookup thực sự dùng, không phải raw encoder magnitude.
- Xác nhận post-query vector là deterministic soft `prototype_context`, không phải nearest prototype và không phải fused hidden.
- Giữ official score hiện tại cho tới khi experiment gate pass.

### Acceptance gate

- Mỗi tên chỉ một runtime object.
- Không còn ambiguity về normalization, reduction, deterministic/MC và threshold source.

## Phase 1: Thêm score extraction tối thiểu

### Proposed code boundary

- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`: thêm pure helper tính point latent displacement từ hai tensor `[B,L,H]`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py`: export raw point latent displacement trong `aux`, không thay top-level `point_scores`.
- `src/models/online_impl/online_adaptation_helpers.py`: expose cùng point tensor; giữ `latent_window_score` cũ để không đổi triage.
- Proposed `scripts/ops/evaluate_latent_point_score_ablation.py`: load checkpoint, collect clean-validation/test score timelines, calibrate score và gọi evaluator hiện có.

### Focused tests

- Shape: `[B,L,H] -> [B,L]`.
- Identity: identical vectors cho score 0.
- Ordering: vector bị lệch nhiều hơn cho score lớn hơn.
- Formula: normalized-vector MSE khớp `2 * cosine_distance / H`.
- Fail-fast khi prototype bank chưa initialized hoặc retrieval bị bypass.
- Forward pass không thay reconstruction, logits, official `point_scores`, `window_scores` hoặc checkpoint state.
- `S_input-raw` và `S_input-calibrated` giữ đúng thứ tự point; q99 predictions phải giống nhau khi không có numerical boundary mismatch.

## Phase 2: One-combination smoke và score audit

Chạy `O1 / machine-1-6 / seed8` bằng `.venv/bin/python`.

### Audit trước metric

- Tất cả raw/transformed score finite.
- `tau_d > 0`; nếu MAD bằng 0 thì dừng, không tự thêm epsilon.
- Timeline sau overlap aggregation cùng absolute length và ordering với official score.
- Clean-validation q99 prediction rate gần 1% theo quantile contract, có ghi tie count.
- Không có test label trong scorer/calibrator call graph.
- Histogram/quantiles của attention entropy và latent displacement không bị collapse.

Attention entropy là diagnostic bắt buộc vì continuous logits hiện chia cho `sqrt(H)`. Nếu softmax gần uniform trên mọi point, `prototype_context` có thể gần bank mean và score mới mất khả năng phân biệt.

### Smoke acceptance

- Tất cả contract checks pass.
- Score candidate không constant, không chỉ có một vài unique values, và không có >1% exact ties tại threshold mà không được giải thích.
- Artifact tóm tắt đủ provenance; không lưu raw forward tensors mặc định.

## Phase 3: Chạy paired matrix

Với mỗi checkpoint identity:

1. Load một Stage-B best checkpoint và frozen banks.
2. Chạy clean validation một lần để collect raw input-space MSE và raw latent-displacement MSE.
3. Giữ raw input-space MSE thành `S_input-raw`; fit calibration riêng cho input-space và latent-space families trên cùng clean-validation points.
4. Chạy test một lần; tạo ba score branches từ cùng forward payload.
5. Overlap-aggregate theo cùng absolute-index rule.
6. Apply threshold đã khóa; sau đó mới join test labels để tính metrics.
7. Ghi summary statistics, config/checkpoint hash và score-contract identity.

Không chạy model forwards riêng cho từng score nếu có thể tái sử dụng cùng payload. Việc này giữ paired comparison thật sự cùng stochastic realization.

## Phase 4: Metrics và phân tích

### Primary metrics

- VUS-PR: maximize.
- Affiliation-F1: maximize.

### Secondary metrics

- VUS-ROC: internal diagnostic.
- Point precision, recall, F1.
- `PosPred`, FP count và false-positive rate trên test-normal points.
- Detection delay theo anomaly span, nếu evaluator hiện có contract rõ; không tự thêm point adjustment.

### Complementarity diagnostics

- Spearman correlation giữa raw input-space và raw latent-space timelines.
- Equivalence audit giữa `S_input-raw` và `S_input-calibrated`: Spearman bằng 1 trong giới hạn số học, identical q99 predictions và identical rank-based metrics.
- Overlap của false-positive sets và false-negative spans.
- Score quantiles tách riêng test-normal và test-anomaly, chỉ dùng cho phân tích sau inference.
- Stratify theo anomaly span length để xem latent score có chỉ hữu ích cho anomaly dài/ngắn hay không.

### Statistical reporting

- Tính paired delta `latent - reconstruction` cho từng variant/entity/seed.
- Báo cáo median, mean và range của paired deltas; không chỉ báo cáo Average.
- Dùng block bootstrap hoặc anomaly-span bootstrap cho confidence interval; không bootstrap các point như quan sát độc lập.
- Không chọn q95/q99 theo test metric. q99 là primary; q95 được báo cáo như sensitivity result đã đăng ký trước.

## Phase 5: Decision gate

Chỉ đề xuất latent score thành official candidate nếu:

1. Mean hoặc median paired delta của cả VUS-PR và Affiliation-F1 dương.
2. Cải thiện không chỉ đến từ một entity hoặc một seed.
3. FP burden không tệ hơn một cách lớn và có hệ thống. Ngưỡng "lớn" phải được chốt trước full run nếu cần hard acceptance rule.
4. Score/calibration không collapse và không phụ thuộc test distribution.
5. Kết quả lặp lại ở cả O0 và O1, hoặc có giải thích rõ vì sao chỉ một variant hưởng lợi.

Nếu gate fail, giữ score này là diagnostic/triage feature. Không fusion với reconstruction score trong cùng experiment batch.

## Phase 6: Chỉ sau khi gate pass

- Cập nhật `documents/spec/` với terminology-change section và score identity.
- Quyết định thay thế hay thêm ablation; không silently overwrite `point_scores` semantics.
- Mở rộng threshold artifact schema với score family, pre/post representation identity, reduction, calibration parameters và checkpoint/prototype hashes.
- Thêm online EWMA experiment sau offline evidence. Online phải calibrate EWMA latent score trên stride-1 clean validation, không dùng test stream.
- Sau đó mới cân nhắc retraining loss hoặc score fusion trong một protocol riêng.

## Artifacts cần giữ

Mỗi identity chỉ giữ:

- resolved config và protocol identity;
- Stage-B checkpoint path/hash và continuous bank hash;
- calibration summary cho từng score family;
- threshold, metrics và paired deltas;
- selected histograms/quantiles, attention entropy và tie diagnostics;
- aggregate manifest.

Không giữ toàn bộ latent tensor, attention matrix hoặc per-forward MC tensor mặc định.

## Rủi ro chính

| Rủi ro | Tác động | Giảm thiểu |
| --- | --- | --- |
| Soft retrieval gần uniform | Score đo khoảng cách tới bank mean thay vì local normal structure | Audit attention entropy; so với `S_latent-nearest` |
| MSE của unit vectors co theo `1/H` | Giá trị rất nhỏ, dễ nhầm reduction | Unit test cosine identity; lưu `H` trong provenance |
| MAD bằng 0 | Không fit được sigmoid scale | Fail-fast và báo score collapse |
| Dùng test min-max/threshold | Leakage và metric lạc quan | Fit mọi transform/threshold chỉ từ clean validation |
| Đổi triage ngoài ý muốn | Online behavior không còn paired | Export score mới trong `aux`; giữ `latent_window_score` cũ |
| Nhiều score variants gây selection bias | Chọn result theo test | Khóa primary candidate/q99 trước run; variants còn lại là diagnostic |

## Tiêu chí hoàn tất

- One-combination smoke pass trước full matrix.
- 18/18 checkpoint identities có paired results hoặc có failure record rõ.
- Không non-finite score, không calibration/test leakage, không mismatch absolute indices.
- Báo cáo có per-entity/per-seed values, Average, paired deltas, FP audit và uncertainty interval.
- Kết luận ghi rõ: `replace`, `retain as ablation`, hoặc `reject`; không đổi official spec khi evidence chưa đủ.

