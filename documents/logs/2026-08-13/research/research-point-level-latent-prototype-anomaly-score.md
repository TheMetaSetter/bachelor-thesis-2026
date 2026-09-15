---
date: 2026-08-13T00:00:00+07:00
researcher: OpenAI Codex
topic: "Point-level latent prototype-displacement anomaly score"
status: complete
revision: 38bcf7027a470213475f3a2f7b7bd7c297338a0c
branch: dev
---

# Research: Point-level latent prototype-displacement anomaly score

## Kết luận chính

Thiết kế đề xuất lấy MSE giữa point-level latent vector trước và sau khi truy vấn `continuous_prototype_bank`. Ý tưởng này có tiền thân trong `full-spec-v1` ở mức window, nhưng chưa phải anomaly score chính thức và chưa được runtime hiện hành tính theo cùng công thức.

Runtime hiện hành phân biệt raw reconstruction error với anomaly score đã calibrate:

1. `raw_point_scores` là raw point-level MSE trong input space.
2. `point_scores` chính thức là shifted-and-scaled logistic sigmoid của raw point-level MSE đó.
3. `latent_window_score` online là trung bình theo window của cosine distance tới normal continuous prototype gần nhất. Đây là auxiliary triage score, không phải một trong ba thiết kế point-level anomaly score được tổng hợp.

Vì vậy, point-level latent MSE mới phải có tên và contract riêng trong thí nghiệm. Không được coi nó là alias của `point_scores` hoặc `latent_window_score` hiện tại.

## Ba thiết kế anomaly score đã được xem xét

| ID thí nghiệm | Nguồn tín hiệu | Granularity gốc | Trạng thái |
| --- | --- | --- | --- |
| `S_input-raw` | Raw point-level reconstruction MSE giữa input point và reconstructed point | Point | Thiết kế 1; raw score trong input space |
| `S_input-calibrated` | Shifted-and-scaled logistic sigmoid của `S_input-raw`, với `c` và `tau` fit trên clean validation | Point | Thiết kế 2; official THESIS v3 |
| `S_latent-displacement` | MSE giữa normalized latent point trước continuous retrieval và normalized `prototype_context` sau retrieval | Point | Thiết kế 3; chưa implement và chưa được chấp nhận là official |

RedLamp-style reconstruction/classification score không thuộc bộ ba này. Nó chỉ là một reference/exploratory score từng được xem xét và đã bị loại khỏi official offline THESIS.

`S_input-raw` và `S_input-calibrated` có cùng thứ tự xếp hạng vì sigmoid với `tau > 0` là phép biến đổi đơn điệu tăng. Nếu threshold của mỗi score được chuyển đổi nhất quán từ cùng clean-validation quantile, hai thiết kế phải cho cùng hard predictions và cùng rank-based metrics. Khác biệt chính nằm ở thang đo, calibration identity và khả năng dùng một threshold có ý nghĩa ổn định.

## Bằng chứng hiện hành

### Official reconstruction score

`documents/spec/full-spec-v3.md:553-604` định nghĩa cả thiết kế 1 và 2: `e` là raw point-level MSE; `q = sigmoid((e-c)/tau)` là anomaly score đã calibrate, với `c = median` và `tau = MAD / 0.6745` fit trên clean validation. `documents/spec/full-spec-v3.md:606-648` loại RedLamp-style scoring khỏi official offline THESIS.

### Continuous retrieval

`src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py:197-230`:

- normalize latent input;
- tính similarity logits với continuous bank;
- softmax thành attention weights;
- lấy weighted prototype mixture;
- normalize mixture thành `prototype_context`.

Hai tensor phù hợp để định nghĩa score mới là normalized latent input và normalized `prototype_context`, cùng shape `[B,L,H]`.

### Latent score lịch sử và runtime

`documents/spec/full-spec-v1.md:1011-1031` từng định nghĩa `s_latent_window(W) = MSE(Z_W, Z_W_cont)`. Đây là window score phục vụ triage, không phải point-level anomaly score.

`src/models/online_impl/online_adaptation_helpers.py:110-118` hiện không dùng công thức trên. Runtime tính cosine distance tới từng continuous prototype, lấy minimum theo prototype, rồi mean theo point để ra một scalar cho window.

## Công thức đề xuất cho thí nghiệm

Với hidden dimension `H`, đặt:

\[
\tilde z_{bt}=\operatorname{normalize}(z_{bt}),\qquad
c_{bt}=\operatorname{continuous\_retrieve}(\tilde z_{bt}),
\]

\[
d_{bt}=\frac{1}{H}\sum_{h=1}^{H}(\tilde z_{bth}-c_{bth})^2.
\]

Primary experiment dùng deterministic soft `prototype_context` hiện có. MC latent displacement là sensitivity ablation, không phải primary design.

Vì cả hai vector được L2-normalize, ta có:

\[
d_{bt}=\frac{2}{H}\left(1-\cos(\tilde z_{bt},c_{bt})\right).
\]

Do đó MSE này đo thay đổi hướng latent và có scale phụ thuộc `H`. Thí nghiệm phải ghi cả raw MSE và cosine-equivalent diagnostic để phát hiện sai reduction.

## Khoảng trống cần khóa trước implementation

- Offline THESIS output hiện chưa export point-level latent displacement.
- `latent_window_score` runtime hiện dùng nearest-prototype cosine distance, không dùng soft-query MSE.
- Threshold artifact chưa có identity cho latent point-score family.
- Spec chưa quyết định latent score có thay thế official reconstruction score hay chỉ là ablation.
- Chưa có bằng chứng latent displacement tách normal/anomaly tốt hơn reconstruction error.

## Phạm vi an toàn cho bước kế tiếp

Nên thêm score extraction theo kiểu post-hoc trên cùng Stage-B checkpoint trước. Không retrain model, không thay loss, không thay prototype bank, không thay online triage, và không thay official score cho tới khi paired experiment đủ bằng chứng.

