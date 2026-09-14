# The Story of the Anomaly Score and Its Notation

> The story starts with one question: when THESIS sees a time-point, which
> score best describes its error? This document follows that question through
> input space, calibration, latent space, EWMA, and the boundary between a
> score, a loss, and a runtime decision.

## Story path

We start with one input window and its reconstructions. We then compare three
score designs, connect each symbol to its runtime field, and record which
choices are official and which are still experimental. The formulas below are
the technical evidence for this story.

## Default score contract

The default operational anomaly score is simple reconstruction MSE with the
identity transform. A current run must declare:

```yaml
score_space: raw_input
point_score_transform: identity
point_score_definition: raw_input_point_mse
```

The scorer averages per-sample MSE values before it makes a point or window
score. It does not apply a sigmoid. A latent-space run is also valid, but it
must declare `score_space: latent` and use a named latent MSE definition. The
latent score and the raw-input score must not be mixed in one threshold
artifact.

For `score_space: raw_input`, compute MSE after inverse-transforming the scaled
input and every reconstruction into the original sensor units. For
`score_space: latent`, compute MSE on the named latent tensors. In both cases,
the identity transform is the default.

The shifted-and-scaled logistic sigmoid is historical and opt-in only. It must
never be selected by omission or by a generic `calibrated` setting.

**Trạng thái:** authoritative-for-notation  
**Phạm vi:** anomaly score mức điểm của THESIS trong đánh giá offline, suy luận online, báo cáo độ bất định và thí nghiệm ablation  
**Nguồn ký hiệu:** Chương 3, “Phương pháp đề xuất”  
**Ngày:** 2026-08-13

## 1. Mục đích

Tài liệu này quy định bộ ký hiệu chuẩn cho ba thiết kế anomaly score mức điểm. Các spec khác trong **documents/spec/** phải dùng bộ ký hiệu này khi mô tả cùng một đại lượng.

Tên trường runtime, config và artifact vẫn giữ nguyên. Đây là các tên được dùng trong code và file kết quả. Ví dụ: **point_scores**, **raw_point_scores**, **aux.point_score_samples** và **latent_window_score**.

Trong tài liệu này, “score space” nghĩa là không gian mà score được tính: input space hoặc latent space. “Center” là giá trị trung tâm của score validation. “Scale” là độ phân tán dùng để chuẩn hóa score.

## 2. Ký hiệu nền

Với cửa sổ đầu vào thứ \(t\), ta có:

\[
\mathbf{X}_t=[\mathbf{x}_{t,1};\ldots;\mathbf{x}_{t,T}]
\in\mathbb{R}^{T\times C}.
\]

Ở đây, \(T\) là số điểm trong cửa sổ, \(C\) là số chiều của mỗi điểm, và \(\mathbf{x}_{t,i}\in\mathbb{R}^{C}\) là điểm thứ \(i\).

Encoder là bộ biến đổi input sang không gian latent. Nó tạo tensor:

\[
\mathbf{Z}_t=f_{\mathrm{enc}}(\mathbf{X}_t;\theta_{\mathrm{enc}})
=[\mathbf{z}_{t,1};\ldots;\mathbf{z}_{t,T}]
\in\mathbb{R}^{T\times H}.
\]

Trong công thức này, \(H\) là số chiều latent và \(\mathbf{z}_{t,i}\) là vector latent của điểm \(\mathbf{x}_{t,i}\).

Continuous prototype bank là ngân hàng prototype liên tục:

\[
\mathbf{P}^{(c)}=[\mathbf{p}^{(c)}_1;\ldots;\mathbf{p}^{(c)}_{K_c}]
\in\mathbb{R}^{K_c\times H}.
\]

Sau lần truy vấn ngẫu nhiên thứ \(m\), vector latent được truy hồi là:

\[
\widetilde{\mathbf{z}}^{(c,m)}_{t,i}
=\sum_{k=1}^{K_c}\alpha^{(c,m)}_{t,i,k}\mathbf{p}^{(c)}_k.
\]

Decoder là bộ biến đổi latent trở lại input. Nó tạo \(\widehat{\mathbf{x}}^{(m)}_{t,i}\). \(M\) là số mẫu Monte Carlo, tức số lần chạy truy vấn ngẫu nhiên để lấy trung bình.

## 3. Ba thiết kế anomaly score

### 3.1 Design 1 — raw point-level MSE in input space

MSE (mean squared error, sai số bình phương trung bình) của điểm \(i\) ở mẫu thứ \(m\) là:

\[
s^{(m)}_{t,i}
=\frac{1}{C}
\left\|\mathbf{x}_{t,i}-\widehat{\mathbf{x}}^{(m)}_{t,i}\right\|_2^2.
\]

This is the default operational score. THESIS computes it by averaging the MSE
of the \(M\) samples. No score transform is applied.

\[
\boxed{
\overline{s}_{t,i}
=\frac{1}{M}\sum_{m=1}^{M}s^{(m)}_{t,i}
}.
\]

Quy trình phải lấy trung bình các MSE theo từng mẫu. Quy trình không được tính MSE giữa input và reconstruction trung bình. Hai cách tính này nói chung cho hai kết quả khác nhau.

### 3.2 Optional legacy design — calibrated point-level MSE in input space

This design is not the default. Use it only when a run explicitly declares
`point_score_transform: shifted-and-scaled logistic sigmoid` and records the
legacy calibration parameters in its artifact.

Calibration là bước đưa raw score về thang điểm ổn định hơn. Tập calibration chỉ dùng raw score của clean validation, tức dữ liệu validation không chứa anomaly theo protocol.

Gọi tập raw score đó là \(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}}\). THESIS tính median và MAD (median absolute deviation, độ lệch tuyệt đối trung vị):

\[
\mu^{(\mathrm{input})}_{\mathrm{val}}
=\operatorname{median}\left(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}}\right),
\]

\[
\gamma^{(\mathrm{input})}_{\mathrm{val}}
=\frac{\operatorname{MAD}\left(\mathcal{S}^{(\mathrm{input})}_{\mathrm{val}}\right)}{0.6745}.
\]

The legacy calibrated score uses the sigmoid \(\sigma\). It is not a default
score and is valid only for an explicit historical or ablation run:

\[
\boxed{
s^{(\mathrm{cal})}_{t,i}
=\sigma\left(
\frac{\overline{s}_{t,i}-\mu^{(\mathrm{input})}_{\mathrm{val}}}
{\gamma^{(\mathrm{input})}_{\mathrm{val}}}
\right)
}.
\]

Thiết kế 2 chỉ biến đổi đơn điệu Thiết kế 1. Nếu threshold được biến đổi cùng cách, hai thiết kế giữ nguyên thứ tự điểm và dự đoán nhị phân. Vì vậy, các metric dựa trên thứ tự điểm cũng giữ nguyên.

So sánh hai thiết kế này đo ảnh hưởng của calibration, threshold artifact và độ ổn định số. So sánh này không tạo thêm một tín hiệu anomaly mới.

### 3.3 Optional design — point-level prototype displacement in latent space

Thiết kế 3 đo khoảng cách giữa vector latent trước và sau khi truy vấn continuous prototype bank. Độ lệch tại mẫu thứ (m) là:

\[
\ell^{(c,m)}_{t,i}
=\frac{1}{H}
\left\|\mathbf{z}_{t,i}-\widetilde{\mathbf{z}}^{(c,m)}_{t,i}\right\|_2^2.
\]

The raw latent MSE is the mean of the \(M\) samples. It is the default score
when a run explicitly selects `score_space: latent`:

\[
\boxed{
\overline{\ell}^{(c)}_{t,i}
=\frac{1}{M}\sum_{m=1}^{M}\ell^{(c,m)}_{t,i}
}.
\]

If an experiment needs a score in \((0,1)\), it may use the legacy sigmoid only
when the run opts in and fits a separate center and scale on clean validation:

\[
s^{(\mathrm{latent})}_{t,i}
=\sigma\left(
\frac{\overline{\ell}^{(c)}_{t,i}-\mu^{(\mathrm{latent})}_{\mathrm{val}}}
{\gamma^{(\mathrm{latent})}_{\mathrm{val}}}
\right).
\]

The run must not use input-space center or scale for latent-space MSE. The two
spaces have different distributions.

## 4. EWMA và quyết định mức điểm

EWMA (exponentially weighted moving average, trung bình trượt có trọng số mũ) làm mượt score trên timeline tuyệt đối. Gọi \(r\) là lần xử lý cửa sổ và \(n\) là chỉ số tuyệt đối của điểm:

\[
\widetilde{s}^{(r)}_n
=\rho s^{(r)}_n+(1-\rho)\widetilde{s}^{(r-1)}_n.
\]

Với điểm cục bộ \((t,i)\), chỉ số tuyệt đối được xác định bởi \(n=\operatorname{start}(t)+i\). Giá trị \(s^{(r)}_n\) là một trong các score đã chọn cho run: \(\overline{s}_{t,i}\), \(s^{(\mathrm{cal})}_{t,i}\), \(\overline{\ell}^{(c)}_{t,i}\) hoặc \(s^{(\mathrm{latent})}_{t,i}\).

The default run uses raw input MSE with `point_score_transform: identity`.
The run may select raw latent MSE instead, but must name that score space.
Each run selects one score space. Do not mix score spaces unless the run defines
the combination.

Quyết định anomaly cứng là:

\[
\widehat{a}_n=\mathbb{I}\left(\widetilde{s}_n>T_{\mathrm{point}}\right).
\]

Quy trình phải tính threshold từ clean validation trong đúng score space và dùng đúng chuỗi xử lý của run. Quy trình không được dùng test labels hoặc phân phối score của test để calibration.

## 5. Mapping giữa ký hiệu và tên runtime

| Khái niệm | Ký hiệu chuẩn | Tên runtime hiện có |
|---|---|---|
| MSE của một reconstruction ngẫu nhiên | \(s^{(m)}_{t,i}\) | **aux.point_score_samples[:,m,i]** |
| Raw input MSE trung bình theo Monte Carlo | \(\overline{s}_{t,i}\) | **raw_point_mse**, **raw_point_scores**, and default **point_scores** |
| Default input point MSE | \(\overline{s}_{t,i}\) | **point_scores**, **window_point_scores** |
| Raw latent prototype displacement | \(\overline{\ell}^{(c)}_{t,i}\) | Chưa có field point-level chuẩn; cần field riêng khi triển khai ablation |
| Raw latent point MSE | \(\overline{\ell}^{(c)}_{t,i}\) | A separate field is required for a latent-MSE run |
| Latent window score dùng cho triage | \(S_t^{(\mathrm{latent})}\) | **latent_window_score** |
| EWMA score trên timeline tuyệt đối | \(\widetilde{s}^{(r)}_n\) | **active_ewma_point_scores**, event **point_scores** |

**latent_window_score** hiện là score ở mức cửa sổ. Runtime không được xem nó là Thiết kế 3 nếu chưa định nghĩa rõ phép gộp từ \(\overline{\ell}^{(c)}_{t,i}\), cách chuẩn hóa và đường đi của truy vấn (query path).

## 6. Ranh giới với loss và RedLamp

**point_score_loss** in Stage A uses raw reconstruction error for training. It
does not use an inference score transform. The default inference score is the
same simple MSE with the identity transform.

RedLamp-style score không thuộc ba thiết kế trên. Score này kết hợp reconstruction và classification theo protocol riêng. Nếu chạy score này, phải ghi rõ đó là baseline hoặc ablation riêng.

## 7. Kế hoạch thí nghiệm cho Thiết kế 3

For the current default, compare raw input MSE and raw latent MSE directly.
Do not fit a sigmoid calibration. The center, scale, and sigmoid steps below
apply only to an explicitly marked legacy ablation.

Giữ nguyên checkpoint, seed, dữ liệu, số mẫu \(M\), cách gộp timeline, EWMA và bộ metric. Chỉ thay đại lượng dùng làm score và threshold artifact. Threshold artifact là file lưu threshold và thông tin calibration.

1. Chạy một smoke flow đầy đủ trên một tổ hợp development đã chốt. Kiểm tra shape **[B,M,T]** của \(\ell^{(c,m)}_{t,i}\), kiểm tra NaN/Inf, và kiểm tra thứ tự điểm có khớp input window hay không.
2. From the same output, create the default raw input MSE
   \(\overline{s}_{t,i}\) and raw latent MSE
   \(\overline{\ell}^{(c)}_{t,i}\). Create calibrated input or latent variants
   only for explicit legacy sigmoid ablations.
3. Fit only the threshold for each score space on clean validation. Fit center,
   scale, and a sigmoid only for an explicitly marked legacy ablation.
4. Báo cáo VUS-PR, VUS-ROC, Affiliation F1, point F1, precision, recall, số điểm được dự đoán là anomaly và số false positive. Báo cáo thêm Spearman correlation giữa input score và latent score.
5. Chỉ mở rộng sang full matrix sau khi smoke flow tạo đủ score timeline, threshold artifact, thông tin provenance (nguồn gốc và cấu hình của kết quả) và summary đủ để lập báo cáo.

## 8. Thay đổi thuật ngữ so với các spec cũ

| Ký hiệu cũ | Ký hiệu chuẩn | Loại thay đổi | Ngữ nghĩa |
|---|---|---|---|
| \(s^{(m)}_{bt}\) | \(s^{(m)}_{t,i}\) | Đổi ký hiệu | Giữ nguyên MSE mức điểm ở mẫu \(m\) |
| \(e_{bt}\) | \(\overline{s}_{t,i}\) | Đổi ký hiệu | Giữ nguyên raw point MSE trung bình theo Monte Carlo |
| \(q_{bt}\) | \(s^{(\mathrm{cal})}_{t,i}\) | Đổi ký hiệu | Giữ nguyên score sigmoid đã calibration |
| \(c\) | \(\mu^{(\mathrm{input})}_{\mathrm{val}}\) | Đổi ký hiệu | Giữ nguyên median của clean validation |
| \(\tau\) dùng làm calibration scale | \(\gamma^{(\mathrm{input})}_{\mathrm{val}}\) | Đổi ký hiệu | Tránh trùng với \(\tau_c,\tau_d\), là nhiệt độ truy vấn |
| Chưa có | \(\overline{\ell}^{(c)}_{t,i}\) | Đại lượng mới | Raw point-level latent prototype displacement |
| Chưa có | \(s^{(\mathrm{latent})}_{t,i}\) | Đại lượng mới | Latent prototype displacement sau calibration |

Các tên runtime không đổi trong lần chuẩn hóa này.
