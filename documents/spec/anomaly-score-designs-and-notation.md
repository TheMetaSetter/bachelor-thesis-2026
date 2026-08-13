# Tổng hợp thiết kế anomaly score và bộ ký hiệu chuẩn

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

### 3.1 Thiết kế 1 — raw point-level MSE trong input space

MSE (mean squared error, sai số bình phương trung bình) của điểm \(i\) ở mẫu thứ \(m\) là:

\[
s^{(m)}_{t,i}
=\frac{1}{C}
\left\|\mathbf{x}_{t,i}-\widehat{\mathbf{x}}^{(m)}_{t,i}\right\|_2^2.
\]

Raw score là score trước bước calibration. THESIS tính raw score bằng trung bình MSE của \(M\) mẫu:

\[
\boxed{
\overline{s}_{t,i}
=\frac{1}{M}\sum_{m=1}^{M}s^{(m)}_{t,i}
}.
\]

Quy trình phải lấy trung bình các MSE theo từng mẫu. Quy trình không được tính MSE giữa input và reconstruction trung bình. Hai cách tính này nói chung cho hai kết quả khác nhau.

### 3.2 Thiết kế 2 — calibrated point-level MSE trong input space

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

Score calibrated dùng hàm sigmoid \(\sigma\):

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

### 3.3 Thiết kế 3 — point-level prototype displacement trong latent space

Thiết kế 3 đo khoảng cách giữa vector latent trước và sau khi truy vấn continuous prototype bank. Độ lệch tại mẫu thứ (m) là:

\[
\ell^{(c,m)}_{t,i}
=\frac{1}{H}
\left\|\mathbf{z}_{t,i}-\widetilde{\mathbf{z}}^{(c,m)}_{t,i}\right\|_2^2.
\]

Raw latent score là trung bình của \(M\) mẫu:

\[
\boxed{
\overline{\ell}^{(c)}_{t,i}
=\frac{1}{M}\sum_{m=1}^{M}\ell^{(c,m)}_{t,i}
}.
\]

Nếu thí nghiệm cần score trong khoảng \((0,1)\), quy trình phải fit một bộ center và scale riêng trên clean validation:

\[
s^{(\mathrm{latent})}_{t,i}
=\sigma\left(
\frac{\overline{\ell}^{(c)}_{t,i}-\mu^{(\mathrm{latent})}_{\mathrm{val}}}
{\gamma^{(\mathrm{latent})}_{\mathrm{val}}}
\right).
\]

Quy trình không được dùng center hoặc scale của input-space MSE cho latent-space displacement. Hai score được tính trong hai không gian khác nhau và có thể có hai phân phối khác nhau.

## 4. EWMA và quyết định mức điểm

EWMA (exponentially weighted moving average, trung bình trượt có trọng số mũ) làm mượt score trên timeline tuyệt đối. Gọi \(r\) là lần xử lý cửa sổ và \(n\) là chỉ số tuyệt đối của điểm:

\[
\widetilde{s}^{(r)}_n
=\rho s^{(r)}_n+(1-\rho)\widetilde{s}^{(r-1)}_n.
\]

Với điểm cục bộ \((t,i)\), chỉ số tuyệt đối được xác định bởi \(n=\operatorname{start}(t)+i\). Giá trị \(s^{(r)}_n\) là một trong các score đã chọn cho run: \(\overline{s}_{t,i}\), \(s^{(\mathrm{cal})}_{t,i}\), \(\overline{\ell}^{(c)}_{t,i}\) hoặc \(s^{(\mathrm{latent})}_{t,i}\).

Mỗi run chỉ được chọn một score space. Không được trộn nhiều score nếu spec của run chưa định nghĩa phép kết hợp.

Quyết định anomaly cứng là:

\[
\widehat{a}_n=\mathbb{I}\left(\widetilde{s}_n>T_{\mathrm{point}}\right).
\]

Quy trình phải tính threshold từ clean validation trong đúng score space và dùng đúng chuỗi xử lý của run. Quy trình không được dùng test labels hoặc phân phối score của test để calibration.

## 5. Mapping giữa ký hiệu và tên runtime

| Khái niệm | Ký hiệu chuẩn | Tên runtime hiện có |
|---|---|---|
| MSE của một reconstruction ngẫu nhiên | \(s^{(m)}_{t,i}\) | **aux.point_score_samples[:,m,i]** |
| Raw input MSE trung bình theo Monte Carlo | \(\overline{s}_{t,i}\) | **raw_point_mse**, **raw_point_scores** hoặc giá trị trung gian của **point_scores** |
| Input score sau calibration | \(s^{(\mathrm{cal})}_{t,i}\) | **point_scores**, **window_point_scores** |
| Raw latent prototype displacement | \(\overline{\ell}^{(c)}_{t,i}\) | Chưa có field point-level chuẩn; cần field riêng khi triển khai ablation |
| Latent score sau calibration | \(s^{(\mathrm{latent})}_{t,i}\) | Chưa có field point-level chuẩn; cần field riêng khi triển khai ablation |
| Latent window score dùng cho triage | \(S_t^{(\mathrm{latent})}\) | **latent_window_score** |
| EWMA score trên timeline tuyệt đối | \(\widetilde{s}^{(r)}_n\) | **active_ewma_point_scores**, event **point_scores** |

**latent_window_score** hiện là score ở mức cửa sổ. Runtime không được xem nó là Thiết kế 3 nếu chưa định nghĩa rõ phép gộp từ \(\overline{\ell}^{(c)}_{t,i}\), cách chuẩn hóa và đường đi của truy vấn (query path).

## 6. Ranh giới với loss và RedLamp

**point_score_loss** trong Stage A dùng reconstruction error thô để huấn luyện. Loss này không dùng score inference đã calibration \(s^{(\mathrm{cal})}_{t,i}\). Lý do là các tham số calibration chỉ được fit sau training trên clean validation.

RedLamp-style score không thuộc ba thiết kế trên. Score này kết hợp reconstruction và classification theo protocol riêng. Nếu chạy score này, phải ghi rõ đó là baseline hoặc ablation riêng.

## 7. Kế hoạch thí nghiệm cho Thiết kế 3

Giữ nguyên checkpoint, seed, dữ liệu, số mẫu \(M\), cách gộp timeline, EWMA và bộ metric. Chỉ thay đại lượng dùng làm score và threshold artifact. Threshold artifact là file lưu threshold và thông tin calibration.

1. Chạy một smoke flow đầy đủ trên một tổ hợp development đã chốt. Kiểm tra shape **[B,M,T]** của \(\ell^{(c,m)}_{t,i}\), kiểm tra NaN/Inf, và kiểm tra thứ tự điểm có khớp input window hay không.
2. Từ cùng output, tạo bốn biến thể: raw input \(\overline{s}_{t,i}\), calibrated input \(s^{(\mathrm{cal})}_{t,i}\), raw latent \(\overline{\ell}^{(c)}_{t,i}\) và calibrated latent \(s^{(\mathrm{latent})}_{t,i}\). Hai biến thể latent đều thuộc Thiết kế 3.
3. Fit center, scale và threshold riêng cho từng score space bằng clean validation.
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
