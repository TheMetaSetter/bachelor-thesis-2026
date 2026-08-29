# Ba phiên bản dàn ý slides thuyết trình

## Phạm vi

Tài liệu này là dàn ý, chưa phải nội dung cuối của từng slide. Cả ba phiên bản đều phải giải thích kèm trực quan hóa. Mỗi slide cần có một hình chính, một thông điệp chính và một câu kết nối sang slide kế tiếp.

Ba phiên bản khác nhau ở mạch kể:

1. **Phiên bản A — từ tín hiệu đến quyết định:** đi theo luồng dữ liệu và runtime.
2. **Phiên bản B — lịch sử các vấn đề:** mỗi thành phần mới xuất hiện để giải quyết một thất bại của thành phần trước.
3. **Phiên bản C — từ bằng chứng thí nghiệm quay về phương pháp:** bắt đầu bằng câu hỏi thực nghiệm, sau đó truy ngược về thiết kế và cách kiểm chứng.

## Quy tắc slide: rất ít chữ, nhiều trực quan

- Mỗi slide chỉ có **một thông điệp chính**.
- Trên slide chỉ giữ từ khóa, công thức ngắn và nhãn của hình. Phần giải thích dài đưa vào lời nói hoặc speaker notes.
- Một slide nên có khoảng (60\%\)–(80\%) diện tích dành cho timeline, matrix, flowchart, state machine, scatter plot hoặc bảng nhỏ.
- Không đặt nguyên đoạn định nghĩa lên slide. Hiển thị một câu ngắn; phần còn lại nói bằng lời.
- Mỗi công thức phải trả lời một câu hỏi cụ thể. Không đưa nhiều công thức lên cùng một slide nếu chúng không tạo thành một chuỗi tính toán.
- Dùng màu nhất quán: xanh cho dữ liệu bình thường, đỏ cho anomaly, vàng cho gray zone, tím cho uncertainty, xám cho module frozen.
- Mỗi hình phải có mũi tên chỉ hướng đọc. Không dùng hình trang trí không mang thông tin.
- Cuối mỗi slide chỉ cần một câu chuyển tiếp, chẳng hạn: “Nhưng reconstruction vẫn có thể học nhầm bất thường; vì vậy cần memory.”

## Kinh nghiệm dàn trang và trình bày

Các quy tắc dưới đây áp dụng cho cả ba phiên bản. Chúng biến dàn ý nội dung thành bố cục slide cụ thể.

### Công thức bố cục mặc định

Quy tắc dùng xuyên suốt deck:

> **Title → 1–2 flat boxes → equations/evidence → one short takeaway.**

Với slide kỹ thuật, ưu tiên hai box phẳng:

1. **Box trên:** cơ chế hoặc định nghĩa.
2. **Box dưới:** kết quả, ý nghĩa hoặc quyết định runtime.

Không dùng nhiều box nhỏ nối bằng quá nhiều mũi tên. Nếu pipeline có nhiều bước, chuyển sang một trong ba dạng sau:

- `columns` ngang cho các nhánh xử lý song song;
- một bảng ngắn cho các thành phần và vai trò;
- các bước đánh số cho một chuỗi tính toán.

### Quy tắc chữ và công thức

- Không cố đưa toàn bộ nội dung báo cáo lên slide. Báo cáo lưu chi tiết; slide chỉ giữ lập luận mà người nghe cần theo trong vài giây.
- Ưu tiên công thức giúp hiểu logic. Công thức đầy đủ, khai triển dài và chứng minh để trong báo cáo hoặc backup slides.
- Các hằng số nhỏ nên nằm cùng dòng với công thức hoặc trong câu, chẳng hạn `K_top = 3`, `rho = 0.9`, `M = 10`.
- Hạn chế display equation nếu một inline equation đã đủ.
- Dùng `\scriptsize` cho slide kỹ thuật có nhiều thông tin. Chỉ dùng `\tiny` cho bảng lớn khi đã thử rút gọn nội dung.
- Không giảm font để cứu một slide có quá nhiều ý. Nếu vẫn chật, tách slide hoặc chuyển chi tiết sang backup.

Mẫu LaTeX có thể dùng trong Beamer:

```latex
\begin{columns}[T,onlytextwidth]
  \column{0.48\textwidth}
  \begin{block}{Mechanism}
    \vspace{-0.35em}
    \scriptsize
    ...
    \vspace{-0.35em}
  \end{block}

  \column{0.48\textwidth}
  \begin{block}{Meaning}
    \vspace{-0.35em}
    \scriptsize
    ...
    \vspace{-0.35em}
  \end{block}
\end{columns}
```

Có thể dùng khoảng `-0.5em` giữa hai box nếu cần giảm chiều cao, nhưng không ép quá mạnh vì sẽ làm slide khó đọc và dễ chạm nội dung.

### Visual grammar cho các slide liên quan

Những slide cùng loại phải dùng cùng một cách đọc. Bốn retrieval operators nên có cùng mẫu:

```text
selection / weighting
    -> retrieved representation
    -> interpretation
```

Ví dụ:

| Slide | Selection/weighting | Retrieved representation | Interpretation |
|---|---|---|---|
| Continuous retrieval | dense soft weights trên 32 prototypes | continuous retrieved vector | biểu diễn gần vùng normal memory |
| Discrete retrieval | stochastic top-(K_{top}), (K_{top}=3) | discrete retrieved vector | codeword evidence và class structure |
| PNN verification | lọc known-anomaly radius và recurrent signature | PNN mask | vùng được phép dùng cho adaptation |
| Monte Carlo inference | 10 stochastic samples | mean và variance | score và mức không chắc chắn |

Mỗi slide chỉ cần hiển thị phần khác nhau của ba bước này. Không thay đổi màu, vị trí hoặc thứ tự đọc giữa các slide.

### Chi tiết thiết kế phải được nói rõ

Khi lấy nội dung từ khóa luận, không chỉ chép công thức. Chọn các chi tiết giải thích vì sao thiết kế có ý nghĩa:

- `K_top = 3` cho discrete retrieval;
- (M=10) stochastic samples cho inference và uncertainty;
- EWMA dùng (0.9) score hiện tại và (0.1) score trước đó;
- chỉ `online_mlp_projector` được cập nhật;
- source encoder, memory, codebook, fusion heads và prediction heads bị frozen;
- triage chạy nhanh trước, verification chỉ chạy khi buffer đủ điều kiện;
- uncertainty được báo cáo nhưng không tham gia trực tiếp vào triage truth table.

Các chi tiết này nên đặt trong câu ngắn cạnh hình, không tách thành một slide định nghĩa riêng nếu chúng chỉ là tham số hoặc boundary của cơ chế.

### Slide kết quả: tách số liệu và diễn giải

Slide kết quả nên có hai vùng ngang:

- **Evidence:** bảng hoặc plot giữ số liệu quan trọng, tên entity, variant, seed, metric và support.
- **Interpretation:** một box `Conclusion` chỉ có một hoặc hai câu giải thích kết quả.

Không để người nghe tự đọc bảng rồi tự suy luận. Câu `Conclusion` phải được rút ra từ số liệu thật. Ví dụ về cách viết, chỉ dùng sau khi kiểm tra artifact:

- “O1 không cải thiện đáng kể so với O0 trên các entity được báo cáo.”
- “EWMA đóng góp phần lớn mức tăng trong online score, còn adaptation chỉ tạo thêm cải thiện ở một số vùng.”
- “VUS-PR tăng nhưng Affiliation F1 không tăng; cải thiện chủ yếu nằm ở ranking score, chưa chắc nằm ở event localization.”

Không trình bày các câu trên như kết luận có sẵn. Chúng là mẫu câu để điền sau khi đã đối chiếu từng entity, seed và protocol status.

### Không trộn kết quả với diễn giải

Trên mọi slide kết quả, dùng nhãn rõ ràng:

```text
Experimental result: metric, support, entity, seed, protocol
Interpretation: what the comparison suggests
Limitation: what the comparison cannot prove
```

Một con số tốt trên selected online range không đủ để kết luận online TTA tổng quát. Một VUS-PR cao cũng không tự động chứng minh event boundary được định vị tốt. Câu diễn giải phải đi cùng phạm vi dữ liệu và giới hạn của phép đo.

### Quy tắc cuối cùng

Slide không phải phiên bản thu nhỏ của báo cáo. Báo cáo trả lời chi tiết “tính như thế nào”; slide phải giúp người nghe trả lời nhanh ba câu hỏi:

1. Thành phần này làm gì?
2. Vì sao cần nó?
3. Bằng chứng nào cho thấy nó có hoặc chưa có ích?

## Phần mở đầu từ đoạn nháp mới

Ba hoặc bốn slide mở đầu này được đặt trước phần nội dung kỹ thuật trong cả ba phiên bản. Có thể giữ tiêu đề tiếng Anh và giải thích bằng tiếng Việt.

### Slide M0 — What is a time-series?

- **Chữ trên slide:** `A sequence of observations ordered by time.`
- **Công thức nhỏ:** (X\in\mathbb{R}^{T\times C}), (x_t\in\mathbb{R}^{C}).
- **Trực quan:** bên trái là timeline (t=1,2,\ldots,T); bên phải là một vector (x_t) gồm (C) channels. Một channel dùng đường nét liền, nhiều channels dùng ma trận màu.
- **Lời nói:** (T) là số mốc thời gian, (C) là số channels tại mỗi mốc. Các mốc có thể cách đều hoặc không cách đều.

### Slide M1 — What is a time-series anomaly?

- **Câu chính trên slide:** `A time point or time sequence that deviates significantly from expected behavior in its relevant context.`
- **Trực quan:** một timeline bình thường, một điểm bất thường và một đoạn bất thường; bên dưới ghi `context: past values / season / operating condition / related channels`.
- **Lời nói:** expected behavior không chỉ là một giá trị. Nó có thể gồm giá trị, pattern theo thời gian, trend, seasonality, duration và quan hệ giữa channels.
- **Ghi chú thận trọng:** (x_t\sim\mathcal{D}_{\mathrm{gen},t}) là cách mô tả lý thuyết. Trong thực tế ta không biết chính xác (mathcal{D}_{\mathrm{gen},t}), nên hệ thống dùng một quy tắc operational để ước lượng expected behavior.

### Slide M2 — What is the relevant context?

- **Chữ trên slide:** `The information needed to decide what is expected.`
- **Trực quan:** cửa sổ quá khứ, vị trí mùa vụ, operating condition và related channels cùng trỏ vào điểm đang đánh giá.
- **Lời nói:** thông tin không ảnh hưởng đến expected behavior của điểm đó không cần đưa vào context của quyết định.
- **Ranh giới phương pháp:** không đưa Granger causality vào flow chính. Runtime hiện tại không có bước ước lượng hoặc dùng Granger causality; nếu cần nói về Granger, chỉ đặt ở slide nền tảng hoặc appendix và ghi rõ đó không phải cơ chế đang chạy.

### Slide M3 — What is time-series anomaly detection?

- **Chữ trên slide:** `Observe → estimate expected behavior → measure deviation → decide.`
- **Công thức nhỏ:** (q_{t,i}=\sigma((\bar{s}_{t,i}-\mu_{val})/\gamma_{val})), (hat y_{t,i}=\mathbb{I}(q_{t,i}>\tau)).
- **Trực quan:** flow bốn bước từ cửa sổ dữ liệu đến point-level anomaly label.
- **Vai trò:** slide này nối định nghĩa chung với bài toán THESIS; từ đây mới giới thiệu prototype, uncertainty và online adaptation.

## Các sự thật phải khóa trước khi làm slides

- Bài toán chính là phát hiện bất thường trên chuỗi thời gian đa biến. Một cửa sổ có dạng \\(\mathbf{X}_t \in \mathbb{R}^{L \times D}\\), với cấu hình hiện tại (L=20) và (D=38).
- Baseline trong code là mô hình reconstruction và phân loại bất thường tổng hợp theo hướng RedLamp-inspired. Baseline không có prototype memory, task-specific prototype fusion, triage, verification buffer hoặc online adaptation.
- Các thành phần kế thừa gồm reconstruction error, synthetic anomaly classification, prototype memory, test-time adaptation, stochastic inference, uncertainty estimation, VUS và Affiliation metric. Không được gọi các thành phần này là đóng góp mới nếu chỉ áp dụng lại.
- Phần đề xuất cần làm nổi bật là cách tích hợp cụ thể: hai nhánh memory liên tục/rời rạc, fusion theo tác vụ, stochastic retrieval, two-stage offline training, Monte Carlo uncertainty, cùng online triage–verification–projector adaptation.
- Official offline THESIS anomaly score là point-wise reconstruction MSE lấy trung bình qua (M=10) stochastic samples rồi biến đổi bằng shifted-and-scaled logistic sigmoid. Official offline THESIS **không** dùng RedLamp-style score, không dùng classification probability để ghép score và không dùng test-dependent min–max normalization.
- Offline và online có hai timeline khác nhau. Offline dùng cửa sổ không chồng lấp, stride (20). Online dùng sliding window, stride (1), và EWMA theo absolute index với trọng số (0.9) cho score hiện tại và (0.1) cho score trước đó.
- Uncertainty là thông tin chẩn đoán. Nó không tham gia trực tiếp vào bảng quyết định triage bốn vùng trong runtime hiện tại.
- Trong online runtime, chỉ `online_mlp_projector` được phép cập nhật. Source encoder, memory, codebook, fusion heads và prediction heads bị đóng băng.

## Khối công thức dùng chung

### 1. Hình thức hóa bài toán

Đặt chuỗi dữ liệu:

\[
\mathbf{x}_{1:N} = (\mathbf{x}_1,\ldots,\mathbf{x}_N),
\qquad \mathbf{x}_t \in \mathbb{R}^{D}.
\]

Cửa sổ tại thời điểm (t):

\[
\mathbf{X}_t = [\mathbf{x}_{t-L+1},\ldots,\mathbf{x}_t]
\in \mathbb{R}^{L \times D}.
\]

Mô hình tạo reconstruction, classification probability và các điểm latent. Với stochastic sample thứ (m):

\[
\hat{\mathbf{X}}_t^{(m)},
\qquad s_{t,i}^{(m)}
=
\frac{1}{D}
\left\|\mathbf{x}_{t,i}-\hat{\mathbf{x}}_{t,i}^{(m)}\right\|_2^2.
\]

Điểm raw trung bình:

\[
\bar{s}_{t,i} = \frac{1}{M}\sum_{m=1}^{M}s_{t,i}^{(m)}.
\]

Tham số hiệu chỉnh được ước lượng trên clean validation:

\[
\mu_{\mathrm{val}}=\operatorname{median}(\bar{s}_{\mathrm{val}}),
\qquad
\gamma_{\mathrm{val}}=\frac{\operatorname{MAD}(\bar{s}_{\mathrm{val}})}{0.6745}.
\]

Official point score:

\[
q_{t,i}
=
\sigma\left(
\frac{\bar{s}_{t,i}-\mu_{\mathrm{val}}}{\gamma_{\mathrm{val}}}
\right).
\]

Quyết định điểm bất thường dùng ngưỡng được calibrate trên validation:

\[
\hat{y}_{t,i}=\mathbb{I}(q_{t,i}>\tau).
\]

**Trực quan nên dùng:** một cửa sổ (20 \times 38), mũi tên đi qua encoder, hai memory branches, reconstruction head, score calibration và quyết định nhị phân.

### 2. Tách phần kế thừa và phần đề xuất

| Thành phần | Kế thừa hoặc nền tảng | Phần cần gọi là đề xuất của bài |
|---|---|---|
| Reconstruction | Autoencoder và reconstruction residual | Dùng point-wise MSE làm score chính, sau đó calibration theo entity |
| Synthetic anomaly | RedLamp-inspired anomaly families và CARLA-informed injection mechanics | Dùng anomaly mask cho score loss và benchmark protocol |
| Prototype memory | Ý tưởng lưu normal prototypes để giảm over-generalization | Hai memory branches: continuous bank và discrete codebook, cùng task-specific fusion |
| Stochastic inference | Monte Carlo sampling, Gumbel-Softmax và sample variance | Vectorized retrieval cho hai branch với (M=10) và báo cáo nhiều loại uncertainty |
| Online adaptation | Test-time adaptation với frozen reference và trainable adapter | Triage bốn vùng, verification buffer, PNN mask, TTL và projector-only atomic update |
| Evaluation | VUS và Affiliation metrics | Áp dụng theo protocol point-level, không point adjustment và không dùng test labels để calibrate |

**Trực quan nên dùng:** sơ đồ ba lớp: “đã có trong nghiên cứu trước” → “được giữ lại trong baseline” → “được tích hợp hoặc cải tiến trong THESIS”.

### 3. Luồng offline

#### Offline training

```text
config và protocol
    -> validate
    -> Stage A multitask training
    -> chọn Stage-A best checkpoint
    -> khởi tạo continuous bank và discrete codebook từ train only
    -> lưu Stage-B initialization checkpoint
    -> đóng băng encoder và memory
    -> Stage B fusion finetuning
    -> chọn Stage-B best checkpoint
```

Stage A không dùng memory retrieval cuối cùng vì memory chưa được khởi tạo. Stage B huấn luyện fusion và prediction heads với encoder và memory đã đóng băng.

#### Offline inference

```text
Stage-B best checkpoint
    -> clean validation inference
    -> raw point MSE timeline
    -> median/MAD calibration
    -> offline threshold
    -> synthetic validation diagnostics
    -> test inference
    -> fixed-score metrics và artifacts
```

**Trực quan nên dùng:** hai swimlane riêng cho **training** và **inference**; checkpoint là vật thể trung gian nối hai lane.

### 4. Luồng online

#### Online inference

```text
causal window, stride 1
    -> source encoder đúng một lần
    -> projector tùy A0/A1/A2
    -> similarity logits tính một lần
    -> 10 stochastic retrieval samples
    -> mean score và variance
    -> absolute-index EWMA
    -> triage bốn vùng
    -> admission hoặc không admission
    -> verification nếu buffer đủ điều kiện
    -> event record và runtime checkpoint
```

#### Online training/adaptation

Online training không huấn luyện lại toàn bộ mô hình. Mỗi accepted event tạo một optimizer AdamW mới, tính một loss hữu hạn, backward một lần, clip gradient và cập nhật đúng một bước cho `online_mlp_projector`.

- **A0:** inference only, không projector update.
- **A1:** update bằng PNN-masked reconstruction sau verification.
- **A2:** update hard-old candidate hoặc verified PNN, có thể thêm source-consistency contrastive regularization.

**Trực quan nên dùng:** tách hai lane “online inference” và “online adaptation”; chỉ một mũi tên nhỏ từ adaptation quay lại cửa sổ tương lai.

### 5. Vì sao cần triage và verification buffer

Triage và verification không làm cùng một việc.

- **Triage** là quyết định nhanh trên cửa sổ hiện tại bằng hai điểm: (S_t^{(input)}) và (S_t^{(latent)}).
- **Verification buffer** giữ các gray-zone windows không chồng lấp để chờ đủ bằng chứng liên cửa sổ.
- Verification mã hóa lại các entry bằng frozen source, lọc known-anomaly codewords, kiểm tra recurrent signatures và tạo PNN mask.
- Chỉ verified PNN hoặc hard-old candidate hợp lệ mới được phép tạo adaptation update.

Bảng triage:

\[
\begin{array}{ll}
S_t^{(input)} \le B_{window} & \text{normal}\\
S_t^{(input)} > B_{window},\ S_t^{(latent)} \le A_{low} & \text{hard old-normality}\\
S_t^{(input)} > B_{window},\ A_{low}<S_t^{(latent)}\le A_{high} & \text{gray zone}\\
S_t^{(input)} > B_{window},\ S_t^{(latent)} > A_{high} & \text{strong anomaly}
\end{array}
\]

Không gộp hai pha vì ba lý do:

1. Triage phải giữ latency thấp cho mọi cửa sổ; verification chỉ chạy khi buffer có ít nhất tám entry và có entry mới.
2. Gray zone thiếu bằng chứng để cập nhật ngay. Buffer cho phép kiểm tra recurrence trên các cửa sổ không chồng lấp.
3. Nếu gộp, mỗi cửa sổ đều phải chạy frozen re-encoding, signature matching và PNN computation; chi phí tăng ngay cả khi cửa sổ rõ ràng là normal hoặc strong anomaly.

**Trực quan nên dùng:** trục hai chiều (S^{(input)})–(S^{(latent)}), sau đó minh họa gray-zone entries đi vào buffer và chỉ một nhóm được chuyển thành PNN mask.

### 6. Uncertainty

Với (M=10) stochastic samples:

\[
\bar{v}=\frac{1}{M}\sum_{m=1}^{M}v^{(m)},
\qquad
\widehat{\operatorname{Var}}(v)
=
\frac{1}{M-1}
\sum_{m=1}^{M}(v^{(m)}-\bar{v})^2.
\]

Slides phải tách:

- point anomaly score variance;
- reconstruction variance;
- continuous retrieval variance;
- discrete retrieval variance;
- classification probability variance.

Không tính variance trên class ID nguyên. Classification uncertainty hiện là window-level.

**Trực quan nên dùng:** cùng một cửa sổ đi qua 10 nhánh stochastic, tạo 10 điểm; độ phân tán của các điểm là uncertainty.

### 7. VUS-PR và Affiliation F1

#### VUS-PR

1. Nhận ground-truth point labels (mathbf{y}) và point scores (mathbf{s}).
2. Chọn một tập hữu hạn anomaly thresholds (	heta).
3. Với mỗi (	heta), tạo binary prediction (hat{y}_n=\mathbb{I}(s_n>\theta)).
4. Với mỗi buffer length (ell=0,\ldots,\ell_{max}), mở rộng ground-truth anomaly spans.
5. Tính range-based precision và recall tại từng cặp ((\ell,\theta)).
6. Tạo PR curve theo (	heta), tính AUC-PR cho từng (ell).
7. Lấy trung bình các AUC-PR theo các buffer lengths để nhận VUS-PR.

\[
\operatorname{VUS\text{-}PR}
=
\frac{1}{\ell_{max}+1}
\sum_{\ell=0}^{\ell_{max}}
\operatorname{AUC\text{-}PR}^{(\ell)}.
\]

#### Affiliation F1

1. Dùng một threshold đã chọn để biến score thành binary prediction.
2. Gom các điểm dương liên tiếp thành predicted events; làm tương tự cho ground-truth events.
3. Tạo affiliation zone quanh mỗi ground-truth event bằng các midpoint giữa các event lân cận.
4. Cắt predicted events theo từng affiliation zone.
5. Tính affiliation precision dựa trên phần giao và khoảng cách tới ground truth.
6. Tính affiliation recall dựa trên mức độ bao phủ và vị trí của prediction trong zone.
7. Lấy trung bình precision/recall theo event rồi tính:

\[
F1_{aff}
=
\frac{2P_{aff}R_{aff}}{P_{aff}+R_{aff}}.
\]

**Trực quan nên dùng:** một timeline có ground-truth spans, predicted spans, buffer zones và các phần giao; không chỉ dùng confusion matrix điểm.

## Bảng dữ liệu thí nghiệm nên đưa vào slides

Các số dưới đây tương ứng với cấu hình hiện tại: (L=20), train stride (1), validation/test stride (20), validation chiếm (20\%) phần train gốc. Mỗi entity có 38 biến.

| Entity | Train points | Validation points | Test points | Train windows | Validation windows | Test windows | Test anomaly points | Test anomaly ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `machine-1-6` | 18,951 | 4,737 | 23,689 | 18,932 | 236 | 1,184 | 3,708 | 15.65% |
| `machine-3-4` | 18,950 | 4,737 | 23,687 | 18,931 | 236 | 1,184 | 977 | 4.12% |
| `machine-3-9` | 22,971 | 5,742 | 28,713 | 22,952 | 287 | 1,435 | 303 | 1.06% |

Online stream hiện dùng các đoạn nhân quả được khóa trong protocol:

| Entity | Absolute stream range | Stream points | Online windows, stride 1 |
|---|---:|---:|---:|
| `machine-1-6` | `[146, 2200)` | 2,054 | 2,035 |
| `machine-3-4` | `[2634, 6116)` | 3,482 | 3,463 |
| `machine-3-9` | `[1099, 10807)` | 9,708 | 9,689 |

### Vì sao chọn ba entity này

Slides nên nói theo bằng chứng, không nói “đại diện cho toàn bộ SMD” nếu chưa có phân tích đầy đủ.

- Đây là ba entity được khóa trong `SMD_BENCHMARK_ENTITIES` và được dùng nhất quán trong offline/online benchmark.
- Chúng tạo ra ba mức anomaly prevalence khác nhau: `machine-1-6` cao, `machine-3-4` trung bình, `machine-3-9` rất thấp.
- Chúng có độ dài test khác nhau và các mức train–normal-test distribution drift khác nhau. Phân tích KL hiện có cho mean KL lần lượt khoảng (4.05), (0.98), (4.72) cho `machine-1-6`, `machine-3-4`, `machine-3-9`.
- Vì vậy, bộ ba này phù hợp để kiểm tra sensitivity với mất cân bằng, độ dài chuỗi và distribution shift; nó không đủ để kết luận cho mọi entity SMD.

### Vì sao thí nghiệm online dùng ít dữ liệu

Không nên nói raw SMD là dữ liệu nhỏ. Raw test timelines vẫn có hàng chục nghìn điểm. Phần online chỉ dùng các đoạn ngắn hơn quanh các anomaly spans đã chọn, có khoảng đệm trước và sau.

Mục tiêu của lát cắt nhỏ là kiểm tra causal ordering, triage, buffer admission, verification, TTL và projector update với chi phí có thể kiểm soát. Đây là mechanism-focused evaluation, chưa phải bằng chứng đủ cho khả năng tổng quát hóa rộng.

## Phiên bản A — Từ tín hiệu đến quyết định

Đây là phiên bản phù hợp nhất khi hội đồng cần hiểu runtime flow và cách các tensor biến đổi.

### Slide A0 — Mở đầu bằng một cửa sổ dữ liệu

- Dùng M0–M3 theo thứ tự: time-series → anomaly → context → TSAD.
- **Mạch trực quan:** bắt đầu từ một vector tại thời điểm (t), mở rộng thành (T\times C), cắt thành cửa sổ (L\times D), rồi đi vào runtime.
- **Câu chuyển:** “Sau khi xác định anomaly bằng độ lệch, câu hỏi tiếp theo là mô hình ước lượng expected behavior bằng cách nào?”

### Slide A1 — Câu hỏi nghiên cứu

- **Thông điệp:** Một hệ thống TSAD phải phát hiện bất thường, biết mức độ không chắc chắn và thích ứng khi dữ liệu thay đổi.
- **Nội dung:** Nêu ba khó khăn: over-generalization, distribution shift và false alarm.
- **Trực quan:** Một timeline có normal span, anomaly span, gray-zone span và điểm score dao động.

### Slide A2 — Hình thức hóa bài toán

- **Nội dung:** Dùng (mathbf{x}_{1:N}), (mathbf{X}_t), (q_{t,i}), (hat y_{t,i}) trong khối công thức dùng chung.
- **Trực quan:** Ma trận cửa sổ (20\times38) và đầu ra point score.

### Slide A3 — Phương pháp nền và ranh giới đóng góp

- **Nội dung:** So sánh RedLamp-inspired baseline với THESIS. Nói rõ phần kế thừa và phần mới.
- **Trực quan:** Bảng baseline → memory/fusion → stochastic inference → online TTA.
- **Câu phải nói:** RedLamp-style scoring không phải official offline THESIS score.

### Slide A4 — Kiến trúc THESIS

- **Nội dung:** Encoder → continuous prototype bank, discrete codebook → task-specific fusion → reconstruction/classification heads.
- **Trực quan:** Hai nhánh màu khác nhau, đánh dấu memory và prediction heads.

### Slide A5 — Offline training: Stage A

- **Nội dung:** Train encoder với reconstruction, classification, contrastive và tùy chọn point-score loss theo O0/O1.
- **Trực quan:** Swimlane training; memory để trống ở Stage A.

### Slide A6 — Offline training: khởi tạo memory và Stage B

- **Nội dung:** Dùng train-only latent tokens để tạo 32 continuous prototypes và 60 discrete codewords; lưu initialization checkpoint; freeze encoder/memory; train fusion heads ở Stage B.
- **Trực quan:** Checkpoint boundary và dấu khóa trên encoder/memory.

### Slide A7 — Offline inference và calibration

- **Nội dung:** Clean validation → raw MSE → median/MAD → sigmoid score → threshold; sau đó mới test inference.
- **Trực quan:** Timeline validation tạo threshold rồi mũi tên sang test.

### Slide A8 — Online inference theo một causal window

- **Nội dung:** Source encoder một lần, projector tùy A0/A1/A2, stochastic retrieval 10 lần, mean/variance, EWMA.
- **Trực quan:** Một cửa sổ trượt stride 1 và absolute-index score map.

### Slide A9 — Triage bốn vùng

- **Nội dung:** Trình bày bảng (S^{(input)})–(S^{(latent)}); uncertainty chưa tham gia truth table.
- **Trực quan:** Scatter plot hai ngưỡng (B_{window}), (A_{low}), (A_{high}).

### Slide A10 — Verification buffer

- **Nội dung:** Chỉ gray-zone được admission; entry không chồng lấp; cycle bắt đầu khi buffer có ít nhất 8 entry và có entry mới; TTL giảm sau cycle.
- **Trực quan:** Buffer queue với trạng thái `unresolved`, `verified`, `adapted`, `removed`.

### Slide A11 — Vì sao không gộp triage và verification

- **Nội dung:** So sánh latency, lượng tính toán, độ mạnh của bằng chứng và rủi ro update sai.
- **Trực quan:** Bảng “mọi window” versus “chỉ gray-zone đủ điều kiện”.

### Slide A12 — Online training/adaptation

- **Nội dung:** A0 inference only; A1 verified PNN; A2 hard-old hoặc verified PNN; chỉ projector cập nhật một bước.
- **Trực quan:** Frozen modules màu xám, projector màu đỏ, future windows nhận projector mới.

### Slide A13 — Độ bất định

- **Nội dung:** Công thức mean và unbiased variance; tách score, retrieval, reconstruction, classification uncertainty.
- **Trực quan:** 10 stochastic outputs và violin/strip plot của 10 scores.

### Slide A14 — VUS-PR

- **Nội dung:** Bảy bước tính VUS-PR.
- **Trực quan:** Từ timeline score → nhiều threshold → nhiều buffer lengths → nhiều PR curves → average surface.

### Slide A15 — Affiliation F1

- **Nội dung:** Event extraction, affiliation zones, probability precision/recall và F1.
- **Trực quan:** Timeline event-level có zone và phần giao.

### Slide A16 — Dữ liệu và protocol

- **Nội dung:** Đưa hai bảng số liệu ở trên; nói rõ stride offline/online và test labels chỉ dùng cho metrics.
- **Trực quan:** Bảng train/validation/test và một timeline stream.

### Slide A17 — Kết quả và cách đọc thận trọng

- **Nội dung:** Báo cáo VUS-PR, Affiliation F1, precision/recall, score variance và ablation O0/O1, A0/A1/A2.
- **Trực quan:** Một bảng kết quả kèm cột support, positive ratio, protocol status và uncertainty.
- **Cảnh báo:** Không suy luận “điểm cao” thành “tổng quát tốt” khi online chỉ dùng selected ranges hoặc khi evaluation bị truncate.

### Slide A18 — Kết luận và giới hạn

- **Nội dung:** Tóm tắt đóng góp, sau đó nêu giới hạn: ba entity, online slices nhỏ, projector-only adaptation, PNN/verification phụ thuộc threshold và protocol.
- **Trực quan:** Một sơ đồ claim → evidence → limitation.

## Phiên bản B — Lịch sử các vấn đề và câu trả lời

Đây là phiên bản có mạch kể gần với lịch sử khoa học: mỗi bước mới xuất hiện vì bước trước đó chưa giải quyết được một vấn đề.

### Slide B0 — Mở đầu bằng sự thay đổi của một tín hiệu

- Dùng M0–M3 nhưng kể theo hai timeline: một timeline có expected behavior ổn định và một timeline xuất hiện deviation.
- **Chữ trên slide:** `Normal behavior is contextual.`
- **Mạch kể:** không bắt đầu bằng kiến trúc; bắt đầu bằng việc một deviation chỉ có ý nghĩa khi biết context của nó.
- **Câu chuyển:** “Khi context được học không đầy đủ, reconstruction model có thể học nhầm anomaly thành normal.”

### Slide B1 — Khởi đầu: reconstruction residual

- **Thông điệp:** Mô hình dự đoán trạng thái bình thường rồi dùng độ lệch để phát hiện bất thường.
- **Trực quan:** Dữ liệu thật và reconstruction trên cùng một cửa sổ.

### Slide B2 — Vấn đề thứ nhất: mô hình học cả bất thường

- **Nội dung:** Giải thích contaminated training data và over-generalization.
- **Trực quan:** Hai reconstruction: một mô hình tái tạo bất thường quá tốt, một mô hình tạo sai số lớn ở anomaly span.

### Slide B3 — Câu trả lời: normal prototype memory

- **Nội dung:** Memory bank lưu normal prototypes và hỗ trợ query.
- **Trực quan:** Latent tokens, vùng normal prototypes và query path.

### Slide B4 — Vấn đề thứ hai: một loại memory chưa đủ

- **Nội dung:** Point, interval và period anomalies có cấu trúc khác nhau.
- **Trực quan:** Ba anomaly spans và hai loại memory: continuous/discrete.

### Slide B5 — Câu trả lời của THESIS: dual memory và fusion

- **Nội dung:** Continuous bank 32 prototypes, discrete codebook 60 codewords, stochastic retrieval và task-specific fusion.
- **Trực quan:** Hai nhánh hội tụ vào reconstruction/classification task heads.

### Slide B6 — Phần cũ và phần mới

- **Nội dung:** Dùng bảng kế thừa/đề xuất ở trên; đánh dấu những thành phần chỉ được áp dụng lại.
- **Trực quan:** Màu xanh cho prior, màu đỏ cho proposed, màu vàng cho integration boundary.

### Slide B7 — Offline training giải quyết vấn đề học memory

- **Nội dung:** Stage A học representation trước; memory khởi tạo sau Stage A; Stage B học fusion khi memory đã freeze.
- **Trực quan:** Dòng thời gian hai stage với checkpoint boundary.

### Slide B8 — Offline inference giải quyết vấn đề threshold leakage

- **Nội dung:** Calibration chỉ trên clean validation; test labels chỉ xuất hiện sau khi score cố định.
- **Trực quan:** Dấu chặn giữa validation calibration và test evaluation.

### Slide B9 — Vấn đề thứ ba: môi trường triển khai thay đổi

- **Nội dung:** Hardware/sensor upgrade tạo distribution shift.
- **Trực quan:** Hai phân phối latent trước và sau shift.

### Slide B10 — Câu trả lời: online inference và triage

- **Nội dung:** Causal window → score → EWMA → bốn triage regions.
- **Trực quan:** Stream timeline và mặt phẳng triage.

### Slide B11 — Vấn đề thứ tư: gray zone chưa đủ bằng chứng

- **Nội dung:** Một cửa sổ gray-zone không đủ để kết luận normal mới hay anomaly.
- **Trực quan:** Ba cửa sổ gray-zone cách nhau theo thời gian, chưa được nối thành một event.

### Slide B12 — Câu trả lời: verification buffer và PNN

- **Nội dung:** Non-overlap, capacity 8, recurrent signature, known-anomaly filter, PNN mask và TTL.
- **Trực quan:** State machine của buffer.

### Slide B13 — Vì sao triage và verification phải tách

- **Nội dung:** Triage nhanh và local; verification chậm hơn và cần nhiều entry; gộp hai pha làm tăng chi phí và tăng nguy cơ update sai.
- **Trực quan:** So sánh đường đi “fast path” và “deferred evidence path”.

### Slide B14 — Online training là adaptation có kiểm soát

- **Nội dung:** Chỉ projector được train; A0/A1/A2; atomic event gồm assert, loss, backward, gradient check, clip và one optimizer step.
- **Trực quan:** Frozen boundary và một update token.

### Slide B15 — Vấn đề thứ năm: mô hình có thể không chắc chắn

- **Nội dung:** 10 stochastic samples, mean score, unbiased sample variance; uncertainty là diagnostics.
- **Trực quan:** 10 outputs hội tụ về mean và tỏa ra theo variance.

### Slide B16 — Vấn đề thứ sáu: metric điểm không đủ cho anomaly spans

- **Nội dung:** Precision/Recall/F1 bị ảnh hưởng bởi imbalance; range-based metric có thể quá rộng.
- **Trực quan:** Một prediction chạm một phần anomaly span nhưng metric điểm và event metric cho kết quả khác nhau.

### Slide B17 — Câu trả lời: VUS-PR và Affiliation F1

- **Nội dung:** Đặt hai flow tính metric cạnh nhau; VUS thay đổi threshold và buffer length, Affiliation đánh giá event và vị trí.
- **Trực quan:** Hai timeline song song.

### Slide B18 — Bằng chứng, giới hạn và kết luận lịch sử

- **Nội dung:** Đưa bảng dữ liệu, lý do chọn ba entity, lý do dùng online slices nhỏ, rồi kết luận rằng phương pháp giải quyết từng failure mode nhưng chưa chứng minh tổng quát rộng.
- **Trực quan:** Chuỗi “vấn đề → giải pháp → bằng chứng → giới hạn”.

## Phiên bản C — Từ bằng chứng thí nghiệm quay về phương pháp

Đây là phiên bản phù hợp khi hội đồng quan tâm trước hết đến tính công bằng của thí nghiệm, cách đọc kết quả và giới hạn của claim.

### Slide C0 — Mở đầu bằng câu hỏi đo lường

- Dùng M0–M3 nhưng đặt ngay một câu hỏi kiểm chứng dưới flow: `What is observed? What is expected? What is measured?`
- **Trực quan:** một data card (X\in\mathbb{R}^{T\times C}), một anomaly span và một score timeline; không đặt đoạn định nghĩa dài.
- **Mạch kể:** định nghĩa time-series và anomaly chỉ là tiền đề; thesis phải chứng minh score, decision và metric có khớp với nhau.
- **Câu chuyển:** “Vì vậy, trước khi xem model, cần khóa dữ liệu, threshold và metric.”

### Slide C1 — Claim nào cần kiểm chứng

- **Nội dung:** Tách ba claim: prototype/fusion có giúp offline không; online TTA có giúp sau distribution shift không; uncertainty có cung cấp thông tin hữu ích không.
- **Trực quan:** Ba claim cards, mỗi card có metric và ablation tương ứng.

### Slide C2 — Bối cảnh dữ liệu và vì sao không được phóng đại claim

- **Nội dung:** Đưa bảng điểm thời gian, window counts, anomaly ratios và online selected ranges.
- **Trực quan:** Data cards cho ba entity.
- **Thông điệp:** Online slice nhỏ phục vụ kiểm tra causal mechanism, không phải bằng chứng đủ cho toàn bộ SMD.

### Slide C3 — Vì sao chọn `machine-1-6`, `machine-3-4`, `machine-3-9`

- **Nội dung:** Entity set được khóa trong protocol; ba mức anomaly prevalence; độ dài khác nhau; drift KL khác nhau.
- **Trực quan:** Bar chart anomaly ratio và bar chart mean KL.
- **Giới hạn:** Không gọi bộ ba là đại diện thống kê cho 28 entity nếu chưa có sampling analysis.

### Slide C4 — Hình thức hóa đầu vào, đầu ra và quyết định

- **Nội dung:** Công thức (mathbf{X}_t), reconstruction, raw MSE, calibration, threshold và (hat y).
- **Trực quan:** Một mẫu dữ liệu đi từ input đến label.

### Slide C5 — Đối tượng so sánh

- **Nội dung:** RedLamp-inspired baseline, THESIS offline O0/O1, online A0/A1/A2, deterministic/stochastic ablation.
- **Trực quan:** Experiment matrix nhỏ, không tạo Cartesian product không cần thiết.

### Slide C6 — Cải tiến dựa trên cái gì

- **Nội dung:** Reconstruction, synthetic anomaly classification, prototype memory, test-time adaptation, stochastic uncertainty là nền tảng trước; THESIS đề xuất cách tích hợp và ranh giới runtime cụ thể.
- **Trực quan:** Prior work stack và THESIS contribution overlay.

### Slide C7 — Official score và calibration protocol

- **Nội dung:** Nhấn mạnh THESIS score khác RedLamp-style score; clean validation calibrates, test labels metrics-only; offline/online thresholds tách nhau.
- **Trực quan:** Score provenance graph từ raw MSE đến calibrated score.

### Slide C8 — Offline training flow

- **Nội dung:** Stage A, memory initialization, Stage B; nêu rõ train-only data boundary và frozen boundary.
- **Trực quan:** Sankey/checkpoint flow với vùng cấm validation/test trong memory initialization.

### Slide C9 — Offline inference flow

- **Nội dung:** Validation calibration → synthetic validation diagnostics → test evaluation → artifact export.
- **Trực quan:** Timeline có nhãn “được phép calibrate” và “chỉ được đánh giá”.

### Slide C10 — Online inference flow

- **Nội dung:** Causal sliding window, source-once, projector, 10 stochastic samples, score/uncertainty/EWMA/triage.
- **Trực quan:** One-event runtime flow.

### Slide C11 — Triage và verification buffer

- **Nội dung:** Bốn vùng triage, gray-zone admission, cycle trigger, PNN verification, TTL.
- **Trực quan:** Triage plane nối vào buffer state machine.

### Slide C12 — Tại sao không gộp hai pha

- **Nội dung:** Tách latency, chi phí, bằng chứng và quyền update; chỉ verification mới kiểm tra recurrence/non-overlap.
- **Trực quan:** Cost per window của fast path và deferred path.

### Slide C13 — Online training/adaptation

- **Nội dung:** A0/A1/A2; projector-only update; atomic update; frozen parameter allowlist.
- **Trực quan:** Gradient flow chỉ đi vào projector.

### Slide C14 — Tính uncertainty

- **Nội dung:** 10 stochastic retrieval samples; mean; unbiased variance; score/retrieval/reconstruction/classification uncertainty.
- **Trực quan:** Bảng tensor shapes và plot phân tán.

### Slide C15 — Tính VUS-PR

- **Nội dung:** Threshold sweep, buffered ground truth, range precision/recall, PR AUC theo buffer length, average.
- **Trực quan:** Từ một curve thành nhiều curve rồi thành surface/average.

### Slide C16 — Tính Affiliation F1

- **Nội dung:** Binary events, affiliation zones, partition, probability precision/recall, harmonic mean.
- **Trực quan:** Event timeline có zone boundaries và phần overlap.

### Slide C17 — Kết quả, ablation và cách báo cáo thận trọng

- **Nội dung:** Báo cáo theo entity, variant, seed, metric, support, anomaly ratio, uncertainty và protocol status. So sánh O0/O1 và A0/A1/A2 đúng reset state.
- **Trực quan:** Bảng kết quả có cột “evidence strength” và “limitation”.
- **Cấm:** Không dùng một kết quả tốt trên selected range để kết luận online TTA tổng quát; không che giấu truncated evaluation hoặc single-class regime.

### Slide C18 — Kết luận: claim nào được phép nói

- **Nội dung:** Chỉ claim những gì dữ liệu chứng minh: runtime đã tách được offline/online, uncertainty đã được định lượng, metrics đã phù hợp hơn với spans; mức cải thiện và khả năng tổng quát phải gắn với entity, seed, protocol và support.
- **Trực quan:** Bảng “đã chứng minh / chưa chứng minh / cần thí nghiệm thêm”.

## Ma trận kiểm tra checklist

| Mục checklist | Phiên bản A | Phiên bản B | Phiên bản C |
|---|---|---|---|
| Công thức hình thức hóa bài toán | A2 | B1 | C4 |
| Định nghĩa time-series, anomaly, context và TSAD | A0 | B0 | C0 |
| Phương pháp nền được cải tiến | A3 | B1–B5 | C5–C6 |
| Phần thuộc nghiên cứu trước | A3 | B6 | C6 |
| Phần đề xuất mới | A3–A4 | B5–B6 | C6 |
| Offline training flow | A5–A6 | B7 | C8 |
| Offline inference flow | A7 | B8 | C9 |
| Online inference flow | A8–A10 | B10–B12 | C10–C11 |
| Online training/adaptation flow | A12 | B14 | C13 |
| Tách training và inference | A5–A8, A12 | B7–B10, B14 | C7–C13 |
| Triage và verification buffer | A9–A11 | B10–B13 | C11–C12 |
| Vì sao không gộp hai pha | A11 | B13 | C12 |
| VUS-PR | A14 | B17 | C15 |
| Affiliation F1 | A15 | B17 | C16 |
| Độ bất định | A13 | B15 | C14 |
| Số điểm và số cửa sổ | A16 | B18 | C2 |
| Lý do chọn ba entity | A16 | B18 | C3 |
| Lý do dùng ít dữ liệu online | A16–A18 | B18 | C2, C17 |
| Báo cáo thận trọng và hạn chế | A17–A18 | B18 | C17–C18 |

## Khuyến nghị chọn phiên bản

- Chọn **A** nếu mục tiêu chính là giải thích code/runtime cho hội đồng kỹ thuật.
- Chọn **B** nếu muốn bài nói dễ theo dõi và nhấn mạnh động cơ lịch sử của từng thiết kế.
- Chọn **C** nếu hội đồng quan tâm mạnh đến fairness, protocol, ablation và giới hạn của kết quả.

Với đề tài hiện tại, **A** là dàn ý an toàn nhất cho phần trình bày chính. Có thể lấy cách mở đầu của **B** và cách báo cáo kết quả thận trọng của **C** để làm phiên bản cuối.

## Nguồn nội bộ cần đối chiếu khi dựng slides

- `documents/spec/full-spec-v3.md`
- `documents/notes/offline_runtime_and_data_flow.md`
- `documents/notes/thesis_online_tta_prepare_event_runtime_and_data_flow.md`
- `documents/notes/online_runtime_flow_debug.md`
- `documents/spec/online_benchmark_contract.md`
- `src/models/baseline_impl/redlamp_baseline.py`
- `src/models/thesis_multitask.py`
- `src/metrics/pointwise.py`
- `src/metrics/affiliation.py`
- `src/protocols/smd_benchmark_protocol.py`
- `scripts/analysis/rank_smd_train_test_normal_drift.py`
