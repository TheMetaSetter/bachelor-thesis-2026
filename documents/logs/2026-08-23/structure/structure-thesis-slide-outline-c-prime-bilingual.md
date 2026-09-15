# Dàn ý trình bày phiên bản C′

Mục tiêu của phiên bản này là làm rõ phần đóng góp, trình bày bằng chứng thực nghiệm và nêu thẳng hạn chế về tỉ lệ báo động giả.

Mỗi trang chỉ giữ một thông điệp. Dùng bố cục ngang, hai khung phẳng, ít chữ và một câu kết luận ngắn.

## Phiên bản thuần tiếng Việt

### Trang 1 — Tên đề tài và câu hỏi nghiên cứu

- **Hình:** tên đề tài, một sơ đồ chuỗi thời gian.
- **Nói:** Làm sao phát hiện bất thường khi dữ liệu thay đổi và mô hình chưa chắc chắn?
- **Chốt:** Khóa luận đề xuất một khung đa nhiệm cho phát hiện bất thường chuỗi thời gian.

### Trang 2 — Bài toán

- **Hình:** chuỗi thời gian → cửa sổ → nhãn bình thường hoặc bất thường.
- **Công thức:** \(X_t \in \mathbb{R}^{L \times D}\), \(s_t=f(X_t)\).
- **Chốt:** Mục tiêu là phát hiện đúng bất thường trên từng cửa sổ và từng điểm thời gian.

### Trang 3 — Hai hạn chế cần giải quyết

Hai khung ngang:

1. Mô hình chưa biểu diễn rõ mức độ không chắc chắn.
2. Mô hình khó thích ứng với dòng dữ liệu không dừng mà vẫn giữ chi phí thấp.

**Chốt:** Cần vừa ước lượng độ bất định, vừa thích ứng có kiểm soát.

### Trang 4 — Ranh giới đóng góp

Hai khung phẳng:

| Thành phần kế thừa | Phần khóa luận đề xuất hoặc phối hợp |
|---|---|
| Bộ mã hoá tích chập dùng chung; tái tạo; truy vấn bộ nhớ; các độ đo đánh giá | Hai toán tử truy vấn; truy vấn ngẫu nhiên để định lượng độ bất định; thích ứng trực tuyến; quy tắc tam phân và bộ đệm xác minh |

**Chốt:** Tính mới nằm ở thiết kế phối hợp và luồng vận hành, không nằm ở từng thành phần quen thuộc.

### Trang 5 — Kiến trúc tổng quát

- **Hình:** bộ mã hoá dùng chung → hai bộ nhớ → hai đầu ra.
- **Ghi ngắn:** tập nguyên mẫu liên tục; tập mã vec-tơ rời rạc; tác vụ tái tạo chuỗi; tác vụ phân loại đa lớp.
- **Chi tiết:** 32 nguyên mẫu liên tục; 60 nguyên mẫu rời rạc thuộc 12 lớp.
- **Chốt:** Hai tác vụ dùng chung đặc trưng nhưng giữ mục tiêu riêng.

### Trang 6 — Luồng dữ liệu của pha ngoại tuyến

Vẽ hai hàng. Tách rõ bước chuẩn bị dữ liệu, bước tiêm và bước đưa vào mạng:

**Chuẩn bị dữ liệu**

1. Chuỗi dữ liệu gốc → làm sạch → tính trung bình và độ lệch chuẩn trên tập huấn luyện.
2. Dùng bộ chuẩn hoá đó cho các tập huấn luyện, xác nhận và kiểm tra.
3. Chuỗi đã chuẩn hoá → WindowDataset → các cửa sổ \(X\in\mathbb{R}^{L\times D}\) → lô dữ liệu \(x\in\mathbb{R}^{B\times L\times D}\).

**Huấn luyện một lô**

4. Lô cửa sổ sạch → training step → chuẩn bị lô trước khi lan truyền xuôi.
5. Nếu bật tiêm, SyntheticAnomalyInjector tạo nhãn lớp cho từng cửa sổ.
6. Cửa sổ lớp bình thường giữ nguyên; cửa sổ lớp bất thường được tiêm một họ bất thường vào một đoạn liên tục và một số kênh.
7. Injector ghi lại \(x\) mới, classification labels, synthetic anomaly mask, nhãn điểm và siêu dữ liệu tiêm.
8. Lô sau tiêm → bộ mã hoá tích chập dùng chung → ten-xơ trạng thái ẩn → hai nhánh truy vấn → hai đầu tác vụ.
9. Đầu tái tạo tạo \(\widehat{X}\); đầu phân loại tạo xác suất lớp; các hàm mất mát dùng các trường nhãn vừa tạo.
10. Tính hàm mất mát → lan truyền ngược → cập nhật tham số của giai đoạn đang huấn luyện.
11. Sau giai đoạn A, nếu bật khởi tạo bộ nhớ bằng cửa sổ nhân tạo, mã nguồn gọi lại bộ tiêm trên một số lô huấn luyện; điểm bình thường đi vào tập nguyên mẫu liên tục, còn các lớp được gắn nhãn đi vào tập mã vec-tơ rời rạc.
12. Sau khi khởi tạo bộ nhớ, bộ mã hoá và hai bộ nhớ được giữ cố định; giai đoạn B tiếp tục học các mạng tổng hợp và mạng dự đoán.

**Chi tiết tiêm trong mã nguồn**

- Với cấu hình đa lớp hiện tại, tập lớp gồm “normal” và 11 họ: “spike”, “flip”, “speedup”, “noise”, “cutoff”, “average”, “scale”, “wander”, “contextual”, “upsidedown”, “mixture”.
- Khi train_balance_classes=true, các lớp được phân bổ gần đều trong lô; anomaly_probability không quyết định tỉ lệ trong nhánh này.
- Với cửa sổ \(L=20\), cấu hình hiện tại chọn đoạn dài khoảng 4–6 điểm; với \(D=38\), chọn ngẫu nhiên khoảng 3–19 kênh.
- Mặt nạ là hợp của các kênh bị thay đổi. Các vị trí không bị tiêm vẫn giữ giá trị cửa sổ sạch.
- Tên hàm và khóa dữ liệu được giữ nguyên khi cần đối chiếu với mã nguồn.

**Hiệu chỉnh suy luận**

1. Chuỗi xác thực sạch → các cửa sổ không chồng lấn → điểm tái tạo ở từng điểm thời gian.
2. Ghép các điểm theo đúng thứ tự chuỗi gốc → hiệu chỉnh các ngưỡng bất thường.
3. Nhánh “val_synth” có thể tiêm nhân tạo riêng để đánh giá phụ; nhánh “val” vẫn giữ dữ liệu sạch.

**Chốt:** Tiêm xảy ra sau khi đã tạo lô cửa sổ và trước bộ mã hoá; nó tạo mục tiêu giám sát cho huấn luyện, không phải bước của dự đoán trực tuyến.

### Trang 7 — Luồng dữ liệu của pha trực tuyến

Vẽ một hàng có đánh số từ trái sang phải:

1. **Dữ liệu dòng đã chuẩn hoá:** một điểm hoặc một đoạn mới từ chuỗi đã được chuẩn hoá bằng thống kê tập huấn luyện.
2. **Cửa sổ trượt:** lấy \(L=20\) điểm thời gian liên tiếp, mỗi điểm có \(D\) kênh.
3. **Nhánh nguồn:** \(X\) → bộ mã hoá đã đóng băng → \(Z^{(\mathrm{src})}\), tính một lần.
4. **Nhánh trực tuyến:** \(Z^{(\mathrm{src})}\) → bộ ánh xạ trực tuyến → \(Z^{(\mathrm{on})}\).
5. **Truy vấn ngẫu nhiên:** \(Z^{(\mathrm{on})}\) → hai bộ nhớ → các biểu diễn được truy vấn → đầu tái tạo.
6. **Sai số điểm:** so sánh \(x_i\) với \(\widehat{x}^{(m)}_i\) ở mỗi lượt truy vấn; lấy trung bình qua \(M\) lượt.
7. **Làm trơn:** cập nhật điểm của cùng một chỉ số tuyệt đối bằng EWMA với \(\rho=0.9\).
8. **Hai điểm quyết định của cửa sổ:** tính độ lệch tái tạo \(S^{(\mathrm{input})}\) và khoảng cách nguyên mẫu \(S^{(\mathrm{latent})}\).

**Ghi chú:** Pha trực tuyến không gọi SyntheticAnomalyInjector và không tạo bất thường nhân tạo.

**Chốt:** Dữ liệu dòng đã chuẩn hoá đi qua cửa sổ, biểu diễn, truy vấn và điểm số trước khi đến bước ra quyết định.

### Trang 8 — Từ điểm số đến quyết định dự đoán

Tách rõ hai đầu ra:

**Đầu ra ở mức điểm thời gian**

- Với mỗi điểm có chỉ số tuyệt đối \(n\), tính điểm đã làm trơn \(\widetilde{s}_n\).
- Quyết định dự đoán:
  \[
  \widehat{a}_n=\mathbb{I}\left(\widetilde{s}_n>\delta_{\mathrm{pt}}\right).
  \]
- Đây là nhãn bình thường hoặc bất thường của điểm thời gian.

**Đầu ra ở mức cửa sổ**

- Nếu \(S^{(\mathrm{input})}\leq\delta_{\mathrm{rec}}\): bình thường.
- Nếu sai số lớn nhưng \(S^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{-}\): bất thường khó.
- Nếu \(\delta_{\mathrm{lat}}^{-}<S^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{+}\): vùng xám.
- Nếu \(S^{(\mathrm{latent})}>\delta_{\mathrm{lat}}^{+}\): bất thường mạnh.

**Chốt:** Nhãn điểm trả lời “điểm này có bất thường không?”; quyết định cửa sổ trả lời “có được dùng để thích ứng không?”.

### Trang 9 — Vì sao cần cả tam phân và bộ đệm xác minh?

| Quy tắc tam phân | Bộ đệm xác minh |
|---|---|
| Xử lý nhanh: bình thường, bất thường khó, vùng xám hoặc bất thường mạnh | Xử lý thêm các cửa sổ vùng xám |
| Dùng cho quyết định tức thời | Chờ đủ cửa sổ không chồng lấn rồi kiểm tra chữ ký lặp lại |

- Cửa sổ bình thường và bất thường mạnh không dùng để thích ứng.
- Cửa sổ bất thường khó cập nhật bộ ánh xạ trực tuyến.
- Cửa sổ vùng xám đi vào bộ đệm; chỉ điểm được xác minh là mẫu hình bình thường mới được dùng để cập nhật.
- Không gộp hai bước: tam phân cần rẻ; xác minh cần thêm tính toán và ngữ cảnh.

**Chốt:** Tách hai bước giúp giảm chi phí và giảm nguy cơ học nhầm bất thường.

### Trang 10 — Định lượng độ bất định

- **Hình:** một cửa sổ qua \(M=10\) lượt truy vấn ngẫu nhiên.
- **Công thức:** \(\bar{s}=\frac{1}{M}\sum_m s^{(m)}\), \(\operatorname{Var}(s)=\frac{1}{M-1}\sum_m(s^{(m)}-\bar{s})^2\).
- **Nói:** Trung bình cho điểm dự đoán; phương sai cho mức không chắc chắn giữa các lượt truy vấn.
- **Chốt:** Độ bất định là tín hiệu hỗ trợ, không phải bằng chứng chắc chắn của bất thường.

### Trang 11 — Hai độ đo chính

Hai khung ngang:

- **VUS-PR:** thay đổi ngưỡng → tính độ chính xác và độ bao phủ → tích hợp trên các ngưỡng và phạm vi cửa sổ.
- **F1 liên kết:** tách các khoảng bất thường → kiểm tra phần giao với dự đoán → tính độ chính xác và độ bao phủ theo liên kết → lấy trung bình điều hoà.

**Chốt:** VUS-PR đánh giá đường cong theo nhiều ngưỡng; F1 liên kết xét quan hệ giữa các khoảng bất thường.

### Trang 12 — Bối cảnh thí nghiệm

| Máy | Điểm huấn luyện / kiểm tra xác nhận / kiểm tra | Cửa sổ huấn luyện / kiểm tra xác nhận / kiểm tra |
|---|---:|---:|
| machine-1-6 | 18.951 / 4.737 / 23.689 | 18.932 / 236 / 1.184 |
| machine-3-4 | 18.950 / 4.737 / 23.687 | 18.931 / 236 / 1.184 |
| machine-3-9 | 22.971 / 5.742 / 28.713 | 22.952 / 287 / 1.435 |

- **Thiết lập:** độ dài cửa sổ \(L=20\), số kênh \(D=38\).
- **Chốt:** Ba máy có quy mô và tỉ lệ bất thường khác nhau.

### Trang 13 — Vì sao chọn ba máy và ít dữ liệu?

- **machine-1-6:** làm mốc đối chiếu.
- **machine-3-4:** kiểm tra trên một máy khác cùng nhóm.
- **machine-3-9:** kiểm tra khi tỉ lệ bất thường thấp hơn.
- **Ít dữ liệu:** buộc cơ chế thích ứng hoạt động trong điều kiện hạn chế và giảm nguy cơ kết luận do dữ liệu quá thuận lợi.

**Chốt:** Đây là một kiểm tra có phạm vi hẹp, không đại diện cho mọi dòng dữ liệu.

### Trang 14 — Kết quả và hạn chế về tỉ lệ báo động giả

Hai khung phẳng:

- **Bằng chứng:** bảng VUS-PR, F1 liên kết, độ bao phủ, tỉ lệ báo động giả, số báo động giả và số điểm bình thường cho từng máy.
- **Diễn giải:** một câu ngắn cho từng so sánh; chỉ nói “cơ chế làm trơn đóng góp phần lớn mức tăng” nếu bảng số liệu xác nhận.

- **Công thức:** \(\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}\).
- Ngưỡng lấy từ tập xác nhận sạch không bảo đảm tỉ lệ báo động giả thấp trên tập kiểm tra.

**Chốt:** Không dùng một chỉ số tốt để che khuất tỉ lệ báo động giả còn cao.

### Trang 15 — Kết luận trung thực

Ba dòng lớn:

1. **Đóng góp:** phối hợp truy vấn, độ bất định và thích ứng trực tuyến trong một khung đa nhiệm.
2. **Bằng chứng:** kết quả phải đọc cùng VUS-PR, F1 liên kết và tỉ lệ báo động giả.
3. **Giới hạn:** phạm vi dữ liệu còn nhỏ; tỉ lệ báo động giả chưa đủ thấp.

**Hướng tiếp theo:** giảm tỉ lệ báo động giả và kiểm tra trên nhiều máy hơn.

## English version

### Slide 1 — Title and research question

- **Visual:** title and one time-series sketch.
- **Say:** How can we detect anomalies when the data changes and the model is uncertain?
- **Takeaway:** This thesis proposes a multitask framework for time-series anomaly detection.

### Slide 2 — Problem formulation

- **Visual:** time series → window → normal or anomalous label.
- **Formula:** \(X_t \in \mathbb{R}^{L \times D}\), \(s_t=f(X_t)\).
- **Takeaway:** The goal is to detect anomalies at both window and point levels.

### Slide 3 — Two limitations

Two horizontal boxes:

1. The model does not express predictive uncertainty clearly.
2. The model adapts poorly to non-stationary streams under a low computation budget.

**Takeaway:** The method needs uncertainty estimation and controlled adaptation.

### Slide 4 — Contribution boundary

Two flat boxes:

| Inherited components | Proposed or integrated components |
|---|---|
| Shared convolutional encoder, reconstruction, memory retrieval, evaluation metrics | Two query operators, stochastic queries for uncertainty, online adaptation, triage, and verification buffer |

**Takeaway:** The claimed novelty is the integrated design and runtime policy, not every familiar component.

### Slide 5 — Overall architecture

- **Visual:** shared encoder → two memories → two task heads.
- **Labels:** continuous prototype bank; discrete codebook; reconstruction task; multi-class classification task.
- **Details:** 32 continuous prototypes; 60 discrete prototypes across 12 classes.
- **Takeaway:** Both tasks share representations but keep separate objectives.

### Slide 6 — Offline data flow

Draw two rows. Separate data preparation, injection, and model execution:

**Data preparation**

1. Raw series → cleaning → fit the mean and standard deviation on the training split.
2. Apply the same scaler to the training, validation, and test splits.
3. Scaled series → WindowDataset → windows \(X\in\mathbb{R}^{L\times D}\) → batches \(x\in\mathbb{R}^{B\times L\times D}\).

**One training batch**

4. Clean window batch → training step → batch preparation before the forward pass.
5. If enabled, SyntheticAnomalyInjector assigns a class label to each window.
6. A normal window stays unchanged; an anomaly window receives one anomaly family on one contiguous segment and selected channels.
7. The injector writes the new \(x\), classification labels, synthetic anomaly mask, point labels, and augmentation metadata.
8. Augmented batch → shared convolutional encoder → hidden-state tensor → two query branches → two task heads.
9. The reconstruction head produces \(\widehat{X}\); the classification head produces class probabilities; losses consume the generated supervision fields.
10. Compute loss → backpropagate → update the parameters of the current stage.
11. After Stage A, if synthetic-window memory initialization is enabled, the code calls the injector again on selected training batches; normal points feed the continuous prototype bank, while labeled classes feed the discrete codebook.
12. After memory initialization, the encoder and both memories are frozen; Stage B continues training the fusion and prediction networks.

**Injection details in code**

- The current multiclass configuration uses “normal” plus 11 families: “spike”, “flip”, “speedup”, “noise”, “cutoff”, “average”, “scale”, “wander”, “contextual”, “upsidedown”, and “mixture”.
- With train_balance_classes=true, classes receive near-uniform batch quotas; anomaly_probability does not control this branch.
- With \(L=20\), the current configuration samples a segment of about 4–6 points; with \(D=38\), it samples about 3–19 channels.
- The mask is the union of changed channels. Untouched positions keep the clean window values.
- Keep function and data-key names when the slide must match the source code.

**Inference calibration**

1. Clean validation series → non-overlapping windows → point-level reconstruction scores.
2. Concatenate scores in the original time order → calibrate anomaly thresholds.
3. The “val_synth” branch may inject synthetic anomalies for an auxiliary evaluation; the “val” branch remains clean.

**Takeaway:** Injection happens after window batching and before the encoder; it creates training supervision, not online prediction input.

### Slide 7 — Online data flow

Draw one numbered row from left to right:

1. **Scaled stream data:** a new point or segment from a sequence scaled with training statistics.
2. **Sliding window:** collect \(L=20\) consecutive time points with \(D\) channels.
3. **Source branch:** \(X\) → frozen encoder → \(Z^{(\mathrm{src})}\), computed once.
4. **Online branch:** \(Z^{(\mathrm{src})}\) → online projector → \(Z^{(\mathrm{on})}\).
5. **Stochastic queries:** \(Z^{(\mathrm{on})}\) → two memories → queried representations → reconstruction head.
6. **Point error:** compare \(x_i\) with \(\widehat{x}^{(m)}_i\) at each stochastic pass; average over \(M\) passes.
7. **Smoothing:** update the score of the same global index with EWMA and \(\rho=0.9\).
8. **Two window quantities:** compute reconstruction deviation \(S^{(\mathrm{input})}\) and prototype distance \(S^{(\mathrm{latent})}\).

**Note:** The online phase does not call SyntheticAnomalyInjector and does not create synthetic anomalies.

**Takeaway:** Scaled stream data passes through windows, representations, queries, and scores before the final decision.

### Slide 8 — From scores to prediction decisions

Separate the two outputs:

**Point-level output**

- For each global index \(n\), compute the smoothed score \(\widetilde{s}_n\).
- Prediction:
  \[
  \widehat{a}_n=\mathbb{I}\left(\widetilde{s}_n>\delta_{\mathrm{pt}}\right).
  \]
- This is the normal or anomalous label for a time point.

**Window-level output**

- If \(S^{(\mathrm{input})}\leq\delta_{\mathrm{rec}}\): normal.
- If the error is high but \(S^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{-}\): hard-old-normality.
- If \(\delta_{\mathrm{lat}}^{-}<S^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{+}\): gray zone.
- If \(S^{(\mathrm{latent})}>\delta_{\mathrm{lat}}^{+}\): strong anomaly.

**Takeaway:** The point label answers “is this point anomalous?”; the window route answers “can this window support adaptation?”.

### Slide 9 — Why triage and verification buffer are separate

| Triage rule | Verification buffer |
|---|---|
| Fast routing: normal, hard-old-normality, gray zone, or strong anomaly | Further processing for gray-zone windows |
| Used for an immediate decision | Waits for repeated signatures across non-overlapping windows |

- Normal and strong-anomaly windows are not used for adaptation.
- Hard-old-normality windows update the online projector.
- Gray-zone windows enter the buffer; only verified pseudo-new-normal points support adaptation.
- Do not merge them: triage must be cheap; verification needs extra computation and context.

**Takeaway:** Separate stages reduce cost and reduce the risk of learning anomalies.

### Slide 10 — Uncertainty estimation

- **Visual:** one window passes through \(M=10\) stochastic queries.
- **Formula:** \(\bar{s}=\frac{1}{M}\sum_m s^{(m)}\), \(\operatorname{Var}(s)=\frac{1}{M-1}\sum_m(s^{(m)}-\bar{s})^2\).
- **Say:** The mean gives the prediction score; the variance measures disagreement across queries.
- **Takeaway:** Uncertainty is supporting evidence, not proof of an anomaly.

### Slide 11 — Two main metrics

Two horizontal boxes:

- **VUS-PR:** vary the threshold → compute precision and recall → integrate across thresholds and window ranges.
- **Affiliation F1:** split anomalies into ranges → check prediction overlap → compute range-based precision and recall → take the harmonic mean.

**Takeaway:** VUS-PR evaluates threshold behavior; Affiliation F1 evaluates range relationships.

### Slide 12 — Experimental context

| Machine | Train / validation / test points | Train / validation / test windows |
|---|---:|---:|
| machine-1-6 | 18,951 / 4,737 / 23,689 | 18,932 / 236 / 1,184 |
| machine-3-4 | 18,950 / 4,737 / 23,687 | 18,931 / 236 / 1,184 |
| machine-3-9 | 22,971 / 5,742 / 28,713 | 22,952 / 287 / 1,435 |

- **Setup:** window length \(L=20\), channel count \(D=38\).
- **Takeaway:** The three machines differ in scale and anomaly ratio.

### Slide 13 — Why these machines and limited data?

- **machine-1-6:** reference case.
- **machine-3-4:** another machine from the same group.
- **machine-3-9:** lower anomaly-rate setting.
- **Limited data:** tests adaptation under restricted context and reduces overly favorable conclusions.

**Takeaway:** This is a focused evaluation, not evidence for every stream.

### Slide 14 — Results and false-positive limitation

Two flat boxes:

- **Evidence:** table of VUS-PR, Affiliation F1, recall, FPR, false positives, and normal-point support for each machine.
- **Interpretation:** one short sentence per comparison; say “smoothing contributes most of the gain” only if the table supports it.

- **Formula:** \(\mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}\).
- A threshold selected on clean validation data does not guarantee low test FPR.

**Takeaway:** Do not let one strong metric hide a high false-positive rate.

### Slide 15 — Honest conclusion

Three large lines:

1. **Contribution:** an integrated multitask framework for querying, uncertainty, and online adaptation.
2. **Evidence:** interpret VUS-PR, Affiliation F1, and FPR together.
3. **Limit:** small evaluation scope and insufficiently low FPR.

**Next step:** reduce FPR and evaluate more machines.
