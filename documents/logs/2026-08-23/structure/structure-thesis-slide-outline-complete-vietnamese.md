# Dàn ý hoàn chỉnh cho bài thuyết trình

Mục tiêu: giữ nội dung chính của `slides.tex`, bổ sung các phần còn thiếu trong bảng kiểm, làm rõ tính mới một cách khiêm tốn và báo cáo kết quả cùng hạn chế về tỉ lệ báo động giả.

Mỗi trang chỉ có một thông điệp chính:

> Tiêu đề kết luận → một hoặc hai khung ngang → hình hoặc công thức ngắn → một câu chốt.

## Mạch kể chuyện xuyên suốt

Bộ trang cần trả lời liên tiếp bảy câu hỏi:

1. **Bài toán là gì?** Dữ liệu thay đổi làm mô hình dễ quá tự tin và khó thích ứng.
2. **Khoảng trống nằm ở đâu?** Các nghiên cứu trước có những thành phần phù hợp, nhưng chưa tạo thành một luồng phối hợp thống nhất trong khóa luận này.
3. **Khóa luận đề xuất gì?** Hai cách truy vấn bộ nhớ, phiên bản ngẫu nhiên để đo độ bất định và cơ chế thích ứng trực tuyến có kiểm soát.
4. **Mô hình học như thế nào?** Pha ngoại tuyến học biểu diễn, khởi tạo bộ nhớ và tinh chỉnh các đầu ra.
5. **Mô hình ra quyết định như thế nào?** Pha trực tuyến biến dữ liệu mới thành điểm số, nhãn dự đoán và quyết định cập nhật.
6. **Bằng chứng là gì?** Đánh giá trên các độ đo, các chuỗi dữ liệu và các biến thể đã chọn.
7. **Kết luận đến đâu?** Kết quả tích cực trong phạm vi nhỏ, nhưng cần báo cáo rõ tỉ lệ báo động giả và giới hạn khái quát hóa.

Mỗi phần dưới đây phải nối với câu hỏi kế tiếp; không trình bày các kỹ thuật như những mảnh rời nhau.

### Các câu chuyển quan trọng

| Chuyển trang | Câu chuyển nên nói |
|---|---|
| 2 → 3 | Bài toán này không chỉ có ý nghĩa trên giấy; nó xuất hiện trong nhiều hệ thống cần giám sát liên tục. |
| 4 → 5 | Vì vậy, ta xem các công trình trước đã giải quyết từng phần của vấn đề này như thế nào. |
| 5 → 6 | Từ các công trình đó, ta tách rõ phần khóa luận kế thừa và phần khóa luận phát triển. |
| 8 → 9 | Sau khi biết bản đồ của toàn hệ thống, ta đi theo dữ liệu từ lúc chuẩn bị đến khi mô hình được huấn luyện. |
| 14 → 15 | Khi hai bộ nhớ đã được khởi tạo, ta cần hiểu cách chúng biến biểu diễn ẩn thành thông tin truy vấn. |
| 20 → 21 | Các lượt truy vấn tạo ra điểm số; bước tiếp theo là dùng dữ liệu xác thực sạch để cố định ngưỡng. |
| 21 → 22 | Khi ngưỡng đã cố định, ta đưa dữ liệu mới vào pha trực tuyến và theo dõi toàn bộ đường đi đến quyết định. |
| 27 → 28 | Sau khi giải thích cơ chế, ta cần xác định cách đo kết quả và cách tránh đọc số liệu quá mức. |
| 32 → 33 | Bây giờ ta biết dữ liệu và giới hạn thiết kế; ta có thể đọc các bảng kết quả trong đúng bối cảnh. |
| 36 → 37 | Cuối cùng, tỉ lệ báo động giả cho biết vì sao kết quả tốt vẫn cần được báo cáo cùng giới hạn. |

## 1. Mở đầu và đặt bài toán

**Câu hỏi của phần:** Vì sao bài toán này cần một phương pháp mới?

### Trang 1 — Tên đề tài

- **Giữ lại:** tên đề tài, tác giả, đơn vị và thời gian bảo vệ.
- **Hình:** một chuỗi thời gian có đoạn bất thường được tô màu.
- **Câu hỏi:** Làm sao phát hiện bất thường khi phân phối dữ liệu thay đổi và mô hình có độ bất định?
- **Chốt:** Khóa luận xây dựng một khung đa nhiệm cho phát hiện bất thường chuỗi thời gian.

### Trang 2 — Bài toán cần giải quyết

- **Hình:** chuỗi thời gian → cửa sổ → điểm số → nhãn dự đoán.
- **Công thức:**

  \[
  \mathbf{X}_t\in\mathbb{R}^{L\times C},\qquad
  s_{t,i}=\frac{1}{C}\left\|\mathbf{x}_{t,i}-\widehat{\mathbf{x}}_{t,i}\right\|_2^2,
  \qquad
  \widehat a_{t,i}=\mathbb{I}(s_{t,i}>\delta).
  \]

- \(L\): độ dài cửa sổ; \(C\): số kênh; \(s_{t,i}\): điểm bất thường tại vị trí \(i\).
- **Chốt:** Mục tiêu là phát hiện đúng bất thường ở cả mức điểm thời gian và mức cửa sổ.

### Trang 3 — Bối cảnh sử dụng

- **Giữ lại:** sản xuất bán dẫn, giám sát hệ thống đám mây và theo dõi dấu hiệu sinh tồn trong hồi sức tích cực.
- **Hình:** ba hình cùng kích thước, mỗi hình có một chú thích ngắn.
- **Chốt:** Một báo động sai trong các hệ thống này có thể làm tăng chi phí hoặc gây mệt mỏi vì báo động.

### Trang 4 — Hai hạn chế của các phương pháp gần đây

Hai khung ngang:

1. Mạng nơ-ron có thể quá tự tin khi gặp dữ liệu khác phân phối huấn luyện.
2. Mạng nơ-ron khó thích ứng với dòng dữ liệu không dừng mà vẫn giữ chi phí thấp.

- **Chốt:** Cần một cơ chế vừa đo độ bất định, vừa thích ứng có kiểm soát.

### Trang 5 — Các nghiên cứu liên quan

- **Hình:** năm thẻ công trình, nối từ công trình nền đến phần được kế thừa hoặc phát triển.

| Công trình | Vai trò trong khóa luận |
|---|---|
| RedLamp — KDD-25 | Khung huấn luyện đa tác vụ tương tự được dùng làm nền. |
| H-PAD — ICLR-25 | Bộ nhớ cho toán tử truy vấn liên tục được kế thừa. |
| VQ-VAE — NIPS-17 | Bộ nhớ cho toán tử truy vấn rời rạc được kế thừa. |
| Ada-ReAlign — NeurIPS-24 | Ý tưởng kéo biểu diễn của nhánh trực tuyến về gần nhánh nguồn được lấy cảm hứng. |
| Stochastic Transformer — AAAI-22 | Cơ chế chèn nhiễu Gumbel vào ma trận tương đồng được dùng làm cảm hứng cho hai truy vấn ngẫu nhiên. |

- **Cách nói:** Khóa luận kế thừa khung huấn luyện và hai loại bộ nhớ; đồng thời phát triển hai phiên bản truy vấn ngẫu nhiên và cơ chế thích ứng trực tuyến.
- **Chốt:** Tính mới nằm ở cách kết hợp các thành phần này thành một khung đa nhiệm có độ bất định và thích ứng trực tuyến.

## 2. Tính mới và kiến trúc đề xuất

**Câu hỏi của phần:** Khóa luận dùng gì từ nghiên cứu trước và đề xuất phần nào?

### Trang 6 — Ranh giới giữa kế thừa và đề xuất mới

Hai khung phẳng:

| Thành phần dùng lại hoặc làm nền | Phần khóa luận đề xuất hoặc phối hợp |
|---|---|
| Bộ mã hóa tích chập dùng chung; đầu tái tạo; đầu phân loại; K-Means; các họ bất thường nhân tạo; các độ đo VUS-PR và Affiliation F1-score | Hai toán tử truy vấn liên tục và rời rạc; phiên bản ngẫu nhiên của hai toán tử; luồng thích ứng trực tuyến; quy tắc tam phân; bộ đệm xác minh; hàm mất mát tương phản trực tuyến |

- **Câu nói cần dùng:** Tính mới chính nằm ở thiết kế phối hợp và chính sách vận hành.
- **Câu không nên dùng:** Không tuyên bố rằng mọi thành phần riêng lẻ đều hoàn toàn mới.
- **Chốt:** Đóng góp được trình bày ở mức khung phương pháp và luồng chạy, không phóng đại nguồn gốc của từng kỹ thuật.

### Trang 7 — Kiến trúc tổng quát của pha ngoại tuyến

- **Giữ lại:** sơ đồ kiến trúc ngoại tuyến.
- **Hình:** bộ mã hóa dùng chung → bộ nhớ nguyên mẫu liên tục và tập mã vec-tơ rời rạc → hai nhánh tái tạo và phân loại.
- **Ghi ngắn:** 32 nguyên mẫu liên tục; 60 nguyên mẫu rời rạc thuộc 12 lớp.
- **Chốt:** Hai tác vụ dùng chung biểu diễn nhưng giữ mục tiêu học riêng.

### Trang 8 — Phân biệt bốn hoạt động chính

|  | Huấn luyện | Suy luận |
|---|---|---|
| **Ngoại tuyến** | Học bộ mã hóa, khởi tạo bộ nhớ, tinh chỉnh mạng tổng hợp và đầu dự đoán | Chạy trên xác thực sạch để chọn ngưỡng, sau đó chạy trên kiểm tra |
| **Trực tuyến** | Cập nhật có điều kiện chỉ mạng ánh xạ trực tuyến trong lúc kiểm thử | Nhận cửa sổ mới, tính điểm, làm trơn, phân luồng và đưa ra dự đoán |

- **Chốt:** Pha trực tuyến có cập nhật tham số, nhưng chỉ sau bước suy luận và chỉ khi dữ liệu được xác minh.

## 3. Luồng dữ liệu của pha ngoại tuyến

**Câu hỏi của phần:** Mô hình biến dữ liệu huấn luyện thành một mô hình sẵn sàng suy luận như thế nào?

### Trang 9 — Luồng chuẩn bị dữ liệu ngoại tuyến

Vẽ một hàng từ trái sang phải:

1. Chuỗi dữ liệu gốc.
2. Tính trung bình và độ lệch chuẩn chỉ trên tập huấn luyện.
3. Chuẩn hóa tập huấn luyện, xác thực và kiểm tra bằng cùng thống kê đó.
4. Chia chuỗi thành các cửa sổ \(\mathbf{X}\in\mathbb{R}^{20\times38}\).
5. Gom cửa sổ thành lô \(\mathbf{x}\in\mathbb{R}^{B\times20\times38}\).

- **Chốt:** Dữ liệu xác thực và kiểm tra không được dùng để học bộ chuẩn hóa.

### Trang 10 — Vị trí tiêm bất thường nhân tạo trong luồng chạy

Vẽ hai hàng dữ liệu:

**Dữ liệu sạch → tạo lô cửa sổ → tiêm bất thường → bộ mã hóa → các đầu ra.**

**Một lô huấn luyện sau khi tiêm chứa:**

- cửa sổ đã thay đổi;
- nhãn lớp của từng cửa sổ;
- mặt nạ điểm bị tiêm;
- nhãn điểm và siêu dữ liệu của phép tiêm.

- **Câu chốt:** Tiêm xảy ra sau khi tạo lô cửa sổ và trước bộ mã hóa; đây là bước tạo dữ liệu huấn luyện, không phải bước dự đoán trực tuyến.

### Trang 11 — Cách mã nguồn tiêm bất thường

- **Bộ tiêm:** lớp `SyntheticAnomalyInjector` trong mã nguồn.
- **Các lớp:** bình thường và 11 họ bất thường: tăng đột ngột, lật ngang, tăng nhịp độ, nhiễu, cắt theo ngưỡng, trung bình hóa, co giãn biên độ, dao động trôi dạt, bất thường theo ngữ cảnh, lật dọc và bất thường hỗn hợp.
- **Cách chọn vùng:** chọn một đoạn liên tục dài từ 20% đến 30% cửa sổ; với \(L=20\), độ dài thường là 4–6 điểm.
- **Cách chọn kênh:** chọn ngẫu nhiên từ 10% đến 50% số kênh; với \(C=38\), khoảng 3–19 kênh.
- **Mặt nạ:** chỉ đánh dấu các vị trí thật sự bị thay đổi; các vị trí khác giữ nguyên dữ liệu sạch.
- **Cân bằng lớp:** khi bật cân bằng lớp, 12 lớp được phân bổ gần đều trong lô; xác suất tiêm không quyết định tỉ lệ ở nhánh này.
- **Hình:** một cửa sổ trước và sau khi tiêm, tô vùng thay đổi cùng mặt nạ.
- **Chốt:** Mã nguồn tạo được cả dữ liệu biến đổi và thông tin cho hàm mất mát biết vị trí nào là bất thường nhân tạo.

### Trang 12 — Giai đoạn học biểu diễn đa nhiệm

- **Giữ lại:** sơ đồ tiền huấn luyện.
- **Luồng:** cửa sổ sạch và cửa sổ nhân tạo → cùng bộ mã hóa → tái tạo, phân loại lớp và học tương phản.
- **Mục tiêu:** học biểu diễn phục vụ đồng thời tái tạo chuỗi và phân biệt 12 lớp.
- **Công thức ngắn:**

  \[
  \mathcal{L}_{A}
  =\lambda_{\mathrm{rec}}\mathcal{L}_{\mathrm{rec}}
  +\lambda_{\mathrm{cls}}\mathcal{L}_{\mathrm{cls}}
  +\lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}.
  \]

- **Cấu hình:** 25 vòng lặp huấn luyện.
- **Chốt:** Tiêm bất thường tạo tín hiệu giám sát cho việc học biểu diễn.

### Trang 13 — Khởi tạo hai bộ nhớ

- Sau giai đoạn học biểu diễn, lấy một số lô huấn luyện và tiêm lại bất thường để tạo 12 lớp.
- Lọc các điểm bình thường và dùng K-Means để tạo 32 nguyên mẫu liên tục.
- Với mỗi lớp trong 12 lớp, dùng K-Means để tạo 5 nguyên mẫu; tổng cộng 60 nguyên mẫu rời rạc.
- Đóng băng bộ mã hóa và hai bộ nhớ sau khi khởi tạo.
- **Hình:** điểm ẩn → K-Means → hai bộ nhớ.
- **Chốt:** Hai bộ nhớ được xây dựng từ cùng biểu diễn đã học và không bị thay đổi trong giai đoạn tinh chỉnh.

### Trang 14 — Giai đoạn tinh chỉnh mạng tổng hợp

- **Giữ lại:** sơ đồ tinh chỉnh.
- Bộ mã hóa và hai bộ nhớ bị đóng băng.
- Chỉ các mạng tổng hợp và các đầu dự đoán được cập nhật.
- Đầu tái tạo tạo \(\widehat{\mathbf X}\); đầu phân loại tạo xác suất 12 lớp.
- **Cấu hình:** 5 vòng lặp huấn luyện.
- **Chốt:** Giai đoạn này học cách kết hợp biểu diễn gốc với hai biểu diễn truy vấn.

## 4. Các toán tử truy vấn và độ bất định

**Câu hỏi của phần:** Mô hình tạo biểu diễn, điểm số và độ bất định bằng cách nào?

### Trang 15 — Biểu diễn ẩn và hai bộ nhớ

- **Giữ lại:** định nghĩa cửa sổ \(\mathbf X\), ten-xơ ẩn \(\mathbf Z\) và các bộ nhớ.
- \(\mathbf Z=f_{\mathrm{enc}}(\mathbf X)\), mỗi vec-tơ ẩn được chuẩn hóa chuẩn Euclid bằng 1.
- Bộ nhớ liên tục chứa \(K_c=32\) nguyên mẫu.
- Bộ nhớ rời rạc chứa \(K_d=60\) nguyên mẫu.
- **Hình:** \(\mathbf X\rightarrow\mathbf Z\rightarrow\) hai bộ nhớ.
- **Chốt:** Hai bộ nhớ cung cấp hai cách mô tả mẫu hình bình thường và bất thường nhân tạo.

### Trang 16 — Truy vấn liên tục xác định

- Tính độ tương đồng giữa vec-tơ ẩn và toàn bộ nguyên mẫu liên tục.
- Dùng hàm mềm hóa để tạo trọng số.
- Lấy tổng có trọng số của tất cả nguyên mẫu.
- **Công thức:**

  \[
  \alpha_{i,k}^{(c)}=\operatorname{softmax}_k
  \left(\frac{\mathbf z_i^{\mathsf T}\mathbf p_k^{(c)}}{\tau_c}\right),
  \qquad
  \widetilde{\mathbf z}_i^{(c)}=\sum_k\alpha_{i,k}^{(c)}\mathbf p_k^{(c)}.
  \]

- **Chốt:** Đây là truy vấn dày; mọi nguyên mẫu đều có thể đóng góp.

### Trang 17 — Truy vấn rời rạc xác định

- Tính khoảng cách đến các mã vec-tơ.
- Chọn ba mã gần nhất.
- Chuẩn hóa trọng số trên ba mã được chọn rồi lấy tổng có trọng số.
- **Công thức ngắn:**

  \[
  \mathcal I_i=\operatorname{TopKMin}_k(d_{i,k};3),
  \qquad
  \widetilde{\mathbf z}_i^{(d)}=
  \sum_{k\in\mathcal I_i}\alpha_{i,k}^{(d)}\mathbf p_k^{(d)}.
  \]

- **Chốt:** Đây là truy vấn thưa; chỉ ba mã gần nhất đóng góp.

### Trang 18 — Truy vấn liên tục ngẫu nhiên

- Giữ toàn bộ nguyên mẫu liên tục có thể tham gia.
- Mỗi lượt lấy nhiễu Gumbel mới rồi tính lại trọng số.
- Mỗi lượt tạo một biểu diễn liên tục khác nhau.
- **Hình:** một đầu vào → nhiều lượt lấy mẫu → nhiều biểu diễn.
- **Chốt:** Sự thay đổi giữa các lượt cung cấp tín hiệu để ước lượng độ bất định nhận thức.

### Trang 19 — Truy vấn rời rạc ngẫu nhiên

- Thêm nhiễu Gumbel vào điểm truy vấn.
- Chọn lại ba mã gần nhất ở mỗi lượt.
- Tập ba mã có thể thay đổi giữa các lượt.
- **Chốt:** Độ bất định được tạo ra từ cả trọng số thay đổi và lân cận mã vec-tơ thay đổi.

### Trang 20 — Tính độ bất định

Vẽ một cửa sổ đi qua (M=10) lượt truy vấn ngẫu nhiên:

1. Giữ bộ mã hóa nguồn cố định và tính biểu diễn nguồn một lần.
2. Lấy nhiễu Gumbel cho từng lượt.
3. Tạo \(\widehat{\mathbf X}^{(m)}\) và điểm \(s_i^{(m)}\) ở mỗi lượt.
4. Tính điểm trung bình:

   \[
   \overline{s}_i=\frac{1}{M}\sum_{m=1}^{M}s_i^{(m)}.
   \]

5. Tính phương sai không chệch:

   \[
   u_i^2=\frac{1}{M-1}\sum_{m=1}^{M}
   \left(s_i^{(m)}-\overline{s}_i\right)^2.
   \]

- Điểm trung bình dùng cho dự đoán; phương sai là tín hiệu chẩn đoán.
- Phương sai không tham gia vào ngưỡng, quy tắc tam phân hoặc quyết định cập nhật.
- **Chốt:** Độ bất định cao không tự động có nghĩa là điểm đó bất thường.

### Trang 21 — Suy luận ngoại tuyến và hiệu chỉnh ngưỡng

Vẽ luồng:

**Mô hình tốt nhất → xác thực sạch không chồng lấp → điểm tái tạo → ghép theo thứ tự thời gian → chọn ngưỡng → kiểm tra.**

- Chỉ xác thực sạch được dùng để chọn ngưỡng.
- Chuỗi xác thực nhân tạo chỉ dùng cho chẩn đoán hoặc đánh giá phụ.
- Không dùng nhãn kiểm tra để chọn ngưỡng.
- **Câu nối:** Sau khi đã biết điểm số và độ bất định được tạo ra thế nào, ta có thể cố định ngưỡng rồi chuyển sang dòng dữ liệu mới.
- **Chốt:** Ngưỡng được cố định trước khi đánh giá tập kiểm tra.

## 5. Luồng dữ liệu của pha trực tuyến

**Câu hỏi của phần:** Khi dữ liệu mới xuất hiện, mô hình phát hiện và thích ứng mà không học nhầm như thế nào?

### Trang 22 — Kiến trúc và dữ liệu đầu vào trực tuyến

- **Giữ lại:** sơ đồ kiến trúc trực tuyến.
- Dữ liệu dòng mới → chuẩn hóa bằng thống kê tập huấn luyện → cửa sổ trượt dài 20.
- Bộ mã hóa nguồn bị đóng băng tạo \(\mathbf Z^{(\mathrm{src})}\) một lần.
- Mạng ánh xạ trực tuyến tạo \(\mathbf Z^{(\mathrm{on})}\).
- Chỉ mạng ánh xạ trực tuyến được cập nhật; bộ mã hóa, hai bộ nhớ, mạng tổng hợp và đầu dự đoán bị đóng băng.
- **Chốt:** Luồng trực tuyến tái sử dụng mô hình ngoại tuyến và chỉ mở một vùng tham số nhỏ để thích ứng.

### Trang 23 — Từ dữ liệu thô đến điểm dự đoán

Vẽ một hàng có đánh số:

1. Nhận điểm dữ liệu mới.
2. Chuẩn hóa điểm dữ liệu.
3. Ghép vào cửa sổ trượt.
4. Tính biểu diễn nguồn.
5. Ánh xạ sang biểu diễn trực tuyến.
6. Truy vấn hai bộ nhớ ngẫu nhiên (M=10) lượt.
7. Tái tạo cửa sổ ở từng lượt.
8. Tính sai số từng điểm và lấy trung bình qua các lượt.
9. Ghép các điểm trùng chỉ số tuyệt đối.
10. Làm trơn bằng trung bình trượt hàm mũ với \(\rho=0.9\).
11. So sánh với ngưỡng và tạo nhãn dự đoán từng điểm.

- **Ghi rõ:** Pha trực tuyến không gọi bộ tiêm bất thường nhân tạo.
- **Chốt:** Dữ liệu thô chỉ trở thành quyết định sau khi đi qua chuẩn hóa, cửa sổ, biểu diễn, truy vấn, điểm số và làm trơn.

### Trang 24 — Điểm số và quyết định ở mức cửa sổ

Hai khung ngang:

**Mức điểm thời gian**

\[
\widehat a_n=\mathbb I(\widetilde s_n>\delta_{\mathrm{pt}}).
\]

**Mức cửa sổ**

- Sai số tái tạo không vượt ngưỡng: bình thường.
- Sai số tái tạo cao, khoảng cách ẩn thấp: bất thường cũ khó nhận biết.
- Khoảng cách ẩn ở vùng giữa: vùng xám.
- Khoảng cách ẩn cao: bất thường mạnh.

- **Chốt:** Nhãn điểm phục vụ phát hiện; nhãn cửa sổ quyết định có được phép thích ứng hay không.

### Trang 25 — Quy tắc tam phân và bộ đệm xác minh

| Kết quả phân luồng | Xử lý ngay | Có đưa vào bộ đệm không? |
|---|---|---|
| Bình thường | Không cập nhật | Không |
| Bất thường cũ khó nhận biết | Có thể cập nhật theo điều kiện | Không |
| Vùng xám | Chưa cập nhật | Có |
| Bất thường mạnh | Không cập nhật | Không |

Với cửa sổ vùng xám:

1. Đưa cửa sổ vào bộ đệm.
2. Chờ đủ các cửa sổ không chồng lấp.
3. Loại các điểm thuộc cụm bất thường đã biết.
4. Tính chữ ký mẫu hình.
5. Tìm chữ ký lặp lại ở nhiều cửa sổ.
6. Chỉ các điểm được xác minh là chuẩn tính mới mới được dùng để cập nhật.

- **Chốt:** Quy tắc tam phân xử lý nhanh; bộ đệm xác minh dùng thêm ngữ cảnh cho các trường hợp chưa rõ.

### Trang 26 — Vì sao không gộp hai bước?

Hai khung ngang:

**Nếu chỉ dùng quy tắc tam phân:**

- rẻ và nhanh;
- nhưng vùng xám có thể chứa bất thường hoặc chuẩn tính mới;
- cập nhật ngay có nguy cơ học nhầm bất thường.

**Nếu xác minh mọi cửa sổ:**

- tốn mã hóa và so khớp chữ ký;
- tăng độ trễ;
- không cần thiết cho các trường hợp đã rõ.

- **Chốt:** Tách hai bước là sự đánh đổi giữa chi phí tính toán và an toàn khi cập nhật.

### Trang 27 — Hàm mất mát tương phản trực tuyến

- **Giữ lại:** hàm mất mát tương phản giữa nhánh trực tuyến và nhánh nguồn.
- Biểu diễn trực tuyến phải gần biểu diễn nguồn.
- Biểu diễn trực tuyến phải tránh các mã vec-tơ của bất thường đã biết.
- Chỉ cập nhật mạng ánh xạ trực tuyến.
- **Công thức logic:**

  \[
  \text{gần biểu diễn nguồn},\qquad
  \text{xa biểu diễn bất thường đã biết}.
  \]

- **Chốt:** Hàm mất mát giúp thích ứng nhưng hạn chế trôi khỏi biểu diễn nguồn.

## 6. Độ đo và thiết kế thí nghiệm

**Câu hỏi của phần:** Ta dùng bằng chứng nào để kiểm tra phương pháp và giới hạn của bằng chứng đó là gì?

### Trang 28 — Cách tính VUS-PR

Vẽ một dải nhãn thật, điểm số và các vùng cửa sổ đánh giá:

1. Ghép điểm số theo đúng thứ tự chuỗi gốc.
2. Chọn nhiều ngưỡng để tạo dự đoán nhị phân.
3. Với mỗi phạm vi cửa sổ đánh giá, tính độ chính xác và độ bao phủ.
4. Vẽ đường cong độ chính xác–độ bao phủ.
5. Tính diện tích dưới đường cong cho từng phạm vi.
6. Lấy trung bình trên các phạm vi để nhận VUS-PR.

- **Chốt:** VUS-PR giảm sự phụ thuộc vào một ngưỡng và một độ dài cửa sổ duy nhất.

### Trang 29 — Cách tính Affiliation F1-score

1. Tách nhãn thật thành các khoảng bất thường.
2. Tách dự đoán thành các khoảng bất thường.
3. Tạo vùng liên kết quanh từng khoảng nhãn thật.
4. Phân chia các khoảng dự đoán theo vùng liên kết.
5. Tính xác suất độ chính xác liên kết.
6. Tính xác suất độ bao phủ liên kết.
7. Lấy trung bình điều hòa:

   \[
   F_1=\frac{2PR}{P+R}.
   \]

- **Chốt:** Độ đo này xét mức độ liên kết giữa các khoảng bất thường, không chỉ đếm từng điểm độc lập.

### Trang 30 — Bối cảnh dữ liệu thí nghiệm

| Chuỗi SMD | Điểm huấn luyện | Điểm xác thực | Điểm kiểm tra | Cửa sổ huấn luyện | Cửa sổ xác thực | Cửa sổ kiểm tra |
|---|---:|---:|---:|---:|---:|---:|
| machine-1-6 | 18.951 | 4.737 | 23.689 | 18.932 | 236 | 1.184 |
| machine-3-4 | 18.950 | 4.737 | 23.687 | 18.931 | 236 | 1.184 |
| machine-3-9 | 22.971 | 5.742 | 28.713 | 22.952 | 287 | 1.435 |

- Độ dài cửa sổ: \(L=20\); số kênh: \(C=38\).
- Mỗi chuỗi chạy ba hạt giống ngẫu nhiên và báo cáo kết quả trung bình.
- **Chốt:** Tập xác thực nhỏ hơn nhiều so với tập huấn luyện; vì vậy độ không chắc chắn của ngưỡng cần được thừa nhận.

### Trang 31 — Vì sao chọn ba chuỗi này?

- Cả ba chuỗi đều thuộc Server Machine Dataset.
- Cả ba có thay đổi phân phối giá trị đủ rõ giữa phần huấn luyện và phần kiểm tra.
- Không nên nói ba chuỗi đại diện cho toàn bộ Server Machine Dataset.
- **Chốt:** Lựa chọn này phù hợp để kiểm tra khả năng thích ứng dưới chuyển dịch phân phối, nhưng chưa đủ để kết luận rộng cho mọi hệ thống.

### Trang 32 — Vì sao phạm vi dữ liệu còn nhỏ?

- Thí nghiệm chỉ dùng ba chuỗi và một cấu hình cửa sổ.
- Phạm vi nhỏ giúp kiểm tra sâu luồng chạy, chi phí và các biến thể của phương pháp.
- Tuy nhiên, phạm vi nhỏ làm giảm khả năng khái quát kết luận.
- Cần kiểm tra thêm nhiều thực thể, nhiều mức chuyển dịch phân phối và nhiều loại bất thường.
- **Chốt:** “Ít dữ liệu” là giới hạn của bằng chứng, không phải bằng chứng rằng phương pháp luôn hoạt động tốt.

## 7. Kết quả, tỉ lệ báo động giả và kết luận

**Câu hỏi của phần:** Kết quả thực nghiệm xác nhận được điều gì, và chưa xác nhận được điều gì?

### Trang 33 — Kết quả pha ngoại tuyến

- **Giữ lại:** bảng so sánh THESIS O0, THESIS O1 và các phương pháp đối sánh.
- Giữ các số liệu VUS-PR, Affiliation F1-score và VUS-ROC quan trọng.
- Làm nổi bật cột trung bình nhưng không bỏ qua từng chuỗi.
- **Diễn giải trung thực:** THESIS đạt kết quả trung bình tốt trên VUS-PR và Affiliation F1-score trong bảng hiện tại.
- **So sánh biến thể:** O1 chỉ cải thiện rất ít so với O0; chưa đủ cơ sở nói hàm mất mát cân bằng tạo ra cải thiện lớn.
- **Chốt:** Kết quả gợi ý bộ nhớ đóng góp nhiều hơn hàm mất mát cân bằng, nhưng đây là diễn giải từ một phép so sánh hạn chế.

### Trang 34 — Kết quả định lượng độ bất định

- **Giữ lại:** bảng theo từng chuỗi và bảng trung bình.
- Hiển thị riêng điểm VUS-PR, Affiliation F1-score, độ bất định xác thực và độ bất định kiểm tra.
- **Diễn giải:** độ bất định trên kiểm tra cao hơn xác thực, trong khi các độ đo phát hiện vẫn ở mức tốt.
- **Cách nói thận trọng:** Kết quả phù hợp với khả năng tổng quát hóa mà không quá tự tin; chưa đủ để chứng minh độ bất định đã được hiệu chuẩn tốt.
- **Chốt:** Độ bất định là bằng chứng bổ trợ, không thay thế đánh giá phát hiện.

### Trang 35 — Kết quả pha trực tuyến

- **Giữ lại:** bảng so sánh THESIS với M2N2 và CANDI.
- Giữ các biến thể A0, A1 và A2.
- **Diễn giải:** các biến thể A0, A1 và A2 khác nhau rất ít trong bảng hiện tại.
- **Diễn giải thiết kế:** mức tăng chính có vẻ đến từ làm trơn điểm bằng EWMA hơn là từ cơ chế thích ứng; cần gọi đây là gợi ý, không phải kết luận nhân quả.
- **Chốt:** Pha trực tuyến cho kết quả tốt trong phạm vi thí nghiệm, nhưng đóng góp riêng của thích ứng chưa được tách rõ hoàn toàn.

### Trang 36 — Tỉ lệ báo động giả

- **Bổ sung bắt buộc:** một bảng nhỏ theo từng chuỗi và biến thể.
- **Công thức:**

  \[
  \text{Tỉ lệ báo động giả}
  =\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}.
  \]

- Báo cáo cùng số điểm báo động giả \(\mathrm{FP}\), số điểm bình thường đúng \(\mathrm{TN}\) và ngưỡng đã dùng.
- So sánh tỉ lệ báo động giả giữa xác thực và kiểm tra.
- Nếu hiện chưa có số liệu này, ghi rõ “chưa được báo cáo trong bảng hiện tại” và tính lại từ nhãn điểm cùng dự đoán điểm.
- Không suy ra tỉ lệ báo động giả từ VUS-PR hoặc Affiliation F1-score.
- **Chốt:** Một kết quả tốt trên VUS-PR hoặc F1 không đủ để kết luận hệ thống ít báo động giả.

### Trang 37 — Kết luận và giới hạn

Ba khung ngang:

1. **Đóng góp:** phối hợp truy vấn liên tục, truy vấn rời rạc, truy vấn ngẫu nhiên và thích ứng trực tuyến trong một khung đa nhiệm.
2. **Bằng chứng:** kết quả tốt trên các độ đo chính trong ba chuỗi đã chọn.
3. **Giới hạn:** phạm vi dữ liệu nhỏ; tỉ lệ báo động giả cần được báo cáo rõ; đóng góp riêng của thích ứng chưa tách hoàn toàn khỏi làm trơn điểm.

- **Hướng tiếp theo:** giảm tỉ lệ báo động giả, đánh giá chất lượng độ bất định và mở rộng số chuỗi.
- **Câu kết:** Phương pháp có tín hiệu thực nghiệm tích cực, nhưng cần thêm bằng chứng trước khi khẳng định khả năng sử dụng rộng rãi.

## 8. Quy tắc trình bày áp dụng cho toàn bộ bộ trang

- Mỗi trang chỉ giữ một thông điệp chính.
- Ưu tiên bố cục ngang và tối đa hai khung phẳng.
- Hình, công thức và câu chốt phải nói cùng một ý.
- Dùng màu xanh cho thành phần bị đóng băng, màu đỏ cho thành phần được cập nhật, màu vàng cho độ bất định và màu xám cho dữ liệu.
- Giữ cùng một mẫu cho bốn toán tử: **chọn hoặc gán trọng số → biểu diễn truy vấn → ý nghĩa**.
- Công thức chỉ giữ phần cần để hiểu luồng tính toán; chi tiết đầy đủ để trong báo cáo hoặc trang dự phòng.
- Bảng kết quả phải có một câu diễn giải ngay bên dưới.
- Không dùng chữ nhỏ để nhồi thêm nội dung; phần chi tiết nên chuyển sang trang dự phòng.
- Các trang phân cách mục nên có một câu hỏi hoặc kết luận, không chỉ lặp lại mục lục.

### Trang 38 — Tài liệu tham khảo

- **Giữ lại:** danh mục tài liệu tham khảo hiện có.
- Đưa trang này về cuối hoặc dùng làm trang dự phòng.
- Trong phần trình bày chính, chỉ hiện trích dẫn ngắn ngay tại hình hoặc bảng cần thiết.

## Căn cứ nội dung

- Tệp trình chiếu hiện tại: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/presentation/thesis_slides_22127208/slides/slides.tex`.
- Chương phương pháp: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/presentation/thesis_report_22127208/Chapter3/chapter3.tex`.
- Chương thực nghiệm: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/presentation/thesis_report_22127208/Chapter4/chapter4.tex`.
- Mã tiêm bất thường: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/data/augment.py`.
