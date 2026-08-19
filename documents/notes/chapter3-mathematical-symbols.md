# Ký hiệu toán học trong Chương 3 — Phương pháp đề xuất

Tài liệu này trích xuất ký hiệu từ nội dung Chương 3 do người dùng cung cấp. Ý nghĩa được giữ theo ngữ cảnh của chương; các ký hiệu bị dùng lại với nghĩa khác nhau được ghi rõ ở cuối tài liệu.

## 1. Chỉ số, kích thước và tập cơ bản

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $t$ | Window/time index | Chỉ số cửa sổ/thời gian | Chỉ số cửa sổ hoặc bước thời gian hiện tại, tùy ngữ cảnh. |
| $i,j$ | Within-window time-point indices | Chỉ số điểm thời gian trong cửa sổ | Chỉ số điểm thời gian bên trong một cửa sổ; $j$ còn được dùng làm chỉ số cộng. |
| $k$ | Prototype index | Chỉ số nguyên mẫu | Chỉ số nguyên mẫu trong một bộ nhớ. |
| $m$ | Stochastic-pass index | Chỉ số lần ngẫu nhiên | Chỉ số lần truy vấn hoặc lần lan truyền xuôi ngẫu nhiên. |
| $n$ | Global time-point index | Chỉ số điểm thời gian toàn cục | Chỉ số tuyệt đối/toàn cục của một điểm trên toàn bộ dòng dữ liệu. |
| $r$ | Class/occurrence index | Chỉ số lớp/lần xuất hiện | Chỉ số lớp trong hàm mất mát phân loại; ở phần EWMA, là số lần một điểm xuất hiện trong các cửa sổ đã được quyết định. |
| $u,v$ | Buffered-window indices | Chỉ số cửa sổ trong bộ đệm | Chỉ số cửa sổ trong bộ đệm xác minh; $v$ còn biểu thị phiên bản sạch hoặc bất thường nhân tạo ở pha ngoại tuyến. |
| $T$ | Window length | Độ dài cửa sổ | Số bước thời gian trong một cửa sổ. |
| $C$ | Number of channels | Số biến/kênh | Số biến/kênh của chuỗi thời gian. |
| $H$ | Latent dimension | Số chiều không gian ẩn | Số chiều của mỗi vec-tơ ẩn. |
| $B$ | Batch size | Kích thước lô | Số cửa sổ trong một lô dữ liệu. |
| $M$ | Number of stochastic passes | Số lần lan truyền ngẫu nhiên | Số lần lan truyền xuôi hoặc truy vấn ngẫu nhiên. |
| $K_c$ | Number of continuous prototypes | Số nguyên mẫu liên tục | Trong cài đặt được mô tả: $K_c=32$. |
| $K_d$ | Number of discrete prototypes | Số nguyên mẫu rời rạc | Trong cài đặt được mô tả: $K_d=60$. |
| $K_{\mathrm{top}}$ | Number of selected nearest prototypes | Số nguyên mẫu gần nhất được chọn | Trong cài đặt được mô tả: $K_{\mathrm{top}}=3$. |
| $\mathbb{R}^{T\times C}$ | Input-window space | Không gian cửa sổ đầu vào | Gồm $T$ bước thời gian và $C$ biến. |
| $\mathbb{R}^{T\times H}$ | Latent-tensor space | Không gian ten-xơ ẩn | Gồm $T$ vec-tơ, mỗi vec-tơ có $H$ chiều. |
| $\mathbb{R}^{B\times T\times H}$ | Batched latent-tensor space | Không gian ten-xơ ẩn theo lô | Không gian ten-xơ ẩn khi có thêm chiều lô dữ liệu. |

## 2. Dữ liệu đầu vào và bộ mã hóa

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{X}_t\in\mathbb{R}^{T\times C}$ | Input window | Cửa sổ đầu vào | Cửa sổ chuỗi thời gian tại bước $t$. |
| $\mathbf{x}_{t,i}$ | Input time-point vector | Vec-tơ điểm thời gian đầu vào | Vec-tơ quan sát tại điểm thời gian $i$ trong cửa sổ $t$. |
| $f_{\mathrm{enc}}$ | Shared encoder | Bộ mã hóa dùng chung | Mạng mã hóa được chia sẻ giữa hai tác vụ. |
| $\boldsymbol{\theta}_{\mathrm{enc}}$ | Encoder parameters | Tham số bộ mã hóa | Tập tham số của bộ mã hóa. |
| $\mathbf{Z}_t$ | Latent tensor | Ten-xơ trạng thái ẩn | $f_{\mathrm{enc}}(\mathbf{X}_t;\boldsymbol{\theta}_{\mathrm{enc}})$. |
| $\mathbf{z}_{t,i}$ | Latent vector | Vec-tơ trạng thái ẩn | Vec-tơ tại điểm $i$, được chuẩn hóa về độ lớn đơn vị. |
| $(\cdot)^\top$ | Transpose | Phép chuyển vị | Phép chuyển vị vec-tơ hoặc ma trận. |
| $\lVert\cdot\rVert_2$ | Euclidean norm | Chuẩn Euclid | $\lVert\cdot\rVert_2^2$ là bình phương chuẩn Euclid. |
| $\lVert\cdot\rVert_F$ | Frobenius norm | Chuẩn Frobenius | Chuẩn Frobenius của ma trận/ten-xơ. |
| $\operatorname{cos}(\mathbf{a},\mathbf{b})$ | Cosine similarity | Độ tương tự cosine | Độ tương tự cosine giữa hai vec-tơ. |

## 3. Bộ nhớ liên tục và truy vấn liên tục

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{P}^{(c)}$ | Continuous prototype bank | Tập nguyên mẫu liên tục | Ma trận nguyên mẫu thuộc $\mathbb{R}^{K_c\times H}$. |
| $\mathbf{p}^{(c)}_k$ | Continuous prototype | Nguyên mẫu liên tục | Nguyên mẫu thứ $k$, đại diện cho một mẫu hình bình thường. |
| $\tau_c>0$ | Continuous-query temperature | Nhiệt độ truy vấn liên tục | Siêu tham số nhiệt độ của toán tử truy vấn liên tục. |
| $\alpha^{(c)}_{t,i,k}$ | Deterministic continuous-query weight | Trọng số truy vấn liên tục tất định | Trọng số của $\mathbf{z}_{t,i}$ đối với nguyên mẫu $k$. |
| $\widetilde{\mathbf{z}}^{(c)}_{t,i}$ | Retrieved continuous-branch latent vector | Vec-tơ ẩn truy xuất từ nhánh liên tục | Đầu ra của truy vấn liên tục tất định. |
| $u^{(m)}_{t,i,k}$ | Uniform random variable | Biến ngẫu nhiên đều | Biến $\operatorname{Uniform}(0,1)$ dùng để sinh nhiễu Gumbel. |
| $g^{(m)}_{t,i,k}$ | Gumbel noise | Nhiễu Gumbel | Nhiễu i.i.d. của truy vấn liên tục ở lần $m$. |
| $\alpha^{(c,m)}_{t,i,k}$ | Stochastic continuous-query weight | Trọng số truy vấn liên tục ngẫu nhiên | Trọng số Gumbel–Softmax của nguyên mẫu $k$ ở lần $m$. |
| $\widetilde{\mathbf{z}}^{(c,m)}_{t,i}$ | Stochastic continuous-branch latent vector | Vec-tơ ẩn nhánh liên tục ngẫu nhiên | Vec-tơ được truy xuất ở lần $m$. |
| $\widetilde{\mathbf{Z}}^{(c,m)}_t$ | Continuous-branch latent tensor | Ten-xơ ẩn nhánh liên tục | Ghép các vec-tơ truy xuất; thuộc $\mathbb{R}^{T\times H}$. |
| $\overset{\mathrm{i.i.d.}}{\sim}$ | Independently and identically distributed | Độc lập và cùng phân phối | Quan hệ lấy mẫu độc lập từ cùng một phân phối. |

## 4. Bộ nhớ rời rạc và truy vấn rời rạc

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{P}^{(d)}$ | Discrete prototype bank/codebook | Tập nguyên mẫu rời rạc/bảng mã | Ma trận nguyên mẫu thuộc $\mathbb{R}^{K_d\times H}$. |
| $\mathbf{p}^{(d)}_k$ | Discrete prototype/code vector | Nguyên mẫu rời rạc/vec-tơ mã | Nguyên mẫu rời rạc thứ $k$. |
| $d^{(d)}_{t,i,k}$ | Squared prototype distance | Bình phương khoảng cách đến nguyên mẫu | Khoảng cách từ $\mathbf{z}_{t,i}$ đến $\mathbf{p}^{(d)}_k$. |
| $\mathcal{I}^{(d)}_{t,i}$ | Deterministic selected-index set | Tập chỉ số được chọn tất định | Chỉ số của $K_{\mathrm{top}}$ nguyên mẫu gần nhất. |
| $\tau_d>0$ | Discrete-query temperature | Nhiệt độ truy vấn rời rạc | Nhiệt độ của toán tử truy vấn rời rạc. |
| $\alpha^{(d)}_{t,i,k}$ | Deterministic discrete-query weight | Trọng số truy vấn rời rạc tất định | Bằng $0$ nếu $k\notin\mathcal{I}^{(d)}_{t,i}$. |
| $\widetilde{\mathbf{z}}^{(d)}_{t,i}$ | Retrieved discrete-branch latent vector | Vec-tơ ẩn truy xuất từ nhánh rời rạc | Đầu ra của truy vấn rời rạc tất định. |
| $u^{(d,m)}_{t,i,k}$ | Uniform random variable | Biến ngẫu nhiên đều | Biến $\operatorname{Uniform}(0,1)$ dùng để sinh nhiễu Gumbel. |
| $g^{(d,m)}_{t,i,k}$ | Discrete-query Gumbel noise | Nhiễu Gumbel truy vấn rời rạc | Nhiễu i.i.d. ở lần $m$. |
| $q^{(d,m)}_{t,i,k}$ | Stochastic discrete-query score | Điểm truy vấn rời rạc ngẫu nhiên | Kết hợp khoảng cách, nhiễu Gumbel và nhiệt độ. |
| $\mathcal{I}^{(d,m)}_{t,i}$ | Stochastic selected-index set | Tập chỉ số được chọn ngẫu nhiên | Chỉ số của $K_{\mathrm{top}}$ điểm truy vấn lớn nhất. |
| $\alpha^{(d,m)}_{t,i,k}$ | Stochastic discrete-query weight | Trọng số truy vấn rời rạc ngẫu nhiên | Trọng số của nguyên mẫu $k$ ở lần $m$. |
| $\widetilde{\mathbf{z}}^{(d,m)}_{t,i}$ | Stochastic discrete-branch latent vector | Vec-tơ ẩn nhánh rời rạc ngẫu nhiên | Vec-tơ được truy xuất ở lần $m$. |
| $\widetilde{\mathbf{Z}}^{(d,m)}_t$ | Discrete-branch latent tensor | Ten-xơ ẩn nhánh rời rạc | Ghép các vec-tơ truy xuất; thuộc $\mathbb{R}^{T\times H}$. |
| $\operatorname{TopKMin}$ | Top-$K$ minimum-index operator | Toán tử lấy chỉ số $K$ giá trị nhỏ nhất | Trả về chỉ số của $K$ giá trị nhỏ nhất. |
| $\operatorname{TopKMax}$ | Top-$K$ maximum-index operator | Toán tử lấy chỉ số $K$ giá trị lớn nhất | Trả về chỉ số của $K$ giá trị lớn nhất. |

## 5. Dữ liệu nhân tạo và giai đoạn A ngoại tuyến

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{X}^{(0)}_t$ | Clean window | Cửa sổ sạch | Cửa sổ lấy từ chuỗi huấn luyện gốc. |
| $a_t$ | Synthetic-anomaly class label | Nhãn lớp bất thường nhân tạo | $a_t\in\{1,\ldots,11\}$ cho cửa sổ $t$. |
| $\mathbf{X}^{(a_t)}_t$ | Synthetic anomalous view | Phiên bản bất thường nhân tạo | Được tạo từ $\mathbf{X}^{(0)}_t$. |
| $\mathbf{1}$ | All-ones vector | Vec-tơ toàn số một | Có kích thước phù hợp với phép toán. |
| $\mathbf{M}_t$ | Synthetic-anomaly mask | Mặt nạ bất thường nhân tạo | Xác định các bước thời gian được tiêm bất thường. |
| $\mathbf{M}_{t,i}$ | Mask value | Giá trị mặt nạ | $0$ là giữ nguyên; $1$ là được tiêm bất thường. |
| $\odot$ | Element-wise product | Phép nhân theo từng phần tử | Nhân các phần tử ở cùng vị trí. |
| $\mathcal{A}_{a_t}$ | Anomaly transformation | Hàm biến đổi bất thường | Hàm tương ứng với lớp $a_t$. |
| $\boldsymbol{\xi}_t$ | Random transformation parameters | Tham số biến đổi ngẫu nhiên | Tham số ngẫu nhiên của hàm biến đổi bất thường. |
| $\mathbf{Z}^{(0)}_t$ | Clean-view latent tensor | Ten-xơ ẩn của phiên bản sạch | Biểu diễn ẩn của cửa sổ sạch. |
| $\mathbf{Z}^{(a_t)}_t$ | Anomalous-view latent tensor | Ten-xơ ẩn của phiên bản bất thường | Biểu diễn ẩn của cửa sổ bất thường nhân tạo. |
| $\widehat{\mathbf{X}}^{(0)}_t$ | Reconstructed clean window | Cửa sổ sạch được tái tạo | Dự đoán từ đầu tái tạo. |
| $f_{\mathrm{rec}}$ | Reconstruction head | Đầu tái tạo | Mạng dự đoán dành cho tác vụ tái tạo. |
| $\boldsymbol{\theta}_{\mathrm{rec}}$ | Reconstruction-head parameters | Tham số đầu tái tạo | Tập tham số của $f_{\mathrm{rec}}$. |
| $f_{\mathrm{cls}}$ | Classification head | Đầu phân loại | Mạng dự đoán dành cho tác vụ phân loại. |
| $\boldsymbol{\theta}_{\mathrm{cls}}$ | Classification-head parameters | Tham số đầu phân loại | Tập tham số của $f_{\mathrm{cls}}$. |
| $\operatorname{Pool}$ | Pooling operator | Toán tử gộp | Văn bản mô tả thường là lấy trung bình. |
| $v\in\{0,a_t\}$ | View index | Chỉ số phiên bản dữ liệu | $v=0$ là sạch; $v=a_t$ là bất thường nhân tạo. |
| $\widehat{\mathbf{y}}^{(v)}_t$ | Predicted class-probability vector | Vec-tơ xác suất lớp dự đoán | Dự đoán cho phiên bản $v$ của cửa sổ $t$. |
| $y^{(v)}_{t,r}$ | One-hot class target | Nhãn lớp one-hot | Nhãn thật của lớp $r$ cho phiên bản $v$. |
| $\widehat y^{(v)}_{t,r}$ | Predicted class probability | Xác suất lớp dự đoán | Xác suất của lớp $r$ cho phiên bản $v$. |
| $\mathcal{Q}$ | Anchor set | Tập điểm neo | Tập các điểm neo trong học tương phản. |
| $\mathcal{P}(\mathbf{z}_q)$ | Positive set | Tập điểm dương | Các điểm dương của điểm neo $\mathbf{z}_q$. |
| $\mathcal{N}(\mathbf{z}_q)$ | Negative set | Tập điểm đối | Các điểm đối của điểm neo $\mathbf{z}_q$. |
| $\mathbf{z}_q$ | Anchor vector | Vec-tơ neo | Vec-tơ được dùng làm điểm neo. |
| $\mathbf{z}_p$ | Positive vector | Vec-tơ dương | Vec-tơ thuộc tập điểm dương. |
| $\mathbf{z}_n$ | Negative vector | Vec-tơ đối | Vec-tơ thuộc tập điểm đối. |
| $\tau_{\mathrm{con}}$ | Contrastive temperature | Nhiệt độ tương phản | Nhiệt độ trong mất mát tương phản ngoại tuyến. |
| $\mathcal{L}_{\mathrm{rec}}$ | Reconstruction loss | Hàm mất mát tái tạo | MSE/Frobenius trên cửa sổ sạch. |
| $\mathcal{L}_{\mathrm{cls}}$ | Classification loss | Hàm mất mát phân loại | Entropy chéo cho phân loại đa lớp. |
| $\mathcal{L}_{\mathrm{con}}$ | Point-level contrastive loss | Hàm mất mát tương phản cấp điểm | Mất mát tương phản ở cấp độ điểm thời gian. |
| $\mathcal{L}_{\mathrm{multi}}$ | Multi-task total loss | Hàm mất mát tổng đa tác vụ | Hàm mất mát tổng của giai đoạn A. |
| $\lambda_{\mathrm{rec}},\lambda_{\mathrm{cls}},\lambda_{\mathrm{con}}$ | Loss weights | Trọng số hàm mất mát | Hệ số của các thành phần mất mát tương ứng. |

## 6. Giai đoạn B và các mạng tổng hợp

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{Z}^{(\mathrm{cat},m)}_t$ | Concatenated latent tensor | Ten-xơ ẩn ghép | Ghép ba nhánh ở lần $m$; thuộc $\mathbb{R}^{T\times 3H}$. |
| $\operatorname{Concat}$ | Concatenation operator | Toán tử ghép | Ghép các ten-xơ theo chiều đặc trưng. |
| $f_{\mathrm{fus}}^{(\mathrm{rec})}$ | Reconstruction fusion network | Mạng tổng hợp tái tạo | Mạng tổng hợp dành cho tác vụ tái tạo. |
| $f_{\mathrm{fus}}^{(\mathrm{cls})}$ | Classification fusion network | Mạng tổng hợp phân loại | Mạng tổng hợp dành cho tác vụ phân loại. |
| $\boldsymbol{\theta}_{\mathrm{fus}}^{(\mathrm{rec})}$ | Reconstruction-fusion parameters | Tham số mạng tổng hợp tái tạo | Tham số của $f_{\mathrm{fus}}^{(\mathrm{rec})}$. |
| $\boldsymbol{\theta}_{\mathrm{fus}}^{(\mathrm{cls})}$ | Classification-fusion parameters | Tham số mạng tổng hợp phân loại | Tham số của $f_{\mathrm{fus}}^{(\mathrm{cls})}$. |
| $\mathbf{H}^{(\mathrm{rec},m)}_t$ | Reconstruction-fusion latent tensor | Ten-xơ ẩn tổng hợp tái tạo | Đầu ra của mạng tổng hợp tái tạo. |
| $\mathbf{H}^{(\mathrm{cls},m)}_t$ | Classification-fusion latent tensor | Ten-xơ ẩn tổng hợp phân loại | Đầu ra của mạng tổng hợp phân loại. |
| $\widehat{\mathbf{X}}^{(m)}_t$ | Reconstructed window | Cửa sổ được tái tạo | Kết quả ở lần lan truyền ngẫu nhiên $m$. |
| $\widehat{\mathbf{y}}^{(m)}_t$ | Predicted class-probability vector | Vec-tơ xác suất lớp dự đoán | Kết quả ở lần lan truyền ngẫu nhiên $m$. |
| $y_{t,r}$ | Class target | Nhãn lớp | Nhãn thật của lớp $r$ trong giai đoạn B. |
| $\widehat y^{(m)}_{t,r}$ | Predicted class probability | Xác suất lớp dự đoán | Xác suất dự đoán của lớp $r$ ở lần $m$. |
| $\mathcal{L}^{(B)}_{\mathrm{rec}}$ | Stage-B reconstruction loss | Hàm mất mát tái tạo giai đoạn B | Thành phần mất mát tái tạo. |
| $\mathcal{L}^{(B)}_{\mathrm{cls}}$ | Stage-B classification loss | Hàm mất mát phân loại giai đoạn B | Thành phần mất mát phân loại. |
| $\mathcal{L}_B$ | Stage-B total loss | Hàm mất mát tổng giai đoạn B | Tổng có trọng số của hai thành phần trên. |

## 7. Nhánh nguồn và nhánh trực tuyến

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathbf{Z}^{(\mathrm{src})}_t$ | Source latent tensor | Ten-xơ trạng thái ẩn nguồn | Do bộ mã hóa đã đóng băng tạo ra. |
| $\mathbf{z}^{(\mathrm{src})}_{t,i}$ | Source latent vector | Vec-tơ trạng thái ẩn nguồn | Vec-tơ nguồn tại vị trí $i$. |
| $g_{\mathrm{proj}}$ | Online MLP projector | Bộ ánh xạ MLP trực tuyến | Được khởi tạo xấp xỉ hàm định danh. |
| $\boldsymbol{\phi}$ | Projector parameters | Tham số bộ ánh xạ | Tham số duy nhất được cập nhật trong pha trực tuyến. |
| $\mathbf{Z}^{(\mathrm{on})}_t$ | Online latent tensor | Ten-xơ trạng thái ẩn trực tuyến | Do $g_{\mathrm{proj}}$ tạo ra. |
| $\mathbf{z}^{(\mathrm{on})}_{t,i}$ | Online latent vector | Vec-tơ trạng thái ẩn trực tuyến | Vec-tơ trực tuyến tại vị trí $i$. |

## 8. Điểm bất thường, EWMA và quy tắc phân luồng

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $s^{(m)}_{t,i}$ | Point-level reconstruction error | Độ lệch tái tạo cấp điểm | MSE tại vị trí $i$, cửa sổ $t$, lần lan truyền $m$. |
| $\overline{s}_{t,i}$ | Mean reconstruction error | Độ lệch tái tạo trung bình | Trung bình qua $M$ lần lan truyền ngẫu nhiên. |
| $\overline{s}^{(r)}_n$ | Occurrence-wise mean reconstruction error | Độ lệch tái tạo trung bình theo lần xuất hiện | Giá trị của điểm toàn cục $n$ ở lần xuất hiện thứ $r$. |
| $\widetilde{s}^{(r)}_n$ | EWMA-smoothed anomaly score | Điểm bất thường được làm trơn bằng EWMA | Giá trị đã làm trơn đến lần xuất hiện thứ $r$. |
| $\rho$ | EWMA smoothing factor | Hệ số làm trơn EWMA | Văn bản đặt $\rho=0.9$. |
| $T_{\mathrm{point}}$ | Point-level anomaly threshold | Ngưỡng bất thường cấp điểm | Được hiệu chỉnh từ chuỗi xác thực gốc. |
| $\widehat a_n$ | Predicted anomaly indicator | Chỉ thị bất thường dự đoán | Nhãn nhị phân của điểm toàn cục $n$. |
| $\mathbb{I}(\cdot)$ | Indicator function | Hàm chỉ thị | Trả về $1$ nếu điều kiện đúng, ngược lại trả về $0$. |
| $S^{(\mathrm{input})}_t$ | Window-level reconstruction error | Độ lệch tái tạo cấp cửa sổ | Trung bình các điểm đã làm trơn trong cửa sổ. |
| $S^{(\mathrm{latent})}_t$ | Window-level latent distance | Khoảng cách ẩn cấp cửa sổ | Khoảng cách đến các nguyên mẫu liên tục gần nhất. |
| $B_{\mathrm{win}}$ | Window reconstruction-error threshold | Ngưỡng độ lệch tái tạo cửa sổ | Ngưỡng cho $S^{(\mathrm{input})}_t$. |
| $A_{\mathrm{win}}^{(\mathrm{low})}$ | Lower latent-distance threshold | Ngưỡng dưới của khoảng cách ẩn | Ngưỡng dưới cho $S^{(\mathrm{latent})}_t$. |
| $A_{\mathrm{win}}^{(\mathrm{high})}$ | Upper latent-distance threshold | Ngưỡng trên của khoảng cách ẩn | Ngưỡng trên cho $S^{(\mathrm{latent})}_t$. |
| $\operatorname{Triage}(\mathbf{X}_t)$ | Triage rule | Quy tắc phân luồng | Phân cửa sổ thành bốn luồng được mô tả trong chương. |
| $\land$ | Logical conjunction | Phép hội logic | Biểu thị điều kiện “và”. |

## 9. Thích ứng hard-old-normality

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $S^{(\mathrm{hard})}_t(\boldsymbol{\phi})$ | Differentiable hard-window reconstruction error | Độ lệch tái tạo khả vi của cửa sổ khó | Lấy trung bình qua lần lan truyền, điểm thời gian và biến. |
| $[x]_+$ | Positive-part operator | Toán tử lấy phần dương | $[x]_+=\max(0,x)$, dùng trong hàm phạt kiểu hinge. |
| $\mathcal{L}_{\mathrm{hard\text{-}rec}}$ | Hard-window reconstruction penalty | Hàm phạt tái tạo cửa sổ khó | Phạt khi $S^{(\mathrm{hard})}_t(\boldsymbol{\phi})>B_{\mathrm{win}}$. |
| $c^{(d)}_k$ | Discrete-prototype class label | Nhãn lớp của nguyên mẫu rời rạc | Lớp $0$ là bình thường. |
| $\mathcal{K}_{\mathrm{anom}}$ | Anomalous-prototype index set | Tập chỉ số nguyên mẫu bất thường | Chứa các nguyên mẫu rời rạc có nhãn khác $0$. |
| $\tau_{\mathrm{on}}$ | Online contrastive temperature | Nhiệt độ tương phản trực tuyến | Nhiệt độ của mất mát tương phản trực tuyến. |
| $\mathcal{L}_{\mathrm{src\text{-}on}}$ | Source–online contrastive loss | Hàm mất mát tương phản nguồn–trực tuyến | Nguyên mẫu rời rạc bất thường làm điểm đối. |
| $\mathcal{L}_{\mathrm{hard}}$ | Hard-old-normality adaptation loss | Hàm mất mát thích ứng bình thường cũ khó | Mất mát tổng cho cửa sổ `hard-old-normality`. |

## 10. Xác minh pseudo-new-normality

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\kappa^{(d)}_{u,i}$ | Nearest discrete-prototype index | Chỉ số nguyên mẫu rời rạc gần nhất | Gần $\mathbf{z}^{(\mathrm{on})}_{u,i}$ nhất. |
| $r^{(d)}_k$ | Discrete-cluster covering radius | Bán kính bao phủ cụm rời rạc | Bán kính của cụm ứng với nguyên mẫu thứ $k$. |
| $M^{(\mathrm{known\text{-}anom})}_{u,i}$ | Known-anomaly mask | Mặt nạ bất thường đã biết | Cho biết điểm có khớp một cụm bất thường đã biết hay không. |
| $\boldsymbol{\sigma}_{u,i}$ | Continuous-prototype signature | Chữ ký nguyên mẫu liên tục | Ba chỉ số nguyên mẫu gần nhất theo thứ tự khoảng cách tăng dần. |
| $\operatorname{OrderedTopKMin}$ | Ordered top-$K$ minimum operator | Toán tử lấy có thứ tự $K$ giá trị nhỏ nhất | Trả về chỉ số và giữ thứ tự khoảng cách tăng dần. |
| $\mathcal{B}$ | Verification buffer | Bộ đệm xác minh | Gồm các cửa sổ không chồng lấp. |
| $R_{\mathcal{B}}(\boldsymbol{\sigma})$ | Cross-window signature recurrence count | Số lần chữ ký lặp lại qua các cửa sổ | Đếm số cửa sổ khác nhau chứa chữ ký hợp lệ. |
| $\exists$ | Existential quantifier | Lượng từ tồn tại | Biểu thị “tồn tại ít nhất một”. |
| $M^{(\mathrm{pnn})}_{u,i}$ | Pseudo-new-normality mask | Mặt nạ giả định bình thường mới | Cho biết điểm đã được xác minh là pseudo-new-normality hay chưa. |
| $\mathcal{V}$ | Verified point set | Tập điểm đã được xác minh | Các cặp $(u,i)$ có $M^{(\mathrm{pnn})}_{u,i}=1$. |
| $\lvert\mathcal{V}\rvert$ | Verified-set cardinality | Lực lượng của tập điểm đã xác minh | Số phần tử trong $\mathcal{V}$. |
| $\mathcal{L}_{\mathrm{pnn\text{-}rec}}$ | PNN point-level reconstruction loss | Hàm mất mát tái tạo cấp điểm PNN | Chỉ tính trên các điểm thuộc $\mathcal{V}$. |
| $\mathcal{L}^{(\mathrm{pnn})}_{\mathrm{src\text{-}on}}$ | PNN source–online contrastive loss | Hàm mất mát tương phản nguồn–trực tuyến PNN | Chỉ tính trên các điểm thuộc $\mathcal{V}$. |
| $\mathcal{L}_{\mathrm{pnn}}$ | PNN adaptation loss | Hàm mất mát thích ứng PNN | Mất mát tổng cho pseudo-new-normality đã xác minh. |

## 11. Ký hiệu chỉ xuất hiện trong giả mã

| Ký hiệu | Tên tiếng Anh | Tên tiếng Việt | Ý nghĩa |
|---|---|---|---|
| $\mathcal{D}_{\mathrm{train}}$ | Training dataset | Tập dữ liệu huấn luyện | Dữ liệu dùng để huấn luyện mô hình. |
| $\mathcal{D}_{\mathrm{val}}$ | Validation dataset | Tập dữ liệu xác thực | Dữ liệu dùng để đánh giá và chọn checkpoint. |
| $\Omega$ | Global configuration | Bộ cấu hình chung | Cấu hình của toàn quy trình. |
| $\Omega_p$ | Stage-specific configuration | Bộ cấu hình theo giai đoạn | Cấu hình dành cho giai đoạn $p$. |
| $\mathcal{P}=\{\mathrm{A},\mathrm{B}\}$ | Offline-stage set | Tập giai đoạn ngoại tuyến | Gồm hai giai đoạn A và B. |
| $p$ | Stage index | Chỉ số giai đoạn | Nhận giá trị A hoặc B. |
| $\Theta^*$ | Best offline checkpoint | Checkpoint ngoại tuyến tốt nhất | Checkpoint được chọn sau pha ngoại tuyến. |
| $X,Y,M$ | Batch input, target, and mask | Đầu vào, nhãn và mặt nạ của lô | Ba thành phần của một lô dữ liệu trong giả mã. |
| $\mathcal{L}$ | Generic total loss | Hàm mất mát tổng dạng chung | Ký hiệu viết gọn trong giả mã. |
| $\mathcal{S}$ | Online data stream | Dòng dữ liệu trực tuyến | Dòng cửa sổ dùng trong pha trực tuyến. |
| $\Theta_{\mathrm{online}}$ | Online model state | Trạng thái mô hình trực tuyến | Được khởi tạo từ $\Theta^*$. |
| $\mathcal{B}_{\mathrm{ver}}$ | Verification buffer | Bộ đệm xác minh | Lưu các cửa sổ cần xác minh. |
| $\mathcal{B}_{\mathrm{ttl}}$ | Time-to-Live buffer | Bộ đệm thời hạn tồn tại | Bộ đệm TTL trong giả mã trực tuyến. |
| $s_t$ | Generic window anomaly score | Điểm bất thường cửa sổ dạng chung | Ký hiệu khái quát trong giả mã. |
| $\delta_{\mathrm{normal}}$ | Normal-routing threshold | Ngưỡng phân luồng bình thường | Ngưỡng dưới của `RouteAndVerifyPNN`. |
| $\delta_{\mathrm{anomaly}}$ | Anomaly-routing threshold | Ngưỡng phân luồng bất thường | Ngưỡng trên của `RouteAndVerifyPNN`. |
| $M_t$ trong giả mã trực tuyến | Adaptation-point mask/set | Mặt nạ/tập điểm thích ứng | $\varnothing$ nghĩa là không có điểm nào được chọn. |
| $\gets$ | Assignment operator | Toán tử gán | Gán giá trị trong giả mã. |
| $\varnothing$ | Empty set | Tập rỗng | Tập không chứa phần tử nào. |

## 12. Các trường hợp dùng lại hoặc chưa đồng nhất ký hiệu

1. $M$ là số lần lan truyền ngẫu nhiên, còn $\mathbf{M}_t$ là mặt nạ tiêm bất thường. Trong giả mã trực tuyến, $M_t$ lại biểu thị mặt nạ/tập điểm dùng để thích ứng.
2. $\mathcal{P}$ vừa là tập các giai đoạn $\{\mathrm{A},\mathrm{B}\}$ trong giả mã, vừa xuất hiện trong $\mathcal{P}(\mathbf{z}_q)$ với nghĩa tập điểm dương.
3. $r$ vừa là chỉ số lớp trong mất mát phân loại, vừa là số lần một điểm xuất hiện trong công thức EWMA.
4. $v$ vừa biểu thị phiên bản dữ liệu $v\in\{0,a_t\}$, vừa là chỉ số cửa sổ trong tổng trên bộ đệm xác minh.
5. $\mathcal{B}$ là bộ đệm xác minh trong phần công thức, còn $B$ là kích thước lô và $B_{\mathrm{win}}$ là ngưỡng cấp cửa sổ.
6. Giả mã dùng $s_t$, $\delta_{\mathrm{normal}}$ và $\delta_{\mathrm{anomaly}}$, trong khi phần công thức chi tiết dùng $S_t^{(\mathrm{input})}$, $S_t^{(\mathrm{latent})}$, $B_{\mathrm{win}}$, $A_{\mathrm{win}}^{(\mathrm{low})}$ và $A_{\mathrm{win}}^{(\mathrm{high})}$. Văn bản chưa nêu ánh xạ tường minh giữa hai hệ ký hiệu.
7. Hai công thức ten-xơ nhánh liên tục/rời rạc trong nguồn viết $\mathbf{R}^{T\times H}$; ký hiệu không gian số thực chuẩn là $\mathbb{R}^{T\times H}$.
8. Cặp công thức định nghĩa $\widetilde{\mathbf{Z}}^{(c,m)}_t$ và $\widetilde{\mathbf{Z}}^{(d,m)}_t$ xuất hiện lặp hai lần trong nội dung nguồn.
