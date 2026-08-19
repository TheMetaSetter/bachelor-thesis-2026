# Bản nháp hệ thống ký hiệu Chương 3 theo hướng giảm tải nhận thức

> **Trạng thái:** Bản nháp để rà soát, chưa phải quy ước chính thức và chưa được áp dụng vào nguồn LaTeX.
>
> **Mục tiêu:** Giúp người đọc hiểu cách cấu tạo ký hiệu trước khi phải ghi nhớ từng ký hiệu. Ký hiệu gốc được giữ khi nó đã rõ; mọi đề xuất đổi tên đều được ánh xạ ở cuối tài liệu.

## 1. Cách đọc ký hiệu trong khoảng một phút

### 1.1. Kiểu chữ cho biết loại đối tượng

| Hình thức | Loại đối tượng | Ví dụ |
|---|---|---|
| Chữ đậm viết hoa | Cửa sổ, ma trận hoặc ten-xơ | $\mathbf X_t$, $\mathbf Z_t$, $\mathbf P^{(b)}$ |
| Chữ đậm viết thường | Vec-tơ tại một điểm thời gian | $\mathbf x_{t,i}$, $\mathbf z_{t,i}$, $\mathbf p_k^{(b)}$ |
| Chữ thường | Đại lượng vô hướng | $s_{t,i}$, $d_{t,i,k}$, $\alpha_{t,i,k}$ |
| Chữ viết hoa kiểu thư pháp | Tập hợp hoặc hàm mất mát | $\mathcal V$, $\mathcal K_{\mathrm{anom}}$, $\mathcal L$ |
| Chữ Hy Lạp đậm | Tập tham số của mô hình | $\boldsymbol\theta$, $\boldsymbol\phi$ |

### 1.2. Chỉ số dưới cho biết vị trí

Các chỉ số luôn được đọc từ phạm vi lớn đến phạm vi nhỏ:

| Chỉ số | English | Tiếng Việt |
|---|---|---|
| $t$ | window index | chỉ số cửa sổ hiện tại trong lô dữ liệu (batch of data samples) |
| $i,j$ | within-window time-point indices | chỉ số điểm thời gian trong cửa sổ |
| $k$ | prototype index | chỉ số nguyên mẫu |
| $m$ | stochastic-pass index | chỉ số lần lan truyền hoặc truy vấn ngẫu nhiên |
| $n$ | global time-point index | chỉ số điểm thời gian trên toàn dòng dữ liệu |
| $r$ | occurrence index | số lần một điểm toàn cục đã xuất hiện |
| $\ell$ | class index | chỉ số lớp |
| $w$ | buffered-window index | chỉ số cửa sổ trong bộ đệm xác minh |

Ví dụ:

$
\alpha^{(c,m)}_{t,i,k}
$

là trọng số của **nguyên mẫu $k$**, dành cho **điểm $i$ trong cửa sổ $t$**, ở **lần truy vấn ngẫu nhiên $m$**, thuộc **nhánh liên tục $c$**.

### 1.3. Chỉ số trên đặt trong ngoặc là nhãn, không phải lũy thừa

| Nhãn | English | Tiếng Việt |
|---|---|---|
| $c$ | continuous branch | nhánh liên tục |
| $d$ | discrete branch | nhánh rời rạc |
| $\mathrm{rec}$ | reconstruction task | tác vụ tái tạo |
| $\mathrm{cls}$ | classification task | tác vụ phân loại |
| $\mathrm{src}$ | source branch | nhánh nguồn |
| $\mathrm{on}$ | online branch | nhánh trực tuyến |
| $\mathrm{pnn}$ | pseudo-new-normality | giả định bình thường mới |
| $\mathrm{anom}$ | anomaly | bất thường |

Hai biến nhãn được dùng để viết các cấu trúc chung:

$
b\in\{c,d\},
\qquad
h\in\{\mathrm{rec},\mathrm{cls}\}.
$

Trong đó, $b$ chọn **nhánh bộ nhớ**, còn $h$ chọn **tác vụ**.

### 1.4. Dấu trang trí cho biết trạng thái xử lý

| Dấu | Cách đọc thống nhất | Ví dụ |
|---|---|---|
| $\widehat{\cdot}$ | giá trị do mô hình dự đoán | $\widehat{\mathbf X}_t$, $\widehat{\mathbf y}_t$ |
| $\overline{\cdot}$ | giá trị trung bình | $\overline s_{t,i}$ |
| $\widetilde{\cdot}$ | giá trị đã được truy xuất hoặc làm trơn | $\widetilde{\mathbf z}_{t,i}$, $\widetilde s_n$ |
| $(\cdot)^{(m)}$ | giá trị tại lần ngẫu nhiên thứ $m$ | $s^{(m)}_{t,i}$ |

Ý nghĩa cụ thể của $\widetilde{\cdot}$ được xác định bởi loại đại lượng: với $\mathbf z$, nó biểu thị vec-tơ đã truy xuất; với $s$, nó biểu thị điểm đã làm trơn.

## 2. Luồng ký hiệu tổng quát

Người đọc có thể theo dõi toàn bộ phương pháp theo chuỗi sau:

$
\underbrace{\mathbf X_t}_{\text{cửa sổ đầu vào}}
\xrightarrow{f_{\mathrm{enc}}}
\underbrace{\mathbf Z_t}_{\text{biểu diễn ẩn}}
\xrightarrow{\mathbf P^{(c)},\,\mathbf P^{(d)}}
\underbrace{\widetilde{\mathbf Z}^{(c,m)}_t,
\widetilde{\mathbf Z}^{(d,m)}_t}_{\text{biểu diễn truy xuất}}
\xrightarrow{f_{\mathrm{fus}}^{(h)}}
\underbrace{\mathbf H_t^{(h,m)}}_{\text{biểu diễn theo tác vụ}}
\xrightarrow{f_{\mathrm{rec}},\,f_{\mathrm{cls}}}
\underbrace{\widehat{\mathbf X}^{(m)}_t,
\widehat{\mathbf y}^{(m)}_t}_{\text{đầu ra}}.
$

Trong pha trực tuyến, đầu ra tái tạo được chuyển thành điểm bất thường:

$
s^{(m)}_{t,i}
\xrightarrow{\text{trung bình qua }m}
\overline s_{t,i}
\xrightarrow{\mathrm{EWMA}}
\widetilde s_n
\xrightarrow{\text{gộp theo cửa sổ}}
S_t^{(\mathrm{input})}.
$

## 3. Các đối tượng cốt lõi

Đây là nhóm ký hiệu người đọc nên biết trước. Các ký hiệu dẫn xuất được định nghĩa tại đúng mục sử dụng.

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $\mathbf X_t\in\mathbb R^{T\times C}$ | input window | cửa sổ đầu vào | Chứa $T$ điểm thời gian và $C$ biến. |
| $\mathbf x_{t,i}\in\mathbb R^C$ | input time-point vector | vec-tơ điểm thời gian đầu vào | Hàng thứ $i$ của $\mathbf X_t$. |
| $\mathbf Z_t\in\mathbb R^{T\times H}$ | latent tensor | ten-xơ trạng thái ẩn | Đầu ra của bộ mã hóa. |
| $\mathbf z_{t,i}\in\mathbb R^H$ | latent vector | vec-tơ trạng thái ẩn | Hàng thứ $i$ của $\mathbf Z_t$; có độ lớn đơn vị. |
| $\mathbf P^{(b)}\in\mathbb R^{K_b\times H}$ | prototype bank | tập nguyên mẫu | Bộ nhớ của nhánh $b$. |
| $\mathbf p_k^{(b)}\in\mathbb R^H$ | prototype | nguyên mẫu | Nguyên mẫu thứ $k$ của nhánh $b$. |
| $\alpha_{t,i,k}^{(b,m)}$ | query weight | trọng số truy vấn | Mức đóng góp của nguyên mẫu $k$. |
| $\widetilde{\mathbf z}_{t,i}^{(b,m)}$ | retrieved latent vector | vec-tơ ẩn được truy xuất | Kết quả truy vấn bộ nhớ nhánh $b$. |
| $\widehat{\mathbf X}_t^{(m)}$ | reconstructed window | cửa sổ được tái tạo | Đầu ra của tác vụ tái tạo. |
| $\widehat{\mathbf y}_t^{(m)}$ | predicted class probabilities | xác suất lớp dự đoán | Đầu ra của tác vụ phân loại. |
| $s_{t,i}^{(m)}$ | point-level reconstruction error | độ lệch tái tạo cấp điểm | Điểm thô ở lần ngẫu nhiên $m$. |
| $S_t^{(\mathrm{input})}$ | window reconstruction error | độ lệch tái tạo cấp cửa sổ | Điểm tái tạo dùng để phân luồng. |
| $S_t^{(\mathrm{latent})}$ | window latent distance | khoảng cách ẩn cấp cửa sổ | Mức xa lạ so với các nguyên mẫu bình thường. |
| $\mathcal L$ | loss function | hàm mất mát | Đại lượng được tối thiểu hóa khi huấn luyện hoặc thích ứng. |

Các kích thước dùng chung là:

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $T$ | window length | số điểm thời gian trong cửa sổ |
| $C$ | number of channels | số biến của chuỗi thời gian |
| $H$ | latent dimension | số chiều của vec-tơ ẩn |
| $K_c$ | number of continuous prototypes | số nguyên mẫu liên tục |
| $K_d$ | number of discrete prototypes | số nguyên mẫu rời rạc |
| $M$ | number of stochastic passes | số lần lan truyền ngẫu nhiên |

## 4. Bộ mã hóa và hai bộ nhớ

### 4.1. Bộ mã hóa

$
\mathbf Z_t
=
f_{\mathrm{enc}}
\left(
\mathbf X_t;
\boldsymbol\theta_{\mathrm{enc}}
\right).
$

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $f_{\mathrm{enc}}$ | shared encoder | bộ mã hóa dùng chung |
| $\boldsymbol\theta_{\mathrm{enc}}$ | encoder parameters | tham số bộ mã hóa |

### 4.2. Quy ước chung cho hai bộ nhớ

Với $b\in\{c,d\}$:

$
\mathbf P^{(b)}
=
\begin{bmatrix}
\mathbf p_1^{(b)};
\ldots;
\mathbf p_{K_b}^{(b)}
\end{bmatrix}.
$

Trọng số truy vấn tạo ra vec-tơ truy xuất:

$
\widetilde{\mathbf z}_{t,i}^{(b,m)}
=
\sum_{k=1}^{K_b}
\alpha_{t,i,k}^{(b,m)}
\mathbf p_k^{(b)}.
$

Với nhánh rời rạc, $\alpha_{t,i,k}^{(d,m)}=0$ nếu nguyên mẫu $k$ không được chọn. Vì vậy, tổng trên toàn bộ $k=1,\ldots,K_d$ vẫn tương đương với tổng chỉ trên tập nguyên mẫu được chọn. Quy ước này chỉ gộp **cấu trúc đầu vào và đầu ra**. Cách tính $\alpha^{(c,m)}$ và $\alpha^{(d,m)}$ vẫn phải trình bày riêng vì hai toán tử truy vấn có ngữ nghĩa khác nhau.

<details>
<summary><strong>Ký hiệu chuyên biệt của truy vấn liên tục</strong></summary>

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $\tau_c$ | continuous-query temperature | nhiệt độ truy vấn liên tục | Điều chỉnh độ tập trung của trọng số. |
| $u^{(c,m)}_{t,i,k}$ | uniform random variable | biến ngẫu nhiên đều | Dùng để sinh nhiễu Gumbel. |
| $g^{(c,m)}_{t,i,k}$ | continuous-query Gumbel noise | nhiễu Gumbel truy vấn liên tục | Tạo tính ngẫu nhiên cho truy vấn. |
| $\alpha^{(c,m)}_{t,i,k}$ | stochastic continuous-query weight | trọng số truy vấn liên tục ngẫu nhiên | Trọng số Gumbel--Softmax của nguyên mẫu $k$. |
| $\widetilde{\mathbf Z}^{(c,m)}_t$ | continuous-branch latent tensor | ten-xơ ẩn nhánh liên tục | Ghép các $\widetilde{\mathbf z}^{(c,m)}_{t,i}$. |

</details>

<details>
<summary><strong>Ký hiệu chuyên biệt của truy vấn rời rạc</strong></summary>

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $d^{(d)}_{t,i,k}$ | squared prototype distance | bình phương khoảng cách đến nguyên mẫu | Đo khoảng cách từ $\mathbf z_{t,i}$ đến $\mathbf p_k^{(d)}$. |
| $\tau_d$ | discrete-query temperature | nhiệt độ truy vấn rời rạc | Điều chỉnh độ tập trung của trọng số. |
| $q^{(d,m)}_{t,i,k}$ | stochastic discrete-query score | điểm truy vấn rời rạc ngẫu nhiên | Kết hợp khoảng cách, nhiễu và nhiệt độ. |
| $\mathcal I^{(d,m)}_{t,i}$ | selected-prototype index set | tập chỉ số nguyên mẫu được chọn | Chứa $K_{\mathrm{top}}$ nguyên mẫu có điểm lớn nhất. |
| $\alpha^{(d,m)}_{t,i,k}$ | stochastic discrete-query weight | trọng số truy vấn rời rạc ngẫu nhiên | Bằng $0$ nếu $k\notin\mathcal I^{(d,m)}_{t,i}$. |
| $\widetilde{\mathbf Z}^{(d,m)}_t$ | discrete-branch latent tensor | ten-xơ ẩn nhánh rời rạc | Ghép các $\widetilde{\mathbf z}^{(d,m)}_{t,i}$. |

</details>

## 5. Pha ngoại tuyến

### 5.1. Hai phiên bản của cùng một cửa sổ

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $\mathbf X_t^{(0)}$ | clean view | phiên bản sạch |
| $a_t\in\{1,\ldots,11\}$ | synthetic-anomaly class | lớp bất thường nhân tạo |
| $\mathbf X_t^{(a_t)}$ | synthetic anomalous view | phiên bản bất thường nhân tạo |
| $\mathbf M_t^{(\mathrm{inj})}$ | injection mask | mặt nạ tiêm bất thường |
| $\mathcal A_{a_t}$ | anomaly transformation | hàm biến đổi bất thường |
| $\boldsymbol\xi_t$ | random transformation parameters | tham số biến đổi ngẫu nhiên |

Chỉ số phiên bản được viết gọn bằng

$
v\in\{0,a_t\}.
$

Do đó, $\mathbf X_t^{(v)}$, $\mathbf Z_t^{(v)}$ và $\widehat{\mathbf y}_t^{(v)}$ lần lượt là cửa sổ, biểu diễn ẩn và dự đoán lớp của phiên bản $v$.

### 5.2. Ba hàm mất mát của giai đoạn A

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $\mathcal L_{\mathrm{rec}}$ | reconstruction loss | hàm mất mát tái tạo |
| $\mathcal L_{\mathrm{cls}}$ | classification loss | hàm mất mát phân loại |
| $\mathcal L_{\mathrm{con}}$ | point-level contrastive loss | hàm mất mát tương phản cấp điểm |
| $\lambda_{\mathrm{rec}}$ | reconstruction-loss weight | trọng số mất mát tái tạo |
| $\lambda_{\mathrm{cls}}$ | classification-loss weight | trọng số mất mát phân loại |
| $\lambda_{\mathrm{con}}$ | contrastive-loss weight | trọng số mất mát tương phản |

$
\mathcal L_{\mathrm{multi}}
=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}.
$

Trong tổng phân loại, dùng $\ell$ làm chỉ số lớp để không trùng với $r$, vốn được dành cho số lần xuất hiện trong EWMA.

### 5.3. Hai mạng tổng hợp của giai đoạn B

Với $h\in\{\mathrm{rec},\mathrm{cls}\}$:

$
\mathbf H_t^{(h,m)}
=
f_{\mathrm{fus}}^{(h)}
\left(
\mathbf Z_t^{(\mathrm{cat},m)};
\boldsymbol\theta_{\mathrm{fus}}^{(h)}
\right).
$

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $\mathbf Z_t^{(\mathrm{cat},m)}$ | concatenated latent tensor | ten-xơ ẩn ghép |
| $f_{\mathrm{fus}}^{(h)}$ | task-specific fusion network | mạng tổng hợp theo tác vụ |
| $\boldsymbol\theta_{\mathrm{fus}}^{(h)}$ | fusion-network parameters | tham số mạng tổng hợp |
| $\mathbf H_t^{(h,m)}$ | task-specific latent tensor | ten-xơ ẩn theo tác vụ |

## 6. Pha trực tuyến

### 6.1. Nhánh nguồn và nhánh trực tuyến

$
\mathbf Z_t^{(\mathrm{src})}
=
f_{\mathrm{enc}}
\left(
\mathbf X_t;
\boldsymbol\theta_{\mathrm{enc}}
\right),
\qquad
\mathbf Z_t^{(\mathrm{on})}
=
g_{\mathrm{proj}}
\left(
\mathbf Z_t^{(\mathrm{src})};
\boldsymbol\phi
\right).
$

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $\mathbf Z_t^{(\mathrm{src})}$ | source latent tensor | ten-xơ trạng thái ẩn nguồn | Biểu diễn cố định từ mô hình ngoại tuyến. |
| $g_{\mathrm{proj}}$ | online MLP projector | bộ ánh xạ MLP trực tuyến | Ánh xạ biểu diễn nguồn sang biểu diễn trực tuyến. |
| $\boldsymbol\phi$ | projector parameters | tham số bộ ánh xạ | Tham số duy nhất được cập nhật trực tuyến. |
| $\mathbf Z_t^{(\mathrm{on})}$ | online latent tensor | ten-xơ trạng thái ẩn trực tuyến | Biểu diễn được dùng cho thích ứng. |

### 6.2. Chuỗi biến đổi điểm bất thường

Điểm thô của một lần lan truyền:

$$
s^{(m)}_{t,i}
=
\frac{1}{C}
\left\lVert
\mathbf x_{t,i}
-
\widehat{\mathbf x}^{(m)}_{t,i}
\right\rVert_2^2.
$$

Điểm trung bình qua $M$ lần lan truyền:

$$
\overline s_{t,i}
=
\frac{1}{M}
\sum_{m=1}^{M}s^{(m)}_{t,i}.
$$

Điểm được làm trơn theo chỉ số toàn cục $n$:

$$
\widetilde s_n^{(r)}
=
\rho\,\overline s_n^{(r)}
+
(1-\rho)\widetilde s_n^{(r-1)}.
$$

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $\rho$ | EWMA smoothing factor | hệ số làm trơn EWMA |
| $\widehat a_n$ | predicted anomaly indicator | chỉ thị bất thường dự đoán |
| $\mathbb I(\cdot)$ | indicator function | hàm chỉ thị |

Nếu cần liên hệ chỉ số cục bộ và toàn cục, định nghĩa một lần:

$
n=n(t,i),
$

trong đó $n(t,i)$ ánh xạ vị trí $i$ của cửa sổ $t$ tới chỉ số toàn cục tương ứng. Sau định nghĩa này, không dùng lẫn $\widetilde s_{t,i}$ và $\widetilde s_n$.

Nhãn cấp điểm và hai điểm cấp cửa sổ được viết nhất quán theo ánh xạ này:

$$
\widehat a_n
=
\mathbb I\!\left(
\widetilde s_n>\delta_{\mathrm{pt}}
\right),
$$

$$
S_t^{(\mathrm{input})}
=
\frac{1}{T}
\sum_{i=1}^{T}
\widetilde s_{n(t,i)},
$$

$$
S_t^{(\mathrm{latent})}
=
\frac{1}{T}
\sum_{i=1}^{T}
\min_{1\leq k\leq K_c}
\left\lVert
\mathbf z_{t,i}^{(\mathrm{on})}
-
\mathbf p_k^{(c)}
\right\rVert_2^2.
$$

### 6.3. Hệ ngưỡng đề xuất thống nhất

Để các ngưỡng có cùng cấu trúc, bản nháp đề xuất:

| Ký hiệu đề xuất | English | Tiếng Việt |
|---|---|---|
| $\delta_{\mathrm{pt}}$ | point-level anomaly threshold | ngưỡng bất thường cấp điểm |
| $\delta_{\mathrm{rec}}$ | window reconstruction-error threshold | ngưỡng độ lệch tái tạo cấp cửa sổ |
| $\delta_{\mathrm{lat}}^{-}$ | lower latent-distance threshold | ngưỡng dưới của khoảng cách ẩn |
| $\delta_{\mathrm{lat}}^{+}$ | upper latent-distance threshold | ngưỡng trên của khoảng cách ẩn |

Hai đại lượng dùng để phân luồng là:

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $S_t^{(\mathrm{input})}$ | window reconstruction error | độ lệch tái tạo cấp cửa sổ |
| $S_t^{(\mathrm{latent})}$ | window latent distance | khoảng cách ẩn cấp cửa sổ |

Quy tắc phân luồng được đọc theo hai câu hỏi theo thứ tự:

1. Độ lệch tái tạo có vượt $\delta_{\mathrm{rec}}$ hay không?
2. Nếu có, khoảng cách ẩn nằm dưới, giữa hay trên hai ngưỡng $\delta_{\mathrm{lat}}^{-}$ và $\delta_{\mathrm{lat}}^{+}$?

| Điều kiện | Luồng |
|---|---|
| $S_t^{(\mathrm{input})}\leq\delta_{\mathrm{rec}}$ | `normal` |
| $S_t^{(\mathrm{input})}>\delta_{\mathrm{rec}}$ và $S_t^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{-}$ | `hard-old-normality` |
| $S_t^{(\mathrm{input})}>\delta_{\mathrm{rec}}$ và $\delta_{\mathrm{lat}}^{-}<S_t^{(\mathrm{latent})}\leq\delta_{\mathrm{lat}}^{+}$ | `gray zone` |
| $S_t^{(\mathrm{input})}>\delta_{\mathrm{rec}}$ và $S_t^{(\mathrm{latent})}>\delta_{\mathrm{lat}}^{+}$ | `strong anomaly` |

## 7. Hai trường hợp thích ứng

### 7.1. Hard-old-normality

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $S_t^{(\mathrm{hard})}(\boldsymbol\phi)$ | differentiable hard-window reconstruction error | độ lệch tái tạo khả vi của cửa sổ khó |
| $[x]_+=\max(0,x)$ | positive-part operator | toán tử lấy phần dương |
| $\mathcal L_{\mathrm{hard\text{-}rec}}$ | hard-window reconstruction penalty | hàm phạt tái tạo cửa sổ khó |
| $\mathcal K_{\mathrm{anom}}$ | anomalous-prototype index set | tập chỉ số nguyên mẫu bất thường |
| $\mathcal L_{\mathrm{src\text{-}on}}$ | source--online contrastive loss | hàm mất mát tương phản nguồn--trực tuyến |
| $\mathcal L_{\mathrm{hard}}$ | hard-old-normality adaptation loss | hàm mất mát thích ứng bình thường cũ khó |

Luồng đọc:

$$
S_t^{(\mathrm{hard})}(\boldsymbol\phi)
\longrightarrow
\mathcal L_{\mathrm{hard\text{-}rec}}
\quad\text{và}\quad
\mathcal L_{\mathrm{src\text{-}on}}
\longrightarrow
\mathcal L_{\mathrm{hard}}.
$$

### 7.2. Pseudo-new-normality đã được xác minh

Quá trình xác minh có bốn bước ký hiệu:

$$
\underbrace{\kappa^{(d)}_{w,i}}_{\text{nguyên mẫu rời rạc gần nhất}}
\longrightarrow
\underbrace{M^{(\mathrm{known\text{-}anom})}_{w,i}}_{\text{lọc bất thường đã biết}}
\longrightarrow
\underbrace{\boldsymbol\sigma_{w,i}}_{\text{chữ ký}}
\longrightarrow
\underbrace{M^{(\mathrm{pnn})}_{w,i}}_{\text{xác minh PNN}}.
$$

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $\kappa^{(d)}_{w,i}$ | nearest discrete-prototype index | chỉ số nguyên mẫu rời rạc gần nhất | Xác định cụm gần điểm đang xét nhất. |
| $c_k^{(d)}$ | discrete-prototype class label | nhãn lớp của nguyên mẫu rời rạc | Lớp $0$ là bình thường. |
| $r_k^{(d)}$ | discrete-cluster covering radius | bán kính bao phủ cụm rời rạc | Kiểm tra điểm có nằm trong cụm hay không. |
| $M^{(\mathrm{known\text{-}anom})}_{w,i}$ | known-anomaly mask | mặt nạ bất thường đã biết | Loại điểm khớp cụm bất thường đã biết. |
| $\boldsymbol\sigma_{w,i}$ | continuous-prototype signature | chữ ký nguyên mẫu liên tục | Ba nguyên mẫu liên tục gần nhất theo thứ tự. |
| $R_{\mathcal B}(\boldsymbol\sigma)$ | cross-window recurrence count | số cửa sổ có chữ ký lặp lại | Đếm trên các cửa sổ khác nhau. |
| $M^{(\mathrm{pnn})}_{w,i}$ | PNN mask | mặt nạ giả định bình thường mới | Đánh dấu điểm vượt qua phép xác minh. |
| $\mathcal V$ | verified point set | tập điểm đã được xác minh | Chứa các cặp $(w,i)$ có mặt nạ PNN bằng $1$. |
| $\mathcal L_{\mathrm{pnn}}$ | PNN adaptation loss | hàm mất mát thích ứng PNN | Chỉ tính trên các điểm thuộc $\mathcal V$. |

## 8. Ký hiệu nên định nghĩa cục bộ thay vì bắt người đọc nhớ trước

Các ký hiệu sau chỉ cần được giải thích tại mục chứa công thức tương ứng:

- nhiễu Gumbel $u^{(b,m)}_{t,i,k}$ và $g^{(b,m)}_{t,i,k}$;
- các tập chỉ số $\mathcal I^{(b,m)}_{t,i}$;
- tập điểm neo, điểm dương và điểm đối $\mathcal Q$, $\mathcal P(\mathbf z_q)$ và $\mathcal N(\mathbf z_q)$;
- tham số biến đổi nhân tạo $\boldsymbol\xi_t$;
- các bộ đệm triển khai như $\mathcal B_{\mathrm{ttl}}$;
- các biến chỉ xuất hiện trong giả mã như `decision`.

Những ký hiệu này không nên nằm trong bảng ký hiệu cốt lõi ở đầu chương vì người đọc chưa có ngữ cảnh để sử dụng chúng.

## 9. Ánh xạ từ ký hiệu gốc sang ký hiệu đề xuất

Bảng này làm rõ mọi thay đổi trong bản nháp. Chưa thay ký hiệu trong nguồn LaTeX trước khi tác giả duyệt.

| Ký hiệu gốc | Ký hiệu đề xuất | Mục đích |
|---|---|---|
| $r$ khi biểu thị lớp | $\ell$ | Dành riêng $r$ cho số lần xuất hiện trong EWMA. |
| $v$ khi biểu thị cửa sổ trong bộ đệm | $w$ | Dành riêng $v$ cho phiên bản sạch/bất thường. |
| $\mathcal P=\{\mathrm A,\mathrm B\}$ | $\mathcal G=\{\mathrm A,\mathrm B\}$ | Tránh trùng với tập điểm dương $\mathcal P(\mathbf z_q)$. |
| $\mathbf M_t$ của phép tiêm | $\mathbf M_t^{(\mathrm{inj})}$ | Nêu rõ đây là mặt nạ tiêm bất thường. |
| $M_t$ trong giả mã trực tuyến | $\mathcal V_t$ | Biểu thị đúng đây là tập điểm được chọn để thích ứng. |
| $u^{(m)}_{t,i,k}$, $g^{(m)}_{t,i,k}$ của nhánh liên tục | $u^{(c,m)}_{t,i,k}$, $g^{(c,m)}_{t,i,k}$ | Làm nhãn nhánh của nhiễu nhất quán với nhánh rời rạc. |
| $T_{\mathrm{point}}$ | $\delta_{\mathrm{pt}}$ | Thống nhất họ ký hiệu ngưỡng. |
| $B_{\mathrm{win}}$ | $\delta_{\mathrm{rec}}$ | Nêu rõ đây là ngưỡng tái tạo. |
| $A_{\mathrm{win}}^{(\mathrm{low})}$ | $\delta_{\mathrm{lat}}^{-}$ | Nêu rõ đây là ngưỡng dưới trong không gian ẩn. |
| $A_{\mathrm{win}}^{(\mathrm{high})}$ | $\delta_{\mathrm{lat}}^{+}$ | Nêu rõ đây là ngưỡng trên trong không gian ẩn. |
| $\delta_{\mathrm{normal}},\delta_{\mathrm{anomaly}}$ trong giả mã | Gọi $\operatorname{Triage}(\mathbf X_t)$ | Loại hệ ngưỡng thứ hai chưa được ánh xạ với công thức chi tiết. |

## 10. Những phần không nên rút gọn thêm

Ba cụm sau chứa ngữ nghĩa chính của phương pháp và cần được trình bày tường minh:

1. Cách tính trọng số truy vấn ngẫu nhiên của nhánh liên tục và nhánh rời rạc.
2. Hai đại lượng $S_t^{(\mathrm{input})}$ và $S_t^{(\mathrm{latent})}$ cùng quy tắc phân luồng.
3. Chuỗi lọc bất thường đã biết, tạo chữ ký và xác minh pseudo-new-normality.

Việc thay các cụm này bằng một tên hàm duy nhất sẽ làm phương trình ngắn hơn nhưng khiến người đọc không còn thấy điều kiện và thông tin mà phương pháp thực sự sử dụng.

## 11. Cấu trúc đề xuất khi đưa vào luận văn

1. Đặt Mục 1 của tài liệu này trước phương trình đầu tiên của Chương 3.
2. Đặt bảng “Các đối tượng cốt lõi” ngay sau phần mô tả kiến trúc tổng quát.
3. Định nghĩa ký hiệu chuyên biệt ngay trước phương trình đầu tiên sử dụng ký hiệu đó.
4. Đưa danh mục đầy đủ vào cuối chương hoặc phụ lục để tra cứu.
5. Không lặp lại định nghĩa dài nếu ký hiệu đã tuân theo quy tắc đọc chung; chỉ nhắc lại vai trò mới trong ngữ cảnh hiện tại.

---

### Các điểm cần tác giả duyệt

- Có chấp nhận dùng $b\in\{c,d\}$ cho hai nhánh và $h\in\{\mathrm{rec},\mathrm{cls}\}$ cho hai tác vụ hay không?
- Có đổi chỉ số lớp từ $r$ sang $\ell$ và chỉ số cửa sổ bộ đệm từ $v$ sang $w$ hay không?
- Có thống nhất bốn ngưỡng về họ ký hiệu $\delta$ hay giữ tên hiện tại?
- Có thay $M_t$ trong giả mã trực tuyến bằng tập điểm thích ứng $\mathcal V_t$ hay không?
- Khối “Quy ước ký hiệu” nên nằm đầu Chương 3 hay ngay trước mục kiến trúc mạng?
