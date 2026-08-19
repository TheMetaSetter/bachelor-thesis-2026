Dưới đây là bảng ký hiệu toán học được trích xuất trực tiếp từ toàn bộ nội dung đính kèm. Em giữ nguyên hệ thống ký hiệu của bản thảo, không thay bằng ký hiệu từ mã nguồn hoặc đặc tả khác.

## 1. Chỉ số và kích thước

| Ký hiệu | Ý nghĩa |
|---|---|
| $t$ | Chỉ số cửa sổ hoặc bước thời gian trực tuyến |
| $i\in\{1,\ldots,T\}$ | Vị trí điểm thời gian bên trong một cửa sổ |
| $n$ | Chỉ số tuyệt đối của điểm trên toàn bộ dòng dữ liệu |
| $u,v$ | Chỉ số cửa sổ nằm trong bộ đệm xác minh |
| $m\in\{1,\ldots,M\}$ | Chỉ số lần truy vấn hoặc lan truyền xuôi ngẫu nhiên |
| $k$ | Chỉ số nguyên mẫu |
| $j$ | Chỉ số phụ dùng trong phép tổng hoặc vị trí trong cửa sổ |
| $r$ | Chỉ số lớp trong $\{0,\ldots,11\}$, hoặc số lần một điểm xuất hiện trong các cửa sổ |
| $B$ | Kích thước lô dữ liệu |
| $T$ | Số bước thời gian trong một cửa sổ |
| $C$ | Số biến/kênh của chuỗi thời gian |
| $H$ | Số chiều của không gian ẩn |
| $K_c$ | Số nguyên mẫu liên tục; trong cài đặt là $32$ |
| $K_d$ | Số nguyên mẫu rời rạc; trong cài đặt là $60$ |
| $K_{\mathrm{top}}$ | Số nguyên mẫu rời rạc gần nhất được chọn; trong cài đặt là $3$ |
| $M$ | Số lần lan truyền xuôi hoặc truy vấn ngẫu nhiên |

## 2. Dữ liệu đầu vào và dữ liệu bất thường nhân tạo

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Cửa sổ đầu vào | $\mathbf{X}_t\in\mathbb{R}^{T\times C}$ | Cửa sổ chuỗi thời gian tại bước $t$ |
| Điểm thời gian | $\mathbf{x}_{t,i}\in\mathbb{R}^{C}$ | Điểm thứ $i$ trong cửa sổ $\mathbf{X}_t$ |
| Cửa sổ sạch | $\mathbf{X}^{(0)}_t$ | Cửa sổ lấy từ chuỗi huấn luyện gốc |
| Cửa sổ bất thường nhân tạo | $\mathbf{X}^{(a_t)}_t$ | Phiên bản đã được tiêm bất thường của $\mathbf{X}^{(0)}_t$ |
| Lớp bất thường nhân tạo | $a_t\in\{1,\ldots,11\}$ | Loại phép biến đổi bất thường áp dụng cho cửa sổ |
| Mặt nạ tiêm bất thường | $\mathbf{M}_t$, $\mathbf{M}_{t,i}$ | Xác định vị trí được tiêm bất thường |
| Hàm tạo bất thường | $\mathcal{A}_{a_t}$ | Phép biến đổi tương ứng với lớp $a_t$ |
| Tham số ngẫu nhiên | $\boldsymbol{\xi}_t$ | Tham số ngẫu nhiên của phép biến đổi |
| Vec-tơ toàn số một | $\mathbf{1}$ | Dùng để giữ lại các vị trí không được tiêm |
| Nhân theo phần tử | $\odot$ | Phép nhân Hadamard |
| Nhãn one-hot | $\mathbf{y}^{(v)}_t$, $y^{(v)}_{t,r}$ | Nhãn thật của phiên bản $v$ |
| Xác suất dự đoán | $\widehat{\mathbf{y}}^{(v)}_t$, $\widehat y^{(v)}_{t,r}$ | Phân phối xác suất dự đoán trên 12 lớp |

Công thức tạo cửa sổ bất thường:

$$
\mathbf{X}^{(a_t)}_t
=
(\mathbf 1-\mathbf M_t)\odot\mathbf X^{(0)}_t
+
\mathbf M_t\odot
\mathcal A_{a_t}(\mathbf X^{(0)}_t;\boldsymbol\xi_t).
$$

## 3. Bộ mã hóa chia sẻ

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Bộ mã hóa | $f_{\mathrm{enc}}$ | Mạng CNN một chiều dùng chung cho hai tác vụ |
| Tham số bộ mã hóa | $\boldsymbol{\theta}_{\mathrm{enc}}$ | Các tham số học được của bộ mã hóa |
| Ten-xơ trạng thái ẩn | $\mathbf Z_t\in\mathbb R^{T\times H}$ | Biểu diễn ẩn của một cửa sổ |
| Ten-xơ ẩn theo lô | $\mathbf Z\in\mathbb R^{B\times T\times H}$ | Biểu diễn ẩn khi đầu vào có $B$ cửa sổ |
| Vec-tơ trạng thái ẩn | $\mathbf z_{t,i}\in\mathbb R^H$ | Biểu diễn ẩn của vị trí $i$ |
| Biểu diễn cửa sổ sạch | $\mathbf Z^{(0)}_t$ | Đầu ra bộ mã hóa của $\mathbf X^{(0)}_t$ |
| Biểu diễn cửa sổ nhân tạo | $\mathbf Z^{(a_t)}_t$ | Đầu ra bộ mã hóa của $\mathbf X^{(a_t)}_t$ |

Ánh xạ chính:

$$
\mathbf Z_t=f_{\mathrm{enc}}
(\mathbf X_t;\boldsymbol\theta_{\mathrm{enc}}).
$$

## 4. Tập nguyên mẫu liên tục

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Tập nguyên mẫu liên tục | $\mathbf P^{(c)}\in\mathbb R^{K_c\times H}$ | Bộ nhớ chứa các nguyên mẫu bình thường |
| Nguyên mẫu liên tục thứ $k$ | $\mathbf p^{(c)}_k\in\mathbb R^H$ | Tâm cụm mẫu hình bình thường thứ $k$ |
| Số nguyên mẫu | $K_c$ | Bằng $32$ trong cài đặt |
| Nhiệt độ truy vấn | $\tau_c>0$ | Điều khiển độ sắc của phân phối truy vấn |
| Trọng số truy vấn tất định | $\alpha^{(c)}_{t,i,k}$ | Mức đóng góp của nguyên mẫu $k$ |
| Vec-tơ truy xuất tất định | $\widetilde{\mathbf z}^{(c)}_{t,i}$ | Vec-tơ ẩn của nhánh liên tục |
| Nhiễu Gumbel | $g^{(m)}_{t,i,k}$ | Nhiễu tại lần truy vấn $m$ |
| Biến Uniform | $u^{(m)}_{t,i,k}$ | Biến ngẫu nhiên dùng sinh nhiễu Gumbel |
| Trọng số truy vấn ngẫu nhiên | $\alpha^{(c,m)}_{t,i,k}$ | Trọng số Gumbel-Softmax |
| Vec-tơ truy xuất ngẫu nhiên | $\widetilde{\mathbf z}^{(c,m)}_{t,i}$ | Đầu ra nhánh liên tục ở lần $m$ |
| Ten-xơ nhánh liên tục | $\widetilde{\mathbf Z}^{(c,m)}_t\in\mathbb R^{T\times H}$ | Tập hợp các vec-tơ nhánh liên tục |

Chuỗi xử lý:

$$
\mathbf z_{t,i}
\longrightarrow
\alpha^{(c,m)}_{t,i,k}
\longrightarrow
\widetilde{\mathbf z}^{(c,m)}_{t,i}
\longrightarrow
\widetilde{\mathbf Z}^{(c,m)}_t.
$$

## 5. Tập mã vec-tơ rời rạc

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Tập nguyên mẫu rời rạc | $\mathbf P^{(d)}\in\mathbb R^{K_d\times H}$ | Codebook chứa nguyên mẫu của 12 lớp nhân tạo |
| Nguyên mẫu rời rạc thứ $k$ | $\mathbf p^{(d)}_k\in\mathbb R^H$ | Tâm cụm thứ $k$ |
| Nhãn nguyên mẫu | $c^{(d)}_k\in\{0,\ldots,11\}$ | Lớp gắn với nguyên mẫu $k$ |
| Bán kính cụm | $r^{(d)}_k$ | Bán kính bao phủ của cụm thứ $k$ |
| Khoảng cách đến nguyên mẫu | $d^{(d)}_{t,i,k}$ | Bình phương khoảng cách Euclid |
| Tập chỉ số Top-$K$ | $\mathcal I^{(d)}_{t,i}$ | Các nguyên mẫu gần nhất trong truy vấn tất định |
| Nhiệt độ truy vấn | $\tau_d>0$ | Nhiệt độ của truy vấn rời rạc |
| Trọng số truy vấn tất định | $\alpha^{(d)}_{t,i,k}$ | Trọng số của nguyên mẫu được chọn |
| Vec-tơ truy xuất tất định | $\widetilde{\mathbf z}^{(d)}_{t,i}$ | Vec-tơ ẩn nhánh rời rạc |
| Nhiễu Gumbel | $g^{(d,m)}_{t,i,k}$ | Nhiễu truy vấn rời rạc |
| Biến Uniform | $u^{(d,m)}_{t,i,k}$ | Biến dùng sinh nhiễu Gumbel |
| Điểm truy vấn ngẫu nhiên | $q^{(d,m)}_{t,i,k}$ | Điểm sau khi kết hợp khoảng cách và nhiễu |
| Tập Top-$K$ ngẫu nhiên | $\mathcal I^{(d,m)}_{t,i}$ | Các nguyên mẫu được chọn ở lần $m$ |
| Trọng số truy vấn ngẫu nhiên | $\alpha^{(d,m)}_{t,i,k}$ | Trọng số softmax trên các nguyên mẫu được chọn |
| Vec-tơ truy xuất ngẫu nhiên | $\widetilde{\mathbf z}^{(d,m)}_{t,i}$ | Đầu ra nhánh rời rạc ở lần $m$ |
| Ten-xơ nhánh rời rạc | $\widetilde{\mathbf Z}^{(d,m)}_t\in\mathbb R^{T\times H}$ | Tập hợp các vec-tơ nhánh rời rạc |

Chuỗi xử lý:

$$
\mathbf z_{t,i}
\rightarrow
d^{(d)}_{t,i,k}
\rightarrow
q^{(d,m)}_{t,i,k}
\rightarrow
\mathcal I^{(d,m)}_{t,i}
\rightarrow
\alpha^{(d,m)}_{t,i,k}
\rightarrow
\widetilde{\mathbf z}^{(d,m)}_{t,i}.
$$

## 6. Đầu tái tạo và đầu phân loại

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Mạng tái tạo | $f_{\mathrm{rec}}$ | Ánh xạ biểu diễn ẩn về không gian dữ liệu |
| Tham số mạng tái tạo | $\boldsymbol\theta_{\mathrm{rec}}$ | Tham số của đầu tái tạo |
| Cửa sổ tái tạo sạch | $\widehat{\mathbf X}^{(0)}_t$ | Tái tạo của cửa sổ sạch ở giai đoạn A |
| Cửa sổ tái tạo ngẫu nhiên | $\widehat{\mathbf X}^{(m)}_t$ | Tái tạo ở lần lan truyền thứ $m$ |
| Điểm được tái tạo | $\widehat{\mathbf x}^{(m)}_{t,i}$ | Giá trị tái tạo của điểm thứ $i$ |
| Mạng phân loại | $f_{\mathrm{cls}}$ | Đầu phân loại 12 lớp |
| Tham số mạng phân loại | $\boldsymbol\theta_{\mathrm{cls}}$ | Tham số của đầu phân loại |
| Phép gộp | $\operatorname{Pool}$ | Gộp biểu diễn theo chiều thời gian |
| Xác suất phân loại | $\widehat{\mathbf y}^{(v)}_t$, $\widehat{\mathbf y}^{(m)}_t$ | Phân phối softmax trên 12 lớp |

## 7. Mạng tổng hợp ở giai đoạn B

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Ten-xơ ghép | $\mathbf Z^{(\mathrm{cat},m)}_t\in\mathbb R^{T\times3H}$ | Ghép biểu diễn gốc, nhánh liên tục và nhánh rời rạc |
| Mạng tổng hợp tái tạo | $f_{\mathrm{fus}}^{(\mathrm{rec})}$ | Fusion network cho tác vụ tái tạo |
| Tham số fusion tái tạo | $\boldsymbol\theta_{\mathrm{fus}}^{(\mathrm{rec})}$ | Tham số tương ứng |
| Ten-xơ fusion tái tạo | $\mathbf H^{(\mathrm{rec},m)}_t$ | Biểu diễn đưa vào đầu tái tạo |
| Mạng tổng hợp phân loại | $f_{\mathrm{fus}}^{(\mathrm{cls})}$ | Fusion network cho tác vụ phân loại |
| Tham số fusion phân loại | $\boldsymbol\theta_{\mathrm{fus}}^{(\mathrm{cls})}$ | Tham số tương ứng |
| Ten-xơ fusion phân loại | $\mathbf H^{(\mathrm{cls},m)}_t$ | Biểu diễn đưa vào đầu phân loại |

Quan hệ ghép:

$$
\mathbf Z^{(\mathrm{cat},m)}_t
=
\operatorname{Concat}
\left[
\mathbf Z_t,
\widetilde{\mathbf Z}^{(c,m)}_t,
\widetilde{\mathbf Z}^{(d,m)}_t
\right].
$$

## 8. Hàm mất mát giai đoạn A

| Thành phần | Ký hiệu |
|---|---|
| Tập điểm neo | $\mathcal Q$ |
| Điểm neo | $\mathbf z_q$ |
| Tập điểm dương của điểm neo | $\mathcal P(\mathbf z_q)$ |
| Điểm dương | $\mathbf z_p$ |
| Tập điểm đối của điểm neo | $\mathcal N(\mathbf z_q)$ |
| Điểm đối | $\mathbf z_n$ |
| Nhiệt độ tương phản | $\tau_{\mathrm{con}}>0$ |
| Mất mát tái tạo | $\mathcal L_{\mathrm{rec}}$ |
| Mất mát phân loại | $\mathcal L_{\mathrm{cls}}$ |
| Mất mát tương phản cấp điểm | $\mathcal L_{\mathrm{con}}$ |
| Mất mát đa tác vụ tổng | $\mathcal L_{\mathrm{multi}}$ |
| Trọng số tái tạo | $\lambda_{\mathrm{rec}}$ |
| Trọng số phân loại | $\lambda_{\mathrm{cls}}$ |
| Trọng số tương phản | $\lambda_{\mathrm{con}}$ |

$$
\mathcal L_{\mathrm{multi}}
=
\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{cls}}\mathcal L_{\mathrm{cls}}
+
\lambda_{\mathrm{con}}\mathcal L_{\mathrm{con}}.
$$

## 9. Hàm mất mát giai đoạn B

| Thành phần | Ký hiệu |
|---|---|
| Mất mát tái tạo giai đoạn B | $\mathcal L^{(B)}_{\mathrm{rec}}$ |
| Mất mát phân loại giai đoạn B | $\mathcal L^{(B)}_{\mathrm{cls}}$ |
| Mất mát tổng giai đoạn B | $\mathcal L_B$ |

$$
\mathcal L_B
=
\lambda_{\mathrm{rec}}\mathcal L^{(B)}_{\mathrm{rec}}
+
\lambda_{\mathrm{cls}}\mathcal L^{(B)}_{\mathrm{cls}}.
$$

## 10. Quy trình ngoại tuyến

| Thành phần | Ký hiệu |
|---|---|
| Tập huấn luyện | $\mathcal D_{\mathrm{train}}$ |
| Tập xác thực | $\mathcal D_{\mathrm{val}}$ |
| Bộ cấu hình | $\Omega$ |
| Cấu hình giai đoạn $p$ | $\Omega_p$ |
| Tập các giai đoạn | $\mathcal P=\{\mathrm A,\mathrm B\}$ |
| Chỉ số giai đoạn | $p$ |
| Checkpoint tốt nhất | $\Theta^*$ |
| Hàm mất mát tổng quát trong thuật toán | $\mathcal L$ |

## 11. Nhánh nguồn và bộ ánh xạ trực tuyến

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Ten-xơ ẩn nguồn | $\mathbf Z^{(\mathrm{src})}_t$ | Đầu ra của bộ mã hóa ngoại tuyến đã đóng băng |
| Vec-tơ ẩn nguồn | $\mathbf z^{(\mathrm{src})}_{t,i}$ | Biểu diễn nguồn tại điểm $i$ |
| Bộ ánh xạ trực tuyến | $g_{\mathrm{proj}}$ | MLP projector được thích ứng trực tuyến |
| Tham số thích ứng | $\boldsymbol\phi$ | Nhóm tham số duy nhất được cập nhật trực tuyến |
| Ten-xơ ẩn trực tuyến | $\mathbf Z^{(\mathrm{on})}_t$ | Đầu ra của MLP projector |
| Vec-tơ ẩn trực tuyến | $\mathbf z^{(\mathrm{on})}_{t,i}$ | Biểu diễn trực tuyến tại điểm $i$ |
| Mô hình trực tuyến | $\Theta_{\mathrm{online}}$ | Mô hình được khởi tạo từ $\Theta^*$ |
| Dòng dữ liệu | $\mathcal S$ | Chuỗi thời gian trực tuyến |
| Bộ đệm xác minh | $\mathcal B_{\mathrm{ver}}$, $\mathcal B$ | Lưu cửa sổ vùng xám |
| Bộ đệm Time-to-Live | $\mathcal B_{\mathrm{ttl}}$ | Bộ đệm quản lý thời gian tồn tại |

## 12. Điểm bất thường và EWMA

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Độ lệch tái tạo tại lần $m$ | $s^{(m)}_{t,i}$ | MSE cấp điểm |
| Độ lệch trung bình Monte Carlo | $\overline s_{t,i}$ | Trung bình của $M$ lần lan truyền |
| Độ lệch tại lần xuất hiện $r$ | $\overline s^{(r)}_n$ | Điểm trung bình của điểm tuyệt đối $n$ |
| Điểm EWMA | $\widetilde s^{(r)}_n$ | Điểm đã được làm trơn |
| Hệ số EWMA | $\rho$ | Trong bản thảo được đặt bằng $0.9$ |
| Nhãn bất thường dự đoán | $\widehat a_n\in\{0,1\}$ | Kết quả phân loại cấp điểm |
| Ngưỡng cấp điểm | $T_{\mathrm{point}}$ | Ngưỡng hiệu chỉnh từ chuỗi xác thực sạch |
| Hàm chỉ báo | $\mathbb I[\cdot]$ | Trả về $1$ nếu điều kiện đúng |

Chuỗi tính điểm:

$$
s^{(m)}_{t,i}
\rightarrow
\overline s_{t,i}
\rightarrow
\widetilde s^{(r)}_n
\rightarrow
\widehat a_n.
$$

## 13. Phân luồng cửa sổ trực tuyến

| Thành phần | Ký hiệu |
|---|---|
| Điểm tái tạo cấp cửa sổ | $S^{(\mathrm{input})}_t$ |
| Khoảng cách ẩn cấp cửa sổ | $S^{(\mathrm{latent})}_t$ |
| Ngưỡng tái tạo cửa sổ | $B_{\mathrm{win}}$ |
| Ngưỡng khoảng cách ẩn dưới | $A_{\mathrm{win}}^{(\mathrm{low})}$ |
| Ngưỡng khoảng cách ẩn trên | $A_{\mathrm{win}}^{(\mathrm{high})}$ |
| Hàm phân luồng | $\operatorname{Triage}(\mathbf X_t)$ |
| Ngưỡng normal trong giả mã | $\delta_{\mathrm{normal}}$ |
| Ngưỡng anomaly trong giả mã | $\delta_{\mathrm{anomaly}}$ |
| Điểm cửa sổ trong giả mã | $s_t$ |
| Quyết định phân luồng | $\mathrm{decision}$ |
| Mặt nạ điểm được phép thích ứng | $M_t$ |

Các luồng đầu ra:

$$
\{\textit{normal},\;
\textit{hard-old-normality},\;
\textit{gray zone},\;
\textit{strong anomaly}\}.
$$

## 14. Thích ứng với hard-old-normality

| Thành phần | Ký hiệu |
|---|---|
| Điểm tái tạo khả vi | $S^{(\mathrm{hard})}_t(\boldsymbol\phi)$ |
| Toán tử phần dương | $[x]_+=\max(0,x)$ |
| Phạt tái tạo kiểu hinge | $\mathcal L_{\mathrm{hard\text{-}rec}}$ |
| Tập chỉ số nguyên mẫu bất thường | $\mathcal K_{\mathrm{anom}}$ |
| Nhiệt độ tương phản trực tuyến | $\tau_{\mathrm{on}}$ |
| Mất mát tương phản nguồn–trực tuyến | $\mathcal L_{\mathrm{src\text{-}on}}$ |
| Mất mát thích ứng hard-old-normality | $\mathcal L_{\mathrm{hard}}$ |

$$
\mathcal K_{\mathrm{anom}}
=
\{k\in\{1,\ldots,K_d\}:c^{(d)}_k\neq0\},
$$

$$
\mathcal L_{\mathrm{hard}}
=
\mathcal L_{\mathrm{hard\text{-}rec}}
+
\lambda_{\mathrm{con}}
\mathcal L_{\mathrm{src\text{-}on}}.
$$

## 15. Xác minh pseudo-new-normality

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Nguyên mẫu rời rạc gần nhất | $\kappa^{(d)}_{u,i}$ | Chỉ số prototype rời rạc gần điểm nhất |
| Mặt nạ bất thường đã biết | $M^{(\mathrm{known\text{-}anom})}_{u,i}$ | Bằng $1$ nếu điểm thuộc một cụm bất thường đã biết |
| Chữ ký nguyên mẫu liên tục | $\boldsymbol\sigma_{u,i}$ | Ba prototype liên tục gần nhất, có thứ tự |
| Số cửa sổ chứa chữ ký | $R_{\mathcal B}(\boldsymbol\sigma)$ | Số cửa sổ khác nhau chứa chữ ký |
| Mặt nạ pseudo-new-normality | $M^{(\mathrm{pnn})}_{u,i}$ | Bằng $1$ nếu điểm vượt qua bước xác minh |
| Tập điểm đã xác minh | $\mathcal V$ | Các cặp $(u,i)$ có $M^{(\mathrm{pnn})}_{u,i}=1$ |
| Mất mát tái tạo PNN | $\mathcal L_{\mathrm{pnn\text{-}rec}}$ | MSE chỉ trên các điểm thuộc $\mathcal V$ |
| Mất mát tương phản PNN | $\mathcal L^{(\mathrm{pnn})}_{\mathrm{src\text{-}on}}$ | Ràng buộc biểu diễn PNN với nhánh nguồn |
| Mất mát thích ứng PNN | $\mathcal L_{\mathrm{pnn}}$ | Mất mát tổng cho PNN |

Chuỗi xác minh:

$$
\mathbf z^{(\mathrm{on})}_{u,i}
\rightarrow
\kappa^{(d)}_{u,i}
\rightarrow
M^{(\mathrm{known\text{-}anom})}_{u,i}
\rightarrow
\boldsymbol\sigma_{u,i}
\rightarrow
R_{\mathcal B}(\boldsymbol\sigma_{u,i})
\rightarrow
M^{(\mathrm{pnn})}_{u,i}
\rightarrow
\mathcal V.
$$

## 16. Những chỗ ký hiệu chưa nhất quán trong bản thảo

Có sáu điểm nên chỉnh trước khi đưa bảng ký hiệu vào luận văn:

1. $\widetilde{\mathbf Z}^{(c,m)}_t$ và $\widetilde{\mathbf Z}^{(d,m)}_t$ được định nghĩa lặp lại hai lần liên tiếp.

2. Không gian thực được viết là $\mathbf{R}^{T\times H}$ tại hai phương trình, trong khi các phần khác dùng $\mathbb{R}$. Nên thống nhất thành $\mathbb R$.

3. $\mathbf M_t$ là mặt nạ tiêm bất thường ở giai đoạn A, nhưng $M_t$ trong giả mã trực tuyến lại là mặt nạ pseudo-new-normality. Hai khái niệm khác nhau nhưng ký hiệu gần như giống hệt nhau.

4. $M$ là số lần lan truyền ngẫu nhiên, trong khi $\mathbf M_t$ là mặt nạ. Vẫn phân biệt được bằng kiểu chữ, nhưng dễ đọc nhầm.

5. $r$ vừa là chỉ số lớp trong $\mathcal L_{\mathrm{cls}}$, vừa là số lần một điểm xuất hiện trong công thức EWMA.

6. Giả mã sử dụng $s_t,\delta_{\mathrm{normal}},\delta_{\mathrm{anomaly}}$, nhưng phần định nghĩa chi tiết lại sử dụng hai điểm $S^{(\mathrm{input})}_t,S^{(\mathrm{latent})}_t$ và ba ngưỡng $B_{\mathrm{win}},A_{\mathrm{win}}^{(\mathrm{low})},A_{\mathrm{win}}^{(\mathrm{high})}$. Hai hệ ký hiệu này chưa được nối với nhau bằng một định nghĩa rõ ràng.

Ngoài ra, đoạn “một cửa sổ được xác định là *bất thường khó*” không khớp với tên luồng $\textit{hard-old-normality}$. Theo phần giải thích ngay sau đó, cách gọi đúng phải là “mẫu hình bình thường khó”.
