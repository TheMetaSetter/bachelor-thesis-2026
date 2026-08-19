# Ký hiệu Chương 3 theo phạm vi một cửa sổ

> **Quy ước phạm vi.** Trong toàn bộ phiên bản này, ta cố định một cửa sổ bất kỳ trong một lô dữ liệu (batch) và chỉ mô tả các phép toán trên cửa sổ đó. Vì vậy, chỉ mục cửa sổ $t$ được lược bỏ để ký hiệu gọn hơn. Chẳng hạn, $\mathbf X$, $\mathbf z_i$ và $\alpha^{(c,m)}_{i,k}$ lần lượt được hiểu là $\mathbf X_t$, $\mathbf z_{t,i}$ và $\alpha^{(c,m)}_{t,i,k}$ nếu viết đầy đủ chỉ mục cửa sổ. Việc lược bỏ này không có nghĩa batch chỉ chứa một cửa sổ; nó chỉ có nghĩa ta đang xét một phần tử cố định nhưng bất kỳ của batch. Các chỉ mục còn lại vẫn được giữ khi chúng phân biệt những đối tượng bên trong cửa sổ hoặc trên toàn dòng dữ liệu.

## 1. Quy tắc đọc chỉ mục

| Chỉ số | English | Tiếng Việt |
|---|---|---|
| $i,j$ | within-window time-point indices | chỉ số điểm thời gian trong cửa sổ đang xét |
| $k$ | prototype index | chỉ số nguyên mẫu |
| $m$ | stochastic-pass index | chỉ số lần lan truyền hoặc truy vấn ngẫu nhiên |
| $n$ | global time-point index | chỉ số điểm thời gian trên toàn dòng dữ liệu |
| $r$ | occurrence index | số lần một điểm toàn cục đã xuất hiện |
| $\ell$ | class index | chỉ số lớp |
| $u,w$ | buffered-window indices | chỉ số cửa sổ trong bộ đệm xác minh |

Ví dụ, $\alpha^{(c,m)}_{i,k}$ là trọng số của nguyên mẫu $k$, dành cho điểm $i$ trong cửa sổ đang xét, ở lần truy vấn ngẫu nhiên $m$, thuộc nhánh liên tục $c$.

## 2. Các đối tượng cốt lõi

| Ký hiệu | English | Tiếng Việt | Vai trò |
|---|---|---|---|
| $\mathbf X\in\mathbb R^{T\times C}$ | input window | cửa sổ đầu vào đang xét | Chứa $T$ điểm thời gian và $C$ biến. |
| $\mathbf x_i\in\mathbb R^C$ | input time-point vector | vec-tơ điểm thời gian đầu vào | Hàng thứ $i$ của $\mathbf X$. |
| $\mathbf Z\in\mathbb R^{T\times H}$ | latent tensor | ten-xơ trạng thái ẩn | Đầu ra của bộ mã hóa. |
| $\mathbf z_i\in\mathbb R^H$ | latent vector | vec-tơ trạng thái ẩn | Hàng thứ $i$ của $\mathbf Z$; có độ lớn đơn vị. |
| $\mathbf P^{(b)}\in\mathbb R^{K_b\times H}$ | prototype bank | tập nguyên mẫu | Bộ nhớ của nhánh $b$. |
| $\mathbf p_k^{(b)}\in\mathbb R^H$ | prototype | nguyên mẫu | Nguyên mẫu thứ $k$ của nhánh $b$. |
| $\alpha_{i,k}^{(b,m)}$ | query weight | trọng số truy vấn | Mức đóng góp của nguyên mẫu $k$. |
| $\widetilde{\mathbf z}_{i}^{(b,m)}$ | retrieved latent vector | vec-tơ ẩn được truy xuất | Kết quả truy vấn bộ nhớ nhánh $b$. |
| $\widehat{\mathbf X}^{(m)}$ | reconstructed window | cửa sổ được tái tạo | Đầu ra của tác vụ tái tạo. |
| $\widehat{\mathbf y}^{(m)}$ | predicted class probabilities | xác suất lớp dự đoán | Đầu ra của tác vụ phân loại. |
| $s_i^{(m)}$ | point-level reconstruction error | độ lệch tái tạo cấp điểm | Điểm thô ở lần ngẫu nhiên $m$. |
| $S^{(\mathrm{input})}$ | window reconstruction error | độ lệch tái tạo cấp cửa sổ | Điểm tái tạo dùng để phân luồng. |
| $S^{(\mathrm{latent})}$ | window latent distance | khoảng cách ẩn cấp cửa sổ | Mức xa lạ so với các nguyên mẫu bình thường. |

## 3. Chuỗi biến đổi chính

$$
\mathbf X
\xrightarrow{f_{\mathrm{enc}}}
\mathbf Z
\xrightarrow{\mathbf P^{(c)},\,\mathbf P^{(d)}}
\left(
\widetilde{\mathbf Z}^{(c,m)},
\widetilde{\mathbf Z}^{(d,m)}
\right)
\xrightarrow{f_{\mathrm{fus}}^{(h)}}
\mathbf H^{(h,m)}
\xrightarrow{f_{\mathrm{rec}},\,f_{\mathrm{cls}}}
\left(
\widehat{\mathbf X}^{(m)},
\widehat{\mathbf y}^{(m)}
\right).
$$

Trong đó

$$
\mathbf Z
=
f_{\mathrm{enc}}
\left(
\mathbf X;
\boldsymbol\theta_{\mathrm{enc}}
\right).
$$

## 4. Truy vấn tập nguyên mẫu liên tục

Với điểm $i$ của cửa sổ đang xét, trọng số truy vấn tất định là

$$
\alpha^{(c)}_{i,k}
=
\frac{
\exp\!\left(\mathbf z_i^\top\mathbf p_k^{(c)}/\tau_c\right)
}{
\sum_{j=1}^{K_c}
\exp\!\left(\mathbf z_i^\top\mathbf p_j^{(c)}/\tau_c\right)
},
$$

và vec-tơ truy xuất là

$$
\widetilde{\mathbf z}^{(c)}_i
=
\sum_{k=1}^{K_c}
\alpha^{(c)}_{i,k}\mathbf p_k^{(c)}.
$$

Ở lần truy vấn ngẫu nhiên $m$, nhiễu Gumbel và trọng số truy vấn được viết là

$$
g^{(c,m)}_{i,k}
=
-\log\!\left[-\log\!\left(u^{(c,m)}_{i,k}\right)\right],
\qquad
u^{(c,m)}_{i,k}
\overset{\mathrm{i.i.d.}}{\sim}
\operatorname{Uniform}(0,1),
$$

$$
\alpha^{(c,m)}_{i,k}
=
\frac{
\exp\!\left(
\left(\mathbf z_i^\top\mathbf p_k^{(c)}+g^{(c,m)}_{i,k}\right)/\tau_c
\right)
}{
\sum_{j=1}^{K_c}
\exp\!\left(
\left(\mathbf z_i^\top\mathbf p_j^{(c)}+g^{(c,m)}_{i,j}\right)/\tau_c
\right)
},
$$

$$
\widetilde{\mathbf z}^{(c,m)}_i
=
\sum_{k=1}^{K_c}
\alpha^{(c,m)}_{i,k}\mathbf p_k^{(c)}.
$$

## 5. Truy vấn tập nguyên mẫu rời rạc

Khoảng cách từ vec-tơ ẩn thứ $i$ đến nguyên mẫu rời rạc thứ $k$ là

$$
d^{(d)}_{i,k}
=
\left\|\mathbf z_i-\mathbf p_k^{(d)}\right\|_2^2
=
2\left[
1-\operatorname{cos}\!\left(\mathbf z_i,\mathbf p_k^{(d)}\right)
\right].
$$

Tập chỉ số $K_{\mathrm{top}}$ nguyên mẫu gần nhất và vec-tơ truy xuất tất định được viết là

$$
\mathcal I^{(d)}_i
=
\operatorname{TopKMin}_{k\in\{1,\ldots,K_d\}}
d^{(d)}_{i,k},
$$

$$
\widetilde{\mathbf z}^{(d)}_i
=
\sum_{k=1}^{K_d}
\alpha^{(d)}_{i,k}\mathbf p_k^{(d)},
\qquad
\alpha^{(d)}_{i,k}=0
\quad\text{khi }k\notin\mathcal I^{(d)}_i.
$$

Phiên bản ngẫu nhiên ở lần $m$ được ký hiệu tương ứng bằng

$$
q^{(d,m)}_{i,k},
\qquad
\mathcal I^{(d,m)}_i,
\qquad
\alpha^{(d,m)}_{i,k},
\qquad
\widetilde{\mathbf z}^{(d,m)}_i.
$$

## 6. Hai phiên bản của cửa sổ trong giai đoạn A

Vì đang cố định một cửa sổ, lớp bất thường nhân tạo được ký hiệu là $a$, không phải $a_t$:

$$
a\in\{1,\ldots,11\},
\qquad
\mathbf X^{(a)}
=
\left(\mathbf 1-\mathbf M^{(\mathrm{inj})}\right)\odot\mathbf X^{(0)}
+
\mathbf M^{(\mathrm{inj})}\odot
\mathcal A_a\!\left(\mathbf X^{(0)};\boldsymbol\xi\right).
$$

Chỉ số phiên bản là $v\in\{0,a\}$. Do đó, $\mathbf X^{(v)}$, $\mathbf Z^{(v)}$ và $\widehat{\mathbf y}^{(v)}$ lần lượt là cửa sổ, biểu diễn ẩn và dự đoán lớp của phiên bản $v$.

## 7. Mạng tổng hợp theo tác vụ

Với $h\in\{\mathrm{rec},\mathrm{cls}\}$,

$$
\mathbf Z^{(\mathrm{cat},m)}
=
\operatorname{Concat}
\left(
\mathbf Z,
\widetilde{\mathbf Z}^{(c,m)},
\widetilde{\mathbf Z}^{(d,m)}
\right),
$$

$$
\mathbf H^{(h,m)}
=
f_{\mathrm{fus}}^{(h)}
\left(
\mathbf Z^{(\mathrm{cat},m)};
\boldsymbol\theta_{\mathrm{fus}}^{(h)}
\right).
$$

## 8. Nhánh nguồn và nhánh trực tuyến

Trong phạm vi cửa sổ đang xét,

$$
\mathbf Z^{(\mathrm{src})}
=
f_{\mathrm{enc}}
\left(
\mathbf X;
\boldsymbol\theta_{\mathrm{enc}}
\right),
\qquad
\mathbf Z^{(\mathrm{on})}
=
g_{\mathrm{proj}}
\left(
\mathbf Z^{(\mathrm{src})};
\boldsymbol\phi
\right).
$$

## 9. Điểm bất thường trong cửa sổ đang xét

Độ lệch tái tạo tại điểm $i$, ở lần lan truyền ngẫu nhiên $m$, là

$$
s_i^{(m)}
=
\frac{1}{C}
\left\|
\mathbf x_i-\widehat{\mathbf x}^{(m)}_i
\right\|_2^2,
\qquad
\overline s_i
=
\frac{1}{M}
\sum_{m=1}^{M}s_i^{(m)}.
$$

Nếu điểm $i$ của cửa sổ đang xét tương ứng với điểm toàn cục $n(i)$, điểm EWMA vẫn phải mang chỉ số toàn cục $n$, vì cùng một điểm có thể xuất hiện trong nhiều cửa sổ trượt:

$$
\widetilde s_n^{(r)}
=
\rho\,\overline s_n^{(r)}
+
(1-\rho)\widetilde s_n^{(r-1)},
\qquad n=n(i).
$$

Hai điểm cấp cửa sổ không cần chỉ mục $t$:

$$
S^{(\mathrm{input})}
=
\frac{1}{T}
\sum_{i=1}^{T}
\widetilde s_{n(i)},
$$

$$
S^{(\mathrm{latent})}
=
\frac{1}{T}
\sum_{i=1}^{T}
\min_{1\le k\le K_c}
\left\|
\mathbf z_i^{(\mathrm{on})}-\mathbf p_k^{(c)}
\right\|_2^2.
$$

Các ngưỡng vẫn giữ theo hệ ký hiệu của bản nháp hướng người đọc:

| Ký hiệu | English | Tiếng Việt |
|---|---|---|
| $\delta_{\mathrm{pt}}$ | point-level anomaly threshold | ngưỡng bất thường cấp điểm |
| $\delta_{\mathrm{rec}}$ | window reconstruction-error threshold | ngưỡng độ lệch tái tạo cấp cửa sổ |
| $\delta_{\mathrm{lat}}^{-}$ | lower latent-distance threshold | ngưỡng dưới của khoảng cách ẩn |
| $\delta_{\mathrm{lat}}^{+}$ | upper latent-distance threshold | ngưỡng trên của khoảng cách ẩn |

Quy tắc phân luồng vì vậy được viết gọn là

$$
\operatorname{Triage}(\mathbf X)
=
\begin{cases}
\textit{normal},
& S^{(\mathrm{input})}\le \delta_{\mathrm{rec}},
\\
\textit{hard-old-normality},
& S^{(\mathrm{input})}>\delta_{\mathrm{rec}}
\land
S^{(\mathrm{latent})}\le \delta_{\mathrm{lat}}^{-},
\\
\textit{gray zone},
& S^{(\mathrm{input})}>\delta_{\mathrm{rec}}
\land
\delta_{\mathrm{lat}}^{-}<S^{(\mathrm{latent})}
\le \delta_{\mathrm{lat}}^{+},
\\
\textit{strong anomaly},
& S^{(\mathrm{input})}>\delta_{\mathrm{rec}}
\land
S^{(\mathrm{latent})}>\delta_{\mathrm{lat}}^{+}.
\end{cases}
$$

## 10. Thích ứng trên cửa sổ hard-old-normality

$$
S^{(\mathrm{hard})}(\boldsymbol\phi)
=
\frac{1}{MTC}
\sum_{m=1}^{M}
\left\|
\mathbf X-
\widehat{\mathbf X}^{(m)}(\boldsymbol\phi)
\right\|_F^2,
$$

$$
\mathcal L_{\mathrm{hard\text{-}rec}}
=
\left[
S^{(\mathrm{hard})}(\boldsymbol\phi)-\delta_{\mathrm{rec}}
\right]_+.
$$

## 11. Xác minh pseudo-new-normality

Phần này phải so sánh nhiều cửa sổ trong bộ đệm, nên chỉ số cục bộ $w$ vẫn cần được giữ:

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

## 12. Khi nào vẫn cần chỉ số cửa sổ khác?

Chỉ mục $t$ đã được bỏ hoàn toàn vì nó chỉ dùng để nhận diện cửa sổ đang xét. Tuy nhiên, ở phần xác minh $\textit{pseudo-new-normality}$, mô hình phải so sánh nhiều cửa sổ khác nhau trong bộ đệm. Khi đó vẫn cần một chỉ số cửa sổ cục bộ như $u$ hoặc $w$, chẳng hạn $\mathbf z^{(\mathrm{on})}_{u,i}$ và $M^{(\mathrm{pnn})}_{u,i}$. Đây không phải là chỉ mục thời gian $t$ đã lược bỏ, mà là chỉ mục dùng để phân biệt các phần tử thật sự khác nhau trong bộ đệm xác minh.
