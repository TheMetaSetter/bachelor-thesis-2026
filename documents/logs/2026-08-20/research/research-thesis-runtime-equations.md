---
date: 2026-08-20T16:58:39+07:00
researcher: OpenAI Codex
topic: "Truy xuất codebase và viết lại các phương trình tương ứng runtime flow của phương pháp THESIS"
status: complete
revision: dfb205b761ba1cc4c9f8bcbf43e81ec902ad8cd5
branch: dev
---

# Research: Phương trình bám sát runtime flow của THESIS

## Tóm tắt

Runtime hiện tại xử lý THESIS theo hai pha lớn. Pha offline huấn luyện mô hình qua Stage A và Stage B. Pha online dùng checkpoint tốt nhất của Stage B để chấm điểm từng cửa sổ nhân quả, phân vùng cửa sổ và chỉ cập nhật một projector nhỏ trong các trường hợp được phép.

Phương trình anomaly score chính thức bắt đầu từ sai số tái tạo bình phương trung bình theo từng điểm. Runtime lấy trung bình sai số qua các mẫu Monte Carlo trong đánh giá offline. Runtime sau đó dùng median và MAD của clean validation để biến đổi sai số bằng sigmoid. Xác suất phân loại không tham gia anomaly score chính thức.

Bản này ưu tiên code đang chạy. Bản này không xem mọi phương trình trong `documents/spec/full-spec-v3.md` là hành vi đã được triển khai. Phần “Điểm chưa khớp” ghi rõ các chỗ mà đặc tả và runtime hiện tại khác nhau.

## Câu hỏi nghiên cứu

Truy xuất codebase và viết lại các phương trình tương ứng runtime flow của phương pháp THESIS. Dùng câu và từ dễ hiểu cho người đọc phổ thông.

## Phạm vi và quy ước ký hiệu

Ta xét một cửa sổ dữ liệu:

$$
\mathbf{X}_t=
[\mathbf{x}_{t,1};\ldots;\mathbf{x}_{t,L}]
\in\mathbb{R}^{L\times C}.
$$

Trong đó:

- $t$ là số thứ tự của cửa sổ;
- $i\in\{1,\ldots,L\}$ là vị trí của một điểm trong cửa sổ;
- $C$ là số đặc trưng của một điểm;
- $H$ là số chiều của vector ẩn;
- $m\in\{1,\ldots,M\}$ là số thứ tự của mẫu Monte Carlo;
- $n=\operatorname{start}(t)+i-1$ là chỉ số tuyệt đối của điểm trên timeline.

Config SMD đang được dùng đặt $L=20$, $C=38$, $H=32$, $K_c=32$, $K_d=60$ và $M=10$ khi đánh giá. $K_c$ là số continuous prototype. $K_d$ là số discrete codeword.

## Runtime flow đã xác nhận

```text
config thí nghiệm
  -> Stage A: encoder + hai prediction head
  -> khởi tạo continuous bank và discrete codebook từ tập train
  -> Stage B: đọc hai memory đã đóng băng + học fusion và prediction head
  -> clean validation: fit center, scale và threshold
  -> offline test hoặc online stream
  -> online: score -> EWMA -> triage -> update hoặc buffer -> PNN verification
```

Entry point `scripts/run_thesis_offline_benchmark.py` chuyển quyền điều khiển cho benchmark wrapper. Wrapper gọi đúng runner hai giai đoạn. Runner tạo config riêng cho Stage A và Stage B, chạy Stage A trước, khởi tạo memory, rồi mới chạy Stage B.

## 1. Stage A: học biểu diễn trực tiếp từ cửa sổ

### 1.1 Encoder

Encoder biến đổi từng cửa sổ đầu vào thành một chuỗi vector ẩn:

$$
\mathbf{Z}_t
=f_{\mathrm{enc}}(\mathbf{X}_t;\theta_{\mathrm{enc}})
=[\mathbf{z}_{t,1};\ldots;\mathbf{z}_{t,L}]
\in\mathbb{R}^{L\times H}.
$$

Stage A chưa dùng continuous prototype bank hoặc discrete codebook. Runtime chuyển thẳng $\mathbf{Z}_t$ đến hai prediction head:

$$
\widehat{\mathbf{X}}_t
=f_{\mathrm{rec}}(\mathbf{Z}_t),
$$

$$
\mathbf{o}_t
=f_{\mathrm{cls}}(\operatorname{vec}(\mathbf{Z}_t)).
$$

$\widehat{\mathbf{X}}_t$ là cửa sổ được tái tạo. $\mathbf{o}_t\in\mathbb{R}^{K}$ là vector logit phân loại cho $K=12$ lớp. Logit là giá trị trước hàm softmax.

### 1.2 Reconstruction loss chỉ tính trên điểm normal

Gọi $a_{t,i}\in\{0,1\}$ là nhãn synthetic tại điểm $i$. Giá trị $a_{t,i}=1$ cho biết bộ sinh dữ liệu đã chèn anomaly vào điểm đó. Runtime tạo mặt nạ normal:

$$
m_{t,i}=1-a_{t,i}.
$$

Runtime tính reconstruction loss như sau:

$$
\mathcal L_{\mathrm{rec}}
=
\frac{
\sum_{t,i,c}m_{t,i}
(\widehat x_{t,i,c}-x_{t,i,c})^2
}{
C\sum_{t,i}m_{t,i}
}.
$$

Nếu batch không có điểm normal, code dùng MSE trên toàn bộ batch để tránh phép chia cho 0.

### 1.3 Classification loss với label refurbishment

Runtime biến nhãn cứng thành phân phối mục tiêu $\mathbf{r}_t$. Với lớp thật $y_t$ và lớp normal có chỉ số $0$, runtime dùng $\alpha=0.1$ và $\beta=0.01$.

Nếu $y_t=0$, runtime đặt:

$$
r_{t,0}=1-(K-1)\beta,
\qquad
r_{t,k}=\beta\quad(k\ne0).
$$

Nếu $y_t\ne0$, runtime đặt:

$$
r_{t,y_t}=1-\alpha-(K-1)\beta,
$$

$$
r_{t,0}=\alpha+\beta,
\qquad
r_{t,k}=\beta
\quad(k\notin\{0,y_t\}).
$$

Các giá trị trên đã có tổng bằng $1$. Classification loss là cross-entropy:

$$
\mathcal L_{\mathrm{cls}}
=-\frac{1}{B}\sum_{t=1}^{B}
\sum_{k=1}^{K}r_{t,k}
\log\operatorname{softmax}(\mathbf{o}_t)_k.
$$

### 1.4 Contrastive loss trên các điểm normal

Runtime tạo một cửa sổ sạch và một cửa sổ đã chèn synthetic anomaly. Runtime chỉ giữ các vị trí vẫn là normal trong cửa sổ đã chèn anomaly.

Gọi $\mathbf{u}_j$ là vector ẩn đã chuẩn hóa từ cửa sổ sạch. Gọi $\mathbf{v}_j$ là vector ẩn đã chuẩn hóa tại cùng vị trí trong cửa sổ synthetic. Với $N_0$ điểm normal trong batch, runtime tính:

$$
\ell_{j,k}
=\frac{\mathbf{u}_j^\top\mathbf{v}_k}{\tau_{\mathrm{con}}},
$$

$$
\mathcal L_{\mathrm{con}}
=-\frac{1}{N_0}\sum_{j=1}^{N_0}
\log
\frac{\exp(\ell_{j,j})}
{\sum_{k=1}^{N_0}\exp(\ell_{j,k})}.
$$

Config đặt $\tau_{\mathrm{con}}=0.1$.

### 1.5 Balanced Point-Score Loss của O1

O0 không dùng loss này. O1 chỉ dùng loss này trong Stage A.

Trước hết, runtime tính MSE thô theo từng điểm:

$$
e_{t,i}
=\frac{1}{C}\sum_{c=1}^{C}
(\widehat x_{t,i,c}-x_{t,i,c})^2.
$$

Runtime chỉ dùng các điểm normal trong batch để tính trung bình và độ lệch chuẩn:

$$
\mu_0
=\frac{1}{|\mathcal N|}
\sum_{(t,i)\in\mathcal N}e_{t,i},
$$

$$
\sigma_0
=\sqrt{
\frac{1}{|\mathcal N|}
\sum_{(t,i)\in\mathcal N}(e_{t,i}-\mu_0)^2
}.
$$

Code ngắt gradient qua $\mu_0$ và $\sigma_0$. Runtime sau đó tính logit chuẩn hóa:

$$
\eta_{t,i}
=\frac{e_{t,i}-\mu_0}{\max(\sigma_0,\varepsilon)}.
$$

Gọi $\operatorname{BCELogit}(\eta,a)$ là binary cross-entropy nhận logit $\eta$ và nhãn $a$. Runtime cân bằng hai nhóm bằng nhau:

$$
\mathcal L_{\mathrm{BPSL}}
=\frac{1}{2|\mathcal N|}
\sum_{(t,i)\in\mathcal N}
\operatorname{BCELogit}(\eta_{t,i},0)
+
\frac{1}{2|\mathcal A|}
\sum_{(t,i)\in\mathcal A}
\operatorname{BCELogit}(\eta_{t,i},1).
$$

Nếu batch thiếu một trong hai nhóm, runtime bỏ $\mathcal L_{\mathrm{BPSL}}$ cho batch đó.

### 1.6 Total loss thật sự của Stage A

Config chính đặt $\lambda_{\mathrm{rec}}=0.5$, $\lambda_{\mathrm{cls}}=0.5$ và $\lambda_{\mathrm{con}}=0.3$. Các optional loss khác có trọng số bằng $0$.

Với O0, runtime dùng:

$$
\boxed{
\mathcal L_A^{(O0)}
=0.5\mathcal L_{\mathrm{rec}}
+0.5\mathcal L_{\mathrm{cls}}
+0.3\mathcal L_{\mathrm{con}}
}.
$$

Với O1, runtime thay classification branch loss bằng trung bình của classification loss và BPSL:

$$
\mathcal L_{\mathrm{cls\text{-}branch}}^{(O1)}
=\frac{1}{2}
(\mathcal L_{\mathrm{cls}}+\mathcal L_{\mathrm{BPSL}}).
$$

Do đó, total loss thật sự là:

$$
\boxed{
\mathcal L_A^{(O1)}
=0.5\mathcal L_{\mathrm{rec}}
+0.25\mathcal L_{\mathrm{cls}}
+0.25\mathcal L_{\mathrm{BPSL}}
+0.3\mathcal L_{\mathrm{con}}
}.
$$

## 2. Ranh giới khởi tạo memory

Sau Stage A, runtime đóng băng encoder và chạy lại dữ liệu train. Runtime lấy vector ẩn normal để tạo continuous prototype bank. Runtime chia vector ẩn theo 12 lớp để tạo discrete codebook.

Với một tập token $\{\mathbf{z}_j\}$, k-means lặp hai bước:

$$
c_j
=\arg\min_k
\|\bar{\mathbf z}_j-\mathbf p_k\|_2,
$$

$$
\mathbf p_k
=\frac{1}{|\mathcal C_k|}
\sum_{j\in\mathcal C_k}\bar{\mathbf z}_j.
$$

$\bar{\mathbf z}_j$ là vector đã chuẩn hóa L2. Runtime chạy 10 vòng lặp. Continuous bank có 32 prototype. Discrete codebook có 60 codeword, tức 5 codeword cho mỗi lớp khi có 12 lớp.

Runtime đánh dấu codeword của lớp $1$ đến lớp $11$ là codeword anomaly. Với một anomaly token, runtime tính khoảng cách cosine đến codeword $k$:

$$
d(\mathbf z,\mathbf e_k)
=1-
\frac{\mathbf z^\top\mathbf e_k}
{\|\mathbf z\|_2\|\mathbf e_k\|_2}.
$$

Runtime gán mỗi anomaly token cho codeword gần nhất. Radius của codeword $k$ là phân vị $0.99$ của các khoảng cách đã được gán cho nó:

$$
R_k
=Q_{0.99}
\left(
\{d(\mathbf z_j,\mathbf e_k):c_j=k\}
\right).
$$

Runtime chỉ dùng dữ liệu train ở ranh giới này. Runtime không dùng validation, test hoặc dữ liệu online để tạo memory.

## 3. Stage B: truy hồi memory và học fusion

Stage B đóng băng encoder, continuous bank và discrete codebook. Runtime chỉ huấn luyện hai projection dùng cho fusion và hai prediction head.

### 3.1 Continuous retrieval trong forward huấn luyện

Runtime chuẩn hóa vector ẩn và prototype. Runtime tính logit:

$$
g^{(c)}_{t,i,k}
=\frac{
\bar{\mathbf z}_{t,i}^{\top}\bar{\mathbf p}^{(c)}_k
}{\sqrt H}.
$$

Runtime tính trọng số và vector truy hồi:

$$
\alpha^{(c)}_{t,i,k}
=\operatorname{softmax}_k(g^{(c)}_{t,i,k}),
$$

$$
\widetilde{\mathbf z}^{(c)}_{t,i}
=\operatorname{norm}
\left(
\sum_{k=1}^{K_c}
\alpha^{(c)}_{t,i,k}\bar{\mathbf p}^{(c)}_k
\right).
$$

### 3.2 Discrete top-3 retrieval trong forward huấn luyện

Runtime dùng cosine similarity:

$$
g^{(d)}_{t,i,k}
=\bar{\mathbf z}_{t,i}^{\top}\bar{\mathbf e}^{(d)}_k.
$$

Gọi $I_{t,i}$ là ba codeword có logit lớn nhất. Với nhiệt độ truy vấn $\tau_d=0.1$, runtime tính:

$$
\alpha^{(d)}_{t,i,k}
=
\frac{
\exp(g^{(d)}_{t,i,k}/\tau_d)
}{
\sum_{j\in I_{t,i}}
\exp(g^{(d)}_{t,i,j}/\tau_d)
},
\qquad k\in I_{t,i}.
$$

$$
\widetilde{\mathbf z}^{(d)}_{t,i}
=\operatorname{norm}
\left(
\sum_{k\in I_{t,i}}
\alpha^{(d)}_{t,i,k}\bar{\mathbf e}^{(d)}_k
\right).
$$

### 3.3 Fusion và prediction

Config chọn `task_specific_concat_projection`. Runtime nối hai vector truy hồi rồi dùng projection riêng cho từng tác vụ:

$$
\mathbf h^{(rec)}_{t,i}
=f_{\mathrm{fuse,rec}}
\left(
[\widetilde{\mathbf z}^{(c)}_{t,i};
\widetilde{\mathbf z}^{(d)}_{t,i}]
\right),
$$

$$
\mathbf h^{(cls)}_{t,i}
=f_{\mathrm{fuse,cls}}
\left(
[\widetilde{\mathbf z}^{(c)}_{t,i};
\widetilde{\mathbf z}^{(d)}_{t,i}]
\right).
$$

Hai prediction head tạo kết quả:

$$
\widehat{\mathbf x}_{t,i}
=f_{\mathrm{rec}}(\mathbf h^{(rec)}_{t,i}),
$$

$$
\mathbf o_t
=f_{\mathrm{cls}}
(\operatorname{vec}(\mathbf H_t^{(cls)})).
$$

Total loss thật sự trong Stage B là:

$$
\boxed{
\mathcal L_B
=0.5\mathcal L_{\mathrm{rec}}
+0.5\mathcal L_{\mathrm{cls}}
}.
$$

Stage B không dùng contrastive loss hoặc BPSL.

## 4. Đánh giá offline và anomaly score chính thức

### 4.1 Monte Carlo retrieval

Trong chế độ đánh giá, runtime thêm Gumbel noise vào continuous logit:

$$
\widetilde g^{(c,m)}_{t,i,k}
=g^{(c)}_{t,i,k}+G^{(m)}_{t,i,k},
$$

$$
\alpha^{(c,m)}_{t,i,k}
=\operatorname{softmax}_k
\left(
\frac{\widetilde g^{(c,m)}_{t,i,k}}{\tau_c}
\right),
$$

$$
\widetilde{\mathbf z}^{(c,m)}_{t,i}
=\operatorname{norm}
\left(
\sum_k\alpha^{(c,m)}_{t,i,k}
\bar{\mathbf p}^{(c)}_k
\right).
$$

Config đặt $\tau_c=0.9$ và $M=10$.

Nhánh discrete dùng cùng top-3 ID và cùng softmax weight cho mọi $m$ khi `discrete_query_mode=cosine_topk`. Code hiện tại không thêm Gumbel noise cho nhánh này. Vì vậy:

$$
\widetilde{\mathbf z}^{(d,m)}_{t,i}
=\widetilde{\mathbf z}^{(d)}_{t,i}
\quad\text{với mọi }m.
$$

Runtime fusion từng cặp sample, rồi tạo reconstruction $\widehat{\mathbf x}^{(m)}_{t,i}$.

### 4.2 Raw point MSE

MSE của một sample là:

$$
s^{(m)}_{t,i}
=\frac{1}{C}
\sum_{c=1}^{C}
(\widehat x^{(m)}_{t,i,c}-x_{t,i,c})^2.
$$

Runtime lấy trung bình của các MSE:

$$
\boxed{
\overline s_{t,i}
=\frac{1}{M}
\sum_{m=1}^{M}s^{(m)}_{t,i}
}.
$$

Đây là MSE trung bình qua từng sample. Runtime không lấy MSE giữa input và reconstruction trung bình.

Raw window score là trung bình theo các điểm trong cửa sổ:

$$
S_t^{(input)}
=\frac{1}{L}
\sum_{i=1}^{L}\overline s_{t,i}.
$$

### 4.3 Calibration bằng clean validation

Lần đánh giá clean validation đầu tiên chạy khi model chưa có calibration. Vì vậy, trường `point_scores` tạm thời chứa raw point MSE. Runtime gộp score về timeline rồi mới fit calibration.

Gọi $\mathcal S_{\mathrm{cv}}$ là toàn bộ raw point MSE đã được gộp trên clean validation. Runtime tính:

$$
\mu_{\mathrm{cv}}
=\operatorname{median}(\mathcal S_{\mathrm{cv}}),
$$

$$
\operatorname{MAD}_{\mathrm{cv}}
=\operatorname{median}
\left(
\{|s-\mu_{\mathrm{cv}}|:s\in\mathcal S_{\mathrm{cv}}\}
\right),
$$

$$
\gamma_{\mathrm{cv}}
=\frac{\operatorname{MAD}_{\mathrm{cv}}}{0.6745}.
$$

Runtime biến đổi raw point MSE bằng shifted-and-scaled logistic sigmoid:

$$
\boxed{
q_{t,i}
=\sigma
\left(
\frac{\overline s_{t,i}-\mu_{\mathrm{cv}}}
{\gamma_{\mathrm{cv}}}
\right)
}.
$$

$q_{t,i}$ là anomaly score chính thức. Runtime field `point_scores` chứa $q_{t,i}$ sau khi calibration đã được gắn vào model.

Code yêu cầu $\gamma_{\mathrm{cv}}>0$. Nếu MAD bằng $0$, hàm fit hiện tại tạo $\gamma_{\mathrm{cv}}=0$ và dataclass sẽ báo lỗi. Code không tự thêm $\varepsilon$ ở bước này.

### 4.4 Gộp cửa sổ về timeline offline

Một điểm tuyệt đối có thể nằm trong nhiều cửa sổ. Gọi $\mathcal W(n)$ là tập cửa sổ chứa điểm $n$. Runtime lấy trung bình:

$$
q_n
=\frac{1}{|\mathcal W(n)|}
\sum_{t\in\mathcal W(n)}q_{t,i(n,t)}.
$$

Protocol offline dùng stride $20$ và thêm một cửa sổ cuối được canh về cuối chuỗi. Vì vậy, phần đuôi có thể chồng lên cửa sổ trước. Runtime cũng lấy trung bình tại phần chồng này.

Threshold offline chỉ dùng clean validation:

$$
T_{\mathrm{off}}
=Q_{\alpha_{\mathrm{off}}}
(\{q_n^{(cv)}\}).
$$

Protocol q95 đặt $\alpha_{\mathrm{off}}=0.95$. Runtime dự đoán anomaly khi và chỉ khi:

$$
\widehat a_n
=\mathbb I[q_n>T_{\mathrm{off}}].
$$

Dấu so sánh là $>$, không phải $\ge$.

## 5. Runtime online

### 5.1 Source encoder và residual projector

Online A0 dùng trực tiếp encoder nguồn đã đóng băng:

$$
\mathbf Z_t^{(src)}
=f_{\mathrm{enc}}^{(frozen)}(\mathbf X_t).
$$

A1 và A2 đưa vector này qua residual projector:

$$
\mathbf Z_t^{(proj)}
=\mathbf Z_t^{(src)}
+\alpha_p f_2
\left(
\operatorname{Dropout}
(\operatorname{GELU}(f_1(\mathbf Z_t^{(src)})))
\right).
$$

Runtime chỉ cập nhật tham số của projector. Encoder, memory và prediction head giữ nguyên.

Online scorer hiện tại dùng deterministic continuous retrieval và deterministic discrete top-3 retrieval trên $\mathbf Z_t^{(src)}$ hoặc $\mathbf Z_t^{(proj)}$. Scorer tạo reconstruction, raw point MSE và calibrated point score theo cùng công thức ở phần offline.

### 5.2 Input window score và latent window score

Runtime tính input window score trực tiếp từ reconstruction:

$$
S_t^{(input)}
=\frac{1}{LC}
\sum_{i=1}^{L}\sum_{c=1}^{C}
(\widehat x_{t,i,c}-x_{t,i,c})^2.
$$

Runtime tính latent window score bằng khoảng cách cosine đến continuous prototype normal gần nhất:

$$
d_{t,i,k}^{(c)}
=1-
\frac{
(\mathbf z_{t,i}^{(query)})^\top\mathbf p_k^{(c)}
}{
\|\mathbf z_{t,i}^{(query)}\|_2
\|\mathbf p_k^{(c)}\|_2
},
$$

$$
\boxed{
S_t^{(latent)}
=\frac{1}{L}\sum_{i=1}^{L}
\min_k d_{t,i,k}^{(c)}
}.
$$

$S_t^{(latent)}$ là score ở mức cửa sổ. Nó không phải point-level latent MSE.

### 5.3 EWMA theo chỉ số tuyệt đối

Mỗi cửa sổ online có stride $1$. Do đó, cùng một điểm tuyệt đối xuất hiện trong nhiều cửa sổ. Runtime lưu EWMA riêng cho từng chỉ số tuyệt đối $n$.

Nếu runtime gặp điểm $n$ lần đầu, nó đặt:

$$
\widetilde q_n^{(r)}=q_n^{(r)}.
$$

Nếu runtime đã có score trước đó cho cùng điểm, nó cập nhật:

$$
\boxed{
\widetilde q_n^{(r)}
=w_{prev}\widetilde q_n^{(r-1)}
+w_{cur}q_n^{(r)}
}.
$$

Protocol q95 đặt $w_{cur}=0.9$ và $w_{prev}=0.1$.

Threshold online cũng chỉ dùng clean validation. Runtime mô phỏng đúng stride $1$ và EWMA trên clean validation:

$$
T_{\mathrm{on}}
=Q_{\alpha_{\mathrm{on}}}
(\{\widetilde q_n^{(cv)}\}).
$$

Runtime dự đoán điểm cuối của sự kiện hiện tại bằng:

$$
\widehat a_n
=\mathbb I[\widetilde q_n>T_{\mathrm{on}}].
$$

### 5.4 Four-region triage

Triage là bước chia cửa sổ thành bốn vùng. Bước này dùng hai window score. Bước này không dùng nhãn thật.

$$
R_t=
\begin{cases}
\texttt{normal},
&S_t^{(input)}\le B_{window},\\
\texttt{hard\_old\_normality},
&S_t^{(input)}>B_{window}
\ \land\ S_t^{(latent)}\le A_{low},\\
\texttt{gray\_zone},
&S_t^{(input)}>B_{window}
\ \land\ A_{low}<S_t^{(latent)}\le A_{high},\\
\texttt{strong\_anomaly},
&S_t^{(input)}>B_{window}
\ \land\ S_t^{(latent)}>A_{high}.
\end{cases}
$$

Protocol q95 lấy $B_{window}$ ở phân vị $0.99$, $A_{low}$ ở phân vị $0.75$ và $A_{high}$ ở phân vị $0.99$ của các timeline calibration tương ứng.

### 5.5 Xác minh pseudo-new-normality

Chỉ cửa sổ `gray_zone` được đưa vào verification buffer. Buffer từ chối hai cửa sổ chồng nhau.

Với mỗi token nguồn $\mathbf z_{t,i}^{(src)}$, runtime tìm discrete codeword gần nhất:

$$
k^*_{t,i}
=\arg\min_k d(\mathbf z_{t,i}^{(src)},\mathbf e_k).
$$

Runtime đánh dấu token thuộc anomaly đã biết khi codeword gần nhất là codeword anomaly và khoảng cách nằm trong radius:

$$
M_{t,i}^{(known)}
=\mathbb I
\left[
M_{k^*_{t,i}}^{(anom)}=1
\ \land\
d(\mathbf z_{t,i}^{(src)},\mathbf e_{k^*_{t,i}})
\le R_{k^*_{t,i}}
\right].
$$

Runtime tạo signature bằng ba continuous prototype gần nhất theo thứ tự:

$$
\boldsymbol\sigma_{t,i}
=\operatorname{Top3Ids}_{k}
\left(
-d(\mathbf z_{t,i}^{(src)},\mathbf p_k^{(c)})
\right).
$$

Gọi $\mathcal R$ là tập signature xuất hiện trong ít nhất hai cửa sổ không chồng nhau. PNN mask là:

$$
\boxed{
M_{t,i}^{(pnn)}
=\mathbb I[\boldsymbol\sigma_{t,i}\in\mathcal R]
\left(1-M_{t,i}^{(known)}\right)
}.
$$

Buffer bắt đầu một verification cycle khi có ít nhất 8 entry và có entry mới. Entry chưa được xác minh có TTL bằng 2 cycle. Runtime giảm TTL sau mỗi verification cycle, không giảm sau mỗi stream step.

### 5.6 Quy tắc update của A0, A1 và A2

A0 không cập nhật tham số.

A1 chỉ cập nhật khi cửa sổ đã được xác minh là `pnn_verified` và PNN mask có ít nhất một điểm. Reconstruction loss của A1 là:

$$
\boxed{
\mathcal L_{A1}
=
\frac{
\sum_{i,c}M_{t,i}^{(pnn)}
(\widehat x_{t,i,c}-x_{t,i,c})^2
}{
C\sum_iM_{t,i}^{(pnn)}
}
}.
$$

A2 cho phép hai đường update.

Với `pnn_verified`, A2 dùng cùng masked reconstruction loss:

$$
\mathcal L_{rec}^{(A2,pnn)}=\mathcal L_{A1}.
$$

Với `hard_old_normality`, A2 dùng hinge loss:

$$
\mathcal L_{rec}^{(A2,hard)}
=\left[
\max(0,\overline q_t-T_{on})
\right]^2,
$$

trong đó $\overline q_t$ là trung bình calibrated point score của cửa sổ trong forward update. Đây là công thức đúng theo lời gọi hàm hiện tại.

A2 còn dùng token-level multi-positive InfoNCE. Với một anchor $\mathbf u$, tập positive $\mathcal P(u)$ gồm vector nguồn cùng vị trí và các token có cùng recurrent signature. Tập negative $\mathcal N(u)$ gồm anomaly codeword và các token được đánh dấu anomaly đã biết. Runtime tính:

$$
\mathcal L_{con}(u)
=
\log
\sum_{\mathbf v\in\mathcal P(u)\cup\mathcal N(u)}
\exp
\left(
\frac{\bar{\mathbf u}^{\top}\bar{\mathbf v}}{\tau}
\right)
-
\log
\sum_{\mathbf v\in\mathcal P(u)}
\exp
\left(
\frac{\bar{\mathbf u}^{\top}\bar{\mathbf v}}{\tau}
\right).
$$

Runtime lấy trung bình loss trên các anchor hợp lệ. Total loss của A2 là:

$$
\boxed{
\mathcal L_{A2}
=\mathcal L_{rec}^{(A2)}
+0.1\mathcal L_{con}^{(A2)}
}.
$$

`strong_anomaly`, `normal` và `gray_zone` chưa được xác minh không làm model update. Cửa sổ `hard_old_normality` chỉ được update khi không chồng lên cửa sổ hard-old đã được chấp nhận trước đó.

## 6. Uncertainty được tính nhưng không điều khiển quyết định

Với một đại lượng $y^{(m)}$ từ $M$ mẫu, runtime tính sample variance với correction bằng $1$:

$$
\operatorname{Var}(y)
=\frac{1}{M-1}
\sum_{m=1}^{M}
(y^{(m)}-\bar y)^2.
$$

Runtime lưu variance của reconstruction, continuous retrieval, discrete retrieval, point score, window score và xác suất phân loại. Runtime hiện tại không dùng các variance này để đặt threshold, triage, admission, PNN hoặc adaptation.

## 7. Mapping từ phương trình sang runtime field

| Đại lượng | Ký hiệu | Runtime field hoặc hàm |
| --- | --- | --- |
| Raw point MSE | $\overline s_{t,i}$ | `aux.raw_point_scores` |
| Anomaly score chính thức | $q_{t,i}$ | `point_scores` |
| Raw input window score | $S_t^{(input)}$ | `input_window_score`; offline `window_scores` |
| Latent window score | $S_t^{(latent)}$ | `latent_window_score` |
| EWMA theo điểm tuyệt đối | $\widetilde q_n$ | `active_ewma_point_scores` |
| Offline point threshold | $T_{off}$ | `offline_point_threshold_nonoverlap` trong threshold artifact |
| Online EWMA threshold | $T_{on}$ | `online_point_threshold_ewma` trong threshold artifact |
| Known-anomaly mask | $M^{(known)}$ | `known_anomaly_mask` |
| PNN mask | $M^{(pnn)}$ | `pnn_mask` |

## 8. Cấu hình đã quan sát

| Setting | Giá trị trong config được kiểm tra | Evidence | Phạm vi |
| --- | --- | --- | --- |
| `window_size` | `20` | `configs/model/thesis_multitask_two_stage_window20.yaml:5` | Model SMD |
| `input_dim` | `38` | `configs/model/thesis_multitask_two_stage_window20.yaml:4` | Model SMD |
| `hidden_dim` | `32` | `configs/model/thesis_multitask_two_stage_window20.yaml:10` | Model SMD |
| `continuous_num_prototypes` | `32` | `configs/model/thesis_multitask_two_stage_window20.yaml:21` | Continuous bank |
| `discrete_codebook_size` | `60` | `configs/model/thesis_multitask_two_stage_window20.yaml:23` | Discrete codebook |
| `discrete_topk` | `3` | `configs/model/thesis_multitask_two_stage_window20.yaml:81` | Stage B và inference |
| `monte_carlo_samples` | `10` | `configs/model/thesis_multitask_two_stage_window20.yaml:25` | Model config; offline evaluation dùng giá trị này |
| `lambda_recon` | `0.5` | `configs/model/thesis_multitask_two_stage_window20.yaml:48` | Stage A và Stage B |
| `lambda_cls` | `0.5` | `configs/model/thesis_multitask_two_stage_window20.yaml:49` | Stage A và Stage B |
| `lambda_contrastive` | `0.3` | `configs/model/thesis_multitask_two_stage_window20.yaml:66` | Chỉ Stage A |
| `offline_window_stride` | `20` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:6` | Offline calibration và test |
| `online_window_stride` | `1` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:7` | Online calibration và stream |
| `online_ewma_current_weight` | `0.9` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:10` | Online EWMA |
| `online_ewma_previous_weight` | `0.1` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:11` | Online EWMA |
| `offline_threshold_quantile` | `0.95` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:5` | Protocol q95 được kiểm tra |
| `online_threshold_quantile` | `0.95` | `configs/protocol/smd_window20_cleanval_q95_ewma09.yaml:9` | Protocol q95 được kiểm tra |

## 9. Điểm chưa khớp giữa runtime và đặc tả

### 9.1 Công thức O1 trong spec không thể hiện đúng cách code ghép loss

`full-spec-v3.md` viết $\mathcal L_{point-score}$ như một số hạng cộng riêng. Runtime không cộng một số hạng có trọng số riêng. Runtime thay classification branch loss bằng $\tfrac12(\mathcal L_{cls}+\mathcal L_{BPSL})$. Vì vậy, với config hiện tại, hệ số thật sự của mỗi loss là $0.25$.

### 9.2 Stage B training không dùng sample Monte Carlo để tính loss

Trong training mode, forward có tạo stochastic query khi `stochastic_inference=true`. Tuy nhiên, code chỉ thay top-level reconstruction bằng Monte Carlo output khi `not self.training`. Stage B loss vì vậy dùng deterministic retrieval path. Runner cũng không đổi `monte_carlo_samples` từ `10` thành `1` cho Stage B.

Điều này khác với câu trong spec rằng Stage B training dùng $M_{train}=1$ stochastic sample.

### 9.3 Discrete top-3 không stochastic trong config hiện tại

Spec mô tả Gumbel perturbation cho discrete top-k. Runtime chỉ thêm Gumbel noise khi query mode không phải `cosine_topk`. Config hiện tại chọn `cosine_topk`. Do đó, 10 discrete samples giống nhau.

### 9.4 Online scorer chưa chạy vectorized Monte Carlo $M=10$

Offline model có đường Monte Carlo riêng trong eval mode. `OnlineAdaptationModel` không gọi đường này. Online scorer gọi deterministic prototype lookup từ hidden đã chiếu. Vì vậy, code hiện tại chưa thực hiện câu trong spec rằng official online scoring dùng 10 stochastic retrieval samples.

### 9.5 Hinge loss hard-old nhận online point threshold

Hàm loss đặt tên tham số là `b_window`, nhưng caller truyền `threshold_value`, tức online EWMA point threshold $T_{on}$. Hàm cũng nhận trung bình calibrated `window_scores`, không nhận raw input window score $S_t^{(input)}$. Công thức ở phần 5.6 phản ánh code đang chạy, không phản ánh tên tham số.

### 9.6 Tên `raw_point_score` có lúc không giữ cùng một nghĩa trong buffer path

Trong luồng cửa sổ hiện tại, `raw_point_score` là raw MSE của điểm cuối. Khi code xử lý entry từ verification buffer, caller truyền `input_window_score` vào tham số `raw_point_score`. Giá trị này chủ yếu đi vào record và không đi vào reconstruction loss, nhưng tên field không còn giữ một nghĩa duy nhất.

## 10. Kiểm chứng

Em đã chạy các test tập trung bằng `.venv/bin/python`:

```text
tests/evaluation/test_point_score_contracts.py
tests/models/test_thesis_multitask_point_score_loss.py
tests/online/test_online_tta_triage.py
tests/online/test_online_signature_verification.py
tests/online/test_online_tta_variants.py
tests/online/test_online_verification_buffer.py
```

Kết quả: `27 passed in 2.40s`.

Các test xác nhận công thức calibration, balanced point-score loss, four-region triage, signature verification, quy tắc update A0/A1/A2 và TTL của verification buffer. Test này không chứng minh toàn bộ benchmark chạy end-to-end tại revision hiện tại.

## 11. Evidence chính

- `scripts/benchmarks/run_thesis_offline_benchmark.py:438-478` — clean validation chạy hai lần; lần đầu fit median/MAD calibration, lần sau tạo calibrated score và threshold.
- `scripts/experiments/run_two_stage_offline_pretraining.py:115-162` — runner tạo config Stage A và Stage B theo đúng thứ tự.
- `scripts/experiments/run_two_stage_offline_pretraining.py:244-337` — runner nạp Stage A, khởi tạo memory bằng dữ liệu train và lưu Stage B initialization checkpoint.
- `src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py:9-64` — Stage A bỏ qua memory và đưa hidden trực tiếp đến prediction head.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:30-52` — reconstruction loss chỉ dùng điểm normal và có fallback khi không có điểm normal.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:94-167` — classification loss và label refurbishment.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:270-341` — BPSL dùng raw point MSE, thống kê normal đã detach và binary-balanced BCE.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py:16-38` — O1 ghép classification loss và BPSL theo trung bình $1/2$.
- `src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py:243-273` — thứ tự tính các loss và total loss.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py:197-238` — deterministic continuous retrieval.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:273-354` — discrete cosine top-3 retrieval.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:357-397` — task-specific concat projection.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_helpers.py:13-157` — stochastic continuous sampling và deterministic discrete top-3 sampling trong config hiện tại.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:138-246` — Monte Carlo fusion, reconstruction, mean score và uncertainty.
- `src/protocols/point_score_calibration.py:71-107` — median, MAD-based scale và shifted-and-scaled logistic sigmoid.
- `src/engine/evaluator.py:122-227` — gộp point score từ cửa sổ về timeline bằng trung bình.
- `src/engine/online_tta/point_ewma.py:8-34` — EWMA theo từng chỉ số tuyệt đối.
- `src/models/online_impl/online_adaptation_helpers.py:50-118` — online deterministic scorer và nearest-normal-prototype latent window score.
- `src/engine/online_tta/triage.py:17-41` — four-region triage.
- `src/engine/online_tta/signature_verification.py:116-139` — known-anomaly filter.
- `src/engine/online_tta/signature_verification.py:171-189` — ordered top-3 continuous signature.
- `src/engine/online_tta/signature_verification.py:230-277` — recurrent signature và PNN mask.
- `src/engine/online_tta/online_engine_step.py:108-182` — điều kiện update và loss của A1/A2.
- `src/engine/online_tta/verification_buffer.py:41-80` — non-overlap admission, capacity và TTL theo verification cycle.

## Kết luận

Phương trình trung tâm của THESIS runtime là chuỗi:

$$
\mathbf X_t
\rightarrow
\mathbf Z_t
\rightarrow
\widehat{\mathbf X}^{(m)}_t
\rightarrow
\overline s_{t,i}
\rightarrow
q_{t,i}
\rightarrow
\widetilde q_n
\rightarrow
\widehat a_n.
$$

Stage A học encoder và prediction head. Ranh giới sau Stage A tạo hai memory từ dữ liệu train. Stage B học cách kết hợp hai memory. Clean validation quyết định calibration và threshold. Online runtime dùng thêm latent window score, four-region triage, verification buffer và PNN mask để giới hạn các lần update.

Code hiện tại đã triển khai đầy đủ các phương trình chấm điểm cơ bản, triage, PNN và A1/A2 loss. Tuy nhiên, đường Monte Carlo online và một số chi tiết stochastic Stage B chưa khớp với `full-spec-v3.md`. Vì vậy, tài liệu luận văn nên mô tả hai nhóm riêng: hành vi runtime hiện có và hợp đồng mà đặc tả yêu cầu nhưng code chưa thực hiện đúng.
