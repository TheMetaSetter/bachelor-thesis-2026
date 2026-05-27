---
date: 2026-05-23T00:00:00+07:00
researcher: Codex (GPT-5)
git_commit: 57aeba72e81071194e6e271faab39fbc1e955c89
branch: dev
repository: bachelor-thesis-2026
topic: "Computational stages of offline pre-training in thesis_multitask.py"
tags: [research, offline-pretraining, multitask, prototypes, contrastive, cka]
status: complete
last_updated: 2026-05-23
last_updated_by: Codex (GPT-5)
---

# Research: Computational Stages of Offline Pre-Training

## Research Question
Duyệt codebase, đặc biệt `src/models/thesis_multitask.py`, và bóc tách đầy đủ các computational stage của offline pre-training: từ input 2-view (normal/anomalous), latent representation, contrastive loss, truy vấn prototype, tái tạo latent, cập nhật prototype, CKA similarity, fusion weights (alpha, beta), reconstruction/classification heads, loss, và cơ chế tạo label.

## Scope và code được truy vết
- Core model: `src/models/thesis_multitask.py`
- Synthetic label creation: `src/data/augment.py`
- Prompt workflow anchor: `prompts/1_research_prompt.md`

## Ký hiệu và kích thước tensor
- Batch size: $$B$$
- Window length: $$L$$
- Input channels/features: $$D$$
- Hidden dimension: $$H$$
- Continuous prototypes: $$K_c$$
- Discrete codebook size: $$K_d$$
- Number of classes: $$C$$
- Input batch: $$X \in \mathbb{R}^{B\times L\times D}$$
- Synthetic mask: $$M \in \{0,1\}^{B\times L}$$, với $$M_{b,t}=1$$ là timestep synthetic-anomalous.
- Classification labels: $$y^{cls} \in \{0,\dots,C-1\}^{B}$$

## Stage 0: Tạo batch và label (1-view hoặc 2-view)
Theo `_shared_step` + `_prepare_contrastive_pair_batches` + `augment_batch`.

### 0.1 Một view sạch (clean view)
- Không tiêm anomaly: $$X^{clean}=X$$
- Mask mặc định: $$M^{clean}=\mathbf{0}_{B\times L}$$
- Label mặc định: $$y^{cls,clean}=\mathbf{0}_{B}$$

### 0.2 Một view augmented (anomalous view)
Với từng window $$b$$, biến quyết định tiêm:
$$
z_b \sim \mathrm{Bernoulli}(p_{anom}),\quad z_b\in\{0,1\}
$$
- Nếu $$z_b=0$$: giữ nguyên clean.
- Nếu $$z_b=1$$: chọn 1 anomaly family và sửa subsequence, tạo mask $$M_{b,:}$$ và metadata.

Label phân loại từ metadata:
$$
y_b^{cls}=
\begin{cases}
0, & \text{nếu clean}\\
1, & \text{nếu binary mode và có anomaly}\\
\text{index}(\text{anomaly\_family}), & \text{nếu redlamp\_multiclass}
\end{cases}
$$

Point label:
$$
y^{point}=\max\left(y^{point}_{orig}, M\right)
$$
(nếu không có `point_labels` gốc thì $$y^{point}=M$$).

## Stage 1: Encoder tạo latent representation
Encoder MLP chạy theo từng timestep:
$$
H^{base}=f_{enc}(X)\in\mathbb{R}^{B\times L\times H}
$$
Trong đó pooled vector:
$$
h^{pool}_b=\frac{1}{L}\sum_{t=1}^{L} H^{base}_{b,t,:}\in\mathbb{R}^{H}
$$

## Stage 2: Hai-view contrastive (khi bật `enable_two_view_contrastive`)
Tạo cặp:
- Anchor: clean view $$X^{clean}$$
- Positive: augmented view $$X^{aug}$$

Latent:
$$
H^{anc}=f_{enc}(X^{clean}),\quad H^{pos}=f_{enc}(X^{aug})
$$
Lấy chỉ token normal theo mask của augmented view:
$$
\mathcal{I}_{normal}=\{(b,t)\mid M^{aug}_{b,t}=0\}
$$
$$
A=\{H^{anc}_{b,t,:}\}_{(b,t)\in\mathcal{I}_{normal}},\quad
P=\{H^{pos}_{b,t,:}\}_{(b,t)\in\mathcal{I}_{normal}}
$$
Chuẩn hóa:
$$
\hat{A}_i=\frac{A_i}{\|A_i\|_2},\quad \hat{P}_j=\frac{P_j}{\|P_j\|_2}
$$
Logits InfoNCE:
$$
S_{ij}=\frac{\hat{A}_i^\top \hat{P}_j}{\tau_{con}}
$$
Target diagonal (identity matching): $$t_i=i$$
$$
\mathcal{L}_{con}=\mathrm{CE}(S,t)
$$

## Stage 3: Cập nhật prototype memory trước khi lookup (train mode, memory enabled)
Trong `forward`, update xảy ra trước lookup.

### 3.1 Continuous memory update
Bank hiện tại:
$$
P^c\in\mathbb{R}^{K_c\times H}
$$
Chuẩn hóa hidden và bank:
$$
\tilde{H}=\mathrm{norm}(H^{base}),\quad \tilde{P}^c=\mathrm{norm}(P^c)
$$
Nếu có token mask thì chỉ giữ tập token tương ứng.

Prototype-to-token attention logits:
$$
\Lambda_{k,b,t}=\frac{\langle \tilde{P}^c_k,\tilde{H}_{b,t}\rangle}{\sqrt{H}}
$$
Softmax theo toàn bộ token của mỗi prototype:
$$
W_{k,b,t}=\mathrm{softmax}_{(b,t)}(\Lambda_{k,b,t})
$$
Weighted summary mỗi prototype:
$$
U_k=\mathrm{norm}\left(\sum_{b,t} W_{k,b,t}\tilde{H}_{b,t}\right)
$$
Gate update (MLP sigmoid):
$$
g_k=\sigma\left(f_{gate}([\tilde{P}^c_k;U_k])\right)\in(0,1)^H
$$
Cập nhật prototype:
$$
P^{c,new}_k=\mathrm{norm}\left((1-g_k)\odot \tilde{P}^c_k + g_k\odot U_k\right)
$$

### 3.2 Discrete codebook update (EMA)
Codebook:
$$
P^d\in\mathbb{R}^{K_d\times H}
$$
Assignment logits:
$$
Z_{b,t,:}=f_{assign}(\tilde{H}_{b,t,:})\in\mathbb{R}^{K_d}
$$
Gumbel-softmax probabilities:
$$
Q_{b,t,:}=\mathrm{GumbelSoftmax}(Z_{b,t,:};\tau_g)
$$
Batch counts/sums:
$$
c_k=\sum_{b,t} Q_{b,t,k},\quad s_k=\sum_{b,t} Q_{b,t,k}\tilde{H}_{b,t,:}
$$
EMA update:
$$
N_k\leftarrow \lambda N_k+(1-\lambda)c_k,
$$
$$
S_k\leftarrow \lambda S_k+(1-\lambda)s_k
$$
Codebook row:
$$
P^{d,new}_k=\mathrm{norm}\left(\frac{S_k}{\max(N_k,\epsilon)}\right)
$$

## Stage 4: Prototype lookup để tái tạo latent branch-wise

### 4.1 Continuous lookup
Token-to-prototype logits:
$$
A_{b,t,k}=\frac{\langle \tilde{H}_{b,t},P^{c,act}_k\rangle}{\sqrt{H}}
$$
Weights theo prototype axis:
$$
\Pi_{b,t,:}=\mathrm{softmax}(A_{b,t,:})
$$
Continuous branch latent:
$$
H^c_{b,t,:}=\mathrm{norm}\left(\sum_{k=1}^{K_c} \Pi_{b,t,k}P^{c,act}_k\right)
$$

### 4.2 Discrete lookup
Nếu chưa có $$Q$$ thì tính từ $$f_{assign}$$ + gumbel-softmax. Sau đó:
$$
H^d_{b,t,:}=\mathrm{norm}\left(\sum_{k=1}^{K_d} Q_{b,t,k}P^{d,act}_k\right)
$$
Hard code index (chỉ để log/diag):
$$
\hat{k}_{b,t}=\arg\max_k Q_{b,t,k}
$$

## Stage 5: CKA similarity và fusion weights $$\alpha,\beta$$

### 5.1 Linear CKA cho từng sample
Với 2 token matrices cùng sample:
$$
U,V\in\mathbb{R}^{L\times H}
$$
Center theo feature:
$$
\bar{U}=U-\frac{1}{L}\mathbf{1}\mathbf{1}^\top U,\quad
\bar{V}=V-\frac{1}{L}\mathbf{1}\mathbf{1}^\top V
$$
Gram:
$$
K_U=\bar{U}\bar{U}^\top,\quad K_V=\bar{V}\bar{V}^\top
$$
HSIC terms:
$$
\mathrm{HSIC}_{UV}=\langle K_U,K_V\rangle_F,
\quad
\mathrm{HSIC}_{UU}=\langle K_U,K_U\rangle_F,
\quad
\mathrm{HSIC}_{VV}=\langle K_V,K_V\rangle_F
$$
CKA:
$$
\mathrm{CKA}(U,V)=\frac{\mathrm{HSIC}_{UV}}{\sqrt{\mathrm{HSIC}_{UU}\mathrm{HSIC}_{VV}+\epsilon_{cka}}}
$$

Batch score vectors:
$$
\mathbf{c}^{recon}_b=\mathrm{CKA}(H^{base}_b,H^c_b),\quad
\mathbf{c}^{cls}_b=\mathrm{CKA}(H^{pair}_b,H^d_b)
$$

### 5.2 Alpha/Beta computation
- Nếu không dùng CKA-gated fusion:
$$
\alpha=\sigma(a),\quad \beta=\sigma(b)
$$
trong đó $$a,b$$ là learnable logits toàn cục.

- Nếu dùng CKA-gated fusion:
$$
\mathbf{g}_b=[\mathbf{c}^{recon}_b,\mathbf{c}^{cls}_b]\in\mathbb{R}^2
$$
$$
\alpha_b=\sigma(f_{cls\_gate}(\mathbf{g}_b)),\quad
\beta_b=\sigma(f_{recon\_gate}(\mathbf{g}_b))
$$

### 5.3 Task-specific fusion
$$
H^{recon}_{b,t,:}=\beta_b H^d_{b,t,:} + (1-\beta_b)H^c_{b,t,:}
$$
$$
H^{cls}_{b,t,:}=\alpha_b H^d_{b,t,:} + (1-\alpha_b)H^c_{b,t,:}
$$

## Stage 6: Reconstruction head và classification head

### 6.1 Reconstruction head
MLP theo timestep:
$$
\hat{X}_{b,t,:}=f_{recon}(H^{recon}_{b,t,:})\in\mathbb{R}^{D}
$$
Toàn batch:
$$
\hat{X}\in\mathbb{R}^{B\times L\times D}
$$

Point anomaly score:
$$
s^{point}_{b,t}=\frac{1}{D}\sum_{d=1}^{D}(\hat{X}_{b,t,d}-X_{b,t,d})^2
$$
Window score:
$$
s^{win}_{b}=\frac{1}{L}\sum_{t=1}^{L}s^{point}_{b,t}
$$

### 6.2 Classification head
Flatten window representation:
$$
\tilde{h}^{cls}_b=\mathrm{vec}(H^{cls}_{b,:,:})\in\mathbb{R}^{L\cdot H}
$$
Logits:
$$
z_b=f_{cls}(\tilde{h}^{cls}_b)\in\mathbb{R}^{C}
$$
Probabilities:
$$
p_b=\mathrm{softmax}(z_b)
$$

## Stage 7: Loss functions

### 7.1 Reconstruction loss
Full MSE:
$$
\mathcal{L}_{recon}=\frac{1}{BLD}\sum_{b,t,d}(\hat{X}_{b,t,d}-X_{b,t,d})^2
$$
Nếu `reconstruction_normal_only=True`, dùng mask normal:
$$
\mathcal{N}=\{(b,t)\mid M_{b,t}=0\}
$$
$$
\mathcal{L}_{recon}^{normal}=\frac{1}{|\mathcal{N}|D}\sum_{(b,t)\in\mathcal{N}}\sum_{d=1}^{D}(\hat{X}_{b,t,d}-X_{b,t,d})^2
$$
Nếu $$|\mathcal{N}|=0$$ fallback về full MSE.

### 7.2 Classification loss
- Hard label CE:
$$
\mathcal{L}_{cls}=\frac{1}{B}\sum_{b=1}^{B}\mathrm{CE}(z_b,y^{cls}_b)
$$
- Nếu label refurbishment bật, thay hard label bằng phân phối mục tiêu $$q_b$$ rồi:
$$
\mathcal{L}_{cls}=\frac{1}{B}\sum_{b=1}^{B}\left(-\sum_{c=1}^{C}q_{b,c}\log p_{b,c}\right)
$$

### 7.3 Optional losses
Cross-branch diversity:
$$
\mathcal{L}_{div}=\frac{1}{H^2}\left\|\frac{\bar{H}^{c\top}\bar{H}^d}{N}\right\|_F^2
$$
(với $$N=B\cdot L$$ token sau chuẩn hóa nội bộ).

Variance floor:
$$
\mathcal{L}_{var}=\sum_{r\in\{c,d\}}\frac{1}{H}\sum_{j=1}^{H}\left[\max(0,\gamma-\mathrm{Std}(H^r_{:,j}))\right]^2
$$

Covariance reduction:
$$
\mathcal{L}_{cov}=\sum_{r\in\{c,d\}}\frac{1}{H(H-1)}\sum_{i\neq j}(\Sigma^r_{ij})^2
$$

Prototype usage (discrete):
$$
\bar{u}_k=\frac{1}{BL}\sum_{b,t}Q_{b,t,k},\quad u_k^*=\frac{1}{K_d}
$$
$$
\mathcal{L}_{use}=\sum_{k=1}^{K_d}(\bar{u}_k-u_k^*)^2
$$

Gate regularization (entropy penalty):
$$
H(x)=-x\log x-(1-x)\log(1-x)
$$
$$
\mathcal{L}_{gate}=\frac{1}{2B}\sum_{b=1}^{B}
\left[
1-\frac{H(\alpha_b)}{\log 2}
+
1-\frac{H(\beta_b)}{\log 2}
\right]
$$

### 7.4 Tổng loss offline pre-training
Loss chính (trước contrastive):
$$
\mathcal{L}_{main}=\mathcal{L}_{recon}+\lambda_{cls}\mathcal{L}_{cls}
+\lambda_{div}\mathcal{L}_{div}
+\lambda_{var}\mathcal{L}_{var}
+\lambda_{cov}\mathcal{L}_{cov}
+\lambda_{use}(e)\mathcal{L}_{use}
+\lambda_{gate}\mathcal{L}_{gate}
$$
trong đó $$\lambda_{use}(e)$$ là lịch theo epoch.

Loss cuối cùng tại `_shared_step`:
$$
\mathcal{L}_{total}=\mathcal{L}_{main}+\lambda_{con}\mathcal{L}_{con}
$$

## Thứ tự thực thi quan trọng (theo code thực tế)
1. Tạo clean/aug pair nếu dùng two-view contrastive.
2. Tính contrastive loss từ encoder(clean) và encoder(aug).
3. Chạy `forward(prepared_batch)`.
4. Trong `forward`, cập nhật memory trước lookup.
5. Lookup continuous/discrete để tái tạo latent branch-wise.
6. Fusion (có thể có CKA-gating) để lấy hidden cho từng task.
7. Chạy reconstruction/classification heads.
8. Tính reconstruction, classification, optional losses.
9. Cộng tổng loss và cộng thêm contrastive term.

## Code references (line-level)
- `src/models/thesis_multitask.py:95-124` (encoder I/O)
- `src/models/thesis_multitask.py:1095-1228` (memory update: continuous + discrete EMA)
- `src/models/thesis_multitask.py:1235-1331` (prototype lookup)
- `src/models/thesis_multitask.py:1333-1427` (fusion, alpha/beta, CKA-gated path)
- `src/models/thesis_multitask.py:1429-1460` (linear CKA)
- `src/models/thesis_multitask.py:1462-1485` (two-view contrastive loss)
- `src/models/thesis_multitask.py:1570-1583` (prepare 2-view pair)
- `src/models/thesis_multitask.py:1585-1759` (forward full path)
- `src/models/thesis_multitask.py:1774-1855` (reconstruction/classification loss)
- `src/models/thesis_multitask.py:1923-2011` (optional losses)
- `src/models/thesis_multitask.py:2027-2044` (main weighted loss)
- `src/models/thesis_multitask.py:2206-2253` (shared_step assembly + final contrastive addition)
- `src/data/augment.py:111-122` (classification label mapping)
- `src/data/augment.py:728-802` (augment_batch: tạo mask + labels + metadata)

## Ghi chú ràng buộc theo code hiện tại
- CKA classification branch chỉ meaningful khi có `paired_hidden_for_fusion` (được gán ở path two-view trong `_shared_step`).
- Khi memory chưa initialized hoặc bootstrap active, lookup có thể bypass memory và dùng hidden normalized trực tiếp.
- `anomaly_token_mask` cho discrete memory update chỉ được tạo khi có `synthetic_anomaly_mask` và `enable_two_view_contrastive=True`.
