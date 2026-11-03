# Experiment flows and equations

This note follows the implemented code from input files to reported tables. It covers the shared SMD path, MUSTARD offline training and evaluation, MUSTARD online test-time adaptation, other configured methods, metrics, and table aggregation. Equations describe code behavior. A note marked **configuration-dependent** means the code has a branch controlled by YAML.

Naming rule: `THESIS` identifies this repository's runtime and code-facing
names. `MUSTARD` is the scientific method name used in explanatory prose. When
both are relevant, write “the MUSTARD method in the THESIS runtime.” The
`O0/O1/O2` and `A0/A1/A2` ablations belong to MUSTARD only.

“Online test-time adaptation” names the MUSTARD method family; A0 is its
no-update control. “Online phase” names the shared execution stage and may
also include methods that remain frozen.

“Experiment” is the term used here for a run or group of runs. A few source file names still contain an older term; those are cited as file names only.

## Retrieval wording and review (2026-11-03)

The proposed comparison uses **no retrieval / dense retrieval / sparse retrieval**. No retrieval passes encoder features directly to the task head. Dense retrieval corresponds to the **continuous attentional prototype bank**; sparse retrieval corresponds to the **discrete attentional codebook**. These describe retrieval behavior, not the numerical type or class contents of stored vectors. Weighted top-k output can remain continuous; hard top-1 is one special case. Use “discrete attentional codebook,” not “anomaly codebook.”

For a review, trace the selected configuration through support selection, weighting, normalization, temperature, any Gumbel hard/soft setting, pooling, and task head. Record initialization pools, supervision, and routing separately. Compare actual source with the equations; keep specified behavior, observed runtime behavior, and verified parity distinct. This note's implementation descriptions are not proof that another library implements the same operators.

Use the [review checklist in the simplification lessons](../logs/2026-11-01/detail/simplification-lessons.md#retrieval-wording-and-review-requirements-2026-11-03). The no/dense/sparse classification study remains proposed and does not replace the existing method protocol or ablation matrix. Existing runtime/config/checkpoint/artifact identifiers and historical equations are unchanged.

## 1. Run identity and execution order

The example YAML set fixes several branch choices: input width $D=38$, hidden width $h=32$, 32 continuous prototypes, a 60-vector discrete codebook, 12 classes, 10 Monte Carlo samples, and top-3 discrete retrieval. Synthetic augmentation, synthetic validation, class-balanced training, normal-only reconstruction, and two-view contrastive learning are enabled. Fusion is task-specific concatenation and projection in this model config. The configured weights are $\lambda_{rec}=\lambda_{cls}=0.5$ and $\lambda_{con}=0.3$; diversity, variance, covariance, usage, and gate weights are zero. The data config uses batch size 512, train stride 1, validation/test stride 20. The task config supplies synthetic seeds 7, segment fractions 0.2–0.3, spike scale 3, and visibility boost 1.5.

One run is identified by its method, entity, seed, phase, and configuration. The active single-entity offline example sets SMD entity `machine_1_6`, seed (6), window length (L=20), and 30 training epochs. Its two training stages use 25 and 5 epochs.

\[
\mathcal{R}=(\text{dataset},\text{entity},\text{seed},\text{method},\text{phase},\text{config})
\]

\[
\text{config files}
\to \text{parse splits}
\to \text{validate and scale}
\to \text{make windows and batches}
\to \text{train or adapt}
\to \text{score}
\to \text{map scores to points}
\to \text{compute metrics}
\to \text{aggregate report tables}.
\]

For offline MUSTARD, the wrapper validates config and protocol, materializes a two-stage plan, runs it, loads the selected checkpoint and scaler, evaluates clean validation, synthetic validation, and test data, then writes score artifacts and a JSON report. MUSTARD online test-time adaptation loads the offline Stage B checkpoint, calibrates or reads thresholds, processes a causal test stream, and writes one record per stream window. The wrapper functions are in `scripts/benchmarks/run_thesis_offline_benchmark.py` and `scripts/benchmarks/run_thesis_online_benchmark.py`; orchestration lives in `scripts/experiments/run_two_stage_offline_pretraining.py` and `src/engine/online_tta/`.

## 2. Data preprocessing

### 2.1 SMD files and train/validation/test split

For entity (e), the parser reads feature arrays and test labels:

\[
X^{\mathrm{raw}}_{e,\mathrm{train}}\in\mathbb{R}^{T_{tr}\times D},\qquad
X^{\mathrm{raw}}_{e,\mathrm{test}}\in\mathbb{R}^{T_{te}\times D},\qquad
y^{\mathrm{raw}}_{e,\mathrm{test}}\in\{0,1\}^{T_{te}}.
\]

The test label length must equal the test feature length. The validation tail length and cutoff are:

\[
T_{val}=\max\left(1,\left\lfloor rT_{tr}\right\rfloor\right),\qquad
t_{cut}=T_{tr}-T_{val},
\]

\[
X_{e,\mathrm{train}}=X^{\mathrm{raw}}_{e,\mathrm{train}}[0:t_{cut}],\qquad
X_{e,\mathrm{val}}=X^{\mathrm{raw}}_{e,\mathrm{train}}[t_{cut}:T_{tr}].
\]

The validation labels are zeros because this tail comes from the SMD training file. The test labels come from `test_label`. The parser checks the three files by entity name and, when no entity list is supplied, expects 28 files in each split. See `src/data/datasets/smd.py:15-183`.

### 2.2 Cleaning and feature scaling

The cleaning stage validates sequence shapes and metadata.
It does not fill missing values, clip outliers, or remove rows. In symbols, it returns the same data values after contract checks:

\[
\widetilde X_{e,s}=X_{e,s}\quad\text{if sequence and metadata checks pass}.
\]

The scaler is fitted on concatenated training points only. For feature (d):

\[
\mu_d=\frac{1}{N_{tr}}\sum_{e}\sum_{t\in\mathrm{train}(e)}x_{e,t,d},
\]

\[
\sigma_d=\sqrt{\frac{1}{N_{tr}}\sum_{e}\sum_{t\in\mathrm{train}(e)}(x_{e,t,d}-\mu_d)^2},
\qquad a_d=\mathbb{1}[\sigma_d>0],
\]

\[
\widehat x_{t,d}=\begin{cases}
(x_{t,d}-\mu_d)/\max(\sigma_d,10^{-3}),&a_d=1,\\
x_{t,d},&a_d=0.
\end{cases}
\]

Validation and test use the same training statistics. A constant feature is passed through unchanged. The inverse map for active features is

\[
\widetilde x_{t,d}=\widehat x_{t,d}\max(\sigma_d,10^{-3})+\mu_d.
\]

Code: `src/data/cleaning.py:1-48`, `src/data/scalers.py:10-108`, and `src/data/loaders.py:150-194`.

### 2.3 Window creation and batches

For sequence length $T$, window length $L$, and stride $s$, the loader creates windows at

\[
u_j=js,\quad j=0,1,\ldots,\left\lfloor\frac{T-L}{s}\right\rfloor,
\qquad W_j=\widehat X[u_j:u_j+L].
\]

Each window carries entity, split, and absolute start/end indices. A batch has $B$ windows:

\[
X_b\in\mathbb{R}^{B\times L\times D},\quad
Y_b\in\{0,1\}^{B\times L},\quad
\text{meta}_b=(e_b,u_b,u_b+L).
\]

Training may shuffle windows; validation and test loaders do not. The per-split stride can override the shared stride. The active loader builds indices and slices each window on demand. Code: `src/data/loaders.py:66-148,231-299` and `src/data/window.py:17-84`.

For test scoring, the evaluator also supports end-aligned non-overlapping windows. Let $U$ be regular starts $0,L,2L,\ldots$ that fit. If the last regular window does not end at $T$, append $T-L$:

\[
U'=U\cup\{T-L\}\quad\text{when }\max(U)+L<T.
\]

This produces one final overlap when needed so the tail is covered. This policy is used by point-score protocol helpers; the ordinary `WindowDataset` above uses configured stride. Code: `src/protocols/point_scores.py:8-29`.

## 3. MUSTARD offline training data

### 3.1 Synthetic class assignment

The multiclass label set is normal $0$ plus 11 anomaly families:

\[
\mathcal C=\{0,1,\ldots,11\}
=\{\text{normal},\text{spike},\text{flip},\text{speedup},\text{noise},\text{cutoff},
\text{average},\text{scale},\text{wander},\text{contextual},\text{upsidedown},\text{mixture}\}.
\]

With balanced training enabled, each minibatch of size $B$ gets nearly equal class quotas. Write $B=12q+r$, $0\le r<12$. Each class receives $q$ windows, and $r$ classes receive one extra. The extra-class starting position rotates between calls; labels are then randomly permuted:

\[
n_c=q+\mathbb{1}[c\in\mathcal E_r],\qquad \sum_{c=0}^{11}n_c=B,
\qquad Y^{class}_{b}=\operatorname{permute}(\underbrace{0,\ldots,0}_{n_0},\ldots,\underbrace{11,\ldots,11}_{n_{11}}).
\]

When balancing is disabled, a window is anomalous with Bernoulli probability $p_a$; if anomalous, its family is sampled from the configured list. In balanced mode, $p_a$ does not set the class proportions. Code: `src/data/augment.py:803-883`.

For an anomalous window, a contiguous segment is selected. With configured fractions $f_{min},f_{max}$:

\[
\ell_{min}=\max(1,\lfloor Lf_{min}\rfloor),\quad
\ell_{max}=\min(L,\max(\ell_{min},\lfloor Lf_{max}\rfloor)),
\]

\[
\ell\sim\operatorname{IntegerUniform}[\ell_{min},\ell_{max}],\quad
u\sim\operatorname{IntegerUniform}[0,L-\ell],\quad v=u+\ell.
\]

The implementation also samples affected channels. For $D$ channels, the selected count is uniformly sampled between $\max(1,\lfloor D/10\rfloor)$ and $\max(\text{minimum},\lfloor D/2\rfloor)$. The selected channel indices are sampled without replacement. A mask marks changed points; the window class label names the family. Code: `src/data/augment.py:197-282,746-819,884-979`.

### 3.2 The 11 implemented anomaly transformations

Let $x_{u:v,c}$ denote a selected segment in channel $c$, $g_c$ its transformed value, and $b$ the visibility multiplier. Where the transform uses visibility boosting, the stored segment is

\[
g^{(b)}=x_{u:v,c}+b(g_c-x_{u:v,c}).
\]

The default $b=1.5$. The exact sampled parameters are recorded in `augmentation_metadata`. Transformations are implemented in `src/data/augment.py:319-744`.

1. **Spike.** Select $m$ positions $P$, where $1\le m\le\min(\ell,m_{max})$. For $i\in P$:

\[
g_i=x_i+\epsilon_i s_x\,s_{spike}\,q_{spike},\quad
\epsilon_i\sim\mathcal N(0,1),\quad
s_x=\max\left(\max_i|x_i-\bar x|,\frac1\ell\sum_i|x_i|,0.1\right).
\]

Other segment points remain $x_i$. Only spike positions receive mask 1.

2. **Flip.** Reverse the segment order:

\[
g_i=x_{u+v-1-i}\quad (i=u,\ldots,v-1).
\]

3. **Speedup.** Sample $r\in\{1.5,2,3\}$ by default. For local index $i$, resample at clamped source position $p_i=\min(i r,\ell-1)$ using linear interpolation:

\[
k_i=\lfloor p_i\rfloor,\quad h_i=\lceil p_i\rceil,\quad
\lambda_i=p_i-k_i,\quad
g_i=(1-\lambda_i)x_{u+k_i}+\lambda_i x_{u+h_i}.
\]

4. **Noise.** Add independent Gaussian noise with standard deviation $0.35s_x$:

\[
g_i=x_i+\epsilon_i(0.35s_x),\qquad \epsilon_i\sim\mathcal N(0,1).
\]

5. **Cutoff.** Sample zero or hold mode:

\[
g_i=0\quad\text{or}\quad g_i=x_u.
\]

6. **Average.** Replace each segment value with its mean:

\[
g_i=\bar x=\frac1\ell\sum_{j=u}^{v-1}x_j.
\]

7. **Scale.** Sample $r\in\{0.25,0.5,1.5,2\}$ by default and scale around the segment mean:

\[
g_i=\bar x+r(x_i-\bar x).
\]

8. **Wander.** Add a cumulative random drift. Let $\delta_i\sim\mathcal N(0,(0.15s_x)^2)$:

\[
d_i=\sum_{j=0}^{i}\delta_j-d_0,\qquad g_i=x_i+d_i.
\]

9. **Contextual.** Compute the mean $\mu_{out}$ of the same channel outside the selected segment (or the full-window mean if no outside points exist):

\[
\Delta=\mu_{out}-\bar x+0.5s_x,\qquad g_i=x_i+\Delta.
\]

10. **Upsidedown.** Reflect segment values about their segment mean:

\[
g_i=2\bar x-x_i.
\]

11. **Mixture.** Select $k\in\{2,3\}$ primitive families (or all available if fewer), apply them in sequence to the same segment and channel, and take the union of their changed-point masks:

\[
x^{(0)}=x,\quad x^{(j)}=T_{c_j}(x^{(j-1)}),\quad
g=x^{(k)},\quad m_i=\max_j m_i^{(j)}.
\]

After transformation, the synthetic mask for the multichannel window is the pointwise maximum over affected channels. If original labels exist, the training point labels are their pointwise maximum with the synthetic mask; otherwise the synthetic mask supplies labels.

## 4. MUSTARD offline model and learning

### 4.1 Encoder, branches, fusion, and heads

The encoder maps each window to one hidden vector per time point:

\[
H=f_\theta(X)\in\mathbb R^{B\times L\times h},\qquad
\bar h_b=\frac1L\sum_{t=1}^{L}H_{b,t}.
\]

The code supports MLP and simple CNN encoders, both required to preserve the window dimension. For memory lookup, hidden vectors are L2-normalized:

\[
q_{b,t}=\frac{H_{b,t}}{\max(\|H_{b,t}\|_2,\epsilon_m)}.
\]

### 4.2 Continuous memory initialization

Initialization reads up to `memory_initialization_batches` from the training loader. In clean-only mode, normal hidden tokens are used. With `memory_initialization_with_synthetic_windows=true`, the injector creates labeled synthetic windows; the continuous pool keeps only points where the synthetic mask is zero, including normal windows. Let the resulting normalized pool be $V$.

The continuous bank selects $K_c$ covering vectors. It starts with the token nearest the pool mean, then repeatedly selects the token farthest from its nearest selected vector:

\[
v_1=\arg\min_{v\in V}\|v-\bar V\|_2^2,
\quad
v_{j+1}=\arg\max_{v\in V}\min_{1\le i\le j}\|v-v_i\|_2^2.
\]

If fewer than $K_c$ candidates exist, the code repeats candidate indices to fill the bank. Code: `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:27-77,283-356`.

### 4.3 Discrete codebook initialization from classes

For codebook size $K_d$ and $C=12$ classes, class $c$ receives

\[
K_c=\left\lfloor\frac{K_d}{C}\right\rfloor+
\mathbb{1}\left[c<K_d\bmod C\right],\qquad
\sum_{c=0}^{C-1}K_c=K_d.
\]

The discrete initialization pool is grouped by synthetic window class. For an anomalous window, the current code puts every time-position hidden vector from that window into its class pool, including positions outside the injected mask. Thus it contains normal-class and synthetic-class window tokens when initialization augmentation is enabled; it is not a pool of only masked anomalous points. The continuous pool instead keeps only unmasked positions from normal-class windows:

$$
\mathcal D_c=\{H_{b,t}:y_b^{class}=c,\ 1\le t\le L\},\qquad
\mathcal C=\{H_{b,t}:y_b^{class}=0,\ m_{b,t}=0\}.
$$

The current code runs normalized-token K-means separately within each class pool, with farthest-covering initialization and 10 update iterations:

\[
a_i^{(j)}=\arg\min_{k\le K_c}\|z_i-\mu_k^{(j)}\|_2,
\qquad
\mu_k^{(j+1)}=\operatorname{normalize}\left(\frac{1}{|C_k^{(j)}|}\sum_{i:a_i^{(j)}=k}z_i\right).
\]

An empty cluster keeps its previous center. If a configured class has no tokens, the current implementation falls back to the concatenated available class tokens for that class's K-means pool. This is code behavior, not an equation from the spec. Code: `src/models/thesis_multitask_impl/thesis_multitask_state_memory_mixin.py:79-151,356-382`.

### 4.4 Runtime memory lookup

The continuous branch computes scaled cosine logits against bank vectors $c_k$, then softmax weights and a weighted context:

\[
\ell^{cont}_{b,t,k}=\frac{q_{b,t}^{\top}c_k}{\sqrt h},\qquad
\alpha_{b,t,k}=\operatorname{softmax}_k(\ell^{cont}_{b,t,k}/\tau_c),
\]

\[
C_{b,t}=\operatorname{normalize}\left(\sum_k\alpha_{b,t,k}c_k\right).
\]

For the `cosine_topk` discrete mode, codebook logits and sparse assignment weights are:

\[
\ell^{disc}_{b,t,k}=q_{b,t}^{\top}d_k,\qquad
\mathcal K_{b,t}=\operatorname{TopK}_k(\ell^{disc}_{b,t,k}),
\]

\[
\pi_{b,t,k}=\begin{cases}
\dfrac{\exp(\ell^{disc}_{b,t,k}/\tau_d)}{\sum_{j\in\mathcal K_{b,t}}\exp(\ell^{disc}_{b,t,j}/\tau_d)},&k\in\mathcal K_{b,t},\\
0,&k\notin\mathcal K_{b,t},
\end{cases}\qquad
D_{b,t}=\operatorname{normalize}\left(\sum_k\pi_{b,t,k}d_k\right).
\]

The alternate learned-assignment path applies a Gumbel-softmax assignment layer. The exact runtime uses that path only when `discrete_query_mode` is not `cosine_topk`. Code: `src/models/thesis_multitask_impl/thesis_multitask_routing_helpers.py:13-159`, `thesis_multitask_routing_geometry_helpers.py:277-359`.

With `direct_branch_routing`, reconstruction reads $C$ and classification reads $D$. Other configured fusion modes concatenate or mix the branches. The learnable scalar mix is:

\[
\alpha=\sigma(a),\quad\beta=\sigma(b),\quad
H^{cls}=\alpha D+(1-\alpha)C,\quad
H^{rec}=\beta D+(1-\beta)C.
\]

The optional CKA gate replaces the scalar values with sigmoid outputs from CKA features; direct routing sets branch outputs directly and bypasses these equations. Code: `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:361-470`.

The example model config selects task-specific concatenation and projection, so the active branch equations are:

$$
U_{b,t}=[C_{b,t};D_{b,t}],\quad
H^{rec}_{b,t}=P_{rec}(U_{b,t}),\quad
H^{cls}_{b,t}=P_{cls}(U_{b,t}).
$$

The reconstruction head returns $\widehat X$; the classification head flattens $H^{cls}$ over time and returns class logits:

\[
\widehat X=g_{rec}(H^{rec}),\qquad
z^{class}=g_{cls}(\operatorname{vec}(H^{cls}))\in\mathbb R^{B\times C},
\qquad p^{class}=\operatorname{softmax}(z^{class}).
\]

Code: `src/models/thesis_multitask_impl/thesis_multitask_routing_forward_helpers.py:131-285`.

### 4.5 Reconstruction and class losses

The default reconstruction error is elementwise squared error and its loss is the mean over batch, time, and features:

\[
e_{b,t,d}=(\widehat x_{b,t,d}-x_{b,t,d})^2,\qquad
\mathcal L_{rec}=\frac{1}{BLD}\sum_{b,t,d}e_{b,t,d}.
\]

If `reconstruction_normal_only` is enabled, only unmasked time-feature cells contribute:

\[
M_{b,t,d}=1-m_{b,t},\qquad
\mathcal L_{rec-normal}=\frac{\sum_{b,t,d}M_{b,t,d}e_{b,t,d}}{\sum_{b,t,d}M_{b,t,d}}.
\]

If there are no normal cells, the implementation falls back to the full mean. Otherwise, the classifier uses multiclass cross entropy:

\[
\mathcal L_{cls}=-\frac1B\sum_b\log\operatorname{softmax}(z^{class}_b)_{y_b}.
\]

When label refurbishment is enabled, one-hot targets are softened by configured $\alpha_r,\beta_r$; the loss is cross entropy against those probability targets. Code: `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:30-177`.

### 4.6 Optional losses and total loss

The code has optional branch and usage penalties. These terms equal zero when disabled or when the prototype path is inactive:

**Cross-branch diversity.** Standardize branch token features to $\widetilde C,\widetilde D$, then penalize squared cross-correlation:

\[
R=\frac{\widetilde C^\top\widetilde D}{N},\qquad
\mathcal L_{div}=\operatorname{mean}(R^2).
\]

**Variance floor.** For branch $B\in\{C,D\}$, feature standard deviation $s_{B,j}$:

\[
\mathcal L_{var}=\sum_{B\in\{C,D\}}\frac1h\sum_{j=1}^{h}
\max(0,\gamma-s_{B,j})^2.
\]

**Covariance reduction.** For standardized branch tokens, let $\Sigma_B=\widetilde B^\top\widetilde B/N$. Remove the diagonal:

\[
\mathcal L_{cov}=\sum_{B\in\{C,D\}}
\frac{\|\Sigma_B-\operatorname{diag}(\Sigma_B)\|_F^2}{h(h-1)}\quad(h>1).
\]

**Code usage.** If $\bar p_k$ is average discrete assignment for code $k$:

\[
\mathcal L_{usage}=\sum_{k=1}^{K_d}\left(\bar p_k-\frac1{K_d}\right)^2.
\]

**Gate regularization.** For gates $g\in\{\alpha,\beta\}$, Bernoulli entropy $H(g)=-g\log g-(1-g)\log(1-g)$:

\[
\mathcal L_{gate}=\frac12\operatorname{mean}\left[
1-\frac{H(\alpha)}{\log2}+1-\frac{H(\beta)}{\log2}\right].
\]

The weighted objective is

For the enabled two-view contrastive path, the code keeps normal tokens from both the clean and augmented views. After L2 normalization, each clean token is paired with the same position in the augmented view; all other augmented tokens in that batch form the competing candidates:

$$
z_i=\frac{h_i^{clean}}{\|h_i^{clean}\|_2},\quad
z_i^+=\frac{h_i^{aug}}{\|h_i^{aug}\|_2},\quad
\ell_{ij}=\frac{z_i^\top z_j^+}{\tau_{con}},\quad
\mathcal L_{con}=-\frac1N\sum_{i=1}^N
\log\frac{\exp(\ell_{ii})}{\sum_{j=1}^{N}\exp(\ell_{ij})}.
$$

If the selected normal-token set is empty, this term is zero. Code: `src/models/thesis_multitask_impl/thesis_multitask_routing_mixin.py:307-330\`.

\[
\mathcal L=\lambda_{rec}\mathcal L_{rec}
+\lambda_{cls}\mathcal L_{cls}
+\sum_j\lambda_j\mathcal L_j
+\lambda_{con}\mathcal L_{con},
\]

where disabled optional terms are zero; usage uses its scheduled weight. A point-score auxiliary loss can replace the classification-branch loss with the mean of class and score losses when explicitly enabled in Stage A. It is only active for point-level synthetic-mask targets and requires both normal and anomalous points:

\[
s_{b,t}=\frac1D\sum_d e_{b,t,d},\quad
\mu_0=\operatorname{mean}(s\mid m=0),\quad
\sigma_0=\max(\operatorname{std}(s\mid m=0),\epsilon),
\quad \widetilde s=(s-\mu_0)/\sigma_0,
\]

\[
\mathcal L_{score}=\tfrac12\operatorname{mean}_{m=0}\operatorname{BCEWithLogits}(\widetilde s,0)
+\tfrac12\operatorname{mean}_{m=1}\operatorname{BCEWithLogits}(\widetilde s,1),
\quad \mathcal L_{cls-branch}=\tfrac12(\mathcal L_{cls}+\mathcal L_{score}).
\]

Code: `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py:189-375`, `thesis_multitask_loss_gradient_mixin.py:13-31`, and `thesis_multitask_loss_step_mixin.py:220-360`.

### 4.7 Two-stage optimizer flow

The example config gives Stage A 25 epochs and Stage B 5 epochs. Stage A learns task paths and initializes class-stratified memories from training data. Stage B starts from Stage A, freezes encoder and memories, and finetunes fusion according to the plan. For each minibatch, the trainer executes:

\[
g_t=\nabla_\theta\mathcal L_t,\qquad
g_t^{clip}=g_t\min\left(1,\frac{c}{\|g_t\|_2}\right),\qquad
\theta_{t+1}=\operatorname{AdamW}(\theta_t,g_t^{clip};\eta,\lambda_{wd}).
\]

It clears gradients, backpropagates, optionally clips global gradient norm $c$, and steps the optimizer. The configured scheduler warms up and then follows cosine decay. Checkpoint selection takes the epoch with the best configured validation monitor; the example monitors synthetic-validation VUS-PR. Code: `src/engine/trainer.py:690-745`, `scripts/experiments/run_two_stage_offline_pretraining.py`, and the example YAML `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml`.

### Offline ablation variants O0, O1, and O2

O0, O1, and O2 are MUSTARD ablation variants. The following uses the later user decision recorded in `documents/logs/2026-09-15/research/research-smd-smoke-wet-launchers.md` and the component matrix in `documents/notes/smd-all-machines-experiment-matrix.md`. It supersedes the conflicting O0/O1 loss lists in `documents/spec/tsad-lib-development-spec.md`, §4. This design contract does not establish that every variant is executable in the reference experiment runner.

1. **O0:** Train Stage A with reconstruction and classification losses: $\mathcal L_{O0}=\lambda_{rec}\mathcal L_{rec}+\lambda_{cls}\mathcal L_{cls}$. Initialize the memories using training data, then train Stage B with encoder and memories frozen. In the next version, Stage B uses direct routing: continuous features go to reconstruction and discrete features to classification; fusion blocks are skipped.
2. **O1:** Follow O0, but in Stage A replace the classification branch term with $\mathcal L_{cls\text{-}branch}=\tfrac12(\mathcal L_{cls}+\mathcal L_{score})$, where balanced point-score loss $\mathcal L_{score}$ is defined in §4.6. It is active only when the point-level synthetic mask contains both normal and anomalous points. Stage B uses the same direct routing as O0.
3. **O2:** Add the two-view contrastive loss to O0: $\mathcal L_{O2}=\mathcal L_{O0}+\lambda_{con}\mathcal L_{con}$; disable balanced point-score loss. Stage B uses direct routing. The matrix name “point-level contrastive loss” refers to the existing two-view contrastive computation in §4.6, counted once. Thus O2 differs from O0 by the contrastive loss. The cited specification marks O2 integration as pending; do not infer successful execution from its name.

Loss weights, memory settings, and checkpoint selection come from the run configuration. Direct routing is the only routing mechanism in the next version. Existing configurations using fusion describe historical behavior, not the new contract. The defined O2 contract alone does not establish an executable O2 experiment path.

## 5. Offline scoring, thresholding, and point timeline

### 5.1 Reconstruction scores

For $M$ stochastic reconstructions, the code computes each sample's error first, then averages errors. For sample $m$:

\[
s^{(m,norm)}_{b,t}=\frac1D\sum_d(x_{b,t,d}-\widehat x^{(m)}_{b,t,d})^2,
\quad
s^{(m,raw)}_{b,t}=\frac1D\sum_d(\widetilde x_{b,t,d}-\widetilde{\widehat x}^{(m)}_{b,t,d})^2.
\]

\[
s^{norm}_{b,t}=\frac1M\sum_m s^{(m,norm)}_{b,t},\quad
s^{raw}_{b,t}=\frac1M\sum_m s^{(m,raw)}_{b,t},
\]

\[
S^{norm}_{b}=\frac1L\sum_t s^{norm}_{b,t},\qquad
S^{raw}_{b}=\frac1L\sum_t s^{raw}_{b,t}.
\]

The implementation does not compute the error of the mean reconstruction. Code: `src/protocols/reconstruction_scores.py:8-68`.

Some older model-output protocols fit a robust logistic transform on clean-validation point MSE. This is an optional score path; the example raw-input protocol uses the identity transform:

$$
c=\operatorname{median}(e^{val}),\quad
MAD=\operatorname{median}(|e^{val}-c|),\quad
\tau_{MAD}=MAD/0.6745,\quad
q_t=\sigma((e_t-c)/\tau_{MAD}).
$$

This transform is implemented in src/protocols/point_score_calibration.py. It must be paired with a threshold calibrated in the transformed score space.

### 5.2 Validation threshold

For the standard clean-validation protocol, with quantile $q$:

\[
\tau_{point}=Q_q(\{s^{val,clean}_{t}:s_t\text{ finite}\}).
\]

Some explicit protocol variants calibrate instead on finite scores of normal synthetic-validation points:

\[
\tau_{point}=Q_q(\{s^{val,synth}_{t}:y^{synth}_t=0,\ s_t\text{ finite}\}).
\]

The same selected score space and transform must be used to calibrate and evaluate. Point prediction is strict greater-than:

\[
\widehat y_t=\mathbb{1}[s_t>\tau_{point}].
\]

Window label is positive if any point in the window is labeled anomalous:

\[
y^{window}_b=\mathbb{1}\left[\sum_{t=1}^{L}y_{b,t}>0\right].
\]

Code: `src/engine/thresholding.py:35-86`, `src/data/api.py:249-255`, and `src/engine/evaluator.py:575-717`.

### 5.3 Overlapping-window aggregation

Every scored window writes its scores to the matching absolute point indices. For point $t$, with covering window set $J_t$:

\[
s_t=\frac{\sum_{j\in J_t}s_{j,t-u_j}}{|J_t|},\qquad
\operatorname{covered}(t)=\mathbb{1}[|J_t|>0].
\]

Only covered points enter metric computation. Raw full-sequence labels remain in the record for audit. Code: `src/engine/evaluator.py:121-335`.

For online stride-1 windows, a window point score is assigned to its causal endpoint. If the same absolute point receives repeated scores, the online scorer updates it by EWMA:

\[
\widetilde s^{(r)}_t=\begin{cases}
s^{(r)}_t,&\text{first score for }t,\\
w_c s^{(r)}_t+w_p\widetilde s^{(r-1)}_t,&\text{otherwise},
\end{cases}\qquad w_c+w_p=1.
\]

Code: `src/protocols/point_scores.py:59-94` and `src/engine/online_tta/`.

## 6. Point metrics

All point metrics use aligned covered arrays $(y_t,s_t)$. At threshold $\tau$,

\[
\widehat y_t=\mathbb 1[s_t>\tau],\quad
TP=\sum_t\mathbb 1[y_t=1,\widehat y_t=1],\quad
FP=\sum_t\mathbb 1[y_t=0,\widehat y_t=1],
\]

\[
TN=\sum_t\mathbb 1[y_t=0,\widehat y_t=0],\quad
FN=\sum_t\mathbb 1[y_t=1,\widehat y_t=0].
\]

\[
Precision=\frac{TP}{TP+FP},\quad Recall=\frac{TP}{TP+FN},\quad
F1=\frac{2\,Precision\,Recall}{Precision+Recall},\quad
FPR=\frac{FP}{FP+TN}.
\]

ROC-AUC is the trapezoid area under the threshold-swept $(FPR,TPR)$ curve, where $TPR=Recall$. Average precision is the step-wise area under the precision-recall curve:

$$
AP=\sum_{k}(R_k-R_{k-1})P_k.
$$

The threshold-dependent metrics above use the clean-validation threshold. ROC-AUC and average precision use the ranking of scores across all covered points and do not use that fixed threshold.

Code delegates ROC-AUC, average precision, precision, recall, F1, and safe undefined-value handling to scikit-learn, in `src/metrics/pointwise.py:679-804`. The threshold is validation-derived; threshold-free metrics use score order.

### 6.1 Affiliation F1

Binary labels and predictions are converted into contiguous event intervals. Each predicted interval is partitioned among the ground-truth events and their affiliation zones. The code computes event probability precision $P_i^{aff}$ and recall $R_i^{aff}$ by integrating their interval-based probability functions, then averages event values (ignoring undefined precision values):

\[
P_{aff}=\frac{1}{|I_P|}\sum_{i\in I_P}P_i^{aff},\qquad
R_{aff}=\frac1{|I_G|}\sum_{j\in I_G}R_j^{aff},
\]

\[
F1_{aff}=\begin{cases}
0,&P_{aff}+R_{aff}=0,\\
\dfrac{2P_{aff}R_{aff}}{P_{aff}+R_{aff}},&\text{otherwise}.
\end{cases}
\]

The event-zone construction and integral calculations are implemented directly in `src/metrics/affiliation.py:21-417`; the metric wrapper is `src/metrics/pointwise.py:654-678`. No point adjustment is applied to ordinary predictions.

### 6.2 Range-aware VUS-PR and VUS-ROC

The implementation builds a finite threshold grid from observed scores, including just-below-minimum and just-above-maximum sentinels. For each buffer size $b=0,\ldots,B_{max}$, threshold $\tau$ gives binary predictions:

\[
\widehat y_t(\tau)=\mathbb{1}[s_t>\tau].
\]

Each true anomaly interval $I=[a,c)$ can give prediction-conditioned soft labels to its left and right buffer. For point $t$ in an adjacent buffer, the code uses:

\[
r_t=\max\left(y_t,\sqrt{\max\left(0,1-\frac{d(t,I)}{b}\right)}\right),
\]

where $d(t,I)$ is the code's one-based distance from the event boundary; a buffer gets these weights only when at least one predicted positive lies in that side-buffer. For $b=0$, $r=y$.

The threshold-specific range precision/recall use:

\[
TP_r=\sum_t r_t\widehat y_t,\quad
FP_r=\sum_t(1-r_t)\widehat y_t,\quad
P_r=\begin{cases}1,&TP_r+FP_r=0,\\TP_r/(TP_r+FP_r),&\text{otherwise},\end{cases}
\]

\[
M_r=\sum_t\frac{y_t+r_t}{2},\qquad
R_r=\frac{TP_r}{M_r}\,E,\quad
E=\frac{\#\{\text{true events hit by any positive prediction inside the event}\}}{\#\{\text{true events}\}}.
\]

Range ROC rates are $TPR_r=\sum_t r_t\widehat y_t/\sum_t r_t$ and $FPR_r=\sum_t(1-r_t)\widehat y_t/\sum_t(1-r_t)$. Curves are made monotone by taking the best precision per recall or best TPR per FPR; area uses trapezoidal integration. VUS is the mean over integer buffer sizes:

\[
VUS\text{-}PR=\frac{1}{B_{max}+1}\sum_{b=0}^{B_{max}} A_{PR}(b),\qquad
VUS\text{-}ROC=\frac{1}{B_{max}+1}\sum_{b=0}^{B_{max}} A_{ROC}(b).
\]

Budgeted variants constrain the PR integration to ROC points with $FPR\le\rho$; the ROC variant integrates only up to $\rho$ and divides by $\rho$. The current budgets are $\rho\in\{0.001,0.005,0.01\}$. The implementation uses $B_{max}=20$ and 200 score thresholds in the example config. Exact helper equations live in `src/metrics/pointwise.py:145-422`; VUS loops are at `423-653`.

## 7. MUSTARD online experiment

### 7.1 Initialization and causal windows

The online run loads the offline Stage B checkpoint. The stream advances one point at a time using window starts $u_r=u_0+r$, each window covering $[u_r,u_r+L)$. The reported point is its endpoint $t_r=u_r+L-1$. No future window is used to score that point.

The calibrated operational point threshold is $\tau_p$. A1/A2 also use an input-window threshold $B_w$ and latent-window band $A_{low},A_{high}$, loaded from a validated threshold artifact. The online score record stores raw point score, EWMA point score, latent-window score, threshold, and prediction:

All four operational thresholds are calibrated by simulating the same stride-1 causal scoring path on clean validation data:

$$
\tau_p=Q_{q_p}(\{\widetilde s^{val}_t\}),\quad
B_w=Q_{q_w}(\{S^{input,val}_r\}),\quad
A_{low}=Q_{q_l}(\{S^{latent,val}_r\}),\quad
A_{high}=Q_{q_h}(\{S^{latent,val}_r\}).
$$

Here $\widetilde s^{val}$ is the point score after overlap updates by EWMA. The quantiles $q_p,q_w,q_l,q_h$ come from protocol config; the threshold artifact stores these choices and the source checkpoint identity. Code: scripts/benchmarks/run_thesis_offline_benchmark.py:841-957 and src/engine/online_tta/online_calibration.py.

\[
\widehat y_{t_r}=\mathbb{1}[\widetilde s_{t_r}>\tau_p].
\]

For the named clean-validation example protocol, $q_p=0.99$, $q_w=0.99$, $q_l=0.75$, $q_h=0.99$, and EWMA weights are $w_c=0.9$, $w_p=0.1$. It uses raw-input MSE with identity transform, end-aligned non-overlapping offline windows, stride-1 online windows, and no point adjustment.

### 7.2 Triage and buffer decisions

For A1/A2, the configured triage function combines input reconstruction score $S^{input}_r$ and latent score $S^{latent}_r$. The broad decision regions are:

\[
S^{input}_r\le B_w\Rightarrow\text{not admitted},
\]

\[
S^{input}_r>B_w,\quad S^{latent}_r\le A_{low}
\Rightarrow\text{hard old-normality candidate},
\]

\[
S^{input}_r>B_w,\quad A_{low}<S^{latent}_r\le A_{high}
\Rightarrow\text{gray-zone verification candidate},
\]

\[
S^{input}_r>B_w,\quad S^{latent}_r>A_{high}
\Rightarrow\text{strong anomaly candidate}.
\]

The exact code in `src/engine/online_tta/triage.py` should be treated as authority for boundary equality. Gray-zone windows enter a verification buffer only if their time intervals do not overlap existing entries. A verification cycle is run at its configured capacity. A0 performs no online update.

### 7.3 A1 and A2 update objectives

For an anchor $q_i$, the implementation uses the log-sum-exp of all positives and negatives minus the log-sum-exp of positives:

$$
\mathcal L_{NCE}
=\frac1N\sum_i
\left[
\log\left(
\sum_{p\in P(i)}e^{q_i^\top k_p/\tau}
+\sum_{n\in N(i)}e^{q_i^\top k_n/\tau}
\right)
-\log\sum_{p\in P(i)}e^{q_i^\top k_p/\tau}
\right].
$$

$P(i)$ includes the same-position source token and optional tokens with the same recurrent signature. $N(i)$ includes anomalous codewords and optional known-anomaly projected/source tokens. Similarity uses L2-normalized vectors. If there are no eligible anchor tokens, the code returns zero loss. Source: src/engine/online_tta/online_losses.py, lines 84-157.

For a verified normal mask $M$, A1 uses masked reconstruction MSE:

\[
\mathcal L_{A1}=\frac{\sum_{b,t,d}M_{b,t}(\widehat x_{b,t,d}-x_{b,t,d})^2}
{D\sum_{b,t}M_{b,t}}.
\]

A2 uses this same masked reconstruction term for verified-normal updates. On a hard-old-normality candidate it uses a hinge penalty that only turns on above threshold (the helper in `src/engine/online_tta/online_losses.py` defines the exact reduction):

\[
\mathcal L_{hinge}=\max(0,S^{window}-B_w)^2.
\]

A2 also uses token multi-positive InfoNCE against anomalous codewords. At high level, for query $q_i$, its positive set $P(i)$, and denominator candidate set $C(i)$:

\[
\mathcal L_{NCE}=-\frac1N\sum_i
\log\frac{\sum_{p\in P(i)}\exp(q_i^\top k_p/\tau)}
{\sum_{j\in C(i)}\exp(q_i^\top k_j/\tau)}.
\]

The actual positive and excluded-token masks are supplied by `compute_token_multi_positive_info_nce`; see `src/engine/online_tta/online_losses.py`. A2 combines the losses as:

\[
\mathcal L_{A2}=\mathcal L_{reconstruction}+0.1\mathcal L_{NCE}.
\]

Updates backpropagate through the online model, clip projector gradients, then step the online optimizer. Strong-anomaly candidates are skipped. Code: `src/engine/online_tta/online_engine_step.py:24-226` and `online_engine_window_core.py:24-219`.

### 7.4 Online metric alignment

Each causal endpoint score is joined to the SMD test label at the same absolute index. The reporting script expects exactly

\[
N_{steps}=t_{end}-t_{start}-L+1
\]

windows for the configured half-open stream interval. Labels are sliced from $t_{start}+L-1$ through $t_{end}-1$, matching causal endpoints. It computes FPR, VUS-PR, affiliation F1, and VUS-ROC with the shared point-metric implementation; point adjustment is false. Code: `scripts/ops/aggregate_online_table3_metrics.py:89-150,215-347`.

**Implementation review (2026-11-02):** For historical reference, the existing main O0/O1 experiment YAMLs select `configs/model/thesis_multitask_two_stage_window20.yaml`, which sets `fusion_mode: task_specific_concat_projection`. O1 enables point-score loss through model overrides; if the score loss is unavailable, `_resolve_score_and_classification_loss` retains the full classification loss. The code also contains direct routing; the next version uses it exclusively. No `__O2__` main experiment configuration was found in `configs/experiment/offline_benchmark/thesis`; its proposed contract is not proof of an implemented O2 path. Sources: `src/models/thesis_multitask_impl/thesis_multitask_loss_step_mixin.py:16`, `configs/model/thesis_multitask_two_stage_window20.yaml:79`, and the O0/O1 main experiment YAMLs.

### 7.4 Online ablation variants A0, A1, and A2

`A0/A1/A2` are online ablation variants of MUSTARD only. Each combines with a
MUSTARD offline ablation when its matching checkpoint is available.

**A0:** Load the selected offline Stage B checkpoint and its matching calibrated threshold artifact. For each causal window, run the frozen source model, obtain reconstruction errors, update the point-score EWMA, and record the endpoint score and $\mathbf1[\widetilde s_t>\tau_p]$. There is no triage, verification buffer, optimizer, or parameter update. O0-A0, O1-A0, and O2-A0 select different offline variants but share this online procedure, subject to the checkpoint being available.

**A1:** Load the checkpoint and thresholds; keep inherited encoders frozen and train only `online_mlp_projector`. Compute the score for each causal window before adaptation. The event record is assembled after adaptation from that previously computed score. Apply the input and latent threshold regions in §7.2. Admit only non-overlapping gray-zone windows to the verification buffer. Once the buffer reaches capacity and has new entries, the frozen-source verification adapter filters known anomaly tokens, finds recurrent signatures, and produces a potential new normal (PNN) mask $M$. For verified entries with a non-empty mask, recompute the reconstruction and minimize $\mathcal L_{A1}$ in §7.3, clipping projector gradients before the optimizer step. Empty masks cause no update. Hard-old candidates do not trigger an A1 update; strong-anomaly candidates are skipped. A1's implemented contrastive term is zero. Updates affect subsequent scoring, not the score already recorded.

This describes the audited reference runtime, which the runtime audit notes differs from an earlier matrix policy for A1. See `documents/logs/2026-09-12/research/research-thesis-runtime-flow-audit.md`, §§3–4, and `src/engine/online_tta/verification_adapter.py` for exact verification masks.

## 8. Other methods in the experiment matrix

All methods use the same entity, seed, data split, scaling, window coordinates, coverage rules, and point metric implementation where the experiment configuration routes through the shared evaluator. Method-specific fit and score rules are:

| Method | Fitting or update rule represented by its implementation | Window score sent to the shared evaluation path |
|---|---|---|
| MUSTARD | Equations in Sections 4–7; learns encoder, class head, memory and configured fusion | Reconstruction MSE, with the selected score space |
| RedLamp baseline | Trains the configured neural reconstruction/classification baseline on the synthetic-labeled windows | Point reconstruction MSE |
| Isolation Forest | Fits tree isolation model on flattened training windows; score is the library's anomaly score (higher is converted to anomaly direction in the adapter) | Per-window score, broadcast or mapped by its adapter |
| KMeansAD | Fits centroids on flattened training windows; score is distance to nearest centroid | Nearest-center distance |
| StumPy channel AB | Computes matrix-profile/AB-join discord score by channel and combines channels as configured | Channel-aggregated matrix-profile score |
| CANDI and M2N2 | Offline source checkpoint, then method-specific online adapter/update | Causal point/window score in online records |

For KMeansAD, the source-level distance is:

\[
d(x,\mathcal M)=\min_{k\in\{1,\ldots,K\}}\|x-\mu_k\|_2.
\]

The local Isolation Forest adapter defines its anomaly score by negating the estimator's decision function:

\[
s_{IF}(x)=-\operatorname{decision\_function}_{IF}(x).
\]

The adapter uses the common “larger means more anomalous” direction. For the STUMPY AB join, each query subsequence is compared with training-reference subsequences per channel, then per-channel validation calibration is applied and channel scores are combined:

\[
MP_{i,d}=\operatorname{ABProfile}(Q_{i,d},R_d),\quad
\widetilde{MP}_{i,d}=\frac{MP_{i,d}-median_d^{val}}{\max(IQR_d^{val},\epsilon)},\quad
s_i=\operatorname{aggregate}_{d}(\widetilde{MP}_{i,d}).
\]

Exact library settings and score orientation are set by `src/baselines/traditional/iforest.py`, `kmeans_ad.py`, and `stumpy_channel_ab.py`. The comparative methods are assembled by experiment configs and scripts in `scripts/benchmarks/`; the table above summarizes their code-facing mathematical contract, not a claim that each library implementation is reimplemented locally.

Traditional offline methods can standardize each flattened window row using that row's mean and sample standard deviation. They map window scores to points and average any overlap. Their clean-validation scores are centered by the median and scaled by the interquartile range:

$$
\mu_i=\operatorname{mean}(w_i),\quad
\sigma_i=\begin{cases}
\operatorname{std}_{ddof=1}(w_i),&\sigma_i\ge10^{-6}\text{ and finite},\\
1,&\text{otherwise},
\end{cases}
\quad z_i=(w_i-\mu_i)/\sigma_i,
$$

$$
c=\operatorname{median}(s^{val}),\quad
r=\max(Q_{0.75}(s^{val})-Q_{0.25}(s^{val}),10^{-8}),\quad
\widetilde s_t=(s_t-c)/r,\quad
\tau=Q_q(\{\widetilde s_t:t\text{ covered}\}).
$$

KMeansAD uses Euclidean distance to the assigned centroid:

$$
a_i=\arg\min_k\|z_i-\mu_k\|_2,\qquad
d_{KM}(z_i)=\|z_i-\mu_{a_i}\|_2.
$$

This follows src/baselines/traditional/base.py and kmeans_ad.py. Isolation Forest negates the estimator's decision function. STUMPY uses its channel-wise AB profile, with channel calibration and aggregation. CANDI and M2N2 use their own online adapters and pretrained neural checkpoints, so their normalization and update equations are method-specific. Their implementations are in src/baselines/traditional/iforest.py, src/baselines/traditional/stumpy_channel_ab.py, src/baselines/online/candi.py, and src/baselines/online/m2n2.py.

The neural online baselines also have explicit score and update equations in their adapters. CANDI's optional SANA wrappers transform the input and reconstruction as:

$$
x'=x+\operatorname{SANA}_{in}(x),\quad
r=f_{\theta}(x'),\quad
\widehat x=r-\operatorname{SANA}_{out}(r),\quad
s^{CANDI}=\operatorname{mean}_{t,d}\left[(\widehat x-x)^2\right].
$$

CANDI freezes the source backbone when these SANA modules are active. M2N2 applies its running mean-only detrender before and after the source reconstruction:

$$
\mu_{r+1}=\gamma\mu_r+(1-\gamma)\operatorname{mean}_{b,t}(x_{b,t}),\quad
x^{norm}=x-\mu_r,\quad
\widehat x^{raw}=\widehat x^{norm}+\mu_r.
$$

During adaptation, M2N2 marks time points with reconstruction error at or above the current threshold as pseudo-anomalous and averages error over all batch/time positions after masking:

$$
e_{b,t}=\frac1D\sum_d(\widehat x_{b,t,d}-x_{b,t,d})^2,\quad
\widehat m_{b,t}=\mathbb{1}[e_{b,t}\ge\tau],\quad
\mathcal L_{M2N2}=\frac1{BL}\sum_{b,t}e_{b,t}(1-\widehat m_{b,t}).
$$

This denominator is $BL$ in the code, not the count of unmasked points. The M2N2 update changes the backbone after one configured optimizer step. CANDI and M2N2 source: src/baselines/online/candi.py and src/baselines/online/m2n2.py.

## 9. Report tables and aggregation equations

### 9.1 Per-run record

For each combination $(m,e,s,v)$ of method, entity, seed, and variant, the evaluation record contains protocol identity, score and label arrays (when retention permits), coverage, selected threshold, and metrics:

\[
R_{m,e,s,v}=\left(\text{identity},\tau,\text{coverage},
ROC\text{-}AUC,PR\text{-}AUC,F1,FPR,F1_{aff},VUS\text{-}PR,VUS\text{-}ROC\right).
\]

Undefined metrics are stored as NaN or null according to the report writer; they must not be silently treated as zero.

### 9.2 Offline method-by-entity table

The offline phase table groups three seeds for each method, variant, and entity. For metric $M$:

\[
\overline M_{m,v,e}=\frac13\sum_{s\in\{6,8,36\}}M_{m,v,e,s}.
\]

The current Table 1 renderer reports $\overline{VUS\text{-}PR}$, mean affiliation F1, and mean VUS-ROC. It ranks methods separately within each entity and metric, displays the largest value in bold and second largest underlined. Code: `scripts/ops/render_offline_phase_metric_table.py:20-153`.

### 9.3 Online summary table

For a method/variant group with finite run values $M_1,\ldots,M_n$, the online report uses the arithmetic mean and population standard deviation:

\[
\bar M=\frac1n\sum_{i=1}^{n}M_i,\qquad
\sigma_{pop}=\sqrt{\frac1n\sum_{i=1}^{n}(M_i-\bar M)^2}.
\]

The entity-specific version groups by seed inside one entity. The all-entity version groups runs across entities. Null values are excluded; the count $n$ is saved next to mean and standard deviation. Code: `scripts/ops/aggregate_online_table3_metrics.py:244-345`.

The report-ready offline table has one row per method and variant, with a metric block for each entity:

| Method and variant | Entity | Mean VUS-PR | Mean affiliation F1 | Mean VUS-ROC |
|---|---|---:|---:|---:|
| MUSTARD O0 | machine-1-6 | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ |
| MUSTARD O1 | machine-1-6 | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ |
| Each baseline | each entity | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ | $\overline{M}_{6,8,36}$ |

The online summary table groups by method and variant, then reports the mean, population standard deviation, and valid run count:

| Method and variant | Entity scope | FPR mean ± SD (n) | VUS-PR mean ± SD (n) | affiliation F1 mean ± SD (n) | VUS-ROC mean ± SD (n) |
|---|---|---:|---:|---:|---:|
| MUSTARD O0-A0 | all / one entity | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ |
| MUSTARD O1-A1 | all / one entity | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ |
| Each online method | all / one entity | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ | $\bar M\pm\sigma_{pop}\ (n)$ |

### 9.4 Classification and uncertainty diagnostics

Training and synthetic-validation diagnostic tables may include accuracy, macro F1, weighted F1, observed class count, hard-prediction ratios, and row-normalized confusion matrices. For class $c$,

\[
Accuracy=\frac{\#\{i:\widehat c_i=c_i\}}{N},\qquad
F1_c=\frac{2P_cR_c}{P_c+R_c},
\]

\[
F1_{macro}=\frac1C\sum_cF1_c,\qquad
F1_{weighted}=\sum_c\frac{n_c}{N}F1_c,
\quad
Confusion^{norm}_{ij}=\frac{Confusion_{ij}}{\sum_jConfusion_{ij}}.
\]

For stochastic forward samples $s^{(1)},\ldots,s^{(M)}$, score uncertainty is reported from sample variance (population convention in tensor diagnostics):

\[
\bar s=\frac1M\sum_m s^{(m)},\qquad
Var(s)=\frac1M\sum_m(s^{(m)}-\bar s)^2.
\]

These diagnostics are supplementary; the offline table above uses the exact three ranking metrics from its renderer. Classification metrics are in `src/metrics/pointwise.py:679-726`; variance construction is in `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:74-141`.

## 10. Equation-to-code index

| Flow | Main implementation |
|---|---|
| SMD parsing and split | `src/data/datasets/smd.py` |
| Cleaning, scaling, windows, batches | `src/data/cleaning.py`, `src/data/scalers.py`, `src/data/loaders.py`, `src/data/window.py` |
| Synthetic taxonomy and transformations | `src/data/augment.py` |
| MUSTARD forward and memory routing | `src/models/thesis_multitask_impl/thesis_multitask_encoder.py`, `thesis_multitask_routing_forward_helpers.py`, `thesis_multitask_routing_helpers.py`, `thesis_multitask_routing_geometry_helpers.py` |
| Memory initialization | `src/models/thesis_multitask_impl/thesis_multitask_state_memory_init_helpers.py`, `thesis_multitask_state_memory_mixin.py` |
| Losses and optimization | `src/models/thesis_multitask_impl/thesis_multitask_loss_core_mixin.py`, `thesis_multitask_loss_step_mixin.py`, `thesis_multitask_loss_gradient_mixin.py`, `src/engine/trainer.py` |
| Score spaces, thresholds, timeline mapping | `src/protocols/reconstruction_scores.py`, `src/engine/thresholding.py`, `src/engine/evaluator.py`, `src/protocols/point_scores.py` |
| Point metrics | `src/metrics/pointwise.py`, `src/metrics/affiliation.py` |
| Online updates and reports | `src/engine/online_tta/online_engine_step.py`, `online_engine_window_core.py`, `scripts/ops/aggregate_online_table3_metrics.py` |
| Offline table aggregation | `scripts/ops/render_offline_phase_metric_table.py` |

## 11. Scope and implementation notes

1. The equations above follow code paths present in this checkout. The YAML config decides which branches are active.
2. The current default MUSTARD class taxonomy has 12 labels: one normal class and 11 synthetic anomaly classes. Class balancing applies to training batches when enabled.
3. The code has two memory-initialization branches. Synthetic-inclusive initialization uses normal points for the continuous memory pool and class-labeled hidden points for the discrete codebook pool. Clean-only initialization supplies normal points only.
4. Source files and generated reports still contain the older “benchmark” word in identifiers. This note uses “experiment” in prose.
5. Some baseline scores come from external libraries. Their local adapters define input and score direction; library internals remain owned by those libraries.

