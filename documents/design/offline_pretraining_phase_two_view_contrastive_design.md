# Offline Pre-Training Phase Two-View Contrastive + CKA-Gated Fusion Design (Window Size 20)

## 1. Scope and Terminology

This document specifies the offline pre-training phase design for two-view learning with contrastive objective and CKA-gated per-sample fusion.

Terminology is fixed:

- `offline pre-training phase`: offline training before deployment.
- `online adaptation phase`: adaptation after deployment during streaming.
- `computational stage`: a sub-step inside a phase.

This document only covers the offline pre-training phase. The online adaptation phase has its own contrastive logic and remains separate.

## 2. Confirmed Experimental Plan

### Experiment 1 (Quick-Win)

- Keep current computation graph unchanged.
- Keep current loss surface unchanged.
- Disable bootstrapping only:

$$
\texttt{bootstrap\_encoder\_epochs} = 0
$$

### Experiment 2 (New Design)

- Also disable bootstrapping:

$$
\texttt{bootstrap\_encoder\_epochs} = 0
$$

- Add two-view InfoNCE in offline pre-training phase.
- Add CKA-gated per-sample fusion with two separate gating MLPs.
- Use:

$$
\tau_c = \texttt{contrastive\_temperature} = 0.1
$$

without tuning in the first integration run.

## 3. Window and Batch Contract

Active window length:

$$
L = 20
$$

Per train or synthetic-validation batch:

- normal view window: $$x$$
- injected anomalous view window: $$x'$$
- synthetic anomaly mask: $$M \in \{0,1\}^{B \times L}$$

with:

$$
M_{b,t}=1 \iff t \in A_b,
\qquad
A_b^c = \{0,\dots,L-1\}\setminus A_b
$$

## 4. Representation and Branch Routing

Shared encoder:

$$
H = f_\theta(x),
\qquad
H' = f_\theta(x')
$$

$$
H, H' \in \mathbb{R}^{B \times L \times d_h}
$$

Per sample $$b$$ and timestep $$t$$:

$$
h_{b,t}, h'_{b,t} \in \mathbb{R}^{d_h}
$$

Hard routing in offline pre-training phase:

- $$h$$ queries only the continuous branch.
- $$h'$$ queries only the discrete branch.

Denote branch outputs:

$$
\hat{H}_c \in \mathbb{R}^{B \times L \times d_h}
\quad\text{(continuous output from }h\text{)}
$$

$$
\hat{H}'_d \in \mathbb{R}^{B \times L \times d_h}
\quad\text{(discrete output from }h'\text{)}
$$

## 5. InfoNCE in Offline Pre-Training Phase

For each task-specific encoder, we take two views:

$$
z_{\text{normal}} = f_\theta(x_{\text{normal}}),
\qquad
z_{\text{synth}} = f_\theta(x_{\text{synth}})
$$

with:

$$
z_{\text{normal}}, z_{\text{synth}} \in \mathbb{R}^{B \times L \times d_h}
$$

For a batch element $b$ and a timestep $t$, define the time-step embeddings:

$$
q_{b,t} = \operatorname{norm}\!\left(z_{\text{normal}, b,t}\right),
\qquad
k^+_{b,t} = \operatorname{norm}\!\left(z_{\text{synth}, b,t}\right)
$$

Positive anchors use non-injected timesteps only:

$$
\mathcal{I}_+ = \{(b,t)\mid M_{b,t}=0\}
$$

For each anchor:

$$
q_{b,t}=\operatorname{norm}(h_{b,t}),
\qquad
k^+_{b,t}=\operatorname{norm}(h'_{b,t})
$$

All other in-batch tokens are candidate negatives:

$$
\mathcal{N}_{b,t}=\mathcal{C}\setminus\{(b,t)\},
\qquad
\mathcal{C}=\{(b',t')\ \forall b',t'\}
$$

Tokens at $$t\in A$$ do not form positives and participate only as negatives.

InfoNCE:

$$
\mathcal{L}_{\text{ctr}}
=
-\frac{1}{|\mathcal{I}_+|}
\sum_{(b,t)\in\mathcal{I}_+}
\log
\frac{
\exp\left(\operatorname{sim}(q_{b,t},k^+_{b,t})/\tau_c\right)
}{
\exp\left(\operatorname{sim}(q_{b,t},k^+_{b,t})/\tau_c\right)
+
\sum_{(b',t')\in\mathcal{N}_{b,t}}
\exp\left(\operatorname{sim}(q_{b,t},\tilde{k}_{b',t'})/\tau_c\right)
}
$$

where:

$$
\tilde{k}_{b',t'}=\operatorname{norm}(z_{\text{synth},b',t'})
$$

and:

$$
\operatorname{sim}(u,v)=u^\top v
$$

The contrastive loss is then combined with the task loss:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{task}}
+ \lambda_{\text{ctr}} \mathcal{L}_{\text{ctr}}
$$

where $\mathcal{L}_{\text{task}}$ is the reconstruction or classification objective for that task-specific encoder.

The backward pass follows:

$$
\frac{\partial \mathcal{L}_{\text{ctr}}}{\partial z_{\text{normal}}}
\neq 0,
\qquad
\frac{\partial \mathcal{L}_{\text{ctr}}}{\partial z_{\text{synth}}}
\neq 0
$$

so the contrastive gradient flows into the shared task-specific encoder parameters:

$$
\frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta}
=
\frac{\partial \mathcal{L}_{\text{task}}}{\partial \theta}
+
\lambda_{\text{ctr}}
\frac{\partial \mathcal{L}_{\text{ctr}}}{\partial \theta}
$$

and the optimizer update is:

$$
\theta \leftarrow
\theta - \eta \frac{\partial \mathcal{L}_{\text{total}}}{\partial \theta}
$$

## 6. CKA-Gated Per-Sample Fusion

### 6.1 Linear CKA with Time-Axis Centering

For any sample-level matrices $$X, Y \in \mathbb{R}^{L \times d_h}$$:

$$
\tilde{X}=JX,
\qquad
\tilde{Y}=JY,
\qquad
J=I_L-\frac{1}{L}\mathbf{1}\mathbf{1}^\top
$$

Linear CKA:

$$
\operatorname{CKA}(X,Y)=
\frac{\left\|\tilde{X}^\top\tilde{Y}\right\|_F^2}
{\left\|\tilde{X}^\top\tilde{X}\right\|_F\cdot\left\|\tilde{Y}^\top\tilde{Y}\right\|_F}
$$

### 6.2 Two CKA Scalars per Sample

Per sample $$b$$:

$$
s_b^{\text{rec}} = \operatorname{CKA}(H_b,\hat{H}_{c,b})
$$

$$
s_b^{\text{cls}} = \operatorname{CKA}(H'_b,\hat{H}'_{d,b})
$$

Gate feature vector:

$$
u_b = [s_b^{\text{rec}},\ s_b^{\text{cls}}] \in \mathbb{R}^{2}
$$

### 6.3 Two Separate Gating MLPs

$$
\alpha_b=\sigma(\mathrm{MLP}_{\text{cls}}(u_b)),
\qquad
\beta_b=\sigma(\mathrm{MLP}_{\text{rec}}(u_b))
$$

Definitions:

- $$\alpha_b$$: classification fusion weight on discrete branch.
- $$\beta_b$$: reconstruction fusion weight on discrete branch.

### 6.4 Per-Head Fusion (Per Sample)

Broadcast $$\alpha_b$$ and $$\beta_b$$ over time and hidden dimensions:

$$
\alpha_b,\beta_b \in (0,1),
\qquad
\alpha_b,\beta_b \to \mathbb{R}^{L \times d_h}\ \text{by broadcast}
$$

Classification fusion:

$$
H^{\text{cls}}_b = \alpha_b\,\hat{H}'_{d,b} + (1-\alpha_b)\,\hat{H}_{c,b}
$$

Reconstruction fusion:

$$
H^{\text{rec}}_b = \beta_b\,\hat{H}'_{d,b} + (1-\beta_b)\,\hat{H}_{c,b}
$$

Then:

- classification head consumes $$H^{\text{cls}}_b$$
- reconstruction head consumes $$H^{\text{rec}}_b$$

## 7. Train Computational Stages and Memory Update Policy

For each train step in Experiment 2:

1. Build pair $$x, x'$$ and mask $$M$$.
2. Encode to $$H, H'$$.
3. Query branches with hard routing: $$h \to \text{continuous}$$, $$h' \to \text{discrete}$$.
4. Compute $$\mathcal{L}_{\text{ctr}}$$.
5. Compute CKA features, then $$\alpha_b,\beta_b$$, then fused per-head representations.
6. Compute task losses and total objective.
7. Backpropagation and optimizer update.

Memory update policy is token-level and train-only:

- Continuous memory update uses only tokens with $$M_{b,t}=0$$.
- Discrete memory update uses only tokens with $$M_{b,t}=1$$.
- Validation/test keep memory read-only.

## 8. Total Objective

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{existing}} + \lambda_{\text{ctr}}\mathcal{L}_{\text{ctr}}
$$

Default first-run value:

$$
\lambda_{\text{ctr}} = 1.0
$$

## 9. Experiment Protocol v2

### Exp1: No-Bootstrap Quick-Win

- Keep current model/loss behavior.
- Set:

$$
\texttt{bootstrap\_encoder\_epochs}=0
$$

### Exp2: No-Bootstrap + Two-View InfoNCE + CKA-Gated Fusion

- Add two-view InfoNCE and CKA-gated per-sample fusion.
- Compute contrastive loss on:
  - train pairs
  - synthetic validation pairs (original window + injected window)

Monitor/checkpoint defaults for Exp2:

- scheduler monitor: `val_synth_vus_pr`
- checkpoint monitor: `val_synth_vus_pr`

## 10. Design-Implementation Mapping

Primary implementation file:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py`

Primary model config family:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/model/thesis_multitask_redlamp_multiclass.yaml`

Primary experiment config family:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/experiment/`
