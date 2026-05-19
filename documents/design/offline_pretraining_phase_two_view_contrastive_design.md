# Offline Pre-Training Phase Two-View Contrastive Design (Window Size 20)

## 1. Scope and Terminology

This document defines a new contrastive-learning design inside the **offline pre-training phase**.

Terminology must remain consistent:

- `offline pre-training phase`: offline training phase before deployment.
- `online adaptation phase`: online adaptation phase during streaming/inference-time adaptation.
- `computational stage`: a smaller computational step inside a phase.

This design is only for the offline pre-training phase. Contrastive logic in the online adaptation phase remains a separate mechanism.

## 2. Confirmed Experimental Plan

### Experiment 1: Quick-Win (Run First)

- Keep the current computational flow unchanged.
- Keep the existing loss surface unchanged: reconstruction, classification, and optional regularizers controlled by current configs.
- Disable bootstrapping only:

$$
\texttt{bootstrap\_encoder\_epochs} = 0
$$

### Experiment 2: New Two-View Contrastive Design

- Also disable bootstrapping:

$$
\texttt{bootstrap\_encoder\_epochs} = 0
$$

- Add two-view contrastive learning in the offline pre-training phase.
- Use InfoNCE.
- Add explicit config parameter:

$$
\tau_c = \texttt{contrastive\_temperature} = 0.1
$$

- First run policy: keep this value fixed, no tuning.

## 3. Window Length Contract

The active input window size for this design is:

$$
L = 20
$$

All design references in this document use 20 time-steps.

## 4. Offline Two-View Batch Contract (Model-Side Construction)

This design uses model-side two-view construction (current preferred path for maintainability and quick integration).

For each train batch:

- `x_normal`: original normal window batch.
- `x_anomalous`: synthetic anomalous version created from `x_normal` by model-side injector.
- `synthetic_anomaly_mask`: timestep-level mask indicating injected positions.

Define:

$$
M \in \{0,1\}^{B \times L}
$$

with:

$$
M_{b,t}=1 \iff t \in A_b
$$

where:

- $$A_b$$ is the injected-position set for sample $$b$$.
- $$A_b^c$$ is the complement (non-injected positions).

## 5. Shared Encoder and Hidden Representations

Two views pass through the same encoder:

$$
H^{(n)} = f_\theta(x_{\text{normal}}), \quad
H^{(a)} = f_\theta(x_{\text{anomalous}})
$$

with shape:

$$
H^{(n)}, H^{(a)} \in \mathbb{R}^{B \times L \times d_h}
$$

Per-token vectors:

$$
h^{(n)}_{b,t},\ h^{(a)}_{b,t} \in \mathbb{R}^{d_h}
$$

## 6. InfoNCE Design for Offline Pre-Training

### 6.1 Positive-anchor index set

Use only non-injected positions as positive anchors:

$$
\mathcal{I}_+ = \{(b,t)\mid M_{b,t}=0\}
$$

For each anchor $$ (b,t) \in \mathcal{I}_+ $$:

$$
q_{b,t} = \operatorname{norm}(h^{(n)}_{b,t}), \quad
k^+_{b,t} = \operatorname{norm}(h^{(a)}_{b,t})
$$

### 6.2 Negative candidates

Use in-batch negatives from all other tokens except the positive itself.

$$
\mathcal{C} = \{(b',t')\ \forall b',t'\}
$$

$$
\mathcal{N}_{b,t}=\mathcal{C}\setminus\{(b,t)\}
$$

Important rule:

- Tokens at injected positions $$t\in A$$ do not form positives.
- They can appear in the negative pool.

### 6.3 Similarity and InfoNCE objective

Similarity:

$$
\operatorname{sim}(u,v)=u^\top v
$$

With normalized vectors, this equals cosine similarity.

InfoNCE loss:

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
\tilde{k}_{b',t'}=\operatorname{norm}(h^{(a)}_{b',t'})
$$

and:

$$
\tau_c = 0.1
$$

## 7. Total Objective for Experiment 2

Keep existing objective and add contrastive term:

$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{existing}} + \lambda_{\text{ctr}}\mathcal{L}_{\text{ctr}}
$$

Initial default for first integration run:

$$
\lambda_{\text{ctr}} = 1.0
$$

Policy: expose via config, but do not tune in first run.

## 8. Memory Updating Mechanism Integration

The new contrastive component must coexist with current memory updates. Keep current mechanism unchanged:

- continuous memory update gate
- discrete codebook EMA update
- updates only in train computational stage
- validation and test are read-only for memory

This avoids introducing unnecessary codepath divergence and keeps behavior aligned with maintainability goals.

## 9. Training Computational Stages (Experiment 2)

For each train step:

1. Build `x_normal`, `x_anomalous`, and `synthetic_anomaly_mask`.
2. Run shared encoder on two views to obtain $$H^{(n)}$$ and $$H^{(a)}$$.
3. Compute $$\mathcal{L}_{\text{ctr}}$$ from mask-based anchor selection.
4. Run existing prototype/fusion/head flow and compute existing losses.
5. Combine losses and backpropagate.

## 10. Design-Implementation Mapping (Files)

Primary implementation context:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py`

Likely config surfaces:

- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/model/thesis_multitask_redlamp_multiclass.yaml`
- experiment configs under `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/experiment/`

## 11. Clarification About Checkpoint Monitoring Metric

When an experiment config sets:

$$
\texttt{checkpoint\_monitor\_metric} = \texttt{val\_vus\_pr}
$$

it means the best checkpoint is selected by `val_vus_pr`, not by `val_synth_vus_pr`.

