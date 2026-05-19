# Brainstorming Notes: Contrastive Loss Function on Two Hidden Representations from Two Input Views `x` and `x'`

Date: 18/05/2026

## 1. Original Idea, Preserved

This is a deep neural network for time series analysis.

This part is actually brandstorming for a contrastive loss function on 2 hidden representations from 2 input views `x` and `x'`.

`x` is normal.

`x'` is anomalous version of `x`.

Anomalies are injected.

Input window length is 20 time-steps.

I expect continuous prototypes to store more fine-grained knowledge (than discrete ones) of normal patterns.

I expect discrete prototypes to store both normal and anomalous patterns but in a more compressed manner than continuous prototypes.

I expect discrete prototypes to be used more by the classification head.

I expect the continuous prototypes to be used more by the reconstruction head, which is expected to be only good at reconstructing NORMAL patterns, bad at anomalous ones.

---

## 2. Core Setting

### 2.1 Input Views

There are 2 input views:

\[
x \in \mathbb{R}^{L \times C}
\]

\[
x' \in \mathbb{R}^{L \times C}
\]

where:

- \(x\) is normal.
- \(x'\) is anomalous version of \(x\).
- \(L\) is the input window length.
- \(C\) is the number of input channels / variables / features per time-step.

The input window length is 20 time-steps:

\[
L = 20
\]

Therefore:

\[
x \in \mathbb{R}^{20 \times C}
\]

\[
x' \in \mathbb{R}^{20 \times C}
\]

### 2.2 Time-Step Index Set

The full set of time-step indices is:

\[
\{0,1,2,\dots,19\}
\]

because the input window length is 20 time-steps.

### 2.3 Injected Anomaly Position Set

Anomalies are injected into a subset of time-step positions:

\[
A = \{a_1, a_2, \dots, a_n\}
\]

where:

\[
a_i \in \{0,1,2,\dots,19\}
\]

and:

\[
|A| = n
\]

The complement of \(A\) is:

\[
A^c = \{0,1,2,\dots,19\} \setminus A
\]

where:

- \(A\) contains the positions where anomalies are injected.
- \(A^c\) contains the positions where anomalies are not injected.

### 2.4 Construction of `x'`

The anomalous view \(x'\) is created from \(x\) by injecting anomalies at positions in \(A\):

\[
x'_t =
\begin{cases}
\operatorname{InjectAnomaly}(x_t), & t \in A, \\
x_t, & t \notin A.
\end{cases}
\]

where:

- \(x_t \in \mathbb{R}^{C}\) is the normal input vector at time-step \(t\).
- \(x'_t \in \mathbb{R}^{C}\) is the anomalous-view input vector at time-step \(t\).
- \(\operatorname{InjectAnomaly}(\cdot)\) is the anomaly injection operation.
- If \(t \in A\), then anomaly is injected.
- If \(t \notin A\), then the time-step remains normal.

---

## 3. Encoder and Hidden Representations

### 3.1 Shared Encoder

Both input views are passed through the encoder:

\[
H = f_\theta(x)
\]

\[
H' = f_\theta(x')
\]

where:

- \(f_\theta\) is the encoder.
- \(\theta\) denotes the trainable parameters of the encoder.
- \(H\) is the hidden representation of \(x\).
- \(H'\) is the hidden representation of \(x'\).

### 3.2 Hidden Representation Shapes

The hidden representations have shape:

\[
H \in \mathbb{R}^{L \times d_{\text{model}}}
\]

\[
H' \in \mathbb{R}^{L \times d_{\text{model}}}
\]

Because \(L = 20\):

\[
H \in \mathbb{R}^{20 \times d_{\text{model}}}
\]

\[
H' \in \mathbb{R}^{20 \times d_{\text{model}}}
\]

where:

- \(d_{\text{model}}\) is the hidden dimension of the encoder output.
- Each time-step has one hidden vector.

### 3.3 Per-Time-Step Hidden Vectors

For each time-step \(t\):

\[
h_t \in \mathbb{R}^{d_{\text{model}}}
\]

\[
h'_t \in \mathbb{R}^{d_{\text{model}}}
\]

where:

- \(h_t\) is the hidden vector of normal input \(x\) at time-step \(t\).
- \(h'_t\) is the hidden vector of anomalous input \(x'\) at time-step \(t\).

The full hidden matrices can be written as:

\[
H = [h_0, h_1, \dots, h_{19}]^\top
\]

\[
H' = [h'_0, h'_1, \dots, h'_{19}]^\top
\]

---

## 4. Brainstorming: Contrastive Loss Function

This part is brainstorming for a contrastive loss function on 2 hidden representations from 2 input views \(x\) and \(x'\).

The key design is based on the injected anomaly positions:

- For \(k \notin A\), \(h_k\) and \(h'_k\) correspond to positions where anomalies are not injected.
- For \(a_i \in A\), \(h_{a_i}\) and \(h'_{a_i}\) correspond to positions where anomalies are injected.

## 4.1 Normal Positions: Pull Together

For normal positions:

\[
k \notin A
\]

The corresponding hidden representations should be close:

\[
h_k \approx h'_k
\]

This is because the input at those positions is not changed by anomaly injection:

\[
x'_k = x_k, \qquad k \notin A
\]

So the intended representation behavior is:

\[
d(h_k, h'_k) \to 0, \qquad k \notin A
\]

where:

- \(d(\cdot,\cdot)\) is a distance function between hidden vectors.
- \(h_k\) is the hidden vector from the normal input view.
- \(h'_k\) is the hidden vector from the anomalous input view.
- \(k \notin A\) means no anomaly is injected at this position.

A normal-position alignment loss can be written as:

\[
\mathcal{L}_{\text{normal-align}}
=
\frac{1}{|A^c|}
\sum_{k \notin A}
d(h_k, h'_k)
\]

where:

- \(\mathcal{L}_{\text{normal-align}}\) pulls together hidden representations at non-anomalous positions.
- \(|A^c|\) is the number of positions where anomalies are not injected.
- The summation is over \(k \notin A\).

## 4.2 Anomalous Positions: Push Apart

For anomalous positions:

\[
a_i \in A
\]

The corresponding hidden representations should be different:

\[
h_{a_i} \not\approx h'_{a_i}
\]

This is because the input at those positions is changed by anomaly injection:

\[
x'_{a_i} = \operatorname{InjectAnomaly}(x_{a_i}), \qquad a_i \in A
\]

So the intended representation behavior is:

\[
d(h_{a_i}, h'_{a_i}) \geq m, \qquad a_i \in A
\]

where:

- \(m > 0\) is a margin.
- The distance between normal and anomalous hidden vectors at injected positions should be at least \(m\).

A margin-based anomaly-separation loss can be written as:

\[
\mathcal{L}_{\text{anomaly-separate}}
=
\frac{1}{|A|}
\sum_{a_i \in A}
\max \left(0,\; m - d(h_{a_i}, h'_{a_i}) \right)
\]

where:

- \(\mathcal{L}_{\text{anomaly-separate}}\) pushes apart hidden representations at anomalous positions.
- \(|A|\) is the number of anomaly-injected positions.
- \(m\) is the separation margin.
- If \(d(h_{a_i}, h'_{a_i}) \geq m\), then this term becomes 0.
- If \(d(h_{a_i}, h'_{a_i}) < m\), then this term is positive and pushes the representations apart.

## 4.3 Combined Contrastive Loss

The contrastive loss combines normal-position alignment and anomalous-position separation:

\[
\mathcal{L}_{\text{contrast}}
=
\mathcal{L}_{\text{normal-align}}
+
\lambda
\mathcal{L}_{\text{anomaly-separate}}
\]

Expanded form:

\[
\mathcal{L}_{\text{contrast}}
=
\frac{1}{|A^c|}
\sum_{k \notin A}
d(h_k, h'_k)
+
\lambda
\frac{1}{|A|}
\sum_{a_i \in A}
\max \left(0,\; m - d(h_{a_i}, h'_{a_i}) \right)
\]

where:

- \(\mathcal{L}_{\text{contrast}}\) is the contrastive loss function.
- \(\lambda\) controls the strength of anomalous-position separation.
- \(d(h_k,h'_k)\) pulls normal positions together.
- \(\max(0, m - d(h_{a_i}, h'_{a_i}))\) pushes anomalous positions apart.

---

## 5. Distance Function Design Choice

The contrastive loss needs a distance function:

\[
d(u,v)
\]

where:

\[
u, v \in \mathbb{R}^{d_{\text{model}}}
\]

## 5.1 Squared Euclidean Distance

One possible choice is squared Euclidean distance:

\[
d(u,v) = \|u-v\|_2^2
\]

Expanded:

\[
\|u-v\|_2^2
=
\sum_{r=1}^{d_{\text{model}}} (u_r - v_r)^2
\]

where:

- \(u_r\) is the \(r\)-th component of vector \(u\).
- \(v_r\) is the \(r\)-th component of vector \(v\).
- \(d_{\text{model}}\) is the hidden dimension.

## 5.2 Cosine Distance

Another possible choice is cosine distance:

\[
d(u,v) = 1 - \operatorname{cos}(u,v)
\]

where:

\[
\operatorname{cos}(u,v)
=
\frac{u^\top v}{\|u\|_2\|v\|_2}
\]

Therefore:

\[
d(u,v)
=
1-
\frac{u^\top v}{\|u\|_2\|v\|_2}
\]

where:

- \(u^\top v\) is the dot product between \(u\) and \(v\).
- \(\|u\|_2\) is the Euclidean norm of \(u\).
- \(\|v\|_2\) is the Euclidean norm of \(v\).

---

## 6. Continuous Prototypes

## 6.1 Intuition Preserved

I expect continuous prototypes to store more fine-grained knowledge (than discrete ones) of normal patterns.

I expect the continuous prototypes to be used more by the reconstruction head, which is expected to be only good at reconstructing NORMAL patterns, bad at anomalous ones.

## 6.2 Continuous Prototype Set

Let the continuous prototypes be:

\[
P_c = \{p^c_1, p^c_2, \dots, p^c_M\}
\]

where:

\[
p^c_m \in \mathbb{R}^{d_{\text{model}}}
\]

and:

- \(P_c\) is the continuous prototype set.
- \(M\) is the number of continuous prototypes.
- \(p^c_m\) is the \(m\)-th continuous prototype.
- Each continuous prototype has the same dimension as a hidden vector.

## 6.3 Continuous Prototype Retrieval

For each hidden vector \(h_t\), retrieve a continuous-prototype-based representation:

\[
\hat{h}^{c}_t
=
\operatorname{Retrieve}_c(h_t, P_c)
\]

where:

- \(h_t\) is the hidden vector at time-step \(t\).
- \(P_c\) is the continuous prototype set.
- \(\hat{h}^{c}_t\) is the retrieved continuous-prototype representation.

For the full sequence:

\[
\hat{H}^{c}
=
\operatorname{Retrieve}_c(H, P_c)
\]

with shape:

\[
\hat{H}^{c} \in \mathbb{R}^{L \times d_{\text{model}}}
\]

and because \(L=20\):

\[
\hat{H}^{c} \in \mathbb{R}^{20 \times d_{\text{model}}}
\]

## 6.4 Continuous Prototypes for Reconstruction Head

The reconstruction branch uses continuous prototypes more.

A fused reconstruction representation can be written as:

\[
H_{\text{rec}}
=
\operatorname{Fusion}_{\text{rec}}(H, \hat{H}^{c})
\]

where:

- \(H\) is the encoder hidden representation.
- \(\hat{H}^{c}\) is the continuous-prototype representation.
- \(\operatorname{Fusion}_{\text{rec}}\) is the fusion operation for the reconstruction branch.
- \(H_{\text{rec}}\) is the representation passed to the reconstruction head.

The reconstruction head produces:

\[
\tilde{x}
=
g_{\text{rec}}(H_{\text{rec}})
\]

where:

- \(g_{\text{rec}}\) is the reconstruction head.
- \(\tilde{x}\) is the reconstructed input.

The desired shape is:

\[
\tilde{x} \in \mathbb{R}^{L \times C}
\]

or:

\[
\tilde{x} \in \mathbb{R}^{20 \times C}
\]

## 6.5 Reconstruction Loss on Normal Patterns

Because the continuous prototypes are expected to store more fine-grained knowledge of normal patterns, the reconstruction head is expected to be only good at reconstructing NORMAL patterns.

For normal input \(x\):

\[
\mathcal{L}_{\text{rec-normal}}
=
\frac{1}{L}
\sum_{t=0}^{L-1}
\ell_{\text{rec}}(x_t, \tilde{x}_t)
\]

Because \(L=20\):

\[
\mathcal{L}_{\text{rec-normal}}
=
\frac{1}{20}
\sum_{t=0}^{19}
\ell_{\text{rec}}(x_t, \tilde{x}_t)
\]

where:

- \(\mathcal{L}_{\text{rec-normal}}\) is the reconstruction loss on normal input.
- \(x_t\) is the original normal input at time-step \(t\).
- \(\tilde{x}_t\) is the reconstructed output at time-step \(t\).
- \(\ell_{\text{rec}}(\cdot,\cdot)\) is the reconstruction error function.

If using squared error:

\[
\ell_{\text{rec}}(x_t, \tilde{x}_t)
=
\|x_t - \tilde{x}_t\|_2^2
\]

Expanded:

\[
\|x_t - \tilde{x}_t\|_2^2
=
\sum_{c=1}^{C}(x_{t,c} - \tilde{x}_{t,c})^2
\]

where:

- \(x_{t,c}\) is the value of channel \(c\) at time-step \(t\).
- \(\tilde{x}_{t,c}\) is the reconstructed value of channel \(c\) at time-step \(t\).

## 6.6 Expected Bad Reconstruction on Anomalous Positions

For anomalous input \(x'\), the reconstruction head is expected to be bad at anomalous ones.

The reconstruction error at time-step \(t\) is:

\[
s_t
=
\ell_{\text{rec}}(x'_t, \tilde{x}'_t)
\]

where:

- \(s_t\) is the reconstruction-based anomaly score at time-step \(t\).
- \(x'_t\) is the anomalous-view input at time-step \(t\).
- \(\tilde{x}'_t\) is the reconstruction of \(x'_t\).

The desired behavior is:

\[
s_{a_i} > s_k,
\qquad
 a_i \in A,\; k \notin A
\]

This means:

- Anomalous positions should have larger reconstruction error.
- Normal positions should have smaller reconstruction error.

---

## 7. Discrete Prototypes

## 7.1 Intuition Preserved

I expect discrete prototypes to store both normal and anomalous patterns but in a more compressed manner than continuous prototypes.

I expect discrete prototypes to be used more by the classification head.

## 7.2 Discrete Prototype Set

Let the discrete prototypes be:

\[
P_d = \{p^d_1, p^d_2, \dots, p^d_K\}
\]

where:

\[
p^d_k \in \mathbb{R}^{d_{\text{model}}}
\]

and:

- \(P_d\) is the discrete prototype set.
- \(K\) is the number of discrete prototypes.
- \(p^d_k\) is the \(k\)-th discrete prototype.
- Each discrete prototype has the same dimension as a hidden vector.

## 7.3 Discrete Prototype Assignment

A discrete prototype assignment can be written as:

\[
q_t
=
\operatorname{Assign}(h_t, P_d)
\]

where:

- \(q_t\) is the assigned discrete prototype index at time-step \(t\).
- \(h_t\) is the hidden vector at time-step \(t\).
- \(P_d\) is the discrete prototype set.

A nearest-prototype assignment can be written as:

\[
q_t
=
\arg\min_{k \in \{1,2,\dots,K\}}
\|h_t - p^d_k\|_2^2
\]

where:

- \(q_t\) is the index of the closest discrete prototype.
- \(p^d_k\) is the \(k\)-th discrete prototype.
- \(\|h_t - p^d_k\|_2^2\) is the squared Euclidean distance between hidden vector and prototype.

The retrieved discrete-prototype representation is:

\[
\hat{h}^{d}_t
=
p^d_{q_t}
\]

where:

- \(\hat{h}^{d}_t\) is the retrieved discrete-prototype representation at time-step \(t\).
- \(p^d_{q_t}\) is the selected discrete prototype.

## 7.4 Soft Discrete Prototype Retrieval

A soft assignment can be written as:

\[
\alpha_{t,k}
=
\frac{
\exp(\operatorname{sim}(h_t,p^d_k)/\tau)
}{
\sum_{j=1}^{K}
\exp(\operatorname{sim}(h_t,p^d_j)/\tau)
}
\]

where:

- \(\alpha_{t,k}\) is the soft assignment weight from hidden vector \(h_t\) to discrete prototype \(p^d_k\).
- \(\operatorname{sim}(h_t,p^d_k)\) is a similarity function.
- \(\tau > 0\) is the temperature.
- Smaller \(\tau\) makes the assignment sharper.
- Larger \(\tau\) makes the assignment smoother.

The retrieved discrete-prototype representation is:

\[
\hat{h}^{d}_t
=
\sum_{k=1}^{K}
\alpha_{t,k}p^d_k
\]

where:

- \(\hat{h}^{d}_t\) is a weighted sum of discrete prototypes.
- \(\alpha_{t,k}\) controls how much prototype \(p^d_k\) contributes.

For the full sequence:

\[
\hat{H}^{d}
=
[\hat{h}^{d}_0, \hat{h}^{d}_1, \dots, \hat{h}^{d}_{19}]^\top
\]

with shape:

\[
\hat{H}^{d} \in \mathbb{R}^{20 \times d_{\text{model}}}
\]

---

## 8. Classification Head

The classification head uses discrete prototypes more.

A fused classification representation can be written as:

\[
H_{\text{cls}}
=
\operatorname{Fusion}_{\text{cls}}(H', \hat{H}^{d})
\]

where:

- \(H'\) is the hidden representation of anomalous version of \(x\).
- \(\hat{H}^{d}\) is the discrete-prototype representation.
- \(\operatorname{Fusion}_{\text{cls}}\) is the fusion operation for the classification branch.
- \(H_{\text{cls}}\) is the representation passed to the classification head.

The sequence representation can be flattened:

\[
z_{\text{cls}}
=
\operatorname{Flatten}(H_{\text{cls}})
\]

Because:

\[
H_{\text{cls}} \in \mathbb{R}^{20 \times d_{\text{model}}}
\]

then:

\[
z_{\text{cls}} \in \mathbb{R}^{20d_{\text{model}}}
\]

The classification prediction is:

\[
\hat{y}
=
g_{\text{cls}}(z_{\text{cls}})
\]

where:

- \(g_{\text{cls}}\) is the classification head.
- \(\hat{y}\) is the predicted label / predicted class probability.

The classification loss can be written as:

\[
\mathcal{L}_{\text{cls}}
=
\operatorname{CE}(y, \hat{y})
\]

where:

- \(\mathcal{L}_{\text{cls}}\) is the classification loss.
- \(y\) is the ground-truth label.
- \(\hat{y}\) is the prediction.
- \(\operatorname{CE}(\cdot,\cdot)\) is cross-entropy loss.

---

## 9. Fusion Design Choices

The current idea contains fusion blocks.

The fusion operation is a design choice.

## 9.1 Reconstruction Fusion

The reconstruction fusion is:

\[
H_{\text{rec}}
=
\operatorname{Fusion}_{\text{rec}}(H, \hat{H}^{c})
\]

This means the reconstruction branch uses:

- encoder hidden representation \(H\), and
- continuous-prototype representation \(\hat{H}^{c}\).

This matches the idea:

The continuous prototypes are used more by the reconstruction head.

## 9.2 Classification Fusion

The classification fusion is:

\[
H_{\text{cls}}
=
\operatorname{Fusion}_{\text{cls}}(H', \hat{H}^{d})
\]

This means the classification branch uses:

- anomalous-view hidden representation \(H'\), and
- discrete-prototype representation \(\hat{H}^{d}\).

This matches the idea:

The discrete prototypes are used more by the classification head.

## 9.3 Possible Fusion Operator: Concatenation

One possible fusion operator is concatenation followed by a projection:

\[
\operatorname{Fusion}(U,V)
=
\phi([U;V]W + b)
\]

where:

- \(U \in \mathbb{R}^{L \times d_{\text{model}}}\).
- \(V \in \mathbb{R}^{L \times d_{\text{model}}}\).
- \([U;V] \in \mathbb{R}^{L \times 2d_{\text{model}}}\) is concatenation along the feature dimension.
- \(W \in \mathbb{R}^{2d_{\text{model}} \times d_{\text{model}}}\) is a trainable weight matrix.
- \(b \in \mathbb{R}^{d_{\text{model}}}\) is a bias vector.
- \(\phi(\cdot)\) is a nonlinear activation function.
- The output is in \(\mathbb{R}^{L \times d_{\text{model}}}\).

## 9.4 Possible Fusion Operator: Gated Fusion

Another possible fusion operator is gated fusion:

\[
G = \sigma([U;V]W_g + b_g)
\]

\[
\operatorname{Fusion}(U,V)
=
G \odot U + (1-G) \odot V
\]

where:

- \(G \in \mathbb{R}^{L \times d_{\text{model}}}\) is the gate.
- \(\sigma(\cdot)\) is the sigmoid function.
- \(W_g \in \mathbb{R}^{2d_{\text{model}} \times d_{\text{model}}}\) is a trainable weight matrix.
- \(b_g \in \mathbb{R}^{d_{\text{model}}}\) is a bias vector.
- \(\odot\) is element-wise multiplication.
- \(U\) and \(V\) are fused by a learned gate.

---

## 10. Full Training Objective

A possible full training objective is:

\[
\mathcal{L}
=
\mathcal{L}_{\text{cls}}
+
\alpha \mathcal{L}_{\text{rec-normal}}
+
\beta \mathcal{L}_{\text{contrast}}
+
\gamma \mathcal{L}_{\text{proto}}
\]

where:

- \(\mathcal{L}\) is the full training loss.
- \(\mathcal{L}_{\text{cls}}\) is the classification loss.
- \(\mathcal{L}_{\text{rec-normal}}\) is the reconstruction loss on normal patterns.
- \(\mathcal{L}_{\text{contrast}}\) is the contrastive loss function on 2 hidden representations from 2 input views \(x\) and \(x'\).
- \(\mathcal{L}_{\text{proto}}\) is an optional prototype-related loss.
- \(\alpha\) controls the strength of reconstruction loss.
- \(\beta\) controls the strength of contrastive loss.
- \(\gamma\) controls the strength of prototype-related loss.

Expanded with the contrastive loss:

\[
\mathcal{L}
=
\mathcal{L}_{\text{cls}}
+
\alpha \mathcal{L}_{\text{rec-normal}}
+
\beta
\left[
\frac{1}{|A^c|}
\sum_{k \notin A}
d(h_k, h'_k)
+
\lambda
\frac{1}{|A|}
\sum_{a_i \in A}
\max \left(0,\; m - d(h_{a_i}, h'_{a_i}) \right)
\right]
+
\gamma \mathcal{L}_{\text{proto}}
\]

---

## 11. Design Choice Summary

## 11.1 What `x` and `x'` Mean

- `x` is normal.
- `x'` is anomalous version of `x`.
- Anomalies are injected.

## 11.2 What `A` Means

\[
A = \{a_1, a_2, \dots, a_n\}
\]

- \(A\) is the set of positions where anomalies are injected.
- Each \(a_i\) belongs to \(\{0,1,2,\dots,19\}\).

## 11.3 What Happens at Positions Not in `A`

For:

\[
k \notin A
\]

The hidden representations should be close:

\[
h_k \approx h'_k
\]

This gives the alignment part:

\[
\frac{1}{|A^c|}
\sum_{k \notin A}
d(h_k, h'_k)
\]

## 11.4 What Happens at Positions in `A`

For:

\[
a_i \in A
\]

The hidden representations should be separated:

\[
h_{a_i} \not\approx h'_{a_i}
\]

This gives the separation part:

\[
\frac{1}{|A|}
\sum_{a_i \in A}
\max \left(0,\; m - d(h_{a_i}, h'_{a_i}) \right)
\]

## 11.5 Continuous Prototypes

Continuous prototypes are expected to store more fine-grained knowledge of normal patterns.

They are expected to be used more by the reconstruction head.

The reconstruction head is expected to be only good at reconstructing NORMAL patterns, bad at anomalous ones.

## 11.6 Discrete Prototypes

Discrete prototypes are expected to store both normal and anomalous patterns but in a more compressed manner than continuous prototypes.

They are expected to be used more by the classification head.

---

## 12. Consistency Check

The proposed mathematical structure preserves the idea:

1. There are 2 input views: \(x\) and \(x'\).
2. \(x\) is normal.
3. \(x'\) is anomalous version of \(x\).
4. Anomalies are injected at positions \(A\).
5. The input window length is 20 time-steps.
6. Hidden representations are compared position-wise.
7. For positions \(k \notin A\), \(h_k\) and \(h'_k\) are pulled together.
8. For positions \(a_i \in A\), \(h_{a_i}\) and \(h'_{a_i}\) are pushed apart.
9. Continuous prototypes store more fine-grained knowledge of normal patterns.
10. Discrete prototypes store both normal and anomalous patterns but in a more compressed manner than continuous prototypes.
11. Discrete prototypes are used more by the classification head.
12. Continuous prototypes are used more by the reconstruction head.
13. The reconstruction head is expected to be only good at reconstructing NORMAL patterns, bad at anomalous ones.
