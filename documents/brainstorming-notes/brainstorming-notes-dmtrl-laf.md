# DMTRL-LAF for CNN Kernels: Forward Pass, Backward Pass, and Tensor Shapes

## 1. Core Idea

In Deep Multi-Task Representation Learning with Last-Axis Flattening (DMTRL-LAF), we do not learn one completely independent CNN kernel for every task.

Instead, for each layer, we represent the task-specific kernel as a linear combination of shared basis kernels.

For task \(t\), the generated kernel is:

\[
K^{(t)}
=
\sum_{k=1}^{K}
L_{:,:,:,:,k}S_{k,t}.
\]

Here:

\[
L \in \mathbb{R}^{H_k \times W_k \times C_{\text{in}} \times C_{\text{out}} \times K}
\]

is the tensor of shared basis kernels, and

\[
S \in \mathbb{R}^{K \times T}
\]

is the task-specific coefficient matrix.

So DMTRL-LAF changes how the CNN kernel is generated, but the convolution operation itself remains ordinary.

---

## 2. Example Tensor Shapes

Suppose each task uses a convolution kernel:

\[
K^{(t)}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

This means:

- 128 output filters,
- each filter sees 64 input channels,
- each input-channel slice is a \(3 \times 3\) matrix.

If there are \(T=2\) tasks, then stacking the two task kernels gives:

\[
\mathcal{K}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times 2}.
\]

In DMTRL-LAF, instead of directly learning \(\mathcal{K}\), we learn:

\[
L
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times K}
\]

and

\[
S
\in
\mathbb{R}^{K \times 2}.
\]

Here \(K\) is the number of latent basis kernels.

---

## 3. Initialization from Single-Task Learning

Assume we first train two independent single-task CNNs.

After training, we have two convolution kernels:

\[
K^{(1)}, K^{(2)}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

We stack them along the task axis:

\[
\mathcal{K}
=
\text{stack}(K^{(1)}, K^{(2)})
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times 2}.
\]

Then we flatten the first four axes:

\[
\mathcal{K}_{\text{flat}}
\in
\mathbb{R}^{(3\cdot 3 \cdot 64 \cdot 128) \times 2}.
\]

Because:

\[
3\cdot 3 \cdot 64 \cdot 128 = 73728,
\]

we get:

\[
\mathcal{K}_{\text{flat}}
\in
\mathbb{R}^{73728 \times 2}.
\]

Then we approximate it by matrix factorization:

\[
\mathcal{K}_{\text{flat}}
\approx
LS.
\]

At this flattened level:

\[
L_{\text{flat}}
\in
\mathbb{R}^{73728 \times K},
\]

\[
S
\in
\mathbb{R}^{K \times 2}.
\]

Finally, reshape \(L_{\text{flat}}\) back into convolution-kernel form:

\[
L
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times K}.
\]

These \(L\) and \(S\) become the actual learnable parameters of DMTRL-LAF.

---

## 4. Forward Pass

For task \(t\), we first generate its convolution kernel:

\[
K^{(t)}
=
\sum_{k=1}^{K}
L_{:,:,:,:,k}S_{k,t}.
\]

Shape check:

\[
L_{:,:,:,:,k}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128},
\]

\[
S_{k,t}
\in
\mathbb{R}.
\]

Therefore:

\[
L_{:,:,:,:,k}S_{k,t}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

After summing over \(k=1,\ldots,K\), we get:

\[
K^{(t)}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

Then the CNN performs ordinary convolution:

\[
Y^{(t)}
=
\text{Conv}(X^{(t)},K^{(t)}).
\]

If the input feature map is:

\[
X^{(t)}
\in
\mathbb{R}^{B \times H_{\text{in}} \times W_{\text{in}} \times 64},
\]

then the output feature map is:

\[
Y^{(t)}
\in
\mathbb{R}^{B \times H_{\text{out}} \times W_{\text{out}} \times 128}.
\]

Then the usual activation is applied:

\[
A^{(t)}
=
\sigma(Y^{(t)} + b).
\]

The bias is not shared by the tensor factorization in this paper.

---

## 5. Loss Function

For task \(t\), suppose the prediction is:

\[
\hat{y}^{(t)}
=
f_t(x^{(t)};L,S).
\]

The task loss is:

\[
\mathcal{L}^{(t)}
=
\ell(\hat{y}^{(t)},y^{(t)}).
\]

For multiple tasks, the total loss can be written as:

\[
\mathcal{L}
=
\sum_{t=1}^{T}
\mathcal{L}^{(t)}.
\]

In the two-task example:

\[
\mathcal{L}
=
\mathcal{L}^{(1)}
+
\mathcal{L}^{(2)}.
\]

---

## 6. Backward Pass: Gradient with Respect to the Generated Kernel

During backpropagation, the convolution operation first gives the gradient with respect to the generated task kernel:

\[
\frac{\partial \mathcal{L}^{(t)}}{\partial K^{(t)}}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

Let:

\[
G^{(t)}
=
\frac{\partial \mathcal{L}^{(t)}}{\partial K^{(t)}}.
\]

Then:

\[
G^{(t)}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

But \(K^{(t)}\) is not the final learnable parameter. It is generated from \(L\) and \(S\). So we must pass this gradient further back into \(L\) and \(S\).

---

## 7. Backward Pass: Gradient with Respect to \(L\)

Recall:

\[
K^{(t)}
=
\sum_{k=1}^{K}
L_{:,:,:,:,k}S_{k,t}.
\]

For each basis kernel \(k\):

\[
\frac{\partial K^{(t)}}{\partial L_{:,:,:,:,k}}
=
S_{k,t}.
\]

Therefore, by the chain rule:

\[
\frac{\partial \mathcal{L}^{(t)}}{\partial L_{:,:,:,:,k}}
=
G^{(t)}S_{k,t}.
\]

Across all tasks:

\[
\frac{\partial \mathcal{L}}{\partial L_{:,:,:,:,k}}
=
\sum_{t=1}^{T}
G^{(t)}S_{k,t}.
\]

So the full gradient tensor is:

\[
\frac{\partial \mathcal{L}}{\partial L}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times K}.
\]

This gradient has the same shape as \(L\).

---

## 8. Backward Pass: Gradient with Respect to \(S\)

Again:

\[
K^{(t)}
=
\sum_{k=1}^{K}
L_{:,:,:,:,k}S_{k,t}.
\]

For a coefficient \(S_{k,t}\):

\[
\frac{\partial K^{(t)}}{\partial S_{k,t}}
=
L_{:,:,:,:,k}.
\]

Therefore:

\[
\frac{\partial \mathcal{L}^{(t)}}{\partial S_{k,t}}
=
\left\langle
G^{(t)},
L_{:,:,:,:,k}
\right\rangle.
\]

The inner product means:

\[
\left\langle
G^{(t)},
L_{:,:,:,:,k}
\right\rangle
=
\sum_{h=1}^{3}
\sum_{w=1}^{3}
\sum_{c=1}^{64}
\sum_{m=1}^{128}
G^{(t)}_{h,w,c,m}
L_{h,w,c,m,k}.
\]

So:

\[
\frac{\partial \mathcal{L}}{\partial S}
\in
\mathbb{R}^{K \times T}.
\]

For the two-task case:

\[
\frac{\partial \mathcal{L}}{\partial S}
\in
\mathbb{R}^{K \times 2}.
\]

This gradient has the same shape as \(S\).

---

## 9. Parameter Update

After computing the gradients, the optimizer updates \(L\) and \(S\).

For simple gradient descent:

\[
L
\leftarrow
L
-
\eta
\frac{\partial \mathcal{L}}{\partial L},
\]

\[
S
\leftarrow
S
-
\eta
\frac{\partial \mathcal{L}}{\partial S}.
\]

Here \(\eta\) is the learning rate.

The generated kernels \(K^{(t)}\) are not stored as independent learnable parameters. They are regenerated from the updated \(L\) and \(S\) at the next forward pass.

---

## 10. Full Computation Pipeline

The whole pipeline is:

\[
L,S
\rightarrow
K^{(t)}
\rightarrow
\text{Conv}(X^{(t)},K^{(t)})
\rightarrow
\hat{y}^{(t)}
\rightarrow
\mathcal{L}^{(t)}.
\]

Then backpropagation goes in the reverse direction:

\[
\mathcal{L}^{(t)}
\rightarrow
\frac{\partial \mathcal{L}^{(t)}}{\partial K^{(t)}}
\rightarrow
\frac{\partial \mathcal{L}}{\partial L},
\frac{\partial \mathcal{L}}{\partial S}.
\]

So the key distinction is:

\[
K^{(t)}
\]

is used for convolution, but

\[
L,S
\]

are the true trainable parameters.

---

## 11. Orthogonality of Basis Kernels

The basis kernels in \(L\) do not need to be orthogonal.

A basis kernel is:

\[
L_{:,:,:,:,k}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128}.
\]

To test whether two basis kernels \(L_i\) and \(L_j\) are orthogonal, we flatten them into vectors and compute an inner product:

\[
\langle L_i,L_j\rangle
=
\sum_{h,w,c,m}
L_{h,w,c,m,i}
L_{h,w,c,m,j}.
\]

They are orthogonal if:

\[
\langle L_i,L_j\rangle = 0.
\]

A normalized version is cosine similarity:

\[
\cos(L_i,L_j)
=
\frac{
\langle L_i,L_j\rangle
}{
\|L_i\|\|L_j\|
}.
\]

But DMTRL-LAF does not require:

\[
\langle L_i,L_j\rangle = 0.
\]

The basis kernels may be correlated. The model only requires that their linear combinations help produce useful task-specific kernels.

---

## 12. Compact Summary

For a convolution layer with task-specific kernels:

\[
K^{(t)}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128},
\]

DMTRL-LAF represents them as:

\[
K^{(t)}
=
\sum_{k=1}^{K}
L_{:,:,:,:,k}S_{k,t}.
\]

The learnable parameters are:

\[
L
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times K},
\]

\[
S
\in
\mathbb{R}^{K \times T}.
\]

The gradients are:

\[
\frac{\partial \mathcal{L}}{\partial L}
\in
\mathbb{R}^{3 \times 3 \times 64 \times 128 \times K},
\]

\[
\frac{\partial \mathcal{L}}{\partial S}
\in
\mathbb{R}^{K \times T}.
\]

The forward pass generates kernels from \(L,S\), then applies ordinary convolution.

The backward pass computes gradients through the generated kernels back into \(L,S\).

The convolution operator is unchanged. Only the parameterization of the convolution kernel is changed.