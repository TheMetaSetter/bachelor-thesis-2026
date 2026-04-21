## Summary

During training, H-PAD updates each prototype by combining its **previous memory state** with a **weighted summary of the current query features**, where the combination weight is controlled by a learned **update gate**. During testing, this write operation is disabled, so the learned prototypes are treated as fixed memory and are used only for reconstruction. 

## Detailed Markdown Description of the Updating Mechanism

### 1. High-level idea

H-PAD maintains two kinds of memory-like prototype vectors:

1. **Patch prototypes** for multi-scale local temporal patterns.
2. **Period prototypes** for variable-wise periodic patterns.

In both cases, the update rule follows the same principle:

* compute how strongly the current encoded queries match a prototype,
* aggregate the matched query information,
* use a learned gate to decide how much of the **old prototype** to keep and how much **new information** to write into memory.

So the mechanism is not a hard replacement of prototypes. It is a **continuous, gated interpolation** between old memory and new evidence from the current training batch. 

---

### 2. Patch-prototype update during training

After average pooling at scale (z), the input subsequence is encoded into query vectors

[
Q^{z} = {q^{z}*{1}, q^{z}*{2}, \dots, q^{z}*{L_z}}, \qquad q^{z}*{j} \in \mathbb{R}^{D}.
]

At the same scale, the memory bank contains (M) patch prototypes

[
B^{z} = {b^{z}*{1}, b^{z}*{2}, \dots, b^{z}*{M}}, \qquad b^{z}*{i} \in \mathbb{R}^{D}.
]

The first step is to compute how relevant each query (q_j^z) is to prototype (b_i^z). H-PAD defines a similarity weight

[
v^{z}_{ij}
==========

\frac{\exp!\left(\langle b^{z}*{i}, q^{z}*{j} \rangle / \tau \right)}
{\sum_{r=1}^{L_z} \exp!\left(\langle b^{z}*{i}, q^{z}*{r} \rangle / \tau \right)},
]

where (\tau) is a temperature parameter and (\langle \cdot, \cdot \rangle) denotes inner product similarity. This is a softmax over all queries for a fixed prototype, so the weights satisfy

[
\sum_{j=1}^{L_z} v^{z}_{ij} = 1.
]

That means each prototype attends to the current query set and forms a weighted summary of the most relevant encoded patches. 

The new candidate information for prototype (b_i^z) is therefore

[
\sum_{j=1}^{L_z} v^{z}*{ij} q^{z}*{j}.
]

However, the model should not discard the prototype’s historical content too aggressively. To control this, H-PAD introduces a learned gate

[
\psi
====

\sigma!\left(
U_1^{z} b_i^{z}
+
U_2^{z} \sum_{k=1}^{L_z} v^{z}*{ik} q^{z}*{k}
\right),
]

where:

* (\sigma(\cdot)) is the sigmoid function,
* (U_1^z) and (U_2^z) are learned parameter matrices,
* (\psi \in (0,1)^D) is a **dimension-wise update gate**. 

The prototype update rule is then

[
b^{z}_{i}
=========

(\mathbf{1}*D - \psi)\odot b^{z}*{i}
+
\psi \odot
\sum_{j=1}^{L_z} v^{z}*{ij} q^{z}*{j},
]

where (\odot) denotes element-wise multiplication. 

### 3. Interpretation of the patch update rule

This equation has a very clear meaning.

The updated prototype is a mixture of two terms:

* **memory preservation term**
  [
  (\mathbf{1}_D - \psi)\odot b_i^z
  ]
  which keeps part of the old prototype;

* **memory writing term**
  [
  \psi \odot \sum_{j=1}^{L_z} v^{z}_{ij} q_j^z
  ]
  which injects the new information extracted from the current encoded batch.

So for each feature dimension:

* if (\psi_d) is close to (0), the model mostly keeps the old prototype value;
* if (\psi_d) is close to (1), the model mostly overwrites that coordinate with new query information.

This is why the mechanism can be called **continuous prototype updating**: the memory evolves gradually over training rather than being replaced in a discrete or hard-assignment way. 

---

### 4. Period-prototype update during training

The period branch uses the same basic idea, but the objects being summarized are encoded period segments rather than multi-scale patches.

For one variable and one chosen period partition, the encoder produces

[
Q^{p} = {q^{p}*{1}, q^{p}*{2}, \dots, q^{p}*{N}}, \qquad q^{p}*{j} \in \mathbb{R}^{D},
]

and the model maintains a single period prototype

[
b^{p} \in \mathbb{R}^{D}.
]

Again, H-PAD computes prototype-to-query relevance weights

[
v^{p}_{j}
=========

\frac{\exp!\left(\langle b^{p}, q^{p}*{j} \rangle / \tau \right)}
{\sum*{r=1}^{N} \exp!\left(\langle b^{p}, q^{p}_{r} \rangle / \tau \right)}.
]

These weights define a weighted summary of the current period-query set:

[
\sum_{j=1}^{N} v^{p}*{j} q^{p}*{j}.
]

The update gate for the period prototype is

[
\psi
====

\sigma!\left(
U_1^{p} b^{p}
+
U_2^{p} \sum_{k=1}^{N} v^{p}*{k} q^{p}*{k}
\right),
]

and the period prototype is updated by

[
b^{p}
=====

(\mathbf{1}*D - \psi)\odot b^{p}
+
\psi \odot
\sum*{j=1}^{N} v^{p}*{j} q^{p}*{j}.
]

So the period branch uses the same gated write mechanism as the patch branch; only the source of the query features is different. 

---

### 5. Unified view of the memory update

Both patch and period updates can be written in a common form.

Let:

* (b) be a prototype vector,
* (\tilde{q}) be the weighted query summary,
* (\psi) be the learned update gate.

Then the generic update is

[
\tilde{q} = \sum_j v_j q_j,
]

[
\psi = \sigma(U_1 b + U_2 \tilde{q}),
]

[
b^{\text{new}} = (\mathbf{1}-\psi)\odot b^{\text{old}} + \psi \odot \tilde{q}.
]

This makes the mechanism easy to interpret:

* (v_j) decides **which current query features matter most** to this prototype,
* (\psi) decides **how much memory should change**,
* the final equation performs the actual **write operation** into memory.

---

### 6. Why the gate is necessary

The paper explicitly notes that prototypes should not only absorb new information, but also retain historical information. Without a gate, repeated updates could make prototypes unstable or overly dependent on the current batch. With the gate, H-PAD can adaptively regulate:

* **preservation of history**, and
* **incorporation of new normal patterns**. 

This is especially important in anomaly detection, because the prototypes are intended to represent the normal structure of the training data. A controlled update helps the memory bank evolve toward robust normal prototypes instead of fluctuating too sharply.

---

### 7. Relation to reconstruction

It is important to distinguish **updating prototypes** from **using prototypes**.

After the prototypes are updated, H-PAD uses them to reconstruct the encoded queries.

For patch reconstruction, the model computes weights from query to prototype:

[
w^{z}_{jk}
==========

\frac{\exp!\left(\langle q^{z}*{j}, b^{z}*{k} \rangle / \tau \right)}
{\sum_{r=1}^{M}\exp!\left(\langle q^{z}*{j}, b^{z}*{r} \rangle / \tau \right)},
]

and reconstructs

[
\hat{q}^{z}*{j} = \sum*{k=1}^{M} w^{z}*{jk} b^{z}*{k}.
]

For period reconstruction, the model similarly reconstructs encoded period queries from the updated period prototypes.  

So the pipeline during training is:

1. encode the current sequence into queries,
2. update the memory prototypes,
3. read from the updated prototypes to reconstruct,
4. optimize the training loss.

The update step is therefore a **write-to-memory** step, while the reconstruction step is a **read-from-memory** step.

---

### 8. Training-time versus testing-time behavior

The updating equations above describe the **training-time** mechanism.

At training time, prototypes are repeatedly refined from normal training sequences:

[
b^{(t+1)} = (\mathbf{1}-\psi^{(t)})\odot b^{(t)} + \psi^{(t)}\odot \tilde{q}^{(t)}.
]

This makes the prototypes evolve across optimization steps and batches.  

At testing time, this update is no longer performed. Instead, the learned prototypes are frozen:

[
b^{\text{test}} = b^{\star},
]

where (b^{\star}) denotes the prototype after training has finished.

Then inference only applies the **read/reconstruction** part, not the **write/update** part. In other words:

* **training:** memory is updated and used,
* **testing:** memory is only used.

This matches the intended role of the memory bank: it stores normal prototypes learned during training and serves as a fixed reference for reconstructing test inputs.

---

### 9. Conceptual significance

This update mechanism gives H-PAD several useful properties.

First, it is **adaptive**: each prototype receives only the information that is most relevant to it through the similarity weights (v).

Second, it is **stable**: the gate prevents abrupt overwriting and preserves useful historical structure.

Third, it is **continuous**: prototypes move smoothly in feature space as training proceeds.

Fourth, it is **memory-based**: the model explicitly separates learning normal patterns into stored prototype vectors, instead of relying only on a standard encoder-decoder bottleneck.

These properties are central to the paper’s goal of learning robust hybrid prototypes for multivariate time-series anomaly detection.  

## Check

The equations for patch updating are exactly the paper’s Eq. (3)–(5), and the equations for period updating are Eq. (8)–(9). The reconstruction equations, Eq. (6) and Eq. (10), are separate read operations and should not be confused with the write/update mechanism. Confidence: High.
