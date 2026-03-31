This model is a novel, multi-task framework designed for robust and highly explainable time series anomaly detection. It synthesizes the core strengths of three distinct research papers:

1. **Explainable Structure (from MtsCID):** It adopts a **two-branch architecture** to separately model **intra-variate** (temporal) and **inter-variate** (relational) patterns. This provides a high-level, interpretable "why" for any detected anomaly.
2. **Robust Training (from RedLamp):** It is trained on a **multi-task objective**. One task is reconstruction (learning "normal") and the second is multi-class classification (learning "abnormal"). This makes the model robust to contaminated training data and allows it to identify *what* type of anomaly it sees.
3. **Uncertainty Quantification (from HSA):** It is **not deterministic**. All Transformer attention layers are replaced with **Hierarchical Stochastic Attention (HSA)**. This enables the model to provide a statistical uncertainty score, indicating its own confidence in a prediction.

---

### Detailed Computation Steps

This details the full forward pass of the model for a single batch of subsequences.

Input: A batch of subsequences X with shape [B, L, C].

- `B` = Batch Size (e.g., 64)
- `L` = Subsequence Length (e.g., 100)
- `C` = Number of Variates / Features

### Step 1: Upper Branch (Intra-Variate & Reconstruction Task)

This branch, based on the `t-AutoEncoder` from MtsCID, is responsible for learning the normal temporal patterns of each variate independently and serves as the primary reconstruction task.

1. **Input:** The batch `X` (`[B, L, C]`).
2. **DFT:** A Discrete Fourier Transform is applied to the time dimension (`L`), moving the data to the frequency domain: `H = DFT(X)`.
3. **Stochastic Transformer:** The frequency-domain data `H` is passed through an **HSA-Transformer** (Hierarchical Stochastic Attention replacing the original `fc-Transformer`). This block learns the normal relationships between different frequency components.
4. **iDFT:** An inverse DFT is applied to return to the time domain, yielding the first latent representation: `z_intra = iDFT(HSA_Transformer(H))`.
5. **Task 1 Head (Decoder):** `z_intra` is passed into a dedicated decoder (as in MtsCID) to create the final reconstruction: `X_hat = Decoder(z_intra)`.
    
    *Ghi chú thêm: Nếu reconstruction error cao thì có nghĩa là bên trong một hoặc nhiều variate của subsequence đó có xuất hiện anomalous pattern. Đây là anomalous diễn ra bên trong từng variate, hay còn gọi là intra-variate.*
    
6. **Branch Outputs:**
    - **Latent Representation:** `z_intra` (`[B, L, C]`).
        
        *Ghi chú thêm: Latent representation $z_{intra}$ chứa thông tin về các time-series pattern bên trong từng variate.*
        
    - **Reconstruction Error:** $\mathcal{L}_{MSE} = \text{criterion}(X, \hat{X})$ (calculated per time point).

### Step 2: Lower Branch (Inter-Variate & Feature Task)

This branch, based on the `i-Encoder` from MtsCID, is responsible for learning the normal *relationships between* different variates.

1. **Input:** The batch `X` (`[B, L, C]`).
2. **Conv1d:** A 1D Convolution is applied to `X` to capture coarse-grained local patterns and handle misalignments, creating `T`.
    
    *Ghi chú thêm: Conv1d giống như một bộ lọc nhiễu, giúp đơn giản hoá thông tin fine-grained của từng variate để transformer block dễ học quan hệ giữa các variate với nhau.*
    
3. **DFT:** A DFT is applied to `T` to move the relational patterns into the frequency domain: `E = DFT(T)`.
4. **Stochastic Transformer:** The frequency-domain data `E` is passed through its own **HSA-Transformer** block to learn the complex *inter-variate* frequency patterns.
5. **iDFT:** An inverse DFT is applied to return to the time domain, yielding the second latent representation: `z_inter = iDFT(HSA_Transformer(E))`.
6. **Branch Output:**
    - **Latent Representation:** `z_inter` (`[B, L, C]`).

### Step 3: Fusion & Classification Head (The "RedLamp" Task)

This stage fuses the knowledge from both branches to perform a multi-class classification of anomaly types.

1. **Inputs:** `z_intra` (from Step 1) and `z_inter` (from Step 2).
2. **Fusion Block:** The two latent representations are fused into a single, comprehensive representation.
    - `z_fused = FusionBlock([z_intra, z_inter])` (e.g., via simple concatenation, `[B, L, 2*C]`).
3. **Task 2 Head (Classifier):** The fused representation is fed into an MLP-based classifier (as in RedLamp).
4. **Branch Output:**
    - **Class Probabilities:** $\hat{y} = \text{Softmax}(\text{MLP}(z_{fused}))$. This is a probability distribution over `K` classes, (e.g., `Class 0: Normal`, `Class 1: Spike`, `Class 2: Noise`, etc.).

---

### Training Process (Multi-Task Objective)

The model is trained end-to-end to minimize two separate losses simultaneously, using data augmented with pseudo-anomalies (as in RedLamp):

1. **Loss 1 (Reconstruction):** The mean squared error from **Step 1**, which forces the upper branch to learn "normal" temporal patterns.
    - $\mathcal{L}_{Recon} = \text{mean}(\mathcal{L}_{MSE})$
2. **Loss 2 (Classification):** The cross-entropy loss from **Step 3**, which forces the classifier to correctly identify the *type* of (pseudo)anomaly. This uses the soft-label "backward correction" technique from RedLamp to handle data contamination.
    - $\mathcal{L}_{Class} = \text{CrossEntropy}(\hat{y}, \tilde{y})$ (where $\tilde{y}$ is the soft target label).
3. **Total Loss:** A weighted sum of the two task losses.
    - $\mathcal{L}_{Total} = (1-\gamma)\mathcal{L}_{Recon} + \gamma\mathcal{L}_{Class}$

---

### Inference & Anomaly Scoring (Stochastic Pass)

To detect an anomaly and get an uncertainty score, we must run inference multiple times, as required by the HSA mechanism.

1. **Run T Passes:** For a new, unseen subsequence `X_test`, run it through the *entire* model `T` times (e.g., T=10). Because the HSA layers are stochastic, this will produce `T` slightly different outputs.
2. **Collect Outputs:**
    - `T` Reconstruction Errors: $[\mathcal{L}_{MSE, 1}, \mathcal{L}_{MSE, 2}, ..., \mathcal{L}_{MSE, T}]$
    - `T` Class Probabilities: $[\hat{y}_1, \hat{y}_2, ..., \hat{y}_T]$
3. **Calculate Final Score:** The final anomaly score is the weighted average of the mean of all task outputs.
    - $\bar{\mathcal{L}}_{MSE} = \text{mean}(\mathcal{L}_{MSE, 1}, ..., \mathcal{L}_{MSE, T})$
    - $\bar{\hat{y}} = \text{mean}(\hat{y}_1, ..., \hat{y}_T)$
    - $Score_{Class} = \text{RedLampAnomalyScore}(\bar{\hat{y}})$ (This uses the FAA logic from RedLamp)
    - **`AnomalyScore`** = $(1-\gamma)\bar{\mathcal{L}}_{MSE} + \gamma \cdot Score_{Class}$
4. **Calculate Uncertainty Score:** The *variance* of the `T` outputs is the model's uncertainty.
    - **`UncertaintyScore`** = $\text{variance}(\hat{y}_1, ..., \hat{y}_T)$
    - *Sử dụng variance của cái gì để làm uncertainty score thì cần suy nghĩ thêm.*
    - *Có thể tính toán uncertainty bằng cách khác chứ không phải variance.*