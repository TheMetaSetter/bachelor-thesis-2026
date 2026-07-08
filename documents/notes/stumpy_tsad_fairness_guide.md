# STUMPY Baseline Guide for Fair TSAD Experiments Against THESIS

## Summary

This note describes how to use **STUMPY** as a Matrix Profile baseline for time-series anomaly detection (TSAD), especially when the main method is **THESIS** with an offline pretraining phase and an online test-time adaptation (TTA) phase. The core fairness rule is simple: **STUMPY must receive the same data split, same preprocessing, same window length, same point-level aggregation rule, and same validation-only threshold calibration as THESIS**.

Strictly speaking, “absolute fairness” is not fully attainable because Matrix Profile and THESIS have different inductive biases. However, the protocol below removes the most common unfair advantages: test-threshold tuning, future leakage in online scoring, inconsistent windowing, different scaling, point adjustment used only for some methods, and hidden post-processing.

---

## 1. Sources and terminology

Use the following terminology in the experiment logs and paper text.

- **Matrix Profile (MP)**: a vector storing, for each subsequence, the nearest-neighbor distance to another non-trivial subsequence. High Matrix Profile values correspond to **discords**, which are often treated as anomaly candidates.
- **STUMPY**: a Python library for efficiently computing Matrix Profiles.
- **STUMP**: STUMPY’s exact z-normalized univariate Matrix Profile routine.
- **mSTUMP / `stumpy.mstump`**: STUMPY’s multidimensional Matrix Profile routine.
- **AB-join**: scoring subsequences in one time series by nearest-neighbor distance to another reference time series.
- **Self-join**: scoring subsequences by nearest-neighbor distance inside the same time series, with trivial matches excluded.
- **Discord score**: an anomaly score derived from high Matrix Profile distance.

Primary references:

1. Law, S. M. (2019). **STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining**. *Journal of Open Source Software*, 4(39), 1504. DOI: 10.21105/joss.01504. URL: https://joss.theoj.org/papers/10.21105/joss.01504
2. Yeh, C.-C. M., Kavantzas, N., & Keogh, E. (2017). **Matrix Profile VI: Meaningful Multidimensional Motif Discovery**. *IEEE ICDM 2017*. DOI: 10.1109/ICDM.2017.66. URL: https://www.cs.ucr.edu/~eamonn/Motif_Discovery_ICDM.pdf
3. STUMPY documentation, API reference. URL: https://stumpy.readthedocs.io/en/latest/api.html
4. STUMPY multidimensional motif tutorial. URL: https://stumpy.readthedocs.io/en/latest/Tutorial_Multidimensional_Motif_Discovery.html
5. Yeh, C.-C. M. (2026). **Matrix Profile for Time-Series Anomaly Detection: A Reproducible Open-Source Benchmark on TSB-AD**. arXiv:2604.02445. URL: https://arxiv.org/abs/2604.02445
6. MMPAD GitHub repository. URL: https://github.com/mcyeh/mmpad_tsb

---

## 2. THESIS protocol assumed as the reference

The STUMPY baseline should be aligned to the current THESIS protocol.

Assumed THESIS settings:

- Task: time-series anomaly detection on multivariate datasets such as SMD, MSL, SMAP, SWaT, and UCR-style data.
- Input shape in the codebase: `X ∈ R^{B × L × D}`.
- Current default window length: `L = 20`.
- Offline phase:
  - train split is used for model training and memory/prototype initialization;
  - validation split is used for threshold calibration and model selection;
  - test split is not used for fitting parameters, calibration, memory initialization, or threshold selection.
- Validation/test windowing:
  - clean validation, synthetic validation, and offline test evaluation use non-overlapping windows where the THESIS protocol requires non-overlap;
  - online TTA uses sliding windows because it processes a stream point by point.
- Online phase:
  - source encoder, memories, reconstruction heads, and classification path are frozen;
  - only the lightweight MLP projector is updated;
  - update candidates come from conservative pseudo-new-normal gating;
  - verification buffer admits non-overlapping windows only;
  - threshold calibration uses clean validation only.
- Current point-level threshold idea:
  - clean validation reconstruction MSE q99 for point-level threshold;
  - for online sliding windows, simulate sliding-window scoring on clean validation and calibrate q99 after the same aggregation/EWMA rule used at test time.

For STUMPY, the equivalent fairness principle is:

> No STUMPY statistic, reference database, threshold, aggregation parameter, smoothing parameter, or chosen dimensionality may be fitted using test labels or future test information that THESIS does not use.

---

## 3. Which STUMPY baseline variants to implement

Implement at least two STUMPY baselines. Keep their names explicit in logs.

### 3.1 `STUMPY-ChannelAB`

This is the safest semi-supervised baseline.

For each channel independently:

1. Use train-normal subsequences as the reference database.
2. Score validation/test subsequences by AB-join against the train reference.
3. Convert channel-level subsequence scores to point-level scores.
4. Aggregate across channels.
5. Calibrate threshold using clean validation only.

Recommended use:

- primary fairness baseline against THESIS offline scoring;
- causal online variant by scoring each newly completed test window against a frozen train reference.

Reason:

- it does not use future test subsequences as nearest-neighbor candidates;
- it behaves more like a non-parametric train-normal memory baseline.

### 3.2 `STUMPY-ChannelSelfJoin`

This is the classical unsupervised Matrix Profile discord baseline.

For each channel independently:

1. Run self-join on the entire evaluation sequence.
2. Use high Matrix Profile values as anomaly scores.
3. Aggregate channels and convert to point-level scores.

Recommended use:

- offline unsupervised baseline only;
- do not compare directly to online causal THESIS unless clearly marked as non-causal/offline.

Reason:

- self-join on the full test sequence can use future test subsequences as nearest-neighbor candidates;
- this is standard for offline Matrix Profile, but not equivalent to online TTA.

### 3.3 `STUMPY-MSTUMP-SelfJoin`

This uses `stumpy.mstump` for multidimensional Matrix Profile.

Recommended use:

- offline unsupervised multidimensional MP baseline;
- useful for studying whether multidimensional subspaces reveal anomalies better than per-channel MP.

Caution:

- `stumpy.mstump` is not merely a stack of 1D Matrix Profiles. It computes multidimensional profiles where each k-dimensional row may select a different subset of dimensions for each subsequence.
- If used on the entire test sequence, it is still an offline self-join protocol and should not be treated as causal online detection.

### 3.4 Optional: `MMPAD-style`

If time permits, add an extra baseline based on the 2026 MMPAD repository rather than plain STUMPY.

Use this only as a separately named baseline because it adds design choices beyond vanilla STUMPY:

- multidimensional aggregation;
- k-nearest-neighbor retrieval;
- exclusion-zone-aware handling of repeated anomalies;
- moving-average post-processing.

This should not silently replace the STUMPY baseline.

---

## 4. Installation and reproducibility

Recommended environment:

```bash
pip install stumpy numpy pandas scipy scikit-learn
```

Record these fields in every experiment report:

```text
python_version:
numpy_version:
stumpy_version:
numba_version:
scipy_version:
sklearn_version:
hardware_cpu:
hardware_gpu:
random_seed:
window_length:
stumpy_function:
join_type: self_join | ab_join
normalize:
p:
k:
channel_aggregation:
subsequence_to_point_rule:
threshold_source:
smoothing_rule:
```

STUMPY is mostly deterministic for the exact routines, but record seeds anyway because the surrounding experiment pipeline may contain randomized preprocessing, sampling, or baseline wrappers.

---

## 5. Data contract

### 5.1 Input shape

The codebase likely stores windows as:

```python
X_windows.shape == (B, L, D)
```

For STUMPY, operate on the continuous time series before window batching:

```python
X_train.shape == (T_train, D)
X_val.shape   == (T_val, D)
X_test.shape  == (T_test, D)
```

For `stumpy.stump`, pass one channel:

```python
x_c.shape == (T,)
```

For `stumpy.mstump`, use the multidimensional series. The STUMPY tutorial describes a multidimensional time series as shape `d × n`, where `d` is the number of dimensions and `n` is the number of time points. If the local dataset is stored as `T × D`, transpose before use when required by the installed STUMPY version:

```python
X_for_mstump = X.T  # shape: (D, T)
```

Before finalizing, verify this with a tiny local smoke test because STUMPY can also accept pandas-style examples in the documentation.

---

## 6. Preprocessing fairness

Use the exact same preprocessing pipeline as THESIS and all other baselines.

Required controls:

1. **Missing values**: use the same imputation rule for all methods.
2. **Channel order**: keep the same sensor order as THESIS.
3. **Constant channels**: log and handle constant channels consistently. Matrix Profile z-normalization can be ill-conditioned for constant subsequences.
4. **Scaling**: fit scalers on train only.
5. **No test scaler**: never fit mean/std/min/max on validation or test.
6. **No label-aware cleaning**: do not remove test anomalies before computing STUMPY scores.

Recommended scaler policy:

```python
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
```

Important detail:

- STUMPY’s exact MP routines are z-normalized by default for subsequences when `normalize=True`.
- This does not replace global train-only scaling if the rest of the benchmark uses train-only scaling.
- Report both the dataset-level scaler and the STUMPY `normalize` flag.

Recommended setting:

```python
normalize = True
p = 2.0
```

Do not tune `normalize` or `p` on test.

---

## 7. Window length

For fairness against THESIS, use:

```python
m = L_THESIS = 20
```

This is the most important STUMPY hyperparameter.

Do not tune `m` using test labels. If a window-size sensitivity study is needed, use a predefined grid and apply it to all methods consistently:

```text
m ∈ {10, 20, 40, 80}
```

But the main table should use the THESIS default:

```text
m = 20
```

If THESIS later changes `L`, the STUMPY default must change with it.

---

## 8. STUMPY univariate per-channel AB-join

This is the recommended primary baseline.

### 8.1 Core idea

For channel `c`, a test subsequence is anomalous if it is far from all train-normal subsequences in that same channel.

Let:

```text
train_c ∈ R^{T_train}
query_c ∈ R^{T_query}
```

Use:

```python
mp = stumpy.stump(
    T_A=query_c,
    m=m,
    T_B=train_c,
    ignore_trivial=False,
    normalize=True,
    p=2.0,
)
```

The first profile column, or `mp.P_` if available, is the subsequence-level anomaly score:

```python
subseq_score = mp[:, 0].astype(float)
```

or:

```python
subseq_score = mp.P_.astype(float)
```

### 8.2 Pseudocode

```python
import numpy as np
import stumpy


def stumpy_channel_ab_subseq_scores(X_query, X_ref, m, normalize=True, p=2.0):
    """
    X_query: np.ndarray, shape (T_query, D)
    X_ref:   np.ndarray, shape (T_ref, D)
    returns: np.ndarray, shape (T_query - m + 1, D)
    """
    scores = []
    D = X_query.shape[1]

    for c in range(D):
        q = np.asarray(X_query[:, c], dtype=np.float64)
        r = np.asarray(X_ref[:, c], dtype=np.float64)

        mp = stumpy.stump(
            T_A=q,
            m=m,
            T_B=r,
            ignore_trivial=False,
            normalize=normalize,
            p=p,
        )

        if hasattr(mp, "P_"):
            s_c = np.asarray(mp.P_, dtype=np.float64)
        else:
            s_c = np.asarray(mp[:, 0], dtype=np.float64)

        scores.append(s_c)

    return np.stack(scores, axis=1)
```

### 8.3 Fair channel aggregation

The raw output is:

```text
S_subseq ∈ R^{(T_query - m + 1) × D}
```

Recommended aggregation:

```python
S_window = np.nanmax(S_subseq, axis=1)
```

Why max?

- Many multivariate TSAD anomalies affect only a subset of sensors.
- Mean aggregation can bury sparse channel anomalies.

However, max can be sensitive to noisy channels. To be stricter, calibrate each channel on clean validation first:

```python
S_channel_z[:, c] = (S_subseq[:, c] - median_val_c) / (iqr_val_c + eps)
S_window = np.nanmax(S_channel_z, axis=1)
```

The median/IQR must be fitted on clean validation only.

Recommended default:

```text
channel_aggregation = robust_val_zscore_then_max
```

Do not compare `max`, `mean`, `topk_mean`, and `median` on test labels and report only the best. That is threshold/post-processing leakage.

---

## 9. STUMPY per-channel self-join

Use this only for an offline unsupervised setting.

```python
mp = stumpy.stump(
    T_A=x_c,
    m=m,
    T_B=None,
    ignore_trivial=True,
    normalize=True,
    p=2.0,
)
```

Fairness note:

- `ignore_trivial=True` is correct for self-join because nearby overlapping subsequences are trivial matches.
- This baseline uses the evaluation sequence itself as the nearest-neighbor search space.
- If this is run on full test, it is not causally comparable to online THESIS.

Recommended name:

```text
STUMPY-ChannelSelfJoin-Offline
```

Do not call it simply “STUMPY baseline” in a table that also includes online TTA methods.

---

## 10. STUMPY multidimensional self-join with `mstump`

### 10.1 Core idea

`stumpy.mstump` computes a multidimensional Matrix Profile. The output contains one row per dimensionality level `k`, where row `k-1` corresponds to the k-dimensional Matrix Profile.

Example:

```python
mps, indices = stumpy.mstump(X.T, m=m)
```

Expected conceptual shape:

```text
mps.shape     ≈ (D, T - m + 1)
indices.shape ≈ (D, T - m + 1)
```

For anomaly detection, use high values in `mps`, not low values. Low values indicate motifs; high values indicate discords.

### 10.2 Choosing dimensionality `k`

Avoid choosing `k` using test labels.

Possible policies:

#### Policy A: fixed k

```text
k_dim = 1
```

This is sensitive to anomalies that appear in a small subset of channels.

#### Policy B: all-k scan with validation calibration

For each `k`, fit robust statistics on clean validation:

```python
z_k = (mps[k] - median_val_k) / (iqr_val_k + eps)
```

Then aggregate:

```python
S_window = max_k z_k
```

Recommended default:

```text
mstump_dimensionality_policy = all_k_robust_val_zscore_then_max
```

This avoids selecting one hand-tuned `k`, but it must be calibrated with the same clean validation protocol because max over `k` increases score scale.

### 10.3 Limitation

Plain `stumpy.mstump` is mainly a self-join routine. For a semi-supervised train-reference multidimensional detector, either implement a custom multidimensional AB-join or use an MMPAD-style implementation. Keep this distinction explicit.

---

## 11. Convert subsequence/window scores to point-level scores

STUMPY returns one score per subsequence start index:

```text
S_window[i] corresponds to window X[i : i + m]
```

THESIS is evaluated using point-level anomaly scores, so STUMPY must use the same point-level score contract.

### 11.1 Offline non-causal pointification

This is acceptable for offline evaluation if all methods use the same rule.

```python
def window_to_point_mean(S_window, T, m):
    values = [[] for _ in range(T)]
    for i, s in enumerate(S_window):
        for t in range(i, i + m):
            values[t].append(s)
    return np.array([
        np.mean(v) if len(v) > 0 else np.nan
        for v in values
    ])
```

Caution:

- This uses future windows for early points if applied online.
- Therefore, do not use this rule for online TTA comparison.

### 11.2 Online causal pointification

Recommended for comparison with online THESIS:

```python
def window_to_point_causal_end(S_window, T, m):
    S_point = np.full(T, np.nan, dtype=float)
    for i, s in enumerate(S_window):
        end_t = i + m - 1
        S_point[end_t] = s
    return S_point
```

Meaning:

- A point can only be scored after the window ending at that point is complete.
- The first `m - 1` points have no complete window and should be ignored or marked as warm-up.

Recommended default for online comparison:

```text
subsequence_to_point_rule = causal_window_end
```

This aligns with the streaming description: receive one point, form a complete window when enough points exist, then output a score.

---

## 12. Threshold calibration

Use clean validation only.

### 12.1 Offline threshold

```python
threshold = np.nanquantile(S_val_point_clean, 0.99)
```

Then:

```python
y_pred = (S_test_point > threshold).astype(int)
```

### 12.2 Online threshold with EWMA

If THESIS uses EWMA on online point scores, STUMPY must use the same EWMA before calibration and testing.

Example:

```python
def ewma_scores(scores, alpha_new=0.9):
    out = np.full_like(scores, np.nan, dtype=float)
    prev = np.nan
    for i, s in enumerate(scores):
        if np.isnan(s):
            continue
        if np.isnan(prev):
            prev = s
        else:
            prev = alpha_new * s + (1.0 - alpha_new) * prev
        out[i] = prev
    return out
```

Fair calibration:

```python
S_val_online = window_to_point_causal_end(S_val_window, T_val, m)
S_val_online_ewma = ewma_scores(S_val_online, alpha_new=0.9)
threshold_online = np.nanquantile(S_val_online_ewma, 0.99)
```

Then apply the same to test.

Do not tune `alpha_new` on test. If THESIS uses `0.9 new / 0.1 previous`, STUMPY uses exactly that.

---

## 13. Online STUMPY protocols

There are two fair online variants. Choose one primary variant and label it clearly.

### 13.1 Frozen-reference online AB-join

This is the cleanest comparator to THESIS if the purpose is anomaly scoring, not adaptation.

For each incoming point:

1. Append point to a stream buffer.
2. Once the buffer has `m` points, form the latest window.
3. Compute nearest-neighbor distance from this latest window to train-normal reference subsequences.
4. Aggregate channels.
5. Convert to point score at the current endpoint.
6. Apply EWMA if THESIS uses EWMA.
7. Compare to validation-calibrated threshold.

No test data is added to the reference database.

This is conservative and leakage-safe.

### 13.2 Incremental streaming Matrix Profile with `stumpi`

STUMPY provides `stumpy.stumpi` for incremental z-normalized Matrix Profile on streaming data. This can be useful, but it is not automatically comparable to THESIS.

Use it only if labeled as:

```text
STUMPY-STUMPI-OnlineSelfJoin
```

Fairness risks:

- the reference set evolves with test stream data;
- anomalous subsequences can enter the search space;
- repeated anomalies can become their own nearest neighbors and reduce anomaly scores;
- this resembles online unsupervised adaptation, but without THESIS-style pseudo-normal gating.

If used, report it separately from frozen-reference baselines.

---

## 14. Handling non-overlapping windows

THESIS uses non-overlapping windows for clean validation, synthetic validation, offline test evaluation, and verification buffer admission. STUMPY’s native Matrix Profile is a sliding-subsequence method. To avoid inconsistency, separate two things:

1. **Scoring granularity**: STUMPY may compute sliding scores internally.
2. **Evaluation/admission granularity**: final evaluation or buffer admission must follow the same rule as THESIS.

For offline fair evaluation:

- If THESIS offline test uses non-overlapping windows, report STUMPY scores only at equivalent non-overlapping endpoints or aggregate STUMPY sliding scores into the same non-overlapping window grid.

Example:

```python
nonoverlap_starts = np.arange(0, T - m + 1, m)
S_window_nonoverlap = S_window[nonoverlap_starts]
```

Then convert to point-level using the same THESIS rule.

For online fair evaluation:

- Use sliding windows for online scoring because the stream produces one new candidate window per new point after warm-up.
- If an online STUMPY variant updates any reference state, only admit non-overlapping pseudo-normal windows under the same gate that THESIS uses.

Primary recommendation:

- Use frozen-reference online AB-join. Then there is no STUMPY update buffer and no risk of buffer policy mismatch.

---

## 15. Metrics

Use exactly the same metric implementation as THESIS and other baselines.

Recommended metrics:

- score-based: `VUS-PR`, `VUS-ROC`, `AUC-PR`, `AUC-ROC`;
- label-based: `F1`, `Precision`, `Recall`, `Affiliation-F1` if already used in the benchmark.

Strict rules:

1. Do not apply point adjustment unless every method uses it and the paper clearly labels it.
2. Do not report best test threshold for STUMPY while THESIS uses validation threshold.
3. Do not drop the last incomplete batch/window for one method but not others.
4. For online evaluation, ignore or separately mark the first `m - 1` warm-up points.
5. Use the same anomaly label timeline for all methods.

Recommended main results:

```text
score-based table: raw point scores, no threshold tuning
label-based table: threshold calibrated on clean validation only
```

Optional oracle table:

```text
oracle-threshold F1, clearly marked as oracle and not used as the main claim
```

---

## 16. Runtime and memory fairness

Record total wall-clock time for the same pipeline stage boundaries.

For STUMPY include:

- preprocessing time;
- Matrix Profile computation time;
- channel aggregation time;
- subsequence-to-point conversion time;
- threshold calibration time;
- inference time.

For THESIS include:

- model loading time only if STUMPY reference initialization is also counted;
- forward time;
- online gating time;
- online projector update time;
- score aggregation time.

Main rule:

> Do not compare only STUMPY core kernel time against full THESIS online TTA time.

Suggested reporting:

```text
train_or_fit_time:
calibration_time:
offline_inference_time:
online_avg_ms_per_point:
peak_cpu_memory_mb:
peak_gpu_memory_mb:
```

If using `gpu_stump`, report GPU model and STUMPY GPU settings. If THESIS uses GPU and STUMPY uses CPU, that is acceptable only if clearly reported.

---

## 17. Recommended final baseline configuration

Use the following as the main STUMPY configuration unless there is a strong reason to change it.

```yaml
baseline_name: STUMPY-ChannelAB-FrozenTrainRef
library: stumpy
function: stump
join_type: AB-join
reference_split: train_only
query_split: val_or_test
window_length: 20
normalize: true
p: 2.0
ignore_trivial: false
channel_processing: independent_per_channel
channel_score_calibration: robust_median_iqr_on_clean_val
channel_aggregation: max_after_robust_val_zscore
subsequence_to_point_offline: same_as_thesis_offline_rule
subsequence_to_point_online: causal_window_end
online_smoothing: same_as_thesis_ewma_if_enabled
online_ewma_alpha_new: 0.9
threshold_source: clean_validation_only
threshold_quantile: 0.99
point_adjustment: false
drop_last: false
test_label_usage: metrics_only
online_reference_update: false
```

Optional offline multidimensional baseline:

```yaml
baseline_name: STUMPY-MSTUMP-SelfJoin-Offline
library: stumpy
function: mstump
join_type: self-join
window_length: 20
normalize: true
p: 2.0
dimensionality_policy: all_k_robust_val_zscore_then_max
subsequence_to_point_offline: same_as_thesis_offline_rule
threshold_source: clean_validation_only
threshold_quantile: 0.99
point_adjustment: false
drop_last: false
test_label_usage: metrics_only
causal_online_comparable: false
```

Optional streaming baseline:

```yaml
baseline_name: STUMPY-STUMPI-OnlineSelfJoin
library: stumpy
function: stumpi
join_type: online_self_join
egress: true
window_length: 20
normalize: true
p: 2.0
subsequence_to_point_online: causal_window_end
online_smoothing: same_as_thesis_ewma_if_enabled
threshold_source: clean_validation_only
threshold_quantile: 0.99
point_adjustment: false
drop_last: false
reference_updates_with_test_stream: true
causal_online_comparable: partially
notes: report separately because the search space evolves with test data
```

---

## 18. Minimal implementation sketch

```python
import numpy as np
import stumpy
from sklearn.preprocessing import StandardScaler


def robust_fit_per_channel(S_val_subseq, eps=1e-8):
    med = np.nanmedian(S_val_subseq, axis=0)
    q75 = np.nanpercentile(S_val_subseq, 75, axis=0)
    q25 = np.nanpercentile(S_val_subseq, 25, axis=0)
    iqr = q75 - q25
    return med, np.maximum(iqr, eps)


def robust_transform_per_channel(S_subseq, med, iqr):
    return (S_subseq - med[None, :]) / iqr[None, :]


def aggregate_channels(S_subseq_z):
    return np.nanmax(S_subseq_z, axis=1)


def window_to_point_causal_end(S_window, T, m):
    S_point = np.full(T, np.nan, dtype=float)
    for i, s in enumerate(S_window):
        S_point[i + m - 1] = s
    return S_point


def ewma_scores(scores, alpha_new=0.9):
    out = np.full_like(scores, np.nan, dtype=float)
    prev = np.nan
    for i, s in enumerate(scores):
        if np.isnan(s):
            continue
        if np.isnan(prev):
            prev = s
        else:
            prev = alpha_new * s + (1.0 - alpha_new) * prev
        out[i] = prev
    return out


def stumpy_channel_ab_subseq_scores(X_query, X_ref, m, normalize=True, p=2.0):
    scores = []
    D = X_query.shape[1]
    for c in range(D):
        q = np.asarray(X_query[:, c], dtype=np.float64)
        r = np.asarray(X_ref[:, c], dtype=np.float64)
        mp = stumpy.stump(
            T_A=q,
            m=m,
            T_B=r,
            ignore_trivial=False,
            normalize=normalize,
            p=p,
        )
        s_c = np.asarray(mp.P_ if hasattr(mp, "P_") else mp[:, 0], dtype=np.float64)
        scores.append(s_c)
    return np.stack(scores, axis=1)


def fit_stumpy_channel_ab_baseline(X_train, X_val_clean, m=20):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val_clean)

    S_val_ch = stumpy_channel_ab_subseq_scores(X_val_s, X_train_s, m=m)
    med, iqr = robust_fit_per_channel(S_val_ch)
    S_val_win = aggregate_channels(robust_transform_per_channel(S_val_ch, med, iqr))
    S_val_point = window_to_point_causal_end(S_val_win, T=X_val_s.shape[0], m=m)
    S_val_point = ewma_scores(S_val_point, alpha_new=0.9)
    threshold = np.nanquantile(S_val_point, 0.99)

    return {
        "scaler": scaler,
        "X_ref": X_train_s,
        "m": m,
        "med": med,
        "iqr": iqr,
        "threshold": threshold,
    }


def score_stumpy_channel_ab_baseline(model, X_query):
    X_query_s = model["scaler"].transform(X_query)
    S_ch = stumpy_channel_ab_subseq_scores(X_query_s, model["X_ref"], m=model["m"])
    S_win = aggregate_channels(robust_transform_per_channel(S_ch, model["med"], model["iqr"]))
    S_point = window_to_point_causal_end(S_win, T=X_query_s.shape[0], m=model["m"])
    S_point = ewma_scores(S_point, alpha_new=0.9)
    y_pred = (S_point > model["threshold"]).astype(int)
    return S_point, y_pred
```

Notes:

- The sketch is intentionally conservative and leakage-safe.
- It uses train-only scaling, train-only reference subsequences, validation-only channel score calibration, validation-only threshold calibration, and causal endpoint scoring.
- Replace `StandardScaler` with the exact scaler already used in the THESIS pipeline if different.

---

## 19. Leakage checklist

Before running final experiments, answer each question with “yes”.

```text
[ ] Same raw train/val/test split as THESIS?
[ ] Same missing-value processing as THESIS?
[ ] Same channel order as THESIS?
[ ] Same train-fitted scaler as THESIS?
[ ] Same window length as THESIS?
[ ] Same offline non-overlap policy as THESIS where required?
[ ] Same online sliding-window policy as THESIS where required?
[ ] Same subsequence-to-point aggregation rule as THESIS?
[ ] Same EWMA/smoothing rule as THESIS, or explicitly disabled for all methods?
[ ] Threshold fitted on clean validation only?
[ ] No threshold chosen by best test F1?
[ ] No test anomaly ratio used as contamination parameter?
[ ] No point adjustment unless applied to all methods and clearly reported?
[ ] No drop-last mismatch?
[ ] Online STUMPY does not use future test points?
[ ] If online STUMPY updates reference state, is it reported separately?
[ ] Runtime includes preprocessing and post-processing for every method?
[ ] Score-based and label-based metrics are reported separately?
```

---

## 20. How to describe this in the thesis/paper

Suggested wording:

> We include Matrix Profile baselines implemented with STUMPY. For the main fair comparison, we use a semi-supervised channel-wise AB-join baseline, denoted STUMPY-ChannelAB-FrozenTrainRef. Each test subsequence is scored by its z-normalized Euclidean nearest-neighbor distance to train-normal subsequences in the same channel. Channel scores are robustly normalized using clean validation statistics and aggregated by maximum across channels. The subsequence score is converted to a point-level score using the same causal endpoint rule as the online THESIS setting. Thresholds are calibrated exclusively on clean validation scores using the same quantile rule as THESIS. Test labels are used only for evaluation.

For an offline unsupervised table:

> We additionally report STUMPY-ChannelSelfJoin-Offline and STUMPY-MSTUMP-SelfJoin-Offline. These methods compute Matrix Profiles on the full evaluation sequence and therefore are reported as offline self-join baselines, not as causal online detectors.

---

## 21. Practical recommendation

Use the following order of implementation:

1. Implement `STUMPY-ChannelAB-FrozenTrainRef` first.
2. Verify score length and point-level alignment on a tiny synthetic sequence.
3. Calibrate threshold on clean validation only.
4. Run one SMD machine end-to-end and inspect plots.
5. Add `STUMPY-ChannelSelfJoin-Offline` only for offline comparison.
6. Add `STUMPY-MSTUMP-SelfJoin-Offline` if runtime is acceptable.
7. Consider MMPAD only after the vanilla STUMPY baselines are stable.

The most defensible main baseline is:

```text
STUMPY-ChannelAB-FrozenTrainRef + L=20 + clean-val q99 + causal endpoint pointification + same EWMA as THESIS
```

This configuration is not necessarily the strongest Matrix Profile detector, but it is the cleanest and most protocol-compatible baseline against THESIS.

---

## 22. Confidence and assumptions

Confidence: **Medium-High**.

Key assumptions:

- THESIS currently uses `L = 20` as the main window length.
- The benchmark evaluates point-level scores.
- Clean validation is available and must be the only source for threshold calibration.
- The online setting must avoid future test leakage.
- The local codebase already has preprocessing, splitting, metric, and plotting utilities; therefore this guide focuses on STUMPY integration and fairness controls rather than full runnable project code.

