# Online THESIS FPR audit

FPR được tính tại mức điểm: `FP / (FP + TN)`. Mỗi ô là trung bình số học của 3 seed; `Average` là trung bình của 3 entity means.

| Variant | machine-1-6 | machine-3-4 | machine-3-9 | Average |
|---|---:|---:|---:|---:|
| **THESIS O1+A2** | **0.000** | **0.969** | 0.070 | <u>0.346</u> |
| &nbsp;&nbsp;w/o BPSL (O0+A2) | 0.000 | **0.969** | 0.073 | 0.347 |
| &nbsp;&nbsp;w/o TTA (O1+A0) | 0.000 | **0.969** | <u>0.070</u> | <u>0.346</u> |
| &nbsp;&nbsp;w/o HO+CL (O1+A1) | 0.000 | **0.969** | <u>0.070</u> | <u>0.346</u> |
| &nbsp;&nbsp;w/o BPSL+TTA (O0+A0) | 0.000 | **0.969** | 0.073 | 0.347 |
| &nbsp;&nbsp;w/o BPSL+HO+CL (O0+A1) | 0.000 | **0.969** | 0.073 | 0.347 |

## Variant semantics

| Variant | Meaning |
|---|---|
| BPSL | Point-wise Balanced Reconstruction-Score Loss |
| TTA | online test-time adaptation |
| HO | guarded hard-old update |
| CL | online contrastive loss |

The canonical mapping is `O0 = w/o BPSL`, `O1 = w/ BPSL`, `A0 = w/o TTA`, `A1 = w/o HO + CL`, and `A2 = full TTA`.

Prediction/score consistency: all 54 THESIS runs have `prediction_mismatch_count = 0`.
