# Development Specification v4: Raw-input-space MSE thresholding and prediction

**Status:** normative raw-score implementation specification  
**Date:** 2026-09-04  
**Supersedes:** raw-score decisions in v3 only; v1-v3 remain historical  
**Primary model:** `ThesisMultitaskModel`  
**Window length:** `L = 20`

## 1. Purpose

Version 4 adds a raw-input-space reconstruction MSE protocol for offline and
online anomaly detection. Raw input space means the original sensor units
before standardization. The model still receives scaled tensors, and the scaler
restores both input and reconstruction before the raw MSE is computed.

Training loss and the historical sigmoid protocol are unchanged. A v4 raw run
must declare `score_space: raw_input` and
`point_score_transform: identity`.

## 2. Score contract

For input `x_scaled[b,t,d]`, fitted scaler `inverse_transform`, and reconstruction
sample `r_scaled[b,m,t,d]`:

```text
x_raw[b,t,d] = inverse_transform(x_scaled[b,t,d])
r_raw[b,m,t,d] = inverse_transform(r_scaled[b,m,t,d])

point_mse[b,m,t] = mean_d((x_raw[b,t,d] - r_raw[b,m,t,d])^2)
raw_input_point_mse[b,t] = mean_m(point_mse[b,m,t])
raw_input_window_mse[b] = mean_t(raw_input_point_mse[b,t])
```

The normalized diagnostic values use the same per-sample order on scaled
tensors:

```text
normalized_input_point_mse[b,t] = mean_m(mean_d((x_scaled-r_scaled)^2))
normalized_input_window_mse[b] = mean_t(normalized_input_point_mse[b,t])
```

The implementation must average each sample MSE before any MC reduction. It
must not compute MSE from the mean reconstruction. A deterministic
reconstruction is treated as one MC sample.

## 3. Labels and predictions

Point labels are ground truth labels. A normal point has label `0`; an
anomalous point has label `1`. A window is anomalous when at least one point
label in that window is anomalous:

```python
window_label = (point_labels.sum(dim=1) > 0).long()
```

A normal window has label `0`; an anomalous window has label `1`. A prediction
is a separate threshold comparison and never replaces the ground-truth label:

```text
point_prediction = raw_input_point_mse > point_threshold
window_prediction = raw_input_window_mse > window_threshold
```

## 4. Operational and diagnostic fields

| Field | Space | Shape | Role |
|---|---|---:|---|
| `raw_input_point_mse` | original sensor units | `[B,L]` | operational point score |
| `raw_input_window_mse` | original sensor units | `[B]` | operational window score |
| `normalized_input_point_mse` | standardized input | `[B,L]` | diagnostic only |
| `normalized_input_window_mse` | standardized input | `[B]` | diagnostic only |
| `point_labels` | ground truth | `[B,L]` | metrics/label category |
| `window_labels` | ground truth | `[B]` | metrics/label category |
| `point_predictions` | threshold output | `[B,L]` | prediction |
| `window_predictions` | threshold output | `[B]` | prediction |

The raw protocol uses the identity transform. It must not fit, load, or apply
the shifted-and-scaled logistic sigmoid. The v3 `point_scores` field remains
available only for historical compatibility and must not be selected by the
raw evaluator or online runtime.

## 5. Terminology mapping

| old_name | new_name | status | semantic_equivalence | owner | migration_boundary |
|---|---|---|---|---|---|
| `raw_point_scores` | `normalized_input_point_mse` | renamed | same normalized intermediate value | model diagnostics | v4 payload/export |
| `point_scores` | `raw_input_point_mse` | split | v3 sigmoid output is not equivalent to v4 raw MSE | evaluator/online scorer | v4 raw protocol only |
| `window_scores` | `raw_input_window_mse` | renamed | raw reconstruction window mean | scorer/evaluator | v4 payload/export |
| `point_labels` | `point_labels` | unchanged | binary ground truth | dataset/evaluator | none |
| derived window label | `window_labels` | new | any anomalous point in window | evaluator/export | v4 payload/export |
| `prediction` | `point_predictions` or `window_predictions` | split | threshold result, not label | evaluator/online runtime | v4 records |

Schema v3 and sigmoid schema v4 artifacts remain historical-readable. Raw
artifacts use schema v5 and must declare the raw identity fields below.

## 6. Threshold artifact identity

```json
{
  "schema_version": 5,
  "score_space": "raw_input",
  "point_score_transform": "identity",
  "point_score_definition": "raw_input_point_mse",
  "window_score_definition": "raw_input_window_mse",
  "calibration_split": "clean_validation"
}
```

Point, online EWMA, and input-window thresholds are calibrated from clean
validation for one entity. Test labels are metrics-only. Latent geometry
thresholds remain separate from input-space MSE thresholds.

## 7. Compatibility and provenance

The raw artifact must record entity, seed, variant, window size, stride,
checkpoint SHA256, resolved-config SHA256, score identity, and threshold source.
The runtime rejects a raw protocol with a historical sigmoid artifact or a
checkpoint/config/entity/variant mismatch. Historical sigmoid files and paths
must not be overwritten.

## 8. Validation scope

The validation order is one end-to-end smoke combination, then
`machine-1-6`, `machine-3-4`, and `machine-3-9` sequentially. Each entity gets
its own clean-validation thresholds. Numeric raw thresholds must not be ranked
across entities because original sensor magnitudes can differ.

Each entity report must separate normal and anomalous point/window categories,
store raw predictions and labels separately, keep normalized MSE as diagnostic,
and include point-level and window-level raw-MSE histograms.

## 9. Terminology-change statement

This version splits the ambiguous v3 `point_scores`/`raw_point_scores` pair
into explicit raw-input and normalized-input names. It introduces
`window_labels`, `point_predictions`, and `window_predictions`. Historical v1-v3
names remain readable at their original compatibility boundary.
