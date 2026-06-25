---
date: 2026-06-25 20:10:00 +0700
researcher: TheMetaSetter
repository: bachelor-thesis-2026
branch: dev
topic: "Official SMD entity lock for comparative CNN experiments by drift in normality"
tags: [research, smd, drift, kl-divergence, entity-selection]
status: complete
last_updated: 2026-06-25
last_updated_by: TheMetaSetter
---

# Research: Official SMD Entity Lock for Comparative CNN Experiments by Drift in Normality

## Objective

Khóa ba entity SMD chính thức cho comparative experiment giữa:

- `src/models/thesis_multitask.py` theo three-stage
- `src/models/redlamp_mlp_baseline.py` theo single-stage

với tiêu chí chọn mẫu là **drift in normality mạnh giữa train và test**, không bị chi phối bởi các timestep test đã gán nhãn anomaly.

## Criterion

Tiêu chí khóa entity là:

- tính `KL(test_normal_only || train)` theo từng channel;
- sau đó tổng hợp theo ba thống kê:
  - `mean`
  - `max`
  - `top-5 mean`

Trong đó `test_normal_only` nghĩa là **loại các timestep test có label anomaly ra trước khi tính drift**.

## Result

Ba entity được khóa chính thức là:

1. `machine-3-9`
2. `machine-3-1`
3. `machine-1-6`

Đây là ba entity nổi bật nhất khi ưu tiên drift in normality.

## Supporting Numbers

### `KL(test_normal_only || train)` by mean

1. `machine-3-9`: `4.0294`
2. `machine-3-1`: `3.7090`
3. `machine-1-6`: `3.4895`

### `KL(test_normal_only || train)` by top-5 mean

1. `machine-1-6`: `15.1649`
2. `machine-3-1`: `13.7182`
3. `machine-3-9`: `13.4203`

### `KL(test_normal_only || train)` by max

Trong top theo `max`, ba entity đã khóa vẫn nằm nhóm drift mạnh:

- `machine-1-6`: `22.4266`
- `machine-3-9`: `18.0436`
- `machine-3-1`: `17.7955`

## Decision

Ba entity chính thức cho comparative experiment family là:

- `machine-3-9`
- `machine-3-1`
- `machine-1-6`

Các comparative data config và experiment config phải bám đúng tập này.

## Implementation Constraint

Mỗi run comparative chỉ được chứa **một entity** trong `data.entity_ids`, để standard scaler luôn được fit trên train split của đúng entity đó rồi apply sang val/test tương ứng, không bị pooled nhiều entity.
