---
date: 2026-06-09 15:13:30 +07
researcher: Artificial Intelligence Agent
git_commit: 82d88af33bc04afb4067ba3fbcafaec43229f825
branch: dev
repository: bachelor-thesis-2026
topic: "Current codebase readiness for Experiment 2: simple CNN backbone, two-view contrastive learning, and CKA-gated fusion"
tags: [research, time-series, anomaly-detection, multi-class]
status: complete
last_updated: 2026-06-09
last_updated_by: Artificial Intelligence Agent
---

# Research: Current codebase readiness for Experiment 2

**Date**: 2026-06-09 15:13:30 +07  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: `82d88af33bc04afb4067ba3fbcafaec43229f825`  
**Branch**: `dev`

## Research Question
Kiểm tra xem tình trạng hiện tại của codebase đã sẵn sàng để chạy thí nghiệm số 2 hay chưa, theo thiết kế đã nêu trong `documents/brainstorming-notes/brainstorming-notes-dmtrl-laf.md`.

## Summary
Codebase hiện tại **đã sẵn sàng một phần**, nhưng **chưa sẵn sàng đầy đủ** cho phiên bản thí nghiệm 2 mà bạn mô tả.

Phần đã có sẵn là nhánh `thesis_multitask` với `encoder_family = cnn_simple`, hai-view contrastive loss, linear CKA, và CKA-gated fusion. Exp2 config hiện tại cũng đã bật các cờ tương ứng và test tối thiểu cho CKA/contrastive đã pass.

Phần chưa có là `DMTRL-LAF` đúng nghĩa như trong brainstorming note: không thấy implementation cho factorized CNN kernel, không thấy quy trình suy ra hạng `K = 2` từ ma trận trọng số, và không thấy orchestrator cho một pipeline gồm single-task pre-train rồi multi-task pre-train theo đúng 30-50 epoch warm-start mà bạn mô tả.

## Detailed Findings

### Data Preparation
- SMD machine-level data cho SMD `2-1` được trỏ qua `configs/data/smd_rtx3090_machine_2_1_20.yaml` trong Exp2 config.
- Exp2 dùng `window_size = 20` và taxonomy multi-class RedLamp với `num_classes = 12`.
- Config model hiện tại đặt `reconstruction_normal_only: true`, nên reconstruction loss chỉ tính trên normal cells khi synthetic anomaly mask có mặt.
- Synthetic augmentation và synthetic validation đều được bật trong model config của thesis multitask path.

### Modeling and Training
- `src/models/thesis_multitask.py` đã có `SimpleWindowCnnEncoder`, và `encoder_family` hỗ trợ cả `mlp` lẫn `cnn_simple`. Encoder giữ contract `[B, L, H]`.
- `src/core/config.py` đã validate `encoder_family in {"mlp", "cnn_simple"}` và đã whitelist các cờ Exp2: `enable_two_view_contrastive`, `contrastive_temperature`, `lambda_contrastive`, `enable_cka_gated_fusion`, `bootstrap_encoder_epochs`.
- `src/models/thesis_multitask.py` đã có:
  - linear CKA helper,
  - batch CKA scoring,
  - two-view contrastive loss,
  - CKA-gated fusion path,
  - total loss assembly có cộng thêm `lambda_contrastive * contrastive_loss`.
- `configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml` đã bật:
  - `bootstrap_encoder_epochs: 0`
  - `enable_two_view_contrastive: true`
  - `enable_cka_gated_fusion: true`
- `tests/test_exp2_two_view_cka.py` và `tests/test_thesis_multitask_cnn_shapes.py` đều pass trong kiểm tra cục bộ.

### What is already runnable
- Code hiện tại có thể chạy path Exp2 của `thesis_multitask` với:
  - `cnn_simple` encoder,
  - two-view contrastive objective,
  - CKA-gated fusion,
  - 300 epochs trên SMD `2-1`.
- Lớp `thesis_multitask` đã có đủ runtime contract để train/val/test theo config Exp2 hiện có.

### What is not yet implemented
- Không thấy implementation cho `DMTRL-LAF` trong `src/`, `configs/`, hay `tests/`; chỉ có note brainstorming trong `documents/brainstorming-notes/brainstorming-notes-dmtrl-laf.md`.
- Không có code path nào để:
  - factorize kernel CNN thành basis `L` và score matrix `S`,
  - suy ra hoặc ép rank `K = 2`,
  - log `SVD energy retention`, `SVD residual`, hoặc `factorization drift`,
  - chạy riêng phase single-task pre-train rồi multi-task pre-train theo đúng protocol bạn mô tả.
- `bootstrap_encoder_epochs` trong code hiện tại chỉ phục vụ bootstrap/memory initialization cho prototype path, không phải một scheduler cho hai phase pretrain tách biệt theo task.

## Code References
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:95-174` - simple CNN encoder and `cnn_simple` support
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:278-282` - Exp2 objective flags
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:1455-1500` - CKA-gated fusion
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:1551-1584` - linear CKA and two-view contrastive helper
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:1692-1714` - two-view pair preparation and stage gating
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/models/thesis_multitask.py:2340-2374` - contrastive loss added to total loss
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py:240-290` - model config whitelist includes Exp2 flags
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/core/config.py:770-825` - encoder family validation and thesis_multitask-specific config validation
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/model/thesis_multitask_redlamp_multiclass.yaml:1-45` - current default thesis multitask config
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/configs/experiment/thesis/exp2/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20-exp2__w20__seed11__default.yaml:1-56` - enabled Exp2 experiment config
- `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/tests/test_exp2_two_view_cka.py:1-58` - CKA/contrastive tests

## Pipeline Documentation
The current thesis multitask pipeline is a standard offline multi-task loop over fixed windows of length `20`. For Exp2, the model path is:

`batch -> simple CNN encoder -> continuous prototype branch + discrete prototype branch -> CKA-gated fusion -> reconstruction head + classification head -> total loss`

This is already runnable in the current codebase. However, the DMTRL-LAF idea is not yet part of this pipeline, so the kernel factorization and rank-selection stage is still only a design note.

## Historical Context (from documents/)
- `documents/design/idea.md` and `documents/design/design_starter.md` describe the thesis-facing hidden-state contract and the modular objective philosophy.
- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md` is the active implementation contract for the current Exp2 path.
- `documents/logs/05-20-2026/detail/detail-exp2-offline-pretraining-two-view-contrastive-cka-gated-fusion.md` documents the intended implementation phases for the current thesis multitask Exp2 path.
- `documents/brainstorming-notes/brainstorming-notes-dmtrl-laf.md` contains the DMTRL-LAF kernel-factorization idea, but it is still a brainstorming note rather than an implemented code path.

## Open Questions
- Do you want Exp2 to stay on the existing `thesis_multitask` implementation, or do you want a separate RedLamp-based Exp2 implementation that actually contains DMTRL-LAF?
- If DMTRL-LAF is the target, what exact tensor factorization should be used for each CNN layer, and what rule should decide `K = 2` beyond manual fixing?
- Should the staged training be implemented as one experiment config with internal phase switching, or as separate configs/runs for single-task pre-train and multi-task pre-train?
