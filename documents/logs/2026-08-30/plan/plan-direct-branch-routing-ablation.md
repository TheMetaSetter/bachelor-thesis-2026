---
date: 2026-08-30 Asia/Ho_Chi_Minh
planner: OpenAI Codex
topic: "Implement direct branch routing ablation for THESIS"
status: implemented_local
revision: 5bae88fc9aa13814633d83eaf182e7ec4aadd990
branch: dev
related_research: documents/logs/2026-08-30/research/research-direct-branch-routing-ablation-rerun.md
---

# Implementation Plan: Direct Branch Routing Ablation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm một mode THESIS trong đó continuous latent đi thẳng vào reconstruction head và discrete latent đi thẳng vào classification head.

**Architecture:** Kiến trúc direct không có hai khối fusion trong đường tính toán. Runtime chỉ đi theo hai đường:

\[
H \rightarrow H_{\mathrm{cont}} \rightarrow \text{reconstruction head},
\qquad
H \rightarrow H_{\mathrm{disc}} \rightarrow \text{classification head}.
\]

Các projection và gate cũ chỉ được giữ lại trong model để checkpoint cũ còn cùng state-dict keys. Direct Stage B không gọi và không cập nhật các module này.

**Tech Stack:** Python, PyTorch, YAML, pytest, `.venv/bin/python`.

**Spec:** `documents/logs/2026-08-30/research/research-direct-branch-routing-ablation-rerun.md`.

## Global Constraints

- Không chạy Stage A.
- Không chạy full benchmark hoặc SSH trong phiên lập trình local.
- Không sửa hai model fusion cũ.
- Giữ nguyên output schema, loss, memory, stochastic inference và evaluation.
- Dùng `scripts.train` với `initialization_checkpoint_path` cho Stage B-only.
- Giữ pilot: SMD `machine-1-6`, O0, seed `6`, window `20`.

---

## Summary

Code hiện chỉ nhận `task_specific_concat_projection` và `learnable_sigmoid_scalars`. Hai mode này có hai khối fusion hoạt động. Direct routing chưa tồn tại và sẽ không tạo hoặc gọi hai khối đó.

## Current state

- `ActiveRuntimeConfig` từ chối `direct_branch_routing`.
- `_compute_fusion_outputs()` đang tạo hidden fusion cho hai task.
- `_build_sampled_fusion_hidden()` đang fusion các tensor Monte Carlo.
- Model luôn đăng ký projection và gate modules.
- Stage B freeze gate/scalar nhưng còn để concat projections trainable.
- `scripts.train` đã có đường nạp `initialization_checkpoint_path`.
- Two-stage runner luôn gọi Stage A trước Stage B.

## Desired end state

- Mode mới được chấp nhận qua khóa `fusion_mode`.
- Direct deterministic path không gọi hai khối fusion.
- Continuous latent đi vào reconstruction head; discrete latent đi vào classification head.
- Monte Carlo path dùng cùng quy tắc cho từng sample.
- Projection, gate, `alpha_logit` và `beta_logit` chỉ còn là module tương thích checkpoint; chúng không tham gia forward hoặc gradient trong direct Stage B.
- Checkpoint cũ vẫn load được.
- Có một config Stage-B-only riêng, không có khối `two_stage`.

## Scope

### In scope

- Runtime validation, deterministic routing, Monte Carlo routing.
- Freeze module không dùng và kiểm tra checkpoint keys.
- Một YAML pilot Stage-B-only.
- Test nhỏ cho các hợp đồng trên.

### Out of scope

- Stage A, memory re-initialization và full benchmark.
- Changes to online adaptation configs or online benchmark execution.
- Thay đổi loss, metric, threshold, preprocessing hoặc two-stage runner.

## Evidence

- `src/models/thesis_multitask_impl/thesis_multitask_components.py:210-239` — validation của `fusion_mode`.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:14-67` — sampled fusion.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:138-270` — Monte Carlo heads.
- `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:357-461` — deterministic fusion.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:95-177` — trainability.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py:293-342` — projection, gate và task heads.
- `scripts/cli/train.py:81-92,181-228` — load initialization checkpoint.
- `src/core/config.py:747-807` — merge `model_overrides`.
- `scripts/experiments/run_two_stage_offline_pretraining.py:364-460` — runner luôn chạy Stage A.
- `configs/experiment/offline_benchmark/thesis/smd__thesis__offline__O0__machine_1_6__w20__seed6__main.yaml` — config pilot hiện tại.

## Terminology mapping

| Existing term | Direct-ablation term | Status | Runtime owner | Evidence |
|---|---|---|---|---|
| `task_specific_concat_projection` | same name | unchanged, not used in direct mode | `_compute_fusion_outputs` | `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:365-397` |
| `learnable_sigmoid_scalars` | same name | unchanged, not used in direct mode | `_compute_fusion_outputs` | `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py:398-460` |
| `phase_direct_passthrough` | not renamed to direct routing | unchanged, separate Stage-A concept | Stage-A passthrough mixin | `src/models/thesis_multitask_impl/thesis_multitask_state_passthrough_mixin.py:43-63` |
| `fusion head` | inactive compatibility modules in direct mode | unchanged name, inactive in direct architecture | projection/gate group | `documents/spec/offline_pretraining_terminology_ontology.md:289-296` |
| no existing mode | `direct_branch_routing` | new | deterministic and MC routing helpers | this plan |
| `stage_b_init.pt` | same filename | unchanged | checkpoint initialization | `scripts/experiments/run_two_stage_offline_pretraining.py:244-336` |

## Implementation approach

Dùng các helper đang có. Mode mới trả hai branch tensor trực tiếp và không tạo khối fusion mới. Module projection/gate cũ vẫn đăng ký để giữ state-dict keys, nhưng direct Stage B freeze và bypass chúng. Config mới gọi `scripts.train` độc lập để nạp `stage_b_init.pt`; không sửa two-stage runner.

## Phase 1: Thêm direct routing vào model

### Goal

Model chấp nhận mode mới và không thực hiện fusion trong deterministic path.

### Sequential stages

#### Stage 1.1 — Tạo test deterministic thất bại

- **File:** `tests/models/test_direct_branch_routing.py` (mới)
- **Action:** Dùng hai tensor khác nhau, gọi `_compute_fusion_outputs`, kiểm tra direct mode và metadata.
- **Tools:** `apply_patch`, `.venv/bin/python -m pytest`.
- **Expected:** Test thất bại vì mode chưa hợp lệ.

**Atomic steps:**

- [ ] Tạo model fixture nhỏ trong `tests/models/test_direct_branch_routing.py`.
- [ ] Tạo hai input tensor có giá trị khác nhau.
- [ ] Gọi `_compute_fusion_outputs` với `fusion_mode="direct_branch_routing"`.
- [ ] Chạy test targeted và ghi nhận lỗi mode chưa được hỗ trợ.

#### Stage 1.2 — Thêm mode và route deterministic

- **Files:** `src/models/thesis_multitask_impl/thesis_multitask_components.py`, `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`.
- **Symbols:** `ActiveRuntimeConfig.__post_init__`, `_compute_fusion_outputs`.
- **Action:** Cho phép `direct_branch_routing`; trả `continuous_hidden` cho reconstruction và `discrete_hidden` cho classification; giữ `alpha`/`beta` dạng tensor zero.
- **Tools:** `apply_patch`.
- **Expected:** Deterministic test pass.

**Atomic steps:**

- [ ] Thêm `direct_branch_routing` vào tập giá trị hợp lệ của `ActiveRuntimeConfig`.
- [ ] Thêm nhánh direct trong `_compute_fusion_outputs`.
- [ ] Gán continuous tensor cho reconstruction.
- [ ] Gán discrete tensor cho classification.
- [ ] Trả metadata fusion direct cùng tensor zero cho `alpha` và `beta`.
- [ ] Chạy lại deterministic test.

#### Stage 1.3 — Kiểm tra mode cũ

- **Action:** Chạy các test shape/config hiện có cho hai fusion mode cũ.
- **Tools:** `.venv/bin/python -m pytest`.
- **Expected:** Không có regression.

**Atomic steps:**

- [ ] Chạy test config hiện có.
- [ ] Chạy test shape cho `task_specific_concat_projection`.
- [ ] Chạy test shape cho `learnable_sigmoid_scalars`.
- [ ] Xác nhận chỉ direct mode có hành vi mới.

### Changes

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_components.py`
- **Symbol:** `ActiveRuntimeConfig.__post_init__`
- **Change:** Thêm `direct_branch_routing` vào tập giá trị hợp lệ và cập nhật thông báo lỗi.
- **Reason:** Đây là runtime contract hiện tại.

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`
- **Symbol:** `_compute_fusion_outputs`
- **Change:** Trả branch tensor trực tiếp và ghi `aux["fusion"]["fusion_mode"]` là `direct_branch_routing`.
- **Reason:** Helper này nằm ngay trước hai task heads.

### Verification

- [x] `.venv/bin/python -m pytest -q tests/models/test_direct_branch_routing.py` — deterministic direct test pass.
- [x] `.venv/bin/python -m pytest -q tests/models/test_thesis_multitask_config_refactor.py tests/models/test_multitask_shapes.py -k 'not shared_three_layer_mlp_depth'` — mode cũ không regression.

## Phase 2: Đồng bộ Monte Carlo và trainability

### Goal

Direct mode có cùng hành vi trong stochastic evaluation và không cập nhật module fusion cũ.

### Sequential stages

#### Stage 2.1 — Tạo test sampled routing

- **File:** `tests/models/test_direct_branch_routing.py`.
- **Action:** Dùng tensor `[B, M, L, H]`, kiểm tra hai output sampled giữ nguyên branch tương ứng.
- **Tools:** `apply_patch`, `.venv/bin/python -m pytest`.
- **Expected:** Test thất bại trước khi sửa helper.

**Atomic steps:**

- [ ] Tạo continuous samples có shape `[B, M, L, H]`.
- [ ] Tạo discrete samples có cùng shape nhưng giá trị khác.
- [ ] Gọi `_build_sampled_fusion_hidden` ở direct mode.
- [ ] Chạy test targeted và ghi nhận lỗi hoặc output chưa đúng.

#### Stage 2.2 — Sửa sampled route và freeze

- **Files:** `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`, `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`.
- **Symbols:** `_build_sampled_fusion_hidden`, `_configure_trainable_parameters_for_phase`.
- **Action:** Trả branch samples trực tiếp; freeze hai concat projection, hai fusion gate, `alpha_logit` và `beta_logit` trong direct Stage B.
- **Tools:** `apply_patch`.
- **Expected:** Sampled route và trainability tests pass.

**Atomic steps:**

- [ ] Thêm nhánh direct trả continuous samples cho reconstruction.
- [ ] Thêm nhánh direct trả discrete samples cho classification.
- [ ] Thêm điều kiện freeze cho hai projection trong direct Stage B.
- [ ] Thêm điều kiện freeze cho hai gate trong direct Stage B.
- [ ] Đặt `requires_grad=False` cho `alpha_logit`.
- [ ] Đặt `requires_grad=False` cho `beta_logit`.
- [ ] Chạy lại sampled routing test.
- [ ] Chạy lại trainability test.

#### Stage 2.3 — Chạy test MC hiện có

- **Action:** Chạy test Monte Carlo hiện có và test direct mới.
- **Tools:** `.venv/bin/python -m pytest`.
- **Expected:** MC mean, uncertainty và shape schema vẫn pass.

**Atomic steps:**

- [ ] Chạy test MC mean và uncertainty.
- [ ] Chạy test single-sample MC.
- [ ] Kiểm tra output schema không đổi.

### Changes

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_routing_geometry_helpers.py`
- **Symbol:** `_build_sampled_fusion_hidden`
- **Change:** Trả `continuous_samples` cho reconstruction và `discrete_samples` cho classification trong direct mode.
- **Reason:** MC evaluation phải dùng cùng routing với deterministic path.

- **File:** `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`
- **Symbol:** `_configure_trainable_parameters_for_phase`
- **Change:** Freeze module fusion và hai scalar trong direct Stage B; không xóa module.
- **Reason:** Direct path không dùng chúng nhưng checkpoint vẫn cần keys cũ.

### Verification

- [x] `.venv/bin/python -m pytest -q tests/models/test_direct_branch_routing.py` — sampled route và trainability pass.
- [x] `.venv/bin/python -m pytest -q tests/models/test_multitask_shapes.py -k 'returns_monte_carlo_means_and_uncertainty_in_eval_mode or eval_without_stochastic_inference_is_safe or handles_single_sample_monte_carlo_without_nan'` — MC contract pass.

## Phase 3: Chuẩn bị checkpoint và config Stage-B-only

### Goal

Direct model load được checkpoint cũ và có YAML riêng để chạy Stage B mà không gọi Stage A.

### Sequential stages

#### Stage 3.1 — Kiểm tra state-dict compatibility

- **File:** `tests/models/test_direct_branch_routing.py`.
- **Action:** Lưu state-dict của model cũ, load vào model direct cùng kiến trúc bằng `CheckpointManager`.
- **Tools:** `apply_patch`, `.venv/bin/python -m pytest`.
- **Expected:** Không có missing hoặc unexpected keys.

**Atomic steps:**

- [ ] Tạo model Stage B dùng mode cũ.
- [ ] Lưu checkpoint tạm bằng `CheckpointManager`.
- [ ] Tạo model Stage B dùng direct mode với cùng kiến trúc.
- [ ] Load checkpoint vào direct model bằng strict mode.
- [ ] Kiểm tra missing keys và unexpected keys đều rỗng.

#### Stage 3.2 — Thêm config pilot

- **File:** `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml` (mới).
- **Action:** Giữ config O0 hiện tại, đặt `epochs: 5`, `experiment_variant: direct_branch_routing_v1`, hai `model_overrides` cho mode và Stage B, dùng initialization path O0 hiện tại, và ghi output vào `outputs/benchmark/smd/machine_1_6/seed6/thesis_direct_branch_routing/offline/stage_b`.
- **Tools:** `apply_patch`.
- **Expected:** YAML không có khối `two_stage`.

**Atomic steps:**

- [ ] Sao chép các tham chiếu data, model và task của O0 pilot.
- [ ] Đặt `epochs` bằng `5`.
- [ ] Đặt `fusion_mode` trong `model_overrides`.
- [ ] Đặt `training_phase` trong `model_overrides`.
- [ ] Đặt variant và output directory riêng.
- [ ] Xóa khối `two_stage` khỏi config mới.

#### Stage 3.3 — Kiểm tra config loader

- **Action:** Load YAML bằng `load_experiment_config` và kiểm tra mode, phase, epoch, checkpoint path, output path.
- **Tools:** `.venv/bin/python -m pytest`.
- **Expected:** Config validate mà không cần dataset hoặc checkpoint thật.

**Atomic steps:**

- [ ] Gọi `load_experiment_config` với config direct mới.
- [ ] Kiểm tra resolved `fusion_mode`.
- [ ] Kiểm tra resolved `training_phase` và `epochs`.
- [ ] Kiểm tra initialization path và output path.
- [ ] Kiểm tra key `two_stage` không tồn tại.

### Changes

- **File:** `tests/models/test_direct_branch_routing.py`
- **Symbol:** checkpoint compatibility và config-loading tests
- **Change:** Kiểm tra strict state-dict load và resolved YAML.
- **Reason:** Đây là hai điều kiện trước khi chạy remote.

- **File:** `configs/experiment/offline_ablation/thesis/smd__thesis__offline__direct_branch_routing__machine_1_6__w20__seed6__stage_b.yaml`
- **Symbol:** top-level experiment YAML
- **Change:** Tạo config Stage-B-only với `initialization_checkpoint_path` là `outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt` và output path riêng.
- **Reason:** `scripts.train` đã hỗ trợ nạp checkpoint trước training; two-stage runner thì luôn gọi Stage A.

### Verification

- [x] `.venv/bin/python -m pytest -q tests/models/test_direct_branch_routing.py` — checkpoint và config tests pass.
- [ ] `test -f outputs/benchmark/smd/thesis/O0/machine_1_6/seed6/two_stage/initializations/stage_b_init.pt` — chỉ kiểm tra trước remote training; nếu thiếu thì dừng.

## Phase 4: Kiểm tra local và handoff

### Goal

Local code pass test cần thiết và sẵn sàng để anh push lên remote.

### Sequential stages

#### Stage 4.1 — Chạy một forward/backward nhỏ

- **File:** `tests/models/test_direct_branch_routing.py`.
- **Action:** Chạy một batch CPU qua direct Stage B và loss hiện có; kiểm tra loss hữu hạn, task heads có gradient, module fusion không có gradient.
- **Tools:** `apply_patch`, `.venv/bin/python -m pytest`.
- **Expected:** Pass; không gọi Stage A.

**Atomic steps:**

- [ ] Tạo một batch nhỏ theo batch contract hiện tại.
- [ ] Chạy forward bằng direct Stage B trên CPU.
- [ ] Tính loss bằng loss path hiện có.
- [ ] Chạy backward một lần.
- [ ] Kiểm tra loss hữu hạn và gradient của task heads.
- [ ] Kiểm tra module fusion không có gradient.

#### Stage 4.2 — Chạy regression set tối thiểu

- **Action:** Chạy test direct mới, config refactor, dry-run orchestration và test shape/MC liên quan.
- **Tools:** `.venv/bin/python -m pytest`.
- **Expected:** Không regression.

**Atomic steps:**

- [ ] Chạy toàn bộ test direct mới.
- [ ] Chạy test config refactor.
- [ ] Chạy test dry-run orchestration.
- [ ] Chạy test shape đã chọn.
- [ ] Chạy test MC đã chọn.
- [ ] Ghi lại test failure nếu có.

#### Stage 4.3 — Kiểm tra diff

- **Action:** Chạy `git diff --check` và `git status --short`.
- **Tools:** `git diff --check`, `git status --short`.
- **Expected:** Chỉ có source, test, config và plan dự kiến.

**Atomic steps:**

- [ ] Chạy `git diff --check`.
- [ ] Đọc danh sách file từ `git status --short`.
- [ ] Xác nhận không có checkpoint hoặc artifact remote trong diff.
- [ ] Xác nhận không có lệnh Stage A trong thay đổi local.

### Changes

- **File:** `tests/models/test_direct_branch_routing.py`
- **Symbol:** one-batch direct Stage-B smoke test
- **Change:** Kiểm tra forward, backward, gradient và output schema.
- **Reason:** Xác nhận public model và loss chạy cùng nhau ở local.

- **File:** config direct mới
- **Symbol:** comments quanh `initialization_checkpoint_path` và `model_overrides`
- **Change:** Ghi ngắn rằng config này chạy Stage B độc lập và không dùng two-stage runner.
- **Reason:** Tránh chạy nhầm Stage A sau khi push remote.

### Verification

- [x] `.venv/bin/python -m pytest -q tests/models/test_direct_branch_routing.py` — test direct pass.
- [x] `.venv/bin/python -m pytest -q tests/models/test_thesis_multitask_config_refactor.py tests/benchmarks/test_two_stage_orchestration_dry_run.py` — test nền pass.
- [x] `git diff --check` — không có whitespace error.

## Testing strategy

Test mới tập trung vào mode, branch identity, MC identity, trainability, checkpoint keys và config resolution. Không chạy dataset training, Stage A, benchmark hoặc SSH trong phiên local.

## Migration and rollback

Không có migration dữ liệu. Hai fusion mode cũ và config baseline không đổi. Nếu checkpoint direct không load được, dừng trước training và kiểm tra state-dict keys; không dùng `strict=False` để che lỗi.

Rollback bằng cách revert mode mới, test mới và config mới. Checkpoint cũ vẫn load được vì module names cũ không bị xóa.

## Final verification

- [x] Direct architecture không gọi hai fusion block.
- [x] Continuous branch đi vào reconstruction head.
- [x] Discrete branch đi vào classification head.
- [x] Monte Carlo path dùng cùng routing.
- [x] Checkpoint cũ load được.
- [x] Config Stage-B-only không gọi Stage A.
- [x] Không chạy remote trong phiên local này.

## Assumptions and non-blocking uncertainties

- Pilot vẫn là O0, SMD `machine-1-6`, seed `6`, window `20`.
- `stage_b_init.pt` hiện chưa có trong local; chỉ kiểm tra sự tồn tại trước khi chạy Stage B sau này.
- Online path không thuộc phạm vi hiện tại; adapter hiện tại sẽ tự dùng mode direct nếu sau này nạp direct checkpoint.
