# Two-Stage Offline Pretraining Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đóng các gap còn thiếu giữa `documents/design/two-stage-offline-pretraining-spec.md` và code hiện tại, ưu tiên bổ sung point-wise balanced reconstruction-score loss cho Stage A mà không làm vỡ contract two-stage base đang chạy được.

**Architecture:** Giữ nguyên runtime two-stage hiện tại làm đường cơ sở. Bổ sung một nhánh score-loss có thể bật/tắt bằng config, implement theo kiểu adapter cục bộ để map ý nghĩa `x_clean` / `x_input` vào batch contract hiện có, và để evaluator tiếp tục dùng `point_scores` / thresholding hiện tại. Không đổi sang kiến trúc mới lớn hơn mức cần thiết.

**Tech Stack:** Python 3.12, PyTorch, PyYAML, pytest, existing thesis-multitask model stack, current runner/evaluator/checkpoint utilities.

---

## Current State

- `src/models/thesis_multitask_loss_mixin.py` đã có reconstruction loss, classification loss, two-view contrastive loss, và một số optional regularizers nội bộ.
- `src/models/thesis_multitask_routing_mixin.py` đã xuất `point_scores` và `window_scores`, nhưng đó là output phục vụ evaluation, không phải training objective.
- `src/engine/evaluator.py` đã có thresholding, VUS-PR, VUS-ROC, affiliation F1, và hợp nhất điểm theo timeline.
- `src/core/config.py` và `src/core/config_model_validation.py` đã có schema cho two-stage base, nhưng chưa có các key mới của point-score-supervised variant.
- `configs/model/thesis_multitask_two_stage_window20.yaml` đã có, nhưng các exp4 YAML trong `configs/experiment/thesis/exp4/` hiện đang thiếu trong checkout này.

## Design Options

- Option A, recommended: giữ batch runtime contract hiện tại, thêm score-loss adapter nội bộ trong model để suy ra nghĩa `x_clean` / `x_input` từ `batch["x"]` và `batch["synthetic_anomaly_mask"]`, rồi bật/tắt bằng config variant. Ưu điểm là diff nhỏ, giữ compatibility, ít rủi ro phá base run.
- Option B: migrate toàn bộ data/model contract sang `x_clean` / `x_input` / `class_labels` / `is_synthetic`. Ưu điểm là khớp spec văn bản hơn, nhưng sẽ đụng nhiều loader, trainer, test, và dễ tạo regression.
- Option C: chỉ bổ sung score metrics ở evaluation và không thêm training-side score-loss. Ưu điểm là an toàn, nhưng không đạt mục tiêu của spec point-score-supervised run.

**Recommendation:** Chọn Option A. Đây là đường ngắn nhất để đóng gap thực sự còn thiếu mà không làm hỏng đường base two-stage hiện tại.

## Risk and Mitigation

- Risk: point-score-supervised logic làm thay đổi base two-stage. Mitigation: default `enable_score_loss = false`, tách rõ base và point-score variant trong config/test.
- Risk: score-loss target dùng sai nguồn label, ví dụ window label thay vì point mask. Mitigation: derive target trực tiếp từ `synthetic_anomaly_mask` và test một batch synthetic cụ thể.
- Risk: batch contract spec và runtime contract bị lẫn. Mitigation: không đổi batch-wide keys trong giai đoạn này; thêm helper cục bộ để map semantic views của score-loss.
- Risk: config schema drift. Mitigation: thêm test load config cho cả base và point-score variants, kể cả smoke YAML.
- Risk: evaluation metric inflation. Mitigation: re-use evaluator hiện có và chỉ so sánh cùng một checkpoint/threshold policy giữa base và point-score variant.

## Open Questions

- Không có blocker thiết kế mới nào ở mức plan này. Quyết định đã chọn là giữ runtime batch contract hiện tại và thêm compatibility adapter nội bộ cho score-loss.

---

### Task 1: Extend config schema for the point-score variant

**Files:**
- Modify: `src/core/config.py`
- Modify: `src/core/config_model_validation.py`
- Modify: `src/models/thesis_multitask_components.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Test: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Test: `tests/test_thesis_multitask_config_refactor.py`

- [ ] **Step 1: Write the failing config tests**

Add assertions that a point-score-supervised experiment config can load and that the following keys survive validation and model config construction:

```python
assert loaded_config["experiment_variant"] == "two_stage_point_score_supervised_v1"
assert loaded_config["model"]["enable_score_loss"] is True
assert loaded_config["model"]["score_loss_granularity"] == "point"
assert loaded_config["model"]["score_loss_type"] == "pointwise_balanced_reconstruction_score"
assert loaded_config["model"]["score_loss_target"] == "synthetic_anomaly_mask"
```

Also add a negative test that base two-stage configs keep `enable_score_loss` false by default and still validate.

- [ ] **Step 2: Run the config tests and confirm the current failure mode**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_thesis_multitask_config_refactor.py
```

Expected: failures for missing keys until the schema is extended.

- [ ] **Step 3: Implement the schema extension**

Add the new score-loss fields to the thesis multitask config dataclasses and validator so the repository accepts the following minimal contract:

```python
enable_score_loss: bool = False
score_loss_granularity: str = "point"
score_loss_type: str = "pointwise_balanced_reconstruction_score"
score_loss_target: str = "synthetic_anomaly_mask"
score_loss_normalization: str = "batch_normal_tokens_detached_mean_std"
score_loss_reduction: str = "pointwise_binary_balanced_mean"
```

Keep the validation narrow. Reject unknown score-loss modes instead of silently accepting arbitrary strings.

- [ ] **Step 4: Re-run the config tests**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_thesis_multitask_config_refactor.py
```

Expected: pass.

---

### Task 2: Implement Stage A point-wise balanced reconstruction-score loss

**Files:**
- Modify: `src/models/thesis_multitask_loss_mixin.py`
- Modify: `src/models/thesis_multitask_routing_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Test: `tests/test_offline_pretraining_two_stage_runner.py`
- Test: `tests/test_thesis_multitask_cnn_shapes.py`
- Test: `tests/test_one_multitask_train_step.py`
- New test: `tests/test_thesis_multitask_point_score_loss.py`

- [ ] **Step 1: Write the failing loss test**

Add a focused test around a synthetic batch with injected anomalies that checks two things:

```python
assert "point_scores" in outputs
assert loss_terms["score_loss"] > 0.0
```

Then add a complementary test for the base run:

```python
assert loss_terms["score_loss"] == 0.0
```

The test should use the existing `synthetic_anomaly_mask` contract, not a new global batch schema.

- [ ] **Step 2: Run the loss test and confirm it fails**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_thesis_multitask_point_score_loss.py
```

Expected: failure because the model does not yet compute a fourth Stage A loss term.

- [ ] **Step 3: Implement the minimal score-loss helper**

Add a local helper in `ThesisMultitaskLossMixin` that:

1. derives point-wise targets from `synthetic_anomaly_mask`,
2. computes point-wise reconstruction scores from `outputs["recon"]` and `batch["x"]`,
3. normalizes scores using only normal tokens in the current batch,
4. computes a balanced point-wise loss,
5. returns zero when `enable_score_loss` is false.

Use the current runtime batch contract:

```python
batch["x"]
batch["synthetic_anomaly_mask"]
batch["classification_labels"]
```

Do not rename the whole pipeline to `x_clean` / `x_input` in this cycle. Keep the compatibility adapter local to the score-loss path.

- [ ] **Step 4: Route the new loss into Stage A only**

Modify `_compute_optional_loss_terms(...)` or `_shared_step(...)` so that:

```python
if self.training_phase == TWO_STAGE_A_PHASE_NAME and self.enable_score_loss:
    total_loss = total_loss + score_loss_weight * score_loss
```

Keep Stage B unaffected.

Also make sure the logged stage metrics expose the new score loss for train and synthetic validation if the variant is enabled.

- [ ] **Step 5: Re-run the loss and stage-step tests**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_thesis_multitask_point_score_loss.py tests/test_one_multitask_train_step.py tests/test_thesis_multitask_cnn_shapes.py
```

Expected: pass, with base two-stage still returning the original 3-loss objective and the point-score variant returning 4 losses.

---

### Task 3: Restore the exp4 two-stage experiment configs and wire the variant names

**Files:**
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-point-score-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-point-score-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Modify: `scripts/run_two_stage_offline_pretraining.py`
- Test: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Test: `tests/test_offline_pretraining_two_stage_runner.py`

- [ ] **Step 1: Write the failing YAML-loading tests**

Add tests that load the four exp4 files above and assert the variant split:

```python
assert loaded_config["two_stage"]["stage_a_multitask_epochs"] == 80
assert loaded_config["two_stage"]["stage_b_fusion_finetuning_epochs"] == 20
assert loaded_config["model"]["enable_score_loss"] is False
```

And for the point-score variant:

```python
assert loaded_config["experiment_variant"] == "two_stage_point_score_supervised_v1"
assert loaded_config["model"]["enable_score_loss"] is True
assert loaded_config["model"]["score_loss_granularity"] == "point"
```

- [ ] **Step 2: Run the config-loading tests and confirm they fail on missing files**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py
```

Expected: failure until the exp4 YAMLs exist in the repo.

- [ ] **Step 3: Add the YAML files with minimal duplication**

Use the existing two-stage base config as the template, then create the point-score variant by changing only the explicit variant markers:

```yaml
experiment_variant: two_stage_point_score_supervised_v1
model:
  enable_score_loss: true
  score_loss_granularity: point
  score_loss_type: pointwise_balanced_reconstruction_score
```

Keep the smoke config as the exact 5-epoch split already used by the current tests.

- [ ] **Step 4: Ensure the runner preserves the variant**

If `scripts/run_two_stage_offline_pretraining.py` strips unknown metadata or drops variant keys when generating stage configs, keep the variant keys round-trippable so the generated manifest still identifies the point-score variant unambiguously.

- [ ] **Step 5: Re-run the runner tests**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_offline_pretraining_two_stage_config_loading.py tests/test_offline_pretraining_two_stage_runner.py
```

Expected: pass, and the runner still materializes Stage A, Stage B init checkpoint, Stage B config, and evaluation command in the correct order.

---

### Task 4: Lock the new contract with end-to-end smoke verification

**Files:**
- Modify: `tests/test_config_loading.py`
- Modify: `tests/test_trainer_checkpoint_fallback.py`
- Modify: `tests/test_redlamp_realistic_validation_alignment.py`
- Possibly modify: `tests/test_one_multitask_train_step.py`

- [ ] **Step 1: Add a contract test for score-loss off vs on**

Add a test that compares base and point-score configs and asserts:

```python
assert base_model.enable_score_loss is False
assert point_score_model.enable_score_loss is True
assert base_model._phase_uses_contrastive_objective() is True
assert point_score_model._phase_uses_contrastive_objective() is True
```

The purpose is to ensure the new loss does not disturb the existing Stage A topology.

- [ ] **Step 2: Add a checkpoint metadata test**

Verify that the checkpoint metadata still records the evaluation threshold and that the new variant does not break threshold resolution:

```python
assert outputs["metrics"]["threshold_source"].startswith("checkpoint::")
```

- [ ] **Step 3: Run the full narrow verification set**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_offline_pretraining_two_stage_config_loading.py \
  tests/test_offline_pretraining_two_stage_runner.py \
  tests/test_thesis_multitask_point_score_loss.py \
  tests/test_one_multitask_train_step.py \
  tests/test_trainer_checkpoint_fallback.py
```

Expected: pass.

- [ ] **Step 4: Dry-run the two-stage runner**

Run:

```bash
./.venv/bin/python scripts/run_two_stage_offline_pretraining.py \
  --experiment-config configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml \
  --dry-run
```

Expected: manifest and execution report are written, Stage A comes first, Stage B init checkpoint path is present, and no training is launched during dry-run.

---

## Acceptance Criteria

- Base two-stage still runs with the original 3-loss Stage A objective.
- Point-score-supervised variant can be loaded from config and exposes the new score-loss contract.
- Score-loss uses point-wise synthetic anomaly masks, not window labels.
- Evaluation still uses `point_scores`, timeline merging, thresholding, VUS-PR, VUS-ROC, and affiliation F1.
- The repo has real exp4 YAML entrypoints again, so the runner tests stop failing on `FileNotFoundError`.
- No unrelated model or baseline refactor is introduced.

## Suggested Execution Order

1. Task 1: schema and config acceptance.
2. Task 2: score-loss implementation in the model.
3. Task 3: restore experiment YAMLs and keep the variant explicit.
4. Task 4: lock the behavior with tests and dry-run verification.

