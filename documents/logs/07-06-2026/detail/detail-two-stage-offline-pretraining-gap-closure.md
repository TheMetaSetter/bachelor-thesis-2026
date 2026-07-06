# Two-Stage Offline Pretraining Gap Closure Detailed Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đóng các gap còn thiếu giữa spec two-stage offline pretraining và code hiện tại, ưu tiên bổ sung point-wise balanced reconstruction-score loss cho Stage A, khôi phục exp4 two-stage experiment configs, và khóa lại contract bằng test mà không làm vỡ đường base two-stage đang hoạt động.

**Architecture:** Giữ nguyên runtime two-stage hiện tại làm baseline. Thêm một score-loss adapter nội bộ trong model để map ý nghĩa `x_clean` / `x_input` vào contract batch hiện hữu, và mở rộng config/runner tối thiểu để nhận variant point-score một cách tường minh. Không đổi kiến trúc batch-wide sang schema mới trong cùng đợt này; thay vào đó, giữ stable interfaces và chỉ mở rộng theo đúng minimal vertical slice.

**Tech Stack:** Python 3.12, PyTorch, PyYAML, pytest, existing thesis-multitask model stack, current runner/evaluator/checkpoint utilities, repository-local config dataclasses.

---

## Phase 1: Config and Variant Contract Alignment

### Phase Summary
Phase này chuẩn hóa contract cấu hình để code nhận được hai biến thể rõ ràng: base two-stage và point-score-supervised two-stage. Mục tiêu là giữ compatibility với batch/runtime contract hiện tại nhưng cho phép bật score-loss bằng config thay vì hard-code. Đây là bước nền để các phase sau có thể đọc đúng ý nghĩa của variant và không làm lẫn base run với experimental run.

### File-Level Edits

- Modify: `src/core/config.py`
- Modify: `src/core/config_model_validation.py`
- Modify: `src/models/thesis_multitask_components.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Modify: `tests/test_thesis_multitask_config_refactor.py`
- Modify: `tests/test_config_loading.py`

### Explicit Edit Content

- Thêm các field score-loss tối thiểu vào schema model config:

```python
enable_score_loss: bool = False
score_loss_granularity: str = "point"
score_loss_type: str = "pointwise_balanced_reconstruction_score"
score_loss_target: str = "synthetic_anomaly_mask"
score_loss_normalization: str = "batch_normal_tokens_detached_mean_std"
score_loss_reduction: str = "pointwise_binary_balanced_mean"
```

- Cho phép `load_experiment_config(...)` và validator chấp nhận `experiment_variant` ở top-level để phân biệt base run và point-score run.
- Giữ nguyên `training_phase`, `stage_name`, `discrete_query_mode`, `freeze_memories_after_initialization`, và `discrete_memory_label_source` đang có.
- Ở `ThesisMultitaskModelConfig.from_flat_kwargs(...)`, map các field score-loss mới vào group phù hợp để constructor flat kwargs và constructor dataclass cùng ra một runtime contract.
- Không đổi tên batch keys của runtime hiện tại trong phase này.

### Interface and Contract Definitions

- Config contract:

```python
experiment_config = {
    "experiment_variant": "two_stage_base_v1" | "two_stage_point_score_supervised_v1",
    "two_stage": {...},
    "model": {
        "training_phase": "stage_a_multitask_pretraining",
        "enable_score_loss": bool,
        "score_loss_granularity": "point",
        "score_loss_type": "pointwise_balanced_reconstruction_score",
        "score_loss_target": "synthetic_anomaly_mask",
    },
}
```

- Backward compatibility rule:
  - If `enable_score_loss` is absent, treat it as `False`.
  - Base two-stage YAMLs must still validate without any score-loss keys.

### Design Patterns

- Adapter pattern: map the new score-loss meaning vào contract batch hiện có mà không đổi toàn bộ batch schema.
- Stable interface: model config extension phải không làm hỏng constructor hiện tại cho `ThesisMultitaskModel`.
- Composition over inheritance: keep score-loss as config-controlled behavior, not a new model subclass.

### Risk Mitigation

- Prototype redundancy risk: không đụng continuous/discrete branch logic ở phase này.
- Fusion collapse risk: không sửa fusion math ở phase này để tránh kéo theo regression.
- Evaluation inflation risk: không thay evaluation metric, chỉ mở rộng config.

### Test Plan and Validation

- Add assertions that base two-stage config still validates with no score-loss keys.
- Add assertions that point-score config exposes all new score-loss keys after load.
- Add one negative test that rejects unknown score-loss mode strings.

### Acceptance Criteria

- `load_experiment_config(...)` đọc được base and point-score variants.
- Validator chấp nhận score-loss keys mới nhưng vẫn reject unknown modes.
- Base configs vẫn pass mà không cần chỉnh YAML hiện hữu.

---

## Phase 2: Stage A Score-Loss Implementation

### Phase Summary
Phase này bổ sung training-side point-wise balanced reconstruction-score loss vào Stage A. Core objective là tạo thêm một term loss thứ tư, nhưng chỉ bật cho point-score variant và chỉ dùng dữ liệu trong batch hiện tại. Base two-stage vẫn chạy đúng 3-loss objective cũ.

### File-Level Edits

- Modify: `src/models/thesis_multitask_loss_mixin.py`
- Modify: `src/models/thesis_multitask_routing_mixin.py`
- Modify: `src/models/thesis_multitask_setup_mixin.py`
- Modify: `tests/test_one_multitask_train_step.py`
- Modify: `tests/test_thesis_multitask_cnn_shapes.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Add: `tests/test_thesis_multitask_point_score_loss.py`

### Explicit Edit Content

- Thêm helper mới trong `ThesisMultitaskLossMixin` để tính score-loss:

```python
def _compute_point_score_loss(self, outputs: dict[str, Any], batch: dict[str, Any]) -> torch.Tensor:
    if not self.enable_score_loss:
        return self._zero_loss(outputs["recon"])
    if self.training_phase != TWO_STAGE_A_PHASE_NAME:
        return self._zero_loss(outputs["recon"])
    if "synthetic_anomaly_mask" not in batch:
        return self._zero_loss(outputs["recon"])

    point_scores = torch.mean((outputs["recon"] - batch["x"]) ** 2, dim=-1)
    point_labels = batch["synthetic_anomaly_mask"]
    if point_labels.ndim == 3:
        point_labels = point_labels.any(dim=-1)
    point_labels = point_labels.float()

    normal_mask = point_labels == 0
    anomaly_mask = point_labels == 1
    if int(normal_mask.sum().item()) == 0 or int(anomaly_mask.sum().item()) == 0:
        return self._zero_loss(outputs["recon"])

    normal_scores = point_scores[normal_mask]
    score_mean = normal_scores.mean().detach()
    score_std = normal_scores.std(unbiased=False).detach().clamp_min(self.epsilon)
    normalized_scores = (point_scores - score_mean) / score_std
    return self._balanced_point_score_objective(normalized_scores, point_labels)
```

- Keep this logic local to the model file and route it through the existing `_shared_step(...)` path.
- Add the score-loss term to `loss_terms` and to logged train / val_synth metrics only when the variant enables it.
- Use current runtime batch contract:

```python
batch["x"]
batch["classification_labels"]
batch["synthetic_anomaly_mask"]
```

- Do not rename the whole pipeline to `x_clean` / `x_input` in this cycle.
- Preserve base loss assembly:

```python
total_loss = recon_weight * recon_loss + cls_weight * cls_loss + contrastive_weight * contrastive_loss + optional_terms
```

### Interface and Contract Definitions

- Training contract for this phase:

```python
loss_terms = {
    "total_loss": Tensor[],
    "reconstruction_loss": Tensor[],
    "classification_loss": Tensor[],
    "contrastive_loss": Tensor[],
    "score_loss": Tensor[],  # zero when disabled
}
```

- Score-loss contract:
  - Derived from point-wise reconstruction MSE on `batch["x"]`.
  - Target labels derived from `synthetic_anomaly_mask`, not window class labels.
  - Normalization uses only normal tokens in the current batch.
  - Must stay zero outside Stage A or when disabled.

### Design Patterns

- Composition over inheritance: helper method in existing loss mixin, not a new subclass.
- Adapter pattern: interpret current batch semantics as the score-loss input view.
- Stable interface: keep `outputs["point_scores"]` and `outputs["window_scores"]` unchanged for evaluation code.

### Risk Mitigation

- Prototype redundancy risk: do not touch prototype branch behavior while adding score-loss.
- Fusion collapse risk: keep fusion gates and weights unchanged to avoid confounding score-loss effects.
- Adaptation contamination risk: score-loss must never consume validation or test batches.
- Projector drift risk: not applicable in this phase, and must not be introduced prematurely.

### Test Plan and Validation

- Add a dedicated unit test with a synthetic anomalous batch that asserts `score_loss > 0` when enabled.
- Add a complementary test that asserts `score_loss == 0` for the base run.
- Add a train-step integration test that checks Stage A logs include `score_loss` only for the point-score variant.
- Re-run shape tests to ensure output tensor shapes remain unchanged:
  - `hidden: [B, L, H]`
  - `recon: [B, L, D]`
  - `point_scores: [B, L]`
  - `window_scores: [B]`

### Acceptance Criteria

- Base two-stage still returns the original 3-loss Stage A objective.
- Point-score variant returns a fourth score-loss term in Stage A only.
- Score-loss uses point-wise anomaly mask semantics, not window labels.
- Model output shapes stay unchanged.

---

## Phase 3: Experiment YAML Restoration and Runner Preservation

### Phase Summary
Phase này khôi phục các entrypoint experiment exp4 bị thiếu và tạo bản point-score variant rõ ràng để runner có thể khởi động lại đúng contract. Mục tiêu là biến implementation thành runnable experiment surface, không chỉ là model-side support. Runner orchestration hiện có phải được giữ ổn định, chỉ thêm variant metadata nếu cần round-trip.

### File-Level Edits

- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-point-score-machine-3-4-window20__w20__seed11__rtx3090.yaml`
- Create: `configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-point-score-machine-3-4-window20-smoke__w20__seed11__smoke.yaml`
- Modify: `scripts/run_two_stage_offline_pretraining.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`

### Explicit Edit Content

- Reintroduce base two-stage YAMLs using the current working model config and current two-stage budget.
- Add point-score variant YAMLs by changing only the explicit variant markers:

```yaml
experiment_variant: two_stage_point_score_supervised_v1
model:
  enable_score_loss: true
  score_loss_granularity: point
  score_loss_type: pointwise_balanced_reconstruction_score
```

- Ensure smoke YAML keeps the exact 5-epoch split used by the current test contract.
- If runner-generated stage configs strip unknown metadata, preserve `experiment_variant` and score-loss keys in the manifest round-trip.
- Keep Stage A -> Stage B -> evaluation ordering unchanged.

### Interface and Contract Definitions

- Runner contract:

```python
manifest = {
    "training_stages": [
        {"stage_name": "stage_a_multitask_pretraining", ...},
        {"stage_name": "stage_b_fusion_finetuning", ...},
    ],
    "evaluation": {
        "config_path": ...,
        "checkpoint_path": ...,
    },
}
```

- Variant contract:
  - Base variant uses score-loss disabled.
  - Point-score variant uses score-loss enabled and the explicit variant name.

### Design Patterns

- Registry/factory style: reuse the existing config-driven experiment builder and do not hard-wire stage execution.
- Stable interface: preserve runner API and manifest shape.
- Minimal vertical slice: start from existing YAMLs and add only the keys needed for the new variant.

### Risk Mitigation

- Evaluation metric inflation risk: keep the same evaluator and threshold policy.
- Batch contract drift risk: do not rename runtime batch keys as part of YAML restoration.
- Prototype redundancy risk: do not alter k-means initialization semantics in runner-level code.

### Test Plan and Validation

- Add YAML loading tests for both base and point-score exp4 configs.
- Add runner tests that assert:
  - Stage A config path is first,
  - Stage B config path is second,
  - Stage B init checkpoint path exists in manifest,
  - evaluation command points to Stage B best checkpoint.
- Confirm missing-file failures are gone once YAMLs are restored.

### Acceptance Criteria

- The exp4 YAMLs exist in the tree and load successfully.
- The runner still generates manifest and execution report without changing stage ordering.
- Base and point-score variants are distinguishable in config and logs.

---

## Phase 4: Test and Contract Locking

### Phase Summary
Phase này khóa lại hành vi mới bằng test mức config, model, runner, checkpoint, và evaluation metadata. Đây là phase bảo vệ stable interfaces: nếu score-loss bật lên mà contract nào đổi ngoài ý muốn, test sẽ chặn ngay. Base run và point-score run phải coexist mà không phá đường cũ.

### File-Level Edits

- Modify: `tests/test_thesis_multitask_point_score_loss.py`
- Modify: `tests/test_offline_pretraining_two_stage_config_loading.py`
- Modify: `tests/test_offline_pretraining_two_stage_runner.py`
- Modify: `tests/test_one_multitask_train_step.py`
- Modify: `tests/test_trainer_checkpoint_fallback.py`
- Modify: `tests/test_redlamp_realistic_validation_alignment.py`

### Explicit Edit Content

- Add a unit test that exercises a synthetic batch with injected anomalies and verifies `score_loss` is positive only when enabled.
- Add a test that checks `checkpoint_monitor_metric` behavior remains unchanged and threshold metadata still round-trips.
- Add an integration assertion that base and point-score models share the same `hidden`, `recon`, `point_scores`, and `window_scores` shapes.
- Add a runner assertion that the point-score manifest still finishes with evaluation on the Stage B best checkpoint.

### Interface and Contract Definitions

- Output contract remains:

```python
{
    "hidden": Tensor[B, L, H],
    "pooled": Tensor[B, ...],
    "recon": Tensor[B, L, D],
    "logits": Tensor[B, C],
    "point_scores": Tensor[B, L],
    "window_scores": Tensor[B],
    "aux": dict,
}
```

- Evaluation contract remains pointwise:
  - threshold selection uses validation scores,
  - metrics operate on timeline-aligned point scores,
  - VUS-PR/VUS-ROC/affiliation F1 remain unchanged.

### Design Patterns

- Stable interface: tests should enforce no shape drift and no metric contract drift.
- Composition over inheritance: verify helper-based score-loss addition does not leak into unrelated model methods.

### Risk Mitigation

- Fusion collapse risk: assert that base path still logs the same fusion-related metrics.
- Adaptation contamination risk: if any future online-adaptation config appears in this branch, keep it out of scope for this plan.
- Metric inflation risk: keep checkpoint selection and threshold source explicit in tests.

### Test Plan and Validation

- Run the narrow set first:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_offline_pretraining_two_stage_config_loading.py \
  tests/test_offline_pretraining_two_stage_runner.py \
  tests/test_thesis_multitask_point_score_loss.py \
  tests/test_one_multitask_train_step.py \
  tests/test_trainer_checkpoint_fallback.py
```

- Then run the pointwise alignment tests if needed:

```bash
./.venv/bin/python -m pytest -q tests/test_redlamp_realistic_validation_alignment.py
```

### Acceptance Criteria

- Base run remains green.
- Point-score variant is covered by direct unit and integration tests.
- Thresholding and evaluation metadata continue to work.

---

## Phase 5: Dry-Run Verification and Reporting

### Phase Summary
Phase cuối cùng xác nhận toàn bộ flow qua dry-run và một tập pytest hẹp. Mục tiêu là chứng minh config mới, runner, Stage A score-loss, và Stage B execution chain đều hoạt động trước khi mở thêm ablation. Đây là bước chốt để tránh claim hoàn thành khi mới chỉ sửa code cục bộ.

### File-Level Edits

- No new production file expected.
- Possibly modify: `documents/logs/07-06-2026/detail/detail-two-stage-offline-pretraining-gap-closure.md` if verification reveals any mismatch.

### Explicit Edit Content

- Run dry-run orchestration on the smoke config.
- Verify that the manifest contains the expected stage ordering and stage B initialization checkpoint path.
- Verify that no score-loss regression appears in base two-stage metrics.

### Interface and Contract Definitions

- Run contract:

```text
Stage A train
-> Stage B init checkpoint materialization
-> Stage B train
-> evaluate.py on best Stage B checkpoint
```

### Design Patterns

- YAGNI: no extra refactor once the target contract passes.
- Stable interface: do not widen the scope to online adaptation or new encoder families in this cycle.

### Risk Mitigation

- Evaluation metric inflation risk: compare base and point-score runs under the same validation threshold policy.
- Prototype redundancy risk: keep continuous/discrete branches untouched at this stage.

### Test Plan and Validation

Run:

```bash
./.venv/bin/python scripts/run_two_stage_offline_pretraining.py \
  --experiment-config configs/experiment/thesis/exp4/smd__thesis_multitask__offline-pretraining-two-stage-machine-3-4-window20-smoke__w20__seed11__smoke.yaml \
  --dry-run
```

Expected:
- manifest file is written,
- execution report is written,
- executed stages are `stage_a_multitask_pretraining`, `stage_b_fusion_finetuning`, and `evaluation`,
- no `FileNotFoundError` occurs for the exp4 YAMLs.

### Acceptance Criteria

- Dry-run succeeds for the smoke config.
- Narrow pytest set passes.
- The codebase is ready for a real execution of the base or point-score variant without further contract guessing.

---

## Final Acceptance Criteria

- Base two-stage remains backward compatible.
- Point-score-supervised two-stage is explicitly configurable and testable.
- The score-loss term is present only in Stage A and only when enabled.
- Exp4 YAML entrypoints exist again in the repository.
- Evaluation metrics and thresholding stay timeline-compatible and unchanged in semantics.
- No unrelated online-adaptation or encoder rewrite is introduced.

## Execution Order

1. Phase 1: config and variant contract alignment.
2. Phase 2: Stage A score-loss implementation.
3. Phase 3: YAML restoration and runner preservation.
4. Phase 4: test and contract locking.
5. Phase 5: dry-run verification and reporting.

