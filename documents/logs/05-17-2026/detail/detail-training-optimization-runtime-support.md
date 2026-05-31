---
date: 2026-05-17
research_source: documents/logs/05-17-2026/research/research-training-optimization-runtime-support.md
plan_source: documents/logs/05-17-2026/plan/plan-training-optimization-runtime-support.md
topic: "Detailed implementation plan for AdamW, CANDI-style cosine scheduling, configurable gradient clipping, and experiment configurations"
status: draft
---

# Detailed Plan: Training Optimization Runtime Support

## Objective

Implement a readable and configuration-driven training-runtime extension that supports:

1. `optimizer_name: adamw` and `optimizer_name: adam` in both offline training and online adaptation entrypoints;
2. `scheduler_name: cosine` using the CANDI reference mechanism of **iteration-level** learning-rate updates from fractional epoch progress;
3. `gradient_clip_norm` as an explicit YAML-controlled training parameter;
4. `val_vus_pr` as a validation metric computed on every epoch;
5. explicit best-checkpoint selection from `val_vus_pr`;
6. two reproducible RedLamp MLP baseline experiment configurations for SMD machine `2-1`, one with learning rate `1.0e-3` and one with learning rate `1.0e-4`;
7. batch-level learning-rate records for traceability while keeping user-facing progress summaries at the epoch level.

The implementation must preserve the repository's existing contracts, especially the standardized batch schema, model output schema, and one-model-per-file organization.

## Scope Boundaries

### In scope

- Offline training optimizer configurability.
- Online adaptation optimizer configurability.
- Cosine learning-rate policy for offline training only.
- Configurable offline gradient clipping.
- Per-epoch validation `VUS-PR`.
- `VUS-PR`-based best-checkpoint selection.
- Epoch-summary learning-rate logging plus retained batch-level learning-rate history.
- New SMD `2-1` experiment configurations.
- Tests for config validation, optimizer construction, cosine behavior, gradient clipping, and regression preservation.

### Out of scope

- Refactoring the whole training system into a generic strategy framework.
- Changing model architectures, data loaders, anomaly injection, prototype branches, or evaluation metrics.
- Adding cosine scheduling to online adaptation in this pass.
- Changing checkpoint semantics beyond what is necessary to preserve existing functionality.

## Stable Interfaces and Design Patterns

### Existing contracts to preserve

| Contract | Required behavior |
|---|---|
| Batch contract | Models continue to receive dictionaries whose primary tensor is `batch["x"]` with shape `[B, L, D]`. |
| Encoder contract | Models continue to expose the thesis-facing hidden representation with shape `[B, L, H]`. |
| Model output contract | Existing keys such as `hidden`, `pooled`, `recon`, `logits`, `point_scores`, `window_scores`, and `aux` remain unchanged. |
| Training-step contract | Model `training_step()` methods continue to return a scalar `loss`, a `log` dictionary, and model outputs. |

### Metric policy

- `VUS-PR` is the primary thesis-facing anomaly-detection performance metric.
- Cosine scheduling is schedule-based and therefore does not require any monitored metric to update learning rate.
- Checkpoint selection is separate from scheduler stepping. Metric-driven schedulers may still use a scheduler monitor, but cosine experiments must be able to select checkpoints independently from `val_vus_pr`.

### Design pattern use

- **Composition over inheritance:** this feature should extend the runtime through small helper functions and explicit configuration rather than through a deep scheduler inheritance tree.
- **Adapter pattern for encoders:** unchanged; encoder adapters remain outside this work and continue to protect the hidden-state contract.
- **Strategy pattern for runtime behavior:** use explicit named runtime branches for plateau scheduling and cosine scheduling, because these behaviors have different stepping contracts.
- **Registry/factory principle:** unchanged for datasets and models; this work remains at the engine and entrypoint layers and must not bypass the existing registry-driven construction path.

## Phase 1: Extend Configuration Contracts

### Phase summary

This phase makes the intended optimization behavior expressible in YAML without altering model or data contracts. It aligns the repository with its stated design preference that experiments should be explicit and ablation-friendly.

### File-level edits

#### `src/core/config.py`

Add optimizer validation:

```python
optimizer_name = optimizer_config.get("optimizer_name", "adam")
if optimizer_name not in {"adam", "adamw"}:
    raise ValueError("optimizer.optimizer_name must be one of: adam, adamw")
```

Add clipping validation:

```python
gradient_clip_norm = optimizer_config.get("gradient_clip_norm")
if gradient_clip_norm is not None:
    if (
        not isinstance(gradient_clip_norm, (int, float))
        or float(gradient_clip_norm) <= 0.0
    ):
        raise ValueError("optimizer.gradient_clip_norm must be positive when provided")
```

Add checkpoint-monitor validation:

```python
checkpoint_monitor_metric = experiment_config.get(
    "checkpoint_monitor_metric",
    "val_loss",
)
if checkpoint_monitor_metric not in {
    "val_loss",
    "val_synth_loss",
    "val_synth_roc_auc",
    "val_synth_pr_auc",
    "val_vus_pr",
}:
    raise ValueError(
        "checkpoint_monitor_metric must be one of: val_loss, val_synth_loss, "
        "val_synth_roc_auc, val_synth_pr_auc, val_vus_pr"
    )
```

Extend scheduler validation into two explicit branches:

```python
scheduler_name = scheduler_config.get("scheduler_name")
if scheduler_name == "reduce_on_plateau":
    # retain current plateau validation
elif scheduler_name == "cosine":
    # validate cosine-specific fields
else:
    raise ValueError(
        "optimizer.scheduler.scheduler_name must be one of: reduce_on_plateau, cosine"
    )
```

For cosine, validate:

- `warmup_epochs`: non-negative integer;
- `warmup_start_lr`: positive numeric value not exceeding `optimizer.learning_rate`;
- `cosine_end_lr`: non-negative numeric value lower than `optimizer.learning_rate`;
- `cosine_after_warmup`: boolean.

Reject scheduler fields that are incompatible with the selected scheduler family when practical, especially:

- plateau-only: `monitor_metric`, `factor`, `patience`, `threshold`, `threshold_mode`, `cooldown`, `min_lr`;
- cosine-only: `warmup_epochs`, `warmup_start_lr`, `cosine_end_lr`, `cosine_after_warmup`.

#### `tests/test_config_loading.py`

Add tests that:

- accept `optimizer_name: adamw`;
- preserve legacy configs that omit `optimizer_name` by defaulting to `adam`;
- reject unsupported optimizer names;
- accept `gradient_clip_norm: 1.0`;
- reject `gradient_clip_norm <= 0`;
- accept a valid cosine scheduler block;
- reject malformed cosine blocks;
- accept `checkpoint_monitor_metric: val_vus_pr`;
- continue accepting valid `reduce_on_plateau` blocks.

### Acceptance criteria

- A config with `optimizer_name: adamw`, `gradient_clip_norm: 1.0`, and a valid cosine scheduler loads successfully.
- A config with `checkpoint_monitor_metric: val_vus_pr` loads successfully.
- A config with an invalid optimizer name fails before runtime construction.
- Existing scheduler tests for `reduce_on_plateau` still pass unchanged.
- No dataset, model, or task contract changes are required to load the new config fields.

## Phase 2: Add Explicit Optimizer Construction

### Phase summary

This phase removes the current optimizer hard-coding while keeping the code path deliberately small. It also brings online adaptation under the same explicit optimizer naming rule without broadening scheduler scope there.

### File-level edits

#### `scripts/train.py`

Extract or add a helper:

```python
def build_optimizer_from_experiment_config(
    model: torch.nn.Module,
    experiment_config: dict[str, object],
) -> torch.optim.Optimizer:
    optimizer_config = experiment_config["optimizer"]
    optimizer_name = str(optimizer_config.get("optimizer_name", "adam"))
    optimizer_kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
```

Replace direct optimizer creation with this helper and log `optimizer_name`.

#### `scripts/run_online_adaptation.py`

Add a parallel helper or reuse a small local helper with the same accepted optimizer names, but using:

```python
model.get_parameter_group(experiment_config["task"]["target_param_group"])
```

instead of `model.parameters()`.

Keep the online adaptation scheduler behavior unchanged in this phase.

#### Tests

Add focused tests proving:

- offline config `adamw` yields `torch.optim.AdamW`;
- offline config omitting `optimizer_name` yields `torch.optim.Adam`;
- online adaptation config `adamw` yields `torch.optim.AdamW` over the configured target parameter group.

Preferred test homes:

- `tests/test_learning_rate_scheduler.py` or a new small `tests/test_optimizer_building.py` for offline optimizer building;
- `tests/test_online_entrypoint.py` for online optimizer configuration behavior.

### Acceptance criteria

- Offline training and online adaptation both obey `optimizer_name`.
- Existing configs that omit `optimizer_name` retain their previous `Adam` behavior.
- Console logs report the actual optimizer type instead of a hard-coded label.

## Phase 3: Implement CANDI-Style Cosine Policy

### Phase summary

This phase adds the new learning-rate policy while preserving the repository's current plateau scheduling behavior. The implementation should be explicit because cosine scheduling and plateau scheduling use different triggers and should not be hidden behind a misleading common interface.

### File-level edits

#### `scripts/train.py`

Add a pure helper for cosine learning-rate computation:

```python
def compute_candi_style_cosine_learning_rate(
    *,
    base_learning_rate: float,
    current_progress: float,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_lr: float,
    cosine_end_lr: float,
    cosine_after_warmup: bool,
) -> float:
    if current_progress < warmup_epochs:
        cosine_warmup_end_lr = _compute_cosine_learning_rate_without_warmup(...)
        warmup_alpha = (
            cosine_warmup_end_lr - warmup_start_lr
        ) / warmup_epochs
        return current_progress * warmup_alpha + warmup_start_lr
    return _compute_cosine_learning_rate_without_warmup(...)
```

Use fractional progress:

```python
current_progress = epoch_index + float(train_batch_index) / num_training_batches
```

The formula should match the CANDI logic:

- if `cosine_after_warmup` is true, use `warmup_epochs` as the cosine offset;
- otherwise, use `0.0` as the offset;
- cosine decay runs from `base_learning_rate` toward `cosine_end_lr`.

#### `src/engine/trainer.py`

Extend `Trainer.__init__` with:

- `cosine_scheduler_config: dict[str, Any] | None`
- `gradient_clip_norm: float | None`

Add private helpers:

- `_set_optimizer_learning_rate(new_learning_rate: float) -> None`
- `_step_cosine_learning_rate_scheduler(epoch_index: int, train_batch_index: int, num_training_batches: int) -> float`

During the training loop, before the forward/backward path of each batch:

1. compute the cosine LR when cosine config is present;
2. assign it to all optimizer parameter groups;
3. append it to a per-epoch list such as `batch_learning_rates`.

Keep the existing `_step_learning_rate_scheduler()` for plateau scheduling and call it only at epoch end.

#### Logging behavior

Store batch-level LR values internally for traceability, but expose user-facing summaries only at epoch granularity:

- do **not** print one console line per LR batch update;
- at epoch end, add summary metrics such as:
  - `optimizer_lr_start`
  - `optimizer_lr_end`
  - `optimizer_lr_min`
  - `optimizer_lr_max`
  - existing `optimizer_lr` should represent the final LR of the epoch.

If the existing JSONL or WandB metric path supports arbitrary scalar metrics only, keep raw batch LR history in memory only for the current epoch unless a later explicit artifact requirement is added.

#### `tests/test_learning_rate_scheduler.py`

Add deterministic tests for:

- LR at the beginning of training equals `warmup_start_lr`;
- LR differs between two batches in the same epoch;
- LR decreases over later training progress;
- the final LR approaches `cosine_end_lr`;
- plateau scheduling still behaves exactly as before.

### Acceptance criteria

- Cosine learning rate changes within a single epoch when multiple batches are present.
- The LR path matches the configured warmup and cosine schedule numerically in tests.
- User-facing console output remains epoch-oriented rather than batch-spam oriented.
- Existing plateau scheduler tests remain green.

## Phase 4: Add Per-Epoch Validation VUS-PR

### Phase summary

This phase promotes `VUS-PR` from a final evaluation-only metric into a validation metric available after every epoch. This is required because `VUS-PR` is the primary thesis-facing anomaly-detection metric and will determine best-checkpoint selection for the new cosine experiments.

### File-level edits

#### `src/engine/trainer.py`

Extend the validation flow so that after ordinary validation aggregation, the trainer computes validation pointwise metrics on the full validation timeline and adds:

```python
epoch_metrics["val_vus_pr"] = validation_evaluation_outputs["metrics"]["vus_pr"]
```

The trainer must reuse the overlap-aware point-score reconstruction already implemented in `src/engine/evaluator.py`; it must not duplicate that timeline-merging logic locally.

Recommended structure:

- construct or receive a validation evaluator configured with `vus_max_buffer_size` and `vus_num_thresholds`;
- run it on `val_loader` once per epoch while the model is in evaluation mode;
- prefix returned evaluation metrics with `val_`;
- include `val_vus_pr` in epoch metric history and logger payloads before checkpoint selection.

#### `src/engine/evaluator.py`

If needed for reuse, add a small helper that exposes the existing evaluation payload cleanly without changing the public evaluator contract or duplicating logic in the trainer.

#### Tests

Add trainer/evaluator tests proving:

- `val_vus_pr` is present in every epoch metric record when VUS evaluation is configured;
- `val_vus_pr` is computed from validation point scores and point labels, not from synthetic classification logits;
- existing validation metrics remain present alongside `val_vus_pr`.

### Acceptance criteria

- Every epoch exposes `val_vus_pr` when the experiment enables VUS evaluation.
- `VUS-PR` is available before checkpoint selection runs.
- The implementation reuses the evaluator's existing overlap-aware metric path.

## Phase 5: Add Configurable Gradient Clipping

### Phase summary

This phase ports the RedLamp-style gradient-clipping safeguard into the local training runtime while keeping it configuration controlled and opt-in.

### File-level edits

#### `src/engine/trainer.py`

After `loss.backward()` and before `optimizer.step()`:

```python
gradient_norm = None
if self.gradient_clip_norm is not None:
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        self.model.parameters(),
        max_norm=self.gradient_clip_norm,
    )
```

Add epoch-level aggregation for clipping diagnostics:

- number of clipped steps if measurable;
- maximum observed gradient norm before clipping if the returned tensor is used;
- optional `gradient_norm_last` or `gradient_norm_max` scalar in epoch metrics.

Keep console summaries at epoch level. Do not add mandatory per-batch clipping logs.

#### Tests

Add tests proving:

- `clip_grad_norm_` is called when `gradient_clip_norm` is configured;
- it is not called when the field is absent;
- training still completes for a simple dummy model with clipping enabled.

### Acceptance criteria

- Offline training applies clipping when requested and preserves current behavior when clipping is absent.
- The implementation uses the YAML value rather than a hard-coded threshold.
- The returned epoch metrics expose enough evidence to verify clipping without flooding normal console output.

## Phase 6: Reconcile Scheduler and Checkpoint Semantics

### Phase summary

This phase keeps experiment selection behavior understandable after the new non-metric scheduler is introduced.

### File-level edits

#### `src/engine/trainer.py`

Update checkpoint-monitor resolution so that:

- checkpoint selection uses explicit `checkpoint_monitor_metric` when configured;
- `val_vus_pr` is supported with monitor mode `"max"`;
- `ReduceLROnPlateau` keeps its own scheduler monitor independently;
- cosine requires no scheduler monitor, but may still use `checkpoint_monitor_metric: val_vus_pr`.

Document this distinction in helper names and comments.

#### `src/engine/checkpoint.py`

No functional change is required if cosine remains a deterministic arithmetic policy without scheduler state. Leave generic scheduler-state persistence unchanged for plateau schedulers.

#### Tests

Add/adjust tests proving:

- cosine training tracks the best checkpoint from `val_vus_pr` when configured;
- plateau training still tracks the best checkpoint from its configured checkpoint monitor;
- scheduler monitor and checkpoint monitor can differ without conflict;
- checkpoint roundtrip tests for plateau remain unchanged.

### Acceptance criteria

- Adding cosine does not require a scheduler monitor metric.
- Best-checkpoint selection follows the explicit configured checkpoint metric.
- Existing scheduler-state persistence remains intact.
- Test names make the distinction between metric-driven and non-metric scheduling explicit.

## Phase 7: Add SMD 2-1 Experiment Configurations

### Phase summary

This phase creates the first concrete experiment pair needed for the RedLamp baseline study and removes ambiguous metadata from the current legacy config surface.

### File-level edits

#### Create

- `configs/experiment/scale/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-machine-2-1-window20-adamw-cosine-lr1e-3__w20__seed11__default.yaml`
- `configs/experiment/smd_redlamp_mlp_baseline_machine_2_1_window20_adamw_cosine_lr1e-4.yaml`

#### Shared values

```yaml
seed: 11
device: cuda
data_config_path: configs/data/smd_rtx3090_machine_2_1_20.yaml
model_config_path: configs/model/redlamp_mlp_baseline.yaml
task_config_path: configs/task/multitask_tsad_redlamp_multiclass_window20.yaml
optimizer:
  optimizer_name: adamw
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  scheduler:
    scheduler_name: cosine
    warmup_epochs: 5
    warmup_start_lr: 0.0001
    cosine_end_lr: 0.0
    cosine_after_warmup: true
checkpoint_monitor_metric: val_vus_pr
epochs: 300
```

#### Differences

- `learning_rate: 0.001` vs `learning_rate: 0.0001`;
- distinct `experiment_name`, `output_dir`, `checkpoint_dir`, and `wandb_run_name`.

#### Existing file handling

Keep `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml` as the historical plateau config, but correct its `wandb_run_name` so it no longer implies `adamw_cosine`.

#### Tests

Extend config-loading coverage so both new configs load successfully and resolve to:

- `optimizer_name == "adamw"`
- `scheduler_name == "cosine"`
- `gradient_clip_norm == 1.0`
- `checkpoint_monitor_metric == "val_vus_pr"`
- `epochs == 300`

### Acceptance criteria

- There are exactly two explicit AdamW-cosine experiment configs for SMD `2-1`.
- Both configs select the best checkpoint from `val_vus_pr`.
- The older baseline file no longer advertises behavior it does not execute.
- A future researcher can infer the full optimization setup from filename plus YAML contents without opening source code.

## Phase 8: Validation and Regression Pass

### Phase summary

This phase verifies that the new optimization behavior is correct without regressing the existing research pipeline.

### Required commands

Run at minimum:

```bash
pytest -q tests/test_config_loading.py tests/test_learning_rate_scheduler.py tests/test_checkpoint_roundtrip.py
pytest -q tests/test_online_entrypoint.py
pytest -q tests/test_one_train_step.py tests/test_one_multitask_train_step.py
pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py
```

If any affected smoke tests exist for the RedLamp MLP baseline, run those as well.

### Manual verification

Perform one short smoke experiment or reduced-epoch local run using the new `lr1e-3` config variant after temporarily overriding epochs downward in a copied smoke config or test fixture. Verify:

- optimizer log reports `AdamW`;
- epoch summary reports LR start/end values;
- epoch summary includes `val_vus_pr`;
- no per-batch LR spam appears in standard console output;
- batch-level LR history is still available through the intended internal or metric path;
- checkpoints are saved and best checkpoint selection uses `val_vus_pr` for cosine.

### Acceptance criteria

- Focused tests pass.
- Existing plateau-scheduler behavior remains intact.
- Both new SMD configs load.
- The first smoke run demonstrates `AdamW + cosine + gradient_clip_norm=1.0 + val_vus_pr`.

## Cross-Cutting Risk Mitigation

| Repository risk from thesis context | Relevance to this feature | Mitigation in this plan |
|---|---|---|
| Prototype redundancy | Not directly changed | Preserve model contracts and avoid touching prototype files. |
| Fusion collapse | Not directly changed | Preserve model logic and experiment comparability. |
| Adaptation contamination | Online optimizer configurability is extended, but online update policy is unchanged | Limit this pass to optimizer selection only; do not add new online scheduler behavior. |
| Projector drift | Not directly changed | Keep online optimizer extension narrow and avoid changing projector logic. |
| High-variance updates | Directly relevant | Add config-driven gradient clipping for offline training. |
| Evaluation metric inflation | Not directly changed | Preserve evaluator and checkpoint-monitor semantics; do not mix scheduler behavior with evaluation policy. |
| Primary metric unavailable during training | Directly relevant | Compute `val_vus_pr` every epoch and use it for checkpoint selection. |

## Detailed Test Inventory

| Test area | File |
|---|---|
| Config acceptance and rejection | `tests/test_config_loading.py` |
| Optimizer construction | `tests/test_optimizer_building.py` or existing entrypoint tests |
| Cosine schedule numerics | `tests/test_learning_rate_scheduler.py` |
| Iteration-level scheduler behavior | `tests/test_learning_rate_scheduler.py` |
| Gradient clipping behavior | `tests/test_learning_rate_scheduler.py` or dedicated trainer test |
| Plateau regression | `tests/test_learning_rate_scheduler.py` |
| Checkpoint monitor and state behavior | `tests/test_checkpoint_roundtrip.py` |
| Per-epoch validation `VUS-PR` | trainer/evaluator regression tests plus `tests/test_vus_pr_metric.py` |
| Online optimizer configurability | `tests/test_online_entrypoint.py` |
| One-step offline regression | `tests/test_one_train_step.py`, `tests/test_one_multitask_train_step.py` |

## Final Acceptance Criteria

The feature is complete when all of the following are true:

1. A YAML file can request `optimizer_name: adamw` and the runtime constructs `torch.optim.AdamW`.
2. The same `optimizer_name` mechanism works in online adaptation.
3. A YAML file can request `scheduler_name: cosine` with warmup fields and the runtime updates LR on every training iteration using fractional epoch progress.
4. A YAML file can set `gradient_clip_norm: 1.0`, and the trainer applies clipping between backward pass and optimizer step.
5. Every epoch records `val_vus_pr` when VUS evaluation is configured.
6. Best-checkpoint selection can be driven by `checkpoint_monitor_metric: val_vus_pr`.
7. Cosine scheduling updates learning rate without requiring any scheduler monitor metric.
8. Batch-level LR values are retained for traceability, but ordinary console output remains summarized by epoch.
9. Existing `reduce_on_plateau` configurations remain valid and behaviorally unchanged.
10. Two explicit AdamW-cosine SMD `2-1` experiment configs exist for `1e-3` and `1e-4`.
11. The older RedLamp baseline config no longer contains a misleading `adamw_cosine` run name.
12. The focused regression suite passes without changing dataset, model, evaluator, or batch contracts.
