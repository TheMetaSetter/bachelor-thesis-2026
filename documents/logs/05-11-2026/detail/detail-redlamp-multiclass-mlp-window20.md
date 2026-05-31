---
date: 2026-05-11 15:08:12 +07 +0700
researcher: TheMetaSetter
git_commit: 13c67f255bd177f14f83fdf87a13cb156e3faa36
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for RedLamp multiclass MLP baseline with window size 20"
tags: [detail, implementation-plan, redlamp, multiclass, mlp, window20]
status: complete
last_updated: 2026-05-11
last_updated_by: TheMetaSetter
---

# Detail: RedLamp Multiclass MLP Window-20 Implementation

## Source Documents

This detailed implementation document is based on:

- Research note: `documents/logs/05-11-2026/research/research-redlamp-baseline-synthetic-anomaly-alignment.md`
- Plan note: `documents/logs/05-11-2026/plan/plan-redlamp-multiclass-mlp-window20.md`
- Design documents: `documents/design/idea.md` and `documents/design/design_starter.md`
- Repository preferences: `codebase_preferences.md`

The prompt `prompts/4_detail_prompt.md` refers to a `structure/` note. No matching structure note exists for this topic at the time this document is written, so the window-20 implementation plan is used as the concrete outline source.

## Objective

The implementation must produce a fair RedLamp-aligned comparison path in which:

1. The proposed thesis multitask model performs multi-class synthetic anomaly classification over the RedLamp class space.
2. A new RedLamp-inspired MLP baseline is available as an active repository model.
3. Both the proposed model and the RedLamp MLP baseline use the same synthetic anomaly class taxonomy and the same window length of twenty time steps.
4. The implementation remains consistent with the repository's `1 model - 1 file` rule, explicit configuration style, and readability-first research code convention.

## Global Contracts

### Dataset And Batch Contract

All data loaders must continue to produce batches with the existing contract:

```python
batch = {
    "x": Tensor[B, L, D],
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

The new RedLamp comparison configs set:

```yaml
window_size: 20
stride: 20
```

The synthetic anomaly injector adds the following fields after augmentation:

```python
augmented_batch["classification_labels"]      # Tensor[B], int64
augmented_batch["classification_class_names"] # tuple[str, ...]
augmented_batch["synthetic_anomaly_mask"]     # Tensor[B, L], int64
augmented_batch["augmentation_metadata"]      # list[dict]
```

For RedLamp multi-class mode, class index `0` is `normal`, and class indices `1..11` correspond exactly to the eleven RedLamp synthetic anomaly families.

### Model Output Contract

All active models must continue to return:

```python
outputs = {
    "hidden": Tensor[B, L, H],
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor[B, C]],
    "point_scores": Optional[Tensor[B, L]],
    "window_scores": Optional[Tensor[B]],
    "aux": dict,
}
```

The new classification convention is:

- `outputs["logits"]` stores unnormalized logits for loss computation.
- `outputs["aux"]["class_probabilities"]` stores `torch.softmax(outputs["logits"], dim=-1)` for inspection and RedLamp-comparable probability output.

### Class Taxonomy Contract

The class taxonomy must be centralized in `src/data/augment.py`:

```python
REDLAMP_ANOMALY_FAMILIES = (
    "spike",
    "flip",
    "speedup",
    "noise",
    "cutoff",
    "average",
    "scale",
    "wander",
    "contextual",
    "upsidedown",
    "mixture",
)

REDLAMP_MULTICLASS_CLASS_NAMES = ("normal", *REDLAMP_ANOMALY_FAMILIES)
BINARY_SYNTHETIC_CLASS_NAMES = ("normal", "synthetic_anomaly")
```

Both `ThesisMultitaskModel` and `RedLampMLPBaseline` must use this shared taxonomy instead of duplicating label order locally.

## Design Pattern Application

- **Composition over inheritance:** The new baseline composes an encoder MLP, decoder MLP, classifier MLP, and `SyntheticAnomalyInjector`. It should not inherit from the RedLamp reference implementation.
- **Adapter pattern for encoders:** The RedLamp MLP baseline adapts a flattened-window latent vector back to the repository output contract by expanding it to `hidden: Tensor[B, L, H]`. This allows the trainer and evaluator to use the same interface as the thesis model.
- **Strategy pattern for tasks:** `classification_label_mode` acts as the task strategy for binary versus RedLamp multi-class synthetic supervision while preserving one injector implementation.
- **Registry/factory pattern:** The new model is registered through the existing `register_model` and constructed through the existing experiment config path.
- **Self-contained model rule:** `src/models/thesis_multitask.py` remains the single file for the proposed model. `src/models/redlamp_mlp_baseline.py` owns all architecture, forward logic, loss logic, and stage methods for the baseline.

## Phase 1: Shared RedLamp Multi-Class Synthetic Labels

### Phase Summary

This phase establishes the class-space contract that both methods must share. It directly supports the thesis objective that the proposed method and RedLamp baseline classify the same synthetic anomaly classes.

### File-Level Edits

Modify `src/data/augment.py`.

Add class-name constants beside the existing RedLamp family tuple:

```python
REDLAMP_MULTICLASS_CLASS_NAMES: tuple[str, ...] = ("normal", *REDLAMP_ANOMALY_FAMILIES)
BINARY_SYNTHETIC_CLASS_NAMES: tuple[str, ...] = ("normal", "synthetic_anomaly")
```

Extend `SyntheticAnomalyInjector.__init__` with:

```python
classification_label_mode: str = "binary",
```

Validate it explicitly:

```python
if classification_label_mode not in {"binary", "redlamp_multiclass"}:
    raise ValueError(
        "classification_label_mode must be one of: binary, redlamp_multiclass"
    )
self.classification_label_mode = classification_label_mode
```

Add two small helper methods:

```python
def _classification_class_names(self) -> tuple[str, ...]:
    if self.classification_label_mode == "binary":
        return BINARY_SYNTHETIC_CLASS_NAMES
    return REDLAMP_MULTICLASS_CLASS_NAMES

def _classification_label_from_metadata(self, metadata: dict[str, Any]) -> int:
    if not metadata["is_synthetic_anomaly"]:
        return 0
    if self.classification_label_mode == "binary":
        return 1
    return REDLAMP_MULTICLASS_CLASS_NAMES.index(metadata["anomaly_family"])
```

Change `augment_batch` so it no longer assigns all injected windows to class `1` unconditionally. After `_inject_single_window`, set:

```python
classification_labels[batch_index] = self._classification_label_from_metadata(
    window_metadata
)
```

Add the class names to every augmented batch:

```python
augmented_batch["classification_class_names"] = self._classification_class_names()
```

### Tests

Modify `tests/test_synthetic_anomaly_injection.py`.

Add tests that assert:

- `REDLAMP_MULTICLASS_CLASS_NAMES == ("normal", *REDLAMP_ANOMALY_FAMILIES)`.
- Binary mode remains the default and returns `("normal", "synthetic_anomaly")`.
- RedLamp multi-class mode maps metadata family names to class indices correctly.

Run:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py
```

### Acceptance Criteria

- The injector can return binary labels without config changes.
- The injector can return RedLamp multi-class labels with `classification_label_mode="redlamp_multiclass"`.
- The selected anomaly family in metadata matches the emitted class label.
- All existing binary injector tests still pass.

## Phase 2: Proposed Thesis Model Multi-Class Classification

### Phase Summary

This phase changes the proposed method from binary synthetic anomaly detection to RedLamp-style anomaly-type classification while preserving the fused-representation thesis design.

### File-Level Edits

Modify `src/models/thesis_multitask.py`.

Extend `SyntheticAnomalyConfig`:

```python
classification_label_mode: str = "binary"
```

Validate it in `__post_init__`:

```python
if self.classification_label_mode not in {"binary", "redlamp_multiclass"}:
    raise ValueError(
        "classification_label_mode must be one of: binary, redlamp_multiclass"
    )
```

Add `"classification_label_mode"` to `synthetic_keys` inside `ThesisMultitaskModelConfig.from_flat_kwargs`.

Store the mode in `_store_config_values`:

```python
self.classification_label_mode = synthetic.classification_label_mode
```

Pass the mode into both synthetic injectors in `_build_synthetic_injectors`:

```python
classification_label_mode=config.synthetic.classification_label_mode,
```

In `forward`, compute and expose probabilities:

```python
logits = self.classification_head(pooled_classification_hidden)
class_probabilities = torch.softmax(logits, dim=-1)
```

Add to `outputs["aux"]`:

```python
"class_probabilities": class_probabilities,
```

Replace binary-only refurbished targets with a generalized method:

```python
def _build_refurbished_classification_targets(
    self,
    classification_labels: torch.Tensor,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    hard_labels = classification_labels.long()
    target_probabilities = F.one_hot(
        hard_labels,
        num_classes=self.num_classes,
    ).to(dtype=target_dtype)

    if self.classification_label_mode == "binary":
        if self.num_classes != 2:
            raise ValueError("Binary label refurbishment requires num_classes == 2")
        target_probabilities[:, 0] = torch.where(
            hard_labels == 0,
            1.0 - self.refurbishment_beta,
            self.refurbishment_alpha,
        )
        target_probabilities[:, 1] = torch.where(
            hard_labels == 0,
            self.refurbishment_beta,
            1.0 - self.refurbishment_alpha,
        )
        return target_probabilities

    target_probabilities = torch.where(
        target_probabilities > 0.0,
        1.0
        - (
            self.refurbishment_alpha
            + self.refurbishment_beta * self.num_classes
            - self.refurbishment_beta
        ),
        self.refurbishment_beta,
    )
    target_probabilities[:, 0] = target_probabilities[:, 0] + self.refurbishment_alpha
    return target_probabilities / target_probabilities.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(self.epsilon)
```

Update `_compute_classification_loss` to use this method when label refurbishment is enabled.

### Tests

Modify:

- `tests/test_thesis_multitask_config_refactor.py`
- `tests/test_one_multitask_train_step.py`
- `tests/test_multitask_shapes.py`

Add tests that assert:

- `num_classes=12` creates logits with shape `[B, 12]`.
- `outputs["aux"]["class_probabilities"]` has shape `[B, 12]` and sums to one.
- RedLamp-style label smoothing produces normalized soft targets.
- Existing binary label refurbishment tests remain valid.

Run:

```bash
pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_one_multitask_train_step.py tests/test_multitask_shapes.py
```

### Acceptance Criteria

- The thesis model can train with `num_classes: 12`.
- Binary configs remain accepted.
- The classification head produces logits and softmax probabilities for the same class order used by the injector.
- Label smoothing is numerically stable and produces target distributions summing to one.

## Phase 3: Multi-Class Metrics And Trainer Aggregation

### Phase Summary

This phase prevents evaluation metric inflation and binary-only misreporting. It ensures that RedLamp multi-class classification is evaluated with multi-class metrics instead of binary anomaly metrics.

### File-Level Edits

Modify `src/metrics/pointwise.py`.

Add `accuracy_score` to imports:

```python
from sklearn.metrics import accuracy_score
```

Add:

```python
def compute_multiclass_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    label_array = labels.detach().cpu().numpy().astype(np.int64)
    prediction_array = torch.argmax(logits.detach().cpu(), dim=-1).numpy().astype(
        np.int64
    )
    return {
        "accuracy": _safe_metric(accuracy_score, label_array, prediction_array),
        "macro_f1": _safe_metric(
            f1_score,
            label_array,
            prediction_array,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": _safe_metric(
            f1_score,
            label_array,
            prediction_array,
            average="weighted",
            zero_division=0,
        ),
        "num_classes_observed": float(len(np.unique(label_array))),
    }
```

Modify `src/engine/trainer.py`.

Import both metric helpers:

```python
from src.metrics.pointwise import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
)
```

Dispatch inside `_aggregate_multitask_classification_metrics`:

```python
num_classes = concatenated_logits.shape[-1]
if num_classes == 2:
    classification_metrics = compute_binary_classification_metrics(
        logits=concatenated_logits,
        labels=concatenated_labels,
    )
else:
    classification_metrics = compute_multiclass_classification_metrics(
        logits=concatenated_logits,
        labels=concatenated_labels,
    )
```

### Tests

Modify:

- `tests/test_evaluator_thresholding.py`
- `tests/test_multitask_metrics_runtime.py`

Add a unit test for `compute_multiclass_classification_metrics` that verifies accuracy, macro F1, weighted F1, and observed class count.

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
```

### Acceptance Criteria

- Binary classification metrics remain unchanged for two-class logits.
- Multi-class runs report accuracy, macro F1, weighted F1, and observed class count.
- The trainer does not compute binary ROC-AUC or PR-AUC on twelve-class logits.

## Phase 4: RedLamp MLP Baseline Model

### Phase Summary

This phase adds the active MLP baseline requested by the user. It provides a RedLamp-inspired autoencoder plus multi-class classification head, but uses the repository's standardized batch and model output contracts.

### File-Level Edits

Create `src/models/redlamp_mlp_baseline.py`.

The file must define one model class:

```python
class RedLampMLPBaseline(BaseModel):
    ...
```

Constructor arguments:

```python
def __init__(
    self,
    input_dim: int,
    window_size: int,
    latent_dim: int = 128,
    mlp_num_linear_layers: int = 3,
    classifier_dim: int = 32,
    num_classes: int = len(REDLAMP_MULTICLASS_CLASS_NAMES),
    dropout: float = 0.1,
    lambda_cls: float = 0.1,
    use_label_refurbishment: bool = True,
    refurbishment_alpha: float = 0.1,
    refurbishment_beta: float = 0.01,
    anomaly_probability: float = 0.5,
    min_segment_fraction: float = 0.1,
    max_segment_fraction: float = 0.2,
    spike_scale: float = 3.0,
    anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
    use_synthetic_augmentation: bool = True,
    use_synthetic_validation: bool = True,
    synthetic_validation_seed: int = 7,
) -> None:
```

Architecture:

- Flatten `[B, L, D]` to `[B, L * D]`.
- Encoder MLP maps flattened input to `latent_dim`.
- Decoder MLP maps `latent_dim` back to `L * D`.
- Classification head maps `latent_dim` to `num_classes`.
- All three stacks use `mlp_num_linear_layers`, defaulting to three for fairness.

Forward behavior:

```python
flattened_x = x_tensor.reshape(batch_size, window_size * input_dim)
latent = self.encoder(flattened_x)
reconstructed_flat = self.decoder(latent)
recon = reconstructed_flat.reshape(batch_size, window_size, input_dim)
logits = self.classification_head(latent)
point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
hidden = latent.unsqueeze(1).expand(batch_size, window_size, self.latent_dim)
```

Loss behavior:

```python
total_loss = reconstruction_loss + self.lambda_cls * classification_loss
```

Use the same RedLamp-style smoothed target formula as the thesis model in RedLamp multi-class mode. The baseline should expose logits and probabilities, not apply softmax before loss.

Stage methods:

- `training_step`
- `validation_step`
- `synthetic_validation_step`
- `test_step`

Each method delegates to one `_shared_step` helper to minimize codepaths.

### Tests

Create `tests/test_redlamp_mlp_baseline.py`.

Add tests that assert:

- Forward contract returns `recon: [B, 20, D]`.
- `hidden: [B, 20, latent_dim]`.
- `logits: [B, 12]`.
- `class_probabilities.sum(dim=-1)` equals one.
- Encoder, decoder, and classifier head each contain three `nn.Linear` layers when `mlp_num_linear_layers=3`.

Run:

```bash
pytest -q tests/test_redlamp_mlp_baseline.py
```

### Acceptance Criteria

- The baseline is fully self-contained in `src/models/redlamp_mlp_baseline.py`.
- The baseline uses the same injector and class names as the thesis model.
- The baseline supports one forward pass and returns valid reconstruction and classification outputs.
- Layer-count tests confirm the MLP depth contract.

## Phase 5: Window-20 Configurations And Registry Integration

### Phase Summary

This phase makes the new comparison reproducible through YAML configuration and the existing registry/factory path. It ensures both models use exactly the same twenty-step window regime.

### File-Level Edits

Create `configs/data/smd_rtx3090_machine_2_1_20.yaml`:

```yaml
dataset_name: smd
root_dir: data/ServerMachineDataset
entity_ids:
  - machine-2-1
window_size: 20
stride: 20
batch_size: 256
num_workers: 8
validation_split_ratio: 0.2
shuffle_train: true
```

Create `configs/model/thesis_multitask_redlamp_multiclass.yaml` with:

```yaml
model_name: thesis_multitask
input_dim: 38
encoder_dim: 64
hidden_dim: 32
mlp_num_linear_layers: 3
num_classes: 12
dropout: 0.1
continuous_enabled: true
continuous_num_prototypes: 8
discrete_enabled: true
discrete_codebook_size: 16
gumbel_temperature: 1.5
temperature_start: 1.5
temperature_end: 0.7
temperature_anneal_fraction: 0.8
temperature_hold_fraction: 0.0
alpha_logit_init: 0.0
beta_logit_init: 0.0
use_label_refurbishment: true
refurbishment_alpha: 0.1
refurbishment_beta: 0.01
reconstruction_normal_only: true
lambda_cls: 1.0
lambda_div: 0.0
lambda_var: 0.0
lambda_cov: 0.0
lambda_use: 0.0
lambda_gate: 0.0
usage_lambda_start: 0.0
usage_lambda_end: 0.0
usage_lambda_schedule_fraction: 1.0
variance_floor_gamma: 1.0
gate_barrier_margin: 0.25
bootstrap_encoder_epochs: 10
discrete_ema_decay: 0.99
memory_norm_epsilon: 1.0e-6
memory_initialization_batches: 16
memory_initialization_with_synthetic_windows: true
```

Create `configs/model/redlamp_mlp_baseline.yaml`:

```yaml
model_name: redlamp_mlp_baseline
input_dim: 38
window_size: 20
latent_dim: 128
mlp_num_linear_layers: 3
classifier_dim: 32
num_classes: 12
dropout: 0.1
lambda_cls: 0.1
use_label_refurbishment: true
refurbishment_alpha: 0.1
refurbishment_beta: 0.01
```

Create `configs/task/multitask_tsad_redlamp_multiclass_window20.yaml`:

```yaml
task_name: multitask_tsad
use_synthetic_augmentation: true
use_synthetic_validation: true
synthetic_validation_seed: 7
classification_label_mode: redlamp_multiclass
freeze_fusion_for_epochs: 0
warmup_alpha_value: 0.5
warmup_beta_value: 0.5
anomaly_probability: 0.5
balance_binary_classes_within_batch: false
min_segment_fraction: 0.1
max_segment_fraction: 0.2
spike_scale: 3.0
anomaly_families:
  - spike
  - flip
  - speedup
  - noise
  - cutoff
  - average
  - scale
  - wander
  - contextual
  - upsidedown
  - mixture
```

Create experiment configs:

- `configs/experiment/thesis/exp3/smd__thesis_multitask__thesis-multitask-redlamp-multiclass-window20__w20__seed11__default.yaml`
- `configs/experiment/baseline/smd__redlamp_mlp_baseline__redlamp-mlp-baseline-window20__w20__seed11__default.yaml`

Both must reference:

```yaml
data_config_path: configs/data/smd_rtx3090_machine_2_1_20.yaml
task_config_path: configs/task/multitask_tsad_redlamp_multiclass_window20.yaml
```

Modify `src/core/config.py`:

- Add `redlamp_mlp_baseline` to `supported_model_names`.
- Validate `latent_dim`, `mlp_num_linear_layers`, `classifier_dim`, and `num_classes`.
- Validate `classification_label_mode`.
- Require `num_classes == 12` when `classification_label_mode == "redlamp_multiclass"`.

Modify:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`

Add:

```python
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline
```

Register:

```python
register_model("redlamp_mlp_baseline", RedLampMLPBaseline)
```

Pass data window size to baseline construction:

```python
if model_name == "redlamp_mlp_baseline":
    model_kwargs["window_size"] = experiment_config["data"]["window_size"]
```

### Tests

Modify `tests/test_config_loading.py`.

Add tests that load both experiment configs and assert:

- `data.window_size == 20`
- `data.stride == 20`
- `model.num_classes == 12`
- `model.mlp_num_linear_layers == 3`
- `task.classification_label_mode == "redlamp_multiclass"`
- The anomaly family list matches `REDLAMP_ANOMALY_FAMILIES`.

Run:

```bash
pytest -q tests/test_config_loading.py
```

### Acceptance Criteria

- Both experiment configs load through `load_experiment_config`.
- Both configs use the same data and task configs.
- The RedLamp MLP baseline is constructible through the registry path.
- Window size and stride are both twenty.

## Phase 6: End-To-End Verification

### Phase Summary

This phase confirms that the implementation works beyond unit-level behavior by running one training step for both models and checking checkpoint compatibility for the existing model stack.

### File-Level Edits

Create `tests/test_one_redlamp_mlp_train_step.py`.

The test should instantiate `RedLampMLPBaseline` with:

```python
input_dim=4
window_size=20
latent_dim=16
mlp_num_linear_layers=3
classifier_dim=8
num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES)
anomaly_probability=1.0
```

Build a batch:

```python
batch = {
    "x": torch.randn(2, 20, 4),
    "point_labels": torch.zeros(2, 20, dtype=torch.long),
    "mask": None,
    "timestamps": None,
    "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
}
```

Run:

```python
step_output = model.training_step(batch)
step_output["loss"].backward()
```

Assert at least one trainable parameter receives a gradient.

### Validation Commands

Run focused tests:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py
pytest -q tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py
pytest -q tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
pytest -q tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
pytest -q tests/test_config_loading.py
```

Run final relevant regression suite:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_multitask_metrics_runtime.py tests/test_checkpoint_roundtrip.py tests/test_config_loading.py tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
```

### Acceptance Criteria

- The thesis model runs one forward/backward training step with twelve-class labels.
- The RedLamp MLP baseline runs one forward/backward training step with twelve-class labels.
- Relevant regression tests pass.
- Existing checkpoint roundtrip tests pass.
- No old binary behavior is removed.

## Risk Mitigation For Broader Thesis Concerns

### Prototype Redundancy

The new classification mode must not bypass the existing continuous and discrete prototype branches in `ThesisMultitaskModel`. Multi-class classification should continue to use `hidden_classification`, which is derived from task-specific fusion. Existing ablation configs for continuous-only, discrete-only, and fused branches must remain valid.

Acceptance criterion: `ThesisMultitaskModel.forward` still computes continuous branch outputs, discrete branch outputs, and `fusion_outputs` before classification logits.

### Fusion Collapse

The implementation must preserve logging of `alpha` and `beta` fusion values in `outputs["aux"]`. The multi-class change should not hard-code classification to either branch.

Acceptance criterion: `outputs["aux"]["alpha"]` and `outputs["aux"]["beta"]` are still present in thesis model outputs.

### Adaptation Contamination

This implementation does not change online adaptation. It must avoid editing `src/models/online_adaptation.py` except for registry import compatibility if necessary. Synthetic multi-class labels must remain confined to offline multitask and baseline training.

Acceptance criterion: online adaptation tests still pass if included in broader regression.

### Projector Drift

This implementation does not modify the projector, projector warm-start, or anchor regularization. The detailed plan explicitly leaves those mechanisms unchanged to avoid coupling a baseline-classification change to online adaptation behavior.

Acceptance criterion: no new projector parameters or losses are introduced in this work.

### Evaluation Metric Inflation

Binary anomaly metrics must not be applied to twelve-class classification logits. Multi-class classification metrics are separated from pointwise anomaly detection metrics.

Acceptance criterion: trainer aggregation dispatches on `logits.shape[-1]`, and twelve-class logits use `compute_multiclass_classification_metrics`.

## Reproducibility And Logging

The new experiment configs should include W&B metadata but keep smoke runs disabled by default:

```yaml
logging:
  use_wandb: false
  wandb_project: bachelor-thesis-2026
  wandb_mode: disabled
  wandb_tags:
    - smd
    - redlamp_multiclass
    - window20
    - mlp-depth-3
```

When running real experiments later, switch `use_wandb: true` and use separate run names for:

- `smd_thesis_multitask_redlamp_multiclass_window20`
- `smd_redlamp_mlp_baseline_window20`

If generated synthetic datasets are serialized later, add DVC tracking in a separate data-versioning task. This implementation injects synthetic anomalies at batch time and does not introduce serialized derived datasets.

## Final Acceptance Checklist

- [ ] `src/data/augment.py` exports `REDLAMP_MULTICLASS_CLASS_NAMES`.
- [ ] `SyntheticAnomalyInjector` supports `classification_label_mode="binary"` and `"redlamp_multiclass"`.
- [ ] `ThesisMultitaskModel` supports `num_classes: 12` with label smoothing.
- [ ] `ThesisMultitaskModel` exposes `outputs["aux"]["class_probabilities"]`.
- [ ] `src/models/redlamp_mlp_baseline.py` exists and is self-contained.
- [ ] `redlamp_mlp_baseline` is registered in train and evaluation entrypoints.
- [ ] `configs/data/smd_rtx3090_machine_2_1_20.yaml` exists.
- [ ] Both RedLamp comparison experiment configs use `window_size: 20`.
- [ ] Multi-class classification metrics are implemented and trainer-dispatched.
- [ ] Focused and regression tests listed above pass.
