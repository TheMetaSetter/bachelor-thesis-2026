# RedLamp Multiclass MLP Window-10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fair RedLamp-aligned experiment path where the proposed thesis model and a new RedLamp-style MLP baseline both inject the same RedLamp synthetic anomaly classes, classify the same twelve labels, and run with window size 10.

**Architecture:** Keep `src/data/augment.py` as the single synthetic-anomaly source shared by both models. Extend `src/models/thesis_multitask.py` in place so its classification head can run RedLamp-style multi-class anomaly-type classification while preserving the existing binary configs. Add one self-contained model file, `src/models/redlamp_mlp_baseline.py`, because `codebase_preferences.md` requires one model per file and colocated training/inference logic.

**Tech Stack:** Python, PyTorch, YAML configs, Pytest, scikit-learn metrics, existing registry and trainer infrastructure.

---

## Current State

- The research note at `documents/logs/05-11-2026/research/research-redlamp-baseline-synthetic-anomaly-alignment.md` documents that the active injector already lists the eleven RedLamp synthetic anomaly families in `src/data/augment.py`.
- The current thesis model is binary by default. `SyntheticAnomalyInjector.augment_batch` assigns `classification_labels = 0` for clean windows and `classification_labels = 1` for every injected window, regardless of anomaly family.
- RedLamp uses a multi-class label dictionary. Its default label space is twelve classes: `normal`, `spike`, `flip`, `speedup`, `noise`, `cutoff`, `average`, `scale`, `wander`, `contextual`, `upsidedown`, and `mixture`.
- The current thesis model already has a configurable MLP depth through `mlp_num_linear_layers`, and `configs/model/thesis_multitask.yaml` sets it to `3`.
- CANDI uses `WIN_SIZE = 10`, and this repository already has `configs/data/smd_rtx3090_machine_2_1_10.yaml` with `window_size: 10` and `stride: 10`.

## Design Options

- **Option A: Upgrade only the proposed method to multi-class and keep RedLamp reference code external.** This is insufficient because the user explicitly asked to create a RedLamp baseline using MLP inside this repo.
- **Option B: Modify `bsc-thesis-ref-codebases/RedLamp` directly.** This makes comparison code harder to track in the thesis repo and risks mixing external reference code with active experiment code.
- **Option C: Add an active `redlamp_mlp_baseline` model in `src/models/`, sharing the active injector and config system.** This is the recommended option because it keeps one model per file, uses the same data and logging stack as the proposed method, and lets both models run under the same window-10 SMD configs.

## Risk and Mitigation

- **Risk: Binary experiments break when the injector changes.** Mitigation: add `classification_label_mode` with allowed values `binary` and `redlamp_multiclass`; keep existing configs default-compatible unless explicitly overridden.
- **Risk: RedLamp label smoothing is incorrectly applied to logits.** Mitigation: build smoothed target probabilities explicitly and compute `-target * log_softmax(logits)` for both the thesis model and the RedLamp MLP baseline.
- **Risk: Metrics remain binary and silently misreport multi-class runs.** Mitigation: add `compute_multiclass_classification_metrics` and make the trainer dispatch metrics by `num_classes`.
- **Risk: The baseline is not fair because MLP layer counts differ.** Mitigation: give `redlamp_mlp_baseline` the same `mlp_num_linear_layers: 3` contract and add tests that count encoder, decoder, and classifier linear layers.
- **Risk: Window length is accidentally inherited from a non-window-10 config.** Mitigation: add dedicated window-10 model/task/experiment configs and config-loading tests that assert `window_size == 10`.
- **Risk: Label order drifts between proposed method and baseline.** Mitigation: define one exported class-name tuple in `src/data/augment.py` and use it in both models and tests.

## Open Questions

- The plan selects the twelve-class RedLamp default label space because RedLamp includes `normal` in `args.anomaly_types`. If the thesis later needs “synthetic classes only,” create a separate eleven-class ablation after the twelve-class RedLamp comparison is reproducible.
- RedLamp reference uses softmax before `CrossEntropyLoss`. This plan preserves the thesis repo's numerically standard behavior by outputting logits for loss and storing softmax probabilities in `outputs["aux"]["class_probabilities"]`.

## Files To Create Or Modify

- Modify `src/data/augment.py`: add shared RedLamp class-name constants and a multi-class label mode in `SyntheticAnomalyInjector`.
- Modify `src/models/thesis_multitask.py`: add RedLamp multi-class label smoothing, expose class probabilities, and keep binary label refurbishment available for old configs.
- Modify `src/metrics/pointwise.py`: add multi-class classification metrics.
- Modify `src/engine/trainer.py`: dispatch classification aggregation to binary or multi-class metrics from model metadata.
- Modify `src/core/config.py`: validate new model names and new task/model fields.
- Create `src/models/redlamp_mlp_baseline.py`: self-contained RedLamp-style MLP autoencoder baseline with reconstruction and multi-class classification.
- Modify `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py`: register `redlamp_mlp_baseline` where offline models are registered.
- Create `configs/model/thesis_multitask_redlamp_multiclass.yaml`.
- Create `configs/model/redlamp_mlp_baseline.yaml`.
- Create `configs/task/multitask_tsad_redlamp_multiclass_window10.yaml`.
- Create `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window10.yaml`.
- Create `configs/experiment/smd_redlamp_mlp_baseline_window10.yaml`.
- Modify or add tests under `tests/` for injector labels, thesis model multi-class loss, baseline shapes, metrics, config loading, and one train step.

## Contract Decisions

- The batch contract remains `batch["x"]: Tensor[B, L, D]`, with optional `point_labels`, `mask`, `timestamps`, and `meta`.
- Multi-class synthetic supervision adds `classification_labels: Tensor[B]`, where `0` means `normal`, and indices `1..11` match `REDLAMP_MULTICLASS_CLASS_NAMES`.
- The model output contract remains unchanged: `outputs["logits"]` stores unnormalized class logits, and `outputs["aux"]["class_probabilities"]` stores `torch.softmax(outputs["logits"], dim=-1)`.
- The thesis encoder contract remains `hidden: Tensor[B, L, H]`. The RedLamp MLP baseline also exposes a timestep-level hidden tensor by expanding the pooled latent representation to `[B, L, H]` so the existing output contract is satisfied.

---

### Task 1: Add RedLamp Multi-Class Label Mode To The Shared Injector

**Files:**
- Modify: `src/data/augment.py`
- Test: `tests/test_synthetic_anomaly_injection.py`

- [ ] **Step 1: Write failing tests for the RedLamp class list and labels**

Add these tests to `tests/test_synthetic_anomaly_injection.py`:

```python
def test_redlamp_multiclass_class_names_include_normal_then_families() -> None:
    from src.data.augment import REDLAMP_ANOMALY_FAMILIES, REDLAMP_MULTICLASS_CLASS_NAMES

    assert REDLAMP_MULTICLASS_CLASS_NAMES == ("normal", *REDLAMP_ANOMALY_FAMILIES)


def test_synthetic_anomaly_injector_can_emit_redlamp_multiclass_labels() -> None:
    from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES, SyntheticAnomalyInjector

    batch = _build_batch(batch_size=4, window_size=20, num_channels=3)
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        classification_label_mode="redlamp_multiclass",
        anomaly_families=("spike", "noise", "scale"),
        deterministic_seed=123,
    )

    augmented_batch = injector.augment_batch(batch)

    assert augmented_batch["classification_labels"].shape == (4,)
    assert augmented_batch["classification_class_names"] == REDLAMP_MULTICLASS_CLASS_NAMES
    assert augmented_batch["classification_labels"].min().item() >= 1
    assert augmented_batch["classification_labels"].max().item() <= 3
    for label, metadata in zip(
        augmented_batch["classification_labels"].tolist(),
        augmented_batch["augmentation_metadata"],
    ):
        assert REDLAMP_MULTICLASS_CLASS_NAMES[label] == metadata["anomaly_family"]


def test_synthetic_anomaly_injector_keeps_binary_labels_by_default() -> None:
    from src.data.augment import SyntheticAnomalyInjector

    batch = _build_batch(batch_size=3, window_size=20, num_channels=2)
    injector = SyntheticAnomalyInjector(
        anomaly_probability=1.0,
        anomaly_families=("spike",),
        deterministic_seed=7,
    )

    augmented_batch = injector.augment_batch(batch)

    assert augmented_batch["classification_labels"].tolist() == [1, 1, 1]
    assert augmented_batch["classification_class_names"] == ("normal", "synthetic_anomaly")
```

- [ ] **Step 2: Run the injector tests and verify failure**

Run:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py
```

Expected result: the new tests fail because `REDLAMP_MULTICLASS_CLASS_NAMES`, `classification_label_mode`, and `classification_class_names` do not exist yet.

- [ ] **Step 3: Implement shared class names and label mode**

Modify `src/data/augment.py`:

```python
REDLAMP_ANOMALY_FAMILIES: tuple[str, ...] = (
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

REDLAMP_MULTICLASS_CLASS_NAMES: tuple[str, ...] = ("normal", *REDLAMP_ANOMALY_FAMILIES)
BINARY_SYNTHETIC_CLASS_NAMES: tuple[str, ...] = ("normal", "synthetic_anomaly")
```

Extend `SyntheticAnomalyInjector.__init__`:

```python
def __init__(
    self,
    anomaly_probability: float = 0.5,
    min_segment_fraction: float = 0.1,
    max_segment_fraction: float = 0.2,
    spike_scale: float = 3.0,
    anomaly_families: tuple[str, ...] | list[str] = REDLAMP_ANOMALY_FAMILIES,
    balance_binary_classes_within_batch: bool = False,
    deterministic_seed: int | None = None,
    classification_label_mode: str = "binary",
) -> None:
    if classification_label_mode not in {"binary", "redlamp_multiclass"}:
        raise ValueError(
            "classification_label_mode must be one of: binary, redlamp_multiclass"
        )
    self.classification_label_mode = classification_label_mode
```

Add helper methods inside `SyntheticAnomalyInjector`:

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

Replace the hard-coded assignment in `augment_batch`:

```python
classification_labels = torch.zeros(
    batch_size, dtype=torch.long, device=clean_windows.device
)
```

and inside the injection loop:

```python
classification_labels[batch_index] = self._classification_label_from_metadata(
    window_metadata
)
```

Set class names in the returned batch:

```python
augmented_batch["classification_class_names"] = self._classification_class_names()
```

- [ ] **Step 4: Run injector tests and verify pass**

Run:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py
```

Expected result: all tests in the file pass.

- [ ] **Step 5: Commit**

```bash
git add src/data/augment.py tests/test_synthetic_anomaly_injection.py
git commit -m "Add RedLamp multiclass synthetic labels"
```

---

### Task 2: Upgrade Thesis Multitask Classification To RedLamp Multi-Class Mode

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Test: `tests/test_thesis_multitask_config_refactor.py`
- Test: `tests/test_one_multitask_train_step.py`
- Test: `tests/test_multitask_shapes.py`

- [ ] **Step 1: Write failing tests for multi-class config and logits**

Add a helper override in existing test builders where needed:

```python
def test_thesis_multitask_redlamp_multiclass_outputs_twelve_logits() -> None:
    from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES

    model = _build_model(
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        use_label_refurbishment=True,
        classification_label_mode="redlamp_multiclass",
        anomaly_probability=1.0,
        anomaly_families=list(REDLAMP_MULTICLASS_CLASS_NAMES[1:]),
    )
    batch = _build_batch(batch_size=2)

    step_output = model.training_step(batch)

    assert step_output["outputs"]["logits"].shape == (
        2,
        len(REDLAMP_MULTICLASS_CLASS_NAMES),
    )
    assert step_output["outputs"]["aux"]["class_probabilities"].shape == (
        2,
        len(REDLAMP_MULTICLASS_CLASS_NAMES),
    )
    assert torch.allclose(
        step_output["outputs"]["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(2),
    )
    assert step_output["batch"]["classification_labels"].min().item() >= 1
```

Add a label smoothing test:

```python
def test_redlamp_multiclass_label_refurbishment_matches_redlamp_formula() -> None:
    model = _build_model(
        num_classes=12,
        use_label_refurbishment=True,
        classification_label_mode="redlamp_multiclass",
        refurbishment_alpha=0.1,
        refurbishment_beta=0.01,
    )
    labels = torch.tensor([0, 1, 11], dtype=torch.long)

    targets = model._build_refurbished_classification_targets(
        labels,
        target_dtype=torch.float32,
    )

    assert targets.shape == (3, 12)
    assert torch.allclose(targets.sum(dim=-1), torch.ones(3), atol=1.0e-6)
    assert targets[0, 0] > targets[0, 1]
    assert targets[1, 1] > targets[1, 0]
    assert targets[2, 11] > targets[2, 0]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_one_multitask_train_step.py tests/test_multitask_shapes.py
```

Expected result: tests fail because `classification_label_mode`, `class_probabilities`, and the generalized target builder do not exist.

- [ ] **Step 3: Add classification mode to thesis model config**

Modify `SyntheticAnomalyConfig` in `src/models/thesis_multitask.py`:

```python
@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    use_synthetic_augmentation: bool = True
    use_synthetic_validation: bool = True
    synthetic_validation_seed: int = 7
    anomaly_probability: float = 0.5
    min_segment_fraction: float = 0.1
    max_segment_fraction: float = 0.2
    spike_scale: float = 3.0
    balance_binary_classes_within_batch: bool = False
    anomaly_families: tuple[str, ...] = REDLAMP_ANOMALY_FAMILIES
    classification_label_mode: str = "binary"

    def __post_init__(self) -> None:
        object.__setattr__(self, "anomaly_families", tuple(self.anomaly_families))
        if self.classification_label_mode not in {"binary", "redlamp_multiclass"}:
            raise ValueError(
                "classification_label_mode must be one of: binary, redlamp_multiclass"
            )
```

Add `"classification_label_mode"` to `synthetic_keys` in `ThesisMultitaskModelConfig.from_flat_kwargs`.

In `_store_config_values`, store:

```python
self.classification_label_mode = synthetic.classification_label_mode
```

When constructing both synthetic injectors, pass:

```python
classification_label_mode=config.synthetic.classification_label_mode,
```

- [ ] **Step 4: Add class probabilities and generalized label smoothing**

In `forward`, after logits:

```python
class_probabilities = torch.softmax(logits, dim=-1)
```

In `outputs["aux"]`, add:

```python
"class_probabilities": class_probabilities,
```

Replace `_build_refurbished_binary_targets` with:

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

    beta_background = torch.full_like(target_probabilities, self.refurbishment_beta)
    target_probabilities = torch.where(
        target_probabilities > 0.0,
        1.0 - (self.refurbishment_alpha + self.refurbishment_beta * self.num_classes - self.refurbishment_beta),
        beta_background,
    )
    target_probabilities[:, 0] = target_probabilities[:, 0] + self.refurbishment_alpha
    return target_probabilities / target_probabilities.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
```

Modify `_compute_classification_loss`:

```python
if self.use_label_refurbishment:
    target_probabilities = self._build_refurbished_classification_targets(
        batch["classification_labels"],
        outputs["logits"].dtype,
    )
    log_probabilities = F.log_softmax(outputs["logits"], dim=-1)
    return torch.mean(torch.sum(-target_probabilities * log_probabilities, dim=-1))

return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())
```

- [ ] **Step 5: Run focused tests and verify pass**

Run:

```bash
pytest -q tests/test_thesis_multitask_config_refactor.py tests/test_one_multitask_train_step.py tests/test_multitask_shapes.py
```

Expected result: focused tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/models/thesis_multitask.py tests/test_thesis_multitask_config_refactor.py tests/test_one_multitask_train_step.py tests/test_multitask_shapes.py
git commit -m "Add RedLamp multiclass thesis classification"
```

---

### Task 3: Add Multi-Class Classification Metrics To The Trainer

**Files:**
- Modify: `src/metrics/pointwise.py`
- Modify: `src/engine/trainer.py`
- Test: `tests/test_evaluator_thresholding.py`
- Test: `tests/test_multitask_metrics_runtime.py`

- [ ] **Step 1: Write failing metric tests**

Add to `tests/test_evaluator_thresholding.py`:

```python
def test_compute_multiclass_classification_metrics_reports_accuracy_and_macro_f1() -> None:
    from src.metrics.pointwise import compute_multiclass_classification_metrics

    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 2.0, 3.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)

    metrics = compute_multiclass_classification_metrics(logits=logits, labels=labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["num_classes_observed"] == 3.0
```

- [ ] **Step 2: Run metric tests and verify failure**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py::test_compute_multiclass_classification_metrics_reports_accuracy_and_macro_f1
```

Expected result: import failure for `compute_multiclass_classification_metrics`.

- [ ] **Step 3: Implement multi-class metrics**

Modify imports in `src/metrics/pointwise.py`:

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
    prediction_array = torch.argmax(logits.detach().cpu(), dim=-1).numpy().astype(np.int64)
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

- [ ] **Step 4: Dispatch trainer metrics by class count**

Modify `src/engine/trainer.py` imports:

```python
from src.metrics.pointwise import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
)
```

Modify `_aggregate_multitask_classification_metrics`:

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

- [ ] **Step 5: Run trainer and metric tests**

Run:

```bash
pytest -q tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
```

Expected result: tests pass, and binary metric tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/metrics/pointwise.py src/engine/trainer.py tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
git commit -m "Add multiclass classification metrics"
```

---

### Task 4: Create The RedLamp MLP Baseline Model

**Files:**
- Create: `src/models/redlamp_mlp_baseline.py`
- Test: `tests/test_redlamp_mlp_baseline.py`

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_redlamp_mlp_baseline.py`:

```python
import torch
import torch.nn as nn

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def _build_batch(batch_size: int = 3, window_size: int = 10, input_dim: int = 4) -> dict[str, object]:
    return {
        "x": torch.randn(batch_size, window_size, input_dim),
        "point_labels": torch.zeros(batch_size, window_size, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"} for _ in range(batch_size)],
    }


def test_redlamp_mlp_baseline_forward_contract_and_probabilities() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=10,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
        use_label_refurbishment=True,
    )

    step_output = model.training_step(_build_batch())

    assert step_output["outputs"]["recon"].shape == (3, 10, 4)
    assert step_output["outputs"]["hidden"].shape == (3, 10, 16)
    assert step_output["outputs"]["logits"].shape == (3, len(REDLAMP_MULTICLASS_CLASS_NAMES))
    assert torch.allclose(
        step_output["outputs"]["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(3),
        atol=1.0e-6,
    )
    assert step_output["loss_terms"]["reconstruction_loss"].item() >= 0.0
    assert step_output["loss_terms"]["classification_loss"].item() >= 0.0


def test_redlamp_mlp_baseline_uses_three_linear_layers_per_mlp_stack() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=10,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
    )

    encoder_linear_layers = [layer for layer in model.encoder if isinstance(layer, nn.Linear)]
    decoder_linear_layers = [layer for layer in model.decoder if isinstance(layer, nn.Linear)]
    classifier_linear_layers = [
        layer for layer in model.classification_head if isinstance(layer, nn.Linear)
    ]

    assert len(encoder_linear_layers) == 3
    assert len(decoder_linear_layers) == 3
    assert len(classifier_linear_layers) == 3
```

- [ ] **Step 2: Run baseline tests and verify failure**

Run:

```bash
pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result: import failure because `src/models/redlamp_mlp_baseline.py` does not exist.

- [ ] **Step 3: Implement `RedLampMLPBaseline`**

Create `src/models/redlamp_mlp_baseline.py` with this structure:

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import validate_batch, validate_model_outputs
from src.data.augment import (
    REDLAMP_ANOMALY_FAMILIES,
    REDLAMP_MULTICLASS_CLASS_NAMES,
    SyntheticAnomalyInjector,
)
from src.models.base_model import BaseModel
from src.models.thesis_multitask import build_multilayer_perceptron


class RedLampMLPBaseline(BaseModel):
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
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.use_label_refurbishment = use_label_refurbishment
        self.refurbishment_alpha = refurbishment_alpha
        self.refurbishment_beta = refurbishment_beta
        self.use_synthetic_augmentation = use_synthetic_augmentation
        self.use_synthetic_validation = use_synthetic_validation
        self.epsilon = 1.0e-6

        flattened_dim = input_dim * window_size
        self.encoder = build_multilayer_perceptron(
            input_dim=flattened_dim,
            intermediate_dim=max(latent_dim * 2, classifier_dim),
            output_dim=latent_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=True,
        )
        self.decoder = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=max(latent_dim * 2, classifier_dim),
            output_dim=flattened_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
        self.classification_head = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=classifier_dim,
            output_dim=num_classes,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
        self.synthetic_anomaly_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
            deterministic_seed=None,
            classification_label_mode="redlamp_multiclass",
        )
        self.synthetic_validation_injector = SyntheticAnomalyInjector(
            anomaly_probability=anomaly_probability,
            min_segment_fraction=min_segment_fraction,
            max_segment_fraction=max_segment_fraction,
            spike_scale=spike_scale,
            anomaly_families=anomaly_families,
            deterministic_seed=synthetic_validation_seed,
            classification_label_mode="redlamp_multiclass",
        )

    def prepare_synthetic_validation_epoch(self) -> None:
        self.synthetic_validation_injector.reset_rng()

    def _prepare_batch(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        if stage_name == "train" and self.use_synthetic_augmentation:
            return self.synthetic_anomaly_injector.augment_batch(batch)
        if stage_name == "val_synth" and self.use_synthetic_validation:
            return self.synthetic_validation_injector.augment_batch(batch)
        clean_batch = dict(batch)
        batch_size = batch["x"].shape[0]
        clean_batch["classification_labels"] = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=batch["x"].device,
        )
        clean_batch["classification_class_names"] = REDLAMP_MULTICLASS_CLASS_NAMES
        return clean_batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        validate_batch(batch)
        x_tensor = batch["x"]
        batch_size, window_size, input_dim = x_tensor.shape
        if window_size != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {window_size}")
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")
        flattened_x = x_tensor.reshape(batch_size, window_size * input_dim)
        latent = self.encoder(flattened_x)
        reconstructed_flat = self.decoder(latent)
        recon = reconstructed_flat.reshape(batch_size, window_size, input_dim)
        logits = self.classification_head(latent)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
        hidden = latent.unsqueeze(1).expand(batch_size, window_size, self.latent_dim)
        outputs = {
            "hidden": hidden,
            "pooled": latent,
            "recon": recon,
            "logits": logits,
            "point_scores": point_scores,
            "window_scores": point_scores.mean(dim=1),
            "aux": {
                "class_probabilities": torch.softmax(logits, dim=-1),
                "class_names": REDLAMP_MULTICLASS_CLASS_NAMES,
            },
        }
        validate_model_outputs(outputs)
        return outputs

    def _build_redlamp_smoothed_targets(
        self,
        classification_labels: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        hard_labels = classification_labels.long()
        one_hot_targets = F.one_hot(hard_labels, num_classes=self.num_classes).to(
            dtype=target_dtype
        )
        targets = torch.where(
            one_hot_targets > 0.0,
            1.0 - (self.refurbishment_alpha + self.refurbishment_beta * self.num_classes - self.refurbishment_beta),
            self.refurbishment_beta,
        )
        targets[:, 0] = targets[:, 0] + self.refurbishment_alpha
        return targets / targets.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)

    def _compute_classification_loss(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
    ) -> torch.Tensor:
        if self.use_label_refurbishment:
            targets = self._build_redlamp_smoothed_targets(
                batch["classification_labels"],
                outputs["logits"].dtype,
            )
            return torch.mean(
                torch.sum(-targets * F.log_softmax(outputs["logits"], dim=-1), dim=-1)
            )
        return F.cross_entropy(outputs["logits"], batch["classification_labels"].long())

    def _shared_step(self, batch: dict[str, Any], stage_name: str) -> dict[str, Any]:
        prepared_batch = self._prepare_batch(batch, stage_name)
        outputs = self.forward(prepared_batch)
        reconstruction_loss = torch.mean((outputs["recon"] - prepared_batch["x"]) ** 2)
        classification_loss = self._compute_classification_loss(outputs, prepared_batch)
        total_loss = reconstruction_loss + self.lambda_cls * classification_loss
        predicted_labels = torch.argmax(outputs["logits"], dim=-1)
        classification_accuracy = torch.mean(
            (predicted_labels == prepared_batch["classification_labels"]).float()
        )
        return {
            "loss": total_loss,
            "log": {
                f"{stage_name}_loss": float(total_loss.detach().cpu()),
                f"{stage_name}_reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                f"{stage_name}_classification_loss": float(classification_loss.detach().cpu()),
                f"{stage_name}_classification_accuracy": float(classification_accuracy.detach().cpu()),
            },
            "outputs": outputs,
            "loss_terms": {
                "reconstruction_loss": reconstruction_loss,
                "classification_loss": classification_loss,
            },
            "batch": prepared_batch,
        }

    def training_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="train")

    def validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="val")

    def synthetic_validation_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="val_synth")

    def test_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._shared_step(batch=batch, stage_name="test")
```

- [ ] **Step 4: Run baseline tests and verify pass**

Run:

```bash
pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result: all baseline tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/models/redlamp_mlp_baseline.py tests/test_redlamp_mlp_baseline.py
git commit -m "Add RedLamp MLP baseline"
```

---

### Task 5: Add Configs And Registry Support For Window-10 RedLamp Runs

**Files:**
- Modify: `src/core/config.py`
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate.py`
- Modify: `scripts/run_online_adaptation.py`
- Create: `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- Create: `configs/model/redlamp_mlp_baseline.yaml`
- Create: `configs/task/multitask_tsad_redlamp_multiclass_window10.yaml`
- Create: `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window10.yaml`
- Create: `configs/experiment/smd_redlamp_mlp_baseline_window10.yaml`
- Test: `tests/test_config_loading.py`

- [ ] **Step 1: Write failing config-loading tests**

Add to `tests/test_config_loading.py`:

```python
def test_load_thesis_redlamp_multiclass_window10_config() -> None:
    config = load_experiment_config(
        "configs/experiment/smd_thesis_multitask_redlamp_multiclass_window10.yaml"
    )

    assert config["data"]["window_size"] == 10
    assert config["model"]["model_name"] == "thesis_multitask"
    assert config["model"]["num_classes"] == 12
    assert config["task"]["classification_label_mode"] == "redlamp_multiclass"
    assert config["task"]["anomaly_families"] == [
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
    ]


def test_load_redlamp_mlp_baseline_window10_config() -> None:
    config = load_experiment_config(
        "configs/experiment/smd_redlamp_mlp_baseline_window10.yaml"
    )

    assert config["data"]["window_size"] == 10
    assert config["model"]["model_name"] == "redlamp_mlp_baseline"
    assert config["model"]["mlp_num_linear_layers"] == 3
    assert config["model"]["num_classes"] == 12
    assert config["task"]["classification_label_mode"] == "redlamp_multiclass"
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
pytest -q tests/test_config_loading.py::test_load_thesis_redlamp_multiclass_window10_config tests/test_config_loading.py::test_load_redlamp_mlp_baseline_window10_config
```

Expected result: config files and `redlamp_mlp_baseline` validation support are missing.

- [ ] **Step 3: Extend config validation**

Modify `src/core/config.py`:

```python
supported_model_names = {
    "reconstruction_mlp_ae",
    "thesis_multitask",
    "redlamp_mlp_baseline",
    "online_adaptation",
}
```

Treat `redlamp_mlp_baseline` as an offline model with integer fields:

```python
if model_config.get("model_name") == "redlamp_mlp_baseline":
    integer_fields["window_size"] = model_config.get("window_size", data_config.get("window_size"))
    integer_fields["latent_dim"] = model_config.get("latent_dim")
    integer_fields["mlp_num_linear_layers"] = model_config.get("mlp_num_linear_layers", 3)
    integer_fields["classifier_dim"] = model_config.get("classifier_dim")
    integer_fields["num_classes"] = model_config.get("num_classes")
```

Add float validation for baseline:

```python
if model_config.get("model_name") == "redlamp_mlp_baseline":
    float_fields["dropout"] = model_config.get("dropout")
    float_fields["lambda_cls"] = model_config.get("lambda_cls")
    float_fields["refurbishment_alpha"] = model_config.get("refurbishment_alpha")
    float_fields["refurbishment_beta"] = model_config.get("refurbishment_beta")
```

Add task validation:

```python
classification_label_mode = task_config.get("classification_label_mode", "binary")
if classification_label_mode not in {"binary", "redlamp_multiclass"}:
    raise ValueError(
        "classification_label_mode must be one of: binary, redlamp_multiclass"
    )
if classification_label_mode == "redlamp_multiclass" and int(model_config["num_classes"]) != 12:
    raise ValueError("redlamp_multiclass requires num_classes == 12")
```

- [ ] **Step 4: Register the new model**

Modify imports in `scripts/train.py` and `scripts/evaluate.py`:

```python
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline
```

Modify `register_runtime_components` in both files:

```python
register_model("redlamp_mlp_baseline", RedLampMLPBaseline)
```

Modify `build_model_from_experiment_config` in `scripts/train.py`, `scripts/evaluate.py`, and `scripts/run_online_adaptation.py` so model kwargs include `window_size` when the model is `redlamp_mlp_baseline`:

```python
if model_name == "redlamp_mlp_baseline":
    model_kwargs["window_size"] = experiment_config["data"]["window_size"]
```

- [ ] **Step 5: Create model and task configs**

Create `configs/model/thesis_multitask_redlamp_multiclass.yaml`:

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
window_size: 10
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

Create `configs/task/multitask_tsad_redlamp_multiclass_window10.yaml`:

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

- [ ] **Step 6: Create experiment configs**

Create `configs/experiment/smd_thesis_multitask_redlamp_multiclass_window10.yaml`:

```yaml
experiment_name: smd_thesis_multitask_redlamp_multiclass_window10
seed: 11
device: cpu
output_dir: outputs/smd_thesis_multitask_redlamp_multiclass_window10
checkpoint_dir: outputs/smd_thesis_multitask_redlamp_multiclass_window10/checkpoints
data_config_path: configs/data/smd_rtx3090_machine_2_1_10.yaml
model_config_path: configs/model/thesis_multitask_redlamp_multiclass.yaml
task_config_path: configs/task/multitask_tsad_redlamp_multiclass_window10.yaml
optimizer:
  learning_rate: 0.0001
  weight_decay: 0.0
epochs: 1
logging:
  use_wandb: false
  wandb_project: bachelor-thesis-2026
  wandb_mode: disabled
  wandb_tags:
    - smd
    - thesis_multitask
    - redlamp_multiclass
    - window10
    - mlp-depth-3
```

Create `configs/experiment/smd_redlamp_mlp_baseline_window10.yaml`:

```yaml
experiment_name: smd_redlamp_mlp_baseline_window10
seed: 11
device: cpu
output_dir: outputs/smd_redlamp_mlp_baseline_window10
checkpoint_dir: outputs/smd_redlamp_mlp_baseline_window10/checkpoints
data_config_path: configs/data/smd_rtx3090_machine_2_1_10.yaml
model_config_path: configs/model/redlamp_mlp_baseline.yaml
task_config_path: configs/task/multitask_tsad_redlamp_multiclass_window10.yaml
optimizer:
  learning_rate: 0.0001
  weight_decay: 0.0
epochs: 1
logging:
  use_wandb: false
  wandb_project: bachelor-thesis-2026
  wandb_mode: disabled
  wandb_tags:
    - smd
    - redlamp_mlp_baseline
    - redlamp_multiclass
    - window10
    - mlp-depth-3
```

- [ ] **Step 7: Run config tests and verify pass**

Run:

```bash
pytest -q tests/test_config_loading.py
```

Expected result: config loading tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/core/config.py scripts/train.py scripts/evaluate.py scripts/run_online_adaptation.py configs/model/thesis_multitask_redlamp_multiclass.yaml configs/model/redlamp_mlp_baseline.yaml configs/task/multitask_tsad_redlamp_multiclass_window10.yaml configs/experiment/smd_thesis_multitask_redlamp_multiclass_window10.yaml configs/experiment/smd_redlamp_mlp_baseline_window10.yaml tests/test_config_loading.py
git commit -m "Add window10 RedLamp multiclass configs"
```

---

### Task 6: Verify End-To-End Training Surface

**Files:**
- Modify: `tests/test_one_multitask_train_step.py`
- Create: `tests/test_one_redlamp_mlp_train_step.py`

- [ ] **Step 1: Add one-step training tests**

Create `tests/test_one_redlamp_mlp_train_step.py`:

```python
import torch

from src.data.augment import REDLAMP_MULTICLASS_CLASS_NAMES
from src.models.redlamp_mlp_baseline import RedLampMLPBaseline


def test_one_redlamp_mlp_train_step_runs_backward() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=10,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 10, 4),
        "point_labels": torch.zeros(2, 10, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    step_output = model.training_step(batch)
    step_output["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert step_output["batch"]["classification_labels"].shape == (2,)
```

- [ ] **Step 2: Run one-step tests**

Run:

```bash
pytest -q tests/test_one_multitask_train_step.py tests/test_one_redlamp_mlp_train_step.py
```

Expected result: both proposed method and RedLamp MLP baseline can run one forward and backward pass.

- [ ] **Step 3: Run smoke config load plus model construction tests**

Run:

```bash
pytest -q tests/test_config_loading.py tests/test_registry.py tests/test_redlamp_mlp_baseline.py
```

Expected result: model registry, config loading, and baseline shape tests pass.

- [ ] **Step 4: Run broad relevant regression tests**

Run:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_multitask_metrics_runtime.py tests/test_checkpoint_roundtrip.py tests/test_config_loading.py tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
```

Expected result: all listed tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_one_multitask_train_step.py tests/test_one_redlamp_mlp_train_step.py
git commit -m "Verify RedLamp multiclass training steps"
```

---

## Validation Procedure

1. Run focused injector tests:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py
```

2. Run model-level tests:

```bash
pytest -q tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
```

3. Run metrics and trainer tests:

```bash
pytest -q tests/test_evaluator_thresholding.py tests/test_multitask_metrics_runtime.py
```

4. Run config tests:

```bash
pytest -q tests/test_config_loading.py
```

5. Run the final relevant suite:

```bash
pytest -q tests/test_synthetic_anomaly_injection.py tests/test_multitask_shapes.py tests/test_one_multitask_train_step.py tests/test_multitask_metrics_runtime.py tests/test_checkpoint_roundtrip.py tests/test_config_loading.py tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
```

## Self-Review

- **Spec coverage:** Task 1 aligns anomaly labels across both methods. Task 2 changes the proposed method to multi-class classification and adds RedLamp-style label smoothing. Task 4 creates the RedLamp MLP baseline. Task 5 creates window-10 configs following CANDI. Task 6 verifies one-step training.
- **Placeholder scan:** This plan does not use placeholder implementation steps. Each task includes concrete files, code snippets, commands, and expected results.
- **Type consistency:** `classification_label_mode`, `classification_labels`, `classification_class_names`, `REDLAMP_MULTICLASS_CLASS_NAMES`, `class_probabilities`, and `num_classes` are used consistently across injector, models, metrics, trainer, configs, and tests.
