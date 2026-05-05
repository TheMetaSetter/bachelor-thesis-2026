---
date: 2026-05-05 12:44:49 +07
author: Artificial Intelligence Agent
git_commit: a0f3294eedf29ac65debea0794bb3b5f6827121a
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for ThesisMultitaskModel constructor configuration refactor"
tags: [detail, time-series, anomaly-detection, multitask, configuration, refactor]
status: complete
last_updated: 2026-05-05
last_updated_by: Artificial Intelligence Agent
---

# Detailed Plan: ThesisMultitaskModel Constructor Configuration Refactor

## Scope

This document records the detailed implementation decisions for refactoring the
constructor of `ThesisMultitaskModel` in `src/models/thesis_multitask.py`. The
current constructor exposes approximately fifty flat parameters. That shape is
difficult to read, difficult to test, and difficult to maintain as the thesis
model accumulates architecture, prototype, objective, memory, schedule, and
synthetic anomaly settings.

The selected implementation strategy is compatibility-first. The model will
gain explicit same-file configuration dataclasses and private setup helpers, but
existing flat keyword construction from YAML, scripts, tests, and checkpoint
reload paths will continue to work.

This plan is written against commit
`a0f3294eedf29ac65debea0794bb3b5f6827121a` on branch `dev`. A focused
pre-refactor verification run passed with:

```bash
pytest -q tests/test_multitask_shapes.py tests/test_multitask_objective_controls.py tests/test_temperature_schedule.py tests/test_multitask_memory_bootstrap.py tests/test_checkpoint_roundtrip.py tests/test_config_loading.py
```

The observed result was `36 passed in 64.06s`.

There is an existing unstaged local edit in `src/models/thesis_multitask.py`
around comments in the continuous memory update path. The implementation must
preserve that local work.

## Phase 1 - Introduce same-file configuration dataclasses

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make the offline prototype-fusion model
readable and ablation-friendly while preserving the repository rule that one
model belongs in one file. Grouping constructor settings into configuration
objects makes the model easier to inspect without moving model-owned inference,
training, synthetic anomaly, memory, or loss logic into separate files.

### File-level edits

Modify `src/models/thesis_multitask.py`.

Add `dataclass` to the imports:

```python
from dataclasses import dataclass
```

Keep the existing `typing` imports and retain `Any` because the flat YAML and
registry path still pass dictionaries of ordinary Python values.

Add these dataclasses near the top of the file, after
`MultitaskWindowEncoder` and before `ThesisMultitaskModel`:

```python
@dataclass(frozen=True)
class MultitaskArchitectureConfig:
    input_dim: int
    encoder_dim: int
    hidden_dim: int
    mlp_num_linear_layers: int = 3
    num_classes: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PrototypeBranchConfig:
    continuous_enabled: bool = True
    continuous_num_prototypes: int = 8
    discrete_enabled: bool = True
    discrete_codebook_size: int = 16
    gumbel_temperature: float = 1.0
    discrete_ema_decay: float = 0.99


@dataclass(frozen=True)
class ScheduleAndWarmupConfig:
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    temperature_anneal_fraction: float = 1.0
    temperature_hold_fraction: float = 0.0
    usage_lambda_start: float | None = None
    usage_lambda_end: float | None = None
    usage_lambda_schedule_fraction: float = 1.0
    freeze_fusion_for_epochs: int = 0
    warmup_alpha_value: float = 0.5
    warmup_beta_value: float = 0.5


@dataclass(frozen=True)
class ObjectiveConfig:
    alpha_logit_init: float = 0.0
    beta_logit_init: float = 0.0
    use_label_refurbishment: bool = False
    refurbishment_alpha: float = 0.0
    refurbishment_beta: float = 0.0
    reconstruction_normal_only: bool = False
    lambda_cls: float = 1.0
    enable_diversity_loss: bool = False
    enable_variance_loss: bool = False
    enable_covariance_loss: bool = False
    enable_usage_loss: bool = False
    enable_gate_loss: bool = False
    lambda_div: float = 0.0
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    lambda_use: float = 0.0
    lambda_gate: float = 0.0
    variance_floor_gamma: float = 1.0
    gate_barrier_margin: float = 0.25


@dataclass(frozen=True)
class MemoryInitializationConfig:
    bootstrap_encoder_epochs: int = 0
    memory_norm_epsilon: float = 1.0e-6
    memory_initialization_batches: int = 16
    memory_initialization_with_synthetic_windows: bool = True


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


@dataclass(frozen=True)
class ThesisMultitaskModelConfig:
    architecture: MultitaskArchitectureConfig
    prototypes: PrototypeBranchConfig = PrototypeBranchConfig()
    schedule: ScheduleAndWarmupConfig = ScheduleAndWarmupConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    memory: MemoryInitializationConfig = MemoryInitializationConfig()
    synthetic: SyntheticAnomalyConfig = SyntheticAnomalyConfig()
```

If Python raises a mutable-default warning for nested dataclass instances, use
`field(default_factory=...)` for the nested sections. Keep all dataclasses in the
same model file.

### Interface and contract definitions

The dataset contract is unchanged. Batches continue to be dictionaries with
`x`, optional labels, masks, timestamps, and metadata.

The encoder contract is unchanged. `MultitaskWindowEncoder.forward()` continues
to return `hidden`, `pooled`, and `aux`.

The model contract is unchanged. `ThesisMultitaskModel.forward()` and the stage
methods continue to return the current model output dictionary, including
reconstruction, logits, scores, and auxiliary branch artifacts.

The task contract is unchanged. Reconstruction and synthetic anomaly
classification remain owned by the model stage methods.

The training engine contract is unchanged. The trainer continues to consume the
standardized model outputs and does not need to know about the dataclasses.

### Design pattern application

Use composition over inheritance. Configuration grouping should not introduce a
new model base class or subclass family.

The encoder remains an internal adapter from standardized batches to the
thesis-facing hidden representation. No external encoder adapter is introduced
in this refactor.

The existing task strategy remains configuration-driven through model and task
YAML sections. The refactor should not create separate task classes.

The registry and factory path remain unchanged. `register_model("thesis_multitask",
ThesisMultitaskModel)` and `build_model(model_name, **model_kwargs)` must remain
valid.

### Risk mitigation

Prototype redundancy is mitigated by leaving continuous and discrete prototype
lookup, update, and optional-loss logic unchanged.

Fusion collapse is mitigated by leaving `alpha`, `beta`, fusion warmup, and
fusion equations unchanged.

Adaptation contamination is mitigated by preserving the offline reference model
loading path used by `src/models/online_adaptation.py`.

Projector drift is out of scope for this refactor because the online projector
logic is not modified.

Evaluation metric inflation is mitigated by leaving evaluation and thresholding
code unchanged and by not claiming performance changes from a constructor-only
refactor.

### Acceptance criteria

- The new dataclasses are defined in `src/models/thesis_multitask.py`.
- No model logic is moved out of the model file.
- Existing public model outputs are unchanged.
- The registry path remains compatible with flat keyword construction.

## Phase 2 - Add a flat-keyword compatibility factory

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make configuration readable without
breaking reproducibility. Existing resolved experiment configurations and saved
checkpoints store flat model and task dictionaries, so the model must continue
to accept flat keyword arguments.

### File-level edits

Modify `ThesisMultitaskModelConfig` in `src/models/thesis_multitask.py`.

Add a class method:

```python
@classmethod
def from_flat_kwargs(cls, flat_kwargs: dict[str, Any]) -> "ThesisMultitaskModelConfig":
    ...
```

This method must:

- accept every current keyword in the existing constructor;
- preserve current defaults exactly;
- raise `ValueError` when unknown keys are supplied;
- convert `anomaly_families` to `tuple[str, ...]`;
- preserve the current `usage_lambda_start` and `usage_lambda_end` behavior,
  where omitted values fall back to `lambda_use` later in model initialization.

Use explicit key groupings instead of dynamic introspection. This keeps the
configuration surface readable for a thesis codebase.

### Explicit edit content

The flat architecture keys are:

```python
{
    "input_dim",
    "encoder_dim",
    "hidden_dim",
    "mlp_num_linear_layers",
    "num_classes",
    "dropout",
}
```

The prototype keys are:

```python
{
    "continuous_enabled",
    "continuous_num_prototypes",
    "discrete_enabled",
    "discrete_codebook_size",
    "gumbel_temperature",
    "discrete_ema_decay",
}
```

The schedule and warmup keys are:

```python
{
    "temperature_start",
    "temperature_end",
    "temperature_anneal_fraction",
    "temperature_hold_fraction",
    "usage_lambda_start",
    "usage_lambda_end",
    "usage_lambda_schedule_fraction",
    "freeze_fusion_for_epochs",
    "warmup_alpha_value",
    "warmup_beta_value",
}
```

The objective keys are:

```python
{
    "alpha_logit_init",
    "beta_logit_init",
    "use_label_refurbishment",
    "refurbishment_alpha",
    "refurbishment_beta",
    "reconstruction_normal_only",
    "lambda_cls",
    "enable_diversity_loss",
    "enable_variance_loss",
    "enable_covariance_loss",
    "enable_usage_loss",
    "enable_gate_loss",
    "lambda_div",
    "lambda_var",
    "lambda_cov",
    "lambda_use",
    "lambda_gate",
    "variance_floor_gamma",
    "gate_barrier_margin",
}
```

The memory keys are:

```python
{
    "bootstrap_encoder_epochs",
    "memory_norm_epsilon",
    "memory_initialization_batches",
    "memory_initialization_with_synthetic_windows",
}
```

The synthetic anomaly keys are:

```python
{
    "use_synthetic_augmentation",
    "use_synthetic_validation",
    "synthetic_validation_seed",
    "anomaly_probability",
    "min_segment_fraction",
    "max_segment_fraction",
    "spike_scale",
    "balance_binary_classes_within_batch",
    "anomaly_families",
}
```

### Acceptance criteria

- Flat YAML-driven kwargs map cleanly into `ThesisMultitaskModelConfig`.
- Unknown keys fail early with a readable error message.
- The config object can be inspected by section.

## Phase 3 - Refactor constructor entrypoint and setup helpers

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make the model readable from top to
bottom. The constructor should express the model assembly sequence rather than
interleaving every architecture, schedule, memory, objective, and anomaly field
assignment in one long block.

### File-level edits

Modify the constructor signature in `src/models/thesis_multitask.py`:

```python
def __init__(
    self,
    config: ThesisMultitaskModelConfig | None = None,
    **flat_kwargs: Any,
) -> None:
```

Constructor normalization rules:

```python
if config is not None and flat_kwargs:
    raise ValueError("Pass either config or flat keyword arguments, not both")
if config is None:
    config = ThesisMultitaskModelConfig.from_flat_kwargs(flat_kwargs)
```

Then call setup helpers in this order:

```python
self._store_config_values(config)
self._build_encoder(config)
self._build_prototype_memory(config)
self._build_fusion_parameters(config)
self._build_task_heads(config)
self._build_synthetic_injectors(config)
self._build_optional_loss_configs()
self.set_epoch_context(epoch_index=0, total_epochs=1)
self._print_model_summary(config)
```

Add private helpers on `ThesisMultitaskModel`:

- `_store_config_values`
- `_build_encoder`
- `_build_prototype_memory`
- `_build_fusion_parameters`
- `_build_task_heads`
- `_build_synthetic_injectors`
- `_build_optional_loss_configs`
- `_print_model_summary`

These helpers should move existing code without changing behavior. Preserve
module names, parameter names, buffer names, and auxiliary output names.

### Interface and contract definitions

The public constructor gains a config-object path but keeps flat keyword support.
The following calls must remain valid:

```python
ThesisMultitaskModel(input_dim=38, encoder_dim=64, hidden_dim=16)
```

```python
config = ThesisMultitaskModelConfig.from_flat_kwargs(model_kwargs)
ThesisMultitaskModel(config)
```

The following call must fail:

```python
ThesisMultitaskModel(config, input_dim=38)
```

### Acceptance criteria

- The constructor is short and readable.
- Existing scripts do not require schema changes.
- Existing checkpoints can still reconstruct the reference model from saved
  flat config dictionaries.

## Phase 4 - Add focused tests

### Phase summary tied to thesis objectives

The thesis objective in this phase is to make configuration behavior testable
without requiring full training runs. The tests should verify that the refactor
is behavior-preserving and that future configuration changes fail clearly.

### File-level edits

Modify or extend `tests/test_multitask_shapes.py` or add a focused test file
such as `tests/test_thesis_multitask_config.py`.

Add tests for:

- flat keyword construction still works;
- config-object construction works;
- both construction paths preserve key attributes;
- unknown flat keys raise `ValueError`;
- `anomaly_families` is normalized to a tuple;
- omitted `usage_lambda_start` and `usage_lambda_end` still fall back to
  `lambda_use` in the model state.

### Test plan and validation steps

Run:

```bash
pytest -q tests/test_multitask_shapes.py tests/test_multitask_objective_controls.py tests/test_temperature_schedule.py tests/test_multitask_memory_bootstrap.py tests/test_checkpoint_roundtrip.py tests/test_config_loading.py
```

If time permits, run:

```bash
pytest -q tests/test_multitask_memory_updates.py tests/test_multitask_memory_initialization.py tests/test_multitask_validation_alignment.py tests/test_one_multitask_train_step.py tests/test_online_reference_checkpoint.py tests/test_online_state_roundtrip.py tests/test_fusion_ablation_modes.py
```

### Acceptance criteria

- Focused tests pass.
- Constructor compatibility is covered directly.
- Checkpoint roundtrip remains passing.

## Phase 5 - Preserve runtime documentation and experiment behavior

### Phase summary tied to thesis objectives

The thesis objective in this phase is reproducibility. Because the active
pipeline is YAML-driven, the implementation should not silently alter the
meaning of existing experiment files.

### File-level edits

No YAML schema migration is required in this first pass.

Do not change these files unless a test reveals a direct compatibility issue:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_online_adaptation.py`
- `src/models/online_adaptation.py`
- `configs/model/thesis_multitask.yaml`
- `configs/task/multitask_tsad.yaml`

If documentation is added, it should state that flat YAML remains the canonical
runtime input and the dataclasses are the internal model-side organization.

### Acceptance criteria

- Existing experiment YAML files continue to load.
- Existing runtime builders continue to instantiate the model through the
  registry.
- No training, evaluation, or online adaptation entrypoint needs a required
  interface change.

## Final acceptance criteria

- `ThesisMultitaskModel` has a readable grouped configuration surface.
- Existing flat keyword construction remains supported.
- The public model output contract is unchanged.
- The model remains self-contained in `src/models/thesis_multitask.py`.
- State dictionary compatibility is preserved by keeping parameter and buffer
  names unchanged.
- Focused multitask, configuration, schedule, memory, and checkpoint tests pass.
- The implementation does not alter prototype behavior, fusion behavior,
  online adaptation behavior, or evaluation behavior.
