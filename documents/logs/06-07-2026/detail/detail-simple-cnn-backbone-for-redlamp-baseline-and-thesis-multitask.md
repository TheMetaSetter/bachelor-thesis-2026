---
date: 2026-06-07 15:40:00 +0700
researcher: Codex
git_commit: 32417993875f677a86743ab3a770d0ccc67b32fe
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed implementation plan for replacing active MLP encoders with a simple RedLamp-style CNN backbone in redlamp_mlp_baseline.py and thesis_multitask.py"
tags: [detail, plan, cnn, redlamp, baseline, thesis-model, anomaly-detection]
status: complete
last_updated: 2026-06-07
last_updated_by: Codex
---

# Detailed Implementation Plan: Simple CNN Backbone for RedLamp Baseline and Thesis Multitask Models

**Objective.** Replace the active timestep-MLP encoder with a simple RedLamp-style one-dimensional CNN backbone in the baseline model and the thesis multitask model while preserving the repository batch contract, encoder contract, and model output contract.

**Thesis alignment.** The proposed implementation follows the thesis design principle that the encoder interface must remain stable even when the underlying feature extractor changes. The new CNN backbone must therefore behave as an internal adapter that transforms the standard offline batch `batch["x"]: Tensor[B, L, D]` into hidden states `hidden: Tensor[B, L, H]` without changing the downstream prototype branches, reconstruction head, classification head, or trainer interfaces.

**Implementation principle.** Use composition over inheritance, keep each model self-contained in its own file, and preserve a small number of explicit codepaths. The CNN backbone should be introduced as a local adapter inside the owning model file, not as a new cross-cutting subsystem.

---

## Phase 1: Baseline CNN Vertical Slice

### Phase summary

This phase replaces the active MLP encoder in `src/models/redlamp_mlp_baseline.py` with a simple CNN backbone that preserves the existing baseline output geometry. The purpose of this phase is to establish a minimal, testable vertical slice before any thesis-model prototype or fusion logic is touched. The baseline must remain runnable through the current training and evaluation pipeline, and it must continue to support gradient profiling for the encoder parameters.

### File-level edits

- Modify `src/models/redlamp_mlp_baseline.py`
- Modify `configs/model/redlamp_mlp_baseline.yaml`
- Add `tests/test_redlamp_cnn_baseline_shapes.py`
- Add `tests/test_cnn_encoder_config_loading.py`

### Interface and contract definitions

The baseline model must preserve the following repository contracts:

```python
batch["x"].shape == [batch_size, window_size, input_dim]
```

```python
encoder_outputs["hidden"].shape == [batch_size, window_size, latent_dim]
```

```python
outputs["recon"].shape == [batch_size, window_size, input_dim]
outputs["point_scores"].shape == [batch_size, window_size]
outputs["window_scores"].shape == [batch_size]
```

The new CNN encoder must accept `x: Tensor[B, L, D]`, transpose internally to channel-first format `[B, D, L]`, apply temporal convolutions with same-length padding, and return `hidden: Tensor[B, L, H]`. The public baseline head geometry must remain unchanged so that the classifier and decoder continue to operate on the same logical window structure.

### Programming edits in detail

1. Add an explicit `encoder_family` parameter to the baseline constructor.
2. Add a local CNN encoder class inside `src/models/redlamp_mlp_baseline.py`.
3. Preserve the existing MLP path as a selectable ablation branch.
4. Generalize encoder gradient profiling so it is not hard-coded to `nn.Linear`.
5. Extend the baseline YAML config with explicit CNN keys and safe defaults.

### Exact code changes

#### `src/models/redlamp_mlp_baseline.py`

Add a small adapter class in the same file, for example `SimpleWindowCnnEncoder`, with the following responsibilities:

```python
class SimpleWindowCnnEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_channels: int,
        kernel_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        x_channel_first = x.transpose(1, 2)  # [B, D, L]
        hidden_channel_first = self.network(x_channel_first)  # [B, H, L]
        hidden = hidden_channel_first.transpose(1, 2)  # [B, L, H]
        return hidden
```

Update the baseline constructor signature to accept an explicit encoder family selector and CNN hyperparameters:

```python
def __init__(
    self,
    input_dim: int,
    window_size: int,
    latent_dim: int,
    num_classes: int,
    encoder_family: str = "mlp",
    mlp_num_linear_layers: int = 3,
    cnn_num_layers: int = 3,
    cnn_kernel_size: int = 3,
    cnn_hidden_channels: int = 64,
    dropout: float = 0.1,
    ...
) -> None:
```

Introduce a local construction branch:

```python
if encoder_family == "mlp":
    self.encoder = build_multilayer_perceptron(...)
elif encoder_family == "cnn_simple":
    self.encoder = SimpleWindowCnnEncoder(...)
else:
    raise ValueError(...)
```

Update the gradient profiling helper so the code no longer assumes that the encoder is a pure `nn.Linear` stack. The implementation should expose a family-agnostic notion such as `encoder_last_affine` and should cover both `nn.Linear` and `nn.Conv1d` layers when choosing the profiled target.

#### `configs/model/redlamp_mlp_baseline.yaml`

Add explicit keys for the encoder family and the CNN backbone. The config should remain backward compatible by keeping the MLP path as the default. The file should make the choice visible rather than implicit.

Recommended keys:

```yaml
encoder_family: mlp
cnn_num_layers: 3
cnn_kernel_size: 3
cnn_hidden_channels: 64
cnn_dropout: 0.1
```

If the YAML loader uses flattened config groups, the same keys should be added in the location that the current model config parser already reads from.

#### `tests/test_redlamp_cnn_baseline_shapes.py`

Add a focused pytest module that verifies:

1. The CNN baseline instantiates from config.
2. A single forward pass returns the expected keys and shapes.
3. A single backward pass reaches encoder parameters.
4. The decoder still reconstructs to `[B, L, D]`.
5. The gradient profiling code does not crash under the CNN encoder.

Use a minimal synthetic batch with small dimensions to keep the test fast and deterministic.

#### `tests/test_cnn_encoder_config_loading.py`

Add a configuration regression test that verifies:

1. The old MLP config still loads.
2. The CNN config loads.
3. The parsed model object exposes the expected `encoder_family`.
4. Invalid encoder-family values raise a clear `ValueError`.

### Design pattern application

- Composition over inheritance: the CNN encoder should be a local component owned by the baseline file rather than a base class hierarchy.
- Adapter pattern: the encoder class adapts the canonical batch tensor shape to the convolutional internal layout and back to the public hidden-state layout.
- Registry or factory: the baseline constructor should act as a small factory for `mlp` versus `cnn_simple`, without introducing a separate global registry.
- Strategy pattern: the model family selector behaves as a strategy switch, but it must remain explicit and readable in the constructor.

### Risk mitigation

- Prototype redundancy is not applicable in this phase because the baseline does not yet branch into dual prototype heads.
- Fusion collapse is not applicable in this phase because the baseline path has a simpler head structure.
- Adaptation contamination is not applicable in this phase because there is no online update logic yet.
- Projector drift is mitigated by preserving the existing decoder and classification head geometry and changing only the encoder family.
- Evaluation metric inflation is mitigated by keeping the output contract unchanged so the same metrics remain comparable across runs.

### Acceptance criteria

This phase is complete only if all of the following are true:

1. `pytest -q tests/test_cnn_encoder_config_loading.py` passes.
2. `pytest -q tests/test_redlamp_cnn_baseline_shapes.py` passes.
3. The baseline model returns `hidden: [B, L, latent_dim]`, `recon: [B, L, D]`, `point_scores: [B, L]`, and `window_scores: [B]`.
4. The CNN encoder path preserves the time dimension `L`.
5. Gradient profiling still resolves encoder parameters without assuming `nn.Linear` only.

---

## Phase 2: Thesis Multitask CNN Adapter

### Phase summary

This phase extends the same CNN backbone idea to `src/models/thesis_multitask.py`. The aim is not to redesign the thesis model, but to swap the encoder family while preserving the prototype branches, fusion logic, reconstruction head, classification head, and all downstream hidden-state consumers. The thesis model must continue to be readable top-to-bottom in one file.

### File-level edits

- Modify `src/models/thesis_multitask.py`
- Modify `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- Add `tests/test_thesis_multitask_cnn_shapes.py`
- Update `tests/test_cnn_encoder_config_loading.py`

### Interface and contract definitions

The thesis model must preserve the following interfaces:

```python
batch["x"].shape == [batch_size, window_size, input_dim]
```

```python
encoder_outputs["hidden"].shape == [batch_size, window_size, hidden_dim]
encoder_outputs["pooled"].shape == [batch_size, hidden_dim]
```

```python
outputs["recon"].shape == [batch_size, window_size, input_dim]
outputs["logits"].shape == [batch_size, num_classes]
```

The prototype lookup, continuous memory update, discrete codebook update, and fusion computations must remain hidden-state consumers only. They should not need to know whether the hidden states originated from an MLP or a CNN.

### Programming edits in detail

1. Extend `MultitaskArchitectureConfig` with CNN fields and an `encoder_family` selector.
2. Update `ThesisMultitaskModelConfig.from_flat_kwargs(...)` so new encoder-family keys are accepted explicitly.
3. Add a CNN-aware branch to `MultitaskWindowEncoder`.
4. Keep the prototype/fusion/reconstruction/classification path unchanged at the public interface level.
5. Add shape assertions near the encoder boundary to fail early if the CNN changes the temporal geometry.

### Exact code changes

#### `src/models/thesis_multitask.py`

Extend the architecture dataclass with the minimum CNN surface needed for reproducibility:

```python
@dataclass
class MultitaskArchitectureConfig:
    input_dim: int
    window_size: int
    encoder_dim: int
    hidden_dim: int
    mlp_num_linear_layers: int
    num_classes: int
    dropout: float
    encoder_family: str = "mlp"
    cnn_num_layers: int = 3
    cnn_kernel_size: int = 3
    cnn_hidden_channels: int = 64
```

Update the config parser whitelist so that these keys can flow from the YAML config into the model object without being dropped or rejected.

Keep `MultitaskWindowEncoder` as the owning adapter class, but branch internally on the selected encoder family. The implementation should be explicit and readable rather than generalized through nested registries. A representative shape is:

```python
class MultitaskWindowEncoder(nn.Module):
    def __init__(self, architecture: MultitaskArchitectureConfig) -> None:
        if architecture.encoder_family == "mlp":
            self.network = build_multilayer_perceptron(...)
        elif architecture.encoder_family == "cnn_simple":
            self.network = build_simple_cnn_encoder(...)
        else:
            raise ValueError(...)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden = self.network(batch["x"])
        if hidden.ndim != 3:
            raise ValueError("Encoder must return hidden states with shape [B, L, H].")
        if hidden.shape[1] != self.architecture.window_size:
            raise ValueError("Encoder changed temporal length, which is not supported by the current prototype path.")
        pooled = hidden.mean(dim=1)
        return {"hidden": hidden, "pooled": pooled}
```

The prototype, reconstruction, classification, and fusion modules should continue to consume `hidden` exactly as they do now. No new branching logic should leak into those components during the first pass.

#### `configs/model/thesis_multitask_redlamp_multiclass.yaml`

Add explicit CNN-family settings with defaults that keep the old MLP path intact. The config should make the selected encoder family visible for reruns and ablations. The current experiment configs should remain reproducible without requiring hard-coded Python changes.

Recommended keys:

```yaml
encoder_family: mlp
cnn_num_layers: 3
cnn_kernel_size: 3
cnn_hidden_channels: 64
cnn_dropout: 0.1
```

#### `tests/test_thesis_multitask_cnn_shapes.py`

Add a pytest module that verifies the CNN-backed thesis model can execute a single forward and backward pass on a minimal synthetic batch. The test should exercise:

1. Encoder output shape.
2. Continuous prototype lookup.
3. Discrete prototype lookup.
4. Fusion branch outputs.
5. Reconstruction head output.
6. Classification head output.
7. Loss finiteness.
8. Backpropagation into encoder parameters.

#### `tests/test_cnn_encoder_config_loading.py`

Extend the config regression test to cover the thesis config path as well. The test should ensure that the new architecture fields are parsed into `MultitaskArchitectureConfig` and that invalid family values fail cleanly.

### Design pattern application

- Composition over inheritance: the thesis encoder adapter should own its local CNN or MLP implementation rather than inheriting a generic backbone class.
- Adapter pattern: the encoder normalizes the canonical batch tensor into the internal CNN layout and returns the standard hidden-state contract.
- Strategy pattern: the prototype and task-specific branches remain separate strategies that consume the same hidden-state interface.
- Registry or factory: the model config parser acts as the construction boundary where architecture options are resolved.

### Risk mitigation

- Prototype redundancy: keep continuous and discrete prototype branches unchanged in this phase and only vary the encoder family, so any performance shift can be attributed to the backbone change rather than a broader refactor.
- Fusion collapse: preserve the current fusion structure and add assertions on hidden-state rank and temporal length so that one branch does not silently dominate through malformed geometry.
- Adaptation contamination: not active yet, but the future online adaptation path should gate updates by batch quality and anomaly contamination; this phase should not pre-commit to an online update policy.
- Projector drift: avoid introducing new projection modules unless they are necessary for maintaining `hidden_dim`; if a projection is needed, keep it local and explicitly initialized.
- Evaluation metric inflation: preserve the same output keys and the same anomaly-score computation so cross-run comparisons remain meaningful.

### Acceptance criteria

This phase is complete only if all of the following are true:

1. `pytest -q tests/test_thesis_multitask_cnn_shapes.py` passes.
2. The thesis model still returns `hidden`, `pooled`, `recon`, `logits`, and the scoring fields expected by the existing contract.
3. The prototype branches execute without structural changes to their public interfaces.
4. The CNN encoder preserves `window_size` in the hidden tensor.
5. Invalid `encoder_family` values are rejected early and clearly.

---

## Phase 3: Regression Protection and Experiment Readiness

### Phase summary

This phase hardens the implementation with regression tests, checkpoint round-trip coverage, and experiment-facing clarity. The purpose is to ensure that the new CNN backbone is not only functional but also safe to rerun on the rented GPU server without hidden config drift or serialization failures.

### File-level edits

- Modify or extend `tests/test_checkpoint_roundtrip.py`
- Modify `tests/test_cnn_encoder_config_loading.py`
- Optionally update `documents/design/idea.md`
- Optionally add a short run note under `documents/logs/06-07-2026/detail/`

### Interface and contract definitions

Checkpointing must preserve the model object, its selected encoder family, and the exact shape contract across save and load. After reloading a checkpoint, the model should produce the same output keys and compatible tensor ranks on the same synthetic batch.

The config layer must remain explicit enough that a user can select `mlp` or `cnn_simple` without editing source code.

### Programming edits in detail

1. Add at least one checkpoint round-trip test for a CNN-backed model.
2. Ensure configuration loading remains backward compatible for legacy MLP experiments.
3. Add a brief design note only if the documented default encoder assumption is now stale.
4. Keep the change log in `documents/logs/06-07-2026/detail/` aligned with the final runtime behavior.

### Exact code changes

#### `tests/test_checkpoint_roundtrip.py`

Add a CNN-based save-load test if the current file does not already cover the new encoder family. The test should verify that a checkpoint written from the CNN model can be read back into the same model class and that a forward pass after reload still returns the expected keys and shapes.

#### `tests/test_cnn_encoder_config_loading.py`

Expand the config test to confirm that both legacy and new experiment files load correctly after the CNN change. The test should explicitly assert the parsed encoder family for each config path.

#### `documents/design/idea.md`

Update the design note only if the current documentation falsely implies that the active thesis encoder is still MLP-only. The update should be minimal and should preserve the thesis-level hidden-state contract statement.

### Design pattern application

- Composition over inheritance remains the guiding rule for the model files.
- Adapter pattern remains the mechanism by which the CNN stays hidden behind the canonical batch interface.
- Registry or factory behavior remains confined to the config/model construction boundary.
- Strategy pattern remains the mechanism for selecting between encoder families and task branches.

### Risk mitigation

- Prototype redundancy: regression tests should compare the existing MLP path and the new CNN path so the comparison remains fair.
- Fusion collapse: preserve the existing output keys and verify both branch outputs are non-empty and finite.
- Adaptation contamination: not implemented yet, so this phase must avoid introducing any online update side effects.
- Projector drift: checkpoint tests should verify that the CNN path does not require ad hoc parameter initialization to reload successfully.
- Evaluation metric inflation: keep evaluation logic unchanged and validate that the same output keys are produced before and after checkpoint reload.

### Acceptance criteria

This phase is complete only if all of the following are true:

1. At least one CNN-backed checkpoint round-trip test passes.
2. Both legacy MLP and new CNN configs load successfully.
3. The checkpointed model reloads with the same encoder family selection.
4. The design documentation, if touched, still states the encoder contract in the same `[B, L, H]` form.
5. The repository can be handed to the GPU server workflow without additional code edits for the intended rerun set.

---

## Cross-Phase Validation Matrix

The implementation should be validated in the following order:

1. Baseline CNN shape test.
2. Baseline config loading test.
3. Baseline forward and backward pass test.
4. Thesis CNN shape and backward test.
5. Thesis config loading test.
6. Checkpoint round-trip test.

The validation procedure is intentionally staged because the baseline vertical slice is the cheapest place to detect a geometry or config mistake. The thesis model should only be exercised after the baseline proves that the same tensor contract can be preserved through a convolutional encoder.

## Final Deliverable Definition

The plan is complete when the repository contains:

- a baseline model file that can switch between MLP and simple CNN encoders;
- a thesis model file that can switch between MLP and simple CNN encoders while preserving prototype and fusion behavior;
- config files that expose the encoder family explicitly;
- tests that cover config parsing, forward shape, backward propagation, and checkpoint round-trip behavior;
- documentation that remains consistent with the active encoder contract.

## Implementation Sequence Recommendation

The recommended implementation sequence is:

1. Modify the baseline model and its config.
2. Add and run the baseline tests.
3. Fix any gradient-profiling assumptions that remain tied to linear layers.
4. Modify the thesis model and its config.
5. Add and run the thesis tests.
6. Add checkpoint regression coverage.
7. Only then start the GPU rerun campaign.

This order is preferable because it protects the thesis prototype path from early instability and keeps the first validation loop small enough to debug quickly.

