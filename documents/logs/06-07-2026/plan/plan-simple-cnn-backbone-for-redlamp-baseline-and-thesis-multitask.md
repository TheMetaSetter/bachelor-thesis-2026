---
date: 2026-06-07 15:10:00 +0700
researcher: Codex
git_commit: 32417993875f677a86743ab3a770d0ccc67b32fe
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation plan to replace active MLP encoders with a simple RedLamp-style CNN backbone in redlamp_mlp_baseline.py and thesis_multitask.py"
tags: [plan, cnn, redlamp, baseline, thesis-model, anomaly-detection]
status: complete
last_updated: 2026-06-07
last_updated_by: Codex
---

# Simple CNN Backbone For RedLamp Baseline And Thesis Multitask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active timestep-MLP encoder with a simple RedLamp-style convolutional backbone in the baseline model and the thesis multitask model while preserving the repository batch contract and model output contract.

**Architecture:** The implementation should introduce a simple one-dimensional convolutional encoder that internally converts `[B, L, D]` to `[B, D, L]`, performs convolutional feature extraction in channel-first format, and returns hidden states in the established `[B, L, H]` contract. The baseline model should be migrated first as the minimal vertical slice. The thesis multitask model should then reuse the same encoder-family idea through a local adapter inside the same model file, without fragmenting the model across many files.

**Tech Stack:** Python, PyTorch, YAML configuration loading, `pytest`, repository-local contracts in `src/core/contracts.py`.

---

## Current State

The current repository state is favorable for a backbone swap because the data loader and trainer contracts are already decoupled from the exact encoder family. The standardized offline batch contract remains `batch["x"]: Tensor[B, L, D]` under `src/core/contracts.py`, and the evaluator validates output ranks and required keys rather than checking whether the encoder is an MLP or a CNN.

The active RedLamp baseline in `src/models/redlamp_mlp_baseline.py` is still an MLP timestep encoder. The active thesis model in `src/models/thesis_multitask.py` is also MLP-based through `MultitaskWindowEncoder`. Both files already preserve the important downstream contract `hidden: Tensor[B, L, H]`, which means the training engine, checkpointing path, anomaly scoring path, and evaluator can remain stable if the new convolutional backbone returns the same hidden-state geometry.

The key hidden implementation risk is not in the trainer but inside the owning model files. In particular, the baseline gradient-conflict profiling helper currently assumes the encoder is an ordered stack of `nn.Linear` layers. That assumption must be generalized once the encoder family becomes convolutional.

## Design Options

### Option A: Hard replacement with a simple CNN backbone inside each owning model file

This option replaces the current MLP encoder construction in each model file with a small CNN encoder implementation written directly inside that same file. This option best respects the repository preference `1 model - 1 file`, minimizes abstraction overhead, and is the most direct path for the first experimental reruns.

This option is the recommended starting point.

### Option B: Local encoder-family adapter inside each owning model file

This option keeps both encoder families available within each model file through a small local selector such as `encoder_family in {"mlp", "cnn_simple"}`. This is slightly more code than Option A, but it is much better for ablations, reproducibility, and fair baseline comparison because the same config surface can switch between the old MLP and the new CNN.

This option is the recommended implementation shape if the user wants to preserve easy ablation.

### Option C: Shared CNN module under a new file and imported by both models

This option creates a shared reusable convolutional encoder file and imports it into both models. Although this reduces code duplication, it conflicts with the repository preference that model-related logic remain easy to read inside one self-contained model file. It also creates an unnecessary abstraction layer at this stage.

This option should be avoided for the first implementation unless code duplication becomes a real maintenance problem after both models are stable.

## Recommended Direction

The recommended direction is a hybrid of Option A and Option B. The backbone should be implemented locally inside each owning model file, but exposed through a small explicit encoder-family switch so that the previous MLP path remains available for ablation and reruns. This keeps the code readable, keeps file ownership clear, and preserves experimental comparability.

## Stable Contracts To Preserve

The implementation must preserve three contracts.

First, the batch contract remains unchanged.

```python
batch["x"].shape == [batch_size, window_size, input_dim]
```

Second, the encoder contract remains unchanged at the model boundary.

```python
encoder_outputs["hidden"].shape == [batch_size, window_size, hidden_dim]
```

Third, the model output contract remains unchanged.

```python
outputs["recon"].shape == [batch_size, window_size, input_dim]
outputs["point_scores"].shape == [batch_size, window_size]
outputs["window_scores"].shape == [batch_size]
```

The main design principle is therefore simple: the convolutional backbone may change the internal tensor layout, but it must not change the external repository contract.

## File Structure And Responsibilities

### Files to modify

- `src/models/redlamp_mlp_baseline.py`
  - Add a local simple CNN encoder implementation.
  - Extend the model configuration surface to allow explicit encoder-family selection.
  - Generalize encoder gradient profiling so that it works for both linear and convolutional backbones.

- `src/models/thesis_multitask.py`
  - Add a local simple CNN encoder implementation or a CNN-aware alternative to `MultitaskWindowEncoder`.
  - Extend architecture config dataclasses and config parsing to carry CNN hyperparameters.
  - Keep prototype branches, fusion heads, reconstruction head, and classification head unchanged at their public interfaces.

- `configs/model/redlamp_mlp_baseline.yaml`
  - Add explicit encoder-family and CNN hyperparameters for baseline experiments.

- `configs/model/thesis_multitask_redlamp_multiclass.yaml`
  - Add explicit encoder-family and CNN hyperparameters for thesis-model experiments.

### Files to add

- `tests/test_redlamp_cnn_baseline_shapes.py`
  - Verify forward pass, backward pass, and output shapes for the baseline with the CNN encoder.

- `tests/test_thesis_multitask_cnn_shapes.py`
  - Verify forward pass, backward pass, and output shapes for the thesis model with the CNN encoder.

- `tests/test_cnn_encoder_config_loading.py`
  - Verify that config parsing accepts the new encoder-family and CNN-specific keys.

### Optional documentation updates

- `documents/design/idea.md`
  - Add one short note that the active encoder family can now be selected while preserving the stable hidden-state contract.

- `documents/design/design_starter.md`
  - Update only if the documented directory tree or stated default backbone assumptions become inaccurate.

## Programming Plan

### Task 1: Introduce an explicit encoder-family surface in the baseline model

**Files:**
- Modify: `src/models/redlamp_mlp_baseline.py`
- Modify: `configs/model/redlamp_mlp_baseline.yaml`
- Test: `tests/test_cnn_encoder_config_loading.py`

The baseline file should gain an explicit field such as `encoder_family: str = "mlp"` and a minimal CNN hyperparameter block such as `cnn_kernel_size`, `cnn_hidden_channels`, `cnn_num_layers`, and `cnn_dropout`. The constructor should validate that only supported encoder families are accepted.

The implementation should not introduce a separate global module registry for encoder families. A local helper inside the file is sufficient and easier to read.

Suggested constructor shape:

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

The YAML config should spell these fields out explicitly, even when unused by the MLP path, because explicit config is better for ablation and reproducibility.

### Task 2: Implement the local simple CNN encoder in the baseline model

**Files:**
- Modify: `src/models/redlamp_mlp_baseline.py`
- Test: `tests/test_redlamp_cnn_baseline_shapes.py`

A small encoder class should be added inside the baseline file. It should remain simple and close to the RedLamp reference in spirit, but adapted to the thesis repository contract.

Recommended behavior:

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

The convolution blocks should preserve temporal length through padding, because keeping `L` unchanged is the cleanest path for compatibility with the flatten-classifier and reconstruction path.

The first implementation should not attempt a full convolutional autoencoder. Keep the decoder unchanged as long as `hidden` remains `[B, L, latent_dim]`.

### Task 3: Generalize baseline gradient profiling for CNN compatibility

**Files:**
- Modify: `src/models/redlamp_mlp_baseline.py`
- Test: `tests/test_redlamp_cnn_baseline_shapes.py`

This task is required. Without it, the encoder-family migration remains incomplete.

The current profiling helpers should be rewritten so they no longer assume that all profile-worthy encoder parameters belong to `nn.Linear`. A simple and readable rule is better than a highly dynamic one.

Recommended rule:
- profile parameters from the last trainable affine-like layer in the encoder,
- where affine-like means either `nn.Linear` or `nn.Conv1d`.

Suggested helper logic:

```python
def _iter_encoder_affine_layers(self) -> list[nn.Module]:
    return [
        module
        for module in self.encoder.modules()
        if isinstance(module, (nn.Linear, nn.Conv1d))
    ]
```

The focus-layer naming should become family-agnostic. For example, `encoder_last_affine` is semantically cleaner than `encoder_last_linear`.

### Task 4: Verify the baseline vertical slice end-to-end

**Files:**
- Test: `tests/test_redlamp_cnn_baseline_shapes.py`
- Test: existing baseline-related tests if they exist

This is the minimal vertical slice checkpoint. The baseline CNN path is considered ready only if all of the following are true:

- config loads successfully,
- one forward pass runs,
- one backward pass runs,
- reconstruction shape matches input shape,
- classifier path still accepts flattened hidden states,
- anomaly scores remain finite,
- gradient profiling does not crash under the CNN encoder family.

A minimal synthetic test batch is sufficient for the first verification. There is no need to wait for full GPU experiments before validating the backbone swap.

### Task 5: Extend the thesis architecture config surface for encoder-family selection

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Modify: `configs/model/thesis_multitask_redlamp_multiclass.yaml`
- Test: `tests/test_cnn_encoder_config_loading.py`

`MultitaskArchitectureConfig` should be extended with explicit CNN-related fields. The parser `ThesisMultitaskModelConfig.from_flat_kwargs(...)` must then whitelist these keys.

Suggested fields:

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

The implementation should preserve backward compatibility for old configs by giving all new fields safe defaults.

### Task 6: Add a CNN-aware encoder path in the thesis model

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Test: `tests/test_thesis_multitask_cnn_shapes.py`

The cleanest implementation is to keep `MultitaskWindowEncoder` as the owning encoder adapter and let it internally branch on `encoder_family`.

Recommended shape:

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
        pooled = hidden.mean(dim=1)
        return {"hidden": hidden, "pooled": pooled}
```

This approach is preferable to changing prototype branches, because the prototype path should remain blind to whether hidden states came from an MLP or a CNN.

### Task 7: Keep thesis prototype and fusion paths unchanged, but add shape-protection assertions

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Test: `tests/test_thesis_multitask_cnn_shapes.py`

The prototype, memory-bank, and fusion branches should not be structurally rewritten in this phase. However, shape-protection assertions should be added near the encoder boundary so that any future CNN changes fail loudly if they break the hidden-state contract.

Suggested assertion style:

```python
if hidden.ndim != 3:
    raise ValueError("Encoder must return hidden states with shape [B, L, H].")
if hidden.shape[1] != self.architecture.window_size:
    raise ValueError("Encoder changed temporal length, which is not supported by the current prototype path.")
```

These checks are cheap and defend the rest of the model from silent geometry drift.

### Task 8: Verify the thesis model vertical slice with the CNN encoder

**Files:**
- Test: `tests/test_thesis_multitask_cnn_shapes.py`

The thesis CNN path is acceptable only if the following remain true under one full train-step-style test:

- encoder returns `[B, L, hidden_dim]`,
- continuous prototype lookup runs,
- discrete prototype lookup runs,
- fusion heads produce task-specific hidden states,
- reconstruction and classification heads both run,
- the total loss remains finite,
- backward propagation reaches the encoder parameters.

This test should use the smallest viable configuration that still exercises both task branches.

### Task 9: Add checkpoint and config regression protection

**Files:**
- Modify or add: `tests/test_checkpoint_roundtrip.py` if needed
- Add: `tests/test_cnn_encoder_config_loading.py`

The codebase preferences explicitly prioritize checkpoint save-load tests. Therefore, at least one model using the CNN encoder family must be serialized and reloaded successfully.

The config regression test should verify both of the following:
- old MLP configs still load,
- new CNN configs load and instantiate the expected encoder family.

### Task 10: Document experiment-facing usage for baseline-first reruns

**Files:**
- Modify: `documents/design/idea.md` only if the documented active encoder assumption becomes stale
- Add optional short log note under `documents/logs/06-07-2026/detail/` after implementation

This task should be lightweight. The most important thing is to document the exact config keys and the exact command paths needed for GPU reruns. Since the user plans to rent a GPU server, the model-selection knobs must be obvious and fail-fast.

## Test Plan

The test plan should remain minimal but adversarial.

### Unit-level checks

- Verify that the baseline CNN encoder accepts `[B, L, D]` and returns `[B, L, latent_dim]`.
- Verify that the thesis CNN encoder accepts `[B, L, D]` and returns `[B, L, hidden_dim]`.
- Verify that invalid `encoder_family` values raise a clear `ValueError`.
- Verify that CNN-specific config keys are accepted by config loaders.

### Integration-level checks

- One forward and backward pass for the baseline under `encoder_family: cnn_simple`.
- One forward and backward pass for the thesis model under `encoder_family: cnn_simple`.
- One checkpoint round-trip for at least one CNN-backed model.

### Suggested commands

```bash
pytest -q tests/test_cnn_encoder_config_loading.py tests/test_redlamp_cnn_baseline_shapes.py tests/test_thesis_multitask_cnn_shapes.py
```

```bash
pytest -q tests/test_checkpoint_roundtrip.py
```

## Validation Procedures

The implementation should be validated in three layers.

First, validate contract preservation locally through tests.

Second, validate experiment usability by instantiating both models from YAML configs rather than from hand-written Python constructors. This matters because the real failure surface in this repository often sits in config-to-model wiring.

Third, validate comparability by keeping the old MLP path runnable. The user needs fair reruns for the baseline and the main method, so the migration should not destroy the previous reference path.

## Risk And Mitigation

### Risk: temporal length changes inside the CNN break the flatten-classifier and prototype path

**Mitigation:** use same-length convolutions through explicit padding and add shape assertions at the encoder boundary.

### Risk: the baseline gradient-conflict profiling becomes meaningless or crashes under CNN layers

**Mitigation:** define a family-agnostic profiling rule based on the last affine-like encoder layer and cover it with tests.

### Risk: config sprawl creates hidden codepaths

**Mitigation:** keep only one explicit switch `encoder_family` and a minimal CNN hyperparameter set. Do not add multiple partially overlapping backbone flags.

### Risk: thesis prototype branches receive hidden states with the right rank but poor scale or unstable statistics

**Mitigation:** preserve dropout behavior, keep hidden width explicit, and verify one backward pass with both prototype branches active before any experiment rerun.

### Risk: the first implementation overfits to the baseline and then requires a second incompatible CNN in the thesis model

**Mitigation:** keep the CNN contract identical across both files from the beginning: input `[B, L, D]`, output `[B, L, H]`, no temporal downsampling.

### Risk: evaluation metric inflation due to accidental changes in reconstruction or classifier geometry

**Mitigation:** preserve decoder and classifier heads for the first vertical slice and change only the encoder family.

## Open Questions

The first open question is whether the user wants only an encoder swap or a fuller RedLamp-style autoencoder migration in the baseline. The recommended answer for now is encoder-only, because it isolates the effect of the backbone and preserves fairness.

The second open question is whether `cnn_hidden_channels` should equal `latent_dim` or whether the CNN should use an intermediate channel width and a final projection to `latent_dim`. The recommended answer is the second design because it gives a cleaner separation between internal channel width and public hidden-state width.

The third open question is whether the thesis model should expose `encoder_family` in all experiment configs immediately or only in the configs used for the rerun campaign. The recommended answer is to add the field with defaults everywhere new parsing is needed, but only activate the CNN path in the targeted rerun configs.

## Minimal Vertical Slice Sequence

The minimal and safest sequence is:

1. implement baseline encoder-family switch,
2. implement baseline simple CNN encoder,
3. fix baseline gradient profiling,
4. add baseline tests and checkpoint regression,
5. extend thesis config surface,
6. add thesis CNN encoder path,
7. add thesis shape and backward tests,
8. only then prepare GPU experiment configs.

This sequence keeps the first integration target small and gives quick evidence that the core tensor contract is stable before touching prototype logic.

## Execution Recommendation

The first implementation pass should stop after the baseline vertical slice and its tests. That checkpoint is where architectural mistakes are cheapest to catch. After that, the thesis-model CNN path can be implemented with much lower risk because the tensor-layout and config decisions have already been validated.

Plan complete and saved to `documents/logs/06-07-2026/plan/plan-simple-cnn-backbone-for-redlamp-baseline-and-thesis-multitask.md`.

Two execution options:

1. Subagent-Driven (recommended): implement task-by-task with review between tasks.
2. Inline Execution: implement directly in this session with checkpoints.
