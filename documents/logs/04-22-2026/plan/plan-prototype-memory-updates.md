# Prototype Memory Updates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-only continuous-memory writes and discrete EMA-only codebook writes to `thesis_multitask`, with a ten-epoch encoder bootstrap, one-time clean-dominant memory initialization, hard magnitude control, checkpoint persistence of memory state, and tests that lock the `train updates / val-test freeze` contract.

**Architecture:** Keep the repository’s `1 model - 1 file` rule by placing all prototype-memory logic inside `src/models/thesis_multitask.py`. The trainer remains small and only coordinates the bootstrap-to-memory transition by invoking one model-owned initialization hook after the bootstrap window. Memory content becomes model-owned state, not optimizer-owned parameters; the continuous gate remains learnable, while continuous bank writes and discrete EMA writes remain outside the gradient graph.

**Tech Stack:** Python, PyTorch, Pytest, YAML config loading, repository checkpoint manager, Weights & Biases-compatible experiment logging.

---

## Context

This plan is grounded in:

- `codebase_preferences.md`
- `prompts/2_plan_prompt.md`
- `docs/superpowers/specs/2026-04-22-prototype-memory-updates-design.md`
- current offline model code in `src/models/thesis_multitask.py`

The plan preserves the current simple objective surface and follows the repository’s readability-first and self-contained-model constraints.

## Current State

- `src/models/thesis_multitask.py` already contains the encoder, continuous branch read path, discrete branch read path, fusion logic, and stage-specific objective assembly.
- `src/engine/trainer.py` already owns epoch scheduling and can safely trigger a one-time memory initialization hook without learning model-specific mathematics.
- `src/engine/checkpoint.py` already saves model and optimizer state, but memory lifecycle state that lives outside trainable parameters must now be persisted explicitly.
- The current contract already distinguishes `training_step`, `validation_step`, `synthetic_validation_step`, and `test_step`, so the `train-only memory update` rule can be made explicit without adding a second engine path.

## Design Options Considered

- **Option A: Keep prototype banks as trainable parameters and add auxiliary memory writes.** Rejected because it mixes optimizer updates with memory writes and violates the user’s requirement that memories update only through explicit forward-pass mechanisms.
- **Option B: Make memory content pure model state and keep only controllers learnable.** Accepted because it preserves clean semantics: the continuous gate remains learnable, the continuous bank is updated outside the graph, and the discrete branch is EMA-only.
- **Option C: Split the memory logic into helper modules outside the model file.** Rejected because `codebase_preferences.md` explicitly requires all logic for one model to remain in one file.

## Risk and Mitigation

- **Risk: bootstrap and memory-backed phases drift into two unrelated codepaths.**
  Mitigation: keep one `_shared_step()` and gate only the memory path through explicit lifecycle helpers.
- **Risk: memory initialization accidentally writes anomaly patterns into memory.**
  Mitigation: collect initialization hidden states only from clean windows and from synthetic windows at timesteps where `synthetic_anomaly_mask == 0`.
- **Risk: norm drift destabilizes attention logits, EMA updates, or the continuous gate.**
  Mitigation: add explicit magnitude-control helpers and test norm invariants after initialization and after train-time writes.
- **Risk: checkpoint restore loses EMA history or lifecycle mode.**
  Mitigation: register tensor memory state in model buffers and persist non-tensor lifecycle fields through checkpoint `extra_state`.
- **Risk: validation or test silently mutates memory.**
  Mitigation: add dedicated tests that snapshot memory state before `validation_step` and `test_step`.

## File Structure

### Existing Files To Modify

- `src/models/thesis_multitask.py`
  - Add bootstrap lifecycle state.
  - Add memory buffers for the continuous bank, discrete codebook, and discrete EMA statistics.
  - Add one-time data-driven initialization from normal hidden states.
  - Add hard magnitude control helpers.
  - Add continuous write path and discrete EMA write path.
  - Preserve current read direction for query reconstruction.
- `src/engine/trainer.py`
  - Invoke the model’s one-time memory initialization hook after the bootstrap epoch window.
- `src/engine/checkpoint.py`
  - Persist and restore lifecycle state that is not naturally captured by the model `state_dict`.
- `src/core/config.py`
  - Validate new bootstrap and memory hyperparameters.
- `configs/model/thesis_multitask.yaml`
  - Add bootstrap, EMA, and magnitude-control fields while keeping the default loss simple.

### Test Files To Create

- `tests/test_multitask_memory_bootstrap.py`
  - Bootstrap bypass behavior and no-mutation guarantees.
- `tests/test_multitask_memory_initialization.py`
  - One-time initialization and anomaly-aware masking behavior.
- `tests/test_multitask_memory_updates.py`
  - Continuous train-time writes, discrete EMA writes, and val/test freeze behavior.

### Existing Test Files To Modify

- `tests/test_config_loading.py`
  - New config-field validation.
- `tests/test_checkpoint_roundtrip.py`
  - Memory-state roundtrip coverage.
- `tests/test_online_reference_checkpoint.py`
  - Confirm online loading still accepts enriched multitask checkpoints.

---

### Task 1: Extend Config and Checkpoint Contracts

**Files:**
- Modify: `configs/model/thesis_multitask.yaml`
- Modify: `src/core/config.py`
- Modify: `src/engine/checkpoint.py`
- Modify: `tests/test_config_loading.py`
- Modify: `tests/test_checkpoint_roundtrip.py`

- [ ] **Step 1: Write the failing tests for new config and checkpoint fields**

Add these tests.

```python
# tests/test_config_loading.py
def test_multitask_config_accepts_memory_bootstrap_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_name: memory-plan-smoke",
                "seed: 7",
                "device: cpu",
                "output_dir: outputs/test",
                "checkpoint_dir: outputs/test/checkpoints",
                "data:",
                "  dataset_name: smd",
                "  root_dir: data/ServerMachineDataset",
                "  window_size: 100",
                "  stride: 10",
                "  batch_size: 2",
                "  num_workers: 0",
                "  validation_split_ratio: 0.2",
                "model:",
                "  model_name: thesis_multitask",
                "  input_dim: 38",
                "  encoder_dim: 64",
                "  hidden_dim: 16",
                "  mlp_num_linear_layers: 3",
                "  num_classes: 2",
                "  dropout: 0.0",
                "  continuous_num_prototypes: 4",
                "  discrete_codebook_size: 8",
                "  bootstrap_encoder_epochs: 10",
                "  discrete_ema_decay: 0.99",
                "  memory_norm_epsilon: 1.0e-6",
                "task:",
                "  task_name: multitask_tsad",
                "  use_synthetic_augmentation: true",
                "  use_synthetic_validation: true",
                "  synthetic_validation_seed: 7",
                "  anomaly_probability: 0.5",
                "  min_segment_fraction: 0.1",
                "  max_segment_fraction: 0.2",
                "  spike_scale: 3.0",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 12",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_yaml_config(config_path)
    validate_experiment_config(loaded_config)

    assert loaded_config["model"]["bootstrap_encoder_epochs"] == 10
    assert loaded_config["model"]["discrete_ema_decay"] == 0.99


# tests/test_checkpoint_roundtrip.py
def test_checkpoint_roundtrip_restores_extra_memory_state(tmp_path: Path) -> None:
    model = ReconstructionMLPAutoencoder(
        input_dim=38, encoder_dim=64, hidden_dim=16, dropout=0.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_manager = CheckpointManager(tmp_path)

    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="memory_extra_state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={"experiment_name": "memory-extra-state"},
        epoch=1,
        metric_history=[],
        extra_state={
            "memory_training_enabled": True,
            "memory_initialized": True,
            "bootstrap_encoder_epochs": 10,
        },
    )

    loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer)

    assert loaded_checkpoint["extra_state"]["memory_training_enabled"] is True
    assert loaded_checkpoint["extra_state"]["memory_initialized"] is True
    assert loaded_checkpoint["extra_state"]["bootstrap_encoder_epochs"] == 10
```

- [ ] **Step 2: Run the targeted tests to confirm they fail**

Run:

```bash
pytest tests/test_config_loading.py -k memory_bootstrap_fields -v
pytest tests/test_checkpoint_roundtrip.py -k extra_memory_state -v
```

Expected:

- the config-loading test fails because the new model fields are not yet validated
- the checkpoint test fails if `extra_state` handling is missing or incomplete

- [ ] **Step 3: Implement the minimal config and checkpoint support**

Add these fields to the model YAML.

```yaml
# configs/model/thesis_multitask.yaml
bootstrap_encoder_epochs: 10
discrete_ema_decay: 0.99
memory_norm_epsilon: 1.0e-6
memory_initialization_batches: 16
memory_initialization_with_synthetic_windows: true
```

Validate the new fields explicitly.

```python
# src/core/config.py
if model_config.get("model_name") == "thesis_multitask":
    integer_fields["bootstrap_encoder_epochs"] = model_config.get(
        "bootstrap_encoder_epochs"
    )
    integer_fields["memory_initialization_batches"] = model_config.get(
        "memory_initialization_batches"
    )
    float_fields["discrete_ema_decay"] = model_config.get("discrete_ema_decay")
    float_fields["memory_norm_epsilon"] = model_config.get("memory_norm_epsilon")

    if int(model_config["bootstrap_encoder_epochs"]) < 0:
        raise ValueError("bootstrap_encoder_epochs must be non-negative")
    if not 0.0 < float(model_config["discrete_ema_decay"]) < 1.0:
        raise ValueError("discrete_ema_decay must be in (0, 1)")
    if float(model_config["memory_norm_epsilon"]) <= 0.0:
        raise ValueError("memory_norm_epsilon must be positive")
```

Keep checkpoint `extra_state` explicit.

```python
# src/engine/checkpoint.py
if extra_state is not None:
    checkpoint_payload["extra_state"] = extra_state

# load_checkpoint already returns the payload; preserve this behavior and
# document in comments that lifecycle state is restored by the caller.
```

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
pytest tests/test_config_loading.py -k memory_bootstrap_fields -v
pytest tests/test_checkpoint_roundtrip.py -k extra_memory_state -v
```

Expected:

- both tests pass

- [ ] **Step 5: Commit the contract extension**

Run:

```bash
git add configs/model/thesis_multitask.yaml src/core/config.py src/engine/checkpoint.py tests/test_config_loading.py tests/test_checkpoint_roundtrip.py
git commit -m "Add memory config and checkpoint contracts"
```

---

### Task 2: Add Bootstrap Lifecycle and Memory State Scaffolding

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Create: `tests/test_multitask_memory_bootstrap.py`

- [ ] **Step 1: Write the failing bootstrap tests**

Create the new test file with these tests.

```python
from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def test_bootstrap_epochs_bypass_memory_and_keep_state_unchanged() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=10,
    )
    model.set_epoch_context(epoch_index=0, total_epochs=12)
    batch = _build_batch()

    continuous_before = model.continuous_memory_bank.clone()
    discrete_before = model.discrete_codebook_memory.clone()

    step_output = model.training_step(batch)

    assert step_output["log"]["train_memory_mode"] == 0.0
    assert torch.allclose(model.continuous_memory_bank, continuous_before)
    assert torch.allclose(model.discrete_codebook_memory, discrete_before)


def test_memory_mode_turns_on_after_bootstrap_window() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=10,
    )

    model.set_epoch_context(epoch_index=10, total_epochs=12)

    assert model.memory_training_enabled is True
```

- [ ] **Step 2: Run the bootstrap tests to confirm they fail**

Run:

```bash
pytest tests/test_multitask_memory_bootstrap.py -v
```

Expected:

- failure because the model does not yet expose memory lifecycle state or bootstrap bypass behavior

- [ ] **Step 3: Implement bootstrap lifecycle state inside the model file**

Add explicit non-parameter memory state and lifecycle flags inside `ThesisMultitaskModel`.

```python
# src/models/thesis_multitask.py
self.bootstrap_encoder_epochs = bootstrap_encoder_epochs
self.memory_training_enabled = False
self.memories_initialized = False

self.register_buffer(
    "continuous_memory_bank",
    torch.zeros(self.continuous_num_prototypes, hidden_dim),
)
self.register_buffer(
    "discrete_codebook_memory",
    torch.zeros(self.discrete_codebook_size, hidden_dim),
)
self.register_buffer(
    "discrete_ema_counts",
    torch.zeros(self.discrete_codebook_size),
)
self.register_buffer(
    "discrete_ema_sums",
    torch.zeros(self.discrete_codebook_size, hidden_dim),
)

def _in_bootstrap_mode(self) -> bool:
    return self.current_epoch_index < self.bootstrap_encoder_epochs

def _should_update_memory(self, stage_name: str) -> bool:
    return (
        stage_name == "train"
        and self.memory_training_enabled
        and self.memories_initialized
    )

def _should_bypass_memory(self) -> bool:
    return self._in_bootstrap_mode() or not self.memories_initialized
```

Extend epoch context so bootstrap mode is explicit and loggable.

```python
def set_epoch_context(self, epoch_index: int, total_epochs: int) -> None:
    self.current_epoch_index = epoch_index
    self.current_total_epochs = total_epochs
    self.memory_training_enabled = epoch_index >= self.bootstrap_encoder_epochs
    self.schedule_state["memory_mode"] = float(self.memory_training_enabled)
```

- [ ] **Step 4: Re-run the bootstrap tests**

Run:

```bash
pytest tests/test_multitask_memory_bootstrap.py -v
```

Expected:

- both tests pass

- [ ] **Step 5: Commit the lifecycle scaffolding**

Run:

```bash
git add src/models/thesis_multitask.py tests/test_multitask_memory_bootstrap.py
git commit -m "Add bootstrap memory lifecycle scaffolding"
```

---

### Task 3: Implement One-Time Data-Driven Initialization with Anomaly-Aware Masking

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Modify: `src/engine/trainer.py`
- Create: `tests/test_multitask_memory_initialization.py`

- [ ] **Step 1: Write the failing initialization tests**

Create the new test file with these tests.

```python
from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def test_initialization_pool_excludes_anomalous_timesteps() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=8,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=0,
    )
    hidden = torch.arange(0, 2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    synthetic_mask = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=torch.long)

    normal_hidden = model._select_normal_hidden_states_for_memory_initialization(
        hidden=hidden,
        synthetic_anomaly_mask=synthetic_mask,
    )

    assert normal_hidden.shape[0] == 4


def test_memory_initialization_runs_only_once() -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=8,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=0,
    )
    batch = {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }

    model.initialize_memories_from_batches([batch], device="cpu")
    continuous_after_first_init = model.continuous_memory_bank.clone()
    model.initialize_memories_from_batches([batch], device="cpu")

    assert model.memories_initialized is True
    assert torch.allclose(model.continuous_memory_bank, continuous_after_first_init)
```

- [ ] **Step 2: Run the initialization tests to confirm they fail**

Run:

```bash
pytest tests/test_multitask_memory_initialization.py -v
```

Expected:

- failure because initialization helpers and one-time guards do not exist yet

- [ ] **Step 3: Implement one-time initialization helpers and trainer hook**

Add model-owned helpers.

```python
# src/models/thesis_multitask.py
def _select_normal_hidden_states_for_memory_initialization(
    self,
    *,
    hidden: torch.Tensor,
    synthetic_anomaly_mask: torch.Tensor | None,
) -> torch.Tensor:
    if synthetic_anomaly_mask is None:
        return hidden.reshape(-1, self.hidden_dim)
    normal_mask = (synthetic_anomaly_mask == 0).reshape(-1)
    flattened_hidden = hidden.reshape(-1, self.hidden_dim)
    return flattened_hidden[normal_mask]


def initialize_memories_from_batches(
    self,
    batches: list[dict[str, Any]],
    *,
    device: str,
) -> None:
    if self.memories_initialized:
        return

    collected_normal_hidden_states: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in batches:
            batch_on_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            clean_prepared_batch = self._prepare_clean_batch(batch_on_device, "train")
            clean_hidden = self.encoder(clean_prepared_batch)["hidden"]
            collected_normal_hidden_states.append(
                self._select_normal_hidden_states_for_memory_initialization(
                    hidden=clean_hidden,
                    synthetic_anomaly_mask=clean_prepared_batch.get(
                        "synthetic_anomaly_mask"
                    ),
                )
            )

            if self.memory_initialization_with_synthetic_windows:
                synthetic_prepared_batch = self.synthetic_anomaly_injector.augment_batch(
                    batch_on_device
                )
                synthetic_hidden = self.encoder(synthetic_prepared_batch)["hidden"]
                collected_normal_hidden_states.append(
                    self._select_normal_hidden_states_for_memory_initialization(
                        hidden=synthetic_hidden,
                        synthetic_anomaly_mask=synthetic_prepared_batch[
                            "synthetic_anomaly_mask"
                        ],
                    )
                )

    initialization_hidden = torch.cat(collected_normal_hidden_states, dim=0)
    self._initialize_continuous_memory_from_hidden_states(initialization_hidden)
    self._initialize_discrete_memory_from_hidden_states(initialization_hidden)
    self.memories_initialized = True
```

Trigger initialization from the trainer at the first post-bootstrap epoch.

```python
# src/engine/trainer.py
if (
    hasattr(self.model, "initialize_memories_from_batches")
    and hasattr(self.model, "memory_training_enabled")
    and self.model.memory_training_enabled
    and not self.model.memories_initialized
):
    initialization_batches = []
    for batch_index, train_batch in enumerate(train_loader):
        initialization_batches.append(train_batch)
        if batch_index + 1 >= int(self.model.memory_initialization_batches):
            break
    self.model.initialize_memories_from_batches(
        initialization_batches,
        device=self.device,
    )
```

- [ ] **Step 4: Re-run the initialization tests**

Run:

```bash
pytest tests/test_multitask_memory_initialization.py -v
```

Expected:

- both tests pass

- [ ] **Step 5: Commit initialization support**

Run:

```bash
git add src/models/thesis_multitask.py src/engine/trainer.py tests/test_multitask_memory_initialization.py
git commit -m "Add one-time memory initialization"
```

---

### Task 4: Implement Hard Magnitude Control and Continuous Memory Writes

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Create: `tests/test_multitask_memory_updates.py`

- [ ] **Step 1: Write the failing tests for continuous train-only writes and norm control**

Create the test file with these tests first.

```python
from __future__ import annotations

import torch

from src.models.thesis_multitask import ThesisMultitaskModel


def _build_batch() -> dict[str, object]:
    return {
        "x": torch.randn(2, 100, 38),
        "point_labels": torch.zeros(2, 100, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
    }


def _build_initialized_model() -> ThesisMultitaskModel:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=0,
    )
    model.initialize_memories_from_batches([_build_batch()], device="cpu")
    model.set_epoch_context(epoch_index=0, total_epochs=1)
    return model


def test_training_step_updates_continuous_memory_bank() -> None:
    model = _build_initialized_model()
    continuous_before = model.continuous_memory_bank.clone()

    model.training_step(_build_batch())

    assert not torch.allclose(model.continuous_memory_bank, continuous_before)


def test_validation_step_does_not_update_continuous_memory_bank() -> None:
    model = _build_initialized_model()
    continuous_before = model.continuous_memory_bank.clone()

    model.validation_step(_build_batch())

    assert torch.allclose(model.continuous_memory_bank, continuous_before)


def test_continuous_memory_bank_rows_keep_controlled_norm() -> None:
    model = _build_initialized_model()
    model.training_step(_build_batch())
    row_norms = model.continuous_memory_bank.norm(dim=-1)

    assert torch.all(row_norms > 0.0)
    assert torch.max(row_norms) - torch.min(row_norms) < 1.0e-3
```

- [ ] **Step 2: Run the continuous-memory tests to confirm they fail**

Run:

```bash
pytest tests/test_multitask_memory_updates.py -k continuous -v
```

Expected:

- failure because train-time writes and magnitude control are not implemented yet

- [ ] **Step 3: Implement hard scale control and continuous write path**

Add normalization helpers and a learnable gate network.

```python
# src/models/thesis_multitask.py
self.continuous_update_gate = nn.Sequential(
    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
    nn.ReLU(),
    nn.Linear(self.hidden_dim, self.hidden_dim),
    nn.Sigmoid(),
)

def _normalize_hidden_for_memory(self, hidden: torch.Tensor) -> torch.Tensor:
    return F.normalize(hidden, dim=-1, eps=self.memory_norm_epsilon)

def _normalize_memory_rows(self, memory_tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(memory_tensor, dim=-1, eps=self.memory_norm_epsilon)
```

Implement continuous write in H-PAD direction, with the actual write under `torch.no_grad()`.

```python
def _update_continuous_memory_bank(
    self,
    hidden: torch.Tensor,
) -> torch.Tensor:
    normalized_hidden = self._normalize_hidden_for_memory(hidden)
    normalized_memory = self._normalize_memory_rows(self.continuous_memory_bank)

    prototype_to_token_logits = torch.einsum(
        "kh,blh->kbl",
        normalized_memory,
        normalized_hidden,
    ) / math.sqrt(self.hidden_dim)
    prototype_to_token_weights = torch.softmax(
        prototype_to_token_logits.reshape(self.continuous_num_prototypes, -1),
        dim=-1,
    ).reshape_as(prototype_to_token_logits)
    weighted_hidden_summary = torch.einsum(
        "kbl,blh->kh",
        prototype_to_token_weights,
        normalized_hidden,
    )
    weighted_hidden_summary = self._normalize_memory_rows(weighted_hidden_summary)

    gate_input = torch.cat([normalized_memory, weighted_hidden_summary], dim=-1)
    update_gate = self.continuous_update_gate(gate_input)

    updated_memory = (
        (1.0 - update_gate) * normalized_memory
        + update_gate * weighted_hidden_summary
    )
    updated_memory = self._normalize_memory_rows(updated_memory)

    with torch.no_grad():
        self.continuous_memory_bank.copy_(updated_memory.detach())

    return updated_memory
```

Use updated memory for the continuous read path during training and stored memory during validation/test.

```python
if self._should_update_memory(stage_name):
    active_continuous_memory_bank = self._update_continuous_memory_bank(hidden)
else:
    active_continuous_memory_bank = self._normalize_memory_rows(
        self.continuous_memory_bank
    )
```

- [ ] **Step 4: Re-run the continuous-memory tests**

Run:

```bash
pytest tests/test_multitask_memory_updates.py -k continuous -v
```

Expected:

- all continuous-memory tests pass

- [ ] **Step 5: Commit the continuous write path**

Run:

```bash
git add src/models/thesis_multitask.py tests/test_multitask_memory_updates.py
git commit -m "Add continuous memory write path"
```

---

### Task 5: Implement Discrete EMA Writes and Persist Full Memory State

**Files:**
- Modify: `src/models/thesis_multitask.py`
- Modify: `src/engine/checkpoint.py`
- Modify: `tests/test_multitask_memory_updates.py`
- Modify: `tests/test_checkpoint_roundtrip.py`
- Modify: `tests/test_online_reference_checkpoint.py`

- [ ] **Step 1: Write the failing tests for discrete EMA writes and checkpoint restore**

Extend the memory-update tests.

```python
# tests/test_multitask_memory_updates.py
def test_training_step_updates_discrete_memory_and_ema_state() -> None:
    model = _build_initialized_model()
    codebook_before = model.discrete_codebook_memory.clone()
    counts_before = model.discrete_ema_counts.clone()
    sums_before = model.discrete_ema_sums.clone()

    model.training_step(_build_batch())

    assert not torch.allclose(model.discrete_codebook_memory, codebook_before)
    assert not torch.allclose(model.discrete_ema_counts, counts_before)
    assert not torch.allclose(model.discrete_ema_sums, sums_before)


def test_test_step_does_not_update_discrete_memory_or_ema_state() -> None:
    model = _build_initialized_model()
    codebook_before = model.discrete_codebook_memory.clone()
    counts_before = model.discrete_ema_counts.clone()
    sums_before = model.discrete_ema_sums.clone()

    model.test_step(_build_batch())

    assert torch.allclose(model.discrete_codebook_memory, codebook_before)
    assert torch.allclose(model.discrete_ema_counts, counts_before)
    assert torch.allclose(model.discrete_ema_sums, sums_before)
```

Extend checkpoint coverage for the multitask memory state.

```python
# tests/test_checkpoint_roundtrip.py
def test_multitask_checkpoint_roundtrip_restores_memory_buffers(tmp_path: Path) -> None:
    model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.initialize_memories_from_batches(
        [
            {
                "x": torch.randn(2, 100, 38),
                "point_labels": torch.zeros(2, 100, dtype=torch.long),
                "mask": None,
                "timestamps": None,
                "meta": [{"entity_id": "machine-a"}, {"entity_id": "machine-b"}],
            }
        ],
        device="cpu",
    )
    checkpoint_manager = CheckpointManager(tmp_path)

    checkpoint_path = checkpoint_manager.save_checkpoint(
        checkpoint_name="multitask_memory.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler_state={"feature_mean": torch.zeros(38), "feature_std": torch.ones(38)},
        config={"experiment_name": "multitask-memory"},
        epoch=1,
        metric_history=[],
        extra_state={
            "memory_training_enabled": model.memory_training_enabled,
            "memories_initialized": model.memories_initialized,
        },
    )

    reloaded_model = ThesisMultitaskModel(
        input_dim=38,
        encoder_dim=64,
        hidden_dim=16,
        use_synthetic_augmentation=False,
        bootstrap_encoder_epochs=0,
    )
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-3)
    loaded_checkpoint = checkpoint_manager.load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
    )

    assert torch.allclose(
        model.continuous_memory_bank, reloaded_model.continuous_memory_bank
    )
    assert torch.allclose(
        model.discrete_codebook_memory, reloaded_model.discrete_codebook_memory
    )
    assert torch.allclose(model.discrete_ema_counts, reloaded_model.discrete_ema_counts)
    assert loaded_checkpoint["extra_state"]["memories_initialized"] is True
```

- [ ] **Step 2: Run the discrete and checkpoint tests to confirm they fail**

Run:

```bash
pytest tests/test_multitask_memory_updates.py -k discrete -v
pytest tests/test_checkpoint_roundtrip.py -k multitask_memory -v
```

Expected:

- the discrete tests fail because EMA-only writes are missing
- the checkpoint test fails because the model does not yet expose or restore the required memory state

- [ ] **Step 3: Implement discrete EMA writes and explicit lifecycle persistence**

Add EMA-only discrete writes.

```python
# src/models/thesis_multitask.py
def _update_discrete_codebook_memory(
    self,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_hidden = self._normalize_hidden_for_memory(hidden)
    assignment_logits = self.discrete_assignment(normalized_hidden)
    assignment_probabilities = F.gumbel_softmax(
        assignment_logits,
        tau=self.gumbel_temperature,
        hard=False,
        dim=-1,
    )

    flattened_probabilities = assignment_probabilities.reshape(-1, self.discrete_codebook_size)
    flattened_hidden = normalized_hidden.reshape(-1, self.hidden_dim)
    batch_counts = flattened_probabilities.sum(dim=0)
    batch_sums = flattened_probabilities.T @ flattened_hidden

    with torch.no_grad():
        self.discrete_ema_counts.mul_(self.discrete_ema_decay).add_(
            (1.0 - self.discrete_ema_decay) * batch_counts.detach()
        )
        self.discrete_ema_sums.mul_(self.discrete_ema_decay).add_(
            (1.0 - self.discrete_ema_decay) * batch_sums.detach()
        )
        normalized_codebook = self.discrete_ema_sums / self.discrete_ema_counts.clamp_min(
            self.memory_norm_epsilon
        ).unsqueeze(-1)
        normalized_codebook = self._normalize_memory_rows(normalized_codebook)
        self.discrete_codebook_memory.copy_(normalized_codebook)

    return assignment_logits, assignment_probabilities
```

Expose lifecycle state through checkpoint `extra_state`.

```python
# src/engine/checkpoint.py
checkpoint_path = self.checkpoint_manager.save_checkpoint(
    checkpoint_name="best.pt",
    model=self.model,
    optimizer=self.optimizer,
    scheduler=self.scheduler,
    scaler_state=scaler_state,
    config=config,
    epoch=epoch_index + 1,
    metric_history=self.metric_history,
    extra_state={
        "memory_training_enabled": getattr(self.model, "memory_training_enabled", False),
        "memories_initialized": getattr(self.model, "memories_initialized", False),
        "bootstrap_encoder_epochs": getattr(self.model, "bootstrap_encoder_epochs", 0),
    },
)
```

Restore lifecycle state after loading the checkpoint in tests and in any runtime that needs it.

```python
# caller-side restore pattern
loaded_checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path, model, optimizer)
extra_state = loaded_checkpoint.get("extra_state", {})
if hasattr(model, "memory_training_enabled"):
    model.memory_training_enabled = bool(extra_state.get("memory_training_enabled", False))
if hasattr(model, "memories_initialized"):
    model.memories_initialized = bool(extra_state.get("memories_initialized", False))
```

- [ ] **Step 4: Re-run the discrete and checkpoint tests**

Run:

```bash
pytest tests/test_multitask_memory_updates.py -k discrete -v
pytest tests/test_checkpoint_roundtrip.py -k multitask_memory -v
pytest tests/test_online_reference_checkpoint.py -v
```

Expected:

- all discrete-memory and checkpoint tests pass
- online reference checkpoint loading still passes because enriched multitask checkpoints remain loadable

- [ ] **Step 5: Commit the discrete EMA and checkpoint persistence work**

Run:

```bash
git add src/models/thesis_multitask.py src/engine/checkpoint.py tests/test_multitask_memory_updates.py tests/test_checkpoint_roundtrip.py tests/test_online_reference_checkpoint.py
git commit -m "Add discrete EMA memory persistence"
```

---

### Task 6: Run the Full Targeted Regression Slice and Update Plan-Adjacent Documentation

**Files:**
- Modify: `documents/logs/04-22-2026/plan/plan-prototype-memory-updates.md`
- Modify: `documents/logs/04-22-2026/research/research-current-codebase-state-for-train-time-prototype-updates.md`

- [ ] **Step 1: Add final regression commands to the plan and research note**

Append this exact section to the end of the plan or a closing checklist section.

````markdown
## Validation Commands

```bash
pytest tests/test_config_loading.py -k memory_bootstrap_fields -v
pytest tests/test_multitask_memory_bootstrap.py -v
pytest tests/test_multitask_memory_initialization.py -v
pytest tests/test_multitask_memory_updates.py -v
pytest tests/test_checkpoint_roundtrip.py -k "memory or multitask_memory" -v
pytest tests/test_online_reference_checkpoint.py -v
pytest tests/test_one_multitask_train_step.py -v
pytest tests/test_multitask_shapes.py -v
```
````

- [ ] **Step 2: Run the full targeted regression slice after implementation**

Run:

```bash
pytest tests/test_config_loading.py -k memory_bootstrap_fields -v
pytest tests/test_multitask_memory_bootstrap.py -v
pytest tests/test_multitask_memory_initialization.py -v
pytest tests/test_multitask_memory_updates.py -v
pytest tests/test_checkpoint_roundtrip.py -k "memory or multitask_memory" -v
pytest tests/test_online_reference_checkpoint.py -v
pytest tests/test_one_multitask_train_step.py -v
pytest tests/test_multitask_shapes.py -v
```

Expected:

- all targeted tests pass

- [ ] **Step 3: Record the implementation outcome in the same-day detail log if code is implemented**

When implementation is complete, create a same-day detail note that records:

```markdown
- the final bootstrap-to-memory transition design
- the chosen magnitude-control strategy
- the final checkpoint persistence mechanism
- the exact targeted test commands that passed
```

- [ ] **Step 4: Re-read the plan against the approved spec**

Check manually that the following spec requirements each map to an implemented task:

```text
- ten-epoch bootstrap
- one-time clean-dominant initialization
- continuous learned gate with non-differentiable memory writes
- discrete EMA-only writes
- hard magnitude control
- checkpoint persistence
- train-update / val-test freeze tests
```

Expected:

- no uncovered spec requirement remains

- [ ] **Step 5: Commit the final implementation-instructions update**

Run:

```bash
git add documents/logs/04-22-2026/plan/plan-prototype-memory-updates.md documents/logs/04-22-2026/research/research-current-codebase-state-for-train-time-prototype-updates.md
git commit -m "Finalize prototype memory implementation plan"
```

---

## Self-Review

### Spec Coverage

This plan covers:

- bootstrap lifecycle
- one-time initialization
- clean-dominant anomaly-aware masking
- continuous H-PAD-style writes
- discrete EMA-only writes
- hard magnitude control
- checkpoint persistence
- train-only update contract
- regression tests

No approved spec requirement is left without a task.

### Placeholder Scan

No `TODO`, `TBD`, or deferred-implementation placeholders remain. Each task contains concrete file paths, test names, commands, and code snippets.

### Type and Naming Consistency

The plan uses these names consistently:

- `continuous_memory_bank`
- `discrete_codebook_memory`
- `discrete_ema_counts`
- `discrete_ema_sums`
- `memories_initialized`
- `memory_training_enabled`
- `initialize_memories_from_batches`
- `_select_normal_hidden_states_for_memory_initialization`
- `_update_continuous_memory_bank`
- `_update_discrete_codebook_memory`
