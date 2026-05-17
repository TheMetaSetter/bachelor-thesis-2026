---
date: 2026-05-17
research_source: documents/logs/05-17-2026/research/research-training-optimization-runtime-support.md
topic: "Preliminary implementation plan for AdamW, CANDI-style cosine scheduling, configurable gradient clipping, and optimizer experiment configs"
status: draft
---

# Preliminary Implementation Plan: Training Optimization Runtime Support

## Goal

Extend the active offline training path so that thesis experiments can explicitly configure `AdamW`, use a CANDI-style cosine learning-rate policy stepped on every training iteration, apply gradient clipping from YAML, and generate two reproducible RedLamp baseline experiment configurations per entity, beginning with SMD machine `2-1`.

## Current State

- The active offline entrypoint in `scripts/train.py` hard-codes `torch.optim.Adam` and does not read an optimizer family from YAML.
- The active scheduler surface in `src/core/config.py` and `scripts/train.py` accepts only `reduce_on_plateau`.
- The active trainer in `src/engine/trainer.py` only supports metric-driven scheduler stepping at the end of each epoch through `scheduler.step(monitor_value)`.
- No offline training path currently reads or applies a gradient clipping norm from configuration.
- Checkpoint persistence in `src/engine/checkpoint.py` already saves arbitrary scheduler state dictionaries when a scheduler object is present.
- The directly referenced experiment file `configs/experiment/smd_redlamp_mlp_baseline_window20.yaml` already uses a run name that implies `adamw_cosine`, but the executable config still describes `Adam + ReduceLROnPlateau`.
- The batch contract, model-output contract, and one-model-per-file structure are already established and do not need to change for this work.

## Design Decision

The implementation should follow a **hybrid reference policy**:

- optimizer choice follows the RedLamp direction: `AdamW`;
- cosine scheduling mechanics follow CANDI: learning rate is recomputed from fractional training progress `epoch_index + iteration_index / num_iterations_in_epoch`;
- the cosine path should include the CANDI-style warmup controls because they are part of the reference scheduler mechanism;
- the implementation should remain an explicit offline-training extension rather than a broad optimizer subsystem refactor.

This choice best matches the intended experiment family while preserving the repository preference for explicit, readable code paths.

## Design Options

### Option A: Minimal epoch-level cosine scheduler

- Add `scheduler_name: cosine`.
- Construct `torch.optim.lr_scheduler.CosineAnnealingLR`.
- Call `scheduler.step()` once per epoch.

**Advantages:** smallest code change and least invasive trainer edit.  
**Disadvantages:** does not match the CANDI reference behavior, loses fractional-progress updates, and provides a weaker experimental match to the selected baseline.

### Option B: Explicit CANDI-style cosine policy in the offline trainer

- Add a small helper that computes cosine learning rate from fractional epoch progress.
- Update optimizer parameter-group learning rates on each batch.
- Keep the existing `ReduceLROnPlateau` path intact for metric-driven experiments.

**Advantages:** closest match to CANDI, clear semantics, preserves the existing plateau scheduler path, and avoids forcing two scheduler families into one incompatible `step()` contract.  
**Disadvantages:** requires more trainer work and additional tests around iteration-level behavior.

### Option C: Generic scheduler strategy abstraction for all optimizer and scheduler families

- Introduce a fuller runtime abstraction layer for optimizer builders and scheduler strategies.
- Use polymorphic scheduler objects to encapsulate per-iteration and per-epoch behavior.

**Advantages:** strongest long-term extensibility.  
**Disadvantages:** too broad for the immediate thesis experiment need, increases code paths, and conflicts with the repository preference for the least amount of code paths.

### Recommendation

Choose **Option B**. It is the smallest design that still reproduces the selected CANDI-style behavior accurately. Option A is too weak as a baseline match, and Option C is premature for the present scope.

## Planned Architecture

### 1. Configuration layer

Modify `src/core/config.py` so the offline experiment schema can express:

```yaml
optimizer:
  optimizer_name: adamw
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  scheduler:
    scheduler_name: cosine
    warmup_epochs: 5
    warmup_start_lr: 0.0001
    cosine_end_lr: 0.0
    cosine_after_warmup: true
```

The validator should:

- accept `optimizer_name` values `adam` and `adamw`;
- validate `gradient_clip_norm` as a positive numeric value when present;
- continue supporting the existing `reduce_on_plateau` schema unchanged;
- add a separate `cosine` scheduler schema with warmup and cosine-specific fields;
- reject plateau-only fields for cosine and cosine-only fields for plateau when practical, so configuration mistakes fail early and visibly.

### 2. Optimizer construction

Modify `scripts/train.py` so optimizer construction is selected from the validated `optimizer_name`:

- `adam` -> `torch.optim.Adam`
- `adamw` -> `torch.optim.AdamW`

The console logging should emit the configured optimizer type rather than a hard-coded `"Adam"` label.

### 3. Scheduler construction and runtime control

Preserve `build_scheduler_from_experiment_config()` for the existing plateau path, but avoid pretending cosine has the same runtime contract.

Recommended decomposition:

- keep `ReduceLROnPlateau` as a true scheduler object returned by the current builder;
- add explicit cosine configuration extraction and learning-rate computation helpers in `scripts/train.py` or a small focused helper module only if the function becomes too large;
- pass the resolved cosine configuration into `Trainer` so the trainer can update LR on each iteration.

The cosine policy should mirror CANDI:

```python
current_progress = epoch_index + float(batch_index) / num_training_batches
```

Then:

- use linear warmup while `current_progress < warmup_epochs`;
- use cosine decay from `learning_rate` to `cosine_end_lr`;
- if `cosine_after_warmup` is true, start the cosine curve after warmup rather than across the full epoch horizon.

### 4. Trainer responsibilities

Modify `src/engine/trainer.py` with two clearly separated update paths:

- **per-iteration path** for cosine:
  - compute current LR before or after each training batch using fractional progress;
  - set the LR on all optimizer parameter groups;
  - optionally log the batch LR to console if current logging style permits without flooding logs;
- **per-epoch path** for `ReduceLROnPlateau`:
  - preserve current monitor-metric behavior and current metrics.

Add gradient clipping in the training loop between `loss.backward()` and `optimizer.step()`:

```python
gradient_norm = torch.nn.utils.clip_grad_norm_(
    self.model.parameters(),
    max_norm=self.gradient_clip_norm,
)
```

When clipping is disabled by config, the loop should preserve current behavior.

### 5. Checkpoint and best-checkpoint semantics

- Keep generic scheduler-state checkpointing unchanged for plateau schedulers.
- For cosine, if the policy is implemented as deterministic arithmetic from epoch and batch progress rather than as a stateful PyTorch scheduler object, the existing checkpoint payload does not need a new scheduler state object.
- Best-checkpoint selection should be decoupled from cosine scheduling. Preliminary recommendation:
  - retain current `val_loss` fallback when no metric-driven scheduler monitor exists;
  - keep current monitor-driven checkpoint behavior for `ReduceLROnPlateau`;
  - do not overload cosine with a fake monitor metric.

### 6. Experiment configuration outputs

Create two explicit experiment configs for SMD machine `2-1`:

- `configs/experiment/smd_redlamp_mlp_baseline_machine_2_1_window20_adamw_cosine_lr1e-3.yaml`
- `configs/experiment/smd_redlamp_mlp_baseline_machine_2_1_window20_adamw_cosine_lr1e-4.yaml`

Both should share:

- `optimizer_name: adamw`
- `scheduler_name: cosine`
- `gradient_clip_norm: 1.0`
- `epochs: 300`
- the same data/model/task references

They should differ only in:

- `experiment_name`
- `output_dir`
- `checkpoint_dir`
- `learning_rate`
- `wandb_run_name`

The existing `configs/experiment/smd_redlamp_mlp_baseline_window20.yaml` should either be renamed, superseded, or corrected so its name no longer claims `adamw_cosine` while encoding plateau scheduling.

## File Plan

| File | Planned responsibility |
|---|---|
| `src/core/config.py` | Extend optimizer and scheduler schema validation; add gradient clipping validation. |
| `scripts/train.py` | Select optimizer implementation from config; parse/build scheduler controls; pass cosine controls into the trainer. |
| `src/engine/trainer.py` | Apply gradient clipping; add per-iteration cosine LR updates; preserve plateau path. |
| `tests/test_config_loading.py` | Add accepted/rejected config tests for `adamw`, cosine fields, and gradient clipping. |
| `tests/test_learning_rate_scheduler.py` | Add cosine LR trajectory tests and trainer tests for iteration-level stepping while retaining plateau tests. |
| `tests/test_checkpoint_roundtrip.py` | Confirm existing scheduler checkpoint behavior remains intact; add cosine-related checkpoint expectations only if runtime state is persisted. |
| `configs/experiment/*.yaml` | Add two explicit SMD `2-1` AdamW-cosine configs and remove misleading naming from the older baseline config if it remains. |

## Interface Preservation

This work does not modify:

- the batch contract `batch["x"] -> Tensor[B, L, D]`;
- the encoder contract `outputs["hidden"] -> Tensor[B, L, H]`;
- the model output contract used by evaluation;
- the one-model-per-file organization rule.

The change is isolated to the training runtime boundary and should not require model-file edits.

## Testing Plan

### Configuration validation

Add tests proving that:

- `optimizer_name: adamw` is accepted;
- unknown optimizer names are rejected;
- `gradient_clip_norm: 1.0` is accepted;
- non-positive clipping values are rejected;
- valid cosine scheduler blocks are accepted;
- malformed cosine scheduler blocks are rejected;
- current `reduce_on_plateau` configs continue to load unchanged.

### Optimizer construction

Add a focused test proving that the offline builder constructs `torch.optim.AdamW` when `optimizer_name: adamw` is configured.

### Cosine learning-rate behavior

Add deterministic tests proving that:

- warmup returns the configured warmup start LR at the beginning;
- LR changes within an epoch as batch index changes;
- LR decreases toward `cosine_end_lr` over the configured 300-epoch horizon;
- the last training progress approaches the intended floor.

### Trainer behavior

Add trainer tests proving that:

- cosine LR updates happen per training iteration rather than only once per epoch;
- plateau scheduling still steps only from configured epoch metrics;
- gradient clipping is applied when configured and skipped when absent;
- logged `optimizer_lr` remains available for epoch metrics after cosine training.

### Regression coverage

Run at minimum:

```bash
pytest -q tests/test_config_loading.py tests/test_learning_rate_scheduler.py tests/test_checkpoint_roundtrip.py
```

and then the existing focused training smoke tests that already exercise the offline path.

## Risk and Mitigation

| Risk | Mitigation |
|---|---|
| Cosine and plateau schedulers are forced into one incompatible interface | Keep separate per-iteration and per-epoch code paths with explicit names and tests. |
| Config schema becomes ambiguous | Use scheduler-specific validation and reject irrelevant fields where feasible. |
| Best-checkpoint behavior becomes unclear for cosine | Keep checkpoint monitoring independent from cosine and default to `val_loss` unless an explicit future checkpoint-monitor field is introduced. |
| Run metadata claims behavior the runtime does not implement | Update experiment names and WandB names only after the executable config matches them. |
| Gradient clipping silently changes all existing experiments | Make clipping opt-in through config and preserve current behavior when the field is absent. |
| A broad optimizer abstraction increases code complexity | Limit the first pass to `adam` and `adamw` in the offline path only. |

## Open Questions

1. Should `gradient_clip_norm` be the final field name, or should the repository adopt the design-starter spelling `grad_clip_norm`?
2. Should the existing legacy baseline config be corrected in place or left as a historical plateau config and replaced by two new explicit AdamW-cosine configs?
3. Should optimizer configurability be extended to `scripts/run_online_adaptation.py` in the same pass, or kept out of scope because the immediate experiment family is offline only?
4. Should the cosine path log per-batch LR values anywhere beyond console output, or is epoch-level summarized LR sufficient for the first implementation?

## Suggested Implementation Sequence

1. Add config schema support for `optimizer_name`, `gradient_clip_norm`, and cosine scheduler fields.
2. Add unit tests for valid and invalid optimizer/scheduler configurations.
3. Add optimizer selection in `scripts/train.py`.
4. Add pure cosine-policy helper logic with warmup and unit tests.
5. Add trainer-side per-iteration LR updates and gradient clipping.
6. Add trainer regression tests for cosine and clipping while preserving plateau behavior.
7. Add the two SMD `2-1` experiment configs and reconcile the legacy baseline config naming.
8. Run focused tests and one short smoke run before launching the first 300-epoch experiment.

## Preliminary Recommendation

Implement only the offline training extension first. This is sufficient to support the immediate RedLamp-baseline experiment family and keeps scope tight. The online adaptation path should be considered in a later pass unless a concrete experiment requires optimizer configurability there.
