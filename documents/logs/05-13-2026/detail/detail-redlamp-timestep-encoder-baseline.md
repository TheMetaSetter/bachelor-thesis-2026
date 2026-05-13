---
date: 2026-05-13 22:03:39 +07 +0700
researcher: Codex
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for changing the RedLamp MLP baseline to a timestep encoder"
tags: [detail, implementation-plan, redlamp, baseline, mlp, timestep-encoder, ablation]
status: complete
last_updated: 2026-05-13
last_updated_by: Codex
source_request: "Change the RedLamp MLP baseline so it encodes each timestep with input_dim -> latent_dim, preserving only the prototype/fusion/memory differences between the thesis model and baseline."
---

# Detail: RedLamp Timestep Encoder Baseline

## Objective

This detailed plan changes the RedLamp-inspired MLP baseline from a flattened-window encoder to a timestep encoder. The thesis objective is to make the baseline comparison more controlled: the thesis model and the RedLamp baseline should both encode each timestep from `input_dim` to a hidden representation, while the thesis model alone retains continuous prototypes, discrete codebook, fusion `alpha`/`beta`, update gate, and memory/bootstrap logic.

The implementation must preserve the repository contracts:

- Data batches expose `batch["x"]: Tensor[B, L, D]`.
- RedLamp baseline outputs expose `outputs["hidden"]: Tensor[B, L, latent_dim]`.
- RedLamp baseline outputs expose `outputs["pooled"]: Tensor[B, latent_dim]`.
- RedLamp baseline outputs expose `outputs["recon"]: Tensor[B, L, D]`.
- RedLamp baseline outputs expose `outputs["logits"]: Tensor[B, 12]`.
- RedLamp baseline outputs expose `outputs["point_scores"]: Tensor[B, L]`.
- The baseline remains self-contained in `src/models/redlamp_mlp_baseline.py`.
- No prototype, fusion, memory, bootstrap, online adaptation, or projector logic is added to the baseline.

The new baseline should no longer encode `window_size * input_dim` as a single flattened vector. It should encode each timestep independently with the same MLP depth contract:

```text
input_dim -> latent_dim -> latent_dim -> latent_dim
```

for the encoder when `mlp_num_linear_layers == 3`, and should decode each timestep hidden vector back to `input_dim`.

## Phase 1: Add Failing Tests for Timestep Encoding

### Phase Summary

This phase establishes the expected baseline behavior before modifying production code. It directly supports the thesis comparison by proving that the RedLamp baseline uses a true timestep hidden state instead of an expanded window-level latent vector.

### File-Level Edits

Modify `tests/test_redlamp_mlp_baseline.py`.

Replace `test_redlamp_mlp_baseline_forward_contract_and_mlp_depth` with a version that checks the encoder and decoder first linear layer dimensions:

```python
def test_redlamp_mlp_baseline_forward_contract_and_mlp_depth() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=20,
        latent_dim=16,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout=0.0,
        anomaly_probability=1.0,
    )
    batch = {
        "x": torch.randn(2, 20, 4),
        "point_labels": torch.zeros(2, 20, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    outputs = model(batch)

    assert outputs["recon"].shape == (2, 20, 4)
    assert outputs["hidden"].shape == (2, 20, 16)
    assert outputs["pooled"].shape == (2, 16)
    assert outputs["logits"].shape == (2, len(REDLAMP_MULTICLASS_CLASS_NAMES))
    assert outputs["point_scores"].shape == (2, 20)
    assert torch.allclose(
        outputs["aux"]["class_probabilities"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.encoder) == 3
    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.decoder) == 3
    assert (
        sum(isinstance(layer, torch.nn.Linear) for layer in model.classification_head)
        == 3
    )
    assert model.encoder[0].in_features == 4
    assert model.encoder[0].out_features == 16
    assert model.decoder[0].in_features == 16
    assert model.decoder[-1].out_features == 4
```

Add a second test to prove that different timesteps can have different hidden vectors. This prevents an implementation that encodes a window once and broadcasts the latent representation over time:

```python
def test_redlamp_mlp_baseline_hidden_is_not_broadcast_window_latent() -> None:
    model = RedLampMLPBaseline(
        input_dim=4,
        window_size=3,
        latent_dim=8,
        mlp_num_linear_layers=3,
        classifier_dim=8,
        num_classes=len(REDLAMP_MULTICLASS_CLASS_NAMES),
        dropout=0.0,
        anomaly_probability=1.0,
    )
    model.eval()
    batch = {
        "x": torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        "point_labels": torch.zeros(1, 3, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}],
    }

    outputs = model(batch)

    assert outputs["hidden"].shape == (1, 3, 8)
    assert not torch.allclose(outputs["hidden"][:, 0, :], outputs["hidden"][:, 1, :])
```

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result before implementation:

- The first test fails because `model.encoder[0].in_features` is currently `window_size * input_dim`.
- The second test fails because the current implementation expands one latent vector across all timesteps.

### Interfaces and Contracts

- Dataset contract remains `batch["x"]: Tensor[B, L, D]`.
- Model contract remains unchanged for callers.
- The test asserts internal encoder dimensions because this change is specifically about the representation geometry of the baseline.

### Design Pattern Application

- Composition over inheritance: tests continue to instantiate the existing `RedLampMLPBaseline` class without adding a subclass.
- Adapter pattern for encoders: the baseline uses the existing `build_multilayer_perceptron` helper as a small adapter from timestep features to latent features.
- Strategy pattern for tasks: no task strategy changes are needed; `training_step`, `validation_step`, `synthetic_validation_step`, and `test_step` remain model-owned methods.
- Registry/factory: no registry changes are needed because the model name remains `redlamp_mlp_baseline`.

### Risk Mitigation

- Prototype redundancy: the baseline still contains no prototype memory, so it remains a clean comparison point.
- Fusion collapse: the baseline still contains no fusion parameters, so fusion diagnostics remain thesis-model-specific.
- Adaptation contamination: the baseline remains offline-only in this task; online adaptation code is not touched.
- Projector drift: no projector or online adapter is introduced.
- Evaluation metric inflation: pointwise scores and VUS-PR evaluation remain unchanged; only representation geometry changes.

### Acceptance Criteria

- The failing tests specifically identify the flattened-window implementation.
- No production file is modified in this phase.

## Phase 2: Change the RedLamp Baseline to Timestep Encoder and Decoder

### Phase Summary

This phase modifies only the owning model file, following the one-model-one-file rule. The RedLamp baseline becomes a timestep MLP autoencoder plus a window-level classifier over pooled timestep hidden states.

### File-Level Edits

Modify `src/models/redlamp_mlp_baseline.py`.

Change the module docstring from:

```python
"""Self-contained RedLamp-inspired MLP baseline.

The baseline keeps the repository batch and output contracts while matching the
RedLamp comparison setting: flattened windows, an MLP autoencoder, and a
multi-class synthetic anomaly classifier over the shared RedLamp taxonomy.
"""
```

to:

```python
"""Self-contained RedLamp-inspired MLP baseline.

The baseline keeps the repository batch and output contracts while using a
timestep encoder for a controlled comparison against the thesis model. It
remains an MLP autoencoder and multi-class synthetic anomaly classifier without
prototype memory, fusion gates, or online adaptation state.
"""
```

Replace the encoder and decoder construction block:

```python
        flattened_dim = window_size * input_dim
        self.encoder = build_multilayer_perceptron(
            input_dim=flattened_dim,
            intermediate_dim=latent_dim,
            output_dim=latent_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=True,
        )
        self.decoder = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=latent_dim,
            output_dim=flattened_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
```

with:

```python
        self.encoder = build_multilayer_perceptron(
            input_dim=input_dim,
            intermediate_dim=latent_dim,
            output_dim=latent_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=True,
        )
        self.decoder = build_multilayer_perceptron(
            input_dim=latent_dim,
            intermediate_dim=latent_dim,
            output_dim=input_dim,
            num_linear_layers=mlp_num_linear_layers,
            dropout=dropout,
            apply_output_activation=False,
        )
```

Replace the forward path:

```python
        flattened_x = x_tensor.reshape(batch_size, self.window_size * self.input_dim)
        latent = self.encoder(flattened_x)
        reconstructed_flat = self.decoder(latent)
        recon = reconstructed_flat.reshape(batch_size, self.window_size, self.input_dim)
        logits = self.classification_head(latent)
        class_probabilities = torch.softmax(logits, dim=-1)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
        hidden = latent.unsqueeze(1).expand(batch_size, self.window_size, self.latent_dim)
```

with:

```python
        hidden = self.encoder(x_tensor)
        pooled_hidden = hidden.mean(dim=1)
        recon = self.decoder(hidden)
        logits = self.classification_head(pooled_hidden)
        class_probabilities = torch.softmax(logits, dim=-1)
        point_scores = torch.mean((recon - x_tensor) ** 2, dim=-1)
```

Change the output dictionary field:

```python
            "pooled": latent,
```

to:

```python
            "pooled": pooled_hidden,
```

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result after implementation:

- Both RedLamp baseline shape tests pass.
- Encoder first linear layer consumes `input_dim`, not `window_size * input_dim`.
- Decoder last linear layer emits `input_dim`, not `window_size * input_dim`.
- Hidden vectors are produced per timestep and are not a broadcasted window latent.

### Interfaces and Contracts

- `RedLampMLPBaseline.__init__` signature remains backward-compatible.
- `window_size` remains required because the model validates incoming batch shape.
- `latent_dim` continues to control the hidden representation size.
- `classification_head` remains a window-level classifier by consuming `pooled_hidden: Tensor[B, latent_dim]`.
- `recon` remains `Tensor[B, L, D]`.
- `point_scores` remains `Tensor[B, L]`.
- `hidden` changes semantically from an expanded window latent to a true timestep hidden sequence, while preserving the same tensor shape.

### Design Pattern Application

- Composition over inheritance: encoder, decoder, and classifier remain composed `nn.Sequential` modules.
- Adapter pattern for encoders: `build_multilayer_perceptron` adapts `input_dim` timestep features into `latent_dim` features without introducing a new encoder class.
- Strategy pattern for tasks: the existing model-owned stage methods keep the same behavior and require no task branch.
- Registry/factory: `scripts/train.py` and `scripts/evaluate.py` continue to build the model through `build_model("redlamp_mlp_baseline", ...)`.

### Risk Mitigation

- Prototype redundancy: no prototype branch is introduced into the baseline.
- Fusion collapse: no `alpha`, `beta`, or fusion layer is introduced into the baseline.
- Adaptation contamination: no online adaptation state is added, and `test_step` remains a pure offline model step.
- Projector drift: no projector module is touched.
- Evaluation metric inflation: threshold selection, VUS-PR, PR-AUC, ROC-AUC, and pointwise score aggregation remain unchanged.

### Acceptance Criteria

- `src/models/redlamp_mlp_baseline.py` no longer contains `flattened_dim`.
- `forward` no longer calls `x_tensor.reshape(batch_size, self.window_size * self.input_dim)`.
- `outputs["hidden"]` is the direct encoder output.
- `outputs["pooled"]` is `hidden.mean(dim=1)`.
- Existing model callers do not require code changes.

## Phase 3: Validate Training Step, Checkpoint Compatibility Surface, and Config Loading

### Phase Summary

This phase verifies that the representation geometry change does not break training, model construction from YAML, or the offline evaluation protocol. It does not attempt to preserve compatibility with old flattened-window checkpoints because the architecture changes parameter shapes.

### File-Level Edits

No production code edits are expected in this phase.

If checkpoint compatibility needs to be documented in a later phase, add a note to the implementation log. Do not add automatic checkpoint migration because the flattened baseline and timestep baseline are intentionally different scientific baselines.

### Validation Commands

Run the baseline unit and one-step training tests:

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py
```

Run config loading:

```bash
./.venv/bin/pytest -q tests/test_config_loading.py
```

Run RedLamp preflight:

```bash
./.venv/bin/python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  --preflight-only
```

Run a smoke-relevant metric and evaluator suite because this baseline feeds point scores into offline metrics:

```bash
./.venv/bin/pytest -q tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py
```

### Interfaces and Contracts

- Dataset config remains `configs/data/smd_rtx3090_machine_2_1_20.yaml` for the long baseline run.
- Experiment config remains `configs/experiment/smd_redlamp_mlp_baseline_window20.yaml`.
- Model config remains `configs/model/redlamp_mlp_baseline.yaml`.
- The model parameter shapes change, so old flattened-window RedLamp checkpoints should not be loaded into the timestep baseline.

### Design Pattern Application

- Composition over inheritance remains unchanged.
- Adapter pattern remains limited to the local MLP helper.
- Strategy pattern remains unchanged because the same model stage methods are used.
- Registry/factory remains unchanged because model registration does not change.

### Risk Mitigation

- Prototype redundancy: training-step validation confirms the baseline can learn without prototype modules.
- Fusion collapse: no fusion metrics are expected for this baseline; ablation summary should leave thesis-only fusion fields absent or `None` when not logged.
- Adaptation contamination: preflight and tests use offline scripts only.
- Projector drift: no projector tests are needed because projector code is untouched.
- Evaluation metric inflation: evaluator tests ensure VUS-PR and thresholded metrics still consume pointwise outputs without a model-specific adjustment.

### Acceptance Criteria

- `tests/test_redlamp_mlp_baseline.py` passes.
- `tests/test_one_redlamp_mlp_train_step.py` passes.
- `tests/test_config_loading.py` passes.
- RedLamp preflight passes.
- Metric and evaluator tests pass.

## Phase 4: Run Smoke Baseline on RTX3090 Before Long Training

### Phase Summary

This phase validates the timestep RedLamp baseline under the actual training entrypoint before the long 300-epoch baseline run. The smoke run should catch CUDA, data loading, and shape issues early.

### File-Level Edits

No code edits are required if Phases 1 through 3 pass.

If a dedicated RedLamp smoke config is desired, create `configs/experiment/smd_redlamp_mlp_baseline_window20_smoke.yaml` with:

```yaml
experiment_name: smd_redlamp_mlp_baseline_window20_smoke
seed: 11
device: cuda
output_dir: outputs/smd_redlamp_mlp_baseline_window20_smoke
checkpoint_dir: outputs/smd_redlamp_mlp_baseline_window20_smoke/checkpoints
data_config_path: configs/data/smd_rtx3090_smoke.yaml
model_config_path: configs/model/redlamp_mlp_baseline.yaml
task_config_path: configs/task/multitask_tsad_redlamp_multiclass_window20.yaml
optimizer:
  learning_rate: 0.0001
  weight_decay: 0.0
epochs: 1
evaluation:
  vus_max_buffer_size: 20
  vus_num_thresholds: 200
logging:
  use_wandb: false
  wandb_project: bachelor-thesis-2026
  wandb_mode: disabled
  wandb_tags:
    - smd
    - redlamp_multiclass
    - window20
    - smoke
    - timestep-encoder
  wandb_run_name: smd_redlamp_mlp_baseline_window20_smoke
```

This config is optional. If it is not created, use the unit tests and preflight gate before running the full baseline.

### Validation Commands

If the smoke config is created, run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/train.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20_smoke.yaml
```

Then evaluate:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/evaluate.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20_smoke.yaml \
  --checkpoint-path outputs/smd_redlamp_mlp_baseline_window20_smoke/checkpoints/best.pt
```

If no smoke config is created, run the full baseline only after all Phase 3 validations pass:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 ./.venv/bin/python scripts/train.py \
  --experiment-config configs/experiment/smd_redlamp_mlp_baseline_window20.yaml
```

### Interfaces and Contracts

- Smoke config, if created, should reuse the same RedLamp model and task config as the full baseline.
- Smoke config should differ only in dataset scale, output paths, and epoch count.
- Full baseline output remains `outputs/smd_redlamp_mlp_baseline_window20`.

### Design Pattern Application

- Composition over inheritance: no new model class is created for smoke.
- Adapter pattern: no additional adapter is created.
- Strategy pattern: smoke and full configs select behavior through YAML, not code branches.
- Registry/factory: smoke and full experiments both use the same model registry name.

### Risk Mitigation

- Prototype redundancy: smoke does not involve thesis prototypes.
- Fusion collapse: smoke does not involve fusion.
- Adaptation contamination: smoke uses offline `scripts/train.py` and `scripts/evaluate.py`.
- Projector drift: no online projector is run.
- Evaluation metric inflation: smoke evaluation must report `pr_auc` and `vus_pr` from the shared evaluator.

### Acceptance Criteria

- Smoke training writes `outputs/smd_redlamp_mlp_baseline_window20_smoke/checkpoints/best.pt` if the smoke config is created.
- Smoke evaluation writes `evaluation_metrics.json` containing `pr_auc` and `vus_pr`.
- If the optional smoke config is skipped, Phase 3 verification must pass before full training starts.

## Phase 5: Documentation and Reproducibility Notes

### Phase Summary

This phase records the scientific deviation from the original CANDI-style flattened MLP baseline. This is necessary because the new RedLamp baseline is no longer a direct implementation of CANDI's flattened-window MLP geometry; it is a controlled baseline for isolating thesis-specific prototype and fusion mechanisms.

### File-Level Edits

After implementation, create an implementation note:

`documents/logs/05-13-2026/detail/detail-redlamp-timestep-encoder-baseline-implementation.md`

Include this structure. During execution, obtain the frontmatter timestamp with
`date '+%Y-%m-%d %H:%M:%S %z %Z'` and obtain the commit identifier with
`git rev-parse HEAD`; write the actual command outputs into the note.

```markdown
---
date: 2026-05-13 22:03:39 +07 +0700
researcher: Codex
git_commit: afba6086047157d72eb96249544402b1d72bc699
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation notes for RedLamp timestep encoder baseline"
tags: [detail, implementation, redlamp, baseline, mlp, timestep-encoder]
status: complete
source_detail: documents/logs/05-13-2026/detail/detail-redlamp-timestep-encoder-baseline.md
---

# Detail: RedLamp Timestep Encoder Baseline Implementation Notes

## Implemented Scope

- Changed RedLamp baseline encoder from flattened-window input to timestep input.
- Changed RedLamp baseline decoder from flattened-window output to timestep output.
- Kept classifier window-level by mean-pooling timestep hidden states.
- Preserved the existing batch, output, training, validation, testing, and evaluation contracts.

## Scientific Note

The original CANDI MLP reference uses flattened-window encoding. This repository intentionally changes the RedLamp baseline to timestep encoding so the comparison against the thesis model isolates prototype memory, discrete codebook, fusion, update gate, and memory/bootstrap logic.

## Final Verification

- `./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py tests/test_one_redlamp_mlp_train_step.py tests/test_config_loading.py tests/test_vus_pr_metric.py tests/test_evaluator_thresholding.py` -> record the exact pass or failure output from the implementation session.
- `./.venv/bin/python scripts/run_multiseed_experiments.py --config-paths configs/experiment/smd_redlamp_mlp_baseline_window20.yaml --preflight-only` -> record the exact preflight output from the implementation session.
```

### Validation Commands

Run the final compact suite:

```bash
./.venv/bin/pytest -q \
  tests/test_redlamp_mlp_baseline.py \
  tests/test_one_redlamp_mlp_train_step.py \
  tests/test_config_loading.py \
  tests/test_vus_pr_metric.py \
  tests/test_evaluator_thresholding.py
```

Run RedLamp preflight:

```bash
./.venv/bin/python scripts/run_multiseed_experiments.py \
  --config-paths configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  --preflight-only
```

### Interfaces and Contracts

- Documentation must make clear that old RedLamp checkpoints trained with flattened-window geometry are not compatible with the new timestep baseline architecture.
- The experiment name may remain `smd_redlamp_mlp_baseline_window20`, but thesis notes must state that this is the timestep-encoder RedLamp baseline after this implementation.

### Design Pattern Application

- Composition over inheritance remains the preferred model construction pattern.
- Adapter pattern is represented by local use of `build_multilayer_perceptron`.
- Strategy pattern remains YAML-driven.
- Registry/factory path remains unchanged.

### Risk Mitigation

- Prototype redundancy: documentation states the baseline has no prototype modules.
- Fusion collapse: documentation states fusion remains thesis-only.
- Adaptation contamination: documentation states offline-only scope.
- Projector drift: documentation states projector untouched.
- Evaluation metric inflation: documentation records the final metric keys and evaluation path.

### Acceptance Criteria

- Implementation note exists under `documents/logs/05-13-2026/detail/`.
- Final verification commands and results are recorded.
- The note explicitly distinguishes this controlled timestep baseline from CANDI's flattened-window MLP reference.

## Overall Acceptance Criteria

- `RedLampMLPBaseline.encoder[0].in_features == input_dim`.
- `RedLampMLPBaseline.decoder[-1].out_features == input_dim`.
- `outputs["hidden"]` is produced directly by the encoder and has shape `Tensor[B, L, latent_dim]`.
- `outputs["pooled"] == outputs["hidden"].mean(dim=1)` in shape and semantic role.
- `outputs["recon"]` has shape `Tensor[B, L, D]`.
- `outputs["logits"]` has shape `Tensor[B, 12]`.
- `outputs["point_scores"]` has shape `Tensor[B, L]`.
- Baseline remains free of continuous prototypes, discrete codebook, fusion `alpha`/`beta`, update gate, memory/bootstrap logic, online adaptation, and projector modules.
- Existing training and evaluation scripts require no caller changes.
- The long baseline can be run through `scripts/train.py` and `scripts/evaluate.py` after smoke validation.

## Explicit Non-Goals

- Do not add continuous prototypes to the RedLamp baseline.
- Do not add discrete codebook or vector quantization to the RedLamp baseline.
- Do not add fusion `alpha`/`beta` parameters to the RedLamp baseline.
- Do not add memory/bootstrap logic to the RedLamp baseline.
- Do not add online adaptation or projector behavior.
- Do not change synthetic anomaly injection taxonomy.
- Do not change VUS-PR, threshold selection, or evaluator aggregation.
- Do not introduce a new model registry name unless the user explicitly asks for both flattened and timestep baselines to coexist.

## Suggested Commit Sequence

Use short imperative commit messages consistent with repository history:

1. `Test RedLamp timestep encoding`
2. `Use timestep encoder in RedLamp baseline`
3. `Document RedLamp timestep baseline`
