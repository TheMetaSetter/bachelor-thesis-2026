---
date: 2026-05-14 16:54:50 +0700 +07
researcher: Codex
git_commit: a7918159ba6acb949e39deef6601b28a3d6eb39f
branch: dev
repository: bachelor-thesis-2026
topic: "Detailed programming plan for flattening latent representations before classifier heads"
tags: [detail, implementation-plan, redlamp, thesis-multitask, classifier, latent-flattening]
status: complete
last_updated: 2026-05-14
last_updated_by: Codex
source_prompt: prompts/4_detail_prompt.md
source_request: "Make redlamp_mlp_baseline.py and thesis_multitask.py match the RedLamp pattern: flatten latent representation before classifier head, but do not flatten latent representation before reconstruction head."
---

# Detail: Flatten Classifier Latent for RedLamp Baseline and Thesis Multitask Model

## Objective

This detailed plan changes the classification path in `src/models/redlamp_mlp_baseline.py` and `src/models/thesis_multitask.py` so both models follow the original RedLamp codebase pattern: the reconstruction path consumes structured latent tokens, while the classification path consumes a flattened latent representation. The implementation must preserve the repository's thesis-facing hidden contract:

- Data batches expose `batch["x"]: Tensor[B, L, D]`.
- Models expose `outputs["hidden"]: Tensor[B, L, H]`.
- Models expose `outputs["recon"]: Tensor[B, L, D]`.
- Models expose `outputs["logits"]: Tensor[B, C]`.
- Models expose `outputs["point_scores"]: Tensor[B, L]`.

The only intentional semantic change is that `outputs["pooled"]` becomes the classifier input tensor with shape `Tensor[B, L * H]` for the two affected models. This field is no longer mean-pooled for these models. This decision was chosen explicitly so the public output reflects the tensor used by the classifier head.

The reconstruction heads must not receive flattened latent tensors. This matches the RedLamp reference behavior where `x_enc` is passed directly to the decoder, while only the classifier receives `x_enc.reshape(x_enc.size(0), -1)`.

## Phase 1: Add Failing Tests for RedLamp Baseline Flattened Classifier Input

### Phase Summary

This phase locks the RedLamp baseline behavior before production code changes. It verifies that the baseline remains a timestep autoencoder for reconstruction while the classifier consumes all timestep latent vectors flattened into one window-level vector.

### File-Level Edits

Modify `tests/test_redlamp_mlp_baseline.py`.

In `test_redlamp_mlp_baseline_forward_contract_and_mlp_depth`, change the pooled assertion:

```python
assert outputs["pooled"].shape == (2, 20 * 16)
```

Add classifier input dimension assertions after the existing `classification_head` depth assertion:

```python
assert model.classification_head[0].in_features == 20 * 16
assert model.classification_head[-1].out_features == len(
    REDLAMP_MULTICLASS_CLASS_NAMES
)
```

Add a focused test:

```python
def test_redlamp_mlp_baseline_flattens_hidden_before_classifier() -> None:
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
    assert outputs["pooled"].shape == (1, 3 * 8)
    assert torch.allclose(
        outputs["pooled"],
        outputs["hidden"].reshape(1, 3 * 8),
    )
```

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result before implementation:

- The existing forward contract test fails because `outputs["pooled"]` is currently `Tensor[B, latent_dim]`.
- The new flattening test fails because `outputs["pooled"]` currently equals `hidden.mean(dim=1)`.
- The classifier dimension assertion fails because `classification_head[0].in_features` is currently `latent_dim`, not `window_size * latent_dim`.

### Interfaces and Contracts

- The data contract remains `batch["x"]: Tensor[B, L, D]`.
- The reconstruction contract remains `outputs["recon"]: Tensor[B, L, D]`.
- The hidden contract remains `outputs["hidden"]: Tensor[B, L, latent_dim]`.
- The classifier input contract changes to `outputs["pooled"]: Tensor[B, L * latent_dim]`.

### Design Pattern Application

- Composition over inheritance remains unchanged: encoder, decoder, and classifier are composed MLP modules inside one model file.
- The encoder remains an adapter from timestep features to latent features.
- The task strategy remains model-owned through `training_step`, `validation_step`, `synthetic_validation_step`, and `test_step`.
- Registry and factory paths remain unchanged because the model name remains `redlamp_mlp_baseline`.

### Risk Mitigation

- Prototype redundancy does not apply to the RedLamp baseline because it has no prototype branches.
- Fusion collapse does not apply because the baseline has no fusion gates.
- Adaptation contamination does not apply because the baseline remains offline-only.
- Projector drift does not apply because no projector exists in this model.
- Evaluation metric inflation is mitigated by preserving `point_scores` and evaluator input contracts.

### Acceptance Criteria

- RedLamp tests fail for the expected classifier input shape reason before implementation.
- No production files are modified in this phase.

## Phase 2: Implement Flattened Classifier Input in RedLamp Baseline

### Phase Summary

This phase changes only `src/models/redlamp_mlp_baseline.py`, preserving the one-model-one-file rule. The reconstruction path continues to decode `hidden: Tensor[B, L, latent_dim]`, while the classifier consumes `hidden.reshape(B, L * latent_dim)`.

### File-Level Edits

Modify `src/models/redlamp_mlp_baseline.py`.

Change classification head construction from:

```python
self.classification_head = build_multilayer_perceptron(
    input_dim=latent_dim,
    intermediate_dim=classifier_dim,
    output_dim=num_classes,
    num_linear_layers=mlp_num_linear_layers,
    dropout=dropout,
    apply_output_activation=False,
)
```

to:

```python
self.classification_head = build_multilayer_perceptron(
    input_dim=window_size * latent_dim,
    intermediate_dim=classifier_dim,
    output_dim=num_classes,
    num_linear_layers=mlp_num_linear_layers,
    dropout=dropout,
    apply_output_activation=False,
)
```

Change the forward path from:

```python
hidden = self.encoder(x_tensor)
pooled_hidden = hidden.mean(dim=1)
recon = self.decoder(hidden)
logits = self.classification_head(pooled_hidden)
```

to:

```python
hidden = self.encoder(x_tensor)
flattened_classification_hidden = hidden.reshape(
    hidden.shape[0],
    self.window_size * self.latent_dim,
)
recon = self.decoder(hidden)
logits = self.classification_head(flattened_classification_hidden)
```

Change output dictionary from:

```python
"pooled": pooled_hidden,
```

to:

```python
"pooled": flattened_classification_hidden,
```

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_redlamp_mlp_baseline.py
```

Expected result after implementation:

- All RedLamp baseline tests pass.
- `outputs["hidden"]` remains `Tensor[B, L, latent_dim]`.
- `outputs["pooled"]` becomes `Tensor[B, L * latent_dim]`.
- `outputs["recon"]` remains `Tensor[B, L, D]`.
- `classification_head[0].in_features == window_size * latent_dim`.

### Interfaces and Contracts

- Existing callers still use `model(batch)` with the same batch format.
- `outputs["pooled"]` keeps the same key but changes shape and meaning to "classification input representation."
- Existing checkpoints trained with the mean-pooled classifier head are not shape-compatible with this classifier head.

### Design Pattern Application

- No new inheritance is introduced.
- The classification head remains a composed MLP.
- The model remains registry-compatible with the existing `redlamp_mlp_baseline` key.

### Risk Mitigation

- Evaluation metric inflation is mitigated because anomaly scores still come from reconstruction error.
- Scientific comparability improves because the baseline classifier now mirrors the original RedLamp flatten-before-classifier pattern.

### Acceptance Criteria

- `src/models/redlamp_mlp_baseline.py` does not call `hidden.mean(dim=1)` before classification.
- `self.decoder(hidden)` remains unchanged and receives a 3D latent tensor.
- `outputs["pooled"]` exactly equals `outputs["hidden"].reshape(B, L * H)` in the new unit test.

## Phase 3: Add Failing Tests and Config Contract for Thesis Multitask Flattened Classifier Input

### Phase Summary

This phase locks the thesis model behavior before production changes. Unlike the RedLamp baseline, `ThesisMultitaskModel` does not currently store `window_size` in its model config. The implementation must pass `data.window_size` into `model.window_size` during experiment config resolution so the classifier head can be constructed with `window_size * hidden_dim`.

### File-Level Edits

Modify `tests/test_multitask_shapes.py`.

In tests that instantiate `ThesisMultitaskModel`, add `window_size` to model kwargs. For the existing shape test with `window_size=20` and `hidden_dim=16`, assert:

```python
assert outputs["pooled"].shape == (4, 20 * 16)
assert model.classification_head[0].in_features == 20 * 16
```

Add a focused test:

```python
def test_multitask_model_flattens_hidden_classification_before_classifier() -> None:
    model = ThesisMultitaskModel(
        input_dim=4,
        window_size=3,
        encoder_dim=8,
        hidden_dim=6,
        mlp_num_linear_layers=3,
        num_classes=2,
        dropout=0.0,
        continuous_enabled=True,
        continuous_num_prototypes=2,
        discrete_enabled=True,
        discrete_codebook_size=3,
        gumbel_temperature=1.0,
        use_synthetic_augmentation=False,
        use_synthetic_validation=False,
        anomaly_probability=0.0,
        min_segment_fraction=0.1,
        max_segment_fraction=0.2,
        spike_scale=3.0,
        lambda_cls=1.0,
        lambda_div=0.0,
        lambda_var=0.0,
        lambda_cov=0.0,
        lambda_use=0.0,
        lambda_gate=0.0,
        usage_lambda_start=0.0,
        usage_lambda_end=0.0,
        usage_lambda_schedule_fraction=1.0,
        variance_floor_gamma=1.0,
        gate_barrier_margin=0.25,
        bootstrap_encoder_epochs=0,
        discrete_ema_decay=0.99,
        memory_norm_epsilon=1.0e-6,
        memory_initialization_batches=1,
        memory_initialization_with_synthetic_windows=False,
        classification_label_mode="binary",
    )
    batch = {
        "x": torch.randn(2, 3, 4),
        "point_labels": torch.zeros(2, 3, dtype=torch.long),
        "mask": None,
        "timestamps": None,
        "meta": [{"entity_id": "unit-test"}, {"entity_id": "unit-test"}],
    }

    outputs = model(batch, stage_name="test")

    assert outputs["hidden"].shape == (2, 3, 6)
    assert outputs["pooled"].shape == (2, 3 * 6)
    assert torch.allclose(
        outputs["pooled"],
        outputs["aux"]["hidden_classification"].reshape(2, 3 * 6),
    )
```

Modify `tests/test_config_loading.py`.

Add a test for propagation from data config to thesis model config:

```python
def test_load_experiment_config_injects_window_size_into_thesis_model(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "data.yaml"
    model_path = tmp_path / "model.yaml"
    task_path = tmp_path / "task.yaml"
    experiment_path = tmp_path / "experiment.yaml"

    data_path.write_text(
        "\n".join(
            [
                "dataset_name: smd",
                "root_dir: data/ServerMachineDataset",
                "entity_ids: [machine-2-1]",
                "window_size: 20",
                "stride: 20",
                "batch_size: 8",
                "num_workers: 0",
                "validation_split_ratio: 0.2",
                "shuffle_train: true",
            ]
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        "\n".join(
            [
                "model_name: thesis_multitask",
                "input_dim: 38",
                "encoder_dim: 64",
                "hidden_dim: 32",
                "mlp_num_linear_layers: 3",
                "num_classes: 2",
                "dropout: 0.1",
                "continuous_enabled: true",
                "continuous_num_prototypes: 8",
                "discrete_enabled: true",
                "discrete_codebook_size: 16",
                "gumbel_temperature: 1.5",
                "temperature_start: 1.5",
                "temperature_end: 0.7",
                "temperature_anneal_fraction: 0.8",
                "temperature_hold_fraction: 0.0",
                "alpha_logit_init: 0.0",
                "beta_logit_init: 0.0",
                "use_label_refurbishment: false",
                "refurbishment_alpha: 0.0",
                "refurbishment_beta: 0.0",
                "reconstruction_normal_only: false",
                "lambda_cls: 1.0",
                "lambda_div: 0.0",
                "lambda_var: 0.0",
                "lambda_cov: 0.0",
                "lambda_use: 0.0",
                "lambda_gate: 0.0",
                "usage_lambda_start: 0.0",
                "usage_lambda_end: 0.0",
                "usage_lambda_schedule_fraction: 1.0",
                "variance_floor_gamma: 1.0",
                "gate_barrier_margin: 0.25",
                "bootstrap_encoder_epochs: 0",
                "discrete_ema_decay: 0.99",
                "memory_norm_epsilon: 1.0e-6",
                "memory_initialization_batches: 1",
                "memory_initialization_with_synthetic_windows: false",
            ]
        ),
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                "task_name: multitask_tsad",
                "classification_label_mode: binary",
                "freeze_fusion_for_epochs: 0",
                "warmup_alpha_value: 0.5",
                "warmup_beta_value: 0.5",
                "use_synthetic_augmentation: true",
                "use_synthetic_validation: true",
                "synthetic_validation_seed: 7",
                "anomaly_probability: 0.5",
                "min_segment_fraction: 0.1",
                "max_segment_fraction: 0.2",
                "spike_scale: 3.0",
                "balance_binary_classes_within_batch: false",
                "anomaly_families: [spike, flip]",
            ]
        ),
        encoding="utf-8",
    )
    experiment_path.write_text(
        "\n".join(
            [
                "experiment_name: unit_test_window_size_injection",
                "seed: 11",
                "device: cpu",
                "output_dir: outputs/unit_test_window_size_injection",
                "checkpoint_dir: outputs/unit_test_window_size_injection/checkpoints",
                f"data_config_path: {data_path}",
                f"model_config_path: {model_path}",
                f"task_config_path: {task_path}",
                "optimizer:",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "epochs: 1",
            ]
        ),
        encoding="utf-8",
    )

    loaded_config = load_experiment_config(experiment_path)

    assert loaded_config["model"]["window_size"] == 20
```

Add a mismatch test:

```python
def test_load_experiment_config_rejects_thesis_model_window_size_mismatch(
    tmp_path: Path,
) -> None:
    # Use the same fixture style as
    # test_load_experiment_config_injects_window_size_into_thesis_model,
    # but write "window_size: 10" into the model config and "window_size: 20"
    # into the data config.
    # The assertion must be:
    with pytest.raises(ValueError, match="model.window_size must match data.window_size"):
        load_experiment_config(experiment_path)
```

When implementing this test, duplicate the full local YAML setup from the propagation test so the test is self-contained and readable.

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_multitask_shapes.py tests/test_config_loading.py
```

Expected result before implementation:

- Direct thesis model construction fails because `window_size` is unknown.
- Existing thesis shape tests fail if they expect flattened pooled output.
- Config propagation test fails because `model.window_size` is not injected.
- Mismatch test fails because no mismatch validation exists.

### Interfaces and Contracts

- `ThesisMultitaskModelConfig.architecture.window_size` becomes required after config resolution.
- `ThesisMultitaskModel` direct construction must include `window_size`.
- `load_experiment_config` must make YAML experiment configs backward-compatible by copying `data.window_size` into `model.window_size` when absent.

### Design Pattern Application

- Composition over inheritance remains unchanged.
- Encoder adapter contract remains `hidden: Tensor[B, L, H]`.
- Strategy pattern remains task-driven through `task_name: multitask_tsad`.
- Registry and factory path remains unchanged with model key `thesis_multitask`.

### Risk Mitigation

- Prototype redundancy is not worsened because prototype branch outputs remain unchanged.
- Fusion collapse monitoring remains valid because `hidden_classification` remains visible in `outputs["aux"]`.
- Adaptation contamination does not apply to this offline path.
- Projector drift does not apply because online adaptation is untouched.
- Evaluation metric inflation is mitigated because point scores remain reconstruction-based.

### Acceptance Criteria

- The tests fail for missing or mismatched flattened classifier behavior before production edits.
- No production files are modified in this phase.

## Phase 4: Implement Flattened Classifier Input in Thesis Multitask Model and Config Resolution

### Phase Summary

This phase updates the thesis model so classification follows the RedLamp reference pattern while reconstruction remains token-wise. The model keeps all prototype, fusion, reconstruction, and auxiliary diagnostics intact.

### File-Level Edits

Modify `src/models/thesis_multitask.py`.

Add `window_size` to `MultitaskArchitectureConfig`:

```python
@dataclass(frozen=True)
class MultitaskArchitectureConfig:
    input_dim: int
    window_size: int
    encoder_dim: int
    hidden_dim: int
    mlp_num_linear_layers: int = 3
    num_classes: int = 2
    dropout: float = 0.0
```

Add `"window_size"` to `architecture_keys` inside `ThesisMultitaskModelConfig.from_flat_kwargs`:

```python
architecture_keys = {
    "input_dim",
    "window_size",
    "encoder_dim",
    "hidden_dim",
    "mlp_num_linear_layers",
    "num_classes",
    "dropout",
}
```

Change missing required architecture keys from:

```python
{"input_dim", "encoder_dim", "hidden_dim"} - set(architecture_values)
```

to:

```python
{"input_dim", "window_size", "encoder_dim", "hidden_dim"} - set(
    architecture_values
)
```

In `_store_config_values`, add:

```python
self.window_size = architecture.window_size
```

In `_build_task_heads`, change classification head construction from:

```python
self.classification_head = build_multilayer_perceptron(
    input_dim=architecture.hidden_dim,
    intermediate_dim=architecture.hidden_dim,
    output_dim=architecture.num_classes,
    num_linear_layers=architecture.mlp_num_linear_layers,
    dropout=architecture.dropout,
    apply_output_activation=False,
)
```

to:

```python
self.classification_head = build_multilayer_perceptron(
    input_dim=architecture.window_size * architecture.hidden_dim,
    intermediate_dim=architecture.hidden_dim,
    output_dim=architecture.num_classes,
    num_linear_layers=architecture.mlp_num_linear_layers,
    dropout=architecture.dropout,
    apply_output_activation=False,
)
```

In `_print_model_summary`, add:

```python
window_size=architecture.window_size,
```

In `forward`, replace mean pooling before classification:

```python
pooled_classification_hidden = hidden_classification.mean(dim=1)
logits = self.classification_head(pooled_classification_hidden)
```

with:

```python
if hidden_classification.shape[1] != self.window_size:
    raise ValueError(
        "hidden_classification must have window dimension "
        f"{self.window_size}, but received {hidden_classification.shape[1]}"
    )
flattened_classification_hidden = hidden_classification.reshape(
    hidden_classification.shape[0],
    self.window_size * self.hidden_dim,
)
logits = self.classification_head(flattened_classification_hidden)
```

Change `outputs["pooled"]` from:

```python
"pooled": pooled_classification_hidden,
```

to:

```python
"pooled": flattened_classification_hidden,
```

Update the nearby explanatory comments so they state that classification uses all timestep hidden vectors flattened into one window-level representation.

Modify `src/core/config.py`.

Inside `load_experiment_config`, after merging `data`, `model`, and `task`, add:

```python
model_config = resolved_experiment_config["model"]
data_config = resolved_experiment_config["data"]
if model_config.get("model_name") == "thesis_multitask":
    data_window_size = data_config.get("window_size")
    model_window_size = model_config.get("window_size")
    if model_window_size is None:
        model_config["window_size"] = data_window_size
    elif model_window_size != data_window_size:
        raise ValueError("model.window_size must match data.window_size")
```

In `validate_experiment_config`, include `window_size` for thesis model integer validation:

```python
if model_config.get("model_name") == "thesis_multitask":
    integer_fields["window_size"] = model_config.get("window_size")
```

Keep the RedLamp baseline validation unchanged because it already has `window_size` in its model YAML.

### Test Command

Run:

```bash
./.venv/bin/pytest -q tests/test_multitask_shapes.py tests/test_config_loading.py
```

Expected result after implementation:

- Thesis model shape tests pass.
- Config propagation and mismatch tests pass.
- Existing experiment YAML files that reference `thesis_multitask` continue to load without adding `window_size` manually.

### Interfaces and Contracts

- `ThesisMultitaskModel` direct construction now requires `window_size`.
- Experiment configs remain backward-compatible through config resolution.
- `outputs["pooled"]` becomes `Tensor[B, L * H]`.
- `outputs["aux"]["hidden_classification"]` remains `Tensor[B, L, H]`.

### Design Pattern Application

- Configuration layer resolves experiment-level decisions before registry model construction.
- Model file remains self-contained for inference and training logic.
- The model factory still receives a flat kwargs mapping.

### Risk Mitigation

- Prototype redundancy risk is mitigated by preserving branch-local diagnostics.
- Fusion collapse risk is mitigated by preserving `alpha`, `beta`, and fused hidden outputs.
- Adaptation contamination and projector drift remain out of scope because online adaptation is untouched.
- Evaluation metric inflation is mitigated by not changing `point_scores`.

### Acceptance Criteria

- `classification_head[0].in_features == window_size * hidden_dim`.
- `outputs["pooled"] == outputs["aux"]["hidden_classification"].reshape(B, L * H)` in the new test.
- `self.reconstruction_head(hidden_reconstruction)` remains token-wise and does not flatten latent tensors.

## Phase 5: Integration Validation and Documentation

### Phase Summary

This phase verifies that classifier flattening does not break training, checkpoint save/load surfaces, evaluator metrics, or config loading. It also records that old classifier-head checkpoints are shape-incompatible.

### File-Level Edits

Create an implementation note after the code is implemented:

`documents/logs/05-14-2026/detail/detail-flatten-classifier-latent-redlamp-thesis-implementation.md`

Use this structure:

```markdown
---
date: <output of date '+%Y-%m-%d %H:%M:%S %z %Z'>
researcher: Codex
git_commit: <output of git rev-parse HEAD>
branch: dev
repository: bachelor-thesis-2026
topic: "Implementation notes for flattened classifier latent representations"
tags: [detail, implementation, redlamp, thesis-multitask, classifier, latent-flattening]
status: complete
source_detail: documents/logs/05-14-2026/detail/detail-flatten-classifier-latent-redlamp-thesis.md
---

# Detail: Flattened Classifier Latent Implementation Notes

## Implemented Scope

- RedLamp baseline classifier now consumes `hidden.reshape(B, L * H)`.
- Thesis multitask classifier now consumes `hidden_classification.reshape(B, L * H)`.
- Reconstruction heads continue to consume structured token tensors.
- Config resolution injects `data.window_size` into thesis model config when omitted.

## Checkpoint Compatibility Note

Old checkpoints whose classifier heads were trained with mean-pooled input are not shape-compatible with the new classifier head input dimension.

## Final Verification

- `<test command>` -> `<exact output summary>`
```

### Validation Commands

Run the focused shape and config suite:

```bash
./.venv/bin/pytest -q \
  tests/test_redlamp_mlp_baseline.py \
  tests/test_multitask_shapes.py \
  tests/test_thesis_multitask_config_refactor.py \
  tests/test_config_loading.py
```

Run one-step training tests:

```bash
./.venv/bin/pytest -q \
  tests/test_one_redlamp_mlp_train_step.py \
  tests/test_one_multitask_train_step.py
```

Run evaluator regression tests:

```bash
./.venv/bin/pytest -q \
  tests/test_evaluator_thresholding.py \
  tests/test_vus_pr_metric.py
```

Run RedLamp and thesis preflight configs:

```bash
./.venv/bin/python scripts/run_multiseed_experiments.py \
  --config-paths \
  configs/experiment/smd_redlamp_mlp_baseline_window20.yaml \
  configs/experiment/smd_thesis_multitask_redlamp_multiclass_window20.yaml \
  --preflight-only
```

### Interfaces and Contracts

- Dataset, dataloader, trainer, evaluator, and registry interfaces remain stable.
- Model output keys remain stable.
- Only the shape and semantic meaning of `outputs["pooled"]` changes for the two affected models.

### Design Pattern Application

- Composition over inheritance is preserved.
- Encoder adapter contracts are preserved.
- Task strategy remains in model-owned stage methods.
- Registry/factory remains unchanged.

### Risk Mitigation

- Prototype redundancy: no prototype implementation changes are introduced.
- Fusion collapse: fusion diagnostics remain available through `outputs["aux"]["fusion"]` and `outputs["aux"]["hidden_classification"]`.
- Adaptation contamination: online adaptation is not touched.
- Projector drift: projector code is not touched.
- Evaluation metric inflation: pointwise anomaly metrics still consume reconstruction-derived `point_scores`.

### Acceptance Criteria

- All validation commands pass or any failures are documented with exact output.
- Implementation note exists under `documents/logs/05-14-2026/detail/`.
- Both models flatten latent representations before classifier heads.
- Neither model flattens latent representations before reconstruction heads.

## Overall Acceptance Criteria

- `RedLampMLPBaseline.classification_head[0].in_features == window_size * latent_dim`.
- `ThesisMultitaskModel.classification_head[0].in_features == window_size * hidden_dim`.
- `RedLampMLPBaseline.outputs["pooled"] == outputs["hidden"].reshape(B, L * H)`.
- `ThesisMultitaskModel.outputs["pooled"] == outputs["aux"]["hidden_classification"].reshape(B, L * H)`.
- `outputs["recon"]` remains `Tensor[B, L, D]` for both models.
- `outputs["logits"]` remains `Tensor[B, C]` for both models.
- `outputs["point_scores"]` remains `Tensor[B, L]` for both models.
- Existing experiment YAML files continue to load through `load_experiment_config`.

## Explicit Non-Goals

- Do not flatten latent representations before reconstruction heads.
- Do not change RedLamp synthetic anomaly taxonomy.
- Do not change prototype lookup, discrete codebook, continuous memory, alpha/beta fusion, or optional loss definitions.
- Do not change evaluator thresholding, VUS-PR, ROC-AUC, PR-AUC, or pointwise score aggregation.
- Do not introduce new model registry names.
- Do not add online adaptation behavior.
