---
date: 2026-09-14 00:00:00 +07:00
researcher: OpenAI Codex
topic: "Code changes required for tsad.run(..., device=...)"
status: complete
revision: not recorded; tsad-lib is not a Git repository
---

# Research: Code changes for `run(..., device=...)`

## Summary

Way 1 needs one straight path.

```text
run(device) → ExperimentRunner(device) → RunConfig(device) → model runner(device)
```

For the native THESIS model, this path needs four production edits. The THESIS runner already moves its model and tensors to the device that its constructor receives.

For copied reference models, the path needs two more runner edits. The reference runner accepts a device today, but it does not move its model or training tensor to that device. `FiLM` and `MICN` need separate compatibility work before they can promise MPS or arbitrary CUDA-device support.

## Research question

Which lines of code in `tsad-lib` need modification to implement Way 1: `run(..., device=...)`?

## The first part of the story: public request reaches configuration

### 1. `src/tsad/api.py:41-54` — required

`run()` has no `device` parameter. Add `device="cpu"` beside the other runtime keywords. Pass it to `ExperimentRunner`.

This gives notebook users the small API:

```python
run(..., device="cpu")
run(..., device="mps")
run(..., device="cuda:0")
```

### 2. `src/tsad/experiment.py:216-228` — required

`ExperimentRunner.__init__()` stores data root, output root, experiment type, and resume state. Add its `device` argument and store it as `self.device`.

`available()` creates this class with only `data_root`, so a default of `"cpu"` preserves that call.

### 3. `src/tsad/experiment.py:334-346` — required

`run_request()` creates the resolved configuration. Add `"device": self.device` to this dictionary.

`RunConfig` already declares `device` at `src/tsad/types.py:82`. The resolved configuration is already written to `resolved_config.yaml`. Therefore, this one added field also records the chosen device in the run artifact and its configuration hash.

### 4. `src/tsad/config.py:9-29` or `src/tsad/types.py:96-121` — required for a clear early error

Current validation does not check `device`. Add one small validation step after `RunConfig` is built. It should reject an invalid device string and reject a requested unavailable backend, such as `mps` on a machine without MPS or `cuda:0` when CUDA is unavailable.

The simplest ownership is `config.py`: it already turns user values into one resolved `RunConfig`. `RunConfig` should remain the stored data object. The validation helper should not choose a device automatically.

## The second part of the story: THESIS uses the selected device

### 5. `src/tsad/experiment.py:378-404` — required

The experiment runner creates THESIS with:

```python
ThesisRunner(seed=request.seed, output_dir=run_dir)
```

Pass `device=config.device` into this constructor.

No change is needed inside `src/tsad/models/thesis.py:171-360`. Its constructor already creates `torch.device(device)`. Stage A, memory initialization, Stage B, and scoring already use `self.device` for model and tensors.

## The third part of the story: reference models use the selected device

### 6. `src/tsad/experiment.py:461-470` — required when a reference model is selected

The experiment runner creates the copied reference model and then creates `TimeSeriesLibraryRunner` without a device. Pass `device=config.device` to that runner.

### 7. `src/tsad/models/time_series_library/runner.py:11-18` — required when a reference model is selected

The constructor stores `self.device`, but leaves `self.model` where it was built. Move a non-empty model to the selected device in this constructor, or at the first line of `fit()`.

Without this edit, `reconstruct()` creates an input tensor on the selected device at line 21 while model parameters can remain on CPU. PyTorch then rejects the mixed devices.

### 8. `src/tsad/models/time_series_library/runner.py:37-55` — required when a reference model is selected

`fit()` creates the training tensor without a device at lines 42-44. Create it on `self.device`.

This keeps the model, input, reconstruction, loss, and backward pass on one device.

## Model-specific boundary

### 9. `src/tsad/models/time_series_library/models/FiLM.py:8,35-50` — required before claiming FiLM support

FiLM chooses `cuda:0` while the module imports. Its buffers and temporary tensor use that global value. This bypasses `run(device=...)`.

Replace the global choice with the device of the module inputs or registered buffers. Then add a conditional accelerator test. Until then, FiLM should not be described as portable across MPS and arbitrary CUDA devices.

### 10. `src/tsad/models/time_series_library/models/MICN.py:16-28` and its later CUDA defaults — required before claiming MICN support

MICN starts with `device="cuda"` defaults and later creates CUDA tensors. It also bypasses the public device choice.

Use input or module device instead. Then test it separately. This is model compatibility work, not part of the native THESIS path.

## CLI boundary

### 11. `src/tsad/cli.py:17-63` — required only if the Docker tutorial uses `tsad run`

The CLI has no `--device` option. Add `--device` with default `cpu`, then pass `args.device` to `api.run()`.

This edit is not required for a notebook-only Way 1. It is required for an honest Ubuntu Docker terminal tutorial.

## Files that do not need a production change

| File | Why it stays unchanged for native THESIS Way 1 |
| --- | --- |
| `src/tsad/types.py:82` | It already stores `device`. |
| `src/tsad/models/thesis.py:171-360` | It already uses its constructor device for tensors and model. |
| `src/tsad/models/thesis_online.py` | It receives the already configured THESIS runner. |
| `src/tsad/models/isolation_forest.py` | scikit-learn Isolation Forest remains CPU-only. |
| `src/tsad/models/time_series_library/forward.py` | It receives tensors prepared by the reference runner. |
| `src/tsad/models/time_series_library/factory.py` | The factory builds architecture, not runtime placement. |

## Test lines that need additions

### 12. `tests/test_public_runtime.py:100-166` — required

Add one focused public-API test. It calls `run(..., device="cpu")` and asserts that the resolved configuration artifact records `"device": "cpu"`. A mocked runner or a conditional accelerator test should prove that a non-default device reaches `ThesisRunner` without requiring GPU hardware in every test environment.

### 13. `tests/test_reference_models.py:30-52` — required when reference device support is enabled

Add one focused test for the reference runner. It should assert that model parameters and training input use the same selected device. Keep the standard test on CPU. Add MPS or CUDA checks only when that backend is available.

### 14. `tests/test_config.py` — required for early device errors

Add tests for an invalid device string and an unavailable requested backend. The test should verify a clear error before training starts.

### 15. `tests/test_cli.py` — required only with `--device`

Add one parser test that proves `--device cuda:0` reaches the same `api.run()` request path.

## Minimal edit set

For a THESIS notebook, change only these production files:

```text
src/tsad/api.py
src/tsad/experiment.py
src/tsad/config.py
```

`src/tsad/models/thesis.py` already has the required device mechanics.

For certified reference models, also change:

```text
src/tsad/models/time_series_library/runner.py
```

For a Docker CLI tutorial, also change:

```text
src/tsad/cli.py
```

FiLM and MICN remain separate compatibility tasks. The minimal first tutorial should use `thesis`, not either of those copied reference models.
