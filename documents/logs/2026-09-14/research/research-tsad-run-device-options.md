---
date: 2026-09-14 00:00:00 +07:00
researcher: OpenAI Codex
topic: "Three ways to add device to tsad.run()"
status: complete
revision: not recorded; tsad-lib is not a Git repository
---

# Research: Three ways to add `device` to `tsad.run()`

## Summary

Today, `RunConfig` already has `device="cpu"`. The public `run()` function does not accept it. The experiment runner also creates the THESIS and reference runners without passing it. Therefore, a notebook cannot currently select MPS or CUDA through `run()`.

The smallest design is a direct optional keyword: `run(..., device="mps")` or `run(..., device="cuda:0")`.

## Research question

What are three ways to add `device` to `run(...)`?

## The story today

A student calls `tsad.run()`. The function creates `ExperimentRunner`. The runner creates a `RunConfig` inside `run_request()`. It never puts a device value into that configuration.

The THESIS runner can already accept a device in its constructor. Its constructor changes the text into `torch.device`. However, the experiment runner creates it with only a seed and an output directory. It therefore uses the THESIS runner default: CPU.

The reference-model runner also has a device constructor argument. The experiment runner creates it without that argument, so it also uses CPU. Its training tensor is currently created on CPU, and it does not move the model to its selected device. Passing a device through the public API is necessary but not sufficient for reference-model GPU support.

Isolation Forest uses scikit-learn. It stays on CPU. A device choice should be recorded for the run, but it should not claim to accelerate this model.

Two copied reference models need separate care. `FiLM` chooses `cuda:0` at module import time. `MICN` has CUDA defaults. These models cannot honestly promise arbitrary device support until their local device handling is corrected and tested.

## Way 1: one direct keyword

The student writes one extra word only when needed.

```python
report = run(
    data_root=DATA,
    datasets=["ICCAD"],
    models=["thesis"],
    entities=["small-iccad"],
    device="mps",
)
```

On a selected GPU inside Docker, the student writes `device="cuda:0"`.

The public function passes the string to `ExperimentRunner`. The runner puts it in `RunConfig`. It then gives `config.device` to `ThesisRunner` and `TimeSeriesLibraryRunner`. The resolved configuration already becomes an artifact, so the selected device is recorded without a new artifact format.

This path has one public concept and one value. It is the best fit for a first notebook. It should accept only `cpu`, `mps`, and valid PyTorch CUDA device strings such as `cuda:0`. It should fail early when the requested accelerator is unavailable.

This is the recommended design.

## Way 2: a configuration object

The student keeps device with the other advanced execution settings.

```python
from tsad.types import RunConfig

report = run(
    data_root=DATA,
    datasets=["ICCAD"],
    models=["thesis"],
    entities=["small-iccad"],
    config=RunConfig(device="mps"),
)
```

The runner merges this configuration with the request values that it owns, such as dataset, entity, variant, seed, and output root. The variant catalog must still remain the owner of losses and routing. The caller must not override those scientific choices by accident.

This path is useful for research notebooks that also set epochs, window size, or memory-bank sizes. It makes the first notebook longer. It also creates a merge rule that students must learn. The merge rule is extra code and extra tests.

## Way 3: a small execution object

The student keeps machine choices separate from scientific choices.

```python
from tsad import Execution, run

report = run(
    data_root=DATA,
    datasets=["ICCAD"],
    models=["thesis"],
    entities=["small-iccad"],
    execution=Execution(device="mps"),
)
```

`Execution` would contain only runtime facts: `device`, perhaps later `num_workers`, and no loss or model settings. `RunConfig` would keep experiment facts. The runner would turn both objects into one resolved configuration before training.

This story is clean when the library later has many machine settings. Today it adds a class, another import, another validation path, and another object for a beginner. It is not the minimal first release.

## Docker and rented GPUs

The library should select a device visible inside the process, not a physical GPU name such as `RTX 4070 Ti S` or `Tesla V100`. GPU name is useful for capacity planning. It is not a stable API value.

For one smoke run, Docker should expose one chosen host GPU. Inside the container, that chosen GPU should be addressed as `cuda:0`. This keeps the notebook identical across host GPU IDs and GPU models. This is an operational Docker convention, not behavior implemented by `tsad-lib`; the target container must verify it with PyTorch before training.

## Evidence

| Finding | Evidence | Status |
| --- | --- | --- |
| `run()` has no `device` parameter. | `src/tsad/api.py:41-54` | Implemented. |
| `RunConfig` has `device="cpu"`. | `src/tsad/types.py:67-121` | Implemented. |
| The runner resolves configuration without a device value. | `src/tsad/experiment.py:317-346` | Implemented. |
| THESIS can receive a device but is created without one. | `src/tsad/models/thesis.py:171-195`; `src/tsad/experiment.py:378-404` | Implemented. |
| Reference runner has a device argument but is created without one. | `src/tsad/models/time_series_library/runner.py:11-55`; `src/tsad/experiment.py:461-470` | Implemented. |
| CLI has no `--device` argument. | `src/tsad/cli.py:17-63` | Implemented. |
| FiLM and MICN have CUDA-specific defaults. | `src/tsad/models/time_series_library/models/FiLM.py:8-50`; `src/tsad/models/time_series_library/models/MICN.py:16-28` | Implemented. |

## Recommendation

Start with Way 1. Add `device="cpu"` as the default keyword in `run()`. Pass it through one straight path to `RunConfig`, THESIS, and supported reference runners. Keep `isolation_forest` on CPU and report that fact. Treat FiLM and MICN as separate compatibility work. Do not add an `Execution` class or a general configuration merge until a concrete notebook needs more than one runtime setting.
