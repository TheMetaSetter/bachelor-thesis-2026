# Minimal Way 1 Device Parameter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `tsad.run()` select `cpu`, `mps`, or one visible CUDA device for native THESIS.

**Architecture:** One string travels from `run()` to `RunConfig` and then to `ThesisRunner`. The resolver validates the string. THESIS moves input tensors to its own device. NumPy receives CPU tensors only.

**Tech Stack:** Python, PyTorch, pytest, scikit-learn.

**Spec:** `documents/logs/2026-09-14/structure/structure-tsad-way-one-device.md`

## Global Constraints

- Run Python and pytest with `tsad-lib/.venv/bin/python`.
- Keep `device="cpu"` as the default.
- Support acceleration for native `thesis` only.
- Do not add CLI support, reference-model support, automatic selection, or multi-GPU execution.
- Preserve O2 losses and `direct_branch_routing`.

---

## Phase 1: define one honest request

### Stage 1.1: validate device text

### Task 1: Add one device validator

**Files:**

- Modify: `tsad-lib/src/tsad/config.py:1-34`
- Modify: `tsad-lib/tests/test_config.py:1-28`

**Interface:** `validate_device(device: str) -> str`

- [ ] **Step 1: Add the malformed-device test**

Add `test_resolve_config_rejects_malformed_device()`. It calls `resolve_config({"device": "cuda:bad"})` and expects `ValueError` containing `device`.

- [ ] **Step 2: Run the malformed-device test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py::test_resolve_config_rejects_malformed_device`

Expected: FAIL.

- [ ] **Step 3: Add the malformed-device branch**

Add `validate_device()` in `config.py`. Make it reject device text outside `cpu`, `mps`, and `cuda:<index>`.

- [ ] **Step 4: Run the malformed-device test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py::test_resolve_config_rejects_malformed_device`

Expected: PASS.

- [ ] **Step 5: Add the unavailable-MPS test**

Add `test_resolve_config_rejects_unavailable_mps()`. Mock MPS availability as false. Call `resolve_config({"device": "mps"})`. Expect `ValueError` containing `unavailable`.

- [ ] **Step 6: Run the unavailable-MPS test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py::test_resolve_config_rejects_unavailable_mps`

Expected: FAIL.

- [ ] **Step 7: Add the MPS availability branch**

In `validate_device()`, reject `mps` when `torch.backends.mps.is_available()` is false.

- [ ] **Step 8: Run configuration tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py`

Expected: PASS.

### Stage 1.2: carry the value to configuration

### Task 2: Add the public keyword

**Files:**

- Modify: `tsad-lib/src/tsad/api.py:41-54`
- Modify: `tsad-lib/src/tsad/experiment.py:216-228,334-346`
- Modify: `tsad-lib/tests/test_public_runtime.py:151-181`

**Interface:** `run(..., device: str = "cpu") -> Report`

- [ ] **Step 1: Add the public CPU-device test**

Add `test_run_records_explicit_cpu_device()`. Call the existing THESIS fixture with `device="cpu"`. Assert its `resolved_config.yaml` contains `device: cpu`.

- [ ] **Step 2: Run the public CPU-device test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_records_explicit_cpu_device`

Expected: FAIL.

- [ ] **Step 3: Add `device` to `api.run()`**

Add `device="cpu"` to the `run()` signature.

- [ ] **Step 4: Pass `device` to `ExperimentRunner`**

Pass the new `device` argument in the `ExperimentRunner(...)` call inside `api.run()`.

- [ ] **Step 5: Add `device` to `ExperimentRunner.__init__()`**

Add `device="cpu"` to the constructor signature.

- [ ] **Step 6: Store `self.device`**

Assign the constructor device argument to `self.device`.

- [ ] **Step 7: Add device to resolver values**

Add `"device": self.device` to the dictionary passed to `resolve_config()`.

- [ ] **Step 8: Run the public CPU-device test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_records_explicit_cpu_device`

Expected: PASS.

### Stage 1.3: reject unsupported accelerated methods

### Task 3: Keep first-release support honest

**Files:**

- Modify: `tsad-lib/src/tsad/experiment.py:334-377`
- Modify: `tsad-lib/tests/test_public_runtime.py`

**Interface:** A non-THESIS request with non-CPU device returns one failed `Result`.

- [ ] **Step 1: Add the non-THESIS accelerator test**

Add `test_run_rejects_non_thesis_accelerator()`. Mock `resolve_config()` to return a configuration with `device="mps"`. Run `isolation_forest`. Assert the result status is `failed` and its error contains `device=cpu`.

- [ ] **Step 2: Run the non-THESIS accelerator test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_rejects_non_thesis_accelerator`

Expected: FAIL.

- [ ] **Step 3: Add the non-THESIS device guard**

After configuration resolves and before artifact writing, raise `ValueError` when `request.model != "thesis"` and `config.device != "cpu"`.

- [ ] **Step 4: Run the non-THESIS accelerator test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_rejects_non_thesis_accelerator`

Expected: PASS.

## Phase 2: complete native THESIS placement

### Stage 2.1: place offline THESIS

### Task 4: Give THESIS the resolved device

**Files:**

- Modify: `tsad-lib/src/tsad/experiment.py:378-404`
- Modify: `tsad-lib/tests/test_public_runtime.py`

**Interface:** `ThesisRunner(..., device=config.device)`

- [ ] **Step 1: Add the THESIS device-spy test**

Add `test_run_passes_cpu_device_to_thesis_runner()`. Spy on `tsad.experiment.ThesisRunner.__init__`. Run native THESIS with `device="cpu"`. Assert the spy receives `device="cpu"`.

- [ ] **Step 2: Run the THESIS device-spy test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_passes_cpu_device_to_thesis_runner`

Expected: FAIL.

- [ ] **Step 3: Pass `config.device` to THESIS**

Add `device=config.device` to the `ThesisRunner(...)` call.

- [ ] **Step 4: Run the THESIS device-spy test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_run_passes_cpu_device_to_thesis_runner`

Expected: PASS.

### Stage 2.2: place memory initialization

### Task 5: Encode memory windows on the model device

**Files:**

- Modify: `tsad-lib/src/tsad/models/thesis.py:109-140`
- Modify: `tsad-lib/tests/test_thesis_runtime.py`

**Interface:** `MemoryInitializer.fit(training_data, model)` encodes device-resident windows.

- [ ] **Step 1: Add the conditional memory-device test**

Add `test_memory_initialization_uses_available_accelerator()`. Skip when neither MPS nor CUDA is available. Move a small THESIS model to the selected device. Assert `MemoryInitializer.fit()` completes.

- [ ] **Step 2: Run the conditional memory-device test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_thesis_runtime.py::test_memory_initialization_uses_available_accelerator`

Expected: FAIL on an available accelerator, or SKIPPED without one.

- [ ] **Step 3: Derive the model device**

In `MemoryInitializer.fit()`, read the device from the first model parameter.

- [ ] **Step 4: Move memory windows to the model device**

Move the training-window tensor to the derived device before calling `model.encode()`.

- [ ] **Step 5: Run the conditional memory-device test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_thesis_runtime.py::test_memory_initialization_uses_available_accelerator`

Expected: PASS or SKIPPED.

### Stage 2.3: place online execution

### Task 6: Move online windows and NumPy boundaries

**Files:**

- Modify: `tsad-lib/src/tsad/models/thesis_online.py:93-153`
- Modify: `tsad-lib/tests/test_thesis_online.py`

**Interface:** `ThesisOnlineRunner.run()` keeps PyTorch values on `offline_runner.device`.

- [ ] **Step 1: Add the conditional online-accelerator test**

Add `test_a0_runs_on_available_accelerator()`. Skip when neither MPS nor CUDA is available. Build the existing small offline THESIS state on the selected device. Run A0. Assert it returns records.

- [ ] **Step 2: Run the conditional online-accelerator test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_thesis_online.py::test_a0_runs_on_available_accelerator`

Expected: FAIL on an available accelerator, or SKIPPED without one.

- [ ] **Step 3: Move online values to the offline runner device**

Add `device=self.offline_runner.device` to the online `torch.as_tensor()` call.

- [ ] **Step 4: Move original values to CPU before NumPy**

Add `.cpu()` before `.numpy()` for the original online values.

- [ ] **Step 5: Move reconstructed values to CPU before NumPy**

Add `.cpu()` before `.numpy()` for the online reconstruction.

- [ ] **Step 6: Run the conditional online-accelerator test again**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_thesis_online.py::test_a0_runs_on_available_accelerator`

Expected: PASS or SKIPPED.

### Stage 2.4: prove one public accelerator path

### Task 7: Run a small native O2-A0 smoke

**Files:**

- Modify: `tsad-lib/tests/test_public_runtime.py`

**Interface:** A public `run(..., device=selected_device)` call completes for native O2-A0.

- [ ] **Step 1: Add the conditional public accelerator smoke test**

Add `test_thesis_o2_a0_runs_on_available_accelerator()`. Skip when no accelerator exists. Call the existing THESIS fixture with `variant="O2"`, `online_variant="A0"`, and the selected device. Assert completion.

- [ ] **Step 2: Run the public accelerator smoke test**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_public_runtime.py::test_thesis_o2_a0_runs_on_available_accelerator`

Expected: PASS or SKIPPED.

## Phase 3: close the evidence

### Stage 3.1: verify the suite

### Task 8: Run final automated verification

**Files:**

- No source change.

- [ ] **Step 1: Run focused native tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_config.py tests/test_thesis_runtime.py tests/test_thesis_online.py tests/test_public_runtime.py`

Expected: PASS.

- [ ] **Step 2: Run the full sibling suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests`

Expected: PASS.

### Stage 3.2: verify real platforms

### Task 9: Collect real smoke evidence

**Files:**

- No source change.

- [ ] **Step 1: Run the Apple Silicon smoke**

On an Apple Silicon host with MPS available, run the public O2-A0 smoke with `device="mps"`.

- [ ] **Step 2: Inspect the Apple Silicon configuration artifact**

Confirm the completed artifact contains `device: mps`.

- [ ] **Step 3: Run the Docker CUDA smoke**

Inside a container exposing one selected host GPU, run the public O2-A0 smoke with `device="cuda:0"`.

- [ ] **Step 4: Inspect the Docker configuration artifact**

Confirm the completed artifact contains `device: cuda:0`.

## Final verification

- [ ] The full sibling suite passes with `.venv/bin/python`.
- [ ] A malformed device fails before training.
- [ ] A non-THESIS accelerator request returns a failed result.
- [ ] A native THESIS O2-A0 CPU run completes.
- [ ] A native THESIS O2-A0 accelerator run completes where the backend exists.
- [ ] Tutorial work remains separate from this device plan.
