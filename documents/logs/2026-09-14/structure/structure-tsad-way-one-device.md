---
date: 2026-09-14 00:00:00 +07:00
topic: "Minimal Way 1 device parameter for tsad.run()"
status: approved
revision: not recorded; tsad-lib is not a Git repository
related_documents:
  - documents/logs/2026-09-14/research/research-tsad-way-one-device-code-changes.md
  - documents/logs/2026-09-14/plan/plan-tsad-way-one-device.md
---

# Implementation Structure: Minimal Way 1 device parameter for `tsad.run()`

## Summary

The user asked for the whole planning workflow in one request. This structure is approved for detail.

The story has three parts. First, one device string enters and is checked. Second, native THESIS keeps all tensors on that device. Third, tests prove only what the code can honestly claim.

## Phase 1: define one honest request

**Result:** `run(..., device=...)` has one clear meaning.

### Stage 1.1: validate the string

The resolver accepts `cpu`, `mps`, and `cuda:<index>`. It rejects malformed text, unavailable backends, and unavailable CUDA indexes.

### Stage 1.2: carry the value

The API passes the value to `ExperimentRunner`. The runner puts it into `RunConfig`. The existing configuration artifact then records it.

### Stage 1.3: protect unsupported methods

The first release supports acceleration for native `thesis` only. A non-THESIS method with an accelerator request returns a clear failed result. A CPU request remains valid.

**Complete when:** The public API, resolved configuration, and result status tell one device story.

## Phase 2: complete native THESIS placement

**Result:** A native O2-A0 run has no mixed CPU and accelerator tensors.

### Stage 2.1: place the offline runner

The experiment runner gives `config.device` to `ThesisRunner`.

### Stage 2.2: place memory initialization

Memory initialization moves training windows to the THESIS model device before encoding. K-means receives CPU NumPy arrays only after encoding.

### Stage 2.3: place online execution

The online runner creates each causal window on the offline runner device. It uses `.cpu()` before NumPy and scaler work.

### Stage 2.4: prove one accelerator

CPU tests protect the default. A conditional O2-A0 smoke test runs on MPS or CUDA only when that backend exists.

**Complete when:** The smoke run completes and its configuration artifact records `mps` or `cuda:0`.

## Phase 3: close the evidence

**Result:** Later tutorial writing has exact limits and fresh evidence.

### Stage 3.1: run the full suite

The sibling test suite uses `tsad-lib/.venv` and passes.

### Stage 3.2: inspect the artifacts

The CPU and accelerator smoke artifacts contain the selected device. The accelerator artifact exists only when the target backend is available.

### Stage 3.3: preserve the boundary

Later tutorials use native `thesis`. CLI device support and reference-model accelerator support stay separate work.

**Complete when:** Documentation can state only tested native THESIS support.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Existing `RunConfig.device` | Valid runtime request |
| 2 | Valid resolved device | Native accelerator smoke |
| 3 | Completed tests and smoke | Honest later tutorial claims |

## Source correction

Source verification found that `thesis_online.py` creates CPU tensors and calls `.numpy()` directly. The earlier research note marked that file unchanged. This structure follows the source code and includes the required online changes.
