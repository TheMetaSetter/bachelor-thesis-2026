---
date: 2026-09-14 00:00:00 +07:00
planner: OpenAI Codex
topic: "Minimal Way 1 device parameter for tsad.run()"
status: ready
revision: not recorded; tsad-lib is not a Git repository
related_research:
  - documents/logs/2026-09-14/research/research-tsad-way-one-device-code-changes.md
  - documents/logs/2026-09-14/research/research-tsad-run-device-options.md
---

# Implementation Plan: Minimal Way 1 device parameter for `tsad.run()`

## Summary

This plan adds one optional keyword: `device`. The first supported accelerated path is native `thesis`. A student can write `device="mps"` on Apple Silicon or `device="cuda:0"` inside Docker. CPU stays the default.

The plan is intentionally small. It does not add automatic selection, a new runtime class, multi-GPU execution, CLI changes, or reference-model accelerator support.

## Current state

`RunConfig` already stores `device="cpu"`. The public API does not receive it. The experiment runner does not pass it to THESIS. Memory initialization and online THESIS execution also create CPU tensors, so an accelerated end-to-end THESIS run would fail today.

## Desired end state

The following call completes on an available accelerator. Its `resolved_config.yaml` records the same device text.

```python
run(..., models=["thesis"], device="mps")
```

The same call with a malformed or unavailable device stops before training. Non-THESIS methods remain CPU-only in this release and return a clear failed result when given an accelerator device.

## Scope

### In scope

- `device` in `run()`.
- Device validation.
- Native THESIS offline, memory, and online placement.
- CPU tests and one conditional accelerator smoke test.

### Out of scope

- CLI `--device`.
- Reference-model accelerator support.
- FiLM and MICN changes.
- ICCAD split preparation and tutorials.
- Distributed or multi-GPU training.

## Phase 1: define one honest public contract

**Goal:** `run()` accepts one validated device value and records it in the resolved configuration.

**Changes:** Add the keyword in `api.py`, store it in `ExperimentRunner`, validate it in `config.py`, and reject an accelerator choice for non-THESIS methods.

**Verification:** Focused configuration and public-runtime tests pass with `tsad-lib/.venv`.

**Complete when:** CPU remains compatible, bad choices fail early, and the CPU artifact says `device: cpu`.

## Phase 2: keep all native THESIS tensors together

**Goal:** Native THESIS uses `config.device` from Stage A to online A0 scoring.

**Changes:** Pass `config.device` to `ThesisRunner`. Move memory windows and online windows to that device. Convert tensors to CPU only before NumPy and scaler calls.

**Verification:** Existing CPU tests pass. A small conditional O2-A0 test passes on MPS or CUDA when available.

**Complete when:** The accelerator smoke has no mixed-device error and writes its selected device to `resolved_config.yaml`.

## Phase 3: keep evidence ready for the later tutorials

**Goal:** The implementation has small, current evidence before Apple Silicon and Docker tutorial writing begins.

**Changes:** Run the sibling suite. Inspect CPU and accelerator artifacts. Record that the later tutorial must use native `thesis` only.

**Verification:** The full sibling suite passes. One real accelerator smoke completes when that accelerator exists.

**Complete when:** Tests, configuration artifacts, and the later tutorial scope agree.

## Risk and recovery

Mixed-device tensors are the main risk. The accelerator smoke detects them. If a non-THESIS method needs an accelerator later, create a separate plan for that method instead of broadening this path.

No migration is needed. Existing calls use the new default `device="cpu"`.

## Final verification

From the `tsad-lib` directory:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests
```

The command must pass. A small O2-A0 THESIS run must also pass on one available accelerator before the tutorials claim accelerator support.
