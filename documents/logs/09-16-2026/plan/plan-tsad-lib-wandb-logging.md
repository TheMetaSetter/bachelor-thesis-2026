---
date: 2026-09-16 Asia/Ho_Chi_Minh
planner: OpenAI Codex
topic: "W&B logging for tsad-lib"
status: ready
revision: 9aa4c35
related_research: documents/logs/09-16-2026/research/research-tsad-lib-wandb-logging.md
---

# Implementation Plan: W&B logging for tsad-lib

## Summary

Add one run-level W&B publisher. It records the resolved configuration, final metrics, and existing completed run directory. It does not change training or metric logic.

## Chosen approach

Use one proposed `wandb_logging.py` function. `ExperimentRunner.run_request()` calls it after all local artifacts are present and before `completion.json`. A W&B publishing failure returns a failed result and leaves no completion marker.

## Scope

In scope: W&B dependency, one publisher, final metric logging, run-directory artifact upload, two tests.

Out of scope: epoch-level logging, a W&B CLI option, offline mode policy, a second logging backend, changing local artifacts, or uploading data sources.

## Phase 1: Prove the publisher contract before production code

**Outcome:** The tests define the exact W&B side effects without a network call.

**Tools:** `pytest`, fake W&B module, existing `tmp_path` fixture.

**Verification:** A unit test expects one initialized run, one metric log, one `experiment` artifact containing `run_dir`, and one finish. An integration test expects `api.run()` to publish only after the normal local artifacts exist.

## Phase 2: Add the minimal publisher and integration seam

**Outcome:** Each completed experiment records one W&B run and one artifact version.

**Tools:** Python, `wandb`, existing `ExperimentRunner`.

**Verification:** The two new tests pass. Existing public and release contracts stay green.

## Phase 3: Verify the real smoke path deliberately

**Outcome:** A W&B-authenticated smoke run has a completed local artifact directory and a corresponding W&B run.

**Tools:** existing `tsad` CLI, W&B account credentials, one small SMD dataset.

**Verification:** Run one smoke combination only. Confirm the W&B page shows config, final metrics, and the `experiment` artifact. Do not launch the matrix.

## Risks

- Missing W&B credentials cause publication to fail. The local run stays resumable but lacks `completion.json`.
- Uploading `run_dir` mirrors the existing local artifact set. This plan adds no raw tensors or new retention.
- The `Result` remains `failed` when publication fails. This keeps the documented reproducibility requirement strict.
