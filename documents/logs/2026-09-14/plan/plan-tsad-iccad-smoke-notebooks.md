---
date: 2026-09-14 00:00:00 +07:00
planner: OpenAI Codex
topic: "Minimal ICCAD smoke notebooks for tsad-lib"
status: ready
related_research:
  - documents/logs/2026-09-14/research/research-tsad-iccad-smoke-notebook-feasibility.md
  - documents/logs/2026-09-14/research/research-tsad-way-one-device-code-changes.md
related_plan:
  - documents/logs/2026-09-14/plan/plan-tsad-way-one-device.md
---

# Implementation Plan: Minimal ICCAD smoke notebooks

## Summary

Four small notebooks will teach one story twice: once for Apple Silicon and once for Docker CUDA. Each story has a full version and a practice version. The practice notebook changes only a few short code snippets into `# TODO`.

The notebooks use one small, local ICCAD-derived dataset. They do not add a new library API or run the full ICCAD dataset.

## Phase 1: make one tiny ICCAD teaching dataset

**Goal:** Each full notebook can create a reproducible 80-point train and 80-point test dataset in its own workspace.

**Changes:** Read one fixed parquet column and timestamps. Remove missing rows. Map source anomaly windows to known-incident labels. Start after `a7` ends. The first 80 points are outside known incidents, but they are not proven normal and may contain unlabeled anomalies. Put the next 80 points in test, where event `a8` spans four points. The user accepts this teaching rule.

**Complete when:** `run()` discovers `ICCAD/iccad-mini` without changes to `tsad-lib` runtime code.

## Phase 2: write one clear Apple Silicon story

**Goal:** A student can start with `device="mps"`, prepare the tiny data, run native THESIS O2-A0, and read the final table.

**Changes:** Write the full notebook in short cells. Copy it into a practice notebook. Replace four short student actions with `# TODO`.

**Complete when:** The full notebook runs on an MPS host. The practice notebook has the same cell order and explanations.

## Phase 3: write one clear Docker CUDA story

**Goal:** A student can expose one host GPU to Docker, confirm it appears as `cuda:0`, run the same tiny data flow, and read the table.

**Changes:** Write the full notebook with a short Docker launch cell and a CUDA check cell. Copy it into a practice notebook. Replace the same four student actions with `# TODO`.

**Complete when:** The full notebook runs in a container with one selected host GPU. It does not name a physical GPU model or a host GPU index inside `run()`.

## Phase 4: verify only the teaching claim

**Goal:** The notebooks are readable, runnable, and scientifically honest.

**Changes:** Execute each full notebook in its target environment. Check that practice notebooks contain four small TODO snippets and no hidden answer. State that labels are known-incident labels shared by channels and that the run is not a benchmark.

**Complete when:** Each full notebook produces a report table. Each practice notebook stops at its TODOs. The stored configuration shows `mps` or `cuda:0`.

## Scope boundary

The notebooks depend on the completed minimal device plan. They use native `thesis`, O2, A0, window size 20, one epoch defaults, and `benchmark_smoke` outputs.

They exclude reference models, full ICCAD, per-channel root-cause labels, paper-replication claims, and performance conclusions. Candidate train contamination is an accepted teaching constraint.
