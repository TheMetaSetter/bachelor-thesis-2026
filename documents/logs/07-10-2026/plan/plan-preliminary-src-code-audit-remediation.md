---
date: 2026-07-10T03:18:01+0700
researcher: Codex
git_commit: 6dc99c0dd296e96bb28f563d38e00d13a0da94f8
branch: dev
repository: bachelor-thesis-2026
topic: "Preliminary programming plan for src/ audit remediation"
tags: [plan, source-audit, readability, refactor]
status: draft
last_updated: 2026-07-10
last_updated_by: Codex
---

# Plan: preliminary `src/` audit remediation

## Basis

This plan is based on
`research-src-code-audit-target-inventory.md`. It applies only after each
target's existing test contract is characterized.

## Phase 1 — Lock audit gates

1. Keep `pytest.ini` restricted to the repository-owned `tests/` tree.
2. Add an AST test that reports all `src/**/*.py` files above 500 lines and all
   functions/methods above 50 lines. Enable it as a required gate only after
   the current inventory reaches zero.
3. For every Tier 1 file, identify its public imports, YAML keys, output
   dictionaries, state-dict keys, and existing test files before editing.

## Phase 2 — Audit runtime owners before moving code

1. Audit config loading and validation together. Split by configuration
   section only after tests lock alias normalization and exact failure messages.
2. Audit `Trainer.train` as a sequence of epoch setup, batch optimization,
   validation, metric aggregation, and checkpoint/artifact decisions. Each
   extracted helper receives explicit inputs and returns its local result; the
   `Trainer` remains the public owner.
3. Audit synthetic augmentation by separating deterministic sampling, family
   transformation, metadata creation, and batch assembly. Preserve generator
   state, anomaly masks, labels, and class balancing.
4. Audit online-TTA calibration, per-window processing, sequence execution,
   and report finalization independently; do not merge this with `OnlineLoop`.

## Phase 3 — Audit model boundaries

1. Retain one registry-facing entrypoint for each model. Replace thesis
   lifecycle mixins only with explicitly named collaborators or immutable
   configuration objects; do not create a second public model API.
2. Preserve constructor kwargs, `forward`, stage-step output dictionaries,
   parameter names, and checkpoint restoration for thesis, RedLamp, and online
   adaptation models.
3. Keep reusable MLP/CNN blocks model-independent. Audit every remaining
   import so RedLamp and online code do not import model-specific thesis
   implementation helpers.

## Phase 4 — Audit data and metrics

1. Preserve parser → scaler → windowing order and the half-open overlapping
   window contract while splitting SMD/window helpers.
2. Preserve evaluator overlap reconstruction before separating payload
   accumulation from metric calculation.
3. Split pointwise range/VUS helpers only behind the existing public metric
   functions and unchanged metric names.

## Verification

- Run focused characterization tests before and after each target file.
- Run `pytest -q` after every completed runtime/model group.
- Finish by enabling the AST size gate and requiring zero violations.
- Run config-load, model forward/backward, checkpoint roundtrip, synthetic
  injection, evaluator reconstruction, and online state tests as the final
  acceptance matrix.

## Risks

- Moving model modules can alter state-dict key paths; compare saved keys
  before and after each model refactor.
- Splitting config validation can alter error order/message; assert current
  failing cases.
- Splitting augmentation can accidentally share or reset random generators;
  preserve fixed-seed outputs.
