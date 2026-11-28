---
date: 2026-11-28T12:46:25+07:00
topic: "Remove limits.max_epochs_per_stage"
status: complete
---

# Plan: Remove `limits.max_epochs_per_stage`

## Summary

This plan removes the `limits.max_epochs_per_stage` configuration key and its runtime epoch-capping logic from validation, training, progress reporting, smoke planning, and tests. After this change, every training stage runs for the `epochs` value defined in `method.training.stages`.

## Blocking Question

- **Smoke-mode epoch cap (`src/experiments.py`):** Currently, `--smoke` sets `limits.max_epochs_per_stage = 1` in `_cell_config` (`src/experiments.py:270`).
  - **Option A (Recommended):** Set `stage["epochs"] = 1` on `resolved["method"]["training"]["stages"]` when `smoke=True` after `resolve_run(...)`.
  - **Option B:** Remove the smoke epoch cap completely so `--smoke` runs the full stage epoch counts from the method profile.

## Ordered Checklist Plan

### Phase 1: Confirm Smoke-Mode Decision and Remove Schema Key
Complete and check Phase 1 before starting Phase 2.

#### Stage 1.1: Resolve smoke-mode epoch behavior
1. [x] Ask the user to choose Option A or Option B for `--smoke` epoch capping. **Tool:** `ask_question`

#### Stage 1.2: Remove `max_epochs_per_stage` from cell validation
2. [ ] Remove `"max_epochs_per_stage"` from `counts` in `_validate_cell` inside `src/config.py` (`src/config.py:150-152`). **Tool:** `edit_file`
3. [ ] Run `tests/test_config.py` to check config validation behavior. **Tool:** `run_command`

---

### Phase 2: Remove Runtime Epoch Capping
Complete and check Phase 2 before starting Phase 3.

#### Stage 2.1: Remove `max_epochs_per_stage` from neural stage execution
4. [ ] Remove `max_epochs = method.config.get("limits", {}).get("max_epochs_per_stage")` from `run_neural_stages` in `src/method_runtime.py` (`src/method_runtime.py:229`). **Tool:** `edit_file`
5. [ ] Replace the `min(definition["epochs"], max_epochs)` branch with `limit = definition["epochs"]` in `src/method_runtime.py` (`src/method_runtime.py:235-239`). **Tool:** `edit_file`

#### Stage 2.2: Remove `max_epochs_per_stage` from offline progress reporting
6. [ ] Remove `total = config.get("limits", {}).get("max_epochs_per_stage", epochs)` and the `min(total, epochs)` check in `_emit_epoch_progress` inside `src/offline.py` (`src/offline.py:321-323`), and pass `total=epochs` to `progress(...)` (`src/offline.py:328`). **Tool:** `edit_file`

#### Stage 2.3: Update smoke planning in `src/experiments.py`
7. [ ] Remove `"max_epochs_per_stage": 1` from `caps` in `_cell_config` inside `src/experiments.py` (`src/experiments.py:270`). **Tool:** `edit_file`
8. [ ] If Option A is selected, set `stage["epochs"] = 1` on `resolved["method"]["training"]["stages"]` when `smoke=True` in `_cell_config` (`src/experiments.py:280`). **Tool:** `edit_file`

---

### Phase 3: Update Affected Tests
Complete and check Phase 3 before starting Phase 4.

#### Stage 3.1: Update `tests/test_config.py`
9. [ ] Replace `resolved["limits"]["max_epochs_per_stage"] = 1` with `resolved["limits"]["max_train_windows"] = 1` in `test_resolution_is_read_only` (`tests/test_config.py:34`). **Tool:** `edit_file`
10. [ ] Add a test in `tests/test_config.py` asserting that `limits: {"max_epochs_per_stage": 1}` raises `ValueError`. **Tool:** `edit_file`

#### Stage 3.2: Update `tests/test_mustard.py`
11. [x] Replace `cfg["limits"] = {"max_epochs_per_stage": 1}` with setting `stage["epochs"] = 1` on `cfg["method"]["training"]["stages"]` (`tests/test_mustard.py:173`). **Tool:** `edit_file`

#### Stage 3.3: Update `tests/test_offline.py`
12. [ ] Replace `limits={"max_epochs_per_stage": 2}` with setting `stage["epochs"] = 2` on `method_config["training"]["stages"]` (`tests/test_offline.py:374`). **Tool:** `edit_file`

#### Stage 3.4: Update `tests/test_online.py`
13. [ ] Remove `"max_epochs_per_stage": 1` from `cell["limits"]` (`tests/test_online.py:295`) and set `stage["epochs"] = 1` on the resolved `config["method"]["training"]["stages"]` (`tests/test_online.py:301`). **Tool:** `edit_file`

---

### Phase 4: Verify Removal and Review Diff
Complete all checks before reporting completion.

#### Stage 4.1: Search for remaining references
14. [ ] Run `rg -n "max_epochs_per_stage" src/ tests/ configs/ scripts/ README.md` to confirm zero active references remain. **Tool:** `run_command`

#### Stage 4.2: Run test suite and CLI validation
15. [ ] Run `.venv/bin/pytest -v` to verify all tests pass. **Tool:** `run_command`
16. [ ] Run `.venv/bin/tsad validate experiment --name smd_comparison` and `.venv/bin/tsad validate experiment --name smd_mustard_10ep`. **Tool:** `run_command`

#### Stage 4.3: Review diff
17. [ ] Run `git diff` to inspect all modified files against project rules. **Tool:** `run_command`
