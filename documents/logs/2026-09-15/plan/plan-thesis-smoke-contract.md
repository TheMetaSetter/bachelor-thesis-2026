---
date: 2026-09-15 15:30:00 +07
planner: OpenAI Codex
topic: "Apply one fixed THESIS smoke contract across datasets"
status: implemented
related_research: documents/logs/2026-09-15/research/research-thesis-smoke-contract.md
---

# Plan: One THESIS Smoke Contract

## Summary

The library will have one smoke meaning for every THESIS variant: three Stage A epochs, two Stage B epochs, and 4,096 causal online windows. Non-THESIS methods and wet runs keep their current flows.

## Phase 1: make the boundary executable

**Result:** The runner can identify the one branch that receives the fixed smoke work.

**Tools:** Python, pytest, `tsad-lib/.venv`.

### Stage 1.1: write the decision helper

Write tests before code for all four MECE branches. Add a private pure helper in `experiment.py` that returns the fixed values only for THESIS smoke. Keep generic `RunConfig` defaults unchanged.

### Stage 1.2: connect the helper to the THESIS request

Pass the returned epoch values into the existing resolved THESIS configuration. Keep the existing model selection, API, and CLI interfaces unchanged.

**Complete when:** a THESIS smoke resolved configuration contains 3/2, while the other three branches keep their existing values.

## Phase 2: make online work exact

**Result:** Every THESIS smoke run receives exactly 4,096 causal windows or returns one failed report row.

**Tools:** Python, NumPy, pytest.

### Stage 2.1: select the required point range

Derive 4,115 points from the existing window size and online stride. Use the existing anomaly-aware range selector. Reject a short, unlabeled, or anomaly-free test input.

### Stage 2.2: save and inspect evidence

Write `point_count` and `window_count` to `online_range.json`. Change the SMD smoke gate to require both values.

**Complete when:** the artifact proves 4,115 points and 4,096 windows, and an old 2,048-point artifact fails the gate.

## Phase 3: prove the public story

**Result:** The test suite stays fast, and one real run stress-tests the actual computation.

**Tools:** pytest, Python, local artifacts, CPU or MPS.

### Stage 3.1: run small tests

Use pure unit tests for the decision tree and point formula. Use one mocked public integration test to prove that the real runner passes exactly 4,096 filtered windows into online TTA. Do not make the normal pytest suite run 4,096 neural updates.

### Stage 3.2: run one real THESIS smoke

Run O2-A0 for one labelled `ServerMachineDataset` entity with seed 6. Check actual Stage A and Stage B loops, actual 4,096 online records, matching checkpoint and threshold evidence, and the report.

**Complete when:** the focused suite passes and the real smoke artifact is complete.

## Phase 4: teach the same story

**Result:** Specifications, SMD gate text, and ICCAD notebooks have one unambiguous meaning of “THESIS smoke”.

**Tools:** Markdown, Jupyter JSON, pytest notebook contract check.

### Stage 4.1: update the documents

State the THESIS-only rule in the development specification and matrix note. Preserve the wet 2,048-point policy as a different contract.

### Stage 4.2: update the four notebooks

Replace the 80-point test segment with a real labelled 4,115-point ICCAD segment. Validate its size and anomaly presence before `run()`. Keep the student code short.

**Complete when:** each notebook explains the full THESIS smoke contract and its notebook test still finds the intended practice TODO cells.

## Exclusions

The plan does not add a public profile option. It does not change losses, O2 routing, threshold formulas, metrics, reference models, or the 1,764-cell wet matrix.
