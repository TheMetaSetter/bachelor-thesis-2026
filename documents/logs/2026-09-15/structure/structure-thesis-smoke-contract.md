---
date: 2026-09-15 15:30:00 +07
topic: "Apply one fixed THESIS smoke contract across datasets"
status: implemented
related_documents:
  - documents/logs/2026-09-15/research/research-thesis-smoke-contract.md
  - documents/logs/2026-09-15/plan/plan-thesis-smoke-contract.md
---

# Structure: One THESIS Smoke Contract

## Summary

The story has four phases. First the runner identifies THESIS smoke. Next it selects exact online work. Then tests prove the boundary without making every test slow. Last, one real smoke and the student documents record the result.

## Scope

The contract applies when `model == "thesis"` and `experiment_type == "benchmark_smoke"`. It applies to all datasets and all THESIS O/A variants. It does not apply to reference methods, Isolation Forest, or `benchmark` wet runs.

## Phase 1: resolve the fixed smoke values

**Result:** One private rule selects 3 Stage A epochs and 2 Stage B epochs only for THESIS smoke.

### Stage 1.1: protect the four decision leaves

Test THESIS smoke, THESIS wet, non-THESIS smoke, and non-THESIS wet before adding code.

### Stage 1.2: reuse the existing configuration path

Use the selected values while building the existing `RunConfig`. Do not add a new field, public argument, or CLI option.

**Verification:** focused unit tests show only THESIS smoke changes.

## Phase 2: select exact causal work

**Result:** THESIS smoke sees 4,096 windows from one labelled 4,115-point range.

### Stage 2.1: derive the point count

Calculate the point count from window size, stride, and the fixed window count. Reuse `select_online_range()`.

### Stage 2.2: fail closed and write the range story

Reject insufficient, unlabeled, or anomaly-free tests. Save both counts. Require both counts in the SMD smoke gate.

**Verification:** unit tests prove arithmetic and failure cases.

## Phase 3: verify without a large permanent test cost

**Result:** One integration test proves wiring, while one real smoke proves computation.

### Stage 3.1: use a recording online runner

Run public `run()` over one 4,115-point fixture. Replace only `ThesisOnlineRunner` with a fake that records its input length.

### Stage 3.2: run the actual path once

Run an O2-A0 `ServerMachineDataset` smoke. Read its artifacts and verify all requested counts and identities.

**Verification:** integration test passes quickly; the real artifact records actual work.

## Phase 4: make documents and notebooks agree

**Result:** A student and the matrix preflight read the same smoke definition.

### Stage 4.1: update SSOT documents

Describe the THESIS-only 3/2/4096 rule. Preserve the separate 2,048-point wet contract.

### Stage 4.2: update ICCAD tutorials

Build the teaching input from a labelled real segment with 4,115 test points. Keep only the existing full and practice notebook pair differences.

**Verification:** document-contract and notebook tests pass.

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| 1 | Current `ExperimentRunner` branch | Correct fixed configuration |
| 2 | Phase 1 | Exact online artifact |
| 3 | Phases 1–2 | Fast test evidence and real evidence |
| 4 | Phase 3 artifact | Accurate stories and tutorials |
