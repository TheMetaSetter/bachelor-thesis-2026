---
date: 2026-09-15
researcher: OpenAI Codex
topic: "Reduce the tsad-lib pytest suite to at most 30 collected cases"
status: complete
---

# Research: A Small TSAD Test Suite

## Summary

`tsad-lib` currently collects 119 pytest nodes from 97 test functions in 27 files. The main source of extra nodes is repetition. One reference-model test becomes 15 nodes, tutorial tests become four nodes, and several adapter tests repeat the same inspection or loading contract.

The selected strategy keeps one test per risk boundary. A test may loop through the members of one boundary. The new suite will collect 28 pytest nodes. It will still exercise all 15 ready reference anomaly paths and all current dataset adapter families.

The cap means **at most 30 collected pytest nodes**. It does not mean the suite may check only 30 input examples. A single contract test may check several inputs when they share one rule.

## Question

How can `tsad-lib` reduce its pytest suite to at most 30 cases while keeping meaningful TSAD evidence?

## Current story

`pytest --collect-only` reports 119 nodes. The suite has 97 functions. Parameterization adds 22 more nodes.

The clearest example is `tests/test_reference_models.py`. One function uses 15 model names, so pytest makes 15 cases. The function already checks the same contract for every model: construct, reconstruct, fit, and score. The test should remain one collected contract test and loop through the 15 ready models itself.

The adapter tests tell a similar story. `tests/test_adapter_inspection.py`, `test_array_text_adapters.py`, and `test_table_archive_mat_adapters.py` split one family boundary into 22 small examples. The library still needs every adapter family checked. It does not need one pytest node per adapter action.

The public runtime and THESIS tests contain important different contracts. They should not become one large test. A failure in O2 offline training, A1 update isolation, A2 contrastive adaptation, artifact resume, or CLI routing should still identify its boundary.

## The six distinct strategies

| Strategy | Story | Strength | Cost |
| --- | --- | --- | --- |
| 1. Boundary-contract folding | One test loops through all members that share one rule. | Keeps broad evidence with few nodes. | A loop failure needs a model or adapter name in its assertion message. |
| 2. Family representatives | Test one model or adapter from each family. | Fastest small suite. | A non-representative member can break unnoticed. |
| 3. End-to-end replacement | Keep only notebook, CLI, and full THESIS runs. | Tests user-visible paths. | Root causes become hard to locate. |
| 4. Core and extended tiers | Keep a core suite below 30 and move the rest to an opt-in suite. | Fast default feedback. | The whole repository still has more than 30 tests. |
| 5. Static preflight commands | Move document, notebook, and catalog checks outside pytest. | Reduces pytest count. | Creates a second verification command to remember. |
| 6. Risk-ranked deletion | Delete low-risk implementation-detail tests. | Least code to write. | Removes evidence instead of removing repetition. |

## MECE decision tree

```text
Does this check protect a unique risk boundary?
├─ No: delete the duplicate example.                         Strategy 6
└─ Yes
   ├─ Do many members share exactly one contract?
   │  └─ Yes: loop inside one named contract test.           Strategy 1
   ├─ Is the contract only about one public end-to-end path?
   │  └─ Yes: keep one integration test.                     Strategy 3
   ├─ Is it static content rather than runtime behavior?
   │  └─ Yes: fold it into one release-contract test.        Strategy 1
   └─ Is fast feedback more important than total-suite size?
      └─ Yes: split a core and extended tier.                Strategy 4
```

The branches do not overlap. Each existing check is first classified as duplicate or unique. A unique check is then classified by its boundary.

## Eisenhower choice

| | Urgent | Not urgent |
| --- | --- | --- |
| Important | Fold model, adapter, runtime, and artifact contracts now. | Keep optional family sampling only as a future diagnostic tool. |
| Not important | Do not create a second static-check command. | Do not keep duplicate example tests or an extended suite. |

Strategy 1 is selected. It removes repetition without creating a new tool, a hidden test tier, or a gap in ready-model coverage.

## Target suite

The proposed suite has five files and 28 collected tests.

| File | Collected tests | Boundary |
| --- | ---: | --- |
| `tests/test_data_contracts.py` | 5 | types, discovery, adapters, pipeline, scores and metrics |
| `tests/test_baseline_contracts.py` | 4 | catalog, smoke map, Isolation Forest, reference models |
| `tests/test_thesis_contracts.py` | 8 | O2, losses, offline stages, routing override, A0, A1, A2, resume state |
| `tests/test_public_contracts.py` | 6 | notebook API, baseline, reference, THESIS smoke, device, CLI |
| `tests/test_release_contracts.py` | 5 | artifacts, resume, SMD preflight, documents, tutorials |
| **Total** | **28** | **all retained boundaries** |

## Evidence

- `tsad-lib/tests/test_reference_models.py:11-52` — one contract is parameterized over 15 ready reference paths.
- `tsad-lib/tests/test_adapter_inspection.py:20-97` — twelve small inspection and ambiguity examples cover one adapter boundary.
- `tsad-lib/tests/test_public_runtime.py:176-410` — baseline, reference, THESIS, device, and artifact behaviors are separate public boundaries.
- `tsad-lib/tests/test_thesis_runtime.py:63-180` — O2 stage, routing, memory, and threshold contracts exist today.
- `tsad-lib/tests/test_thesis_online.py:57-129` — A0, A1, A2, and state restoration have different update rules.
- `tsad-lib/tests/test_matrix.py:21-108` — matrix counts, selected online range, and preflight gate behavior are one release boundary.
- `tsad-lib/pyproject.toml:1-23` — pytest is the existing development test dependency; no second test runner is configured.

## Limits

This proposal changes test organization and collection count. It does not reduce the number of ready reference models that execute inside the folded test. It does not certify the 25 catalog models whose readiness is not `ready`. It does not change TSAD runtime code or scientific contracts.
