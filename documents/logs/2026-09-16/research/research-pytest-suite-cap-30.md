---
date: 09-15-2026
researcher: OpenAI Codex
topic: "Reduce the repository pytest suite to at most 30 collected test cases"
status: complete
revision: 61cb84bb450624031f6a27fde86262fd180565e9
branch: dev
---

# Research: Pytest Suite Cap

## Summary

The repository currently collects 615 pytest nodes. The safest reduction is **boundary-contract folding**: keep one test for each distinct risk boundary, and loop inside that test only when several members share exactly the same contract. This strategy can meet the at-most-30 requirement without removing the required tensor, training, checkpoint, loader, synthetic-injection, visualization, and configuration evidence.

## Research question

How can the repository keep no more than 30 collected pytest cases while retaining meaningful evidence for the research runtime?

## Confirmed context

- `pytest.ini:1-8` defines `tests` as the test root and configures one pytest collection path.
- `pytest --collect-only -q` currently reports **615 tests collected**.
- `codebase_preferences.md:142-152` requires focused Pytest checks for tensor shapes, one forward and backward pass, checkpoint save/load, data-loader batches, synthetic anomaly injection with visualization, and YAML-driven configuration.
- `codebase_preferences.md:52-56` identifies `documents/` as the source of design and workflow guidance.
- `documents/abstract-design-notes/design_starter.md:34-41` defines the relevant runtime boundaries as configuration, data, model, and engine layers.
- Existing tests span data, models, evaluation, runtime, online behavior, benchmarks, operations, visualization, demo behavior, and compliance. This breadth means deleting whole domains would create more risk than folding repeated examples.

## Six distinct strategies

| Strategy | Reduction mechanism | Main benefit | Main cost |
| --- | --- | --- | --- |
| 1. Boundary-contract folding | Replace repeated cases with one named test per shared contract; loop over contract members inside it. | Preserves broad behavior evidence while reducing collection nodes. | Folded failures need clear member names. |
| 2. Representative sampling | Keep one representative model, dataset, adapter, and configuration from each family. | Produces the smallest and fastest suite. | A non-representative family member can regress unnoticed. |
| 3. Vertical-slice prioritization | Retain only a few complete paths from input to report or checkpoint. | Protects user-visible workflows. | Local failures become harder to diagnose, and reusable primitives lose direct evidence. |
| 4. Core-plus-extended suites | Keep a default suite below 30 and move other tests to an opt-in suite. | Improves default feedback time. | The repository still contains more than 30 tests, so it does not satisfy a whole-suite cap. |
| 5. Static-check extraction | Move documentation, configuration, notebook, and catalog checks to scripts or CI checks. | Removes non-runtime cases from pytest. | Adds a second verification mechanism and can hide regressions from the default test command. |
| 6. Risk-ranked deletion | Delete tests judged low risk or redundant without replacing every one. | Requires the least restructuring. | Removes evidence rather than removing duplicated collection nodes. |

## MECE decision tree

```text
Does the check protect a distinct risk boundary?
├─ No: remove it as duplicate evidence.                         Strategy 6
└─ Yes
   ├─ Do several members share the same input, output, and failure contract?
   │  └─ Yes: fold them into one named contract test.             Strategy 1
   ├─ Is the check a complete user-visible path?
   │  └─ Yes: retain one vertical-slice test.                     Strategy 3
   ├─ Is the check static content rather than runtime behavior?
   │  └─ Yes: extract it into the chosen release check.           Strategy 5
   └─ Is the requirement a speed preference rather than a cap?
      ├─ Yes: use a core-plus-extended suite.                    Strategy 4
      └─ No: use representatives only when coverage loss is accepted.
                                                                    Strategy 2
```

The first split separates duplicate evidence from unique risk boundaries. The remaining branches classify each unique boundary by its testing shape or by whether the requirement is really about speed. The selected strategy is therefore not a mixture of incompatible reduction rules.

## Eisenhower matrix

|  | Urgent | Not urgent |
| --- | --- | --- |
| Important | Fold duplicated model, data, evaluation, checkpoint, online, and configuration checks now. | Keep broader family diagnostics available in version control history or a later review, but do not include them in the capped suite. |
| Not important | Do not add a second runner or a new static-check command merely to lower the pytest number. | Do not preserve one test node per parameter value or delete evidence without a replacement contract. |

## Decision

Select **Strategy 1: boundary-contract folding**. It directly satisfies the whole-suite cap, keeps the existing pytest workflow, and preserves the repository’s required coverage categories. Use a target of **28 collected nodes**, leaving two nodes of safety under the hard cap.

## Proposed boundary budget

| Boundary group | Target nodes | Required evidence |
| --- | ---: | --- |
| Configuration and data | 5 | Config loading, loader shapes and batch size, data contracts, anomaly injection, visualization artifact |
| Models and baselines | 5 | Output shapes, one forward/backward pass, baseline/reference selection, model-family contract |
| Evaluation and metrics | 4 | Score transformations, threshold identity, metric behavior, unavailable or malformed inputs |
| Engine and checkpoints | 4 | Training loop, checkpoint round trip, scheduler or fallback behavior, artifact provenance |
| Online runtime | 5 | Causal stream, adaptation isolation, verification buffer, online state round trip, threshold contract |
| Public and release paths | 5 | Benchmark smoke path, CLI or script routing, output structure, documentation/compliance, final preflight |
| **Total** | **28** | **At most 30 collected pytest nodes** |

The exact old-to-new test mapping must be verified before deletion. A folded test must include the member name in assertion output and must not use `pytest.mark.parametrize` when parameterization would increase collected nodes.

## Limits and uncertainties

- The collection count is confirmed, but the final 28-node implementation is not yet present.
- This proposal reduces pytest nodes, not the number of inputs checked inside a contract test.
- The repository contains existing untracked planning documents from other work. This research does not modify or remove them.
- The available evidence does not establish which individual existing tests are safe to delete until each replacement contract passes.
