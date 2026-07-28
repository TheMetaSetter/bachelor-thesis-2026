---
date: 2026-07-26
researcher: TheMetaSetter
repository: bachelor-thesis-2026
source_research: documents/logs/07-26-2026/research/research-code-simplification-hotspots.md
status: planning-baseline
---

# Vision: Behavior-Preserving Code Simplification Program

## Purpose

The program will reduce cognitive load in the current multivariate time-series
anomaly-detection codebase without changing scientific behavior, public entry
points, experiment identity, tensor shapes, checkpoint meaning, metric
semantics, or artifact provenance.

The target is not a smaller line count. The target is a reader being able to
follow one runtime concept through one canonical owner, with narrow adapters
only where an old public name or artifact format must remain supported.

## Selected direction

The selected direction is incremental, behavior-preserving simplification. It
uses the existing implementation as the source of truth and proceeds in one
area per phase. Every phase begins with a caller and contract inventory, makes
one reviewable simplification slice, and ends with parity verification.

The program does not select a new encoder, redesign the anomaly detector, change
the online adaptation algorithm, or alter an evaluation protocol. The design
patterns are constraints for clarity, not reasons to introduce new abstractions:

- composition is preferred when removing lifecycle mixins from the thesis model
  can preserve the same public class and state behavior;
- adapters are retained for explicit legacy names and public facades;
- strategies are used only where existing offline/online variants already have
  distinct behavior and a direct dispatch would otherwise duplicate branches;
- the existing registry/factory boundary is simplified only if it removes
  implicit global ownership rather than creating another factory layer.

## Non-negotiable contracts

- The public model entrypoint remains in `src/models/` and owns construction,
  training/inference APIs, configuration, and checkpoint behavior.
- Data batches retain the dictionary contract, `x=[B,L,D]`, point labels, and
  entity metadata. Baseline flattening remains `[B,L*D]`.
- Encoder outputs retain `hidden=[B,L,H]` and the current named output fields.
- Thesis outputs retain reconstruction, classification, point-score, and
  auxiliary payload semantics, including uncertainty and diagnostic payloads.
- Offline execution remains the active Stage A / Stage B flow unless live code
  and tests prove a different path is canonical.
- Online A0/A1/A2 causal ordering, update gates, checkpoint resolution,
  thresholds, and artifact metadata remain unchanged.
- Existing config field names, generated experiment names, output paths,
  checkpoint roles, and report provenance remain stable.
- A simplification is rejected if it requires weakening error handling,
  modifying tests to hide a behavior change, or mechanically splitting files.

## First delivery target

The first delivery contains seven phases:

1. configuration orchestration;
2. configuration validation ownership;
3. thesis model lifecycle ownership;
4. trainer/evaluator metric ownership;
5. canonical online runtime path;
6. compatibility facade boundaries;
7. runtime component registration ownership.

These seven phases form the first executable simplification wave. The remaining
26 phases are planned, but they must wait for the contracts and canonical paths
established by this wave.

## Completion definition

The program is complete only when each phase has a clean, reviewable diff,
focused tests pass without test weakening, the relevant smoke flow passes, and
the corresponding plan/detail record documents the preserved contracts and
remaining uncertainty. A phase may be marked blocked when the active path or
behavioral contract cannot be established from code, tests, configuration, and
artifact evidence.

