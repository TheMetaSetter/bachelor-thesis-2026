---
date: 2026-11-03
researcher: OpenAI Codex
topic: "Meaning and kinds of protocol in the two codebases"
status: complete
revision: thesis 3041e44b; tsad-lib 20cfa32
---

# Research: Meaning and kinds of protocol

## Summary

The shared next-version definition of **protocol** is still open. The current evidence supports this working explanation: a protocol is a named set of rules for how a run handles data, scoring, thresholds, or online updates. These are overlapping scopes, not an agreed list of protocol types.

The two codebases use the word differently. The thesis configuration records concrete rules. In `tsad-lib`, a run request accepts a protocol label and stores it in run identity and reports. The inspected run path does not show those labels selecting different data or scoring behavior.

## Research question

What does “protocol” mean in the next version of the two codebases, and what kinds of protocol appear in their current files?

## Findings

### Thesis project

The next-version terminology note makes a run include a protocol, but the shared definitions file explicitly leaves the protocol boundary, required contents, and owner unresolved. Therefore, there is no confirmed shared definition yet.

The current thesis configuration shows several rule groups that people call protocol:

- **Data and stream rules:** dataset, entity, window size, strides, and selected test-stream range.
- **Score and threshold rules:** score space and transform, threshold source and quantile, and score smoothing.
- **Evaluation rules:** test labels are used for metrics only; point adjustment is disabled.
- **Method update rules:** each method keeps its own scoring and update behavior, while the experiment shares data and measurement rules.

These groups can form one experiment setup, but the files do not establish a canonical protocol object or say that these are mutually exclusive types. The protocol YAML contains concrete values; the thesis contract describes which settings are shared and which remain method-specific.

### `tsad-lib`

`RunRequest` accepts three protocol labels: `strict_full_test`, `smd_legacy_2048`, and `paper_batch`. It rejects `paper_batch` for online runs. It also stores a version, score rule, reference policy, and optional digest. The resolved identity includes the protocol label, version, and digest, and result rows report the label and digest.

In the inspected runtime sources, the label is validated and recorded. A project-wide search did not find a path where these three labels select separate data or scoring procedures. So their names alone do not establish what each procedure does.

The word also appears in Python typing. `DatasetAdapter`, `MethodRunner`, and `Method` inherit from `typing.Protocol`; these are structural interface contracts for objects. They are unrelated to experiment procedure rules.

## Working explanation

For research runs, I understand a protocol as the named, recorded rules that constrain how a run is performed and evaluated. The rules may cover data selection and order, scoring, threshold calibration, measurement, or method updates. A run may use several such rule groups together.

This is an evidence-based explanation, not an approved definition for the next versions. The shared terminology note correctly keeps the term open until its boundary and ownership are agreed.

## Kinds seen in the files

| Scope | What it controls | Evidence status |
|---|---|---|
| Run or evaluation | Test extent and evaluation procedure | Labels exist in `tsad-lib`; their full behavior is not established by the inspected runtime path. |
| Data and stream | Splits, windows, strides, and stream range | Explicit values appear in thesis configuration and specifications. |
| Scoring and calibration | Score space, transform, threshold source, quantile, and smoothing | Explicit values appear in thesis configuration and validation code. |
| Method update | Whether and how a method changes state online | Thesis specifications say this is method-specific. It is not the same as the shared data and measurement rules. |
| Python interface | Required methods an object must provide | Implemented with `typing.Protocol`; not an experiment protocol. |

These are useful scopes for describing current usage, not a confirmed protocol taxonomy. They overlap: a run-level protocol can combine data, scoring, calibration, evaluation, and update rules.

## Evidence

- `documents/specs/definitions.md:34-36` — says the shared protocol definition and ownership remain unresolved.
- `documents/spec/experiment-terminology-next-version.md:7-16` — puts a protocol in run identity and distinguishes execution phase from adaptation behavior.
- `documents/spec/online_tta_terminology_ontology.md:297-312` — describes score selection through a named score rule and threshold artifact.
- `configs/protocol/smd_window20_cleanval_q99_ewma09.yaml:1-18` — records score space, windowing, calibration splits and quantiles, smoothing, test-label use, and point-adjustment rule.
- `src/tsad/request.py:20-38,50-81,170-205` — defines accepted labels, validates them, and places protocol identity fields in resolved run identity.
- `src/tsad/report_objects.py:45-72` — writes the label and digest to result rows.
- `src/tsad/data/adapters.py:1-14` and `src/tsad/models/base.py:1-35` — use Python `Protocol` for structural interfaces.

## Uncertainty

The evidence does not confirm one shared definition or official list of protocol types across the next versions. The current `tsad-lib` runtime also does not establish the exact execution meaning of each accepted protocol label. Those meanings should remain open until their behavior and ownership are specified and verified.
