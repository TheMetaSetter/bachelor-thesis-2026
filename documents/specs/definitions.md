# Unified definitions

Status: agreed terms for the next version of `tsad-lib` and `bachelor-thesis-2026`.

These definitions guide naming and design. They do not claim that every next-version structure is implemented in the current code. For current behavior, follow the owning code and mark any difference from this note.

## Core terms

- **Experiment:** one planned comparison made of multiple runs. A request for one execution is a standalone run.
- **Run:** one execution with a fixed dataset, entity, method, seed, phase, protocol, and configuration. An online run records the offline run it depends on when the method requires one.
- **Cell:** one resolved item in an experiment matrix. It fixes the settings for one run. Use *run* when describing the execution itself.
- **Dataset:** the named collection and format of time-series data. Its dataset-specific code owns loading and preprocessing choices. Fit data-dependent transforms on training data; apply the fitted transforms to validation and test data.
- **Entity:** one dataset-defined series or unit selected for a run, such as a machine or asset. The dataset defines what an entity means.
- **Method:** a time-series anomaly-detection approach. A method owns its fitting, scoring, optional updating, and state rules. `MUSTARD` is a method name. Online support is a method capability; it is separate from whether a method is called a baseline.

## Execution terms

- **Offline phase:** the stage that fits a method from training data and selects validation-dependent state, such as a model checkpoint or threshold. Test labels do not select these values.
- **Online phase:** the execution stage that processes test data in stream order. It names when work happens, not whether the method adapts.
- **Online test-time adaptation:** optional method behavior that updates learned model parameters while processing the test stream. Score each window before any update; an update can affect later scores, not the score for that same window.
- **Scoring state:** state used to produce or smooth scores, such as an EWMA value. It may change during an online phase even when learned model parameters stay fixed. A scoring-state change alone is not online test-time adaptation.
- **Offline and online variants:** method-specific settings for the respective phases. The codes `O0`–`O2` and `A0`–`A2` name MUSTARD ablations only; they are not shared names for every method.

## Retrieval terms

Use these labels when comparing retrieval behavior:

- **No retrieval:** send the encoder representation directly to the task head.
- **Dense retrieval:** weight all eligible entries in a continuous attentional prototype bank.
- **Sparse retrieval:** select a subset of entries from a discrete attentional codebook and combine them. The stored entries may still be real-valued vectors. Weighted top-k output may be continuous; hard top-1 is one special case.

Record the selection rule, weighting, normalization, temperature, and any hard/soft Gumbel setting when they apply. The name *discrete attentional codebook* does not mean that its entries contain anomalies only.

## Engineering terms

- **Protocol:** in the next version of `bachelor-thesis-2026`, a set of validation rules grouped by related context. This is a project-specific meaning, not a general definition of *protocol* or a definition for `tsad-lib`.

  The four protocols in that codebase are:

  1. Run and evaluation
  2. Data and stream
  3. Scoring and calibration
  4. Method updates

- Checking whether an implementation provides required methods is a separate procedural check. It can use a Markdown checklist and simple `grep`; it is not one of these four protocols.

## Evidence and use

Evidence checked:

1. In `tsad-lib`:
   1. `documents/specs/2026-11-01-experiment-codebase-simplification.md`
   2. `documents/specs/2026-11-01-simplified-interfaces.md`
   3. `documents/specs/02-experiment-runtime-flow.md`
2. In `bachelor-thesis-2026`:
   1. `documents/spec/experiment-terminology-next-version.md`
   2. `documents/spec/online_tta_terminology_ontology.md`
   3. `documents/notes/experiment-flows-and-equations.md` — documents the current reference scoring order and retrieval behavior.

The next-version trees and interfaces remain design material until implementation and parity checks confirm them. When current code differs, describe current behavior separately instead of changing the definition to fit an unverified path.
