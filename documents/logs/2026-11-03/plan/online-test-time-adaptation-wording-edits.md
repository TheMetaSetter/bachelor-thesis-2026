# Online Test-Time Adaptation Wording Edit Plan

> **For agentic workers:** Follow the checklist in order. This is a documentation and prose-only plan; do not change runtime behavior.

**Goal:** Use “online test-time adaptation” precisely across bachelor-thesis-2026 and tsad-lib, while distinguishing it from a generic online execution phase.

**Architecture:** Keep one shared meaning for the term, then edit each repository's own documents. Preserve repository-specific method naming and all code, configuration, checkpoint, metric, and artifact identifiers.

**Tech Stack:** Markdown; `rg` for terminology searches; `git diff` for review; no runtime tests for prose-only changes.

**Source review:** `documents/logs/2026-11-03/detail/online-tta-terminology-review.md` and `documents/logs/2026-11-03/plan/cross-codebase-online-tta-terminology-audit.md`.

## Global constraints

- Use **online test-time adaptation** for the method family that adapts on test-stream data. State each variant's update behavior explicitly; MUSTARD A0 is the no-update control in its online test-time adaptation ablation family.
- Use **online phase** or **online-phase evaluation** for the execution stage, including frozen methods.
- In bachelor-thesis-2026 prose, call the method MUSTARD and identify THESIS as the runtime when needed.
- In tsad-lib prose, call the method MUSTARD; do not use THESIS as its method name.
- Keep `online_tta_phase`, `online_adaptation`, `BaseTTAMethod`, and other source/API identifiers unchanged.
- Do not change run identity, configuration keys, checkpoint roles, metric names, serialized fields, or historical artifacts.
- Compare relevant specification versions and record a terminology mapping. State that runtime object names are unchanged.
- Keep documentation concise and write dates as `yyyy-mm-dd`.

## Review focus

- A frozen baseline can run in an online phase without performing online test-time adaptation.
- Scoring state such as EWMA can change even when learned model parameters remain frozen.
- “Online adaptation” can refer to a broader operation; replace it only where the source means online test-time adaptation.
- Keep historical research titles, literal file names, and stable runtime identifiers intact.
- Do not imply that a proposed tsad-lib API is implemented behavior.

## Phase 1: lock the cross-repository terminology

Tools: `rg`, `sed`, `git`; read both audit files and the named specifications.

- [x] Compare the bachelor-thesis terminology ontology and online contract with relevant earlier full-spec versions. Earlier full specs use mixed long and short forms; they remain historical.
- [x] Check the offline ontology and development specification. They contain current cross-phase wording to update; `online-tta-thesis-spec.md` explicitly labels itself historical and remains unchanged.
- [x] Compare tsad-lib's runtime-flow, simplified-interface, proposed notebook API, and simplification documents.
- [x] Build the terminology map: “MUSTARD online adaptation” → “MUSTARD online test-time adaptation” (wording correction); “online adaptation phase” → deprecated prose alias; `online phase` remains the generic stage. Runtime objects are unchanged.
- [x] Search both repositories for remaining prose uses of “online adaptation,” “online TTA,” and related terms. Matches were classified as behavior, generic phase, identifier, current prose, historical text, or literature title.

## Phase 2: edit bachelor-thesis-2026

Repository: `bachelor-thesis-2026`. Tools: Markdown editor, `rg`, `git diff`.

- [x] In `documents/spec/online_tta_terminology_ontology.md`, use the full canonical term in prose and remove “online adaptation phase” from the approved alias table. Keep `online_tta_phase` unchanged.
- [x] In `documents/notes/experiment-flows-and-equations.md`, replace prose that calls the MUSTARD flow “online adaptation” with “online test-time adaptation.”
- [x] In `documents/spec/online_benchmark_contract.md`, describe the whole cohort as an online-phase experiment/evaluation. Use “online test-time adaptation” for TTA-capable method families, state variant-level updates explicitly (including MUSTARD A0 as a no-update control), and keep frozen baseline rows explicitly frozen.
- [x] In `documents/spec/offline_pretraining_terminology_ontology.md`, clarify the generic `online_variant` profile and use the full term for the specific TTA engine. Preserve `online_tta_phase` and all object identities.
- [x] In `documents/spec/tsad-lib-development-spec.md`, use “online test-time adaptation” where the contract restricts test labels from online model updates. Preserve the broader prohibition on labels reaching other online update paths.
- [x] Apply the same wording to additional active design notes found in Phase 1. Preserve broad online-learning terminology where the note covers multiple methods; preserve historical logs and versioned full-spec v1/v2/v3.
- [x] In active `documents/abstract-design-notes/idea.md`, `design_starter.md`, `offline_pretraining_phase_two_view_contrastive_design.md`, and `long_term_codebase_roadmap.md`, use “online test-time adaptation” for the MUSTARD test-stream method and its update family.
- [x] In `documents/abstract-design-notes/stream_design.md`, add a short scope note that the note discusses generic stream-time adaptation strategies, while the MUSTARD method is online test-time adaptation.
- [x] In `documents/notes/online_runtime_flow_debug.md`, `documents/notes/thesis_online_tta_prepare_event_runtime_and_data_flow.md`, and `documents/notes/smd-all-machines-experiment-matrix.md`, use the full term in prose and retain literal paths/identifiers.
- [x] In `documents/notes/terminology_ontology_naming_conventions.md`, update the link text for the renamed ontology title; keep the `online_tta_phase` mapping unchanged.
- [x] In `documents/notes/tsad-lib-ontology-audit.md`, clarify that `online_variant` is an online-phase method/profile choice, not a synonym for online test-time adaptation.
- [x] In relevant THESIS runtime docstrings, comments, and diagnostic messages, use the full term where they mean test-stream parameter adaptation. No branch logic, API identifiers, or artifact fields changed.
- [x] Add a short terminology-change note to each edited specification: wording and alias status changed; runtime object names did not.

## Phase 3: edit tsad-lib

Repository: `/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib`. Tools: Markdown editor, `rg`, `git diff`.

- [x] In `documents/specs/02-experiment-runtime-flow.md`, state that every online-phase run scores before its update step, while only methods with online test-time adaptation update learned parameters.
- [x] In `documents/specs/2026-11-01-simplified-interfaces.md`, replace ambiguous prose “methods without online adaptation” with “methods without online test-time adaptation”; clarify that their learned parameters stay unchanged while scoring state may still change.
- [x] In `documents/specs/2026-11-01-experiment-codebase-simplification.md`, change broad flow prose that treats online adaptation as the phase name to “online-phase evaluation” or equivalent wording.
- [x] In the proposed `documents/specs/2026-10-05-notebook-online-tta-api.md`, spell out “online test-time adaptation” on first prose use and distinguish that behavior from the generic online phase. Preserve its proposed-design status.
- [x] In `documents/specs/method-mathematics/mustard.md`, replace the broad phrase “online adaptation” in the test-label rule with “online test-time adaptation.” Leave unrelated method-paper terminology unchanged.
- [x] In `src/tsad/experiment_identity.py`, expand the TTA abbreviation in the relevant error message only if the exception-message contract has no exact-string callers; preserve the validation condition.
- [x] Apply the same wording to any additional current, non-historical prose found in Phase 1. Do not rename method, class, API, or artifact identifiers.
- [x] Add a terminology-change note to each edited specification: wording changed; runtime object names did not.

## Phase 4: cross-repository consistency review

Tools: `rg`, `git diff`, Markdown link/path review.

- [x] Search both repositories again for the old prose uses and inspect each remaining match.
- [x] Confirm that “online test-time adaptation” names the learned-parameter behavior and “online phase” names the broader execution stage.
- [x] Confirm that bachelor-thesis prose uses MUSTARD for the method and THESIS only for its runtime when needed; tsad-lib prose uses MUSTARD without THESIS as the method name.
- [x] Confirm the terminology map marks all runtime objects as unchanged and no config, checkpoint, metric, or artifact migration is proposed.
- [x] Review the two repository diffs. Do not run runtime tests for prose-only edits; record that tests were not run.
- [x] Update the terminology review with the completed edits and any remaining historical or identifier exceptions.
