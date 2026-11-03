# Online test-time adaptation terminology review

Date: 2026-11-03  
Scope: bachelor-thesis-2026 and tsad-lib specifications, active design notes, and runtime wording.  
Method: static search, source/spec reading, and diff review; runtime behavior and object identities did not change.

## Canonical wording

Use **online test-time adaptation** for the method family that adapts on test-stream data. State each variant's update behavior explicitly; MUSTARD A0 is the no-update control in its online test-time adaptation ablation family. In the bachelor-thesis repository, name the scientific method as “MUSTARD online test-time adaptation in the THESIS runtime.” In tsad-lib prose, use “MUSTARD online test-time adaptation” and do not use THESIS as the method name.

Use **online phase** or **online-phase evaluation** for the execution stage shared by methods that update learned parameters, methods that remain frozen, and methods whose scoring state changes without learned-parameter updates. An online phase is not automatically online test-time adaptation.

## Findings

| Codebase | Evidence | Finding and precise wording action |
|---|---|---|
| bachelor-thesis-2026 | `documents/spec/online_tta_terminology_ontology.md:4,30,108-120` | The authoritative ontology correctly scopes `online_tta_phase` to the THESIS runtime and MUSTARD. Its alias table also permits “online adaptation phase” as a contextual alias. That alias is less precise and conflicts with the requested canonical wording. Remove it from approved prose aliases; retain `online_tta_phase` as the code-facing identifier. |
| bachelor-thesis-2026 | `documents/notes/experiment-flows-and-equations.md:3,42` | “MUSTARD online adaptation” names the method flow without saying that it is test-time adaptation. Replace prose uses with “MUSTARD online test-time adaptation.” |
| bachelor-thesis-2026 | `documents/spec/online_benchmark_contract.md:7,302-312,403-406` | The contract covers THESIS, adaptive M2N2/CANDI, and frozen traditional baselines. Its broad scope and section heading call the whole comparison “online test-time adaptation” or “online adaptation,” although some methods do not adapt. Name the full cohort an “online-phase experiment/evaluation”; use “online test-time adaptation” for the TTA-capable method families and state each variant's update behavior, including MUSTARD A0 as a no-update control. |
| bachelor-thesis-2026 | `src/engine/online_tta/` and `scripts/experiments/run_online_adaptation.py` | `online_tta` and `online_adaptation` are literal package/API names. Keep them as identifiers. In prose, explain which MUSTARD online test-time adaptation behavior they implement rather than treating the identifier as the preferred term. |
| tsad-lib | `documents/specs/02-experiment-runtime-flow.md:15-18`; `src/tsad/engine.py:45,176-179` | The runtime has a generic online phase for one method, and the spec says methods may or may not have test-time adaptation. This is the correct distinction. Tighten the sentence to say that all online-phase runs score before their update step, while only methods with online test-time adaptation update learned parameters. |
| tsad-lib | `documents/specs/2026-11-01-simplified-interfaces.md:7-10,195` | The spec correctly allows a no-op model update and notes that frozen methods may still change scoring state. “Methods without online adaptation” is ambiguous; replace it with “methods without online test-time adaptation” and state that scoring state may still change. |
| tsad-lib | `documents/specs/2026-10-05-notebook-online-tta-api.md:1-16`; `src/tsad/models/tta.py:67-68` | The API note is explicitly a proposed design and `BaseTTAMethod` is a method-family identifier. Neither makes the generic `online` phase itself TTA. Expand “online TTA” to “online test-time adaptation” on first prose use and state the phase/behavior distinction. Keep API and class identifiers unchanged. |

## Review conclusion

The main cross-codebase confusion is using “online adaptation” as a short synonym for the precise method family. It can also mean a generic online-phase operation or a broader class of updates. Use **online test-time adaptation** for the method family/setting and state variant-level updates explicitly; A0 is a no-update control in the MUSTARD ablation family. Use “online phase” for the execution stage, including frozen methods.

The bachelor-thesis specs already define the precise term but permit a looser alias and use loose prose in two current documents. tsad-lib separates the concepts in its runtime design, but some prose says “online adaptation” and its proposed notebook API uses only the abbreviation. These are wording risks, not evidence that the code performs the wrong updates.

This review did not test execution or establish scientific equivalence between repositories. It did not rename API identifiers, class names, directories, or historical documents.


## Execution outcome (2026-11-03)

The wording edits in the checklist plan are complete. Current bachelor-thesis specs and active design prose use the full term for the MUSTARD method family. tsad-lib's current runtime-flow and design specs distinguish online-phase evaluation from optional online test-time adaptation; its MUSTARD method note now states that test labels are never used for online test-time adaptation. The relevant identity error message spells out the term; its validation condition is unchanged, and no exact-string callers were found.

Terminology mapping: `online adaptation phase` → deprecated prose alias; `online phase` → execution stage; `online test-time adaptation` → optional learned-parameter behavior on test-stream data; `online_tta_phase`, `online_adaptation`, `BaseTTAMethod`, and other runtime/API identifiers → unchanged. No configuration, checkpoint, metric, artifact, or run-identity migration is proposed.

Remaining matches are historical/versioned thesis specs, the explicitly historical online-TTA thesis spec, literature titles and method-paper discussion, generic stream/recovery wording, or stable identifiers. These retain their original scope. Proposed tsad-lib API documents remain marked as proposals. Runtime tests were not run because the edits are prose and one diagnostic string only.
