---
researcher: OpenAI Codex
topic: "Audit the phases, stages, and atomic steps in tsad-lib-development-spec.md"
status: complete
revision: a1bcb1b3fdcf7cae51f7e77544a8a7873b8f8994
branch: dev
---

# Research: Audit the `tsad-lib` development steps

## Summary

The development specification has a clear implementation story, but it does
not yet satisfy the strict meaning of “indivisible atomic step.” The document
contains 12 phase sections, 36 stages, and 200 numbered steps. Every phase has
a completion condition, and every stage has an atomic-step section. Several
numbered items still group many actions together. The `6A` and `6B` labels also
make the phase sequence less simple than the other numeric labels.

The document is a future development contract, not evidence that `tsad-lib`
already implements these paths. The research prompt requires this distinction.

## Research question

Does `tsad-lib-development-spec.md` provide complete high-level phases,
sequential stages, and indivisible atomic steps, according to the rules in
`prompts/1_research_prompt.md`?

## System context

The inspected document describes a new sibling library at
`/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/tsad-lib`.
It is marked `draft-development-contract` and calls itself a development
story. Its phase section begins at
`documents/spec/tsad-lib-development-spec.md:967`.

The research prompt says that evidence must be separated into implemented,
configured, tested, documented, planned, inferred, and unknown behavior. The
development specification therefore supplies documented or planned evidence;
it does not prove runtime behavior.

## Execution path in the plan

The phase map gives this order:

```text
Phase 1
→ Phase 2
→ Phase 3
→ Phase 4
→ Phase 5
→ Phase 6
→ Phase 6A
→ Phase 6B
→ Phase 7
→ Phase 8
→ Phase 9
→ Phase 10
```

The map records a dependency for every phase. Phase 6A depends on Phase 6,
Phase 6B depends on Phase 6A, and Phase 7 depends on Phase 6B. Inside each
phase, the stages appear in numeric order. The document does not give a
separate dependency field for each stage, so the stage order is inferred from
the listing order.

## Detailed findings

### 1. High-level phases

The phase layer is present and mostly coherent.

The table lists 12 phase rows, including the two lettered phases, and gives a
result, tools, and dependency for each row at
`documents/spec/tsad-lib-development-spec.md:974-989`.

The ten core phases form a sensible runtime path: package shell, common data,
dataset loading, preparation, baseline, metrics, THESIS offline execution,
online execution, matrix execution, and SMD readiness. This is documented
structure, not implemented behavior.

The two inserted phases are logically ordered. Phase 6A registers all 40
reference models. Phase 6B certifies the 15 current anomaly paths. Phase 7
then consumes that catalog and adapter boundary. The dependency cells at
`documents/spec/tsad-lib-development-spec.md:983-986` support this order.

The remaining issue is naming. The story says “ten core phases and two short
Time-Series-Library phases,” but the phase identifiers are `1, 2, 3, 4, 5, 6,
6A, 6B, 7, 8, 9, 10`. A simple phase runner may treat `6A` and `6B` as special
strings rather than ordinary sequential numbers. The dependency relation is
clear, but the numbering is not uniform.

### 2. Sequential stages

The stage layer is complete in shape.

Each phase contains named stages. Every stage has a tools line, an
`Atomic steps` heading, numbered items, and a phase completion condition. The
structural scan found all 36 stages and found a non-empty step list in every
stage.

Examples include the package stages at
`documents/spec/tsad-lib-development-spec.md:1004-1026`, the dataset stages at
`documents/spec/tsad-lib-development-spec.md:1075-1123`, and the THESIS offline
stages at `documents/spec/tsad-lib-development-spec.md:1390-1447`.

The stage sequence is readable, but stage dependencies are implicit. For
example, Stage 3.4 follows the adapter stages and tests them, but the document
does not state this dependency in a machine-readable field. A coding agent can
follow the story, while an automated planner would need to infer the order.

The completion condition is written at phase level only. The document does not
give a separate exit condition for each stage. This makes it harder to decide
whether the agent may safely move from one stage to the next when a stage has
several tests or several model sources.

### 3. Atomic-step audit

Some steps are close to indivisible. For example, the package shell creates one
file per step at `documents/spec/tsad-lib-development-spec.md:1008-1013`.
The O2 settings are also separated into individual configuration actions at
`documents/spec/tsad-lib-development-spec.md:1394-1402`. The online save and
load actions are separated at `documents/spec/tsad-lib-development-spec.md:1486-1492`.

Several steps are not indivisible under a strict reading:

| Location | Step | Why it is grouped |
| --- | --- | --- |
| `:1081-1084` | “Add ignored-name rules for cache files and unrelated metadata.” | It names two rule groups with different reasons and likely different tests. |
| `:1114-1120` | “Add one inspection test for each adapter family.” | It represents several test additions, one per family. |
| `:1114-1120` | “Add one loading test for each adapter family.” | It represents several test additions, one per family. |
| `:1286-1291` | “Add the 40 reference model names.” | It contains 40 catalog additions. |
| `:1286-1291` | “Add the source module for each model.” | It contains up to 40 source mappings. |
| `:1347-1360` | “Run one reconstruction test for each model.” | It runs four or eleven tests, depending on the stage. |
| `:1436-1444` | “Run one complete O2 offline smoke test.” | It contains many runtime operations and is an end-to-end gate, not one action. |
| `:1574-1581` | “Run the complete offline and online flow.” | It combines two large flows in one step. |
| `:1587-1594` | “Check metric formula and serialization status.” | It combines formula validation and serialization validation. |
| `:1587-1594` | “Select a new output root or an explicit resume policy.” | It contains a decision between two different policies. |

The specification therefore has an atomic-step format, but not atomicity for
every numbered item. The grouped items are the main audit finding.

### 4. Alignment with the research prompt

The document follows several useful rules from the prompt. It names files,
modules, tools, tests, and completion conditions. It also warns the coding
agent to keep implemented runtime behavior, documented research intent, and
future library design separate at
`documents/spec/tsad-lib-development-spec.md:35-53`.

The document does not follow the prompt’s research-report format because it is
not a report about an existing implementation. It proposes a future codebase.
That is appropriate for a development specification, but every phase, stage,
and step must remain labeled as planned until source code and tests confirm it.

The source story at
`documents/spec/tsad-lib-development-spec.md:1645-1652` identifies the
documents used to design the contract. It does not provide executable evidence
that the listed future modules, classes, methods, or tests already exist.

## Evidence

- `prompts/1_research_prompt.md:30-40` — the research task must explain the current implementation, separate evidence types, and avoid presenting planned behavior as implemented.
- `prompts/1_research_prompt.md:100-115` — a reliable investigation follows the trigger, inputs, configuration, calls, transformations, outputs, and confirming tests.
- `prompts/1_research_prompt.md:120-130` — the report must classify implemented, configured, tested, documented, planned, inferred, and unknown claims.
- `prompts/1_research_prompt.md:287-325` — the required report structure includes summary, question, context, execution path, findings, evidence, configuration, conflicts, and open questions.
- `documents/spec/tsad-lib-development-spec.md:967-972` — the document defines its phase, stage, and atomic-step story.
- `documents/spec/tsad-lib-development-spec.md:974-993` — the phase map lists 12 rows, tools, and dependencies.
- `documents/spec/tsad-lib-development-spec.md:1004-1026` — Phase 1 shows the stage and completion pattern.
- `documents/spec/tsad-lib-development-spec.md:1075-1123` — Phase 3 shows the dataset stage sequence and grouped family tests.
- `documents/spec/tsad-lib-development-spec.md:1271-1374` — Phases 6A and 6B show the model inventory and adapter certification sequence.
- `documents/spec/tsad-lib-development-spec.md:1390-1447` — Phase 7 shows the O2 loss, routing, checkpoint, threshold, and smoke-test sequence.
- `documents/spec/tsad-lib-development-spec.md:1546-1605` — Phase 10 shows the SMD validation, smoke flow, and launch gate.
- `git rev-parse --show-toplevel`, `git branch --show-current`, and `git rev-parse HEAD` — the audit used repository revision `a1bcb1b3fdcf7cae51f7e77544a8a7873b8f8994` on branch `dev`.

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| Phase count in the map | 12 rows | `documents/spec/tsad-lib-development-spec.md:976-989` | Planned development sequence |
| Core phase count stated by the story | 10 core phases plus 2 library phases | `documents/spec/tsad-lib-development-spec.md:969-970` | Documentation wording |
| Stage count | 36 stages | Section 17 stage headings | Planned development sequence |
| Numbered step count | 200 numbered items | Section 17 atomic-step lists | Planned development sequence |
| Phase completion gates | Present for all 12 phase sections | `documents/spec/tsad-lib-development-spec.md:1026-1605` | Planned verification |
| O2 routing | `direct_branch_routing` | `documents/spec/tsad-lib-development-spec.md:1380-1383`, `:1400-1402` | Planned THESIS offline path |

## Conflicts and uncertainties

The phase order is not contradictory, but the `6A` and `6B` labels are not
uniform with the numeric phase labels. The available document does not say
whether an implementation tool should treat them as ordinary ordered phase
identifiers or as subphases of Phase 6.

The document calls every numbered item atomic, but several items explicitly
refer to sets of models, adapter families, tests, or complete workflows. The
available document does not define the intended unit of work for those sets.

The document gives phase completion conditions but no stage completion
conditions. The available document therefore cannot prove that each stage is
independently checkable before the next stage starts.

The available files do not establish that the sibling `tsad-lib` directory
already contains the planned modules or tests. This audit assesses the plan’s
structure only.

## Open questions

1. Should `6A` and `6B` remain subphases of Phase 6, or should they become ordinary phases with uniform numeric identifiers?
2. Should each stage receive its own completion condition?
3. Should grouped model and adapter work be split into one atomic step per model or per test case?
4. Should end-to-end smoke runs remain stage gates, or should they be moved outside the atomic-step lists as verification gates?

## Audit result

The high-level phase map is present and dependency-ordered. The stage map is
present and readable. The atomic-step layer is only partially compliant: the
document uses the right headings and ordering, but several numbered items are
compound actions. The plan is suitable for human reading, but it is not yet a
strict indivisible execution plan.
