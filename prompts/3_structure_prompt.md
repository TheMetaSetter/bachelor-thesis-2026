---
description: Create and validate a high-level implementation structure without assuming a codebase, agent, language, framework, or project type
---

# Create an Implementation Structure

## Goal

Create a high-level structure for implementing the user's task.

The structure must show the main phases, their order, their dependencies, and the result of each phase. It must remain short enough for the user to review before anyone writes a detailed implementation plan or changes the code.

Base the structure on evidence from the available project and documents. Do not assume a particular:

- codebase layout;
- programming language;
- framework or library;
- architecture or design pattern;
- build or test system;
- deployment platform;
- document name;
- ticket system;
- version-control system;
- artificial intelligence agent or development tool.

Use the project's own concepts, names, commands, and conventions.

## Scope

## CRITICAL: YOUR ONLY JOB IS TO CREATE AND VALIDATE THE IMPLEMENTATION STRUCTURE

- Identify the outcomes that the implementation must produce.
- Divide the work into coherent phases.
- Order phases by real dependencies.
- State what each phase changes and how the team can verify it.
- Keep the structure at a high level.
- **DO NOT** implement the task.
- **DO NOT** write a detailed file-by-file implementation plan yet.
- **DO NOT** force a fixed number of phases.
- **DO NOT** impose a familiar architecture or design pattern on the project.
- **NEVER** invent project files, symbols, commands, or constraints.
- **IF A BLOCKING DECISION WOULD CHANGE THE STRUCTURE, RESOLVE IT BEFORE WRITING THE FINAL OUTLINE.**

## First response

If the user has not provided a task or source document, respond with:

> I am ready to create an implementation structure. Please provide the task, specification, research note, implementation plan, or relevant project context.

Then wait for the user's input.

If the user has already provided the task or source documents, begin immediately.

## Process

### 1. Understand the task

Restate the requested change in one or two sentences.

Identify:

- the current problem;
- the desired outcome;
- explicit requirements;
- explicit exclusions;
- constraints;
- acceptance criteria;
- decisions that may change the phase order.

Do not convert vague wording into a technical decision without evidence.

### 2. Read named documents first

Start with every document, ticket, file, or link that the user names.

**IMPORTANT: READ USER-NAMED DOCUMENTS BEFORE SEARCHING DEFAULT LOCATIONS.**

Read enough context to understand the complete requirement. Read a short or central document fully. For a large document, inspect every relevant section and state what you did not inspect.

### 3. Discover supporting documents

If the user does not provide all required context, search the default document locations:

```text
documents/logs/<dd-mm-yyyy>/research/
documents/logs/<dd-mm-yyyy>/plan/
```

Replace `<dd-mm-yyyy>` with the relevant local date.

Look for documents that match the task topic. Prefer content and evidence over filename similarity.

Possible supporting documents include:

- research findings;
- vision or requirements documents;
- implementation plans;
- architecture decisions;
- risk assessments;
- migration plans;
- test plans;
- earlier structure outlines.

Treat these categories as examples. A project may organize documents differently.

If the default paths do not exist, inspect the project's documentation conventions. If no relevant document exists, use direct project evidence.

**DO NOT ASSUME THAT A DOCUMENT IS CURRENT. VERIFY IMPORTANT CLAIMS AGAINST THE PROJECT WHEN POSSIBLE.**

### 4. Inspect the project when needed

Inspect the project only as far as needed to validate the structure.

Look for:

- local agent or contributor instructions;
- entry points related to the task;
- components that own the affected responsibilities;
- interfaces and data boundaries;
- configuration and storage changes;
- tests and verification commands;
- migration, rollout, and compatibility constraints.

Use fast project-wide search tools when available.

The prompt must work with any capable agent. Refer to actions such as “search the project,” “read the file,” and “track the work.” Do not require a tool or agent name that belongs to one product.

### 5. Establish the change boundary

Define:

- what the task must change;
- what the task must preserve;
- what is out of scope;
- which components or teams depend on the change;
- which compatibility or migration rules apply;
- which risks affect phase order.

**DO NOT EXPAND THE TASK BEYOND THE USER'S REQUEST WITHOUT EXPLICIT APPROVAL.**

### 6. Identify implementation outcomes

Write each phase around an observable outcome, not around a vague activity.

Prefer:

> Phase 1 establishes the request-validation contract and keeps existing callers compatible.

Avoid:

> Phase 1 works on validation.

Each outcome should answer:

- What becomes possible after this phase?
- Which project boundary changes?
- Which later phase depends on it?
- How can the team verify the result?

### 7. Order the phases

Order phases by dependency, risk, and reversibility.

A common order may include:

1. contracts, schemas, or shared foundations;
2. a minimal end-to-end path;
3. integrations and remaining behavior;
4. migration and compatibility work;
5. validation, rollout, documentation, and cleanup.

Use this order only when it fits the task.

Prefer an early minimal end-to-end path when it can test the main assumptions safely. A minimal end-to-end path is the smallest working path from input to observable result.

A small task may need one phase. A large task may need several. Do not add phases only to make the outline look complete.

### 8. Check each phase

Each phase must include:

- a clear outcome;
- its main scope;
- dependencies on earlier phases;
- affected components or responsibilities;
- important constraints;
- automated verification;
- manual verification when human judgment is required;
- relevant risks;
- a clear completion condition.

Keep file-level changes for the detailed implementation plan. Include a file path only when it is already confirmed and necessary to understand the phase boundary.

### 9. Resolve blocking decisions

A blocking decision changes:

- phase boundaries or order;
- a public interface;
- stored data or migration;
- security or permission behavior;
- compatibility requirements;
- user-visible behavior;
- rollout or verification strategy.

Research the decision first. If the available evidence cannot resolve it, ask the user a focused question.

**NEVER FINALIZE THE STRUCTURE WHILE A BLOCKING DECISION REMAINS OPEN.**

### 10. Ask for feedback

Present the proposed structure before anyone writes detailed implementation instructions.

Ask the user:

- Does the phase order match the intended delivery path?
- Is any phase too broad or too narrow?
- Is any required outcome missing?
- Does any out-of-scope item need to move into scope?

**IMPORTANT: GET USER FEEDBACK ON THE STRUCTURE BEFORE EXPANDING IT INTO A DETAILED IMPLEMENTATION PLAN.**

If the user already approved the structure or explicitly asks to skip review, continue without asking again.

## Default output file

## IMPORTANT: WRITE THE IMPLEMENTATION STRUCTURE TO A MARKDOWN FILE BY DEFAULT

Store the structure under:

```text
documents/logs/<dd-mm-yyyy>/structure/
```

Replace `<dd-mm-yyyy>` with the current local date. For example, use `28-07-2026` for 28 July 2026.

Create the date directory and its `structure` subdirectory when they do not exist.

Use a descriptive lowercase filename with hyphen-separated words:

```text
documents/logs/<dd-mm-yyyy>/structure/structure-<topic>.md
```

For example:

```text
documents/logs/28-07-2026/structure/structure-request-routing.md
```

If the user provides another output path or asks for a conversation-only outline, follow the user's instruction instead.

After saving the structure, give the user a concise summary and the exact file path.

## Document format

Use this structure. Remove sections that do not apply.

```markdown
---
date: [Current date and time with timezone]
topic: "[Task or change]"
status: proposed
revision: [Version-control revision, if available]
related_documents:
  - [Source-document path]
---

# Implementation Structure: [Task or change]

## Summary

[State the intended result and the proposed delivery path.]

## Request

[Preserve the user's task, constraints, and exclusions.]

## Confirmed context

- [Current behavior or constraint with evidence]

## Scope

### In scope

- [Included outcome]

### Out of scope

- [Excluded outcome]

## Proposed phases

### Phase 1: [Observable outcome]

**Result:** [What becomes true after this phase]

**Scope:**

- [Main responsibility or component]

**Depends on:**

- [Earlier decision, phase, or external condition]

**Verification:**

- Automated: [Existing project check or expected automated evidence]
- Manual: [Human check, only when needed]

**Risks:**

- [Relevant risk and mitigation]

**Complete when:**

- [Clear completion condition]

### Phase 2: [Observable outcome]

[Repeat the phase structure when needed.]

## Dependency summary

| Phase | Requires | Enables |
| --- | --- | --- |
| [Phase] | [Dependency] | [Later outcome] |

## Decisions confirmed

- [Decision and supporting evidence]

## Non-blocking uncertainties

- [Uncertainty and how the detailed plan should verify it]

## Feedback requested

- [Focused question about order, scope, or granularity]
```

After the user approves the outline, change `status: proposed` to `status: approved` and update the same file unless the user requests a separate version.

## Quality check

Before presenting the structure, verify that:

1. The outline answers the user's actual task.
2. The phases follow real dependencies.
3. Each phase produces an observable result.
4. The outline separates current evidence from proposed work.
5. The scope and exclusions are explicit.
6. Verification is possible at the end of each phase.
7. The number of phases matches the task size.
8. No agent-specific tool, command, or role appears as a requirement.
9. No blocking decision remains open.
10. The default discovery and output paths are clear.

**DO NOT FINISH THE STRUCTURE UNTIL EVERY PHASE HAS A CLEAR RESULT, DEPENDENCY, AND COMPLETION CONDITION.**

## Writing rules

Use plain language in every language.

- Follow the user's language unless the user requests another language.
- State the main point before background details.
- Name the actor and action in each sentence.
- Prefer concrete verbs. Write “the service validates the request,” not “request validation is performed.”
- Explain each technical term when it first appears.
- Present one main idea at a time.
- Use the natural sentence structure of the output language. Do not translate English syntax word for word.
- Use connectors only when they show a real relationship.
- Make instructions operational. Name the phase, condition, outcome, or expected result.
- Preserve uncertainty and limitations.
- Avoid unexplained abbreviations and informal jargon.
- Keep official technical terms when simpler wording would change their meaning.
- Split long sentences that contain several actions, conditions, or conclusions.

When clarity and elegance conflict, prefer clarity. When simplicity and accuracy conflict, preserve accuracy and add a short explanation.
