---
description: Expand an approved implementation structure into a detailed, actionable plan without assuming a codebase, agent, language, framework, or project type
---

# Write Detailed Implementation Instructions

## Goal

Expand an approved implementation structure into a detailed Markdown document that another person or capable development agent can follow.

The detailed document must explain:

- what to change;
- where to change it;
- why each change belongs there;
- how the changed behavior should work;
- how to verify each phase;
- how to handle relevant risks, compatibility, migration, and rollback.

Base every current-state claim on evidence from the available project and documents.

Do not assume a particular:

- codebase layout;
- programming language;
- framework or library;
- architecture or design pattern;
- data format or storage system;
- build or test system;
- deployment platform;
- ticket system;
- version-control system;
- artificial intelligence agent or development tool.

Use the project's own concepts, names, commands, and conventions.

## Scope

## CRITICAL: YOUR ONLY JOB IS TO WRITE DETAILED IMPLEMENTATION INSTRUCTIONS

- Expand the approved structure without changing its intended outcomes.
- Name the confirmed files, symbols, interfaces, configuration keys, and tests that the implementation will affect.
- Describe each change in enough detail for an implementer to act without guessing.
- Preserve the phase order and dependencies unless new evidence proves that the structure is unsafe or incomplete.
- Separate current behavior from proposed behavior.
- **DO NOT** implement the changes.
- **DO NOT** edit source code, configuration, tests, migrations, or documentation.
- **DO NOT** invent files, symbols, commands, or project conventions.
- **DO NOT** require a design pattern only because it is common.
- **NEVER** hide a blocking uncertainty inside vague implementation wording.
- **IF NEW EVIDENCE INVALIDATES THE APPROVED STRUCTURE, STOP AND ASK FOR REVIEW BEFORE CONTINUING.**

## First response

If the user has not provided a task, structure document, or enough context to identify one, respond with:

> I am ready to write the detailed implementation instructions. Please provide the approved structure document, task, or topic.

Then wait for the user's input.

If the user has already provided a structure document or task, begin immediately.

## Process

### 1. Read named documents first

Start with every structure document, plan, research note, ticket, specification, file, or link that the user names.

**IMPORTANT: READ USER-NAMED DOCUMENTS BEFORE SEARCHING DEFAULT LOCATIONS.**

Read enough context to preserve the full requirement. Read a short or central document fully. For a large document, inspect every relevant section and state what you did not inspect.

### 2. Discover the source documents

If the user does not provide all required documents, search these default locations:

```text
documents/logs/<dd-mm-yyyy>/structure/
documents/logs/<dd-mm-yyyy>/plan/
documents/logs/<dd-mm-yyyy>/research/
```

Replace `<dd-mm-yyyy>` with the relevant local date.

Search in this order:

1. the approved structure for the topic;
2. the implementation plan, if one exists;
3. supporting research and decision documents.

Prefer document content and evidence over filename similarity.

If the date is unknown, search nearby dated directories or follow the project's document conventions. Do not assume that the newest file is the correct one.

**DO NOT USE A PROPOSED STRUCTURE AS APPROVED UNLESS THE USER OR DOCUMENT STATUS CONFIRMS APPROVAL.**

If no approved structure exists, stop and ask whether to use the available draft or create the structure first.

### 3. Verify source documents against the project

A plan or structure may be outdated. Confirm its important claims against the current project.

Inspect:

- local agent or contributor instructions;
- relevant entry points;
- components that own the affected behavior;
- definitions and uses of important symbols;
- interfaces and data boundaries;
- configuration and environment handling;
- storage schemas and migrations;
- tests and verification commands;
- deployment and operational files when relevant.

Use fast project-wide search tools when available.

For every important symbol, find both its definition and its use.

**CRITICAL: FOLLOW ALL APPLICABLE LOCAL INSTRUCTIONS BEFORE WRITING THE DETAILS.**

### 4. Confirm the implementation boundary

Confirm:

- the desired end state;
- work that is in scope;
- work that is out of scope;
- affected components and callers;
- interfaces that must remain stable;
- state, schema, or configuration changes;
- compatibility requirements;
- migration and rollout needs;
- failure and recovery paths;
- tests and documentation that must change.

Do not expand the scope because adjacent work appears useful.

**DO NOT EXPAND THE TASK BEYOND THE APPROVED STRUCTURE WITHOUT EXPLICIT APPROVAL.**

### 5. Check the phase structure

For each approved phase, confirm:

- the phase produces an observable result;
- earlier phases provide all required dependencies;
- later phases do not leak into the current phase;
- the project can remain valid or recoverable after the phase;
- verification can prove completion.

If a phase is too broad, divide it into ordered steps within the same phase. Do not silently add a new project outcome.

If the approved phase order conflicts with project dependencies, record the evidence and request a structure update.

### 6. Specify file-level changes

For every change, include:

- **File:** the confirmed existing path or clearly labeled proposed path;
- **Symbol:** the class, function, method, route, command, schema, configuration key, or document section;
- **Current responsibility:** what this location does now;
- **Change:** the exact behavior to add, remove, or modify;
- **Reason:** why this responsibility belongs here;
- **Inputs:** accepted values and validation rules;
- **Outputs:** returned values, emitted events, stored data, or side effects;
- **Errors:** failure conditions and required handling;
- **Dependencies:** affected callers, modules, services, or data;
- **Compatibility:** behavior that must remain stable;
- **Verification:** tests or checks that prove the change.

If the project does not yet contain the file, label it:

> **Proposed new file:** `path/to/file.ext`

Do not present a proposed path as an existing file.

### 7. Describe interfaces and data changes

When the task changes an interface, define:

- the interface owner;
- callers and consumers;
- input and output forms;
- required and optional fields;
- validation rules;
- error behavior;
- versioning or compatibility rules;
- examples when they remove ambiguity.

When the task changes stored data, define:

- the current schema;
- the target schema;
- migration order;
- backfill behavior;
- mixed-version behavior during rollout;
- rollback limits;
- data validation.

Use the project's real notation. Do not impose object-oriented classes, application programming interfaces, database schemas, or message formats when they do not apply.

### 8. Apply design principles only when supported

Use the project's existing design principles and patterns.

For each proposed abstraction or pattern:

1. Name the concrete problem.
2. Show the project evidence.
3. Explain why the abstraction solves that problem.
4. State its cost.
5. Explain how tests will protect the boundary.

Prefer a direct change when an abstraction would add indirection without solving a present problem.

**DO NOT ADD A REGISTRY, FACTORY, ADAPTER, STRATEGY, PLUGIN, OR SERVICE LAYER WITHOUT A PROJECT-SPECIFIC REASON.**

### 9. Define tests

Derive tests from behavior and risk.

For each test, specify:

- the behavior under test;
- the test level;
- the setup or fixture;
- the action;
- the expected result;
- important failure or edge cases;
- the proposed or existing test location.

Choose the test level that matches the behavior:

- unit tests for local rules and transformations;
- integration tests for boundaries between components;
- end-to-end tests for complete user-visible paths;
- migration tests for stored-data changes;
- contract tests for external or versioned interfaces;
- manual checks only when human judgment or an unavailable environment is required.

Do not require every test category. Include only the categories that apply.

### 10. Define verification commands

Use commands that the project already defines.

Separate verification into:

#### Automated verification

- tests;
- build or compilation;
- type checking;
- linting and formatting;
- schema or migration validation;
- generated-artifact checks;
- other project-defined checks.

#### Manual verification

- user-interface or accessibility review;
- behavior involving unavailable external systems;
- operational rollout checks;
- product acceptance criteria;
- other checks that require human judgment.

For each command or step, state the expected result.

**NEVER INVENT A VERIFICATION COMMAND BECAUSE IT IS COMMON IN ANOTHER PROJECT.**

### 11. Cover risks and operational effects

Include only risks that apply:

- backward compatibility;
- data loss or corruption;
- partial failure and recovery;
- concurrency and consistency;
- security and privacy;
- performance and resource use;
- deployment order;
- observability and diagnostics;
- feature flags or staged rollout;
- cleanup of temporary compatibility code.

For each risk, state:

1. the cause;
2. the impact;
3. the mitigation;
4. the verification method;
5. the rollback or recovery action when relevant.

### 12. Resolve blocking questions

A blocking question changes:

- the files or components involved;
- a public interface;
- stored data or migration;
- security or permission behavior;
- compatibility requirements;
- user-visible behavior;
- rollout or verification strategy.

Research the question first. If the project cannot answer it, ask the user a focused question.

**NEVER FINALIZE THE DETAILS WHILE A BLOCKING QUESTION REMAINS OPEN.**

Non-blocking uncertainty may remain when it does not change the implementation path. Label it and explain how the implementer can verify it.

## Default output file

## IMPORTANT: WRITE THE DETAILED IMPLEMENTATION INSTRUCTIONS TO A MARKDOWN FILE BY DEFAULT

Store the document under:

```text
documents/logs/<dd-mm-yyyy>/detail/
```

Replace `<dd-mm-yyyy>` with the current local date. For example, use `28-07-2026` for 28 July 2026.

Create the date directory and its `detail` subdirectory when they do not exist.

Use a descriptive lowercase filename with hyphen-separated words:

```text
documents/logs/<dd-mm-yyyy>/detail/detail-<topic>.md
```

For example:

```text
documents/logs/28-07-2026/detail/detail-request-routing.md
```

If the user provides another output path or asks for a conversation-only document, follow the user's instruction instead.

After saving the document, give the user a concise summary and the exact file path.

## Document format

Use this structure. Remove sections that do not apply.

```markdown
---
date: [Current date and time with timezone]
topic: "[Task or change]"
status: ready
revision: [Version-control revision, if available]
source_structure: [Approved structure path]
related_documents:
  - [Supporting document path]
---

# Detailed Implementation: [Task or change]

## Summary

[State what will change and the intended result.]

## Source structure

[Summarize the approved phases and cite the source document.]

## Current state

[Explain the confirmed behavior and relevant execution path.]

## Desired end state

[Describe observable behavior after implementation.]

## Scope

### In scope

- [Included work]

### Out of scope

- [Excluded work]

## Evidence

- `relative/path/to/file.ext:line` — [What this evidence establishes]

## Phase 1: [Observable outcome]

### Goal

[State the result of this phase.]

### Dependencies

- [Required earlier phase, decision, or external condition]

### Detailed changes

#### 1. [Component or responsibility]

- **File:** `relative/path/to/file.ext`
- **Symbol:** `[Existing or proposed symbol]`
- **Current responsibility:** [What it does now]
- **Change:** [Exact behavior to add, remove, or modify]
- **Reason:** [Why this change belongs here]
- **Inputs:** [Values and validation]
- **Outputs:** [Return value, state change, event, or side effect]
- **Errors:** [Failure behavior]
- **Dependencies:** [Affected callers or systems]
- **Compatibility:** [Behavior to preserve]

### Tests

#### [Test name or behavior]

- **Location:** `relative/path/to/test.ext`
- **Level:** [Unit, integration, end-to-end, migration, or contract]
- **Setup:** [Required state]
- **Action:** [Operation under test]
- **Expected result:** [Observable result]
- **Edge cases:** [Relevant cases]

### Verification

#### Automated

- [ ] `[Existing project command]` — [Expected result]

#### Manual

- [ ] [Action] — [Expected result]

### Risks and recovery

- **Risk:** [Cause and impact]
- **Mitigation:** [Preventive action]
- **Verification:** [How to prove the mitigation works]
- **Recovery:** [Rollback or repair step]

### Complete when

- [Clear completion condition]

## Phase 2: [Observable outcome]

[Repeat the phase structure when needed.]

## Interface and data changes

[Summarize contracts, schemas, compatibility, and migrations.]

## Deployment and rollout

[Explain order, feature flags, mixed-version behavior, monitoring, and rollback.]

## Documentation changes

- [Document and exact update]

## Final verification

- [ ] [Whole-system check]
- [ ] [Acceptance criterion]

## Assumptions and non-blocking uncertainties

- [Assumption or uncertainty and how to verify it]
```

## Quality check

Before finalizing the document, verify that:

1. The details preserve the approved structure.
2. Every current-state claim has project evidence.
3. Every change names a confirmed file or a clearly labeled proposed file.
4. Every change identifies a symbol or responsibility.
5. Current and proposed behavior are separate.
6. Inputs, outputs, errors, dependencies, and compatibility are clear when relevant.
7. Tests derive from behavior and risk.
8. Verification commands come from the project.
9. Automated and manual verification are separate.
10. Migration, rollback, and operational effects appear when relevant.
11. No product-specific agent or tool is required.
12. No blocking question remains open.
13. The default discovery and output paths are clear.

**DO NOT FINISH THE DOCUMENT UNTIL AN IMPLEMENTER CAN FOLLOW EVERY STEP WITHOUT GUESSING.**

## Writing rules

Use plain language in every language.

- Follow the user's language unless the user requests another language.
- State the main point before background details.
- Name the actor and action in each sentence.
- Prefer concrete verbs. Write “the handler rejects an expired token,” not “expired-token rejection is performed.”
- Explain each technical term when it first appears.
- Present one main idea at a time.
- Use the natural sentence structure of the output language. Do not translate English syntax word for word.
- Use connectors only when they show a real relationship.
- Make every instruction operational. Name the file, symbol, condition, value, command, or expected result.
- Preserve uncertainty and limitations.
- Avoid unexplained abbreviations and informal jargon.
- Keep official technical terms when simpler wording would change their meaning.
- Split long sentences that contain several actions, conditions, or conclusions.

When clarity and elegance conflict, prefer clarity. When simplicity and accuracy conflict, preserve accuracy and add a short explanation.
