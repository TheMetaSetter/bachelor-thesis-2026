---
description: Create a detailed implementation plan for an existing codebase without assuming its structure, language, framework, or purpose
---

# Create an Implementation Plan for an Existing Codebase

## Goal

Create a complete, actionable implementation plan for the user's task.

Base the plan on evidence from the current project. Evidence may come from source code, configuration files, tests, documentation, schemas, scripts, version history, research notes, tickets, or generated outputs.

Do not assume that the project uses a particular:

- directory structure;
- programming language;
- framework or library;
- architecture or design pattern;
- data format or storage system;
- build or test system;
- deployment platform;
- ticket system;
- version-control system.

Use the project's own concepts, names, tools, and conventions.

## Scope

## CRITICAL: YOUR ONLY JOB IS TO RESEARCH THE TASK AND WRITE AN IMPLEMENTATION PLAN

- Explain what must change and why.
- Identify the files, symbols, interfaces, configurations, tests, and documentation that the implementation will affect.
- Divide the work into ordered, verifiable phases.
- Separate confirmed project behavior from assumptions and proposed changes.
- Resolve decisions that would change the implementation before finalizing the plan.
- **DO NOT** implement the changes unless the user explicitly asks.
- **DO NOT** edit source code, configuration, tests, or documentation while planning.
- **DO NOT** force a familiar architecture or design pattern onto the project.
- **NEVER** invent files, modules, commands, interfaces, or project conventions.
- **IF THE AVAILABLE EVIDENCE DOES NOT SUPPORT A DETAIL, MARK IT AS UNKNOWN AND INVESTIGATE IT.**

## First response

If the user has not provided a task, respond with:

> I am ready to create an implementation plan. Please provide the task, issue, feature request, bug report, or specification. Include any relevant files, constraints, and expected behavior.

Then wait for the user's input.

If the user has already provided a task or referenced a file, begin the planning process immediately.

## Planning process

### 1. Understand the request

Restate the task in one or two sentences.

Identify:

- the current behavior;
- the desired behavior;
- the reason for the change;
- explicit requirements;
- explicit exclusions;
- acceptance criteria;
- constraints that limit the solution.

Do not turn vague wording into a technical decision without evidence. Record the uncertainty and investigate it.

### 2. Read the supplied context first

Start with every file, ticket, research note, specification, or link that the user names.

**IMPORTANT: READ USER-NAMED FILES BEFORE STARTING A BROAD PROJECT SEARCH.**

Read enough context to understand the complete requirement. For a short or central file, read it fully. For a large file, inspect all relevant sections and state what you did not inspect.

Next, inspect the project structure and local instructions. Look for:

- `AGENTS.md` and other agent instructions;
- readme and contributor documents;
- architecture and design records;
- dependency and environment files;
- build and package definitions;
- configuration files;
- application and command-line entry points;
- tests and small executable examples;
- deployment and automation files.

Treat these names as examples, not requirements.

**CRITICAL: FOLLOW ALL APPLICABLE LOCAL INSTRUCTIONS BEFORE CONTINUING.**

### 3. Find existing research

Look for a research note related to the task. The default research location is:

```text
documents/logs/<yyyy-mm-dd>/research/
```

Replace `<yyyy-mm-dd>` with the relevant local date.

If the user names a research note, use that file. If several notes may apply, select them by topic and evidence rather than filename alone.

A research note can guide the investigation, but it does not replace code verification. Check its important claims against the current project because the code may have changed.

If no research note exists, continue with direct project research. Do not create a separate research report unless the user asks or the task cannot be planned safely without one.

### 4. Create a working plan

Use Codex's planning tool when the investigation has several steps. Keep the working plan short. Update it when new evidence changes the task scope.

Skip the working plan when one or two direct checks can establish the implementation path.

### 5. Research the current implementation

Use fast project-wide search tools such as `rg` when available.

Trace the real execution path:

1. Find the event that starts the current behavior.
2. Follow the code that receives the input.
3. Trace configuration values into the code that uses them.
4. Follow calls across modules and process boundaries.
5. Trace data transformations and state changes.
6. Find outputs, stored results, emitted events, side effects, and errors.
7. Find tests that establish the current behavior.
8. Find similar completed changes or established patterns in the project.

For every important symbol, find both its definition and its use.

Do not treat unused, deprecated, experimental, or unreachable code as part of the active system. State its status when it affects the plan.

### 6. Define the change boundary

Identify:

- the entry points affected by the change;
- the components that own the relevant responsibilities;
- the interfaces that must remain stable;
- the data or state that will change;
- external systems and dependencies involved;
- compatibility requirements;
- migration or rollout needs;
- failure and recovery paths;
- tests and documentation that must change;
- work that is explicitly out of scope.

Prefer the smallest change that fully satisfies the task and fits the existing project.

**DO NOT EXPAND THE TASK BEYOND THE USER'S REQUEST WITHOUT EXPLICIT APPROVAL.**

### 7. Develop the implementation approach

Choose an approach that follows the project's current structure and conventions.

Use a design pattern only when the project already uses it or when the task provides clear evidence that it solves a specific problem. Name the problem that the pattern solves.

For each meaningful design decision:

1. State the decision.
2. Cite the project evidence.
3. Explain why it fits the current code.
4. State its effect on implementation and testing.

If more than one approach remains valid, compare only the options that would lead to materially different implementations.

Use a table when it makes the comparison clearer:

| Option | Fits current project | Benefits | Costs or risks | Evidence |
| --- | --- | --- | --- | --- |
| [Option] | [How it fits] | [Benefits] | [Costs] | `path:line` |

Ask the user to choose when the decision depends on product intent, business rules, risk tolerance, or another preference that the code cannot answer.

Do not ask the user to decide something that the project evidence already resolves.

### 8. Resolve blocking questions

A blocking question is an unanswered question that would change:

- the files or components involved;
- a public interface;
- stored data or a migration;
- security or permission behavior;
- compatibility requirements;
- user-visible behavior;
- testing or rollout strategy.

Research each blocking question first. If the project cannot answer it, ask the user a focused question.

**DO NOT FINALIZE THE PLAN WHILE A BLOCKING QUESTION REMAINS OPEN.**

Non-blocking uncertainty may remain when it does not change the implementation path. Label it and state how the implementer can verify it.

### 9. Divide the work into phases

Each phase must produce a coherent and verifiable result.

Order phases by dependency:

1. shared contracts, schemas, or infrastructure;
2. core behavior;
3. integrations and callers;
4. migrations or compatibility work;
5. tests, documentation, rollout, and cleanup.

Use this order only when it matches the task. A small change may need one phase. Do not create extra phases to make the plan appear thorough.

For risky changes, prefer an incremental path that keeps the project runnable between phases.

### 10. Specify each change

For every planned change, include:

- the existing file path;
- the relevant symbol, section, or configuration key;
- the responsibility of that code;
- the exact behavior to add, remove, or change;
- inputs, outputs, state changes, and error behavior;
- affected callers or dependencies;
- tests that prove the change;
- compatibility, migration, or rollback details when applicable.

If a new file is required, give its proposed path and purpose. Clearly label the path as proposed.

Do not include large implementation code blocks. Use short pseudocode, function signatures, schemas, or examples only when they remove ambiguity.

### 11. Define verification

Separate verification into two groups.

#### Automated verification

Include commands and checks that an execution agent can run, such as:

- unit tests;
- integration tests;
- end-to-end tests;
- type checking;
- compilation or build checks;
- linting and formatting;
- schema or migration validation;
- generated-artifact checks.

Use commands that the project already defines. Do not invent a command because it is common in another project.

#### Manual verification

Include only checks that require human judgment or a real environment, such as:

- user-interface behavior;
- accessibility or visual review;
- behavior involving unavailable external systems;
- operational rollout checks;
- product acceptance criteria.

State the exact steps and expected result.

Do not use manual verification as a substitute for an automated test that the project can reasonably support.

### 12. Check risks and operational effects

Include only risks that apply to the task:

- backward compatibility;
- data migration and rollback;
- partial failure and recovery;
- concurrency and consistency;
- security and privacy;
- performance and resource use;
- deployment order;
- observability and diagnostics;
- feature flags or staged rollout;
- cleanup of temporary compatibility code.

For each risk, state the cause, impact, mitigation, and verification method.

Do not turn a normal implementation plan into a full security or performance review unless the user asks.

## Default output file

## IMPORTANT: WRITE THE IMPLEMENTATION PLAN TO A MARKDOWN FILE BY DEFAULT

Store the plan under:

```text
documents/logs/<yyyy-mm-dd>/plan/
```

Replace `<yyyy-mm-dd>` with the current local date. For example, use `2026-07-28` for 28 July 2026.

Create the date directory and its `plan` subdirectory when they do not exist.

Use a descriptive lowercase filename with hyphen-separated words:

```text
documents/logs/<yyyy-mm-dd>/plan/plan-<topic>.md
```

For example:

```text
documents/logs/2026-07-28/plan/plan-request-routing.md
```

If the user provides another output path or asks for a conversation-only plan, follow the user's instruction instead.

After saving the plan, give the user a concise summary and the exact file path.

## Plan format

Use this structure. Remove sections that do not apply.

```markdown
---
date: [Current date and time with timezone]
planner: OpenAI Codex
topic: "[Task or change]"
status: ready
revision: [Version-control revision, if available]
branch: [Branch name, if available]
related_research: [Research-note path, if used]
---

# Implementation Plan: [Task or change]

## Summary

[State what will change and the intended result.]

## Request

[Preserve the user's task and important constraints.]

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

## Implementation approach

[Explain the selected approach and why it fits the project.]

## Phase 1: [Outcome]

### Goal

[State the result of this phase.]

### Changes

#### 1. [Component or responsibility]

- **File:** `relative/path/to/file.ext`
- **Symbol:** `[Existing or proposed symbol]`
- **Change:** [Exact behavior to add, remove, or modify]
- **Reason:** [Why the change belongs here]
- **Dependencies:** [Affected callers, data, or systems]

### Verification

#### Automated

- [ ] `[Existing project command]` — [Expected result]

#### Manual

- [ ] [Action] — [Expected result]

### Risks

- [Risk, mitigation, and verification]

## Phase 2: [Outcome]

[Repeat the phase structure when needed.]

## Testing strategy

[Summarize coverage, important cases, and test boundaries.]

## Migration and rollback

[Explain data, configuration, compatibility, deployment, and recovery steps.]

## Documentation

[List user, developer, operational, or API documentation changes.]

## Final verification

- [ ] [Whole-system check]
- [ ] [Acceptance criterion]

## Assumptions and non-blocking uncertainties

- [Assumption or uncertainty and how to verify it]
```

## Quality check

Before finalizing the plan, verify that:

1. The plan answers the user's actual request.
2. Every current-state claim has project evidence.
3. Every proposed change names a file or a clearly identified component.
4. The phases follow real dependencies.
5. The plan separates current behavior from proposed behavior.
6. Automated and manual verification are separate.
7. Commands come from the project.
8. Compatibility, migration, rollback, and operational effects are included when relevant.
9. Out-of-scope work is explicit.
10. No blocking question remains open.

**DO NOT FINISH THE PLAN UNTIL EVERY STEP IS ACTIONABLE AND EVERY BLOCKING DECISION IS RESOLVED.**

## Writing rules

Use plain language in every language.

- Follow the user's language unless the user requests another language.
- State the main point before background details.
- Name the actor and action in each sentence.
- Prefer concrete verbs. Write “the handler validates the request,” not “request validation is performed.”
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
