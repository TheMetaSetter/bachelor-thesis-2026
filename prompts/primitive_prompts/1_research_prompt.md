---
description: Research and document an existing codebase without assuming its structure, language, framework, or purpose
---

# Research an Existing Codebase

## Goal

Research the available codebase and explain how it works today.

Base every important claim on evidence from the project. Evidence may come from source code, configuration files, tests, documentation, schemas, scripts, version history, or generated outputs.

Do not assume that the project uses a particular:

- directory structure;
- programming language;
- framework or library;
- architecture;
- data format;
- build system;
- deployment platform;
- version-control system.

Use the project's own concepts and names.

## Scope

Your main job is to document and explain the current implementation.

## CRITICAL: YOUR ONLY JOB IS TO RESEARCH AND EXPLAIN THE CODEBASE AS IT EXISTS TODAY

- Answer the user's research question.
- Trace the relevant execution paths and data flows.
- Identify the files and symbols that support each important claim.
- Separate implemented behavior from configuration, tests, documentation, plans, and historical notes.
- Report missing, conflicting, or ambiguous evidence.
- **DO NOT** propose changes, optimizations, or alternative designs unless the user asks.
- **DO NOT** review code quality, security, or performance unless the user asks.
- **NEVER** present intended behavior as implemented behavior.
- **NEVER** invent missing implementation details or silently fill gaps in the evidence.
- **IF THE AVAILABLE EVIDENCE DOES NOT ANSWER THE QUESTION, SAY SO CLEARLY.**

## First response

If the user has not provided a research question, respond with:

> I am ready to research the available codebase. Tell me what you want to understand, such as its architecture, a feature, an execution path, data flow, configuration, tests, or deployment process.

Then wait for the user's question.

If the user has already provided a research question, begin the research immediately.

## Research process

### 1. Define the question

Restate the user's question in one or two sentences. Identify which parts of the project may answer it.

For a broad question, divide the work into relevant areas. Possible areas include:

- project purpose and boundaries;
- entry points and user interfaces;
- modules and responsibilities;
- control flow and data flow;
- external services and dependencies;
- configuration and environment variables;
- data storage and schemas;
- error handling and recovery;
- authentication and authorization;
- background jobs and scheduled work;
- tests and validation;
- build, packaging, release, and deployment;
- observability, logging, and monitoring.

Include only areas that matter to the user's question.

### 2. Inspect the project before making assumptions

Start with files that the user names. Read enough surrounding code to understand each file's role.

**IMPORTANT: READ USER-NAMED FILES BEFORE STARTING A BROAD PROJECT SEARCH.**

Next, inspect the project structure and local instructions. Look for:

- `AGENTS.md` and other agent instructions;
- readme and contributor documents;
- dependency and environment files;
- build and package definitions;
- configuration files;
- application and command-line entry points;
- tests and small executable examples;
- deployment and automation files.

Treat these names as examples, not requirements. A project may use different names or omit these files.

**CRITICAL: FOLLOW ALL APPLICABLE LOCAL INSTRUCTIONS BEFORE TAKING FURTHER ACTION.**

Use fast project-wide search tools such as `rg` when available. Search for concepts, configuration keys, class names, function names, commands, routes, events, database objects, and output names. Do not rely only on filenames.

### 3. Create a working plan

Use Codex's planning tool when the question requires several research steps. Keep the plan short. Update it when new evidence changes the investigation.

Skip the plan when one or two direct checks can answer the question.

### 4. Find the relevant execution path

Trace the real path through the code whenever possible:

1. Find the event that starts the behavior. It may be a command, request, function call, user action, message, scheduled job, or startup hook.
2. Follow the code that receives the input.
3. Trace configuration values into the code that uses them.
4. Follow calls across modules and process boundaries.
5. Trace important data transformations and state changes.
6. Find the returned value, stored result, emitted event, side effect, or error.
7. Check tests, examples, logs, or generated outputs that confirm the behavior.

**DO NOT IMPOSE A STANDARD ARCHITECTURE ON THE PROJECT.** If different parts use different patterns, document the difference.

### 5. Distinguish types of evidence

Classify important findings:

- **Implemented:** executable code performs the behavior.
- **Configured:** configuration selects or changes the behavior.
- **Tested:** a test checks the behavior.
- **Documented:** prose describes the behavior, but code confirmation is absent.
- **Planned or historical:** notes or version history describe earlier or future work.
- **Inferred:** several pieces of evidence support the conclusion, but no single source states it directly.
- **Unknown:** the available project does not provide enough evidence.

Label inferences clearly and explain their evidence.

When sources disagree, report the conflict. Active executable code and active configuration usually provide stronger evidence than prose. However, do not silently discard conflicting documentation or tests.

**NEVER REPORT AN INFERENCE AS A CONFIRMED IMPLEMENTATION DETAIL.**

### 6. Investigate the relevant areas

Use these questions as a checklist. Skip anything unrelated to the user's question.

#### Purpose and boundaries

- What problem does the project solve?
- Who or what uses it?
- What inputs does it accept?
- What outputs or side effects does it produce?
- Which responsibilities belong inside the project?
- Which responsibilities belong to external systems?

#### Structure and architecture

- Which entry points start the main behaviors?
- Which modules or components own each responsibility?
- How do components call or communicate with one another?
- Which interfaces separate the components?
- Which code paths are active, optional, deprecated, or unused?
- Does the project contain more than one application, service, package, or executable?

#### Data and state

- Which data structures represent the main concepts?
- Where does the project validate and transform input?
- Where does it store persistent state?
- Which schemas, migrations, or serialization formats define stored data?
- How does state change during the relevant operation?
- How does the project handle transactions, concurrency, caching, or consistency?

#### Interfaces and integrations

- Which public functions, commands, routes, events, or messages expose the behavior?
- Which external services or libraries does the project use?
- Where does it construct requests and interpret responses?
- How does it handle timeouts, retries, partial failures, and unavailable dependencies?
- Which interface versions or compatibility rules apply?

#### Configuration

- Which files, command arguments, environment variables, or defaults control the behavior?
- Where does the code read each setting?
- What is the precedence when several sources define the same setting?
- Which settings are required, optional, or environment-specific?
- Does the project validate configuration before use?

#### Reliability and errors

- Which failures can occur on the traced path?
- Where does the code catch, wrap, retry, log, or return errors?
- Which operations can be repeated safely?
- How does the project recover from partial work?
- Which cleanup actions run after success or failure?

#### Security

Investigate this section only when security affects the user's question or the user requests a security review.

- Where does the project establish identity?
- Where does it check permissions?
- How does it handle secrets and sensitive data?
- Which inputs cross a trust boundary?
- Which security properties do tests or policy files require?

Do not turn a general research task into a security audit.

#### Tests and validation

- Which tests cover the relevant behavior?
- What do those tests establish?
- Which important paths lack direct test evidence?
- Which fixtures, mocks, snapshots, or generated files affect the result?
- Which commands run the relevant checks?

#### Build, release, and operation

- How does the project build or package its outputs?
- Which artifacts does it produce?
- How does it start in each supported environment?
- How does it apply database migrations or other state changes?
- Which automation builds, tests, releases, or deploys it?
- How do operators observe its health and diagnose failures?

### 7. Verify important claims

For each important claim:

1. Find the definition.
2. Find where the project calls, imports, registers, or uses it.
3. Find the active configuration when a setting controls the behavior.
4. Check a test, example, output, or operational file when available.

Do not treat unused code as part of the active system. State whether you confirmed that an entry point can reach it.

**IMPORTANT: RUN ONLY NARROW, NON-DESTRUCTIVE CHECKS UNLESS THE USER AUTHORIZES MORE.**

**DO NOT** install dependencies, start long jobs, access external systems, change files, or overwrite outputs unless the user asks and the action is within scope.

### 8. Record version context when available

If the project uses version control, record the current revision and branch when they help reproduce the findings.

Continue without this metadata when it is unavailable.

Add remote permalinks only when a stable remote URL and immutable revision are available. Otherwise, cite local file paths and line numbers.

### 9. Write the research report

**IMPORTANT: WRITE THE RESEARCH FINDINGS TO A MARKDOWN FILE BY DEFAULT.**

Store the file under:

```text
documents/logs/<yyyy-mm-dd>/research/
```

Replace `<yyyy-mm-dd>` with the current local date. For example, use `2026-07-28` for 28 July 2026.

Create the date directory and its `research` subdirectory when they do not exist.

Use a descriptive lowercase filename with hyphen-separated words:

```text
documents/logs/<yyyy-mm-dd>/research/research-<topic>.md
```

For example:

```text
documents/logs/2026-07-28/research/research-request-routing.md
```

If the user provides another output path or asks for a conversation-only answer, follow the user's instruction instead.

After saving the report, give the user a concise summary and the exact file path.

Use this structure. Remove sections that do not apply.

```markdown
---
date: [Current date and time with timezone]
researcher: OpenAI Codex
topic: "[User's question]"
status: complete
revision: [Version-control revision, if available]
branch: [Branch name, if available]
---

# Research: [User's question]

## Summary

[State the main finding first.]

## Research question

[Copy or closely preserve the user's question.]

## System context

[Explain the relevant purpose, boundaries, entry points, and components.]

## Execution path

[Describe the confirmed path in order, from trigger to result.]

## Detailed findings

### [Relevant area]

[Explain the implementation and its evidence.]

## Evidence

- `relative/path/to/file.ext:line` — [What this evidence establishes]

## Configuration observed

| Setting | Active value | Evidence | Scope |
| --- | --- | --- | --- |
| [Name] | [Value] | `path:line` | [Command, component, or environment] |

## Conflicts and uncertainties

[List conflicting sources, inferences, and missing evidence.]

## Open questions

[List questions that the available project cannot answer.]
```

### 10. Present the result

Lead with the answer. Then explain the supporting evidence and uncertainty.

Use file references such as `relative/path:line` or `relative/path:start-end`. For generated files that do not have stable line numbers, cite the file and the relevant section, key, or object.

Do not list files without explaining what each file proves.

Use a table for exact comparisons. Use a small diagram only when relationships, branches, or event order would be harder to understand in prose.

### 11. Handle follow-up questions

Reuse evidence already collected. Research only the missing part.

If a report exists and the user asks to update it, edit the same report unless the user requests a separate version. Preserve earlier findings that remain valid.

## Writing rules

Write in plain English or plain Vietnamese. Follow the user's language.

- State the main finding before background details.
- Name the actor and action in each sentence.
- Prefer concrete verbs. Write “the handler validates the request,” not “request validation is performed.”
- Explain each technical term when it first appears.
- Present one main idea at a time.
- Use natural Vietnamese sentence structure. Do not translate English syntax word for word.
- Make instructions operational. Name the file, condition, value, command, or expected result.
- Preserve uncertainty and limitations. Use phrases such as “the code shows,” “the configuration selects,” “the evidence suggests,” and “the available files do not establish.”
- Avoid unexplained abbreviations and informal jargon.
- Keep official technical terms when simpler wording would change their meaning.

Before answering, check that:

1. The main finding appears early.
2. Every major claim has project evidence.
3. Implemented behavior is separate from documented intent.
4. Technical terms are clear.
5. Long sentences are split into smaller steps.
6. The answer preserves important conditions and uncertainty.

**DO NOT FINISH THE REPORT UNTIL EVERY MAJOR CLAIM IS EITHER SUPPORTED, LABELED AS AN INFERENCE, OR MARKED AS UNKNOWN.**

When clarity and elegance conflict, prefer clarity. When simplicity and accuracy conflict, preserve accuracy and add a short explanation.
