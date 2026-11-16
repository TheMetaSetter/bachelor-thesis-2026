# Agent instructions

## Required reading

1. Read [codebase-agnostic-preferences.md](codebase-agnostic-preferences.md) for general work preferences.
2. Read [codebase-specific-preferences.md](codebase-specific-preferences.md) for project preferences.
3. Read the documents that apply to the task.

Apply both preference files within their stated scope.
Report missing references and conflicting requirements before affected changes.

## Work sequence

1. Inspect the current files and relevant evidence.
2. Make a short checklist-style plan with phases, stages, atomic steps, and tools.
   - Put phases in execution order. Complete and check one phase before you start the next.
   - Put stages inside phases.
   - Put atomic steps inside stages. Each step must contain one action.
   - Name the tools for each step.
3. Make the smallest change that meets the requirements.
4. Run the checks that apply to the change.
5. Review the diff.
6. Report changes, check results, and remaining problems.

For a requested detailed plan, write an ordered Markdown checklist.
For plan execution, complete each checklist item in order.

## Project rules

- Use `documents/` for project requirements.
- Before Python work, read [python_semantics.md](python_semantics.md).
- Before a full experiment, test one combination through the complete flow.
- Keep changes outside the task unchanged.
- Keep credentials out of code, documents, and output.
- Separate proposed behavior from current behavior.
- Use ASD-STE100 Simplified Technical English.