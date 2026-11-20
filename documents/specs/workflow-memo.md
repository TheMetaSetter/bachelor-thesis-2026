# Workflow memo

Date: 2026-11-16
Status: observed workflow, recorded at the user's request.

Keep code and maintenance work small. Preserve scientific meaning and reproducibility.

1. **Inspect evidence.** Read project rules, specifications, current code, reference code, and saved results. Separate proposals from verified behavior.
2. **Define terms.** Agree on each concept before designing its interface.
3. **Describe the desired result.** Start with user commands, outputs, and simple diagrams. Work backward toward code.
4. **Clarify important choices.** Give short options with benefits and costs. Infer obvious answers from existing decisions.
5. **Record decisions.** Save confirmed choices in Markdown and update related specifications.
6. **Design for reuse.** Specify files, signatures, and configuration fields. Share useful computations; keep method-specific behavior and state separate.
7. **Plan in order.** A plan contains phases. A phase contains stages. A stage contains atomic steps. Name the tool for each step.
8. **Implement and check in stages.** Follow the steps in order. Check scientific rules, saved-state recovery, and one complete command flow before broader execution.
9. **Remove replaced code.** Remove old code after replacement checks pass. Preserve specifications, detailed mathematics, required artifacts, and historical evidence.
10. **Review and simplify.** Review the diff, update outdated documents, and list remaining work. Remove unnecessary files and duplicate execution paths.
