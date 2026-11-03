# Phase 2 source-baseline audit plan

Goal: identify verified duplicate code and nonessential source features that could meet the 40-file, 100-lines-per-file, 4,000-line limits while preserving the required experiment matrix.

## Phase 1: establish audit inputs

- [x] Read the Phase 1 boundary and simplification lessons. Tools: `sed`, `rg`.
- [x] Count owned source files and lines with the agreed counter. Tool: `.venv/bin/python`.
- [x] Inventory `scripts/` by folder and counted lines. Tool: `.venv/bin/python`.

## Phase 2: verify repeated-code candidates

- [x] Check repeated function signatures and callers in source, tests, configs, and documents. Tools: AST, `rg`, file reads. Evidence: `detail/phase2-reexecution/source-baseline-audit.md`, “Repeated-body review”.
- [x] Classify each candidate as safe deduplication, contract-sensitive, or retained. Tool: Markdown evidence table.
- [x] Recount only verified duplicate code savings. Tool: the agreed counter rule. Result: 206 gross span-helper lines and 41 RedLamp helper lines are supported candidates; 347 lines of uncalled THESIS helper workflows are separately classified as retirement, not deduplication. The 712-line AST estimate is not counted as savings.

## Phase 3: assess script features against the experiment matrix

- [x] Mark each script group as required runtime, matrix generation/execution, report or analysis support, compatibility entrypoint, or one-off. Tools: Phase 1 boundary, specs, `rg`. Evidence: script-family inventory and retirement-candidate sections in the source audit.
- [x] Record affected callers, configs, tests, and documented commands for retirement candidates. Tools: `rg`, AST, file reads. Evidence: “Known callers affected by these candidates”; unresolved external commands and artifact consumers are stated as static-review limits.
- [x] Select and count a conservative current-matrix script set. Tool: `.venv/bin/python`. Result: 18 files, 6,672 lines, 15 files over 100 lines. This is a candidate support set, not a final target tree or proof of minimum file count.

## Phase 4: revise the proposal

- [x] Update the feature-pruning proposal with evidence, source actions, and retirement effects. Tool: Markdown edit.
- [x] Verify the source baseline and candidate set against all three limits. Tool: the agreed counter plus per-file counter. Result: target not met; baseline is 244 files/42,297 lines, and the 18-file candidate is 6,672 lines with 15 files above 100 lines.
- [x] Record unresolved contracts and static-review limits. Tool: Markdown review. Runtime parity, external callers, and artifact-consumer equivalence remain unverified; no source implementation changed and no tests were run.

## Phase 5: apply clarified runtime priorities

- [x] Record the user's selected runtime responsibilities in the feature proposal. Tool: Markdown edit.
- [x] Preserve baseline selection as unresolved until the retained methods are named; state the next decision needed. Tool: Markdown review.
