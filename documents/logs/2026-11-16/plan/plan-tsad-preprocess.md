# Plan: tsad preprocess

Date: 2026-11-16
Status: initial planning record. Later decisions replaced parts of this plan. The standalone preprocessing script and all required data processing are complete. The installed `tsad preprocess` command remains pending.

## Current status

1. Use [the confirmed decisions](decisions-tsad-preprocess.md) for current behavior. One self-contained `scripts/preprocess.py` implements the five readers and shared processing flow.
2. All required ICCAD, SMD, IOPS, SWaT, and NASA data is processed. NASA excludes P-2 and T-10. The original source folders were removed after checks.
3. Split rounding, training-only scaling, approved missing-value rules, labels, versioning, and saved-output checks are implemented. The recorded temporary suite reached 46 tests.
4. The script module documentation replaces the proposed separate `documents/specs/data-preprocessing.md`. Built-in profiles replace required external default profiles.
5. The checklist below records the initial plan. Unchecked items do not mean that all corresponding behavior is missing. Do not mark old inspection steps complete without evidence that each step was performed.
6. Remaining work includes the installed command, updates to the broader CLI specification, source provenance, and runtime window checks.

## Goal

Provide a CLI command that preprocesses local data. Use the agreed decisions to update the specifications before implementation.

The earlier task list is a backlog. Start with the data and CLI work needed for this command. Complete each phase and its checks before the next phase.

## Evidence and instruction issues

1. `AGENTS.md` requires ordered phases, stages, atomic steps, and tools.
2. Both preference files require simple code, explicit inputs, and versioned data outputs.
3. `src/data/api.py` builds data settings for current loaders. Its presence does not establish a standalone preprocessing command.
4. `src/data/cleaning.py` validates sequence contracts. Its name does not establish missing-value repair.
5. `documents/specs/config-profiles-flow.md` separates profile settings from protocol validation rules.
6. `documents/logs/2026-11-10/research/research-local-datasets.md` records incomplete exploration. Later decisions select one Console entity, 5XX count features, and five-minute interval overlap labels.
7. `documents/specs/desired-cli-20261109.md` does not specify `tsad preprocess`.
8. Instruction links to `prompts/1_research_prompt.md` and `prompts/2_plan_prompt.md` are missing. The matching files exist under `prompts/primitive_prompts/`.
9. The earlier output hierarchy needs review against the current maximum path depth. Do not copy that hierarchy into preprocessing outputs without review.

## Initial decisions

| Decision | Option | Pros | Cons |
| --- | --- | --- | --- |
| First dataset | A: ICCAD only | Focused work; uses the current investigation | Other datasets need later support |
| First dataset | C: ICCAD, SMD, IOPS, SWaT, and NASA; selected | Checks several formats within a fixed scope | Needs rules and checks for five datasets |
| First dataset | B: all local time-series datasets | Broad coverage | Several formats and meanings remain unresolved |
| Output boundary | A: cleaned data in a common format | Reusable across experiments; keeps split and scaling choices separate | Training needs a later preparation step |
| Output boundary | B: train, validation, and test data with fitted scaling; selected | Direct input for training | Split rules, missing-value rules, and scaler state must be settled now |
| Command input | A: dataset and data profile; selected | Can prepare data before an experiment exists | Experiment commands must later reference the prepared version |
| Command input | B: experiment name | Reuses experiment settings | Requires more of the proposed experiment CLI first |

Confirmed: the output includes train, validation, and test data with fitted scaling (2B). Define the default split rules and scaler state now. Fit the scaler on training data only. The later decision file and script define robust scaling, saved state, split defaults, and integer rounding.

Confirmed: option 1C covers ICCAD, SMD, IOPS, SWaT, and NASA. Option 3A selects a dataset and data profile directly. The script uses the separate SMD machine files; equivalence with the earlier aggregate representation remains unverified. See [today's decisions](decisions-tsad-preprocess.md).

Question 3 asks where the command gets its settings. Option A names the dataset and its data profile directly. Option B names an experiment, then reads that experiment's dataset and data profile selections. Both options can produce the output selected in 2B. Neither option requires model training.

## Phase 1: agree on the scope

### Stage 1.1: select the first behavior

1. [x] Select the first dataset scope: ICCAD, SMD, IOPS, SWaT, and NASA. Tool: user decision.
2. [x] Select the preprocessing output boundary: 2B, including default split rules and scaler state design now. Tool: user decision.
3. [x] Select the command input: dataset and data profile (3A). Tool: user decision.
4. [x] Record the answers in this plan. Tool: Markdown editor.

### Stage 1.2: check the scope

1. [ ] List the exploration gaps that affect the selected behavior. Tool: existing dataset report.
2. [ ] Mark unrelated backlog items as later work. Tool: Markdown editor.
3. [ ] Check that the scope supports one complete CLI example. Tool: document review.

## Phase 2: establish the data rules

### Stage 2.1: inspect existing behavior

1. [ ] Read `python_semantics.md`. Tool: file reader.
2. [ ] Trace the current dataset loaders and their callers. Tool: `rg` and file reader.
3. [ ] Inspect the relevant Time-Series-Library reference loader. Tool: file reader.
4. [ ] Inspect the available CLI entry points and package settings. Tool: `rg` and file reader.
5. [ ] Complete each exploration check required by Phase 1. Tool: available bounded data readers.
6. [ ] Record which existing code can be reused. Tool: Markdown editor.

### Stage 2.2: settle the scientific choices

1. [ ] Define the entity selection rule. Tool: source evidence and user decision where needed.
2. [ ] Define timestamp handling. Tool: source evidence and user decision where needed.
3. [ ] Define missing-value handling. Tool: source evidence and user decision where needed.
4. [ ] Define the label mapping, if labels are included. Tool: annotation evidence and user decision where needed.
5. [ ] Define split rules, if splitting is included. Tool: source evidence and user decision where needed.
6. [ ] Define scaling rules, if scaling is included. Tool: data profile and existing definitions.
7. [ ] Check that test data cannot fit a transform or select preprocessing settings. Tool: contract review.

Stop at an unresolved scientific choice. Present only options that change the result.

## Phase 3: update specifications

### Stage 3.1: record the accepted behavior

1. [ ] Add the accepted command to `documents/specs/desired-cli-20261109.md`. Tool: Markdown editor.
2. Superseded: use the script module documentation as the usage and interface specification. No separate preprocessing specification is required. Tool: file reader.
3. Superseded: retain the terminology-change note in the confirmed decision file. Tool: Markdown editor.
4. [ ] Define input arguments and profile fields. Tool: specification review.
5. [ ] Define output format and directory ownership. Tool: specification review.
6. [ ] Define data version identity and provenance. Tool: specification review.
7. [ ] Define repeat execution and partial-failure behavior. Tool: specification review.
8. [ ] Define input and output checks under the data and stream protocol. Tool: specification review.
9. [ ] Update conflicting data sections in the older config plan. Tool: Markdown editor.
10. [ ] Review one example command and its expected files with the user. Tool: document review.

### Stage 3.2: prepare the implementation detail

1. [ ] Name each affected file and its responsibility. Tool: source review.
2. [ ] Define the required function signatures. Tool: specification editor.
3. [ ] Update the proposed tree for the selected scope. Tool: Markdown editor.
4. [ ] Check that no blocking decision remains. Tool: plan review.

Use simple functions with one CLI entry point unless the code review establishes another need. A design pattern is not required by default.

## Phase 4: implement the accepted command

### Stage 4.1: build one execution path

1. [ ] Implement the selected data transformation. Tool: Python and available data libraries.
2. [ ] Implement the input and output checks. Tool: Python.
3. [ ] Implement output persistence and provenance. Tool: the storage tools selected in Phase 3.
4. [ ] Implement the CLI entry point. Tool: the existing CLI framework, if suitable.
5. [ ] Add short help text and one example. Tool: CLI framework and Markdown editor.

Exact files and signatures must come from Phase 3. Do not choose them from this draft alone.

## Phase 5: verify and deliver

### Stage 5.1: check the result

1. [ ] Test one small input with known expected values. Tool: Pytest.
2. [ ] Test invalid input rejection. Tool: Pytest.
3. [ ] Test saved-data loading. Tool: the selected output reader.
4. [ ] Test repeat execution against the agreed policy. Tool: Pytest.
5. [ ] Check training-only fitting if scaling is included. Tool: Pytest.
6. [ ] Run one selected dataset through the complete command. Tool: local CLI.
7. Later policy: keep sources through preprocessing and saved-output checks, then remove them. All five source folders are now removed. Tool: source inventory and scoped checks.
8. [ ] Check CLI help. Tool: local CLI.
9. [ ] Run `git diff --check`. Tool: Git.
10. [ ] Review the changed files. Tool: Git diff and file reader.
11. [ ] Report the command, output location, checks, and remaining limits. Tool: Markdown response.

## Later backlog, in order

1. Finish exploration for datasets outside the first command scope.
2. Extend preprocessing through the same data contract.
3. Complete experiment defaults, smoke execution, and resume rules.
4. Complete method profiles and the shared encoder decision.
5. Implement experiment execution and reporting.
6. Verify W&B, Kaggle backup, and Cleandocs coverage.
7. Resolve notebook retention as a separate small task.

Hugging Face download remains a future feature. Do not add it as an implicit part of preprocessing.
