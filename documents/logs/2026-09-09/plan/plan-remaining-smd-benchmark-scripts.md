# Plan: Remaining-SMD benchmark scripts

**Spec:** `documents/logs/09-09-2026/research/design-remaining-smd-benchmark-scripts.md`

## Phase 1: Lock the executable contract

### Stage 1.1: Inspect current runners and result schemas

1. Read the existing THESIS, traditional baseline, RedLamp, and online runner entrypoints.
2. Record the accepted config fields and output report locations.
3. Record the existing metric key names and threshold semantics.

Tools: `rg`, `sed`, `.venv/bin/python`.

### Stage 1.2: Define the remaining-entity matrix

1. Discover entity IDs from the dataset train files.
2. Validate matching test and test-label files.
3. Exclude the three completed entities.
4. Define the method, variant, seed, phase, and mode combinations.

Tools: Python `pathlib`, pytest.

## Phase 2: Add failing tests first

### Stage 2.1: Test entity discovery and counts

1. Create a temporary four-entity SMD-like dataset.
2. Assert that discovery excludes the three excluded IDs.
3. Assert that the matrix count formulas are correct.

Tools: pytest, `tmp_path`.

### Stage 2.2: Test smoke and wet config contracts

1. Assert smoke Stage A is `3` and Stage B is `2`.
2. Assert wet Stage A is `25` and Stage B is `5`.
3. Assert smoke online steps are `16` and wet online steps are unlimited.
4. Assert generated configs retain only the requested report metrics.

Tools: pytest.

### Stage 2.3: Test launcher and metric extraction

1. Assert dry-run output contains two GPU assignments.
2. Assert metric extraction maps `fpr` to `raw-FPR`.
3. Assert the exact phrase `VUS-PR@FPR-budget` appears in the report schema.

Tools: pytest, Bash.

## Phase 3: Implement config generation

### Stage 3.1: Implement dataset discovery

1. Add a small Python module for remaining-entity discovery.
2. Add explicit file validation and exclusion constants.
3. Add a command-line interface for dataset root, mode, and output root.

Tools: Python, YAML writer already used by the repository.

### Stage 3.2: Implement the run manifest and configs

1. Generate one compact data config per remaining entity and mode.
2. Generate THESIS offline configs for O0 and O1.
3. Generate RedLamp and traditional offline configs.
4. Generate THESIS online configs for A0, A1, and A2.
5. Generate CANDI, M2N2, STUMPY, KMeansAD, and Isolation Forest online configs.
6. Write one manifest containing config paths and minimal run identity.
7. Select one deterministic `2048`-point test subsequence per entity from ground-truth labels.
8. Put the selected half-open range into every online config and reuse it across all methods and seeds.
9. Set CANDI and M2N2 to CUDA while keeping non-adaptive online baselines on CPU.

Tools: Python, existing config builders where their contracts match.

## Phase 4: Implement execution and reporting

### Stage 4.1: Implement one GPU worker

1. Read the manifest.
2. Select entity indices assigned to the worker.
3. Run offline configs before dependent online configs.
4. Set `CUDA_VISIBLE_DEVICES` for the worker.
5. Skip a run only when its expected report already exists and `--skip-completed` is set.
6. Pass the configured device into CANDI and M2N2 constructors so their model tensors are placed on GPU.

Tools: Bash, tmux, `.venv/bin/python`.

### Stage 4.2: Implement the two-GPU coordinator

1. Start two workers with GPU indices `0` and `1`.
2. Wait for both workers.
3. Collect compact metrics after both workers finish.
4. Return a nonzero status when any worker fails.

Tools: Bash, tmux.

### Stage 4.3: Implement compact metric collection

1. Read only final metric JSON objects from completed runs.
2. Normalize offline and online metric key prefixes.
3. Emit the five requested metric fields.
4. Emit one JSON summary and one Markdown table.

Tools: Python, Markdown.

## Phase 5: Verify and document commands

### Stage 5.1: Run focused verification

1. Run the new pytest module.
2. Run existing benchmark config-generation tests.
3. Run `bash -n` on all new shell scripts.
4. Run generator preflight against a temporary fixture.
5. Run smoke dry-run without starting tmux.

Tools: pytest, Bash, `.venv/bin/python`.

### Stage 5.2: Record CLI commands

1. Add smoke preflight, dry-run, and execution commands.
2. Add wet preflight, dry-run, and execution commands.
3. Add the post-run metric collection command.
4. State intentional defaults and explicit overrides.

Tools: Markdown.
