---
date: 2026-09-16
status: completed
source_structure: ../structure/structure-kaggle-output-publisher.md
---

# Small atomic steps for Kaggle output uploads

The agent works in the local `tsad-lib` checkout only. The GPU server is not
edited by this plan.

## Phase 1: one batch has one meaning

### Stage 1.1: define completed output

- [x] Add a failing contract for a valid completed stage.
- [x] Run the focused contract and see it fail.
- [x] Add a reader for `completion.json`.
- [x] Run the focused contract again.
- [x] Add a failing scenario for a missing artifact hash.
- [x] Run the focused contract and see it fail.
- [x] Reject that stage as incomplete.
- [x] Run the focused contract again.

### Stage 1.2: define upload state

- [x] Add a failing scenario for an empty ledger.
- [x] Run the focused contract and see it fail.
- [x] Add the smallest ledger schema.
- [x] Run the focused contract again.
- [x] Add a failing scenario for a repeated stage hash.
- [x] Run the focused contract and see it fail.
- [x] Exclude the repeated stage from the next batch.
- [x] Run the focused contract again.
- [x] Add a failing scenario for a changed hash at the same stage identity.
- [x] Run the focused contract and see it fail.
- [x] Return a clear conflict instead of replacing the old record.
- [x] Run the focused contract again.

### Stage 1.3: define unique batches

- [x] Add a failing assertion that two batch IDs differ.
- [x] Run the focused contract and see it fail.
- [x] Build each ID from UTC time and the batch content hash.
- [x] Run the focused contract again.
- [x] Add a failing assertion for a batch manifest.
- [x] Run the focused contract and see it fail.
- [x] Write stage identities and hashes into the manifest.
- [x] Run the focused contract again.

## Phase 2: one upload is safe

### Stage 2.1: collect and stage

- [x] Add a failing scenario for a partial stage beside a completed stage.
- [x] Run the focused contract and see it fail.
- [x] Copy only the completed stage into a temporary batch directory.
- [x] Run the focused contract again.
- [x] Add a failing assertion that the batch contains dataset metadata.
- [x] Run the focused contract and see it fail.
- [x] Write minimal `dataset-metadata.json` with the configured dataset ID.
- [x] Run the focused contract again.

### Stage 2.2: lock and upload

- [x] Add a failing scenario for a held upload lock.
- [x] Run the focused contract and see it fail.
- [x] Add one non-blocking file lock around the upload command.
- [x] Run the focused contract again.
- [x] Add a failing scenario that records the exact Kaggle arguments.
- [x] Run the focused contract and see it fail.
- [x] Call `kaggle datasets version -p BATCH -m MESSAGE -q` with an argument list.
- [x] Run the focused contract again.

### Stage 2.3: publish state

- [x] Add a failing scenario for a Kaggle command that exits nonzero.
- [x] Run the focused contract and see it fail.
- [x] Keep the ledger unchanged after that failure.
- [x] Run the focused contract again.
- [x] Add a failing scenario for a successful Kaggle command.
- [x] Run the focused contract and see it fail.
- [x] Write the ledger through a temporary file and atomic rename.
- [x] Run the focused contract again.

## Phase 3: time makes repeated checks

### Stage 3.1: make one-shot execution

- [x] Add a failing CLI contract for a missing `KAGGLE_DATASET`.
- [x] Run the focused contract and see it fail.
- [x] Reject the call before scanning or uploading.
- [x] Run the focused contract again.
- [x] Add a one-shot command that returns the publisher status.
- [x] Run the focused contract again.

### Stage 3.2: make the tmux loop

- [x] Add a failing Bach assertion for the fixed 1,800-second interval.
- [x] Run the Bash contract and see it fail.
- [x] Pass the interval to the upload loop.
- [x] Run the Bash contract again.
- [x] Add a failing Bach assertion for a colliding tmux session.
- [x] Run the Bash contract and see it fail.
- [x] Reject the second session.
- [x] Run the Bash contract again.

### Stage 3.3: keep versions separate

- [x] Add a failing integration scenario with two completed stages.
- [x] Run the focused integration contract and see it fail.
- [x] Upload the first stage in batch one.
- [x] Create the second batch with only the second stage.
- [x] Run the focused integration contract again.
- [x] Add a retry scenario after a failed first upload.
- [x] Run the focused integration contract and see it fail.
- [x] Retry the same unrecorded stage without creating a duplicate ledger row.
- [x] Run the focused integration contract again.

## Phase 4: test and document

### Stage 4.1: test local rules

- [x] Keep all publisher cases inside the existing release-contract test budget.
- [x] Run the focused publisher contract.
- [x] Run all sibling pytest tests with `tsad-lib/.venv/bin/python`.

### Stage 4.2: test process boundaries

- [x] Add the publisher wrapper to the existing Bash contract.
- [x] Run `bash tests/launchers.test.sh`.
- [x] Check shell syntax with `bash -n`.

### Stage 4.3: write the operator story

- [x] Document `KAGGLE_DATASET` and the server credential location.
- [x] Document the tmux start command.
- [x] Document the tmux attach command.
- [x] Document how to inspect the ledger and failed batch.
- [x] Document that the full wet run needs durable output storage.
- [x] Run the final local test commands.

## Completion rule

The plan is complete only when the local tests prove that two publisher cycles
cannot overlap, a failed upload can retry, a successful upload cannot repeat a
stage, and no secret is written to an artifact.
