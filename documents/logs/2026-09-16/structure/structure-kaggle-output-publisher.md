---
date: 2026-09-16
status: proposed
source_plan: ../plan/plan-kaggle-output-publisher.md
---

# The order in which uploads become trustworthy

## Phase 1: one batch has one meaning

### Stage 1.1: define completed output

Read `completion.json` and its artifact hashes. Ignore temporary folders,
partial stages, and failed runs.

### Stage 1.2: define upload state

Read one ledger. A recorded stage hash is already sent. An unrecorded valid
stage is new. A changed hash is a failure, not a silent replacement.

### Stage 1.3: define unique batches

Give every batch a unique ID and a manifest. Put each batch in its own staging
directory. Never reuse a batch directory after a failed upload.

## Phase 2: one upload is safe

### Stage 2.1: collect and stage

Collect only new completed stages. Copy them to a temporary directory on the
same filesystem. Write metadata and the batch manifest.

### Stage 2.2: lock and upload

Take one non-blocking lock. Call the Kaggle CLI with an argument list. Release
the lock after success or failure.

### Stage 2.3: publish state

Write the ledger only after the CLI succeeds. Use an atomic file replacement.
Leave the staged batch after failure so the operator can inspect it.

## Phase 3: time makes repeated checks

### Stage 3.1: make one-shot execution

Expose a short Python command for one upload attempt. Return nonzero on a
missing credential, lock collision, invalid stage, or Kaggle failure.

### Stage 3.2: make the tmux loop

Call the one-shot command every 1,800 seconds. Keep the loop free of upload
rules. Refuse a colliding tmux session.

### Stage 3.3: keep versions separate

Use the ledger and unique batch folders. A later cycle cannot upload an earlier
stage again. A failed cycle can retry the same stage without marking it twice.

## Phase 4: test the story

### Stage 4.1: test local rules

Use temporary folders and fake commands to test discovery, hashing, and ledger
changes.

### Stage 4.2: test process boundaries

Use the existing Bash contract to test quoting, tmux session collision, fixed
interval arguments, and exit codes.

### Stage 4.3: write the operator story

Document credential checks, the first manual upload, log inspection, and safe
restart. Do not put secrets in the repository.
