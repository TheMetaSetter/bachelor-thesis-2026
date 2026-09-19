---
date: 2026-09-16
status: complete
topic: Upload completed tsad outputs to Kaggle every 30 minutes
---

# The output uploader story

The GPU server runs long experiments. Every 30 minutes, a small publisher
should copy finished output batches to one Kaggle dataset. Two uploads must
not run at the same time. The publisher must never mark a batch as uploaded
before Kaggle accepts it.

## What exists today

The sibling library already writes output folders under `outputs`. A completed
stage has `completion.json` with hashes. `ArtifactStore` writes files through a
temporary file and rename. It also has a lock for one stage
(`tsad-lib/src/tsad/artifacts.py:117-177`).

`ExperimentRunner` writes run artifacts and supports resume checks
(`tsad-lib/src/tsad/experiment.py:449-465` and `:811-860`). The Bash launchers
already use tmux (`tsad-lib/scripts/run_smd_smoke.sh:23-33`).

The library has no Kaggle publisher, upload ledger, upload lock, or upload
timer. `pyproject.toml` has no Kaggle dependency. This is good for a small
design: call the installed `kaggle` command from one short Python publisher.

Kaggle documents `kaggle datasets version -p FOLDER -m MESSAGE` for a new
dataset version. The upload folder must contain dataset metadata. Kaggle also
accepts credentials through `~/.kaggle/kaggle.json` or environment variables.

Sources:

- [Kaggle dataset version command](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md)
- [Kaggle CLI credentials](https://github.com/Kaggle/kaggle-cli/blob/main/docs/configuration.md)

## The important boundary

The experiment writes files. The publisher reads them. The publisher does not
change model code, checkpoints, or experiment results.

It accepts only a stage whose `completion.json` is valid. It copies that stage
to a fresh upload batch. It gives every batch a unique ID. It uploads the
batch. Only after success does it record the batch hash in
`kaggle_upload_state.json`.

The lock protects time overlap. The hash ledger protects duplicate content.
The completion file protects unfinished runs.

## Six possible designs

| Design | Story | Main cost |
| --- | --- | --- |
| A. tmux timer plus one-shot publisher | tmux wakes a small publisher every 1,800 seconds | It needs one live tmux session |
| B. cron plus one-shot publisher | cron starts the publisher every 30 minutes | cron may be absent in the container |
| C. systemd timer | systemd starts and restarts the publisher | systemd may be absent and adds setup work |
| D. publisher thread inside the coordinator | the running experiment sleeps and uploads | an experiment crash also stops uploads |
| E. file watcher plus timer | filesystem events wake an uploader | event logic is more code than needed |
| F. external CI or cloud scheduler | another service starts each upload | it needs network setup and another secret boundary |

## Evidence-based choice

Use this MECE decision tree:

1. Must the design work inside the current rented container? Keep B, C, and D
   only if the service is known to exist. It is not known to exist.
2. Must the uploader stay independent from model training? Remove D.
3. Must the design remain inside the existing tmux workflow? Keep A.
4. Must it avoid a new external service? Remove E and F.

The selected design is A: a tmux loop calls one one-shot publisher.

## Three challenges to the choice

First, cron is shorter. It is shorter only when cron is already installed and
running. The current server facts do not prove that. tmux is already part of
the launch flow, so A has fewer environment assumptions.

Second, the in-process thread needs no new script. It couples uploads to the
experiment coordinator. A broken coordinator could then stop both training and
backup. A keeps the publisher independent.

Third, a simple timestamp is enough to avoid duplicate uploads. It is not
enough after a retry or clock change. A content hash ledger gives one stable
answer: a completed batch is either recorded as uploaded or it is not.

## Remaining uncertainty

Kaggle dataset versions are snapshots. “No overlap” therefore has two meanings.
This plan guarantees no time overlap and no duplicate completed stage in the
publisher ledger. It also puts each new stage in a unique batch directory. If
the user means that every Kaggle version must contain a full cumulative
snapshot, the staging step will copy all verified stages instead. That choice
does not change the lock, hash, or retry rules.

No code changed during this research.
