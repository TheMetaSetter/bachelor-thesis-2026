---
date: 2026-09-16
status: proposed
related_research: ../research/research-kaggle-output-publisher.md
---

# A small story for safe Kaggle uploads

The experiment keeps writing local outputs. A separate one-shot publisher
looks for completed stages. A tmux loop calls it every 30 minutes. The
publisher makes a unique batch, uploads it, and records success only after the
Kaggle command returns zero.

The publisher never stores the Kaggle key in source code, output files, or
logs. The operator provides `KAGGLE_DATASET` and the server's Kaggle
credentials.

## Selected design

Use one Python module and one Bash wrapper:

- `src/tsad/kaggle_publish.py` owns discovery, hashes, staging, locking, and
  the Kaggle command.
- `scripts/run_kaggle_upload_loop.sh` owns tmux and the 1,800-second wait.
- `tests/test_release_contracts.py` receives one compact upload contract with
  several named scenarios.

Do not add a Kaggle Python dependency. Use `subprocess.run` to call the
installed `kaggle` command. This keeps local unit tests fast and lets the
server choose its Kaggle CLI version.

## Phase 1: freeze the upload contract

The first phase gives every upload one clear identity. A batch contains only
verified completed stages that are not in the local ledger. The batch ID is a
UTC timestamp plus a short content hash. The ledger stores batch ID, stage
hashes, Kaggle dataset name, and success time.

The phase ends when the contract can answer three questions: what is ready,
what was already sent, and what must be retried.

## Phase 2: build one safe upload

The publisher scans `outputs`, verifies `completion.json`, copies selected
stages into a fresh batch directory, writes `dataset-metadata.json`, and calls
`kaggle datasets version`. It holds one non-blocking lock for the whole upload.

The phase ends when a second publisher cannot overlap the first one, and a
failed Kaggle command leaves the ledger unchanged.

## Phase 3: add the 30-minute loop

The Bash script checks that the Kaggle dataset name and credentials exist. It
starts one tmux session and calls the one-shot publisher on a fixed 1,800-second
schedule. It refuses a second session. The Python publisher remains the only
place that knows upload rules.

The phase ends when SSH disconnection does not stop the loop and a second loop
cannot start.

## Phase 4: verify and document the operation

Tests use a temporary output tree, a fake Kaggle executable, and a fake clock
or one-shot function. The tests cover successful upload, empty output, failed
upload, retry, duplicate prevention, lock collision, and command quoting.

Keep the current 30 top-level test-case limit. Extend one existing release
contract and the existing Bash launcher contract. Put the smaller cases inside
those contracts instead of creating a new test family.

The phase ends with a local test pass and a short server command story.

## Verification command

Run from the sibling library:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
bash tests/launchers.test.sh
```

Do not run a real Kaggle upload in the test suite. Run one manual upload only
after the server operator checks the dataset name, credentials, disk space,
and persistent output location.
