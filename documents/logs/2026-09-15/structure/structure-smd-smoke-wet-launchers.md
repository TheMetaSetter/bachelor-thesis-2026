---
date: 2026-09-15
status: proposed; contract questions pending
source_plan: ../plan/plan-smd-smoke-wet-launchers.md
---

# The order in which the launchers become trustworthy

The agent first defines a job, then makes its output safe, then starts several jobs, and finally proves the experiment on the server. Design D supplies exactly two production Bash entry points and one shared Python execution path.

## Phase 1: every job has one meaning

### Stage 1.1: settle the experiment contract

Read the matrix beside the variant catalog and generator. Resolve the pending method scope. Apply the user's confirmed O0/O1/O2 loss distinction. Record wet epochs and training stride. Tools: Markdown, source inspection. Complete when no scientific choice is implicit.

### Stage 1.2: build the manifest

Expand the resolved method list across entities and seeds. Give every cell one identity, resource class, range, and dependency list. Tools: `matrix.py`, Python, existing release contract test. Complete when counts and unique paths are proven.

### Stage 1.3: check runnable methods

Check actual runner capabilities, CUDA parsing, online scoring, adaptation gates, checkpoint reuse, and metric provenance. Tools: existing public, baseline, and THESIS tests. Complete when each cell is runnable or explicitly blocked. A blocked scientific path prevents the real smoke and wet stages.

## Phase 2: completed stages survive interruption

### Stage 2.1: separate phase ownership

Train each offline variant once. Start A0/A1/A2 from the same verified offline artifact with fresh online state. Tools: `experiment.py`, runner APIs, existing THESIS and public tests. Complete when phase identities and online metrics are distinct.

### Stage 2.2: publish complete artifacts

Write temporary stage files, verify them, and publish completion last. Tools: `artifacts.py`, local filesystem, existing release test. Complete when a failed write cannot become a completed result.

### Stage 2.3: resume verified work

Hold one campaign lock. Reuse only matching completed stages. Preserve partial output for inspection. Tools: file locks, hashes, existing release test. Complete when a restart repeats only incomplete work and rejects mismatches.

## Phase 3: workers stay inside their resources

### Stage 3.1: discover the rental's resources

Probe allowed GPU UUIDs, CPU IDs, CPU quota, RAM, disk, and the installed runtime. Tools: `nvidia-smi`, Python, Linux resource files. Complete when four single-GPU probes succeed and the 44-slot budget is supported.

### Stage 3.2: run fixed queues

Assign four GPU queues and two CPU queues. Wait at the offline barrier. Collect every child exit status. Tools: Python subprocesses and affinity. Complete when fake jobs prove allocation, failure handling, and dependency order.

### Stage 3.3: expose two Bash scripts

Both scripts start the shared coordinator under server-side tmux. Tools: Bash 4.3+, Bach, tmux. Complete when argument quoting, mode selection, session collision, and child failures pass tests.

## Phase 4: earn permission to start the wet matrix

### Stage 4.1: stress the boundaries

Interrupt fake writers and workers. Exercise duplicate launch, bad hashes, full disk, hidden GPUs, nonzero GPU IDs, and sparse CPU IDs. Tools: pytest and Bach. Complete when 28 pytest cases plus two Bach cases pass and no final artifact is mistaken for partial work.

### Stage 4.2: run the real smoke

Use the approved remote checkout. Run O2-A0 first, then the remaining smoke cells. Disconnect and reconnect tmux while a job runs. Tools: SSH, tmux, the smoke script, artifact verifier. Complete when all selected methods and variants have real training/evaluation evidence.

### Stage 4.3: launch and audit wet results

Validate the smoke evidence and frozen wet manifest. Run the wet script only on explicit execution authorization. Tools: the wet script, coordinator, metrics collector, W&B. Complete when all expected logical cells have verified results or explicit failure records; claim the experiment complete only when none are missing or failed.

Phase 2 depends on Phase 1. Phase 3 depends on the stage contract from Phase 2. Real execution in Phase 4 depends on every scientific gate, not only launcher tests.
