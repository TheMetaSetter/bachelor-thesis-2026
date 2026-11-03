---
date: 2026-09-15
status: proposed; scientific contract questions remain open
related_research: ../research/research-smd-smoke-wet-launchers.md
---

# Two scripts, one execution path

The student starts `run_smd_smoke.sh`. Later, the student starts `run_smd_wet.sh`. Both proposed scripts call the same small Python coordinator. The scripts only locate the library, select the mode, and enter tmux. Python owns the manifest, worker allocation, stage dependencies, and result checks. The user confirmed O0 = reconstruction + classification, O1 = O0 + BPSL, and O2 = O0 + the existing two-view contrastive loss.

This is the selected design. It is a proposal, not implemented behavior. The user requested all four planning documents together and delegated the design choice, so this plan includes the stages and detailed steps without an intermediate approval loop. Scientific questions remain explicitly open.

## Six ways the story could work

| Design | How it runs | Main benefit | Main cost |
| --- | --- | --- | --- |
| A. Two serial Bash scripts | Each script loops through every cell | Smallest scheduler | Leaves three GPUs idle |
| B. Two Bash scripts with fixed queues | Shell owns four GPU queues and two CPU queues | Few dependencies | JSON, hashes, and recovery become awkward shell code |
| C. Two Bash scripts with GNU Parallel | Parallel assigns cells to worker slots | Good load balancing | Adds a tool and still needs dependency and artifact logic |
| D. Two thin Bash scripts with fixed Python queues | One coordinator partitions an immutable manifest | Reuses Python types and keeps shell tiny | Needs one coordinator module |
| E. Two Bash scripts with a SQLite task queue | Workers claim pending cells transactionally | Flexible retries and load balancing | Database state must agree with artifacts |
| F. Two Bash scripts submitting Slurm jobs | Slurm owns resource allocation and dependencies | Strong cluster scheduling | Requires a Slurm service not established on this rental |

## One decision framework: a MECE tree

The first branch asks whether a server scheduler is required. It is not established here, so F is excluded. The second branch asks whether execution is serial or concurrent. Four GPUs require concurrent work, so A is excluded. The third branch asks whether workers share a dynamic queue or receive fixed lists. Fixed lists meet this one-server need with fewer states, so C and E lose their main advantage. The last branch asks who owns hashes and structured results. Those objects already live in Python, so D is simpler than B across the whole system.

MECE means each branch separates alternatives without overlap at that decision. This tree judges the complete execution path, not the number of shell lines alone.

## Three challenges to D

First, perhaps a Bash-only solution is smaller. It is smaller only until it must parse manifests, check checkpoint identities, and merge reports. D keeps those actions beside their Python owners. The scripts remain easy to mock.

Second, fixed queues may finish unevenly. That is a real cost. Sort jobs by method and entity, then distribute each resource class round-robin. Four GPU queues receive comparable method mixes. Accept an idle tail for now. Add dynamic scheduling only after measured imbalance justifies it.

Third, one coordinator might fail midway. A shared queue would not repair a half-written checkpoint either. D uses one writer per stage, a held lock, staged output publication, and a completion manifest written last. Restart skips verified completed stages and reruns an incomplete stage from its last completed dependency. This avoids a second database of truth.

The final choice is D. A fourth check strengthens it: scheduler tests do not establish scientific correctness. Therefore the coordinator rejects missing methods, ambiguous contracts, and unverified online behavior before starting training.

## How the four GPUs and 44 CPU slots work

| Worker group | Count | CPU slots each | GPU allocation |
| --- | ---: | ---: | --- |
| GPU workers | 4 | 6 | One discovered UUID each |
| CPU workers | 2 | 8 | CUDA hidden |
| Coordinator and system reserve | 1 | 4 | None |
| Total | | 44 | 4 GPUs |

These are positions in the allowed CPU list, not hard-coded CPU IDs. Pin all child processes. Set OMP, MKL, OpenBLAS, NumExpr, NumBA, and PyTorch thread limits within each worker's allocation. Start data loaders with zero subprocess workers. This bounds oversubscription while preserving simple loops. Forty-four refers to usable logical CPU slots until topology is checked.

Use one model process at a time per GPU. The supplied names are RTX 3070, but query actual free VRAM and usable RAM before launch. Do not change batch size automatically after OOM: that changes the experiment. Stop that cell, retain its failure, and require a new resolved configuration for a different batch size.

The coordinator validates the whole manifest first. It runs the offline queues, waits for all exits, and checks every required artifact. Any offline failure blocks the online phase. Independent offline cells may finish. After a clean barrier, it runs online queues and writes one final report itself.

## Phase 1: give every job one meaning

Use Python, the existing matrix, and pytest. Resolve the baseline list and apply the confirmed loss identity. Freeze entities, seeds, ranges, epochs, strides, metrics, phase identities, and dependency hashes. Replace optimistic readiness text with evidence. This phase ends with a manifest that either passes or names a blocking cell.

The proposed base smoke scope is one entity (`machine-1-3`) and seed 6, with all matrix families: 7 offline and 14 online logical units. First run O2-A0 as the canary; reuse its completed stages when finishing this 21-unit smoke. THESIS uses 3/2 epochs and 4,096 causal windows. Other methods keep native flows and use the recorded smoke range. Wet uses 28 entities and seeds 6, 8, 36, giving 588 + 1,176 units for the original matrix.

Propose wet Stage A 25 and Stage B 5 from the main generator; propose RedLamp smoke 5 and wet 30 epochs. Record these as explicit experiment settings rather than inheriting sibling 1/1 defaults. The disputed training stride must be resolved before freezing the manifest. All methods share the same wet 2,048-point range per entity; labels select the range once and are hidden from inference/adaptation.

## Phase 2: make one stage safe to run again

Use the existing runners, ArtifactStore, pathlib, hashes, and pytest. Give offline and online separate identities and outputs. Write one stage into a temporary sibling directory. Publish it only after required files can be read and verified. Resume means reuse complete stages, not continue an arbitrary interrupted optimizer step.

The existing missing baseline adapters and online scientific gaps are launch prerequisites. This scheduler plan does not replace those methods with approximations. Complete a source-level adapter/online audit before authoring their implementation details. Until then, their manifest rows remain blocked.

## Phase 3: connect the two scripts to bounded workers

Use Bash, Bach, tmux, Python subprocesses, CPU affinity, and local file locks. Discover actual resources. Assign fixed worker lists. Preserve child exit codes. Stop scheduling on termination and wait for owned children. Let one coordinator merge reports after workers stop.

## Phase 4: prove the path before wet execution

Use existing pytest contracts, two Bach cases, fake child processes, and a real server smoke. First test failures and interruptions without training. Then run the canary and full smoke coverage under tmux. A wet run starts only when every gate has evidence for the same source and scientific contract.

Keep 28 pytest cases and add two Bach cases: at most 30 top-level cases across both frameworks. Extend existing contract tests with named scenarios. Count scenario rows separately in the report so the small case count does not hide the work done.

## Recovery and limits

Use a fresh campaign directory containing canonical `outputs/benchmark_smoke` or `outputs/benchmark` trees. Include the variant in the stage token, such as `O2_stage_a` and `O2_A1_evaluation`, to keep paths distinct without another directory layer. Old artifacts remain readable; do not rename or overwrite them.

Store an immutable manifest, per-stage results, short worker logs, and essential checkpoints. Write the final report once. Hash code, data, configuration, and upstream artifacts. Separate scheduling fields such as GPU assignment from scientific identity so a restart can use a different physical GPU while recording that change.

Use W&B local/offline logging when the server loses internet, with stable stage IDs. Local artifacts remain authoritative; later sync must not create duplicate stage identities. The original matrix targets 2,352 W&B runs. Adding TSLib methods changes that target and requires an explicit expansion.

The scripts must not upload code, store credentials, kill unrelated jobs, or delete incomplete outputs automatically. tmux protects against client disconnects; durable artifacts protect completed work after process loss. A lost rental disk cannot be repaired by tmux.

## Terminology

`point-level contrastive loss` and `two_view_contrastive_loss` remain one computation. `benchmark_smoke` and `benchmark` retain their names. O0/O1/O2 and A0/A1/A2 are THESIS/MUSTARD ablations only. `matrix cell` keeps its method/phase/variant/entity/seed meaning. `worker` and `campaign manifest` are new orchestration objects; neither is a new model or scientific variant.
