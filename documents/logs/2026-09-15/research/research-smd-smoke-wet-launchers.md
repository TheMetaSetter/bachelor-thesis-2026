---
date: 2026-09-15
status: complete
topic: Two SMD launchers for four GPUs
revision: 61cb84bb (main repository; working tree has existing changes)
---

# The experiment needs more than two launch commands

Two short Bash scripts can start this experiment. They cannot yet make the current sibling runtime satisfy the whole matrix. The missing work includes phase reuse, online metrics, several baseline runners, CUDA validation, and safe concurrent writes.

This is local research. The connection file in `tsad-lib/gpu-server.txt` was read. Its password is not copied here. The supplied server snapshot shows four RTX 3070 GPUs. The claimed 44 CPU cores, remote checkout, RAM, disk, CUDA packages, CPU quota, and GPU availability have not been checked on the server.

## What happens today

The CLI calls `api.run()`, which expands requests and calls `ExperimentRunner.run_request()`. A THESIS request trains Stage A, initializes memory, trains Stage B, fits a threshold, scores the test split, and then runs one online variant. It stores these outputs in one O–A directory.

Running O2-A0, O2-A1, and O2-A2 therefore trains O2 three times. These calls do not share one offline checkpoint. A matrix launcher must separate the offline result from its three online consumers.

The final metric call still uses the offline `scores` variable. Online records are written, but do not supply the final metric table. Separate online metric rows must come from the causal online stream.

The online projector can change during adaptation. However, the next score calls `stage_b()`, whose reconstruction path does not use that projector. `_verify_nonempty_pnn()` only checks whether the input and bank have elements. The buffer does not trigger verification at capacity, and the hard-old update has no interval guard. These observations are runtime gaps against the matrix stories; parameter-change tests alone cannot certify A1/A2.

## Evidence that changes the design

Paths below are relative to their named repository.

| Owner and location | Observed behavior | Consequence |
| --- | --- | --- |
| sibling `src/tsad/cli.py:18` | No device or phase argument | A shell loop cannot select CUDA or reuse an offline phase through this CLI |
| sibling `src/tsad/config.py:19` | CUDA regex contains an extra backslash | A mocked available `cuda:0` is rejected |
| sibling `src/tsad/experiment.py:349` | Accepts THESIS, Isolation Forest, and ready TSLib models | Matrix-only baselines are missing |
| sibling `src/tsad/experiment.py:363` | Isolation Forest uses window size 2 | Does not use the matrix window size 20 |
| sibling `src/tsad/experiment.py:388` | Non-THESIS accelerated runs are rejected | Do not advertise GPU baseline execution from catalog readiness |
| sibling `src/tsad/experiment.py:421` | One call trains offline and executes one A variant | Offline training is repeated across A variants |
| sibling `src/tsad/experiment.py:533` | Metrics consume offline scores | Online results need a separate evaluation path |
| sibling `src/tsad/experiment.py:585` | Every call writes the shared root report | Concurrent CLI calls can replace each other's report |
| sibling `src/tsad/experiment.py:627` | Resume checks manifest identity and metric rows | It does not verify every checkpoint hash or resume a partial stage |
| sibling `src/tsad/artifacts.py:14` | JSON is written directly to its final name | An interruption can leave a partial file |
| sibling `src/tsad/models/thesis_online.py:98` | Scores use `stage_b()` | Adapted projector is absent from this score path |
| sibling `src/tsad/models/thesis_online.py:205` | PNN verification checks nonempty tensors | This is not prototype verification or a verified point mask |
| sibling `src/tsad/models/thesis.py:47` | Direct routing reconstructs retrieved continuous states | Online adaptation needs an explicit, tested inference connection |
| sibling `src/tsad/variants.py:33` | O0 and O2 use the same three losses | The table's intended ablation remains ambiguous |
| sibling `src/tsad/models/thesis.py:194` | Stage A loads all training windows and always adds contrastive loss | A catalog edit alone cannot change the ablation; batch size does not bound this allocation |
| sibling `src/tsad/matrix.py:128` | Several gates are hard-coded ready | Gate status must be derived from evidence |
| main `scripts/benchmarks/generate_remaining_smd_benchmark_configs.py:118` | Wet defaults are Stage A 25, Stage B 5, RedLamp 30 | Useful proposed wet values, not current sibling defaults |
| main generator `:315` | Training stride is 1 | Conflicts with sibling offline stride 20 |

The main repository has baseline sources under `src/baselines/neural/redlamp.py`, `src/baselines/online/candi.py`, `src/baselines/online/m2n2.py`, `src/baselines/traditional/stumpy_channel_ab.py`, and `src/baselines/traditional/kmeans_ad.py`. Their existence does not prove that they can be copied without dependencies. A separate adapter audit must establish that boundary before implementation.

## The matrix tells two different stories

Its total table gives 588 offline and 1,176 online units: 1,764 units. Later instructions still say 504 and 924. Its resource section still targets six GPUs. The current request replaces that resource target with four GPUs and 44 available CPU slots.

The O0 flow includes two-view contrastive loss while its text excludes point-level contrastive loss. The user has defined these names as the same computation. O2 must keep exactly reconstruction, classification, and two-view contrastive losses, with no BPSL and direct routing. O0's distinction needs a decision before a scientific run.

The matrix baselines are RedLamp, CANDI, M2N2, Stumpy, KMeansAD, and Isolation Forest. The sibling ready catalog instead has Isolation Forest and 15 TSLib models. These are different sets. A pending question asks whether the latter should be added. The 1,764 count applies only to the named matrix.

## The hardware story

At launch, read index, UUID, name, and memory from `nvidia-smi`. Retain the index for the report. Assign each worker by its full GPU UUID. Under one UUID in `CUDA_VISIBLE_DEVICES`, the worker uses logical `cuda:0`. Probe that mapping in a fresh child before training; do not equate a container's displayed index with a PyTorch ordinal. Preserve the container's allowed-device boundary. [NVIDIA CUDA environment variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

Read the allowed CPU IDs with `os.sched_getaffinity(0)`. Read the container quota as well. A list of 44 allowed IDs does not prove a quota of 44 CPUs. Pin workers to disjoint subsets of the allowed IDs. [Linux CPU affinity](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html).

One server-side tmux session keeps the coordinator running after SSH disconnects. It does not survive server reboot, container deletion, or lost storage. [tmux documentation](https://github.com/tmux/tmux/wiki/Getting-Started).

A nonblocking lock prevents a second coordinator from using the same experiment root. Lock behavior must be checked on the actual filesystem; NFS and CIFS may differ. [flock manual](https://man7.org/linux/man-pages/man1/flock.1.html).

## The test story

A fresh collection found 28 pytest tests. This was collection only, not a full test pass. A focused mocked CUDA probe reproduced the rejection of `cuda:0`.

Bach can mock shell commands and compare calls. It requires Bash 4.3 or newer. It cannot intercept absolute executable paths or shell redirections automatically. The shell tests must mock the launch function before invoking it and use temporary paths. Real process interruption and filesystem checks belong in the existing Python integration contracts. [Bach documentation](https://bach.sh/).

## Decisions still needed

The user resolved loss identity during this research: O0 uses reconstruction and classification; O1 adds BPSL; O2 adds the existing two-view contrastive loss without BPSL. The code still needs that change. Baseline scope remains pending. The proposed launcher design is independent of that answer; final counts must wait for it.

No runtime code, scripts, tests, server files, or experiment outputs were changed by this research.
