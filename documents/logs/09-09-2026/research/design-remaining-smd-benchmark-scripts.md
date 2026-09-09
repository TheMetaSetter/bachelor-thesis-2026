# Design: Remaining-SMD benchmark scripts

Design status: approved by anh in the current task.

## Scope

The scripts cover every SMD entity except `machine-1-6`, `machine-3-4`, and `machine-3-9` when no explicit entity selection is supplied.

The cloud run selects eight machines from the drift-ranked list: `machine-3-1`, `machine-3-5`, `machine-3-2`, `machine-3-11`, `machine-3-10`, `machine-1-3`, `machine-1-1`, and `machine-2-8`.

The scripts use seeds `6`, `8`, and `36`, window size `20`, and the existing test protocol.

The scripts run THESIS offline Stage A and Stage B, THESIS online test-time adaptation, RedLamp offline, traditional offline baselines, and online CANDI, M2N2, STUMPY, KMeansAD, and Isolation Forest.

RedLamp is the offline deep-learning baseline because the current codebase exposes CANDI and M2N2 as the online deep-learning adapters.

## Modes

Smoke mode uses THESIS Stage A `3` epochs and Stage B `2` epochs.

Smoke mode evaluates only the first `16` causal windows of the selected online test stream.

Wet mode uses THESIS Stage A `25` epochs and Stage B `5` epochs.

Wet mode evaluates one deterministic `2048`-point test subsequence per entity.

The subsequence maximizes the number of ground-truth anomaly points and breaks ties by choosing the earliest start index.

All methods and seeds for one entity reuse the same subsequence.

The labels define the requested evaluation subset only and remain unavailable to thresholding and adaptation decisions.

Because the subset is selected with ground-truth labels, these results are targeted event-window results and are not full-test estimates.

Smoke and wet modes therefore share the same event-containing subsequence policy.

Smoke RedLamp uses `5` single-stage epochs to keep the smoke flow short.

## Execution

The legacy shared launcher keeps two GPU workers for backward compatibility.

The cloud wrapper creates four GPU workers and two CPU workers through `tmux`.

The four GPU workers use `CUDA_VISIBLE_DEVICES=0`, `1`, `2`, and `3` with CPU masks `0-7`, `8-15`, `16-23`, and `24-31`.

The two CPU workers use CPU masks `32-37` and `38-43`.

Each worker completes offline training and evaluation before starting the dependent online runs for the same entity and seed.

The launcher can run in tmux or foreground mode and supports dry-run and skip-completed execution.

## Metrics

The report retains only run identity and these requested metrics: `VUS-PR@FPR-budget`, `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR`.

The three `VUS-PR@FPR-budget` values use budgets `0.1%`, `0.5%`, and `1%`.

`raw-FPR` maps to the existing unadjusted `fpr` value at the protocol threshold.

The report does not retain score arrays, predictions, labels, traces, or forward-pass outputs.

CANDI and M2N2 use the configured CUDA device for their backbone, adapter modules, optimizer tensors, and inference batches.

STUMPY, KMeansAD, and Isolation Forest remain CPU baselines because their current implementations do not expose a GPU execution path.

## Expected run counts

For `N` remaining entities, each seed produces `2` THESIS offline runs, `1` RedLamp offline run, `3` traditional offline runs, `6` THESIS online runs, and `5` online baseline runs.

With the standard SMD dataset, `N` is `25`, so the all-remaining matrix contains `450` offline runs and `825` online runs.

The eight-machine cloud matrix contains `144` offline runs, `264` online runs, and `408` total runs.

## Output

Generated configs and compact reports live below `outputs/benchmark/smd_remaining` or `outputs/benchmark_smoke/smd_remaining`.

The output hierarchy keeps entity, seed, method, phase, and variant as the smallest human-readable identity.
