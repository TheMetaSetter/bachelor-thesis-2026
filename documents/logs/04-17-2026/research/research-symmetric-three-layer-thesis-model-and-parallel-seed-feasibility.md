---
date: 2026-04-17 21:53:58 +07+0700
researcher: Artificial Intelligence Agent
git_commit: 70e8186778862f559bff6c73af31dd2ad5a25327
branch: dev
repository: bachelor-thesis-2026
topic: "Changing the offline thesis model to three linear layers per encoder and head, and assessing three parallel seed runs on one RTX3090"
tags: [research, time-series, anomaly-detection, multi-task, architecture, experiment-orchestration]
status: complete
last_updated: 2026-04-17
last_updated_by: Artificial Intelligence Agent
---

# Research: Changing the Offline Thesis Model to Three Linear Layers per Encoder and Head, and Assessing Three Parallel Seed Runs on One RTX3090

**Date**: 2026-04-17 21:53:58 +07+0700  
**Researcher**: Artificial Intelligence Agent  
**Git Commit**: 70e8186778862f559bff6c73af31dd2ad5a25327  
**Branch**: dev

## Research Question

The research question is whether the current offline thesis model already supports changing the encoder, reconstruction head, and classification head from two linear layers to three linear layers each, with symmetric depth across all three modules, and whether the current repository supports running three experiments with three different random seeds in parallel on a single NVIDIA RTX3090 graphics processing unit.

## Summary

The current repository does not expose encoder depth or head depth as configuration parameters. In the active offline thesis model, the encoder is hard-coded as a two-linear-layer multilayer perceptron, the reconstruction head is hard-coded as a two-linear-layer decoder, and the classification head is hard-coded as a two-linear-layer multilayer perceptron. Therefore, changing the model to a three-linear-layer symmetric design is not currently a configuration-only operation. It would require direct code edits in the active model file.

The current experiment orchestration surface is also single-run oriented. One experiment configuration contains one scalar seed, one output directory, one checkpoint directory, and one Weights and Biases run name. The training entrypoint builds one model, one optimizer, one trainer, and one scheduler for a single process. The repository does not contain a built-in multi-seed launcher, distributed training path, multiprocessing experiment runner, or device-partitioning mechanism for concurrent runs on a single graphics processing unit.

The active server configuration already targets a single long run on one CUDA device with batch size 512 and eight data loader workers. The repository contains no profiling artifact, memory estimator, or concurrency validation showing that three such runs have been tested concurrently on one RTX3090. Consequently, the codebase as it exists today does not provide implementation evidence that three parallel full experiments on one RTX3090 are a supported or validated execution mode.

## Detailed Findings

### Data Preparation

- The active full server configuration uses the Server Machine Dataset through a dedicated SMD data configuration file. The data configuration fixes:
  - window size `100`
  - stride `10`
  - batch size `512`
  - number of data loader workers `8`
  - validation split ratio `0.2`
- The active experiment configuration points to CUDA execution and a 300-epoch offline multitask run.
- The design documents continue to describe the thesis-facing hidden-state contract as a fixed-length multivariate window pipeline with window length 100 and a hidden representation shared across prototype branches.

### Modeling and Training

#### Current encoder depth

The active encoder in the offline thesis model is defined by `MultitaskWindowEncoder`. Its sequential network is currently:

1. `Linear(input_dim, encoder_dim)`
2. `ReLU()`
3. `Dropout(dropout)`
4. `Linear(encoder_dim, hidden_dim)`
5. `ReLU()`

This means the encoder currently contains **two linear layers**, not three.

#### Current reconstruction-head depth

The reconstruction head is currently defined as:

1. `Linear(hidden_dim, encoder_dim)`
2. `ReLU()`
3. `Dropout(dropout)`
4. `Linear(encoder_dim, input_dim)`

This means the reconstruction head currently contains **two linear layers**, not three.

#### Current classification-head depth

The classification head is currently defined as:

1. `Linear(hidden_dim, hidden_dim)`
2. `ReLU()`
3. `Dropout(dropout)`
4. `Linear(hidden_dim, num_classes)`

This means the classification head currently contains **two linear layers**, not three.

#### Current configuration surface for architecture depth

The active model configuration file contains:

- `input_dim`
- `encoder_dim`
- `hidden_dim`
- prototype settings
- temperature settings
- regularization weights

It does **not** contain any field such as:

- `num_encoder_layers`
- `num_reconstruction_layers`
- `num_classification_layers`
- `mlp_depth`

Therefore, layer depth is currently fixed by code, not by YAML configuration.

#### Consequence for the requested symmetric three-layer design

In the current repository, making the encoder, reconstruction head, and classification head all use three linear layers would mean editing the active model definition in `src/models/thesis_multitask.py`. It is not currently implemented as a configurable ablation switch.

#### Current seed handling

The repository seeds Python, NumPy, and PyTorch from a single scalar experiment seed through `seed_everything(seed)`. The active full experiment configuration sets:

- `seed: 7`

The training script calls `seed_everything(int(experiment_config["seed"]))` once at experiment startup. This confirms that the current offline runtime is designed around **one seed per process**.

### Evaluation and Orchestration

#### Current experiment assembly

The training entrypoint performs the following sequence:

1. load one resolved experiment configuration
2. seed all random number generators once
3. build one dataset bundle
4. build one model
5. build one optimizer
6. optionally build one learning-rate scheduler
7. build one trainer
8. execute one training loop

This is a single-process execution graph.

#### Current output and logging surface

The active full experiment configuration specifies one:

- `experiment_name`
- `output_dir`
- `checkpoint_dir`
- `wandb_run_name`

These values are scalar and fixed for the run. Reusing the same configuration in multiple concurrent processes would therefore target the same local output locations and the same nominal run identity unless those fields were changed.

#### Current support for multiple simultaneous seed runs

A repository-wide search of the active source and scripts did not identify built-in support for:

- `torch.nn.DataParallel`
- `DistributedDataParallel`
- `torch.distributed`
- a multi-seed launcher
- a multiprocessing experiment runner
- a sweep orchestrator for simultaneous local seed runs

The current codebase therefore does not contain an internal mechanism that takes a list of seeds and dispatches them in parallel as a managed local experiment bundle.

#### Current evidence regarding one RTX3090 handling three concurrent runs

The repository provides evidence for a **single** full offline server configuration on CUDA with:

- batch size `512`
- eight data loader workers
- 300 epochs

The repository does **not** provide:

- a measured memory budget for the active full run
- a measured throughput budget for the active full run
- a concurrency benchmark for two or three simultaneous full runs
- a local scheduler that partitions one graphics processing unit across several experiments

As a result, the repository as it exists today does not document or validate that three full experiments can run concurrently on one RTX3090. The only grounded statement available from the codebase is that the current intended full server path is configured as a single CUDA experiment.

## Code References

- `src/models/thesis_multitask.py:30` - current `MultitaskWindowEncoder` definition
- `src/models/thesis_multitask.py:42` - current two-linear-layer encoder stack
- `src/models/thesis_multitask.py:191` - current reconstruction head definition
- `src/models/thesis_multitask.py:198` - current classification head definition
- `configs/model/thesis_multitask.yaml:1` - current model configuration surface, which does not expose layer-depth controls
- `src/core/seed.py:11` - single-seed runtime seeding function
- `scripts/train.py:104` - single-experiment training entrypoint
- `scripts/train.py:121` - training startup seeding call
- `scripts/train.py:153` - one-model build from one experiment configuration
- `scripts/train.py:157` - one optimizer per experiment process
- `scripts/train.py:201` - one trainer per experiment process
- `configs/experiment/smd_multitask_rtx3090_full.yaml:1` - active full single-run server experiment configuration
- `configs/data/smd_rtx3090_512.yaml:1` - active full-run SMD batch and worker configuration

## Pipeline Documentation

The current offline multitask pipeline remains:

1. resolve one experiment configuration
2. seed one runtime from one scalar seed
3. build SMD windows with window length 100 and stride 10
4. move one model to one CUDA device
5. train one offline multitask model with:
   - one encoder
   - one continuous prototype branch
   - one discrete prototype branch
   - one reconstruction head
   - one classification head
6. log one experiment history to one output directory and one Weights and Biases run

Within this pipeline, the encoder and both heads are currently fixed-depth two-linear-layer modules. No part of the current runtime automatically expands them to three layers.

Likewise, the current pipeline does not expose a higher-order launcher that constructs three seed-specific subprocesses or three independent output namespaces automatically. Multi-seed parallelism is therefore not a first-class repository feature at the present commit.

## Historical Context (from documents/)

The design documents continue to emphasize:

- one-model-one-file readability
- a thesis-facing hidden representation
- fixed-length windows of length 100
- prototype-based multitask learning with reconstruction and classification
- explicit experiment configuration and logging

The current implementation remains aligned with those principles. In particular, the active offline thesis model keeps the encoder, prototype branches, fusion logic, optional losses, and task heads inside one file. The present finding about layer depth is therefore not a contradiction of the design documents. It is a statement that the current implementation realizes this design with fixed two-layer multilayer perceptrons in the encoder and heads.

## Open Questions

1. The repository documents a single full server configuration with batch size 512 on CUDA, but it does not record the actual peak graphics memory usage of that run at this commit.
2. The repository does not currently document whether the active full configuration was profiled on the user’s RTX3090 before considering concurrent execution.
3. The repository does not currently contain a canonical multi-seed launcher or a seed-sweep execution script, so concurrent seed execution remains outside the current built-in experiment surface.
4. The current full experiment configuration still contains scheduler comments and tags that describe the scheduler as “planned,” even though scheduler support is now present in the training code. That comment mismatch is a documentation inconsistency rather than a model-behavior inconsistency.

## Follow-up (2026-04-17 22:12:02 +07+0700)

This research note documented the repository state before the implementation pass that followed it. The repository now includes:

- a symmetric multilayer perceptron depth contract for the offline thesis model through `mlp_num_linear_layers`
- seed-specific RTX3090 experiment configurations with isolated output and checkpoint paths
- a multi-seed launcher script for dry-run, preflight, sequential execution, and explicit parallel execution

The unresolved hardware question remains unchanged. The repository now contains the orchestration surface for multi-seed runs, but it still does not contain measured evidence that three full 300-epoch batch-size-512 runs can safely coexist on one RTX3090 without smoke-level concurrency validation.
