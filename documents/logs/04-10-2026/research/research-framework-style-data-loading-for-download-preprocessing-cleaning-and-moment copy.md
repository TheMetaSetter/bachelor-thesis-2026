# Research: Research how to make all data downloading, data preprocessing, and data cleaning become a framework that can load data in a few notebook lines and feed a model such as Moment-1

**Date**: 2026-04-10 15:46:00 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: 33f0e4ef21ad9862ee5d979ae9143084497736e4  
**Branch**: dev

## Research Question

The user asked that the requirement wording be preserved exactly. The quoted text below is therefore copied without rewording:

> I want to make all the data downloading, data preprocessing and data cleaning in this repo, ALSO, become a framework, such that within few lines of code in a Jupyter notebook, I can EASILY load all the data and feed into a model such as Moment-1. Please see [smd_colab_window_preprocessing_template.ipynb](notebooks/smd_colab_window_preprocessing_template.ipynb) , documents under [design](documents/design) , and code under [data](configs/data) [data](src/data) . I wish I can do that in something like scikit-learn or PyTorch style.

> See https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html

> Invoke [1_research_prompt.md](prompts/1_research_prompt.md) to research how to meet the requirements above. Note the requirements carefully and do not change my words.

## Summary

The repository already contains the core of a framework-style SMD data path, but that path is split across two surfaces rather than exposed as one stable public notebook API. The packaged implementation in `src/data/` already performs per-machine parsing, train-only standardization, fixed-length window creation, collate into the repository batch contract, and registry-driven bundle construction. The notebook `notebooks/smd_colab_window_preprocessing_template.ipynb` already demonstrates the missing user-facing pieces: dataset download, quick inspection utilities, and a MOMENT adapter that converts `[B, L, D]` windows into the `x_enc` and `input_mask` expected by `MOMENTPipeline`.

The shortest path to meeting the requirement is not to replace the current contracts. It is to promote them into a first-class public data framework surface. In concrete terms, the repository needs one public API that owns five stages in one readable path: `download -> parse -> clean/validate -> preprocess -> window/batch -> model adapter`. Today, `parse -> preprocess -> window/batch` already exists in `src/data/`, `download` exists only in notebooks, `clean/validate` exists partly as contract validators rather than as a named pipeline stage, and the MOMENT handoff exists only in the notebook rather than inside `src/`.

The repository is therefore close to the desired direction, but it does not yet provide the same kind of single-entry usage surface that official dataset APIs expose. The current public entrypoint is a config-driven builder that returns a rich `data_bundle`. That is useful for scripts, but it is not yet the few-lines-of-code notebook interface the user requested.

## Detailed Findings

### Data Preparation

#### Current downloading path

- The repository has an explicit SMD downloader workflow, but it is notebook-only today.
- `notebooks/time_series_loading_template.ipynb` implements a GitHub-API-based downloader for `ServerMachineDataset`.
- `notebooks/smd_colab_window_preprocessing_template.ipynb` reuses the same downloader mechanism and writes directly to `DATA_ROOT = /content/ServerMachineDataset`.
- The packaged code under `src/data/` does not currently expose a `download=True` or equivalent public constructor flag.
- The packaged configs instead assume the dataset already exists locally at `data/ServerMachineDataset` (`configs/data/smd.yaml`).
- The local workspace currently does contain the expected SMD folder with 28 `train`, 28 `test`, and 28 `test_label` files, so the present parser can run against local data without using the notebook downloader.

#### Current preprocessing path

- The parser is already separated cleanly from the scaler and the loader, which matches the design documents.
- `src/data/datasets/smd.py` parses raw SMD per machine, preserves entity boundaries, creates per-machine train, validation, and test splits, and emits the repository raw-sequence contract.
- `src/data/scalers.py` fits feature-wise mean and standard deviation on training sequences only, then transforms all splits with the same fitted state.
- `src/data/loaders.py` constructs a `WindowDataset` that materializes windows lazily from sequence index records rather than copying every window up front.
- `src/data/collate.py` stacks windows into the batch contract with `x`, `point_labels`, `mask`, `timestamps`, and `meta`.
- `src/core/contracts.py` validates raw sequences, windows, and batches so later code can assume shape and key consistency.

#### Current cleaning path

- The repository does not yet expose a standalone, named data-cleaning abstraction in `src/data/`.
- What exists today is a combination of validation and normalization:
  - validation through `validate_raw_sequence`, `validate_window`, and `validate_batch`;
  - normalization through `SequenceStandardScaler`;
  - shape-preserving split logic in the parser.
- There is no packaged `Cleaner`, `TransformPipeline`, or equivalent object that a notebook user can call explicitly as a first-class framework step.
- This matters for the requirement wording because “data cleaning” is not yet a visible public layer of the framework, even though some cleaning-adjacent behavior already exists implicitly.

#### Current outputs

- The active builder already returns a reusable bundle:
  - `parser`
  - `scaler`
  - `raw_sequences`
  - `scaled_sequences`
  - `datasets`
  - `loaders`
- This is a strong internal surface for scripts and tests.
- It is not yet a minimal public surface for notebook users because it still expects a config dictionary and knowledge of internal keys.

### Modeling and Training

#### Current contract alignment with the design documents

- `documents/design/design_starter.md` already fixes the intended framework philosophy:
  - a small number of fixed contracts;
  - four stable runtime layers;
  - a stable batch contract;
  - a stable model output contract.
- The active code follows that direction well in the data layer.
- `scripts/train.py` and `scripts/evaluate.py` already use registry-based dataset construction through `build_dataset(...)`, so the repository no longer has the older direct-builder divergence documented in earlier March research logs.

#### Current MOMENT path

- The repository already contains a concrete MOMENT handoff, but it is notebook-only.
- `notebooks/smd_colab_window_preprocessing_template.ipynb` defines:
  - `prepare_moment_x_enc(...)`, which transposes `[B, L, D]` into `[B, D, context_length]` and pads to the expected context length;
  - `embed_windows_with_moment(...)`, which iterates over the project `DataLoader`, moves batches to device, calls `MOMENTPipeline`, and returns embeddings, window labels, and metadata.
- This is important evidence that the desired user flow is already understood in the repository:
  1. load windows using the project batch contract;
  2. adapt the batch to MOMENT’s contract;
  3. obtain embeddings without rewriting the loader.
- However, this adapter has not yet been promoted into `src/` as a reusable public API.

#### Current synthetic augmentation and streaming relevance

- `src/data/augment.py` already provides a self-contained synthetic anomaly injector with the 11 RedLamp-family anomaly taxonomy.
- `src/data/stream.py` already provides sequential window streaming and online view construction.
- These modules show that the repository is already thinking in framework components rather than notebook-only helpers.
- For the present requirement, their relevance is architectural: they prove that the data layer can support multiple downstream consumers while keeping the same core window contract.

### Evaluation

- Evaluation is not the main blocker for the requirement, but the current evaluation path confirms that the data contract is already stable enough to serve more than one runtime consumer.
- `scripts/evaluate.py` rebuilds the dataset through the same registry-driven data path, restores the scaler state from checkpoint, and evaluates the model on the test loader.
- This means the missing work is not “can the repository build reusable data objects?” but “can those reusable data objects be exposed through a concise public notebook API?”

## Framework-Oriented Research Conclusion

To meet the requirement above, the repository should keep the current thin-waist batch contract and elevate it into a public data framework whose notebook-facing surface is closer to three primary-source conventions:

1. A `torchvision.datasets.MNIST`-style dataset constructor surface.
   - The official MNIST API exposes `root`, split selection, transform hooks, and a `download` flag in one constructor.
   - The important pattern is not image-specific behavior. The important pattern is that download and local materialization are part of the public dataset API rather than being notebook-only glue code.

2. A `torch.utils.data.Dataset` plus `DataLoader` separation.
   - PyTorch’s data documentation separates sample storage from iterable batching.
   - The repository already follows this internally through `WindowDataset` and `collate_windows`.
   - The missing step is to expose that separation under a stable notebook import path rather than only through `build_smd_dataset_bundle(data_config)`.

3. A scikit-learn-style transformer pipeline for preprocessing and cleaning.
   - Scikit-learn’s estimator conventions revolve around explicit `fit`, `transform`, and `fit_transform` style behavior.
   - The repository already has part of this shape in `SequenceStandardScaler.fit(...)` and `transform_sequences(...)`.
   - The missing step is to package cleaning and preprocessing as named, composable public transformers rather than hidden internal steps inside the builder or notebook.

The MOMENT model card also supports the intended direction. The official usage surface is `MOMENTPipeline.from_pretrained(..., model_kwargs={"task_name": "embedding"})`, which means the repository does not need to invent a new embedding model interface. It needs to supply a stable adapter from the repository batch contract into MOMENT’s expected input format.

## What Already Exists Versus What Is Missing

### Already exists

- An SMD parser that preserves entity boundaries and split semantics.
- A train-only standardization flow with checkpointable scaler state.
- A lazy window dataset and collate function that produce `[B, L, D]`.
- Runtime validators for raw sequences, windows, and batches.
- A registry-driven script path that rebuilds the same data bundle for train and evaluation.
- A notebook proof-of-concept for SMD download.
- A notebook proof-of-concept for MOMENT embedding handoff.

### Missing for the requirement

- A packaged dataset-level `download` surface in `src/data/`.
- A named, public cleaning stage instead of only implicit validation plus scaling.
- A single notebook-facing public API that hides internal config-dictionary details.
- A packaged MOMENT adapter under `src/` rather than only notebook code.
- A public split-oriented dataset object or helper that behaves in a more PyTorch-like way.
- A public preprocessing pipeline object or helper that behaves in a more scikit-learn-like way.
- Data-versioning wiring for downloaded and processed artifacts through `dvc.yaml`, which `codebase_preferences.md` explicitly calls for and which is not present in the current repository root.

## Recommended Public Surface

The current codebase suggests one dominant design choice: keep the current internals, but add a simpler public layer on top of them.

The public layer should expose three notebook-facing entrypoints:

### 1. Dataset surface

This should own raw data materialization and split access.

Conceptually:

```python
train_dataset = SMD(
    root="data",
    split="train",
    window_size=100,
    stride=10,
    download=True,
)
```

This surface would be the framework answer to the user’s “something like scikit-learn or PyTorch style” requirement. It should wrap the existing parser, scaler, and `WindowDataset`, not replace them.

### 2. Preprocessing and cleaning surface

This should own explicit, named pipeline stages.

Conceptually:

```python
pipeline = SMDPreprocessingPipeline(
    validation_split_ratio=0.2,
    cleaning_steps=[...],
    scaler="standard",
)
pipeline.fit(raw_train_sequences)
processed_train = pipeline.transform(raw_train_sequences)
```

The key research conclusion is that the current repository already has enough functionality to support this surface. What is missing is the public wrapper and the naming of “cleaning” as a visible stage.

### 3. Model-adapter surface

This should own conversion from the repository batch contract into model-specific contracts.

Conceptually:

```python
moment_adapter = MomentWindowAdapter(model_name="AutonLab/MOMENT-1-small")
moment_inputs = moment_adapter.prepare_batch(batch)
embeddings = moment_adapter.embed_batch(batch)
```

This is the cleanest way to preserve the repository’s `[B, L, D]` thesis-facing contract while still supporting pretrained backbones such as MOMENT.

## Proposed Few-Line Notebook Experience

The user asked for a few-line notebook flow. Based on the current repository, the desired public usage shape is most defensibly this:

```python
from src.data.api import load_smd_data
from src.adapters.moment import MomentWindowAdapter

data = load_smd_data(root="data", download=True, window_size=100, stride=10, batch_size=32)
batch = next(iter(data.loaders["train"]))

moment = MomentWindowAdapter("AutonLab/MOMENT-1-small", task_name="embedding")
embeddings = moment.embed_batch(batch)
```

This usage does not exist yet. It is included here because it is the shortest framework-shaped expression of the current codebase direction, and every component in the example already has a partial implementation in the repository.

## Code References

- `prompts/1_research_prompt.md` - research workflow invoked for this document
- `documents/design/idea.md:2` - thesis direction and stable window length context
- `documents/design/design_starter.md:2` - small-number-of-contracts framework direction
- `documents/design/design_starter.md:15` - four stable runtime layers
- `documents/design/design_starter.md:30` - conceptual batch contract
- `documents/design/design_starter.md:42` - conceptual model output contract
- `documents/design/design_starter.md:400` - MOMENT adapter direction
- `documents/design/design_starter.md:760` - framework reuse across datasets and encoders
- `configs/data/smd.yaml:1` - active SMD config root and window defaults
- `src/core/contracts.py:41` - raw-sequence validation
- `src/core/contracts.py:62` - window validation
- `src/core/contracts.py:75` - batch validation
- `src/core/config.py:16` - YAML config loading
- `src/core/config.py:45` - config validation
- `src/core/registry.py:10` - dataset registry
- `src/core/registry.py:18` - dataset builder dispatch
- `src/data/datasets/smd.py:13` - SMD parser definition
- `src/data/datasets/smd.py:52` - SMD split parsing
- `src/data/scalers.py:10` - standard scaler definition
- `src/data/scalers.py:16` - scaler fitting on sequences
- `src/data/scalers.py:27` - scaler transform
- `src/data/scalers.py:40` - scaler checkpoint state
- `src/data/window.py:16` - window slicing helper
- `src/data/collate.py:10` - collate into repository batch contract
- `src/data/loaders.py:19` - lazy `WindowDataset`
- `src/data/loaders.py:71` - SMD dataset builder
- `src/data/loaders.py:133` - public SMD dataset-bundle helper
- `src/data/loaders.py:138` - public SMD dataloader helper
- `src/data/augment.py:31` - synthetic anomaly injector
- `src/data/augment.py:495` - augmentation batch surface
- `src/data/stream.py:35` - sequential online SMD stream
- `src/data/stream.py:118` - online window batcher
- `scripts/train.py:32` - runtime dataset registration
- `scripts/train.py:56` - training experiment orchestration
- `scripts/train.py:82` - registry-based dataset construction in training
- `scripts/evaluate.py:31` - runtime dataset registration in evaluation
- `scripts/evaluate.py:55` - evaluation experiment orchestration
- `scripts/evaluate.py:63` - registry-based dataset construction in evaluation
- `notebooks/time_series_loading_template.ipynb:146` - notebook HTTP session for SMD download
- `notebooks/time_series_loading_template.ipynb:591` - recursive downloader implementation
- `notebooks/smd_colab_window_preprocessing_template.ipynb:190` - notebook HTTP session for SMD download
- `notebooks/smd_colab_window_preprocessing_template.ipynb:207` - GitHub contents API URL helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:290` - recursive downloader implementation
- `notebooks/smd_colab_window_preprocessing_template.ipynb:531` - raw-sequence builder
- `notebooks/smd_colab_window_preprocessing_template.ipynb:570` - per-machine loader with validation split
- `notebooks/smd_colab_window_preprocessing_template.ipynb:636` - notebook standard-scaler fit helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:729` - notebook `SMDWindowDataset`
- `notebooks/smd_colab_window_preprocessing_template.ipynb:843` - notebook collate helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1206` - device move helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1281` - MOMENT input preparation helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1354` - MOMENT embedding helper

## Pipeline Documentation

The current repository path for SMD is:

1. local raw dataset under `data/ServerMachineDataset`
2. parse raw per-machine files into repository raw sequences
3. create per-machine train, validation, and test splits
4. fit train-only standardization statistics
5. transform all splits with the fitted scaler
6. create lazy fixed-length windows with `window_size = 100` and `stride = 10`
7. collate windows into the batch contract `x: [B, L, D]`
8. feed the batch either into repository models directly or, in the notebook, into a MOMENT adapter that converts the batch into MOMENT inputs

The missing framework step is that stages 1 and 8 are not yet packaged into the same public `src/` data surface as stages 2 through 7.

## Historical Context (from documents/)

`documents/design/design_starter.md` already argues for a thin-waist framework with fixed contracts and composition over deep inheritance. That design document is directly consistent with the present `src/data` implementation. `documents/design/idea.md` fixes the thesis-facing data shape at windows of length one hundred and explicitly leaves room for backbone adapters such as MOMENT. Earlier March planning documents also pushed the repository toward registry-driven construction and a readable data path that keeps parsing, scaling, windowing, and batching distinct.

The new part of the present requirement is not the desire for a framework itself. That desire is already present in the design documents. The new part is the demand that download, preprocessing, and cleaning all become part of one concise notebook-facing public API rather than remaining split across notebook utilities and internal builders.

## External Reference Patterns

- Torchvision MNIST constructor pattern:
  - [MNIST documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
  - Search result snippet confirms the public constructor shape `root`, `train`, `transform`, `target_transform`, and `download`.
- PyTorch dataset and dataloader separation:
  - [Datasets and DataLoaders tutorial](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  - The official explanation states that `Dataset` stores samples and labels while `DataLoader` wraps an iterable around the dataset.
- Scikit-learn transformer pattern:
  - [TransformerMixin documentation](https://scikit-learn.org/0.23/modules/generated/sklearn.base.TransformerMixin.html)
  - The relevant reusable pattern is explicit `fit_transform` and `transform` behavior.
- MOMENT public model surface:
  - [AutonLab/MOMENT-1-small model card](https://huggingface.co/AutonLab/MOMENT-1-small)
  - The model card documents `MOMENTPipeline.from_pretrained(...)` with `task_name="embedding"` as an official representation-learning usage pattern.

## Open Questions

- Should the primary public notebook surface be dataset-class-first, bundle-helper-first, or should the repository expose both?
- What exact operations should count as “data cleaning” in this repository beyond shape validation, split integrity checks, and feature standardization?
- Should download and processed-data caching be tracked only for raw SMD, or also for windowed and MOMENT-ready artifacts?
- Should the public MOMENT adapter return only embeddings, or also preserve labels and metadata in a structured bundle?
- How should DVC version the boundary between raw data, cleaned data, standardized data, and windowed data so that the requirement in `codebase_preferences.md` is satisfied without multiplying confusing codepaths?

## Follow-up 2026-04-10 16:20:42 +0700 +07

### Follow-up Question

The follow-up request expanded the scope from data loading alone to the broader framework workflow and persistence surface. The user asked that the repository support:

> This codebase will allow me to run experiments by modifying configuration yaml files, running scripts in the terminal, or I can use Jupyter notebook to run experiments easily, also.

and also that:

> the framework should be compatible with whatever the baseline non-neural network or neural network, classical machine learning or deep learning, whatever the baseline models

and further that:

> automatically save best checkpoints, best weights, onto a Kaggle dataset specified by the user, using Kaggle API

### Follow-up Findings

#### Current experiment workflow surface

- The current repository already supports configuration-driven terminal experiments.
- `configs/experiment/*.yaml` files define experiment-level settings such as `output_dir`, `checkpoint_dir`, and the referenced `data`, `model`, and `task` YAML files.
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_ablation.py`, and `scripts/run_online_adaptation.py` all load experiment configuration through `src/core/config.py` and build runtime components from the registry.
- This means the repository already implements two parts of the stated workflow vision as code that exists today:
  - modify YAML files;
  - run scripts in the terminal.

#### Current notebook workflow surface

- The repository also already supports notebook-based experimentation, but only through explicit notebook code rather than a packaged public notebook API.
- `notebooks/smd_colab_window_preprocessing_template.ipynb` provides a complete SMD walkthrough including download, preprocessing, windowing, batch inspection, classical-baseline flattening, and MOMENT embedding.
- The notebook therefore functions as an experiment surface today, but it is still a notebook template rather than a stable public framework layer under `src/`.

#### Current model-family compatibility

- The current runtime is PyTorch-specific rather than universally model-agnostic.
- `src/models/base_model.py` defines `BaseModel` as a subclass of `torch.nn.Module`.
- `src/engine/trainer.py` expects a PyTorch optimizer, calls `loss.backward()`, and steps the optimizer directly.
- `src/core/config.py` validates only three model names:
  - `reconstruction_mlp_ae`
  - `thesis_multitask`
  - `online_adaptation`
- The script registration path also registers only these three model builders.
- Therefore, compatibility with arbitrary classical machine learning and arbitrary neural baselines is not an implemented repository fact today.

#### Current evidence for partial baseline generality

- The notebook does contain explicit preparation helpers for non-neural or classical baselines.
- `notebooks/smd_colab_window_preprocessing_template.ipynb` includes `flatten_windows_for_baseline(...)` and explicitly states that classical baselines often expect a two-dimensional feature matrix.
- The same notebook also states that the handoff cell is the standard entrypoint to the thesis model or any PyTorch baseline.
- This is evidence that the repository is being shaped toward broader baseline compatibility, but that broader compatibility currently exists only as data-format handoff helpers in notebooks rather than as a unified training and evaluation engine.

#### Current checkpointing and artifact persistence

- The current repository already saves best checkpoints automatically on local storage.
- `src/engine/trainer.py` tracks `best_val_loss` and calls `CheckpointManager.save_checkpoint(...)` whenever validation loss improves.
- `src/engine/checkpoint.py` writes the checkpoint payload into the configured `checkpoint_dir`.
- `scripts/train.py` then logs the resolved config, metrics file, and best checkpoint path through `ExperimentLogger`.
- `src/engine/logger.py` supports remote artifact logging only through Weights & Biases when `logging.use_wandb` is enabled.
- `configs/experiment/smd_multitask.yaml` currently enables Weights & Biases online mode.

#### Current Kaggle persistence status

- The repository currently contains no Kaggle integration code.
- Repository search did not find any usage of `kaggle`, `kagglehub`, or Kaggle dataset-upload commands in `src/`, `scripts/`, `configs/`, or notebooks.
- The current environment files include `wandb`, but they do not include `kaggle` or `kagglehub`.
- The current configuration schema also contains no Kaggle-specific fields such as dataset handle, upload policy, API credential source, or version notes.
- As the repository exists today, local checkpoints are protected only by the local filesystem and, when enabled, optional Weights & Biases artifact logging. Kaggle-backed checkpoint persistence is not implemented.

#### Official Kaggle documentation verified on 2026-04-10

- The official Kaggle GitHub organization currently lists:
  - `Kaggle/kaggle-api` as the official Kaggle API;
  - `Kaggle/kagglehub` as a Python library to access Kaggle resources.
- The Kaggle GitHub organization page shown by the official source indicates these repositories were updated recently relative to this research pass:
  - `kaggle-api`: November 25, 2025;
  - `kagglehub`: November 21, 2025.
- The official Kaggle CLI repository states that it can list, create, update, download, or delete datasets and models.
- The official `Dataset Metadata` wiki for `kaggle-api` documents `dataset-metadata.json` and the `kaggle datasets init -p /path/to/dataset` command for dataset creation and versioning.
- The official `kagglehub` README documents:
  - authentication through a generated Kaggle token;
  - continued support for legacy `~/.kaggle/kaggle.json`;
  - `dataset_upload(handle, local_dataset_dir, version_notes=...)` for creating a new dataset or a new version of an existing dataset.

### Follow-up Interpretation of the Current State

The repository already satisfies part of the stated framework vision. It supports YAML-driven terminal experiments, local automatic best-checkpoint saving, notebook-based data exploration, and optional remote artifact persistence through Weights & Biases. However, the repository does not yet implement the broader vision as an existing fact in three important respects.

First, the notebook workflow is still template-driven rather than exposed as a stable public framework API. Second, universal compatibility with arbitrary classical and deep-learning baselines is not yet implemented, because the current training engine is explicitly built around `torch.nn.Module`, PyTorch optimizers, and a fixed set of registered model names. Third, Kaggle-backed checkpoint persistence is not currently part of the repository’s logging, checkpoint, or configuration surface, even though official Kaggle-maintained interfaces now exist that could support dataset-version uploads outside the repository.

### Additional Code References

- `src/models/base_model.py:9` - current model interface is `torch.nn.Module`-based
- `src/core/config.py:70` - supported model names
- `src/core/config.py:71` - supported task names
- `src/engine/trainer.py:62` - best-checkpoint tracking begins
- `src/engine/trainer.py:100` - best checkpoint is saved on validation improvement
- `src/engine/checkpoint.py:9` - checkpoint manager definition
- `src/engine/checkpoint.py:14` - checkpoint save implementation
- `src/engine/logger.py:14` - experiment logger definition
- `src/engine/logger.py:47` - optional Weights & Biases integration gate
- `src/engine/logger.py:87` - remote artifact file logging surface
- `scripts/train.py:82` - registry-based dataset construction in training
- `scripts/train.py:99` - local checkpoint manager construction
- `scripts/train.py:141` - resolved-config artifact logging
- `scripts/train.py:148` - metrics artifact logging
- `scripts/train.py:156` - best-checkpoint artifact logging
- `scripts/evaluate.py:63` - registry-based dataset construction in evaluation
- `scripts/run_ablation.py:86` - repeated YAML-driven train/evaluate orchestration
- `scripts/run_online_adaptation.py:72` - registry-based dataset construction in online adaptation
- `configs/experiment/smd_vertical_slice.yaml:4` - experiment output directory
- `configs/experiment/smd_vertical_slice.yaml:5` - experiment checkpoint directory
- `configs/experiment/smd_multitask.yaml:14` - `use_wandb: true`
- `configs/experiment/smd_multitask.yaml:17` - `wandb_mode: online`
- `requirements.txt:11` - `scikit-learn` dependency
- `requirements.txt:14` - `wandb` dependency
- `environment.yml:19` - `wandb` dependency in the conda environment
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1164` - explicit classical-baseline note
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1173` - baseline flattening helper
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1197` - PyTorch baseline handoff note
- `notebooks/smd_colab_window_preprocessing_template.ipynb:1425` - toy PyTorch baseline class

### Additional External References

- [Kaggle GitHub organization](https://github.com/kaggle)
- [Official Kaggle CLI repository](https://github.com/Kaggle/kaggle-api)
- [Official Kaggle Dataset Metadata wiki](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata)
- [Official kagglehub repository](https://github.com/Kaggle/kagglehub)
- [Official kagglehub README](https://github.com/Kaggle/kagglehub/blob/main/README.md)

### Additional Open Questions

- Should remote persistence, if later added, upload only `best.pt`, or the entire experiment output directory including resolved config and metric history?
- Should Kaggle persistence, if later added, create a new dataset automatically or require an existing dataset handle supplied by the user?
- Should notebook-driven classical baselines remain notebook-only, or should the script engine eventually support a second model interface that is not based on `torch.nn.Module`?
