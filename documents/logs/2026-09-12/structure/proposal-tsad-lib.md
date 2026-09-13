---
date: 2026-09-12
topic: Architecture proposal for tsad in tsad-lib
status: proposal
scope: notebook API and multi-dataset architecture
---

# Proposal: `tsad` in `tsad-lib`

`tsad` means time-series anomaly detection.
It is the short public package name and CLI name.
The sibling directory keeps the project name `tsad-lib`.

## The story starts in a notebook

The user should not need to understand the internal classes.
The user should be able to load data, choose a method, run an experiment, and inspect the result.

The smallest useful notebook should let a student see what is available and choose from it:

```python
from tsad import available, run

DATA = "/path/to/bachelor-thesis-2026/data"

available(root=DATA).show()

report = run(
    root=DATA,
    datasets=["SMD"],
    models=["thesis"],
    entities=["machine-1-6"],
    variant="O2",
    window=20,
    seed=6,
)

report.table()
```

The same story should work for another dataset:

```python
report = run(
    root=DATA,
    datasets=["SMD", "SWaT"],
    models=["thesis", "isolation_forest"],
    entities="all",
    window=20,
    seed=6,
)
report.table()
```

The first example trains one model on one entity.
The second example trains several models on several datasets and their available entities.
The student does not write a loop.
The library writes the loop.

For a file whose structure is not known yet, the user first asks the library to explain it:

```python
from tsad import inspect

info = inspect(DATA + "/extra-industrial-data/DNA_C1.mat")
info.show()
```

If a file contains several possible variables, the error should name the available variables and show the next valid command.
The library should not silently guess a scientific meaning for an unknown variable.

The notebook API has five simple verbs:

| Verb | Meaning |
| --- | --- |
| `inspect` | Explain a file or dataset family before loading it |
| `available` | List datasets, models, and entities that can be selected |
| `load` | Convert a source into one common time-series object |
| `run` | Train, score, evaluate, and report selected combinations |
| `report.table` | Show the metrics table |

The end-user sees a short story.
The codebase handles the longer story behind the scenes.

## The same story from the command line

The notebook is optional.
The same experiment can run from one short command:

```bash
tsad run \
  --data-root /path/to/bachelor-thesis-2026/data \
  --datasets SMD SWaT \
  --models thesis isolation_forest \
  --entities all \
  --window 20 \
  --seed 6 \
  --report reports/first-run
```

The CLI expands the selected datasets, models, and entities into runs.
It trains and evaluates each run.
It writes one report table at the end.
The student does not need to write Python loops, configuration files, or output-path code.

The CLI also has two discovery commands:

```bash
tsad list datasets --data-root /path/to/bachelor-thesis-2026/data
tsad list models
```

The command should fail before training when a selected model cannot use a selected dataset.
The error should name the missing capability and show a valid alternative.

## The report appears automatically

After evaluation, `run()` and `tsad run` create one report directory.
The first report version contains:

```text
reports/first-run/
├── metrics.csv
├── metrics.md
└── run_manifest.json
```

The main table has one row for each dataset, entity, and model combination.
Its first required metric is `VUS-PR@FPR-budget`.
The default table also contains `VUS-PR`, `Affiliation F1-score`, `VUS-ROC`, and `raw-FPR` when the dataset has the required labels.

The table also records `dataset`, `entity`, `model`, `variant`, `seed`, `status`, and `warning`.
If a metric is undefined, the cell stays undefined.
The system never changes an undefined value to zero.

The Markdown table is for a student to read.
The CSV table is for later analysis.
The JSON manifest explains exactly which runs produced the rows.

## What the data directory contains

The loader must support every dataset family currently present under `data/`.
It must also distinguish datasets from cache files and operating-system files.

| Data family | Current form | First adapter |
| --- | --- | --- |
| SMD | `.npy` arrays and labels | `NpyArrayAdapter` |
| ServerMachineDataset | train/test/label `.txt` files | `ServerMachineAdapter` |
| NASA | train/test `.npy` files and `labeled_anomalies.csv` | `NasaAdapter` |
| SWaT | timestamped `.csv` files | `SwatAdapter` |
| IOPS | train/test `.out` files with values and labels | `IopsAdapter` |
| AnomalyArchive | univariate `.txt` files | `UcrTextAdapter` |
| TSB-AD-M and TSB-AD-U | `.zip` archives containing `.csv` files | `TsbArchiveAdapter` |
| IBM Cloud ICCAD | `.parquet`, `.csv`, and anomaly-window files | `IccadAdapter` |
| Extra industrial data | `.mat` and large `.csv` files | `MatAdapter` and `CsvAdapter` |

`data/.cache`, `.DS_Store`, and unrelated metadata files are not datasets.
The two TSB-AD zip files are dataset sources even though they are not directories.

The loader will support all these physical forms.
It will not claim that every model has the same scientific meaning on every family.
For example, a model that needs multivariate windows cannot run on a univariate source without an explicit conversion rule.
The library will report that limitation before training starts.

## The common language inside the codebase

Every adapter tells the same internal story.
It returns a `SeriesSet` with these fields:

```text
values          numeric array with shape [time, channels]
timestamps      optional array with length time
labels          optional point labels with length time
channel_names   optional list with length channels
entity          source entity or file name
dataset_name    logical dataset family
split           train, validation, test, or unknown
provenance      source path, file hash, parser name, and parser settings
capabilities    labels, timestamps, multivariate, train_test_split, streaming
```

The object is intentionally small.
It does not contain a model, optimizer, or metric.
It only describes data.

The data flow then becomes:

```text
source path
    ↓
inspect
    ↓
adapter
    ↓
SeriesSet
    ↓
split and scale
    ↓
window
    ↓
model runner
    ↓
scores and labels
    ↓
Result
```

This boundary lets the rest of the code ignore whether the source was NPY, CSV, TXT, MAT, Parquet, or ZIP.

## The public classes tell the same story

The notebook uses functions.
The functions use a few small classes.

### `SeriesSet`

`SeriesSet` owns one normalized source.

Attributes:

- `values`
- `timestamps`
- `labels`
- `channel_names`
- `dataset_name`
- `entity`
- `split`
- `provenance`
- `capabilities`

Methods:

- `validate()` checks lengths, dimensions, and finite values.
- `select_channels(names)` returns a smaller `SeriesSet`.
- `with_labels(labels)` returns a labeled copy.
- `describe()` returns a short human-readable description.

`SeriesSet` does not load files by itself.
An adapter creates it.

### `DatasetAdapter`

`DatasetAdapter` is the small boundary between a source format and `SeriesSet`.

Methods:

- `can_read(path) -> bool`
- `inspect(path) -> DatasetInfo`
- `load(path, options) -> SeriesSet`

Each adapter owns one clear parsing responsibility.
It does not train a model.
It does not compute metrics.

### `DatasetInfo`

`DatasetInfo` explains a source before loading it.

Attributes:

- `family`
- `source_path`
- `entities`
- `file_format`
- `shape_hint`
- `has_labels`
- `has_timestamps`
- `warnings`

Method:

- `show()` prints the next useful action for a notebook user.

### `Catalog`

`Catalog` is what `available()` returns.
It contains dataset names, model names, and entity names for each dataset.
Its only public method is `show()`.
Students can copy the shown names into `run()` or the CLI command.

### `RunRequest`

`RunRequest` is the internal description of one selected combination.
It contains one dataset, one entity, one model, one variant when needed, one window, and one seed.
The multi-run function creates these requests automatically.
Students do not need to create them.

### `DataPipeline`

`DataPipeline` prepares a `SeriesSet` for a model.

Its methods follow one order:

1. `split()` creates train, validation, and test parts.
2. `fit_scaler()` learns scaling from train only.
3. `transform()` applies the saved scaler.
4. `window()` creates windows and absolute indices.
5. `stream()` returns sequential windows for online methods.

The pipeline stores the scaler and window settings in `PipelineState`.
It never uses test labels to fit a scaler or update a model.

### `ExperimentConfig`

`ExperimentConfig` holds the small set of choices needed for one run:

- `model`
- `variant`
- `window`
- `seed`
- `train_stride`
- `eval_stride`
- `output_root`
- `dataset_options`

Defaults should make a first notebook run short and clear.
The full matrix can pass a larger configuration later.

### `Experiment`

`Experiment` tells the complete offline or online story.

Methods:

- `prepare()` loads and prepares the data.
- `fit()` trains the selected method.
- `score()` produces point scores.
- `evaluate()` computes metrics when labels exist.
- `run()` calls the required steps in order.

The notebook-level `run()` function creates `Experiment` internally.
Most users never need to construct this class directly.

### `ModelRunner`

`ModelRunner` is the common method boundary.

Methods:

- `fit(data, config) -> ModelState`
- `score(data, state) -> ScoreSet`
- `predict(scores, threshold) -> PredictionSet`

THESIS, RedLamp, CANDI, M2N2, Stumpy, KMeansAD, and Isolation Forest each implement this boundary.
Their native computations remain separate.

### `Result`

`Result` is what the notebook user receives.

Attributes:

- `scores`
- `predictions`
- `metrics`
- `coverage`
- `artifacts`
- `provenance`
- `warnings`

Methods:

- `summary()` prints the important numbers.
- `plot()` shows scores and predictions when plotting is available.
- `save(path)` writes a small JSON result and selected arrays.

### `Report`

`Report` collects the results from one `run()` call.

Methods:

- `table()` prints the core metric table.
- `save(path)` writes `metrics.csv`, `metrics.md`, and `run_manifest.json`.
- `failed()` lists combinations that could not run.

The student sees `Report`.
The adapter, pipeline, and model classes stay inside the codebase.

## The modules keep the story readable

The proposed project tree is small:

```text
tsad-lib/
├── pyproject.toml
├── README.md
├── notebooks/
│   └── 01_first_run.ipynb
├── src/tsad/
│   ├── __init__.py
│   ├── api.py
│   ├── types.py
│   ├── config.py
│   ├── data/
│   │   ├── adapters.py
│   │   ├── discovery.py
│   │   ├── normalize.py
│   │   ├── windows.py
│   │   └── adapters/
│   │       ├── npy.py
│   │       ├── server_machine.py
│   │       ├── nasa.py
│   │       ├── swat.py
│   │       ├── iops.py
│   │       ├── ucr_text.py
│   │       ├── tsb_archive.py
│   │       ├── iccad.py
│   │       ├── mat.py
│   │       └── csv.py
│   ├── models/
│   │   ├── thesis.py
│   │   ├── redlamp.py
│   │   ├── candi.py
│   │   ├── m2n2.py
│   │   └── traditional.py
│   ├── pipeline.py
│   ├── metrics.py
│   ├── artifacts.py
│   └── reporting.py
└── tests/
    ├── test_api.py
    ├── test_data_adapters.py
    ├── test_pipeline.py
    └── test_models.py
```

The modules have one-way dependencies:

```text
api
  → pipeline
    → data and models
      → types
  → metrics and reporting
```

Data adapters must not import model modules.
Model modules must not inspect raw file formats.
The notebook API must not contain parsing rules.

## How each dataset enters the story

The adapter first identifies the source.
Then it returns the same `SeriesSet` shape.

### NPY sources

`NpyArrayAdapter` handles array files with optional companion label files.
`NasaAdapter` adds the NASA train/test layout and reads `labeled_anomalies.csv` as metadata.
`SMD` and `ServerMachineDataset` remain separate adapters because their file layouts and identity rules differ.

### Server machine sources

`ServerMachineAdapter` matches train, test, and test-label files by machine name.
It exposes one machine as one entity.
It checks that all three files agree on time length and channel count.

### CSV sources

`SwatAdapter` treats the first column as timestamps and the remaining columns as channels.
It keeps normal and attack files as distinct splits when both are available.
`CsvAdapter` handles explicit column and label options for sources such as ICCAD and the extra industrial CSV.
It never assumes that the first numeric column is a label without a rule.

### IOPS and UCR sources

`IopsAdapter` reads the value and label columns in `.out` files.
`UcrTextAdapter` reads univariate values from text files and uses filename metadata only as provenance unless labels are explicitly available.

### TSB-AD archives

`TsbArchiveAdapter` reads members from `TSB-AD-M.zip` and `TSB-AD-U.zip` without requiring the user to unpack them first.
It exposes archive member names as entities.
It records the archive hash and member name in provenance.

### MAT sources

`MatAdapter` inspects variable names and shapes before loading a selected variable.
The user must choose a variable when more than one candidate exists.
This keeps the design safe for `DNA_C1.mat` and the chromosome file.

### Parquet and ICCAD sources

`IccadAdapter` reads the Parquet table and the anomaly-window CSV.
It preserves timestamps, service or location identifiers, and anomaly windows as metadata.
It creates point labels only when the window-to-point conversion is explicit.

## Three design pattern choices

The data formats are different, so some separation is necessary.
The question is how much structure to add around that separation.

### Option A — Direct loader functions

Each dataset has one function such as `load_smd()` or `load_swat()`.
The public API calls the correct function with a small `if` statement.

This is the smallest design.
It is easy for a beginner to read.
It becomes repetitive when more formats and archives are added.

### Option B — Adapter plus Strategy

Each dataset family has one adapter.
The adapter converts its source into `SeriesSet`.
The pipeline uses strategies for split, scaling, and window rules.
The public API remains only `inspect`, `load`, and `run`.

This keeps format-specific code out of models and keeps the user API small.
It adds a few classes, but each class has one clear job.
This is the recommended option.

### Option C — Registry plus plugins

Adapters register themselves by name.
Third-party packages can add new datasets without changing the core package.

This is useful for a library used by many teams.
It adds discovery rules, plugin loading, and version compatibility.
It is too large for the first thesis codebase.

## Recommended choice

Choose a small version of Option B.
Use an adapter boundary because the data directory contains NPY, TXT, CSV, MAT, Parquet, and ZIP sources.
Use direct functions for split and window rules at first.
Add a strategy object only when one concrete dataset conflict requires it.
Do not add a plugin system.
Do not add a general registry until a real external extension requires it.

The public API remains a Facade.
The dataset adapters use the Adapter pattern.
Split and window differences use the Strategy pattern.
The model boundary uses a small Protocol rather than a deep inheritance tree.

## The full notebook journey

The first notebook starts by setting one data root.
The user calls `available()` and chooses names from the printed lists.
The user calls `run()` with one or more datasets, models, and entities.
The experiment prepares each dataset, trains each model, evaluates each result, and returns one `Report`.
The user reads `report.table()`.
The table is generated automatically after training and evaluation.

The same journey works for a clean labeled benchmark and for an unlabeled exploratory file.
When a method needs labels or multivariate input, the library says so before running.
When a source is ambiguous, the library asks the user to select the variable or columns.
When a dataset is large, the pipeline streams or memory-maps it instead of loading everything into memory.

## Boundaries and safety rules

The loader never fits a scaler on test data.
The model never receives test labels during fitting or online updates.
The adapter never invents labels.
The evaluator reports undefined metrics instead of replacing them with zero.
Every result records source path, hash, parser, config, seed, and code revision.
Smoke outputs and research outputs use different roots.

The new codebase supports all current dataset families at the loading boundary.
A method may still reject a dataset when its required capability is missing.
That rejection is part of the design, not a hidden conversion.

## Suggested implementation order

The story should be implemented in small chapters.

First, write `types.py` and make one synthetic `SeriesSet` pass validation.
Next, implement `load()` and the SMD adapter.
Then, make the notebook run one baseline end to end.
After that, add the other adapters one family at a time and test each against a small real file.
Only then add THESIS, online adaptation, metrics, and the full matrix runner.

The first acceptance target is not the full benchmark.
It is a short notebook that loads one SMD machine, runs one model, and returns one readable result.
The second target is the same notebook flow for every dataset family listed above.
The third target is the same multi-run flow from the CLI.

## Open choices for review

The architecture is ready for review, but three choices should be confirmed before implementation:

1. Choose Option A, B, or C for dataset organization.
2. Confirm that unknown MAT variables require an explicit notebook selection.
3. Confirm whether `run()` should default to a simple baseline or require an explicit model name.

No code has been created from this proposal.
