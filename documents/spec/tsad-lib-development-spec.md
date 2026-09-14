---
title: "Development Specification for tsad-lib"
status: draft-development-contract
scope: "Minimal sibling library for time-series anomaly detection"
public_name: tsad
project_name: tsad-lib
---

# The Development Story of `tsad-lib`

The story begins with a student who has a data folder and wants to run one
time-series anomaly detector. The student should write a few clear lines in a
notebook, or one short command in a terminal. The library should do the long
work: find the data, prepare it, train the selected method, evaluate it, and
write a readable report.

The library must stay small. It must still preserve the scientific rules in the
current ontology, THESIS specifications, and SMD experiment matrix.

The story now has one larger method shelf. The reference
`Time-Series-Library` contains 40 model files, but those files serve several
tasks. `tsad-lib` may register all 40 names, while each name must state whether
it has a verified anomaly-detection adapter. A forecasting-only or zero-shot
model must not appear as a ready TSAD method merely because its file exists.

This document is the development contract for the sibling directory:

```text
/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/tsad-lib
```

The document is written in the thesis repository because `documents/` is the
current source of truth.

## How the coding agent should read this story

The coding agent is the actor in every chapter. It starts with the smallest
working path, checks the result, and then opens the next chapter.

At the start of each phase, the coding agent reads the named contracts and
checks the current repository state. During the phase, it changes only the
files needed for that chapter. At the end, it runs the smallest test that proves
the chapter works.

The coding agent must not silently decide a scientific meaning. When a name,
loss, metric, artifact, or matrix rule is marked pending, the agent stops that
path and records the missing decision. It may continue with independent work
that has a complete contract.

The coding agent must keep three things separate:

```text
implemented runtime behavior
documented research intent
future library design
```

This separation lets the agent build a simple library without claiming that an
unfinished experiment is ready.

## 1. What the first user sees

The first notebook should be short enough for a high-school student to read:

```python
from tsad import available, run

DATA = "/path/to/bachelor-thesis-2026/data"

available(DATA).show()

report = run(
    data_root=DATA,
    datasets=["SMD"],
    models=["thesis"],
    entities=["machine-1-6"],
    variant="O0",
    seed=6,
)

report.table()
```

The table keeps the student’s choice simple while making readiness visible:

| model | source | task | status | note |
| --- | --- | --- | --- | --- |
| `TimesNet` | `Time-Series-Library` | anomaly detection | `reference_anomaly_path` | Adapter tests are still required. |
| `KANAD` | `Time-Series-Library` | anomaly detection | `reference_anomaly_path` | Adapter tests are still required. |
| `Chronos` | `Time-Series-Library` | zero-shot forecasting | `not_yet_tsad_ready` | No verified TSAD score path yet. |
| `isolation_forest` | `tsad-lib` | anomaly detection | `ready` | Native `tsad-lib` method. |

The table may contain all 40 reference models. A student can select a model by
name, but `run()` must explain and reject a model whose status is not `ready`.

The student may choose more than one item:

```python
report = run(
    data_root=DATA,
    datasets=["SMD", "SWaT"],
    models=["thesis", "isolation_forest"],
    entities="all",
    seed=6,
)
```

The library creates the internal loop. The student does not write nested loops
over datasets, models, and entities.

An unfamiliar file has a separate first step:

```python
from tsad import inspect

inspect(DATA + "/extra-industrial-data/DNA_C1.mat").show()
```

If a file has several possible variables or columns, `inspect()` explains the
choice. The loader must not guess a scientific meaning.

The public API has five verbs:

| Verb | Job |
| --- | --- |
| `inspect` | Explain one source before loading it |
| `available` | List datasets, entities, models, and variants |
| `load` | Convert a source into the common data object |
| `run` | Train, score, evaluate, and collect selected runs |
| `report.table` | Show the final metric table |

## 2. The same story in the CLI

The terminal path must use the same Python runtime as the notebook:

```bash
tsad run \
  --data-root /path/to/bachelor-thesis-2026/data \
  --datasets SMD SWaT \
  --models thesis isolation_forest \
  --entities all \
  --seed 6 \
  --report reports/first-run
```

Discovery commands are small:

```bash
tsad list datasets --data-root /path/to/bachelor-thesis-2026/data
tsad list models
```

The CLI and notebook call the same `run()` function. They must not contain
separate parsing or training logic.

## 3. Rules inherited from the current research documents

The public API may be simple. The runtime must still obey these rules.

| Rule | Development requirement |
| --- | --- |
| Data leakage | Fit scalers and memories from training data only. Use clean validation for thresholds. Use test labels only for final metrics. |
| Default score | Use raw-input MSE with `score_space: raw_input` and `point_score_transform: identity`. |
| Offline THESIS flow | Stage A → memory initialization → Stage B → threshold calibration → offline evaluation. |
| Stage B routing | Use `direct_branch_routing` by default. Fusion blocks are skipped by default. An explicit override must be recorded. |
| Contrastive name | `point-level contrastive loss` and `_compute_two_view_contrastive_loss` refer to the same computation. Do not add a second copy. |
| Online state | Freeze the offline source model and update only the allowed online module. |
| Result meaning | Keep point scores, window scores, predictions, metrics, coverage, warnings, and provenance separate. |
| Undefined metrics | Keep an undefined metric undefined. Never replace it with zero. |
| Reproducibility | Record dataset, entity, method, variant, seed, configuration, checkpoint, and code identity. |

The detailed scientific rules remain in the [offline ontology](offline_pretraining_terminology_ontology.md), [online ontology](online_tta_terminology_ontology.md), [full specification v3](full-spec-v3.md), and [raw-input score specification](full-spec-v4.md).

## 4. The four directed graphs

The library keeps four small graphs. Each graph has one clear direction.

### 4.1 Dataset graph

```text
dataset → entity → file → file_extension
```

One dataset family contains entities. One entity may use several files. Every
file has one physical extension. A ZIP archive is a source container; its inner
files still have their own extensions.

### 4.2 Method graph

```text
method_source → method → variant → component
method → component
```

The method source records where an implementation came from. The method owns
the complete component set. A variant selects a subset or a changed loss set.
The contract is:

```text
num_components(method) >= num_components(corresponding method_variant)
```

Components include model modules and loss modules. Artifact files are not
counted as method components.

### 4.3 Metric graph

```text
metric_family → metric → metric_priority
```

The family groups related metrics. The priority tells the report which metric
must appear first. The priority does not change the metric calculation.

### 4.4 Runtime graph

```text
student or CLI
→ run request
→ dataset adapter
→ common sequence
→ prepared windows
→ method runner
→ scores and predictions
→ metrics
→ result
→ report files
```

The graph is a dependency graph, not a promise that every method uses the same
internal algorithm.

## 5. Common data objects

Every dataset adapter ends its work by creating `SeriesSet`.

```python
SeriesSet(
    values=[time, channels],
    timestamps=None,
    labels=None,
    channel_names=None,
    dataset_name="SMD",
    entity="machine-1-6",
    split="train",
    provenance={...},
    capabilities={...},
)
```

`SeriesSet` owns data only. It does not own a model, optimizer, or metric.

Required rules:

1. `values` is numeric with shape `[time, channels]`.
2. `labels` is optional and, when present, has length `time`.
3. `timestamps` is optional and, when present, has length `time`.
4. The object records the dataset family, entity, split, source path, parser,
   file hash, and parser settings.
5. `validate()` rejects wrong dimensions, wrong lengths, non-finite values, and
   empty time axes.

The pipeline converts `SeriesSet` into these smaller runtime objects:

```text
SeriesSet
→ raw_sequence
→ offline_window or causal_window
→ offline_batch
→ ScoreSet
```

An offline window records `start_index` and `end_index`. An online causal
window records absolute indices and never contains labels in the scorer.

## 6. Dataset support

The first release must discover every current top-level dataset source below.
It must ignore `.cache`, `.DS_Store`, notebooks, lock files, and unrelated
metadata.

| Dataset family | Current source | First adapter | Entity rule |
| --- | --- | --- | --- |
| SMD | `.npy` arrays | `SmdAdapter` | Each machine is one entity |
| ServerMachineDataset | train/test/test-label `.txt` files | `ServerMachineAdapter` | The machine name is the entity |
| NASA | train/test `.npy` files and `labeled_anomalies.csv` | `NasaAdapter` | Each matching NASA series is one entity |
| SWaT | timestamped `.csv` files | `SwatAdapter` | The selected file or declared stream is the entity |
| IOPS | train/test `.out` files | `IopsAdapter` | The KPI identifier is the entity |
| AnomalyArchive | univariate `.txt` files | `UcrTextAdapter` | The file stem is the entity |
| TSB-AD-M and TSB-AD-U | ZIP archives containing `.csv` members | `TsbArchiveAdapter` | The archive member is the entity |
| IBM Cloud ICCAD | `.parquet` and anomaly-window `.csv` files | `IccadAdapter` | The declared service or location is the entity |
| Extra industrial data | `.mat` and `.csv` files | `MatAdapter` and `CsvAdapter` | The selected variable or declared series is the entity |

An adapter must implement:

```python
class DatasetAdapter(Protocol):
    def can_read(self, path: Path) -> bool: ...
    def inspect(self, path: Path) -> DatasetInfo: ...
    def load(self, path: Path, options: dict) -> list[SeriesSet]: ...
```

`inspect()` must expose candidate entities, file format, shapes, labels,
timestamps, warnings, and required user choices.

Future dataset support needs only three additions:

1. Add one adapter module.
2. Add one catalog entry.
3. Add focused loader tests and one small fixture.

The first release does not use a third-party plugin system.

## 7. Data preparation

`DataPipeline` follows one order:

```text
load
→ split
→ fit scaler on train only
→ transform all selected splits
→ create windows
→ keep absolute indices
```

The pipeline stores its scaler, window size, stride, and split identity in
`PipelineState`. A test label can be carried to the evaluator, but it cannot
reach scaler fitting, model fitting, memory initialization, threshold fitting,
or online adaptation.

The default research values are:

```yaml
window_size: 20
offline_stride: 20
online_stride: 1
```

Dataset-specific rules must be explicit. The pipeline must reject a method
when a required capability is missing. It must not silently convert a
univariate source into a multivariate source.

## 8. Method and variant contract

The public method boundary is small:

```python
class MethodRunner(Protocol):
    def fit(self, data: PreparedData, config: RunConfig) -> ModelState: ...
    def score(self, data: PreparedData, state: ModelState) -> ScoreSet: ...
    def predict(self, scores: ScoreSet, thresholds: Thresholds) -> PredictionSet: ...
```

The boundary hides internal differences. THESIS, RedLamp, CANDI, M2N2,
Stumpy, KMeansAD, and Isolation Forest keep their own native computations.

The first catalog contains these method names:

```text
thesis
redlamp
candi
m2n2
stumpy
kmeans_ad
isolation_forest
```

The second catalog contains the 40 model names from the reference
`Time-Series-Library` source. The names are catalog entries, not automatic
claims of anomaly-detection readiness.

```text
Autoformer, Chronos, Chronos2, Crossformer, DLinear,
ETSformer, FEDformer, FiLM, FreTS, Informer, KANAD, Koopa,
LightTS, MICN, MSGNet, Mamba, MambaSimple, Moirai,
MultiPatchFormer, Nonstationary_Transformer, PAttn, PatchTST,
Pyraformer, Reformer, SCINet, SegRNN, Sundial, TSMixer,
TemporalFusionTransformer, TiDE, TiRex, TimeFilter, TimeMixer,
TimeMoE, TimeXer, TimesFM, TimesNet, Transformer, WPMixer,
iTransformer
```

The library stores these method-source edges:

```text
Time-Series-Library → ModelSpec
ModelSpec → TimeSeriesLibraryRunner
TimeSeriesLibraryRunner → MethodRunner
```

The model file and its reusable layer files are implementation components.
They are not separate student-selectable methods. The reference `exp/` files
are task runners, so they are not copied as `tsad-lib` experiment runners.

Each reference model receives one readiness state:

| State | Meaning |
| --- | --- |
| `reference_anomaly_path` | The reference repository contains an anomaly-detection script for the model. `tsad-lib` still needs its own adapter tests. |
| `not_yet_tsad_ready` | The model is registered, but the current evidence does not yet prove a valid `tsad-lib` anomaly path. |
| `blocked_dependency` | The model needs an unavailable or optional package before it can run. |
| `ready` | The model passes the `tsad-lib` anomaly adapter, score, threshold, and artifact tests. |

The first 15 reference anomaly paths are:

```text
Autoformer, Crossformer, DLinear, ETSformer, FEDformer,
FiLM, Informer, KANAD, LightTS, MICN, Pyraformer, Reformer,
TimesNet, Transformer, iTransformer
```

The remaining models stay visible in `available(DATA)` but remain pending until
their task, output shape, dependencies, and anomaly score meaning are checked.

The library must distinguish:

```text
method_variant  = the method's own variant
offline_variant = O0, O1, or a later approved offline variant
online_variant  = A0, A1, A2, or a method-specific name such as main
combined_label  = offline_variant + online_variant
```

`O2` is a defined offline variant in the current ontology and SMD matrix. Its
scientific contract is fixed: it uses `reconstruction_loss`,
`classification_loss`, and `two_view_contrastive_loss`. The matrix calls the
last loss point-level contrastive loss. The ontology says that both names mean
the same computation, `_compute_two_view_contrastive_loss`.

The O2 implementation uses that computation once. It does not create a second
point-level loss. O2 does not use Balanced Point-Score Loss. In Stage B, O2
uses `direct_branch_routing`, so the fusion blocks are skipped by default.
O2 still needs generator, configuration, dependency, checkpoint, and preflight
integration before it becomes an executable all-machine matrix entry.

The minimum offline variant table is:

| Variant | Loss contract | Stage B routing | Status |
| --- | --- | --- | --- |
| `O0` | Reconstruction, classification, and the existing two-view contrastive loss | `direct_branch_routing`; fusion blocks skipped | Defined |
| `O1` | `O0` plus Balanced Point-Score Loss in Stage A | `direct_branch_routing`; fusion blocks skipped | Defined |
| `O2` | Reconstruction, classification, and the existing two-view contrastive loss; no Balanced Point-Score Loss | `direct_branch_routing`; fusion blocks skipped | Defined contract; integration pending |

The O2 configuration tells the same story in one place:

```yaml
offline_variant: O2
losses:
  - reconstruction_loss
  - classification_loss
  - two_view_contrastive_loss
balanced_point_score_loss: false
fusion_mode: direct_branch_routing
```

## 9. THESIS offline runtime

The THESIS offline runner follows this story for one entity and seed:

```text
Stage A multitask training
→ stage_a_best_checkpoint
→ stage_b_memory_initialization
→ stage_b_initialization_checkpoint
→ Stage B training
→ stage_b_best_checkpoint
→ clean-validation threshold artifact
→ offline evaluation
```

Stage A uses the exact configured loss set. The named losses are:

```text
reconstruction_loss
classification_loss
two_view_contrastive_loss
point_score_loss      # only for its approved ablation
```

For O2, Stage A selects the first three losses and disables Balanced Point-Score
Loss. The point-level name in the matrix still selects
`two_view_contrastive_loss`; it does not add another computation.

The contrastive computation keeps clean points from both views, flattens them
across the complete batch, uses the matching point as the positive, and uses
other filtered points in the batch as negatives. It returns zero when no clean
point remains.

Memory initialization reads training data only. It creates:

```text
continuous_prototype_bank
discrete_codebook
anomaly_verification_metadata
```

Stage B freezes the encoder and both memory banks. The default is:

```yaml
fusion_mode: direct_branch_routing
```

The continuous branch goes to the reconstruction head. The discrete branch
goes to the classification head. Fusion blocks are skipped by default. An
explicit fusion override must be present in the resolved configuration and
checkpoint metadata.

The reason is part of the contract. The discrete codebook may carry anomaly
patterns, so those patterns must not leak into the reconstruction head. The
continuous prototype bank represents normal patterns, so it must not add noise
to the classification head.

## 10. Stochastic retrieval and uncertainty

When the selected THESIS configuration uses the current stochastic protocol,
the runner must preserve these values:

```yaml
continuous_bank_size: 32
discrete_codebook_size: 60
continuous_temperature: 0.9
discrete_temperature: 0.9
inference_samples: 10
sample_variance_correction: unbiased
precision: FP32
```

The encoder runs once before the Monte Carlo sample dimension. Similarity
matrices are computed once. Retrieval and downstream heads are vectorized over
the ten samples.

The stable output keeps mean predictions at the top level. Per-sample outputs
and uncertainty remain under `aux`:

```text
aux.point_score_samples
aux.window_score_samples
aux.reconstruction_samples
aux.classification_probability_samples
aux.point_score_variance
```

Uncertainty is reported. It does not change thresholding, triage, buffers, or
adaptation in the current protocol.

## 11. Score, threshold, and metric contract

The default score is:

```yaml
score_space: raw_input
point_score_definition: raw_input_point_mse
point_score_transform: identity
```

The runtime inverse-transforms the input and every reconstruction sample with
the train-fitted scaler before computing raw-input MSE. It averages per-sample
MSE values before point or window reduction. It does not compute MSE against
the average reconstruction.

A latent-MSE run is allowed only when it names its latent score space and uses
a separate threshold artifact. The historical sigmoid transform is opt-in
only.

The evaluator restores overlapping window scores to the absolute entity
timeline before computing point or event metrics.

The report metric graph is:

```text
VUS
├── VUS-PR@FPR-budget  priority: primary_required
├── VUS-PR             priority: default
└── VUS-ROC            priority: default
Affiliation
└── Affiliation F1-score priority: default
FPR
└── raw-FPR             priority: default
```

The SMD matrix requires `VUS-PR@FPR-budget` at budgets `0.1%`, `0.5%`, and
`1%`. The metric implementation must record its budget, support, and
availability. It must not hide failed or unavailable cells.

The current documents have not yet locked the final serialization and formula
name for every library metric. Until they do, the library must keep metric
calculation behind one small interface and record the exact metric definition
used by each report.

## 12. Online runtime

The online runner receives the matching `stage_b_best_checkpoint` and
`threshold_artifact`. It does not rebuild either from the test stream.

The causal story is:

```text
causal_window
→ source score
→ EWMA score
→ four-region triage
→ optional verification
→ optional projector update
→ online_event_record
```

The four regions are:

```text
normal
hard_old_normality
gray_zone
strong_anomaly
```

`A0` performs inference only. `A1` updates through the verified non-empty PNN
reconstruction path. `A2` may update through the guarded hard-old path or the
verified PNN path. Only `online_mlp_projector` may change in the accepted
THESIS online paths.

The online runner keeps `VerificationBuffer`, TTL state, non-overlap guards,
absolute indices, event records, and resumable runtime state. Test labels stay
outside scoring, triage, verification, and adaptation.

Traditional methods remain frozen. CANDI and M2N2 keep their own adapter
policies. They must not inherit THESIS triage or PNN rules automatically.

## 13. Run, result, and report objects

The smallest useful internal objects are:

| Object | Owns |
| --- | --- |
| `DatasetInfo` | Source description and warnings |
| `SeriesSet` | One normalized source sequence |
| `Catalog` | Available datasets, entities, models, and variants |
| `RunRequest` | One dataset/entity/method/variant/seed selection |
| `RunConfig` | Resolved runtime values |
| `DataPipeline` | Split, scaling, windowing, and metadata |
| `MethodRunner` | Fit, score, and prediction boundary |
| `ScoreSet` | Point and window scores plus coverage |
| `Result` | One completed or failed run |
| `Report` | Results and metric rows from one multi-run call |

`Report.table()` must show at least:

```text
dataset
entity
method
variant
seed
status
VUS-PR@FPR-budget
VUS-PR
VUS-ROC
Affiliation F1-score
raw-FPR
warning
```

## 14. Artifacts and directory layout

The sibling library uses the canonical experiment layout:

```text
outputs/<experiment_type>/<dataset_name>/<entity_name>/<seed_value>/<method_name>/<phase_name>/<stage_name>/
```

A stage directory keeps only report-ready values by default:

```text
run_manifest.json
resolved_config.yaml
metrics.json
metrics.csv
metrics.md
provenance.json
checkpoint.pt          # only for stages that create one
thresholds.json        # for calibration stages
```

The library computes intermediate values on the fly. It does not save every
forward-pass output unless the user explicitly selects a diagnostic retention
mode.

Every artifact records:

```text
dataset identity
entity identity
split identity
method and variant
seed
window and stride
score protocol
checkpoint role and hash
threshold identity
parser and source hash
configuration hash
code revision
device
status and warnings
```

The default is fail-closed. A missing, corrupt, incomplete, or mismatched
artifact is reported as a failed run. The library does not overwrite an
existing research output without an explicit resume policy.

W&B support is an optional tracker boundary. The local artifacts remain the
source of truth for a run. Human-readable W&B names keep phase, method or
variant, entity, and seed; detailed provenance belongs in metadata.

## 15. Minimal module layout

The first codebase should have this shape:

```text
tsad-lib/
├── pyproject.toml
├── README.md
├── src/tsad/
│   ├── __init__.py
│   ├── api.py
│   ├── cli.py
│   ├── types.py
│   ├── catalog.py
│   ├── config.py
│   ├── experiment.py
│   ├── pipeline.py
│   ├── scores.py
│   ├── metrics.py
│   ├── reporting.py
│   ├── artifacts.py
│   ├── data/
│   │   ├── common.py
│   │   ├── discovery.py
│   │   ├── adapters.py
│   │   ├── windows.py
│   │   └── adapters/
│   │       ├── smd.py
│   │       ├── server_machine.py
│   │       ├── nasa.py
│   │       ├── swat.py
│   │       ├── iops.py
│   │       ├── anomaly_archive.py
│   │       ├── tsb_ad.py
│   │       ├── iccad.py
│   │       ├── mat.py
│   │       └── csv.py
│   └── models/
│       ├── base.py
│       ├── thesis.py
│       ├── redlamp.py
│       ├── candi.py
│       ├── m2n2.py
│       ├── stumpy.py
│       ├── kmeans_ad.py
│       ├── isolation_forest.py
│       └── time_series_library/
│           ├── registry.py
│           ├── runner.py
│           ├── forward.py
│           ├── dependencies.py
│           ├── models/
│           └── layers/
└── tests/
    ├── test_api.py
    ├── test_catalog.py
    ├── test_data_adapters.py
    ├── test_pipeline.py
    ├── test_scores.py
    ├── test_metrics.py
    ├── test_artifacts.py
    └── test_models.py
```

The dependency direction is:

```text
api and cli
→ experiment orchestration
→ pipeline, models, metrics, reporting
→ data adapters and common types
```

Data adapters must not import models. Models must not inspect raw file formats.
The public API must not contain parsing rules.

## 15.1 Implementation inventory

The sibling directory starts empty. The coding agent therefore creates the
following small set of modules and objects. These are proposed runtime names,
not claims about code that already exists.

### Public and orchestration modules

| Module | Objects or functions | Small responsibility |
| --- | --- | --- |
| `src/tsad/__init__.py` | public exports | Expose the five public verbs. |
| `src/tsad/api.py` | `available`, `inspect`, `load`, `run` | Give notebook users one simple entry point. |
| `src/tsad/cli.py` | `main`, `build_parser` | Convert short CLI commands into API calls. |
| `src/tsad/catalog.py` | `Catalog` | List built-in datasets, models, variants, and metrics. |
| `src/tsad/config.py` | `load_config`, `resolve_config`, `validate_config` | Turn defaults and user choices into one checked `RunConfig`. |
| `src/tsad/experiment.py` | `ExperimentRunner` | Expand selections, run one request at a time, and collect results. |
| `src/tsad/models/time_series_library/registry.py` | `ModelSpec`, `TimeSeriesLibraryCatalog` | Register all 40 reference model names and their readiness states. |
| `src/tsad/models/time_series_library/runner.py` | `TimeSeriesLibraryRunner` | Adapt one reference model to the common anomaly method boundary. |
| `src/tsad/models/time_series_library/forward.py` | `ForwardAdapter`, adapter implementations | Convert model-specific inputs and outputs into reconstruction scores. |
| `src/tsad/models/time_series_library/dependencies.py` | `check_dependencies` | Report missing optional packages before a run starts. |

### Core data, score, and result classes

| Class | Attributes | Methods |
| --- | --- | --- |
| `SeriesSet` | `values`, `timestamps`, `labels`, `channel_names`, `dataset_name`, `entity`, `split`, `provenance`, `capabilities` | `validate()` |
| `DatasetInfo` | `dataset_name`, `source_path`, `entities`, `file_extensions`, `shapes`, `warnings`, `required_choices`, `capabilities` | `to_rows()` |
| `PipelineState` | `scaler`, `window_size`, `stride`, `split_names`, `score_space`, `provenance` | `validate()` |
| `PreparedData` | `train`, `validation`, `test`, `state` | `splits()` |
| `Window` | `values`, `start_index`, `end_index`, `split` | `validate()` |
| `RunRequest` | `dataset`, `entity`, `model`, `offline_variant`, `online_variant`, `seed` | `label()` |
| `RunConfig` | `data_root`, `output_root`, `window_size`, `offline_stride`, `online_stride`, `score_space`, `point_score_transform`, `fusion_mode`, `losses`, `metric_budgets`, `device` | `validate()` |
| `ScoreSet` | `point_scores`, `window_scores`, `start_indices`, `end_indices`, `coverage`, `labels`, `aux` | `validate()` |
| `Thresholds` | `point_threshold`, `window_threshold`, `fit_split`, `score_space`, `protocol` | `validate()` |
| `PredictionSet` | `point_predictions`, `window_predictions`, `start_indices`, `end_indices`, `coverage` | `validate()` |
| `ModelState` | `model_name`, `variant`, `seed`, `checkpoint_role`, `payload`, `metadata` | `validate()` |
| `Result` | `request`, `status`, `metrics`, `coverage`, `artifact_paths`, `warnings`, `error` | `to_row()` |
| `Report` | `results`, `metric_rows`, `output_root` | `table()`, `to_csv()`, `to_json()`, `to_markdown()` |

The class rules are deliberately small. A data class stores data. A pipeline
prepares data. A method runner owns model computation. A report formats results.
No class reaches into a responsibility owned by another class.

### Dataset and preparation classes

`src/tsad/data/common.py` owns `SeriesSet`, `DatasetInfo`, and source metadata.
`src/tsad/data/discovery.py` owns file discovery and `discover_sources()`.
`src/tsad/data/adapters.py` owns the `DatasetAdapter` protocol:

```python
class DatasetAdapter(Protocol):
    def can_read(self, path: Path) -> bool: ...
    def inspect(self, path: Path) -> DatasetInfo: ...
    def load(self, path: Path, options: dict) -> list[SeriesSet]: ...
```

`src/tsad/data/adapters/` contains one small adapter class for each required
family: `SmdAdapter`, `ServerMachineAdapter`, `NasaAdapter`, `SwatAdapter`,
`IopsAdapter`, `UcrTextAdapter`, `TsbArchiveAdapter`, `IccadAdapter`,
`MatAdapter`, and `CsvAdapter`. Each class implements the same three methods.

`src/tsad/data/windows.py` owns `make_windows()` and absolute-index rules.
`src/tsad/pipeline.py` owns `DataPipeline`, whose methods are:

```python
class DataPipeline:
    def split(self, series: SeriesSet) -> dict[str, SeriesSet]: ...
    def fit_scaler(self, train: SeriesSet) -> PipelineState: ...
    def transform(self, series: SeriesSet, state: PipelineState) -> SeriesSet: ...
    def prepare(self, series: SeriesSet) -> PreparedData: ...
```

Only `train` reaches `fit_scaler()`. The pipeline keeps labels available for
final evaluation but never sends test labels to training, memory initialization,
threshold fitting, or online adaptation.

### Method and runtime classes

`src/tsad/models/base.py` owns the small `MethodRunner` protocol:

```python
class MethodRunner(Protocol):
    def fit(self, data: PreparedData, config: RunConfig) -> ModelState: ...
    def score(self, data: PreparedData, state: ModelState) -> ScoreSet: ...
    def predict(self, scores: ScoreSet, thresholds: Thresholds) -> PredictionSet: ...
```

The model modules contain `ThesisRunner`, `RedLampRunner`, `CandiRunner`,
`M2N2Runner`, `StumpyRunner`, `KMeansADRunner`, and `IsolationForestRunner`.
Each runner implements the three protocol methods. Each runner keeps native
model computation inside its own module.

`src/tsad/models/time_series_library/registry.py` owns these additional
objects:

| Object | Attributes | Methods |
| --- | --- | --- |
| `ModelSpec` | `name`, `source_module`, `task`, `status`, `required_packages`, `input_mode`, `output_mode`, `default_config` | `supports()`, `to_row()` |
| `TimeSeriesLibraryCatalog` | `specs`, `source_root` | `list()`, `get()`, `ready()` |
| `DependencyReport` | `model_name`, `missing_packages`, `warnings` | `ok()` |

`src/tsad/models/time_series_library/runner.py` owns
`TimeSeriesLibraryRunner`. Its attributes are `spec`, `model`, `adapter`, and
`dependency_report`. Its methods are `fit()`, `score()`, and `predict()`.

`src/tsad/models/time_series_library/forward.py` owns the small
`ForwardAdapter` protocol:

```python
class ForwardAdapter(Protocol):
    def build_inputs(self, window: Tensor, config: RunConfig) -> tuple: ...
    def call(self, model: Module, inputs: tuple) -> Tensor: ...
    def extract_reconstruction(self, output: Tensor, window: Tensor) -> Tensor: ...
```

The first adapter families are `DirectReconstructionAdapter`,
`ForecastingSignatureAdapter`, and `StatsAwareAdapter`. A separate
`FoundationModelAdapter` is added only for a model whose zero-shot output has a
defined anomaly score. An adapter must preserve the raw-input score contract;
it must not copy the reference threshold or detection-adjustment procedure.

`ThesisRunner` additionally owns these small internal objects:

| Object | Attributes | Methods |
| --- | --- | --- |
| `ThesisOfflineState` | `stage_a_state`, `continuous_prototype_bank`, `discrete_codebook`, `stage_b_state`, `thresholds` | `validate()` |
| `MemoryInitializer` | `continuous_bank_size`, `discrete_codebook_size`, `seed` | `fit(training_data)`, `save_metadata()` |
| `ThesisOnlineState` | `thresholds`, `ewma_state`, `verification_state`, `projector_state`, `absolute_index` | `validate()`, `save()`, `load()` |

The THESIS training methods are `fit_stage_a()`, `initialize_memory()`,
`fit_stage_b()`, and `fit_thresholds()`. The THESIS scoring method uses the
existing `_compute_two_view_contrastive_loss` computation when O2 selects the
matrix name `point-level contrastive loss`. It does not create a second loss.
The default Stage B method is `direct_branch_routing`; it sends the continuous
branch to reconstruction, sends the discrete branch to classification, and
skips the fusion blocks.

### Scores, metrics, artifacts, and reports

`src/tsad/scores.py` owns point and window reduction functions. Its main
functions are `reduce_windows_to_points()` and `build_score_set()`.

`src/tsad/metrics.py` owns the metric definitions and `compute_metrics()`. The
initial metric names are `VUS-PR@FPR-budget`, `VUS-PR`, `VUS-ROC`,
`Affiliation F1-score`, and `raw-FPR`. Each metric row carries its priority,
budget when applicable, availability, and warning.

`src/tsad/artifacts.py` owns `ArtifactStore`. Its methods are
`write_manifest()`, `write_config()`, `write_metrics()`, `write_checkpoint()`,
`read_checkpoint()`, and `validate_run()`. It writes the canonical stage files
and refuses mismatched or incomplete research artifacts.

`src/tsad/reporting.py` owns `ReportBuilder`. Its methods are
`add_result()`, `build()`, and `write()`. The builder keeps undefined metrics
undefined and writes the same rows to CSV, JSON, and Markdown.

The first release uses functions and protocols where a class adds no value.
It does not add a plugin manager, dependency-injection container, workflow
engine, event bus, or class hierarchy for its own sake.

## 16. Chosen design patterns

The design uses only patterns that solve a current problem:

| Pattern | Small use |
| --- | --- |
| Facade | `available`, `inspect`, `load`, and `run` hide internal objects |
| Adapter | Each physical dataset format becomes `SeriesSet` |
| Strategy | A method owns its native split, score, or update policy when needed |
| Protocol | Small structural interfaces avoid a deep inheritance tree |
| Explicit catalog | One dictionary lists built-in datasets, methods, and metrics |

The first release does not use a plugin framework, dependency injection
container, event bus, or general workflow engine.

## 17. Development phases, stages, and atomic steps

The coding agent tells this story in ten core phases and two short
Time-Series-Library phases. The phase map gives the large order. Each phase then
opens into short stages. Every numbered step performs one action, so the agent
can finish and check it before moving on.

Phase 6A and Phase 6B are subphases of Phase 6. They are not new numeric
phases. The fixed order is `1 → 2 → 3 → 4 → 5 → 6 → 6A → 6B → 7 → 8 → 9 →
10`. A phase runner must keep this order.

Each stage below ends with a stage completion check. The agent must pass that
check before opening the next stage. A stage verification gate may run several
commands or a complete smoke flow. It is a gate, not an indivisible atomic
step.

### 17.0 Checklist status

This section is also the implementation checklist. `[x]` means that current
source files and tests provide evidence for the atomic step. `[ ]` means that
the step is still open. A checked atomic step does not by itself close the
stage or phase; the stage and phase completion checks still apply.

The checklist was last audited on 2026-09-14. The sibling test suite passed
45 tests. The public notebook API, full experiment runner, and CLI remain open
where their source code is still a placeholder.

### 17.1 High-level phase map

| Phase | Result | Main tools and technologies | Depends on |
| --- | --- | --- | --- |
| 1 | An importable empty package exists. | Python, `pyproject.toml`, pytest | None |
| 2 | One validated common data object exists. | Python, NumPy, pytest | Phase 1 |
| 3 | The current dataset families can be discovered and loaded. | Python, NumPy, pandas, SciPy, PyArrow, ZIP reader, pytest | Phase 2 |
| 4 | Windows and scaling preserve split boundaries. | Python, NumPy, scikit-learn, pytest | Phase 3 |
| 5 | One baseline produces a report. | Python, scikit-learn, pytest | Phase 4 |
| 6 | Metrics and reports have one stable contract. | Python, NumPy, pandas, pytest | Phase 5 |
| 6A subphase | All 40 reference models appear in the catalog with readiness states. | Python, pathlib, pytest | Phase 6 |
| 6B subphase | The 15 reference anomaly paths use the common adapter boundary. | Python, PyTorch, NumPy, pytest | Phase 6A subphase |
| 7 | THESIS offline O0, O1, and O2 paths are represented and checked. | Python, PyTorch, scikit-learn, pytest | Phase 6B subphase |
| 8 | The THESIS online causal path can resume safely. | Python, PyTorch, pytest | Phase 7 |
| 9 | Notebook and CLI use one matrix runner. | Python, `argparse`, pytest | Phase 8 |
| 10 | One smoke combination proves readiness for the SMD matrix. | Python, pytest, local artifacts, optional W&B | Phase 9 |

The tools named here are implementation choices already required by the data
formats and research runtime. The agent must verify dependency availability
before using a family-specific reader.

### Phase 1: create the empty library

**Story:** The agent first gives the sibling directory a door. The door opens
when `import tsad` works and the five public verbs have names.

**Modules:** `pyproject.toml`, `src/tsad/__init__.py`, `api.py`, `cli.py`.

**Tools:** Python, `pyproject.toml`, pytest.

#### Stage 1.1: create the package shell

**Tools:** Python, filesystem, `pyproject.toml`.

**Atomic steps:**

- [x] 1. Create `pyproject.toml`.
- [x] 2. Create `src/tsad/__init__.py`.
- [x] 3. Create `src/tsad/api.py`.
- [x] 4. Create `src/tsad/cli.py`.

**Stage complete when:** the four package files exist in the sibling directory.

#### Stage 1.2: prove the public import

**Tools:** pytest, Python import system.

**Atomic steps:**

- [x] 1. Create `tests/test_api.py`.
- [x] 2. Add an import test for `tsad`.
- [x] 3. Add a symbol test for `available`, `inspect`, `load`, `run`, and `report`.
- [x] 4. Run the focused API test.

**Stage complete when:** the focused API test passes.

**Phase complete when:** the package imports and the focused test passes.

### Phase 2: make one common data object

**Story:** The agent next gives every future adapter one shared language. A
synthetic sequence enters as `SeriesSet`, and invalid data stops at the door.

**Modules:** `types.py`, `data/common.py`, `tests/test_types.py`.

**Tools:** Python, NumPy, pytest.

#### Stage 2.1: define the source contract

**Tools:** Python dataclasses, NumPy.

**Atomic steps:**

- [x] 1. Define `SeriesSet`.
- [x] 2. Add `SeriesSet.validate()`.
- [x] 3. Define `DatasetInfo`.
- [x] 4. Add provenance and capability fields.

**Stage complete when:** `SeriesSet` and `DatasetInfo` expose the required source contract.

#### Stage 2.2: prove valid and invalid shapes

**Tools:** pytest, NumPy.

**Atomic steps:**

- [x] 1. Create a valid synthetic `SeriesSet` fixture.
- [x] 2. Test that the valid fixture passes validation.
- [x] 3. Test that a one-dimensional value array fails validation.
- [x] 4. Test that a label with the wrong length fails validation.

**Stage complete when:** the valid fixture passes and both invalid fixtures fail clearly.

**Phase complete when:** one valid source passes and each required invalid shape
fails with a clear error.

### Phase 3: discover and load every current dataset family

**Story:** The common language now meets the real data folder. The agent walks
through each file family, asks what it contains, and loads only what the source
can prove. When a MAT variable or CSV column is ambiguous, the story pauses and
asks the user instead of guessing.

**Modules:** `data/discovery.py`, `data/adapters.py`, `data/adapters/*.py`,
`catalog.py`.

**Tools:** Python, NumPy, pandas, SciPy, PyArrow, standard-library ZIP reader,
pytest.

#### Stage 3.1: create discovery and catalog contracts

**Tools:** Python, pathlib, pytest.

**Atomic steps:**

- [x] 1. Define `DatasetAdapter`.
- [x] 2. Implement `discover_sources()`.
- [x] 3. Add cache-file ignore rules.
- [x] 4. Add unrelated-metadata ignore rules.
- [x] 5. Define `Catalog` dataset entries.

**Stage complete when:** discovery and catalog contracts exist and both ignore rules are tested by inspection.

#### Stage 3.2: add array and text adapters

**Tools:** NumPy, pandas, pytest.

**Atomic steps:**

- [x] 1. Implement `SmdAdapter`.
- [x] 2. Implement `ServerMachineAdapter`.
- [x] 3. Implement `NasaAdapter`.
- [x] 4. Implement `IopsAdapter`.
- [x] 5. Implement `UcrTextAdapter`.

**Stage complete when:** the five array and text adapters return the common data contract for their fixtures.

#### Stage 3.3: add table, archive, and MATLAB adapters

**Tools:** pandas, SciPy, PyArrow, ZIP reader, pytest.

**Atomic steps:**

- [x] 1. Implement `SwatAdapter`.
- [x] 2. Implement `TsbArchiveAdapter`.
- [x] 3. Implement `IccadAdapter`.
- [x] 4. Implement `MatAdapter`.
- [x] 5. Implement `CsvAdapter`.

**Stage complete when:** the five table, archive, and MATLAB adapters return the common data contract for their fixtures.

#### Stage 3.4: prove inspection before loading

**Tools:** pytest, small fixtures from the current data folder.

**Atomic steps:**

- [x] 1. Add the SMD inspection test.
- [x] 2. Add the server-machine inspection test.
- [x] 3. Add the NASA inspection test.
- [x] 4. Add the IOPS inspection test.
- [x] 5. Add the UCR-text inspection test.
- [x] 6. Add the SWAT inspection test.
- [x] 7. Add the TSB-archive inspection test.
- [x] 8. Add the ICCAD inspection test.
- [x] 9. Add the MAT inspection test.
- [x] 10. Add the CSV inspection test.
- [x] 11. Add the SMD loading test.
- [x] 12. Add the server-machine loading test.
- [x] 13. Add the NASA loading test.
- [x] 14. Add the IOPS loading test.
- [x] 15. Add the UCR-text loading test.
- [x] 16. Add the SWAT loading test.
- [x] 17. Add the TSB-archive loading test.
- [x] 18. Add the ICCAD loading test.
- [x] 19. Add the MAT loading test.
- [x] 20. Add the CSV loading test.
- [x] 21. Add the MAT-variable ambiguity test.
- [x] 22. Add the CSV-column ambiguity test.

**Stage verification:** run the focused adapter tests.

**Stage complete when:** every adapter family has separate inspection and loading evidence, and both ambiguity tests fail safely.

**Phase complete when:** every current family has an adapter, inspection names
its entities, and loading returns validated `SeriesSet` objects.

### Phase 4: prepare windows without leakage

**Story:** The agent now prepares learning data. The training split fits the
scaler. Validation and test data are transformed by that fitted scaler. Windows
keep their absolute positions, so later metrics can return to the entity
timeline.

**Modules:** `pipeline.py`, `data/windows.py`, `types.py`.

**Tools:** Python, NumPy, scikit-learn, pytest.

#### Stage 4.1: fit and apply the train-only scaler

**Tools:** NumPy, scikit-learn, pytest.

**Atomic steps:**

- [x] 1. Define `PipelineState`.
- [x] 2. Implement `DataPipeline.fit_scaler()`.
- [x] 3. Implement `DataPipeline.transform()`.
- [x] 4. Test that only training values affect the scaler.

**Stage complete when:** the scaler is fitted from training values and transforms later splits.

#### Stage 4.2: create indexed offline and causal windows

**Tools:** Python, NumPy, pytest.

**Atomic steps:**

- [x] 1. Define `Window`.
- [x] 2. Implement `make_windows()`.
- [x] 3. Implement `DataPipeline.prepare()`.
- [x] 4. Test absolute start and end indices.
- [x] 5. Test offline stride `20`.
- [x] 6. Test online stride `1`.

**Stage complete when:** offline and causal windows carry the required indices and strides.

#### Stage 4.3: block label leakage

**Tools:** pytest, synthetic labels.

**Atomic steps:**

- [x] 1. Add a test with anomalous test labels.
- [x] 2. Assert that scaler fitting ignores test labels.
- [x] 3. Assert that training windows do not carry test labels.
- [x] 4. Run the focused pipeline tests.

**Stage complete when:** the focused pipeline tests show no test-label path into preparation.

**Phase complete when:** prepared data has correct splits, indices, scales, and
no test-label path into preparation.

### Phase 5: run one simple baseline

**Story:** Before the research model enters the story, one small frozen model
proves that a student can move from a data path to a report.

**Modules:** `models/base.py`, one baseline module, `experiment.py`,
`reporting.py`, `artifacts.py`.

**Tools:** Python, scikit-learn, pytest, local JSON/CSV/Markdown files.

#### Stage 5.1: define the method boundary

**Tools:** Python protocols, dataclasses.

**Atomic steps:**

- [x] 1. Define `MethodRunner`.
- [x] 2. Define `RunRequest`.
- [x] 3. Define `RunConfig`.
- [x] 4. Define `ModelState`.

**Stage complete when:** the method boundary can represent one fit, score, and predict request.

#### Stage 5.2: connect one baseline to one SMD entity

**Tools:** scikit-learn, NumPy, pytest.

**Atomic steps:**

- [x] 1. Implement `IsolationForestRunner.fit()`.
- [x] 2. Implement `IsolationForestRunner.score()`.
- [x] 3. Implement `IsolationForestRunner.predict()`.
- [x] 4. Add one SMD entity to a `RunRequest`.

**Stage verification:** run one end-to-end baseline test.

**Stage complete when:** the baseline accepts one SMD request and returns a score set and predictions.

#### Stage 5.3: write the first result

**Tools:** JSON, CSV, Markdown, pytest.

**Atomic steps:**

- [x] 1. Define `Result`.
- [x] 2. Define `Report`.
- [x] 3. Implement one report row.
- [x] 4. Write one JSON result.
- [x] 5. Write one CSV result.
- [x] 6. Write one Markdown result.

**Stage complete when:** one result is readable in JSON, CSV, and Markdown.

**Phase complete when:** one SMD entity runs through the public path and leaves
one readable report without a student-written loop.

### Phase 6: add the metric and report contract

**Story:** The first report now learns the project’s measurement language. Each
metric keeps its family, priority, budget, availability, and warning visible.

**Modules:** `metrics.py`, `scores.py`, `reporting.py`, `artifacts.py`.

**Tools:** Python, NumPy, pandas, pytest.

#### Stage 6.1: restore scores to the entity timeline

**Tools:** NumPy, pytest.

**Atomic steps:**

- [x] 1. Implement `reduce_windows_to_points()`.
- [x] 2. Implement `build_score_set()`.
- [x] 3. Define `ScoreSet` validation.
- [x] 4. Test overlapping-window coverage.

**Stage complete when:** overlapping window scores return to one validated entity timeline.

#### Stage 6.2: add the required metric rows

**Tools:** Python, NumPy, pandas, pytest.

**Atomic steps:**

- [x] 1. Define the metric-family catalog.
- [x] 2. Implement `VUS-PR@FPR-budget` rows.
- [x] 3. Implement `VUS-PR` rows.
- [x] 4. Implement `VUS-ROC` rows.
- [x] 5. Implement `Affiliation F1-score` rows.
- [x] 6. Implement `raw-FPR` rows.

**Stage complete when:** the metric catalog can create every required metric row.

#### Stage 6.3: preserve undefined results

**Tools:** pytest, synthetic labeled and unlabeled fixtures.

**Atomic steps:**

- [x] 1. Add a metric availability field.
- [x] 2. Add a metric warning field.
- [x] 3. Test that an unavailable metric is not zero.
- [x] 4. Test the `0.001` SMD FPR budget.
- [x] 5. Test the `0.005` SMD FPR budget.
- [x] 6. Test the `0.01` SMD FPR budget.

**Stage verification:** run the focused metric tests.

**Stage complete when:** unavailable metrics and all three SMD budgets remain explicit in the report.

**Phase complete when:** one report shows all required metric columns and keeps
unavailable values explicit.

### Phase 6A: register all reference models

**Story:** The metric table is now stable, so the agent can name every model
without pretending that every model is ready. The catalog records all 40
reference names, their task family, dependencies, and readiness state.

**Modules:** `catalog.py`, `models/time_series_library/registry.py`,
`models/time_series_library/dependencies.py`.

**Tools:** Python, pathlib, package metadata, pytest.

#### Stage 6A.1: create the reference model inventory

**Tools:** Python, pathlib, pytest.

**Atomic steps:**

- [x] 1. Add one `ModelSpec` row for `Autoformer`.
- [x] 2. Add one `ModelSpec` row for `Chronos`.
- [x] 3. Add one `ModelSpec` row for `Chronos2`.
- [x] 4. Add one `ModelSpec` row for `Crossformer`.
- [x] 5. Add one `ModelSpec` row for `DLinear`.
- [x] 6. Add one `ModelSpec` row for `ETSformer`.
- [x] 7. Add one `ModelSpec` row for `FEDformer`.
- [x] 8. Add one `ModelSpec` row for `FiLM`.
- [x] 9. Add one `ModelSpec` row for `FreTS`.
- [x] 10. Add one `ModelSpec` row for `Informer`.
- [x] 11. Add one `ModelSpec` row for `KANAD`.
- [x] 12. Add one `ModelSpec` row for `Koopa`.
- [x] 13. Add one `ModelSpec` row for `LightTS`.
- [x] 14. Add one `ModelSpec` row for `MICN`.
- [x] 15. Add one `ModelSpec` row for `MSGNet`.
- [x] 16. Add one `ModelSpec` row for `Mamba`.
- [x] 17. Add one `ModelSpec` row for `MambaSimple`.
- [x] 18. Add one `ModelSpec` row for `Moirai`.
- [x] 19. Add one `ModelSpec` row for `MultiPatchFormer`.
- [x] 20. Add one `ModelSpec` row for `Nonstationary_Transformer`.
- [x] 21. Add one `ModelSpec` row for `PAttn`.
- [x] 22. Add one `ModelSpec` row for `PatchTST`.
- [x] 23. Add one `ModelSpec` row for `Pyraformer`.
- [x] 24. Add one `ModelSpec` row for `Reformer`.
- [x] 25. Add one `ModelSpec` row for `SCINet`.
- [x] 26. Add one `ModelSpec` row for `SegRNN`.
- [x] 27. Add one `ModelSpec` row for `Sundial`.
- [x] 28. Add one `ModelSpec` row for `TSMixer`.
- [x] 29. Add one `ModelSpec` row for `TemporalFusionTransformer`.
- [x] 30. Add one `ModelSpec` row for `TiDE`.
- [x] 31. Add one `ModelSpec` row for `TiRex`.
- [x] 32. Add one `ModelSpec` row for `TimeFilter`.
- [x] 33. Add one `ModelSpec` row for `TimeMixer`.
- [x] 34. Add one `ModelSpec` row for `TimeMoE`.
- [x] 35. Add one `ModelSpec` row for `TimeXer`.
- [x] 36. Add one `ModelSpec` row for `TimesFM`.
- [x] 37. Add one `ModelSpec` row for `TimesNet`.
- [x] 38. Add one `ModelSpec` row for `Transformer`.
- [x] 39. Add one `ModelSpec` row for `WPMixer`.
- [x] 40. Add one `ModelSpec` row for `iTransformer`.

Each row contains the model name, source module, task family, and initial
readiness state.

**Stage complete when:** all 40 `ModelSpec` rows exist.

#### Stage 6A.2: show model readiness to students

**Tools:** Python, table formatting, pytest.

**Atomic steps:**

- [x] 1. Add `ModelSpec.to_row()`.
- [x] 2. Add `TimeSeriesLibraryCatalog.list()`.
- [x] 3. Add the model columns to `available(DATA)`.
- [x] 4. Test that pending models remain visible.

**Stage complete when:** `available(DATA)` shows every model and its readiness state.

**Subphase complete when:** `available(DATA)` lists all 40 reference models and
shows why each model is ready, pending, or blocked.

### Phase 6B: certify reference anomaly adapters

**Story:** The agent now gives the 15 script-backed anomaly models a common
runtime path. The agent copies or isolates only the model and layer code needed
by a selected model. The reference experiment wrapper is not reused as the
scientific contract.

**Modules:** `models/time_series_library/runner.py`,
`models/time_series_library/forward.py`, `models/time_series_library/models/`,
`models/time_series_library/layers/`, `tests/test_models.py`.

**Tools:** Python, PyTorch, NumPy, pytest, optional model packages.

#### Stage 6B.1: define the common adapter boundary

**Tools:** Python protocols, PyTorch, pytest.

**Atomic steps:**

- [x] 1. Define `ForwardAdapter`.
- [x] 2. Define `TimeSeriesLibraryRunner`.
- [x] 3. Define reconstruction output validation.
- [x] 4. Test a synthetic model with the adapter boundary.

**Stage complete when:** a synthetic model passes through the common adapter boundary.

#### Stage 6B.2: add the first anomaly model group

**Tools:** PyTorch, NumPy, pytest.

**Atomic steps:**

- [x] 1. Add `TimesNet` model sources.
- [x] 2. Add `Autoformer` model sources.
- [x] 3. Add `Transformer` model sources.
- [x] 4. Add `KANAD` model sources.
- [x] 5. Run the `TimesNet` reconstruction test.
- [x] 6. Run the `Autoformer` reconstruction test.
- [x] 7. Run the `Transformer` reconstruction test.
- [x] 8. Run the `KANAD` reconstruction test.

**Stage complete when:** the four first-group reconstruction tests pass.

#### Stage 6B.3: add the remaining script-backed group

**Tools:** PyTorch, NumPy, pytest.

**Atomic steps:**

- [x] 1. Add `Crossformer` model sources.
- [x] 2. Add `DLinear` model sources.
- [x] 3. Add `ETSformer` model sources.
- [x] 4. Add `FEDformer` model sources.
- [x] 5. Add `FiLM` model sources.
- [x] 6. Add `Informer` model sources.
- [x] 7. Add `LightTS` model sources.
- [x] 8. Add `MICN` model sources.
- [x] 9. Add `Pyraformer` model sources.
- [x] 10. Add `Reformer` model sources.
- [x] 11. Add `iTransformer` model sources.
- [x] 12. Run the `Crossformer` reconstruction test.
- [x] 13. Run the `DLinear` reconstruction test.
- [x] 14. Run the `ETSformer` reconstruction test.
- [x] 15. Run the `FEDformer` reconstruction test.
- [x] 16. Run the `FiLM` reconstruction test.
- [x] 17. Run the `Informer` reconstruction test.
- [x] 18. Run the `LightTS` reconstruction test.
- [x] 19. Run the `MICN` reconstruction test.
- [x] 20. Run the `Pyraformer` reconstruction test.
- [x] 21. Run the `Reformer` reconstruction test.
- [x] 22. Run the `iTransformer` reconstruction test.

**Stage complete when:** the eleven remaining script-backed reconstruction tests pass.

#### Stage 6B.4: keep unverified models visible but blocked

**Tools:** Python, dependency checks, pytest.

**Atomic steps:**

- [x] 1. Mark the remaining 25 models `not_yet_tsad_ready`.
- [x] 2. Record missing dependency warnings.
- [x] 3. Reject a run for a model without a valid anomaly adapter.
- [x] 4. Test that the report records the rejected cell.

**Stage complete when:** pending models remain visible and non-ready runs are rejected with a recorded reason.

**Subphase complete when:** the 15 script-backed models pass the common anomaly
adapter tests and the other 25 models remain visible without false readiness.

### Phase 7: add THESIS offline training

**Story:** The agent now opens the research model’s offline chapters in order.
Stage A learns the task. Memory initialization reads training data only. Stage B
uses the frozen encoder and memories. For O2, the story selects reconstruction,
classification, and the existing two-view contrastive computation, disables
Balanced Point-Score Loss, and uses `direct_branch_routing` with fusion blocks
skipped by default.

**Modules:** `models/thesis.py`, `config.py`, `scores.py`, `artifacts.py`,
`experiment.py`.

**Tools:** Python, PyTorch, scikit-learn k-means, NumPy, pytest.

#### Stage 7.1: represent offline variants and losses

**Tools:** Python, YAML or equivalent resolved configuration, pytest.

**Atomic steps:**

- [x] 1. Add O0 to the variant catalog.
- [x] 2. Add O1 to the variant catalog.
- [x] 3. Add O2 to the variant catalog.
- [x] 4. Add O2’s three-loss configuration.
- [x] 5. Set O2 `balanced_point_score_loss` to `false`.
- [x] 6. Set O2 `fusion_mode` to `direct_branch_routing`.
- [x] 7. Test that O2 rejects Balanced Point-Score Loss.

**Stage complete when:** O0, O1, and O2 have validated variant records, and O2 rejects Balanced Point-Score Loss.

#### Stage 7.2: implement Stage A and memory initialization

**Tools:** PyTorch, NumPy, scikit-learn k-means, pytest.

**Atomic steps:**

- [x] 1. Implement `ThesisRunner.fit_stage_a()`.
- [x] 2. Implement `MemoryInitializer.fit()`.
- [x] 3. Create the continuous prototype bank.
- [x] 4. Create the discrete codebook.
- [x] 5. Write the Stage A checkpoint.
- [x] 6. Write the memory initialization checkpoint.

**Stage complete when:** Stage A and training-only memory initialization produce their two checkpoints.

#### Stage 7.3: implement Stage B and direct routing

**Tools:** PyTorch, pytest.

**Atomic steps:**

- [x] 1. Implement `ThesisRunner.fit_stage_b()`.
- [x] 2. Freeze the shared encoder.
- [x] 3. Freeze the continuous prototype bank.
- [x] 4. Freeze the discrete codebook.
- [x] 5. Route the continuous branch to reconstruction.
- [x] 6. Route the discrete branch to classification.
- [x] 7. Skip the fusion blocks by default.
- [x] 8. Test the explicit fusion override record.

**Stage complete when:** Stage B freezes the required objects, records direct routing, and records any explicit fusion override.

#### Stage 7.4: implement O2 scoring and threshold artifacts

**Tools:** PyTorch, NumPy, pytest, JSON.

**Atomic steps:**

- [x] 1. Reuse `_compute_two_view_contrastive_loss` for O2.
- [x] 2. Test clean-point filtering across the complete batch.
- [x] 3. Test zero loss when no clean point remains.
- [x] 4. Implement `ThesisRunner.fit_thresholds()`.
- [x] 5. Write the clean-validation threshold artifact.
- [x] 6. Test raw-input MSE with identity transformation.

**Stage verification:** run one complete O2 offline smoke test.

**Stage complete when:** the O2 score and threshold artifacts pass their focused tests and the complete smoke test produces provenance.

**Phase complete when:** O2 runs one entity with its fixed losses, direct branch
routing, checkpoints, threshold artifact, scores, metrics, and provenance.

### Phase 8: add THESIS online execution

**Story:** The offline source is now frozen. The online runner receives its
checkpoint and threshold artifact, moves through causal windows, and changes only
the allowed online module.

**Modules:** `models/thesis.py`, `pipeline.py`, `artifacts.py`, `experiment.py`.

**Tools:** Python, PyTorch, NumPy, pytest.

#### Stage 8.1: add causal scoring and triage

**Tools:** NumPy, PyTorch, pytest.

**Atomic steps:**

- [x] 1. Implement causal window creation.
- [x] 2. Implement EWMA score updates.
- [x] 3. Implement the four triage regions.
- [x] 4. Test absolute online indices.

**Stage complete when:** the online scorer produces causal scores, EWMA values, triage regions, and absolute indices.

#### Stage 8.2: add verification and allowed updates

**Tools:** Python, PyTorch, pytest.

**Atomic steps:**

- [x] 1. Define `VerificationBuffer` state.
- [x] 2. Implement A0 inference-only behavior.
- [x] 3. Implement A1 verified updates.
- [x] 4. Implement A2 guarded updates.
- [x] 5. Test that only `online_mlp_projector` changes.

**Stage complete when:** A0, A1, and A2 obey their update rules and only the allowed module changes.

#### Stage 8.3: save and resume online state

**Tools:** JSON, PyTorch checkpoints, pytest.

**Atomic steps:**

- [x] 1. Define `ThesisOnlineState`.
- [x] 2. Implement `ThesisOnlineState.save()`.
- [x] 3. Implement `ThesisOnlineState.load()`.
- [x] 4. Test a resumed absolute index.
- [x] 5. Test a mismatched threshold artifact.

**Stage complete when:** online state resumes with the same absolute index and rejects a mismatched threshold artifact.

**Phase complete when:** A0 runs causally, A1/A2 obey their update gates, and a
resumed run preserves state and artifact identity.

### Phase 9: add the matrix runner and CLI

**Story:** The notebook path and terminal path now meet at one runner. A student
names models and datasets; the runner creates the small requests and records
every success or failure.

**Modules:** `api.py`, `cli.py`, `catalog.py`, `experiment.py`, `config.py`.

**Tools:** Python, `argparse`, pytest.

#### Stage 9.1: expand selections into requests

**Tools:** Python, catalog objects, pytest.

**Atomic steps:**

- [x] 1. Implement `Catalog.list_datasets()`.
- [x] 2. Implement `Catalog.list_models()`.
- [x] 3. Implement `Catalog.list_variants()`.
- [x] 4. Implement request expansion for `entities="all"`.
- [x] 5. Test one multi-model request set.

**Stage complete when:** one selection expands into the expected dataset, model, variant, and entity requests.

#### Stage 9.2: connect CLI and notebook paths

**Tools:** Python, `argparse`, pytest.

**Atomic steps:**

- [x] 1. Implement `available(DATA)` table output.
- [x] 2. Implement the CLI dataset-list command.
- [x] 3. Implement the CLI model-list command.
- [x] 4. Connect CLI `run` to `api.run()`.
- [x] 5. Test that notebook and CLI produce the same request objects.

**Stage complete when:** notebook and CLI paths produce the same request objects.

#### Stage 9.3: add preflight and failed-cell reporting

**Tools:** Python, pytest, local artifact checks.

**Atomic steps:**

- [x] 1. Implement capability validation.
- [x] 2. Implement dependency validation.
- [x] 3. Record one failed matrix cell.
- [x] 4. Preserve successful cells beside failed cells.
- [x] 5. Run the focused CLI tests.

**Stage complete when:** the preflight explains failed cells and preserves successful cells in the same report.

**Phase complete when:** the CLI and notebook call one runner and the report
accounts for every selected cell.

### Phase 10: prepare the all-machine SMD run

**Story:** The final chapter checks the whole SMD map before a large experiment.
The agent proves one complete combination, then checks the matrix without
silently replacing missing work or overwriting old research outputs.

**Modules:** `experiment.py`, `catalog.py`, `config.py`, `artifacts.py`,
`reporting.py`.

**Tools:** Python, pytest, local filesystem, optional W&B client.

#### Stage 10.1: validate the SMD matrix inputs

**Tools:** Python, NumPy, pytest.

**Atomic steps:**

- [x] 1. Discover all 28 SMD entities.
- [x] 2. Validate seed `6`.
- [x] 3. Validate seed `8`.
- [x] 4. Validate seed `36`.
- [x] 5. Validate window size `20`.
- [x] 6. Validate online stride `1`.
- [x] 7. Validate the selected 2048-point ranges.
- [x] 8. Validate the `0.001` FPR budget.
- [x] 9. Validate the `0.005` FPR budget.
- [x] 10. Validate the `0.01` FPR budget.

**Stage complete when:** all 28 entities, three seeds, required strides, ranges, and budgets pass validation.

#### Stage 10.2: run the required smoke combination

**Tools:** Python, pytest, local artifacts.

**Atomic steps:**

- [x] 1. Select one SMD entity.
- [x] 2. Select one seed.
- [x] 3. Select one complete model and variant combination.
- [x] 4. Read the generated report.
- [x] 5. Check the generated provenance.

**Stage verification:** run the complete offline and online flow for the selected combination.

**Stage complete when:** the selected combination produces a readable report and complete provenance.

#### Stage 10.3: close the matrix launch gate

**Tools:** Python, artifact validation, optional W&B client.

**Atomic steps:**

- [x] 1. Count the logical run units.
- [x] 2. Check the O2 integration gate.
- [x] 3. Check the A1/A2 ontology reconciliation gate.
- [x] 4. Check the metric formula status.
- [x] 5. Check the metric serialization status.
- [x] 6. Check non-SMD entity-selection status.
- [x] 7. Record the selected output-root policy.
- [x] 8. Record the selected resume policy.

**Stage complete when:** every launch gate has a recorded result and the output and resume policies are explicit.

The target remains:

```text
1,764 logical run units
2,352 W&B jobs after O2 integration and logger completion
```

**Phase complete when:** one smoke combination passes, every matrix cell is
accounted for, and the open launch gates are either closed or explicitly shown
as blocking.

**Evidence story:** The library counted 1,764 logical units and wrote a
preflight report. One `ServerMachineDataset/machine-1-1`, seed `6`, `O2-A0`
smoke run completed under `benchmark_smoke`. Its threshold records the matching
Stage B checkpoint hash. Its online range records 2,048 points. The wet matrix
has not started.

## 18. Acceptance story

The first acceptance gate passes when a student can load one SMD machine, run
one model, and read a metric table without writing a loop.

The model-catalog gate passes when `available(DATA)` lists all 40 reference
models, shows their task and readiness state, and rejects a non-ready selection
with a clear reason.

The second gate passes when every current dataset family can be inspected and
loaded into `SeriesSet`, with clear capability errors for unsupported methods.

The third gate passes when THESIS offline and online contracts produce complete
checkpoints, thresholds, scores, metrics, and provenance.

The final gate passes when the all-machine SMD matrix is structurally complete,
each logical cell is accounted for, unavailable metrics remain explicit, and no
run violates the leakage or artifact identity rules.

## 19. Open launch gates

Only these decisions remain open in this development contract:

1. Integrate the defined O2 contract into the generator, configuration,
   dependency checks, checkpoint inventory, and preflight validation.
2. Reconcile the SMD matrix A1/A2 component table with the authoritative online
   ontology and runtime policy.
3. Lock the final formula and serialization for `VUS-PR@FPR-budget` and
   `Affiliation F1-score` in the library metric interface.
4. Confirm the entity-selection rules for non-SMD datasets after adapter
   inspection.
5. Certify the 15 reference anomaly paths and record the readiness decision for
   the remaining 25 models.

Until these gates close, the library may implement discovery, loading, simple
baselines, reporting, and the already locked THESIS paths. It must not claim
that the complete O2 matrix is scientifically ready.

## Source story

This development contract combines the public API and adapter design from
[proposal-tsad-lib.md](../logs/2026-09-12/structure/proposal-tsad-lib.md), the
object relations from [tsad-lib-directed-graphs.md](tsad-lib-directed-graphs.md),
the naming rules from the two ontology documents, the scientific runtime rules
from the THESIS specifications, and the all-machine counts and dependencies
from [smd-all-machines-experiment-matrix.md](../notes/smd-all-machines-experiment-matrix.md).
