# Proposed runtime signatures

Date: 2026-11-16
Status: the replacement runtime is installed. The [execution record](runtime-replacement-review.md) lists the behavior checked by tests and bounded runs.

Binding details: [profile fields and defaults](runtime-profile-contract.md) and [ordered decisions](../plan/decisions-runtime-replacement.md). These records supersede earlier open-item statements below for the decisions they explicitly settle. The approved bindings control implementation.

## Confirmed decisions: 2026-11-16

Follow the [runtime replacement decisions](../plan/decisions-runtime-replacement.md). They settle the common CNN, separate method-owned weights, causal test-data use, RedLamp scoring changes, staged replacement, smoke selection, and result hierarchy. See the execution record for method-specific checks and limits.

## 1. Scope and design choices

1. Follow the [updated tree](../../2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md). Keep one public method class per method and functions for the other workflows.
2. Keep `scripts/preprocess.py` self-contained. All required data is processed. Runtime loading reads saved splits; it does not split or scale them again.
3. Reuse ordinary dictionaries, NumPy arrays, PyTorch modules, paths, and iterators. Do not add a plugin system, service classes, abstract base classes, mixins, or `typing.Protocol`.
4. Write only helpers that actual code needs. The signatures below define public calls, not a requirement to implement every helper in advance. Keep functions within 50 lines and code files within 500 lines.
5. Proposed choice: resolve configuration without side effects; allocate a run ID only when execution starts. This separates inspection from execution and changes the earlier resolver proposal.
6. The updated tree places methods in `src/methods/`, replacing the older preference for `src/models/`. Keep the single-public-entrypoint rule. File placement does not change method mathematics.
7. The thesis has no `documents/specs/README.md` index. This draft uses the named decision files and their stated statuses. Broader schemas and method readiness remain open. The confirmed decision record now settles encoder sharing and replaces the older result-depth limit.

The selected approach uses functions for workflow operations and classes for learned state. An all-class design would add lifecycle code to simple workflows. A large shared base class would couple methods with different training and update rules. Neither is needed here.

## 2. Small shared vocabulary

These are annotation conventions, not a proposed `types.py` or class hierarchy. Import standard names in the module that uses them. Repeat a short alias when that is clearer than adding another file.

```python
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn

Config = dict[str, Any]
Record = dict[str, Any]
Array = NDArray[np.generic]
```

1. `Config` is resolved settings. Required keys have one owner and are checked before use. `Record` is a named result shape described below, not an unrestricted replacement for a schema.
2. Config resolution does not mutate caller dictionaries. Arrays passed to scoring are read-only by contract. Method fitting and updates may change only method-owned state.
3. A prepared split record contains `x` with shape `[T, C]`, labels with unknown values preserved, optional timestamps, and source file/row mapping. Feature order comes from the manifest, never from incidental column order.
4. A prepared entity record contains `splits`, `features`, `scaler`, and `identity`. `splits` maps `train`, `validation`, and `test` to split records. `identity` includes dataset, entity, prepared version, manifest reference, and saved scaler reference/hash.
5. A batch record contains `x` with shape `[B, L, C]`, integer `indices` with shape `[B, L]` into its prepared split, optional labels/timestamps, and split identity. These indices differ from original source row numbers, which remain in source mapping.
6. A score record contains `score_kind`, `raw_scores`, and matching integer `indices`. Kind `point` uses shape `[N]`; kind `window` uses shape `[B, L]`. Native endpoint scores are point scores indexed at their actual endpoints. Optional auxiliary fields support calibration, uncertainty, or triage. Do not broadcast scores to force another shape.
7. A point-score record contains scores and a coverage mask of length `T`. Unknown labels and uncovered points are excluded only under the metric's documented eligibility rule. Do not replace them with normal labels or valid zero scores.
8. Method checkpoint state is a mapping of serializable values and tensors. It includes method identity, configuration identity, learned state, and state needed for exact continuation. It is not a JSON-only mapping.
9. Resume state also includes cursor, optimizer/scheduler state, random-generator state, scoring state, and method-specific buffers when used. A best checkpoint and a resume checkpoint have different roles.

### Public record fields

These tables define the common envelope. A method or metric may add only its documented fields. Exact scientific child schemas remain open; this document is not yet a complete implementation handoff.

| Record | Required fields | Optional fields |
| --- | --- | --- |
| Resolved config | `project_root`, `dataset`, `entity`, `method_name`, `variant`, `phase`, `seed`, `data`, `method`, `evaluation`, `profile_paths`, `prepared_identity` | `experiment_id`, standalone `run_id`, `output_dir`, `limits`, `required_offline_variant`, standalone parent directory |
| Execution config | resolved fields, `run_id`, `started_at`, `output_root` | `offline_parent` after selection |
| Prepared split | `x`, `labels`, `source_file`, `source_row`, `segments`, `identity` | `timestamps` |
| Prepared entity | `splits`, `features`, `scaler`, `identity` | none |
| Batch | `x`, `indices`, `segment_ids`, `identity` | `labels`, `timestamps`; remove labels before method scoring/updating |
| Method scores | `score_kind`, `raw_scores`, `indices`, `segment_ids`, `identity` | documented `auxiliary` fields |
| Point scores | `scores`, `coverage`, `segment_ids`, `identity` | uncertainty, `scored_at_index`, `finalized_at_index`, and `finalization_event` |
| Saved point scores | one nullable `score` column in selected split order | no per-point identity columns |
| Metric evaluation | `metrics`, `eligibility`, `segments`, `metric_metadata` | none |
| Run/result record | `run_id`, `status`, `started_at`, configured dimensions, `profile_paths`, `prepared_identity` | `completed_at`, `split`, `metrics`, `artifact_paths`, `offline_parent`, `error` |
| Training summary | `stages`, `selected_checkpoint` | validation summaries and method-specific calibration references |
| Update result | `did_update`, `loss` (`None` if unavailable) | documented diagnostics |
| Resume snapshot | recorded execution config, method state, cursor, RNG state, scoring state | optimizer/scheduler/buffers when used; recorded offline parent and calibration when required |

1. Proposed status values are `running`, `successful`, `failed`, `blocked`, and `interrupted`. A non-successful record may have no split or metrics. Terminal status is independent of split results.
2. Every identity names its dataset, entity, and prepared version. Score/calibration/checkpoint identities additionally record the settings and state needed by their compatibility check.
3. Unknown labels remain missing values in the loaded label array. Metric eligibility uses an explicit known-label mask. Timestamps remain absent when the source had none.
4. Calibration and auxiliary field schemas depend on the reviewed method and evaluation profiles. Reject unsupported kinds or missing applicable fields; a generic mapping annotation must not bypass checks.

## 3. `scripts/preprocess.py`: preserve the existing interface

```python
def preprocess(dataset, profile_path=None, input_root=None,
               output_root=None, entities=None): ...

def main(argv=None): ...
```

1. These are the existing parameter names. The current implementation returns the prepared version path from `preprocess`; `main` returns an exit code. Verify annotations against that implementation before adding them.
2. Keep its existing `EntityData` record and private processing helpers. Do not replace them with runtime records or import `src/` into this script.
3. The CLI forwards the supported arguments to this implementation. The default output root is `data/preprocessed/`. Source files are required for processing and source-based reuse checks. `--show-profile` does not read source data.

## 4. `src/cli.py`

```python
def main(argv: Sequence[str] | None = None) -> int: ...

def complete_names(kind: str, prefix: str, *, project_root: Path) -> list[str]: ...
```

1. `main` parses the desired command, calls its owner, and prints short results or errors. It returns zero on success and a nonzero exit code on failure. Parse/help functions remain private.
2. `complete_names` reuses config discovery for methods and datasets and experiment discovery for experiment names. It does not create files. Shell integration remains a CLI implementation detail.
3. `tsad preprocess` calls the script entrypoint with supported arguments. Other commands call the functions below; no second preprocessing implementation is introduced.

## 5. `src/config.py`

```python
def load_yaml(path: Path) -> Config: ...

def list_names(kind: str, *, project_root: Path) -> list[str]: ...

def resolve_run(cell: Mapping[str, Any], *, project_root: Path) -> Config: ...

def create_method(config: Config, *, encoder: nn.Module | None = None) -> Any: ...
```

1. `load_yaml` rejects duplicate keys, an empty document, and a non-mapping root. Missing files raise `FileNotFoundError`; invalid values raise `ValueError` with file and field context.
2. `list_names` accepts only the known discovery categories. Use one small explicit name-to-constructor/default table where needed, not runtime plugin discovery.
3. `resolve_run` resolves one run's profiles and defaults, preserves profile paths, and checks currently available inputs with the four protocol modules. It does not expand an experiment, create IDs, choose an offline parent, or write files.
4. The resolved record contains an absolute project root, dataset, entity, method, variant, phase, seed, profiles, prepared-data identity, limits, and output settings. Execution adds the run ID, selected parent, and timestamps. Exact method child fields remain method-specific.
5. `create_method` dispatches to the selected method class. `Any` marks the structurally checked method object; it does not authorize arbitrary methods. Check required methods with a Markdown checklist and text search. No fifth protocol is added.
6. Create a separate time-preserving CNN instance for MUSTARD, RedLamp, CANDI, and M2N2 using common architecture and settings. Each method owns its weights and update state. Do not share trained weights or a mutable module. MOMENT retains its pretrained encoder; traditional methods need no encoder.

## 6. `src/experiments.py`

```python
def list_experiments(*, project_root: Path) -> list[str]: ...

def create_experiment(name: str, *, project_root: Path) -> Path: ...

def edit_experiment(path: Path, operation: str, field: str,
                    values: Sequence[str], *, project_root: Path) -> Config: ...

def plan_experiment(path: Path, *, project_root: Path,
                    smoke: bool = False) -> list[Config]: ...

def run_experiment(path: Path, *, project_root: Path,
                   smoke: bool = False) -> list[Record]: ...

def resume_experiment(path: Path, *, project_root: Path) -> list[Record]: ...
```

1. `create_experiment` writes one definition and rejects an existing name. `edit_experiment` implements only the desired `add`, `remove`, and `set` operations and their allowed fields. It parses values by field; it is not a general override language.
2. `plan_experiment` is read-only. It resolves defaults and applicable method–dataset–entity–variant–seed combinations in stable order. Each dataset uses one data profile. Return fully specified run settings without run IDs.
3. The desired CLI's applicable combinations take precedence over the earlier seed-only expansion proposal. Exact supported combinations, schemas, and numeric smoke limits still need specification. Stop validation on unsupported combinations. Smoke uses the first configured dataset, entity, applicable variant, and seed per method and supported phase, with required offline dependencies. Keep prepared data unchanged and exclude smoke results from normal reports.
4. `show` uses the plan and displays default origins. `validate` checks the same plan and prepared-data readiness. Neither allocates a run ID.
5. `run_experiment` allocates execution identity and calls offline or online execution in dependency order. It returns records for completed, failed, or blocked runs; success must not hide failures.
6. `resume_experiment` finds interrupted executions and restores their exact recorded configuration and run IDs. It does not rerun successful runs or apply current defaults to old state. Detailed eligibility remains a recovery design task.
7. Standalone execution resolves one cell and calls the same offline or online function. A separate standalone runner class is unnecessary.

## 7. `src/prepared_data.py`

```python
def load_entity(prepared_root: Path, entity: str) -> Record: ...

def iter_batches(split: Record, data_config: Config, *,
                 phase: str, seed: int) -> Iterator[Record]: ...
```

1. `load_entity` checks the selected manifest, saved files, hashes, feature order, shapes, labels, and scaler identity. It reads prepared values without transforming them again. It never requires removed source files.
2. `iter_batches` creates windows lazily. Offline training may shuffle under the data profile and seed. Validation and test order follow their documented evaluation policy. Online iteration yields one causal window at a time in increasing split index order.
3. Do not join different splits to make a window. Validate nonempty usable windows and retain source gaps and duplicate observations. Tail, stride, gap-crossing, and overlapping online-score alignment require explicit policies before implementation.
4. Tensor conversion belongs where the method needs it. Native NumPy baselines need not depend on a PyTorch training abstraction.

## 8. Method entrypoints

Each file below contains one public class. The following signatures apply to its supported capabilities. They do not require a common parent class.

| File | Public class |
| --- | --- |
| `src/methods/mustard/model.py` | `Mustard` |
| `src/methods/baselines/redlamp.py` | `RedLamp` |
| `src/methods/baselines/stumpy_channel_ab.py` | `StumpyChannelAB` |
| `src/methods/baselines/kmeans_ad.py` | `KMeansAD` |
| `src/methods/baselines/iforest.py` | `IsolationForestAD` |
| `src/methods/baselines/candi.py` | `CANDI` |
| `src/methods/baselines/m2n2.py` | `M2N2` |
| `src/methods/baselines/moment.py` | `Moment` |

```python
class Mustard:
    def __init__(self, config: Config, *, encoder: nn.Module | None = None): ...

    def fit(self, train: Record, validation: Record, *,
            checkpoint_dir: Path, resume: Record | None = None,
            on_checkpoint: Callable[[Record], None] | None = None) -> Record: ...

    def score(self, batch: Record) -> Record: ...

    def update(self, batch: Record, context: Record) -> Record: ...

    def state_dict(self) -> Record: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
```

1. The seven baseline classes use the same constructor, `fit`, `score`, and state signatures where applicable. Only methods with confirmed online updates implement `update`. Capability metadata controls calls; absence is not silently treated as successful adaptation.
2. A fixed pretrained method need not run fitting. Its profile declares that capability and the workflow skips `fit`. A0 similarly skips updates. Do not create dummy optimizers or empty training stages.
3. `fit` owns scientific training, synthetic augmentation, validation monitoring, and state selection. It uses `artifacts.save_checkpoint` under the supplied directory. Its return record contains `selected_checkpoint`, `stages`, and applicable calibration references, not all training tensors.
4. Methods call `prepared_data.iter_batches` as needed for repeated epochs; do not hand them a one-use iterator that silently exhausts after one epoch. Sequence-based methods can use the prepared sequence directly.
5. `score` produces raw scores and required auxiliary outputs without fitting on test labels. It does not perform external shared smoothing or reporting. Preserve each method's score orientation and alignment; define unsupported mappings before exposing the method as ready.
6. `update` receives one window and the already recorded pre-update score, triage decision, and required calibration in `context`. It returns `did_update`, an optional loss, and method-specific diagnostics. The method owns its optimizer and buffers.
7. `state_dict` and `load_state_dict` round-trip all learned and adaptation state. Load rejects incompatible identities. The workflow selects a checkpoint by role before calling these methods. A best checkpoint need not contain an exact-resume snapshot; reject a resume request when continuation state is missing.
8. No method receives test labels, including adaptation diagnostics. A batch passed to `score` or `update` has labels removed; the evaluator holds labels for metrics only. Test values support scoring and causal adaptation after recording the score. Prohibit future observations, complete-test statistics, and test-based tuning.
9. Review method equations and author code before replacement. Preserve core calculations where the confirmed decisions permit; document architecture departures. RedLamp retains reviewed multiclass training but fixes class selection and score scaling from validation and uses causal smoothing. MUSTARD O2, MOMENT readiness, pooling, and uncertainty outputs still need review.

## 9. MUSTARD helper files

### `src/methods/mustard/retrieval.py`

```python
def retrieve(features: Tensor, memory: Mapping[str, Tensor],
             settings: Config) -> Record: ...
```

Return the selected representation and required selection diagnostics. Preserve no, dense, and sparse retrieval semantics. Memory initialization and changes remain owned by `Mustard`; retrieval does not hide lifecycle changes.

### `src/methods/mustard/losses.py`

```python
def compute_loss(outputs: Record, targets: Record,
                 settings: Config) -> dict[str, Tensor]: ...
```

Return the total scalar loss and named terms. Settings select the reviewed stage and ablation terms. Do not silently substitute one loss family for another.

### `src/methods/mustard/state.py`

```python
def pack_state(method: Any) -> Record: ...

def restore_state(method: Any, state: Mapping[str, Any]) -> None: ...
```

Private helpers for the public state methods. They encode and restore method-owned state; they do not own file paths or create another method object.

### `src/methods/mustard/online_update.py`

```python
def update_window(method: Any, batch: Record, context: Record) -> Record: ...
```

Private computation called only through `Mustard.update`. It may change the method's allowed parameters, optimizer, and buffers. It must not start another stream or record a replacement score for the same window.

### `src/methods/augmentation.py`

```python
def augment(x: Tensor, settings: Config, *,
            generator: torch.Generator) -> tuple[Tensor, Record]: ...
```

The generator is a local `torch.Generator`, supplied explicitly. Return new values and supervision/provenance needed by the selected family. Do not alter the source batch or make synthetic labels indistinguishable from source labels.

## 10. `src/encoders.py`

```python
def create_encoder(settings: Config, *, input_features: int,
                   window_size: int) -> nn.Module: ...
```

Return a fresh time-preserving CNN for the four selected methods. It maps `[B, L, C]` to `[B, L, H]`. Reuse the implementation and settings, with separate learned weights. Method-specific pooling and heads remain with each method; CANDI's SANA network is separate. MOMENT keeps its pretrained encoder. Use existing `nn.Module.forward`, parameter, and state behavior; do not add a second encoder lifecycle.

## 11. Execution files

### `src/offline.py`

```python
def run_offline(config: Config, *,
                resume: Record | None = None) -> Record: ...
```

Load prepared data, create the method, fit when supported, restore the selected best checkpoint, calibrate on the approved source, evaluate required splits, and save results. Record initialization and best checkpoints for each required stage. Test labels never choose state or thresholds.

### `src/online.py`

```python
def run_online(config: Config, *,
               resume: Record | None = None) -> Record: ...
```

1. For a new run, select the required offline parent when applicable. For resume, restore the recorded parent and artifact identities; never select a newer parent. Load the appropriate best or resume checkpoint, check prepared-data and calibration compatibility, then iterate causal windows.
2. For each window: validate input, score, apply the documented scoring state, record the pre-update score, decide whether an update is allowed, update if allowed, and validate the update result. Changes affect later scores only.
3. Checkpoint enough state for continuation. Restore cursor, model/optimizer/buffers, random generators, and scoring state together. Return status and artifact references rather than all stream outputs.
4. The caller supplies an allocated `run_id`; resume keeps the old one. Execution functions record failures and release integration resources before propagating errors.

## 12. `src/scoring.py`

```python
def reconstruction_scores(x: Tensor, reconstruction: Tensor,
                          settings: Config) -> Tensor: ...

def calibrate(scores: Record, settings: Config, *, identity: Record) -> Record: ...

def score_batch(raw: Record, calibration: Record | None,
                settings: Config, *, state: Record | None = None
                ) -> tuple[Record, Record]: ...

def aggregate_points(batches: Iterator[Record], *,
                     length: int, settings: Config) -> Record: ...
```

1. `reconstruction_scores` is shared only by methods with the same reviewed loss-space calculation. Method-specific reconstruction layouts remain explicit.
2. `calibrate` uses the source that the evaluation profile approves.
   For the retained normalized-input SMD q99 profile, offline execution generates synthetic windows from saved validation and scores them with the selected checkpoint.
   `calibrate` uses eligible point and window scores with their respective masks.
   Its result records the source identity, generation settings when applicable, score identity, and fitted thresholds or transforms.
   Test data does not set thresholds.
3. `score_batch` applies calibration and smoothing, returning scores plus the next scoring state. It does not mutate a supplied state. Persist that state for exact online continuation.
4. `aggregate_points` reduces aligned window outputs using the specified policy and returns coverage. It does not store every batch. Online aggregation must preserve already recorded pre-update decisions; it cannot revise a past decision using future windows.
5. The resolved evaluation settings select the score space. MUSTARD supplies a raw point score for each window position. For latent MSE it uses its normalized query and continuous prototype reconstruction before fusion. Both score options use the same shared alignment and smoothing interfaces.
6. Calibration records the selected score space, stochastic sample reduction, overlap reduction, EWMA weights when used, and validation source. It produces a separate q95 point threshold for each comparison profile. Triage thresholds remain distinct.
7. The [runtime profile contract](runtime-profile-contract.md#mustard-validation-q95-comparison) specifies the mean of the per-sample latent MSE values. Both phases use the same score definition and point EWMA. Offline fits the detection threshold on validation. Online loads the matching offline threshold and keeps it fixed during adaptation. The existing APIs accept these settings.

## 13. `src/evaluation_metrics.py`

```python
def compute_metrics(scores: Record, labels: Array,
                    settings: Config) -> Record: ...
```

Compute requested metrics and return `metrics`, `eligibility`, `segments`, and `metric_metadata`. Scalar values remain under `metrics`; the run/result `metrics` field receives that mapping. Preserve per-segment bounds, counts, undefined reasons, and coverage. Use distinct `segment_mean_*` names for length-weighted means of defined segment metrics. Canonical range metrics remain unavailable for a discontinuous entity. Unknown labels remain unknown. Store unrounded values. A missing requested dependency is an error; a mathematically undefined value is null with a reason.

## 14. `src/artifacts.py`

```python
def allocate_run(config: Config, *, project_root: Path) -> Config: ...

def find_runs(root: Path, *, filters: Mapping[str, Any]) -> list[Record]: ...

def select_offline_parent(config: Config, *, project_root: Path) -> Record: ...

def result_directory(config: Config, split: str) -> Path: ...

def write_run_status(config: Config, status: str, *,
                     details: Record | None = None) -> Path: ...

def write_result(config: Config, split: str, result: Record) -> Path: ...

def save_checkpoint(path: Path, state: Mapping[str, Any], *,
                    identity: Record, role: str) -> Path: ...

def load_checkpoint(path: Path, *, expected_identity: Record) -> Record: ...

def save_resume(config: Config, state: Record) -> Path: ...

def load_resume(run: Record) -> Record: ...
```

1. `allocate_run` returns a new config with an execution ID, start time, and resolved output information. Generate UUIDv7 IDs for experiment runs; honor a unique standalone user ID. Allocate atomically and reject collisions. Do not derive identity from a folder's position alone.
2. `find_runs` follows the configured result hierarchy and metadata, including omitted single-value levels. Filters use exact recorded dimensions. It is shared by reporting, parent lookup, and resume discovery; each caller adds its own eligibility rule.
3. `select_offline_parent` searches the same experiment for matching successful, compatible runs and orders by start time, then completion time. Standalone selection uses the supplied parent directory. Missing required artifacts after selection cause an error; do not fall back silently.
4. `write_run_status` owns whole-run lifecycle records, including failures or blocked dependencies without split output. It records transitions atomically and makes them discoverable by `find_runs`. `result_directory` implements the agreed hierarchy. `write_result` saves split JSON with ID, status, times, dimensions, prepared identity, profile paths, and metrics. It must not mark a whole run successful before all required work is complete.
5. Checkpoint save/load owns storage, checksums, role, and compatibility checks. Method state contents stay with the method. Save required stage initialization and best states; save resume state separately.
6. `save_resume` and `load_resume` include the complete execution snapshot. A resume config comes from the saved run, not newly resolved defaults. Atomic writes must leave either the old complete snapshot or a new complete one.
7. Exact filenames, whole-run status location, and equal-timestamp final tie order remain open. The confirmed result hierarchy replaces the older depth limit. Do not turn this interface into a claim that recovery or path migration is implemented.

## 15. `src/reporting.py`

```python
def aggregate_table(runs: Sequence[Record], reporting: Config, *,
                    expected_seeds: Sequence[int]) -> Record: ...

def report_experiment(path: Path, *, project_root: Path) -> list[Path]: ...
```

1. `aggregate_table` is a pure calculation. For each dataset, phase, split, method variant, entity, metric, and seed, select the latest successful run with a valid metric by start time. Then calculate mean, sample standard deviation, and valid/expected seed count. One valid seed has unavailable standard deviation.
2. A table covers one dataset and one phase; it may cover several splits. Keep split identity in reporting cells. Do not mix offline and online results or average repeated executions as extra seeds.
3. `report_experiment` loads selected reporting profiles, discovers saved results through `artifacts.find_runs`, and writes detailed and mean-only CSV tables. Shared metric precision applies by default; rare profile overrides affect formatting only. Missing values display as `N/A`.
4. Use complete recorded configurations and identity checks. Profiles are expected to stay fixed within an experiment, but changed or incompatible results must not be combined silently.

## 16. `src/integrations.py`

```python
def log_wandb(event: Record, *, session: Any | None = None) -> Any | None: ...

def backup_kaggle(experiment_root: Path, settings: Config) -> Record: ...
```

1. The W&B session is the provider's existing object, not a new wrapper class. Events declare start, metrics, or finish; `log_wandb` returns the active session or `None`. Disabled logging performs no provider call.
2. Backup receives exactly one experiment root and saves only the approved artifact selection. It must not sweep other experiments or credentials into an upload. Scheduling and failure policy follow the integration specification; they are not invented here.
3. Keep logging and backup independent. Local result validity does not depend on remote availability unless an explicit policy requires it.

## 17. Four protocol files

Each module has separate input and output checks. They raise a contextual `ValueError` and return `None` on success. No automatic repair, hidden conversion, or method execution occurs in a validator.

```python
# src/protocols/run_evaluation.py
def validate_input(config: Config) -> None: ...
def validate_output(result: Record, config: Config) -> None: ...

# src/protocols/data_stream.py
def validate_input(data: Record, config: Config) -> None: ...
def validate_output(batch: Record, config: Config) -> None: ...

# src/protocols/scoring_calibration.py
def validate_input(values: Record, config: Config) -> None: ...
def validate_output(values: Record, config: Config) -> None: ...

# src/protocols/method_updates.py
def validate_input(method: Any, batch: Record, context: Record) -> None: ...
def validate_output(result: Record, context: Record) -> None: ...
```

1. Import each module by its name, such as `data_stream.validate_output`. Do not build a central validator router or import all checks under the same unqualified function name.
2. Records identify their documented kind when one context has several shapes, such as resolved run, prepared entity, calibration, or evaluation. Use a small explicit branch per supported shape. Exact fields must be specified before implementation; do not accept arbitrary dictionaries.
3. Run/evaluation checks identity, profile consistency, output structure, label restrictions, and result alignment. Data/stream checks prepared input identity, shapes, feature order, source mapping, ranges, and causal indices.
4. Scoring/calibration checks supported score identity, finite eligible values, threshold ordering, calibration provenance, and required coverage. It does not require unused online fields for an offline-only method.
5. Method-update checks batch size, allowed trainable parameters, triage inputs, finite results, and actual allowed state changes. The caller provides compact before/after evidence in `context` where needed; a finite loss alone does not prove an allowed update.
6. Reuse the [existing validation inventory](../../../specs/current-protocol-validation-rules.md), but do not copy raw-preprocessing checks into runtime. Newly proposed output checks are not claims about current enforcement.

## 18. Package and test files

Package `thesis_tsad` maps to `src/`; package `thesis_scripts` maps to `scripts/`. Distribution `thesis-tsad` registers `tsad = thesis_tsad.cli:main` in the thesis environment only. Check existing command ownership before installation. Explicit package lists exclude retained legacy packages. The two descriptive module names above avoid collisions with old `src/data/` and `src/metrics/` packages.

1. `__init__.py` files add no workflow or state. Avoid importing every method eagerly.
2. Test files listed in the tree exercise the owning public calls above. They do not require another application interface. Use one parameterized checkpoint round-trip contract across supported method classes.
3. Important checks are read-only planning, prepared-data loading without raw files, no second scaling, batch/index alignment, method state round-trip, exact resume, score-before-update order, parent compatibility, and seed-aware reporting.
4. Test method mathematics separately with small known inputs and reference outputs. Passing structural checks does not establish numerical parity.
5. Keep preprocessing tests temporary under the current delivery decision. A permanent `tests/test_preprocess.py` remains a proposed future test artifact; it is not another delivered preprocessing script.

## 19. Small execution example

This is an interface sketch, not runnable implementation.

```python
plans = plan_experiment(experiment_path, project_root=project_root)
for plan in plans:
    config = allocate_run(plan, project_root=project_root)
    if config["phase"] == "offline":
        result = run_offline(config)
    else:
        result = run_online(config)
```

The production workflow also records blocked dependencies, failed status, and cleanup. It must not allocate a new ID when resuming an interrupted run.

## 20. Changes from earlier signature proposals

| Earlier design | New proposal | Status | Owner |
| --- | --- | --- | --- |
| `src/core/config.py` | `src/config.py` | renamed path; general resolver role unchanged | config |
| `resolve_cell` allocates run identity during resolution | `resolve_run` resolves settings; `allocate_run` creates identity at execution | split; behavior changed | config and artifacts |
| `resolve_experiment` combines parsing, expansion, and IDs | `plan_experiment` plans; `run_experiment` executes | split | experiments |
| Runtime data profile supplies raw split ratio and scaler | `prepared_root` selects processed data | deprecated raw fields; new prepared reference | preprocessing and data |
| Raw loaders, split module, and scaler module in runtime | one source-preparation script; one prepared-data loader | merged source preparation; split from runtime loading | preprocess and data |
| Per-task runner and collector scripts | CLI calls owning workflow functions | merged command surface | cli, experiments, reporting |
| Proposed `src/data.py` and `src/metrics.py` | `src/prepared_data.py` and `src/evaluation_metrics.py` | renamed paths; avoid retained package collisions | prepared data and metrics |
| Method-profile singular `phase` | `phases` list; one run still has one phase | changed profile cardinality | config |
| `fit` without continuation inputs | optional `resume` and `on_checkpoint` | added durable epoch/stage recovery | method and workflow |
| Flat `compute_metrics` result | metric evaluation envelope | expanded output; scalar metrics unchanged | evaluation metrics |
| Prepared/batch scores without segments | explicit segment bounds and IDs | added gap-safe metric provenance | prepared data and scoring |

No domain term is renamed. `Mustard` is a proposed Python class identifier for the method MUSTARD. Required-method checks remain a checklist and simple text search. No fifth protocol or new mandatory type hierarchy is introduced.

## 21. Review limits and references

1. This is a proposed public interface, not permission to remove current code or an exact code-size estimate. Keep required current callers, scientific scope, checkpoints, and tests until replacement checks pass.
2. Phase 1 bindings now specify profile fields and defaults, capabilities, heads, alignment, smoke limits, gaps/tails, metric inputs, artifact status, compatibility, recovery, and provider policy. Review them through the linked contract and decision record. Source pins, licensed implementations, dependency compatibility, scientific tests, and runtime evidence still belong to their later plan stages.

## 22. Recovery and metric bindings

1. Fitted neural methods invoke `on_checkpoint` only at complete epoch or stage boundaries. They supply method/optimizer/scheduler state and their next cursor. The workflow adds identities, RNGs, and writer state before `save_resume`. Native fits commit before and after fitting without invented epoch settings.
2. Online execution saves all method state, pending point EWMA, finalized accumulators, cursor, and RNGs together. Use the two-slot commit-pointer protocol and default eight-window interval. Resume pins the same parent. Never infer exact continuation from a best checkpoint alone.
3. A segment is a half-open range into the unchanged prepared split. Window batches do not cross a known gap. Metric eligibility can divide it further at unknown labels or uncovered scores. Preserve those boundaries without compressing elapsed time.
4. Point score availability and finalization are separate runtime concepts. A point uses its last EWMA estimate and may have a reporting delay. Pending estimates may change; committed window events and finalized points cannot change. Reconstruct positions and availability from shared metadata and the deterministic schedule; do not persist per-point identity or time columns. Natural end and an explicit work limit have distinct run-level end events.
5. Metric metadata records the requested definition, parameters, segment policy, contributing and omitted point counts, and coverage. A segment summary is not renamed as a canonical whole-series VUS or Affiliation F1.
6. Artifact schema version 1, UTC timestamps, global ID reservation, hierarchy omissions, state branches, tie order, provider policy, and package isolation follow the numbered Stage 1.2 bindings. Do not implement another fallback policy from an older open-item sentence.

## 23. Compact persistence and epoch review

1. Point, batch, and score records above describe runtime values. Their index and segment arrays remain temporary. Save scores as one nullable column with shared split/run metadata. Do not serialize the whole runtime record as the score artifact.
2. Recovery stores ordered score/EWMA values and sufficient cursor state. Reconstruct their indices from frozen prepared identity and the window schedule. Reconstruct finalization metadata rather than saving it per point.
3. Unequal training-epoch defaults are withdrawn. The user confirmed 30 total offline epochs: MUSTARD uses 25 + 5; other fitted neural methods use 30. Method profiles own `training.stages[].epochs`; run/evaluation validation checks equal totals. No duplicate experiment field or public function signature change is needed. Zero-shot and native estimators have no epoch budget.
3. The confirmed result hierarchy supersedes older path-depth preferences for the next thesis version. No existing result path is migrated here.
4. The signature proposal adds no dependencies and no source files. Refine a signature when a reviewed method contract demonstrates a missing input or output; do not grow speculative interfaces in advance.
5. Evidence: [definitions](../../../specs/definitions.md), [desired CLI](../../../specs/desired-cli-20261109.md), [configuration plan](../../2026-11-09/plan/plan-config-signatures-and-fields.md), [preprocessing decisions](../plan/decisions-tsad-preprocess.md), and [updated tree](../../2026-11-03/detail/phase2-reexecution/proposed-next-version-tree.md).
