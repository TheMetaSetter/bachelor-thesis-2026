**Summary:**
The simplest reusable framework is built around a **small number of fixed contracts** and **composition instead of deep inheritance**. For your thesis, the most important contract is still: every model must expose a thesis-facing hidden representation

$$
H \in \mathbb{R}^{B \times L \times d_h},
$$

so datasets, models, and the trainer stay decoupled while each model file stays self-contained. The current offline objective should also follow the same philosophy: keep the default objective small, keep optional loss terms modular, and only enable extra regularizers when diagnostics justify them.

## SSOT synchronization note

This design starter is synchronized with `documents/design/` as the single source of truth.

Active window length for current thesis experiments is:

$$
L = 20
$$

For the detailed offline pre-training phase two-view contrastive specification, see:

- `documents/design/offline_pretraining_phase_two_view_contrastive_design.md`

**Reasoning:**
Let us start from the real design question. What usually makes a research codebase impossible to reuse?

It is usually not “lack of abstraction.” It is the wrong abstraction. A codebase becomes fragile when dataset-specific logic leaks into models, model-specific assumptions leak into the trainer, and every experiment invents a new batch format. So the goal is not “many abstractions.” The goal is a **thin waist**: a very small number of interfaces that everything agrees on.

For your case, I would design the framework around four stable runtime layers:

1. **Configuration layer**
2. **Data layer**
3. **Model layer**
4. **Engine layer**

The key idea is this: datasets should only care about producing standardized batches, models should consume those batches and return standardized outputs, and the engine should only decide how to loop, checkpoint, and log. Model-specific losses, scoring, and stage behavior should remain inside the corresponding model file. That is required by `codebase_preferences.md` and it removes the phase-1 to phase-3 split debt identified in the research log.

So the framework’s central question becomes:

> “What are the minimum input and output objects that every dataset and every model must agree on?”

For your thesis, I recommend these two contracts.

A batch should always look like this conceptually:

```python
batch = {
    "x": Tensor[B, L, D],                  # input window
    "point_labels": Optional[Tensor[B, L]],
    "mask": Optional[Tensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],                    # dataset-specific metadata
}
```

A model output should always look like this conceptually:

```python
outputs = {
    "hidden": Tensor[B, L, H],     # thesis-facing representation
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor],
    "point_scores": Optional[Tensor[B, L]],
    "window_scores": Optional[Tensor[B]],
    "aux": dict,                   # prototype assignments, uncertainty, etc.
}
```

Contract note for the current codebase:

- Use `hidden` as the single thesis-facing learned representation.
- Do not expose both `hidden` and a second semantically equivalent `latent` field in the same model output.
- If a model has an internal bottleneck or intermediate width, that is an implementation detail. The external contract should still expose only `hidden: Tensor[B, L, H]`.
- Naming rule: if an autoencoder or variational autoencoder intentionally uses a bottleneck as its primary public representation, that model may call the representation `latent` instead of `hidden`. But one model should use one public name only unless `hidden` and `latent` are truly different tensors with different semantic roles.

That is the core of reusability. Once these two are stable, you can swap datasets and models much more easily.

Now let us translate that into a concrete folder design.

```text
bachelor-thesis-2026/
├── configs/
│   ├── data/
│   │   ├── smd.yaml
│   │   ├── msl.yaml
│   │   └── custom.yaml
│   ├── model/
│   │   ├── reconstruction_mlp_ae.yaml
│   │   ├── thesis_multitask.yaml
│   │   └── online_adaptation.yaml
│   └── experiment/
│       ├── smd_reconstruction.yaml
│       ├── smd_multitask.yaml
│       └── smd_online_adaptation.yaml
│
├── src/
│   ├── core/
│   │   ├── registry.py
│   │   ├── config.py
│   │   ├── contracts.py
│   │   ├── seed.py
│   │   └── paths.py
│   │
│   ├── data/
│   │   ├── base.py
│   │   ├── api.py
│   │   ├── cleaning.py
│   │   ├── download.py
│   │   ├── public_types.py
│   │   ├── window.py
│   │   ├── scalers.py
│   │   ├── collate.py
│   │   ├── loaders.py
│   │   └── datasets/
│   │       ├── smd.py
│   │       ├── msl.py
│   │       └── custom_csv.py
│   │
│   ├── models/
│   │   ├── base_model.py
│   │   ├── reconstruction_mlp_ae.py
│   │   ├── thesis_multitask.py
│   │   └── online_adaptation.py
│   │
│   ├── adapters/
│   │   ├── base.py
│   │   └── moment.py
│   │
│   ├── metrics/
│   │   ├── pointwise.py
│   │   ├── eventwise.py
│   │   └── uncertainty.py
│   │
│   ├── engine/
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── checkpoint.py
│   │   ├── logger.py
│   │   └── artifact_sinks.py
│   │
│   └── utils/
│       ├── io.py
│       ├── device.py
│       └── debug.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize_synthetic_anomalies.py
│   ├── run_ablation.py
│   └── export_results.py
│
└── tests/
    ├── test_smd_dataset_shapes.py
    ├── test_windowizer.py
    ├── test_model_shapes.py
    ├── test_one_train_step.py
    ├── test_checkpoint_roundtrip.py
    ├── test_synthetic_anomaly_injection.py
    ├── test_multitask_shapes.py
    ├── test_one_multitask_train_step.py
    ├── test_synthetic_anomaly_visualization.py
    ├── test_online_adaptation_step.py
    ├── test_online_state_roundtrip.py
    └── test_registry.py
```

The notebook-facing public path should stay thin and additive:

- `from src.data import load_smd_data`
- `from src.data import flatten_windows_for_baseline, point_labels_to_window_labels`
- `from src.adapters import MomentWindowAdapter`

Those imports should wrap the existing config-driven internals rather than replace them. The active script-facing path remains the registry plus YAML configuration.

This structure is simple because each folder has one job. It is reusable because a new dataset still means one parser file and one config, while a new model normally means one new file under `src/models/` and one config, without changing the trainer.

Now let us make the contracts more precise.

Your **data layer** should be split into two different concerns:

* **dataset parsing**
* **window construction**

This separation matters a lot. SMD, MSL, SWaT, and other datasets differ in raw storage format, split conventions, and label formats. But the operation “turn a long multivariate series into windows of length $20$” is common. So do not embed windowing inside each dataset parser.

A good design is:

* `datasets/smd.py` reads raw SMD files and returns full sequences
* `window.py` converts full sequences into windows with size $L=20$, stride $s$
* `scalers.py` normalizes data
* `loaders.py` builds PyTorch `Dataset` and `DataLoader`

That way, when you add a new dataset, you only rewrite the raw parser, not the whole pipeline.

Your **model layer** should stay self-contained at the file level.

Inside one model file, it is still fine to organize the code into three conceptual sections:

* **encoder block**
* **prototype and fusion block**
* **output heads and stage methods**

This is exactly what your thesis needs. For example:

$$
X \xrightarrow{\text{encoder}} H
$$

then

$$
H \xrightarrow{\text{prototype modules}} H^{(c)}, H^{(d)}
$$

then

$$
(H^{(c)}, H^{(d)}) \xrightarrow{\text{fusion}} H_{\text{rec}}, H_{\text{cls}}
$$

then

$$
H_{\text{rec}} \xrightarrow{\text{reconstruction head}} \hat X,
\qquad
H_{\text{cls}} \xrightarrow{\text{classification head}} \hat y.
$$

This decomposition is important because it lets you replace the encoder logic without rewriting the trainer, but it does not force you to fragment one thesis model across `modules/`, `heads/`, `losses/`, and `tasks/` files. In this repository, readability comes first, so the continuous prototypes, discrete prototypes, fusion, scoring, and loss computation that belong to one model should be read top-to-bottom in that same model file.

Now let us write the minimal base classes. Keep them small.

```python
# src/models/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: standardized batch dict
        Returns:
            standardized model output dict
        """
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def test_step(self, batch: dict) -> dict:
        raise NotImplementedError
```

This may look almost too small, but that is good. A reusable framework should not force every model into a giant class hierarchy. The base classes should only enforce the common contract.

Now the **registry**. Since you want many datasets and many models, you do need a registry, but keep it tiny.

```python
# src/core/registry.py
DATASETS = {}
MODELS = {}

def register_dataset(name):
    def wrapper(cls):
        DATASETS[name] = cls
        return cls
    return wrapper

def register_model(name):
    def wrapper(cls):
        MODELS[name] = cls
        return cls
    return wrapper
```

Then each dataset or model can register itself. This gives you flexibility without adding a heavy plugin system. It also keeps `scripts/train.py` and `scripts/evaluate.py` on one registry-based construction path instead of mixing registered and direct builder calls.

For **configuration**, I would not start with Hydra unless you already know you need it. It is powerful, but it can also add conceptual overhead. For a bachelor thesis framework, I would start with plain YAML + a small config loader.

A single experiment config can look like this:

```yaml
seed: 42

data:
  name: smd
  root: ./data/SMD
  window_size: 100
  stride: 10
  batch_size: 64
  num_workers: 4
  scaler: standard

model:
  name: thesis_multitask
  hidden_dim: 256
  continuous_prototypes:
    num_prototypes: 32
  discrete_prototypes:
    codebook_size: 64
    init_temperature: 2.0
    final_temperature: 0.5
    anneal_fraction: 0.4
  fusion:
    alpha_init: 0.5
    beta_init: 0.5
    learnable: true
  loss_weights:
    recon: 1.0
    cls: 1.0
    div: 0.05
    var: 1.0
    cov: 0.1
    use: 0.01
    gate: 0.001
  synthetic_anomaly:
    enabled: true
    family: carla_subsequence

train:
  epochs: 50
  warmup_epochs: 5
  optimizer: adamw
  lr_encoder: 0.0001
  lr_new_modules: 0.001
  weight_decay: 0.0001
  grad_clip_norm: 1.0

eval:
  metrics: [f1, auc, pr_auc]
```

For the current offline thesis model, the default prediction rule should also be fixed in the design docs. The real task heads stay on the two fused representations only:

$$
H_{\text{rec}} = \beta \hat H^{(d)} + (1-\beta)\hat H^{(c)},
\qquad
H_{\text{cls}} = \alpha \hat H^{(d)} + (1-\alpha)\hat H^{(c)}.
$$

The reconstruction head consumes only $H_{\text{rec}}$, and the anomaly-type classification head consumes only $H_{\text{cls}}$. By default, the architecture does not add a branch-local decoder on $\hat H^{(c)}$ or a branch-local classifier on $\hat H^{(d)}$. Pre-fusion regularizers still belong to the same model file, but they should operate on $\hat H^{(c)}$, $\hat H^{(d)}$, and the discrete assignments before fusion rather than creating separate prediction paths.

The loss design should follow **objective modularity** or, equivalently, an **ablation-friendly objective surface**. That means:

* the default starting objective is only reconstruction plus classification
* this simple objective remains the beginning-of-training default until
  concrete observed failure modes justify extra regularizers
* variance and covariance regularization are the first anti-collapse additions if collapse appears
* cross-branch decorrelation, code-usage balancing, and gate entropy regularization are activated only for observed failure modes
* every loss term remains in the same model file as the model that owns it
* every optional term is controlled explicitly through YAML rather than through ad hoc code edits

So the current design intent is not “turn on every regularizer by default.” The design intent is “start small, observe failure modes, then add the smallest justified regularizer.”

In this design surface, the `gate` weight denotes gate entropy regularization. Current design target: gate entropy regularization. Current implementation status: the code still uses a barrier-style gate term and should be updated separately.

That is enough for reuse. A new dataset should mean a new `data` config. A new model or training stage should mean a new `model` config or `experiment` config, not a second file that steals its loss logic away from the model.

Now, because your thesis has both **offline training** and **online adaptation**, I strongly recommend that you represent these as different **self-contained model files**, not as one fragmented model plus several task files. That is cleaner for this repository.

So for example:

* `reconstruction_mlp_ae.py`
  reconstruction architecture, scoring, and reconstruction-stage step methods

* `thesis_multitask.py`
  encoder, continuous and discrete prototype logic, fused reconstruction and classification heads, CARLA-aligned synthetic anomaly training path, and the full offline multitask objective

* `online_adaptation.py`
  frozen reference encoder, online encoder, near-identity residual projector, optional Fisher- or NGD-style preconditioning for the small adapted subset, and online alignment losses

This is a very important design choice. Why? Because `codebase_preferences.md` is explicit: all logic related to one model, including inference and training logic, should live in one readable file. In this repository, pushing losses and stage behavior into separate `tasks/` or `losses/` files recreates the exact debt called out in problem 1.

Now the **engine**. Keep the trainer extremely plain. Do not hide too much magic.

Conceptually:

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        step_out = model.training_step(batch)
        loss = step_out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    for batch in val_loader:
        model.validation_step(batch)
```

That is enough. You do not need a deep framework inside your framework.

For your thesis specifically, I would define the trainer to know nothing about SMD, nothing about prototypes, and nothing about anomaly detection. It should only know:

* how to loop
* how to backpropagate
* how to checkpoint
* how to log

Everything else belongs elsewhere.

Now let us discuss the most important architectural principle for your thesis model.

The right decomposition is not:

> one giant `ThesisModel` that knows everything

The better decomposition is:

```python
class ThesisMultiTaskModel(BaseModel):
    def __init__(self, encoder, cont_proto, disc_proto, fusion, recon_head, cls_head):
        ...
```

That is, the architecture should be **assembled from modules**. Then you can instantiate:

* `LSTMEncoder + simple head`
* `MOMENTAdapter + simple head`
* `MOMENTAdapter + thesis prototype modules`
* `TimesNetAdapter + thesis prototype modules`

without rewriting the rest of the framework.

That is exactly how you make a codebase reusable across many models.

Here is a simple conceptual sketch:

```python
class ThesisMultiTaskModel(BaseModel):
    def __init__(
        self,
        encoder,
        continuous_proto,
        discrete_proto,
        fusion,
        recon_head,
        cls_head,
    ):
        super().__init__()
        self.encoder = encoder
        self.continuous_proto = continuous_proto
        self.discrete_proto = discrete_proto
        self.fusion = fusion
        self.recon_head = recon_head
        self.cls_head = cls_head

    def forward(self, batch: dict) -> dict:
        x = batch["x"]                       # [B, L, D]
        enc_out = self.encoder(x)
        h = enc_out["hidden"]               # [B, L, H]

        h_cont = self.continuous_proto(h)
        h_disc = self.discrete_proto(h)

        fused = self.fusion(h_cont, h_disc)
        h_rec = fused["h_rec"]
        h_cls = fused["h_cls"]
        pooled_cls = h_cls.mean(dim=1)

        recon = self.recon_head(h_rec)
        logits = self.cls_head(pooled_cls)

        return {
            "hidden": h,
            "pooled": pooled_cls,
            "recon": recon,
            "logits": logits,
            "aux": {
                "h_cont": h_cont,
                "h_disc": h_disc,
                **fused.get("aux", {}),
            },
        }
```

This design is simple, but it already supports your whole thesis.

The important rule in that sketch is where supervision lives. The default thesis model should predict only from the fused task-specialized states `h_rec` and `h_cls`. The branch outputs `h_cont` and `h_disc` remain observable in `aux` for regularization, monitoring, and ablations, but they are not separate default prediction pathways.

Now, since your professor specifically wants reusability across datasets, you should also standardize **dataset metadata**. Not every dataset has the same properties. Some are fully labeled, some are weakly labeled, some are univariate, some are multivariate, some have train/test-only splits, some have validation splits.

So each dataset builder should expose metadata like:

```python
dataset_info = {
    "num_channels": D,
    "window_size": L,
    "has_point_labels": True,
    "has_event_labels": False,
    "task_type": "tsad",
}
```

This helps the model and evaluator decide what is legal.

Now the practical question: what should you code first?

I would do it in this exact order.

First, implement the framework skeleton:

* `registry.py`
* `config.py`
* `contracts.py`
* `base_model.py`
* `trainer.py`

Second, implement one dataset only:

* `datasets/smd.py`
* `window.py`
* `loaders.py`

Third, implement one trivial baseline:

* `reconstruction_mlp_ae.py`

Fourth, make sure one experiment runs end-to-end.

Fifth, add your thesis model gradually:

* `thesis_multitask.py` with internal continuous prototypes, discrete prototypes, fusion, and classification
* CARLA-aligned synthetic anomaly injection
* anomaly-visualization export
* `online_adaptation.py` only after the earlier debt is closed

That order is important. It is the same “minimal vertical slice first” principle from the earlier conversation, and it is exactly the right way to avoid chaos 

There are also a few things I would explicitly avoid.

Do not create abstract base classes for everything.
Do not let datasets return different batch formats.
Do not split one model's losses across separate `tasks/`, `losses/`, and helper files.
Do not make the trainer aware of anomaly-specific logic.
Do not mix raw-data parsing with window generation.
Do not make your framework dependent on one encoder family.

A good framework is not the one with the most folders. It is the one where adding a new dataset means adding one dataset file and one config, and adding a new model means adding one model file and one config.

Finally, for your thesis, I would define three non-negotiable contracts:

First, the **batch contract**:

$$
x \in \mathbb{R}^{B \times L \times D}
$$

Second, the **encoder contract**:

$$
H \in \mathbb{R}^{B \times L \times d_h}
$$

Third, the **step output contract**:

every training step returns at least a total loss and a log dictionary.

If these three remain stable, your framework will stay clean even as the thesis model grows.

**Check:**
A good consistency test is this:

* Can I add a new dataset without touching the trainer?
* Can I add a new encoder without touching the dataset code?
* Can I switch from `reconstruction_mlp_ae.py` to `thesis_multitask.py` without rewriting the trainer?
* Can I keep the same downstream prototype modules even if I swap MOMENT for another encoder?

If the answer is yes, then the framework is genuinely reusable. If the answer is no, then some boundary is in the wrong place.

## Implementation sequencing in the current repository

The earlier generic `Phase 0-10` roadmap should now be translated into the
repository's present phase language rather than treated as a literal active
sequence.

- The older contract-freezing and minimum-vertical-slice phases are already
  closed through the current SMD-first offline path, the stable batch contract,
  and the stable model-output contract.
- The earlier modularization phases are partly closed through the current
  `configuration -> data -> model -> engine` layering, the registry-driven
  scripts, and the one-model-one-file rule from `codebase_preferences.md`.
  Broad multi-dataset generalization is still deferred.
- The earlier experiment-logging phase is partly closed through YAML-controlled
  experiments, resolved-config persistence, JSONL metrics, and optional Weights
  & Biases logging. The remaining reproducibility debt is DVC-backed synthetic
  or derived-data versioning when those artifacts are materialized.
- The earlier thesis-model phase is already realized in the current offline
  multitask implementation. The repository now exposes the continuous branch,
  discrete branch, task-specific fusion, RedLamp-default synthetic anomaly
  injection with CARLA as a mechanism reference, and the ablation-ready
  objective surface.
- The earlier online-adaptation phase is also partly realized: the accepted
  first online slice is projector-first, clean-stream-only, and deliberately
  narrow. Drift injection, non-adaptive online baselines under drift, broader
  adaptation policies, and NGD-style optimization remain later-slice scope.
- The older final “generalize only after one successful result” phase remains an
  active policy for this repository. The codebase should expand to broader
  datasets, drift families, and adaptation strategies only after the current
  accepted offline and conservative online paths are stable.

This repository should therefore be read as a translated realization of the old
roadmap, not as a codebase still waiting to begin that roadmap.

## Frozen interface contracts

Before implementing many modules, freeze the native interfaces that every dataset, stream wrapper, windowizer, model, and evaluator must obey.

The purpose of this section is to remove ambiguity early. The question is not "what can this one module accept?" The question is "what must remain unchanged if I swap the dataset or the model?"

### Global invariants

- All native thesis tensors are **time-major**. A full sequence is $[T, D]$, a window is $[L, D]$, and a batch of windows is $[B, L, D]$.
- No native module in `src/` should use $[D, T]$ as its public format. If a reference codebase uses $[D, T]$, the conversion must happen inside an adapter at the boundary.
- Machine or entity boundaries must always be preserved in metadata, even if some storage format is concatenated on disk.
- Point labels must always align one-to-one with timesteps. If labels are missing for a split, the label field is `None`, not a shape-changing substitute.
- Optional fields may be `None`, but their key names must not change across datasets.

### 1. Raw sequence format

The raw parser for one entity must return a dictionary with this shape contract:

```python
raw_sequence = {
    "x": FloatTensor[T, D],
    "point_labels": Optional[IntTensor[T]],
    "mask": Optional[BoolTensor[T, D]],
    "timestamps": Optional[Tensor[T]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "num_channels": int,
        "sequence_length": int,
    },
}
```

Rules:

- `x` is the canonical raw multivariate sequence and is always `$[time, channel]$`.
- `point_labels` stores timestep-level ground truth when available.
- `mask` indicates missing or invalid observations at the same resolution as `x`.
- `timestamps` is optional because some benchmark datasets are indexed by order only.
- `meta` must keep the dataset name, entity name, and split visible so later stages never infer them from filenames.

### 2. Stream output format

The streaming wrapper must expose `next_point()` and `next_window()` without inventing a new schema.

`next_point()` must return:

```python
stream_point = {
    "x": FloatTensor[D],
    "point_label": Optional[IntTensor[()]],
    "mask": Optional[BoolTensor[D]],
    "timestamp": Optional[Tensor[()]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "time_index": int,
        "is_start_of_sequence": bool,
        "is_end_of_sequence": bool,
    },
}
```

Rules:

- `next_point()` is the pointwise streaming contract.
- `next_window()` must return the exact window contract defined below.
- Drift injectors may modify values, but they must preserve the same keys and axis order.

### 3. Window format

The native window object must be:

```python
window = {
    "x": FloatTensor[L, D],
    "point_labels": Optional[IntTensor[L]],
    "mask": Optional[BoolTensor[L, D]],
    "timestamps": Optional[Tensor[L]],
    "meta": {
        "dataset_name": str,
        "entity_id": str,
        "split": str,
        "start_index": int,
        "end_index": int,
        "window_size": int,
        "stride": int,
    },
}
```

Rules:

- One native window is always `X in R^{L x D}`.
- Windowing is responsible only for slicing the sequence and copying aligned labels, masks, timestamps, and metadata.
- Stage-specific targets such as anomaly-type classes may be added later by model preparation code, but the base window contract does not change.

### 4. Model input and output format

The batched input to every model must be:

```python
batch = {
    "x": FloatTensor[B, L, D],
    "point_labels": Optional[IntTensor[B, L]],
    "mask": Optional[BoolTensor[B, L, D]],
    "timestamps": Optional[Tensor[B, L]],
    "meta": list[dict],
}
```

Every model must return:

```python
outputs = {
    "hidden": FloatTensor[B, L, H],
    "pooled": Optional[FloatTensor[B, H]],
    "recon": Optional[FloatTensor[B, L, D]],
    "logits": Optional[FloatTensor],
    "window_scores": Optional[FloatTensor[B]],
    "point_scores": Optional[FloatTensor[B, L]],
    "aux": dict,
}
```

Rules:

- `hidden` is non-negotiable. Every encoder or model adapter must expose a thesis-facing representation of shape $[B, L, H]$.
- `pooled` is optional because some backbones naturally expose a sequence summary and some do not.
- `recon` is optional because not every model is reconstruction-based.
- `logits` is optional because some models are classification-style and some are score-only.
- `window_scores` and `point_scores` are separated explicitly so the codebase never relies on an ambiguous generic `scores` tensor.
- `aux` is where prototype assignments, uncertainty statistics, attention maps, and other model-specific artifacts belong.

### 5. Evaluation record format

The evaluator must convert predictions into a serialization-friendly record format. One evaluated window must become:

```python
evaluation_record = {
    "dataset_name": str,
    "entity_id": str,
    "split": str,
    "start_index": int,
    "end_index": int,
    "window_size": int,
    "window_label": Optional[int],
    "window_score": Optional[float],
    "point_labels": Optional[list[int]],
    "point_scores": Optional[list[float]],
    "threshold": Optional[float],
    "prediction": Optional[int],
    "meta": dict,
}
```

Rules:

- Evaluation records should be JSON-friendly so they can be written to disk, converted to a dataframe, or logged to Weights and Biases without extra schema work.
- `window_label` is a derived evaluation field, not a mandatory dataset-storage field.
- `point_labels` and `point_scores` preserve timestep-level evidence for eventwise or pointwise metrics.
- Aggregated metrics are computed from collections of `evaluation_record`, not from ad hoc tensors with dataset-specific shapes.

### 6. Threshold calibration and leakage rule

The evaluator must keep anomaly-score computation separate from threshold calibration.

For the default static-threshold protocol:

- reconstruction-based models may compute point scores as reconstruction MSE and may derive window scores by averaging point scores inside a window;
- overlapping window point scores may be averaged back onto the original entity timeline before metrics are computed;
- the scalar threshold used for validation, test, or streaming inference must be calibrated on a known calibration split such as train or validation, not on the same test sequence being evaluated;
- a test evaluator must receive an already-calibrated threshold or a serialized threshold-calibration artifact;
- computing a 95th quantile or any high quantile from the full test score vector is allowed only as an explicitly named oracle/offline diagnostic, never as the default reported protocol.

This rule matters because full-test quantile thresholding assumes that all future test windows are already available. That assumption is incompatible with streaming evaluation and can also leak the anomaly-score distribution of the evaluation set into the decision rule.

For stream testing, the stricter rule is:

- the threshold at time `t` may depend on train/validation calibration and on stream observations available before or at `t`;
- the threshold at time `t` must not depend on windows or scores from future stream positions;
- online adaptive thresholding may use rolling, learned, or probabilistic updates, but the state update must be causal and checkpointable.

Current unresolved implementation problem: `src/engine/evaluator.py` currently computes `quantile=0.95` from the concatenated scores of whichever loader is passed to `evaluate()`. Since `scripts/evaluate.py` passes the test loader, the active CLI path calibrates the threshold on the full test timeline. This must be refactored into an explicit calibration step before the default evaluation results are treated as thesis-valid.

### What must remain unchanged when swapping datasets or models

- Dataset modules may change how raw files are read, but they must still emit `raw_sequence` with the same keys and tensor orientation.
- Stream wrappers may change how they iterate, but they must still emit `stream_point` and `window`.
- Windowizers may change stride, padding policy, or overlap, but they must still emit `window` with `$[L, D]$`.
- Encoders may change internally, but they must still emit `hidden` with `$[B, L, H]$`.
- Models may add auxiliary outputs, but they must not remove or rename the fixed keys in `outputs`.
- Evaluators may compute different metrics, but they must still reduce predictions into the same `evaluation_record` schema.

If these interfaces remain fixed, then the codebase can swap SMD for another dataset, or MOMENT for another encoder, without forcing broad rewrites across the framework.
