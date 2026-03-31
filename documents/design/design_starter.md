**Summary:**
The simplest reusable framework is built around a **small number of fixed contracts** and **composition instead of deep inheritance**. For your thesis, the most important contract is still: every encoder must expose a thesis-facing hidden representation

[
H \in \mathbb{R}^{B \times L \times d_h},
]

so datasets, models, heads, and training logic stay decoupled 

**Reasoning:**
Let us start from the real design question. What usually makes a research codebase impossible to reuse?

It is usually not “lack of abstraction.” It is the wrong abstraction. A codebase becomes fragile when dataset-specific logic leaks into models, model-specific assumptions leak into the trainer, and every experiment invents a new batch format. So the goal is not “many abstractions.” The goal is a **thin waist**: a very small number of interfaces that everything agrees on.

For your case, I would design the framework around five stable layers:

1. **Configuration layer**
2. **Data layer**
3. **Model layer**
4. **Task layer**
5. **Engine layer**

The key idea is this: datasets should only care about producing standardized batches, models should only care about consuming those batches and producing standardized outputs, and tasks should decide how losses and metrics are computed.

So the framework’s central question becomes:

> “What are the minimum input and output objects that every dataset and every model must agree on?”

For your thesis, I recommend these two contracts.

A batch should always look like this conceptually:

```python
batch = {
    "x": Tensor[B, L, D],          # input window
    "y": Optional[Tensor],         # labels if available
    "mask": Optional[Tensor],      # missing-value mask if needed
    "timestamps": Optional[Tensor],
    "meta": dict,                  # dataset-specific metadata
}
```

A model output should always look like this conceptually:

```python
outputs = {
    "hidden": Tensor[B, L, H],     # thesis-facing representation
    "pooled": Optional[Tensor[B, H]],
    "recon": Optional[Tensor[B, L, D]],
    "logits": Optional[Tensor],
    "scores": Optional[Tensor],
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
│   │   ├── lstm_ae.yaml
│   │   ├── moment_adapter.yaml
│   │   └── thesis_model.yaml
│   ├── task/
│   │   ├── reconstruction.yaml
│   │   ├── multitask_tsad.yaml
│   │   └── online_adaptation.yaml
│   └── experiment/
│       ├── smd_baseline.yaml
│       └── smd_thesis_v1.yaml
│
├── src/
│   ├── core/
│   │   ├── registry.py
│   │   ├── config.py
│   │   ├── typing.py
│   │   ├── seed.py
│   │   └── paths.py
│   │
│   ├── data/
│   │   ├── base.py
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
│   │   ├── base_encoder.py
│   │   ├── base_model.py
│   │   ├── encoders/
│   │   │   ├── lstm.py
│   │   │   ├── moment_adapter.py
│   │   │   └── timesnet_adapter.py
│   │   ├── modules/
│   │   │   ├── continuous_prototypes.py
│   │   │   ├── discrete_prototypes.py
│   │   │   ├── fusion.py
│   │   │   └── projector.py
│   │   ├── heads/
│   │   │   ├── reconstruction_head.py
│   │   │   ├── classification_head.py
│   │   │   └── score_head.py
│   │   └── architectures/
│   │       ├── recon_baseline.py
│   │       └── thesis_multitask.py
│   │
│   ├── tasks/
│   │   ├── base_task.py
│   │   ├── reconstruction_task.py
│   │   ├── multitask_tsad_task.py
│   │   └── online_adaptation_task.py
│   │
│   ├── losses/
│   │   ├── reconstruction.py
│   │   ├── contrastive.py
│   │   ├── classification.py
│   │   └── prototype.py
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
│   │   └── logger.py
│   │
│   └── utils/
│       ├── io.py
│       ├── device.py
│       └── debug.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
└── tests/
    ├── test_dataset_shapes.py
    ├── test_model_shapes.py
    ├── test_one_train_step.py
    └── test_registry.py
```

This structure is simple because each folder has one job. It is reusable because new datasets and new models only require adding files inside `datasets/` or `encoders/` or `architectures/`, without changing the trainer.

Now let us make the contracts more precise.

Your **data layer** should be split into two different concerns:

* **dataset parsing**
* **window construction**

This separation matters a lot. SMD, MSL, SWaT, and other datasets differ in raw storage format, split conventions, and label formats. But the operation “turn a long multivariate series into windows of length 100” is common. So do not embed windowing inside each dataset parser.

A good design is:

* `datasets/smd.py` reads raw SMD files and returns full sequences
* `window.py` converts full sequences into windows with size `L=100`, stride `s`
* `scalers.py` normalizes data
* `loaders.py` builds PyTorch `Dataset` and `DataLoader`

That way, when you add a new dataset, you only rewrite the raw parser, not the whole pipeline.

Your **model layer** should also be split into three concerns:

* **encoder**
* **intermediate modules**
* **task heads**

This is exactly what your thesis needs. For example:

[
X \xrightarrow{\text{encoder}} H
]

then

[
H \xrightarrow{\text{prototype modules}} H^{(c)}, H^{(d)}
]

then

[
(H^{(c)}, H^{(d)}) \xrightarrow{\text{fusion}} H_{\text{rec}}, H_{\text{cls}}
]

then

[
H_{\text{rec}} \xrightarrow{\text{reconstruction head}} \hat X,
\qquad
H_{\text{cls}} \xrightarrow{\text{classification head}} \hat y.
]

This decomposition is important because it lets you replace the encoder without rewriting the prototype logic, and lets you replace the prototype logic without rewriting the trainer. It is also consistent with the earlier decision that the encoder output interface must stay fixed across backbones 

Now let us write the minimal base classes. Keep them small.

```python
# src/models/base_encoder.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Tensor[B, L, D]
        Returns:
            {
                "hidden": Tensor[B, L, H],
                "pooled": Optional[Tensor[B, H]],
                "aux": dict,
            }
        """
        raise NotImplementedError
```

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
```

```python
# src/tasks/base_task.py
from abc import ABC, abstractmethod

class BaseTask(ABC):
    @abstractmethod
    def training_step(self, model, batch: dict) -> dict:
        """
        Returns:
            {
                "loss": tensor,
                "log": dict,
                "outputs": dict,
            }
        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, model, batch: dict) -> dict:
        raise NotImplementedError
```

This may look almost too small, but that is good. A reusable framework should not force every model into a giant class hierarchy. The base classes should only enforce the common contract.

Now the **registry**. Since you want many datasets and many models, you do need a registry, but keep it tiny.

```python
# src/core/registry.py
DATASETS = {}
MODELS = {}
TASKS = {}

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

def register_task(name):
    def wrapper(cls):
        TASKS[name] = cls
        return cls
    return wrapper
```

Then each dataset or model can register itself. This gives you flexibility without adding a heavy plugin system.

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
  encoder:
    name: moment_adapter
    hidden_dim: 256
  continuous_prototypes:
    num_prototypes: 32
  discrete_prototypes:
    codebook_size: 64
    temperature: 0.5

task:
  name: multitask_tsad
  loss_weights:
    recon: 1.0
    cls: 1.0
    proto: 0.1

train:
  epochs: 50
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001

eval:
  metrics: [f1, auc, pr_auc]
```

That is enough for reuse. A new dataset should mean a new `data` config. A new model should mean a new `model` config. A new objective should mean a new `task` config.

Now, because your thesis has both **offline training** and **online adaptation**, I strongly recommend that you represent these as different **tasks**, not as one huge model. That is cleaner.

So for example:

* `reconstruction_task.py`
  only reconstruction loss

* `multitask_tsad_task.py`
  reconstruction + anomaly-type classification + prototype regularization

* `online_adaptation_task.py`
  frozen reference encoder + online encoder + projector + contrastive alignment

This is a very important design choice. Why? Because a task defines how the same model is trained, not what the model is. If you encode all training logic inside the model class, reuse becomes difficult. The same thesis architecture might be trained in different stages with different losses. That belongs in `tasks/`, not in the architecture file.

Now the **engine**. Keep the trainer extremely plain. Do not hide too much magic.

Conceptually:

```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        step_out = task.training_step(model, batch)
        loss = step_out["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    for batch in val_loader:
        task.validation_step(model, batch)
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

        recon = self.recon_head(h_rec)
        logits = self.cls_head(h_cls)

        return {
            "hidden": h,
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

This helps the task and evaluator decide what is legal.

Now the practical question: what should you code first?

I would do it in this exact order.

First, implement the framework skeleton:

* `registry.py`
* `config.py`
* `base_encoder.py`
* `base_model.py`
* `base_task.py`
* `trainer.py`

Second, implement one dataset only:

* `datasets/smd.py`
* `window.py`
* `loaders.py`

Third, implement one trivial baseline:

* `LSTMEncoder`
* `ReconBaselineModel`
* `ReconstructionTask`

Fourth, make sure one experiment runs end-to-end.

Fifth, add your thesis model gradually:

* continuous prototypes
* discrete prototypes
* fusion
* classification head
* multitask task
* online adaptation task

That order is important. It is the same “minimal vertical slice first” principle from the earlier conversation, and it is exactly the right way to avoid chaos 

There are also a few things I would explicitly avoid.

Do not create abstract base classes for everything.
Do not let datasets return different batch formats.
Do not put loss computation inside every model class.
Do not make the trainer aware of anomaly-specific logic.
Do not mix raw-data parsing with window generation.
Do not make your framework dependent on one encoder family.

A good framework is not the one with the most folders. It is the one where adding a new dataset means adding one dataset file and one config, and adding a new model means adding one model file and one config.

Finally, for your thesis, I would define three non-negotiable contracts:

First, the **batch contract**:

[
x \in \mathbb{R}^{B \times L \times D}
]

Second, the **encoder contract**:

[
H \in \mathbb{R}^{B \times L \times d_h}
]

Third, the **task output contract**:

every training step returns at least a total loss and a log dictionary.

If these three remain stable, your framework will stay clean even as the thesis model grows.

**Check:**
A good consistency test is this:

* Can I add a new dataset without touching the trainer?
* Can I add a new encoder without touching the dataset code?
* Can I change from reconstruction-only training to multitask training without rewriting the model?
* Can I keep the same downstream prototype modules even if I swap MOMENT for another encoder?

If the answer is yes, then the framework is genuinely reusable. If the answer is no, then some boundary is in the wrong place.

## Frozen interface contracts

Before implementing many modules, freeze the native interfaces that every dataset, stream wrapper, windowizer, model, task, and evaluator must obey.

The purpose of this section is to remove ambiguity early. The question is not "what can this one module accept?" The question is "what must remain unchanged if I swap the dataset or the model?"

### Global invariants

- All native thesis tensors are **time-major**. A full sequence is `[T, D]`, a window is `[L, D]`, and a batch of windows is `[B, L, D]`.
- No native module in `src/` should use `[D, T]` as its public format. If a reference codebase uses `[D, T]`, the conversion must happen inside an adapter at the boundary.
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

- `x` is the canonical raw multivariate sequence and is always `[time, channel]`.
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
- Task-specific targets such as anomaly-type classes may be added later by task builders, but the base window contract does not change.

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

- `hidden` is non-negotiable. Every encoder or model adapter must expose a thesis-facing representation of shape `[B, L, H]`.
- `pooled` is optional because some backbones naturally expose a sequence summary and some do not.
- `recon` is optional because not every model is reconstruction-based.
- `logits` is optional because some tasks are classification-style and some are score-only.
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

### What must remain unchanged when swapping datasets or models

- Dataset modules may change how raw files are read, but they must still emit `raw_sequence` with the same keys and tensor orientation.
- Stream wrappers may change how they iterate, but they must still emit `stream_point` and `window`.
- Windowizers may change stride, padding policy, or overlap, but they must still emit `window` with `[L, D]`.
- Encoders may change internally, but they must still emit `hidden` with `[B, L, H]`.
- Models may add auxiliary outputs, but they must not remove or rename the fixed keys in `outputs`.
- Evaluators may compute different metrics, but they must still reduce predictions into the same `evaluation_record` schema.

If these interfaces remain fixed, then the codebase can swap SMD for another dataset, or MOMENT for another encoder, without forcing broad rewrites across the framework.
