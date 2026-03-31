This block is written for direct paste into your idea file. It reflects the tool choice that best fits your datasets and architecture, and it aligns with the current idea text you shared earlier.  The selected tools and URLs below come from the official project pages and documentation for River, TSGM, tsaug, MOA, TimeSynth, DeepEcho, and agots. ([riverml.xyz][1])

````markdown
## Streaming simulation and drift generation design

To support the online adaptation part of the thesis, I will not rely on a single all-in-one simulator. Instead, I will use a **hybrid streaming framework** composed of a real-dataset stream wrapper, a drift injection layer, and an optional synthetic multivariate time-series generator.

This choice is more suitable for my datasets: **SMD, MSL, SMAP, SWaT, and UCR Anomaly Archive**. These datasets are naturally offline benchmark datasets, so the most practical solution is to convert them into sequential streams and then inject controlled drift or non-stationarity on top of them.

### Final choice

I choose the following stack as the main solution:

1. **River** as the primary Python streaming backbone.
2. **Custom dataset stream wrappers** for SMD, MSL, SMAP, SWaT, and UCR.
3. **Custom drift injection operators** as the main mechanism for generating non-stationarity.
4. **tsaug** as a helper library for simple time-series augmentation primitives.
5. **TSGM** as the optional synthetic multivariate time-series generator.
6. **MOA** as an optional benchmark-oriented concept drift engine, not as the main codebase dependency.

I do **not** choose scikit-multiflow as the main dependency because its project states that it is merging into River, so River is the cleaner long-term choice.

### Why this is the best fit for my case

My thesis does not only need synthetic streams. It needs a framework that can:

- evaluate models on realistic benchmark datasets in an online manner,
- simulate multiple kinds of drift on top of those datasets,
- remain modular enough to support different encoders, anomaly detectors, and online adaptation strategies,
- stay Python-first and easy to integrate into a reusable thesis codebase.

A pure synthetic generator would not be enough, because the final evaluation should still be performed on realistic benchmark streams derived from SMD, MSL, SMAP, SWaT, and UCR. Therefore, the best design is:

```text
offline dataset
-> sequential stream wrapper
-> sliding window construction
-> drift injection / non-stationarity injection
-> online model update and evaluation
````

### Selected tools and their roles

#### 1. River

**Role:** main stream-learning backbone for the Python codebase.

I will use River for the stream abstraction and for concept-drift-oriented experimentation. River already provides a `ConceptDriftStream`, which is useful when I want abrupt or gradual switching between two concepts. However, since my thesis focuses on multivariate time series rather than generic tabular streams, River will serve as the infrastructure layer rather than the full simulator.

**Official URLs**

* [https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream/](https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream/)
* [https://riverml.xyz/](https://riverml.xyz/)
* [https://github.com/online-ml/river](https://github.com/online-ml/river)

#### 2. Custom dataset stream wrappers

**Role:** convert each benchmark dataset into an online stream.

I will implement dataset-specific wrappers for:

* SMD
* MSL
* SMAP
* SWaT
* UCR Anomaly Archive

Each wrapper will expose a unified interface and emit either:

* one multivariate time point at a time, or
* one sliding window at a time.

This is the core adaptation needed for my thesis, because no existing library directly provides these benchmark datasets in exactly the streaming form I need.

#### 3. Custom drift injection operators

**Role:** inject controlled non-stationarity into real benchmark streams.

This is the most important part of the streaming simulation design. I will implement a `DriftInjector` module that can operate on top of the real dataset stream. This module will support controlled start time, duration, strength, affected channels, and drift type.

The first set of drift operators should include:

* **Mean drift**: add a channel-wise offset.
* **Variance drift**: inflate or shrink channel variance.
* **Trend drift**: introduce a time-varying slope.
* **Seasonality / frequency drift**: modify oscillation frequency or amplitude.
* **Correlation drift**: change dependency structure between variables.
* **Sensor dropout**: zero out or mask selected channels for a duration.
* **Delay / phase drift**: shift selected channels relative to others.
* **Noise drift**: gradually increase noise level.
* **Regime switch**: replace one segment by data sampled from another operating regime.
* **Mixed drift**: compose several drift operators together.

This approach is better than depending only on a generic concept-drift package, because multivariate time-series drift often occurs at the signal level rather than only at the label-distribution level.

#### 4. tsaug

**Role:** helper library for low-level augmentation primitives.

I will use tsaug only as a convenience layer for some local perturbation operators such as magnitude warping, window warping, cropping-like distortions, or noise-like perturbations. It is not the main drift framework, but it is useful as a building block inside my custom drift injector.

**Official URLs**

* [https://github.com/arundo/tsaug](https://github.com/arundo/tsaug)
* [https://tsaug.readthedocs.io/en/stable/](https://tsaug.readthedocs.io/en/stable/)

#### 5. TSGM

**Role:** optional synthetic multivariate time-series generation.

When I need fully synthetic data, extra pretraining data, or controlled simulator-based experiments, I will use TSGM. This is the best secondary tool because it is specifically designed for synthetic time-series generation and evaluation, including generative and simulator-based methods.

However, TSGM is **secondary**, not primary, because my thesis evaluation should still be anchored in real benchmark datasets.

**Official URLs**

* [https://tsgm.readthedocs.io/](https://tsgm.readthedocs.io/)
* [https://github.com/alexandervnikitin/tsgm](https://github.com/alexandervnikitin/tsgm)

#### 6. MOA

**Role:** optional benchmark tool for classical concept drift experiments.

MOA is useful when I want to compare against classical stream-mining drift setups or reproduce formal concept-drift benchmarks. But it is Java-first, so I will not use it as the center of my codebase.

**Official URLs**

* [https://moa.cms.waikato.ac.nz/](https://moa.cms.waikato.ac.nz/)
* [https://moa.cms.waikato.ac.nz/details/classification/streams/](https://moa.cms.waikato.ac.nz/details/classification/streams/)
* [https://github.com/Waikato/moa](https://github.com/Waikato/moa)

### Optional tools that I may use only for side experiments

#### TimeSynth

Useful for simple handcrafted synthetic signal generation and sanity checks, especially when I want quick experiments with regular or irregular synthetic time series.

**URLs**

* [https://github.com/timesynth/timesynth](https://github.com/timesynth/timesynth)
* [https://github.com/TimeSynth](https://github.com/TimeSynth)

#### DeepEcho

Useful only if I later want to learn a generative model for multivariate time series and sample synthetic sequences from it. It is not the first choice for the main thesis framework.

**URLs**

* [https://github.com/sdv-dev/DeepEcho](https://github.com/sdv-dev/DeepEcho)
* [https://pypi.org/project/deepecho/](https://pypi.org/project/deepecho/)

#### agots

Useful if I want an anomaly-generation helper for synthetic anomaly injection. It is closer to an anomaly generator than a general drift simulator, so it is not the main choice.

**URLs**

* [https://github.com/KDD-OpenSource/agots](https://github.com/KDD-OpenSource/agots)

### Software design decision

The streaming part of the thesis codebase should be built around the following modules:

```text
DatasetStream
    -> loads one dataset and exposes sequential access

Windowizer
    -> converts raw sequential stream into windows of length L

DriftInjector
    -> injects controlled non-stationarity into raw streams or windows

StreamScenario
    -> defines when drift starts, ends, and which channels are affected

OnlineEvaluator
    -> feeds streamed windows to the model and records online metrics
```

A clean data flow is:

```text
raw dataset
-> DatasetStream
-> optional DriftInjector
-> Windowizer
-> model
-> online adaptation
-> online evaluation
```

### Unified interface

For reusability, each dataset stream should expose a unified interface such as:

```python
next_point()
next_window()
reset()
state_dict()
```

and each drift injector should expose a unified interface such as:

```python
apply_point(x_t, t)
apply_window(X_win, t_start, t_end)
reset()
```

This will let me swap:

* datasets,
* drift types,
* window lengths,
* anomaly detectors,
* online adaptation strategies

without changing the high-level training and evaluation loop.

### Recommended experimental protocol

I will evaluate under three regimes:

1. **Clean streaming**

   * No injected drift.
   * Purpose: verify whether the online pipeline itself works.

2. **Real-data streaming with injected drift**

   * Benchmark datasets streamed sequentially.
   * Controlled drift inserted at selected segments.
   * Purpose: evaluate robustness to non-stationarity.

3. **Fully synthetic multivariate streaming**

   * Generated by TSGM or small handcrafted generators.
   * Purpose: controlled ablation and stress tests.

This is a stronger design than using only one regime, because it separates:

* realism,
* controllability,
* debugging convenience.

### Final conclusion for the thesis idea

The main streaming simulation solution for this thesis is:

* **River** for stream infrastructure,
* **custom dataset wrappers** for SMD, MSL, SMAP, SWaT, and UCR,
* **custom drift injectors** for multivariate non-stationarity,
* **tsaug** for simple augmentation primitives,
* **TSGM** for optional synthetic multivariate generation,
* **MOA** only for optional classical concept-drift benchmarking.

This design is the best fit because it is modular, Python-first, compatible with real anomaly-detection benchmarks, and flexible enough for online adaptation research.

## Updated current recommended plan

The cleanest current plan is:

1. Use **TSLib-style structure** and build the codebase around a stable encoder contract.
2. Start with **SMD** and a minimal vertical slice.
3. Implement the streaming layer as:

   * `DatasetStream`
   * `Windowizer`
   * `DriftInjector`
   * `OnlineEvaluator`
4. Use **River** as the stream backbone.
5. Use **custom drift injectors** as the primary way to simulate non-stationarity on real benchmark streams.
6. Use **tsaug** only as a helper library inside the drift injector.
7. Use **TSGM** only when fully synthetic multivariate sequences are needed.
8. Keep **MOA** as an optional external benchmark tool, not as the main codebase dependency.
9. Then add, in order:

   * continuous prototypes,
   * discrete prototypes,
   * task-specific fusion,
   * online adaptation.
10. Evaluate under:

* clean streaming,
* real streaming with injected drift,
* fully synthetic streaming.

## Necessary URLs

### Main choices

* River: [https://riverml.xyz/](https://riverml.xyz/)

* River ConceptDriftStream: [https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream/](https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream/)

* River GitHub: [https://github.com/online-ml/river](https://github.com/online-ml/river)

* TSGM docs: [https://tsgm.readthedocs.io/](https://tsgm.readthedocs.io/)

* TSGM GitHub: [https://github.com/alexandervnikitin/tsgm](https://github.com/alexandervnikitin/tsgm)

* tsaug GitHub: [https://github.com/arundo/tsaug](https://github.com/arundo/tsaug)

* tsaug docs: [https://tsaug.readthedocs.io/en/stable/](https://tsaug.readthedocs.io/en/stable/)

* MOA homepage: [https://moa.cms.waikato.ac.nz/](https://moa.cms.waikato.ac.nz/)

* MOA streams docs: [https://moa.cms.waikato.ac.nz/details/classification/streams/](https://moa.cms.waikato.ac.nz/details/classification/streams/)

* MOA GitHub: [https://github.com/Waikato/moa](https://github.com/Waikato/moa)

### Optional tools

* TimeSynth GitHub: [https://github.com/timesynth/timesynth](https://github.com/timesynth/timesynth)

* TimeSynth org page: [https://github.com/TimeSynth](https://github.com/TimeSynth)

* DeepEcho GitHub: [https://github.com/sdv-dev/DeepEcho](https://github.com/sdv-dev/DeepEcho)

* DeepEcho PyPI: [https://pypi.org/project/deepecho/](https://pypi.org/project/deepecho/)

* agots GitHub: [https://github.com/KDD-OpenSource/agots](https://github.com/KDD-OpenSource/agots)

### Not chosen as the main dependency

* scikit-multiflow GitHub: [https://github.com/scikit-multiflow/scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow)
* scikit-multiflow docs: [https://scikit-multiflow.github.io/](https://scikit-multiflow.github.io/)

```

**Confidence:** High.  
**Key assumptions:** you want a Python-first reusable framework, the stream is built from real TSAD benchmarks rather than only synthetic data, and you want drift injection to be signal-level and multivariate rather than only generic tabular concept drift. :contentReference[oaicite:2]{index=2}
::contentReference[oaicite:3]{index=3}
```

[1]: https://riverml.xyz/dev/api/datasets/synth/ConceptDriftStream/?utm_source=chatgpt.com "ConceptDriftStream"
