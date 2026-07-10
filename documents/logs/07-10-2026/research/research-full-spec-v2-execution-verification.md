---
date: 2026-07-10
topic: "full-spec-v2 execution verification"
status: partial_verified
---

# Verification result

The active SMD benchmark matrix is loadable and the required smoke runs were
executed with `.venv/bin/python`:

- offline O0: completed;
- offline O1: completed;
- online O0-A0: completed;
- online O0-A2: completed;
- online O1-A0: completed;
- online O1-A2: completed.

Each completed run wrote its benchmark report, threshold artifact, score files,
metrics, and checkpoint artifacts under `outputs/benchmark_smoke/`.

The matrix preflight reports 18 THESIS offline configs, 54 THESIS online
configs, 9 RedLamp configs, and 81 traditional/online baseline configs. Online
YAML generation now points to the two-stage Stage-B checkpoint:

```text
.../{offline_variant}/{entity}/seed{seed}/two_stage/
    stage_b_fusion_finetuning/checkpoints/best.pt
```

The online model also accepts legacy flat checkpoint paths and resolves them to
the Stage-B location when the flat path is absent.

# Remaining gap

`full-spec-v2.md` describes stricter A1/A2 semantics than the current first
online-TTA slice. In particular, the runtime still needs dedicated
window-threshold calibration, recurrent-signature PNN masks, verification-cycle
TTL semantics, and the exact A2 hard-old/contrastive loss composition before
the implementation can be called fully spec-complete.

The repository-wide `pytest -q` is not green because legacy tests still import
removed `configs/experiment/smoke/*w100*` files. The active full-spec focused
tests and matrix preflight pass.
