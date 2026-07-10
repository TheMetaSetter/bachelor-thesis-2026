---
date: 2026-07-10T18:30:00+0700
researcher: Codex
topic: "Current code flow for full-spec-v2 experiments"
status: current_snapshot
---

# Flow theo code hiện tại

## Offline THESIS và các baseline

```text
experiment YAML
    |
    v
config loader + validation
    |
    +--> dataset builder --> cleaning --> train-fitted scaler
    |                              |
    |                              v
    |                       overlapping windows [B,L,C]
    |
    +--> registry model
    |       |
    |       +--> THESIS / RedLamp / reconstruction AE
    |       +--> traditional ML adapter (IForest, KMeansAD, STUMPY)
    |
    v
Trainer.train()
    |
    +--> forward --> loss --> backward --> optimizer/scheduler
    +--> validation --> point scores --> threshold selection
    +--> checkpoint + metric/artifact sinks
    |
    v
Evaluator.evaluate()
    |
    +--> window scores --> overlap aggregation --> point/range/VUS metrics
    +--> report JSON + visual artifacts
```

## Online THESIS O0/O1 + A0/A1/A2

```text
Stage-B reference checkpoint + online YAML
    |
    v
OnlineAdaptationModel
    |  reference encoder frozen
    |  online encoder frozen
    |  online_mlp_projector trainable
    v
clean validation per entity
    |
    +--> point scores --> EWMA --> threshold artifact
    +--> input-window / latent-window thresholds
    |
    v
test stream, one causal window at a time
    |
    +--> score_window + EWMA
    +--> exact triage: normal / hard-old / gray / strong anomaly
    +--> optional prototype signature helpers --> PNN mask
    +--> VerificationBuffer admission and cycle TTL
    +--> A0: no update
    +--> A1: masked PNN reconstruction update
    +--> A2: hard-old hinge or contrastive update
    |       fresh AdamW, lr=1e-4, wd=1e-4, clip=0.5
    v
online records + diagnostics + checkpoint extra state
```

## Demo queue

```text
sequence --> StreamQueueController (producer thread)
                         |
                         v
                 demo consumer/replay
                         |
                         +--> wait until L points
                         +--> latest window callback
                         +--> provisional plot state
                         v
                         demo/app.py renders only
```

## Trạng thái hiện tại

- THESIS, RedLamp, reconstruction và traditional adapters dùng chung config,
  loader, evaluator và report pipeline.
- Online queue lifecycle đã tách thành `demo/stream_queue.py`.
- PNN, buffer-cycle TTL, hard-old hinge và optimizer factory đã có helper và
  test focused; việc nối PNN prototype state trực tiếp vào từng online event
  vẫn là phần cần xác nhận thêm bằng integration smoke.
