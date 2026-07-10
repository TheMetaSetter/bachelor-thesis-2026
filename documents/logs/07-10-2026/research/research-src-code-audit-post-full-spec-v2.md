---
date: 2026-07-10T18:35:00+0700
researcher: Codex
git_commit: 8e9b208c7ef485eb5d74b5128a97e68b2a8dcdb1
branch: dev
repository: bachelor-thesis-2026
topic: "Post-full-spec-v2 source audit refresh"
tags: [research, source-audit, online-tta, tests, current-state]
status: complete
last_updated: 2026-07-10
last_updated_by: Codex
---

# Research: trạng thái source audit sau full-spec-v2 remediation

## Research question

Xác định các source path, runtime flow và test groups hiện đang là cơ sở thật
cho việc làm mới `detail-src-code-audit-remediation.md` sau các thay đổi
online-TTA và demo queue.

## Summary

AST scanner hiện ghi nhận 12 file `src/` vượt 500 dòng và 71 callable vượt 50
dòng. Vì vậy kế hoạch source-audit cũ không còn đúng baseline. Các helper mới
đã tách threshold calibration, signature verification, optimizer, non-overlap
guard và demo queue, nhưng `online_engine.py` vẫn là entrypoint lớn cần chia
theo responsibility. Public registry và checkpoint surfaces vẫn nằm ở
`src/core/runtime_components.py`, `src/models/online_adaptation.py` và các
script benchmark.

## Current implementation map

### Runtime

- `src/core/config.py`, `src/core/config_model_validation.py`: load và validate
  experiment YAML.
- `src/core/registry.py`, `src/core/runtime_components.py`: model/dataset
  registry; public names gồm `thesis_multitask`, `redlamp_baseline`,
  `reconstruction_mlp_ae`, `online_adaptation`.
- `src/engine/trainer.py`: offline loop, validation, checkpoint và logging.
- `src/engine/evaluator.py`: overlap aggregation và evaluation metrics.
- `src/engine/online_tta/online_engine.py`: calibration, causal scoring,
  triage, online update, output serialization.
- `src/engine/online_tta/threshold_calibration.py`: pure score/quantile helpers.
- `src/engine/online_tta/signature_verification.py`: codeword/radius,
  continuous signature và PNN mask helpers.
- `src/engine/online_tta/verification_buffer.py`: admission, status và cycle
  TTL; `non_overlap_guard.py` bảo vệ hard-old intervals.
- `demo/stream_queue.py`: producer/consumer lifecycle độc lập với model.

### Tests

Focused online tests hiện gồm `test_online_tta_variants.py`,
`test_online_tta_triage.py`, `test_online_verification_buffer.py`,
`test_online_signature_verification.py`, `test_threshold_artifact.py`,
`test_online_entrypoint.py`, `test_online_streaming_benchmark_wrapper.py` và
`test_demo_stream_queue.py`. Compliance được quét bởi
`tests/codebase_compliance.py` và `test_codebase_compliance_scanner.py`.

## AST audit result

```text
files > 500 lines: 12
callables > 50 lines: 71
```

Các file cần đưa vào source-refactor plan mới: `src/core/config.py`,
`src/core/config_model_validation.py`, `src/data/augment.py`,
`src/engine/online_tta/online_engine.py`, `src/engine/trainer.py`,
`src/metrics/pointwise.py`, `src/models/online_adaptation.py`,
`src/models/redlamp_baseline.py`, và bốn thesis mixin hiện còn tồn tại.

## Open implementation facts

- Entity threshold artifact fields đã được mở rộng nhưng calibration engine
  hiện vẫn cần hoàn chỉnh đường multi-entity và artifact selection.
- Signature helpers hiện là pure functions; engine chưa tự động lấy codebook,
  anomaly radius và recurrent history cho mọi event.
- Demo queue lifecycle đã có test; `demo/online_replay.py` vẫn chủ yếu đọc
  report artifact và chưa phải một live consumer hoàn chỉnh.
- Các test cũ vẫn import trực tiếp nhiều module lớn; khi refactor phải giữ
  public constructor, registry key, output dictionaries và checkpoint keys.

## Verification after refresh

Focused online/demo/compliance tests pass (`14 passed`). Repository-wide
`pytest -x -q` currently stops at `tests/test_ablation_runner.py` because its
legacy `w100` smoke YAML path no longer exists. This is a test/config ownership
gap, not evidence that the online helper tests are green repository-wide.

## Updated planning basis

Kế hoạch source-audit mới phải bắt đầu bằng contract tests, sau đó tách
`online_engine.py`, config validation, trainer/evaluator, augmentation và
metrics. Không được dùng số liệu 13 files/69 callables trong plan cũ nữa.
