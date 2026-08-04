---
date: 2026-07-28 18:58:21 +07
researcher: TheMetaSetter
git_commit: 71e798d488ddbd9edcbd100a41dc0384f96558be
branch: dev
repository: bachelor-thesis-2026
topic: "Plan to remove legacy processing flows while preserving full-spec-v2/v3 experiments"
tags: [plan, legacy-flow, simplification, smd, anomaly-detection]
status: draft
last_updated: 2026-07-28
last_updated_by: TheMetaSetter
---

# Preliminary plan: Simplify legacy processing flows

## 1. Purpose and scope

Kế hoạch này chuyển research markers L01-L13 trong [research note](../research/research-legacy-processing-flow-markers.md) thành các phase triển khai có thể kiểm chứng. Mục tiêu là giảm số processing path, nhưng vẫn giữ đúng các flow được yêu cầu bởi `full-spec-v2.md` và `full-spec-v3.md`:

- SMD với cửa sổ `L=20`;
- THESIS offline O0/O1 với Stage A, memory initialization và Stage B;
- threshold calibration trên clean validation;
- THESIS online A0/A1/A2 với projector-only update;
- stochastic query của v3 và `M=10` khi thí nghiệm yêu cầu;
- native baseline flows thuộc full benchmark của v2;
- checkpoint, metric, provenance, report và demo contracts.

Kế hoạch này chưa sửa hoặc xoá source code. Mỗi phase sẽ được lập detail plan riêng sau khi scope gate của phase đó được chấp nhận.

## 2. Design basis

Kế hoạch tuân theo `documents/abstract-design-notes/idea.md`, `documents/abstract-design-notes/design_starter.md` và `codebase_preferences.md`:

1. Giữ thin waist của hệ thống: batch contract, model output contract và checkpoint/artifact contract phải ổn định.
2. Giữ composition và các public entrypoint hiện hành; không thay bằng một hierarchy mới.
3. Mỗi model chính tiếp tục có forward, scoring và stage-specific behavior dễ đọc trong cùng public model surface.
4. Xoá nhánh legacy theo dependency graph, không xoá theo tên file hoặc theo số dòng.
5. Sau mỗi phase phải có test nhỏ, test contract và một smoke flow thích hợp.

## 3. Locked contracts

### 3.1 Offline and online batch

```python
batch = {
    "x": Tensor[B, 20, D],
    "point_labels": optional Tensor[B, 20],
    "mask": optional Tensor[B, 20, D],
    "timestamps": optional Tensor[B, 20],
    "meta": list[dict],
}
```

Online scorer không được nhận label. `view_a` và `view_b` không thuộc active online batch contract; nếu còn cần tạo augmentation thì việc đó phải xảy ra trong adaptation step, không phải là một compatibility field của stream output.

### 3.2 Model output

```python
outputs = {
    "hidden": Tensor[B, 20, H],
    "pooled": optional Tensor[B, H],
    "recon": optional Tensor[B, 20, D],
    "logits": optional Tensor,
    "point_scores": optional Tensor[B, 20],
    "window_scores": optional Tensor[B],
    "aux": dict,
}
```

### 3.3 Lifecycle

```text
Stage A multitask pretraining
  -> train-only continuous/discrete memory initialization
  -> freeze encoder and memories
  -> Stage B fusion/prediction-head fine-tuning
  -> clean-validation calibration
  -> offline evaluation or online A0/A1/A2
```

## 4. Scope gates before implementation

Các gate sau phải được trả lời hoặc kiểm chứng trong phase tương ứng:

| Gate | Câu hỏi | Mặc định an toàn trong kế hoạch |
|---|---|---|
| G1 | Main experiments có bao gồm toàn bộ v2 benchmark cells không? | Có, vì v2 quy định baseline là một phần full benchmark; không xoá baseline |
| G2 | Có cần đọc lại three-stage checkpoint/config cũ không? | Chưa xoá cho tới khi xác nhận không cần tái lập hoặc đã có migration policy |
| G3 | `reconstruction_mlp_ae` có còn là smoke/test-only model được duy trì không? | Xem là non-main candidate; chỉ xoá sau caller audit và test migration |
| G4 | Project có chính thức chỉ giữ SMD không? | Chưa xoá AnomalyArchive; chỉ tách khỏi active runtime sau khi có quyết định |
| G5 | Có cho phép metadata-only checkpoint resolution không? | Chuyển dần sang metadata-only, giữ fallback trong thời gian migration |

## 5. Phased implementation plan

### Phase 0 — Baseline inventory and safety snapshot

**Mục tiêu:** tạo bằng chứng trước khi thay đổi, để biết chính xác các active caller và tránh ảnh hưởng các thay đổi đang có trong worktree.

**Phạm vi:** repository state, registry, configs, tests, active benchmark entrypoints và các artifact/checkpoint fixture liên quan đến L01-L13.

**Kế hoạch:**

1. Ghi lại `git status`, active branch, current commit và các file đang modified; không reset hoặc overwrite thay đổi có sẵn.
2. Lập caller matrix cho `ThreeStageRuntimeConfig`, `_validate_three_stage_config`, `reconstruction_mlp_ae`, `anomaly_archive`, `legacy_stride`, `resolve_legacy_reference_checkpoint_path`, `online_encoder_params` và `validate_legacy_two_view_batch`.
3. Chạy test groups liên quan trước khi refactor để phân biệt regression mới với baseline failure.
4. Lập danh sách config được phép trong active v2/v3 benchmark và đánh dấu config historical/test-only.

**Exit criteria:** có caller matrix, test baseline và danh sách file bị ảnh hưởng; không có deletion nào ở phase này.

### Phase 1 — Migrate THESIS lifecycle to two-stage-only runtime

**Ưu tiên:** cao nhất.

**Mục tiêu:** loại bỏ processing branches Stage 1/Stage 2/Stage 3 khỏi model runtime, chỉ giữ Stage A/B theo v2/v3.

**File và symbol chính:**

- `src/models/thesis_multitask_impl/thesis_multitask_components.py`: thu hẹp `ThreeStageRuntimeConfig` về active two-stage runtime hoặc thay bằng cấu trúc two-stage hiện hành; loại bỏ tên Stage 3 khỏi active model surface.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`: loại bỏ nhánh `_phase_uses_*` và trainable-parameter logic cho `stage1_*`, `stage2_recovery`, `multitask_pretraining` và Stage 3.
- `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py`: loại bỏ memory/fusion substeps riêng của Stage 3; giữ schedule cho Stage A, memory initialization và Stage B.
- Các caller của `runtime: ThreeStageRuntimeConfig`: cập nhật về contract two-stage, không tạo compatibility alias mới.

**Bảo toàn:** objective O0/O1, memory initialization từ train, freeze semantics, Stage B fusion/prediction heads, output contract và checkpoint payload active.

**Test plan:**

- cập nhật `tests/models/test_multitask_config_refactor.py` và các phase/state tests để chỉ kiểm tra Stage A/B;
- giữ test xác nhận Stage B freeze encoder/memory;
- thêm test rejection cho config phase legacy thay vì chạy legacy flow;
- chạy model shape, one train step, checkpoint round-trip và một two-stage smoke.

**Exit criteria:** active THESIS config không đi qua Stage 3; một Stage A → memory init → Stage B smoke pass; không còn branch runtime cho phase legacy.

### Phase 2 — Remove three-stage configuration and compatibility aliases

**Mục tiêu:** làm config layer phản ánh đúng một lifecycle thay vì vừa đọc active config vừa giữ historical three-stage validator.

**File và symbol chính:**

- `src/core/config.py`: loại bỏ constants/normalizers cho `three_stage`, `stage3_prototype_warmup`, `three_stage_phase` và các alias phase tương ứng sau khi Phase 1 hoàn tất.
- `src/core/config_experiment_validation.py`: bỏ nhánh validate three-stage; giữ validate two-stage.
- `src/core/config_model_validation.py`: thu hẹp allowed keys và phase validation theo active model/config contract.
- `configs/model/thesis_multitask_three_stage_window20.yaml` và `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`: xoá hoặc chuyển sang archival location theo quyết định G2; không để chúng được runtime discovery.
- Các config snapshots/compliance fixtures: cập nhật một lần theo active schema.

**Test plan:**

- active two-stage config load/validation;
- rejection rõ ràng cho `three_stage` và historical aliases;
- alias conflict tests được thay bằng unknown/unsupported config tests;
- kiểm tra generated online benchmark configs vẫn có `stage_b_fusion_finetuning` và metadata cần thiết.

**Exit criteria:** config loader có một canonical lifecycle; không còn three-stage config được chấp nhận bởi runtime; error message chỉ dẫn sang two-stage schema.

### Phase 3 — Remove standalone reconstruction model path

**Mục tiêu:** loại bỏ model không thuộc main v2/v3 experiment flow sau khi xác nhận không còn caller cần thiết.

**File và symbol chính:**

- `src/models/reconstruction_mlp_ae.py`;
- `configs/model/reconstruction_mlp_ae.yaml`;
- `src/core/runtime_components.py` registration;
- `src/core/config_model_validation.py` allowed model keys và model-specific validation;
- test registry, shape, train-step, checkpoint và launcher fixtures đang dùng model này;
- compliance snapshot liên quan.

**Migration rule:** không thay `reconstruction_mlp_ae` bằng một facade giả. Nếu một test chỉ cần checkpoint container hoặc generic model registry, chuyển test sang `thesis_multitask` active model hoặc test primitive đúng mục tiêu.

**Test plan:**

- tìm kiếm toàn repository để chứng minh không còn active experiment config dùng model;
- chạy registry/config snapshot sau khi gỡ registration;
- chạy Stage A/B model shape, one train step và checkpoint round-trip;
- kiểm tra các benchmark launcher không sinh lại `reconstruction_mlp_ae`.

**Exit criteria:** registry chỉ chứa model thuộc active scope; không còn test/launcher/config production nào yêu cầu standalone reconstruction model.

### Phase 4 — Narrow online adaptation to the active projector-only contract

**Mục tiêu:** loại bỏ parameter target branch không được active online engine hỗ trợ.

**File và symbol chính:**

- `src/models/online_impl/online_adaptation.py`: xoá `online_encoder_params` branch trong `_parameters_for_target_group` và thu hẹp error message/type surface.
- `src/core/config_model_validation.py`: chỉ chấp nhận `target_param_group: projector_params`.
- `src/engine/online_tta/online_engine_shared.py`: giữ invariant projector-only và đồng bộ message/validation.
- tests online target-group/trainable-surface: giữ projector update, frozen reference/encoder và reject unsupported target.

**Bảo toàn:** A0 frozen, A1 reconstruction-only, A2 reconstruction-plus-contrastive, projector anchor và reset/state round-trip.

**Test plan:**

- assert only projector parameters change;
- assert reference encoder and online encoder remain frozen;
- A0/A1/A2 step tests;
- online state round-trip;
- online smoke với config main và `L=20`.

**Exit criteria:** không còn public path để update online encoder; config, model và engine cùng dùng một target-group contract.

### Phase 5 — Remove obsolete stream mode and legacy online batch fixture contract

**Mục tiêu:** làm online data path dùng một active window policy và bỏ test/contract residue của hai-view stream batch.

**File và symbol chính:**

- `src/data/stream.py`: sau caller audit, loại bỏ `legacy_stride`; giữ `sliding_stride_1` và/hoặc `nonoverlap_tail` theo benchmark protocol.
- `src/core/contracts.py`: giữ active `validate_online_batch`; loại bỏ `validate_legacy_two_view_batch` nếu không còn caller.
- `src/core/console.py`: bỏ instrumentation cho `view_a/view_b` nếu không còn production batch caller.
- `tests/online/test_online_stream.py`, `test_online_tta_variants.py`, `test_online_adaptation_step.py`, `test_online_engine_max_steps.py`: chuyển fixture sang active batch contract.

**Test plan:**

- stream index and tail policy tests;
- assert window shape `[B,20,D]`;
- assert labels are not passed to scorer;
- assert active batch has no `view_a/view_b`;
- one online engine smoke with sliding policy and one with non-overlap tail if v2 requires both.

**Exit criteria:** online stream có ít policy hơn, mọi active caller dùng cùng contract, và legacy two-view validator không còn reachable.

### Phase 6 — Replace flat checkpoint fallback with metadata-only resolution

**Mục tiêu:** giữ một canonical checkpoint resolution path cho benchmark, tránh path guessing và layout fallback.

**File và symbol chính:**

- `src/engine/online_tta/checkpoint_resolution.py`: loại bỏ `resolve_legacy_reference_checkpoint_path` sau migration; giữ `resolve_stage_b_checkpoint` và metadata validation.
- `src/models/online_impl/online_adaptation.py`: bỏ wrapper/fallback call; yêu cầu canonical Stage B checkpoint.
- `src/engine/online_tta/online_engine_run.py` và `scripts/benchmarks/run_thesis_online_benchmark.py`: luôn truyền metadata hoặc canonical resolved path.
- online checkpoint tests và benchmark wrapper tests: thay fallback-success tests bằng missing/ambiguous metadata rejection tests.

**Bảo toàn:** SHA256/provenance, stage name, benchmark mode, entity, seed và Stage B checkpoint identity.

**Exit criteria:** một input contract cho online checkpoint; missing/ambiguous metadata fail early; không còn flat-path guessing.

### Phase 7 — Decide and isolate non-main dataset and operational paths

**Mục tiêu:** xử lý các thành phần không thuộc SMD main processing flow mà không xoá nhầm research evidence.

**Phạm vi:** `anomaly_archive` runtime registration, parser, analysis/visualization scripts và `scripts/ops/*`.

**Kế hoạch:**

1. Dựa trên G4, quyết định AnomalyArchive là active dataset, archival analysis dataset hay out of project scope.
2. Nếu out of scope, bỏ nó khỏi active runtime registry/config validation trước; giữ research artifacts và chuyển parser/analysis vào archival boundary trước khi xoá.
3. Không xoá operational prune/backfill/re-evaluate tools trong simplification batch. Tạo retention/operations plan riêng nếu chúng không còn cần cho provenance.
4. Cập nhật documentation và tests để phân biệt active scientific processing với archival/operational tooling.

**Exit criteria:** runtime scope được ghi rõ; non-main paths không còn bị hiểu là active experiment path; archival evidence vẫn truy cập được nếu cần.

## 6. Validation matrix

Sau mỗi phase, chạy tối thiểu các nhóm phù hợp bằng `.venv/bin/python` và `pytest`:

| Layer | Validation |
|---|---|
| Config | active config load, schema validation, rejection of removed schema |
| Data | SMD loader, window shape, stream policy, no-label online contract |
| Model | model shapes, one forward/backward step, phase trainable set |
| Checkpoint | save/load, Stage B identity, online state round-trip |
| Online | A0/A1/A2 behavior, projector-only update, triage contract |
| Benchmark | one concrete v2/v3 smoke combination before scaling |
| Documentation | spec links, active path map, changelog/research provenance |

Một phase chỉ được coi là hoàn tất khi test của phase pass và một smoke flow end-to-end tương ứng pass. Các baseline failures đã tồn tại trước refactor phải được ghi nhận riêng, không được gán cho phase mới.

## 7. Risk controls

- **Mất khả năng tái lập checkpoint cũ:** giữ migration fixture trong Phase 0; chỉ xoá compatibility sau khi G2 được quyết định.
- **Xoá nhầm baseline v2:** đóng scope G1 trước Phase 7; giữ toàn bộ native baseline branch.
- **Thay đổi online semantics:** trước/sau Phase 4-6 kiểm tra trainable parameter set, stream cursor, threshold metadata và state round-trip.
- **Registry/config drift:** mỗi model hoặc dataset bị gỡ phải đồng thời cập nhật registry, validator, config, test và compliance snapshot.
- **Refactor quá rộng:** mỗi phase chỉ xử lý một boundary; không đổi architecture, loss hay metric nếu không cần để xoá legacy branch.
- **Artifact không còn truy nguyên:** giữ metadata, selected diagnostics, initialization/best checkpoints và report summary theo retention policy hiện hành.

## 8. Dependency order

```text
Phase 0
  -> Phase 1
  -> Phase 2
  -> Phase 3
  -> Phase 4
  -> Phase 5
  -> Phase 6
  -> Phase 7
```

Phase 3 có thể tách thành một change set độc lập sau Phase 0 nếu G3 đã được chốt. Phase 7 chỉ được thực hiện sau khi quyết định rõ phạm vi dataset và benchmark; baseline L12 và operational tooling L13 không nằm trong các phase xoá mặc định.

## 9. Next step

Bước kế tiếp là dùng `prompts/3_structure_prompt.md` để chuyển kế hoạch này thành cấu trúc phase/stage chính thức. Chi tiết code-level của từng stage sẽ được viết bằng `prompts/4_detail_prompt.md` ngay trước khi triển khai phase tương ứng.

