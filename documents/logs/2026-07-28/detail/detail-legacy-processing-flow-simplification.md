---
date: 2026-07-28 18:58:21 +07
topic: "Detailed atomic steps for legacy processing-flow simplification"
status: ready
revision: 29981c0ec294d26b3f31e421f83db012762ff73b
source_structure: documents/logs/07-28-2026/structure/structure-legacy-processing-flow-simplification.md
related_documents:
  - documents/logs/07-28-2026/plan/plan-legacy-processing-flow-simplification.md
  - documents/logs/07-28-2026/research/research-legacy-processing-flow-markers.md
  - documents/spec/full-spec-v2.md
  - documents/spec/full-spec-v3.md
  - prompts/4_detail_prompt.md
---

# Detailed implementation: Simplify legacy processing flows

## Summary

Tài liệu này mở rộng cấu trúc Phase → Stage thành các atomic steps. Mỗi step chỉ thực hiện một thay đổi nhỏ, có thể kiểm tra độc lập và phải dừng nếu phát hiện caller, config hoặc artifact dependency chưa được xử lý.

Phạm vi được giữ nguyên theo structure document:

- giữ SMD, `L=20`, THESIS O0/O1, Stage A, memory initialization, Stage B, calibration và online A0/A1/A2;
- giữ stochastic query và `M=10` của v3 khi experiment config yêu cầu;
- giữ native baseline flows của full-spec-v2;
- đánh dấu và loại bỏ từng legacy path theo dependency order;
- không xoá operational tooling hoặc AnomalyArchive trước khi scope gate tương ứng được chốt.

Tài liệu này là hướng dẫn triển khai. Chưa có source code, config, test hoặc artifact nào được thay đổi bởi việc viết tài liệu này.

## 1. Global execution rules

### 1.1 Unit of work

1. Thực hiện đúng một atomic step.
2. Chạy verification được ghi ngay sau step đó.
3. Nếu verification fail, giữ thay đổi ở trạng thái có thể đảo ngược và dừng tại stage hiện tại.
4. Không bắt đầu stage kế tiếp khi completion condition của stage hiện tại chưa đạt.
5. Không sửa architecture, loss, metric hoặc baseline protocol nếu step đó không cần để loại bỏ legacy path.
6. Bảo toàn các thay đổi đã có trong worktree; không dùng destructive reset hoặc checkout để làm sạch worktree.

### 1.2 Stable contracts

Offline và online batch active phải giữ `x` với shape `[B, 20, D]`. Online scorer không nhận label. Active model output phải giữ các field đã được dùng trong contracts: `hidden`, `recon`, `logits`, `point_scores`, `window_scores` và `aux` khi field đó có ý nghĩa với model.

Online THESIS chỉ cho phép `target_param_group: projector_params`. Baseline online vẫn giữ protocol riêng và không được kế thừa THESIS triage, PNN hoặc projector update.

### 1.3 Verification rule

Verification chính dùng `.venv/bin/python -m pytest ...` với test paths đã tồn tại trong repository. Trước full suite, ưu tiên test group của stage và một smoke flow có một tổ hợp cụ thể. Các baseline failures đã biết phải được ghi riêng, không được coi là bằng chứng rằng stage mới đúng hoặc sai.

## 2. Phase 0 — Establish safety boundary

### Stage 0.1 — Capture repository state

**Boundary**

- Repository root, `documents/spec/`, `documents/abstract-design-notes/`, `configs/`, `src/`, `scripts/` và `tests/`.
- Current documentation: research note và structure document của task.

**Atomic steps**

1. Đọc `AGENTS.md`, `codebase_preferences.md`, `documents/spec/full-spec-v2.md`, `documents/spec/full-spec-v3.md` và active design notes.
2. Ghi `git rev-parse HEAD`, branch và `git status --short` vào execution record của work session.
3. Liệt kê các file đang modified hoặc untracked trước khi bắt đầu refactor.
4. Xác định file nào thuộc task và file nào là thay đổi có sẵn của anh; không gộp hai nhóm vào một change set.
5. Ghi revision của research, plan và structure documents vào phần metadata của detail document hoặc execution record.

**Verification**

- Repository state có commit, branch và worktree snapshot.
- Không có lệnh destructive được chạy.

**Completion condition**

Có thể xác định source change nào được tạo bởi từng stage sau này.

**Stop/rollback**

Nếu worktree chứa thay đổi chồng lấp trực tiếp với target symbols, dừng trước khi edit và tách phạm vi thay đổi.

### Stage 0.2 — Build legacy caller map

**Boundary**

- `src/models/thesis_multitask_impl/thesis_multitask_components.py`
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`
- `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py`
- `src/core/config.py`
- `src/core/config_experiment_validation.py`
- `src/core/config_model_validation.py`
- `src/models/reconstruction_mlp_ae.py`
- `src/data/stream.py`
- `src/core/contracts.py`
- `src/engine/online_tta/checkpoint_resolution.py`
- `src/models/online_impl/online_adaptation.py`

**Atomic steps**

1. Tìm definition và caller của `ThreeStageRuntimeConfig`.
2. Tìm definition và caller của `_validate_three_stage_config`.
3. Tìm tất cả config và test chứa `three_stage`, `stage3_`, `stage1_`, `stage2_recovery` và `multitask_pretraining`.
4. Tìm tất cả caller của `reconstruction_mlp_ae` và phân biệt production, launcher, test-only.
5. Tìm tất cả caller/config của `legacy_stride`.
6. Tìm tất cả caller của `validate_legacy_two_view_batch`, `view_a` và `view_b`.
7. Tìm tất cả caller của `resolve_legacy_reference_checkpoint_path` và `reference_checkpoint_path`.
8. Tìm tất cả caller/config của `online_encoder_params`.
9. Ghi từng kết quả vào caller matrix; mỗi row phải có definition, caller, config/test evidence và classification.

**Verification**

- Mỗi marker L01-L13 có ít nhất một evidence về definition và một evidence về use hoặc được ghi rõ là chưa tìm thấy caller.

**Completion condition**

Không còn marker nào được đánh dấu “xoá ngay” nếu chưa có caller evidence.

**Stop/rollback**

Nếu caller map phát hiện active v2/v3 flow phụ thuộc marker, chuyển marker thành migration candidate và không xoá trong phase tương ứng.

### Stage 0.3 — Classify active versus historical paths

**Boundary**

- `documents/spec/full-spec-v2.md` và `documents/spec/full-spec-v3.md`.
- `src/engine/online_tta/triage.py` và `src/baselines/online/*`.
- Main benchmark configs và launchers.

**Atomic steps**

1. Lập danh sách THESIS offline active path: O0/O1, Stage A, memory initialization, Stage B, calibration và evaluation.
2. Lập danh sách THESIS online active path: A0/A1/A2, projector-only update và metadata checkpoint.
3. Lập danh sách baseline path được v2 yêu cầu giữ lại.
4. Đánh dấu `classify_legacy_baseline_window` là baseline-native path, không phải deletion candidate của THESIS runtime.
5. Đánh dấu AnomalyArchive và `scripts/ops/*` là scope decision, không phải automatic deletion.
6. Đối chiếu mọi candidate deletion với active entrypoint và benchmark config.

**Verification**

- Không có baseline path nằm trong deletion list mặc định.
- THESIS active path được mô tả bằng một flow duy nhất.

**Completion condition**

Có deletion boundary được chấp nhận cho các phase 1-6 và decision boundary riêng cho phase 7.

### Stage 0.4 — Establish baseline verification

**Boundary**

- Focused tests dưới `tests/models/`, `tests/core/`, `tests/online/`, `tests/runtime/`, `tests/benchmarks/`.
- One concrete SMD smoke combination theo repository guidance.

**Atomic steps**

1. Chọn test files bảo vệ phase lifecycle, config, model registry, online contract và checkpoint.
2. Chạy từng test group riêng để biết failure location.
3. Ghi pass/fail/skip và nguyên nhân quan sát được.
4. Ghi rõ các failure baseline đã tồn tại từ trước.
5. Chọn một smoke command đã có trong repository cho active two-stage hoặc benchmark path; không tự tạo command mới chỉ để phục vụ plan.
6. Lưu expected output shape, trainable parameter set và checkpoint metadata của smoke baseline.

**Verification**

- Có baseline test result trước refactor.
- Có expected active contract để so sánh sau mỗi phase.

**Completion condition**

Phase 1 có thể bắt đầu mà không phải đoán regression baseline.

## 3. Phase 1 — Simplify THESIS runtime lifecycle

### Stage 1.1 — Freeze the active two-stage contract

**Files and symbols**

- `src/models/thesis_multitask_impl/thesis_multitask_components.py`: phase constants, `ThreeStageRuntimeConfig`, `ThesisMultitaskModelConfig`.
- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`: phase helper methods.
- `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py`: state schedule helpers.
- `src/engine/trainer.py`: stage dispatch and training lifecycle caller.

**Current responsibility**

Model runtime vừa hỗ trợ active two-stage names vừa giữ phase names của three-stage history. Mixins quyết định objective weights, freeze state và trainable set theo nhiều phase.

**Target behavior**

Chỉ giữ ba active training states: Stage A multitask pretraining, memory initialization và Stage B fusion fine-tuning. Public output, checkpoint payload active và Stage B freeze semantics không thay đổi.

**Atomic steps**

1. Đọc các active two-stage config để ghi canonical phase names và fields đang được caller dùng.
2. Đối chiếu canonical names với `ThesisMultitaskModelConfig` và trainer stage dispatch.
3. Liệt kê mỗi phase helper hiện tại và đánh dấu nhánh giữ lại hoặc nhánh legacy.
4. Ghi invariant cho Stage A objective, memory initialization, Stage B objective, encoder freeze và memory freeze.
5. Ghi invariant cho output fields và checkpoint state fields mà active evaluator/online loader đọc.
6. Không thay đổi implementation trong stage này; chỉ tạo acceptance checklist cho các stage xoá tiếp theo.

**Verification**

- Checklist có mapping một-một từ active config phase đến runtime state.
- Không có active config cần phase legacy để chạy.

**Completion condition**

Các stage sau có một contract rõ ràng để thu hẹp runtime mà không đổi experiment semantics.

### Stage 1.2 — Remove legacy objective branches

**Files and symbols**

- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`: `_phase_uses_prototype_path`, `_phase_uses_contrastive_objective`, `_phase_reconstruction_weight`, `_phase_classification_weight`, `_phase_contrastive_weight`.
- `src/models/thesis_multitask_impl/thesis_multitask_components.py`: phase runtime configuration.

**Current responsibility**

Objective helper methods có các nhánh cho `stage1_classification`, `stage1_reconstruction`, `stage2_recovery`, `multitask_pretraining`, Stage 3 và Stage B.

**Target behavior**

Objective logic chỉ nhận active stage state. Stage A giữ multitask objective theo O0/O1; Stage B giữ fusion/prediction-head objective. Memory initialization không bị biến thành một training objective mới.

**Atomic steps**

1. Tạo test expectation cho từng active objective branch trước khi xoá branch cũ.
2. Xoá một nhánh legacy phase khỏi helper tương ứng.
3. Chạy test helper/objective ngay sau thay đổi đó.
4. Lặp lại cho `stage1_classification` và `stage1_reconstruction`.
5. Lặp lại cho `stage2_recovery` và `multitask_pretraining`.
6. Lặp lại cho Stage 3 objective branch.
7. Thu hẹp error/phase handling để phase không hợp lệ fail rõ ràng.
8. Kiểm tra O0/O1 không bị đổi loss weights ngoài phạm vi legacy branch.

**Verification**

- Active Stage A/B objective tests pass.
- Legacy phase input bị reject hoặc không còn entrypoint để gọi.
- O0/O1 smoke giữ đúng loss components.

**Completion condition**

Không còn objective helper branch dành cho phase legacy.

**Rollback**

Nếu active checkpoint hoặc config cần một branch bị đánh dấu legacy, revert riêng atomic step đó và chuyển dependency về Phase 0/2 migration list.

### Stage 1.3 — Remove legacy trainable-state branches

**Files and symbols**

- `src/models/thesis_multitask_impl/thesis_multitask_setup_mixin.py`: `_phase_freezes_encoder`, `_configure_trainable_parameters_for_phase`.
- `src/models/thesis_multitask_impl/thesis_multitask_state_schedule_mixin.py`: `_memory_initialization_substep_active`, `_fusion_warmup_substep_active` và trainable-state helpers.

**Current responsibility**

Code còn điều khiển encoder/memory/fusion theo Stage 1, Stage 2, Stage 3 và Stage B.

**Target behavior**

Trainable surface chỉ chuyển theo active lifecycle: Stage A trainable set → memory initialization state → Stage B trainable fusion/prediction heads.

**Atomic steps**

1. Chụp expected `requires_grad` state cho Stage A và Stage B bằng test fixture hiện có.
2. Chụp expected memory initialization state và freeze boundary.
3. Xoá Stage 1/Stage 2 trainable branches khỏi `_configure_trainable_parameters_for_phase`.
4. Chạy trainable-surface tests.
5. Xoá Stage 3-specific encoder freeze branch.
6. Chạy freeze tests và kiểm tra Stage B vẫn freeze encoder/memory.
7. Thu hẹp state schedule về memory initialization và Stage B fusion warm-up nếu active config còn dùng warm-up đó.
8. Bỏ state transition không còn caller.

**Verification**

- Stage A, memory initialization và Stage B có `requires_grad` state đúng.
- Không có state transition vào Stage 1/2/3 legacy.

**Completion condition**

Runtime trainable-state graph chỉ còn active two-stage graph.

### Stage 1.4 — Reconnect active model callers

**Files and symbols**

- `src/engine/trainer.py`: stage dispatch, scheduler/lifecycle caller.
- `src/engine/evaluator.py`: active model output/scoring consumer.
- `src/models/thesis_multitask.py` và `src/models/thesis_multitask_impl/*`: public model surface.
- Active two-stage scripts/configs.

**Current responsibility**

Trainer/evaluator/model facade có thể đi qua compatibility surface được xây cho nhiều lifecycle.

**Target behavior**

Active entrypoints chỉ gọi two-stage runtime và giữ public model facade.

**Atomic steps**

1. Tìm từng caller truyền phase/runtime config vào model.
2. Cập nhật caller đầu tiên sang canonical two-stage state nếu caller còn dùng phase alias.
3. Chạy focused caller test.
4. Lặp lại cho trainer, evaluator và active script entrypoints.
5. Kiểm tra public facade vẫn import implementation đúng module.
6. Không đổi baseline trainer/evaluator path nếu nó không gọi THESIS legacy branch.

**Verification**

- Active two-stage script constructs one canonical runtime.
- Evaluator nhận output contract cũ.
- Baseline entrypoints không bị import hoặc triage theo THESIS.

**Completion condition**

Không còn active caller nào phụ thuộc phase legacy.

### Stage 1.5 — Verify two-stage vertical slice

**Files and tests**

- `tests/models/test_multitask_config_refactor.py`.
- Các test multitask shape/train-step/checkpoint hiện có.
- Active two-stage SMD config và benchmark smoke entrypoint.

**Atomic steps**

1. Chạy config validation cho active Stage A/B config.
2. Chạy model shape test với input `[B,20,D]`.
3. Chạy một forward pass và một backward pass cho Stage A.
4. Chạy memory initialization từ train-only fixture.
5. Chạy một Stage B step và kiểm tra encoder/memory freeze.
6. Lưu và load active checkpoint.
7. Chạy một end-to-end smoke từ training đến checkpoint.
8. So sánh output fields, shape và trainable set với Phase 0 baseline.

**Verification**

- Focused tests pass.
- Smoke flow pass.
- Không có legacy phase marker trong active runtime trace.

**Completion condition**

Phase 1 tạo được active two-stage vertical slice có thể bàn giao cho Phase 2.

## 4. Phase 2 — Simplify configuration lifecycle

### Stage 2.1 — Migrate active configs

**Files and symbols**

- `configs/experiment/**` active v2/v3 configs.
- `configs/model/thesis_multitask_two_stage_*.yaml`.
- `src/core/config.py` canonical two-stage loading.
- `src/core/config_model_validation.py` active model validation.

**Current responsibility**

Config layer chứa active two-stage config và historical three-stage schema/aliases.

**Target behavior**

Mọi active experiment dùng canonical two-stage keys, `stage_b_fusion_finetuning`, `L=20` và online metadata đầy đủ.

**Atomic steps**

1. Liệt kê active v2/v3 config files được benchmark launcher đọc.
2. Kiểm tra từng config có `two_stage` và không có `three_stage`.
3. Kiểm tra từng online config có offline variant, entity, seed, benchmark mode và stage name.
4. Kiểm tra active online config dùng `projector_params`.
5. Sửa từng active config còn dùng alias, mỗi config một bước.
6. Chạy config validation sau từng nhóm config.
7. Ghi lại config historical chưa migrate để Phase 2.4 xử lý.

**Verification**

- Active configs load/validate.
- Không có active config còn chứa three-stage keys hoặc online encoder target.

**Completion condition**

Config set active là input duy nhất cần bảo vệ trong các stage xoá compatibility.

### Stage 2.2 — Remove three-stage validation path

**Files and symbols**

- `src/core/config.py`: `_normalize_three_stage_config_keys`, `_validate_three_stage_config`.
- `src/core/config_experiment_validation.py`: three-stage validation dispatch.
- `src/core/config_model_validation.py`: three-stage allowed keys/validation entries.

**Current responsibility**

Validator vẫn nhận và normalize three-stage config dù active rerun dùng two-stage.

**Target behavior**

Runtime chỉ validate two-stage schema; three-stage config fail ở boundary config với lỗi rõ ràng.

**Atomic steps**

1. Thêm hoặc cập nhật test chứng minh active two-stage validation.
2. Xoá dispatch từ experiment validation vào three-stage validator.
3. Chạy config tests.
4. Xoá `_normalize_three_stage_config_keys` sau khi không còn caller.
5. Xoá `_validate_three_stage_config` sau khi không còn caller.
6. Xoá allowed keys và constants chỉ phục vụ three-stage.
7. Thêm test cho three-stage config bị reject ở public loader.
8. Chạy toàn bộ config test group.

**Verification**

- Active two-stage config pass.
- Three-stage config fail trước khi tạo model/trainer.
- Error không làm lộ một fallback path khác.

**Completion condition**

Không còn three-stage validator reachable từ runtime.

### Stage 2.3 — Remove historical aliases

**Files and symbols**

- `src/core/config.py`: phase metadata alias normalization và legacy key constants.
- `src/core/config_model_validation.py`: alias-dependent validation.
- Tests cho alias conflict và compatibility loading.

**Current responsibility**

Config loader chấp nhận nhiều tên lịch sử cho phase, global epoch và variance/config fields.

**Target behavior**

Chỉ canonical key được chấp nhận trong active scope; key lịch sử bị reject thay vì silently normalize.

**Atomic steps**

1. Lập danh sách alias còn được active config sử dụng sau Stage 2.1.
2. Xoá một alias normalization rule không còn caller.
3. Chạy test canonical config.
4. Lặp lại cho phase aliases và global epoch aliases.
5. Giữ lại alias chỉ khi caller evidence chứng minh active v2/v3 cần nó; ghi rõ exception.
6. Thay alias-conflict test bằng unsupported-key rejection test nếu compatibility đã kết thúc.
7. Chạy config validation và snapshot tests.

**Verification**

- Canonical keys load bình thường.
- Historical aliases fail rõ ràng hoặc đã được tách khỏi runtime theo migration policy.

**Completion condition**

Config normalization không còn là processing path cho legacy experiment.

### Stage 2.4 — Remove or archive three-stage config files

**Files**

- `configs/model/thesis_multitask_three_stage_window20.yaml`.
- `configs/model/thesis_multitask_three_stage_comparative_smd.yaml`.
- Docs/tests/config snapshots tham chiếu các file trên.

**Current responsibility**

Các file này giữ historical three-stage model config và có thể bị runtime discovery hoặc test fixture đọc.

**Target behavior**

Three-stage config không còn là active runtime input. Nếu cần tái lập lịch sử, file phải được đánh dấu archival và không được active loader tự chọn.

**Atomic steps**

1. Tìm toàn repository mọi reference đến hai config.
2. Phân loại reference thành active, test-only, documentation hoặc archival.
3. Nếu G2 chưa chốt, dừng xoá và chỉ đánh dấu migration candidate.
4. Nếu G2 cho phép xoá, cập nhật từng active/test reference sang two-stage config.
5. Nếu cần giữ lịch sử, chuyển file vào archival location theo convention đã có; không tạo runtime alias.
6. Chạy config discovery/validation tests.

**Verification**

- Runtime không discover three-stage config.
- Active benchmark config vẫn đủ.
- Archival file, nếu giữ, không được active launcher sử dụng.

**Completion condition**

Không còn three-stage config trong active config surface.

**Stop/rollback**

Nếu checkpoint reproducibility cần file cũ, dừng tại step 3 và mở migration decision thay vì xoá.

### Stage 2.5 — Verify config rejection and active loading

**Files and tests**

- `tests/core/` config tests.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`.
- Active config files.

**Atomic steps**

1. Chạy active config load tests.
2. Chạy rejection tests cho three-stage schema và aliases.
3. Chạy online benchmark wrapper tests.
4. Kiểm tra generated online config giữ Stage B metadata và projector target.
5. Chạy một config-backed smoke trước khi sang Phase 3.

**Verification**

- Active config pass.
- Unsupported legacy config fail early.
- Generated benchmark config không bị schema drift.

**Completion condition**

Phase 2 tạo ra một canonical config lifecycle.

## 5. Phase 3 — Remove non-main reconstruction model path

### Stage 3.1 — Confirm no active experiment dependency

**Files and symbols**

- `src/models/reconstruction_mlp_ae.py`.
- `src/core/runtime_components.py` registration.
- `src/core/config_model_validation.py` model validation.
- `configs/model/reconstruction_mlp_ae.yaml`.
- Benchmark scripts, tests và compliance fixtures.

**Current responsibility**

Standalone reconstruction model vẫn được registry, validator, config và nhiều tests dùng, dù không nằm trong main v2/v3 matrix.

**Target behavior**

Chỉ tiếp tục model này nếu caller evidence chứng minh nó cần cho active scope hoặc smoke policy.

**Atomic steps**

1. Tìm mọi `model_name: reconstruction_mlp_ae` trong configs, scripts và tests.
2. Tìm import trực tiếp `ReconstructionMLPAutoencoder`.
3. Tách callers thành active benchmark, generic test, online reference fixture và compliance snapshot.
4. Kiểm tra benchmark launcher có sinh model này trong runtime hay chỉ trong fixture.
5. Kiểm tra demo và checkpoint reader có yêu cầu class này không.
6. Ghi quyết định G3 và danh sách caller cần migrate.

**Verification**

- Không còn caller chưa phân loại.
- Không xoá model nếu active caller chưa có migration path.

**Completion condition**

Có quyết định rõ ràng: remove hoàn toàn hoặc giữ model ngoài deletion scope.

### Stage 3.2 — Migrate test-only dependencies

**Files and tests**

- `tests/models/test_model_shapes.py`.
- `tests/models/test_one_train_step.py`.
- `tests/runtime/test_checkpoint_roundtrip.py`.
- `tests/online/test_online_reference_checkpoint.py`.
- `tests/benchmarks/test_multiseed_launcher.py`.
- `tests/compliance/test_src_refactor_contracts.py` và fixture.

**Current responsibility**

Một số tests dùng reconstruction model để test shape, checkpoint hoặc registry thay vì test behavior đặc thù của model.

**Target behavior**

Generic tests kiểm tra contract active; tests đặc thù chỉ giữ nếu model được scope duy trì.

**Atomic steps**

1. Gắn mục tiêu behavior cho từng test dùng reconstruction model.
2. Chuyển test generic shape/step sang active THESIS model nếu contract tương đương.
3. Chuyển checkpoint container test sang checkpoint type active nếu không cần reconstruction architecture.
4. Cập nhật launcher fixture để sinh active model config.
5. Cập nhật compliance fixture và registry expectation.
6. Chạy từng test file sau mỗi nhóm migration.

**Verification**

- Tests vẫn kiểm tra đúng behavior ban đầu, không chỉ đổi import để làm pass.
- Không có fixture ẩn yêu cầu model cũ.

**Completion condition**

Không còn test-only dependency bắt buộc vào reconstruction model.

### Stage 3.3 — Remove model/config registration

**Files and symbols**

- `src/models/reconstruction_mlp_ae.py`.
- `configs/model/reconstruction_mlp_ae.yaml`.
- `src/core/runtime_components.py`: `register_model("reconstruction_mlp_ae", ...)`.
- `src/core/config_model_validation.py`: allowed keys/validation branches.

**Current responsibility**

Registry và validator làm standalone reconstruction model trở thành một public runtime option.

**Target behavior**

Model không còn xuất hiện trong active registry/config surface sau khi Stage 3.1-3.2 hoàn tất.

**Atomic steps**

1. Xác nhận Stage 3.2 tests đã migrate.
2. Xoá registration khỏi runtime component setup.
3. Chạy registry tests.
4. Xoá model-specific validator branch.
5. Chạy config validation tests.
6. Xoá model config file.
7. Xoá source model file chỉ khi không còn import.
8. Chạy import/search check cho class và model name.

**Verification**

- Registry không nhận model name cũ.
- Active model registry vẫn build THESIS/RedLamp/baselines cần thiết.
- Import/search không còn production reference.

**Completion condition**

Standalone model path bị loại bỏ hoàn toàn hoặc được chứng minh là nằm ngoài scope nếu G3 chọn giữ.

### Stage 3.4 — Update compliance and snapshots

**Files and tests**

- `tests/models/test_registry.py`.
- `tests/compliance/fixtures/src_refactor_contracts.json`.
- Các config snapshot tests được Stage 0.2 phát hiện.

**Atomic steps**

1. Xoá model cũ khỏi expected registry snapshot.
2. Cập nhật expected model keys theo active scope.
3. Chạy compliance tests.
4. Đọc failure để phát hiện caller chưa migrate.
5. Migrate caller còn sót hoặc dừng phase nếu đó là active dependency.

**Verification**

- Snapshot phản ánh active registry, không chỉ được sửa để bỏ failure.

**Completion condition**

Compliance surface đồng nhất với runtime surface.

### Stage 3.5 — Verify active model registry

**Atomic steps**

1. Chạy registry tests.
2. Chạy active THESIS model shape tests.
3. Chạy one train-step tests.
4. Chạy checkpoint round-trip tests.
5. Chạy benchmark launcher/config generation test.
6. Chạy một active two-stage smoke.
7. So sánh output contract và artifact layout với Phase 0 baseline.

**Completion condition**

Model removal không ảnh hưởng active model flow hoặc checkpoint contract.

## 6. Phase 4 — Narrow online trainable surface

### Stage 4.1 — Confirm projector-only invariant

**Files and symbols**

- `src/models/online_impl/online_adaptation.py`: `_parameters_for_target_group`, `_set_trainable_parameter_group`.
- `src/engine/online_tta/online_engine_shared.py`: optimizer construction and target validation.
- `src/core/config_model_validation.py`: target group validation.
- Online experiment configs and trainable-surface tests.

**Current responsibility**

Model surface accepts both `projector_params` and `online_encoder_params`; engine active path rejects the latter.

**Target behavior**

Một target group duy nhất được biểu diễn ở config, model và optimizer.

**Atomic steps**

1. Tìm tất cả active config chứa `target_param_group`.
2. Tìm direct caller gọi `get_parameter_group`.
3. Kiểm tra mọi active caller dùng `projector_params`.
4. Ghi expected parameters thay đổi ở A1/A2 và parameters phải freeze.
5. Ghi expected A0 no-update behavior.

**Verification**

- Không có active direct caller cần online encoder target.

**Completion condition**

Removal không làm mất một active experiment required by v2/v3.

### Stage 4.2 — Remove online encoder target branch

**Files and symbols**

- `src/models/online_impl/online_adaptation.py`: `_parameters_for_target_group` và error message.

**Target behavior**

Method chỉ trả projector parameters; mọi target khác fail rõ ràng.

**Atomic steps**

1. Cập nhật test projector group trước khi xoá branch.
2. Xoá nhánh `online_encoder_params`.
3. Thu hẹp error message về `projector_params`.
4. Chạy model target-group tests.
5. Kiểm tra không còn caller tham chiếu error text cũ.

**Verification**

- `projector_params` trả đúng projector parameters.
- `online_encoder_params` bị reject.

**Completion condition**

Model không còn public path để chọn online encoder parameters.

### Stage 4.3 — Narrow config validation

**Files and symbols**

- `src/core/config_model_validation.py`: online target validation.
- `src/engine/online_tta/online_engine_shared.py`: optimizer guard.

**Atomic steps**

1. Đổi allowed-value validation thành chỉ `projector_params`.
2. Giữ engine guard như defense-in-depth.
3. Cập nhật invalid-config test cho `online_encoder_params`.
4. Chạy config và online engine tests.
5. Kiểm tra active generated configs không bị đổi ngoài target key.

**Verification**

- Config fail trước optimizer construction khi target không hợp lệ.
- Engine vẫn reject nếu được gọi trực tiếp với target sai.

**Completion condition**

Config và engine có cùng online trainable contract.

### Stage 4.4 — Verify A0/A1/A2 trainable behavior

**Files and tests**

- `tests/online/test_online_tta_trainable_surface.py`.
- `tests/online/test_online_tta_variants.py`.
- `tests/online/test_online_adaptation_step.py`.
- `tests/online/test_online_state_roundtrip.py`.

**Atomic steps**

1. Chạy A0 test và assert không có projector update.
2. Chạy A1 test và assert projector-only update.
3. Chạy A2 test và assert projector-only update.
4. Assert reference encoder và online encoder không đổi.
5. Chạy projector anchor/reset tests.
6. Chạy online state round-trip.
7. Chạy một online smoke với active `L=20` config.

**Completion condition**

A0/A1/A2 semantics và state persistence không đổi sau khi thu hẹp target surface.

## 7. Phase 5 — Simplify online stream and batch contract

### Stage 5.1 — Confirm active stream policies

**Files and symbols**

- `src/data/stream.py`: `SMDOnlineStream.__init__` và index-record construction.
- `src/engine/online_tta/online_calibration.py`.
- `src/baselines/online/base.py`.
- Online configs and stream tests.

**Current responsibility**

Stream hỗ trợ `legacy_stride`, `sliding_stride_1` và `nonoverlap_tail`; active code evidence cho thấy hai policy mới được dùng.

**Atomic steps**

1. Tìm mọi explicit `stream_window_mode` caller.
2. Tách THESIS online caller và baseline caller.
3. Kiểm tra spec yêu cầu sliding, non-overlap hoặc cả hai ở từng flow.
4. Ghi policy cần giữ cho từng caller.
5. Nếu `legacy_stride` còn active caller, chuyển caller đó sang policy canonical trước khi xoá mode.

**Verification**

- Mọi caller có policy canonical hoặc được ghi là out of scope.

**Completion condition**

Có thể xoá `legacy_stride` mà không đổi cửa sổ active ngoài ý muốn.

### Stage 5.2 — Remove `legacy_stride` mode

**Files and symbols**

- `src/data/stream.py`: accepted modes, validation và start-index branch.
- Callers/configs được Stage 5.1 phát hiện.

**Atomic steps**

1. Cập nhật stream tests cho hai policy được giữ lại.
2. Xoá `legacy_stride` khỏi accepted values.
3. Xoá start-index branch chỉ phục vụ mode đó.
4. Thu hẹp validation error message.
5. Chạy stream index tests.
6. Chạy THESIS online calibration/engine smoke.

**Verification**

- Canonical policies tạo đúng index records.
- `legacy_stride` bị reject.

**Completion condition**

Stream không còn một policy xử lý historical stride.

### Stage 5.3 — Remove legacy two-view validator path

**Files and symbols**

- `src/core/contracts.py`: `validate_legacy_two_view_batch`.
- `src/core/console.py`: optional `view_a`/`view_b` instrumentation.
- Direct callers found by Stage 0.2.

**Current responsibility**

Contracts và console vẫn biết batch có `view_a`/`view_b`, dù active online batch contract không có hai field này.

**Target behavior**

Online stream batch không chứa hai field này; nếu adaptation cần augmentation, augmentation được tạo tại adaptation-step boundary.

**Atomic steps**

1. Tìm direct caller của `validate_legacy_two_view_batch`.
2. Nếu caller thuộc active adaptation-step construction, chuyển augmentation tạo view vào đúng adaptation step và giữ stream batch sạch.
3. Chạy focused online contract tests.
4. Xoá validator khi không còn caller.
5. Xoá console summary branch khi không còn production batch caller.
6. Chạy core contract và console tests.

**Verification**

- Active online batch không có `view_a`/`view_b`.
- Augmentation behavior cần cho A1/A2 vẫn chạy bên trong adaptation step.

**Completion condition**

Legacy two-view validator không còn reachable từ production online path.

### Stage 5.4 — Migrate online test fixtures

**Files and tests**

- `tests/online/test_online_stream.py`.
- `tests/online/test_online_tta_variants.py`.
- `tests/online/test_online_adaptation_step.py`.
- `tests/online/test_online_engine_max_steps.py`.
- `tests/online/test_full_spec_online_contract.py`.

**Atomic steps**

1. Sửa stream fixture assertion để kiểm tra `x` shape `[B,20,D]` và không kiểm tra `view_a/view_b`.
2. Sửa adaptation-step fixture để tạo view nội bộ chỉ tại boundary yêu cầu augmentation.
3. Sửa max-step fixture thành active online batch.
4. Giữ full-spec contract assertions rằng labels và legacy views không vào scorer.
5. Chạy từng test file sau mỗi migration group.

**Verification**

- Tests kiểm tra behavior active thay vì chỉ xoá assertions cũ.

**Completion condition**

Test suite không còn củng cố legacy two-view stream contract.

### Stage 5.5 — Verify stream and no-label boundary

**Atomic steps**

1. Chạy stream shape/index tests.
2. Chạy no-label online contract tests.
3. Chạy online triage contract tests.
4. Chạy state round-trip với stream cursor.
5. Chạy một sliding online smoke.
6. Chạy non-overlap tail smoke nếu policy đó thuộc active v2 flow.
7. Kiểm tra artifact metadata ghi đúng stream policy và window size.

**Completion condition**

Online stream và adaptation boundary dùng một contract active thống nhất.

## 8. Phase 6 — Simplify checkpoint resolution

### Stage 6.1 — Confirm metadata completeness

**Files and symbols**

- `src/engine/online_tta/checkpoint_resolution.py`: `ONLINE_BENCHMARK_METADATA_FIELDS`, `_resolve_checkpoint_from_metadata`.
- `src/core/config_model_validation.py`: online task validation.
- `configs/experiment/online_benchmark/thesis/*`.

**Atomic steps**

1. Kiểm tra active online configs có đủ `offline_variant`, `entity_id`, `seed`, `benchmark_mode`, `stage_name`.
2. Kiểm tra benchmark mode chỉ thuộc active values `main` hoặc `smoke`.
3. Kiểm tra stage name map tới `stage_b_fusion_finetuning` artifact.
4. Ghi config nào chỉ có flat `reference_checkpoint_path`.
5. Migrate config thiếu metadata trước khi xoá fallback.

**Verification**

- Metadata resolver tìm đúng một Stage B checkpoint cho mỗi active fixture.

**Completion condition**

Active online benchmark không cần legacy flat layout để resolve checkpoint.

### Stage 6.2 — Migrate callers to metadata resolution

**Files and symbols**

- `src/engine/online_tta/checkpoint_resolution.py`: `resolve_stage_b_checkpoint`.
- `scripts/benchmarks/run_thesis_online_benchmark.py`.
- `src/engine/online_tta/online_engine_run.py`.
- `scripts/experiments/run_online_adaptation.py`.
- Online wrapper tests.

**Atomic steps**

1. Tìm caller truyền `reference_checkpoint_path`.
2. Cập nhật benchmark wrapper để truyền metadata đầy đủ.
3. Cập nhật experiment wrapper chỉ dùng canonical resolved path sau metadata resolution.
4. Chạy wrapper tests.
5. Kiểm tra provenance vẫn ghi checkpoint path và SHA256 khi file tồn tại.
6. Ghi caller nào còn cần direct path cho local unit fixture.

**Verification**

- Active benchmark path resolve bằng metadata.
- Local direct-file fixture không bị nhầm với flat legacy fallback.

**Completion condition**

Không còn active benchmark caller phụ thuộc path guessing.

### Stage 6.3 — Remove flat checkpoint fallback

**Files and symbols**

- `src/engine/online_tta/checkpoint_resolution.py`: `resolve_legacy_reference_checkpoint_path` và fallback branch trong `resolve_stage_b_checkpoint`.
- `src/models/online_impl/online_adaptation.py`: `_resolve_reference_checkpoint_path`.
- Tests reference checkpoint.

**Atomic steps**

1. Xác nhận Stage 6.2 active caller migration pass.
2. Thay fallback branch bằng canonical metadata/direct canonical path handling theo active interface.
3. Xoá `resolve_legacy_reference_checkpoint_path` khi không còn caller.
4. Xoá wrapper fallback trong online model implementation.
5. Cập nhật error message để chỉ dẫn metadata/canonical Stage B checkpoint.
6. Chạy reference-checkpoint tests.

**Verification**

- Canonical checkpoint vẫn resolve.
- Flat legacy path không được tự động remap.
- Missing checkpoint fail rõ ràng.

**Completion condition**

Checkpoint resolver chỉ còn một canonical path.

**Stop/rollback**

Nếu active reproducibility test cần fallback, dừng xoá và mở migration policy cho checkpoint cũ.

### Stage 6.4 — Verify failure behavior and provenance

**Files and tests**

- `tests/online/test_online_reference_checkpoint.py`.
- `tests/benchmarks/test_thesis_online_benchmark_wrapper.py`.
- `src/engine/online_tta/online_engine_run.py` provenance fields.

**Atomic steps**

1. Test missing required metadata.
2. Test unknown benchmark mode.
3. Test no matching Stage B checkpoint.
4. Test ambiguous Stage B candidates.
5. Test valid Stage B resolution.
6. Test provenance stores benchmark mode, entity, seed, stage and checksum.
7. Chạy online smoke với một active metadata config.

**Completion condition**

Checkpoint failures xảy ra sớm, deterministic và không làm mất provenance.

## 9. Phase 7 — Decide and isolate non-main paths

### Stage 7.1 — Decide dataset scope

**Files and symbols**

- `src/core/runtime_components.py`: dataset registration.
- `src/core/config.py`: accepted dataset names and validation.
- `src/data/api.py`.
- `src/data/datasets/anomaly_archive.py`.
- AnomalyArchive configs/tests/analysis scripts.

**Current responsibility**

Runtime registry và config layer vẫn hỗ trợ `anomaly_archive`; nhiều analysis/evaluation tests cũng dùng dataset này.

**Target behavior**

AnomalyArchive được phân loại chính thức là active dataset, archival dataset hoặc out of project scope trước khi xoá.

**Atomic steps**

1. Tìm mọi AnomalyArchive config, runtime caller, analysis caller và test.
2. Đối chiếu với scope dataset trong v2/v3 và thesis design.
3. Kiểm tra có report/artifact nào cần giữ parser để audit không.
4. Ghi decision G4 trong decision record hoặc research note update.
5. Không sửa registry trước khi decision được ghi.

**Verification**

- Có một scope decision rõ ràng và danh sách callers bị ảnh hưởng.

**Completion condition**

Stage 7.2 biết chính xác có được gỡ AnomalyArchive khỏi runtime hay không.

### Stage 7.2 — Isolate non-main dataset processing

**Files and symbols**

- `src/core/runtime_components.py` dataset registration.
- `src/core/config.py` dataset validation.
- `src/data/api.py` và `src/data/datasets/anomaly_archive.py`.
- AnomalyArchive-specific analysis/visualization scripts and tests.

**Atomic steps**

1. Nếu G4 giữ active, không xoá runtime path; chỉ ghi rõ nó không thuộc THESIS main matrix.
2. Nếu G4 chọn archival, ngăn active benchmark launcher tự chọn dataset này.
3. Chuyển hoặc đánh dấu parser/analysis theo archival convention hiện có.
4. Nếu G4 chọn out of scope, migrate/remove registry and validation references từng bước.
5. Chạy active SMD config tests.
6. Chạy archival tests riêng nếu parser được giữ.

**Verification**

- SMD active flow không bị ảnh hưởng.
- Archival evidence còn truy cập được nếu decision yêu cầu giữ.

**Completion condition**

AnomalyArchive không còn mơ hồ giữa active benchmark và archival processing.

### Stage 7.3 — Review operational tooling boundary

**Files and symbols**

- `scripts/ops/*` gồm prune, backfill và re-evaluate tools.
- Artifact/report/provenance documentation.

**Current responsibility**

Operational scripts không phải scientific processing flow, nhưng có thể cần để giữ provenance, report và artifact retention.

**Target behavior**

Tooling được phân loại riêng; không bị xoá như legacy model/data path.

**Atomic steps**

1. Liệt kê từng operational script và input/output artifact của nó.
2. Tìm caller từ benchmark/report workflows.
3. Đánh dấu script active operational, historical hoặc unused.
4. Kiểm tra retention policy yêu cầu artifact nào.
5. Chỉ mở deletion task riêng cho script được chứng minh unused và không ảnh hưởng provenance.

**Verification**

- Main benchmark report và provenance workflow vẫn có đường chạy được.

**Completion condition**

Operational tooling nằm ngoài deletion batch mặc định hoặc có plan riêng được phê duyệt.

### Stage 7.4 — Update documentation and tests

**Files and documents**

- `documents/spec/full-spec-v2.md` và `documents/spec/full-spec-v3.md` chỉ sửa nếu có decision/documentation drift đã được xác nhận.
- Research, plan, structure và detail logs.
- Registry/config/evaluation tests liên quan.

**Atomic steps**

1. Cập nhật active flow map theo processing paths sau cleanup.
2. Ghi rõ baseline retained, archival dataset và operational tooling scope.
3. Cập nhật tests để phản ánh active versus archival behavior.
4. Không thay normative spec chỉ để làm cho code trông đơn giản hơn.
5. Chạy documentation consistency checks nếu repository có test tương ứng.

**Verification**

- Người đọc có thể xác định active entrypoint, dataset và artifact flow từ `documents/`.

**Completion condition**

Documentation và test contract không còn mô tả một legacy path là active.

### Stage 7.5 — Verify final processing-flow inventory

**Files and tests**

- `src/core/runtime_components.py`.
- Active config directories.
- THESIS and baseline benchmark entrypoints.
- Focused test groups and one concrete smoke.

**Atomic steps**

1. Liệt kê dataset registry còn lại.
2. Liệt kê model registry còn lại.
3. Liệt kê active offline và online entrypoints.
4. Tìm lại marker names L01-L13 và phân loại mọi kết quả còn tồn tại.
5. Chạy config tests.
6. Chạy model/online contract tests.
7. Chạy một concrete v2/v3 smoke combination.
8. Đối chiếu artifact layout, provenance và report output với full-spec requirements.

**Verification**

- Mọi marker còn lại đều có lý do active, baseline-native, archival hoặc operational.
- Active SMD THESIS flow và required v2 baselines vẫn chạy.

**Completion condition**

Codebase có một processing-flow inventory cuối cùng, không còn legacy path chưa được phân loại.

## 10. Cross-phase verification matrix

| Boundary | Required check | Expected result |
|---|---|---|
| Config | Active two-stage load and legacy rejection | Active config pass; unsupported legacy schema fail early |
| Model | Shape and one train step | Input `[B,20,D]`; output contract unchanged |
| Lifecycle | Stage A → memory init → Stage B | No Stage 1/2/3 transition |
| Trainable state | Stage B and online projector | Encoder/memory/reference freeze preserved |
| Online batch | Stream and scorer boundary | No labels or `view_a/view_b` in scorer input |
| Online update | A0/A1/A2 | Only projector updates when adaptation is enabled |
| Checkpoint | Metadata resolution and round-trip | Deterministic Stage B identity and provenance |
| Baselines | Native v2 paths | No THESIS triage/PNN/projector inheritance |
| Scope cleanup | Registry and final inventory | Every remaining path classified |

## 11. Risk, rollback and stopping rules

### Backward compatibility

- **Cause:** old checkpoint/config may depend on three-stage names or flat layout.
- **Impact:** old experiment cannot be loaded or reproduced.
- **Mitigation:** Phase 0 caller/artifact inventory; Phase 2 and Phase 6 only remove compatibility after G2/G5 evidence.
- **Verification:** migration fixtures and explicit rejection tests.
- **Rollback:** revert only the compatibility removal step, not the whole repository.

### Baseline regression

- **Cause:** a function named `legacy` is actually native baseline logic.
- **Impact:** v2 benchmark becomes incomplete or incomparable.
- **Mitigation:** retain `classify_legacy_baseline_window` and baseline modules unless v2 scope changes.
- **Verification:** baseline smoke and protocol audit.
- **Rollback:** restore baseline path before continuing cleanup.

### Contract drift

- **Cause:** removing `view_a/view_b` or changing stream mode changes a hidden caller assumption.
- **Impact:** online adaptation step or state round-trip fails.
- **Mitigation:** migrate caller and fixture before deleting validator/mode.
- **Verification:** full-spec online contract, stream tests and online smoke.
- **Rollback:** restore the single removed branch and stop; do not reintroduce multiple new adapters.

### Registry/config drift

- **Cause:** removing a model or dataset from only one layer.
- **Impact:** import, config validation and benchmark discovery disagree.
- **Mitigation:** update source registration, validator, config, tests and snapshot in one stage.
- **Verification:** registry tests, config tests and repository-wide search.
- **Rollback:** restore registration without restoring unrelated legacy branches.

### Provenance loss

- **Cause:** checkpoint fallback or operational tooling is removed without preserving metadata.
- **Impact:** result identity and reproducibility become unclear.
- **Mitigation:** keep Stage B metadata, checksum, selected diagnostics and report artifacts.
- **Verification:** checkpoint/provenance tests and one benchmark smoke.
- **Rollback:** stop Phase 6 or Phase 7 cleanup until artifact retention is restored.

## 12. Delivery order

```text
Phase 0 / Stages 0.1 → 0.4
  → Phase 1 / Stages 1.1 → 1.5
  → Phase 2 / Stages 2.1 → 2.5
  → Phase 3 / Stages 3.1 → 3.5
  → Phase 4 / Stages 4.1 → 4.4
  → Phase 5 / Stages 5.1 → 5.5
  → Phase 6 / Stages 6.1 → 6.4
  → Phase 7 / Stages 7.1 → 7.5
```

Không bắt đầu Phase 7 chỉ vì các code paths còn lại có vẻ không chính. Phase 7 phụ thuộc vào scope decision cho AnomalyArchive, baseline và operational tooling.

## 13. Final handoff criteria

Trước khi triển khai source code, người thực hiện phải có:

1. Structure order được chấp nhận.
2. G2/G3/G4/G5 được quyết định tại các stage tương ứng hoặc được ghi rõ là chưa đủ điều kiện xoá.
3. Baseline test result và active smoke baseline từ Phase 0.
4. Một change set nhỏ cho mỗi stage, không gộp nhiều phase.
5. Test result sau từng atomic step.
6. Diff review chứng minh không xoá baseline, output contract hoặc provenance ngoài scope.

