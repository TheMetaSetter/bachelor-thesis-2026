---
title: "Nghiên cứu lại ablation direct branch routing của THESIS"
date: 2026-08-30
type: research
status: completed_with_cloud_check_blocked
scope: local_codebase_and_read_only_cloud_probe
stage_a_executed: false
---

# Tóm tắt

Mục tiêu là chuẩn bị thí nghiệm bỏ hai fusion head. Discrete branch sẽ đi thẳng vào classification head. Continuous branch sẽ đi thẳng vào reconstruction head. Các phần còn lại giữ nguyên.

Code hiện tại chưa có `direct_branch_routing` như một `fusion_mode` hợp lệ. Runtime hiện vẫn tạo hai nhánh, chạy fusion, rồi đưa hai hidden đã fusion vào hai head. Runner two-stage cũng luôn chạy lệnh Stage A trước Stage B khi không ở chế độ dry-run.

Em đã không chạy Stage A, không chạy benchmark và không sửa code hoặc cấu hình. Kiểm tra cloud GPU chỉ đọc đã thử nhưng máy chủ đóng kết nối ngay sau khi mở cổng, nên trạng thái checkpoint hiện tại trên cloud chưa được xác minh.

# Câu hỏi nghiên cứu

1. Runtime hiện tạo và sử dụng hai fusion head ở đâu?
2. Những file code và cấu hình nào liên quan đến ablation direct branch routing?
3. Có thể bắt đầu thí nghiệm từ checkpoint Stage A/Stage B đã có mà không chạy lại Stage A không?

# Bối cảnh hệ thống

Model chính là `ThesisMultitaskModel`. Encoder tạo hidden chung \(H\). Ở Stage B, continuous branch tạo `prototype_context` và discrete branch tạo `quantized_hidden`. Hàm fusion hiện tạo hai tensor đầu ra:

\[
H_{\mathrm{recon}} = F_{\mathrm{recon}}(H_{\mathrm{cont}}, H_{\mathrm{disc}}),
\qquad
H_{\mathrm{cls}} = F_{\mathrm{cls}}(H_{\mathrm{cont}}, H_{\mathrm{disc}}).
\]

Reconstruction head nhận \(H_{\mathrm{recon}}\). Classification head nhận \(H_{\mathrm{cls}}\) sau khi flatten theo chiều cửa sổ.

Đề xuất thí nghiệm của anh tương ứng với routing trực tiếp:

\[
H_{\mathrm{recon}} = H_{\mathrm{cont}},
\qquad
H_{\mathrm{cls}} = H_{\mathrm{disc}}.
\]

Đây là mô tả mục tiêu thí nghiệm, chưa phải hành vi đã được code triển khai.

# Đường chạy hiện tại

Trong Stage B, `thesis_multitask_routing_forward_helpers.py` thực hiện các bước sau:

1. Tra cứu continuous prototype từ hidden chung.
2. Tra cứu discrete codebook từ hidden chung.
3. Gọi `_compute_fusion_outputs(...)`.
4. Gửi `hidden_reconstruction` vào `reconstruction_head`.
5. Flatten `hidden_classification`, rồi gửi vào `classification_head`.
6. Tính reconstruction score và classification output.

Trong Stage A, `thesis_multitask_state_passthrough_mixin.py` bypass prototype memory. Hai nhánh đều tạm nhận hidden encoder, và fusion metadata là `phase_direct_passthrough`. Vì vậy metadata này không phải `direct_branch_routing` của thí nghiệm; nó chỉ mô tả passthrough của Stage A.

# Phát hiện chi tiết

## 1. `fusion_mode` chưa nhận direct routing

`ActiveRuntimeConfig` chỉ cho phép:

- `task_specific_concat_projection`;
- `learnable_sigmoid_scalars`.

Giá trị mặc định là `learnable_sigmoid_scalars`. Khi khởi tạo model với `fusion_mode="direct_branch_routing"`, code hiện trả về `ValueError`.

## 2. Hai fusion projection vẫn được tạo

Setup mixin luôn tạo `reconstruction_concat_projection` và `classification_concat_projection`. Với `task_specific_concat_projection`, hai projection này nhận phép nối của continuous và discrete hidden. Với `learnable_sigmoid_scalars`, code dùng hai trọng số sigmoid để trộn hai nhánh:

\[
H_{\mathrm{recon}} = \beta H_{\mathrm{disc}} + (1-\beta)H_{\mathrm{cont}},
\]

\[
H_{\mathrm{cls}} = \alpha H_{\mathrm{disc}} + (1-\alpha)H_{\mathrm{cont}}.
\]

Direct routing cần một nhánh runtime khác trước khi gọi hai head. Không nên suy ra hành vi này chỉ từ tên metadata `phase_direct_passthrough`.

## 3. Checkpoint và trainability

Stage B hiện làm frozen fusion gates, alpha và beta. Hai concat projection vẫn được đặt trainable trong phần cấu hình chung của setup mixin. Khi ablation bỏ fusion projection khỏi forward, cần kiểm tra lại hợp đồng checkpoint và danh sách tham số trainable. Checkpoint cũ có thể vẫn chứa projection weights; loader Stage B hiện dùng `strict=False` nhưng chỉ cho phép một nhóm unexpected key cụ thể. Vì vậy cần giữ tương thích state-dict khi triển khai, thay vì tự ý xóa module.

## 4. Runner chưa có đường bỏ qua Stage A

`execute_two_stage_plan(...)` luôn gọi `subprocess.run(training_commands[0], check=True)` ở nhánh chạy thật. CLI hiện có `--dry-run`, `--skip-completed` và `--stop-after-stage-b-init`, nhưng chưa có tùy chọn `--skip-stage-a` hoặc `--stage-b-init-checkpoint`.

Do yêu cầu hiện tại là không chạy lại Stage A, runner chưa đáp ứng trực tiếp mục tiêu này. Có thể dùng evaluation-only nếu đã có checkpoint phù hợp, nhưng việc chọn checkpoint và tính nhất quán với manifest phải được xác minh trước.

## 5. Online path dùng lại fusion của offline model

Online adapter gọi trực tiếp `_compute_fusion_outputs` của offline model. Vì vậy nếu checkpoint direct-routing được nạp thành công, online scoring sẽ nhận hành vi direct routing từ cùng model path. Checkpoint resolution vẫn yêu cầu đúng `offline_variant`, entity, seed, benchmark mode và stage name; threshold artifact cũng phải khớp identity này.

# Cấu hình đã quan sát

| Hạng mục | Giá trị hiện tại | Nguồn |
|---|---|---|
| Git revision | `5bae88fc9aa13814633d83eaf182e7ec4aadd990` | `git rev-parse HEAD` |
| Branch | `dev` | `git branch --show-current` |
| Model config | `configs/model/thesis_multitask_two_stage_window20.yaml` | local checkout |
| Current fusion mode | `task_specific_concat_projection` | model YAML |
| Stage A / Stage B | 25 / 5 epochs trong benchmark config mẫu | `configs/experiment/offline_benchmark/thesis/` |
| Cloud endpoint | `root@159.48.242.1:20710` | `ssh-gpu.txt` |
| Cloud authentication | Có mật khẩu trong file, đã redacted trong báo cáo | `ssh-gpu.txt` |
| Local checkpoint | Không tìm thấy `stage_b_init.pt` trong output benchmark mẫu | local `outputs/` |
| Historical checkpoint paths | Có trong inventory cũ, nhưng không chứng minh trạng thái hiện tại | `documents/inventories/detail-remote-gpu-checkpoints-inventory.md` |

# Kiểm tra đã chạy

Các kiểm tra sau chỉ đọc hoặc chạy test hẹp, không gọi training:

- `.venv/bin/python -m pytest -q tests/models/test_thesis_multitask_config_refactor.py tests/benchmarks/test_two_stage_orchestration_dry_run.py` → `11 passed`.
- Test shape được chọn theo tên test → `9 passed, 1 skipped, 2 deselected`.
- Khởi tạo model với `fusion_mode="direct_branch_routing"` → `ValueError`, đúng với việc mode chưa được hỗ trợ.
- SSH read-only đến endpoint trong `ssh-gpu.txt` → lần đầu bị sandbox chặn `Operation not permitted`; lần thử có quyền mạng vẫn bị máy chủ đóng kết nối. Không có lệnh train/eval nào được gửi lên cloud.

# Kết luận và giới hạn

Codebase hiện chưa sẵn sàng chạy ablation direct branch routing bằng cách chỉ đổi YAML. Phần cần được điều chỉnh nằm ở validation của `fusion_mode`, đường tính `fusion_outputs`, việc bảo toàn checkpoint state-dict, và cách nạp checkpoint Stage B mà không gọi Stage A. Tuy nhiên, theo prompt nghiên cứu, báo cáo này chỉ xác định hiện trạng; nó chưa đề xuất patch hay thay đổi cấu hình.

Không thể xác nhận checkpoint nào còn tồn tại trên cloud vì SSH bị đóng kết nối. Inventory cũ chỉ là bằng chứng lịch sử. Trước khi chạy thí nghiệm, cần một lần đọc-only thành công để xác nhận `stage_b_init.pt` hoặc Stage B checkpoint, manifest, model config và threshold artifact cùng một run identity.
