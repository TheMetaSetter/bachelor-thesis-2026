# Real Run Argument Checklist

Mục tiêu của note này là nhắc các argument và option dễ lệch khi chạy THESIS real run, để tránh nhầm default, lệch variant, hoặc trỏ sai checkpoint.

## 1. Các argument cần kiểm tra trước khi chạy

- `--experiment-config`
  - Phải trỏ đúng file YAML cho đúng stage và đúng variant.
  - Không dùng nhầm smoke config cho real run.

- `--protocol-config`
  - Phải khớp với cửa sổ dữ liệu và protocol đã chốt trong SSOT.
  - Không để mặc định khi benchmark online.

- `--online-variant`
  - Phải truyền tường minh `A0`, `A1`, hoặc `A2`.
  - Không được để rơi về default `A0` nếu đang chạy A1/A2.

- `task_overrides.offline_variant`
  - Phải khớp với checkpoint Stage A / Stage B tương ứng, ví dụ `O0` hoặc `O1`.

- `task_overrides.entity_id`
  - Phải khớp entity thật đang benchmark, ví dụ `machine_1_6`.

- `task_overrides.seed`
  - Phải khớp seed của tổ hợp đã chọn trong development spec.

- `task_overrides.reference_checkpoint_path`
  - Khi online cần trỏ đúng checkpoint Stage B thật, không trỏ nhầm Stage A.

- `task_overrides.max_online_steps`
  - Smoke có thể dùng số nhỏ.
  - Real run thường phải để đúng semantics của full stream, không vô tình để lại cap smoke.

- `epochs`
  - Smoke và real run phải được tách rõ.
  - Không được để giá trị smoke lọt vào real run.

- `logging.use_wandb` và `logging.wandb_mode`
  - Phải khớp chính sách logging của lần chạy.
  - Nếu cần tắt W&B để smoke, phải xác nhận lại trước khi real run.

- `output_dir` và `checkpoint_dir`
  - Phải đặt tên riêng cho từng run để tránh ghi đè artifact.

## 2. Các điểm provenance cần soi

- `report["online_variant"]`
- `extra_state.online_variant`
- `threshold_artifact.variant_name`
- `extra_state.verification_metadata_source`
- `checkpoint_metadata.verification_metadata_source`

## 3. Quy tắc an toàn khi chạy

- Luôn kiểm tra default option của CLI trước khi bấm run.
- Không tin `--help` là nguồn duy nhất cho default thật.
- Chạy thử 1 tổ hợp đại diện trước khi bung toàn bộ combination.
- Giữ command ngắn, dễ debug, và tách từng bước rõ ràng.
- Không dùng lại output dir cũ nếu artifact cần giữ provenance sạch.

