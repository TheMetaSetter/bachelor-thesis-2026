# Two-Stage Offline Pretraining Gap Closure Structure

## Overview
Mục tiêu của đợt lập trình này là đóng các gap còn thiếu giữa spec two-stage offline pretraining và code hiện tại, nhưng vẫn giữ nguyên đường base two-stage đang hoạt động. Ý tưởng chính ở đây là thêm điểm thiếu thật sự: score-loss supervised cho Stage A, schema/config tương ứng, và các YAML exp4 bị thiếu, trong khi evaluator timeline và base runner hiện có được giữ nguyên.

## Implementation Phases

1. **Phase 1: Config and variant contract alignment** - Chuẩn hóa schema để code nhận được variant base và variant point-score một cách tường minh, thêm các field score-loss cần thiết, và giữ backward compatibility với batch/runtime contract hiện tại. Giai đoạn này theo nguyên tắc separation of concerns vì chỉ chạm vào config/validation layer, không đụng model logic. Dùng adapter-style compatibility để map ý nghĩa mới vào contract cũ thay vì đổi toàn bộ input schema.

2. **Phase 2: Stage A score-loss implementation** - Bổ sung point-wise balanced reconstruction-score loss vào `ThesisMultitaskLossMixin`, tính từ `point_scores` và `synthetic_anomaly_mask`, và chỉ bật trong Stage A khi variant yêu cầu. Giai đoạn này giữ single responsibility cho model loss logic và composition over inheritance vì logic mới nằm trong helper cục bộ của model, không tạo lớp mới. Base two-stage vẫn chỉ dùng reconstruction, classification, và contrastive loss.

3. **Phase 3: Experiment YAML restoration and runner preservation** - Khôi phục các file exp4 two-stage bị thiếu và tạo thêm bản point-score-supervised với tên variant rõ ràng. Runner `scripts/run_two_stage_offline_pretraining.py` chỉ cần bảo toàn manifest, stage ordering, và stage-b init checkpoint path, không cần đổi topology. Giai đoạn này dùng factory/registry style hiện có của repo cho experiment config, đồng thời giữ minimal vertical slice bằng cách sao chép YAML cơ sở rồi bật cờ mới.

4. **Phase 4: Test and contract locking** - Thêm test cho config loading, runner orchestration, stage-loss behavior, và checkpoint/threshold metadata để khóa contract mới. Mục tiêu là bảo đảm model output contract, batch contract hiện tại, và evaluation contract không bị lệch khi bật score-loss. Đây là bước bảo vệ stable interfaces: test phải chứng minh base run không đổi và point-score variant chỉ mở rộng hành vi đã chốt.

5. **Phase 5: Dry-run verification and reporting** - Chạy narrow pytest set và dry-run runner để xác nhận cấu hình mới, manifest generation, Stage A -> Stage B -> evaluation ordering, và score-loss toggle đều vận hành đúng. Giai đoạn này giữ engineering rigor bằng cách xác minh trước khi mở rộng thêm ablation hoặc thay batch contract rộng hơn. Nếu cần báo cáo, chỉ ghi nhận kết quả và mở câu hỏi tiếp theo về migration batch naming, không mở thêm scope.

Does this phasing make sense? Nếu muốn, em có thể chỉnh lại độ hạt mịn của Phase 2 và Phase 3 trước khi đi vào detailed prompt.

