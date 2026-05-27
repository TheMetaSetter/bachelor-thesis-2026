Dựa trên phân tích hiện trạng codebase và định hướng thesis, đây là kế hoạch lập trình sơ bộ để triển khai thí nghiệm gradient conflict trước trên baseline `RedLampMLPBaseline`.

## Current State
- `src/models/redlamp_mlp_baseline.py` đã có đủ 2 nhánh nhiệm vụ dùng chung encoder MLP theo đúng contract output của repo: `recon`, `logits`, `point_scores`, `hidden`, `pooled`.
- Hàm `_shared_step(...)` hiện cộng loss theo dạng:
  - `total_loss = reconstruction_loss + lambda_cls * classification_loss`
  - chưa có cơ chế tách gradient theo từng objective để đo giao thoa gradient.
- Baseline đã tích hợp synthetic anomaly injection (`SyntheticAnomalyInjector`) cho `train` và `val_synth`, nên có thể dùng trực tiếp pipeline hiện tại để đo metric theo từng iteration mà không đổi contract batch.
- `BaseModel` contract đang ổn định, nên chỗ phù hợp nhất để thêm profiling là ngay trong `RedLampMLPBaseline` (giữ nguyên nguyên tắc 1 model - 1 file).

## Design Options
- Option A (khuyến nghị cho bước sơ bộ):
  - Giữ nguyên forward/loss semantics của baseline.
  - Thêm một cơ chế profiling gradient “ngoại vi” trong chính model file:
    - tính riêng `g_CE`, `g_MSE`, `g_total` cho các layer encoder;
    - chỉ log metric, không can thiệp trajectory update mặc định của optimizer trong production path.
  - Mục tiêu: kiểm chứng giả thuyết conflict trước, sau đó mới nâng cấp sang manual gradient injection cho nghiên cứu sâu.
- Option B:
  - Tích hợp ngay quy trình backprop 2 lượt + manual `.grad` injection trong train loop baseline.
  - Ưu điểm: đo trực tiếp đúng cơ chế can thiệp.
  - Nhược điểm: dễ sai update ở các layer khác nếu scope inject không đầy đủ; tăng độ phức tạp ngay từ đầu.
- Option C:
  - Tạo “diagnostic mode” tách biệt: 1 step chuẩn + 1 step chỉ đo gradient bằng `autograd.grad` trên cùng batch.
  - Ít rủi ro làm lệch training nhưng tốn compute hơn.

## Risk and Mitigation
- Risk: chỉ inject `.grad` cho 1 layer sẽ làm các layer còn lại update sai hoặc không update.
  - Mitigation: mặc định không can thiệp optimizer path ở giai đoạn sơ bộ; nếu bật chế độ injection thì phải inject full parameter set nằm trong profile scope.
- Risk: metric nhiễu mạnh theo batch gây kết luận sai về conflict.
  - Mitigation: log đồng thời raw + EMA(alpha=0.1) + SMA(window=50), lấy EMA làm đường chính.
- Risk: graph/backward retain sai gây memory blow-up.
  - Mitigation: tách rõ diagnostic path, zero-grad đúng thứ tự, và chỉ retain graph khi thật sự cần.
- Risk: drift giữa metric logged và metric thực dùng evaluator.
  - Mitigation: giữ nguyên thresholding evaluator hiện tại `q=0.95`, không thay đổi luồng chấm điểm anomaly.

## Open Questions (cần chốt trước khi code)
- Profiling scope theo layer ở baseline MLP:
  - đo toàn bộ encoder linear layers hay ưu tiên một tập con (ví dụ layer cuối encoder) làm “focus layer” tương đương bottleneck?
- Chiến lược tích hợp vào trainer:
  - đặt logic profiling trong `training_step` của model hay ở `trainer` để dễ bật/tắt theo config?
- Tần suất ghi log:
  - log mọi iteration hay sampling theo chu kỳ (vd mỗi `k` step) để giảm overhead?
- Định dạng artifact:
  - ngoài W&B, có cần thêm file cục bộ JSONL chuyên biệt cho gradient metrics không?

## Recommended Approach For Preliminary Implementation
- Chọn Option A cho vòng đầu vì đúng mục tiêu “chẩn đoán trước, can thiệp sau” và ít rủi ro làm sai đường train baseline.
- Triển khai theo lát cắt dọc tối thiểu:
  1. Thêm config bật/tắt gradient profiling cho baseline.
  2. Thêm helper trong `RedLampMLPBaseline` để:
     - trích parameter groups của encoder;
     - tính `g_CE`, `g_MSE`, `g_total` ở mỗi iteration;
     - tính `cosine_similarity` và `R-ratio` cho từng layer.
  3. Thêm bộ nhớ state để giữ raw/EMA/SMA theo iteration.
  4. Đẩy metric vào hệ log hiện tại (và tùy chọn JSONL nếu cần).
  5. Giữ nguyên loss train/eval gốc và ngưỡng `q=0.95` của evaluator.

## Planned File-Level Changes (Sơ bộ)
- `src/models/redlamp_mlp_baseline.py`
  - thêm dataclass/config block cho gradient profiling.
  - thêm helper enumerate encoder layers, gradient flattening, metric computation, EMA/SMA update.
  - thêm diagnostic logging path ở `training_step`.
- `configs/model/*.yaml` hoặc config baseline tương ứng
  - thêm cờ `enable_gradient_conflict_profiling`, `profiling_scope`, `ema_alpha`, `sma_window`, `log_every_n_steps`.
- `src/engine/trainer.py` (nếu cần)
  - nhận thêm các metric dictionary từ model step để log nhất quán.
- `tests/`
  - test shape + finite-value cho cosine/R-ratio.
  - test consistency của EMA/SMA update.
  - smoke test 1 train step với profiling bật.

## Validation Procedure (Sơ bộ)
- Chạy 1 epoch ngắn với profiling bật trên tập nhỏ.
- Xác nhận:
  - không crash graph/backward;
  - metric raw/EMA/SMA đều được log;
  - train loss vẫn giảm tương tự baseline không profiling (sai khác trong ngưỡng chấp nhận do overhead đo).
- Kiểm tra `focus layer` có log đầy đủ `||g_CE||`, `||g_MSE||`, `||g_total||`, cosine, R-ratio.

## Decision Gate
- Nếu Option A xác nhận có conflict đáng kể (cosine âm kéo dài + R-ratio thấp kéo dài), chuyển sang kế hoạch chi tiết để triển khai chế độ manual gradient injection có kiểm soát cho toàn encoder scope.
- Nếu conflict yếu/không ổn định, ưu tiên giữ baseline training path và dùng profiling như công cụ quan sát, chưa cần can thiệp optimizer.
