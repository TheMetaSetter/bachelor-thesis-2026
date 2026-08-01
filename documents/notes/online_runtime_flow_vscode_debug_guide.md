# Hướng dẫn debug online runtime flow bằng VSCode

Tài liệu này hướng dẫn cách dừng chương trình và xem dữ liệu trong runtime flow hiện tại của THESIS. Cách này chỉ dùng để chẩn đoán. Không dùng config có `debug_timing: true` cho run báo cáo performance.

## 1. Chuẩn bị

Mở codebase bằng VSCode. Nếu chạy trên máy GPU, mở thư mục bằng VSCode Remote-SSH để Python debugger chạy trên máy remote.

Codebase hiện chưa có thư mục `.vscode`. Có thể tạo một cấu hình debug tạm thời trong VSCode, không cần commit file đó.

Trong `Run and Debug`, chọn `create a launch.json file`, chọn `Python`, rồi dùng cấu hình tương đương sau:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug THESIS online A2",
      "type": "debugpy",
      "request": "launch",
      "module": "scripts.run_thesis_online_benchmark",
      "args": [
        "--experiment-config",
        "configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml",
        "--protocol-config",
        "configs/protocol/smd_window20_cleanval_q99_ewma09.yaml",
        "--online-variant",
        "A2"
      ],
      "cwd": "${workspaceFolder}",
      "justMyCode": true,
      "console": "integratedTerminal"
    }
  ]
}
```

Config mẫu dùng `O1`, `A2`, entity `machine_1_6`, và stream `[5608,5909)`. Các giá trị này nằm trong [config chẩn đoán transfer timing](../../configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml#L1-L42).

## 2. Đặt breakpoint theo thứ tự runtime

Các breakpoint dưới đây là điểm kiểm tra gợi ý. Anh có thể tự đặt thêm breakpoint ở bất kỳ hàm hoặc dòng nào. Điều này hữu ích khi cần xem một tensor, một nhánh `if`, hoặc một lần gọi hàm cụ thể. Sau mỗi lần dừng, nhấn `F10` để đi qua dòng hiện tại hoặc `F5` để chạy tới breakpoint kế tiếp.

1. [Benchmark entrypoint](../../scripts/benchmarks/run_thesis_online_benchmark.py#L218-L243): kiểm tra `online_variant`, config, checkpoint Stage B và lời gọi vào online experiment.
2. [`run_thesis_online_tta_experiment`](../../src/engine/online_tta/online_engine_run.py#L487-L545): kiểm tra model, device, `verification_buffer`, `signature_history` và đoạn stream đã được chọn.
3. Vòng lặp từng cửa sổ trong [`_run_online_sequence`](../../src/engine/online_tta/online_engine_run.py#L261-L284): kiểm tra `batch`, `meta[0]`, `start_index`, `end_index` và `stream_step`.
4. [`_process_online_window`](../../src/engine/online_tta/online_engine_window_core.py#L53-L108): đây là điểm bao quanh bốn nhóm việc chính: chuẩn bị event, buffer/verification, adaptation step và tạo output.
5. [`_prepare_online_window_event`](../../src/engine/online_tta/online_engine_window_core.py#L141-L194): kiểm tra thứ tự thực tế. Code hiện tại tính preliminary `pnn_mask` trước khi gọi `_classify_event_window`.
6. [`_build_event_pnn_mask`](../../src/engine/online_tta/online_engine_window_metrics.py#L149-L193): kiểm tra `hidden`, `known_anomaly`, `signatures`, `signature_history` và số point trong `pnn_mask`.
7. [`_update_online_window_buffers`](../../src/engine/online_tta/online_engine_window_metrics.py#L196-L222): chỉ cửa sổ có `triage_decision == "gray_zone"` mới được thêm vào `verification_buffer`.
8. [`_verify_and_adapt_entries`](../../src/engine/online_tta/online_engine_window_metrics.py#L33-L79): kiểm tra nhánh verification có chạy hay không và dữ liệu được tính lại cho các entry trong buffer.
9. [`_run_online_variant_update`](../../src/engine/online_tta/online_engine_step.py#L108-L184): kiểm tra điều kiện update của A1/A2, loss, `pnn_mask`, backward và `optimizer.step()`.

## 3. Những biến nên xem trong VSCode

Khi chương trình dừng, thêm các biểu thức sau vào `Watch`:

```text
batch["meta"][0]
event["triage_decision"]
event["input_window_score"]
event["latent_window_score"]
event["ewma_point_score"]
batch.get("pnn_mask")
len(signature_history)
len(verification_buffer.entries)
step_result["did_update"]
```

Ở bước verification, xem thêm:

```text
entry["entry_id"]
entry["window_start"]
entry["window_end"]
result.pseudo_normal_points
result.adapted
```

Nếu tên biến không tồn tại tại breakpoint đó, bỏ qua biến đó. VSCode chỉ hiển thị biến thuộc scope của frame hiện tại.

## 4. Cách kiểm tra một cửa sổ

Với mỗi cửa sổ, kiểm tra theo bốn câu hỏi:

- Cửa sổ bắt đầu và kết thúc ở index nào?
- `triage_decision` là `normal`, `hard_old_normality`, `gray_zone` hay `strong_anomaly`?
- Có preliminary `pnn_mask` trước triage hay không?
- Có entry mới trong `verification_buffer` hoặc có adaptation step hay không?

Với A2, chỉ xem là có update khi `step_result["did_update"]` là `True`. Sau đó kiểm tra `reconstruction_loss`, `contrastive_loss` và `projector_grad_norm` trong `model._last_online_diagnostics`.

## 5. Phân biệt breakpoint và timing log

Breakpoint giúp xem dữ liệu của một cửa sổ tại một thời điểm. Timing log giúp so sánh thời gian của nhiều cửa sổ.

`OnlineTtaTimingLogger` chỉ in log khi được bật và đo các nhóm `prepare_event`, `buffer_and_verification`, `adaptation_step` và `build_outputs`; xem [timing logger](../../src/engine/online_tta/timing_debug.py#L11-L42). Config chẩn đoán bật `debug_timing` ở [dòng 42](../../configs/experiment/online_diagnostic/thesis/smd__thesis__online__O1_A2__machine_1_6__w20__seed6__transfer_timing_5608_5909.yaml#L39-L43).

Khi debug bằng VSCode, nên đặt breakpoint ở component nghi ngờ. Khi đo thời gian, chạy đoạn stream ngắn và giữ `debug_timing: true`. Test xác nhận logger im lặng khi tắt và in entity, interval, component, elapsed time khi bật tại [test timing debug](../../tests/online/test_online_timing_debug.py#L18-L36).

## 6. Luồng cần nhớ

```text
benchmark entrypoint
  -> run_thesis_online_tta_experiment
  -> _run_online_sequence
  -> _process_online_window
  -> prepare_event
  -> preliminary pnn_mask/signatures
  -> triage
  -> gray-zone admission and verification
  -> A1/A2 adaptation step
  -> output and runtime-state sync
```

Đây là luồng đang được lập trình. Nó chưa phải luồng triage-first trong ý tưởng gốc vì preliminary `pnn_mask` và signature được tính trong `prepare_event` trước khi phân loại cửa sổ.

Tài liệu này chỉ hướng dẫn debug và không thay đổi source code, config hay kết quả thí nghiệm.
