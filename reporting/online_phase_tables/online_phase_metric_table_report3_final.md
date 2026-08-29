# Online Phase — Table 3 (Final)

Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng method, variant và entity.

VUS-PR và affiliation F1: càng cao càng tốt; FPR: càng thấp càng tốt. Giá trị tốt nhất được in đậm; giá trị tốt thứ hai được gạch chân.

<style>
  .report-shared { border-collapse: collapse; }
  .report-shared th, .report-shared td { padding: 0.55rem 1.25rem; text-align: center; }
  .report-shared .blank-corner { background: #fff; border: 0; }
  .report-shared thead th { background: #dcebf8; }
  .report-shared tbody th { text-align: left; white-space: nowrap; }
</style>

<table class="report-shared">
  <thead>
    <tr>
      <th rowspan="2" class="blank-corner"></th>
      <th colspan="3">machine-1-6</th>
      <th colspan="3">machine-3-4</th>
      <th colspan="3">machine-3-9</th>
      <th colspan="3">Average</th>
    </tr>
    <tr>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>FPR</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>FPR</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>FPR</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>FPR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>THESIS O0 + A0</th>
      <td><u>0.8083</u></td><td><u>0.9986</u></td><td><strong>0.0000</strong></td>
      <td>0.9796</td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td>0.6593</td><td>0.7648</td><td>0.0729</td>
      <td>0.8157</td><td><u>0.8281</u></td><td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O0 + A1</th>
      <td><u>0.8083</u></td><td><u>0.9986</u></td><td><strong>0.0000</strong></td>
      <td>0.9796</td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td>0.6593</td><td>0.7648</td><td>0.0729</td>
      <td>0.8157</td><td><u>0.8281</u></td><td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O0 + A2</th>
      <td><u>0.8083</u></td><td><u>0.9986</u></td><td><strong>0.0000</strong></td>
      <td>0.9795</td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td>0.6574</td><td>0.7648</td><td>0.0729</td>
      <td>0.8151</td><td><u>0.8281</u></td><td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O1 + A0</th>
      <td><strong>0.8241</strong></td><td><strong>0.9986</strong></td><td><strong>0.0000</strong></td>
      <td><strong>0.9804</strong></td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td><strong>0.6667</strong></td><td><u>0.8177</u></td><td><u>0.0700</u></td>
      <td><strong>0.8237</strong></td><td><strong>0.8458</strong></td><td><u>0.3463</u></td>
    </tr>
    <tr>
      <th>THESIS O1 + A1</th>
      <td><strong>0.8241</strong></td><td><strong>0.9986</strong></td><td><strong>0.0000</strong></td>
      <td><strong>0.9804</strong></td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td><strong>0.6667</strong></td><td><u>0.8177</u></td><td><u>0.0700</u></td>
      <td><strong>0.8237</strong></td><td><strong>0.8458</strong></td><td><u>0.3463</u></td>
    </tr>
    <tr>
      <th>THESIS O1 + A2</th>
      <td><strong>0.8241</strong></td><td><strong>0.9986</strong></td><td><strong>0.0000</strong></td>
      <td><u>0.9803</u></td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td><u>0.6661</u></td><td><u>0.8177</u></td><td><u>0.0700</u></td>
      <td><u>0.8235</u></td><td><strong>0.8458</strong></td><td><u>0.3463</u></td>
    </tr>
    <tr>
      <th>M2N2</th>
      <td>0.5743</td><td>0.7905</td><td><u>0.0119</u></td>
      <td>0.8147</td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td><strong>0.7165</strong></td><td><strong>0.9972</strong></td><td><strong>0.0037</strong></td>
      <td>0.7019</td><td><u>0.8362</u></td><td><strong>0.3281</strong></td>
    </tr>
    <tr>
      <th>CANDI</th>
      <td><u>0.5902</u></td><td><u>0.7929</u></td><td>0.0147</td>
      <td><u>0.9194</u></td><td><strong>0.7210</strong></td><td><strong>0.9688</strong></td>
      <td>0.6084</td><td>0.8079</td><td>0.1715</td>
      <td><u>0.7060</u></td><td>0.7739</td><td>0.3850</td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `reporting/online_phase_tables/online_table3_metrics.json`
- Bảng gồm 8 hàng được chọn từ Table 3 online gốc: 6 biến thể THESIS, M2N2 và CANDI; mỗi ô là trung bình của 3 seed.
- Mục đích: giữ đầy đủ các biến thể THESIS để đối chiếu cùng M2N2 và CANDI.
- Ba cột cuối là trung bình số học của từng metric theo 3 entity.
- FPR được tính bằng `FP / (FP + TN)` và được xếp hạng theo hướng thấp hơn là tốt hơn.
- Score dùng để tính metric là `online/ewma_point_score`; threshold và prediction giữ nguyên từ runtime online.
- Protocol VUS dùng `vus_max_buffer_size = 20` và `vus_num_thresholds = 200`.
