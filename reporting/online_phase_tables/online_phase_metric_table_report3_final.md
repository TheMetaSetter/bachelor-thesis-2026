# Online Phase — Table 3 (Final)

Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng method, variant và entity.

Giá trị cao nhất được in đậm; giá trị cao thứ hai được gạch chân. Các metric đều được hiểu là càng cao càng tốt.

<style>
  .report-shared { border-collapse: collapse; }
  .report-shared th, .report-shared td { padding: 0.55rem 1.25rem; text-align: center; }
  .report-shared thead th { background: #dcebf8; }
  .report-shared tbody th { text-align: left; white-space: nowrap; }
</style>

<table class="report-shared">
  <thead>
    <tr>
      <th rowspan="2">Method + variant</th>
      <th colspan="3">machine_1_6</th>
      <th colspan="3">machine_3_4</th>
      <th colspan="3">machine_3_9</th>
      <th colspan="3">Trung bình theo entity</th>
    </tr>
    <tr>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>VUS-ROC</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>VUS-ROC</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>VUS-ROC</th>
      <th>VUS-PR</th>
      <th>affiliation F1</th>
      <th>VUS-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>THESIS O1 + A2</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td>0.8975</td>
      <td><strong>0.9803</strong></td>
      <td><strong>0.7210</strong></td>
      <td><strong>0.9767</strong></td>
      <td><u>0.6661</u></td>
      <td><u>0.8177</u></td>
      <td>0.7335</td>
      <td><strong>0.8235</strong></td>
      <td><strong>0.8458</strong></td>
      <td>0.8692</td>
    </tr>
    <tr>
      <th>M2N2</th>
      <td>0.5743</td>
      <td>0.7905</td>
      <td><strong>0.9825</strong></td>
      <td>0.8147</td>
      <td><strong>0.7210</strong></td>
      <td><u>0.9763</u></td>
      <td><strong>0.7165</strong></td>
      <td><strong>0.9972</strong></td>
      <td><u>0.8568</u></td>
      <td>0.7019</td>
      <td><u>0.8362</u></td>
      <td><strong>0.9385</strong></td>
    </tr>
    <tr>
      <th>CANDI</th>
      <td><u>0.5902</u></td>
      <td><u>0.7929</u></td>
      <td><u>0.9726</u></td>
      <td><u>0.9194</u></td>
      <td><strong>0.7210</strong></td>
      <td>0.9615</td>
      <td>0.6084</td>
      <td>0.8079</td>
      <td><strong>0.8714</strong></td>
      <td><u>0.7060</u></td>
      <td>0.7739</td>
      <td><u>0.9352</u></td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `reporting/online_phase_tables/online_table3_metrics.json`
- Bảng gồm 3 hàng được chọn từ Table 3 online gốc; mỗi ô là trung bình của 3 seed.
- Mục đích: bảng kết quả cuối cùng với THESIS O1 + A2, M2N2 và CANDI.
- Ba cột cuối là trung bình số học của từng metric theo 3 entity.
- Score dùng để tính metric là `online/ewma_point_score`; threshold và prediction giữ nguyên từ runtime online.
- Protocol VUS dùng `vus_max_buffer_size = 20` và `vus_num_thresholds = 200`.
