# Online Phase — Ablation Study

Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng method, variant và entity.

Giá trị cao nhất được in đậm; giá trị cao thứ hai được gạch chân. Các metric đều được hiểu là càng cao càng tốt.

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
      <th>THESIS O0 + A0</th>
      <td><u>0.8083</u></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8952</u></td>
      <td><u>0.9796</u></td>
      <td><strong>0.7210</strong></td>
      <td><strong>0.9776</strong></td>
      <td><u>0.6593</u></td>
      <td><u>0.7648</u></td>
      <td><u>0.7300</u></td>
      <td><u>0.8157</u></td>
      <td><u>0.8281</u></td>
      <td><u>0.8676</u></td>
    </tr>
    <tr>
      <th>THESIS O0 + A1</th>
      <td><u>0.8083</u></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8952</u></td>
      <td><u>0.9796</u></td>
      <td><strong>0.7210</strong></td>
      <td><strong>0.9776</strong></td>
      <td><u>0.6593</u></td>
      <td><u>0.7648</u></td>
      <td><u>0.7300</u></td>
      <td><u>0.8157</u></td>
      <td><u>0.8281</u></td>
      <td><u>0.8676</u></td>
    </tr>
    <tr>
      <th>THESIS O0 + A2</th>
      <td><u>0.8083</u></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8952</u></td>
      <td>0.9795</td>
      <td><strong>0.7210</strong></td>
      <td><u>0.9775</u></td>
      <td>0.6574</td>
      <td><u>0.7648</u></td>
      <td>0.7284</td>
      <td>0.8151</td>
      <td><u>0.8281</u></td>
      <td>0.8671</td>
    </tr>
    <tr>
      <th>THESIS O1 + A0</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><strong>0.8975</strong></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9767</td>
      <td><strong>0.6667</strong></td>
      <td><strong>0.8177</strong></td>
      <td><strong>0.7335</strong></td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td><strong>0.8692</strong></td>
    </tr>
    <tr>
      <th>THESIS O1 + A1</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><strong>0.8975</strong></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9767</td>
      <td><strong>0.6667</strong></td>
      <td><strong>0.8177</strong></td>
      <td><strong>0.7335</strong></td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td><strong>0.8692</strong></td>
    </tr>
    <tr>
      <th>THESIS O1 + A2</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8975</u></td>
      <td><u>0.9803</u></td>
      <td><strong>0.7210</strong></td>
      <td>0.9767</td>
      <td><u>0.6661</u></td>
      <td><u>0.8177</u></td>
      <td>0.7335</td>
      <td><u>0.8235</u></td>
      <td><strong>0.8458</strong></td>
      <td>0.8692</td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `reporting/online_phase_tables/online_table3_metrics.json`
- Bảng gồm 6 biến thể THESIS được chọn từ Table 3 online gốc; mỗi ô là trung bình của 3 seed.
- Mục đích: so sánh các biến thể THESIS O0/O1 và A0/A1/A2.
- Ba cột cuối là trung bình số học của từng metric theo 3 entity.
- Score dùng để tính metric là `online/ewma_point_score`; threshold và prediction giữ nguyên từ runtime online.
- Protocol VUS dùng `vus_max_buffer_size = 20` và `vus_num_thresholds = 200`.
