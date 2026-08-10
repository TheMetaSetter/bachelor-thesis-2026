# Online Phase — Table 3 (Final)

Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng method, variant và entity.

Giá trị cao nhất được in đậm; giá trị cao thứ hai được gạch chân. Các metric đều được hiểu là càng cao càng tốt.

<table>
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
      <td><u>0.8975</u></td>
      <td><strong>0.9803</strong></td>
      <td><strong>0.7210</strong></td>
      <td><u>0.9767</u></td>
      <td><strong>0.6661</strong></td>
      <td><strong>0.8177</strong></td>
      <td><strong>0.7335</strong></td>
      <td><strong>0.8235</strong></td>
      <td><strong>0.8458</strong></td>
      <td><u>0.8692</u></td>
    </tr>
    <tr>
      <th>M2N2</th>
      <td><u>0.5849</u></td>
      <td><u>0.7932</u></td>
      <td><strong>0.9766</strong></td>
      <td><u>0.8833</u></td>
      <td><strong>0.7210</strong></td>
      <td><strong>0.9776</strong></td>
      <td><u>0.5861</u></td>
      <td><u>0.7284</u></td>
      <td><u>0.7204</u></td>
      <td><u>0.6848</u></td>
      <td><u>0.7475</u></td>
      <td><strong>0.8915</strong></td>
    </tr>
    <tr>
      <th>CANDI</th>
      <td><u>0.5849</u></td>
      <td><u>0.7932</u></td>
      <td><strong>0.9766</strong></td>
      <td><u>0.8833</u></td>
      <td><strong>0.7210</strong></td>
      <td><strong>0.9776</strong></td>
      <td><u>0.5861</u></td>
      <td><u>0.7284</u></td>
      <td><u>0.7204</u></td>
      <td><u>0.6848</u></td>
      <td><u>0.7475</u></td>
      <td><strong>0.8915</strong></td>
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
