# Online Phase — Table 3

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
      <th>THESIS O0 + A0</th>
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td>0.8952</td>
      <td>0.9796</td>
      <td><strong>0.7210</strong></td>
      <td>0.9776</td>
      <td>0.6593</td>
      <td>0.7648</td>
      <td>0.7300</td>
      <td>0.8157</td>
      <td><u>0.8281</u></td>
      <td>0.8676</td>
    </tr>
    <tr>
      <th>THESIS O0 + A1</th>
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td>0.8952</td>
      <td>0.9796</td>
      <td><strong>0.7210</strong></td>
      <td>0.9776</td>
      <td>0.6593</td>
      <td>0.7648</td>
      <td>0.7300</td>
      <td>0.8157</td>
      <td><u>0.8281</u></td>
      <td>0.8676</td>
    </tr>
    <tr>
      <th>THESIS O0 + A2</th>
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td>0.8952</td>
      <td>0.9795</td>
      <td><strong>0.7210</strong></td>
      <td>0.9775</td>
      <td>0.6574</td>
      <td>0.7648</td>
      <td>0.7284</td>
      <td>0.8151</td>
      <td><u>0.8281</u></td>
      <td>0.8671</td>
    </tr>
    <tr>
      <th>THESIS O1 + A0</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8975</u></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9767</td>
      <td><strong>0.6667</strong></td>
      <td><u>0.8177</u></td>
      <td><u>0.7335</u></td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td>0.8692</td>
    </tr>
    <tr>
      <th>THESIS O1 + A1</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><u>0.8975</u></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9767</td>
      <td><strong>0.6667</strong></td>
      <td><u>0.8177</u></td>
      <td><u>0.7335</u></td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td>0.8692</td>
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
    <tr>
      <th>M2N2</th>
      <td>0.5849</td>
      <td>0.7932</td>
      <td><strong>0.9766</strong></td>
      <td>0.8833</td>
      <td><strong>0.7210</strong></td>
      <td><u>0.9776</u></td>
      <td>0.5861</td>
      <td>0.7284</td>
      <td>0.7204</td>
      <td>0.6848</td>
      <td>0.7475</td>
      <td><u>0.8915</u></td>
    </tr>
    <tr>
      <th>CANDI</th>
      <td>0.5849</td>
      <td>0.7932</td>
      <td><strong>0.9766</strong></td>
      <td>0.8833</td>
      <td><strong>0.7210</strong></td>
      <td><u>0.9776</u></td>
      <td>0.5861</td>
      <td>0.7284</td>
      <td>0.7204</td>
      <td>0.6848</td>
      <td>0.7475</td>
      <td><u>0.8915</u></td>
    </tr>
    <tr>
      <th>Isolation Forest</th>
      <td>0.0588</td>
      <td>0.5498</td>
      <td>0.2749</td>
      <td>0.7654</td>
      <td>0.1142</td>
      <td>0.9363</td>
      <td>0.0354</td>
      <td>0.6984</td>
      <td>0.4612</td>
      <td>0.2865</td>
      <td>0.4541</td>
      <td>0.5575</td>
    </tr>
    <tr>
      <th>KMeansAD</th>
      <td>0.2419</td>
      <td>0.2090</td>
      <td>0.8623</td>
      <td>0.9533</td>
      <td>0.3208</td>
      <td><strong>0.9837</strong></td>
      <td>0.5157</td>
      <td><strong>0.9211</strong></td>
      <td><strong>0.8623</strong></td>
      <td>0.5703</td>
      <td>0.4836</td>
      <td><strong>0.9028</strong></td>
    </tr>
    <tr>
      <th>StumPy</th>
      <td>0.0372</td>
      <td>0.4266</td>
      <td>0.4621</td>
      <td>0.3320</td>
      <td><u>0.7194</u></td>
      <td>0.6126</td>
      <td>0.0431</td>
      <td>0.7249</td>
      <td>0.5968</td>
      <td>0.1374</td>
      <td>0.6236</td>
      <td>0.5572</td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `reporting/online_phase_tables/online_table3_metrics.json`
- Bảng có 11 hàng, 3 entity và 3 metric; tổng cộng 99 run được gộp thành 33 combination theo entity.
- Ba cột cuối là trung bình số học của từng metric theo 3 entity.
- Score dùng để tính metric là `online/ewma_point_score`; threshold và prediction giữ nguyên từ runtime online.
- Protocol VUS dùng `vus_max_buffer_size = 20` và `vus_num_thresholds = 200`.
