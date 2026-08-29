# Online Phase — Table 3

Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng method, variant và entity.

VUS-PR và affiliation F1: càng cao càng tốt; FPR: càng thấp càng tốt. Giá trị tốt nhất được in đậm; giá trị tốt thứ hai được gạch chân.

<style>
  .report-shared { border-collapse: collapse; }
  .report-shared th, .report-shared td { padding: 0.55rem 1.25rem; text-align: center; }
  .report-shared .blank-corner { background: #fff; border: 0; }
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
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td><strong>0.0000</strong></td>
      <td>0.9796</td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td>0.6593</td>
      <td>0.7648</td>
      <td>0.0729</td>
      <td>0.8157</td>
      <td>0.8281</td>
      <td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O0 + A1</th>
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td><strong>0.0000</strong></td>
      <td>0.9796</td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td>0.6593</td>
      <td>0.7648</td>
      <td>0.0729</td>
      <td>0.8157</td>
      <td>0.8281</td>
      <td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O0 + A2</th>
      <td><u>0.8083</u></td>
      <td><u>0.9986</u></td>
      <td><strong>0.0000</strong></td>
      <td>0.9795</td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td>0.6574</td>
      <td>0.7648</td>
      <td>0.0729</td>
      <td>0.8151</td>
      <td>0.8281</td>
      <td>0.3472</td>
    </tr>
    <tr>
      <th>THESIS O1 + A0</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><strong>0.0000</strong></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td><u>0.6667</u></td>
      <td>0.8177</td>
      <td>0.0700</td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td>0.3463</td>
    </tr>
    <tr>
      <th>THESIS O1 + A1</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><strong>0.0000</strong></td>
      <td><strong>0.9804</strong></td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td><u>0.6667</u></td>
      <td>0.8177</td>
      <td>0.0700</td>
      <td><strong>0.8237</strong></td>
      <td><strong>0.8458</strong></td>
      <td>0.3463</td>
    </tr>
    <tr>
      <th>THESIS O1 + A2</th>
      <td><strong>0.8241</strong></td>
      <td><strong>0.9986</strong></td>
      <td><strong>0.0000</strong></td>
      <td><u>0.9803</u></td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td>0.6661</td>
      <td>0.8177</td>
      <td>0.0700</td>
      <td><u>0.8235</u></td>
      <td><strong>0.8458</strong></td>
      <td>0.3463</td>
    </tr>
    <tr>
      <th>M2N2</th>
      <td>0.5743</td>
      <td>0.7905</td>
      <td>0.0119</td>
      <td>0.8147</td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td><strong>0.7165</strong></td>
      <td><strong>0.9972</strong></td>
      <td><strong>0.0037</strong></td>
      <td>0.7019</td>
      <td><u>0.8362</u></td>
      <td>0.3281</td>
    </tr>
    <tr>
      <th>CANDI</th>
      <td>0.5902</td>
      <td>0.7929</td>
      <td>0.0147</td>
      <td>0.9194</td>
      <td><strong>0.7210</strong></td>
      <td>0.9688</td>
      <td>0.6084</td>
      <td>0.8079</td>
      <td>0.1715</td>
      <td>0.7060</td>
      <td>0.7739</td>
      <td>0.3850</td>
    </tr>
    <tr>
      <th>Isolation Forest</th>
      <td>0.0588</td>
      <td>0.5498</td>
      <td>0.0207</td>
      <td>0.7654</td>
      <td>0.1142</td>
      <td><u>0.0009</u></td>
      <td>0.0354</td>
      <td>0.6984</td>
      <td>0.0752</td>
      <td>0.2865</td>
      <td>0.4541</td>
      <td>0.0323</td>
    </tr>
    <tr>
      <th>KMeansAD</th>
      <td>0.2419</td>
      <td>0.2090</td>
      <td><u>0.0046</u></td>
      <td>0.9533</td>
      <td>0.3208</td>
      <td><strong>0.0000</strong></td>
      <td>0.5157</td>
      <td><u>0.9211</u></td>
      <td><u>0.0064</u></td>
      <td>0.5703</td>
      <td>0.4836</td>
      <td><strong>0.0037</strong></td>
    </tr>
    <tr>
      <th>StumPy</th>
      <td>0.0372</td>
      <td>0.4266</td>
      <td>0.0208</td>
      <td>0.3320</td>
      <td><u>0.7194</u></td>
      <td>0.0162</td>
      <td>0.0431</td>
      <td>0.7249</td>
      <td>0.0417</td>
      <td>0.1374</td>
      <td>0.6236</td>
      <td><u>0.0262</u></td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `reporting/online_phase_tables/online_table3_metrics.json`
- Bảng có 11 hàng, 3 entity và 3 metric; tổng cộng 99 run được gộp thành 33 combination theo entity.
- Ba cột cuối là trung bình số học của từng metric theo 3 entity.
- Score dùng để tính metric là `online/ewma_point_score`; threshold và prediction giữ nguyên từ runtime online.
- Protocol VUS dùng `vus_max_buffer_size = 20` và `vus_num_thresholds = 200`.
