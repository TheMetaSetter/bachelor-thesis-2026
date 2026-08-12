# Offline Phase — Table 1

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thesis main + O0</th>
      <td><u>0.7978</u></td>
      <td><u>0.7468</u></td>
      <td><strong>0.8923</strong></td>
      <td><strong>0.7051</strong></td>
      <td><u>0.7075</u></td>
      <td><strong>0.9502</strong></td>
      <td><u>0.5778</u></td>
      <td>0.7891</td>
      <td><u>0.6452</u></td>
    </tr>
    <tr>
      <th>Thesis main + O1</th>
      <td><strong>0.7985</strong></td>
      <td><strong>0.7477</strong></td>
      <td><u>0.8922</u></td>
      <td><u>0.7045</u></td>
      <td>0.7075</td>
      <td><u>0.9502</u></td>
      <td><u>0.5778</u></td>
      <td><u>0.7925</u></td>
      <td><u>0.6452</u></td>
    </tr>
    <tr>
      <th>RedLamp + baseline</th>
      <td>0.5861</td>
      <td>0.7008</td>
      <td>0.5004</td>
      <td>0.6139</td>
      <td><strong>0.7078</strong></td>
      <td>0.8900</td>
      <td>0.5177</td>
      <td><strong>0.8244</strong></td>
      <td>0.5468</td>
    </tr>
    <tr>
      <th>iForest</th>
      <td>0.2651</td>
      <td>0.4336</td>
      <td>0.6234</td>
      <td>0.1536</td>
      <td>0.1509</td>
      <td>0.8304</td>
      <td>0.0277</td>
      <td>0.6445</td>
      <td>0.2494</td>
    </tr>
    <tr>
      <th>KMeans-AD</th>
      <td>0.3002</td>
      <td>0.3004</td>
      <td>0.7663</td>
      <td>0.3915</td>
      <td>0.1364</td>
      <td>0.9493</td>
      <td><strong>0.6745</strong></td>
      <td>0.6987</td>
      <td><strong>0.8996</strong></td>
    </tr>
    <tr>
      <th>STUMPY + channel AB</th>
      <td>0.1751</td>
      <td>0.1897</td>
      <td>0.4989</td>
      <td>0.0579</td>
      <td>0.6235</td>
      <td>0.5060</td>
      <td>0.0450</td>
      <td>0.5923</td>
      <td>0.5504</td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `outputs/reporting/offline_phase_tables/offline_report_data.json`
- Bảng có 6 hàng, 3 entity và 3 metric; tổng cộng 54 report record được gộp thành 18 combination theo entity.
- Bảng này chỉ dùng `vus_pr`, `affiliation_f1` và `vus_roc`; UQ được trình bày ở bảng report riêng.

<table class="report-shared">
  <thead>
    <tr>
      <th rowspan="2">Method + variant</th>
      <th colspan="3">machine_1_6</th>
      <th colspan="3">machine_3_4</th>
      <th colspan="3">machine_3_9</th>
      <th colspan="3">Average across machines</th>
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
      <th>Thesis main + O0</th>
      <td><u>0.7978</u></td>
      <td><u>0.7468</u></td>
      <td><strong>0.8923</strong></td>
      <td><strong>0.7051</strong></td>
      <td><u>0.7075</u></td>
      <td><strong>0.9502</strong></td>
      <td><u>0.5778</u></td>
      <td>0.7891</td>
      <td><u>0.6452</u></td>
      <td>0.6936</td>
      <td>0.7478</td>
      <td>0.8292</td>
    </tr>
    <tr>
      <th>Thesis main + O1</th>
      <td><strong>0.7985</strong></td>
      <td><strong>0.7477</strong></td>
      <td><u>0.8922</u></td>
      <td><u>0.7045</u></td>
      <td>0.7075</td>
      <td><u>0.9502</u></td>
      <td><u>0.5778</u></td>
      <td><u>0.7925</u></td>
      <td><u>0.6452</u></td>
      <td>0.6936</td>
      <td>0.7492</td>
      <td>0.8292</td>
    </tr>
    <tr>
      <th>RedLamp + baseline</th>
      <td>0.5861</td>
      <td>0.7008</td>
      <td>0.5004</td>
      <td>0.6139</td>
      <td><strong>0.7078</strong></td>
      <td>0.8900</td>
      <td>0.5177</td>
      <td><strong>0.8244</strong></td>
      <td>0.5468</td>
      <td>0.5726</td>
      <td>0.7443</td>
      <td>0.6457</td>
    </tr>
    <tr>
      <th>iForest</th>
      <td>0.2651</td>
      <td>0.4336</td>
      <td>0.6234</td>
      <td>0.1536</td>
      <td>0.1509</td>
      <td>0.8304</td>
      <td>0.0277</td>
      <td>0.6445</td>
      <td>0.2494</td>
      <td>0.1488</td>
      <td>0.4097</td>
      <td>0.5677</td>
    </tr>
    <tr>
      <th>KMeans-AD</th>
      <td>0.3002</td>
      <td>0.3004</td>
      <td>0.7663</td>
      <td>0.3915</td>
      <td>0.1364</td>
      <td>0.9493</td>
      <td><strong>0.6745</strong></td>
      <td>0.6987</td>
      <td><strong>0.8996</strong></td>
      <td>0.4554</td>
      <td>0.3785</td>
      <td>0.8717</td>
    </tr>
    <tr>
      <th>STUMPY + channel AB</th>
      <td>0.1751</td>
      <td>0.1897</td>
      <td>0.4989</td>
      <td>0.0579</td>
      <td>0.6235</td>
      <td>0.5060</td>
      <td>0.0450</td>
      <td>0.5923</td>
      <td>0.5504</td>
      <td>0.0927</td>
      <td>0.4685</td>
      <td>0.5184</td>
    </tr>
  </tbody>
</table>
