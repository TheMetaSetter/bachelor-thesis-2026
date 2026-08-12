# Offline Phase — Table 2

So sánh uncertainty của THESIS trên clean validation và test. Mỗi ô là trung bình số học của 3 seed (`seed6`, `seed8`, `seed36`) cho cùng variant và entity. Metric được báo cáo là `mean_of_variances`, làm tròn đến 3 chữ số sau dấu thập phân.

<style>
  .report2 { border-collapse: collapse; }
  .report2 th, .report2 td { padding: 0.55rem 1.25rem; text-align: center; }
  .report2 .blank-corner { background: #fff; border: 0; }
  .report2 .entity-header, .report2 .validation-header, .report2 .test-header { background: #dcebf8; }
  .report2 tbody th { text-align: left; white-space: nowrap; }
</style>

<table class="report2">
  <thead>
    <tr>
      <th rowspan="2" class="blank-corner"></th>
      <th colspan="2" class="entity-header">machine_1_6</th>
      <th colspan="2" class="entity-header">machine_3_4</th>
      <th colspan="2" class="entity-header">machine_3_9</th>
    </tr>
    <tr>
      <th class="validation-header">Validation</th>
      <th class="test-header">Test</th>
      <th class="validation-header">Validation</th>
      <th class="test-header">Test</th>
      <th class="validation-header">Validation</th>
      <th class="test-header">Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>THESIS + O0</th>
      <td>0.012</td>
      <td>0.250</td>
      <td>0.013</td>
      <td>0.432</td>
      <td>0.013</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>THESIS + O1</th>
      <td>0.006</td>
      <td>0.089</td>
      <td>0.019</td>
      <td>0.573</td>
      <td>0.024</td>
      <td>0.061</td>
    </tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `outputs/reporting/offline_phase_tables/canonical_run_manifest.json`
- Trường dữ liệu: `report_fields.uq_summary.splits.clean_validation.mean_of_variances` và `report_fields.uq_summary.splits.test.mean_of_variances`.
- Bảng chỉ gồm THESIS với hai variant `O0` và `O1`, ba entity, và hai split được so sánh.
