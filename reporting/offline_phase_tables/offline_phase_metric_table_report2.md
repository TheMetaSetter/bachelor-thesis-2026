# Offline Phase — Table 2

So sánh `VUS-PR`, `Aff. F1` và uncertainty của THESIS trên clean validation và test. Mỗi ô là trung bình của 3 seed (`seed6`, `seed8`, `seed36`) và được làm tròn đến 3 chữ số sau dấu thập phân.

<style>
  .report2 { border-collapse: collapse; }
  .report2 th, .report2 td { padding: 0.55rem 1.25rem; text-align: center; }
  .report2 thead th { background: #dcebf8; }
  .report2 .blank-corner { background: #fff; border: 0; }
  .report2 tbody th { text-align: left; white-space: nowrap; }
  .report2 .average-section th,
  .report2 .average-header th,
  .report2 .average-row th,
  .report2 .average-row td { border: 0; }
  .report2 .average-section th:first-child,
  .report2 .average-header th:first-child,
  .report2 .average-row th:first-child { border-top: 1.5px solid #111; }
  .report2 .average-section th:first-child { border-bottom: 2.5px solid #111; padding-right: 0.9rem; }
  .report2 .average-header th:nth-child(-n+5),
  .report2 .average-row:last-child th:nth-child(-n+5),
  .report2 .average-row:last-child td:nth-child(-n+4) { border-bottom: 2.5px solid #111; }
  .report2 .average-header th:nth-child(5),
  .report2 .average-row:last-child td:nth-child(4) { padding-right: 0.9rem; }
</style>

<table class="report2">
  <thead>
    <tr>
      <th rowspan="2" class="blank-corner"></th>
      <th colspan="4">machine-1-6</th>
      <th colspan="4">machine-3-4</th>
      <th colspan="4">machine-3-9</th>
    </tr>
    <tr>
      <th>VUS-PR</th><th>Aff. F1</th><th>Validation</th><th>Test</th>
      <th>VUS-PR</th><th>Aff. F1</th><th>Validation</th><th>Test</th>
      <th>VUS-PR</th><th>Aff. F1</th><th>Validation</th><th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>THESIS O0</th><td>0.798</td><td>0.747</td><td>0.012</td><td>0.250</td><td>0.705</td><td>0.708</td><td>0.013</td><td>0.432</td><td>0.578</td><td>0.789</td><td>0.013</td><td>0.045</td></tr>
    <tr><th>THESIS O1</th><td>0.799</td><td>0.748</td><td>0.006</td><td>0.089</td><td>0.704</td><td>0.708</td><td>0.019</td><td>0.573</td><td>0.578</td><td>0.793</td><td>0.024</td><td>0.061</td></tr>
  </tbody>
  <tbody>
    <tr class="average-section"><th colspan="5">Average</th><th colspan="8"></th></tr>
    <tr class="average-header">
      <th></th><th>VUS-PR</th><th>Aff. F1</th><th>Validation</th><th>Test</th><th colspan="8"></th>
    </tr>
    <tr class="average-row"><th>THESIS O0</th><td><strong>0.694</strong></td><td>0.748</td><td><strong>0.013</strong></td><td><strong>0.242</strong></td><td colspan="8"></td></tr>
    <tr class="average-row"><th>THESIS O1</th><td><strong>0.694</strong></td><td><strong>0.749</strong></td><td>0.016</td><td>0.241</td><td colspan="8"></td></tr>
  </tbody>
</table>

## Nguồn dữ liệu

- `outputs/reporting/offline_phase_tables/canonical_run_manifest.json`
- Trường dữ liệu: `report_fields.metrics.vus_pr`, `report_fields.metrics.affiliation_f1`, `report_fields.uq_summary.splits.clean_validation.mean_of_variances` và `report_fields.uq_summary.splits.test.mean_of_variances`.
