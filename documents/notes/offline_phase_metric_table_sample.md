# Offline Phase Metric Table Sample

This is a layout sample only. No real values are used here.

Interpretation:
- Each cell is the mean over 3 seeds.
- Metrics are `VUS-PR`, `affiliation F1`, and `VUS-ROC`.
- Layout is modeled after the reference figure: methods on rows, entities on grouped columns.
- Best values may be bold, second-best may be underlined.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method + variant</th>
      <th colspan="3">machine_1_6</th>
      <th colspan="3">machine_3_4</th>
      <th colspan="3">machine_3_9</th>
    </tr>
    <tr>
      <th>VUS-PR</th><th>affiliation F1</th><th>VUS-ROC</th>
      <th>VUS-PR</th><th>affiliation F1</th><th>VUS-ROC</th>
      <th>VUS-PR</th><th>affiliation F1</th><th>VUS-ROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Thesis main + O0</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <th>Thesis main + O1</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <th>RedLamp + baseline</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <th>Traditional ML 1</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <th>Traditional ML 2</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
    <tr>
      <th>Traditional ML 3</th>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
      <td>--</td><td>--</td><td>--</td>
    </tr>
  </tbody>
</table>

## Suggested aggregation rule

For each `method + variant + entity`:

1. collect the 3 seed results,
2. compute the arithmetic mean per metric,
3. report the mean in the table cell,
4. optionally bold the best and underline the second best across methods.

## If you want the exact thesis style

Use one table per entity and keep methods as rows, or keep one wide table with grouped entity headers like above.
