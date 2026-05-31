# Detail: Sampling Rules Chosen (Train Balanced + Val Realistic)

Date: 2026-05-30
Status: User-confirmed decisions

## Confirmed Decisions

1) Train per-batch balance
- Use `task.train_balance_classes: true` for relative class balancing within each training batch.
- Target classes: 12 classes (1 normal + 11 anomaly families).

2) Remainder allocation policy
- Use round-robin allocation across class indices to distribute leftover samples when `batch_size` is not divisible by 12.
- If `batch_size < 12`, rotate class coverage across consecutive batches.

3) Val realistic source options
- `task.val_realistic_source: test_same_scope`
- `task.val_realistic_source: test_smd_all`

4) Window-level anomaly definition for realistic rate
- A window is anomalous if it contains at least one anomalous point label.
- Compute with the exact current config windowization (`window_size`, `stride`).

5) Override behavior
- `task.val_anomaly_rate_override: null` means auto-derive anomaly rate from test prior source.
- Numeric value in `[0, 1]` means force that anomaly rate.

6) Val realistic anomaly-family distribution
- Keep the 11 anomaly families uniformly distributed (equal probability).

7) Minimal config fields (confirmed)
- `task.train_balance_classes: bool`
- `task.val_realistic: bool`
- `task.val_realistic_source: test_same_scope|test_smd_all`
- `task.val_anomaly_rate_override: float|null`

8) Backward compatibility policy
- Do not preserve old incompatible configs.
- Remove outdated configs if they no longer match the new strict semantics.

## Notes for Implementation
- Validation must fail hard on invalid combinations or invalid values.
- Error messages must include clear English fix instructions.
