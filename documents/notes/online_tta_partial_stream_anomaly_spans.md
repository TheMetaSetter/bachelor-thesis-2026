# SMD anomaly spans and partial online-TTA stream windows

## Purpose and interval convention

This note defines the labeled SMD test intervals used to select one short,
contiguous online test-time adaptation (TTA) stream for each target entity.
All indices are entity-local, 0-based, and half-open: `[start, end)` includes
`start` and excludes `end`.

The chosen stream contains the first three labeled anomaly spans for that
entity, with a 100-time-step prefix before the first selected span and a
100-time-step suffix after the third selected span. It is a partial-test
evaluation protocol, not full-test benchmark evidence. Preserve the absolute
indices in result provenance and report partial coverage explicitly.

| Entity | Test length | Selected anomaly spans | Absolute stream interval | Stream length | Rule |
|---|---:|---|---|---:|---|
| `machine-1-6` | 23,689 | `[246,252)`, `[653,658)`, `[2092,2100)` | `[146,2200)` | 2,054 | `246 - 100`, `2100 + 100` |
| `machine-3-4` | 23,687 | `[2734,3520)`, `[4474,4550)`, `[6013,6016)` | `[2634,6116)` | 3,482 | `2734 - 100`, `6016 + 100` |
| `machine-3-9` | 28,713 | `[1199,1230)`, `[5361,5487)`, `[10662,10707)` | `[1099,10807)` | 9,708 | `1199 - 100`, `10707 + 100` |

## Complete labeled anomaly spans

### `machine-1-6`

| No. | Span `[start,end)` | Length |
|---:|---|---:|
| 1 | `[246,252)` | 6 |
| 2 | `[653,658)` | 5 |
| 3 | `[2092,2100)` | 8 |
| 4 | `[2884,2888)` | 4 |
| 5 | `[3534,3539)` | 5 |
| 6 | `[4647,5045)` | 398 |
| 7 | `[5167,5172)` | 5 |
| 8 | `[5708,5713)` | 5 |
| 9 | `[5873,5885)` | 12 |
| 10 | `[6022,6027)` | 5 |
| 11 | `[6412,6419)` | 7 |
| 12 | `[7851,7856)` | 5 |
| 13 | `[9291,9298)` | 7 |
| 14 | `[10731,10736)` | 5 |
| 15 | `[11467,11471)` | 4 |
| 16 | `[12171,12176)` | 5 |
| 17 | `[13069,13073)` | 4 |
| 18 | `[13277,13280)` | 3 |
| 19 | `[13613,13619)` | 6 |
| 20 | `[14603,14607)` | 4 |
| 21 | `[15052,15055)` | 3 |
| 22 | `[15397,15401)` | 4 |
| 23 | `[15802,15805)` | 3 |
| 24 | `[16491,16499)` | 8 |
| 25 | `[16718,16721)` | 3 |
| 26 | `[16972,16976)` | 4 |
| 27 | `[17931,17939)` | 8 |
| 28 | `[18600,21761)` | 3,161 |
| 29 | `[22252,22260)` | 8 |
| 30 | `[22417,22420)` | 3 |

### `machine-3-4`

| No. | Span `[start,end)` | Length |
|---:|---|---:|
| 1 | `[2734,3520)` | 786 |
| 2 | `[4474,4550)` | 76 |
| 3 | `[6013,6016)` | 3 |
| 4 | `[10963,10969)` | 6 |
| 5 | `[11565,11569)` | 4 |
| 6 | `[13699,13709)` | 10 |
| 7 | `[18589,18640)` | 51 |
| 8 | `[18784,18825)` | 41 |

### `machine-3-9`

| No. | Span `[start,end)` | Length |
|---:|---|---:|
| 1 | `[1199,1230)` | 31 |
| 2 | `[5361,5487)` | 126 |
| 3 | `[10662,10707)` | 45 |
| 4 | `[27849,27950)` | 101 |

## Current runtime limitation

The checked-in online benchmark configs expose `max_online_steps`, which caps
the number of steps starting at the beginning of the test sequence. They do
not expose an absolute start/end slice for the test sequence. Therefore setting
only `max_online_steps` cannot produce the selected intervals above for all
three methods. A later implementation must add the same
`absolute_start_index` and `absolute_end_index` contract before windowization
to the shared sequence input used by THESIS, M2N2, and CANDI. It must retain
the original sequence length and absolute timestamps in metadata so evaluation
and reporting remain auditable.
