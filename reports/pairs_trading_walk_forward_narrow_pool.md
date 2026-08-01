# Pairs Trading (Narrow 2-Sector Pool -- REFERENCE ONLY, not full-universe corrected) — Walk-Forward Report

## Windows

| Window | IS End | OOS Period | Result | OOS Sharpe | Permutation p | Agreement |
|---|---|---|---|---|---|---|
| 1 | 2024-06-28 00:00:00 | 2024-07-01 00:00:00 to 2024-12-31 00:00:00 | META-WBD | 2.13 | 0.0730 | No |
| 2 | 2024-12-31 00:00:00 | 2025-01-02 00:00:00 to 2025-06-30 00:00:00 | AEP-FE | 0.94 | 0.3038 | Yes |
| 3 | 2025-06-30 00:00:00 | 2025-07-01 00:00:00 to 2025-12-31 00:00:00 | no statistically-justified pair survived selection | N/A | N/A | N/A |
| 4 | 2025-12-31 00:00:00 | 2026-01-02 00:00:00 to 2026-07-24 00:00:00 | no statistically-justified pair survived selection | N/A | N/A | N/A |

## Consistency Summary

- Windows with a result: 2/4
- Windows with no tested result: 2/4 -- these were never tested, which is not the same as tested and unremarkable:
  - Window 3: no statistically-justified pair survived selection
  - Window 4: no statistically-justified pair survived selection
- OOS Sharpe among populated windows: mean=1.53, min=0.94, max=2.13
- Windows with permutation p < 0.05: 0/2 -- checking multiple windows is itself a form of multiple comparisons; a single window's apparent significance should not be treated as confirmed edge without correction.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*