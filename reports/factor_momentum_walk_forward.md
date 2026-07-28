# Factor Momentum (Global Mode) — Walk-Forward Report

## Windows

| Window | IS End | OOS Period | Result | OOS Sharpe | Permutation p | Agreement |
|---|---|---|---|---|---|---|
| 1 | 2024-06-28 00:00:00 | 2024-07-01 00:00:00 to 2024-12-31 00:00:00 | factor-momentum (global) | 0.94 | 0.2574 | Yes |
| 2 | 2024-12-31 00:00:00 | 2025-01-02 00:00:00 to 2025-06-30 00:00:00 | factor-momentum (global) | -0.09 | 0.5152 | Yes |
| 3 | 2025-06-30 00:00:00 | 2025-07-01 00:00:00 to 2025-12-31 00:00:00 | factor-momentum (global) | -0.92 | 0.7391 | Yes |
| 4 | 2025-12-31 00:00:00 | 2026-01-02 00:00:00 to 2026-07-24 00:00:00 | factor-momentum (global) | 1.86 | 0.0950 | Yes |

## Consistency Summary

- Windows with a result: 4/4
- OOS Sharpe among populated windows: mean=0.45, min=-0.92, max=1.86
- Windows with permutation p < 0.05: 0/4 -- checking multiple windows is itself a form of multiple comparisons; a single window's apparent significance should not be treated as confirmed edge without correction.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*