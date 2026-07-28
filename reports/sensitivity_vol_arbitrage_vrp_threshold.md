# Vol Arbitrage VRP Threshold (VIX/SPY, SVXY) — Sensitivity Analysis

## Grid

| vrp_threshold | Result | OOS Sharpe | Permutation p | Agreement |
|---|---|---|---|---|
| -0.05 | vol-arbitrage | 0.06 | 0.4583 | Yes |
| -0.02 | vol-arbitrage | 0.29 | 0.3553 | Yes |
| 0 | vol-arbitrage (default) | 0.06 | 0.4638 | Yes |
| 0.02 | vol-arbitrage | 0.43 | 0.2949 | Yes |
| 0.05 | vol-arbitrage | 0.18 | 0.4093 | Yes |

## Consistency Summary

- Configurations with a result: 5/5
- Configurations with permutation p < 0.05: 0/5
- **Checking multiple parameter configurations is itself a form of multiple comparisons.** A single configuration's apparent success (or apparent significance) should not be treated as validating that specific parameter choice without correction. The default configuration was chosen a priori (before running this grid) based on independent, established conventions -- not tuned to this data. This report deliberately does not rank or highlight a "best" configuration.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*