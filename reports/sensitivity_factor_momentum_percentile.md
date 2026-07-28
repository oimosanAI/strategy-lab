# Factor Momentum Long/Short Percentile (Global Mode) — Sensitivity Analysis

## Grid

| percentile | Result | OOS Sharpe | Permutation p | Agreement |
|---|---|---|---|---|
| 0.1 | factor-momentum | -0.27 | 0.6392 | Yes |
| 0.2 | factor-momentum (default) | 0.44 | 0.2899 | Yes |
| 0.25 | factor-momentum | 0.52 | 0.2564 | Yes |
| 0.3 | factor-momentum | 0.66 | 0.1964 | Yes |
| 0.4 | factor-momentum | 0.24 | 0.3803 | Yes |

## Consistency Summary

- Configurations with a result: 5/5
- Configurations with permutation p < 0.05: 0/5
- **Checking multiple parameter configurations is itself a form of multiple comparisons.** A single configuration's apparent success (or apparent significance) should not be treated as validating that specific parameter choice without correction. The default configuration was chosen a priori (before running this grid) based on independent, established conventions -- not tuned to this data. This report deliberately does not rank or highlight a "best" configuration.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*