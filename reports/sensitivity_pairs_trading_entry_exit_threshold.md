# Pairs Trading Entry/Exit Threshold (AEP-FE) — Sensitivity Analysis

## Grid

| entry_threshold | exit_threshold | Result | OOS Sharpe | Permutation p | Agreement |
|---|---|---|---|---|---|
| 1.5 | 0 | AEP-FE | -0.42 | 0.6877 | Yes |
| 1.5 | 0.25 | AEP-FE | -0.13 | 0.5552 | Yes |
| 1.5 | 0.5 | AEP-FE | -0.18 | 0.5842 | Yes |
| 1.75 | 0 | AEP-FE | 0.11 | 0.4533 | Yes |
| 1.75 | 0.25 | AEP-FE | -0.68 | 0.7786 | Yes |
| 1.75 | 0.5 | AEP-FE | -0.66 | 0.7731 | Yes |
| 2 | 0 | AEP-FE (default) | -0.28 | 0.6342 | Yes |
| 2 | 0.25 | AEP-FE | -0.44 | 0.6887 | Yes |
| 2 | 0.5 | AEP-FE | -0.45 | 0.6942 | Yes |
| 2.25 | 0 | AEP-FE | -0.87 | 0.8591 | Yes |
| 2.25 | 0.25 | AEP-FE | -0.60 | 0.7721 | Yes |
| 2.25 | 0.5 | AEP-FE | -0.25 | 0.6132 | Yes |
| 2.5 | 0 | AEP-FE | -0.74 | 0.8161 | Yes |
| 2.5 | 0.25 | AEP-FE | -0.65 | 0.7921 | Yes |
| 2.5 | 0.5 | AEP-FE | -0.29 | 0.6407 | Yes |

## Consistency Summary

- Configurations with a result: 15/15
- Configurations with permutation p < 0.05: 0/15
- **Checking multiple parameter configurations is itself a form of multiple comparisons.** A single configuration's apparent success (or apparent significance) should not be treated as validating that specific parameter choice without correction. The default configuration was chosen a priori (before running this grid) based on independent, established conventions -- not tuned to this data. This report deliberately does not rank or highlight a "best" configuration.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*