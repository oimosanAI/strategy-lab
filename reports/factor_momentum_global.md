# Factor Momentum: 12-1 Momentum + 60d Low-Volatility — Performance Report

**Period**: 2025-01-02 00:00:00 to 2026-07-24 00:00:00 (390 observations)

## What Was Tested

Full S&P 500 universe; quintile long/short, monthly rebalance, leg-normalized.

## Causality (Pillar 1)

assert_backtest_causal PASSED (n_trials=8, seed=0) on the full S&P 500 panel (503 tickers).

## Performance Metrics

| Metric | Value |
|---|---|
| Annualized Return | 4.12% |
| Annualized Volatility | 10.44% |
| Sharpe Ratio | 0.44 |
| Sortino Ratio | 0.61 |
| Calmar Ratio | 0.33 |
| Max Drawdown | -12.52% (216 days) |
| Win Rate | 54.62% |
| Profit/Loss Ratio | 0.89 |

## Statistical Significance (Pillar 2)

- Sign-flip permutation test: p = 0.2899 (n=2000, alternative=greater)
- Moving-block bootstrap 95% CI: [-0.0005, 0.0008] (n_resamples=2000)
- **Agreement: Yes**

## Walk-Forward / Over-fitting (Pillar 3)

Anchored split: IS Sharpe=0.683, OOS Sharpe=0.439. See strategies/factor_momentum/README.md for the full degradation record and, for sector-neutral mode, the global-vs-sector-neutral comparison.

## Known Limitations

- Level A survivorship bias (current S&P 500 constituents only).
- Sector neutralization not applied in this mode -- see factor_momentum_sector_neutral.md for the sector-neutral comparison; the global mode's apparent outperformance may partly reflect incidental sector tilts (Utilities/Financials) rather than pure factor exposure.
- check_exposure_limits() is a post-hoc monitor, not enforced inside run_backtest.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*