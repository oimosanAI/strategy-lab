# Factor Momentum: 12-1 Momentum + 60d Low-Volatility (Sector-Neutral) — Performance Report

**Period**: 2025-01-02 00:00:00 to 2026-07-24 00:00:00 (390 observations)

## What Was Tested

Full S&P 500 universe; sector-neutral quintile long/short (within each GICS sector), monthly rebalance, per-sector equal-budget normalization.

## Causality (Pillar 1)

assert_backtest_causal PASSED (n_trials=8, seed=0) on the full S&P 500 panel (503 tickers).

## Performance Metrics

| Metric | Value |
|---|---|
| Annualized Return | -0.11% |
| Annualized Volatility | 7.99% |
| Sharpe Ratio | 0.03 |
| Sortino Ratio | 0.04 |
| Calmar Ratio | -0.01 |
| Max Drawdown | -10.70% (216 days) |
| Win Rate | 53.08% |
| Profit/Loss Ratio | 0.89 |

## Statistical Significance (Pillar 2)

- Sign-flip permutation test: p = 0.4953 (n=2000, alternative=greater)
- Moving-block bootstrap 90% CI: [-0.0004, 0.0004] (n_resamples=2000; tail-matched to the greater test above, which is why a one-sided test is paired with a 90% and not a 95% interval)
- **Agreement: Yes**

## Walk-Forward / Over-fitting (Pillar 3)

Anchored split: IS Sharpe=1.470, OOS Sharpe=0.026. See strategies/factor_momentum/README.md for the full degradation record and, for sector-neutral mode, the global-vs-sector-neutral comparison.

## Known Limitations

- Level A survivorship bias (current S&P 500 constituents only).
- Sector-neutral construction reduces sector concentration (see strategies/factor_momentum/README.md Sec 3-3) but showed WORSE IS->OOS degradation than the global mode in this same evaluation window -- a documented trade-off, not a bug.
- min_names_per_sector=10 validated only partially on real data (no sector in this universe actually fell below the threshold during this run).
- Portfolio exposure limits are checked automatically inside run_backtest, but not enforced: the default BacktestConfig uses exposure_limits_strict=False, so a breach is recorded in BacktestResult.exposure_violations and warned about rather than raised. The default caps (max_gross=max_net=5.0) are a catastrophic sanity net, not a per-strategy-tuned constraint -- see BacktestConfig.exposure_limits.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*