# Pairs Trading: AEP-FE (Static Hedge) — Performance Report

**Period**: 2025-01-02 00:00:00 to 2026-07-24 00:00:00 (390 observations)

## What Was Tested

S&P 500 Utilities/Communication Services subset; single pair, static hedge.

## Causality (Pillar 1)

assert_backtest_causal PASSED (n_trials=8, seed=0) on the full AEP/FE panel.

## Performance Metrics

| Metric | Value |
|---|---|
| Annualized Return | -3.50% |
| Annualized Volatility | 10.76% |
| Sharpe Ratio | -0.28 |
| Sortino Ratio | -0.44 |
| Calmar Ratio | -0.24 |
| Max Drawdown | -14.42% (352 days) |
| Win Rate | 50.00% |
| Profit/Loss Ratio | 0.92 |

## Statistical Significance (Pillar 2)

- Sign-flip permutation test: p = 0.6342 (n=2000, alternative=greater)
- Moving-block bootstrap 90% CI: [-0.0007, 0.0005] (n_resamples=2000; tail-matched to the greater test above, which is why a one-sided test is paired with a 90% and not a 95% interval)
- **Agreement: Yes**

## Walk-Forward / Over-fitting (Pillar 3)

See strategies/pairs_trading/README.md for the full IS/OOS degradation record.

## Known Limitations

- Level A survivorship bias (current S&P 500 constituents only).
- Candidate pool: AEP-FE is the single pair surviving Bonferroni correction within the NARROW 2-sector scan (308 pairs = 616 tests, adjusted p=0.0472 -- barely inside alpha=0.05). It does NOT survive the full-universe family (5,551 pairs = 11,102 tests), where zero pairs survive. The family size counts 2 tests per pair because engle_granger_test selects the stronger of both regression directions -- see strategies/pairs_trading/README.md Step F.
- Static hedge ratio frozen from in-sample OLS; not re-estimated OOS.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*