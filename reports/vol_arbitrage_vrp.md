# Vol Arbitrage: VIX/SPY Volatility Risk Premium (SVXY, long-only) — Performance Report

**Period**: 2025-01-02 00:00:00 to 2026-07-24 00:00:00 (390 observations)

## What Was Tested

VRP(t) = VIX(t)/100 - realized_volatility(SPY, window=21). Long SVXY when VRP > 0.0, flat otherwise. VolTargetSizer (target_vol=15%, vol_window=60, max_leverage=2x). Costs: 0.5bps commission + 5.0bps slippage per REQUIREMENTS.md 7.1.

## Causality (Pillar 1)

assert_backtest_causal PASSED (n_trials=8, seed=0) on the full VIX/SPY/SVXY panel (2023-01-03 to 2026-07-24).

## Performance Metrics

| Metric | Value |
|---|---|
| Annualized Return | -0.26% |
| Annualized Volatility | 15.57% |
| Sharpe Ratio | 0.06 |
| Sortino Ratio | 0.08 |
| Calmar Ratio | -0.02 |
| Max Drawdown | -14.22% (34 days) |
| Win Rate | 55.39% |
| Profit/Loss Ratio | 0.82 |

## Statistical Significance (Pillar 2)

- Sign-flip permutation test: p = 0.4638 (n=2000, alternative=greater)
- Moving-block bootstrap 95% CI: [-0.0008, 0.0008] (n_resamples=2000)
- **Agreement: Yes**

## Walk-Forward / Over-fitting (Pillar 3)

Single anchored split only (walk-forward not yet performed for this strategy): IS Sharpe=0.307, OOS Sharpe=0.061. Degradation = 0.246. Buy & Hold SVXY OOS Sharpe (no signal, no cost model) = 0.411, cumulative return=13.50%, vs strategy cumulative return=-0.41%.

## Known Limitations

- Level A survivorship bias: SVXY's own leverage was reduced from -1x to -0.5x after the 2018 Volmageddon event; this backtest only covers 2023-2026, so that structural risk is not reflected in-sample even though it remains a real, documented tail risk for this instrument.
- vrp_threshold=0.0 has no strong literature backing -- see strategies/vol_arbitrage/README.md Sec 3-4; the strategy's signal uses TRAILING realized volatility, a different (causal) measure from the forward-looking realized volatility the academic VRP literature usually reports against.
- Long-only design: no short-vol-to-long-vol reversal when VRP is negative, due to this project's cost model having no borrow-cost modeling.
- A single Memorial Day holiday row (2026-05-25) was dropped from the panel as a data-hygiene step (both SVXY and SPY showed NaN on that date), not a results-shaping exclusion.
- No walk-forward or parameter sensitivity analysis has been performed yet for this strategy (unlike pairs_trading/factor_momentum) -- see Known Limitations in strategies/vol_arbitrage/README.md.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*