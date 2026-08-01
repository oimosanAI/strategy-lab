# Strategy Comparison Report

**Strategies compared**: 2

## Period

| | Pairs Trading: AEP-FE (Static Hedge) | Factor Momentum: 12-1 Momentum + 60d Low-Volatility |
|---|---|---|
| Start | 2025-01-02 00:00:00 | 2025-01-02 00:00:00 |
| End | 2026-07-24 00:00:00 | 2026-07-24 00:00:00 |
| Observations | 390 | 390 |

## Causality (Pillar 1)

- **Pairs Trading: AEP-FE (Static Hedge)**: assert_backtest_causal PASSED (n_trials=8, seed=0) on the full AEP/FE panel.
- **Factor Momentum: 12-1 Momentum + 60d Low-Volatility**: assert_backtest_causal PASSED (n_trials=8, seed=0) on the full S&P 500 panel (503 tickers).

## Performance Metrics

| | Pairs Trading: AEP-FE (Static Hedge) | Factor Momentum: 12-1 Momentum + 60d Low-Volatility |
|---|---|---|
| Annualized Return | -3.50% | 4.12% |
| Annualized Volatility | 10.76% | 10.44% |
| Sharpe Ratio | -0.28 | 0.44 |
| Sortino Ratio | -0.44 | 0.61 |
| Calmar Ratio | -0.24 | 0.33 |
| Max Drawdown | -14.42% (352 days) | -12.52% (216 days) |
| Win Rate | 50.00% | 54.62% |
| Profit/Loss Ratio | 0.92 | 0.89 |

## Statistical Significance (Pillar 2)

| | Pairs Trading: AEP-FE (Static Hedge) | Factor Momentum: 12-1 Momentum + 60d Low-Volatility |
|---|---|---|
| Permutation p-value | 0.6342 | 0.2899 |
| Bootstrap CI | [-0.0007, 0.0005] | [-0.0004, 0.0007] |
| Agreement | Yes | Yes |

## Walk-Forward / Over-fitting (Pillar 3)

- **Pairs Trading: AEP-FE (Static Hedge)**: See strategies/pairs_trading/README.md for the full IS/OOS degradation record.
- **Factor Momentum: 12-1 Momentum + 60d Low-Volatility**: Anchored split: IS Sharpe=0.683, OOS Sharpe=0.439. See strategies/factor_momentum/README.md for the full degradation record and, for sector-neutral mode, the global-vs-sector-neutral comparison.

## Known Limitations

### Pairs Trading: AEP-FE (Static Hedge)

- Level A survivorship bias (current S&P 500 constituents only).
- Candidate pool: single surviving pair (AEP-FE) after full-universe Bonferroni correction across 5,551 tests -- see strategies/pairs_trading/README.md.
- Static hedge ratio frozen from in-sample OLS; not re-estimated OOS.

### Factor Momentum: 12-1 Momentum + 60d Low-Volatility

- Level A survivorship bias (current S&P 500 constituents only).
- Sector neutralization not applied in this mode -- see factor_momentum_sector_neutral.md for the sector-neutral comparison; the global mode's apparent outperformance may partly reflect incidental sector tilts (Utilities/Financials) rather than pure factor exposure.
- Portfolio exposure limits are checked automatically inside run_backtest, but not enforced: the default BacktestConfig uses exposure_limits_strict=False, so a breach is recorded in BacktestResult.exposure_violations and warned about rather than raised. The default caps (max_gross=max_net=5.0) are a catastrophic sanity net, not a per-strategy-tuned constraint -- see BacktestConfig.exposure_limits.

---
*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above before drawing any conclusion.*