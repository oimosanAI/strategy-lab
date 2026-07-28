"""Tests for core.evaluation.metrics.

Fixes the public API surface core/evaluation/metrics.py must implement.
Deliberately decoupled from core.backtest: every function here takes a
plain pd.Series (or two, for benchmark comparison) of periodic returns,
never a BacktestResult. This lets the module be reused for anything that
produces a daily-return series -- live trading records included -- not
just this repo's own backtests.

Ported from the sibling backtest-framework repo (src/backtest/metrics.py),
verified to have no single-asset-specific assumptions (every function
operates purely on an already-realized return series, agnostic to how
many tickers produced it): annualized_return, annualized_volatility,
sharpe_ratio, max_drawdown, win_rate, profit_loss_ratio, summary(),
equity_curve(). Same annualization constants (252 trading days,
sqrt(252), ddof=1).

New for this project (not in the reference, per REQUIREMENTS.md 6):
sortino_ratio, calmar_ratio, benchmark_comparison (alpha/beta via OLS).

Test fixtures for annualized_return / max_drawdown reuse the exact
sanity-check cases the backtest-verification skill requires: a 2x-over-
2-years CAGR check (-> sqrt(2)-1) and a 50% drawdown fixture.

All expected values were derived independently of the implementation
(see the conversation record for the standalone verification script that
computed them -- no import of core.evaluation).

IMPORTANT (per the backtest-verification skill): these metrics are not,
by themselves, evidence of a real edge. A Sharpe/Sortino number is only
reportable after Pillar 1 (causality, assert_causal) and Pillar 2
(significance, statistical_tests.py) have been checked -- this module
computes the numbers, it does not certify them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.evaluation.metrics import (
    BenchmarkComparison,
    annualized_return,
    annualized_volatility,
    benchmark_comparison,
    calmar_ratio,
    equity_curve,
    max_drawdown,
    profit_loss_ratio,
    sharpe_ratio,
    sortino_ratio,
    summary,
    win_rate,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


# ---------------------------------------------------------------------------
# annualized_return / max_drawdown: skill-mandated sanity fixtures
# ---------------------------------------------------------------------------


def test_annualized_return_doubles_over_two_years_gives_sqrt2_minus_1() -> None:
    # Arrange: a single +100% day, then 503 flat days -> exactly 2x over
    # 504 = 2*252 trading days, regardless of the path taken to get there.
    returns = _series([1.0] + [0.0] * 503)

    # Act
    result = annualized_return(returns)

    # Assert
    assert result == pytest.approx(np.sqrt(2) - 1)


def test_max_drawdown_50_percent_fixture() -> None:
    # Arrange: wealth goes 1.0 -> 2.0 -> 1.0, a clean 50% peak-to-trough decline.
    returns = _series([1.0, -0.5])

    # Act
    result = max_drawdown(returns)

    # Assert
    assert result.max_drawdown == pytest.approx(-0.5)
    assert result.duration == 1


def test_max_drawdown_is_zero_for_empty_series() -> None:
    # Arrange
    returns = pd.Series([], dtype=float)

    # Act
    result = max_drawdown(returns)

    # Assert
    assert result.max_drawdown == 0.0
    assert result.duration == 0


def test_annualized_return_is_nan_for_empty_series() -> None:
    # Arrange
    returns = pd.Series([], dtype=float)

    # Act / Assert
    assert np.isnan(annualized_return(returns))


def test_annualized_return_is_negative_100_percent_on_total_wipeout() -> None:
    # Arrange: a single -100% day wipes out the account entirely.
    returns = _series([-1.0])

    # Act / Assert
    assert annualized_return(returns) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# equity_curve
# ---------------------------------------------------------------------------


def test_equity_curve_compounds_returns_from_initial_capital() -> None:
    # Arrange
    returns = _series([0.1, -0.05])

    # Act
    result = equity_curve(returns, initial=1.0)

    # Assert
    assert result.tolist() == pytest.approx([1.1, 1.045])


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_is_nan_for_fewer_than_two_observations() -> None:
    # Arrange
    returns = _series([0.01])

    # Act / Assert
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_hand_computable() -> None:
    # Arrange
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0])

    # Act
    result = sharpe_ratio(returns)

    # Assert: mean=0.01, std(ddof=1)=0.015811388300841896
    assert result == pytest.approx(10.039920318408905)


def test_sharpe_ratio_is_nan_when_std_is_zero() -> None:
    # Arrange
    returns = _series([0.01] * 5)

    # Act / Assert
    assert np.isnan(sharpe_ratio(returns))


# ---------------------------------------------------------------------------
# annualized_volatility
# ---------------------------------------------------------------------------


def test_annualized_volatility_hand_computable() -> None:
    # Arrange
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0])

    # Act
    result = annualized_volatility(returns)

    # Assert: std(ddof=1)=0.015811388300841896, * sqrt(252)
    assert result == pytest.approx(0.25099800796022265)


def test_annualized_volatility_is_nan_for_fewer_than_two_observations() -> None:
    # Arrange
    returns = _series([0.01])

    # Act / Assert
    assert np.isnan(annualized_volatility(returns))


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_is_nan_for_fewer_than_two_observations() -> None:
    # Arrange
    returns = _series([0.01])

    # Act / Assert
    assert np.isnan(sortino_ratio(returns))


def test_sortino_ratio_hand_computable() -> None:
    # Arrange
    returns = _series([0.05, -0.02, 0.03, -0.01, 0.04])

    # Act
    result = sortino_ratio(returns)

    # Assert: mean=0.018, downside_dev (over ALL periods, upside clipped to
    # 0) = 0.01
    assert result == pytest.approx(28.574114159497576)


def test_sortino_exceeds_sharpe_when_returns_are_upside_skewed() -> None:
    # Arrange: one large gain, several small consistent losses -- Sortino
    # should reward this shape much more than Sharpe, which penalizes the
    # large (desirable) upside move as if it were risk.
    returns = _series([0.10, -0.01, -0.01, -0.01, -0.01])

    # Act
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)

    # Assert
    assert sharpe == pytest.approx(3.8723431307561142)
    assert sortino == pytest.approx(21.29788721915862)
    assert sortino > sharpe


def test_sortino_ratio_is_nan_when_no_downside() -> None:
    # Arrange: every period positive -> downside deviation is exactly 0.
    returns = _series([0.01, 0.02, 0.03])

    # Act / Assert
    assert np.isnan(sortino_ratio(returns))


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


def test_calmar_ratio_hand_computable() -> None:
    # Arrange: flat for 251 days, one -20% day, flat for the remaining 252
    # -- 504 observations total (2 years), one clean drawdown.
    returns = _series([0.0] * 251 + [-0.2] + [0.0] * 252)

    # Act
    result = calmar_ratio(returns)

    # Assert: CAGR = 0.8**0.5 - 1, max_drawdown = -0.2
    assert result == pytest.approx(-0.5278640450004208)


def test_calmar_ratio_is_nan_when_no_drawdown() -> None:
    # Arrange: monotonically increasing equity curve.
    returns = _series([0.01] * 10)

    # Act / Assert
    assert np.isnan(calmar_ratio(returns))


# ---------------------------------------------------------------------------
# win_rate / profit_loss_ratio
# ---------------------------------------------------------------------------


def test_win_rate_and_profit_loss_ratio_hand_computable() -> None:
    # Arrange: 3 wins, 2 losses, 1 flat (excluded from win_rate's denominator).
    returns = _series([0.02, -0.01, 0.03, 0.0, -0.02, 0.01])

    # Act
    wr = win_rate(returns)
    plr = profit_loss_ratio(returns)

    # Assert: 3 of 5 non-flat periods are wins; mean win=0.02, mean |loss|=0.015
    assert wr == pytest.approx(0.6)
    assert plr == pytest.approx(1.3333333333333335)


def test_win_rate_is_nan_when_all_periods_are_flat() -> None:
    # Arrange
    returns = _series([0.0, 0.0, 0.0])

    # Act / Assert
    assert np.isnan(win_rate(returns))


def test_profit_loss_ratio_is_nan_when_no_losers() -> None:
    # Arrange: every non-zero period is a win.
    returns = _series([0.01, 0.02, 0.0])

    # Act / Assert
    assert np.isnan(profit_loss_ratio(returns))


def test_profit_loss_ratio_is_nan_when_no_winners() -> None:
    # Arrange: every non-zero period is a loss.
    returns = _series([-0.01, -0.02, 0.0])

    # Act / Assert
    assert np.isnan(profit_loss_ratio(returns))


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_returns_all_expected_keys() -> None:
    # Arrange
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0])

    # Act
    result = summary(returns)

    # Assert
    assert set(result.keys()) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "max_drawdown_duration",
        "win_rate",
        "profit_loss_ratio",
    }


# ---------------------------------------------------------------------------
# benchmark_comparison
# ---------------------------------------------------------------------------


def test_benchmark_comparison_recovers_known_alpha_and_beta() -> None:
    # Arrange: strategy_returns constructed EXACTLY as
    # known_alpha_daily + known_beta * benchmark_returns, with zero noise,
    # so OLS must recover the construction parameters (near) exactly.
    known_alpha_daily = 0.0002
    known_beta = 1.5
    benchmark_returns = _series([0.01, -0.02, 0.03, 0.005, -0.01, 0.02, -0.015, 0.008, 0.012, -0.005])
    strategy_returns = known_alpha_daily + known_beta * benchmark_returns

    # Act
    result = benchmark_comparison(strategy_returns, benchmark_returns)

    # Assert
    assert isinstance(result, BenchmarkComparison)
    assert result.beta == pytest.approx(known_beta, abs=1e-8)
    assert result.alpha == pytest.approx(known_alpha_daily * 252, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-8)


def test_benchmark_comparison_raises_on_insufficient_overlap() -> None:
    # Arrange: only one overlapping date between the two series.
    strategy_returns = pd.Series([0.01], index=pd.bdate_range("2020-01-01", periods=1))
    benchmark_returns = pd.Series([0.02], index=pd.bdate_range("2020-01-01", periods=1))

    # Act / Assert
    with pytest.raises(ValueError):
        benchmark_comparison(strategy_returns, benchmark_returns)
