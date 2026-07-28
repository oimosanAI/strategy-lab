"""Tests for strategies.factor_momentum.factors.

Fixes the public API surface:

- momentum_score(prices, lookback_days=252, skip_days=21) -> pd.DataFrame:
  12-1 momentum, price[t - skip_days] / price[t - lookback_days] - 1. NaN
  during the warm-up period (t < lookback_days), matching the
  strategies.pairs_trading.signal.rolling_zscore precedent of leaving
  warm-up rows as NaN rather than inventing a value.
- low_volatility_score(prices, window=60) -> pd.DataFrame: rolling
  annualized realized volatility of daily returns (same formula as
  core.backtest.position_sizing.VolTargetSizer's realized_vol, reused
  here as a ranking score rather than a sizing input). NaN during warm-up.

Both are pure rolling-window functions -- causal by construction, the
same pattern as rolling_zscore. Expected values are derived independently
via numpy (np.std with ddof=1, hand-picked ratios), never by running the
implementation and copying its output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.factor_momentum.factors import low_volatility_score, momentum_score


def test_momentum_score_matches_hand_computed_ratio() -> None:
    # Arrange: a simple increasing price series, small lookback/skip so the
    # exact ratio at a specific date can be verified by hand.
    idx = pd.bdate_range("2020-01-01", periods=15)
    prices = pd.DataFrame(
        {
            "A": [100.0 + i for i in range(15)],
            "B": [200.0 + 2 * i for i in range(15)],  # same shape, different scale
        },
        index=idx,
    )

    # Act
    result = momentum_score(prices, lookback_days=10, skip_days=2)

    # Assert: at t=10 (0-indexed), score = price[t-2] / price[t-10] - 1
    t = 10
    expected_a = prices["A"].iloc[t - 2] / prices["A"].iloc[t - 10] - 1.0
    expected_b = prices["B"].iloc[t - 2] / prices["B"].iloc[t - 10] - 1.0
    assert result["A"].iloc[t] == pytest.approx(expected_a)
    assert result["B"].iloc[t] == pytest.approx(expected_b)


def test_momentum_score_is_nan_during_warmup() -> None:
    idx = pd.bdate_range("2020-01-01", periods=15)
    prices = pd.DataFrame({"A": [100.0 + i for i in range(15)]}, index=idx)

    result = momentum_score(prices, lookback_days=10, skip_days=2)

    # Defined starting exactly at row index 10 (0-indexed): rows 0..9 NaN.
    assert result["A"].iloc[:10].isna().all()
    assert result["A"].iloc[10:].notna().all()


def test_momentum_score_is_scale_invariant() -> None:
    # A momentum ratio should be identical for a price series and its
    # constant multiple (e.g. a stock split / currency rescaling) --
    # confirms the formula is a pure ratio, not accidentally
    # level-dependent.
    idx = pd.bdate_range("2020-01-01", periods=15)
    base = pd.Series([100.0 + i for i in range(15)], index=idx, name="A")
    prices = pd.DataFrame({"A": base, "B": base * 3.7})

    result = momentum_score(prices, lookback_days=10, skip_days=2)

    pd.testing.assert_series_equal(result["A"], result["B"], check_names=False)


def test_low_volatility_score_matches_hand_computed_std() -> None:
    # Arrange: build prices from a known, hand-picked returns sequence so
    # the rolling std can be independently derived via numpy.
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03])
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    prices_values = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    prices = pd.DataFrame({"A": prices_values}, index=idx)

    # Act
    result = low_volatility_score(prices, window=5)

    # Assert: prices.pct_change() reproduces `returns` at rows 1..7 (row 0
    # is NaN). The rolling(5) window at row 5 covers rows 1..5, i.e.
    # returns[0:5] -- the first window that excludes row 0's NaN entirely.
    expected = np.std(returns[0:5], ddof=1) * np.sqrt(252)
    assert result["A"].iloc[5] == pytest.approx(expected)


def test_low_volatility_score_is_nan_during_warmup() -> None:
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03])
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    prices_values = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    prices = pd.DataFrame({"A": prices_values}, index=idx)

    result = low_volatility_score(prices, window=5)

    # Rows 0..4 all fall in windows that still include row 0's NaN return
    # -> NaN (min_periods=5 default). Row 5 is the first fully-valid window.
    assert result["A"].iloc[:5].isna().all()
    assert result["A"].iloc[5:].notna().all()
