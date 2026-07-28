"""Tests for strategies.vol_arbitrage.signal.

Fixes the public API surface:

- realized_volatility(prices, window=21) -> pd.Series: same rolling
  formula as strategies.factor_momentum.factors.low_volatility_score
  (annualized std of daily returns), reused here for SPY rather than
  duplicating the math.
- compute_vrp(vix, spy_prices, window=21) -> pd.Series: VIX (yfinance's
  ^VIX Close is quoted in PERCENTAGE POINTS, e.g. 18.67 meaning 18.67%
  annualized IV -- confirmed against real data before implementation)
  minus realized_volatility(spy_prices) (a decimal fraction, e.g. 0.05
  for 5%). VIX must be divided by 100 before subtracting -- otherwise
  VRP is dominated by VIX's raw ~15-30 scale and the RV term becomes
  numerically irrelevant, a "runs but means nothing" bug caught here
  before implementation, not after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.vol_arbitrage.signal import compute_vrp, realized_volatility


def test_realized_volatility_matches_hand_computed_std() -> None:
    # Same technique as test_factors.py's low_volatility_score test:
    # build prices from a known, hand-picked returns sequence so the
    # rolling std can be independently derived via numpy.
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03])
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    prices_values = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    prices = pd.Series(prices_values, index=idx)

    result = realized_volatility(prices, window=5)

    # pct_change() reproduces `returns` at rows 1..7 (row 0 is NaN). The
    # rolling(5) window at row 5 covers rows 1..5, i.e. returns[0:5] --
    # the first window excluding row 0's NaN entirely.
    expected = np.std(returns[0:5], ddof=1) * np.sqrt(252)
    assert result.iloc[5] == pytest.approx(expected)


def test_realized_volatility_is_nan_during_warmup() -> None:
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03])
    idx = pd.bdate_range("2020-01-01", periods=len(returns) + 1)
    prices_values = 100.0 * np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    prices = pd.Series(prices_values, index=idx)

    result = realized_volatility(prices, window=5)

    assert result.iloc[:5].isna().all()
    assert result.iloc[5:].notna().all()


def test_compute_vrp_converts_vix_percentage_points_to_decimal() -> None:
    # A constant-vol SPY series so realized_volatility is exactly known
    # in advance without needing to hand-derive a rolling std: zero
    # returns give exactly 0.0 realized volatility everywhere it's defined.
    idx = pd.bdate_range("2020-01-01", periods=10)
    spy_prices = pd.Series([100.0] * 10, index=idx)  # flat -> RV = 0.0 exactly
    vix = pd.Series([20.0] * 10, index=idx)  # 20.0 = "20.67-style" percentage points

    result = compute_vrp(vix, spy_prices, window=5)

    # VRP = (VIX / 100) - RV = 0.20 - 0.0 = 0.20 -- NOT 20.0 - 0.0 = 20.0,
    # which is what a missing /100 conversion would produce.
    assert result.iloc[-1] == pytest.approx(0.20)


def test_compute_vrp_is_nan_during_rv_warmup() -> None:
    idx = pd.bdate_range("2020-01-01", periods=10)
    spy_prices = pd.Series([100.0 + i for i in range(10)], index=idx)
    vix = pd.Series([20.0] * 10, index=idx)

    result = compute_vrp(vix, spy_prices, window=5)

    assert result.iloc[:5].isna().all()
    assert result.iloc[5:].notna().all()
