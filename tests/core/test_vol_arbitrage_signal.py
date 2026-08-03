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
- compute_term_structure_ratio(vix9d, vix) -> pd.Series: VIX9D / VIX,
  no window (an instantaneous ratio, not a rolling quantity -- see
  strategies/vol_arbitrage/README.md Step F/§3-7 for why this was added
  to address trailing RV's inability to de-risk ahead of a vol spike).
  UNLIKE compute_vrp, no unit conversion is needed: yfinance's ^VIX9D
  and ^VIX are both quoted in the same percentage-point scale (confirmed
  against real data -- mean ~16.5 vs ~17.4 over 2023-2026 -- before
  implementation), so the ratio is scale-consistent as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.vol_arbitrage.signal import compute_term_structure_ratio, compute_vrp, realized_volatility


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


def test_compute_term_structure_ratio_matches_hand_computed_division() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5)
    vix9d = pd.Series([15.0, 18.0, 20.0, 12.0, 25.0], index=idx)
    vix = pd.Series([20.0, 20.0, 20.0, 20.0, 20.0], index=idx)

    result = compute_term_structure_ratio(vix9d, vix)

    expected = pd.Series([0.75, 0.90, 1.00, 0.60, 1.25], index=idx)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_compute_term_structure_ratio_needs_no_unit_conversion() -> None:
    # Both ^VIX9D and ^VIX are quoted in the same percentage-point scale
    # (unlike VIX vs realized_volatility in compute_vrp) -- a naive /100
    # on only one side would be the equivalent numerically-wrong bug here.
    idx = pd.bdate_range("2020-01-01", periods=1)
    vix9d = pd.Series([18.0], index=idx)
    vix = pd.Series([20.0], index=idx)

    result = compute_term_structure_ratio(vix9d, vix)

    assert result.iloc[0] == pytest.approx(0.9)


def test_compute_term_structure_ratio_has_no_warmup_period() -> None:
    # Unlike compute_vrp (which depends on a rolling RV window and is
    # therefore NaN for the first `window` rows), the ratio is a same-row
    # quantity with no lookback -- every row with valid inputs is valid.
    idx = pd.bdate_range("2020-01-01", periods=3)
    vix9d = pd.Series([15.0, 16.0, 17.0], index=idx)
    vix = pd.Series([20.0, 20.0, 20.0], index=idx)

    result = compute_term_structure_ratio(vix9d, vix)

    assert result.notna().all()
