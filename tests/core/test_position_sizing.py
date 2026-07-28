"""Tests for core.backtest.position_sizing and portfolio.check_exposure_limits.

Fixes the public API surface that core/backtest/position_sizing.py must
implement:

- PositionSizingConfig: target_vol, vol_window, kelly_window, kelly_fraction,
  max_leverage, min_var_threshold.
- PositionSizer (Protocol): apply(signals, prices) -> DataFrame, same
  (date, ticker) shape as both inputs. Causal: row t depends only on
  rows <= t of signals and prices.
- VolTargetSizer: weight = signal * (target_vol / realized_vol), clipped to
  +/- max_leverage, flat during warm-up.
- HalfKellySizer: weight = signal * (kelly_fraction * mu/var) of the
  signal's own historical strategy_return, clipped to the same
  max_leverage, and additionally floored to flat whenever var falls below
  min_var_threshold (a var-based reciprocal is far more explosive near
  zero than vol-targeting's single reciprocal, hence the extra floor).

Role separation (documented, tested implicitly through these cases):
max_leverage is a PER-TICKER safety net against a single position's
estimation blowing up; it says nothing about portfolio-level gross/net
exposure, which is what core.backtest.portfolio.check_exposure_limits
checks separately and explicitly.

Also fixes core/backtest/portfolio.py's exposure-limit surface:

- ExposureLimits(max_gross, max_net, max_per_ticker)
- ExposureViolation(date, kind, ticker, value, limit)
- check_exposure_limits(positions, limits) -> list[ExposureViolation]
  Never raises; returns facts only. Policy (warn/raise/ignore) is the
  caller's decision, deferred deliberately since REQUIREMENTS.md 4.5's
  risk-response policy isn't finalized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal
from core.backtest.portfolio import ExposureLimits, check_exposure_limits
from core.backtest.position_sizing import (
    HalfKellySizer,
    PositionSizer,
    PositionSizingConfig,
    VolTargetSizer,
)


def _geometric_prices(n_days: int, daily_return: float, start: float = 100.0) -> pd.DataFrame:
    """A single-ticker price path with an EXACTLY constant daily return,
    so realized volatility over any window is exactly zero -- useful for
    forcing the leverage-cap path deterministically."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    values = start * (1.0 + daily_return) ** np.arange(n_days)
    return pd.DataFrame({"AAPL": values}, index=idx)


def _constant_signal(prices: pd.DataFrame, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(value, index=prices.index, columns=prices.columns)


# ---------------------------------------------------------------------------
# Test doubles (negative control for the sizer-level causality guard)
# ---------------------------------------------------------------------------


class _SizerAsStrategy:
    """Adapts a PositionSizer + fixed signals into the Strategy protocol so
    the existing assert_causal can be reused unchanged to verify a sizer's
    own causality with respect to price history (e.g. a volatility or
    Kelly estimation window). No new perturbation function needed."""

    name = "sizer-under-test"

    def __init__(self, sizer: PositionSizer, signals: pd.DataFrame) -> None:
        self.sizer = sizer
        self.signals = signals

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.sizer.apply(self.signals.reindex(prices.index), prices)


class _FullSampleVolSizer:
    """Deliberately non-causal: estimates realized volatility using the
    FULL sample (std over the whole series, not a rolling window) and
    applies it as a constant to every date. Mirrors the full-sample-OLS
    pitfall from test_engine.py, one layer down in the sizing module --
    proves the _SizerAsStrategy adapter has real detection power, not
    just that a correct sizer happens to pass."""

    name = "full-sample-vol (deliberately non-causal)"

    def apply(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        full_sample_vol = prices.pct_change().std() * np.sqrt(252)  # uses ALL rows
        return signals * (0.15 / full_sample_vol)


# ---------------------------------------------------------------------------
# VolTargetSizer
# ---------------------------------------------------------------------------


def test_vol_target_sizer_is_flat_during_warmup() -> None:
    # Arrange: fewer bars than vol_window means realized_vol is NaN.
    config = PositionSizingConfig(vol_window=60)
    prices = _geometric_prices(30, daily_return=0.001)
    signals = _constant_signal(prices)
    sizer = VolTargetSizer(config)

    # Act
    weight = sizer.apply(signals, prices)

    # Assert
    assert (weight["AAPL"] == 0.0).all()


def test_vol_target_sizer_clips_to_max_leverage_when_vol_near_zero() -> None:
    # Arrange: an exactly-constant daily return gives realized_vol == 0
    # exactly, so raw_weight = signal * (target_vol / 0) = +inf. This must
    # be clipped, not left to blow up or silently become NaN.
    config = PositionSizingConfig(vol_window=20, max_leverage=2.0)
    prices = _geometric_prices(25, daily_return=0.001)
    signals = _constant_signal(prices, value=1.0)
    sizer = VolTargetSizer(config)

    # Act
    weight = sizer.apply(signals, prices)

    # Assert: clipped to exactly the cap, sign preserved from the signal.
    assert weight["AAPL"].iloc[-1] == pytest.approx(2.0)


def test_vol_target_sizer_passes_assert_causal() -> None:
    # Arrange
    config = PositionSizingConfig(vol_window=20)
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = pd.DataFrame(
        {"AAPL": 100 + np.cumsum(rng.normal(0, 1, 120)), "MSFT": 100 + np.cumsum(rng.normal(0, 1, 120))},
        index=idx,
    ).abs() + 1.0
    signals = _constant_signal(prices, value=1.0)

    # Act / Assert: perturbing future prices must not change past weights.
    assert_causal(_SizerAsStrategy(VolTargetSizer(config), signals), prices, n_trials=12)


def test_full_sample_vol_sizer_is_rejected_by_assert_causal() -> None:
    # Arrange: the guard must actually bite for a broken sizer, not just
    # pass for a correct one -- otherwise the "positive control" test above
    # proves nothing.
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = pd.DataFrame({"AAPL": 100 + np.cumsum(rng.normal(0, 1, 120))}, index=idx).abs() + 1.0
    signals = _constant_signal(prices, value=1.0)

    # Act / Assert
    with pytest.raises(LookAheadError):
        assert_causal(_SizerAsStrategy(_FullSampleVolSizer(), signals), prices, n_trials=12)


# ---------------------------------------------------------------------------
# HalfKellySizer
# ---------------------------------------------------------------------------


def test_half_kelly_sizer_is_flat_during_warmup() -> None:
    # Arrange
    config = PositionSizingConfig(kelly_window=20)
    prices = _geometric_prices(10, daily_return=0.001)
    signals = _constant_signal(prices)
    sizer = HalfKellySizer(config)

    # Act
    weight = sizer.apply(signals, prices)

    # Assert
    assert (weight["AAPL"] == 0.0).all()


def test_half_kelly_sizer_floors_to_flat_when_variance_below_threshold() -> None:
    # Arrange: daily-return noise with std=1e-5 -> var ~= 1e-10, far below
    # the default min_var_threshold (1e-6). A raw mu/var here would still
    # be numerically finite (not inf/NaN) but is not a trustworthy
    # estimate -- this must floor to flat regardless of what mu/var says.
    rng = np.random.default_rng(7)
    tiny_returns = rng.normal(0, 1e-5, 25)
    prices = pd.DataFrame({"AAPL": 100 * np.cumprod(1.0 + tiny_returns)}, index=pd.bdate_range("2020-01-01", periods=25))
    signals = _constant_signal(prices, value=1.0)
    config = PositionSizingConfig(kelly_window=20, min_var_threshold=1e-6)
    sizer = HalfKellySizer(config)

    # Act
    weight = sizer.apply(signals, prices)

    # Assert
    assert weight["AAPL"].iloc[-1] == 0.0


def test_half_kelly_sizer_clips_to_max_leverage() -> None:
    # Arrange: a clearly profitable, low-variance-but-not-degenerate signal
    # should produce a Kelly fraction well beyond max_leverage, forcing the
    # clip to engage.
    idx = pd.bdate_range("2020-01-01", periods=25)
    # Alternating small returns: positive mean, small-but-not-floored variance.
    returns = np.array([0.01 if i % 2 == 0 else 0.008 for i in range(25)])
    prices = pd.DataFrame({"AAPL": 100 * np.cumprod(1.0 + returns)}, index=idx)
    signals = _constant_signal(prices, value=1.0)
    config = PositionSizingConfig(kelly_window=20, max_leverage=2.0, min_var_threshold=1e-9)
    sizer = HalfKellySizer(config)

    # Act
    weight = sizer.apply(signals, prices)

    # Assert
    assert weight["AAPL"].iloc[-1] == pytest.approx(2.0)


def test_half_kelly_sizer_passes_assert_causal() -> None:
    # Arrange
    config = PositionSizingConfig(kelly_window=20)
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2020-01-01", periods=120)
    prices = pd.DataFrame({"AAPL": 100 + np.cumsum(rng.normal(0, 1, 120))}, index=idx).abs() + 1.0
    signals = _constant_signal(prices, value=1.0)

    # Act / Assert
    assert_causal(_SizerAsStrategy(HalfKellySizer(config), signals), prices, n_trials=12)


# ---------------------------------------------------------------------------
# check_exposure_limits: reports facts, never raises, policy left to caller
# ---------------------------------------------------------------------------


def test_check_exposure_limits_detects_gross_violation() -> None:
    # Arrange: gross = |0.8| + |0.8| = 1.6 > 1.0
    positions = pd.DataFrame({"A": [0.8], "B": [0.8]}, index=pd.bdate_range("2020-01-01", periods=1))
    limits = ExposureLimits(max_gross=1.0)

    # Act
    violations = check_exposure_limits(positions, limits)

    # Assert
    assert len(violations) == 1
    assert violations[0].kind == "gross"


def test_check_exposure_limits_detects_net_violation() -> None:
    # Arrange: net = 0.9 + 0.8 = 1.7 > 1.0 (both long, no offset)
    positions = pd.DataFrame({"A": [0.9], "B": [0.8]}, index=pd.bdate_range("2020-01-01", periods=1))
    limits = ExposureLimits(max_net=1.0)

    # Act
    violations = check_exposure_limits(positions, limits)

    # Assert
    assert any(v.kind == "net" for v in violations)


def test_check_exposure_limits_detects_per_ticker_violation() -> None:
    # Arrange: ticker A at 0.5 exceeds a 0.2 per-name cap; B does not.
    positions = pd.DataFrame({"A": [0.5], "B": [-0.05]}, index=pd.bdate_range("2020-01-01", periods=1))
    limits = ExposureLimits(max_per_ticker=0.2)

    # Act
    violations = check_exposure_limits(positions, limits)

    # Assert
    assert any(v.kind == "per_ticker" and v.ticker == "A" for v in violations)
    assert not any(v.ticker == "B" for v in violations)


def test_check_exposure_limits_returns_empty_list_when_within_limits() -> None:
    # Arrange
    positions = pd.DataFrame({"A": [0.1], "B": [-0.1]}, index=pd.bdate_range("2020-01-01", periods=1))
    limits = ExposureLimits(max_gross=1.0, max_net=0.5, max_per_ticker=0.3)

    # Act
    violations = check_exposure_limits(positions, limits)

    # Assert
    assert violations == []
