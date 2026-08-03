"""Tests for strategies.vol_arbitrage.strategy.

Fixes the public API surface:

- VolArbitrageConfig(rv_window=21, vrp_threshold=..., traded_ticker="SVXY",
  vix_column="VIX", spy_column="SPY").
- VolArbitrageStrategy(config=None): implements strategies.base.Strategy
  directly (generate_signals(prices) -> pd.DataFrame), no adapter needed
  -- same pattern as PairsTradingStrategy/FactorMomentumStrategy.
  `prices` carries 3 columns: the traded ticker (SVXY) plus VIX and SPY
  as REFERENCE-ONLY columns that are never assigned a nonzero position
  (their presence lets assert_causal's existing prices-only perturbation
  mechanism cover the VRP computation's causality too, with no new guard
  -- see the module docstring).

Negative control (_LookaheadVRPStrategy) mirrors pairs_trading's
_CenteredZScoreStrategy: a realistic 'forgot to leave center at its
default' bug in the realized-volatility rolling window, proven to be
CAUGHT by assert_causal, not merely assumed to be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal
from strategies.vol_arbitrage.signal import realized_volatility
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

_TEST_CONFIG = VolArbitrageConfig(rv_window=5, vrp_threshold=0.10)


def _synthetic_price_panel(n_days: int = 90, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    spy = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days)))
    svxy = 50.0 * np.exp(np.cumsum(rng.normal(-0.0001, 0.02, n_days)))
    vix = np.clip(15.0 + rng.normal(0.0, 3.0, n_days).cumsum() * 0.1, 8.0, 60.0)
    return pd.DataFrame({"SVXY": svxy, "VIX": vix, "SPY": spy}, index=idx)


def test_vol_arbitrage_strategy_name() -> None:
    strategy = VolArbitrageStrategy()
    assert strategy.name == "vol-arbitrage"


def test_vol_arbitrage_strategy_only_trades_configured_ticker() -> None:
    prices = _synthetic_price_panel()
    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)

    result = strategy.generate_signals(prices)

    assert (result["VIX"] == 0.0).all()
    assert (result["SPY"] == 0.0).all()


def test_vol_arbitrage_strategy_signals_svxy_when_vrp_above_threshold() -> None:
    idx = pd.bdate_range("2020-01-01", periods=15)
    spy = pd.Series([100.0] * 15, index=idx)  # flat -> RV = 0.0 once warmed up
    vix = pd.Series([5.0] * 10 + [20.0] * 5, index=idx)  # low then high
    svxy = pd.Series([50.0] * 15, index=idx)
    prices = pd.DataFrame({"SVXY": svxy, "VIX": vix, "SPY": spy}, index=idx)

    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)
    result = strategy.generate_signals(prices)

    # rows 0-4: RV warmup (NaN VRP) -> flat. rows 5-9: VRP=0.05 < 0.10 ->
    # flat. rows 10-14: VRP=0.20 > 0.10 -> long SVXY.
    assert (result["SVXY"].iloc[:10] == 0.0).all()
    assert (result["SVXY"].iloc[10:] == 1.0).all()


@pytest.mark.parametrize("missing", ["SVXY", "VIX", "SPY"])
def test_vol_arbitrage_strategy_rejects_panel_missing_a_required_column(missing: str) -> None:
    # Arrange: a misconfigured traded_ticker used to be SILENT -- assigning
    # signal[traded_ticker] created a brand-new column that run_backtest
    # (which iterates prices.columns) then ignored, so the backtest
    # completed and reported an all-zero-return "no edge" strategy instead
    # of failing. The VIX/SPY columns raised KeyError on the same mistake.
    # All three must now fail the same way, loudly, before any number is
    # produced.
    prices = _synthetic_price_panel().drop(columns=[missing])
    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)

    # Act / Assert
    with pytest.raises(ValueError, match=missing):
        strategy.generate_signals(prices)


def test_vol_arbitrage_strategy_error_names_every_missing_column() -> None:
    prices = _synthetic_price_panel().drop(columns=["SVXY", "VIX"])
    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)

    with pytest.raises(ValueError) as excinfo:
        strategy.generate_signals(prices)

    message = str(excinfo.value)
    assert "SVXY" in message and "VIX" in message


def test_vol_arbitrage_strategy_passes_assert_causal() -> None:
    prices = _synthetic_price_panel()
    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)

    assert_causal(strategy, prices, n_trials=12)


# ---------------------------------------------------------------------------
# Negative control: assert_causal must actually CATCH this.
# ---------------------------------------------------------------------------


class _LookaheadVRPStrategy:
    """Deliberately non-causal: realized volatility uses a centered
    rolling window (rolling(window, center=True)), peeking at future
    rows within the window -- the same realistic 'forgot to leave
    center at its default' bug as pairs_trading's _CenteredZScoreStrategy."""

    name = "lookahead-vrp (deliberately non-causal)"

    def __init__(self, config: VolArbitrageConfig) -> None:
        self._config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        spy = prices[self._config.spy_column]
        returns = spy.pct_change()
        rv = returns.rolling(self._config.rv_window, center=True).std() * np.sqrt(252)  # BUG: center=True
        vrp = (prices[self._config.vix_column] / 100.0) - rv

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[self._config.traded_ticker] = (vrp > self._config.vrp_threshold).astype(float)
        return signal


def _near_threshold_price_panel(n_days: int = 90, seed: int = 1) -> pd.DataFrame:
    # A generic random-walk fixture (like _synthetic_price_panel) turned
    # out to have near-zero detection power for this negative control:
    # VIX and SPY are independent random walks, so baseline VRP usually
    # sits far from vrp_threshold at whichever date assert_causal happens
    # to check -- a random walk can shift RV a lot without ever crossing
    # the discrete 0/1 boundary. Verified empirically: even n_trials=500
    # only caught the bug in 4/10 independent assert_causal seeds.
    #
    # Instead, engineer baseline VRP to sit just ABOVE vrp_threshold at
    # every date (SPY has small, roughly-constant realized vol; VIX is a
    # constant set from that same realized vol + threshold + a small
    # epsilon), so the signal is right at the decision boundary
    # everywhere -- any RV shift from the center=True lookahead flips it
    # reliably. Verified empirically: n_trials=12 (the default) then
    # catches it in 10/10 independent assert_causal seeds.
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n_days))), index=idx)
    svxy = pd.Series([50.0] * n_days, index=idx)

    avg_rv = realized_volatility(spy, window=_TEST_CONFIG.rv_window).dropna().mean()
    vix_level = (avg_rv + _TEST_CONFIG.vrp_threshold + 0.005) * 100.0
    vix = pd.Series([vix_level] * n_days, index=idx)

    return pd.DataFrame({"SVXY": svxy, "VIX": vix, "SPY": spy}, index=idx)


def test_lookahead_vrp_is_rejected_by_assert_causal() -> None:
    prices = _near_threshold_price_panel()
    with pytest.raises(LookAheadError):
        assert_causal(_LookaheadVRPStrategy(_TEST_CONFIG), prices, n_trials=12)
