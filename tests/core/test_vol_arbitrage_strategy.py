"""Tests for strategies.vol_arbitrage.strategy.

Fixes the public API surface:

- VolArbitrageConfig(rv_window=21, vrp_threshold=..., ts_threshold=1.0,
  mode="vrp"|"term_structure", traded_ticker="SVXY", vix_column="VIX",
  spy_column="SPY", vix9d_column="VIX9D").
- VolArbitrageStrategy(config=None): implements strategies.base.Strategy
  directly (generate_signals(prices) -> pd.DataFrame), no adapter needed
  -- same pattern as PairsTradingStrategy/FactorMomentumStrategy.
  `prices` carries the traded ticker (SVXY) plus VIX/SPY (and, in
  "term_structure" mode, VIX9D) as REFERENCE-ONLY columns that are never
  assigned a nonzero position (their presence lets assert_causal's
  existing prices-only perturbation mechanism cover the signal
  computation's causality too, with no new guard -- see the module
  docstring). mode="vrp" (default) preserves the original trailing-RV
  VRP signal unchanged; mode="term_structure" is the new VIX9D/VIX
  variant added per strategies/vol_arbitrage/README.md Step F -- see
  that section for why the two are kept side by side rather than one
  replacing the other.

Negative control (_LookaheadVRPStrategy) mirrors pairs_trading's
_CenteredZScoreStrategy: a realistic 'forgot to leave center at its
default' bug in the realized-volatility rolling window, proven to be
CAUGHT by assert_causal, not merely assumed to be.
_LookaheadTermStructureStrategy is the term_structure-mode equivalent: a
realistic 'used tomorrow's VIX9D' off-by-one bug (`.shift(-1)`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal
from strategies.vol_arbitrage.signal import compute_term_structure_ratio, realized_volatility
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

_TEST_CONFIG = VolArbitrageConfig(rv_window=5, vrp_threshold=0.10)
_TS_TEST_CONFIG = VolArbitrageConfig(mode="term_structure", ts_threshold=1.0)


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


# ---------------------------------------------------------------------------
# mode="term_structure" (VIX9D / VIX ratio) -- strategies/vol_arbitrage/
# README.md Step F/section 3-7.
# ---------------------------------------------------------------------------


def _synthetic_price_panel_with_vix9d(n_days: int = 90, seed: int = 0) -> pd.DataFrame:
    panel = _synthetic_price_panel(n_days=n_days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    # VIX9D need not track VIX exactly -- independent small noise around
    # the same level is enough for the "only trades configured ticker" /
    # "missing column" tests below, which don't depend on any particular
    # ratio value.
    panel["VIX9D"] = np.clip(panel["VIX"] + rng.normal(0.0, 1.0, n_days), 5.0, 70.0)
    return panel


def test_vol_arbitrage_strategy_term_structure_only_trades_configured_ticker() -> None:
    prices = _synthetic_price_panel_with_vix9d()
    strategy = VolArbitrageStrategy(config=_TS_TEST_CONFIG)

    result = strategy.generate_signals(prices)

    assert (result["VIX"] == 0.0).all()
    assert (result["SPY"] == 0.0).all()
    assert (result["VIX9D"] == 0.0).all()


def test_vol_arbitrage_strategy_term_structure_signals_svxy_when_ratio_below_threshold() -> None:
    idx = pd.bdate_range("2020-01-01", periods=6)
    svxy = pd.Series([50.0] * 6, index=idx)
    spy = pd.Series([100.0] * 6, index=idx)
    vix = pd.Series([20.0] * 6, index=idx)
    # ratio = vix9d / vix: [0.5, 0.9, 1.0, 1.1, 1.5, 1.0] -> below 1.0,
    # below 1.0, at 1.0 (not below), above, above, at (boundary again).
    vix9d = pd.Series([10.0, 18.0, 20.0, 22.0, 30.0, 20.0], index=idx)
    prices = pd.DataFrame({"SVXY": svxy, "VIX": vix, "SPY": spy, "VIX9D": vix9d}, index=idx)

    strategy = VolArbitrageStrategy(config=_TS_TEST_CONFIG)
    result = strategy.generate_signals(prices)

    expected = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert result["SVXY"].tolist() == expected


@pytest.mark.parametrize("missing", ["SVXY", "VIX", "SPY", "VIX9D"])
def test_vol_arbitrage_strategy_term_structure_rejects_panel_missing_a_required_column(
    missing: str,
) -> None:
    prices = _synthetic_price_panel_with_vix9d().drop(columns=[missing])
    strategy = VolArbitrageStrategy(config=_TS_TEST_CONFIG)

    with pytest.raises(ValueError, match=missing):
        strategy.generate_signals(prices)


def test_vol_arbitrage_strategy_vrp_mode_does_not_require_vix9d_column() -> None:
    # mode="vrp" (the default/original behavior) must be unaffected by the
    # term_structure addition -- no VIX9D column required.
    prices = _synthetic_price_panel()
    strategy = VolArbitrageStrategy(config=_TEST_CONFIG)

    result = strategy.generate_signals(prices)

    assert "VIX9D" not in result.columns


def test_vol_arbitrage_strategy_term_structure_passes_assert_causal() -> None:
    prices = _synthetic_price_panel_with_vix9d()
    strategy = VolArbitrageStrategy(config=_TS_TEST_CONFIG)

    assert_causal(strategy, prices, n_trials=12)


class _LookaheadTermStructureStrategy:
    """Deliberately non-causal: uses TOMORROW's VIX9D (`.shift(-1)`), a
    realistic off-by-one alignment bug, instead of today's."""

    name = "lookahead-term-structure (deliberately non-causal)"

    def __init__(self, config: VolArbitrageConfig) -> None:
        self._config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        vix9d_tomorrow = prices[self._config.vix9d_column].shift(-1)  # BUG: should be .shift(0)
        ratio = compute_term_structure_ratio(vix9d_tomorrow, prices[self._config.vix_column])

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[self._config.traded_ticker] = (ratio < self._config.ts_threshold).astype(float)
        return signal


def _near_threshold_term_structure_panel(n_days: int = 90, seed: int = 1) -> pd.DataFrame:
    # Same rationale as _near_threshold_price_panel above (see its comment):
    # a generic random-walk VIX9D/VIX pair usually sits far from
    # ts_threshold, giving the shift(-1) bug near-zero detection power.
    # Here, VIX9D is constructed as VIX * ts_threshold plus small
    # independent daily noise, so the true ratio hugs the threshold at
    # every date and swapping in tomorrow's (independent) noise via the
    # bug flips the discrete decision reliably.
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    vix = pd.Series(np.clip(15.0 + rng.normal(0.0, 3.0, n_days).cumsum() * 0.05, 8.0, 60.0), index=idx)
    svxy = pd.Series([50.0] * n_days, index=idx)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n_days))), index=idx)

    noise = rng.normal(0.0, 0.15, n_days)  # small, independent day-to-day wobble
    vix9d = _TS_TEST_CONFIG.ts_threshold * vix + noise

    return pd.DataFrame({"SVXY": svxy, "VIX": vix, "SPY": spy, "VIX9D": vix9d}, index=idx)


def test_lookahead_term_structure_is_rejected_by_assert_causal() -> None:
    prices = _near_threshold_term_structure_panel()
    with pytest.raises(LookAheadError):
        assert_causal(_LookaheadTermStructureStrategy(_TS_TEST_CONFIG), prices, n_trials=12)


def test_lookahead_term_structure_negative_control_has_reliable_detection_power() -> None:
    # Mirrors strategies/vol_arbitrage/README.md Step C: a negative
    # control is only trustworthy once its detection power is measured,
    # not merely asserted to pass once. Runs assert_causal across 10
    # independent seeds and requires the bug be caught in all 10 -- the
    # same bar Step C established for _LookaheadVRPStrategy.
    catches = 0
    for seed in range(10):
        prices = _near_threshold_term_structure_panel(seed=seed)
        try:
            assert_causal(_LookaheadTermStructureStrategy(_TS_TEST_CONFIG), prices, n_trials=12)
        except LookAheadError:
            catches += 1
    assert catches == 10
