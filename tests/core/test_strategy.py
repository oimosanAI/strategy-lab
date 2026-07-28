"""Tests for strategies.pairs_trading.strategy.

Fixes the public API surface:

- PairsTradingSignalConfig(entry_threshold, exit_threshold,
  zscore_window_multiplier, min_zscore_window, max_zscore_window,
  max_holding_multiplier).
- PairsTradingStrategy(candidate, hedge_ratio_mode, config=None,
  kalman_hyperparams=None): implements strategies.base.Strategy directly
  (generate_signals(prices) -> pd.DataFrame), so no adapter is needed to
  reuse core.backtest.engine.assert_causal, unlike the sizer/Kalman-filter
  test adapters used earlier in this project.
- PairsTradingStrategy.last_forced_exit_dates: an auxiliary property
  (not part of the Strategy protocol) exposing the max-holding-period
  forced-exit dates from the most recent generate_signals() call, so the
  safety valve's activation can be tracked and its effect on performance
  evaluated separately later.

The forced_exit_dates wiring test does not just check
"isinstance(..., list)" (that would pass even for a
"always returns []" bug): it uses a deliberately aggressive
max_holding_multiplier so a forced exit is known to occur, and
cross-checks the recorded dates against an INDEPENDENT call to
generate_entry_exit_positions on the same reconstructed z-score series --
generate_entry_exit_positions' own correctness is already covered by
test_signal.py; this test verifies only that PairsTradingStrategy wires
it correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal
from core.backtest.sample_split import SamplePeriod
from strategies.pairs_trading.cointegration import (
    PairCandidate,
    compute_half_life,
    engle_granger_test,
    johansen_test,
)
from strategies.pairs_trading.kalman_hedge import calibrate_kalman_hyperparameters
from strategies.pairs_trading.signal import (
    compute_spread,
    generate_entry_exit_positions,
    rolling_zscore,
    zscore_window_from_half_life,
)
from strategies.pairs_trading.static_hedge import static_hedge_ratio
from strategies.pairs_trading.strategy import PairsTradingSignalConfig, PairsTradingStrategy


def _cointegrated_panel(n: int = 200, beta: float = 1.5, phi: float = 0.5) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng_a = np.random.default_rng(30)
    a = 100 + np.cumsum(rng_a.normal(0, 1, n))
    rng_spread = np.random.default_rng(31)
    innov = rng_spread.normal(0, 2, n)  # wider innovations -> real z-score swings
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + innov[t]
    b = beta * a + spread
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def _make_candidate(prices: pd.DataFrame) -> PairCandidate:
    eg = engle_granger_test(prices["A"], prices["B"], "A", "B")
    jh = johansen_test(prices[["A", "B"]])
    dep = prices[eg.dependent]
    indep = prices[eg.independent]
    spread = dep - eg.intercept - eg.hedge_ratio * indep
    half_life = compute_half_life(spread)
    return PairCandidate(
        ticker_a="A",
        ticker_b="B",
        sector="Tech",
        engle_granger=eg,
        johansen=jh,
        half_life=half_life,
        # n_tests=1: this fixture builds a single, standalone pair outside
        # select_pairs' multi-candidate scanning, so there is no
        # multiple-testing context -- the adjusted value is just the raw
        # p-value (see cointegration._adjust_p_value's n_tests<=1 case).
        adjusted_engle_granger_p_value=eg.p_value,
        n_tests=1,
    )


# ---------------------------------------------------------------------------
# 4 (continued). forced_exit_dates wiring: not just isinstance(..., list)
# ---------------------------------------------------------------------------


def test_pairs_trading_strategy_wires_forced_exit_dates_correctly() -> None:
    # Arrange: a deliberately aggressive max_holding_multiplier so a
    # forced exit is known to occur, not merely possible.
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    config = PairsTradingSignalConfig(max_holding_multiplier=0.5)
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static", config=config)

    # Act
    strategy.generate_signals(prices)

    # Independently reconstruct the same pipeline (generate_entry_exit_
    # positions' own correctness is already covered by test_signal.py;
    # this only checks that the strategy wires it correctly).
    hedge = static_hedge_ratio(candidate, prices.index)
    spread = compute_spread(hedge, prices)
    window = zscore_window_from_half_life(
        candidate.half_life,
        multiplier=config.zscore_window_multiplier,
        min_window=config.min_zscore_window,
        max_window=config.max_zscore_window,
    )
    zscore = rolling_zscore(spread, window)
    max_holding_periods = round(candidate.half_life * config.max_holding_multiplier)
    expected = generate_entry_exit_positions(
        zscore, config.entry_threshold, config.exit_threshold, max_holding_periods=max_holding_periods
    )

    # Assert: actually triggered (not vacuously empty), and matches exactly.
    assert len(expected.forced_exit_dates) > 0
    assert strategy.last_forced_exit_dates == expected.forced_exit_dates


def test_pairs_trading_strategy_forced_exit_dates_empty_before_any_call() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")
    assert strategy.last_forced_exit_dates == []


def test_pairs_trading_strategy_requires_kalman_hyperparams_for_kalman_mode() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    with pytest.raises(ValueError):
        PairsTradingStrategy(candidate, hedge_ratio_mode="kalman")


def test_pairs_trading_strategy_name_reflects_pair_and_mode() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")
    assert strategy.name == "pairs-trading(A,B,static)"


# ---------------------------------------------------------------------------
# 5. PairsTradingStrategy: Strategy protocol + assert_causal (both modes)
# ---------------------------------------------------------------------------


def test_pairs_trading_strategy_passes_assert_causal_static_mode() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")

    assert_causal(strategy, prices, n_trials=12)


def test_pairs_trading_strategy_passes_assert_causal_kalman_mode() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    in_sample = SamplePeriod(prices.index[0], prices.index[-1])
    hyperparams = calibrate_kalman_hyperparameters(
        prices, in_sample, candidate.engle_granger.dependent, candidate.engle_granger.independent
    )
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="kalman", kalman_hyperparams=hyperparams)

    assert_causal(strategy, prices, n_trials=12)


# ---------------------------------------------------------------------------
# 6. center=True negative control
# ---------------------------------------------------------------------------


class _CenteredZScoreStrategy:
    """Deliberately non-causal: uses rolling(window, center=True), which
    peeks at future rows within the window -- a realistic 'forgot to
    leave center at its default' bug."""

    name = "centered-zscore (deliberately non-causal)"

    def __init__(self, candidate: PairCandidate, window: int) -> None:
        self._candidate = candidate
        self._window = window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        hedge = static_hedge_ratio(self._candidate, prices.index)
        spread = compute_spread(hedge, prices)
        mean = spread.rolling(self._window, center=True).mean()
        std = spread.rolling(self._window, center=True).std()
        zscore = (spread - mean) / std
        position = np.sign(zscore).fillna(0.0)
        return pd.DataFrame(
            {
                hedge.ticker_dependent: position,
                hedge.ticker_independent: -position * hedge.ratio,
            },
            index=prices.index,
        )


def test_centered_zscore_is_rejected_by_assert_causal() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)

    with pytest.raises(LookAheadError):
        assert_causal(_CenteredZScoreStrategy(candidate, window=20), prices, n_trials=12)
