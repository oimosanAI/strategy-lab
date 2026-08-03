"""Tests for strategies.factor_momentum.strategy.

Fixes the public API surface:
- FactorMomentumStrategy(config=None): implements strategies.base.Strategy
  directly (generate_signals(prices) -> pd.DataFrame), no adapter needed
  to reuse core.backtest.engine.assert_causal -- same pattern as
  PairsTradingStrategy. No separate "universe" argument: the traded
  universe is whatever columns `prices` has when generate_signals is
  called.

Negative controls (_LookaheadMomentumStrategy, _BackfilledRebalanceStrategy)
mirror pairs_trading's _CenteredZScoreStrategy: minimal, realistic
one-line bugs layered on top of the correct implementation, each PROVEN
to be caught by assert_causal, not merely assumed to be -- a guard only
shown to pass on the correct implementation has never been shown to
actually work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal
from strategies.factor_momentum.factors import low_volatility_score, momentum_score
from strategies.factor_momentum.ranking import FactorMomentumConfig, build_long_short_signal
from strategies.factor_momentum.rebalance import hold_until_next_rebalance, month_end_dates
from strategies.factor_momentum.strategy import FactorMomentumStrategy

# Shortened windows so a compact synthetic panel exercises several
# rebalance cycles without needing a full 252-day momentum lookback.
_TEST_CONFIG = FactorMomentumConfig(momentum_lookback_days=20, momentum_skip_days=5, low_vol_window=10)


def _synthetic_price_panel(n_days: int = 90, n_tickers: int = 10, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    log_returns = rng.normal(0.0002, 0.01, size=(n_days, n_tickers))
    prices = 100.0 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def test_factor_momentum_strategy_name() -> None:
    strategy = FactorMomentumStrategy()
    assert strategy.name == "factor-momentum"


def test_factor_momentum_strategy_generate_signals_shape_matches_prices() -> None:
    prices = _synthetic_price_panel()
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG)

    result = strategy.generate_signals(prices)

    assert list(result.index) == list(prices.index)
    assert list(result.columns) == list(prices.columns)


def test_factor_momentum_strategy_sector_neutral_shape_matches_prices() -> None:
    # strategies.base.Strategy's contract is "same shape as prices", and
    # global mode already honours it. Sector-neutral mode builds its output
    # by concatenating per-sector frames, so a ticker absent from
    # sector_map used to vanish from the returned columns entirely --
    # surviving only because VolTargetSizer's pandas alignment happened to
    # refill it with 0.0. The contract must hold on its own, in BOTH modes.
    prices = _synthetic_price_panel()
    sector_map = {t: ("SectorA" if i % 2 else "SectorB") for i, t in enumerate(prices.columns)}
    unmapped = str(prices.columns[0])
    del sector_map[unmapped]
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG, mode="sector_neutral", sector_map=sector_map)

    result = strategy.generate_signals(prices)

    assert list(result.columns) == list(prices.columns)
    assert list(result.index) == list(prices.index)
    assert (result[unmapped] == 0.0).all()
    assert not result.isna().to_numpy().any()


def test_factor_momentum_strategy_each_leg_normalizes_to_one_or_zero() -> None:
    # ranking.build_long_short_signal normalizes each leg to sum to
    # 1.0 (long) / -1.0 (short), or 0.0 on a day with no qualifying
    # candidates (e.g. warm-up) -- verified here at the full-strategy
    # level, not just ranking.py's own unit tests, since this is the
    # property that fixes the unbounded portfolio-gross-exposure gap
    # found during the real-data E2E run.
    prices = _synthetic_price_panel()
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG)

    result = strategy.generate_signals(prices)

    long_leg_sum = result.where(result > 0, 0.0).sum(axis=1)
    short_leg_sum = result.where(result < 0, 0.0).sum(axis=1)
    assert np.all(np.isclose(long_leg_sum, 0.0) | np.isclose(long_leg_sum, 1.0))
    assert np.all(np.isclose(short_leg_sum, 0.0) | np.isclose(short_leg_sum, -1.0))


def test_factor_momentum_strategy_passes_assert_causal() -> None:
    prices = _synthetic_price_panel()
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG)

    assert_causal(strategy, prices, n_trials=12)


# ---------------------------------------------------------------------------
# Negative controls: assert_causal must actually CATCH these.
# ---------------------------------------------------------------------------


class _LookaheadMomentumStrategy:
    """Deliberately non-causal: momentum's skip_days shift has its sign
    flipped, so it reads price[t + skip_days] instead of
    price[t - skip_days] -- a realistic 'wrong sign on shift()' bug."""

    name = "lookahead-momentum (deliberately non-causal)"

    def __init__(self, config: FactorMomentumConfig) -> None:
        self._config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        lagged = prices.shift(-self._config.momentum_skip_days)  # BUG: wrong sign
        momentum = (
            lagged / lagged.shift(self._config.momentum_lookback_days - self._config.momentum_skip_days) - 1.0
        )
        low_vol = low_volatility_score(prices, self._config.low_vol_window)
        daily_signal = build_long_short_signal(momentum, low_vol, self._config)
        rebalance_dates = month_end_dates(prices.index)
        return hold_until_next_rebalance(daily_signal, rebalance_dates)


class _BackfilledRebalanceStrategy:
    """Deliberately non-causal: uses bfill() instead of ffill() when
    holding a signal between rebalance dates, so a date BEFORE a
    rebalance can end up depending on that (future, relative to it)
    rebalance decision -- a realistic 'wrong fill direction' bug,
    directly analogous to pairs_trading's rolling(center=True) control."""

    name = "backfilled-rebalance (deliberately non-causal)"

    def __init__(self, config: FactorMomentumConfig) -> None:
        self._config = config

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = momentum_score(prices, self._config.momentum_lookback_days, self._config.momentum_skip_days)
        low_vol = low_volatility_score(prices, self._config.low_vol_window)
        daily_signal = build_long_short_signal(momentum, low_vol, self._config)
        rebalance_dates = month_end_dates(prices.index)

        non_rebalance_mask = ~daily_signal.index.isin(rebalance_dates)
        masked = daily_signal.copy()
        masked.loc[non_rebalance_mask] = np.nan
        return masked.bfill().fillna(0.0)  # BUG: bfill instead of ffill


def test_lookahead_momentum_is_rejected_by_assert_causal() -> None:
    # n_trials=12 is NOT enough here and was verified to intermittently
    # miss: monthly hold_until_next_rebalance discards ~95% of the daily
    # signal (only the value AT each rebalance date survives), so the
    # look-ahead leak into a pre-cut index k is only visible when some
    # rebalance date r satisfies k-5 < r <= k (the bug reads prices[r+5]).
    # With 5 rebalance dates in this 90-day panel, that "danger window" is
    # only ~20% of possible k draws -- P(miss all 12) ~= 0.795**12 ~= 7%,
    # which is exactly what happened with n_trials=12 during development.
    # n_trials=60 pushes P(miss all) to ~= 0.795**60 ~= 3e-7.
    prices = _synthetic_price_panel()
    with pytest.raises(LookAheadError):
        assert_causal(_LookaheadMomentumStrategy(_TEST_CONFIG), prices, n_trials=60)


def test_backfilled_rebalance_is_rejected_by_assert_causal() -> None:
    prices = _synthetic_price_panel()
    with pytest.raises(LookAheadError):
        assert_causal(_BackfilledRebalanceStrategy(_TEST_CONFIG), prices, n_trials=12)


# ---------------------------------------------------------------------------
# mode="sector_neutral" wiring (strategies.factor_momentum.sector_neutral
# itself is unit-tested in tests/core/test_sector_neutral.py -- these tests
# only verify the Strategy class is actually connected to it, not the
# ranking math itself).
# ---------------------------------------------------------------------------

_SECTOR_MAP = {f"T{i}": ("SectorA" if i < 5 else "SectorB") for i in range(10)}


def test_factor_momentum_strategy_sector_neutral_mode_requires_sector_map() -> None:
    with pytest.raises(ValueError):
        FactorMomentumStrategy(config=_TEST_CONFIG, mode="sector_neutral")


def test_factor_momentum_strategy_sector_neutral_mode_passes_assert_causal() -> None:
    prices = _synthetic_price_panel()
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG, mode="sector_neutral", sector_map=_SECTOR_MAP)

    assert_causal(strategy, prices, n_trials=12)


def test_factor_momentum_strategy_sector_neutral_mode_produces_different_signals_than_global_mode() -> None:
    # Wiring check: mode actually changes generate_signals' behavior,
    # rather than being silently ignored. (The algorithmic reason sector
    # neutrality differs from global ranking -- guaranteed per-sector
    # representation -- is proven precisely in test_sector_neutral.py.)
    prices = _synthetic_price_panel()
    global_strategy = FactorMomentumStrategy(config=_TEST_CONFIG, mode="global")
    neutral_strategy = FactorMomentumStrategy(config=_TEST_CONFIG, mode="sector_neutral", sector_map=_SECTOR_MAP)

    global_result = global_strategy.generate_signals(prices)
    neutral_result = neutral_strategy.generate_signals(prices)

    assert not global_result.equals(neutral_result)


def test_factor_momentum_strategy_sector_neutral_mode_name_is_distinct() -> None:
    strategy = FactorMomentumStrategy(config=_TEST_CONFIG, mode="sector_neutral", sector_map=_SECTOR_MAP)
    assert strategy.name == "factor-momentum(sector_neutral)"
