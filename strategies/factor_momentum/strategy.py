"""FactorMomentumStrategy: momentum + low-volatility multi-factor
long/short, monthly rebalanced (REQUIREMENTS.md 5.2).

Implements strategies.base.Strategy directly (generate_signals(prices) ->
pd.DataFrame) -- no adapter needed to reuse core.backtest.engine.
assert_causal, mirroring PairsTradingStrategy's pattern. Unlike
PairsTradingStrategy, there is no per-pair hedge_ratio to embed in the
signal: generate_signals returns a leg-normalized weight (see
ranking.build_long_short_signal -- each leg sums to +-1.0, or 0.0 with no
qualifying candidates) for every ticker in `prices`, leaving each name's
own volatility-based sizing to core.backtest.position_sizing -- the same
division of responsibility already established there. The leg
normalization itself (rather than a raw +-1) exists specifically so that
VolTargetSizer's per-ticker-only max_leverage cap cannot combine into
unbounded portfolio-level gross exposure regardless of how many names a
quintile happens to contain (see ranking.py's module docstring).

No separate "universe" constructor argument: the traded universe is
whatever columns `prices` has when generate_signals is called, matching
the Strategy protocol's pure-function contract.

mode="sector_neutral" (REQUIREMENTS.md 5.2 extension) reuses
ranking.build_long_short_signal's global cross-sectional construction by
default; passing mode="sector_neutral" switches to
sector_neutral.build_sector_neutral_signal instead, requiring a
sector_map (ticker -> GICS sector, same shape strategies.pairs_trading
already uses) -- mirroring PairsTradingStrategy's
hedge_ratio_mode="kalman" requiring kalman_hyperparams. sector_map is a
static, time-independent side input bound at construction (like
PairsTradingStrategy's candidate); it introduces no new causality risk
since assert_causal only perturbs `prices`, never anything bound to the
strategy instance.
"""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from strategies.factor_momentum.factors import low_volatility_score, momentum_score
from strategies.factor_momentum.ranking import FactorMomentumConfig, build_long_short_signal
from strategies.factor_momentum.rebalance import hold_until_next_rebalance, month_end_dates
from strategies.factor_momentum.sector_neutral import build_sector_neutral_signal


class FactorMomentumStrategy:
    def __init__(
        self,
        config: FactorMomentumConfig | None = None,
        mode: Literal["global", "sector_neutral"] = "global",
        sector_map: dict[str, str] | None = None,
    ) -> None:
        if mode == "sector_neutral" and sector_map is None:
            raise ValueError("sector_map is required when mode='sector_neutral'")
        self._config = config or FactorMomentumConfig()
        self._mode = mode
        self._sector_map = sector_map

    @property
    def name(self) -> str:
        if self._mode == "global":
            return "factor-momentum"
        return f"factor-momentum({self._mode})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        momentum = momentum_score(prices, self._config.momentum_lookback_days, self._config.momentum_skip_days)
        low_vol = low_volatility_score(prices, self._config.low_vol_window)
        if self._mode == "sector_neutral":
            assert self._sector_map is not None  # guaranteed by __init__
            daily_signal = build_sector_neutral_signal(momentum, low_vol, self._sector_map, self._config)
        else:
            daily_signal = build_long_short_signal(momentum, low_vol, self._config)
        # DataLoader's price panels are always datetime-indexed at
        # runtime; pandas-stubs types DataFrame.index as the generic
        # pd.Index, so this narrowing mirrors the same pandas-stubs
        # limitation worked around elsewhere in this project (see
        # core/data/loader.py's Hashable-ambiguity cast).
        rebalance_dates = month_end_dates(cast(pd.DatetimeIndex, prices.index))
        return hold_until_next_rebalance(daily_signal, rebalance_dates)
