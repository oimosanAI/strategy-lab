"""Pairs-trading Strategy implementation (REQUIREMENTS.md 5.1).

Assembles cointegration.py's selected pair, either static_hedge.py's or
kalman_hedge.py's hedge-ratio estimate, and signal.py's z-score entry/exit
logic into strategies.base.Strategy's generate_signals(prices) ->
DataFrame contract. Switching between the static and Kalman hedge ratio
is a single constructor argument (``hedge_ratio_mode``) -- running the
SAME class both ways on the SAME pair/period is what makes the two
directly comparable, per REQUIREMENTS.md 5.1's "両方実装し比較できる".

PairsTradingStrategy implements Strategy directly (no test adapter is
needed to reuse core.backtest.engine.assert_causal, unlike the sizer/
Kalman-filter-alone test adapters used earlier): every piece it composes
(a constant or already-causal Kalman hedge ratio, a backward-looking
rolling z-score, a forward-only entry/exit state machine) is causal by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from strategies.pairs_trading.cointegration import PairCandidate
from strategies.pairs_trading.kalman_hedge import KalmanHyperparameters, kalman_hedge_ratio
from strategies.pairs_trading.signal import (
    compute_spread,
    generate_entry_exit_positions,
    rolling_zscore,
    zscore_window_from_half_life,
)
from strategies.pairs_trading.static_hedge import HedgeRatioSeries, static_hedge_ratio


@dataclass(frozen=True)
class PairsTradingSignalConfig:
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    zscore_window_multiplier: float = 4.0
    min_zscore_window: int = 10
    max_zscore_window: int = 90
    # A multiple of the pair's own half-life; None disables the safety
    # valve entirely. Kept ON by default (REQUIREMENTS.md 1.1's
    # institutional risk-management framing) -- see
    # PairsTradingStrategy.last_forced_exit_dates for why its activations
    # are tracked separately rather than silently blended into the raw
    # signal's results.
    max_holding_multiplier: float | None = 3.0


class PairsTradingStrategy:
    """strategies.base.Strategy implementation for a single pair."""

    def __init__(
        self,
        candidate: PairCandidate,
        hedge_ratio_mode: Literal["static", "kalman"],
        config: PairsTradingSignalConfig | None = None,
        kalman_hyperparams: KalmanHyperparameters | None = None,
    ) -> None:
        if hedge_ratio_mode == "kalman" and kalman_hyperparams is None:
            raise ValueError("kalman_hyperparams is required when hedge_ratio_mode='kalman'")

        self._candidate = candidate
        self._mode = hedge_ratio_mode
        self._config = config or PairsTradingSignalConfig()
        self._kalman_hyperparams = kalman_hyperparams
        self._last_forced_exit_dates: list[pd.Timestamp] = []

    @property
    def name(self) -> str:
        return f"pairs-trading({self._candidate.ticker_a},{self._candidate.ticker_b},{self._mode})"

    @property
    def last_forced_exit_dates(self) -> list[pd.Timestamp]:
        """Dates where the max-holding-period safety valve forced an exit
        during the most recent generate_signals() call (empty before any
        call, or if max_holding_multiplier is None)."""
        return self._last_forced_exit_dates

    def _hedge_ratio(self, prices: pd.DataFrame) -> HedgeRatioSeries:
        if self._mode == "static":
            return static_hedge_ratio(self._candidate, prices.index)
        assert self._kalman_hyperparams is not None  # guaranteed by __init__
        eg = self._candidate.engle_granger
        return kalman_hedge_ratio(prices, self._kalman_hyperparams, eg.dependent, eg.independent)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        hedge = self._hedge_ratio(prices)
        spread = compute_spread(hedge, prices)

        window = zscore_window_from_half_life(
            self._candidate.half_life,
            multiplier=self._config.zscore_window_multiplier,
            min_window=self._config.min_zscore_window,
            max_window=self._config.max_zscore_window,
        )
        zscore = rolling_zscore(spread, window)

        max_holding_periods = (
            round(self._candidate.half_life * self._config.max_holding_multiplier)
            if self._config.max_holding_multiplier is not None
            else None
        )
        entry_exit = generate_entry_exit_positions(
            zscore,
            entry_threshold=self._config.entry_threshold,
            exit_threshold=self._config.exit_threshold,
            max_holding_periods=max_holding_periods,
        )
        self._last_forced_exit_dates = entry_exit.forced_exit_dates

        position_state = entry_exit.position_state
        return pd.DataFrame(
            {
                hedge.ticker_dependent: position_state,
                hedge.ticker_independent: -position_state * hedge.ratio,
            },
            index=prices.index,
        )
