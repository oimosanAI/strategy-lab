"""z-score based entry/exit signal generation (REQUIREMENTS.md 5.1).

Every function here is causal by construction:

- compute_spread / rolling_zscore only ever use ``.rolling()`` (a
  backward-looking window).
- generate_entry_exit_positions is a forward-only sequential loop: the
  state at t depends only on zscore(t) and the state carried from t-1.

This module deliberately does not know about strategies.base.Strategy --
strategy.py composes these pieces (plus a chosen hedge-ratio source) into
the actual Strategy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from strategies.pairs_trading.static_hedge import HedgeRatioSeries


def compute_spread(hedge: HedgeRatioSeries, prices: pd.DataFrame) -> pd.Series:
    """spread(t) = y(t) - alpha(t) - beta(t) * x(t).

    Subtracting the intercept matters most for the Kalman case, where
    alpha(t) is itself time-varying: omitting it would let alpha's own
    drift leak into the spread as extra noise. For the static case it is
    a constant and would cancel out of the rolling z-score anyway, but
    including it keeps the spread's absolute scale interpretable either way.
    """
    y = prices[hedge.ticker_dependent]
    x = prices[hedge.ticker_independent]
    return y - hedge.intercept - hedge.ratio * x


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """(spread - rolling_mean) / rolling_std over a trailing window."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return (spread - mean) / std


def zscore_window_from_half_life(
    half_life: float,
    multiplier: float = 4.0,
    min_window: int = 10,
    max_window: int = 90,
) -> int:
    """Derive the z-score rolling window from a pair's own half-life:
    long enough (relative to the mean-reversion speed) to average out
    noise, short enough to stay responsive. Clamped to [min_window,
    max_window] since a pair with an extreme half-life would otherwise
    produce an unusably tiny or unusably huge window."""
    return int(np.clip(round(half_life * multiplier), min_window, max_window))


@dataclass(frozen=True)
class EntryExitResult:
    """position_state: -1 (short the spread), 0 (flat), +1 (long the
    spread), per date. forced_exit_dates: dates where max_holding_periods
    triggered an exit rather than the z-score reverting naturally --
    tracked separately so the safety valve's activation frequency and
    effect on performance can be evaluated on their own, since it is on
    by default and would otherwise silently blend into the raw signal's
    results."""

    position_state: pd.Series
    forced_exit_dates: list[pd.Timestamp] = field(default_factory=list)


def generate_entry_exit_positions(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    max_holding_periods: int | None = None,
) -> EntryExitResult:
    """Forward-only state machine over ``zscore``.

    Rules, in priority order, evaluated at each date:

    1. Flat and z > +entry_threshold -> enter short the spread.
    2. Flat and z < -entry_threshold -> enter long the spread.
    3. In a position and |z| <= exit_threshold -> exit to flat.
    4. In a position, max_holding_periods is set, and the position has
       been held for at least that many periods without triggering rule 3
       -> forced exit to flat (recorded in ``forced_exit_dates``).
    5. Otherwise -> hold the current position unchanged.

    While in a position, only exit conditions (3, 4) are evaluated --
    a new entry signal in the opposite direction does NOT flip the
    position directly; the standard pairs-trading convention is to
    return to flat first.

    NaN z-score values (rolling-window warm-up) are treated as flat and
    never trigger any transition.
    """
    state = 0
    periods_in_position = 0
    states: list[int] = []
    forced_exit_dates: list[pd.Timestamp] = []

    for date, z in zscore.items():
        if pd.isna(z):
            states.append(0)
            continue

        if state == 0:
            if z > entry_threshold:
                state = -1
                periods_in_position = 0
            elif z < -entry_threshold:
                state = 1
                periods_in_position = 0
        else:
            periods_in_position += 1
            if abs(z) <= exit_threshold:
                state = 0
                periods_in_position = 0
            elif max_holding_periods is not None and periods_in_position >= max_holding_periods:
                forced_exit_dates.append(cast(pd.Timestamp, date))
                state = 0
                periods_in_position = 0

        states.append(state)

    return EntryExitResult(
        position_state=pd.Series(states, index=zscore.index),
        forced_exit_dates=forced_exit_dates,
    )
