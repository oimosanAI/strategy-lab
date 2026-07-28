"""Multi-asset backtest engine core -- structural and empirical causality
guards.

A trading signal generated at time ``t`` -- using information available
only up to and including ``t`` -- can **never** earn the return realised
over ``(t-1, t]``. It can only govern positions from ``t+1`` onward.

This module provides two independent layers of defence against violating
that property, matching the pattern proven in the sibling
``backtest-framework`` repo (``src/backtest/engine.py``), generalised here
from a single price Series to a multi-ticker price panel:

1. **Structural guard** (mechanism-based): ``EXECUTION_LAG`` is applied in
   exactly one place -- the engine's own position-from-signal step, not
   inside any strategy -- so no strategy can accidentally or deliberately
   trade on the same bar that produced its signal.

2. **Empirical guard** (:func:`assert_causal`): proves, by perturbation,
   that a strategy's signals at time ``t`` do not depend on any price at
   time ``> t``, for every ticker. This catches violations regardless of
   *how* a strategy computes its signals -- vectorised rolling windows,
   explicit sequential loops (e.g. a Kalman filter), or a parameter fitted
   once over the whole sample all get the same test. The guard is only
   trustworthy if it has been shown to catch a planted bug, not merely to
   pass -- see the negative controls in ``tests/core/test_engine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtest import portfolio
from core.backtest.position_sizing import PositionSizer
from strategies.base import Strategy

# Number of bars between a signal being *decided* and it being *executed*.
# A signal decided at the close of bar t is acted on at bar t + 1. Applied
# in exactly one place downstream (position sizing / the engine's run
# loop), never inside a strategy.
EXECUTION_LAG: int = 1


class LookAheadError(AssertionError):
    """Raised when a strategy is shown to depend on future information."""


def assert_causal(
    strategy: Strategy,
    prices: pd.DataFrame,
    *,
    n_trials: int = 8,
    perturbation: float = 0.10,
    seed: int = 0,
) -> None:
    """Prove, by perturbation, that ``strategy`` has no look-ahead bias.

    Picks a cut point ``k``, perturbs every ticker's price *after* ``k`` by
    a random multiplicative shock, regenerates the signals, and requires
    that all signals at indices ``<= k`` (every ticker) are unchanged. If
    the strategy peeked into the future -- directly, or indirectly via a
    parameter fitted over the whole sample -- changing future prices would
    ripple back into earlier signals and this check fails.

    Raises
    ------
    LookAheadError
        If any signal at or before a cut point changes when only prices
        after that cut point are altered.
    """
    n = len(prices)
    if n < 3:
        raise ValueError("need at least 3 observations to test causality")

    baseline = strategy.generate_signals(prices).reindex(index=prices.index, columns=prices.columns)
    rng = np.random.default_rng(seed)

    for _ in range(n_trials):
        k = int(rng.integers(1, n - 1))

        shocked = prices.copy()
        future = shocked.iloc[k + 1 :]
        multipliers = 1.0 + rng.uniform(-perturbation, perturbation, size=future.shape)
        shocked.iloc[k + 1 :] = future.to_numpy() * multipliers

        perturbed = strategy.generate_signals(shocked).reindex(index=prices.index, columns=prices.columns)

        past_before = baseline.iloc[: k + 1]
        past_after = perturbed.iloc[: k + 1]

        if not past_before.equals(past_after):
            mismatch = ~_frame_equal_elementwise(past_before, past_after)
            rows, cols = np.where(mismatch.to_numpy())
            diff_locations = [
                (past_before.index[r], past_before.columns[c]) for r, c in zip(rows, cols)
            ]
            raise LookAheadError(
                "Look-ahead bias detected: perturbing prices after index "
                f"{k} changed earlier signals at {diff_locations!r}. A causal "
                "strategy's past signals must not depend on future prices."
            )


def _frame_equal_elementwise(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Element-wise equality treating NaN == NaN as True."""
    both_nan = a.isna() & b.isna()
    return (a == b) | both_nan


@dataclass(frozen=True)
class BacktestConfig:
    """Execution frictions and capital scale.

    Attributes
    ----------
    commission, slippage:
        Proportional cost per unit of turnover (REQUIREMENTS.md 7.1
        defaults: 0.5bps commission, 5bps slippage, one-way). Applied
        together as a single cost_rate = commission + slippage.
    initial_capital:
        Starting capital for the equity curve; affects scale only.
    """

    commission: float = 0.00005
    slippage: float = 0.0005
    initial_capital: float = 1.0


@dataclass(frozen=True)
class BacktestResult:
    """Full output of a backtest. Every field shares an index/columns
    layout aligned to the input price panel, so they can be sliced and
    compared directly.

    Attributes
    ----------
    signals:
        Raw target exposure the strategy decided at each date (not yet
        sized or lagged).
    target_position:
        ``signals`` after position sizing, still not lagged.
    position:
        ``target_position`` shifted by EXECUTION_LAG -- the position
        actually held during each bar. This is what meets the returns.
    turnover, costs, gross_returns, returns, equity_curve:
        Portfolio-level (summed across tickers) series; see
        core.backtest.portfolio for how turnover and per-ticker returns
        are computed.
    """

    signals: pd.DataFrame
    target_position: pd.DataFrame
    position: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_returns: pd.Series
    returns: pd.Series
    equity_curve: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    strategy: Strategy,
    sizer: PositionSizer,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a vectorised, multi-asset backtest.

    Parameters
    ----------
    prices:
        Close-price panel (index=date, columns=ticker) used for signal
        generation and position sizing.
    open_prices:
        Open-price panel, same shape, used as the fill price
        (REQUIREMENTS.md 4.3: next-day-open execution) and for the
        partial-day return on entry/exit days (see
        core.backtest.portfolio.compute_ticker_returns).
    strategy, sizer:
        Anything satisfying the Strategy / PositionSizer protocols.
    config:
        Execution frictions and capital scale; defaults to
        BacktestConfig().
    """
    config = config or BacktestConfig()

    signals = strategy.generate_signals(prices)
    target_position = sizer.apply(signals, prices)

    # === LOOK-AHEAD GUARD ==================================================
    # Applied in exactly this one place, regardless of what strategy or
    # sizer produced target_position: a decision made using information up
    # to bar t cannot be live any earlier than bar t+1.
    position = target_position.shift(EXECUTION_LAG).fillna(0.0)
    # ========================================================================

    per_ticker_returns = pd.DataFrame(
        {
            ticker: portfolio.compute_ticker_returns(position[ticker], open_prices[ticker], prices[ticker])
            for ticker in prices.columns
        },
        index=prices.index,
    )
    gross_returns = per_ticker_returns.sum(axis=1)

    turnover = portfolio.compute_turnover(position)
    costs = turnover * (config.commission + config.slippage)
    net_returns = gross_returns - costs
    equity_curve = (1.0 + net_returns).cumprod() * config.initial_capital

    return BacktestResult(
        signals=signals,
        target_position=target_position,
        position=position,
        turnover=turnover,
        costs=costs,
        gross_returns=gross_returns,
        returns=net_returns,
        equity_curve=equity_curve,
    )


def assert_backtest_causal(
    strategy: Strategy,
    sizer: PositionSizer,
    close_prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    *,
    n_trials: int = 8,
    perturbation: float = 0.10,
    seed: int = 0,
) -> None:
    """Whole-pipeline causality guard.

    :func:`assert_causal` only verifies a Strategy's own signal generation.
    A bug could still be introduced later -- in position sizing (e.g. a
    volatility estimate that reads the whole sample) or in portfolio
    accounting -- and slip past that check entirely. This perturbs both
    close and open prices after a cut point ``k`` and asserts that
    ``BacktestResult.position`` and ``.returns`` (what any downstream
    evaluation would consume) are unchanged at or before ``k``, covering
    the sizing and accounting stages assert_causal alone doesn't reach.

    Raises
    ------
    LookAheadError
        If position or returns at or before a cut point change when only
        prices after that cut point are altered.
    """
    n = len(close_prices)
    if n < 3:
        raise ValueError("need at least 3 observations to test causality")

    baseline = run_backtest(close_prices, open_prices, strategy, sizer, config)
    rng = np.random.default_rng(seed)

    for _ in range(n_trials):
        k = int(rng.integers(1, n - 1))

        shocked_close = close_prices.copy()
        shocked_open = open_prices.copy()
        future_close = shocked_close.iloc[k + 1 :]
        future_open = shocked_open.iloc[k + 1 :]
        shocked_close.iloc[k + 1 :] = future_close.to_numpy() * (
            1.0 + rng.uniform(-perturbation, perturbation, size=future_close.shape)
        )
        shocked_open.iloc[k + 1 :] = future_open.to_numpy() * (
            1.0 + rng.uniform(-perturbation, perturbation, size=future_open.shape)
        )

        perturbed = run_backtest(shocked_close, shocked_open, strategy, sizer, config)

        for field_name in ("position", "returns"):
            before = getattr(baseline, field_name).iloc[: k + 1]
            after = getattr(perturbed, field_name).iloc[: k + 1]
            if not before.equals(after):
                raise LookAheadError(
                    "Look-ahead bias detected in run_backtest: perturbing "
                    f"close/open prices after index {k} changed '{field_name}' "
                    f"at or before that index. A causal backtest's past "
                    "position/returns must not depend on future prices."
                )
