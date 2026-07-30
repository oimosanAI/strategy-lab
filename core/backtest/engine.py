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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.backtest import portfolio
from core.backtest.portfolio import ExposureLimits
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
    """Execution frictions, capital scale, and portfolio exposure policy.

    Attributes
    ----------
    commission, slippage:
        Proportional cost per unit of turnover (REQUIREMENTS.md 7.1
        defaults: 0.5bps commission, 5bps slippage, one-way). Applied
        together as a single cost_rate = commission + slippage.
    initial_capital:
        Starting capital for the equity curve; affects scale only.
    exposure_limits:
        Portfolio-level gross/net exposure caps, checked automatically by
        run_backtest (see portfolio.check_exposure_limits). Defaults to a
        deliberately generous "catastrophic sanity net" (max_gross=5.0,
        max_net=5.0) -- NOT a per-strategy-tuned value. It is derived from
        PositionSizingConfig.max_leverage's own default (2.0) times the
        largest number of simultaneously-uncancelled legs any of this
        project's current strategies can have (2, for pairs_trading),
        plus headroom, so it never fires under any of the 3 strategies'
        legitimate operation while still catching a repeat of the
        historical factor_momentum blowout (median 39.9x, max 56.3x
        gross -- see strategies/factor_momentum/README.md) almost
        immediately. max_net equals max_gross rather than a tighter value
        because net <= gross always holds; a tighter default would
        misfire on vol_arbitrage, whose long-only single-ticker design
        makes net == gross by construction, not by malfunction.
        max_per_ticker defaults to None (unchecked) because
        PositionSizingConfig.max_leverage already clips each ticker at
        the sizing stage -- a default here would be redundant, not
        additional protection.
    exposure_limits_strict:
        False (default): a breach emits ONE aggregated warnings.warn()
        per violation kind (gross/net/per_ticker) via
        portfolio.ExposureLimitWarning, and the backtest still completes
        normally with the breach recorded in
        BacktestResult.exposure_violations -- matching this project's
        walk-forward/sensitivity convention of never letting one bad
        configuration abort a batch run. True: raises
        portfolio.ExposureLimitError instead, for contexts (e.g. a CI
        gate) that want a breach to fail loudly. Note: if True, a
        perturbation trial inside assert_backtest_causal that happens to
        push exposure over the limit would raise ExposureLimitError
        instead of confirming causality -- harmless under the default
        False, since no causality test in this project opts into strict
        mode.
    """

    commission: float = 0.00005
    slippage: float = 0.0005
    initial_capital: float = 1.0
    exposure_limits: ExposureLimits = field(default_factory=lambda: ExposureLimits(max_gross=5.0, max_net=5.0))
    exposure_limits_strict: bool = False


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
        are computed. equity_curve is floored at 0.0 from ruin_date
        onward (see ruin_date below) -- returns itself is left
        unmodified, since it is a per-day rate and remains a faithful
        record of what the (still fully leveraged, unrealistically
        continuing) position would have returned; only the compounded
        equity_curve is nonsensical past total loss and is what gets
        floored.
    exposure_violations:
        Always present (empty list if config.exposure_limits found
        nothing to flag, or if exposure_limits itself is unchecked). See
        BacktestConfig.exposure_limits/.exposure_limits_strict for the
        policy this is checked against.
    ruin_date:
        The first date on which the RAW (unfloored) equity curve would
        reach <= 0.0, or None if it never does. exposure_limits (even at
        strict=False's aggregated-warning default) is meant to catch
        this long before it happens; ruin_date existing at all on a real
        run is itself a signal something upstream is badly misconfigured
        -- see BacktestConfig.exposure_limits's docstring for why the
        default (max_gross=max_net=5.0) is sized as a catastrophic
        sanity net, not a per-strategy-tuned value. NOT compared by
        assert_backtest_causal (see that function's docstring for why:
        equity_curve floors to a constant 0.0 for every date from
        ruin_date onward regardless of what happens to prices afterward,
        which would make a look-ahead bug occurring entirely within an
        already-ruined stretch invisible to a perturbation test that
        only inspects equity_curve -- returns and the other compared
        fields are NOT floored and remain fully sensitive to such a bug).
    """

    signals: pd.DataFrame
    target_position: pd.DataFrame
    position: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_returns: pd.Series
    returns: pd.Series
    equity_curve: pd.Series
    exposure_violations: list[portfolio.ExposureViolation]
    ruin_date: pd.Timestamp | None


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

    Raises
    ------
    ValueError
        If ``prices`` and ``open_prices`` do not share the same
        (index, columns). Without this check, pandas would silently
        align the two DataFrames on their shared dates/tickers -- close
        and open panels loaded from two independently-fetched or
        independently-regenerated sources (e.g. dashboard/data.py's
        close/open snapshots, saved as two separate parquet files) can
        drift apart with no runtime signal otherwise, producing a
        plausible-looking but silently wrong result rather than an
        error at the boundary where the mistake was made.
    """
    if not prices.index.equals(open_prices.index):
        raise ValueError(
            "prices and open_prices must share the same index (dates); "
            f"got {len(prices.index)} vs {len(open_prices.index)} rows, "
            "first mismatch may be due to independently loaded/regenerated panels"
        )
    if not prices.columns.equals(open_prices.columns):
        raise ValueError(
            f"prices and open_prices must share the same columns (tickers) in the same order; "
            f"got {list(prices.columns)} vs {list(open_prices.columns)}"
        )

    config = config or BacktestConfig()

    signals = strategy.generate_signals(prices)
    target_position = sizer.apply(signals, prices)

    # === LOOK-AHEAD GUARD ==================================================
    # Applied in exactly this one place, regardless of what strategy or
    # sizer produced target_position: a decision made using information up
    # to bar t cannot be live any earlier than bar t+1.
    position = target_position.shift(EXECUTION_LAG).fillna(0.0)
    # ========================================================================

    # === EXPOSURE-LIMIT GUARD ===============================================
    # Checked on `position` (actually held, post-lag) rather than
    # target_position: that is the economically real exposure that earns
    # or loses money, and matches the level at which the factor_momentum
    # gross-exposure blowout was originally diagnosed. check_exposure_
    # limits itself stays a pure fact-finder; enforce_exposure_limits is
    # the only place the raise/warn policy decision is made.
    exposure_violations = portfolio.check_exposure_limits(position, config.exposure_limits)
    portfolio.enforce_exposure_limits(exposure_violations, strict=config.exposure_limits_strict)
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

    # === RUIN FLOOR ==========================================================
    # (1 + net_returns).cumprod() alone has no floor: a single day with
    # net_returns < -1.0 (a >100% loss -- possible under leverage, and
    # what exposure_limits exists to make rare, not impossible) sends
    # equity negative, after which a POSITIVE return makes equity worse,
    # not better. `ever_ruined` is a purely left-to-right cumulative scan
    # (each date depends only on raw equity at or before it), so this
    # introduces no look-ahead of its own.
    raw_equity_curve = (1.0 + net_returns).cumprod() * config.initial_capital
    ruined = raw_equity_curve <= 0.0
    ever_ruined = ruined.cummax()
    equity_curve = raw_equity_curve.where(~ever_ruined, 0.0)
    ruin_date = raw_equity_curve.index[int(ruined.to_numpy().argmax())] if ruined.any() else None
    # ========================================================================

    return BacktestResult(
        signals=signals,
        target_position=target_position,
        position=position,
        turnover=turnover,
        costs=costs,
        gross_returns=gross_returns,
        returns=net_returns,
        equity_curve=equity_curve,
        exposure_violations=exposure_violations,
        ruin_date=ruin_date,
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
    close and open prices after a cut point ``k`` and asserts that every
    stage of ``BacktestResult`` -- ``signals`` and ``target_position``
    (pre-EXECUTION_LAG) as well as ``position`` and ``returns``
    (post-lag, what any downstream evaluation would consume) -- is
    unchanged at or before ``k``, covering the sizing and accounting
    stages assert_causal alone doesn't reach.

    ``signals``/``target_position`` must be compared in addition to
    ``position``/``returns``, not instead of them: EXECUTION_LAG shifts
    ``target_position`` forward by exactly one bar before it becomes
    ``position``, which means a signal that peeks exactly one bar into
    the future produces a ``position`` that is *still* causal by
    construction -- the lag absorbs precisely that one bar of leakage.
    Comparing only the post-lag fields would let a one-bar look-ahead in
    signal generation or sizing pass this guard silently; comparing the
    pre-lag fields as well closes that gap. See the negative control in
    tests/core/test_backtest_engine.py that plants exactly this bug.

    Raises
    ------
    LookAheadError
        If signals, target_position, position, or returns at or before a
        cut point change when only prices after that cut point are
        altered.
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

        for field_name in ("signals", "target_position", "position", "returns"):
            before = getattr(baseline, field_name).iloc[: k + 1]
            after = getattr(perturbed, field_name).iloc[: k + 1]
            if not before.equals(after):
                raise LookAheadError(
                    "Look-ahead bias detected in run_backtest: perturbing "
                    f"close/open prices after index {k} changed '{field_name}' "
                    f"at or before that index. A causal backtest's past "
                    "position/returns must not depend on future prices."
                )
