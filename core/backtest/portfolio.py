"""Position and portfolio return accounting.

Per REQUIREMENTS.md 4.3, fills happen at the NEXT day's open, not the same
day's close a signal was decided on. That single fact means a day where a
position CHANGES cannot be scored with a naive ``position * close_to_close``
formula: the changed portion of the position was only actually held for
part of that day.

Every day's return for a single ticker decomposes into up to three pieces,
based on how much of position(t) overlaps with position(t-1) in sign:

    carried = same-sign overlap between position(t-1) and position(t)
              -> held all day -> close-to-close
    new     = position(t) - carried
              -> established at today's open -> open-to-close
    exited  = position(t-1) - carried
              -> held overnight, sold at today's open -> close(t-1)-to-open(t)

The "exited" piece matters because it is NOT present in position(t) at all,
so it would be silently dropped by any formula that only looks at
position(t) times a same-day return. It is real economic exposure: the
shares were held through the close(t-1)-to-open(t) gap before being sold.

The very first observation in any Series passed to compute_ticker_returns
is always treated as coming from a flat (zero) book -- there is no prior
bar to reference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd


def compute_ticker_returns(position: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
    """Per-day return contribution of one ticker's position.

    Parameters
    ----------
    position:
        Position held during each day (already lagged by EXECUTION_LAG
        upstream in the engine).
    open_, close:
        Same-index open/close price Series for this ticker.
    """
    prev_position = position.shift(1).fillna(0.0)
    prev_close = close.shift(1)

    pos_sign = np.sign(position)
    prev_sign = np.sign(prev_position)
    same_sign = pos_sign == prev_sign

    carried = pd.Series(
        np.where(same_sign, pos_sign * np.minimum(position.abs(), prev_position.abs()), 0.0),
        index=position.index,
    )
    new = position - carried
    exited = prev_position - carried

    # carried/exited multiply prev_close, which is NaN on the very first
    # bar (no prior close to reference). carried and exited are already
    # structurally 0 there (prev_position is forced to 0), but 0 * NaN is
    # NaN, not 0 -- clean up that arithmetic artifact explicitly.
    carry_return = (carried * (close / prev_close - 1.0)).fillna(0.0)
    # new_return multiplies open_, which can be NaN on a data gap. Unlike
    # carried/exited (whose NaN-triggering factor, prev_close, is NaN only
    # on the very first bar -- where prev_position is ALSO structurally
    # forced to 0), open_ can be NaN on ANY bar, independent of whether
    # that bar has a real entry (new != 0). So a blanket .fillna(0.0)
    # would silently zero out a genuine unpriceable trade, not just the
    # 0 * NaN arithmetic artifact. Mask only where new is structurally
    # zero (no entry that day); let a real entry with a missing open
    # price propagate as NaN rather than being hidden as a phantom 0.0
    # return.
    new_return = (new * (close / open_ - 1.0)).where(new != 0, 0.0)
    exit_return = (exited * (open_ / prev_close - 1.0)).fillna(0.0)

    return carry_return + new_return + exit_return


def compute_turnover(positions: pd.DataFrame) -> pd.Series:
    """Portfolio-level turnover per day.

    Sums the absolute per-ticker position change across ALL tickers --
    deliberately NOT netted. A pairs trade that goes +0.3/-0.3 in the same
    bar is two real trades (turnover 0.6), not a netted no-op (turnover
    0.0); netting here would silently understate transaction costs for any
    long/short strategy.
    """
    prev = positions.shift(1).fillna(0.0)
    return (positions - prev).abs().sum(axis=1)


@dataclass(frozen=True)
class ExposureLimits:
    """Portfolio-level exposure caps. Any field left as ``None`` is not
    checked. Sector-concentration limits are deliberately out of scope here:
    they need a ticker -> sector mapping, and core.data.universe's
    get_constituents() (which would provide GICS sector) was itself
    deferred to Phase 3 -- this is a known, documented limitation, not an
    oversight.
    """

    max_gross: float | None = None
    max_net: float | None = None
    max_per_ticker: float | None = None


class ExposureLimitWarning(UserWarning):
    """Raised (via warnings.warn) by run_backtest when exposure_limits are
    breached and exposure_limits_strict=False (the default): loud enough
    to notice, but never stops a batch job (walk-forward/sensitivity grids,
    an interactive dashboard slider) the way an exception would."""


class ExposureLimitError(RuntimeError):
    """Raised by run_backtest instead of warning when
    BacktestConfig.exposure_limits_strict=True. Opt-in hard-stop for
    contexts (e.g. a CI gate) that want a breach to fail loudly rather
    than merely warn -- mirrors PairSelectionConfig.multiple_testing_
    correction's pattern of making a policy trade-off an explicit config
    choice rather than a single global decision."""


@dataclass(frozen=True)
class ExposureViolation:
    date: pd.Timestamp
    kind: str  # "gross" | "net" | "per_ticker"
    ticker: str | None
    value: float
    limit: float


def check_exposure_limits(positions: pd.DataFrame, limits: ExposureLimits) -> list[ExposureViolation]:
    """Report exposure-limit violations; never raises.

    This deliberately returns facts only -- an empty list means no
    violations were found. Whether a violation should be logged, warned
    about, or treated as fatal is left entirely to the caller:
    REQUIREMENTS.md 4.5's own risk-response policy isn't finalized yet, so
    hard-coding a raise/warn choice here would be premature.

    Role separation from position_sizing.PositionSizingConfig.max_leverage:
    that is a PER-TICKER safety net against a single position's own
    volatility/Kelly estimate blowing up. It says nothing about the
    portfolio as a whole -- a multi-factor strategy holding many names can
    have every single position comfortably within its own leverage cap
    while the portfolio's summed gross exposure still climbs far beyond any
    sane limit. check_exposure_limits is the only place that checks that.
    """
    violations: list[ExposureViolation] = []

    if limits.max_gross is not None:
        gross = positions.abs().sum(axis=1)
        for date, value in gross[gross > limits.max_gross].items():
            violations.append(
                ExposureViolation(
                    date=cast(pd.Timestamp, date),
                    kind="gross",
                    ticker=None,
                    value=value,
                    limit=limits.max_gross,
                )
            )

    if limits.max_net is not None:
        net = positions.sum(axis=1).abs()
        for date, value in net[net > limits.max_net].items():
            violations.append(
                ExposureViolation(
                    date=cast(pd.Timestamp, date),
                    kind="net",
                    ticker=None,
                    value=value,
                    limit=limits.max_net,
                )
            )

    if limits.max_per_ticker is not None:
        exceeded = positions.abs() > limits.max_per_ticker
        for ticker in positions.columns:
            for date, value in positions.loc[exceeded[ticker], ticker].items():
                violations.append(
                    ExposureViolation(
                        date=cast(pd.Timestamp, date),
                        kind="per_ticker",
                        ticker=ticker,
                        value=abs(value),
                        limit=limits.max_per_ticker,
                    )
                )

    return violations


def enforce_exposure_limits(violations: list[ExposureViolation], *, strict: bool) -> None:
    """Given already-computed violations (from check_exposure_limits),
    resolve the policy question that function deliberately leaves open:
    raise (strict=True) or emit one AGGREGATED warning per violation kind
    (strict=False, the default). check_exposure_limits itself stays an
    unmodified, pure fact-finder -- this is the only place a raise/warn
    choice is made."""
    if not violations:
        return

    if strict:
        kinds = sorted({v.kind for v in violations})
        raise ExposureLimitError(f"Exposure limit(s) breached: {', '.join(kinds)} (see BacktestResult.exposure_violations for detail)")

    by_kind: dict[str, list[ExposureViolation]] = {}
    for v in violations:
        by_kind.setdefault(v.kind, []).append(v)

    for kind, kind_violations in by_kind.items():
        max_value = max(v.value for v in kind_violations)
        limit = kind_violations[0].limit
        n_days = len({v.date for v in kind_violations})
        warnings.warn(
            f"{kind} exposure exceeded limit on {n_days} day(s): max={max_value:.2f}, limit={limit:.2f}",
            ExposureLimitWarning,
            stacklevel=3,
        )
