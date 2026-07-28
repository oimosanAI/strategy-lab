"""Monthly rebalancing: decide signals only at month-end, hold between.

REQUIREMENTS.md 5.2 calls for monthly rebalancing. The chosen design (see
strategies/pairs_trading/README.md-adjacent design discussion) is: the
daily factor-ranking signal is computed continuously (see factors.py,
ranking.py) but only actually ACTED ON at each month-end -- every other
day simply holds the previous month-end's decision forward. Position
*sizing* (volatility targeting) still runs daily via
core.backtest.position_sizing, independently of this monthly hold.

Both functions here are causal by construction:
- month_end_dates reads only the index already given to it.
- hold_until_next_rebalance's forward-fill only ever copies a PAST value
  into later rows, never a future one -- combined with a causal daily
  signal (factors.py's rolling windows), the composition stays causal,
  verified empirically via core.backtest.engine.assert_causal on the full
  strategy (strategy.py).
"""

from __future__ import annotations

import pandas as pd


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last date ACTUALLY PRESENT in `index` for each (year, month).

    Deliberately data-driven rather than a calendar-formula month-end
    (e.g. pandas' BMonthEnd): a real price panel's index is the source of
    truth for which dates actually traded, matching how
    core.backtest.sample_split treats the price panel's own index rather
    than reconstructing an idealized trading calendar that could desync
    from real gaps/holidays in the data.
    """
    grouped = index.to_series().groupby([index.year, index.month]).max()
    return pd.DatetimeIndex(grouped.to_numpy())


def hold_until_next_rebalance(daily_signal: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Keep `daily_signal` only at `rebalance_dates` rows, forward-filled
    to every date until the next rebalance date.

    Before the first rebalance date -- or while the value AT a rebalance
    date is itself NaN (e.g. a factor score still in its warm-up period)
    -- the result is 0.0 (flat), never a stale or invented value.
    """
    non_rebalance_mask = ~daily_signal.index.isin(rebalance_dates)
    masked = daily_signal.copy()
    masked.loc[non_rebalance_mask] = float("nan")
    return masked.ffill().fillna(0.0)
