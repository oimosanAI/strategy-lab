"""Factor definitions (REQUIREMENTS.md 5.2): 12-1 momentum and
60-day low-volatility.

Both functions are pure rolling-window computations over a full price
panel -- causal by construction (row t depends only on prices.iloc[:t+1]),
the same pattern as strategies.pairs_trading.signal.rolling_zscore.
Neither reads REQUIREMENTS.md's default window lengths from any
in-sample-fitted value; they are fixed, literature-standard definitions,
not parameters calibrated from performance -- so no
core.backtest.sample_split.assert_selection_ignores_out_of_sample-style
guard applies here (see strategies/factor_momentum/README.md once written
for the fuller reasoning). core.backtest.engine.assert_causal is the
relevant guard, applied to the full strategy in strategy.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_score(prices: pd.DataFrame, lookback_days: int = 252, skip_days: int = 21) -> pd.DataFrame:
    """12-1 momentum: price[t - skip_days] / price[t - lookback_days] - 1.

    Skipping the most recent `skip_days` avoids the well-documented
    short-term reversal effect contaminating the momentum signal. NaN for
    t < lookback_days (insufficient history) -- left as NaN rather than
    an invented value, matching rolling_zscore's warm-up convention.
    """
    lagged = prices.shift(skip_days)
    return lagged / lagged.shift(lookback_days - skip_days) - 1.0


def low_volatility_score(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling annualized realized volatility of daily returns.

    Same formula as core.backtest.position_sizing.VolTargetSizer's
    realized_vol, reused here as a ranking score (lower = more attractive
    for the low-volatility factor) rather than a sizing input -- ranking
    into long/short buckets is ranking.py's responsibility, not this
    function's.
    """
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)
