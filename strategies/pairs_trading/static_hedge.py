"""Static (constant) hedge ratio -- the baseline REQUIREMENTS.md 5.1 asks
to compare against the Kalman-filter dynamic estimate.

This module does NOT re-estimate anything. cointegration.py's
select_pairs() already fits the Engle-Granger OLS regression, in-sample,
as part of testing for cointegration; re-running that same regression
here on the same data would just reproduce the identical number. This
module's only job is to broadcast that already-selected value across a
date index, in the SAME shape (HedgeRatioSeries) kalman_hedge.py produces
-- so strategies/pairs_trading/signal.py (and any later comparison
tooling) can treat "static" and "dynamic" hedge ratios uniformly without
caring which one produced a given series.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.pairs_trading.cointegration import PairCandidate


@dataclass(frozen=True)
class HedgeRatioSeries:
    """The shared output shape for both static_hedge_ratio and
    kalman_hedge_ratio: a per-date hedge ratio of ``ticker_dependent`` in
    terms of ``ticker_independent`` (constant for the static case,
    time-varying for the Kalman case)."""

    ticker_dependent: str
    ticker_independent: str
    intercept: pd.Series
    ratio: pd.Series


def static_hedge_ratio(candidate: PairCandidate, index: pd.Index) -> HedgeRatioSeries:
    """Broadcast candidate.engle_granger's already-selected (in-sample)
    intercept/hedge ratio as constant series across ``index`` (typically
    the full in-sample+out-of-sample backtest date range)."""
    eg = candidate.engle_granger
    return HedgeRatioSeries(
        ticker_dependent=eg.dependent,
        ticker_independent=eg.independent,
        intercept=pd.Series(eg.intercept, index=index),
        ratio=pd.Series(eg.hedge_ratio, index=index),
    )
