"""Volatility Risk Premium (VRP) signal computation (REQUIREMENTS.md 5.3).

VRP(t) = VIX(t)/100 - realized_volatility(SPY, window)(t): the gap
between the market's forward-looking implied volatility (VIX) and the
TRAILING realized volatility of the S&P 500 itself. Both sides are
causal rolling-window quantities -- row t depends only on data up to and
including t, the same pattern as strategies.factor_momentum.factors.

IMPORTANT UNIT CONVERSION: yfinance's ^VIX "Close" is quoted in
PERCENTAGE POINTS (e.g. 18.67 meaning 18.67% annualized implied
volatility), confirmed against real data before this was implemented --
NOT a decimal fraction like realized_volatility returns (e.g. 0.05 for
5%). Dividing by 100 here is not cosmetic: without it, VRP would be
dominated by VIX's raw ~8-60 scale and the RV term would be numerically
irrelevant -- a "runs without error but means nothing" bug, caught by
test_compute_vrp_converts_vix_percentage_points_to_decimal before this
module was ever wired into a strategy.

IMPORTANT SCOPE NOTE: this uses TRAILING (backward-looking) realized
volatility, not the FORWARD (subsequently-realized) volatility that the
well-documented academic "VIX overstates realized volatility" finding
is usually measured against. Trailing RV assumes some persistence in
volatility (volatility clustering) to serve as a usable, CAUSAL proxy for
"what RV is likely to look like going forward" -- an assumption, not a
certainty. See strategies/vol_arbitrage/README.md for the fuller
discussion of this distinction, including Step F/section 3-7's real-data
diagnostic showing this trailing-RV design leaves the signal on through
87% of days and does not de-risk ahead of a spike -- the motivation for
compute_term_structure_ratio below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility(prices: pd.Series, window: int = 21) -> pd.Series:
    """Rolling annualized realized volatility of daily returns. Same
    formula as strategies.factor_momentum.factors.low_volatility_score,
    reused here for SPY rather than duplicating the math."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def compute_vrp(vix: pd.Series, spy_prices: pd.Series, window: int = 21) -> pd.Series:
    """VIX (percentage points, divided by 100) minus trailing realized
    volatility of SPY (decimal fraction) -- see module docstring for the
    unit-conversion and trailing-vs-forward-RV caveats."""
    rv = realized_volatility(spy_prices, window)
    return (vix / 100.0) - rv


def compute_term_structure_ratio(vix9d: pd.Series, vix: pd.Series) -> pd.Series:
    """VIX9D / VIX: near-term (9-day) implied volatility relative to the
    standard 30-day VIX. Both are forward-looking, options-implied
    quantities in the SAME percentage-point scale, so no unit conversion
    is needed (unlike compute_vrp, which compares VIX against a
    decimal-fraction realized volatility).

    ratio < 1 (contango): near-term IV priced below 30-day IV -- the
    normal, calm-market state. ratio >= 1 (backwardation): near-term IV
    exceeds 30-day IV -- the market is pricing elevated near-term risk, a
    well-documented volatility term-structure inversion. Unlike
    compute_vrp's trailing realized volatility, this has no rolling
    window and therefore no warm-up period: it is a same-row quantity,
    causal by construction as long as VIX9D(t) and VIX(t) are both known
    as of t (the same assumption already made for VIX/SPY elsewhere in
    this module).
    """
    return vix9d / vix
