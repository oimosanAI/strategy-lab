"""Cross-sectional ranking: combine momentum and low-volatility scores
into a single composite percentile, bucket into long/short/neutral, and
normalize each leg to sum to 1.0 (equal-weight within the leg).

Both factor scores are on incomparable scales (momentum is a return
ratio; low-volatility is an annualized standard deviation), so they are
combined via cross-sectional PERCENTILE RANK (pandas' rank(axis=1,
pct=True), scale-invariant) rather than a raw weighted average.
rank(pct=True) excludes NaN inputs from both the ranking itself and the
percentile denominator, which this module relies on to implement
"exclude NaN-scored tickers from the ranking pool" (REQUIREMENTS.md 5.2)
without any extra NaN-handling code.

PER-LEG NORMALIZATION (added after the first real-data E2E run surfaced
it as a genuine gap, not merely a style choice -- see
strategies/factor_momentum/README.md once written): an un-normalized
+-1 signal, applied to a broad universe where a whole quintile (~100
names in a full S&P 500 scan) is simultaneously long or short, lets
core.backtest.position_sizing.VolTargetSizer's PER-TICKER-only
max_leverage cap combine into portfolio-level gross exposure with no
ceiling at all (56x was observed in that run) -- the number of names in
a leg was never accounted for anywhere in the pipeline. Normalizing each
leg to sum to exactly 1.0 (long leg) / -1.0 (short leg) here fixes the
root cause once, rather than leaving every future caller to remember to
divide by leg size themselves.

CONTRACT: build_long_short_signal never emits NaN -- every cell is a
real number (a leg-normalized weight, or 0.0 for neutral/excluded/no-
candidates-that-day). This matters downstream: rebalance.
hold_until_next_rebalance treats a NaN value AT a rebalance date as "no
decision, keep holding the previous one" -- if this module ever leaked
NaN through, a ticker temporarily missing data would incorrectly keep
last month's stale position instead of flattening to 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorMomentumConfig:
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    low_vol_window: int = 60
    long_percentile: float = 0.80
    short_percentile: float = 0.20
    # Sector-neutral construction only (strategies.factor_momentum.
    # sector_neutral.build_sector_neutral_signal): a sector with fewer than
    # this many VALID (non-NaN-scored) candidates on a given date is
    # excluded entirely for that date, rather than forcing a degenerate
    # quintile split on too few names (e.g. 1 name representing a whole
    # sector's factor bet). 10 is chosen so quintile (20%) still yields at
    # least ~2 names per leg even in the smallest real GICS sectors
    # (Energy had 21 names in the full-universe E2E scan).
    min_names_per_sector: int = 10


def build_long_short_signal(
    momentum: pd.DataFrame, low_vol: pd.DataFrame, config: FactorMomentumConfig
) -> pd.DataFrame:
    """Composite cross-sectional rank of momentum (higher=better) and
    low_vol (lower=better, hence 1 - rank), bucketed by config's
    percentile thresholds, then equal-weighted within each leg so the
    long leg sums to 1.0 and the short leg sums to -1.0 (0.0 for either
    leg on a day with no qualifying candidates). See module docstring for
    the normalization rationale and the NaN contract."""
    momentum_rank = momentum.rank(axis=1, pct=True)
    low_vol_rank = 1.0 - low_vol.rank(axis=1, pct=True)
    composite = (momentum_rank + low_vol_rank) / 2.0

    long_mask = (composite >= config.long_percentile).astype(float)
    short_mask = (composite <= config.short_percentile).astype(float)

    n_long = long_mask.sum(axis=1).replace(0.0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0.0, np.nan)

    long_weight = long_mask.div(n_long, axis=0).fillna(0.0)
    short_weight = short_mask.div(n_short, axis=0).fillna(0.0)
    return long_weight - short_weight
