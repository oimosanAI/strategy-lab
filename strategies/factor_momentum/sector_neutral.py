"""Sector-neutral long/short construction (REQUIREMENTS.md 5.2 extension:
sector neutralization).

Reuses ranking.build_long_short_signal UNCHANGED, once per sector (a
sub-slice of momentum/low_vol restricted to that sector's own tickers),
then re-normalizes across sectors so every sector gets an EQUAL budget in
the final portfolio (long leg sums to 1.0, short leg to -1.0) regardless
of how many names that sector contains. The global, non-sector-neutral
ranking.build_long_short_signal remains available and unchanged for
callers who don't want sector neutrality -- both are legitimate
configurations, not a replacement of one by the other.

CROSS-SECTOR NORMALIZATION relies on a property of
build_long_short_signal's own output: an ACTIVE sector's long leg sums to
EXACTLY 1.0 (short leg to EXACTLY -1.0, see ranking.py's own contract).
Summing the combined long-leg values across ALL tickers therefore IS the
count of active long sectors that day -- no separate counting logic is
needed, which is also why there is little room for a new class of bug
here: the same per-leg-normalization invariant that already fixed the
gross-exposure blowup (see this package's README.md) does the counting
for free.

config.min_names_per_sector: a sector with fewer than this many VALID
(non-NaN momentum AND non-NaN low_vol) candidates on a date is excluded
entirely for that date -- all its tickers become 0.0 -- rather than
forcing a degenerate quintile split on too few names (e.g. a single stock
representing an entire sector's factor bet).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from strategies.factor_momentum.ranking import FactorMomentumConfig, build_long_short_signal


def _group_tickers_by_sector(tickers: Iterable[str], sector_map: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for ticker in tickers:
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        groups[sector].append(ticker)
    return groups


def build_sector_neutral_signal(
    momentum: pd.DataFrame,
    low_vol: pd.DataFrame,
    sector_map: dict[str, str],
    config: FactorMomentumConfig,
) -> pd.DataFrame:
    """Sector-neutral composite: quintile within each sector, then an
    equal budget per sector. See module docstring for the normalization
    rationale and the NaN/threshold contract."""
    groups = _group_tickers_by_sector(momentum.columns, sector_map)

    per_sector_signals: list[pd.DataFrame] = []
    for tickers in groups.values():
        sector_momentum = momentum[tickers]
        sector_low_vol = low_vol[tickers]

        sector_signal = build_long_short_signal(sector_momentum, sector_low_vol, config)

        valid_counts = (sector_momentum.notna() & sector_low_vol.notna()).sum(axis=1)
        below_threshold = valid_counts < config.min_names_per_sector
        sector_signal = sector_signal.mask(below_threshold, 0.0)

        per_sector_signals.append(sector_signal)

    combined = pd.concat(per_sector_signals, axis=1)

    long_part = combined.where(combined > 0, 0.0)
    short_part = combined.where(combined < 0, 0.0)

    n_active_long = long_part.sum(axis=1).replace(0.0, np.nan)
    n_active_short = short_part.sum(axis=1).abs().replace(0.0, np.nan)

    final_long = long_part.div(n_active_long, axis=0).fillna(0.0)
    final_short = short_part.div(n_active_short, axis=0).fillna(0.0)

    return final_long + final_short
