"""Tests for strategies.factor_momentum.sector_neutral.

Fixes the public API surface:

- build_sector_neutral_signal(momentum, low_vol, sector_map, config) ->
  pd.DataFrame: reuses ranking.build_long_short_signal UNCHANGED, once per
  sector (a sub-slice of momentum/low_vol restricted to that sector's
  tickers), then re-normalizes across sectors so the OVERALL long leg
  sums to 1.0 and the overall short leg to -1.0 -- giving every sector an
  EQUAL budget regardless of how many names it contains.

  The cross-sector normalization relies on a property of
  build_long_short_signal's own per-sector output: an ACTIVE sector's
  long leg sums to EXACTLY 1.0 (short leg to EXACTLY -1.0), so summing
  the combined long-leg values across all tickers IS the count of active
  long sectors -- no separate counting logic is needed, which is also
  why there is little room for a new class of bug here.

- config.min_names_per_sector: a sector with fewer than this many VALID
  (non-NaN momentum AND non-NaN low_vol) candidates on a date is excluded
  entirely for that date (all its tickers -> 0.0), rather than forcing a
  degenerate quintile split on too few names.

Expected values below are hand-derived from the same rank(pct=True)
reasoning already used in test_ranking.py, applied independently within
each sector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.factor_momentum.ranking import FactorMomentumConfig, build_long_short_signal
from strategies.factor_momentum.sector_neutral import build_sector_neutral_signal


def _sector_map(tickers: list[str], sector: str) -> dict[str, str]:
    return {t: sector for t in tickers}


# ---------------------------------------------------------------------------
# 1. within-sector quintile selection + reuse of build_long_short_signal
# ---------------------------------------------------------------------------


def test_build_sector_neutral_signal_selects_within_sector_and_normalizes_across_sectors() -> None:
    # Arrange: 2 sectors, 5 tickers each, same monotonic pattern used in
    # test_ranking.py (composite(T_i) = (2i+1)/10 for n=5 -> long: i=4
    # (0.9), short: i=0 (0.1)). Each sector's own leg sums to 1.0/-1.0;
    # with 2 active sectors, cross-sector normalization divides by 2.
    idx = pd.bdate_range("2020-01-01", periods=1)
    a_tickers = [f"A{i}" for i in range(5)]
    b_tickers = [f"B{i}" for i in range(5)]
    tickers = a_tickers + b_tickers

    momentum = pd.DataFrame(
        [[0.01 * (i + 1) for i in range(5)] + [0.01 * (i + 1) for i in range(5)]], index=idx, columns=tickers
    )
    low_vol = pd.DataFrame(
        [[0.30 - 0.02 * i for i in range(5)] + [0.30 - 0.02 * i for i in range(5)]], index=idx, columns=tickers
    )
    sector_map = {**_sector_map(a_tickers, "SectorA"), **_sector_map(b_tickers, "SectorB")}
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    result = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    expected = pd.DataFrame(
        [[-0.5, 0.0, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.5]], index=idx, columns=tickers
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_sector_neutral_signal_gives_equal_sector_budget_despite_unequal_sector_sizes() -> None:
    # Arrange: SectorA has 5 tickers (n_long=1 within sector), SectorB has
    # 10 tickers (n_long=2 within sector, per test_ranking.py's original
    # fixture). Both sectors are active -> each gets an equal 0.5 total
    # budget, split among however many names qualify within that sector.
    idx = pd.bdate_range("2020-01-01", periods=1)
    a_tickers = [f"A{i}" for i in range(5)]
    b_tickers = [f"B{i}" for i in range(10)]
    tickers = a_tickers + b_tickers

    momentum = pd.DataFrame(
        [[0.01 * (i + 1) for i in range(5)] + [0.01 * (i + 1) for i in range(10)]], index=idx, columns=tickers
    )
    low_vol = pd.DataFrame(
        [[0.30 - 0.02 * i for i in range(5)] + [0.30 - 0.02 * i for i in range(10)]], index=idx, columns=tickers
    )
    sector_map = {**_sector_map(a_tickers, "SectorA"), **_sector_map(b_tickers, "SectorB")}
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    result = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    # SectorA's single long name (A4) takes the whole 0.5 sector budget.
    assert result["A4"].iloc[0] == 0.5
    # SectorB's two long names (B8, B9) split the same 0.5 sector budget.
    assert result["B8"].iloc[0] == 0.25
    assert result["B9"].iloc[0] == 0.25
    # Each sector's total long contribution is equal (0.5), despite
    # SectorB having twice as many tickers.
    a_total = result[a_tickers].where(result[a_tickers] > 0, 0.0).sum(axis=1).iloc[0]
    b_total = result[b_tickers].where(result[b_tickers] > 0, 0.0).sum(axis=1).iloc[0]
    assert a_total == b_total == 0.5


def test_build_sector_neutral_signal_ignores_tickers_missing_from_sector_map() -> None:
    # A ticker present in the price/score data but absent from sector_map
    # (unknown sector) is simply excluded from grouping entirely, the
    # same "sector = sector_map.get(ticker); if None: skip" pattern
    # already used by strategies.pairs_trading.cointegration.select_pairs.
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"A{i}" for i in range(5)] + ["Unknown"]
    momentum = pd.DataFrame(
        [[0.01 * (i + 1) for i in range(5)] + [0.99]], index=idx, columns=tickers
    )
    low_vol = pd.DataFrame(
        [[0.30 - 0.02 * i for i in range(5)] + [0.01]], index=idx, columns=tickers
    )
    sector_map = _sector_map([f"A{i}" for i in range(5)], "SectorA")  # "Unknown" deliberately absent
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    result = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    assert "Unknown" not in result.columns


def test_build_sector_neutral_signal_raises_when_no_ticker_has_a_sector() -> None:
    # Skipping SOME unmapped tickers is legitimate (the test above). A
    # sector_map matching NOTHING is a configuration error -- e.g. a map
    # keyed on company names instead of tickers. Returning an all-zero
    # signal there would let a misconfiguration masquerade as "the
    # strategy chose to stay flat", the same silent-failure mode fixed in
    # VolArbitrageStrategy. It previously surfaced as pandas' opaque
    # "No objects to concatenate" from pd.concat([]).
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"A{i}" for i in range(5)]
    momentum = pd.DataFrame([[0.01 * (i + 1) for i in range(5)]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.30 - 0.02 * i for i in range(5)]], index=idx, columns=tickers)
    config = FactorMomentumConfig(min_names_per_sector=2)

    with pytest.raises(ValueError, match="sector_map"):
        build_sector_neutral_signal(momentum, low_vol, {"NotAColumn": "SectorA"}, config)


# ---------------------------------------------------------------------------
# 2. min_names_per_sector threshold boundary
# ---------------------------------------------------------------------------


def test_build_sector_neutral_signal_includes_sector_exactly_at_min_names_threshold() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"A{i}" for i in range(4)]
    momentum = pd.DataFrame([[0.01 * (i + 1) for i in range(4)]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.30 - 0.02 * i for i in range(4)]], index=idx, columns=tickers)
    sector_map = _sector_map(tickers, "SectorA")
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=4)

    result = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    # 4 valid names == threshold -> included, normal quintile applies
    # (only sector active -> cross-sector normalization divides by 1).
    assert result["A3"].iloc[0] > 0.0
    assert result["A0"].iloc[0] < 0.0


def test_build_sector_neutral_signal_excludes_sector_below_min_names_threshold() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"A{i}" for i in range(4)]
    momentum = pd.DataFrame([[np.nan, 0.02, 0.03, 0.04]], index=idx, columns=tickers)  # only 3 valid
    low_vol = pd.DataFrame([[0.30, 0.25, 0.20, 0.15]], index=idx, columns=tickers)
    sector_map = _sector_map(tickers, "SectorA")
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=4)

    result = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    # 3 valid < threshold 4 -> whole sector excluded, including the 3
    # otherwise-valid names (not just A0's NaN, which would be 0 anyway).
    assert (result.iloc[0] == 0.0).all()


# ---------------------------------------------------------------------------
# 3. cross-sector normalization / active-sector-count derivation
# ---------------------------------------------------------------------------


def test_build_sector_neutral_signal_total_long_and_short_legs_sum_to_one() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)
    sectors = {}
    all_tickers: list[str] = []
    momentum_row: list[float] = []
    low_vol_row: list[float] = []
    for s in range(3):
        tks = [f"S{s}T{i}" for i in range(5)]
        all_tickers += tks
        momentum_row += [0.01 * (i + 1) for i in range(5)]
        low_vol_row += [0.30 - 0.02 * i for i in range(5)]
        sectors.update(_sector_map(tks, f"Sector{s}"))

    momentum = pd.DataFrame([momentum_row], index=idx, columns=all_tickers)
    low_vol = pd.DataFrame([low_vol_row], index=idx, columns=all_tickers)
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    result = build_sector_neutral_signal(momentum, low_vol, sectors, config)

    long_sum = result.where(result > 0, 0.0).sum(axis=1).iloc[0]
    short_sum = result.where(result < 0, 0.0).sum(axis=1).iloc[0]
    assert long_sum == 1.0
    assert short_sum == -1.0


def test_build_sector_neutral_signal_total_leg_still_sums_to_one_when_one_sector_excluded() -> None:
    idx = pd.bdate_range("2020-01-01", periods=1)
    sectors = {}
    all_tickers: list[str] = []
    momentum_row: list[float] = []
    low_vol_row: list[float] = []
    for s in range(2):
        tks = [f"S{s}T{i}" for i in range(5)]
        all_tickers += tks
        momentum_row += [0.01 * (i + 1) for i in range(5)]
        low_vol_row += [0.30 - 0.02 * i for i in range(5)]
        sectors.update(_sector_map(tks, f"Sector{s}"))
    # A third, excluded sector: only 1 valid name (< threshold).
    excluded_tickers = ["Ex0", "Ex1"]
    all_tickers += excluded_tickers
    momentum_row += [np.nan, 0.5]
    low_vol_row += [0.30, 0.05]
    sectors.update(_sector_map(excluded_tickers, "ExcludedSector"))

    momentum = pd.DataFrame([momentum_row], index=idx, columns=all_tickers)
    low_vol = pd.DataFrame([low_vol_row], index=idx, columns=all_tickers)
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    result = build_sector_neutral_signal(momentum, low_vol, sectors, config)

    # ExcludedSector has only 1 valid name (< threshold 2) -> excluded.
    assert (result[excluded_tickers].iloc[0] == 0.0).all()
    # The remaining 2 active sectors still normalize to a full 1.0/-1.0
    # total, not diluted by the excluded sector's absence.
    long_sum = result.where(result > 0, 0.0).sum(axis=1).iloc[0]
    short_sum = result.where(result < 0, 0.0).sum(axis=1).iloc[0]
    assert long_sum == 1.0
    assert short_sum == -1.0


# ---------------------------------------------------------------------------
# Existence-justifying test: sector-neutral guarantees representation that
# global cross-sectional ranking denies to a systematically weaker sector.
# ---------------------------------------------------------------------------


def test_build_sector_neutral_signal_gives_sector_a_long_exposure_that_global_mode_denies() -> None:
    # SectorA's raw momentum AND low_vol are uniformly dominated by
    # SectorB's (SectorB is always more momentum-positive AND less
    # volatile), so in a GLOBAL cross-sectional ranking, no SectorA name
    # ever reaches the top quintile. Sector-neutral construction ranks
    # within each sector independently, so SectorA's own best name (A4)
    # still gets long exposure.
    idx = pd.bdate_range("2020-01-01", periods=1)
    a_tickers = [f"A{i}" for i in range(5)]
    b_tickers = [f"B{i}" for i in range(5)]
    tickers = a_tickers + b_tickers

    momentum = pd.DataFrame(
        [[0.01 * (i + 1) for i in range(5)] + [0.10 + 0.01 * j for j in range(5)]], index=idx, columns=tickers
    )
    low_vol = pd.DataFrame(
        [[0.50 - 0.02 * i for i in range(5)] + [0.10 - 0.02 * j for j in range(5)]], index=idx, columns=tickers
    )
    sector_map = {**_sector_map(a_tickers, "SectorA"), **_sector_map(b_tickers, "SectorB")}
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20, min_names_per_sector=2)

    global_signal = build_long_short_signal(momentum, low_vol, config)
    neutral_signal = build_sector_neutral_signal(momentum, low_vol, sector_map, config)

    assert global_signal["A4"].iloc[0] == 0.0
    assert neutral_signal["A4"].iloc[0] > 0.0
