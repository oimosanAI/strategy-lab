"""Re-verify the multiple-testing conclusions after the H1 family-size fix.

H1: ``select_pairs`` set the Bonferroni family size to the number of
candidate PAIRS, but ``engle_granger_test`` runs ``coint()`` in BOTH
directions per pair and reports the lower of the two p-values. That
minimum-of-two is itself a selection, so the family size is 2 tests per
pair. The fix doubles ``n_tests``; this script measures what that does to
every published pair-selection conclusion, on the real cached data.

METHOD -- one scan, both answers. Rather than running ``select_pairs``
twice (once per family-size rule) and once more per sensitivity-grid
point, this scans ONCE at the loosest correlation pre-filter with the
correction disabled (``multiple_testing_correction="none"``,
``cointegration_alpha=1.0``). That makes the Engle-Granger gate a
pass-through, so the returned candidates are exactly the pairs clearing
the Johansen and half-life gates, each carrying its RAW p-value. Every
published configuration is then reproduced offline by applying the
Bonferroni arithmetic to those raw p-values:

- family size N (the old, under-counted rule) vs 2N (the fixed one);
- any correlation pre-filter >= the scan's, since the set of pairs tested
  at a stricter threshold is a subset of those tested at a looser one.

This is exact, not an approximation: ``adjusted_p = raw_p * n_tests`` and
the Johansen/half-life gates do not depend on the family size at all.

Run with --count-only first to size the job before committing to a scan.
"""

from __future__ import annotations

import argparse
import json
import time
import dataclasses
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import pandas as pd

from core.backtest.sample_split import SamplePeriod
from core.data.cache import ParquetCache
from core.data.loader import DataLoader, YFinanceProvider
from core.data.universe import get_sp500_constituents, get_sp500_tickers
from strategies.pairs_trading.cointegration import PairSelectionConfig, select_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors scripts/generate_reports.py exactly -- these must not drift, or
# the re-verification would not be measuring the published runs.
START = date(2023, 1, 1)
END = date(2026, 7, 25)
IN_SAMPLE = SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31"))
EXCLUDED_SPARSE_HISTORY = ["FDXF", "HONA", "SNDK", "Q", "GEV", "SOLV", "VLTO", "KVUE", "FISV"]
NARROW_SECTORS = {"Utilities", "Communication Services"}

PREFILTER_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
SCAN_PREFILTER = min(PREFILTER_GRID)
PUBLISHED_ALPHA = 0.05


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class Pool:
    name: str
    prices: pd.DataFrame
    sector_map: dict[str, str]


def _load_pools() -> tuple[Pool, Pool]:
    loader = DataLoader(provider=YFinanceProvider(), cache=ParquetCache(REPO_ROOT / "data" / "cache"))
    all_tickers = get_sp500_tickers()
    wf_tickers = [t for t in all_tickers if t not in EXCLUDED_SPARSE_HISTORY]
    constituents = get_sp500_constituents()
    sector_map_full = {str(r.ticker): str(r.gics_sector) for r in constituents.itertuples()}
    sector_map_wf = {t: s for t, s in sector_map_full.items() if t not in EXCLUDED_SPARSE_HISTORY}
    narrow_tickers = [t for t in wf_tickers if sector_map_wf.get(t) in NARROW_SECTORS]

    close_all = loader.get_price_panel(all_tickers, START, END, field="close")
    close_wf = close_all[wf_tickers]

    _log(f"Full pool: {len(wf_tickers)} tickers. Narrow pool: {len(narrow_tickers)} tickers.")
    return (
        Pool("narrow (2-sector, Step A)", close_wf[narrow_tickers], {t: sector_map_wf[t] for t in narrow_tickers}),
        Pool("full universe (Step C)", close_wf, sector_map_wf),
    )


def _pair_correlations(pool: Pool) -> dict[frozenset[str], float]:
    """|corr| on IN-SAMPLE data for every within-sector pair, so the pair
    count at any pre-filter threshold can be derived without re-scanning."""
    is_prices = pool.prices.loc[IN_SAMPLE.start : IN_SAMPLE.end]
    groups: dict[str, list[str]] = {}
    for ticker in is_prices.columns:
        sector = pool.sector_map.get(ticker)
        if sector is not None:
            groups.setdefault(sector, []).append(ticker)

    correlations: dict[frozenset[str], float] = {}
    for tickers in groups.values():
        for a, b in combinations(sorted(tickers), 2):
            correlations[frozenset((a, b))] = abs(is_prices[a].corr(is_prices[b]))
    return correlations


def _n_pairs_at(correlations: dict[frozenset[str], float], prefilter: float) -> int:
    return sum(1 for c in correlations.values() if c >= prefilter)


@dataclass(frozen=True)
class ScannedPair:
    """One pair that cleared the Johansen and half-life gates, with its RAW
    Engle-Granger p-value -- the input every family-size/pre-filter
    combination is then re-derived from offline."""

    ticker_a: str
    ticker_b: str
    sector: str
    raw_p: float
    half_life: float
    johansen_significant: bool


def _scan(pool: Pool) -> list[ScannedPair]:
    """One pass with the Engle-Granger gate disabled -- see module docstring."""
    config = PairSelectionConfig(
        correlation_prefilter=SCAN_PREFILTER,
        cointegration_alpha=1.0,
        multiple_testing_correction="none",
    )
    started = time.time()
    candidates = select_pairs(pool.prices, IN_SAMPLE, pool.sector_map, config)
    _log(f"  scan finished in {time.time() - started:.0f}s: {len(candidates)} pairs cleared Johansen + half-life")
    return [
        ScannedPair(
            ticker_a=c.ticker_a,
            ticker_b=c.ticker_b,
            sector=c.sector,
            raw_p=c.engle_granger.p_value,
            half_life=c.half_life,
            johansen_significant=c.johansen.is_cointegrated,
        )
        for c in candidates
    ]


def _survivors(
    scanned: list[ScannedPair],
    correlations: dict[frozenset[str], float],
    prefilter: float,
    family_multiplier: int,
    alpha: float = PUBLISHED_ALPHA,
) -> list[tuple[str, str, float]]:
    """Pairs whose Bonferroni-adjusted p clears ``alpha``, for one
    (pre-filter, family-size rule) combination."""
    n_tests = family_multiplier * _n_pairs_at(correlations, prefilter)
    out: list[tuple[str, str, float]] = []
    for row in scanned:
        if correlations.get(frozenset((row.ticker_a, row.ticker_b)), 0.0) < prefilter:
            continue
        adjusted = min(1.0, row.raw_p * n_tests)
        if adjusted < alpha:
            out.append((row.ticker_a, row.ticker_b, adjusted))
    return sorted(out, key=lambda r: r[2])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-only", action="store_true", help="Report pair counts and exit (no scan).")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "h1_reverification.json")
    args = parser.parse_args()

    narrow, full = _load_pools()
    results: dict[str, object] = {}

    for pool in (narrow, full):
        _log(f"\n=== {pool.name} ===")
        correlations = _pair_correlations(pool)
        _log(f"  within-sector pairs (no pre-filter): {len(correlations)}")
        for prefilter in PREFILTER_GRID:
            n = _n_pairs_at(correlations, prefilter)
            _log(f"  prefilter>={prefilter}: {n} pairs -> n_tests: {n} (old) / {2 * n} (fixed)")

        if args.count_only:
            continue

        _log(f"  scanning at prefilter>={SCAN_PREFILTER} (correction disabled)...")
        scanned = _scan(pool)

        grid: dict[str, object] = {}
        pool_result: dict[str, object] = {
            "scanned_pairs": [dataclasses.asdict(p) for p in scanned],
            "grid": grid,
        }
        for prefilter in PREFILTER_GRID:
            n = _n_pairs_at(correlations, prefilter)
            old = _survivors(scanned, correlations, prefilter, family_multiplier=1)
            new = _survivors(scanned, correlations, prefilter, family_multiplier=2)
            _log(
                f"  prefilter>={prefilter}: n_pairs={n} | survivors old(n_tests={n})={len(old)} "
                f"new(n_tests={2 * n})={len(new)}"
            )
            for label, rows in (("old", old), ("new", new)):
                for a, b, adj in rows[:5]:
                    _log(f"      [{label}] {a}-{b}: adjusted_p={adj:.6g}")
            grid[str(prefilter)] = {
                "n_pairs": n,
                "n_tests_old": n,
                "n_tests_new": 2 * n,
                "survivors_old": old,
                "survivors_new": new,
            }
        results[pool.name] = pool_result

    if not args.count_only:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        _log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
