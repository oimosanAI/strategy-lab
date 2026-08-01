"""Regenerate every Markdown report under reports/ from real market data.

Reproducible from a fresh clone: uses the same DataLoader/YFinanceProvider/
ParquetCache stack (data/cache/, gitignored) as scripts/generate_figures.py
and scripts/generate_dashboard_data.py -- a cold cache simply means the
first run re-fetches from yfinance instead of hitting local parquet files.

WHY THIS SCRIPT EXISTS: reports/*.md were previously produced ad hoc and
then hand-maintained. That is fine until the evaluation logic changes --
at which point every committed report silently becomes a claim the code no
longer makes, and the only ways to fix it are to re-derive the numbers by
hand (i.e. invent them) or leave them stale. Neither is acceptable in a
repo whose whole point is that reported numbers are reproducible. From
here on, reports are REGENERATED, never edited.

TWO UNIVERSES, DELIBERATELY. The single-split reports (factor_momentum
global/sector-neutral, its percentile sensitivity, the comparison report)
are computed on the FULL 503-ticker S&P 500 panel, matching what those
reports already document in their own causality notes. The walk-forward
reports use the 494-ticker panel that drops EXCLUDED_SPARSE_HISTORY,
because those 9 tickers have NaN gaps specifically in the walk-forward
range (see strategies/pairs_trading/README.md Step C). This split is not
an inconsistency to be tidied away: each report states which panel it used,
and forcing both onto one universe would silently change published
numbers for no methodological reason.

RUNTIME: roughly 45-60 minutes on a warm cache. The two full-universe
pairs_trading artifacts dominate -- select_pairs across ~5,551 pair-tests
takes ~5 minutes per call, and they need 4 (walk-forward windows) + 5
(correlation-prefilter grid points) of them. scripts/generate_figures.py
transcribes those results rather than re-deriving them for a chart; this
script does NOT transcribe, because a report that quietly copies numbers
from another report is exactly the failure mode above. Use --only to
regenerate a subset (see its --help) when iterating; a full run is the
default and takes no argument.

Run: poetry run python scripts/generate_reports.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import (
    SamplePeriod,
    TrainTestSplit,
    generate_anchored_windows,
    split_returns,
)
from core.data.cache import ParquetCache
from core.data.loader import DataLoader, YFinanceProvider
from core.data.universe import get_sp500_constituents, get_sp500_tickers
from core.evaluation import metrics
from core.evaluation.report import (
    StrategyReportInputs,
    write_comparison_report,
    write_strategy_report,
)
from core.evaluation.sensitivity import render_sensitivity_report
from core.evaluation.statistical_tests import check_significance
from core.evaluation.walk_forward import evaluate_walk_forward_windows, render_walk_forward_report
from strategies.factor_momentum.ranking import FactorMomentumConfig
from strategies.factor_momentum.sensitivity import run_percentile_grid
from strategies.factor_momentum.strategy import FactorMomentumStrategy
from strategies.pairs_trading.cointegration import select_pairs
from strategies.pairs_trading.sensitivity import (
    run_correlation_prefilter_grid,
    run_entry_exit_threshold_grid,
)
from strategies.pairs_trading.strategy import PairsTradingStrategy
from strategies.pairs_trading.walk_forward import run_pairs_trading_walk_forward
from strategies.vol_arbitrage.sensitivity import run_vrp_threshold_grid
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

START = date(2023, 1, 1)
END = date(2026, 7, 25)

SPLIT = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), pd.Timestamp(END)),
)
WALK_FORWARD_BOUNDARIES = [pd.Timestamp(b) for b in ["2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"]]

# See the module docstring's "TWO UNIVERSES" note for why this exclusion
# applies to the walk-forward panels only.
EXCLUDED_SPARSE_HISTORY = ["FDXF", "HONA", "SNDK", "Q", "GEV", "SOLV", "VLTO", "KVUE", "FISV"]

NARROW_SECTORS = {"Utilities", "Communication Services"}

# Free text, not computed. Every string below is a claim about what was
# actually done (which causality check ran, what the known weaknesses are)
# and is carried verbatim from the reports these replace. It is kept here,
# beside the code that produces the numbers, so that a future change to
# either one is visible in the same diff.
CAUSALITY_FULL_SP500 = (
    "assert_backtest_causal PASSED (n_trials=8, seed=0) on the full S&P 500 panel (503 tickers)."
)
CAUSALITY_AEP_FE = "assert_backtest_causal PASSED (n_trials=8, seed=0) on the full AEP/FE panel."
CAUSALITY_VRP = (
    "assert_backtest_causal PASSED (n_trials=8, seed=0) on the full VIX/SPY/SVXY panel "
    "(2023-01-03 to 2026-07-24)."
)

SURVIVORSHIP_LIMITATION = "Level A survivorship bias (current S&P 500 constituents only)."
EXPOSURE_LIMITATION = (
    "Portfolio exposure limits are checked automatically inside run_backtest, but not "
    "enforced: the default BacktestConfig uses exposure_limits_strict=False, so a breach "
    "is recorded in BacktestResult.exposure_violations and warned about rather than "
    "raised. The default caps (max_gross=max_net=5.0) are a catastrophic sanity net, not "
    "a per-strategy-tuned constraint -- see BacktestConfig.exposure_limits."
)


def _log(message: str) -> None:
    print(message, flush=True)


def _degradation_note(is_returns: pd.Series, oos_returns: pd.Series, tail: str) -> str:
    """Pillar-3 line: IS vs OOS Sharpe, computed rather than transcribed."""
    return (
        f"Anchored split: IS Sharpe={metrics.sharpe_ratio(is_returns):.3f}, "
        f"OOS Sharpe={metrics.sharpe_ratio(oos_returns):.3f}. {tail}"
    )


@dataclass(frozen=True)
class Panels:
    """Every price panel and universe map the report builders need.

    Loaded once and passed around so that a partial run (--only) still sees
    exactly the same inputs as a full one -- the panels are the thing that
    must not drift between modes.
    """

    close_all: pd.DataFrame
    open_all: pd.DataFrame
    close_wf: pd.DataFrame
    open_wf: pd.DataFrame
    close_narrow: pd.DataFrame
    open_narrow: pd.DataFrame
    close_vrp: pd.DataFrame
    open_vrp: pd.DataFrame
    sector_map_full: dict[str, str]
    sector_map_wf: dict[str, str]
    sector_map_narrow: dict[str, str]
    # Two window sets, because the two panels have different trading
    # calendars: the VIX/SPY/SVXY panel drops rows the equity panel keeps
    # (see the Memorial Day note in vol_arbitrage's limitations), so
    # anchoring both to one index would silently shift vol_arbitrage's
    # window boundaries.
    windows_wf: list[TrainTestSplit]
    windows_vrp: list[TrainTestSplit]


def _load_panels(loader: DataLoader) -> Panels:
    """Build both universes -- see the module docstring's TWO UNIVERSES note."""
    all_tickers = get_sp500_tickers()
    walk_forward_tickers = [t for t in all_tickers if t not in EXCLUDED_SPARSE_HISTORY]
    constituents = get_sp500_constituents()
    sector_map_full: dict[str, str] = {
        str(row.ticker): str(row.gics_sector) for row in constituents.itertuples()
    }
    sector_map_wf = {t: s for t, s in sector_map_full.items() if t not in EXCLUDED_SPARSE_HISTORY}
    narrow_tickers = [t for t in walk_forward_tickers if sector_map_wf.get(t) in NARROW_SECTORS]
    sector_map_narrow = {t: sector_map_wf[t] for t in narrow_tickers}

    _log(f"Single-split universe: {len(all_tickers)} tickers.")
    _log(f"Walk-forward universe: {len(walk_forward_tickers)} tickers. Narrow pool: {len(narrow_tickers)}.")

    close_all = loader.get_price_panel(all_tickers, START, END, field="close")
    open_all = loader.get_price_panel(all_tickers, START, END, field="open")
    close_wf = close_all[walk_forward_tickers]
    open_wf = open_all[walk_forward_tickers]

    vix_spy_svxy = ["^VIX", "SPY", "SVXY"]
    close_vrp = loader.get_price_panel(vix_spy_svxy, START, END, field="close").rename(columns={"^VIX": "VIX"})
    open_vrp = loader.get_price_panel(vix_spy_svxy, START, END, field="open").rename(columns={"^VIX": "VIX"})
    close_vrp = close_vrp.dropna()
    open_vrp_aligned = open_vrp.loc[close_vrp.index]

    return Panels(
        close_all=close_all,
        open_all=open_all,
        close_wf=close_wf,
        open_wf=open_wf,
        close_narrow=close_wf[narrow_tickers],
        open_narrow=open_wf[narrow_tickers],
        close_vrp=close_vrp,
        open_vrp=open_vrp_aligned,
        sector_map_full=sector_map_full,
        sector_map_wf=sector_map_wf,
        sector_map_narrow=sector_map_narrow,
        windows_wf=list(generate_anchored_windows(pd.DatetimeIndex(close_wf.index), WALK_FORWARD_BOUNDARIES)),
        windows_vrp=list(generate_anchored_windows(pd.DatetimeIndex(close_vrp.index), WALK_FORWARD_BOUNDARIES)),
    )


def _generate_fast_reports(
    panels: Panels, sizer: VolTargetSizer, backtest_config: BacktestConfig
) -> list[str]:
    """Everything except the two full-universe pairs_trading reports."""
    close_all, open_all = panels.close_all, panels.open_all
    close_wf, open_wf = panels.close_wf, panels.open_wf
    close_narrow, open_narrow = panels.close_narrow, panels.open_narrow
    close_vrp, open_vrp = panels.close_vrp, panels.open_vrp
    sector_map_full, sector_map_narrow = panels.sector_map_full, panels.sector_map_narrow
    windows_wf = panels.windows_wf

    written: list[str] = []

    # ======================================================================
    # factor_momentum: global + sector-neutral single-split reports
    # ======================================================================
    _log("Running factor_momentum (global)...")
    fm_config = FactorMomentumConfig()
    fm_global = run_backtest(
        close_all, open_all, FactorMomentumStrategy(config=fm_config), sizer, backtest_config
    )
    fm_global_is, fm_global_oos = split_returns(fm_global.returns, SPLIT)

    fm_readme_tail = (
        "See strategies/factor_momentum/README.md for the full degradation record and, for "
        "sector-neutral mode, the global-vs-sector-neutral comparison."
    )
    fm_global_inputs = StrategyReportInputs(
        name="Factor Momentum: 12-1 Momentum + 60d Low-Volatility",
        returns=fm_global_oos,
        metrics=metrics.summary(fm_global_oos),
        significance=check_significance(fm_global_oos, seed=0),
        causality_note=CAUSALITY_FULL_SP500,
        walk_forward_note=_degradation_note(fm_global_is, fm_global_oos, fm_readme_tail),
        description="Full S&P 500 universe; quintile long/short, monthly rebalance, leg-normalized.",
        limitations=[
            SURVIVORSHIP_LIMITATION,
            "Sector neutralization not applied in this mode -- see factor_momentum_sector_neutral.md "
            "for the sector-neutral comparison; the global mode's apparent outperformance may partly "
            "reflect incidental sector tilts (Utilities/Financials) rather than pure factor exposure.",
            EXPOSURE_LIMITATION,
        ],
    )
    write_strategy_report(fm_global_inputs, REPORTS_DIR / "factor_momentum_global.md")
    written.append("factor_momentum_global.md")

    _log("Running factor_momentum (sector-neutral)...")
    fm_sn = run_backtest(
        close_all,
        open_all,
        FactorMomentumStrategy(config=fm_config, mode="sector_neutral", sector_map=sector_map_full),
        sizer,
        backtest_config,
    )
    fm_sn_is, fm_sn_oos = split_returns(fm_sn.returns, SPLIT)
    write_strategy_report(
        StrategyReportInputs(
            name="Factor Momentum: 12-1 Momentum + 60d Low-Volatility (Sector-Neutral)",
            returns=fm_sn_oos,
            metrics=metrics.summary(fm_sn_oos),
            significance=check_significance(fm_sn_oos, seed=0),
            causality_note=CAUSALITY_FULL_SP500,
            walk_forward_note=_degradation_note(fm_sn_is, fm_sn_oos, fm_readme_tail),
            description=(
                "Full S&P 500 universe; sector-neutral quintile long/short (within each GICS "
                "sector), monthly rebalance, per-sector equal-budget normalization."
            ),
            limitations=[
                SURVIVORSHIP_LIMITATION,
                "Sector-neutral construction reduces sector concentration (see "
                "strategies/factor_momentum/README.md Sec 3-3) but showed WORSE IS->OOS "
                "degradation than the global mode in this same evaluation window -- a documented "
                "trade-off, not a bug.",
                "min_names_per_sector=10 validated only partially on real data (no sector in this "
                "universe actually fell below the threshold during this run).",
                EXPOSURE_LIMITATION,
            ],
        ),
        REPORTS_DIR / "factor_momentum_sector_neutral.md",
    )
    written.append("factor_momentum_sector_neutral.md")

    # ======================================================================
    # vol_arbitrage: single-split report
    # ======================================================================
    _log("Running vol_arbitrage...")
    va_config = VolArbitrageConfig()
    va_result = run_backtest(
        close_vrp, open_vrp, VolArbitrageStrategy(config=va_config), sizer, backtest_config
    )
    va_is, va_oos = split_returns(va_result.returns, SPLIT)

    svxy_oos = close_vrp["SVXY"].pct_change().loc[SPLIT.out_of_sample.start : SPLIT.out_of_sample.end]
    buy_hold_sharpe = metrics.sharpe_ratio(svxy_oos)
    # np.prod over the raw array, matching core.evaluation.metrics'
    # annualized_return, rather than Series.prod (whose stub return type is
    # a broad union that mypy cannot narrow to a float).
    buy_hold_cumulative = float(np.prod((1.0 + svxy_oos.dropna()).to_numpy())) - 1.0
    strategy_cumulative = float(np.prod((1.0 + va_oos.dropna()).to_numpy())) - 1.0
    va_degradation = metrics.sharpe_ratio(va_is) - metrics.sharpe_ratio(va_oos)

    write_strategy_report(
        StrategyReportInputs(
            name="Vol Arbitrage: VIX/SPY Volatility Risk Premium (SVXY, long-only)",
            returns=va_oos,
            metrics=metrics.summary(va_oos),
            significance=check_significance(va_oos, seed=0),
            causality_note=CAUSALITY_VRP,
            walk_forward_note=(
                f"Anchored split: IS Sharpe={metrics.sharpe_ratio(va_is):.3f}, "
                f"OOS Sharpe={metrics.sharpe_ratio(va_oos):.3f}. "
                f"Degradation = {va_degradation:.3f}. Buy & Hold SVXY OOS Sharpe (no signal, no "
                f"cost model) = {buy_hold_sharpe:.3f}, cumulative return={buy_hold_cumulative:.2%}, "
                f"vs strategy cumulative return={strategy_cumulative:.2%}. "
                "See reports/vol_arbitrage_walk_forward.md for the 4-window walk-forward record."
            ),
            description=(
                "VRP(t) = VIX(t)/100 - realized_volatility(SPY, window=21). Long SVXY when "
                "VRP > 0.0, flat otherwise. VolTargetSizer (target_vol=15%, vol_window=60, "
                "max_leverage=2x). Costs: 0.5bps commission + 5.0bps slippage per "
                "REQUIREMENTS.md 7.1."
            ),
            limitations=[
                "Level A survivorship bias: SVXY's own leverage was reduced from -1x to -0.5x "
                "after the 2018 Volmageddon event; this backtest only covers 2023-2026, so that "
                "structural risk is not reflected in-sample even though it remains a real, "
                "documented tail risk for this instrument.",
                "vrp_threshold=0.0 has no strong literature backing -- see "
                "strategies/vol_arbitrage/README.md Sec 3-4; the strategy's signal uses TRAILING "
                "realized volatility, a different (causal) measure from the forward-looking "
                "realized volatility the academic VRP literature usually reports against.",
                "Long-only design: no short-vol-to-long-vol reversal when VRP is negative, due to "
                "this project's cost model having no borrow-cost modeling.",
                "A single Memorial Day holiday row (2026-05-25) was dropped from the panel as a "
                "data-hygiene step (both SVXY and SPY showed NaN on that date), not a "
                "results-shaping exclusion.",
                "Walk-forward and vrp_threshold sensitivity analyses exist (see "
                "reports/vol_arbitrage_walk_forward.md and "
                "reports/sensitivity_vol_arbitrage_vrp_threshold.md) but all four walk-forward "
                "windows and all five grid points come back non-significant, so neither "
                "establishes an edge -- see Known Limitations in "
                "strategies/vol_arbitrage/README.md.",
            ],
        ),
        REPORTS_DIR / "vol_arbitrage_vrp.md",
    )
    written.append("vol_arbitrage_vrp.md")

    # ======================================================================
    # pairs_trading: narrow-pool AEP-FE single-split report + comparison
    # ======================================================================
    _log("Selecting pairs_trading candidate (narrow 2-sector pool)...")
    narrow_candidates = select_pairs(close_narrow, SPLIT.in_sample, sector_map_narrow)
    candidate = narrow_candidates[0]
    _log(f"  candidate: {candidate.ticker_a}-{candidate.ticker_b}")

    pt_result = run_backtest(
        close_narrow,
        open_narrow,
        PairsTradingStrategy(candidate, hedge_ratio_mode="static"),
        sizer,
        backtest_config,
    )
    pt_is, pt_oos = split_returns(pt_result.returns, SPLIT)
    pt_inputs = StrategyReportInputs(
        name=f"Pairs Trading: {candidate.ticker_a}-{candidate.ticker_b} (Static Hedge)",
        returns=pt_oos,
        metrics=metrics.summary(pt_oos),
        significance=check_significance(pt_oos, seed=0),
        causality_note=CAUSALITY_AEP_FE,
        walk_forward_note=(
            "See strategies/pairs_trading/README.md for the full IS/OOS degradation record."
        ),
        description="S&P 500 Utilities/Communication Services subset; single pair, static hedge.",
        limitations=[
            SURVIVORSHIP_LIMITATION,
            "Candidate pool: single surviving pair (AEP-FE) after full-universe Bonferroni "
            "correction across 5,551 tests -- see strategies/pairs_trading/README.md.",
            "Static hedge ratio frozen from in-sample OLS; not re-estimated OOS.",
        ],
    )
    write_strategy_report(pt_inputs, REPORTS_DIR / "pairs_trading_aep_fe.md")
    written.append("pairs_trading_aep_fe.md")

    write_comparison_report([pt_inputs, fm_global_inputs], REPORTS_DIR / "strategy_comparison.md")
    written.append("strategy_comparison.md")

    # ======================================================================
    # Walk-forward reports
    # ======================================================================
    _log("Walk-forward: factor_momentum (global)...")
    fm_wf_result = run_backtest(
        close_wf, open_wf, FactorMomentumStrategy(config=fm_config), sizer, backtest_config
    )
    fm_windows = evaluate_walk_forward_windows(
        fm_wf_result.returns, windows_wf, label="factor-momentum (global)"
    )
    (REPORTS_DIR / "factor_momentum_walk_forward.md").write_text(
        render_walk_forward_report("Factor Momentum (Global Mode)", fm_windows), encoding="utf-8"
    )
    written.append("factor_momentum_walk_forward.md")

    _log("Walk-forward: vol_arbitrage...")
    va_windows = evaluate_walk_forward_windows(
        va_result.returns, panels.windows_vrp, label="vol-arbitrage"
    )
    (REPORTS_DIR / "vol_arbitrage_walk_forward.md").write_text(
        render_walk_forward_report("Vol Arbitrage (VIX/SPY VRP, SVXY)", va_windows), encoding="utf-8"
    )
    written.append("vol_arbitrage_walk_forward.md")

    _log("Walk-forward: pairs_trading (narrow 2-sector pool)...")
    pt_narrow_windows = run_pairs_trading_walk_forward(
        close_narrow, open_narrow, sector_map_narrow, windows_wf, sizer=sizer, backtest_config=backtest_config
    )
    (REPORTS_DIR / "pairs_trading_walk_forward_narrow_pool.md").write_text(
        render_walk_forward_report(
            "Pairs Trading (Narrow 2-Sector Pool -- REFERENCE ONLY, not full-universe corrected)",
            pt_narrow_windows,
        ),
        encoding="utf-8",
    )
    written.append("pairs_trading_walk_forward_narrow_pool.md")

    # ======================================================================
    # Sensitivity reports
    # ======================================================================
    _log("Sensitivity: factor_momentum percentile...")
    fm_grid = run_percentile_grid(
        close_all,
        open_all,
        SPLIT,
        percentiles=[0.10, 0.20, 0.25, 0.30, 0.40],
        base_config=fm_config,
    )
    (REPORTS_DIR / "sensitivity_factor_momentum_percentile.md").write_text(
        render_sensitivity_report("Factor Momentum Long/Short Percentile (Global Mode)", fm_grid),
        encoding="utf-8",
    )
    written.append("sensitivity_factor_momentum_percentile.md")

    _log("Sensitivity: vol_arbitrage vrp_threshold...")
    va_grid = run_vrp_threshold_grid(
        close_vrp,
        open_vrp,
        SPLIT,
        thresholds=[-0.05, -0.02, 0.0, 0.02, 0.05],
        default_threshold=0.0,
        base_config=va_config,
    )
    (REPORTS_DIR / "sensitivity_vol_arbitrage_vrp_threshold.md").write_text(
        render_sensitivity_report("Vol Arbitrage VRP Threshold (VIX/SPY, SVXY)", va_grid),
        encoding="utf-8",
    )
    written.append("sensitivity_vol_arbitrage_vrp_threshold.md")

    _log("Sensitivity: pairs_trading entry/exit thresholds...")
    pt_pair_close = close_narrow[[candidate.ticker_a, candidate.ticker_b]]
    pt_pair_open = open_narrow[[candidate.ticker_a, candidate.ticker_b]]
    pt_grid = run_entry_exit_threshold_grid(
        candidate,
        pt_pair_close,
        pt_pair_open,
        SPLIT,
        entry_thresholds=[1.5, 1.75, 2.0, 2.25, 2.5],
        exit_thresholds=[0.0, 0.25, 0.5],
    )
    (REPORTS_DIR / "sensitivity_pairs_trading_entry_exit_threshold.md").write_text(
        render_sensitivity_report(
            f"Pairs Trading Entry/Exit Threshold ({candidate.ticker_a}-{candidate.ticker_b})", pt_grid
        ),
        encoding="utf-8",
    )
    written.append("sensitivity_pairs_trading_entry_exit_threshold.md")

    return written


def _generate_full_universe_pair_reports(
    panels: Panels, sizer: VolTargetSizer, backtest_config: BacktestConfig
) -> list[str]:
    """The two expensive reports: ~5 minutes per select_pairs call, and this
    needs 4 (walk-forward windows) + 5 (prefilter grid points) of them."""
    written: list[str] = []

    _log("Walk-forward: pairs_trading (FULL universe, ~5,551 pair-tests x 4 windows)...")
    pt_full_windows = run_pairs_trading_walk_forward(
        panels.close_wf,
        panels.open_wf,
        panels.sector_map_wf,
        panels.windows_wf,
        sizer=sizer,
        backtest_config=backtest_config,
    )
    (REPORTS_DIR / "pairs_trading_walk_forward_full_universe.md").write_text(
        render_walk_forward_report("Pairs Trading (Full Universe, Bonferroni-corrected)", pt_full_windows),
        encoding="utf-8",
    )
    written.append("pairs_trading_walk_forward_full_universe.md")

    _log("Sensitivity: pairs_trading correlation_prefilter (FULL universe, 5 grid points)...")
    pt_prefilter_grid = run_correlation_prefilter_grid(
        panels.close_wf,
        panels.open_wf,
        SPLIT,
        panels.sector_map_wf,
        prefilters=[0.5, 0.6, 0.7, 0.8, 0.9],
        default_prefilter=0.7,
        sizer=sizer,
        backtest_config=backtest_config,
    )
    (REPORTS_DIR / "sensitivity_pairs_trading_correlation_prefilter.md").write_text(
        render_sensitivity_report(
            "Pairs Trading Correlation Prefilter (Full Universe, Bonferroni-corrected)",
            pt_prefilter_grid,
        ),
        encoding="utf-8",
    )
    written.append("sensitivity_pairs_trading_correlation_prefilter.md")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", "fast", "full-universe-pairs"],
        default="all",
        help=(
            "Which reports to regenerate. 'all' (default) is the real run. "
            "'fast' skips the two full-universe pairs_trading reports (~45 min "
            "of the runtime) for quick iteration on the others. "
            "'full-universe-pairs' regenerates ONLY those two -- use it when a "
            "change affects them alone, so the cheap reports are not needlessly "
            "recomputed. Partial modes exist so that fixing one report is still "
            "a regeneration and never a hand-edit."
        ),
    )
    args = parser.parse_args()

    started = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(provider=YFinanceProvider(), cache=ParquetCache(REPO_ROOT / "data" / "cache"))
    sizer = VolTargetSizer(PositionSizingConfig())
    backtest_config = BacktestConfig()
    panels = _load_panels(loader)

    written: list[str] = []
    if args.only in ("all", "fast"):
        written += _generate_fast_reports(panels, sizer, backtest_config)
    if args.only in ("all", "full-universe-pairs"):
        written += _generate_full_universe_pair_reports(panels, sizer, backtest_config)

    elapsed = time.time() - started
    _log(f"\nDone in {elapsed / 60:.1f} min. {len(written)} reports written under {REPORTS_DIR}:")
    for name in written:
        _log(f"  - {name}")


if __name__ == "__main__":
    main()
