"""Generate all supplementary visualizations (walk-forward, sensitivity
grids, equity curves, multiple-testing, 3D sensitivity scatter) from real
market data and save them under reports/figures/.

Reproducible from a fresh clone: uses the same DataLoader/YFinanceProvider/
UniverseLoader/ParquetCache stack (data/cache/, gitignored) as every other
part of this project -- a cold cache simply means the first run re-fetches
from yfinance instead of hitting local parquet files. This script is
independent of core/evaluation/report.py's Markdown generation; it does
not modify or depend on that module's logic.

One deliberate exception: the full-universe pairs_trading walk-forward
(Step D in strategies/pairs_trading/README.md) re-runs select_pairs across
~5,551 candidate pairs per window (4 windows) -- prohibitively expensive to
redo just for a chart. Its 4 window results and the two Bonferroni scan
points (308 pairs = 616 tests / 5,551 pairs = 11,102 tests -- the family
size counts 2 tests per pair because engle_granger_test selects the
stronger of both regression directions, see that README's Step F) are
transcribed verbatim from the real, already-verified results in
strategies/pairs_trading/README.md and
reports/pairs_trading_walk_forward_full_universe.md, not re-derived here.
Everything else in this script is computed fresh from real data.

Run: poetry run python scripts/generate_figures.py
"""

from __future__ import annotations

import sys
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
from core.evaluation.statistical_tests import BootstrapResult, PermutationResult, SignificanceCheck
from core.evaluation.visualization import (
    BonferroniScanPoint,
    EquityCurveSeries,
    plot_entry_exit_3d_interactive,
    plot_entry_exit_3d_static,
    plot_equity_curves,
    plot_multiple_testing_bonferroni,
    plot_sensitivity_grid,
    plot_walk_forward_sharpe,
)
from core.evaluation.walk_forward import WalkForwardWindowResult, evaluate_walk_forward_windows
from strategies.factor_momentum.ranking import FactorMomentumConfig
from strategies.factor_momentum.sensitivity import run_percentile_grid
from strategies.factor_momentum.strategy import FactorMomentumStrategy
from strategies.pairs_trading.cointegration import select_pairs
from strategies.pairs_trading.sensitivity import run_entry_exit_threshold_grid
from strategies.pairs_trading.strategy import PairsTradingStrategy
from strategies.vol_arbitrage.sensitivity import run_vrp_threshold_grid
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
INTERACTIVE_DIR = FIGURES_DIR / "interactive"

START = date(2023, 1, 1)
END = date(2026, 7, 25)

SPLIT = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), pd.Timestamp(END)),
)
WALK_FORWARD_BOUNDARIES = [pd.Timestamp(b) for b in ["2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"]]

# 9 tickers excluded for NaN gaps in the IS/walk-forward range -- see
# strategies/pairs_trading/README.md Step C and strategies/factor_momentum
# /README.md's walk-forward section for the real-data derivation of this
# list (data-hygiene exclusion, not a results-shaping one).
EXCLUDED_SPARSE_HISTORY = ["FDXF", "HONA", "SNDK", "Q", "GEV", "SOLV", "VLTO", "KVUE", "FISV"]

NARROW_SECTORS = {"Utilities", "Communication Services"}


def _significance(p_value: float) -> SignificanceCheck:
    """Placeholder significance for the transcribed (not re-derived)
    full-universe pairs_trading walk-forward points -- only p_value is
    used by the plotting functions here."""
    permutation = PermutationResult(
        observed=0.0, p_value=p_value, n_permutations=2000, alternative="greater",
        null_distribution=np.array([0.0]), seed=0,
    )
    bootstrap = BootstrapResult(
        point_estimate=0.0, lower=0.0, upper=0.0, confidence_level=0.95,
        standard_error=0.0, distribution=np.array([0.0]), n_resamples=2000, seed=0,
    )
    return SignificanceCheck(permutation=permutation, bootstrap=bootstrap, agree=True)


def _transcribed_pairs_trading_full_universe_walk_forward() -> list[WalkForwardWindowResult]:
    """Verbatim from reports/pairs_trading_walk_forward_full_universe.md
    -- see this module's docstring for why this one series is transcribed
    rather than recomputed."""
    window = SPLIT  # placeholder window shape; only used for label rendering elsewhere
    populated: list[tuple[float, float]] = [(-0.73, 0.697), (-0.62, 0.613)]
    empty_window_positions = {1, 2}  # windows 2-3 (0-indexed 1,2): no statistically-justified candidate

    results: list[WalkForwardWindowResult] = []
    populated_iter = iter(populated)
    for i in range(4):
        if i in empty_window_positions:
            results.append(
                WalkForwardWindowResult(window=window, label="no candidate", is_metrics=None, oos_metrics=None, significance=None)
            )
        else:
            sharpe, p_value = next(populated_iter)
            results.append(
                WalkForwardWindowResult(
                    window=window, label="pairs-trading (full-universe)",
                    is_metrics=None, oos_metrics={"sharpe_ratio": sharpe},
                    significance=_significance(p_value),
                )
            )
    return results


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(provider=YFinanceProvider(), cache=ParquetCache(REPO_ROOT / "data" / "cache"))
    sizer = VolTargetSizer(PositionSizingConfig())
    backtest_config = BacktestConfig()

    # --- Universe setup ---
    all_tickers = [t for t in get_sp500_tickers() if t not in EXCLUDED_SPARSE_HISTORY]
    constituents = get_sp500_constituents()
    sector_map_full: dict[str, str] = {
        str(row.ticker): str(row.gics_sector)
        for row in constituents.itertuples()
        if str(row.ticker) not in EXCLUDED_SPARSE_HISTORY
    }
    narrow_tickers = [t for t in all_tickers if sector_map_full.get(t) in NARROW_SECTORS]
    sector_map_narrow = {t: sector_map_full[t] for t in narrow_tickers}

    print(f"Full universe: {len(all_tickers)} tickers. Narrow pool: {len(narrow_tickers)} tickers.")

    close_full = loader.get_price_panel(all_tickers, START, END, field="close")
    open_full = loader.get_price_panel(all_tickers, START, END, field="open")
    close_narrow = close_full[narrow_tickers]
    open_narrow = open_full[narrow_tickers]

    vix_spy_svxy = ["^VIX", "SPY", "SVXY"]
    close_vrp = loader.get_price_panel(vix_spy_svxy, START, END, field="close").rename(columns={"^VIX": "VIX"})
    open_vrp = loader.get_price_panel(vix_spy_svxy, START, END, field="open").rename(columns={"^VIX": "VIX"})
    close_vrp = close_vrp.dropna()
    open_vrp = open_vrp.loc[close_vrp.index]

    windows_full = generate_anchored_windows(pd.DatetimeIndex(close_full.index), WALK_FORWARD_BOUNDARIES)
    windows_vrp = generate_anchored_windows(pd.DatetimeIndex(close_vrp.index), WALK_FORWARD_BOUNDARIES)

    # --- factor_momentum: backtest once, reuse for walk-forward + equity curve ---
    print("Running factor_momentum backtest...")
    fm_strategy = FactorMomentumStrategy(config=FactorMomentumConfig())
    fm_result = run_backtest(close_full, open_full, fm_strategy, sizer, backtest_config)
    fm_is, fm_oos = split_returns(fm_result.returns, SPLIT)
    fm_walk_forward = evaluate_walk_forward_windows(fm_result.returns, windows_full, label="factor-momentum")
    fm_percentile_grid = run_percentile_grid(
        close_full, open_full, SPLIT, percentiles=[0.10, 0.20, 0.25, 0.30, 0.40], base_config=FactorMomentumConfig(),
    )

    # --- vol_arbitrage: backtest once, reuse for walk-forward + equity curve ---
    print("Running vol_arbitrage backtest...")
    va_config = VolArbitrageConfig()
    va_strategy = VolArbitrageStrategy(config=va_config)
    va_result = run_backtest(close_vrp, open_vrp, va_strategy, sizer, backtest_config)
    va_is, va_oos = split_returns(va_result.returns, SPLIT)
    va_walk_forward = evaluate_walk_forward_windows(va_result.returns, windows_vrp, label="vol-arbitrage")
    va_threshold_grid = run_vrp_threshold_grid(
        close_vrp, open_vrp, SPLIT, thresholds=[-0.05, -0.02, 0.0, 0.02, 0.05], default_threshold=0.0, base_config=va_config,
    )

    # --- pairs_trading: narrow-pool AEP-FE selection + backtest + entry/exit grid ---
    print("Selecting pairs_trading candidate (narrow 2-sector pool)...")
    candidates = select_pairs(close_narrow, SPLIT.in_sample, sector_map_narrow)
    candidate = candidates[0]
    print(f"  candidate: {candidate.ticker_a}-{candidate.ticker_b}")
    pt_strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")
    pt_result = run_backtest(close_narrow, open_narrow, pt_strategy, sizer, backtest_config)
    pt_is, pt_oos = split_returns(pt_result.returns, SPLIT)
    pt_entry_exit_grid = run_entry_exit_threshold_grid(
        candidate, close_narrow[[candidate.ticker_a, candidate.ticker_b]], open_narrow[[candidate.ticker_a, candidate.ticker_b]],
        SPLIT, entry_thresholds=[1.5, 1.75, 2.0, 2.25, 2.5], exit_thresholds=[0.0, 0.25, 0.5],
    )

    # ==========================================================
    # (a) Walk-forward OOS Sharpe by window (3 strategies)
    # ==========================================================
    print("Plotting (a) walk-forward OOS Sharpe...")
    results_by_strategy = {
        "pairs-trading (full-universe)": _transcribed_pairs_trading_full_universe_walk_forward(),
        "factor-momentum (global)": fm_walk_forward,
        "vol-arbitrage": va_walk_forward,
    }
    fig_a = plot_walk_forward_sharpe(results_by_strategy)
    fig_a.savefig(FIGURES_DIR / "walk_forward_oos_sharpe.png", dpi=150)

    # ==========================================================
    # (b) Sensitivity grids (factor_momentum percentile, vol_arbitrage vrp_threshold)
    # ==========================================================
    print("Plotting (b) sensitivity grids...")
    grids = {
        "Factor Momentum: Long/Short Percentile": fm_percentile_grid,
        "Vol Arbitrage: vrp_threshold": va_threshold_grid,
    }
    fig_b = plot_sensitivity_grid(grids)
    fig_b.savefig(FIGURES_DIR / "sensitivity_grids.png", dpi=150)

    # ==========================================================
    # (c) Equity curve comparison (3 strategies)
    # ==========================================================
    print("Plotting (c) equity curve comparison...")
    curves = [
        EquityCurveSeries(
            returns=pt_oos, label=f"Pairs Trading ({candidate.ticker_a}-{candidate.ticker_b})",
            caveat="reference only, narrow 2-sector pool, not significant at full-universe correction",
        ),
        EquityCurveSeries(returns=fm_oos, label="Factor Momentum (global)"),
        EquityCurveSeries(returns=va_oos, label="Vol Arbitrage (VRP)"),
    ]
    fig_c = plot_equity_curves(curves)
    fig_c.savefig(FIGURES_DIR / "equity_curves_comparison.png", dpi=150)

    # ==========================================================
    # (d) Multiple testing: Bonferroni-adjusted p-value vs n_tests
    # ==========================================================
    print("Plotting (d) multiple testing visualization...")
    raw_p = 7.666e-05  # AEP-FE's raw Engle-Granger p-value, see strategies/pairs_trading/README.md Sec 3-1
    scans = [
        # n_tests counts TESTS, not pairs: engle_granger_test runs coint()
        # in both directions and keeps the stronger, so each pair costs 2
        # (see strategies/pairs_trading/README.md Step F).
        BonferroniScanPoint(n_tests=616, survivors_before=18, survivors_after=1, label="2-sector\n(308 pairs\n=616 tests)"),
        BonferroniScanPoint(n_tests=11102, survivors_before=237, survivors_after=0, label="full-universe\n(5,551 pairs\n=11,102 tests)"),
    ]
    fig_d = plot_multiple_testing_bonferroni(raw_p, scans)
    fig_d.savefig(FIGURES_DIR / "multiple_testing_bonferroni.png", dpi=150)

    # ==========================================================
    # 3D scatter: pairs_trading entry_threshold x exit_threshold (15 raw points)
    # ==========================================================
    print("Plotting 3D entry/exit threshold scatter...")
    fig_3d_static = plot_entry_exit_3d_static(pt_entry_exit_grid)
    fig_3d_static.savefig(FIGURES_DIR / "pairs_trading_entry_exit_3d.png", dpi=150)

    fig_3d_interactive = plot_entry_exit_3d_interactive(pt_entry_exit_grid)
    fig_3d_interactive.write_html(INTERACTIVE_DIR / "pairs_trading_entry_exit_3d.html", include_plotlyjs=True)

    print(f"\nDone. 5 PNGs + 1 interactive HTML saved under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
