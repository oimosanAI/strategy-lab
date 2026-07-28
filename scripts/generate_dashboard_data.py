"""Generate the dashboard's frozen data (Tier 1 precomputed results +
Tier 2 price-panel snapshots) and write them under dashboard/data/.

The Streamlit app (app.py) never calls yfinance or DataLoader itself --
by design, so a public deployment (Streamlit Community Cloud) never
depends on live network access or risks yfinance rate-limiting a
stranger's page load. This script is the ONLY place that touches real
market data for the dashboard; its output is committed to the repo.

Tier 1 (precomputed, non-reactive in the app): pairs_trading's
full-universe walk-forward and the two Bonferroni scan points are
transcribed from the real, already-verified results in
strategies/pairs_trading/README.md -- re-deriving them here would mean
re-running select_pairs across ~5,551 pair-tests x 4 windows, the same
cost concern already documented in scripts/generate_figures.py.
factor_momentum's and vol_arbitrage's walk-forward ARE recomputed fresh
here (cheap, single continuous run_backtest + window slicing).

Tier 2 snapshots are small frozen (close, open) panels: 2 tickers for
pairs_trading (AEP-FE), 3 for vol_arbitrage (VIX/SPY/SVXY), and the full
~494-ticker universe for factor_momentum (needed for the percentile
slider to bucket rank meaningfully across the whole universe -- this is
the one genuinely non-trivial snapshot, a few MB, not "tiny").

Run: poetry run python scripts/generate_dashboard_data.py
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, generate_anchored_windows
from core.data.cache import ParquetCache
from core.data.loader import DataLoader, YFinanceProvider
from core.data.universe import get_sp500_constituents, get_sp500_tickers
from core.evaluation.walk_forward import evaluate_walk_forward_windows
from strategies.factor_momentum.ranking import FactorMomentumConfig
from strategies.factor_momentum.strategy import FactorMomentumStrategy
from strategies.pairs_trading.cointegration import select_pairs
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard" / "data"

START = date(2023, 1, 1)
END = date(2026, 7, 25)

SPLIT = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), pd.Timestamp(END)),
)
WALK_FORWARD_BOUNDARIES = [pd.Timestamp(b) for b in ["2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31"]]

EXCLUDED_SPARSE_HISTORY = ["FDXF", "HONA", "SNDK", "Q", "GEV", "SOLV", "VLTO", "KVUE", "FISV"]
NARROW_SECTORS = {"Utilities", "Communication Services"}

# Transcribed verbatim from strategies/pairs_trading/README.md -- see
# this module's docstring for why (prohibitively expensive to re-derive
# just for the dashboard).
PAIRS_TRADING_FULL_UNIVERSE_WALK_FORWARD = [
    {"label": "Window 1 (MTB-SCHW)", "oos_sharpe": -0.73, "permutation_p": 0.697},
    {"label": "Window 2", "oos_sharpe": None, "permutation_p": None},
    {"label": "Window 3", "oos_sharpe": None, "permutation_p": None},
    {"label": "Window 4 (AVGO-FLEX)", "oos_sharpe": -0.62, "permutation_p": 0.613},
]
RAW_P = 7.666e-05
BONFERRONI_SCANS = [
    {"n_tests": 308, "survivors_before": 18, "survivors_after": 1, "label": "2-sector"},
    {"n_tests": 5551, "survivors_before": 237, "survivors_after": 0, "label": "full-universe"},
]


def _windows_to_json(results: list) -> list[dict]:  # type: ignore[type-arg]
    out = []
    for i, r in enumerate(results, start=1):
        if r.oos_metrics is None:
            out.append({"label": f"Window {i}", "oos_sharpe": None, "permutation_p": None})
        else:
            out.append(
                {
                    "label": f"Window {i}",
                    "oos_sharpe": r.oos_metrics["sharpe_ratio"],
                    "permutation_p": r.significance.permutation.p_value,
                }
            )
    return out


def main() -> None:
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(provider=YFinanceProvider(), cache=ParquetCache(REPO_ROOT / "data" / "cache"))
    sizer = VolTargetSizer(PositionSizingConfig())
    backtest_config = BacktestConfig()

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

    # --- factor_momentum: full snapshot + walk-forward (real, recomputed) ---
    print("Running factor_momentum backtest...")
    fm_strategy = FactorMomentumStrategy(config=FactorMomentumConfig())
    fm_result = run_backtest(close_full, open_full, fm_strategy, sizer, backtest_config)
    fm_walk_forward = evaluate_walk_forward_windows(fm_result.returns, windows_full, label="factor-momentum")
    close_full.to_parquet(DASHBOARD_DATA_DIR / "snapshot_factor_momentum_close.parquet")
    open_full.to_parquet(DASHBOARD_DATA_DIR / "snapshot_factor_momentum_open.parquet")

    # --- vol_arbitrage: snapshot + walk-forward (real, recomputed) ---
    print("Running vol_arbitrage backtest...")
    va_strategy = VolArbitrageStrategy(config=VolArbitrageConfig())
    va_result = run_backtest(close_vrp, open_vrp, va_strategy, sizer, backtest_config)
    va_walk_forward = evaluate_walk_forward_windows(va_result.returns, windows_vrp, label="vol-arbitrage")
    close_vrp.to_parquet(DASHBOARD_DATA_DIR / "snapshot_vol_arbitrage_close.parquet")
    open_vrp.to_parquet(DASHBOARD_DATA_DIR / "snapshot_vol_arbitrage_open.parquet")

    # --- pairs_trading: narrow-pool AEP-FE selection + snapshot ---
    print("Selecting pairs_trading candidate (narrow 2-sector pool)...")
    candidates = select_pairs(close_narrow, SPLIT.in_sample, sector_map_narrow)
    candidate = candidates[0]
    print(f"  candidate: {candidate.ticker_a}-{candidate.ticker_b}")
    pair_close = close_narrow[[candidate.ticker_a, candidate.ticker_b]]
    pair_open = open_narrow[[candidate.ticker_a, candidate.ticker_b]]
    pair_close.to_parquet(DASHBOARD_DATA_DIR / "snapshot_pairs_trading_close.parquet")
    pair_open.to_parquet(DASHBOARD_DATA_DIR / "snapshot_pairs_trading_open.parquet")
    with open(DASHBOARD_DATA_DIR / "snapshot_pairs_trading_candidate.pkl", "wb") as f:
        pickle.dump(candidate, f)

    # --- Tier 1 JSON ---
    tier1 = {
        "as_of": close_full.index[-1].date().isoformat(),
        "pairs_trading_candidate": [candidate.ticker_a, candidate.ticker_b],
        "pairs_trading_walk_forward": PAIRS_TRADING_FULL_UNIVERSE_WALK_FORWARD,
        "factor_momentum_walk_forward": _windows_to_json(fm_walk_forward),
        "vol_arbitrage_walk_forward": _windows_to_json(va_walk_forward),
        "raw_p": RAW_P,
        "bonferroni_scans": BONFERRONI_SCANS,
    }
    with open(DASHBOARD_DATA_DIR / "tier1_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(tier1, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Dashboard data written to {DASHBOARD_DATA_DIR}")


if __name__ == "__main__":
    main()
