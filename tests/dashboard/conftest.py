"""Shared fixture: a small SYNTHETIC dashboard data directory (not the
real committed snapshot) so Layer 2 AppTest wiring tests stay fast and
independent of scripts/generate_dashboard_data.py's real-data output."""

from __future__ import annotations

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategies.pairs_trading.cointegration import EngleGrangerResult, JohansenResult, PairCandidate
from strategies.vol_arbitrage.signal import realized_volatility


def _price_panel(tickers: list[str], n_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2023-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days))) for t in tickers}, index=idx
    )
    open_ = close.shift(1).bfill()
    return close, open_


def _vol_arbitrage_panel(n_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VIX set from SPY's own trailing realized vol (not an independent
    random walk): a slider sweep across [-0.05, 0.05] must actually
    change which days are long, or the wiring test can't tell threshold
    values apart -- the same detection-power lesson as
    tests/core/test_vol_arbitrage_strategy.py's _near_threshold_price_panel."""
    idx = pd.bdate_range("2023-01-03", periods=n_days)
    rng = np.random.default_rng(seed)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n_days))), index=idx)
    svxy = pd.Series(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n_days))), index=idx)

    trailing_rv = realized_volatility(spy, window=21).bfill()
    # Center VIX so VRP = VIX/100 - trailing_rv oscillates with a small
    # sinusoidal wobble around 0, keeping most days within [-0.05, 0.05].
    wobble = 0.03 * np.sin(np.arange(n_days) / 15.0)
    vix = (trailing_rv + wobble) * 100.0

    close = pd.DataFrame({"VIX": vix.to_numpy(), "SPY": spy.to_numpy(), "SVXY": svxy.to_numpy()}, index=idx)
    open_ = close.shift(1).bfill()
    return close, open_


@pytest.fixture()
def dashboard_test_data_dir(tmp_path: Path) -> Path:
    n_days = 620  # spans IS (2023-01-03..2024-12-31) and some OOS (2025-01-02..)

    pt_close, pt_open = _price_panel(["TICKA", "TICKB"], n_days, seed=0)
    pt_close.to_parquet(tmp_path / "snapshot_pairs_trading_close.parquet")
    pt_open.to_parquet(tmp_path / "snapshot_pairs_trading_open.parquet")

    va_close, va_open = _vol_arbitrage_panel(n_days, seed=1)
    va_close.to_parquet(tmp_path / "snapshot_vol_arbitrage_close.parquet")
    va_open.to_parquet(tmp_path / "snapshot_vol_arbitrage_open.parquet")

    fm_tickers = [f"T{i}" for i in range(10)]
    fm_close, fm_open = _price_panel(fm_tickers, n_days, seed=2)
    fm_close.to_parquet(tmp_path / "snapshot_factor_momentum_close.parquet")
    fm_open.to_parquet(tmp_path / "snapshot_factor_momentum_open.parquet")

    candidate = PairCandidate(
        ticker_a="TICKA",
        ticker_b="TICKB",
        sector="Utilities",
        engle_granger=EngleGrangerResult(
            dependent="TICKA", independent="TICKB", intercept=0.0, hedge_ratio=1.0,
            test_statistic=-4.0, p_value=7.666e-05, is_cointegrated=True,
        ),
        johansen=JohansenResult(trace_statistic=20.0, critical_value_95=15.0, is_cointegrated=True, cointegrating_vector=(1.0, -1.0)),
        half_life=15.0,
        adjusted_engle_granger_p_value=0.0236,
        n_tests=308,
    )
    with open(tmp_path / "snapshot_pairs_trading_candidate.pkl", "wb") as f:
        pickle.dump(candidate, f)

    tier1 = {
        "as_of": date(2026, 7, 25).isoformat(),
        "pairs_trading_candidate": ["TICKA", "TICKB"],
        "pairs_trading_walk_forward": [
            {"label": "Window 1", "oos_sharpe": -0.73, "permutation_p": 0.697},
            {"label": "Window 2", "oos_sharpe": None, "permutation_p": None},
            {"label": "Window 3", "oos_sharpe": None, "permutation_p": None},
            {"label": "Window 4", "oos_sharpe": -0.62, "permutation_p": 0.613},
        ],
        "factor_momentum_walk_forward": [
            {"label": "Window 1", "oos_sharpe": 0.94, "permutation_p": 0.2},
            {"label": "Window 2", "oos_sharpe": -0.08, "permutation_p": 0.4},
            {"label": "Window 3", "oos_sharpe": -0.92, "permutation_p": 0.5},
            {"label": "Window 4", "oos_sharpe": 1.86, "permutation_p": 0.1},
        ],
        "vol_arbitrage_walk_forward": [
            {"label": "Window 1", "oos_sharpe": -1.30, "permutation_p": 0.81},
            {"label": "Window 2", "oos_sharpe": -1.38, "permutation_p": 0.83},
            {"label": "Window 3", "oos_sharpe": 1.72, "permutation_p": 0.11},
            {"label": "Window 4", "oos_sharpe": -0.18, "permutation_p": 0.57},
        ],
        "raw_p": 7.666e-05,
        "bonferroni_scans": [
            {"n_tests": 308, "survivors_before": 18, "survivors_after": 1, "label": "2-sector"},
            {"n_tests": 5551, "survivors_before": 237, "survivors_after": 0, "label": "full-universe"},
        ],
    }
    with open(tmp_path / "tier1_precomputed.json", "w", encoding="utf-8") as f:
        json.dump(tier1, f)

    return tmp_path
