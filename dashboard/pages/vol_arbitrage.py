"""Vol Arbitrage (VIX/SPY VRP) dashboard page.

Tier 1 (precomputed walk-forward) then Tier 2 (live vrp_threshold
slider against the frozen VIX/SPY/SVXY snapshot) -- this ordering is
deliberate: the rigorous, already-verified result is shown first, the
explorable parameter comes second.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from core.evaluation.statistical_tests import check_significance
from dashboard.components import render_significance_panel, tier1_banner_text, tier2_banner_text
from dashboard.data import load_tier1_precomputed, load_vol_arbitrage_snapshot, resolve_data_dir
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy

st.title("Vol Arbitrage: VIX/SPY Volatility Risk Premium")

data_dir = resolve_data_dir()
tier1 = load_tier1_precomputed(data_dir)

st.header("Tier 1: Walk-Forward (事前計算)")
st.info(tier1_banner_text())
for window in tier1.vol_arbitrage_walk_forward:
    if window.oos_sharpe is None:
        st.write(f"- {window.label}: N/A")
    else:
        st.write(f"- {window.label}: OOS Sharpe={window.oos_sharpe:.2f}, permutation p={window.permutation_p:.4f}")

st.header("Tier 2: vrp_threshold（ライブ再計算）")
st.info(tier2_banner_text(tier1.as_of))

close_panel, open_panel = load_vol_arbitrage_snapshot(data_dir)
split = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), close_panel.index[-1]),
)

threshold = st.slider("vrp_threshold", min_value=-0.10, max_value=0.10, value=0.0, step=0.01)
st.caption("デフォルト値（vrp_threshold=0.0）は事前に別の根拠で選定された値であり、この感度分析のために選んだものではありません。")

config = VolArbitrageConfig(vrp_threshold=threshold)
strategy = VolArbitrageStrategy(config=config)
sizer = VolTargetSizer(PositionSizingConfig())
result = run_backtest(close_panel, open_panel, strategy, sizer, BacktestConfig())
_, oos_returns = split_returns(result.returns, split)
oos_metrics = metrics_mod.summary(oos_returns)
significance = check_significance(oos_returns, seed=0)

render_significance_panel(oos_metrics, significance)
