"""Factor Momentum (global mode) dashboard page.

Tier 1 (precomputed walk-forward) then Tier 2 (live long/short
percentile slider against the frozen full-universe snapshot).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import streamlit as st

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from core.evaluation.statistical_tests import check_significance
from dashboard.components import render_significance_panel, tier1_banner_text, tier2_banner_text
from dashboard.data import load_factor_momentum_snapshot, load_tier1_precomputed, resolve_data_dir
from strategies.factor_momentum.ranking import FactorMomentumConfig
from strategies.factor_momentum.strategy import FactorMomentumStrategy

st.title("Factor Momentum: Momentum + Low-Volatility (Global Mode)")

data_dir = resolve_data_dir()
tier1 = load_tier1_precomputed(data_dir)

st.header("Tier 1: Walk-Forward (事前計算)")
st.info(tier1_banner_text())
for window in tier1.factor_momentum_walk_forward:
    if window.oos_sharpe is None:
        st.write(f"- {window.label}: N/A")
    else:
        st.write(f"- {window.label}: OOS Sharpe={window.oos_sharpe:.2f}, permutation p={window.permutation_p:.4f}")

st.header("Tier 2: Long/Short Percentile（ライブ再計算）")
st.info(tier2_banner_text(tier1.as_of))

close_panel, open_panel = load_factor_momentum_snapshot(data_dir)
split = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), close_panel.index[-1]),
)

percentile = st.slider("percentile (long=1-p, short=p)", min_value=0.05, max_value=0.45, value=0.20, step=0.05)
st.caption("デフォルト値（percentile=0.20、quintile）は事前に別の根拠で選定された値であり、この感度分析のために選んだものではありません。")

config = dataclasses.replace(FactorMomentumConfig(), long_percentile=1.0 - percentile, short_percentile=percentile)
strategy = FactorMomentumStrategy(config=config)
sizer = VolTargetSizer(PositionSizingConfig())
result = run_backtest(close_panel, open_panel, strategy, sizer, BacktestConfig())
_, oos_returns = split_returns(result.returns, split)
oos_metrics = metrics_mod.summary(oos_returns)
significance = check_significance(oos_returns, seed=0)

render_significance_panel(oos_metrics, significance)
