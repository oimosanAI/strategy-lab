"""Pairs Trading (AEP-FE, narrow 2-sector pool) dashboard page.

Tier 1 (precomputed full-universe walk-forward + Bonferroni scan) then
Tier 2 (live entry/exit threshold sliders on the already-selected AEP-FE
candidate -- select_pairs itself is never re-run in this app, see
scripts/generate_dashboard_data.py).
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
from dashboard.data import (
    load_pairs_trading_candidate,
    load_pairs_trading_snapshot,
    load_tier1_precomputed,
    resolve_data_dir,
)
from strategies.pairs_trading.strategy import PairsTradingSignalConfig, PairsTradingStrategy

st.title("Pairs Trading: AEP-FE (Narrow 2-Sector Pool — Reference Only)")
st.warning(
    "この候補（AEP-FE）は狭い2セクタープールでの参考情報です。全宇宙（503銘柄、5,551検定）での"
    "Bonferroni補正では生存ペア0本であり、統計的に有意なエッジではありません。"
)

data_dir = resolve_data_dir()
tier1 = load_tier1_precomputed(data_dir)

st.header("Tier 1: 全宇宙Walk-Forward・多重検定（事前計算）")
st.info(tier1_banner_text())
for window in tier1.pairs_trading_walk_forward:
    if window.oos_sharpe is None:
        st.write(f"- {window.label}: N/A (no candidate)")
    else:
        st.write(f"- {window.label}: OOS Sharpe={window.oos_sharpe:.2f}, permutation p={window.permutation_p:.4f}")

st.subheader("多重検定（Bonferroni補正）")
for scan in tier1.bonferroni_scans:
    adjusted_p = min(tier1.raw_p * scan.n_tests, 1.0)
    st.write(
        f"- {scan.label}（n_tests={scan.n_tests}）: adjusted p={adjusted_p:.4f}, "
        f"生存数 {scan.survivors_before}→{scan.survivors_after}"
    )

st.header("Tier 2: Entry/Exit Threshold（ライブ再計算）")
st.info(tier2_banner_text(tier1.as_of))

candidate = load_pairs_trading_candidate(data_dir)
close_panel, open_panel = load_pairs_trading_snapshot(data_dir)
split = TrainTestSplit(
    in_sample=SamplePeriod(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31")),
    out_of_sample=SamplePeriod(pd.Timestamp("2025-01-02"), close_panel.index[-1]),
)

entry_threshold = st.slider("entry_threshold", min_value=1.5, max_value=2.5, value=2.0, step=0.25)
exit_threshold = st.slider("exit_threshold", min_value=0.0, max_value=0.5, value=0.0, step=0.25)
st.caption(
    "デフォルト値（entry_threshold=2.0, exit_threshold=0.0）は事前に別の根拠（機関投資家慣行）で"
    "選定された値であり、この感度分析のために選んだものではありません。"
)

signal_config = PairsTradingSignalConfig(entry_threshold=entry_threshold, exit_threshold=exit_threshold)
strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static", config=signal_config)
sizer = VolTargetSizer(PositionSizingConfig())
result = run_backtest(close_panel, open_panel, strategy, sizer, BacktestConfig())
_, oos_returns = split_returns(result.returns, split)
oos_metrics = metrics_mod.summary(oos_returns)
significance = check_significance(oos_returns, seed=0)

render_significance_panel(oos_metrics, significance)
