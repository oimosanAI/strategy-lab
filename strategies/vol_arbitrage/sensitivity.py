"""Parameter sensitivity analysis for vol_arbitrage.

vrp_threshold only affects signal generation
(VolArbitrageStrategy.generate_signals' threshold comparison) -- no
selection step to re-run (same structure as
strategies.factor_momentum.sensitivity.run_percentile_grid), so this is
backtest-only and cheap regardless of grid size.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizer, PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import TrainTestSplit, split_returns
from core.evaluation import metrics
from core.evaluation.sensitivity import SensitivityGridPoint
from core.evaluation.statistical_tests import check_significance
from strategies.vol_arbitrage.strategy import VolArbitrageConfig, VolArbitrageStrategy


def run_vrp_threshold_grid(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    split: TrainTestSplit,
    thresholds: Sequence[float],
    default_threshold: float = 0.0,
    base_config: VolArbitrageConfig | None = None,
    sizer: PositionSizer | None = None,
    backtest_config: BacktestConfig | None = None,
) -> list[SensitivityGridPoint]:
    """1D grid over vrp_threshold. Backtest-only -- no re-selection needed."""
    base_config = base_config or VolArbitrageConfig()
    resolved_sizer = sizer or VolTargetSizer(PositionSizingConfig())
    resolved_backtest_config = backtest_config or BacktestConfig()

    points: list[SensitivityGridPoint] = []
    for threshold in thresholds:
        config = dataclasses.replace(base_config, vrp_threshold=threshold)
        strategy = VolArbitrageStrategy(config=config)
        result = run_backtest(prices, open_prices, strategy, resolved_sizer, resolved_backtest_config)
        _, oos_returns = split_returns(result.returns, split)
        points.append(
            SensitivityGridPoint(
                params={"vrp_threshold": threshold},
                label="vol-arbitrage",
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, seed=0),
                is_default=(threshold == default_threshold),
            )
        )
    return points
