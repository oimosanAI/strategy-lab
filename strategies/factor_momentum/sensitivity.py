"""Parameter sensitivity analysis for factor_momentum.

long_percentile/short_percentile only affect signal generation
(ranking.build_long_short_signal) -- no selection step to re-run (unlike
pairs_trading's correlation_prefilter), so this is backtest-only and
cheap regardless of grid size. Parametrized as a single symmetric
"bucket fraction" p (long=1-p, short=p), matching how quintile/quartile/
decile cuts are conventionally discussed as a single symmetric threshold
rather than two independently-chosen values.
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
from strategies.factor_momentum.ranking import FactorMomentumConfig
from strategies.factor_momentum.strategy import FactorMomentumStrategy


def run_percentile_grid(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    split: TrainTestSplit,
    percentiles: Sequence[float],
    default_percentile: float = 0.20,
    base_config: FactorMomentumConfig | None = None,
    sizer: PositionSizer | None = None,
    backtest_config: BacktestConfig | None = None,
) -> list[SensitivityGridPoint]:
    """1D grid over the symmetric bucket fraction p (long=1-p, short=p).
    Backtest-only -- no re-selection needed."""
    base_config = base_config or FactorMomentumConfig()
    resolved_sizer = sizer or VolTargetSizer(PositionSizingConfig())
    resolved_backtest_config = backtest_config or BacktestConfig()

    points: list[SensitivityGridPoint] = []
    for p in percentiles:
        config = dataclasses.replace(base_config, long_percentile=1.0 - p, short_percentile=p)
        strategy = FactorMomentumStrategy(config=config)
        result = run_backtest(prices, open_prices, strategy, resolved_sizer, resolved_backtest_config)
        _, oos_returns = split_returns(result.returns, split)
        points.append(
            SensitivityGridPoint(
                params={"percentile": p},
                label="factor-momentum",
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, seed=0),
                is_default=(p == default_percentile),
            )
        )
    return points
