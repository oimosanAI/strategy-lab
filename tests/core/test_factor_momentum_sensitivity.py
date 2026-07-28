"""Tests for strategies.factor_momentum.sensitivity.

Fixes the public API surface:

- run_percentile_grid(prices, open_prices, split, percentiles,
  default_percentile=0.20, base_config=None, sizer=None,
  backtest_config=None) -> list[SensitivityGridPoint]: a 1D grid over a
  single symmetric "bucket fraction" p, mapped to
  long_percentile=1-p / short_percentile=p (quintile/quartile/decile
  terminology stays a single parameter, matching how these cuts are
  conventionally discussed as symmetric long/short buckets). Backtest-
  only -- factor_momentum has no selection step to re-run (see
  strategies/factor_momentum/README.md's walk-forward discussion for why).
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from strategies.factor_momentum.sensitivity import run_percentile_grid
from strategies.factor_momentum.strategy import FactorMomentumStrategy
from tests.core.test_strategy_factor_momentum import _TEST_CONFIG, _synthetic_price_panel


def _split(index: pd.DatetimeIndex, is_end_idx: int) -> TrainTestSplit:
    return TrainTestSplit(
        in_sample=SamplePeriod(index[0], index[is_end_idx]),
        out_of_sample=SamplePeriod(index[is_end_idx + 1], index[-1]),
    )


def test_run_percentile_grid_produces_one_point_per_percentile() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_percentile_grid(prices, prices, split, percentiles=[0.10, 0.20, 0.30], base_config=_TEST_CONFIG)

    assert len(points) == 3
    assert {p.params["percentile"] for p in points} == {0.10, 0.20, 0.30}


def test_run_percentile_grid_maps_percentile_to_long_short_correctly() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_percentile_grid(prices, prices, split, percentiles=[0.10], base_config=_TEST_CONFIG)

    config = dataclasses.replace(_TEST_CONFIG, long_percentile=0.90, short_percentile=0.10)
    strategy = FactorMomentumStrategy(config=config)
    sizer = VolTargetSizer(PositionSizingConfig())
    result = run_backtest(prices, prices, strategy, sizer, BacktestConfig())
    _, expected_oos = split_returns(result.returns, split)
    expected_metrics = metrics_mod.summary(expected_oos)

    assert points[0].oos_metrics == expected_metrics


def test_run_percentile_grid_marks_default_correctly() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_percentile_grid(
        prices, prices, split, percentiles=[0.10, 0.20], default_percentile=0.20, base_config=_TEST_CONFIG
    )

    defaults = [p for p in points if p.is_default]
    assert len(defaults) == 1
    assert defaults[0].params == {"percentile": 0.20}
