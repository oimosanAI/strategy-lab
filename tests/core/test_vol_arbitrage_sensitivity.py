"""Tests for strategies.vol_arbitrage.sensitivity.

Fixes the public API surface:

- run_vrp_threshold_grid(prices, open_prices, split, thresholds,
  default_threshold=0.0, base_config=None, sizer=None,
  backtest_config=None) -> list[SensitivityGridPoint]: a 1D grid over
  vrp_threshold. Backtest-only -- vrp_threshold only affects
  VolArbitrageStrategy.generate_signals' threshold comparison, no
  selection step to re-run (same structure as factor_momentum's
  percentile grid).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from strategies.vol_arbitrage.sensitivity import run_vrp_threshold_grid
from strategies.vol_arbitrage.strategy import VolArbitrageStrategy
from tests.core.test_vol_arbitrage_strategy import _TEST_CONFIG, _synthetic_price_panel


def _split(index: pd.DatetimeIndex, is_end_idx: int) -> TrainTestSplit:
    return TrainTestSplit(
        in_sample=SamplePeriod(index[0], index[is_end_idx]),
        out_of_sample=SamplePeriod(index[is_end_idx + 1], index[-1]),
    )


def test_run_vrp_threshold_grid_produces_one_point_per_threshold() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_vrp_threshold_grid(
        prices, prices, split, thresholds=[-0.05, 0.0, 0.05], base_config=_TEST_CONFIG
    )

    assert len(points) == 3
    assert {p.params["vrp_threshold"] for p in points} == {-0.05, 0.0, 0.05}


def test_run_vrp_threshold_grid_wiring_matches_direct_backtest() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_vrp_threshold_grid(prices, prices, split, thresholds=[0.05], base_config=_TEST_CONFIG)

    config = dataclasses.replace(_TEST_CONFIG, vrp_threshold=0.05)
    strategy = VolArbitrageStrategy(config=config)
    sizer = VolTargetSizer(PositionSizingConfig())
    result = run_backtest(prices, prices, strategy, sizer, BacktestConfig())
    _, expected_oos = split_returns(result.returns, split)
    expected_metrics = metrics_mod.summary(expected_oos)

    # threshold=0.05 happens to produce a fully-flat (never-triggered)
    # OOS window on this synthetic panel/split, so several metrics are
    # NaN (0/0 division) on BOTH sides -- plain dict `==` fails on NaN
    # (nan != nan), even though both paths compute via the identical
    # run_backtest call. nan_ok=True treats matching NaNs as equal.
    assert points[0].oos_metrics == pytest.approx(expected_metrics, nan_ok=True)


def test_run_vrp_threshold_grid_marks_default_correctly() -> None:
    prices = _synthetic_price_panel()
    split = _split(prices.index, 59)

    points = run_vrp_threshold_grid(
        prices, prices, split, thresholds=[-0.05, 0.0], default_threshold=0.0, base_config=_TEST_CONFIG
    )

    defaults = [p for p in points if p.is_default]
    assert len(defaults) == 1
    assert defaults[0].params == {"vrp_threshold": 0.0}
