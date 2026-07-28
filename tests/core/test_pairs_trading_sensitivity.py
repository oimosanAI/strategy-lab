"""Tests for strategies.pairs_trading.sensitivity.

Fixes the public API surface:

- run_entry_exit_threshold_grid(candidate, prices, open_prices, split,
  entry_thresholds, exit_thresholds, default_entry=2.0, default_exit=0.0,
  sizer=None, backtest_config=None) -> list[SensitivityGridPoint]: a
  full 2D grid (backtest-only, no re-selection -- entry/exit thresholds
  only affect signal generation for an ALREADY-selected candidate).
- run_correlation_prefilter_grid(prices, open_prices, split, sector_map,
  prefilters, default_prefilter=0.7, base_config=None, sizer=None,
  backtest_config=None) -> list[SensitivityGridPoint]: re-runs
  select_pairs at EACH grid point (correlation_prefilter changes the
  candidate pool itself, hence n_tests and the Bonferroni threshold) --
  mirrors run_pairs_trading_walk_forward's "no candidate" handling.
"""

from __future__ import annotations

import pandas as pd

from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.evaluation import metrics as metrics_mod
from strategies.pairs_trading.cointegration import select_pairs
from strategies.pairs_trading.sensitivity import run_correlation_prefilter_grid, run_entry_exit_threshold_grid
from strategies.pairs_trading.strategy import PairsTradingSignalConfig, PairsTradingStrategy
from tests.core.test_cointegration import _cointegrated_pair, _non_cointegrated_pair
from tests.core.test_strategy import _cointegrated_panel, _make_candidate


def _split(index: pd.DatetimeIndex, is_end_idx: int) -> TrainTestSplit:
    return TrainTestSplit(
        in_sample=SamplePeriod(index[0], index[is_end_idx]),
        out_of_sample=SamplePeriod(index[is_end_idx + 1], index[-1]),
    )


# ---------------------------------------------------------------------------
# run_entry_exit_threshold_grid
# ---------------------------------------------------------------------------


def test_run_entry_exit_threshold_grid_produces_full_2d_grid() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    split = _split(prices.index, 149)

    points = run_entry_exit_threshold_grid(
        candidate, prices, prices, split, entry_thresholds=[1.5, 2.0, 2.5], exit_thresholds=[0.0, 0.5]
    )

    assert len(points) == 6
    param_pairs = {(p.params["entry_threshold"], p.params["exit_threshold"]) for p in points}
    assert param_pairs == {(1.5, 0.0), (1.5, 0.5), (2.0, 0.0), (2.0, 0.5), (2.5, 0.0), (2.5, 0.5)}


def test_run_entry_exit_threshold_grid_marks_default_correctly() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    split = _split(prices.index, 149)

    points = run_entry_exit_threshold_grid(
        candidate,
        prices,
        prices,
        split,
        entry_thresholds=[1.5, 2.0],
        exit_thresholds=[0.0, 0.5],
        default_entry=2.0,
        default_exit=0.0,
    )

    defaults = [p for p in points if p.is_default]
    assert len(defaults) == 1
    assert defaults[0].params == {"entry_threshold": 2.0, "exit_threshold": 0.0}


def test_run_entry_exit_threshold_grid_matches_direct_backtest() -> None:
    prices = _cointegrated_panel()
    candidate = _make_candidate(prices)
    split = _split(prices.index, 149)

    points = run_entry_exit_threshold_grid(
        candidate, prices, prices, split, entry_thresholds=[2.0], exit_thresholds=[0.0]
    )

    config = PairsTradingSignalConfig(entry_threshold=2.0, exit_threshold=0.0)
    strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static", config=config)
    sizer = VolTargetSizer(PositionSizingConfig())
    result = run_backtest(prices, prices, strategy, sizer, BacktestConfig())
    _, expected_oos = split_returns(result.returns, split)
    expected_metrics = metrics_mod.summary(expected_oos)

    assert points[0].oos_metrics == expected_metrics


# ---------------------------------------------------------------------------
# run_correlation_prefilter_grid
# ---------------------------------------------------------------------------


def test_run_correlation_prefilter_grid_handles_no_candidate_without_stopping() -> None:
    a, b = _non_cointegrated_pair(n=300)
    prices = pd.DataFrame({"A": a, "B": b})
    sector_map = {"A": "Tech", "B": "Tech"}
    split = _split(prices.index, 199)

    points = run_correlation_prefilter_grid(prices, prices, split, sector_map, prefilters=[0.5, 0.9])

    assert len(points) == 2
    for point in points:
        assert point.label == "no candidate"
        assert point.oos_metrics is None
        assert point.significance is None


def test_run_correlation_prefilter_grid_matches_direct_select_pairs() -> None:
    a, b = _cointegrated_pair(n=300)
    prices = pd.DataFrame({"A": a, "B": b})
    sector_map = {"A": "Tech", "B": "Tech"}
    split = _split(prices.index, 199)

    points = run_correlation_prefilter_grid(prices, prices, split, sector_map, prefilters=[0.7])

    expected_candidates = select_pairs(prices, split.in_sample, sector_map)
    assert expected_candidates, "fixture must actually be cointegrated for this test to be meaningful"
    assert points[0].label == f"{expected_candidates[0].ticker_a}-{expected_candidates[0].ticker_b}"
    assert points[0].oos_metrics is not None


def test_run_correlation_prefilter_grid_marks_default_correctly() -> None:
    a, b = _cointegrated_pair(n=300)
    prices = pd.DataFrame({"A": a, "B": b})
    sector_map = {"A": "Tech", "B": "Tech"}
    split = _split(prices.index, 199)

    points = run_correlation_prefilter_grid(
        prices, prices, split, sector_map, prefilters=[0.5, 0.7], default_prefilter=0.7
    )

    defaults = [p for p in points if p.is_default]
    assert len(defaults) == 1
    assert defaults[0].params == {"correlation_prefilter": 0.7}
