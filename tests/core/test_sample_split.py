"""Tests for core.backtest.sample_split.

Fixes the public API surface core/backtest/sample_split.py must
implement:

- SamplePeriod(start, end) / TrainTestSplit(in_sample, out_of_sample):
  out_of_sample.start must be strictly after in_sample.end. Gaps between
  the two periods are allowed (not required to be adjacent) -- the
  validation only forbids overlap/wrong order, so a caller wanting a gap
  just constructs SamplePeriod boundaries further apart; no change to
  these types is needed for that.
- InSampleData / slice_in_sample(prices, split): the ONLY sanctioned way
  to obtain data that pair-selection / static-parameter-estimation
  functions are allowed to see.
- assert_selection_ignores_out_of_sample(selection_fn, prices, split):
  the IS/OOS-boundary analogue of core.backtest.engine.assert_causal --
  perturbs only the out-of-sample portion of `prices` and asserts
  selection_fn's result (computed from the re-sliced InSampleData) is
  unchanged. Reuses LookAheadError: out_of_sample is definitionally later
  in time than in_sample (enforced by TrainTestSplit itself), so using
  OOS data during IS-only selection is a form of look-ahead bias too.
- split_returns(returns, split) -> (is_returns, oos_returns): the
  sanctioned way to slice a CONTINUOUSLY-run BacktestResult.returns into
  per-period pieces for separate Pillar-3 scoring. Deliberately NOT
  "run run_backtest twice, once per period" -- see
  test_continuous_execution_avoids_warmup_gap_that_oos_only_execution_has
  for why that would be wrong.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, run_backtest
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import (
    InSampleData,
    SamplePeriod,
    TrainTestSplit,
    assert_selection_ignores_out_of_sample,
    generate_anchored_windows,
    slice_in_sample,
    split_returns,
)
from tests.core.test_backtest_engine import _FixedLongShortStrategy


def _panel(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"A": 100 + np.cumsum(rng.normal(0, 1, n_days)), "B": 100 + np.cumsum(rng.normal(0, 1, n_days))},
        index=idx,
    ).abs() + 1.0


# ---------------------------------------------------------------------------
# 1. SamplePeriod / TrainTestSplit / InSampleData validation
# ---------------------------------------------------------------------------


def test_train_test_split_rejects_overlapping_periods() -> None:
    in_sample = SamplePeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    out_of_sample = SamplePeriod(pd.Timestamp("2020-06-15"), pd.Timestamp("2020-12-31"))  # overlaps
    with pytest.raises(ValueError):
        TrainTestSplit(in_sample=in_sample, out_of_sample=out_of_sample)


def test_train_test_split_accepts_adjacent_periods() -> None:
    in_sample = SamplePeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    out_of_sample = SamplePeriod(pd.Timestamp("2020-07-01"), pd.Timestamp("2020-12-31"))
    split = TrainTestSplit(in_sample=in_sample, out_of_sample=out_of_sample)
    assert split.out_of_sample.start == pd.Timestamp("2020-07-01")


def test_train_test_split_accepts_gap_between_periods() -> None:
    # A deliberate gap (e.g. an embargo period) requires no change to
    # SamplePeriod/TrainTestSplit -- just further-apart boundaries.
    in_sample = SamplePeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    out_of_sample = SamplePeriod(pd.Timestamp("2020-09-01"), pd.Timestamp("2020-12-31"))
    split = TrainTestSplit(in_sample=in_sample, out_of_sample=out_of_sample)
    assert split.out_of_sample.start > split.in_sample.end + timedelta(days=60)


def test_slice_in_sample_returns_exact_date_range() -> None:
    # Arrange
    prices = _panel(n_days=100, seed=1)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[99]),
    )

    # Act
    result = slice_in_sample(prices, split.in_sample)

    # Assert
    assert isinstance(result, InSampleData)
    assert result.prices.index[0] == prices.index[0]
    assert result.prices.index[-1] == prices.index[59]
    assert len(result.prices) == 60


def test_slice_in_sample_excludes_out_of_sample_rows() -> None:
    # Arrange
    prices = _panel(n_days=100, seed=1)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[99]),
    )

    # Act
    result = slice_in_sample(prices, split.in_sample)

    # Assert: no OOS date is present at all.
    oos_dates = set(prices.index[60:])
    assert oos_dates.isdisjoint(set(result.prices.index))


# ---------------------------------------------------------------------------
# 2 & 3. assert_selection_ignores_out_of_sample: positive and negative control
# ---------------------------------------------------------------------------


def _correct_selector(prices: pd.DataFrame, in_sample: SamplePeriod) -> float:
    """Positive control: receives the FULL panel (mirroring
    Strategy.generate_signals in assert_causal) but restricts itself to
    the in-sample period internally, via slice_in_sample."""
    data = slice_in_sample(prices, in_sample)
    return float(data.prices["A"].mean())


def _leaking_selector(prices: pd.DataFrame, in_sample: SamplePeriod) -> float:  # noqa: ARG001
    """Negative control: ignores `in_sample` entirely and computes its
    statistic from the WHOLE panel -- the realistic 'forgot to slice'
    bug this guard exists to catch. Structurally identical to
    test_engine.py's _FullSampleOLSHedgeStrategy, one layer up (IS/OOS
    boundary instead of the day-t time boundary)."""
    return float(prices["A"].mean())


def test_assert_selection_ignores_oos_passes_for_correct_selector() -> None:
    # Arrange
    prices = _panel(n_days=100, seed=2)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[99]),
    )

    # Act / Assert: no exception.
    assert_selection_ignores_out_of_sample(_correct_selector, prices, split)


def test_assert_selection_ignores_oos_catches_leaking_selector() -> None:
    # Arrange
    prices = _panel(n_days=100, seed=3)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[99]),
    )

    # Act / Assert: this guard must bite, or it is asleep at the wheel.
    with pytest.raises(LookAheadError):
        assert_selection_ignores_out_of_sample(_leaking_selector, prices, split)


# ---------------------------------------------------------------------------
# 4. run_backtest: continuous IS+OOS execution, then split_returns()
# ---------------------------------------------------------------------------


def test_split_returns_gives_correct_date_ranges_and_lengths() -> None:
    # Arrange
    prices = _panel(n_days=100, seed=4)
    strategy = _FixedLongShortStrategy()
    sizer = VolTargetSizer(PositionSizingConfig(vol_window=10))
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[99]),
    )
    result = run_backtest(prices, prices, strategy, sizer)

    # Act
    is_returns, oos_returns = split_returns(result.returns, split)

    # Assert
    assert is_returns.index[0] == prices.index[0]
    assert is_returns.index[-1] == prices.index[59]
    assert len(is_returns) == 60
    assert oos_returns.index[0] == prices.index[60]
    assert oos_returns.index[-1] == prices.index[99]
    assert len(oos_returns) == 40


def test_continuous_execution_avoids_warmup_gap_that_oos_only_execution_has() -> None:
    # Arrange: same strategy/sizer, same underlying prices.
    prices = _panel(n_days=90, seed=5)
    strategy = _FixedLongShortStrategy()
    sizer = VolTargetSizer(PositionSizingConfig(vol_window=10))
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[59]),
        out_of_sample=SamplePeriod(prices.index[60], prices.index[89]),
    )

    # Act: (a) run once, continuously, over the full IS+OOS panel.
    continuous_result = run_backtest(prices, prices, strategy, sizer)
    oos_position_continuous = continuous_result.position.loc[
        split.out_of_sample.start : split.out_of_sample.end
    ]

    # (b) run independently on the OOS-only slice.
    oos_prices = prices.loc[split.out_of_sample.start : split.out_of_sample.end]
    oos_only_result = run_backtest(oos_prices, oos_prices, strategy, sizer)

    # Assert: continuous execution inherited the rolling window's memory
    # from the tail of the in-sample period, so OOS's first vol_window
    # days are ALREADY populated (not flat) -- no warm-up gap at the
    # IS/OOS boundary.
    first_10_days_continuous = oos_position_continuous.iloc[:10]
    assert (first_10_days_continuous != 0.0).all(axis=None)

    # Whereas running the OOS slice in isolation restarts the rolling
    # window from scratch, forcing a fresh warm-up: its first 10 days are
    # flat (zero), an artifact of slicing before running rather than after.
    first_10_days_oos_only = oos_only_result.position.iloc[:10]
    assert (first_10_days_oos_only == 0.0).all(axis=None)


# ---------------------------------------------------------------------------
# 5. generate_anchored_windows: walk-forward window generation
# ---------------------------------------------------------------------------


def test_generate_anchored_windows_matches_hand_computed_boundaries() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    boundaries = [idx[39], idx[69]]

    windows = generate_anchored_windows(idx, boundaries)

    assert len(windows) == 2
    assert windows[0].in_sample.start == idx[0]
    assert windows[0].in_sample.end == idx[39]
    assert windows[0].out_of_sample.start == idx[40]
    assert windows[0].out_of_sample.end == idx[69]
    # Anchored: window 2's in-sample still starts at idx[0], just extends
    # further -- not a fresh, independent in-sample period.
    assert windows[1].in_sample.start == idx[0]
    assert windows[1].in_sample.end == idx[69]
    assert windows[1].out_of_sample.start == idx[70]
    assert windows[1].out_of_sample.end == idx[99]


def test_generate_anchored_windows_snaps_boundary_to_latest_actual_date() -> None:
    # A deliberately gappy index -- 2020-01-07 is NOT a real trading day
    # here, so the boundary must snap to the latest real date before it
    # (mirroring the same "read the real index, not a theoretical
    # calendar" reasoning already used for month-end detection elsewhere
    # in this project).
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-08", "2020-01-09"])
    boundaries = [pd.Timestamp("2020-01-07")]

    windows = generate_anchored_windows(idx, boundaries)

    assert windows[0].in_sample.end == pd.Timestamp("2020-01-06")
    assert windows[0].out_of_sample.start == pd.Timestamp("2020-01-08")
    assert windows[0].out_of_sample.end == idx[-1]


def test_generate_anchored_windows_single_boundary_extends_oos_to_last_date() -> None:
    idx = pd.bdate_range("2020-01-01", periods=50)

    windows = generate_anchored_windows(idx, [idx[29]])

    assert len(windows) == 1
    assert windows[0].out_of_sample.end == idx[-1]


def test_generate_anchored_windows_rejects_boundary_before_earliest_date() -> None:
    idx = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        generate_anchored_windows(idx, [idx[0] - timedelta(days=10)])


def test_generate_anchored_windows_rejects_boundary_at_last_available_date() -> None:
    # No room for an out-of-sample period if the boundary snaps to the
    # very last available date.
    idx = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        generate_anchored_windows(idx, [idx[-1]])


def test_generate_anchored_windows_rejects_non_increasing_boundaries() -> None:
    idx = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        generate_anchored_windows(idx, [idx[30], idx[10]])


def test_generate_anchored_windows_rejects_empty_boundaries() -> None:
    idx = pd.bdate_range("2020-01-01", periods=50)
    with pytest.raises(ValueError):
        generate_anchored_windows(idx, [])
