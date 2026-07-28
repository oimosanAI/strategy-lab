"""Tests for strategies.factor_momentum.rebalance.

Fixes the public API surface:

- month_end_dates(index) -> pd.DatetimeIndex: the last date ACTUALLY
  PRESENT in `index` for each (year, month) group -- not a theoretical
  calendar month-end. Deliberately data-driven, mirroring how
  core.backtest.sample_split treats the price panel's own index as the
  source of truth rather than reconstructing an idealized trading
  calendar (holidays/gaps in the real panel would otherwise desync a
  calendar-formula month-end from any date that actually exists).
- hold_until_next_rebalance(daily_signal, rebalance_dates) -> pd.DataFrame:
  keeps daily_signal's value only at rebalance_dates rows and forward-
  fills it to every date until the next rebalance date. Before the first
  rebalance date -- or while the value AT a rebalance date is itself NaN
  (e.g. a factor score still in its warm-up period) -- the result is 0
  (flat). ffill() only ever copies a PAST value forward, so this
  composition stays causal: identical reasoning to
  core.backtest.engine.assert_causal's causal-rolling-window pattern,
  verified structurally here and empirically via assert_causal on the
  full strategy once strategy.py exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.factor_momentum.rebalance import hold_until_next_rebalance, month_end_dates


def test_month_end_dates_picks_last_actual_date_per_calendar_month() -> None:
    # Arrange: a deliberately gappy index -- Feb's last entry (02-04) is
    # NOT a real calendar month-end, proving the function reads the
    # index's own last-present date rather than computing one from a
    # calendar rule.
    index = pd.DatetimeIndex(
        [
            "2021-01-04", "2021-01-05", "2021-01-06",
            "2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04",
            "2021-03-01", "2021-03-02",
        ]
    )

    # Act
    result = month_end_dates(index)

    # Assert
    expected = pd.DatetimeIndex(["2021-01-06", "2021-02-04", "2021-03-02"])
    pd.testing.assert_index_equal(result, expected)


def test_hold_until_next_rebalance_holds_value_between_rebalance_dates() -> None:
    idx = pd.bdate_range("2020-01-01", periods=12)
    daily_signal = pd.DataFrame({"A": [np.nan] * 12}, index=idx)
    daily_signal.loc[idx[2], "A"] = 1.0
    daily_signal.loc[idx[7], "A"] = -1.0
    rebalance_dates = pd.DatetimeIndex([idx[2], idx[7]])

    result = hold_until_next_rebalance(daily_signal, rebalance_dates)

    expected = pd.Series(
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        index=idx,
        name="A",
    )
    pd.testing.assert_series_equal(result["A"], expected)


def test_hold_until_next_rebalance_ignores_non_rebalance_date_values() -> None:
    # A value present on a NON-rebalance date must have no effect on the
    # held signal -- only rebalance_dates rows are ever read.
    idx = pd.bdate_range("2020-01-01", periods=6)
    daily_signal = pd.DataFrame({"A": [np.nan, np.nan, 1.0, 999.0, np.nan, np.nan]}, index=idx)
    rebalance_dates = pd.DatetimeIndex([idx[2]])  # only idx[2] is a real rebalance date

    result = hold_until_next_rebalance(daily_signal, rebalance_dates)

    # idx[3]'s 999.0 sits on a non-rebalance date and must be ignored; the
    # held value stays 1.0 (from idx[2]) all the way through.
    expected = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=idx, name="A")
    pd.testing.assert_series_equal(result["A"], expected)


def test_hold_until_next_rebalance_treats_nan_at_rebalance_date_as_flat_until_first_valid_decision() -> None:
    # This is the warm-up interaction explicitly flagged during design
    # review: a factor score can itself be NaN AT a rebalance date (e.g.
    # momentum's lookback window isn't fully populated yet at the first
    # few month-ends). idx[3] IS a designated rebalance date, but its own
    # value is NaN -- this must resolve to 0 (flat), not to some
    # incorrectly-propagated stale value, and must keep reading as 0 all
    # the way through the next rebalance date (idx[7]), which has a
    # genuine value.
    idx = pd.bdate_range("2020-01-01", periods=10)
    daily_signal = pd.DataFrame(
        {"A": [np.nan, np.nan, np.nan, np.nan, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]}, index=idx
    )
    rebalance_dates = pd.DatetimeIndex([idx[3], idx[7]])  # idx[3]'s own value is NaN (warm-up)

    result = hold_until_next_rebalance(daily_signal, rebalance_dates)

    # idx[4..6]'s 0.5 sits on non-rebalance dates and must be ignored, same
    # as the previous test -- the only real decision before idx[7] is
    # idx[3]'s NaN, which must read as flat (0.0) through idx[6].
    expected = pd.Series(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], index=idx, name="A"
    )
    pd.testing.assert_series_equal(result["A"], expected)


def test_hold_until_next_rebalance_is_zero_before_first_rebalance_date() -> None:
    idx = pd.bdate_range("2020-01-01", periods=5)
    daily_signal = pd.DataFrame({"A": [np.nan, np.nan, 1.0, np.nan, np.nan]}, index=idx)
    rebalance_dates = pd.DatetimeIndex([idx[2]])

    result = hold_until_next_rebalance(daily_signal, rebalance_dates)

    expected = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0], index=idx, name="A")
    pd.testing.assert_series_equal(result["A"], expected)
