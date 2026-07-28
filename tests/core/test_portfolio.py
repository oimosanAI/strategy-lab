"""Tests for core.backtest.portfolio.

Fixes the public API surface that core/backtest/portfolio.py must implement:

- compute_ticker_returns(position, open_, close) -> Series
  Decomposes each day's return into up to three components, per
  REQUIREMENTS.md 4.3 ("翌日始値約定"): a signal decided at close(t-1) is
  executed at open(t), so the day a position CHANGES needs a partial-day
  return, not a naive close-to-close figure.

    carried = same-sign overlap between position(t-1) and position(t)
              -> close-to-close (continuously held all day)
    new     = position(t) - carried
              -> open(t)-to-close(t) (only entered/added to at today's open)
    exited  = position(t-1) - carried
              -> close(t-1)-to-open(t) (held overnight, sold at today's open;
                 this is the piece a naive position(t)*bar_return formula
                 drops entirely, since it's no longer in position(t))

  day_return(t) = carried*(close/close[-1]-1) + new*(close/open-1)
                + exited*(open/close[-1]-1)

  The very first bar of any series passed in is always treated as coming
  from a flat (zero) book -- there is no bar -1 to reference.

- compute_turnover(positions: DataFrame) -> Series
  Portfolio-level turnover = SUM ACROSS TICKERS of |position(t) - position(t-1)|
  per ticker, per day. Deliberately NOT netted across tickers: a pairs trade
  that goes +0.3/-0.3 in the same bar is two real trades (turnover 0.6), not
  a netted no-op (turnover 0.0). Getting this wrong silently understates
  transaction costs for any long/short strategy.

Every fixture below uses deliberately mismatched open/close values (e.g. a
day-1 open of 999 when it shouldn't be referenced at all) specifically so
that a wrong basis (using close instead of open, or vice versa) produces an
obviously wrong number rather than coincidentally matching.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.backtest.portfolio import compute_ticker_returns, compute_turnover


def _series(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=index, dtype=float)


# ---------------------------------------------------------------------------
# compute_ticker_returns: partial-day / full-day switching
# ---------------------------------------------------------------------------


def test_pure_hold_uses_close_to_close() -> None:
    # Arrange: position unchanged across day0 -> day1; day1's open is a
    # deliberately absurd value (999) that must NOT be referenced at all.
    position = _series([1.0, 1.0])
    open_ = _series([100.0, 999.0])
    close = _series([100.0, 105.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: day1 is a continuously-held day -> close-to-close only.
    assert returns.iloc[1] == pytest.approx(105.0 / 100.0 - 1.0)


def test_entry_day_uses_open_to_close() -> None:
    # Arrange: flat on day0, a fresh long established on day1.
    position = _series([0.0, 1.0])
    open_ = _series([100.0, 102.0])
    close = _series([100.0, 110.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: entry day return is open(1)-to-close(1), NOT close(0)-to-close(1)
    # (110/100-1 = 0.10 would be the wrong-basis answer; 110/102-1 is correct).
    assert returns.iloc[1] == pytest.approx(110.0 / 102.0 - 1.0)


def test_exit_day_captures_overnight_gap() -> None:
    # Arrange: long on day0, fully closed out on day1. day1's close (999) is
    # deliberately absurd and must NOT be referenced -- once sold at the
    # open, the position no longer participates in the rest of the day.
    position = _series([1.0, 0.0])
    open_ = _series([100.0, 97.0])
    close = _series([100.0, 999.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: the exit still captures the close(0)->open(1) overnight gap;
    # it is not zero, and it is not based on close(1).
    assert returns.iloc[1] == pytest.approx(97.0 / 100.0 - 1.0)


def test_position_flip_decomposes_into_exited_and_new_portions() -> None:
    # Arrange: a direct long-to-short flip, skipping flat entirely.
    position = _series([1.0, -1.0])
    open_ = _series([100.0, 103.0])
    close = _series([100.0, 106.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: no overlap (signs differ) -> carried=0; the old long is
    # exited (close-to-open) and the new short is entered (open-to-close).
    expected = 1.0 * (103.0 / 100.0 - 1.0) + (-1.0) * (106.0 / 103.0 - 1.0)
    assert returns.iloc[1] == pytest.approx(expected)


def test_partial_increase_blends_carried_and_new() -> None:
    # Arrange: same-direction resize, 0.5 -> 1.0 (adding to a winning long).
    position = _series([0.5, 1.0])
    open_ = _series([100.0, 103.0])
    close = _series([100.0, 106.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: carried=0.5 (close-to-close), new=0.5 (open-to-close).
    expected = 0.5 * (106.0 / 100.0 - 1.0) + 0.5 * (106.0 / 103.0 - 1.0)
    assert returns.iloc[1] == pytest.approx(expected)


def test_partial_reduction_blends_carried_and_exited() -> None:
    # Arrange: same-direction resize, 1.0 -> 0.5 (trimming a long, not
    # flattening it). carried=0.5 stays held all day; exited=0.5 was sold
    # at today's open and only captures the overnight gap.
    position = _series([1.0, 0.5])
    open_ = _series([100.0, 103.0])
    close = _series([100.0, 106.0])

    # Act
    returns = compute_ticker_returns(position, open_, close)

    # Assert: NOT the same as a full hold (1.0 * 0.06 = 0.06) and NOT the
    # same as treating the trimmed half as close-to-close too.
    expected = 0.5 * (106.0 / 100.0 - 1.0) + 0.5 * (103.0 / 100.0 - 1.0)
    assert returns.iloc[1] == pytest.approx(expected)
    assert returns.iloc[1] != pytest.approx(1.0 * (106.0 / 100.0 - 1.0))


# ---------------------------------------------------------------------------
# compute_turnover: per-ticker absolute sum, never netted across tickers
# ---------------------------------------------------------------------------


def test_turnover_sums_absolute_per_ticker_changes_not_netted() -> None:
    # Arrange: a classic pairs-trade rebalance in one bar -- long one leg,
    # short the other, in equal size. Net portfolio weight change is ~0,
    # but two real trades happened.
    positions = pd.DataFrame(
        {"A": [0.0, 0.3], "B": [0.0, -0.3]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )

    # Act
    turnover = compute_turnover(positions)

    # Assert: 0.6, not the netted (and wrong) 0.0.
    assert turnover.iloc[1] == pytest.approx(0.6)


def test_turnover_at_first_bar_from_flat_book() -> None:
    # Arrange: the very first bar, entering from an implicit flat book.
    positions = pd.DataFrame(
        {"A": [0.5], "B": [-0.2]},
        index=pd.bdate_range("2020-01-01", periods=1),
    )

    # Act
    turnover = compute_turnover(positions)

    # Assert
    assert turnover.iloc[0] == pytest.approx(0.7)


def test_turnover_zero_when_positions_unchanged() -> None:
    # Arrange
    positions = pd.DataFrame(
        {"A": [0.4, 0.4], "B": [-0.1, -0.1]},
        index=pd.bdate_range("2020-01-01", periods=2),
    )

    # Act
    turnover = compute_turnover(positions)

    # Assert
    assert turnover.iloc[1] == pytest.approx(0.0)
