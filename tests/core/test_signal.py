"""Tests for strategies.pairs_trading.signal.

Fixes the public API surface:

- compute_spread(hedge, prices) -> pd.Series: spread = y - alpha - beta*x.
- rolling_zscore(spread, window) -> pd.Series.
- zscore_window_from_half_life(half_life, multiplier, min_window,
  max_window) -> int.
- EntryExitResult(position_state, forced_exit_dates) /
  generate_entry_exit_positions(zscore, entry_threshold, exit_threshold,
  max_holding_periods=None) -> EntryExitResult: a forward-only state
  machine (causal by construction: state(t) depends only on
  zscore(t) and state(t-1)).
"""

from __future__ import annotations

import pandas as pd
import pytest

from strategies.pairs_trading.signal import (
    EntryExitResult,
    compute_spread,
    generate_entry_exit_positions,
    rolling_zscore,
    zscore_window_from_half_life,
)
from strategies.pairs_trading.static_hedge import HedgeRatioSeries


def _z_series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


# ---------------------------------------------------------------------------
# 1. compute_spread / rolling_zscore
# ---------------------------------------------------------------------------


def test_compute_spread_hand_computable() -> None:
    # Arrange
    hedge = HedgeRatioSeries(
        ticker_dependent="Y",
        ticker_independent="X",
        intercept=pd.Series([1.0, 1.0, 1.0]),
        ratio=pd.Series([2.0, 2.0, 2.0]),
    )
    prices = pd.DataFrame({"Y": [10.0, 12.0, 14.0], "X": [3.0, 4.0, 5.0]})

    # Act
    spread = compute_spread(hedge, prices)

    # Assert: spread = Y - intercept - ratio*X = [10-1-6, 12-1-8, 14-1-10]
    assert spread.tolist() == pytest.approx([3.0, 3.0, 3.0])


def test_rolling_zscore_hand_computable() -> None:
    # Arrange
    spread = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    # Act
    z = rolling_zscore(spread, window=3)

    # Assert: window [1,2,3] -> mean=2, std(ddof=1)=1 -> z=(3-2)/1=1.0
    assert z.iloc[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. zscore_window_from_half_life
# ---------------------------------------------------------------------------


def test_window_applies_multiplier() -> None:
    assert zscore_window_from_half_life(5.0, multiplier=4.0, min_window=10, max_window=90) == 20


def test_window_clamps_to_min() -> None:
    assert zscore_window_from_half_life(1.0, multiplier=4.0, min_window=10, max_window=90) == 10


def test_window_clamps_to_max() -> None:
    assert zscore_window_from_half_life(100.0, multiplier=4.0, min_window=10, max_window=90) == 90


# ---------------------------------------------------------------------------
# 3. generate_entry_exit_positions: state machine
# ---------------------------------------------------------------------------


def test_enters_long_when_zscore_very_negative() -> None:
    z = _z_series([0.0, -2.5])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0)
    assert isinstance(result, EntryExitResult)
    assert result.position_state.tolist() == [0, 1]


def test_enters_short_when_zscore_very_positive() -> None:
    z = _z_series([0.0, 2.5])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0)
    assert result.position_state.tolist() == [0, -1]


def test_holds_existing_position_when_no_trigger() -> None:
    z = _z_series([-2.5, -1.0, -0.5])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0)
    assert result.position_state.tolist() == [1, 1, 1]


def test_exits_when_zscore_reverts_to_threshold() -> None:
    z = _z_series([-2.5, -1.0, 0.0])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0)
    assert result.position_state.tolist() == [1, 1, 0]


def test_does_not_flip_directly_while_in_position() -> None:
    # Long from t0 (z=-2.5); z jumps straight past +entry_threshold at t1
    # without passing through the exit zone -- only exit conditions are
    # evaluated while in a position, so it stays long instead of flipping.
    z = _z_series([-2.5, 2.5])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0)
    assert result.position_state.tolist() == [1, 1]


# ---------------------------------------------------------------------------
# 4. max_holding_periods: forced exit + recorded trigger dates
# ---------------------------------------------------------------------------


def test_forced_exit_triggers_after_max_holding_periods() -> None:
    # Arrange: entry at t0 (z=-2.5), never reverts to 0.
    z = _z_series([-2.5, -2.0, -1.8, -1.9])

    # Act
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0, max_holding_periods=2)

    # Assert
    assert result.position_state.tolist() == [1, 1, 0, 0]
    assert result.forced_exit_dates == [z.index[2]]


def test_no_forced_exit_when_natural_exit_happens_first() -> None:
    z = _z_series([-2.5, 0.0])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0, max_holding_periods=5)
    assert result.forced_exit_dates == []


def test_no_forced_exit_when_max_holding_periods_is_none() -> None:
    z = _z_series([-2.5, -2.0, -1.8, -1.9, -2.0, -1.5])
    result = generate_entry_exit_positions(z, entry_threshold=2.0, exit_threshold=0.0, max_holding_periods=None)
    assert result.forced_exit_dates == []
    assert result.position_state.tolist() == [1, 1, 1, 1, 1, 1]
