"""In-sample / out-of-sample separation for pair-selection and static
parameter estimation.

REQUIREMENTS.md 4.1 requires that pair selection (cointegration tests) and
static parameter estimation (e.g. an OLS hedge ratio) be done using
in-sample data only, then validated out-of-sample. This module provides
the structural machinery for that separation, deliberately mirroring
core.backtest.engine's causality guards -- one level up the stack:

- core.backtest.engine.assert_causal guards TIME causality: a strategy's
  signal at t must not depend on prices after t. It applies to the
  ONGOING, continuously-running signal/z-score/Kalman-state computation.
- assert_selection_ignores_out_of_sample (this module) guards the IS/OOS
  PARTITION boundary: a ONE-TIME selection or calibration decision (which
  pair, what static hedge ratio, what Kalman hyperparameters) must be
  made using in-sample data only, never re-tuned by peeking at
  out-of-sample data.

These are two ORTHOGONAL guards. A Kalman filter's hyperparameters must
clear this module's guard (chosen from in-sample data only); its actual
sequential state recursion runs continuously through in-sample AND
out-of-sample data and is instead covered by assert_causal/
assert_backtest_causal -- running it continuously is correct, not a
violation of anything this module checks.

IMPORTANT: run_backtest should be called ONCE, continuously, over the
full in-sample+out-of-sample price panel -- NOT separately per period.
Running it only on an out-of-sample slice restarts every rolling window
(z-score, realized volatility, ...) from scratch at the OOS boundary,
creating an artificial warm-up gap that does not reflect how a strategy
would actually be run live (continuing to hold and trade a position from
in-sample into out-of-sample). split_returns() below slices the
CONTINUOUS result's returns after the fact for separate Pillar-3 (IS vs
OOS) scoring; see
tests/core/test_sample_split.py::test_continuous_execution_avoids_warmup_gap_that_oos_only_execution_has
for a concrete demonstration of why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.backtest.engine import LookAheadError


@dataclass(frozen=True)
class SamplePeriod:
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class TrainTestSplit:
    """A single in-sample / out-of-sample boundary.

    ``out_of_sample.start`` must be strictly after ``in_sample.end`` --
    out-of-sample is definitionally later in time than in-sample. A gap
    between the two (e.g. an embargo period) is allowed and requires no
    change here: just construct ``out_of_sample`` further from
    ``in_sample.end``.
    """

    in_sample: SamplePeriod
    out_of_sample: SamplePeriod

    def __post_init__(self) -> None:
        if self.out_of_sample.start <= self.in_sample.end:
            raise ValueError("out_of_sample must start strictly after in_sample ends")


@dataclass(frozen=True)
class InSampleData:
    """The recommended shape for pair-selection / static-parameter-
    estimation functions to work with internally, obtained via
    slice_in_sample(). A type alone cannot stop a function from ignoring
    it and reaching for the full panel instead -- that is exactly the
    class of bug assert_selection_ignores_out_of_sample below is designed
    to catch by perturbation, not by trusting that a function which
    "should" use InSampleData actually does.
    """

    prices: pd.DataFrame
    period: SamplePeriod


def slice_in_sample(prices: pd.DataFrame, period: SamplePeriod) -> InSampleData:
    """The sanctioned way to obtain an InSampleData: slices strictly to
    [period.start, period.end]. A correct selection_fn (see
    assert_selection_ignores_out_of_sample) typically calls this
    internally to restrict itself to the in-sample period."""
    sliced = prices.loc[period.start : period.end]
    return InSampleData(prices=sliced, period=period)


def split_returns(returns: pd.Series, split: TrainTestSplit) -> tuple[pd.Series, pd.Series]:
    """Slice a CONTINUOUSLY-run return series into (in_sample,
    out_of_sample) pieces, for separate Pillar-3 (IS vs OOS) scoring.

    See the module docstring for why run_backtest should be run once,
    continuously, over the full period rather than separately per period.
    """
    is_returns = returns.loc[split.in_sample.start : split.in_sample.end]
    oos_returns = returns.loc[split.out_of_sample.start : split.out_of_sample.end]
    return is_returns, oos_returns


def assert_selection_ignores_out_of_sample(
    selection_fn: Callable[[pd.DataFrame, SamplePeriod], object],
    prices: pd.DataFrame,
    split: TrainTestSplit,
    *,
    perturbation: float = 0.10,
    seed: int = 0,
) -> None:
    """Prove, by perturbation, that ``selection_fn`` ignores out-of-sample
    data.

    ``selection_fn`` receives the FULL (possibly perturbed) price panel
    plus the in-sample period boundary -- deliberately mirroring how
    ``core.backtest.engine.assert_causal`` hands a Strategy the full price
    panel and verifies, by perturbation, that it behaves causally rather
    than pre-slicing on its behalf. A correct ``selection_fn`` restricts
    itself to ``in_sample`` internally (typically via ``slice_in_sample``);
    that restriction is verified here, not assumed or enforced by having
    this function do the slicing for it -- pre-slicing here would make
    every selection_fn pass trivially regardless of whether it actually
    respects the boundary, defeating the point of the test (this was
    caught during development: see the negative control in
    tests/core/test_sample_split.py, which only has real detection power
    under this "full panel in" calling convention).

    Perturbs only the out-of-sample portion of ``prices`` and asserts
    ``selection_fn``'s result is unchanged. Reuses LookAheadError:
    out-of-sample data is definitionally later in time than in-sample data
    (enforced by TrainTestSplit itself), so a selector affected by
    out-of-sample prices is exhibiting a form of look-ahead bias, one
    level up from single-day time causality.

    Raises
    ------
    LookAheadError
        If perturbing only the out-of-sample prices changes
        ``selection_fn``'s result.
    """
    baseline = selection_fn(prices, split.in_sample)

    shocked = prices.copy()
    oos_mask = (shocked.index >= split.out_of_sample.start) & (shocked.index <= split.out_of_sample.end)
    rng = np.random.default_rng(seed)
    oos_values = shocked.loc[oos_mask]
    multipliers = 1.0 + rng.uniform(-perturbation, perturbation, size=oos_values.shape)
    shocked.loc[oos_mask] = oos_values.to_numpy() * multipliers

    perturbed = selection_fn(shocked, split.in_sample)

    if baseline != perturbed:
        raise LookAheadError(
            "Out-of-sample leakage detected: perturbing prices strictly "
            f"after in_sample.end ({split.in_sample.end}) changed the "
            "result of selection_fn, which should only ever depend on "
            "in-sample data."
        )


def generate_anchored_windows(
    index: pd.DatetimeIndex, is_end_boundaries: Sequence[pd.Timestamp]
) -> list[TrainTestSplit]:
    """Anchored (expanding in-sample) walk-forward windows from explicit
    IS-end boundary dates.

    In-sample always starts at ``index[0]``; window i's in-sample end is
    ``is_end_boundaries[i]`` snapped to the latest date ACTUALLY PRESENT
    in ``index`` at or before it -- the panel's real index is the source
    of truth, not a theoretical calendar boundary that might fall on a
    holiday/gap (the same reasoning already used for month-end detection
    elsewhere in this project). Window i's out-of-sample period runs from
    the day after its in-sample end to the NEXT window's (snapped)
    in-sample end, or to ``index[-1]`` for the final window.

    Deliberately independent of any "monthly"/"quarterly" cadence concept
    (unlike strategies.factor_momentum.rebalance.month_end_dates): a
    walk-forward window boundary is a Pillar-3 evaluation concern, not a
    trading rebalance concern, so the caller supplies whatever boundaries
    are appropriate rather than this function inventing a stepping
    scheme.

    Raises
    ------
    ValueError
        If ``is_end_boundaries`` is empty, not strictly increasing, falls
        before ``index[0]``, or snaps to ``index[-1]`` (leaving no room
        for an out-of-sample period).
    """
    if not is_end_boundaries:
        raise ValueError("is_end_boundaries must not be empty")
    for prev, curr in zip(is_end_boundaries, is_end_boundaries[1:]):
        if curr <= prev:
            raise ValueError("is_end_boundaries must be strictly increasing")

    snapped: list[pd.Timestamp] = []
    for boundary in is_end_boundaries:
        eligible = index[index <= boundary]
        if len(eligible) == 0:
            raise ValueError(f"boundary {boundary} is before the earliest available date {index[0]}")
        snapped_date = eligible[-1]
        if snapped_date == index[-1]:
            raise ValueError(
                f"boundary {boundary} snaps to the last available date {index[-1]}, "
                "leaving no room for an out-of-sample period"
            )
        snapped.append(snapped_date)

    windows: list[TrainTestSplit] = []
    for i, is_end in enumerate(snapped):
        oos_start = index[index > is_end][0]
        oos_end = snapped[i + 1] if i + 1 < len(snapped) else index[-1]
        windows.append(
            TrainTestSplit(
                in_sample=SamplePeriod(index[0], is_end),
                out_of_sample=SamplePeriod(oos_start, oos_end),
            )
        )
    return windows
