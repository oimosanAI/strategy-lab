"""Tests for strategies.factor_momentum.ranking.

Fixes the public API surface:

- FactorMomentumConfig(momentum_lookback_days, momentum_skip_days,
  low_vol_window, long_percentile=0.80, short_percentile=0.20).
- build_long_short_signal(momentum, low_vol, config) -> pd.DataFrame:
  combines momentum (higher=better) and low_vol (lower=better) into a
  single cross-sectional composite percentile per date
  (rank(axis=1, pct=True), scale-invariant since momentum and low_vol
  are on incomparable raw scales), buckets into long (composite >=
  long_percentile) / short (composite <= short_percentile) / neutral,
  then NORMALIZES each leg to sum to 1.0 (long) / -1.0 (short) --
  equal-weight within the leg, 0.0 for either leg on a day with no
  qualifying candidates. Added after the full-universe E2E run showed
  an un-normalized +-1 signal lets portfolio-level gross exposure scale
  with however many names happen to be in a quintile (56x was observed
  with ~100 names/leg) -- see ranking.py's module docstring.
  NEVER emits NaN -- any ticker with a NaN momentum or low_vol score is
  excluded from ranking (via pandas rank(pct=True)'s own NaN handling,
  which shrinks the percentile denominator rather than including a
  phantom middle-of-pack value) and explicitly assigned 0.0, not left as
  NaN. This matters downstream: rebalance.hold_until_next_rebalance
  treats a NaN value AT a rebalance date as "no decision, keep holding
  the previous one" -- if this module ever leaked NaN through, a ticker
  temporarily missing data would incorrectly keep last month's stale
  position instead of flattening to 0.

Expected composite values below are derived independently by hand from
the rank(pct=True) definition (rank / count of non-null values in that
row), not by running the implementation and copying its output. Only the
final discretized/normalized signal is asserted, not the raw composite
float, since none of the hand-picked composites in these fixtures land
near a threshold boundary -- the comparison is robust to any
floating-point noise in the intermediate rank arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.factor_momentum.ranking import FactorMomentumConfig, build_long_short_signal


def test_build_long_short_signal_assigns_quintile_buckets_correctly() -> None:
    # Arrange: 10 tickers, single date. momentum strictly increasing with
    # ticker index; low_vol strictly DECREASING with ticker index, so the
    # inversion of low_vol's rank must be exercised correctly (if the
    # inversion were missing, momentum_rank + low_vol_rank would be a
    # CONSTANT 1.1 for every ticker, and nothing would ever cross either
    # threshold -- this fixture would catch that bug).
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"T{i}" for i in range(10)]
    momentum = pd.DataFrame([[0.01 * (i + 1) for i in range(10)]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.30 - 0.02 * i for i in range(10)]], index=idx, columns=tickers)
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20)

    # Act
    result = build_long_short_signal(momentum, low_vol, config)

    # Assert: composite(T_i) = ((i+1)/10 + i/10) / 2 = (2i+1)/20 (hand-derived)
    # -> long: T8 (0.85), T9 (0.95) => n_long=2, each weight=1/2=0.5;
    # short: T0 (0.05), T1 (0.15) => n_short=2, each weight=-1/2=-0.5;
    # neutral: T2..T7.
    expected = pd.DataFrame(
        [[-0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]], index=idx, columns=tickers
    )
    pd.testing.assert_frame_equal(result, expected)


def test_build_long_short_signal_normalizes_each_leg_to_sum_to_one() -> None:
    # The equal-weight-within-leg contract, asserted directly rather than
    # only incidentally implied by the discrete bucket test above.
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"T{i}" for i in range(10)]
    momentum = pd.DataFrame([[0.01 * (i + 1) for i in range(10)]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.30 - 0.02 * i for i in range(10)]], index=idx, columns=tickers)
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20)

    result = build_long_short_signal(momentum, low_vol, config)

    long_leg_sum = result.where(result > 0, 0.0).sum(axis=1)
    short_leg_sum = result.where(result < 0, 0.0).sum(axis=1)
    assert long_leg_sum.iloc[0] == pytest.approx(1.0)
    assert short_leg_sum.iloc[0] == pytest.approx(-1.0)


def test_build_long_short_signal_leg_is_zero_when_no_candidates_qualify() -> None:
    # A day where every score is NaN (e.g. the whole universe still in
    # warm-up) must produce an all-flat row -- 0.0, not a division-by-zero
    # artifact -- for both legs.
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = ["A", "B", "C"]
    momentum = pd.DataFrame([[np.nan, np.nan, np.nan]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.2, 0.3, 0.4]], index=idx, columns=tickers)
    config = FactorMomentumConfig()

    result = build_long_short_signal(momentum, low_vol, config)

    assert (result.iloc[0] == 0.0).all()


def test_build_long_short_signal_excludes_nan_score_from_ranking_pool() -> None:
    # Arrange: 5 tickers, T2 has a NaN momentum score. The ranking among
    # the remaining 4 tickers must be computed as if T2 did not exist
    # (percentile denominator = 4, not 5) -- hand-derived below.
    idx = pd.bdate_range("2020-01-01", periods=1)
    tickers = ["T0", "T1", "T2", "T3", "T4"]
    momentum = pd.DataFrame([[0.01, 0.02, np.nan, 0.04, 0.05]], index=idx, columns=tickers)
    low_vol = pd.DataFrame([[0.30, 0.25, 0.10, 0.15, 0.05]], index=idx, columns=tickers)
    config = FactorMomentumConfig(long_percentile=0.80, short_percentile=0.20)

    # Act
    result = build_long_short_signal(momentum, low_vol, config)

    # Assert: among {T0,T1,T3,T4} (T2 excluded), momentum_rank(pct,/4) =
    # T0=0.25, T1=0.5, T3=0.75, T4=1.0; low_vol_rank(pct,/4) = T0=1.0
    # (highest vol), T1=0.75, T3=0.5, T4=0.25 (lowest vol) -> inverted =
    # T0=0.0, T1=0.25, T3=0.5, T4=0.75. composite = T0=0.125, T1=0.375,
    # T3=0.625, T4=0.875 -> T4 long (>=0.80), T0 short (<=0.20), T1/T3
    # neutral, T2 excluded (0.0, not NaN).
    expected = pd.DataFrame([[-1.0, 0.0, 0.0, 0.0, 1.0]], index=idx, columns=tickers)
    pd.testing.assert_frame_equal(result, expected)


def test_build_long_short_signal_never_emits_nan() -> None:
    # Contract test: regardless of how much NaN the inputs contain
    # (warm-up periods, data gaps), the output must always be a real
    # number (0.0/1.0/-1.0), never NaN -- rebalance.
    # hold_until_next_rebalance's correctness depends on this contract
    # (see module docstring).
    idx = pd.bdate_range("2020-01-01", periods=5)
    tickers = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    momentum = pd.DataFrame(rng.normal(size=(5, 4)), index=idx, columns=tickers)
    low_vol = pd.DataFrame(rng.uniform(0.1, 0.5, size=(5, 4)), index=idx, columns=tickers)
    momentum.iloc[0, :] = np.nan  # entire warm-up row
    low_vol.iloc[2, 1] = np.nan  # scattered data gap
    config = FactorMomentumConfig()

    result = build_long_short_signal(momentum, low_vol, config)

    assert not result.isna().any().any()
