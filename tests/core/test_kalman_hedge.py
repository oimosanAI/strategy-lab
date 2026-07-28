"""Tests for strategies.pairs_trading.kalman_hedge.

Fixes the public API surface:

- KalmanHyperparameters(initial_state, initial_covariance, process_noise,
  observation_noise).
- calibrate_kalman_hyperparameters(prices, in_sample, ticker_dependent,
  ticker_independent, *, delta=1e-5) -> KalmanHyperparameters. Matches the
  (prices, in_sample) -> object contract verified by
  core.backtest.sample_split.assert_selection_ignores_out_of_sample
  (bind ticker_dependent/ticker_independent/delta with functools.partial).
- kalman_hedge_ratio(prices, hyperparams, ticker_dependent,
  ticker_independent) -> HedgeRatioSeries: the sequential state
  recursion, run continuously across whatever `prices` panel it is given.
  Its time-causality (state(t) depends only on prices <= t) is verified
  via core.backtest.engine.assert_causal through the _KalmanHedgeAsStrategy
  adapter below -- a DIFFERENT, orthogonal guard from the one above (see
  core.backtest.sample_split's module docstring on why these are two
  separate concerns).

Two levels of verification, matching the project's "trust well-tested
libraries, verify wiring/logic" pattern used throughout core.evaluation:
- The single Kalman UPDATE STEP is verified by an exact, independently
  hand-computed value (matrix arithmetic small enough to check by hand).
- Multi-step tracking behaviour is verified qualitatively (does the
  filter's estimate move in the correct direction, does a near-zero
  process noise keep it near-constant) rather than by hand-deriving a
  20+ step recursion, which is not a reasonable "by hand" computation.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from core.backtest.engine import LookAheadError, assert_causal
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, assert_selection_ignores_out_of_sample
from strategies.pairs_trading.kalman_hedge import (
    KalmanHyperparameters,
    calibrate_kalman_hyperparameters,
    kalman_hedge_ratio,
)
from tests.core.test_engine import _FullSampleOLSHedgeStrategy


# ---------------------------------------------------------------------------
# 2. kalman_hedge_ratio: single-step exact hand computation
# ---------------------------------------------------------------------------


def test_kalman_filter_single_step_hand_computable() -> None:
    # Arrange: one observation, x(0)=2, y(0)=3; initial_state=(alpha=0,
    # beta=1), P0=diag(1,1), Q=0 (no drift, isolates a single clean
    # update), R=1.
    #
    # Hand computation:
    #   predict: P = P0 + Q = diag(1, 1)
    #   H = [1, x(0)] = [1, 2]
    #   y_pred = H @ state = 1*0 + 2*1 = 2
    #   innovation = y(0) - y_pred = 3 - 2 = 1
    #   S = H @ P @ H.T + R = (1*1*1 + 2*1*2) + 1 = 5 + 1 = 6
    #   K = (P @ H.T) / S = [1, 2] / 6 = [1/6, 1/3]
    #   new_state = state + K * innovation = [0, 1] + [1/6, 1/3] = [1/6, 4/3]
    idx = pd.bdate_range("2020-01-01", periods=1)
    prices = pd.DataFrame({"X": [2.0], "Y": [3.0]}, index=idx)
    hyperparams = KalmanHyperparameters(
        initial_state=(0.0, 1.0),
        initial_covariance=(1.0, 1.0),
        process_noise=(0.0, 0.0),
        observation_noise=1.0,
    )

    # Act
    result = kalman_hedge_ratio(prices, hyperparams, "Y", "X")

    # Assert
    assert result.intercept.iloc[0] == pytest.approx(1.0 / 6.0)
    assert result.ratio.iloc[0] == pytest.approx(1.0 + 1.0 / 3.0)


# ---------------------------------------------------------------------------
# 2 (continued). Multi-step qualitative tracking behaviour
# ---------------------------------------------------------------------------


def _time_varying_beta_pair(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    true_beta = 1.0 + 0.5 * np.arange(n) / n  # drifts 1.0 -> 1.5
    noise = rng.normal(0, 0.1, n)
    y = true_beta * x + noise
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"X": x, "Y": y}, index=idx), true_beta


def test_kalman_filter_tracks_slowly_drifting_beta() -> None:
    # Arrange
    prices, true_beta = _time_varying_beta_pair()
    hyperparams = KalmanHyperparameters(
        initial_state=(0.0, 1.0),
        initial_covariance=(1.0, 1.0),
        process_noise=(1e-6, 1e-4),
        observation_noise=0.01,
    )

    # Act
    result = kalman_hedge_ratio(prices, hyperparams, "Y", "X")

    # Assert: the estimate moves in the correct direction and ends closer
    # to the true final value than to the true initial value.
    estimated_early = result.ratio.iloc[20]
    estimated_late = result.ratio.iloc[-1]
    assert estimated_late > estimated_early
    assert abs(estimated_late - true_beta[-1]) < 0.3


def test_kalman_filter_beta_stays_near_constant_when_process_noise_is_tiny() -> None:
    # Arrange: a GENUINELY constant true beta (no drift at all) -- NOT the
    # drifting-beta fixture used above. Discovered during development:
    # reusing the drifting fixture here gave a false failure, because with
    # x centered far from zero (~100) and a true zero intercept, alpha and
    # beta become nearly collinear (a ridge in the likelihood): beta can
    # drift along that ridge even with tiny Q while alpha absorbs the
    # compensating change and the fitted value (alpha + beta*x) stays
    # accurate. That is a real property of this state-space parameterization
    # under near-collinearity, not a bug in kalman_hedge_ratio -- see the
    # conversation record for the diagnostic trace that confirmed it. A
    # constant-truth fixture sidesteps the ambiguity entirely: there is no
    # persistent one-directional pressure pushing the state off-truth, so
    # a tiny Q should keep the estimate flat, unambiguously.
    n = 300
    rng = np.random.default_rng(1)
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    true_beta_constant = 1.2
    noise = rng.normal(0, 0.1, n)
    y = true_beta_constant * x + noise
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"X": x, "Y": y}, index=idx)
    hyperparams = KalmanHyperparameters(
        initial_state=(0.0, 1.0),
        initial_covariance=(1.0, 1.0),
        process_noise=(1e-10, 1e-10),
        observation_noise=0.01,
    )

    # Act
    result = kalman_hedge_ratio(prices, hyperparams, "Y", "X")

    # Assert: after a short convergence window, essentially no drift is
    # allowed -> the estimate stays flat near the true constant value.
    converged = result.ratio.iloc[50:]
    assert converged.std() < 0.05
    assert converged.mean() == pytest.approx(true_beta_constant, abs=0.05)


# ---------------------------------------------------------------------------
# 4. kalman_hedge_ratio: time-causality via assert_causal
# ---------------------------------------------------------------------------


class _KalmanHedgeAsStrategy:
    """Adapts kalman_hedge_ratio into the Strategy protocol so the
    existing assert_causal can be reused unchanged, mirroring
    position_sizing's _SizerAsStrategy test adapter."""

    name = "kalman-hedge-under-test"

    def __init__(self, hyperparams: KalmanHyperparameters, ticker_dependent: str, ticker_independent: str) -> None:
        self._hyperparams = hyperparams
        self._dep = ticker_dependent
        self._indep = ticker_independent

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        result = kalman_hedge_ratio(prices, self._hyperparams, self._dep, self._indep)
        return pd.DataFrame({self._dep: result.ratio}, index=prices.index)


def test_kalman_hedge_ratio_passes_assert_causal() -> None:
    # Arrange
    prices, _ = _time_varying_beta_pair(n=120, seed=5)
    hyperparams = KalmanHyperparameters(
        initial_state=(0.0, 1.0), initial_covariance=(1.0, 1.0), process_noise=(1e-6, 1e-5), observation_noise=0.01
    )

    # Act / Assert: no exception -- the recursion is causal by construction.
    assert_causal(_KalmanHedgeAsStrategy(hyperparams, "Y", "X"), prices, n_trials=12)


def test_full_sample_hedge_strategy_is_still_rejected_by_assert_causal() -> None:
    # Reuses the ALREADY-PROVEN-BROKEN negative control from Phase 1
    # (test_engine.py) rather than inventing a new one: fitting a hedge
    # ratio over the whole sample is exactly the same class of bug here.
    prices, _ = _time_varying_beta_pair(n=120, seed=6)
    prices = prices.rename(columns={"X": "AAPL", "Y": "MSFT"})

    with pytest.raises(LookAheadError):
        assert_causal(_FullSampleOLSHedgeStrategy(), prices, n_trials=12)


# ---------------------------------------------------------------------------
# 3. calibrate_kalman_hyperparameters: IS/OOS separation contract
# ---------------------------------------------------------------------------


def _leaking_calibrate(
    prices: pd.DataFrame, in_sample: SamplePeriod, ticker_dependent: str, ticker_independent: str, delta: float = 1e-5
) -> KalmanHyperparameters:
    """Negative control: ignores `in_sample`, always calibrates from the
    full panel -- the same 'forgot to respect the split' bug as
    _leaking_select_pairs in test_cointegration.py."""
    full_period = SamplePeriod(prices.index[0], prices.index[-1])
    return calibrate_kalman_hyperparameters(prices, full_period, ticker_dependent, ticker_independent, delta=delta)


def test_calibrate_kalman_hyperparameters_passes_assert_selection_ignores_oos() -> None:
    # Arrange
    prices, _ = _time_varying_beta_pair(n=260, seed=7)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[199]),
        out_of_sample=SamplePeriod(prices.index[200], prices.index[259]),
    )
    bound = functools.partial(calibrate_kalman_hyperparameters, ticker_dependent="Y", ticker_independent="X")

    # Act / Assert
    assert_selection_ignores_out_of_sample(bound, prices, split)


def test_leaking_calibrate_is_caught_by_assert_selection_ignores_oos() -> None:
    # Arrange
    prices, _ = _time_varying_beta_pair(n=260, seed=8)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[199]),
        out_of_sample=SamplePeriod(prices.index[200], prices.index[259]),
    )
    bound_leaking = functools.partial(_leaking_calibrate, ticker_dependent="Y", ticker_independent="X")

    # Act / Assert
    with pytest.raises(LookAheadError):
        assert_selection_ignores_out_of_sample(bound_leaking, prices, split)


# ---------------------------------------------------------------------------
# 5. calibrate_kalman_hyperparameters: Q/R derivation (wiring check)
# ---------------------------------------------------------------------------


def test_calibrate_kalman_hyperparameters_matches_independent_ols_fit() -> None:
    # Arrange: an independent OLS fit, called directly in the test --
    # a wiring check (like benchmark_comparison / one_sample_ttest
    # elsewhere in this project), not a re-derivation of OLS itself.
    prices, _ = _time_varying_beta_pair(n=100, seed=9)
    in_sample = SamplePeriod(prices.index[0], prices.index[-1])
    delta = 1e-5

    predictors = sm.add_constant(prices["X"])
    expected = sm.OLS(prices["Y"], predictors).fit()
    expected_alpha = float(expected.params.iloc[0])
    expected_beta = float(expected.params.iloc[1])
    expected_se_alpha = float(expected.bse.iloc[0])
    expected_se_beta = float(expected.bse.iloc[1])
    expected_r = float(expected.mse_resid)
    expected_q = expected_r * delta / (1 - delta)

    # Act
    result = calibrate_kalman_hyperparameters(prices, in_sample, "Y", "X", delta=delta)

    # Assert
    assert result.initial_state == pytest.approx((expected_alpha, expected_beta))
    assert result.initial_covariance == pytest.approx((expected_se_alpha**2, expected_se_beta**2))
    assert result.observation_noise == pytest.approx(expected_r)
    assert result.process_noise == pytest.approx((expected_q, expected_q))
