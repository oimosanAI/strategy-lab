"""Dynamic hedge ratio via a Kalman filter (REQUIREMENTS.md 5.1).

State-space model (a 2-state random-walk Kalman filter, the standard
pairs-trading formulation):

    Observation: y(t) = alpha(t) + beta(t) * x(t) + eps(t),  eps ~ N(0, R)
    State:       [alpha(t), beta(t)]' = [alpha(t-1), beta(t-1)]' + eta(t),
                 eta ~ N(0, Q)

Implemented by hand (not via statsmodels.tsa.statespace) for two reasons:
this repo's own convention of self-implementing the numerically-central
pieces (REQUIREMENTS.md 2: "自前実装"), and because a hand-rolled 2-state
recursion is simple enough to verify with an exact single-step
hand-computation (see tests/core/test_kalman_hedge.py) and to wrap for
core.backtest.engine.assert_causal -- a black-box statespace framework
would obscure both.

Two ORTHOGONAL guards apply here, exactly as for cointegration.py's pair
selection (see core.backtest.sample_split's module docstring):

- calibrate_kalman_hyperparameters (this module) chooses Q, R, and the
  initial state/covariance from IN-SAMPLE data only -- a ONE-TIME
  calibration decision, verified by
  core.backtest.sample_split.assert_selection_ignores_out_of_sample.
- kalman_hedge_ratio's sequential state recursion runs CONTINUOUSLY
  through in-sample AND out-of-sample data using those FIXED
  hyperparameters -- this is correct, not a violation of anything the
  IS/OOS guard checks. Its TIME causality (state(t) depends only on
  prices <= t) is instead verified by
  core.backtest.engine.assert_causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.backtest.sample_split import SamplePeriod, slice_in_sample
from strategies.pairs_trading.static_hedge import HedgeRatioSeries


@dataclass(frozen=True)
class KalmanHyperparameters:
    initial_state: tuple[float, float]  # (alpha_0, beta_0)
    initial_covariance: tuple[float, float]  # diagonal (var_alpha_0, var_beta_0)
    process_noise: tuple[float, float]  # diagonal Q (q_alpha, q_beta)
    observation_noise: float  # R


def calibrate_kalman_hyperparameters(
    prices: pd.DataFrame,
    in_sample: SamplePeriod,
    ticker_dependent: str,
    ticker_independent: str,
    *,
    delta: float = 1e-5,
) -> KalmanHyperparameters:
    """Calibrate Q, R, and the initial state/covariance from in-sample
    data only.

    Matches the (prices, in_sample) -> object contract verified by
    core.backtest.sample_split.assert_selection_ignores_out_of_sample:
    slices to in-sample data via slice_in_sample before computing
    anything. ``ticker_dependent``/``ticker_independent``/``delta`` are
    time-independent extra arguments; bind them with functools.partial
    when verifying the IS/OOS contract.

    The initial state/covariance come from an in-sample OLS fit (alpha_0,
    beta_0 from the fitted coefficients; the initial covariance from
    their standard errors -- a principled reflection of genuine estimation
    uncertainty, not an arbitrary constant). R is the OLS residual
    variance. Q follows the common heuristic ``Q = R * delta / (1 - delta)``
    for a small ``delta`` (Chan's textbook convention): a small fraction
    of the observation noise, controlling how quickly the state is allowed
    to drift per period.
    """
    data = slice_in_sample(prices, in_sample)
    dependent = data.prices[ticker_dependent]
    independent = data.prices[ticker_independent]

    predictors = sm.add_constant(independent)
    model = sm.OLS(dependent, predictors).fit()

    alpha_0 = float(model.params.iloc[0])
    beta_0 = float(model.params.iloc[1])
    se_alpha = float(model.bse.iloc[0])
    se_beta = float(model.bse.iloc[1])
    r = float(model.mse_resid)
    q = r * delta / (1 - delta)

    return KalmanHyperparameters(
        initial_state=(alpha_0, beta_0),
        initial_covariance=(se_alpha**2, se_beta**2),
        process_noise=(q, q),
        observation_noise=r,
    )


def kalman_hedge_ratio(
    prices: pd.DataFrame,
    hyperparams: KalmanHyperparameters,
    ticker_dependent: str,
    ticker_independent: str,
) -> HedgeRatioSeries:
    """Run the Kalman filter recursion continuously across the full
    ``prices`` index (in-sample AND out-of-sample) using FIXED
    hyperparameters calibrated from in-sample data only.

    Causal by construction: state(t) is a function of observations
    <= t only. Verified by core.backtest.engine.assert_causal /
    assert_backtest_causal via the _KalmanHedgeAsStrategy test adapter,
    not by core.backtest.sample_split.assert_selection_ignores_out_of_sample
    -- that guard covers calibrate_kalman_hyperparameters above, an
    orthogonal, one-time decision.
    """
    dependent = prices[ticker_dependent]
    independent = prices[ticker_independent]

    state = np.array(hyperparams.initial_state, dtype=float)
    covariance = np.diag(hyperparams.initial_covariance).astype(float)
    process_noise = np.diag(hyperparams.process_noise)
    observation_noise = hyperparams.observation_noise

    alphas = np.empty(len(prices))
    betas = np.empty(len(prices))

    for t in range(len(prices)):
        # Predict
        covariance = covariance + process_noise

        # Update
        observation_matrix = np.array([1.0, independent.iloc[t]])
        prediction = observation_matrix @ state
        innovation = dependent.iloc[t] - prediction
        innovation_variance = observation_matrix @ covariance @ observation_matrix.T + observation_noise
        kalman_gain = (covariance @ observation_matrix) / innovation_variance
        state = state + kalman_gain * innovation
        covariance = covariance - np.outer(kalman_gain, observation_matrix) @ covariance

        alphas[t] = state[0]
        betas[t] = state[1]

    return HedgeRatioSeries(
        ticker_dependent=ticker_dependent,
        ticker_independent=ticker_independent,
        intercept=pd.Series(alphas, index=prices.index),
        ratio=pd.Series(betas, index=prices.index),
    )
