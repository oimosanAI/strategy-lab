"""Statistical-significance tests for evaluation return series.

Every function here takes a plain ``pd.Series`` of periodic (typically
daily) returns -- deliberately decoupled from ``core.backtest``, same as
``core.evaluation.metrics`` -- so this module is reusable for any daily-
return record, not just this repo's own backtests.

Per the project's backtest-verification standard, a clean Sharpe ratio is
not evidence: this module answers "could this be luck?" (permutation /
t-test) and "how uncertain is the number itself?" (bootstrap CI).

Three complementary, DIFFERENT null hypotheses live in this module -- do
not conflate them:

- ``one_sample_ttest``: assumes returns are i.i.d. and (asymptotically)
  normal; tests whether the mean differs from ``popmean``.
- ``sign_flip_permutation_test``: assumes returns are independent and
  symmetric around zero (each observed r_i could equally likely have been
  -r_i); makes no normality assumption, but -- like the t-test -- still
  assumes independence across periods. KNOWN LIMITATION: real daily
  strategy returns are often autocorrelated (momentum, volatility
  clustering), which this test ignores entirely. It can therefore
  understate the true uncertainty and overstate significance. This is
  exactly why it must never be read alone -- see ``SignificanceCheck``.
- ``bootstrap_statistic``: no distributional assumption; with
  ``block_size > 1`` (moving-block), it is comparatively robust to serial
  correlation, which is what makes it a meaningful cross-check for the
  permutation test's blind spot.

A fourth null (positions-vs-returns permutation, testing whether a
strategy's *timing* carries information) exists in the sibling
backtest-framework repo but is deliberately NOT ported here this cycle: it
needs per-ticker (positions, returns) pairs to generalise correctly to a
multi-asset portfolio (naively netting positions across simultaneously
long/short legs -- e.g. a pairs trade -- would erase the very timing
signal being tested). Revisit when Phase 3 pairs-trading needs it.

Every function takes an explicit ``seed`` so every reported p-value and
interval is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from core.evaluation.metrics import sharpe_ratio

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_RESAMPLES: int = 2000
DEFAULT_N_PERMUTATIONS: int = 2000
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_SEED: int = 0

Statistic = Callable[[pd.Series], float]
Alternative = Literal["greater", "less", "two-sided"]


def _default_statistic(returns: pd.Series) -> float:
    """Annualised Sharpe ratio -- bootstrap_statistic's default score."""
    return sharpe_ratio(returns)


def _mean_statistic(returns: pd.Series) -> float:
    """Mean return -- sign_flip_permutation_test's default score: the most
    directly interpretable statistic under a sign-symmetry null."""
    return float(returns.mean())


def _as_series(values: np.ndarray) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# One-sample t-test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTestResult:
    statistic: float
    p_value: float
    mean: float
    n: int


def one_sample_ttest(
    returns: pd.Series,
    popmean: float = 0.0,
    alternative: Alternative = "two-sided",
) -> TTestResult:
    """One-sample t-test of H0: mean(returns) == popmean.

    Thin wrapper around ``scipy.stats.ttest_1samp`` -- the t-distribution
    CDF is trusted from scipy the same way this project trusts numpy's
    sqrt or statsmodels' OLS elsewhere, rather than being re-derived here.

    Raises
    ------
    ValueError
        If fewer than two observations are provided.
    """
    r = returns.astype(float).dropna()
    if len(r) < 2:
        raise ValueError("need at least two observations for a t-test")
    result = stats.ttest_1samp(r.to_numpy(), popmean, alternative=alternative)
    return TTestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        mean=float(r.mean()),
        n=len(r),
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (ported from backtest-framework)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    distribution: np.ndarray
    n_resamples: int
    seed: int


def _resample_indices(rng: np.random.Generator, n: int, block_size: int) -> np.ndarray:
    """Return ``n`` resampled positions, i.i.d. or in moving blocks.

    With ``block_size == 1`` this is the ordinary i.i.d. bootstrap. With
    ``block_size > 1`` it is the moving-block bootstrap, which resamples
    contiguous runs and so preserves short-range autocorrelation.
    """
    if block_size <= 1:
        return rng.integers(0, n, size=n)

    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size, 0)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    offsets = np.arange(block_size)
    idx: np.ndarray = (starts[:, None] + offsets[None, :]).ravel()[:n]
    return idx


def bootstrap_statistic(
    returns: pd.Series,
    statistic: Statistic = _default_statistic,
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    block_size: int = 1,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap a percentile-method confidence interval for ``statistic``.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than two observations, ``n_resamples`` is
        not positive, ``confidence_level`` is outside ``(0, 1)``, or
        ``block_size`` is not positive or exceeds the number of
        observations.
    """
    if confidence_level <= 0.0 or confidence_level >= 1.0:
        raise ValueError("confidence_level must lie strictly in (0, 1)")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    r = returns.astype(float).dropna()
    n = len(r)
    if n < 2:
        raise ValueError("need at least two observations to bootstrap")
    if block_size > n:
        raise ValueError(
            f"block_size ({block_size}) must not exceed the number of "
            f"observations ({n}); a block larger than the sample makes every "
            "resample identical to the original series"
        )

    point = float(statistic(r))
    values = r.to_numpy()
    rng = np.random.default_rng(seed)

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = _resample_indices(rng, n, block_size)
        samples[i] = statistic(_as_series(values[idx]))

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        raise ValueError(
            "every bootstrap resample produced an undefined statistic; "
            "the statistic may be ill-defined for this series"
        )

    tail = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(finite, tail))
    upper = float(np.quantile(finite, 1.0 - tail))
    standard_error = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    return BootstrapResult(
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence_level=confidence_level,
        standard_error=standard_error,
        distribution=finite,
        n_resamples=n_resamples,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Sign-flip permutation test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    n_permutations: int
    alternative: Alternative
    null_distribution: np.ndarray
    seed: int


def _p_value_from_null(observed: float, null: np.ndarray, alternative: Alternative) -> float:
    """Add-one-corrected p-value: the observed statistic itself counts as
    one valid draw from the null, so the p-value can never be exactly 0."""
    n = null.size
    if alternative == "greater":
        hits = int(np.sum(null >= observed))
    elif alternative == "less":
        hits = int(np.sum(null <= observed))
    elif alternative == "two-sided":
        centre = float(np.mean(null))
        hits = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown alternative {alternative!r}")
    return (1 + hits) / (1 + n)


def sign_flip_permutation_test(
    returns: pd.Series,
    statistic: Statistic = _mean_statistic,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    alternative: Alternative = "greater",
    seed: int = DEFAULT_SEED,
) -> PermutationResult:
    """Sign-flip permutation test.

    H0: the return-generating process is symmetric around zero and each
    period is independent -- each observed r_i could equally likely have
    been -r_i. Under H0, independently flipping every period's sign
    produces a valid draw from the null distribution of ``statistic``.

    KNOWN LIMITATION: this ignores autocorrelation and volatility
    clustering in daily returns entirely. An autocorrelated series can
    make this test's p-value overstate significance. Never read this
    result alone -- pair it with a moving-block ``bootstrap_statistic``
    CI (see ``check_significance``) before drawing any conclusion.

    Raises
    ------
    ValueError
        If ``n_permutations`` is not positive or fewer than two
        observations are provided.
    """
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")

    r = returns.astype(float).dropna()
    if len(r) < 2:
        raise ValueError("need at least two observations for a permutation test")

    values = r.to_numpy()
    observed = float(statistic(_as_series(values)))

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(values))
        null[i] = statistic(_as_series(values * signs))

    null = null[np.isfinite(null)]
    p_value = _p_value_from_null(observed, null, alternative)

    return PermutationResult(
        observed=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        alternative=alternative,
        null_distribution=null,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Composite: sign-flip permutation + moving-block bootstrap, cross-checked
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceCheck:
    """Pairs sign_flip_permutation_test with a moving-block bootstrap CI so
    neither is read in isolation.

    Attributes
    ----------
    permutation, bootstrap:
        The two component results.
    agree:
        True iff both checks reach the same conclusion about significance
        (see ``_check_agreement``). False signals a genuine disagreement
        -- e.g. the permutation test claims significance but the bootstrap
        CI still straddles zero -- worth investigating, not averaging
        away. When ``agree`` is False, trust the (comparatively
        autocorrelation-robust) bootstrap CI and treat the result as NOT
        established; do not split the difference between the two numbers.
    """

    permutation: PermutationResult
    bootstrap: BootstrapResult
    agree: bool


def _check_agreement(permutation: PermutationResult, bootstrap: BootstrapResult, alpha: float) -> bool:
    """Pure function: do the two checks reach the same conclusion?

    Kept separate from any randomness so disagreement scenarios can be
    tested deterministically with hand-built results.
    """
    permutation_significant = permutation.p_value < alpha
    bootstrap_excludes_zero = not (bootstrap.lower <= 0.0 <= bootstrap.upper)
    return permutation_significant == bootstrap_excludes_zero


def check_significance(
    returns: pd.Series,
    statistic: Statistic = _mean_statistic,
    *,
    alpha: float = 0.05,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: int = 5,
    seed: int = DEFAULT_SEED,
) -> SignificanceCheck:
    """Run sign_flip_permutation_test and a moving-block bootstrap_statistic
    on the same returns/statistic, and report whether they agree.

    ``block_size`` defaults to 5 here (unlike bootstrap_statistic's own
    neutral default of 1/i.i.d.): this function exists specifically to
    counter the permutation test's i.i.d.-sign-symmetry blind spot, so its
    bootstrap component should not itself default to the same blind
    assumption.
    """
    permutation = sign_flip_permutation_test(returns, statistic, n_permutations=n_permutations, seed=seed)
    bootstrap = bootstrap_statistic(
        returns, statistic, n_resamples=n_resamples, block_size=block_size, seed=seed
    )
    agree = _check_agreement(permutation, bootstrap, alpha)
    return SignificanceCheck(permutation=permutation, bootstrap=bootstrap, agree=agree)


# ---------------------------------------------------------------------------
# Multiple-testing / data-snooping corrections
#
# Ported from the sibling backtest-framework repo. Originally scoped here
# for multi-strategy comparison; first ACTUALLY used by
# strategies.pairs_trading.cointegration.select_pairs, which scans many
# candidate pairs for cointegration and needs the same correction for the
# same reason (many hypotheses tested -> chance false positives, unless
# corrected).
# ---------------------------------------------------------------------------


def bonferroni_adjust(p_value: float, n_trials: int) -> float:
    """Bonferroni-correct a p-value for having tried ``n_trials``
    hypotheses (strategies, parameter settings, candidate pairs, ...).

    Multiplies by ``n_trials`` (capped at 1.0). Valid under ANY dependence
    structure among the trials, which makes it conservative -- often too
    conservative when trials are highly correlated -- but it is the
    simplest honest adjustment and a good first sanity check.

    Raises
    ------
    ValueError
        If ``p_value`` is outside [0, 1] or ``n_trials`` < 1.
    """
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must lie in [0, 1]")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return min(1.0, p_value * n_trials)


def sidak_adjust(p_value: float, n_trials: int) -> float:
    """Sidak-correct a p-value for ``n_trials`` INDEPENDENT trials.

    ``1 - (1 - p) ** n_trials`` is the exact family-wise correction when
    trials are independent, and is marginally less conservative than
    Bonferroni. Independence rarely holds for nearby/overlapping trials
    (e.g. candidate pairs sharing a ticker) -- use Bonferroni instead when
    that assumption is in doubt.

    Raises
    ------
    ValueError
        If ``p_value`` is outside [0, 1] or ``n_trials`` < 1.
    """
    if not 0.0 <= p_value <= 1.0:
        raise ValueError("p_value must lie in [0, 1]")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    return 1.0 - (1.0 - p_value) ** n_trials
