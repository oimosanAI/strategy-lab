"""Tests for core.evaluation.statistical_tests.

Fixes the public API surface core/evaluation/statistical_tests.py must
implement. Deliberately decoupled from core.backtest (pd.Series in, same
as core.evaluation.metrics), per the same reasoning: reusable for any
daily-return record, not just this repo's own backtests.

- one_sample_ttest(returns, popmean=0.0, alternative="two-sided") ->
  TTestResult(statistic, p_value, mean, n). Wraps scipy.stats.ttest_1samp;
  the t-statistic itself is verified independently by hand-formula below,
  but the p-value is cross-checked directly against a scipy call in the
  test rather than re-derived -- we trust scipy's t-distribution CDF the
  same way we trust numpy's sqrt or statsmodels' OLS elsewhere in this
  project. That is a wiring check, not a re-proof of the t-distribution.

- bootstrap_statistic: ported verbatim from the sibling backtest-framework
  repo (src/backtest/significance.py), confirmed to have no single-asset
  assumption. Only the default `statistic` callable now points at
  core.evaluation.metrics.sharpe_ratio.

- sign_flip_permutation_test(returns, statistic=mean, ...) ->
  PermutationResult. New for this project. H0: the return-generating
  process is symmetric around zero and each period is independent (each
  r_i could equally likely have been -r_i). This is NOT the same null as
  the reference's positions-vs-returns permutation_test (deliberately not
  ported this cycle -- see the design discussion) and NOT the t-test's
  normality assumption. KNOWN LIMITATION: ignores autocorrelation /
  volatility clustering in daily returns entirely; an autocorrelated
  series can make this test overstate significance. That is exactly why
  it must never be read alone -- see SignificanceCheck below.

- SignificanceCheck / check_significance / _check_agreement: composite
  result pairing sign_flip_permutation_test with a moving-block
  bootstrap_statistic CI, plus a boolean `agree` computed by the pure
  `_check_agreement` helper (tested directly with hand-built
  PermutationResult/BootstrapResult instances, deterministically --
  no reliance on random simulation happening to produce a disagreement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.evaluation.statistical_tests import (
    Alternative,
    BootstrapResult,
    PermutationResult,
    SignificanceCheck,
    TTestResult,
    _check_agreement,
    bonferroni_adjust,
    bootstrap_statistic,
    check_significance,
    one_sample_ttest,
    sidak_adjust,
    sign_flip_permutation_test,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


# ---------------------------------------------------------------------------
# one_sample_ttest
# ---------------------------------------------------------------------------


def test_ttest_statistic_hand_computable() -> None:
    # Arrange: mean=0.01, std(ddof=1)=0.015811388300841896, n=5
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0])

    # Act
    result = one_sample_ttest(returns)

    # Assert: t = mean / (std/sqrt(n)) = sqrt(2), computed independently
    # by hand-formula, not by calling scipy.
    assert isinstance(result, TTestResult)
    assert result.statistic == pytest.approx(np.sqrt(2))
    assert result.mean == pytest.approx(0.01)
    assert result.n == 5


def test_ttest_matches_scipy_reference() -> None:
    # Arrange: a wiring check -- our wrapper's output vs. calling scipy
    # directly, not a re-derivation of the t-distribution's math.
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0])
    expected = stats.ttest_1samp(returns.to_numpy(), 0.0, alternative="two-sided")

    # Act
    result = one_sample_ttest(returns)

    # Assert
    assert result.statistic == pytest.approx(float(expected.statistic))
    assert result.p_value == pytest.approx(float(expected.pvalue))


def test_ttest_raises_for_insufficient_observations() -> None:
    # Arrange
    returns = _series([0.01])

    # Act / Assert
    with pytest.raises(ValueError):
        one_sample_ttest(returns)


# ---------------------------------------------------------------------------
# bootstrap_statistic (ported)
# ---------------------------------------------------------------------------


def test_bootstrap_reproducible_with_seed() -> None:
    # Arrange
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0, 0.015, -0.02, 0.01])

    # Act
    result1 = bootstrap_statistic(returns, statistic=lambda r: float(r.mean()), n_resamples=200, seed=42)
    result2 = bootstrap_statistic(returns, statistic=lambda r: float(r.mean()), n_resamples=200, seed=42)

    # Assert
    assert result1.point_estimate == result2.point_estimate
    assert result1.lower == result2.lower
    assert result1.upper == result2.upper
    np.testing.assert_array_equal(result1.distribution, result2.distribution)


def _make_ar1_returns(n: int, phi: float, sigma: float, seed: int) -> pd.Series:
    """Strongly autocorrelated synthetic returns (AR(1), phi=0.9). Verified
    independently (standalone script, see conversation record) across 10
    generation seeds that moving-block bootstrap SE robustly exceeds the
    i.i.d. bootstrap SE for this phi/n/block_size combination -- not a
    single-seed coincidence."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.empty(n)
    r[0] = eps[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    return pd.Series(r, index=pd.bdate_range("2020-01-01", periods=n))


def test_bootstrap_moving_block_gives_wider_ci_for_autocorrelated_returns() -> None:
    # Arrange
    returns = _make_ar1_returns(n=150, phi=0.9, sigma=0.01, seed=0)
    mean_statistic = lambda r: float(r.mean())  # noqa: E731

    # Act
    iid_result = bootstrap_statistic(returns, mean_statistic, n_resamples=1000, block_size=1, seed=100)
    block_result = bootstrap_statistic(returns, mean_statistic, n_resamples=1000, block_size=15, seed=200)

    # Assert: i.i.d. resampling destroys the autocorrelation and
    # understates the true sampling uncertainty; moving-block preserves it.
    assert block_result.standard_error > iid_result.standard_error
    assert (block_result.upper - block_result.lower) > (iid_result.upper - iid_result.lower)


def test_bootstrap_rejects_invalid_confidence_level() -> None:
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns, confidence_level=1.5)


def test_bootstrap_rejects_block_size_exceeding_sample_size() -> None:
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns, block_size=10)


def test_bootstrap_rejects_non_positive_n_resamples() -> None:
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns, n_resamples=0)


def test_bootstrap_rejects_non_positive_block_size() -> None:
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns, block_size=0)


def test_bootstrap_rejects_fewer_than_two_observations() -> None:
    returns = _series([0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns)


def test_bootstrap_rejects_when_every_resample_is_undefined() -> None:
    # Arrange: a statistic that is always NaN, regardless of the resample.
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        bootstrap_statistic(returns, statistic=lambda r: float("nan"))


def test_bootstrap_default_statistic_is_sharpe_ratio() -> None:
    # Arrange: exercise the default `statistic` argument (Sharpe ratio),
    # not an explicitly-passed one.
    returns = _series([0.01, 0.02, -0.01, 0.03, 0.0, 0.015, -0.02, 0.01])

    # Act
    result = bootstrap_statistic(returns, n_resamples=200, seed=5)

    # Assert: point estimate matches sharpe_ratio computed directly.
    from core.evaluation.metrics import sharpe_ratio

    assert result.point_estimate == pytest.approx(sharpe_ratio(returns))


# ---------------------------------------------------------------------------
# sign_flip_permutation_test
# ---------------------------------------------------------------------------


def test_sign_flip_rejects_non_positive_n_permutations() -> None:
    returns = _series([0.01, 0.02, -0.01])
    with pytest.raises(ValueError):
        sign_flip_permutation_test(returns, n_permutations=0)


def test_sign_flip_rejects_fewer_than_two_observations() -> None:
    returns = _series([0.01])
    with pytest.raises(ValueError):
        sign_flip_permutation_test(returns)


def test_sign_flip_supports_less_and_two_sided_alternatives() -> None:
    # Arrange: exercise the "less" and "two-sided" branches of
    # _p_value_from_null (only "greater" is used elsewhere in this file).
    returns = _series([-0.05, -0.04, -0.06, -0.05, -0.045, -0.055])

    # Act
    less_result = sign_flip_permutation_test(returns, n_permutations=500, alternative="less", seed=1)
    two_sided_result = sign_flip_permutation_test(returns, n_permutations=500, alternative="two-sided", seed=1)

    # Assert: a clearly negative series is significant under "less" (luck
    # rarely goes this negative) and under "two-sided" (luck rarely
    # deviates this far from typical in either direction).
    assert less_result.p_value < 0.05
    assert two_sided_result.p_value < 0.05


def test_sign_flip_finds_no_edge_in_symmetric_zero_mean_returns() -> None:
    # Arrange: perfectly symmetric around zero -> observed statistic (mean)
    # is exactly 0, sitting at the center of its own null distribution.
    returns = _series([0.02, -0.02, 0.01, -0.01, 0.03, -0.03])

    # Act
    result = sign_flip_permutation_test(returns, n_permutations=2000, seed=1)

    # Assert: not significant.
    assert result.observed == pytest.approx(0.0)
    assert result.p_value > 0.3


def test_sign_flip_finds_edge_in_clearly_positive_returns() -> None:
    # Arrange: every period positive and of similar magnitude. With only
    # n=6 observations there are just 2**6=64 discrete sign patterns, so
    # the smallest p-value this test can ever report is bounded around
    # 1/64 ~ 0.016 (only the all-plus pattern matches or exceeds the
    # observed mean) -- 0.05 is the honest threshold for this sample size,
    # not an arbitrarily loose one.
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])

    # Act
    result = sign_flip_permutation_test(returns, n_permutations=2000, seed=1)

    # Assert
    assert result.p_value < 0.05


def test_sign_flip_null_distribution_is_symmetric_around_zero() -> None:
    # Arrange: even though the INPUT is all-positive (asymmetric), the
    # sign-flip mechanism itself guarantees a null distribution symmetric
    # around zero -- flipping every sign in a drawn pattern is equally
    # likely and exactly negates that draw's statistic.
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])

    # Act
    result = sign_flip_permutation_test(returns, n_permutations=5000, seed=2)

    # Assert: the null's own mean should be close to 0, within Monte Carlo
    # error (a rough 4-sigma bound on the null mean's own sampling error).
    null = result.null_distribution
    mc_error = null.std(ddof=1) / np.sqrt(len(null))
    assert abs(null.mean()) < 4 * mc_error


def test_permutation_rejects_when_every_draw_is_undefined() -> None:
    # Arrange: a statistic that is always NaN, so every sign-flipped draw
    # is non-finite and the null distribution is empty after filtering.
    # Previously this returned p = (1 + 0) / (1 + 0) = 1.0 in silence --
    # a confident-looking "not significant" backed by zero draws, while
    # the sibling bootstrap_statistic raises in exactly this situation
    # (test_bootstrap_rejects_when_every_resample_is_undefined).
    returns = _series([0.01, 0.02, -0.01])

    # Act / Assert
    with pytest.raises(ValueError, match="undefined"):
        sign_flip_permutation_test(returns, statistic=lambda r: float("nan"), n_permutations=50)


def test_permutation_null_distribution_size_is_the_p_value_denominator() -> None:
    # Arrange: a statistic that is NaN for roughly half the draws (those
    # whose sign-flipped sum happens to be negative), so some are dropped.
    def flaky(r: pd.Series) -> float:
        total = float(r.sum())
        return total if total > 0 else float("nan")

    returns = _series([0.01, 0.02, -0.01, 0.03, 0.005])

    # Act
    result = sign_flip_permutation_test(returns, statistic=flaky, n_permutations=200, seed=1)

    # Assert: some draws really were dropped (otherwise this test proves
    # nothing), and the reported p-value is the add-one-corrected ratio
    # over the SURVIVING draws -- which is null_distribution.size, not the
    # requested n_permutations. The two fields must not be read as
    # interchangeable.
    assert result.null_distribution.size < result.n_permutations
    expected = (1 + int(np.sum(result.null_distribution >= result.observed))) / (
        1 + result.null_distribution.size
    )
    assert result.p_value == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _check_agreement: deterministic, no randomness
# ---------------------------------------------------------------------------


def _permutation(p_value: float, alternative: Alternative = "greater") -> PermutationResult:
    return PermutationResult(
        observed=0.01, p_value=p_value, n_permutations=100, alternative=alternative,
        null_distribution=np.array([0.0]), seed=0,
    )


def _bootstrap(lower: float, upper: float, confidence_level: float = 0.90) -> BootstrapResult:
    # confidence_level defaults to 0.90, NOT 0.95: these fixtures pair with
    # a one-sided ("greater") permutation test at alpha=0.05, and the
    # interval whose tail mass matches that is 1 - 2*alpha = 0.90. A 0.95
    # interval here would put 2.5% in the relevant tail against the test's
    # 5% -- the exact mismatch _check_agreement now rejects outright.
    return BootstrapResult(
        point_estimate=0.01, lower=lower, upper=upper, confidence_level=confidence_level,
        standard_error=0.005, distribution=np.array([0.0]), n_resamples=100, seed=0,
    )


def test_check_agreement_true_when_both_significant() -> None:
    assert _check_agreement(_permutation(0.01), _bootstrap(0.002, 0.02), alpha=0.05) is True


def test_check_agreement_true_when_both_not_significant() -> None:
    assert _check_agreement(_permutation(0.5), _bootstrap(-0.01, 0.02), alpha=0.05) is True


def test_check_agreement_false_when_permutation_significant_but_ci_straddles_zero() -> None:
    # The disagreement case this whole module exists to catch.
    assert _check_agreement(_permutation(0.01), _bootstrap(-0.01, 0.02), alpha=0.05) is False


def test_check_agreement_false_when_ci_excludes_zero_but_permutation_not_significant() -> None:
    assert _check_agreement(_permutation(0.5), _bootstrap(0.002, 0.02), alpha=0.05) is False


def test_check_agreement_greater_ignores_an_interval_entirely_below_zero() -> None:
    # A strategy that LOSES money significantly: the interval excludes zero,
    # but on the negative side. An `alternative="greater"` permutation test
    # asks "is the mean significantly ABOVE zero?" and correctly says no.
    # These two are NOT in disagreement -- a direction-agnostic
    # "interval excludes zero" rule would wrongly report agree=False here,
    # because it silently answers a different question from the test it is
    # supposed to be cross-checking. This is the case that distinguishes a
    # directional agreement rule from a symmetric one.
    assert _check_agreement(_permutation(0.9), _bootstrap(-0.02, -0.002), alpha=0.05) is True


def test_check_agreement_less_is_significant_when_interval_lies_below_zero() -> None:
    # Mirror image: with alternative="less", an interval entirely below
    # zero IS the significant case.
    permutation = _permutation(0.01, alternative="less")
    assert _check_agreement(permutation, _bootstrap(-0.02, -0.002), alpha=0.05) is True
    assert _check_agreement(permutation, _bootstrap(-0.01, 0.02), alpha=0.05) is False


def test_check_agreement_two_sided_uses_a_one_minus_alpha_interval() -> None:
    # A two-sided permutation test at alpha pairs with a 1 - alpha
    # interval, and only there is "excludes zero" the right rule.
    permutation = _permutation(0.01, alternative="two-sided")
    assert _check_agreement(permutation, _bootstrap(-0.02, -0.002, 0.95), alpha=0.05) is True
    assert _check_agreement(permutation, _bootstrap(0.002, 0.02, 0.95), alpha=0.05) is True
    assert _check_agreement(permutation, _bootstrap(-0.01, 0.02, 0.95), alpha=0.05) is False


def test_check_agreement_rejects_a_mismatched_confidence_level() -> None:
    # The MEDIUM-2 defect, made structurally impossible rather than merely
    # documented: a one-sided test at alpha=0.05 compared against a 95%
    # two-sided interval puts 5% in the test's tail against the interval's
    # 2.5%. The function refuses rather than silently returning a verdict
    # built on mismatched tail masses.
    with pytest.raises(ValueError, match="confidence_level"):
        _check_agreement(_permutation(0.01), _bootstrap(0.002, 0.02, 0.95), alpha=0.05)


def test_check_agreement_rejects_alpha_outside_zero_to_half() -> None:
    # alpha >= 0.5 would make the one-sided pairing 1 - 2*alpha <= 0, i.e.
    # not a confidence level at all.
    with pytest.raises(ValueError, match="alpha"):
        _check_agreement(_permutation(0.01), _bootstrap(0.002, 0.02), alpha=0.5)


# ---------------------------------------------------------------------------
# check_significance: wiring
# ---------------------------------------------------------------------------


def test_check_significance_wires_permutation_and_bootstrap_correctly() -> None:
    # Arrange
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])

    # Act
    result = check_significance(returns, n_permutations=500, n_resamples=500, seed=3)

    # Assert: correct types, and `agree` matches independently recomputing
    # the same pure function on the components (verifies wiring, not the
    # agreement logic itself -- that's covered by the deterministic tests
    # above).
    assert isinstance(result, SignificanceCheck)
    assert isinstance(result.permutation, PermutationResult)
    assert isinstance(result.bootstrap, BootstrapResult)
    assert result.agree == _check_agreement(result.permutation, result.bootstrap, alpha=0.05)


def test_check_significance_pairs_a_one_sided_test_with_a_matched_interval() -> None:
    # Arrange / Act: the default alternative is one-sided ("greater").
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])
    result = check_significance(returns, n_permutations=500, n_resamples=500, seed=3)

    # Assert: alpha=0.05 one-sided pairs with a 1 - 2*alpha = 0.90
    # interval, so the two components put the same 5% in the tail that
    # actually decides the verdict.
    assert result.permutation.alternative == "greater"
    assert result.bootstrap.confidence_level == pytest.approx(0.90)


def test_check_significance_pairs_a_two_sided_test_with_a_one_minus_alpha_interval() -> None:
    # Arrange / Act
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])
    result = check_significance(
        returns, n_permutations=500, n_resamples=500, alternative="two-sided", seed=3
    )

    # Assert: alternative must reach the permutation component, and the
    # interval must follow it to 1 - alpha = 0.95.
    assert result.permutation.alternative == "two-sided"
    assert result.bootstrap.confidence_level == pytest.approx(0.95)


def test_check_significance_rejects_alpha_outside_zero_to_half() -> None:
    returns = _series([0.05, 0.04, 0.06, 0.05, 0.045, 0.055])
    with pytest.raises(ValueError, match="alpha"):
        check_significance(returns, alpha=0.5, n_permutations=100, n_resamples=100, seed=3)


# ---------------------------------------------------------------------------
# bonferroni_adjust / sidak_adjust
# ---------------------------------------------------------------------------


def test_bonferroni_adjust_multiplies_by_n_trials() -> None:
    assert bonferroni_adjust(0.01, n_trials=10) == pytest.approx(0.1)


def test_bonferroni_adjust_caps_at_one() -> None:
    assert bonferroni_adjust(0.5, n_trials=10) == pytest.approx(1.0)


def test_bonferroni_adjust_rejects_invalid_p_value() -> None:
    with pytest.raises(ValueError):
        bonferroni_adjust(1.5, n_trials=10)


def test_bonferroni_adjust_rejects_invalid_n_trials() -> None:
    with pytest.raises(ValueError):
        bonferroni_adjust(0.01, n_trials=0)


def test_sidak_adjust_matches_formula() -> None:
    expected = 1 - (1 - 0.01) ** 10
    assert sidak_adjust(0.01, n_trials=10) == pytest.approx(expected)


def test_sidak_adjust_rejects_invalid_p_value() -> None:
    with pytest.raises(ValueError):
        sidak_adjust(-0.1, n_trials=10)


def test_sidak_adjust_rejects_invalid_n_trials() -> None:
    with pytest.raises(ValueError):
        sidak_adjust(0.01, n_trials=0)


def test_sidak_adjust_is_less_conservative_than_bonferroni() -> None:
    # A well-known property: for the same (p, n_trials), Sidak's
    # adjustment is <= Bonferroni's (Sidak assumes independence, which is
    # a stronger, more favorable assumption).
    p, n = 0.01, 20
    assert sidak_adjust(p, n) <= bonferroni_adjust(p, n)
