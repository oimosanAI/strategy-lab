"""Pair selection via cointegration testing (REQUIREMENTS.md 5.1).

Both Engle-Granger (2-step) and Johansen tests are implemented so their
results can be compared, per REQUIREMENTS.md 5.1's explicit requirement.
By default (``PairSelectionConfig.require_both_tests_agree=True``), a
candidate pair is only selected when BOTH tests agree it is cointegrated
-- a deliberately conservative choice, since a false-positive pair
selection in pairs trading turns directly into real trading losses.
Setting it to ``False`` accepts either test alone.

Selection funnel (cheapest checks first):

1. Sector (or sub-industry) match -- candidate generation only.
2. Optional correlation pre-filter -- cheap, avoids running a
   cointegration test on every combinatorial candidate in large sectors.
3. Cointegration test (Engle-Granger and/or Johansen, per
   ``require_both_tests_agree``).
4. Half-life bounds -- a pair that is technically cointegrated but reverts
   too slowly (or the estimated coefficient isn't even mean-reverting) or
   implausibly fast is excluded; this is a hard gate, not a ranking
   criterion (a shorter half-life is not necessarily "better").

``select_pairs`` matches the ``(prices, in_sample) -> object`` contract
verified by ``core.backtest.sample_split.assert_selection_ignores_out_of_sample``
-- it slices to in-sample data via ``slice_in_sample`` before computing
anything. ``sector_map`` and ``config`` are time-independent extra
arguments; bind them with ``functools.partial`` when verifying the IS/OOS
contract (see tests/core/test_cointegration.py).

statsmodels' ``coint``, ``coint_johansen`` and ``OLS`` are trusted the
same way scipy's t-distribution and statsmodels' own OLS were trusted
elsewhere in this project -- the underlying math is not re-derived here,
only wired up and fixture-verified.

MULTIPLE-TESTING CORRECTION (``PairSelectionConfig.multiple_testing_correction``,
default ``"bonferroni"``): scanning a sector for candidate pairs means
running a cointegration test on every pair that survives the correlation
pre-filter -- e.g. 308 tests in a real 54-ticker/2-sector run during this
project's own E2E validation. At alpha=0.05 with 308 tests, ~15 false
positives are expected by CHANCE ALONE, uncorrected -- and 18 candidates
were found in that run, meaning a substantial fraction could be spurious.
One confirmed case: a pair with strong price-LEVEL correlation (a classic
spurious-regression trap -- two trending series look "related" in levels)
but a daily-RETURN correlation of only 0.15, i.e. no real short-term
co-movement. Bonferroni (not Sidak) is used because candidate pairs
sharing a ticker (e.g. three different pairs all involving the same
utility) are correlated tests, not independent ones -- Sidak's
independence assumption does not hold here, while Bonferroni is valid
under any dependence structure.

The correction is applied ONLY to the Engle-Granger p-value
(``PairCandidate.adjusted_engle_granger_p_value``); ``EngleGrangerResult
.is_cointegrated`` and ``JohansenResult.is_cointegrated`` themselves stay
RAW/uncorrected, since those functions are usable standalone and have no
way to know how many other tests a caller is running. KNOWN LIMITATION:
Johansen's decision is NOT corrected here -- ``statsmodels.coint_johansen``
only exposes fixed 90/95/99% critical values, not a computable p-value or
quantile function to Bonferroni-adjust. The ``require_both_tests_agree``
AND-combination still benefits from the Engle-Granger-side correction,
but this is not a fully rigorous joint correction -- do not present it as
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from core.backtest.sample_split import SamplePeriod, slice_in_sample
from core.evaluation.statistical_tests import bonferroni_adjust, sidak_adjust


@dataclass(frozen=True)
class EngleGrangerResult:
    dependent: str
    independent: str
    intercept: float
    hedge_ratio: float
    test_statistic: float
    p_value: float
    is_cointegrated: bool


@dataclass(frozen=True)
class JohansenResult:
    trace_statistic: float
    critical_value_95: float
    is_cointegrated: bool
    cointegrating_vector: tuple[float, float]


@dataclass(frozen=True)
class PairSelectionConfig:
    group_by: str = "gics_sector"
    correlation_prefilter: float | None = 0.7
    cointegration_alpha: float = 0.05
    require_both_tests_agree: bool = True
    min_half_life_days: float = 1.0
    max_half_life_days: float = 60.0
    # See the module docstring's "MULTIPLE-TESTING CORRECTION" section:
    # Bonferroni is the default because candidate pairs sharing a ticker
    # are correlated tests, not independent ones (Sidak's assumption).
    multiple_testing_correction: Literal["bonferroni", "sidak", "none"] = "bonferroni"


@dataclass(frozen=True)
class PairCandidate:
    ticker_a: str
    ticker_b: str
    sector: str
    engle_granger: EngleGrangerResult  # raw (uncorrected) p_value/is_cointegrated
    johansen: JohansenResult  # raw; NOT multiple-testing corrected (see module docstring)
    half_life: float
    adjusted_engle_granger_p_value: float
    n_tests: int


def engle_granger_test(
    series_a: pd.Series,
    series_b: pd.Series,
    name_a: str,
    name_b: str,
    alpha: float = 0.05,
) -> EngleGrangerResult:
    """Engle-Granger 2-step cointegration test, tried in both directions
    (the test is not symmetric); reports whichever direction gave the
    stronger (lower p-value) result."""
    stat_b_on_a, pval_b_on_a, _ = coint(series_b, series_a)
    stat_a_on_b, pval_a_on_b, _ = coint(series_a, series_b)

    if pval_b_on_a <= pval_a_on_b:
        dependent, independent = name_b, name_a
        dep_series, indep_series = series_b, series_a
        stat, pval = stat_b_on_a, pval_b_on_a
    else:
        dependent, independent = name_a, name_b
        dep_series, indep_series = series_a, series_b
        stat, pval = stat_a_on_b, pval_a_on_b

    predictors = sm.add_constant(indep_series)
    model = sm.OLS(dep_series, predictors).fit()
    intercept = float(model.params.iloc[0])
    hedge_ratio = float(model.params.iloc[1])

    return EngleGrangerResult(
        dependent=dependent,
        independent=independent,
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        test_statistic=float(stat),
        p_value=float(pval),
        is_cointegrated=bool(pval < alpha),
    )


def johansen_test(prices: pd.DataFrame, alpha: float = 0.05) -> JohansenResult:
    """Johansen trace test for cointegration between exactly 2 series.

    ``prices`` must have exactly 2 columns; the cointegrating vector is
    reported in that column order, normalized so its first component is 1.
    """
    result = coint_johansen(prices.to_numpy(), det_order=0, k_ar_diff=1)
    trace_statistic = float(result.lr1[0])
    critical_value_95 = float(result.cvt[0, 1])

    vector = np.real(result.evec[:, 0])
    vector = vector / vector[0]

    return JohansenResult(
        trace_statistic=trace_statistic,
        critical_value_95=critical_value_95,
        is_cointegrated=bool(trace_statistic > critical_value_95),
        cointegrating_vector=(float(vector[0]), float(vector[1])),
    )


def compute_half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life of ``spread``, in bars.

    Fits ``delta_spread(t) = lambda * spread(t-1) + eps`` via OLS
    (the discretized Ornstein-Uhlenbeck / AR(1) form) and returns
    ``-ln(2) / ln(1 + lambda)``. Returns ``inf`` when ``lambda`` is at or
    above (a small negative epsilon below) zero -- not meaningfully
    mean-reverting at all. The epsilon guards against floating-point
    OLS estimates of a truly-zero lambda landing at e.g. -1e-16: without
    it, ``1 + lambda`` can round to exactly 1.0, making ``log(1 + lambda)``
    exactly 0.0 and dividing by zero -- discovered via
    tests/core/test_cointegration.py's non-mean-reverting fixture.

    Returns ``nan`` when ``1 + lambda <= 0`` (an oscillating, not
    exponentially-decaying, spread -- e.g. strong negative
    autocorrelation): the half-life formula assumes 0 < 1+lambda < 1 and
    is not meaningful outside that range. ``nan`` fails any
    min/max half-life bound check, so callers filtering on half-life
    correctly exclude such a candidate without a spurious log-of-negative
    warning.
    """
    delta = spread.diff().dropna()
    lagged = spread.shift(1).dropna()
    lagged = lagged.loc[delta.index]

    predictors = sm.add_constant(lagged)
    model = sm.OLS(delta, predictors).fit()
    lam = float(model.params.iloc[1])

    if lam >= -1e-10:
        return float("inf")
    if 1 + lam <= 0:
        return float("nan")
    return -np.log(2) / np.log(1 + lam)


def _adjust_p_value(p_value: float, n_tests: int, method: Literal["bonferroni", "sidak", "none"]) -> float:
    """Multiple-testing correction for one Engle-Granger p-value across
    ``n_tests`` candidate pairs actually tested in one select_pairs() call
    (see the module docstring's "MULTIPLE-TESTING CORRECTION" section for
    why Bonferroni is the default). Reuses
    core.evaluation.statistical_tests's existing adjustment functions --
    the same ones already used for multi-strategy comparison -- rather
    than re-deriving the math.
    """
    if method == "none" or n_tests <= 1:
        return p_value
    if method == "bonferroni":
        return bonferroni_adjust(p_value, n_tests)
    if method == "sidak":
        return sidak_adjust(p_value, n_tests)
    raise ValueError(f"unknown multiple_testing_correction {method!r}")  # pragma: no cover - guarded by Literal


def _passes_cointegration_filter(eg_significant: bool, jh_significant: bool, config: PairSelectionConfig) -> bool:
    """Pure function: does this candidate clear the cointegration bar?

    Takes plain booleans (already-adjusted Engle-Granger significance,
    raw Johansen significance) rather than full result objects, so the
    multiple-testing correction lives entirely in select_pairs and this
    function only ever combines two yes/no answers. Kept separate from
    any data/randomness so the agreement policy can be tested
    deterministically with hand-built booleans.
    """
    if config.require_both_tests_agree:
        return eg_significant and jh_significant
    return eg_significant or jh_significant


def _rank_candidates(candidates: list[PairCandidate]) -> list[PairCandidate]:
    """Rank by the (possibly multiple-testing corrected) Engle-Granger
    p-value, strongest evidence first. Half-life is a hard gate (already
    applied before a candidate reaches this list), not a ranking
    criterion."""
    return sorted(candidates, key=lambda c: c.adjusted_engle_granger_p_value)


def select_pairs(
    prices: pd.DataFrame,
    in_sample: SamplePeriod,
    sector_map: dict[str, str],
    config: PairSelectionConfig | None = None,
) -> list[PairCandidate]:
    """Select candidate pairs using in-sample data only.

    Matches the (prices, in_sample) -> object contract verified by
    core.backtest.sample_split.assert_selection_ignores_out_of_sample:
    slices to in-sample data via slice_in_sample before computing
    anything else. ``sector_map`` and ``config`` are time-independent;
    bind them with functools.partial when verifying the IS/OOS contract.
    """
    config = config or PairSelectionConfig()
    is_prices = slice_in_sample(prices, in_sample).prices

    groups: dict[str, list[str]] = {}
    for ticker in is_prices.columns:
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        groups.setdefault(sector, []).append(ticker)

    # First pass: the correlation-prefilter survivors define the
    # multiple-testing family size (n_tests) below. Pairs excluded here
    # never had a p-value computed, so they don't inflate the correction.
    surviving_pairs: list[tuple[str, str, str]] = []
    for sector, tickers in groups.items():
        for ticker_a, ticker_b in combinations(sorted(tickers), 2):
            if config.correlation_prefilter is not None:
                correlation = is_prices[ticker_a].corr(is_prices[ticker_b])
                if abs(correlation) < config.correlation_prefilter:
                    continue
            surviving_pairs.append((sector, ticker_a, ticker_b))

    n_tests = len(surviving_pairs)

    candidates: list[PairCandidate] = []
    for sector, ticker_a, ticker_b in surviving_pairs:
        series_a = is_prices[ticker_a]
        series_b = is_prices[ticker_b]

        eg = engle_granger_test(series_a, series_b, ticker_a, ticker_b, config.cointegration_alpha)
        jh = johansen_test(is_prices[[ticker_a, ticker_b]], config.cointegration_alpha)

        adjusted_p = _adjust_p_value(eg.p_value, n_tests, config.multiple_testing_correction)
        eg_significant = adjusted_p < config.cointegration_alpha

        if not _passes_cointegration_filter(eg_significant, jh.is_cointegrated, config):
            continue

        dep_series = is_prices[eg.dependent]
        indep_series = is_prices[eg.independent]
        spread = dep_series - eg.intercept - eg.hedge_ratio * indep_series
        half_life = compute_half_life(spread)
        if not (config.min_half_life_days <= half_life <= config.max_half_life_days):
            continue

        candidates.append(
            PairCandidate(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                sector=sector,
                engle_granger=eg,
                johansen=jh,
                half_life=half_life,
                adjusted_engle_granger_p_value=adjusted_p,
                n_tests=n_tests,
            )
        )

    return _rank_candidates(candidates)
