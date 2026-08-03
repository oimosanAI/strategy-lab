"""Tests for strategies.pairs_trading.cointegration.

Fixes the public API surface strategies/pairs_trading/cointegration.py
must implement:

- EngleGrangerResult / engle_granger_test(series_a, series_b, name_a,
  name_b, alpha) -> EngleGrangerResult. Tests BOTH regression directions
  (statsmodels.tsa.stattools.coint is not symmetric) and reports the
  stronger (lower p-value) direction, recording which ticker was the
  dependent variable.
- JohansenResult / johansen_test(prices, alpha) -> JohansenResult, using
  statsmodels.tsa.vector_ar.vecm.coint_johansen.
- compute_half_life(spread) -> float: OU/AR(1)-based mean-reversion speed.
- PairSelectionConfig / PairCandidate.
- select_pairs(prices, in_sample, sector_map, config) -> list[PairCandidate]:
  matches the (prices, in_sample) -> object contract verified by
  core.backtest.sample_split.assert_selection_ignores_out_of_sample (via
  functools.partial to bind sector_map/config).

Per the project's standard, statsmodels' coint()/coint_johansen()/OLS are
trusted the same way scipy's t-distribution and statsmodels' OLS were
trusted in core.evaluation -- the math itself is not re-derived, but
fixtures are independently verified (see conversation record for the
standalone script confirming p-values/half-life/hedge-ratio behave as
expected before any test literal was written) and direction-selection /
filter / ranking LOGIC is verified directly.

Every test constructs its own PairSelectionConfig instance (no shared
fixture is mutated or reused with only one field overridden) so that one
test's threshold choice can never leak into another's.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from core.backtest.engine import LookAheadError
from core.backtest.sample_split import SamplePeriod, TrainTestSplit, assert_selection_ignores_out_of_sample
from strategies.pairs_trading.cointegration import (
    EngleGrangerResult,
    JohansenResult,
    PairCandidate,
    PairSelectionConfig,
    _adjust_p_value,
    _passes_cointegration_filter,
    _rank_candidates,
    compute_half_life,
    engle_granger_test,
    johansen_test,
    select_pairs,
)


# ---------------------------------------------------------------------------
# Fixtures (independently verified via a standalone script -- see the
# conversation record for the exact p-values/correlations/half-life
# confirmed before these were finalized)
# ---------------------------------------------------------------------------


def _cointegrated_pair(n: int = 200, beta: float = 1.5, phi: float = 0.5) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng_a = np.random.default_rng(10)
    a = 100 + np.cumsum(rng_a.normal(0, 1, n))
    rng_spread = np.random.default_rng(0)
    innov = rng_spread.normal(0, 1, n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + innov[t]
    b = beta * a + spread
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def _non_cointegrated_pair(n: int = 200) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng_a = np.random.default_rng(20)
    rng_b = np.random.default_rng(21)
    a = 100 + np.cumsum(rng_a.normal(0, 1, n))
    b = 100 + np.cumsum(rng_b.normal(0, 1, n))
    return pd.Series(a, index=idx, name="A"), pd.Series(b, index=idx, name="B")


def _six_ticker_panel(n: int = 200) -> pd.DataFrame:
    """Tech: A, B (cointegrated), C (correlated with A, NOT cointegrated),
    F (uncorrelated with everything). Financials: D, E (unrelated)."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    a_series, b_series = _cointegrated_pair(n=n)
    rng_c = np.random.default_rng(11)
    c = a_series.to_numpy() + np.cumsum(rng_c.normal(0, 1, n))
    rng_f = np.random.default_rng(12)
    f = 100 + np.cumsum(rng_f.normal(0, 1, n))
    rng_d = np.random.default_rng(13)
    d = 100 + np.cumsum(rng_d.normal(0, 1, n))
    rng_e = np.random.default_rng(14)
    e = 100 + np.cumsum(rng_e.normal(0, 1, n))
    return pd.DataFrame(
        {"A": a_series.to_numpy(), "B": b_series.to_numpy(), "C": c, "F": f, "D": d, "E": e}, index=idx
    )


_SECTOR_MAP = {"A": "Tech", "B": "Tech", "C": "Tech", "F": "Tech", "D": "Financials", "E": "Financials"}


# ---------------------------------------------------------------------------
# 1. EngleGrangerResult / JohansenResult / compute_half_life
# ---------------------------------------------------------------------------


def test_engle_granger_detects_cointegrated_pair() -> None:
    a, b = _cointegrated_pair()
    result = engle_granger_test(a, b, "A", "B", alpha=0.05)
    assert isinstance(result, EngleGrangerResult)
    assert result.is_cointegrated is True
    assert result.p_value < 0.05


def test_engle_granger_rejects_non_cointegrated_pair() -> None:
    a, b = _non_cointegrated_pair()
    result = engle_granger_test(a, b, "A", "B", alpha=0.05)
    assert result.is_cointegrated is False
    assert result.p_value > 0.05


def test_engle_granger_recovers_known_hedge_ratio() -> None:
    a, b = _cointegrated_pair(beta=1.5)
    result = engle_granger_test(a, b, "A", "B")
    assert result.hedge_ratio == pytest.approx(1.5, abs=0.05)


def test_engle_granger_selects_stronger_direction() -> None:
    # Arrange: a wiring check -- call statsmodels' coint() directly in
    # both directions, not a re-derivation of the cointegration test math.
    a, b = _cointegrated_pair()
    stat_b_on_a, pval_b_on_a, _ = coint(b, a)
    stat_a_on_b, pval_a_on_b, _ = coint(a, b)
    expected_dependent = "B" if pval_b_on_a <= pval_a_on_b else "A"
    expected_pvalue = min(pval_b_on_a, pval_a_on_b)

    # Act
    result = engle_granger_test(a, b, "A", "B")

    # Assert
    assert result.dependent == expected_dependent
    assert result.p_value == pytest.approx(expected_pvalue)


def test_johansen_detects_cointegrated_pair() -> None:
    a, b = _cointegrated_pair()
    prices = pd.DataFrame({"A": a, "B": b})
    result = johansen_test(prices, alpha=0.05)
    assert isinstance(result, JohansenResult)
    assert result.is_cointegrated is True
    assert result.trace_statistic > result.critical_value_95


def test_johansen_rejects_non_cointegrated_pair() -> None:
    a, b = _non_cointegrated_pair()
    prices = pd.DataFrame({"A": a, "B": b})
    result = johansen_test(prices, alpha=0.05)
    assert result.is_cointegrated is False
    assert result.trace_statistic <= result.critical_value_95


def test_compute_half_life_hand_computable() -> None:
    # Arrange: a NOISELESS AR(1) decay (spread(t) = 0.5 * spread(t-1)
    # exactly) so the OLS fit is exact, not a stochastic estimate --
    # half_life = -ln(2)/ln(0.5) = 1.0 exactly.
    idx = pd.bdate_range("2020-01-01", periods=20)
    spread = pd.Series([1.0 * (0.5**t) for t in range(20)], index=idx)

    # Act
    result = compute_half_life(spread)

    # Assert
    assert result == pytest.approx(1.0)


def test_compute_half_life_is_infinite_for_non_mean_reverting_spread() -> None:
    # Arrange: a spread that trends rather than reverts (lambda >= 0).
    idx = pd.bdate_range("2020-01-01", periods=20)
    spread = pd.Series([float(t) for t in range(20)], index=idx)

    # Act / Assert
    assert compute_half_life(spread) == float("inf")


# ---------------------------------------------------------------------------
# 2. _passes_cointegration_filter: deterministic, no randomness
# ---------------------------------------------------------------------------


def _jh(is_cointegrated: bool) -> JohansenResult:
    return JohansenResult(
        trace_statistic=30.0 if is_cointegrated else 5.0,
        critical_value_95=15.5,
        is_cointegrated=is_cointegrated,
        cointegrating_vector=(1.0, -1.5),
    )


def test_adjust_p_value_bonferroni() -> None:
    assert _adjust_p_value(0.01, n_tests=10, method="bonferroni") == pytest.approx(0.1)


def test_adjust_p_value_bonferroni_caps_at_one() -> None:
    assert _adjust_p_value(0.5, n_tests=10, method="bonferroni") == pytest.approx(1.0)


def test_adjust_p_value_sidak() -> None:
    expected = 1 - (1 - 0.01) ** 10
    assert _adjust_p_value(0.01, n_tests=10, method="sidak") == pytest.approx(expected)


def test_adjust_p_value_none_leaves_unchanged() -> None:
    assert _adjust_p_value(0.01, n_tests=10, method="none") == pytest.approx(0.01)


def test_adjust_p_value_single_test_leaves_unchanged() -> None:
    assert _adjust_p_value(0.01, n_tests=1, method="bonferroni") == pytest.approx(0.01)


def test_passes_filter_requires_both_when_agree_true() -> None:
    config = PairSelectionConfig(require_both_tests_agree=True)
    assert _passes_cointegration_filter(True, False, config) is False


def test_passes_filter_accepts_either_when_agree_false() -> None:
    config = PairSelectionConfig(require_both_tests_agree=False)
    assert _passes_cointegration_filter(True, False, config) is True


def test_passes_filter_true_when_both_agree_cointegrated() -> None:
    config = PairSelectionConfig(require_both_tests_agree=True)
    assert _passes_cointegration_filter(True, True, config) is True


def test_passes_filter_false_when_both_agree_not_cointegrated() -> None:
    config = PairSelectionConfig(require_both_tests_agree=False)
    assert _passes_cointegration_filter(False, False, config) is False


# ---------------------------------------------------------------------------
# 3. _rank_candidates: deterministic
# ---------------------------------------------------------------------------


def _candidate(p_value: float, adjusted_p_value: float | None = None, n_tests: int = 1) -> PairCandidate:
    return PairCandidate(
        ticker_a="X", ticker_b="Y", sector="Tech",
        engle_granger=EngleGrangerResult("Y", "X", 0.0, 1.0, -5.0, p_value, True),
        johansen=_jh(True),
        half_life=5.0,
        adjusted_engle_granger_p_value=adjusted_p_value if adjusted_p_value is not None else p_value,
        n_tests=n_tests,
    )


def test_rank_candidates_sorts_by_adjusted_engle_granger_p_value_ascending() -> None:
    # Arrange: raw p_value order is DELIBERATELY the reverse of adjusted
    # order, so this only passes if ranking actually uses the adjusted
    # field rather than the raw one.
    candidates = [
        _candidate(p_value=0.001, adjusted_p_value=0.09),
        _candidate(p_value=0.002, adjusted_p_value=0.05),
        _candidate(p_value=0.003, adjusted_p_value=0.01),
    ]

    # Act
    ranked = _rank_candidates(candidates)

    # Assert
    assert [c.adjusted_engle_granger_p_value for c in ranked] == [0.01, 0.05, 0.09]
    assert [c.engle_granger.p_value for c in ranked] == [0.003, 0.002, 0.001]


# ---------------------------------------------------------------------------
# 4. select_pairs: staged filters (each test builds its OWN config)
# ---------------------------------------------------------------------------


def test_select_pairs_only_generates_within_sector_candidates() -> None:
    prices = _six_ticker_panel()
    config = PairSelectionConfig()
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, _SECTOR_MAP, config)

    pairs_found = {(c.ticker_a, c.ticker_b) for c in result}
    cross_sector_pairs = {(a, b) for a in ("A", "B", "C", "F") for b in ("D", "E")}
    assert pairs_found.isdisjoint(cross_sector_pairs)
    assert pairs_found.isdisjoint({(b, a) for a, b in cross_sector_pairs})


def test_select_pairs_skips_tickers_missing_from_sector_map() -> None:
    # Arrange: the price panel includes a ticker with no sector mapping
    # (e.g. missing universe metadata) -- it must be silently skipped,
    # not crash select_pairs or appear in any candidate pair.
    prices = _six_ticker_panel()
    incomplete_sector_map = {k: v for k, v in _SECTOR_MAP.items() if k != "F"}
    config = PairSelectionConfig()
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, incomplete_sector_map, config)

    assert all("F" not in (c.ticker_a, c.ticker_b) for c in result)


def test_select_pairs_excludes_uncorrelated_candidates() -> None:
    prices = _six_ticker_panel()
    config = PairSelectionConfig(correlation_prefilter=0.7)
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, _SECTOR_MAP, config)

    pairs_found = {frozenset((c.ticker_a, c.ticker_b)) for c in result}
    assert frozenset(("A", "F")) not in pairs_found
    assert frozenset(("B", "F")) not in pairs_found


def test_select_pairs_excludes_non_cointegrated_pairs() -> None:
    prices = _six_ticker_panel()
    config = PairSelectionConfig(correlation_prefilter=0.7, require_both_tests_agree=True)
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, _SECTOR_MAP, config)

    pairs_found = {frozenset((c.ticker_a, c.ticker_b)) for c in result}
    assert frozenset(("A", "C")) not in pairs_found


def test_select_pairs_excludes_pairs_outside_half_life_bounds() -> None:
    prices = _six_ticker_panel()
    # A-B's estimated half-life is roughly 1-1.5 days (see conversation
    # record); an independently-constructed, deliberately tight upper
    # bound excludes it purely on half-life grounds.
    config = PairSelectionConfig(correlation_prefilter=0.7, max_half_life_days=0.5)
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, _SECTOR_MAP, config)

    pairs_found = {frozenset((c.ticker_a, c.ticker_b)) for c in result}
    assert frozenset(("A", "B")) not in pairs_found


def test_select_pairs_includes_pair_passing_all_filters() -> None:
    prices = _six_ticker_panel()
    config = PairSelectionConfig()
    split = SamplePeriod(prices.index[0], prices.index[-1])

    result = select_pairs(prices, split, _SECTOR_MAP, config)

    pairs_found = {frozenset((c.ticker_a, c.ticker_b)) for c in result}
    assert frozenset(("A", "B")) in pairs_found


def _prefilter_survivor_pair_count(
    prices: pd.DataFrame, sector_map: dict[str, str], prefilter: float | None
) -> int:
    """Independently re-derive how many candidate PAIRS survive the
    correlation pre-filter, without calling select_pairs.

    Deliberately duplicates select_pairs' grouping/pre-filter logic rather
    than reading ``PairCandidate.n_tests`` back: a test that derives its
    expectation from the value under test can only prove self-consistency,
    never correctness. This helper is what gives the n_tests tests below
    their detection power.
    """
    groups: dict[str, list[str]] = {}
    for ticker in prices.columns:
        sector = sector_map.get(ticker)
        if sector is None:
            continue
        groups.setdefault(sector, []).append(ticker)

    count = 0
    for tickers in groups.values():
        for ticker_a, ticker_b in itertools.combinations(sorted(tickers), 2):
            if prefilter is not None and abs(prices[ticker_a].corr(prices[ticker_b])) < prefilter:
                continue
            count += 1
    return count


def test_select_pairs_n_tests_counts_both_regression_directions() -> None:
    # Arrange: engle_granger_test runs coint() in BOTH directions and
    # reports the stronger (lower p-value) one -- that minimum-of-two
    # selection is itself a second layer of multiple testing, so the
    # Bonferroni family size is 2 tests per surviving pair, not 1.
    prices = _six_ticker_panel()
    config = PairSelectionConfig(correlation_prefilter=0.7)
    split = SamplePeriod(prices.index[0], prices.index[-1])
    n_pairs = _prefilter_survivor_pair_count(prices, _SECTOR_MAP, 0.7)

    # Act
    result = select_pairs(prices, split, _SECTOR_MAP, config)
    ab = next(c for c in result if {c.ticker_a, c.ticker_b} == {"A", "B"})

    # Assert: expectation derived from the panel itself, NOT from the
    # recorded n_tests.
    assert n_pairs == 3  # A-B, A-C, B-C (F is uncorrelated; D-E cross-sector)
    assert ab.n_tests == 2 * n_pairs


def test_select_pairs_records_n_tests_and_adjusted_p_value() -> None:
    # Arrange
    prices = _six_ticker_panel()
    config = PairSelectionConfig(correlation_prefilter=0.7)
    split = SamplePeriod(prices.index[0], prices.index[-1])
    n_pairs = _prefilter_survivor_pair_count(prices, _SECTOR_MAP, 0.7)

    # Act
    result = select_pairs(prices, split, _SECTOR_MAP, config)
    ab = next(c for c in result if {c.ticker_a, c.ticker_b} == {"A", "B"})

    # Assert: the Bonferroni factor is re-derived from the independently
    # counted pair total (2 tests per pair), never from ab.n_tests -- an
    # expectation built out of the value under test would pass for ANY
    # family size and so could not catch an under-counted correction.
    expected_adjusted = min(1.0, ab.engle_granger.p_value * 2 * n_pairs)
    assert ab.adjusted_engle_granger_p_value == pytest.approx(expected_adjusted)
    assert ab.adjusted_engle_granger_p_value >= ab.engle_granger.p_value


def test_select_pairs_excludes_pair_significant_only_under_undercounted_family() -> None:
    # Arrange: the boundary case that is the ENTIRE behavioural difference
    # between counting N pairs and counting 2N tests. alpha is placed
    # strictly between the two adjusted p-values, so A-B is significant
    # under the (wrong) per-pair family size and NOT significant under the
    # (correct) per-test one. Derived from the fixture's own raw p-value
    # rather than hardcoded, so it cannot silently drift.
    prices = _six_ticker_panel()
    split = SamplePeriod(prices.index[0], prices.index[-1])
    n_pairs = _prefilter_survivor_pair_count(prices, _SECTOR_MAP, 0.7)

    raw_p = next(
        c
        for c in select_pairs(
            prices, split, _SECTOR_MAP, PairSelectionConfig(correlation_prefilter=0.7)
        )
        if {c.ticker_a, c.ticker_b} == {"A", "B"}
    ).engle_granger.p_value

    # alpha sits at 1.5x the under-counted adjusted p: above it (so the
    # old, per-pair correction would admit A-B) but below 2x it (so the
    # correct, per-test correction rejects A-B).
    boundary_alpha = raw_p * n_pairs * 1.5
    assert raw_p * n_pairs < boundary_alpha < raw_p * 2 * n_pairs

    config = PairSelectionConfig(correlation_prefilter=0.7, cointegration_alpha=boundary_alpha)

    # Act
    result = select_pairs(prices, split, _SECTOR_MAP, config)

    # Assert
    pairs_found = {frozenset((c.ticker_a, c.ticker_b)) for c in result}
    assert frozenset(("A", "B")) not in pairs_found


def test_select_pairs_with_correction_none_matches_raw_p_value() -> None:
    # Arrange
    prices = _six_ticker_panel()
    config = PairSelectionConfig(correlation_prefilter=0.7, multiple_testing_correction="none")
    split = SamplePeriod(prices.index[0], prices.index[-1])

    # Act
    result = select_pairs(prices, split, _SECTOR_MAP, config)
    ab = next(c for c in result if {c.ticker_a, c.ticker_b} == {"A", "B"})

    # Assert
    assert ab.adjusted_engle_granger_p_value == pytest.approx(ab.engle_granger.p_value)


# ---------------------------------------------------------------------------
# 5. select_pairs: IS/OOS separation contract
# ---------------------------------------------------------------------------


def _leaking_select_pairs(
    prices: pd.DataFrame, in_sample: SamplePeriod, sector_map: dict[str, str], config: PairSelectionConfig
) -> list[PairCandidate]:
    """Negative control: ignores `in_sample` and always evaluates the
    FULL panel -- the realistic 'forgot to respect the split' bug."""
    full_period = SamplePeriod(prices.index[0], prices.index[-1])
    return select_pairs(prices, full_period, sector_map, config)


def test_select_pairs_passes_assert_selection_ignores_oos() -> None:
    prices = _six_ticker_panel(n=260)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[199]),
        out_of_sample=SamplePeriod(prices.index[200], prices.index[259]),
    )
    config = PairSelectionConfig()
    bound_select_pairs = functools.partial(select_pairs, sector_map=_SECTOR_MAP, config=config)

    assert_selection_ignores_out_of_sample(bound_select_pairs, prices, split)


def test_leaking_select_pairs_is_caught_by_assert_selection_ignores_oos() -> None:
    prices = _six_ticker_panel(n=260)
    split = TrainTestSplit(
        in_sample=SamplePeriod(prices.index[0], prices.index[199]),
        out_of_sample=SamplePeriod(prices.index[200], prices.index[259]),
    )
    config = PairSelectionConfig()
    bound_leaking = functools.partial(_leaking_select_pairs, sector_map=_SECTOR_MAP, config=config)

    with pytest.raises(LookAheadError):
        assert_selection_ignores_out_of_sample(bound_leaking, prices, split)
