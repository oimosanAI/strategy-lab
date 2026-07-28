"""Tests for strategies.pairs_trading.static_hedge.

Fixes the public API surface: HedgeRatioSeries (the shared output shape
both static_hedge.py and kalman_hedge.py produce, so signal.py can treat
either uniformly) and static_hedge_ratio(candidate, index) ->
HedgeRatioSeries.

static_hedge_ratio does NOT re-estimate anything: it broadcasts the
already-selected Engle-Granger hedge_ratio (computed once, in-sample,
during pair selection in cointegration.py) as a constant series across
the requested index. Re-running OLS here would just reproduce the exact
same number from the exact same data -- pure duplication.
"""

from __future__ import annotations

import pandas as pd

from strategies.pairs_trading.cointegration import EngleGrangerResult, JohansenResult, PairCandidate
from strategies.pairs_trading.static_hedge import HedgeRatioSeries, static_hedge_ratio


def _candidate(intercept: float = 2.0, hedge_ratio: float = 1.5) -> PairCandidate:
    return PairCandidate(
        ticker_a="A",
        ticker_b="B",
        sector="Tech",
        engle_granger=EngleGrangerResult(
            dependent="B",
            independent="A",
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            test_statistic=-5.0,
            p_value=0.01,
            is_cointegrated=True,
        ),
        johansen=JohansenResult(
            trace_statistic=30.0, critical_value_95=15.5, is_cointegrated=True, cointegrating_vector=(1.0, -1.5)
        ),
        half_life=5.0,
        adjusted_engle_granger_p_value=0.01,
        n_tests=1,
    )


def test_static_hedge_ratio_broadcasts_engle_granger_value_as_constant() -> None:
    # Arrange
    candidate = _candidate(hedge_ratio=1.5)
    index = pd.bdate_range("2020-01-01", periods=10)

    # Act
    result = static_hedge_ratio(candidate, index)

    # Assert
    assert isinstance(result, HedgeRatioSeries)
    assert result.ticker_dependent == "B"
    assert result.ticker_independent == "A"
    assert (result.intercept == 2.0).all()
    assert (result.ratio == 1.5).all()
    assert list(result.ratio.index) == list(index)


def test_static_hedge_ratio_uses_the_selected_direction_not_a_fixed_order() -> None:
    # Arrange: dependent/independent could have gone either way during
    # selection (Engle-Granger is not symmetric) -- confirm
    # static_hedge_ratio respects whichever direction was actually chosen,
    # rather than assuming e.g. ticker_b is always the dependent one.
    flipped_eg = EngleGrangerResult(
        dependent="A",
        independent="B",
        intercept=2.0,
        hedge_ratio=1.5,
        test_statistic=-5.0,
        p_value=0.01,
        is_cointegrated=True,
    )
    flipped_candidate = PairCandidate(
        ticker_a="A",
        ticker_b="B",
        sector="Tech",
        engle_granger=flipped_eg,
        johansen=_candidate().johansen,
        half_life=5.0,
        adjusted_engle_granger_p_value=0.01,
        n_tests=1,
    )
    index = pd.bdate_range("2020-01-01", periods=5)

    # Act
    result = static_hedge_ratio(flipped_candidate, index)

    # Assert
    assert result.ticker_dependent == "A"
    assert result.ticker_independent == "B"
