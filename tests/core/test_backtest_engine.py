"""Integration tests for core.backtest.engine.run_backtest.

Fixes the public API surface run_backtest must implement:

- BacktestConfig(commission, slippage, initial_capital)
- BacktestResult(signals, target_position, position, turnover, costs,
  gross_returns, returns, equity_curve)
- run_backtest(prices, open_prices, strategy, sizer, config) -> BacktestResult
- assert_backtest_causal(strategy, sizer, close_prices, open_prices, config,
  ...) -> None: whole-pipeline generalization of assert_causal, covering
  the sizing and portfolio-accounting stages assert_causal alone doesn't
  reach.

All expected numeric values below (turnover, costs, gross/net returns,
equity_curve) were derived independently of the production implementation:
via a standalone script (no import of core.backtest at all) that
reimplements the same formulas as plain arithmetic. See the conversation
record for that script's output; the literals here are copied from its
printed results, not from running run_backtest itself.

Fixture design: two tickers on a 6-day panel, with EXACTLY constant daily
returns (pure geometric growth: A +1%/day, B +2%/day). This makes
VolTargetSizer's rolling realized-volatility exactly zero once its window
is filled, so weight clips to a clean +/-max_leverage -- exercising the
real VolTargetSizer.apply() (rolling std, clip, warm-up masking) rather
than a stub, while keeping the arithmetic hand-tractable. Signal is a
trivial constant (+1 for A, -1 for B, never reads prices) so this test's
focus stays on wiring, not re-proving Strategy-level causality (already
covered in test_engine.py).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    LookAheadError,
    assert_backtest_causal,
    run_backtest,
)
from core.backtest.portfolio import ExposureLimitError, ExposureLimitWarning
from core.backtest.position_sizing import PositionSizingConfig, VolTargetSizer
from tests.core.test_position_sizing import _FullSampleVolSizer


class _FixedLongShortStrategy:
    """Trivially causal: signal never reads prices at all."""

    name = "fixed-long-short"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"A": 1.0, "B": -1.0}, index=prices.index)


def _e2e_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 6
    idx = pd.bdate_range("2020-01-01", periods=n)
    close_a = [100.0 * 1.01**t for t in range(n)]
    close_b = [50.0 * 1.02**t for t in range(n)]
    open_a = list(close_a)
    open_b = list(close_b)
    open_a[3] = 102.5  # only day 3 (the only day with turnover) matters
    open_b[3] = 52.5
    close = pd.DataFrame({"A": close_a, "B": close_b}, index=idx)
    open_ = pd.DataFrame({"A": open_a, "B": open_b}, index=idx)
    return close, open_


SIZER = VolTargetSizer(PositionSizingConfig(vol_window=2, target_vol=0.15, max_leverage=2.0))


# ---------------------------------------------------------------------------
# 1 & 3. End-to-end wiring + BacktestResult field verification
# ---------------------------------------------------------------------------


def test_run_backtest_end_to_end_matches_independently_derived_values() -> None:
    # Arrange
    close, open_ = _e2e_fixture()
    config = BacktestConfig(commission=0.0005, slippage=0.0005, initial_capital=1.0)

    # Act
    result = run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, config)

    # Assert: signals
    assert (result.signals["A"] == 1.0).all()
    assert (result.signals["B"] == -1.0).all()

    # Assert: target_position (warm-up flat for 2 days, then clipped)
    assert result.target_position["A"].tolist() == pytest.approx([0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    assert result.target_position["B"].tolist() == pytest.approx([0.0, 0.0, -2.0, -2.0, -2.0, -2.0])

    # Assert: position (EXECUTION_LAG=1 shift of target_position)
    assert result.position["A"].tolist() == pytest.approx([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
    assert result.position["B"].tolist() == pytest.approx([0.0, 0.0, 0.0, -2.0, -2.0, -2.0])

    # Assert: turnover -- only day 3 has any (entry from flat)
    assert result.turnover.tolist() == pytest.approx([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])

    # Assert: costs = turnover * (commission + slippage) = turnover * 0.001
    assert result.costs.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.004, 0.0, 0.0])

    # Assert: gross_returns (independently derived, see module docstring)
    assert result.gross_returns.tolist() == pytest.approx(
        [0.0, 0.0, 0.0, -0.01100515679442493, -0.020000000000000018, -0.020000000000000018]
    )

    # Assert: returns (net of costs)
    assert result.returns.tolist() == pytest.approx(
        [0.0, 0.0, 0.0, -0.01500515679442493, -0.020000000000000018, -0.020000000000000018]
    )

    # Assert: equity_curve
    assert result.equity_curve.tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 0.9849948432055751, 0.9652949463414635, 0.9459890474146342]
    )


# ---------------------------------------------------------------------------
# 2. Whole-pipeline causality guard
# ---------------------------------------------------------------------------


def test_run_backtest_passes_assert_backtest_causal() -> None:
    # Arrange
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.DataFrame(
        {"A": 100 + np.cumsum(rng.normal(0, 1, 120)), "B": 100 + np.cumsum(rng.normal(0, 1, 120))}, index=idx
    ).abs() + 1.0
    open_ = close * 1.001  # a small, fixed, deterministic gap -- not perturbed independently here

    # Act / Assert: perturbing future close/open prices must not change
    # past positions or returns.
    assert_backtest_causal(
        _FixedLongShortStrategy(),
        VolTargetSizer(PositionSizingConfig(vol_window=10)),
        close,
        open_,
        n_trials=12,
    )


def test_run_backtest_causality_guard_catches_broken_sizer() -> None:
    # Arrange: reuse the ALREADY-PROVEN-BROKEN sizer from
    # test_position_sizing.py rather than inventing a new bug, to show the
    # whole-pipeline guard propagates a sizing-layer leak correctly.
    rng = np.random.default_rng(10)
    idx = pd.bdate_range("2020-01-01", periods=120)
    close = pd.DataFrame({"A": 100 + np.cumsum(rng.normal(0, 1, 120))}, index=idx).abs() + 1.0
    open_ = close * 1.001

    # Act / Assert
    with pytest.raises(LookAheadError):
        assert_backtest_causal(
            _FixedLongShortStrategy(),
            _FullSampleVolSizer(),
            close,
            open_,
            n_trials=12,
        )


def test_assert_backtest_causal_rejects_too_few_observations() -> None:
    # Arrange
    idx = pd.bdate_range("2020-01-01", periods=2)
    close = pd.DataFrame({"A": [100.0, 101.0]}, index=idx)
    open_ = close.copy()

    # Act / Assert
    with pytest.raises(ValueError):
        assert_backtest_causal(_FixedLongShortStrategy(), SIZER, close, open_)


# ---------------------------------------------------------------------------
# 4. Costs are correctly subtracted
# ---------------------------------------------------------------------------


def test_costs_are_correctly_subtracted() -> None:
    # Arrange
    close, open_ = _e2e_fixture()
    no_cost_config = BacktestConfig(commission=0.0, slippage=0.0)
    with_cost_config = BacktestConfig(commission=0.0005, slippage=0.0005)

    # Act
    no_cost = run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, no_cost_config)
    with_cost = run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, with_cost_config)

    # Assert: gross returns are identical regardless of cost config.
    pd.testing.assert_series_equal(no_cost.gross_returns, with_cost.gross_returns)

    # Assert: zero-cost config has zero costs and returns == gross_returns.
    assert (no_cost.costs == 0.0).all()
    pd.testing.assert_series_equal(no_cost.returns, no_cost.gross_returns, check_names=False)

    # Assert: the difference between the two returns series equals exactly
    # the costs charged under the nonzero-cost config.
    diff = no_cost.returns - with_cost.returns
    pd.testing.assert_series_equal(diff, with_cost.costs, check_names=False)

    # Assert: independently-derived no-cost equity curve (see module docstring).
    assert no_cost.equity_curve.tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 0.9889948432055751, 0.9692149463414635, 0.9498306474146343]
    )


# ---------------------------------------------------------------------------
# 5. Exposure-limit enforcement (structural, wired into run_backtest)
# ---------------------------------------------------------------------------


class _AllLongStrategy:
    """Every ticker gets the SAME sign (+1.0) -- a stand-in for
    factor_momentum's original un-normalized gross-exposure blowout (see
    strategies/factor_momentum/README.md): no offsetting short leg, so
    gross and net grow together as more tickers are added."""

    name = "all-long"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


def _blowout_fixture(n_tickers: int = 5, n_days: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """n_tickers, each with constant +1%/day growth (realized vol -> 0
    once SIZER's vol_window warms up), so VolTargetSizer clips EVERY
    ticker to +max_leverage (2.0). Combined with _AllLongStrategy (all
    signal=+1.0, no offsetting leg), gross = net = n_tickers * 2.0 once
    warmed up -- 10.0 for n_tickers=5, deliberately exceeding
    BacktestConfig's default ExposureLimits(max_gross=5.0, max_net=5.0)."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    close = pd.DataFrame({t: [100.0 * 1.01**d for d in range(n_days)] for t in tickers}, index=idx)
    open_ = close.copy()
    return close, open_


def test_backtest_config_default_exposure_limits_are_5_and_5() -> None:
    config = BacktestConfig()

    assert config.exposure_limits.max_gross == 5.0
    assert config.exposure_limits.max_net == 5.0
    assert config.exposure_limits.max_per_ticker is None
    assert config.exposure_limits_strict is False


def test_run_backtest_populates_exposure_violations_when_default_limits_breached() -> None:
    close, open_ = _blowout_fixture(n_tickers=5)
    config = BacktestConfig()

    result = run_backtest(close, open_, _AllLongStrategy(), SIZER, config)

    assert len(result.exposure_violations) > 0
    assert any(v.kind == "gross" for v in result.exposure_violations)
    assert any(v.kind == "net" for v in result.exposure_violations)


def test_run_backtest_exposure_violations_empty_when_within_limits() -> None:
    close, open_ = _e2e_fixture()
    config = BacktestConfig()

    result = run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, config)

    assert result.exposure_violations == []


def test_run_backtest_warns_once_per_violation_kind_not_once_per_date() -> None:
    close, open_ = _blowout_fixture(n_tickers=5, n_days=10)
    config = BacktestConfig()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_backtest(close, open_, _AllLongStrategy(), SIZER, config)

    exposure_warnings = [w for w in caught if issubclass(w.category, ExposureLimitWarning)]
    # Exactly one warning per violation KIND (gross, net), never one per
    # violating date -- with 10 violating days x 2 kinds, a naive
    # per-violation warn() would emit 20, not 2.
    assert len(exposure_warnings) == 2
    messages = [str(w.message) for w in exposure_warnings]
    assert sum("gross" in m for m in messages) == 1
    assert sum("net" in m for m in messages) == 1


def test_run_backtest_emits_no_exposure_warning_when_within_limits() -> None:
    close, open_ = _e2e_fixture()
    config = BacktestConfig()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, config)

    exposure_warnings = [w for w in caught if issubclass(w.category, ExposureLimitWarning)]
    assert exposure_warnings == []


def test_run_backtest_strict_mode_raises_exposure_limit_error() -> None:
    close, open_ = _blowout_fixture(n_tickers=5)
    config = BacktestConfig(exposure_limits_strict=True)

    with pytest.raises(ExposureLimitError):
        run_backtest(close, open_, _AllLongStrategy(), SIZER, config)


def test_run_backtest_strict_mode_does_not_raise_when_within_limits() -> None:
    close, open_ = _e2e_fixture()
    config = BacktestConfig(exposure_limits_strict=True)

    result = run_backtest(close, open_, _FixedLongShortStrategy(), SIZER, config)

    assert isinstance(result, BacktestResult)


def test_run_backtest_non_strict_returns_full_result_despite_violations() -> None:
    close, open_ = _blowout_fixture(n_tickers=5)
    config = BacktestConfig()  # exposure_limits_strict=False (default)

    result = run_backtest(close, open_, _AllLongStrategy(), SIZER, config)

    assert isinstance(result, BacktestResult)
    assert len(result.returns) == len(close)
    assert not result.returns.isna().all()
