"""Tests for core.evaluation.walk_forward.

Fixes the public API surface:

- WalkForwardWindowResult: the SHARED result type produced by both
  evaluate_walk_forward_windows (this module, generic) and
  strategies.pairs_trading.walk_forward.run_pairs_trading_walk_forward
  (strategy-specific) -- sharing one type lets render_walk_forward_report
  render either without knowing which produced it.
- evaluate_walk_forward_windows(returns, windows, label) -> list[...]:
  slices an ALREADY-COMPUTED continuous returns series into multiple
  windows. No new backtest is run here -- this is a wiring test only;
  metrics.summary()/check_significance() themselves are already verified
  independently in test_metrics.py / test_statistical_tests.py.
- render_walk_forward_report(strategy_name, results) -> str: deliberately
  NOT render_comparison_report reused -- a period mismatch across windows
  is the entire point here, not a caveat to warn about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.sample_split import SamplePeriod, TrainTestSplit, split_returns
from core.evaluation import metrics as metrics_mod
from core.evaluation.statistical_tests import check_significance
from core.evaluation.walk_forward import (
    WalkForwardWindowResult,
    evaluate_walk_forward_windows,
    render_walk_forward_report,
)


def _returns(n: int = 100, seed: int = 0) -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.01, n), index=idx)


def _windows(idx: pd.DatetimeIndex) -> list[TrainTestSplit]:
    return [
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74])
        ),
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[99])
        ),
    ]


# ---------------------------------------------------------------------------
# evaluate_walk_forward_windows
# ---------------------------------------------------------------------------


def test_evaluate_walk_forward_windows_matches_direct_metrics_calls() -> None:
    returns = _returns()
    windows = _windows(returns.index)

    results = evaluate_walk_forward_windows(returns, windows, label="my-strategy")

    assert len(results) == 2
    for result, window in zip(results, windows):
        expected_is, expected_oos = split_returns(returns, window)
        assert result.window == window
        assert result.label == "my-strategy"
        assert result.is_metrics == metrics_mod.summary(expected_is)
        assert result.oos_metrics == metrics_mod.summary(expected_oos)
        # significance is stochastic-but-seeded (seed=0 internally); check
        # it matches an independent direct call with the same seed.
        expected_significance = check_significance(expected_oos, seed=0)
        assert result.significance.permutation.p_value == expected_significance.permutation.p_value
        assert result.significance.agree == expected_significance.agree


def test_evaluate_walk_forward_windows_scales_beyond_two_windows() -> None:
    returns = _returns(n=150)
    idx = returns.index
    windows = [
        TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74])),
        TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[99])),
        TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[99]), out_of_sample=SamplePeriod(idx[100], idx[124])),
    ]

    results = evaluate_walk_forward_windows(returns, windows)

    assert len(results) == 3


def _short_oos_windows(idx: pd.DatetimeIndex) -> list[TrainTestSplit]:
    """Two windows whose OOS slices sit either side of the mechanical
    floor: 3 observations (below block_size=5, not computable at all) and
    10 observations (computable, but too few blocks to be informative)."""
    return [
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[52])
        ),
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[59]), out_of_sample=SamplePeriod(idx[60], idx[69])
        ),
    ]


def test_evaluate_walk_forward_windows_skips_a_window_below_the_mechanical_floor() -> None:
    # Arrange: 3 OOS observations, fewer than the default block_size of 5.
    # Previously this escaped as bootstrap_statistic's low-level
    # "block_size (5) must not exceed the number of observations (3)"
    # ValueError, aborting every remaining window -- contradicting the
    # harness's own "no candidate does not stop the run" design.
    returns = _returns()
    window = _short_oos_windows(returns.index)[0]

    # Act
    results = evaluate_walk_forward_windows(returns, [window])

    # Assert: recorded, not raised, and all three payload fields are None
    # together (the type's documented invariant).
    assert len(results) == 1
    assert results[0].is_metrics is None
    assert results[0].oos_metrics is None
    assert results[0].significance is None
    assert results[0].skip_reason is not None
    assert "3" in results[0].skip_reason


def test_evaluate_walk_forward_windows_skips_a_window_below_the_statistical_floor() -> None:
    # Arrange: 10 OOS observations -- above block_size=5, so the bootstrap
    # would run, but far too few blocks for its interval to carry
    # information. Emitting a p-value here would be reporting precision
    # the data cannot support.
    returns = _returns()
    window = _short_oos_windows(returns.index)[1]

    # Act
    results = evaluate_walk_forward_windows(returns, [window])

    # Assert
    assert results[0].oos_metrics is None
    assert results[0].significance is None
    assert results[0].skip_reason is not None
    assert "10" in results[0].skip_reason


def test_evaluate_walk_forward_windows_distinguishes_the_two_skip_reasons() -> None:
    # The whole point of carrying a reason rather than a bare None: the
    # two skips must not read identically.
    returns = _returns()
    results = evaluate_walk_forward_windows(returns, _short_oos_windows(returns.index))

    assert results[0].skip_reason != results[1].skip_reason


def test_evaluate_walk_forward_windows_does_not_stop_at_a_skipped_window() -> None:
    # Arrange: a skipped window sandwiched between two evaluable ones.
    returns = _returns()
    idx = returns.index
    windows = [
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74])
        ),
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[77])
        ),
        TrainTestSplit(
            in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[99])
        ),
    ]

    # Act
    results = evaluate_walk_forward_windows(returns, windows)

    # Assert: the run continues past the skip and still evaluates window 3.
    assert results[0].oos_metrics is not None
    assert results[1].skip_reason is not None
    assert results[2].oos_metrics is not None


def test_evaluate_walk_forward_windows_leaves_skip_reason_none_when_evaluable() -> None:
    returns = _returns()
    results = evaluate_walk_forward_windows(returns, _windows(returns.index))

    assert all(r.skip_reason is None for r in results)


# ---------------------------------------------------------------------------
# render_walk_forward_report
# ---------------------------------------------------------------------------


def _populated_result(window: TrainTestSplit, label: str, sharpe: float) -> WalkForwardWindowResult:
    oos_idx = pd.bdate_range(window.out_of_sample.start, window.out_of_sample.end)
    is_idx = pd.bdate_range(window.in_sample.start, window.in_sample.end)
    rng = np.random.default_rng(0)
    oos_returns = pd.Series(rng.normal(0.001, 0.01, len(oos_idx)), index=oos_idx)
    is_returns = pd.Series(rng.normal(0.001, 0.01, len(is_idx)), index=is_idx)
    oos_metrics = dict(metrics_mod.summary(oos_returns), sharpe_ratio=sharpe)
    return WalkForwardWindowResult(
        window=window,
        label=label,
        is_metrics=metrics_mod.summary(is_returns),
        oos_metrics=oos_metrics,
        significance=check_significance(oos_returns, seed=0),
    )


def _empty_result(window: TrainTestSplit) -> WalkForwardWindowResult:
    return WalkForwardWindowResult(window=window, label="no candidate", is_metrics=None, oos_metrics=None, significance=None)


def test_render_walk_forward_report_includes_populated_window_values() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    window = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74]))
    result = _populated_result(window, "AEP-FE", sharpe=1.23)

    report = render_walk_forward_report("Pairs Trading", [result])

    assert "Pairs Trading" in report
    assert "AEP-FE" in report
    assert "1.23" in report


def test_render_walk_forward_report_handles_empty_window_without_crashing() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    window = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74]))
    result = _empty_result(window)

    report = render_walk_forward_report("Pairs Trading", [result])

    assert "no candidate" in report
    assert "N/A" in report


def test_render_walk_forward_report_shows_skip_reason_instead_of_the_bare_label() -> None:
    # Arrange: a window skipped for being too short, versus the existing
    # "no candidate" case. Rendering both as an undifferentiated N/A row
    # would conflate "selection found nothing" with "we could not
    # evaluate this window" -- two very different facts about the run.
    idx = pd.bdate_range("2020-01-01", periods=100)
    window = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[52]))
    skipped = WalkForwardWindowResult(
        window=window,
        label="strategy",
        is_metrics=None,
        oos_metrics=None,
        significance=None,
        skip_reason="too short to evaluate (3 OOS observations; need at least 5)",
    )

    report = render_walk_forward_report("Factor Momentum", [skipped])

    assert "too short to evaluate" in report
    assert "3 OOS observations" in report


def test_render_walk_forward_report_summary_breaks_out_skipped_windows() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    good_window = TrainTestSplit(
        in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74])
    )
    short_window = TrainTestSplit(
        in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[77])
    )
    # The two genuinely different routes to an N/A row, side by side.
    results = [
        _populated_result(good_window, "strategy", sharpe=1.0),
        WalkForwardWindowResult(
            window=short_window,
            label="no candidate",
            is_metrics=None,
            oos_metrics=None,
            significance=None,
            skip_reason="no statistically-justified pair survived selection",
        ),
        WalkForwardWindowResult(
            window=short_window,
            label="strategy",
            is_metrics=None,
            oos_metrics=None,
            significance=None,
            skip_reason="too short to evaluate (3 OOS observations; need at least 5)",
        ),
    ]

    report = render_walk_forward_report("Factor Momentum", results)

    # The summary must say how many windows were skipped for lack of data,
    # separately from the overall populated count, so a reader cannot
    # mistake an unevaluable run for an evaluated-but-negative one.
    assert "Windows with a result: 1/3" in report
    assert "Windows with no tested result: 2/3" in report
    # Neutral summary wording: the per-window reasons differ (selection
    # found nothing vs. too little data) and must not be summarised as if
    # they were all the same kind of failure.
    assert "insufficient out-of-sample data" not in report
    assert "no statistically-justified pair survived selection" in report
    assert "too short to evaluate" in report


def test_render_walk_forward_report_consistency_summary_counts_populated_windows() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    window1 = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74]))
    window2 = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[74]), out_of_sample=SamplePeriod(idx[75], idx[99]))
    results = [_populated_result(window1, "AEP-FE", sharpe=1.0), _empty_result(window2)]

    report = render_walk_forward_report("Pairs Trading", results)

    assert "1/2" in report


def test_render_walk_forward_report_includes_multiple_comparisons_reminder() -> None:
    idx = pd.bdate_range("2020-01-01", periods=100)
    window = TrainTestSplit(in_sample=SamplePeriod(idx[0], idx[49]), out_of_sample=SamplePeriod(idx[50], idx[74]))
    result = _populated_result(window, "AEP-FE", sharpe=1.0)

    report = render_walk_forward_report("Pairs Trading", [result])

    assert "multiple comparisons" in report.lower()


def test_render_walk_forward_report_raises_on_empty_results() -> None:
    with pytest.raises(ValueError):
        render_walk_forward_report("Pairs Trading", [])
