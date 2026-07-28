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
