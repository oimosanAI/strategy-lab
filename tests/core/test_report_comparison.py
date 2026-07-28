"""Tests for core.evaluation.report.render_comparison_report /
write_comparison_report (REQUIREMENTS.md 6: "戦略間比較").

Fixes the public API surface:

- render_comparison_report(strategies: Sequence[StrategyReportInputs]) -> str:
  renders a side-by-side Markdown comparison. Must scale to N strategies,
  not just the 2 currently available (pairs_trading, factor_momentum) --
  vol_arbitrage or others may join the comparison later without a
  breaking rewrite.
- Deliberately does NOT re-verify anything (same "render, don't enforce"
  philosophy as render_strategy_report): a period mismatch across
  strategies renders a warning banner rather than raising, mirroring how
  significance disagreement is rendered as a warning, not an exception.
- Empty input raises ValueError (no strategies to compare is a genuine
  unusable-state boundary, unlike a period mismatch which is a caveat a
  caller might knowingly accept).
- write_comparison_report(strategies, path) is the file-writing
  counterpart to write_strategy_report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.evaluation.metrics import BenchmarkComparison
from core.evaluation.report import (
    StrategyReportInputs,
    render_comparison_report,
    write_comparison_report,
)
from core.evaluation.statistical_tests import BootstrapResult, PermutationResult, SignificanceCheck


def _permutation(p_value: float = 0.0234) -> PermutationResult:
    return PermutationResult(
        observed=0.01,
        p_value=p_value,
        n_permutations=2000,
        alternative="greater",
        null_distribution=np.array([0.0]),
        seed=0,
    )


def _bootstrap(lower: float = 0.0123, upper: float = 0.0456) -> BootstrapResult:
    return BootstrapResult(
        point_estimate=0.03,
        lower=lower,
        upper=upper,
        confidence_level=0.95,
        standard_error=0.008,
        distribution=np.array([0.0]),
        n_resamples=2000,
        seed=0,
    )


def _metrics(sharpe: float = 1.0) -> dict[str, float]:
    return {
        "annualized_return": 0.1234,
        "annualized_volatility": 0.0890,
        "sharpe_ratio": sharpe,
        "sortino_ratio": 1.567,
        "calmar_ratio": 0.891,
        "max_drawdown": -0.1567,
        "max_drawdown_duration": 23.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 1.2,
    }


def _inputs(
    name: str = "Strategy",
    start: str = "2020-01-01",
    periods: int = 5,
    sharpe: float = 1.0,
    agree: bool = True,
    benchmark: BenchmarkComparison | None = None,
    walk_forward_note: str | None = None,
    limitations: list[str] | None = None,
) -> StrategyReportInputs:
    return StrategyReportInputs(
        name=name,
        returns=pd.Series(
            [0.01, -0.005, 0.02, -0.01, 0.015][:periods], index=pd.bdate_range(start, periods=periods)
        ),
        metrics=_metrics(sharpe=sharpe),
        significance=SignificanceCheck(permutation=_permutation(), bootstrap=_bootstrap(), agree=agree),
        causality_note=f"assert_backtest_causal passed for {name}",
        benchmark=benchmark,
        walk_forward_note=walk_forward_note,
        limitations=limitations if limitations is not None else [f"{name}'s own limitation"],
    )


# ---------------------------------------------------------------------------
# 1. basic rendering, two strategies
# ---------------------------------------------------------------------------


def test_render_comparison_includes_all_sections_for_two_strategies() -> None:
    strategies = [
        _inputs(name="Pairs Trading", sharpe=-0.28, limitations=["Single pair only"]),
        _inputs(name="Factor Momentum", sharpe=0.44, limitations=["Sector neutrality not enforced"]),
    ]

    report = render_comparison_report(strategies)

    assert "Pairs Trading" in report
    assert "Factor Momentum" in report
    assert "assert_backtest_causal passed for Pairs Trading" in report
    assert "assert_backtest_causal passed for Factor Momentum" in report
    assert "Single pair only" in report
    assert "Sector neutrality not enforced" in report
    assert "-0.28" in report
    assert "0.44" in report


# ---------------------------------------------------------------------------
# 2. must scale to N strategies, not hardcoded to 2
# ---------------------------------------------------------------------------


def test_render_comparison_scales_to_four_strategies_not_hardcoded_to_two() -> None:
    strategies = [_inputs(name=f"Strategy{i}", sharpe=float(i)) for i in range(1, 5)]

    report = render_comparison_report(strategies)

    # Header row: all 4 names appear on one line (columns of the same row,
    # not 4 separate single-strategy blocks).
    header_lines = [line for line in report.splitlines() if "Strategy1" in line and "Strategy4" in line]
    assert header_lines, "expected a single header line containing all 4 strategy names"

    # The Sharpe Ratio row must have exactly 4 data cells (one per
    # strategy), not a hardcoded 2 or 3.
    sharpe_lines = [line for line in report.splitlines() if line.strip().startswith("| Sharpe Ratio")]
    assert len(sharpe_lines) == 1
    cells = [c.strip() for c in sharpe_lines[0].strip().strip("|").split("|")]
    data_cells = cells[1:]  # first cell is the "Sharpe Ratio" label
    assert len(data_cells) == 4
    assert data_cells == ["1.00", "2.00", "3.00", "4.00"]


# ---------------------------------------------------------------------------
# 3. period mismatch warning
# ---------------------------------------------------------------------------


def test_render_comparison_warns_when_periods_differ() -> None:
    strategies = [
        _inputs(name="A", start="2020-01-01"),
        _inputs(name="B", start="2021-06-01"),
    ]

    report = render_comparison_report(strategies)

    assert "⚠" in report


def test_render_comparison_no_warning_when_periods_match() -> None:
    strategies = [
        _inputs(name="A", start="2020-01-01", periods=5),
        _inputs(name="B", start="2020-01-01", periods=5),
    ]

    report = render_comparison_report(strategies)

    assert "⚠" not in report


# ---------------------------------------------------------------------------
# 4. empty sequence / single strategy
# ---------------------------------------------------------------------------


def test_render_comparison_raises_on_empty_sequence() -> None:
    with pytest.raises(ValueError):
        render_comparison_report([])


def test_render_comparison_handles_single_strategy() -> None:
    strategies = [_inputs(name="Solo Strategy")]

    report = render_comparison_report(strategies)

    assert "Solo Strategy" in report


def test_render_comparison_shows_none_noted_for_strategy_with_no_limitations() -> None:
    strategies = [_inputs(name="A", limitations=[]), _inputs(name="B")]

    report = render_comparison_report(strategies)

    assert "### A" in report
    assert "- None noted." in report


# ---------------------------------------------------------------------------
# 5. significance disagreement warning
# ---------------------------------------------------------------------------


def test_render_comparison_warns_when_any_strategy_disagrees() -> None:
    strategies = [
        _inputs(name="Agrees", agree=True),
        _inputs(name="Disagrees", agree=False),
    ]

    report = render_comparison_report(strategies)

    assert "Disagrees" in report
    assert "not established" in report


# ---------------------------------------------------------------------------
# 6. benchmark section presence/absence
# ---------------------------------------------------------------------------


def test_render_comparison_omits_benchmark_section_when_none_provided() -> None:
    strategies = [_inputs(name="A"), _inputs(name="B")]

    report = render_comparison_report(strategies)

    assert "Benchmark Comparison" not in report


def test_render_comparison_shows_benchmark_with_na_for_missing() -> None:
    benchmark = BenchmarkComparison(alpha=0.05, beta=0.8, r_squared=0.6, alpha_pvalue=0.01, beta_pvalue=0.02)
    strategies = [
        _inputs(name="Has Benchmark", benchmark=benchmark),
        _inputs(name="No Benchmark", benchmark=None),
    ]

    report = render_comparison_report(strategies)

    assert "Benchmark Comparison" in report
    assert "N/A" in report


# ---------------------------------------------------------------------------
# 7. write_comparison_report
# ---------------------------------------------------------------------------


def test_write_comparison_report_writes_file_matching_render_output(tmp_path: Path) -> None:
    strategies = [_inputs(name="A"), _inputs(name="B")]
    path = tmp_path / "comparison.md"

    write_comparison_report(strategies, path)

    assert path.read_text(encoding="utf-8") == render_comparison_report(strategies)
