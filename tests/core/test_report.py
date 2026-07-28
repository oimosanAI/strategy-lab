"""Tests for core.evaluation.report.

Fixes the public API surface core/evaluation/report.py must implement.
Deliberately decoupled from core.backtest (same reasoning as metrics.py /
statistical_tests.py): StrategyReportInputs is built from plain
pd.Series/dict/dataclass pieces, never a BacktestResult.

`causality_note` is a REQUIRED field (no default) -- the report cannot
call assert_causal itself (no core.backtest dependency), so the caller is
forced to state, in their own words, what causality verification (if any)
was actually done. An empty/omitted claim is not an option; "NOT VERIFIED"
is an acceptable and honest value for that field, but it must be written.
This mirrors assert_causal's own "verify, don't trust" philosophy one
level up the stack.

render_comparison_report()/write_comparison_report() are tested separately
in test_report_comparison.py, once Phase 3 had at least two real
strategies to compare.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.evaluation.report import StrategyReportInputs, render_strategy_report, write_strategy_report
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


def _returns() -> pd.Series:
    return pd.Series([0.01, -0.005, 0.02, -0.01, 0.015], index=pd.bdate_range("2020-01-01", periods=5))


def _metrics() -> dict[str, float]:
    return {
        "annualized_return": 0.1234,
        "annualized_volatility": 0.0890,
        "sharpe_ratio": 1.234,
        "sortino_ratio": 1.567,
        "calmar_ratio": 0.891,
        "max_drawdown": -0.1567,
        "max_drawdown_duration": 23.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 1.2,
    }


def _inputs(**overrides: object) -> StrategyReportInputs:
    defaults = dict(
        name="Test Strategy",
        returns=_returns(),
        metrics=_metrics(),
        significance=SignificanceCheck(permutation=_permutation(), bootstrap=_bootstrap(), agree=True),
        causality_note="assert_backtest_causal passed (n_trials=20, seed=0)",
    )
    defaults.update(overrides)
    return StrategyReportInputs(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render_strategy_report: key sections and formatting
# ---------------------------------------------------------------------------


def test_render_includes_name_period_and_causality_note() -> None:
    # Arrange
    inputs = _inputs()

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "Test Strategy" in report
    assert "2020-01-01" in report
    assert "assert_backtest_causal passed (n_trials=20, seed=0)" in report


def test_render_formats_metrics_table_correctly() -> None:
    # Arrange
    inputs = _inputs()

    # Act
    report = render_strategy_report(inputs)

    # Assert: percentage-style metrics at 2 decimals
    assert "12.34%" in report
    assert "8.90%" in report
    assert "-15.67%" in report
    assert "55.00%" in report
    # ratio-style metrics at 2 decimals
    assert "1.23" in report
    assert "1.57" in report
    assert "0.89" in report
    assert "1.20" in report
    # drawdown duration
    assert "23" in report


def test_render_includes_significance_section() -> None:
    # Arrange
    inputs = _inputs()

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "0.0234" in report
    assert "0.0123" in report
    assert "0.0456" in report


def test_render_shows_disagreement_warning_when_agree_is_false() -> None:
    # Arrange
    inputs = _inputs(
        significance=SignificanceCheck(permutation=_permutation(), bootstrap=_bootstrap(), agree=False)
    )

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "not established" in report.lower() or "未確立" in report


def test_render_does_not_warn_when_agree_is_true() -> None:
    # Arrange
    inputs = _inputs(significance=SignificanceCheck(permutation=_permutation(), bootstrap=_bootstrap(), agree=True))

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "not established" not in report.lower() and "未確立" not in report


# ---------------------------------------------------------------------------
# Optional sections
# ---------------------------------------------------------------------------


def test_render_includes_benchmark_section_when_provided() -> None:
    # Arrange
    from core.evaluation.metrics import BenchmarkComparison

    benchmark = BenchmarkComparison(alpha=0.03, beta=0.8, r_squared=0.65, alpha_pvalue=0.01, beta_pvalue=0.0001)
    inputs = _inputs(benchmark=benchmark)

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "Benchmark" in report
    assert "0.80" in report  # beta


def test_render_omits_benchmark_section_when_not_provided() -> None:
    # Arrange
    inputs = _inputs()

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "Benchmark Comparison" not in report


def test_render_includes_limitations_list() -> None:
    # Arrange
    inputs = _inputs(limitations=["Level A survivorship bias", "single cost model"])

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "Level A survivorship bias" in report
    assert "single cost model" in report


def test_render_includes_description_when_provided() -> None:
    # Arrange
    inputs = _inputs(description="Tested on S&P 500 universe, 2015-2025, 5bps slippage.")

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "What Was Tested" in report
    assert "Tested on S&P 500 universe, 2015-2025, 5bps slippage." in report


def test_render_includes_walk_forward_note_when_provided() -> None:
    # Arrange
    inputs = _inputs(walk_forward_note="IS 2015-2022, OOS 2023-2025, degradation=0.3")

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "IS 2015-2022, OOS 2023-2025, degradation=0.3" in report


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_render_handles_nan_metrics_as_na_not_raw_nan() -> None:
    # Arrange: calmar_ratio undefined (no drawdown at all).
    metrics = _metrics()
    metrics["calmar_ratio"] = float("nan")
    inputs = _inputs(metrics=metrics)

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "N/A" in report
    assert "nan" not in report.lower()


def test_render_handles_nan_percentage_and_duration_metrics_as_na() -> None:
    # Arrange: annualized_return NaN (percentage formatter) and
    # max_drawdown_duration NaN (duration formatter) -- distinct code
    # paths from the ratio-formatter NaN case above.
    metrics = _metrics()
    metrics["annualized_return"] = float("nan")
    metrics["max_drawdown_duration"] = float("nan")
    inputs = _inputs(metrics=metrics)

    # Act
    report = render_strategy_report(inputs)

    # Assert
    assert "N/A" in report
    assert "nan" not in report.lower()


# ---------------------------------------------------------------------------
# write_strategy_report
# ---------------------------------------------------------------------------


def test_write_strategy_report_creates_file_with_rendered_content(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Arrange
    inputs = _inputs()
    path = tmp_path / "report.md"

    # Act
    write_strategy_report(inputs, path)

    # Assert
    assert path.exists()
    assert path.read_text(encoding="utf-8") == render_strategy_report(inputs)
