"""Tests for core.evaluation.sensitivity.

Fixes the public API surface:

- SensitivityGridPoint: one grid point's result (params dict, label,
  optional oos_metrics/significance for a "no result" case, is_default
  flag for orientation only -- not a ranking signal).
- render_sensitivity_report(analysis_name, points) -> str: deliberately
  does NOT rank, sort, or highlight a "best" configuration. Checking
  multiple parameter configurations is itself a form of multiple
  comparisons (same lesson as core.evaluation.walk_forward's per-window
  caution, applied to parameter space instead of time). The report
  preserves whatever order `points` was given in and never bolds/
  emphasizes any particular row -- verified here as explicit NEGATIVE
  tests, since this property could silently regress in a future "helpful"
  refactor (e.g. someone adding a sort-by-Sharpe for readability).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.evaluation.sensitivity import SensitivityGridPoint, render_sensitivity_report
from core.evaluation.statistical_tests import BootstrapResult, PermutationResult, SignificanceCheck


def _permutation(p_value: float = 0.3) -> PermutationResult:
    return PermutationResult(
        observed=0.01, p_value=p_value, n_permutations=2000, alternative="greater",
        null_distribution=np.array([0.0]), seed=0,
    )


def _bootstrap() -> BootstrapResult:
    return BootstrapResult(
        point_estimate=0.03, lower=-0.001, upper=0.001, confidence_level=0.95,
        standard_error=0.008, distribution=np.array([0.0]), n_resamples=2000, seed=0,
    )


def _point(
    params: dict[str, float], label: str, sharpe: float, p_value: float = 0.3, is_default: bool = False
) -> SensitivityGridPoint:
    return SensitivityGridPoint(
        params=params,
        label=label,
        oos_metrics={"sharpe_ratio": sharpe},
        significance=SignificanceCheck(permutation=_permutation(p_value), bootstrap=_bootstrap(), agree=True),
        is_default=is_default,
    )


def _empty_point(params: dict[str, float]) -> SensitivityGridPoint:
    return SensitivityGridPoint(params=params, label="no candidate", oos_metrics=None, significance=None)


# ---------------------------------------------------------------------------
# 1. basic rendering
# ---------------------------------------------------------------------------


def test_render_sensitivity_report_includes_grid_point_values() -> None:
    points = [_point({"entry_threshold": 2.0}, "AEP-FE", sharpe=1.23)]

    report = render_sensitivity_report("Entry Threshold Sensitivity", points)

    assert "Entry Threshold Sensitivity" in report
    assert "AEP-FE" in report
    assert "1.23" in report
    assert "2" in report  # the param value


def test_render_sensitivity_report_handles_empty_result_without_crashing() -> None:
    points = [_empty_point({"correlation_prefilter": 0.9})]

    report = render_sensitivity_report("Correlation Prefilter Sensitivity", points)

    assert "no candidate" in report
    assert "N/A" in report


def test_render_sensitivity_report_raises_on_empty_points() -> None:
    with pytest.raises(ValueError):
        render_sensitivity_report("Empty", [])


# ---------------------------------------------------------------------------
# 2. does NOT rank, sort, or highlight -- negative tests
# ---------------------------------------------------------------------------


def test_render_sensitivity_report_preserves_input_order_not_sorted_by_sharpe() -> None:
    # Deliberately non-monotonic Sharpe order: A=0.1, B=0.9, C=0.5. If the
    # report sorted (ascending or descending) by Sharpe, the row order
    # would differ from input order.
    points = [
        _point({"p": 1.0}, "A", sharpe=0.1),
        _point({"p": 2.0}, "B", sharpe=0.9),
        _point({"p": 3.0}, "C", sharpe=0.5),
    ]

    report = render_sensitivity_report("Order Check", points)

    label_positions = {label: report.index(f"| {label} |") for label in ("A", "B", "C")}
    assert label_positions["A"] < label_positions["B"] < label_positions["C"]


def test_render_sensitivity_report_does_not_bold_or_highlight_any_cell() -> None:
    # Scoped to the "## Grid" table specifically -- the Consistency
    # Summary's multiple-comparisons CAUTION text legitimately uses bold
    # to emphasize the warning itself, which is a different thing from
    # highlighting a "best" result row.
    points = [
        _point({"p": 1.0}, "A", sharpe=0.1),
        _point({"p": 2.0}, "B", sharpe=2.5),  # the "best" looking one
    ]

    report = render_sensitivity_report("Highlight Check", points)

    grid_section = report.split("## Consistency Summary")[0]
    assert "**" not in grid_section


# ---------------------------------------------------------------------------
# 3. (default) annotation
# ---------------------------------------------------------------------------


def test_render_sensitivity_report_marks_only_the_default_point() -> None:
    points = [
        _point({"p": 1.0}, "A", sharpe=0.1),
        _point({"p": 2.0}, "B", sharpe=0.5, is_default=True),
        _point({"p": 3.0}, "C", sharpe=0.9),
    ]

    report = render_sensitivity_report("Default Marker", points)

    assert report.count("(default)") == 1
    assert "B (default)" in report


# ---------------------------------------------------------------------------
# 4. Consistency Summary + multiple-comparisons caution
# ---------------------------------------------------------------------------


def test_render_sensitivity_report_includes_multiple_comparisons_caution() -> None:
    points = [_point({"p": 1.0}, "A", sharpe=0.1)]

    report = render_sensitivity_report("Caution Check", points)

    assert "multiple" in report.lower()
    assert "does not rank" in report.lower() or "not rank" in report.lower()


def test_render_sensitivity_report_consistency_summary_counts_correctly() -> None:
    points = [
        _point({"p": 1.0}, "A", sharpe=0.1, p_value=0.6),
        _point({"p": 2.0}, "B", sharpe=0.5, p_value=0.02),
        _empty_point({"p": 3.0}),
    ]

    report = render_sensitivity_report("Count Check", points)

    assert "2/3" in report  # populated / total
    assert "1/2" in report  # p < 0.05 / populated
