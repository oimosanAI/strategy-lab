"""Tests for core.evaluation.visualization.

Fixes the public API surface:

- plot_walk_forward_sharpe(results_by_strategy: dict[str,
  Sequence[WalkForwardWindowResult]]) -> Figure: one line per strategy,
  OOS Sharpe vs window index. A None oos_metrics window (e.g. "no
  candidate") must NOT be interpolated across -- it must show as a gap
  (NaN in the plotted ydata), not a straight line joining its neighbors.
- plot_sensitivity_grid(grids: dict[str, Sequence[SensitivityGridPoint]])
  -> Figure: one subplot per grid, OOS Sharpe vs the grid's single
  parameter. The is_default point must be visually distinguishable ONLY
  by marker shape (never color/size) from the other points -- this is
  the graphical equivalent of render_sensitivity_report's "never
  rank/highlight" rule.
- EquityCurveSeries(returns, label, caveat=None): caveat is an optional
  free-text legend annotation (e.g. AEP-FE's "reference only, not
  significant" note) -- it must appear in the legend only when set.
- plot_equity_curves(curves: Sequence[EquityCurveSeries]) -> Figure:
  cumulative return (1+returns).cumprod()-1 per series, all on one axes.
- BonferroniScanPoint(n_tests, survivors_before, survivors_after, label).
- plot_multiple_testing_bonferroni(raw_p, scans, alpha=0.05) -> Figure:
  left panel = adjusted-p-vs-n_tests curve with the real scan points
  marked + an alpha reference line; right panel = grouped before/after
  survivor-count bars.
- plot_entry_exit_3d_static(points: Sequence[SensitivityGridPoint]) ->
  Figure: matplotlib 3D scatter of the RAW grid points only (no
  interpolated surface). is_default distinguished by marker shape only.
- plot_entry_exit_3d_interactive(points) -> plotly.graph_objects.Figure:
  same data, plotly Scatter3d with per-point hover text.

matplotlib.use("Agg") is set at module import time (headless, no
display needed). All assertions are on Figure/Axes STRUCTURE (line
data, marker paths/colors/sizes, legend text) -- never pixel/image
comparison.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from core.evaluation.sensitivity import SensitivityGridPoint
from core.evaluation.statistical_tests import BootstrapResult, PermutationResult, SignificanceCheck
from core.evaluation.visualization import (
    BonferroniScanPoint,
    EquityCurveSeries,
    plot_entry_exit_3d_interactive,
    plot_entry_exit_3d_static,
    plot_equity_curves,
    plot_multiple_testing_bonferroni,
    plot_sensitivity_grid,
    plot_walk_forward_sharpe,
)
from core.evaluation.walk_forward import WalkForwardWindowResult


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


def _significance(p_value: float = 0.3) -> SignificanceCheck:
    return SignificanceCheck(permutation=_permutation(p_value), bootstrap=_bootstrap(), agree=True)


def _wf_result(sharpe: float | None, label: str = "strategy") -> WalkForwardWindowResult:
    from core.backtest.sample_split import SamplePeriod, TrainTestSplit

    window = TrainTestSplit(
        in_sample=SamplePeriod(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")),
        out_of_sample=SamplePeriod(pd.Timestamp("2020-07-01"), pd.Timestamp("2020-12-31")),
    )
    if sharpe is None:
        return WalkForwardWindowResult(window=window, label="no candidate", is_metrics=None, oos_metrics=None, significance=None)
    return WalkForwardWindowResult(
        window=window, label=label, is_metrics={"sharpe_ratio": 0.5},
        oos_metrics={"sharpe_ratio": sharpe}, significance=_significance(),
    )


def _sens_point(param_name: str, value: float, sharpe: float, is_default: bool = False) -> SensitivityGridPoint:
    return SensitivityGridPoint(
        params={param_name: value}, label="strategy",
        oos_metrics={"sharpe_ratio": sharpe}, significance=_significance(), is_default=is_default,
    )


def _entry_exit_point(entry: float, exit_: float, sharpe: float, is_default: bool = False) -> SensitivityGridPoint:
    return SensitivityGridPoint(
        params={"entry_threshold": entry, "exit_threshold": exit_}, label="AEP-FE",
        oos_metrics={"sharpe_ratio": sharpe}, significance=_significance(), is_default=is_default,
    )


# ---------------------------------------------------------------------------
# 1. plot_walk_forward_sharpe
# ---------------------------------------------------------------------------


def test_plot_walk_forward_sharpe_draws_one_line_per_strategy() -> None:
    results_by_strategy = {
        "pairs-trading": [_wf_result(1.0), _wf_result(-0.5)],
        "factor-momentum": [_wf_result(0.2), _wf_result(0.8)],
    }

    fig = plot_walk_forward_sharpe(results_by_strategy)

    ax = fig.axes[0]
    # ax.axhline(0.0, ...) also adds a Line2D to the same axes -- exclude
    # that flat reference line (checked separately below) when counting
    # per-strategy data lines.
    data_lines = [line for line in ax.get_lines() if len(set(line.get_ydata())) > 1]
    assert len(data_lines) == 2


def test_plot_walk_forward_sharpe_line_ydata_matches_input_sharpe() -> None:
    results_by_strategy = {"pairs-trading": [_wf_result(1.23), _wf_result(-0.45)]}

    fig = plot_walk_forward_sharpe(results_by_strategy)

    line = fig.axes[0].get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), [1.23, -0.45])


def test_plot_walk_forward_sharpe_does_not_interpolate_missing_window() -> None:
    # window 2 is "no candidate" (oos_metrics=None) -- must show as a gap,
    # not a straight line from window 1 to window 3.
    results_by_strategy = {"pairs-trading": [_wf_result(1.0), _wf_result(None), _wf_result(-1.0)]}

    fig = plot_walk_forward_sharpe(results_by_strategy)

    line = fig.axes[0].get_lines()[0]
    ydata = line.get_ydata()
    assert math.isnan(ydata[1])


def test_plot_walk_forward_sharpe_draws_zero_reference_line() -> None:
    results_by_strategy = {"pairs-trading": [_wf_result(1.0)]}

    fig = plot_walk_forward_sharpe(results_by_strategy)

    ax = fig.axes[0]
    horizontal_lines_at_zero = [
        line for line in ax.get_lines() if len(set(line.get_ydata())) == 1 and line.get_ydata()[0] == 0.0
    ]
    assert len(horizontal_lines_at_zero) >= 1


# ---------------------------------------------------------------------------
# 2. plot_sensitivity_grid
# ---------------------------------------------------------------------------


def test_plot_sensitivity_grid_draws_one_subplot_per_grid() -> None:
    grids = {
        "factor-momentum": [_sens_point("percentile", 0.1, 0.2), _sens_point("percentile", 0.2, 0.4)],
        "vol-arbitrage": [_sens_point("vrp_threshold", -0.05, 0.06), _sens_point("vrp_threshold", 0.0, 0.06)],
    }

    fig = plot_sensitivity_grid(grids)

    assert len(fig.axes) == 2


def test_plot_sensitivity_grid_line_ydata_matches_input_sharpe() -> None:
    grids = {"factor-momentum": [_sens_point("percentile", 0.1, 0.2), _sens_point("percentile", 0.2, 0.4)]}

    fig = plot_sensitivity_grid(grids)

    line = fig.axes[0].get_lines()[0]
    np.testing.assert_allclose(sorted(line.get_ydata()), sorted([0.2, 0.4]))


def test_plot_sensitivity_grid_default_point_differs_only_in_marker_shape() -> None:
    grids = {
        "factor-momentum": [
            _sens_point("percentile", 0.1, 0.2),
            _sens_point("percentile", 0.2, 0.9, is_default=True),  # "best looking" AND default
            _sens_point("percentile", 0.3, 0.1),
        ]
    }

    fig = plot_sensitivity_grid(grids)

    ax = fig.axes[0]
    collections = ax.collections
    assert len(collections) == 2  # one "other points" group, one "default" group
    other_coll, default_coll = collections[0], collections[1]
    # same color and size for both groups -- only the marker path differs
    np.testing.assert_allclose(other_coll.get_facecolor(), default_coll.get_facecolor())
    np.testing.assert_allclose(other_coll.get_sizes(), default_coll.get_sizes())
    assert other_coll.get_paths()[0].vertices.tolist() != default_coll.get_paths()[0].vertices.tolist()


def test_plot_sensitivity_grid_draws_zero_reference_line() -> None:
    grids = {"factor-momentum": [_sens_point("percentile", 0.1, 0.2)]}

    fig = plot_sensitivity_grid(grids)

    ax = fig.axes[0]
    horizontal_lines_at_zero = [
        line for line in ax.get_lines() if len(set(line.get_ydata())) == 1 and line.get_ydata()[0] == 0.0
    ]
    assert len(horizontal_lines_at_zero) >= 1


# ---------------------------------------------------------------------------
# 3. plot_equity_curves
# ---------------------------------------------------------------------------


def _returns(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2025-01-02", periods=len(values))
    return pd.Series(values, index=idx)


def test_plot_equity_curves_draws_one_line_per_series() -> None:
    curves = [
        EquityCurveSeries(returns=_returns([0.01, -0.005, 0.02]), label="factor-momentum"),
        EquityCurveSeries(returns=_returns([0.0, 0.01, 0.0]), label="vol-arbitrage"),
    ]

    fig = plot_equity_curves(curves)

    # ax.axhline(0.0, ...) also adds a Line2D to the same axes -- exclude
    # that flat reference line when counting per-series data lines.
    data_lines = [line for line in fig.axes[0].get_lines() if len(set(line.get_ydata())) > 1]
    assert len(data_lines) == 2


def test_plot_equity_curves_ydata_matches_cumulative_return() -> None:
    returns = _returns([0.10, -0.05, 0.02])
    curves = [EquityCurveSeries(returns=returns, label="factor-momentum")]

    fig = plot_equity_curves(curves)

    expected = ((1 + returns).cumprod() - 1).to_numpy()
    line = fig.axes[0].get_lines()[0]
    np.testing.assert_allclose(line.get_ydata(), expected)


def test_plot_equity_curves_includes_caveat_in_legend() -> None:
    curves = [
        EquityCurveSeries(
            returns=_returns([0.01]), label="pairs-trading (AEP-FE)",
            caveat="reference only, not significant at full-universe correction",
        ),
    ]

    fig = plot_equity_curves(curves)

    _, labels = fig.axes[0].get_legend_handles_labels()
    assert any("reference only" in label for label in labels)


def test_plot_equity_curves_omits_caveat_annotation_when_none() -> None:
    curves = [EquityCurveSeries(returns=_returns([0.01]), label="factor-momentum")]

    fig = plot_equity_curves(curves)

    _, labels = fig.axes[0].get_legend_handles_labels()
    assert labels == ["factor-momentum"]


# ---------------------------------------------------------------------------
# 4. plot_multiple_testing_bonferroni
# ---------------------------------------------------------------------------


def test_plot_multiple_testing_bonferroni_left_panel_matches_real_scan_points() -> None:
    raw_p = 7.666e-05
    scans = [
        BonferroniScanPoint(n_tests=308, survivors_before=18, survivors_after=1, label="2-sector"),
        BonferroniScanPoint(n_tests=5551, survivors_before=237, survivors_after=0, label="full-universe"),
    ]

    fig = plot_multiple_testing_bonferroni(raw_p, scans)

    ax_left = fig.axes[0]
    scatter_collections = [c for c in ax_left.collections if len(c.get_offsets()) > 0]
    assert len(scatter_collections) == 1
    offsets = scatter_collections[0].get_offsets()
    expected = [(308, min(raw_p * 308, 1.0)), (5551, min(raw_p * 5551, 1.0))]
    np.testing.assert_allclose(offsets, expected)


def test_plot_multiple_testing_bonferroni_left_panel_has_alpha_reference_line() -> None:
    scans = [BonferroniScanPoint(n_tests=308, survivors_before=18, survivors_after=1, label="2-sector")]

    fig = plot_multiple_testing_bonferroni(7.666e-05, scans, alpha=0.05)

    ax_left = fig.axes[0]
    alpha_lines = [line for line in ax_left.get_lines() if set(line.get_ydata()) == {0.05}]
    assert len(alpha_lines) >= 1


def test_plot_multiple_testing_bonferroni_right_panel_bars_match_survivor_counts() -> None:
    scans = [
        BonferroniScanPoint(n_tests=308, survivors_before=18, survivors_after=1, label="2-sector"),
        BonferroniScanPoint(n_tests=5551, survivors_before=237, survivors_after=0, label="full-universe"),
    ]

    fig = plot_multiple_testing_bonferroni(7.666e-05, scans)

    ax_right = fig.axes[1]
    heights = [p.get_height() for p in ax_right.patches]
    assert heights[:2] == [18, 237]
    assert heights[2:4] == [1, 0]


# ---------------------------------------------------------------------------
# 5. plot_entry_exit_3d_static
# ---------------------------------------------------------------------------


def test_plot_entry_exit_3d_static_plots_all_15_raw_points() -> None:
    points = [_entry_exit_point(e, x, e + x) for e in [1.5, 1.75, 2.0, 2.25, 2.5] for x in [0.0, 0.25, 0.5]]
    assert len(points) == 15

    fig = plot_entry_exit_3d_static(points)

    ax = fig.axes[0]
    total_points = sum(len(c._offsets3d[0]) for c in ax.collections)
    assert total_points == 15


def test_plot_entry_exit_3d_static_coordinates_match_input() -> None:
    points = [_entry_exit_point(2.0, 0.0, 1.5, is_default=True), _entry_exit_point(1.5, 0.25, -0.3)]

    fig = plot_entry_exit_3d_static(points)

    ax = fig.axes[0]
    all_x = [v for c in ax.collections for v in c._offsets3d[0]]
    all_y = [v for c in ax.collections for v in c._offsets3d[1]]
    all_z = [v for c in ax.collections for v in c._offsets3d[2]]
    assert sorted(all_x) == [1.5, 2.0]
    assert sorted(all_y) == [0.0, 0.25]
    assert sorted(all_z) == [-0.3, 1.5]


def test_plot_entry_exit_3d_static_default_point_differs_only_in_marker_shape() -> None:
    points = [_entry_exit_point(2.0, 0.0, 1.0, is_default=True), _entry_exit_point(1.5, 0.25, -0.3)]

    fig = plot_entry_exit_3d_static(points)

    ax = fig.axes[0]
    assert len(ax.collections) == 2
    other_coll, default_coll = ax.collections[0], ax.collections[1]
    np.testing.assert_allclose(other_coll.get_facecolor(), default_coll.get_facecolor())
    assert other_coll.get_paths()[0].vertices.tolist() != default_coll.get_paths()[0].vertices.tolist()


# ---------------------------------------------------------------------------
# 6. plot_entry_exit_3d_interactive
# ---------------------------------------------------------------------------


def test_plot_entry_exit_3d_interactive_is_scatter3d() -> None:
    points = [_entry_exit_point(2.0, 0.0, 1.0)]

    fig = plot_entry_exit_3d_interactive(points)

    assert fig.data[0].type == "scatter3d"


def test_plot_entry_exit_3d_interactive_coordinates_match_input() -> None:
    points = [_entry_exit_point(2.0, 0.0, 1.5), _entry_exit_point(1.5, 0.25, -0.3)]

    fig = plot_entry_exit_3d_interactive(points)

    assert list(fig.data[0].x) == [2.0, 1.5]
    assert list(fig.data[0].y) == [0.0, 0.25]
    assert list(fig.data[0].z) == [1.5, -0.3]


def test_plot_entry_exit_3d_interactive_hover_includes_permutation_p_value() -> None:
    points = [_entry_exit_point(2.0, 0.0, 1.5)]

    fig = plot_entry_exit_3d_interactive(points)

    assert "0.3" in fig.data[0].text[0]  # default _significance() p_value=0.3
