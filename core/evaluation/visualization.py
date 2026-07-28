"""Supplementary matplotlib/plotly visualizations, independent of
core.evaluation.report's Markdown generation (that module is unchanged).

These plots exist to communicate the HONESTY of the verification process,
not to make results look better than they are:

- Missing/None results (e.g. a walk-forward window with "no candidate")
  are plotted as gaps (NaN), never interpolated across.
- Sensitivity-grid points are drawn with identical color/size for every
  point; the a-priori "default" configuration is distinguished ONLY by
  marker SHAPE, mirroring core.evaluation.sensitivity.render_sensitivity_
  report's rule of never ranking or highlighting a "best" result.
- Axes are not rescaled/zoomed to exaggerate small differences (e.g.
  equity curves always share a 0%-anchored y-axis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

from core.evaluation.sensitivity import SensitivityGridPoint
from core.evaluation.walk_forward import WalkForwardWindowResult

_DEFAULT_COLOR = "tab:blue"
_OTHER_MARKER = "o"
_DEFAULT_MARKER = "s"
_MARKER_SIZE = 60
_OTHER_MARKER_PLOTLY = "circle"
_DEFAULT_MARKER_PLOTLY = "square"
_DEFAULT_COLOR_PLOTLY = "steelblue"  # plotly needs a CSS color name, unlike matplotlib's "tab:blue"


def plot_walk_forward_sharpe(results_by_strategy: Mapping[str, Sequence[WalkForwardWindowResult]]) -> Figure:
    """One line per strategy: OOS Sharpe vs window index. A window with
    oos_metrics=None (e.g. no statistically-justified candidate) is
    plotted as NaN, which matplotlib renders as a gap, not an
    interpolated line segment."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, results in results_by_strategy.items():
        x = list(range(1, len(results) + 1))
        y = [r.oos_metrics["sharpe_ratio"] if r.oos_metrics is not None else float("nan") for r in results]
        ax.plot(x, y, marker="o", label=label)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Window")
    ax.set_ylabel("OOS Sharpe Ratio")
    ax.set_title("Walk-Forward OOS Sharpe by Window")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sensitivity_grid(grids: Mapping[str, Sequence[SensitivityGridPoint]]) -> Figure:
    """One subplot per grid: OOS Sharpe vs the grid's single parameter.
    All points share one color/size; the is_default point is drawn as a
    separate scatter group with a different MARKER SHAPE only -- never a
    different color, size, or highlight."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (label, points) in zip(axes, grids.items()):
        param_name = next(iter(points[0].params))
        xs = [p.params[param_name] for p in points]
        ys = [p.oos_metrics["sharpe_ratio"] if p.oos_metrics is not None else float("nan") for p in points]
        ax.plot(xs, ys, color=_DEFAULT_COLOR, linewidth=1, zorder=1)

        others = [(x, y) for p, x, y in zip(points, xs, ys) if not p.is_default]
        defaults = [(x, y) for p, x, y in zip(points, xs, ys) if p.is_default]
        if others:
            ox, oy = zip(*others)
            ax.scatter(ox, oy, marker=_OTHER_MARKER, color=_DEFAULT_COLOR, s=_MARKER_SIZE, zorder=2, label="grid point")
        if defaults:
            dx, dy = zip(*defaults)
            ax.scatter(dx, dy, marker=_DEFAULT_MARKER, color=_DEFAULT_COLOR, s=_MARKER_SIZE, zorder=2, label="default")

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(param_name)
        ax.set_ylabel("OOS Sharpe Ratio")
        ax.set_title(label)
        ax.legend()

    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class EquityCurveSeries:
    """One strategy's OOS return series for equity-curve comparison.
    `caveat`, when set, is appended to the legend label verbatim (e.g.
    AEP-FE's "reference only, not significant" disclosure) -- it does
    not affect the plotted data, only the legend text."""

    returns: pd.Series
    label: str
    caveat: str | None = None


def plot_equity_curves(curves: Sequence[EquityCurveSeries]) -> Figure:
    """Cumulative return (1+returns).cumprod()-1 per series, all on one
    shared, 0%-anchored y-axis -- no per-series rescaling that would make
    curves harder to compare directly."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for curve in curves:
        cumulative = (1 + curve.returns).cumprod() - 1
        label = f"{curve.label} ({curve.caveat})" if curve.caveat else curve.label
        ax.plot(cumulative.index, cumulative.to_numpy(), label=label)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("OOS Equity Curve Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class BonferroniScanPoint:
    """One test-family scan: n_tests examined, and survivor counts
    before/after Bonferroni correction."""

    n_tests: int
    survivors_before: int
    survivors_after: int
    label: str


def plot_multiple_testing_bonferroni(
    raw_p: float, scans: Sequence[BonferroniScanPoint], alpha: float = 0.05
) -> Figure:
    """Left panel: adjusted_p = min(raw_p * n_tests, 1) as a function of
    n_tests (log scale), with the ACTUAL scanned n_tests values marked,
    plus an alpha reference line. Right panel: grouped bars of survivor
    counts before/after correction, per scan."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    n_values = [s.n_tests for s in scans]
    n_range = np.linspace(min(n_values) * 0.5, max(n_values) * 1.5, 200)
    adjusted_curve = np.minimum(raw_p * n_range, 1.0)
    ax_left.plot(n_range, adjusted_curve, color=_DEFAULT_COLOR)

    adjusted_points = [min(raw_p * n, 1.0) for n in n_values]
    ax_left.scatter(n_values, adjusted_points, color="black", zorder=3)

    ax_left.axhline(alpha, color="red", linestyle="--", label=f"alpha={alpha}")
    ax_left.set_xscale("log")
    ax_left.set_xlabel("n_tests")
    ax_left.set_ylabel("Bonferroni-adjusted p-value")
    ax_left.set_title("Same raw p-value, different test family")
    ax_left.legend()

    x = np.arange(len(scans))
    width = 0.35
    ax_right.bar(x - width / 2, [s.survivors_before for s in scans], width, label="before correction")
    ax_right.bar(x + width / 2, [s.survivors_after for s in scans], width, label="after correction")
    ax_right.set_xticks(list(x))
    ax_right.set_xticklabels([s.label for s in scans])
    ax_right.set_ylabel("Surviving candidates")
    ax_right.set_title("Survivor count before/after Bonferroni correction")
    ax_right.legend()

    fig.tight_layout()
    return fig


def _entry_exit_xyz(points: Sequence[SensitivityGridPoint]) -> tuple[list[float], list[float], list[float]]:
    # entry_threshold x exit_threshold is backtest-only (no re-selection
    # step, see strategies/pairs_trading/sensitivity.py), so oos_metrics
    # is always populated -- unlike correlation_prefilter's grid, which
    # can have a "no candidate" point. Asserted (not just assumed) so
    # mypy can narrow away the Optional.
    for p in points:
        assert p.oos_metrics is not None, "entry/exit threshold grid points are always backtested"
    return (
        [p.params["entry_threshold"] for p in points],
        [p.params["exit_threshold"] for p in points],
        [p.oos_metrics["sharpe_ratio"] for p in points],  # type: ignore[index]
    )


def plot_entry_exit_3d_static(points: Sequence[SensitivityGridPoint]) -> Figure:
    """Raw 3D scatter of the actual grid points tested -- no interpolated
    surface. is_default distinguished by marker shape only, same as
    plot_sensitivity_grid."""
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    others = [p for p in points if not p.is_default]
    defaults = [p for p in points if p.is_default]

    if others:
        ox, oy, oz = _entry_exit_xyz(others)
        ax.scatter(ox, oy, oz, marker=_OTHER_MARKER, color=_DEFAULT_COLOR, s=_MARKER_SIZE, label="grid point")
    if defaults:
        dx, dy, dz = _entry_exit_xyz(defaults)
        ax.scatter(dx, dy, dz, marker=_DEFAULT_MARKER, color=_DEFAULT_COLOR, s=_MARKER_SIZE, label="default")

    ax.set_xlabel("entry_threshold")
    ax.set_ylabel("exit_threshold")
    ax.set_zlabel("OOS Sharpe Ratio")
    ax.set_title("Pairs Trading: Entry/Exit Threshold Grid (raw points only)")
    ax.legend()
    return fig


def plot_entry_exit_3d_interactive(points: Sequence[SensitivityGridPoint]) -> go.Figure:
    """Same raw grid points as plot_entry_exit_3d_static, as an
    interactive plotly Scatter3d with per-point hover text (OOS Sharpe +
    permutation p-value)."""
    xs, ys, zs = _entry_exit_xyz(points)
    for p in points:
        assert p.oos_metrics is not None and p.significance is not None, (
            "entry/exit threshold grid points are always backtested"
        )
    hover_text = [
        f"entry_threshold={p.params['entry_threshold']}, exit_threshold={p.params['exit_threshold']}<br>"
        f"OOS Sharpe={p.oos_metrics['sharpe_ratio']:.2f}<br>"  # type: ignore[index]
        f"permutation p={p.significance.permutation.p_value:.4f}"  # type: ignore[union-attr]
        for p in points
    ]
    symbols = [_DEFAULT_MARKER_PLOTLY if p.is_default else _OTHER_MARKER_PLOTLY for p in points]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                marker=dict(size=6, symbol=symbols, color=_DEFAULT_COLOR_PLOTLY),
                text=hover_text, hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="entry_threshold", yaxis_title="exit_threshold", zaxis_title="OOS Sharpe Ratio",
        ),
        title="Pairs Trading: Entry/Exit Threshold Grid (raw points only, interactive)",
    )
    return fig
