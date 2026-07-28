"""Parameter sensitivity analysis and reporting.

SensitivityGridPoint is the shared result type produced by:
- strategies.pairs_trading.sensitivity.run_entry_exit_threshold_grid /
  run_correlation_prefilter_grid
- strategies.factor_momentum.sensitivity.run_percentile_grid

render_sensitivity_report deliberately does NOT rank, sort, or highlight
a "best" configuration. Checking multiple parameter configurations is
itself a form of multiple comparisons -- the same lesson as
core.evaluation.walk_forward's per-window caution, applied to parameter
space instead of time. A single configuration's apparent success (or
apparent significance) must not be read as validating that specific
parameter choice; the default configuration was chosen a priori, before
running any grid, from independent established conventions -- not tuned
to this data. Ranking by Sharpe would actively invite exactly the
cherry-picking this report exists to guard against, so the report
preserves whatever order `points` is given in and never bolds/emphasizes
any row's numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.evaluation.statistical_tests import SignificanceCheck


@dataclass(frozen=True)
class SensitivityGridPoint:
    """One grid point's result. oos_metrics/significance are both None
    together when nothing was found for that configuration (e.g. no
    statistically-justified pair survived selection at that
    correlation_prefilter value). is_default marks the a-priori-chosen
    configuration for orientation only -- it is not a ranking signal and
    must not be rendered as a recommendation."""

    params: dict[str, float]
    label: str
    oos_metrics: dict[str, float] | None
    significance: SignificanceCheck | None
    is_default: bool = False


def render_sensitivity_report(analysis_name: str, points: Sequence[SensitivityGridPoint]) -> str:
    """Render a Markdown sensitivity report: one row per grid point, in
    the SAME ORDER as `points` (never sorted/ranked), plus a consistency
    summary with a standing multiple-comparisons caution."""
    if not points:
        raise ValueError("render_sensitivity_report requires at least one grid point")

    param_names = list(points[0].params.keys())
    header = "| " + " | ".join(param_names) + " | Result | OOS Sharpe | Permutation p | Agreement |"
    separator = "|" + "---|" * (len(param_names) + 4)

    lines: list[str] = [f"# {analysis_name} — Sensitivity Analysis", "", "## Grid", "", header, separator]

    for point in points:
        param_cells = [f"{point.params[name]:.3g}" for name in param_names]
        label = f"{point.label} (default)" if point.is_default else point.label
        if point.oos_metrics is None or point.significance is None:
            row = "| " + " | ".join(param_cells) + f" | {label} | N/A | N/A | N/A |"
        else:
            sharpe = point.oos_metrics["sharpe_ratio"]
            p_value = point.significance.permutation.p_value
            agree = "Yes" if point.significance.agree else "No"
            row = "| " + " | ".join(param_cells) + f" | {label} | {sharpe:.2f} | {p_value:.4f} | {agree} |"
        lines.append(row)

    lines += ["", "## Consistency Summary", ""]

    populated = [p for p in points if p.oos_metrics is not None]
    lines.append(f"- Configurations with a result: {len(populated)}/{len(points)}")

    if populated:
        significant = [
            p for p in populated if p.significance is not None and p.significance.permutation.p_value < 0.05
        ]
        lines.append(f"- Configurations with permutation p < 0.05: {len(significant)}/{len(populated)}")

    lines.append(
        "- **Checking multiple parameter configurations is itself a form of multiple comparisons.** "
        "A single configuration's apparent success (or apparent significance) should not be treated as "
        "validating that specific parameter choice without correction. The default configuration was "
        "chosen a priori (before running this grid) based on independent, established conventions -- "
        "not tuned to this data. This report deliberately does not rank or highlight a \"best\" "
        "configuration."
    )

    lines += [
        "",
        "---",
        "*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above "
        "before drawing any conclusion.*",
    ]
    return "\n".join(lines)
