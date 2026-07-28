"""Walk-forward window evaluation and reporting (REQUIREMENTS.md 8,
Phase 6 "拡張": walk-forward robustness analysis).

WalkForwardWindowResult is the SHARED result type produced by both:
- evaluate_walk_forward_windows (this module): the generic, strategy-
  agnostic case -- slices an ALREADY-COMPUTED continuous returns series
  (from a single run_backtest call) into multiple windows. Used directly
  by any continuously-run strategy with no per-window recalibration step
  (e.g. FactorMomentumStrategy -- see strategies/factor_momentum/
  README.md for why it has nothing to re-select per window).
- strategies.pairs_trading.walk_forward.run_pairs_trading_walk_forward:
  the strategy-specific case -- re-runs select_pairs (a genuine one-time
  selection decision) at each window's in-sample cutoff.

Sharing one result type lets render_walk_forward_report render either
without knowing which produced it -- deliberately a NEW function, not a
reuse of core.evaluation.report.render_comparison_report: that function's
period-mismatch warning exists to flag comparing different STRATEGIES
over different time periods (a caveat); here, different periods across
windows are the entire point of a walk-forward table, not a caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from core.backtest.sample_split import TrainTestSplit, split_returns
from core.evaluation import metrics
from core.evaluation.statistical_tests import SignificanceCheck, check_significance


@dataclass(frozen=True)
class WalkForwardWindowResult:
    """One window's result. is_metrics/oos_metrics/significance are all
    None together when nothing was found for that window (e.g. no
    statistically-justified pair survived selection) -- see
    strategies.pairs_trading.walk_forward for that case."""

    window: TrainTestSplit
    label: str
    is_metrics: dict[str, float] | None
    oos_metrics: dict[str, float] | None
    significance: SignificanceCheck | None


def evaluate_walk_forward_windows(
    returns: pd.Series, windows: Sequence[TrainTestSplit], label: str = "strategy"
) -> list[WalkForwardWindowResult]:
    """Slice an already-computed continuous returns series into multiple
    IS/OOS windows. No new backtest is run -- purely a generic wiring
    utility over split_returns/metrics.summary/check_significance."""
    results = []
    for window in windows:
        is_returns, oos_returns = split_returns(returns, window)
        results.append(
            WalkForwardWindowResult(
                window=window,
                label=label,
                is_metrics=metrics.summary(is_returns),
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, seed=0),
            )
        )
    return results


def render_walk_forward_report(strategy_name: str, results: Sequence[WalkForwardWindowResult]) -> str:
    """Render a Markdown walk-forward report: one row per window, plus a
    consistency summary across windows (fraction populated, OOS Sharpe
    distribution, and a standing reminder that checking multiple windows
    is itself a form of multiple comparisons)."""
    if not results:
        raise ValueError("render_walk_forward_report requires at least one window result")

    lines: list[str] = [f"# {strategy_name} — Walk-Forward Report", "", "## Windows", ""]
    lines += [
        "| Window | IS End | OOS Period | Result | OOS Sharpe | Permutation p | Agreement |",
        "|---|---|---|---|---|---|---|",
    ]

    for i, result in enumerate(results, start=1):
        is_end = str(result.window.in_sample.end)
        oos_period = f"{result.window.out_of_sample.start} to {result.window.out_of_sample.end}"
        if result.oos_metrics is None or result.significance is None:
            lines.append(f"| {i} | {is_end} | {oos_period} | {result.label} | N/A | N/A | N/A |")
        else:
            sharpe = result.oos_metrics["sharpe_ratio"]
            p_value = result.significance.permutation.p_value
            agree = "Yes" if result.significance.agree else "No"
            lines.append(f"| {i} | {is_end} | {oos_period} | {result.label} | {sharpe:.2f} | {p_value:.4f} | {agree} |")

    lines += ["", "## Consistency Summary", ""]

    populated = [r for r in results if r.oos_metrics is not None]
    lines.append(f"- Windows with a result: {len(populated)}/{len(results)}")

    if populated:
        # The `if r.oos_metrics is not None` re-check (redundant given
        # `populated`'s own filter) lets mypy narrow within this
        # comprehension -- it cannot narrow through populated's separate
        # list-comprehension filter above.
        sharpes = [r.oos_metrics["sharpe_ratio"] for r in populated if r.oos_metrics is not None]
        lines.append(
            f"- OOS Sharpe among populated windows: mean={sum(sharpes) / len(sharpes):.2f}, "
            f"min={min(sharpes):.2f}, max={max(sharpes):.2f}"
        )
        significant = [
            r for r in populated if r.significance is not None and r.significance.permutation.p_value < 0.05
        ]
        lines.append(
            f"- Windows with permutation p < 0.05: {len(significant)}/{len(populated)} -- checking "
            "multiple windows is itself a form of multiple comparisons; a single window's apparent "
            "significance should not be treated as confirmed edge without correction."
        )
    else:
        lines.append(
            "- No window produced a populated result (e.g. no statistically-justified candidate "
            "survived selection in any window)."
        )

    lines += [
        "",
        "---",
        "*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above "
        "before drawing any conclusion.*",
    ]
    return "\n".join(lines)
