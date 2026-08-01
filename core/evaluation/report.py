"""Human-readable Markdown performance reports.

Deliberately decoupled from core.backtest, same as metrics.py and
statistical_tests.py: StrategyReportInputs is built from plain
pd.Series/dict/dataclass pieces, never a BacktestResult. This module's
only job is to render already-computed evaluation results legibly; it
does not compute anything itself and it does not (cannot) re-verify
causality or significance.

`causality_note` is a REQUIRED field on StrategyReportInputs. This module
has no dependency on core.backtest, so it cannot call assert_causal
itself -- the caller must state, in their own words, what causality
verification was actually done (or write "NOT VERIFIED" if none was).
An empty or omitted claim is not an option. This mirrors, one level up
the stack, assert_causal's own philosophy: a property is only credible
once someone has actually checked it, not merely assumed it.

Per the project's backtest-verification standard, a Sharpe ratio is not
evidence of edge by itself -- every rendered report repeats that
reminder, and the significance section renders a prominent warning
whenever the sign-flip permutation test and the moving-block bootstrap CI
disagree (SignificanceCheck.agree is False), rather than only in
docstrings a reader might not see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.evaluation.metrics import BenchmarkComparison
from core.evaluation.statistical_tests import PermutationResult, SignificanceCheck


@dataclass(frozen=True)
class StrategyReportInputs:
    """Everything needed to render a single-strategy report.

    Attributes
    ----------
    name:
        Strategy identifier, used as the report title.
    returns:
        Periodic returns; only its index bounds (period start/end) and
        length are used for display.
    metrics:
        The dict produced by core.evaluation.metrics.summary().
    significance:
        The composite result from
        core.evaluation.statistical_tests.check_significance().
    causality_note:
        Required, free text. What causality verification was actually
        performed (e.g. "assert_backtest_causal passed (n_trials=20,
        seed=0)"), or an honest "NOT VERIFIED" if none was.
    benchmark:
        Optional OLS alpha/beta comparison; the section is omitted
        entirely when not provided.
    walk_forward_note:
        Optional free text for Pillar 3 (in-sample vs out-of-sample
        degradation); not yet produced by any module in this repo, so
        typically None until a walk-forward harness exists.
    limitations:
        Free-text bullet points (e.g. survivorship bias, cost-model
        simplifications).
    description:
        Optional free text: what was tested, on what data, under what
        cost assumptions.
    """

    name: str
    returns: pd.Series
    metrics: dict[str, float]
    significance: SignificanceCheck
    causality_note: str
    benchmark: BenchmarkComparison | None = None
    walk_forward_note: str | None = None
    limitations: list[str] = field(default_factory=list)
    description: str | None = None


def _fmt_pct(value: float, decimals: int = 2) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def _fmt_ratio(value: float, decimals: int = 2) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def _fmt_pvalue(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _fmt_dropped_draws(permutation: PermutationResult) -> str:
    """Disclose discarded permutation draws, or nothing when none were.

    Silent in the overwhelmingly common case (no draw is undefined), so
    the line stays readable; explicit the moment the p-value rests on
    fewer draws than were requested.
    """
    dropped = permutation.n_permutations - permutation.null_distribution.size
    if dropped <= 0:
        return ""
    return f" of {permutation.n_permutations} requested; {dropped} discarded as undefined"


def _fmt_duration(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{int(value)} days"


def render_strategy_report(inputs: StrategyReportInputs) -> str:
    """Render a Markdown report string for a single strategy."""
    m = inputs.metrics
    sig = inputs.significance

    lines: list[str] = [f"# {inputs.name} — Performance Report", ""]

    period_start = inputs.returns.index[0]
    period_end = inputs.returns.index[-1]
    lines.append(f"**Period**: {period_start} to {period_end} ({len(inputs.returns)} observations)")
    lines.append("")

    if inputs.description:
        lines += ["## What Was Tested", "", inputs.description, ""]

    lines += ["## Causality (Pillar 1)", "", inputs.causality_note, ""]

    lines += [
        "## Performance Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Annualized Return | {_fmt_pct(m['annualized_return'])} |",
        f"| Annualized Volatility | {_fmt_pct(m['annualized_volatility'])} |",
        f"| Sharpe Ratio | {_fmt_ratio(m['sharpe_ratio'])} |",
        f"| Sortino Ratio | {_fmt_ratio(m['sortino_ratio'])} |",
        f"| Calmar Ratio | {_fmt_ratio(m['calmar_ratio'])} |",
        f"| Max Drawdown | {_fmt_pct(m['max_drawdown'])} ({_fmt_duration(m['max_drawdown_duration'])}) |",
        f"| Win Rate | {_fmt_pct(m['win_rate'])} |",
        f"| Profit/Loss Ratio | {_fmt_ratio(m['profit_loss_ratio'])} |",
        "",
    ]

    lines += [
        "## Statistical Significance (Pillar 2)",
        "",
        # n is the SURVIVING draw count (null_distribution.size), which is
        # what the p-value's denominator actually was; n_permutations is
        # only what was requested. Where they differ, both are shown --
        # quietly printing the larger one would overstate the evidence.
        f"- Sign-flip permutation test: p = {_fmt_pvalue(sig.permutation.p_value)} "
        f"(n={sig.permutation.null_distribution.size}{_fmt_dropped_draws(sig.permutation)}, "
        f"alternative={sig.permutation.alternative})",
        f"- Moving-block bootstrap {sig.bootstrap.confidence_level:.0%} CI: "
        f"[{sig.bootstrap.lower:.4f}, {sig.bootstrap.upper:.4f}] (n_resamples={sig.bootstrap.n_resamples}; "
        f"tail-matched to the {sig.permutation.alternative} test above, which is why a one-sided "
        "test is paired with a 90% and not a 95% interval)",
    ]
    if sig.agree:
        lines.append("- **Agreement: Yes**")
    else:
        lines.append(
            "- **Agreement: No** — the permutation test and bootstrap CI disagree. "
            "Treat this result as **not established**, whichever of the two claims "
            "significance, and do not average the two numbers together. Two methods "
            "disagreeing about the same null indicate an unstable estimate; this is a "
            "weaker position than a concordant null, not a firmer one."
        )
    lines.append("")

    if inputs.benchmark is not None:
        b = inputs.benchmark
        lines += [
            "## Benchmark Comparison",
            "",
            f"- Alpha (annualized): {_fmt_pct(b.alpha)} (p = {_fmt_pvalue(b.alpha_pvalue)})",
            f"- Beta: {_fmt_ratio(b.beta)} (p = {_fmt_pvalue(b.beta_pvalue)})",
            f"- R²: {_fmt_ratio(b.r_squared)}",
            "",
        ]

    lines += [
        "## Walk-Forward / Over-fitting (Pillar 3)",
        "",
        inputs.walk_forward_note or "Not yet performed.",
        "",
    ]

    lines += ["## Known Limitations", ""]
    if inputs.limitations:
        lines += [f"- {item}" for item in inputs.limitations]
    else:
        lines.append("- None noted.")
    lines.append("")

    lines += [
        "---",
        "*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above "
        "before drawing any conclusion.*",
    ]

    return "\n".join(lines)


def write_strategy_report(inputs: StrategyReportInputs, path: Path) -> None:
    """Render and write a single-strategy report to ``path`` (UTF-8)."""
    path.write_text(render_strategy_report(inputs), encoding="utf-8")


def render_comparison_report(strategies: Sequence[StrategyReportInputs]) -> str:
    """Render a side-by-side Markdown comparison of multiple strategies
    (REQUIREMENTS.md 6: "戦略間比較").

    Every section iterates over ``strategies`` directly -- there is no
    hardcoded column count, so this scales to any number of strategies
    (2 today, more once vol_arbitrage or others join) without a rewrite.

    Same "render, don't enforce" philosophy as render_strategy_report:
    this function does not re-verify anything. A period mismatch across
    strategies renders a warning banner rather than raising (mirroring
    how a significance disagreement is rendered as a warning, not an
    exception) -- comparing across different periods may be a caveat a
    caller knowingly accepts, not necessarily a mistake. An EMPTY
    sequence, by contrast, is a genuine unusable state (there is nothing
    to compare) and raises ValueError.
    """
    if not strategies:
        raise ValueError("render_comparison_report requires at least one strategy")

    names = [s.name for s in strategies]
    header = "| | " + " | ".join(names) + " |"
    separator = "|---|" + "---|" * len(names)

    def _row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |"

    lines: list[str] = ["# Strategy Comparison Report", "", f"**Strategies compared**: {len(strategies)}", ""]

    periods = {(s.returns.index[0], s.returns.index[-1]) for s in strategies}
    if len(periods) > 1:
        lines += [
            "⚠️ **Period mismatch**: the strategies below were evaluated over different "
            "date ranges -- comparing metrics across mismatched periods is not "
            "apples-to-apples.",
            "",
        ]

    lines += ["## Period", "", header, separator]
    lines.append(_row("Start", [str(s.returns.index[0]) for s in strategies]))
    lines.append(_row("End", [str(s.returns.index[-1]) for s in strategies]))
    lines.append(_row("Observations", [str(len(s.returns)) for s in strategies]))
    lines.append("")

    lines += ["## Causality (Pillar 1)", ""]
    lines += [f"- **{s.name}**: {s.causality_note}" for s in strategies]
    lines.append("")

    lines += ["## Performance Metrics", "", header, separator]
    lines.append(_row("Annualized Return", [_fmt_pct(s.metrics["annualized_return"]) for s in strategies]))
    lines.append(_row("Annualized Volatility", [_fmt_pct(s.metrics["annualized_volatility"]) for s in strategies]))
    lines.append(_row("Sharpe Ratio", [_fmt_ratio(s.metrics["sharpe_ratio"]) for s in strategies]))
    lines.append(_row("Sortino Ratio", [_fmt_ratio(s.metrics["sortino_ratio"]) for s in strategies]))
    lines.append(_row("Calmar Ratio", [_fmt_ratio(s.metrics["calmar_ratio"]) for s in strategies]))
    lines.append(
        _row(
            "Max Drawdown",
            [
                f"{_fmt_pct(s.metrics['max_drawdown'])} ({_fmt_duration(s.metrics['max_drawdown_duration'])})"
                for s in strategies
            ],
        )
    )
    lines.append(_row("Win Rate", [_fmt_pct(s.metrics["win_rate"]) for s in strategies]))
    lines.append(_row("Profit/Loss Ratio", [_fmt_ratio(s.metrics["profit_loss_ratio"]) for s in strategies]))
    lines.append("")

    lines += ["## Statistical Significance (Pillar 2)", "", header, separator]
    lines.append(
        _row("Permutation p-value", [_fmt_pvalue(s.significance.permutation.p_value) for s in strategies])
    )
    lines.append(
        _row(
            "Bootstrap CI",
            [f"[{s.significance.bootstrap.lower:.4f}, {s.significance.bootstrap.upper:.4f}]" for s in strategies],
        )
    )
    lines.append(_row("Agreement", ["Yes" if s.significance.agree else "No ⚠️" for s in strategies]))
    lines.append("")

    disagreeing = [s.name for s in strategies if not s.significance.agree]
    for name in disagreeing:
        lines.append(
            f"⚠️ **{name}**: permutation test and bootstrap CI disagree — treat this "
            "result as **not established**, whichever of the two claims significance, "
            "and do not average the two numbers together."
        )
    if disagreeing:
        lines.append("")

    if any(s.benchmark is not None for s in strategies):
        lines += ["## Benchmark Comparison", "", header, separator]
        lines.append(
            _row(
                "Alpha (annualized)",
                [_fmt_pct(s.benchmark.alpha) if s.benchmark is not None else "N/A" for s in strategies],
            )
        )
        lines.append(
            _row("Beta", [_fmt_ratio(s.benchmark.beta) if s.benchmark is not None else "N/A" for s in strategies])
        )
        lines.append(
            _row("R²", [_fmt_ratio(s.benchmark.r_squared) if s.benchmark is not None else "N/A" for s in strategies])
        )
        lines.append("")

    lines += ["## Walk-Forward / Over-fitting (Pillar 3)", ""]
    lines += [f"- **{s.name}**: {s.walk_forward_note or 'Not yet performed.'}" for s in strategies]
    lines.append("")

    lines += ["## Known Limitations", ""]
    for s in strategies:
        lines.append(f"### {s.name}")
        lines.append("")
        if s.limitations:
            lines += [f"- {item}" for item in s.limitations]
        else:
            lines.append("- None noted.")
        lines.append("")

    lines += [
        "---",
        "*A clean Sharpe ratio is not evidence of edge — see Pillars 1-2 above "
        "before drawing any conclusion.*",
    ]

    return "\n".join(lines)


def write_comparison_report(strategies: Sequence[StrategyReportInputs], path: Path) -> None:
    """Render and write a multi-strategy comparison report to ``path`` (UTF-8)."""
    path.write_text(render_comparison_report(strategies), encoding="utf-8")
