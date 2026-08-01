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
from core.evaluation.statistical_tests import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_SEED,
    SignificanceCheck,
    check_significance,
)

# A moving-block bootstrap draws contiguous runs of `block_size` bars, so a
# window of n bars offers only n - block_size + 1 distinct start positions.
# At n barely above block_size every resample is a near-copy of the original
# series and the interval collapses towards zero width -- it stops measuring
# sampling uncertainty and starts measuring nothing. Requiring room for at
# least this many blocks (n >= 4 * block_size, i.e. 20 bars at the default
# block_size of 5) keeps the interval informative. Expressed as a block
# COUNT, deliberately, so it stays tied to block_size instead of becoming a
# free-floating "20" that would silently stop matching if block_size moved.
_MIN_BLOCKS_FOR_SIGNIFICANCE: int = 4


@dataclass(frozen=True)
class WalkForwardWindowResult:
    """One window's result. is_metrics/oos_metrics/significance are all
    None together when the window produced nothing evaluable, and
    ``skip_reason`` then says WHY in the report's own words.

    Two distinct situations reach that state and must not be conflated:
    nothing was FOUND (e.g. no statistically-justified pair survived
    selection -- see strategies.pairs_trading.walk_forward), or the window
    could not be EVALUATED because its out-of-sample slice was too short
    (see evaluate_walk_forward_windows). Both render as an N/A row, so the
    reason is the only thing distinguishing "selection came up empty" from
    "we never got to test this window" -- a difference that changes how the
    whole walk-forward record should be read.
    """

    window: TrainTestSplit
    label: str
    is_metrics: dict[str, float] | None
    oos_metrics: dict[str, float] | None
    significance: SignificanceCheck | None
    skip_reason: str | None = None


def _skip_reason_for(n_oos: int, block_size: int) -> str | None:
    """Why this window cannot be significance-tested, or None if it can.

    Two separate floors, reported separately because they mean different
    things. Below ``max(2, block_size)`` the bootstrap is not COMPUTABLE at
    all. Between there and ``_MIN_BLOCKS_FOR_SIGNIFICANCE * block_size`` it
    is computable but not INFORMATIVE, and emitting a p-value would report
    precision the data cannot support -- the quieter and more misleading of
    the two failures, which is exactly why it gets its own message rather
    than being lumped in with the first.
    """
    mechanical_floor = max(2, block_size)
    if n_oos < mechanical_floor:
        return (
            f"too short to evaluate ({n_oos} OOS observations; a moving-block "
            f"bootstrap with block_size={block_size} needs at least {mechanical_floor})"
        )

    statistical_floor = _MIN_BLOCKS_FOR_SIGNIFICANCE * block_size
    if n_oos < statistical_floor:
        return (
            f"not tested ({n_oos} OOS observations: computable, but below the "
            f"{statistical_floor} needed for {_MIN_BLOCKS_FOR_SIGNIFICANCE} blocks of "
            f"size {block_size}; a p-value here would overstate what the data supports)"
        )

    return None


def evaluate_walk_forward_windows(
    returns: pd.Series,
    windows: Sequence[TrainTestSplit],
    label: str = "strategy",
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[WalkForwardWindowResult]:
    """Slice an already-computed continuous returns series into multiple
    IS/OOS windows. No new backtest is run -- purely a generic wiring
    utility over split_returns/metrics.summary/check_significance.

    A window whose out-of-sample slice is too short to support a
    significance test is RECORDED (all payload fields None, skip_reason
    set) rather than raised on, so one unusable window does not abort the
    remaining ones -- the same "the harness does not stop" contract
    strategies.pairs_trading.walk_forward already honours for windows
    where selection found no candidate.
    """
    results = []
    for window in windows:
        is_returns, oos_returns = split_returns(returns, window)
        skip_reason = _skip_reason_for(len(oos_returns.dropna()), block_size)
        if skip_reason is not None:
            results.append(
                WalkForwardWindowResult(
                    window=window,
                    label=label,
                    is_metrics=None,
                    oos_metrics=None,
                    significance=None,
                    skip_reason=skip_reason,
                )
            )
            continue

        results.append(
            WalkForwardWindowResult(
                window=window,
                label=label,
                is_metrics=metrics.summary(is_returns),
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, block_size=block_size, seed=seed),
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
            # skip_reason, when present, REPLACES the label in the Result
            # column: an N/A row's only informative content is why it is
            # N/A, and "strategy" tells the reader nothing there.
            reason = result.skip_reason or result.label
            lines.append(f"| {i} | {is_end} | {oos_period} | {reason} | N/A | N/A | N/A |")
        else:
            sharpe = result.oos_metrics["sharpe_ratio"]
            p_value = result.significance.permutation.p_value
            agree = "Yes" if result.significance.agree else "No"
            lines.append(f"| {i} | {is_end} | {oos_period} | {result.label} | {sharpe:.2f} | {p_value:.4f} | {agree} |")

    lines += ["", "## Consistency Summary", ""]

    populated = [r for r in results if r.oos_metrics is not None]
    lines.append(f"- Windows with a result: {len(populated)}/{len(results)}")

    # Indices, not the results themselves: WalkForwardWindowResult holds
    # numpy arrays (via SignificanceCheck), so `result in skipped` would
    # trigger elementwise __eq__ and raise "truth value is ambiguous".
    skipped = [i for i, r in enumerate(results, start=1) if r.oos_metrics is None and r.skip_reason is not None]
    if skipped:
        # Broken out from the populated count on purpose: a run where
        # windows were never testable is a different (weaker) claim than a
        # run where they were tested and came back negative, and the two
        # must not collapse into one "2/4" figure.
        # Deliberately neutral about WHY: the reasons differ per window
        # (selection found nothing / too little out-of-sample data) and are
        # listed individually below. An earlier version hardcoded
        # "insufficient out-of-sample data" here and so mislabelled every
        # "no candidate" window as a data problem.
        lines.append(
            f"- Windows with no tested result: {len(skipped)}/{len(results)} "
            "-- these were never tested, which is not the same as tested and unremarkable:"
        )
        lines += [f"  - Window {i}: {results[i - 1].skip_reason}" for i in skipped]

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
