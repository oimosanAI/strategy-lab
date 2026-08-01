"""Walk-forward validation for pairs_trading (REQUIREMENTS.md 8, Phase 6
"拡張": walk-forward robustness analysis).

Re-runs select_pairs -- a genuine one-time selection decision -- at EACH
window's in-sample cutoff, because a pair chosen with information up to
one point in time cannot be assumed still valid at an earlier point
without leaking future information (see the design discussion recorded in
strategies/pairs_trading/README.md and core/evaluation/walk_forward.py's
module docstring for why factor_momentum, which has no such one-time
decision, does NOT need this and instead reuses a single continuous
backtest).

Produces core.evaluation.walk_forward.WalkForwardWindowResult -- the same
shared type factor_momentum's walk-forward evaluation produces -- so both
can be rendered by the same render_walk_forward_report.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizer, PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import TrainTestSplit, split_returns
from core.evaluation import metrics
from core.evaluation.statistical_tests import check_significance
from core.evaluation.walk_forward import WalkForwardWindowResult
from strategies.pairs_trading.cointegration import PairSelectionConfig, select_pairs
from strategies.pairs_trading.strategy import PairsTradingStrategy


def run_pairs_trading_walk_forward(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    sector_map: dict[str, str],
    windows: Sequence[TrainTestSplit],
    selection_config: PairSelectionConfig | None = None,
    sizer: PositionSizer | None = None,
    backtest_config: BacktestConfig | None = None,
) -> list[WalkForwardWindowResult]:
    """Run select_pairs + backtest independently at each window. A window
    with no statistically-justified candidate is recorded as
    label="no candidate" with None metrics -- the harness does not stop.
    """
    resolved_sizer = sizer or VolTargetSizer(PositionSizingConfig())
    resolved_backtest_config = backtest_config or BacktestConfig()

    results: list[WalkForwardWindowResult] = []
    for window in windows:
        window_prices = prices.loc[: window.out_of_sample.end]
        window_open = open_prices.loc[: window.out_of_sample.end]

        candidates = select_pairs(window_prices, window.in_sample, sector_map, selection_config)
        if not candidates:
            results.append(
                WalkForwardWindowResult(
                    window=window,
                    label="no candidate",
                    is_metrics=None,
                    oos_metrics=None,
                    significance=None,
                    # Stated explicitly rather than left to the reader to
                    # infer from the label: an N/A row here means selection
                    # ran and found nothing, NOT that the window went
                    # untested for lack of data (the other route to an N/A
                    # row -- see core.evaluation.walk_forward).
                    skip_reason="no statistically-justified pair survived selection",
                )
            )
            continue

        # Only the single top-ranked candidate (candidates[0], sorted by
        # adjusted p-value) is walk-forward tested per window, matching
        # the single-split E2E validation's approach of picking one best
        # pair. A future extension could walk-forward multiple
        # simultaneously-held pairs per window (a small pairs-trading
        # "book" rather than a single pair); out of scope here.
        candidate = candidates[0]
        strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")
        result = run_backtest(window_prices, window_open, strategy, resolved_sizer, resolved_backtest_config)
        is_returns, oos_returns = split_returns(result.returns, window)

        results.append(
            WalkForwardWindowResult(
                window=window,
                label=f"{candidate.ticker_a}-{candidate.ticker_b}",
                is_metrics=metrics.summary(is_returns),
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, seed=0),
            )
        )
    return results
