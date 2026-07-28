"""Parameter sensitivity analysis for pairs_trading.

Two grids, with very different cost profiles:

- run_entry_exit_threshold_grid: entry/exit thresholds only affect signal
  generation for an ALREADY-selected candidate (PairsTradingSignalConfig).
  No re-selection needed -- backtest-only, cheap regardless of grid size.
- run_correlation_prefilter_grid: correlation_prefilter changes which
  pairs enter cointegration testing at all, hence n_tests and the
  Bonferroni threshold itself -- a genuine re-run of select_pairs is
  required per grid point (the expensive one; see
  strategies/pairs_trading/README.md for the real-data cost/finding).

Both produce core.evaluation.sensitivity.SensitivityGridPoint so they can
be rendered by the same render_sensitivity_report.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.position_sizing import PositionSizer, PositionSizingConfig, VolTargetSizer
from core.backtest.sample_split import TrainTestSplit, split_returns
from core.evaluation import metrics
from core.evaluation.sensitivity import SensitivityGridPoint
from core.evaluation.statistical_tests import check_significance
from strategies.pairs_trading.cointegration import PairCandidate, PairSelectionConfig, select_pairs
from strategies.pairs_trading.strategy import PairsTradingSignalConfig, PairsTradingStrategy


def run_entry_exit_threshold_grid(
    candidate: PairCandidate,
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    split: TrainTestSplit,
    entry_thresholds: Sequence[float],
    exit_thresholds: Sequence[float],
    default_entry: float = 2.0,
    default_exit: float = 0.0,
    sizer: PositionSizer | None = None,
    backtest_config: BacktestConfig | None = None,
) -> list[SensitivityGridPoint]:
    """Full 2D grid over (entry_threshold, exit_threshold) for a single,
    already-selected candidate. Backtest-only -- select_pairs is never
    called here."""
    resolved_sizer = sizer or VolTargetSizer(PositionSizingConfig())
    resolved_backtest_config = backtest_config or BacktestConfig()

    points: list[SensitivityGridPoint] = []
    for entry in entry_thresholds:
        for exit_ in exit_thresholds:
            signal_config = PairsTradingSignalConfig(entry_threshold=entry, exit_threshold=exit_)
            strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static", config=signal_config)
            result = run_backtest(prices, open_prices, strategy, resolved_sizer, resolved_backtest_config)
            _, oos_returns = split_returns(result.returns, split)
            points.append(
                SensitivityGridPoint(
                    params={"entry_threshold": entry, "exit_threshold": exit_},
                    label=f"{candidate.ticker_a}-{candidate.ticker_b}",
                    oos_metrics=metrics.summary(oos_returns),
                    significance=check_significance(oos_returns, seed=0),
                    is_default=(entry == default_entry and exit_ == default_exit),
                )
            )
    return points


def run_correlation_prefilter_grid(
    prices: pd.DataFrame,
    open_prices: pd.DataFrame,
    split: TrainTestSplit,
    sector_map: dict[str, str],
    prefilters: Sequence[float],
    default_prefilter: float = 0.7,
    base_config: PairSelectionConfig | None = None,
    sizer: PositionSizer | None = None,
    backtest_config: BacktestConfig | None = None,
) -> list[SensitivityGridPoint]:
    """1D grid over correlation_prefilter. Re-runs select_pairs at EACH
    grid point (the prefilter changes n_tests and hence the Bonferroni
    threshold, not just which names happen to be tested) -- a window with
    no statistically-justified candidate is recorded as label="no
    candidate" with None metrics; the harness does not stop."""
    base_config = base_config or PairSelectionConfig()
    resolved_sizer = sizer or VolTargetSizer(PositionSizingConfig())
    resolved_backtest_config = backtest_config or BacktestConfig()

    points: list[SensitivityGridPoint] = []
    for prefilter in prefilters:
        config = PairSelectionConfig(
            group_by=base_config.group_by,
            correlation_prefilter=prefilter,
            cointegration_alpha=base_config.cointegration_alpha,
            require_both_tests_agree=base_config.require_both_tests_agree,
            min_half_life_days=base_config.min_half_life_days,
            max_half_life_days=base_config.max_half_life_days,
            multiple_testing_correction=base_config.multiple_testing_correction,
        )
        candidates = select_pairs(prices, split.in_sample, sector_map, config)
        if not candidates:
            points.append(
                SensitivityGridPoint(
                    params={"correlation_prefilter": prefilter},
                    label="no candidate",
                    oos_metrics=None,
                    significance=None,
                    is_default=(prefilter == default_prefilter),
                )
            )
            continue

        candidate = candidates[0]
        strategy = PairsTradingStrategy(candidate, hedge_ratio_mode="static")
        result = run_backtest(prices, open_prices, strategy, resolved_sizer, resolved_backtest_config)
        _, oos_returns = split_returns(result.returns, split)
        points.append(
            SensitivityGridPoint(
                params={"correlation_prefilter": prefilter},
                label=f"{candidate.ticker_a}-{candidate.ticker_b}",
                oos_metrics=metrics.summary(oos_returns),
                significance=check_significance(oos_returns, seed=0),
                is_default=(prefilter == default_prefilter),
            )
        )
    return points
