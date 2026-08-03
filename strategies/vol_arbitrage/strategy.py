"""VolArbitrageStrategy: harvest the Volatility Risk Premium (VRP) via a
long-only position in an inverse-VIX-futures ETF (REQUIREMENTS.md 5.3).

Implements strategies.base.Strategy directly (generate_signals(prices) ->
pd.DataFrame), matching PairsTradingStrategy/FactorMomentumStrategy's
pattern. Unlike those two, the SIGNAL depends on data (VIX, SPY, and in
"term_structure" mode VIX9D) that is NOT the traded instrument (SVXY)
itself. Rather than binding these as side data at construction (which
assert_causal's prices-only perturbation could never reach, since bound
side data never changes across trials), `prices` carries them all as
columns: the traded ticker plus the reference-only inputs.

Why this works with run_backtest UNCHANGED: the reference columns always
get signal 0.0, hence position 0.0 throughout -- core.backtest.portfolio.
compute_ticker_returns' carried/new/exited terms are all proportional to
position, so a 0 position contributes exactly 0 return and 0 turnover
regardless of the reference columns' own price levels. No changes to
core/backtest/engine.py are needed.

Why this is better than binding side data at construction: with the
reference columns as literal `prices` columns, assert_causal's existing
perturbation mechanism (which only ever touches `prices`) automatically
covers the signal computation's causality too -- no new guard needed,
verified via the same assert_causal call used for every other strategy in
this project.

TWO SIGNAL MODES (``PairSelectionConfig``-style ``mode`` field, same
pattern as FactorMomentumStrategy's global/sector_neutral):

- ``mode="vrp"`` (default): the original trailing-realized-volatility VRP
  signal (``strategies.vol_arbitrage.signal.compute_vrp``), unchanged
  since Phase 3.
- ``mode="term_structure"``: VIX9D/VIX ratio
  (``compute_term_structure_ratio``), added per
  strategies/vol_arbitrage/README.md Step F/section 3-7. A real-data
  diagnostic there found the VRP signal spends 87% of days "on" and does
  not de-risk ahead of a spike, because trailing RV is a backward-looking
  proxy that only rises AFTER a spike has partly already happened. The
  term-structure ratio compares two forward-looking (options-implied)
  quantities instead, with no rolling window at all -- see that module's
  docstring for the full rationale.

The two modes are kept side by side rather than one replacing the other:
the VRP mode's documented negative result (Step A-F) is itself a
validated finding worth preserving, and having both lets an E2E run
compare them directly on the same universe/split, the same way
FactorMomentumStrategy's global vs sector_neutral modes were compared.

Long-only (0/+1), never short, in both modes: this project's cost model
(commission + slippage only) has no borrow-cost modeling, so a short
SVXY / long volatility leg is deliberately out of scope -- see
strategies/vol_arbitrage/README.md for the fuller reasoning and the
Volmageddon-style tail-risk disclosure for short-vol ETPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from strategies.vol_arbitrage.signal import compute_term_structure_ratio, compute_vrp


@dataclass(frozen=True)
class VolArbitrageConfig:
    rv_window: int = 21
    # See strategies/vol_arbitrage/README.md for why 0.0 (trade whenever
    # the trailing-RV-based VRP proxy is positive at all) was chosen as a
    # starting point rather than a literature-derived magnitude -- this
    # project's trailing-RV definition of VRP does not have a precise
    # published reference value the way the ex-post academic VRP measure
    # does. Flagged as a sensitivity-analysis target.
    vrp_threshold: float = 0.0
    # Unlike vrp_threshold, 1.0 is not an arbitrary starting point: it is
    # the literal contango/backwardation boundary of the VIX term
    # structure, a named and widely-recognized market phenomenon (see
    # strategies/vol_arbitrage/README.md Step F). Still a
    # sensitivity-analysis target, but on stronger a priori footing.
    ts_threshold: float = 1.0
    mode: Literal["vrp", "term_structure"] = "vrp"
    traded_ticker: str = "SVXY"
    vix_column: str = "VIX"
    spy_column: str = "SPY"
    vix9d_column: str = "VIX9D"


class VolArbitrageStrategy:
    def __init__(self, config: VolArbitrageConfig | None = None) -> None:
        self._config = config or VolArbitrageConfig()

    @property
    def name(self) -> str:
        return "vol-arbitrage"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        config = self._config
        required = {config.traded_ticker, config.vix_column, config.spy_column}
        if config.mode == "term_structure":
            required.add(config.vix9d_column)

        # Fail loudly and identically for every required column. Without
        # this, a misconfigured traded_ticker was SILENT: the assignment
        # below would create a brand-new column, run_backtest (which
        # iterates prices.columns) would never see it, and the run would
        # complete reporting an all-zero-return "no edge" strategy rather
        # than a configuration error -- while the same mistake in the
        # other columns raised a bare KeyError from the lookups. A
        # misconfiguration must never be able to masquerade as a result.
        missing = sorted(required - set(prices.columns))
        if missing:
            expected = [repr(config.traded_ticker), repr(config.vix_column), repr(config.spy_column)]
            if config.mode == "term_structure":
                expected.append(repr(config.vix9d_column))
            raise ValueError(
                f"prices is missing required column(s) {missing}; "
                f"expected the traded ticker plus the reference-only signal inputs "
                f"({', '.join(expected)}). Got columns: {list(prices.columns)}"
            )

        if config.mode == "vrp":
            vrp = compute_vrp(prices[config.vix_column], prices[config.spy_column], config.rv_window)
            active = vrp > config.vrp_threshold
        else:
            ratio = compute_term_structure_ratio(prices[config.vix9d_column], prices[config.vix_column])
            active = ratio < config.ts_threshold

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[config.traded_ticker] = active.astype(float)
        return signal
