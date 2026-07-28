"""VolArbitrageStrategy: harvest the Volatility Risk Premium (VRP) via a
long-only position in an inverse-VIX-futures ETF (REQUIREMENTS.md 5.3).

Implements strategies.base.Strategy directly (generate_signals(prices) ->
pd.DataFrame), matching PairsTradingStrategy/FactorMomentumStrategy's
pattern. Unlike those two, the SIGNAL depends on data (VIX, SPY) that is
NOT the traded instrument (SVXY) itself. Rather than binding VIX/SPY as
side data at construction (which assert_causal's prices-only perturbation
could never reach, since bound side data never changes across trials),
`prices` carries all THREE as columns: the traded ticker plus VIX and SPY
as REFERENCE-ONLY columns.

Why this works with run_backtest UNCHANGED: VIX/SPY always get signal
0.0, hence position 0.0 throughout -- core.backtest.portfolio.
compute_ticker_returns' carried/new/exited terms are all proportional to
position, so a 0 position contributes exactly 0 return and 0 turnover
regardless of VIX/SPY's own price levels. No changes to core/backtest/
engine.py are needed.

Why this is better than binding VIX/SPY at construction: with VIX/SPY as
literal `prices` columns, assert_causal's existing perturbation mechanism
(which only ever touches `prices`) automatically covers the VRP
computation's causality too -- no new guard needed, verified via the same
assert_causal call used for every other strategy in this project.

Long-only (0/+1), never short: this project's cost model (commission +
slippage only) has no borrow-cost modeling, so a short SVXY / long
volatility leg is deliberately out of scope -- see
strategies/vol_arbitrage/README.md for the fuller reasoning and the
Volmageddon-style tail-risk disclosure for short-vol ETPs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.vol_arbitrage.signal import compute_vrp


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
    traded_ticker: str = "SVXY"
    vix_column: str = "VIX"
    spy_column: str = "SPY"


class VolArbitrageStrategy:
    def __init__(self, config: VolArbitrageConfig | None = None) -> None:
        self._config = config or VolArbitrageConfig()

    @property
    def name(self) -> str:
        return "vol-arbitrage"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        vrp = compute_vrp(prices[self._config.vix_column], prices[self._config.spy_column], self._config.rv_window)

        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[self._config.traded_ticker] = (vrp > self._config.vrp_threshold).astype(float)
        return signal
