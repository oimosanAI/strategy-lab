"""Strategy interface shared by every strategy in this repository.

A *strategy* here is anything that turns a wide price panel (index=date,
columns=ticker -- the same shape as
``core.data.loader.DataLoader.get_price_panel()``) into a target-exposure
signal panel of the same shape.

The contract every strategy must honour:

    ``signals.loc[t]`` is a function of ``prices.loc[:t]`` only, for every
    ticker column. A strategy must never look into the future.

This contract is **enforced by verification, not by trust**:
``core.backtest.engine.assert_causal`` proves it by perturbation -- picking
a cut point, changing every price after it, and asserting that no signal
at or before that cut point changes. A strategy that peeks into the
future -- directly (e.g. reading tomorrow's close) or indirectly (e.g.
fitting a hedge ratio or any other parameter over the *entire* sample,
including future rows) -- will fail that check. See
``tests/core/test_engine.py`` for the positive control (a genuinely causal
strategy that passes) and the negative controls (deliberately non-causal
strategies proven to be caught) that keep this guard honest.

Concretely, a strategy is only required to expose a single method:
``generate_signals(prices) -> DataFrame``. This thin surface is what lets
``core.backtest.engine.run_backtest`` and ``assert_causal`` treat every
strategy uniformly, regardless of internal implementation -- a vectorised
rolling-window computation (e.g. moving averages, rolling z-scores) and an
explicit sequential loop (e.g. a Kalman filter hedge-ratio update) are both
just "a causal strategy" from the engine's point of view.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    """Structural type for anything the engine can backtest.

    A strategy is any object exposing ``generate_signals``. The contract --
    enforced by ``core.backtest.engine.assert_causal``, not by trust -- is
    that ``signals.loc[t]`` is a function of ``prices.loc[:t]`` only, for
    every ticker column.
    """

    # Declared as a read-only property (not a plain `name: str` attribute)
    # so that implementers exposing `name` via `@property` (e.g.
    # PairsTradingStrategy, FactorMomentumStrategy) satisfy this protocol
    # structurally -- a plain attribute requirement in a Protocol demands
    # a SETTABLE member, which a read-only property is not. A plain
    # attribute implementer still satisfies a read-only property
    # requirement (it's strictly more capable), so this is a pure
    # widening, not a behavior change.
    @property
    def name(self) -> str: ...

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target-exposure signals.

        Parameters
        ----------
        prices:
            Wide price panel: index=date, columns=ticker (the same shape
            as ``core.data.loader.DataLoader.get_price_panel()``).

        Returns
        -------
        pd.DataFrame
            Same shape as ``prices``: target exposure per ticker per date.
        """
        ...
