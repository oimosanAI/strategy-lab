"""Position sizing: converts a directional signal into a target weight.

Two methods, per REQUIREMENTS.md 4.4: volatility targeting (the primary,
production method) and half-Kelly (an illustrative comparison method --
REQUIREMENTS.md itself frames raw Kelly sizing as prone to over-leveraging,
hence comparing it against a half-Kelly scaling).

Both share the same per-ticker safety net (``max_leverage`` clip). That
clip protects a SINGLE position's own estimate from blowing up when its
volatility/variance estimate is tiny; it says nothing about the portfolio
as a whole. Portfolio-level gross/net exposure across many simultaneously
held names is a separate concern, checked by
``core.backtest.portfolio.check_exposure_limits`` -- not enforced here. A
multi-factor strategy can have every single position comfortably within
its own ``max_leverage`` cap while the portfolio's summed gross exposure is
still far beyond any sane limit; only ``check_exposure_limits`` catches
that.

Known limitation -- stale/frozen prices are NOT distinguished from a
genuinely low-volatility asset: ``VolTargetSizer`` divides by
``realized_vol``, which is exactly ``0.0`` both when an asset is truly
flat over the whole window AND when its price feed has stopped updating
(a data outage, a halted ticker). In either case ``target_vol /
realized_vol`` clips to ``+/-max_leverage`` -- the single largest
position size this sizer can produce, applied on exactly the input it
should trust least. This is deliberately NOT handled here (e.g. via a
``min_vol_threshold`` mirroring ``PositionSizingConfig.
min_var_threshold``): whether a price series is "stale" is a data
QUALITY question, not a sizing question, and conflating the two would
force one numeric threshold to arbitrate between two scenarios with
opposite meanings -- a real low-vol asset (where clipping to
max_leverage is the documented, tested, intended behavior; see
test_vol_target_sizer_clips_to_max_leverage_when_vol_near_zero) and a
broken feed (where it is dangerous) are mathematically identical inputs
here and cannot be told apart from realized_vol alone. Stale-price
detection belongs in core.data (see core/data/loader.py's module
docstring) as an input-validation concern surfaced before prices ever
reach a sizer, the same way check_exposure_limits was kept a portfolio-
level concern rather than folded into this per-ticker cap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PositionSizingConfig:
    target_vol: float = 0.15
    vol_window: int = 60
    kelly_window: int = 252
    kelly_fraction: float = 0.5
    max_leverage: float = 2.0
    # Half-Kelly's mu/var divides by var, which is far more explosive near
    # zero than vol-targeting's single reciprocal -- a tiny var is not just
    # numerically dangerous but also a much less trustworthy estimate to
    # divide by at all. 1e-6 corresponds to a daily-return variance whose
    # implied volatility is ~0.1% daily (~1.6% annualized): a level
    # essentially never seen in a real, moving equity price series, so it
    # only triggers on genuinely degenerate input (stale prices, a data
    # gap). Typical single-stock daily-return variance is on the order of
    # 1e-4.
    min_var_threshold: float = 1e-6


class PositionSizer(Protocol):
    # Declared as a read-only property (not a plain `name: str` attribute)
    # so that frozen-dataclass implementers (VolTargetSizer, HalfKellySizer)
    # satisfy this protocol structurally -- mypy treats a frozen dataclass
    # field as read-only, which does not satisfy a plain (settable)
    # attribute requirement in a Protocol. Mirrors the identical fix
    # applied to strategies.base.Strategy.name for the same reason.
    @property
    def name(self) -> str: ...

    def apply(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert signals into target weights, same (date, ticker) shape.

        Causal contract: row t of the output depends only on rows <= t of
        both ``signals`` and ``prices`` -- verified via ``assert_causal``
        and the ``_SizerAsStrategy`` test adapter, never merely trusted.
        """
        ...


@dataclass(frozen=True)
class VolTargetSizer:
    config: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    name: str = "vol-target"

    def apply(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change()
        realized_vol = returns.rolling(self.config.vol_window).std() * np.sqrt(252)

        raw_weight = signals * (self.config.target_vol / realized_vol)
        weight = raw_weight.clip(lower=-self.config.max_leverage, upper=self.config.max_leverage)

        # Warm-up (fewer than vol_window observations): realized_vol is NaN
        # -> stay flat rather than propagate NaN into the position.
        return weight.where(realized_vol.notna(), 0.0)


@dataclass(frozen=True)
class HalfKellySizer:
    config: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    name: str = "half-kelly"

    def apply(self, signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        asset_returns = prices.pct_change()

        # Proxy for "this signal's own historical edge", not the raw
        # asset's unconditional return -- a mean-reversion signal's edge
        # has nothing to do with the underlying asset's own long-run mu.
        # This is a same-bar estimation proxy (not lagged by
        # EXECUTION_LAG); an intentional simplification appropriate for
        # Kelly's role here as a comparison method rather than the
        # production sizer (vol targeting).
        strategy_returns = signals * asset_returns
        mu = strategy_returns.rolling(self.config.kelly_window).mean()
        var = strategy_returns.rolling(self.config.kelly_window).var()

        # kelly_fraction already carries the sign of the signal's own
        # historical edge; multiplying by today's signal scales *today's*
        # directional call by that historical conviction level. A negative
        # historical edge inverts today's call -- a documented consequence
        # of applying Kelly literally to a signal's own track record, not a
        # bug.
        kelly_fraction = mu / var
        raw_weight = signals * (self.config.kelly_fraction * kelly_fraction)
        weight = raw_weight.clip(lower=-self.config.max_leverage, upper=self.config.max_leverage)

        valid = var.notna() & (var >= self.config.min_var_threshold)
        return weight.where(valid, 0.0)
