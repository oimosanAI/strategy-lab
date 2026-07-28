"""Tests for core.backtest.engine's causality guard.

Fixes the public API surface that core/backtest/engine.py must implement:

- Strategy (Protocol): generate_signals(prices: DataFrame) -> DataFrame,
  same (date index, ticker columns) shape as
  core.data.loader.DataLoader.get_price_panel(). Contract: signals.loc[t]
  depends only on prices.loc[:t]. Never trusted -- always verified.
- LookAheadError(AssertionError)
- assert_causal(strategy, prices, *, n_trials=8, perturbation=0.10, seed=0)
  Multi-ticker generalization of backtest-framework's assert_causal
  (C:\\Users\\oimos\\backtest-framework\\src\\backtest\\engine.py): perturb
  every ticker's price after a cut point k; assert no ticker's signal at or
  before k changes.

Per the project's quant standard (a guard is only proven if it's shown to
catch a planted bug, not merely to pass), this file keeps THREE fixture
strategies as permanent negative/positive controls:

- _RollingZScoreStrategy: causal by construction (rolling window only looks
  backward) -> assert_causal must PASS.
- _PerfectForesightPanel: reads tomorrow's close directly, per ticker ->
  assert_causal must CATCH it. Mirrors backtest-framework's _PerfectForesight.
- _FullSampleOLSHedgeStrategy: estimates a hedge ratio using the ENTIRE
  sample including future rows, then applies it uniformly. This is the
  pairs-trading-specific look-ahead trap REQUIREMENTS.md 4.1 calls out
  explicitly ("全期間データでペアを選ぶのはlook-ahead biasの典型的な誤り")
  -- a subtler violation than "reads tomorrow's price" since no single
  signal value literally reads a future row, yet every signal is still
  contaminated by future information through the fitted parameter.
  assert_causal must CATCH this too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest.engine import LookAheadError, assert_causal


def _panel(n_days: int, tickers: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n_days)
    data = {ticker: 100 + np.cumsum(rng.normal(0, 1, n_days)) for ticker in tickers}
    return pd.DataFrame(data, index=index).abs() + 1.0


class _RollingZScoreStrategy:
    """Causal: rolling mean/std at t only ever look at rows <= t."""

    name = "rolling-zscore (causal)"

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        mean = prices.rolling(self.window).mean()
        std = prices.rolling(self.window).std()
        z = (prices - mean) / std
        return (-np.sign(z)).fillna(0.0)


class _PerfectForesightPanel:
    """Deliberately non-causal: longs whenever TOMORROW's close is higher,
    independently per ticker. The canonical look-ahead bug."""

    name = "perfect-foresight-panel (deliberately non-causal)"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        future = prices.shift(-1)
        return np.sign(future - prices).fillna(0.0)


class _FullSampleOLSHedgeStrategy:
    """Deliberately non-causal: fits a hedge ratio using the FULL sample
    (including rows after any given t) and applies it as a constant for
    every date. No signal literally reads a future row, but every signal
    is contaminated because the fitted scalar itself depends on the whole
    series. This is the REQUIREMENTS.md 4.1 pairs-trading pitfall."""

    name = "full-sample-ols-hedge (deliberately non-causal)"

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        a, b = prices.columns[0], prices.columns[1]
        beta = np.polyfit(prices[b], prices[a], 1)[0]  # uses ALL rows, incl. future
        spread = prices[a] - beta * prices[b]
        z = (spread - spread.mean()) / spread.std()
        return pd.DataFrame({a: -np.sign(z), b: np.sign(z)}, index=prices.index)


def test_assert_causal_passes_for_causal_multi_ticker_strategy() -> None:
    # Arrange
    prices = _panel(150, ["AAPL", "MSFT", "GOOG"], seed=1)

    # Act / Assert: no exception.
    assert_causal(_RollingZScoreStrategy(window=10), prices, n_trials=12)


def test_assert_causal_rejects_perfect_foresight_panel() -> None:
    # Arrange
    prices = _panel(100, ["AAPL", "MSFT"], seed=2)

    # Act / Assert: this guard must bite, or it is asleep at the wheel.
    with pytest.raises(LookAheadError):
        assert_causal(_PerfectForesightPanel(), prices, n_trials=20)


def test_assert_causal_rejects_full_sample_parameter_estimation() -> None:
    # Arrange
    prices = _panel(150, ["AAPL", "MSFT"], seed=3)

    # Act / Assert: catches contamination via a globally-fitted parameter,
    # not just direct future-row access.
    with pytest.raises(LookAheadError):
        assert_causal(_FullSampleOLSHedgeStrategy(), prices, n_trials=20)


def test_assert_causal_rejects_too_few_observations() -> None:
    # Arrange: a cut point needs both a non-empty past and future, so
    # fewer than 3 observations can never be tested meaningfully.
    prices = _panel(2, ["AAPL"], seed=4)

    # Act / Assert
    with pytest.raises(ValueError):
        assert_causal(_RollingZScoreStrategy(), prices)
