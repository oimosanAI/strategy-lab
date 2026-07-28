"""Tests for strategies.pairs_trading.walk_forward.

Fixes the public API surface:

- run_pairs_trading_walk_forward(prices, open_prices, sector_map, windows,
  selection_config=None, sizer=None, backtest_config=None) ->
  list[WalkForwardWindowResult]: re-runs select_pairs (a genuine one-time
  selection decision) at EACH window's in-sample cutoff -- unlike
  factor_momentum, which reuses a single continuous backtest (see
  core.evaluation.walk_forward's module docstring for why the two
  strategies need different walk-forward treatments). Must not stop when
  a window finds no statistically-justified candidate -- that window's
  result is populated with label="no candidate" and None metrics, and the
  harness continues to the next window.

Reuses strategies.pairs_trading's own test fixtures (_cointegrated_pair /
_non_cointegrated_pair from test_cointegration.py) rather than
re-deriving synthetic cointegrated data from scratch.
"""

from __future__ import annotations

import pandas as pd

from core.backtest.sample_split import generate_anchored_windows
from strategies.pairs_trading.cointegration import select_pairs
from strategies.pairs_trading.walk_forward import run_pairs_trading_walk_forward
from tests.core.test_cointegration import _cointegrated_pair, _non_cointegrated_pair


def _panel_from_pair(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"A": a, "B": b})


def test_run_pairs_trading_walk_forward_finds_candidate_matching_direct_select_pairs() -> None:
    a, b = _cointegrated_pair(n=300)
    prices = _panel_from_pair(a, b)
    sector_map = {"A": "Tech", "B": "Tech"}
    windows = generate_anchored_windows(prices.index, [prices.index[149], prices.index[249]])

    results = run_pairs_trading_walk_forward(prices, prices, sector_map, windows)

    assert len(results) == 2
    # Cross-check window 1 against a direct, independent select_pairs call
    # -- not re-deriving the cointegration math, just verifying wiring.
    expected_candidates = select_pairs(prices, windows[0].in_sample, sector_map)
    assert expected_candidates, "fixture must actually be cointegrated for this test to be meaningful"
    assert results[0].label == f"{expected_candidates[0].ticker_a}-{expected_candidates[0].ticker_b}"
    assert results[0].is_metrics is not None
    assert results[0].oos_metrics is not None
    assert results[0].significance is not None


def test_run_pairs_trading_walk_forward_handles_no_candidate_without_stopping() -> None:
    a, b = _non_cointegrated_pair(n=300)
    prices = _panel_from_pair(a, b)
    sector_map = {"A": "Tech", "B": "Tech"}
    windows = generate_anchored_windows(prices.index, [prices.index[149], prices.index[249]])

    results = run_pairs_trading_walk_forward(prices, prices, sector_map, windows)

    # Both windows processed (harness did not stop after window 1's
    # empty result), both correctly populated as "no candidate".
    assert len(results) == 2
    for result in results:
        assert result.label == "no candidate"
        assert result.is_metrics is None
        assert result.oos_metrics is None
        assert result.significance is None


def test_run_pairs_trading_walk_forward_scales_beyond_two_windows() -> None:
    a, b = _cointegrated_pair(n=400)
    prices = _panel_from_pair(a, b)
    sector_map = {"A": "Tech", "B": "Tech"}
    windows = generate_anchored_windows(
        prices.index, [prices.index[149], prices.index[249], prices.index[349]]
    )

    results = run_pairs_trading_walk_forward(prices, prices, sector_map, windows)

    assert len(results) == 3
