"""Tests for core.data.loader.DataLoader.

Fixes the public API surface that core/data/loader.py and core/data/cache.py
must implement:

- DataProvider.fetch(ticker, start, end) -> DataFrame[PRICE_COLUMNS]
  fetches a single ticker; may return an empty (but correctly-shaped) frame
  when no data exists in the range (e.g. pre-IPO).
- CacheBackend.read/write(ticker) <-> CachedSeries(data, covered_start, covered_end)
  where covered_start/covered_end record the range that has already been
  queried, independent of whether any rows exist in it. This is what lets
  the loader distinguish "never asked" from "asked, nothing there".
- DataLoader.get_prices(tickers, start, end) -> tidy DataFrame[TIDY_COLUMNS],
  end inclusive, fail-fast across tickers.
- DataLoader.get_price_panel(tickers, start, end, field) -> wide DataFrame,
  NaN left as-is (no fill).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from core.data.cache import CachedSeries, ParquetCache
from core.data.loader import (
    DataLoader,
    DataProviderError,
    PRICE_COLUMNS,
    TIDY_COLUMNS,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeDataProvider:
    """Records every fetch() call; serves data from an in-memory table.

    `data_by_ticker` holds the "ground truth" full history per ticker.
    A ticker absent from `data_by_ticker` behaves like a symbol with no
    data at all in the requested range (e.g. pre-IPO) rather than an error.
    Tickers in `failing_tickers` raise DataProviderError on every fetch.
    """

    def __init__(
        self,
        data_by_ticker: dict[str, pd.DataFrame],
        failing_tickers: set[str] | None = None,
    ) -> None:
        self._data_by_ticker = data_by_ticker
        self._failing_tickers = failing_tickers or set()
        self.call_log: list[tuple[str, date, date]] = []

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.call_log.append((ticker, start, end))
        if ticker in self._failing_tickers:
            raise DataProviderError(f"simulated failure fetching {ticker}")
        full = self._data_by_ticker.get(ticker)
        if full is None:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        mask = (full["date"] >= pd.Timestamp(start)) & (full["date"] <= pd.Timestamp(end))
        return full.loc[mask].reset_index(drop=True)


class InMemoryCache:
    """Dict-backed CacheBackend for tests that don't need filesystem I/O."""

    def __init__(self) -> None:
        self._store: dict[str, CachedSeries] = {}

    def read(self, ticker: str) -> CachedSeries | None:
        return self._store.get(ticker)

    def write(self, ticker: str, series: CachedSeries) -> None:
        self._store[ticker] = series


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _ohlcv_frame(dates: pd.DatetimeIndex, base_price: float = 100.0) -> pd.DataFrame:
    closes = [base_price + i for i in range(len(dates))]
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(dates),
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "adj_close": closes,
            "volume": [1_000_000 + i for i in range(len(dates))],
        }
    )


JAN_FULL_MONTH = _business_days(date(2020, 1, 1), date(2020, 1, 31))


# ---------------------------------------------------------------------------
# Schema / range filtering
# ---------------------------------------------------------------------------


def test_get_prices_returns_correct_schema_and_filters_range() -> None:
    # Arrange
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))

    # Assert
    assert list(result.columns) == TIDY_COLUMNS
    assert (result["ticker"] == "AAPL").all()
    assert result["date"].min() >= pd.Timestamp(date(2020, 1, 10))
    assert result["date"].max() <= pd.Timestamp(date(2020, 1, 15))


def test_get_prices_end_boundary_is_inclusive() -> None:
    # Arrange: Jan 15, 2020 is a business day (Wed) with data available.
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 1), end=date(2020, 1, 15))

    # Assert: the end date itself must be included, not excluded.
    assert pd.Timestamp(date(2020, 1, 15)) in set(result["date"])
    assert result["date"].max() == pd.Timestamp(date(2020, 1, 15))


# ---------------------------------------------------------------------------
# Cache hit / gap-only refetch
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_provider_refetch() -> None:
    # Arrange
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())
    loader.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))
    assert len(provider.call_log) == 1

    # Act: a narrower request fully inside the already-cached range.
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 13), end=date(2020, 1, 14))

    # Assert: no new provider call was made.
    assert len(provider.call_log) == 1
    assert set(result["date"]) == {pd.Timestamp(date(2020, 1, 13)), pd.Timestamp(date(2020, 1, 14))}


def test_extending_range_fetches_only_forward_gap() -> None:
    # Arrange: cache [Jan10, Jan15] first.
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())
    loader.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))
    assert provider.call_log == [("AAPL", date(2020, 1, 10), date(2020, 1, 15))]

    # Act: request an earlier start but the *same* end -> head gap only.
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 6), end=date(2020, 1, 15))

    # Assert: exactly one additional call, covering only the new head gap.
    assert provider.call_log[1:] == [("AAPL", date(2020, 1, 6), date(2020, 1, 9))]
    assert len(provider.call_log) == 2
    assert result["date"].min() == pd.Timestamp(date(2020, 1, 6))
    assert result["date"].max() == pd.Timestamp(date(2020, 1, 15))


def test_extending_range_fetches_only_backward_gap() -> None:
    # Arrange: cache [Jan10, Jan15] first.
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())
    loader.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))
    assert provider.call_log == [("AAPL", date(2020, 1, 10), date(2020, 1, 15))]

    # Act: request a later end but the *same* start -> tail gap only.
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 20))

    # Assert: exactly one additional call, covering only the new tail gap.
    assert provider.call_log[1:] == [("AAPL", date(2020, 1, 16), date(2020, 1, 20))]
    assert len(provider.call_log) == 2
    assert result["date"].min() == pd.Timestamp(date(2020, 1, 10))
    assert result["date"].max() == pd.Timestamp(date(2020, 1, 20))


def test_extending_range_fetches_both_head_and_tail_gaps_simultaneously() -> None:
    # Arrange: cache a middle slice, [2020-06-01, 2020-12-31], then request
    # a much wider range that needs new data on *both* sides at once.
    full_range = _business_days(date(2020, 1, 1), date(2021, 6, 30))
    aapl_history = _ohlcv_frame(full_range)
    provider = FakeDataProvider({"AAPL": aapl_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())
    loader.get_prices(["AAPL"], start=date(2020, 6, 1), end=date(2020, 12, 31))
    assert provider.call_log == [("AAPL", date(2020, 6, 1), date(2020, 12, 31))]

    # Act
    result = loader.get_prices(["AAPL"], start=date(2020, 1, 1), end=date(2021, 6, 30))

    # Assert: exactly two additional calls, one per side, not a full refetch.
    assert provider.call_log[1:] == [
        ("AAPL", date(2020, 1, 1), date(2020, 5, 31)),
        ("AAPL", date(2021, 1, 1), date(2021, 6, 30)),
    ]
    assert len(provider.call_log) == 3
    assert result["date"].min() == pd.Timestamp(date(2020, 1, 1))
    assert result["date"].max() == pd.Timestamp(date(2021, 6, 30))


def test_multiple_tickers_have_independent_cache_state() -> None:
    # Arrange
    provider = FakeDataProvider(
        {
            "AAPL": _ohlcv_frame(JAN_FULL_MONTH, base_price=100.0),
            "MSFT": _ohlcv_frame(JAN_FULL_MONTH, base_price=200.0),
        }
    )
    loader = DataLoader(provider=provider, cache=InMemoryCache())
    loader.get_prices(["AAPL", "MSFT"], start=date(2020, 1, 10), end=date(2020, 1, 15))
    assert len(provider.call_log) == 2  # one fetch per ticker

    # Act: widen only MSFT's range; AAPL's cached range is untouched.
    loader.get_prices(["MSFT"], start=date(2020, 1, 6), end=date(2020, 1, 20))

    # Assert: AAPL was not refetched; MSFT got exactly its two gap fetches.
    aapl_calls = [c for c in provider.call_log if c[0] == "AAPL"]
    msft_calls = [c for c in provider.call_log if c[0] == "MSFT"]
    assert len(aapl_calls) == 1
    assert len(msft_calls) == 3


# ---------------------------------------------------------------------------
# ParquetCache persistence
# ---------------------------------------------------------------------------


def test_parquet_cache_persists_across_loader_instances(tmp_path: Path) -> None:
    # Arrange
    aapl_history = _ohlcv_frame(JAN_FULL_MONTH)
    provider = FakeDataProvider({"AAPL": aapl_history})
    cache_dir = tmp_path / "cache"

    loader1 = DataLoader(provider=provider, cache=ParquetCache(cache_dir))
    loader1.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))
    assert len(provider.call_log) == 1

    # Act: a fresh DataLoader/ParquetCache pointed at the same directory.
    loader2 = DataLoader(provider=provider, cache=ParquetCache(cache_dir))
    result = loader2.get_prices(["AAPL"], start=date(2020, 1, 10), end=date(2020, 1, 15))

    # Assert: served entirely from disk, no new provider call.
    assert len(provider.call_log) == 1
    assert len(result) == 4  # Jan 10, 13, 14, 15


# ---------------------------------------------------------------------------
# Fail-fast on provider errors
# ---------------------------------------------------------------------------


def test_provider_error_propagates_as_data_provider_error() -> None:
    # Arrange
    provider = FakeDataProvider({}, failing_tickers={"BAD"})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act / Assert
    with pytest.raises(DataProviderError):
        loader.get_prices(["BAD"], start=date(2020, 1, 1), end=date(2020, 1, 10))


def test_partial_ticker_failure_raises_and_returns_no_data() -> None:
    # Arrange: GOOD has real data and would otherwise succeed; BAD fails.
    provider = FakeDataProvider(
        {"GOOD": _ohlcv_frame(JAN_FULL_MONTH)},
        failing_tickers={"BAD"},
    )
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act / Assert: the whole call fails-fast; there is no partial-success
    # return value containing GOOD's rows.
    with pytest.raises(DataProviderError):
        loader.get_prices(["GOOD", "BAD"], start=date(2020, 1, 10), end=date(2020, 1, 15))


# ---------------------------------------------------------------------------
# "Queried, no data" vs "never queried"
# ---------------------------------------------------------------------------


def test_repeated_query_for_no_data_range_does_not_refetch() -> None:
    # Arrange: NEWCO has no rows anywhere (e.g. requested range predates IPO).
    provider = FakeDataProvider({})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    first = loader.get_prices(["NEWCO"], start=date(2020, 1, 1), end=date(2020, 1, 10))
    assert len(provider.call_log) == 1
    assert len(first) == 0

    second = loader.get_prices(["NEWCO"], start=date(2020, 1, 3), end=date(2020, 1, 8))

    # Assert: the sub-range is already known to be empty; no new fetch.
    assert len(provider.call_log) == 1
    assert len(second) == 0


def test_extending_from_no_data_range_only_fetches_new_gap_with_data() -> None:
    # Arrange: NEWCO IPOs on Jan 15, 2020 -- no rows exist before that date.
    ipo_history = _ohlcv_frame(_business_days(date(2020, 1, 15), date(2020, 1, 31)))
    provider = FakeDataProvider({"NEWCO": ipo_history})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # First query covers only the pre-IPO period: cached as "queried, empty".
    pre_ipo = loader.get_prices(["NEWCO"], start=date(2020, 1, 1), end=date(2020, 1, 10))
    assert provider.call_log == [("NEWCO", date(2020, 1, 1), date(2020, 1, 10))]
    assert len(pre_ipo) == 0

    # Act: widen the request to straddle the pre-IPO/post-IPO boundary.
    result = loader.get_prices(["NEWCO"], start=date(2020, 1, 1), end=date(2020, 1, 20))

    # Assert: the already-covered no-data head is NOT refetched; only the
    # new tail gap (which happens to contain real data) is fetched.
    assert provider.call_log[1:] == [("NEWCO", date(2020, 1, 11), date(2020, 1, 20))]
    assert len(provider.call_log) == 2
    assert len(result) == 4  # Jan 15, 16, 17, 20 -- the only trading days post-IPO
    assert result["date"].min() == pd.Timestamp(date(2020, 1, 15))


# ---------------------------------------------------------------------------
# get_price_panel: wide-format pivot, no fill
# ---------------------------------------------------------------------------


def test_get_price_panel_pivots_to_wide_format() -> None:
    # Arrange
    days = _business_days(date(2020, 1, 10), date(2020, 1, 15))
    provider = FakeDataProvider(
        {
            "AAPL": _ohlcv_frame(days, base_price=100.0),
            "MSFT": _ohlcv_frame(days, base_price=200.0),
        }
    )
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    panel = loader.get_price_panel(
        ["AAPL", "MSFT"], start=date(2020, 1, 10), end=date(2020, 1, 15), field="close"
    )

    # Assert
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert list(panel.index) == list(pd.DatetimeIndex(days))
    assert panel.loc[pd.Timestamp(date(2020, 1, 10)), "AAPL"] == 100.0
    assert panel.loc[pd.Timestamp(date(2020, 1, 10)), "MSFT"] == 200.0


def test_get_price_panel_leaves_gaps_as_nan_for_misaligned_trading_days() -> None:
    # Arrange: MSFT is missing Jan 13 entirely (e.g. a trading halt),
    # while AAPL trades every day in the range.
    all_days = _business_days(date(2020, 1, 10), date(2020, 1, 15))
    msft_days = pd.DatetimeIndex([d for d in all_days if d != pd.Timestamp(date(2020, 1, 13))])
    provider = FakeDataProvider(
        {
            "AAPL": _ohlcv_frame(all_days, base_price=100.0),
            "MSFT": _ohlcv_frame(msft_days, base_price=200.0),
        }
    )
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    panel = loader.get_price_panel(
        ["AAPL", "MSFT"], start=date(2020, 1, 10), end=date(2020, 1, 15), field="close"
    )

    # Assert: the hole is NaN, not forward-filled from Jan 10's value.
    # all_days = [Jan10, Jan13, Jan14, Jan15] -> AAPL closes = [100, 101, 102, 103].
    assert pd.isna(panel.loc[pd.Timestamp(date(2020, 1, 13)), "MSFT"])
    assert panel.loc[pd.Timestamp(date(2020, 1, 13)), "AAPL"] == pytest.approx(101.0)


def test_get_price_panel_returns_empty_frame_when_no_data_exists() -> None:
    # Arrange: requested range predates NEWCO's IPO entirely.
    provider = FakeDataProvider({})
    loader = DataLoader(provider=provider, cache=InMemoryCache())

    # Act
    panel = loader.get_price_panel(
        ["NEWCO"], start=date(2020, 1, 1), end=date(2020, 1, 10), field="close"
    )

    # Assert
    assert len(panel) == 0
    assert list(panel.columns) == ["NEWCO"]
