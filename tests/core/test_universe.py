"""Tests for core.data.universe.

Fixes the public API surface that core/data/universe.py must implement:

- UniverseSource.fetch_constituents() -> DataFrame[ticker, security,
  gics_sector, gics_sub_industry], raw (not yet normalized for yfinance).
  A single attempt: no retrying inside the source itself.
- WikipediaUniverseSource: the real UniverseSource. Its HTML parsing is
  tested directly via _parse_html() so no real HTTP call is needed.
- UniverseCache: a point-in-time snapshot cache with a freshness TTL --
  NOT core.data.cache.CacheBackend, which models date-range coverage for
  time series and doesn't fit a single "as of now" snapshot.
- UniverseLoader: owns retry/backoff, cache-then-fetch-then-stale-cache-
  then-bundled-fallback degradation, and ticker normalization for the
  yfinance-ready get_sp500_tickers() -> list[str] output.

Known limitation (Level A survivorship bias, REQUIREMENTS.md 7.1):
get_sp500_tickers() always returns *today's* constituents; it does not
reconstruct historical membership. Not this module's concern to test --
it is a documented limitation of the approach, not a bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from core.data.universe import (
    UniverseCache,
    UniverseLoader,
    UniverseSourceError,
    WikipediaUniverseSource,
    _normalize_ticker,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeUniverseSource:
    """Records fetch_constituents() calls; fails the first `fail_times`
    calls (simulating transient failures), then returns `data`."""

    def __init__(self, data: pd.DataFrame, fail_times: int = 0) -> None:
        self._data = data
        self._fail_times = fail_times
        self.call_count = 0

    def fetch_constituents(self) -> pd.DataFrame:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise UniverseSourceError("simulated transient failure")
        return self._data


def _sample_constituents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "BRK.B"],
            "security": ["Apple Inc.", "Microsoft Corp.", "Berkshire Hathaway"],
            "gics_sector": ["Information Technology", "Information Technology", "Financials"],
            "gics_sub_industry": [
                "Technology Hardware",
                "Systems Software",
                "Multi-Sector Holdings",
            ],
        }
    )


NO_SLEEP = lambda seconds: None  # noqa: E731


# ---------------------------------------------------------------------------
# 1. Ticker normalization on the public output
# ---------------------------------------------------------------------------


def test_get_sp500_tickers_returns_normalized_list() -> None:
    # Arrange
    source = FakeUniverseSource(_sample_constituents())
    loader = UniverseLoader(source=source, cache=None, sleep_fn=NO_SLEEP)

    # Act
    tickers = loader.get_sp500_tickers()

    # Assert: "BRK.B" (Wikipedia notation) becomes "BRK-B" (yfinance notation).
    assert tickers == ["AAPL", "MSFT", "BRK-B"]


# ---------------------------------------------------------------------------
# 2 & 3. Cache freshness / expiry
# ---------------------------------------------------------------------------


def test_fresh_cache_avoids_refetch(tmp_path: Path) -> None:
    # Arrange
    source = FakeUniverseSource(_sample_constituents())
    cache = UniverseCache(tmp_path / "sp500.csv", ttl=timedelta(days=1))
    loader = UniverseLoader(source=source, cache=cache, sleep_fn=NO_SLEEP)
    loader.get_sp500_tickers()
    assert source.call_count == 1

    # Act: a second call well within the TTL window.
    loader.get_sp500_tickers()

    # Assert: served entirely from cache, no new fetch.
    assert source.call_count == 1


def test_expired_cache_triggers_refetch(tmp_path: Path) -> None:
    # Arrange
    cache_path = tmp_path / "sp500.csv"
    source = FakeUniverseSource(_sample_constituents())
    real_now = datetime.now()

    fresh_cache = UniverseCache(cache_path, ttl=timedelta(days=1), now_fn=lambda: real_now)
    UniverseLoader(source=source, cache=fresh_cache, sleep_fn=NO_SLEEP).get_sp500_tickers()
    assert source.call_count == 1

    # Act: read the same cache file, but as if 30 days had passed.
    expired_cache = UniverseCache(
        cache_path, ttl=timedelta(days=1), now_fn=lambda: real_now + timedelta(days=30)
    )
    UniverseLoader(source=source, cache=expired_cache, sleep_fn=NO_SLEEP).get_sp500_tickers()

    # Assert: the TTL had lapsed, so the loader refetched.
    assert source.call_count == 2


# ---------------------------------------------------------------------------
# 4. Retry absorbs transient failures
# ---------------------------------------------------------------------------


def test_transient_failures_are_absorbed_by_retry() -> None:
    # Arrange: fails twice, then succeeds on the 3rd attempt.
    source = FakeUniverseSource(_sample_constituents(), fail_times=2)
    loader = UniverseLoader(source=source, cache=None, retries=3, sleep_fn=NO_SLEEP)

    # Act
    tickers = loader.get_sp500_tickers()

    # Assert: exactly N+1 = 3 calls, and the eventual success is returned.
    assert source.call_count == 3
    assert "AAPL" in tickers


# ---------------------------------------------------------------------------
# 5 & 6. Fallback degradation order
# ---------------------------------------------------------------------------


def test_exhausted_retries_without_cache_falls_back_to_bundled_list(tmp_path: Path) -> None:
    # Arrange: source always fails; no cache; a bundled fallback CSV exists.
    fallback_path = tmp_path / "fallback.csv"
    _sample_constituents().to_csv(fallback_path, index=False)
    source = FakeUniverseSource(_sample_constituents(), fail_times=999)
    loader = UniverseLoader(
        source=source,
        cache=None,
        fallback_path=fallback_path,
        retries=2,
        sleep_fn=NO_SLEEP,
    )

    # Act / Assert: falls back, and warns that stale/fallback data is in use.
    with pytest.warns(UserWarning):
        tickers = loader.get_sp500_tickers()

    assert source.call_count == 2
    assert "AAPL" in tickers


def test_exhausted_retries_prefers_stale_cache_over_bundled_fallback(tmp_path: Path) -> None:
    # Arrange: a stale (TTL-expired) cache exists with different data than
    # the bundled fallback. Both are available; stale cache should win.
    cache_path = tmp_path / "sp500.csv"
    fallback_path = tmp_path / "fallback.csv"

    stale_data = pd.DataFrame(
        {
            "ticker": ["OLDCO"],
            "security": ["Old Co"],
            "gics_sector": ["Industrials"],
            "gics_sub_industry": ["Machinery"],
        }
    )
    UniverseCache(cache_path, ttl=timedelta(days=1)).write(stale_data)
    _sample_constituents().to_csv(fallback_path, index=False)

    expired_cache = UniverseCache(
        cache_path, ttl=timedelta(days=1), now_fn=lambda: datetime.now() + timedelta(days=30)
    )
    source = FakeUniverseSource(_sample_constituents(), fail_times=999)
    loader = UniverseLoader(
        source=source,
        cache=expired_cache,
        fallback_path=fallback_path,
        retries=2,
        sleep_fn=NO_SLEEP,
    )

    # Act / Assert
    with pytest.warns(UserWarning):
        tickers = loader.get_sp500_tickers()

    assert tickers == ["OLDCO"]


# ---------------------------------------------------------------------------
# Full exhaustion: no cache, no fallback -> a clear error, not a hang
# ---------------------------------------------------------------------------


def test_exhausted_retries_without_cache_or_fallback_raises() -> None:
    # Arrange: source always fails; no cache and no fallback configured.
    source = FakeUniverseSource(_sample_constituents(), fail_times=999)
    loader = UniverseLoader(source=source, cache=None, fallback_path=None, retries=2, sleep_fn=NO_SLEEP)

    # Act / Assert
    with pytest.raises(UniverseSourceError):
        loader.get_sp500_tickers()

    assert source.call_count == 2


# ---------------------------------------------------------------------------
# 7. Clear error when the Wikipedia table structure no longer matches
# ---------------------------------------------------------------------------


def test_wikipedia_source_parses_valid_table_successfully() -> None:
    # Arrange: a well-formed constituents table, including an extra column
    # (Headquarters Location) that real Wikipedia has but we don't use.
    html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
          <th>GICS Sub-Industry</th><th>Headquarters Location</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td>
          <td>Technology Hardware</td><td>Cupertino, California</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
          <td>Multi-Sector Holdings</td><td>Omaha, Nebraska</td></tr>
    </table>
    """
    source = WikipediaUniverseSource()

    # Act
    result = source._parse_html(html)

    # Assert: only the expected columns are kept, values parsed correctly.
    assert list(result.columns) == ["ticker", "security", "gics_sector", "gics_sub_industry"]
    assert result.loc[0, "ticker"] == "AAPL"
    assert result.loc[1, "ticker"] == "BRK.B"


def test_wikipedia_source_raises_clear_error_when_table_not_found() -> None:
    # Arrange: no table with id="constituents" at all.
    html = "<table id='some-other-table'><tr><th>Foo</th></tr></table>"
    source = WikipediaUniverseSource()

    # Act / Assert
    with pytest.raises(UniverseSourceError) as exc_info:
        source._parse_html(html)

    assert "constituents" in str(exc_info.value)


def test_wikipedia_source_raises_clear_error_when_columns_missing() -> None:
    # Arrange: a constituents table missing the GICS Sector/Sub-Industry
    # columns, simulating a Wikipedia page structure change.
    html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th></tr>
      <tr><td>AAPL</td><td>Apple Inc.</td></tr>
    </table>
    """
    source = WikipediaUniverseSource()

    # Act / Assert
    with pytest.raises(UniverseSourceError) as exc_info:
        source._parse_html(html)

    message = str(exc_info.value)
    assert "GICS Sector" in message
    assert "GICS Sub-Industry" in message


# ---------------------------------------------------------------------------
# 8. Ticker normalization edge cases
# ---------------------------------------------------------------------------


def test_normalize_ticker_handles_period_whitespace_and_case() -> None:
    assert _normalize_ticker("BRK.B") == "BRK-B"
    assert _normalize_ticker("AAPL") == "AAPL"
    assert _normalize_ticker("  msft  ") == "MSFT"


def test_universe_cache_read_stale_returns_none_when_never_written(tmp_path: Path) -> None:
    cache = UniverseCache(tmp_path / "never_written.csv")
    assert cache.read_stale() is None
