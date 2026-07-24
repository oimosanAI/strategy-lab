"""S&P 500 constituent universe.

`UniverseLoader.get_sp500_tickers()` is the single entry point strategies
use to obtain the current S&P 500 ticker list, ready to hand straight to
`core.data.loader.DataLoader.get_prices(tickers=...)`. As with
`core.data.loader`, the actual data source is injected (`UniverseSource`)
so it can be swapped later; `UniverseLoader` owns everything about making
that source reliable (retry, caching, fallback) and about producing
yfinance-ready ticker strings.

KNOWN LIMITATION (Level A survivorship bias, REQUIREMENTS.md section 7.1):
this always returns *today's* constituents. It does not reconstruct S&P 500
membership as of a past date, so backtests built on this universe are
subject to survivorship bias -- see README for the documented limitation.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd
import requests

CONSTITUENT_COLUMNS: list[str] = ["ticker", "security", "gics_sector", "gics_sub_industry"]

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_CACHE_PATH = Path("data/cache/sp500_constituents.csv")
DEFAULT_FALLBACK_PATH = Path(__file__).parent / "sp500_fallback.csv"

_WIKIPEDIA_EXPECTED_COLUMNS = ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]
_WIKIPEDIA_COLUMN_RENAME = {
    "Symbol": "ticker",
    "Security": "security",
    "GICS Sector": "gics_sector",
    "GICS Sub-Industry": "gics_sub_industry",
}


class UniverseSourceError(Exception):
    """Raised when a UniverseSource fails to fetch/parse constituent data."""


class UniverseSource(Protocol):
    def fetch_constituents(self) -> pd.DataFrame:
        """Fetch the current constituent table in a single attempt.

        Must return a DataFrame with columns CONSTITUENT_COLUMNS, tickers
        in the source's native notation (not yet normalized for yfinance).
        Raises UniverseSourceError on failure; does not retry internally
        -- retrying is UniverseLoader's responsibility.
        """
        ...


def _normalize_ticker(ticker: str) -> str:
    """Convert a source-native ticker to yfinance notation, e.g. the
    Wikipedia-style 'BRK.B' becomes yfinance-style 'BRK-B'."""
    return ticker.strip().upper().replace(".", "-")


class WikipediaUniverseSource:
    """Scrapes the "List of S&P 500 companies" Wikipedia page."""

    def __init__(self, url: str = WIKIPEDIA_URL, timeout_seconds: float = 10.0) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    def fetch_constituents(self) -> pd.DataFrame:
        try:
            response = requests.get(
                self._url,
                timeout=self._timeout_seconds,
                headers={"User-Agent": "strategy-lab/0.1 (github.com/oimosanAI/strategy-lab)"},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UniverseSourceError(f"Failed to fetch {self._url}: {exc}") from exc
        return self._parse_html(response.text)

    def _parse_html(self, html: str) -> pd.DataFrame:
        try:
            tables = pd.read_html(StringIO(html), attrs={"id": "constituents"}, flavor="lxml")
        except ValueError as exc:
            raise UniverseSourceError(
                "Could not find a table with id='constituents' in the Wikipedia page; "
                "the page structure may have changed."
            ) from exc

        table = tables[0]
        missing = [col for col in _WIKIPEDIA_EXPECTED_COLUMNS if col not in table.columns]
        if missing:
            raise UniverseSourceError(
                f"Expected columns {_WIKIPEDIA_EXPECTED_COLUMNS} in the S&P 500 "
                f"constituents table, but {missing} were missing "
                f"(found columns: {list(table.columns)}). "
                "The Wikipedia page structure may have changed."
            )
        return table.rename(columns=_WIKIPEDIA_COLUMN_RENAME)[CONSTITUENT_COLUMNS]


@dataclass(frozen=True)
class UniverseCache:
    """Point-in-time snapshot cache with a freshness TTL.

    Deliberately unrelated to core.data.cache.CacheBackend: that models
    per-ticker date-range coverage for time series, whereas this models a
    single "as of now" snapshot that is either fresh enough to reuse or
    isn't. Forcing the two into one abstraction would be a false shared
    interface.
    """

    cache_path: Path
    ttl: timedelta = timedelta(days=1)
    now_fn: Callable[[], datetime] = datetime.now

    def read_if_fresh(self) -> pd.DataFrame | None:
        if not self.cache_path.exists():
            return None
        mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
        if self.now_fn() - mtime > self.ttl:
            return None
        return pd.read_csv(self.cache_path)

    def read_stale(self) -> pd.DataFrame | None:
        if not self.cache_path.exists():
            return None
        return pd.read_csv(self.cache_path)

    def write(self, data: pd.DataFrame) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self.cache_path, index=False)


class UniverseLoader:
    """Owns retry/backoff, cache-then-fetch-then-stale-cache-then-bundled-
    fallback degradation, and ticker normalization."""

    def __init__(
        self,
        source: UniverseSource,
        cache: UniverseCache | None = None,
        fallback_path: Path | None = None,
        retries: int = 3,
        backoff_seconds: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._source = source
        self._cache = cache
        self._fallback_path = fallback_path
        self._retries = retries
        self._backoff_seconds = backoff_seconds
        self._sleep_fn = sleep_fn

    def get_sp500_tickers(self) -> list[str]:
        """Return current S&P 500 tickers, normalized for yfinance, ready
        to pass straight into DataLoader.get_prices(tickers=...).

        See module docstring for the Level A survivorship-bias limitation.
        """
        data = self._get_constituents()
        return [_normalize_ticker(ticker) for ticker in data["ticker"]]

    def _get_constituents(self) -> pd.DataFrame:
        if self._cache is not None:
            fresh = self._cache.read_if_fresh()
            if fresh is not None:
                return fresh

        last_error: Exception | None = None
        for attempt in range(self._retries):
            try:
                data = self._source.fetch_constituents()
            except UniverseSourceError as exc:
                last_error = exc
                if attempt < self._retries - 1:
                    self._sleep_fn(self._backoff_seconds)
                continue
            if self._cache is not None:
                self._cache.write(data)
            return data

        if self._cache is not None:
            stale = self._cache.read_stale()
            if stale is not None:
                warnings.warn(
                    f"S&P 500 universe fetch failed after {self._retries} attempts; "
                    "using stale cached data, which may not reflect the current "
                    "index composition.",
                    stacklevel=2,
                )
                return stale

        if self._fallback_path is not None and Path(self._fallback_path).exists():
            warnings.warn(
                f"S&P 500 universe fetch failed after {self._retries} attempts and no "
                "cache is available; using the bundled fallback list, which is a "
                "point-in-time snapshot and may be stale.",
                stacklevel=2,
            )
            return pd.read_csv(self._fallback_path)

        raise UniverseSourceError(
            f"Failed to fetch S&P 500 constituents after {self._retries} attempts, "
            "and no cache or fallback data is available."
        ) from last_error


def get_sp500_tickers() -> list[str]:
    """Convenience wrapper: WikipediaUniverseSource + default on-disk cache
    + the bundled fallback list, using the default retry/backoff settings."""
    loader = UniverseLoader(
        source=WikipediaUniverseSource(),
        cache=UniverseCache(DEFAULT_CACHE_PATH),
        fallback_path=DEFAULT_FALLBACK_PATH,
    )
    return loader.get_sp500_tickers()
