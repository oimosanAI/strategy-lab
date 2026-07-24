"""Data-source abstraction layer.

`DataLoader` is the single entry point strategies and the backtest engine
use to obtain price history. It knows nothing about yfinance (or any other
vendor) -- that is confined to whatever `DataProvider` implementation is
injected -- and it knows nothing about the storage format used to avoid
redundant API calls -- that is confined to whatever `CacheBackend` is
injected. Swapping data vendors or cache storage later means writing a new
adapter, not touching this module.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol, Sequence

import pandas as pd

from core.data.cache import CacheBackend, CachedSeries

PRICE_FIELDS: list[str] = ["open", "high", "low", "close", "adj_close", "volume"]
PRICE_COLUMNS: list[str] = ["date", *PRICE_FIELDS]
TIDY_COLUMNS: list[str] = ["date", "ticker", *PRICE_FIELDS]


class DataProviderError(Exception):
    """Raised when a DataProvider fails to fetch data for a ticker."""


class DataProvider(Protocol):
    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV rows for a single ticker within [start, end] inclusive.

        Must return a DataFrame with columns PRICE_COLUMNS. An empty frame
        (not an exception) is the correct response when the ticker simply
        has no data in the range, e.g. a request predating its IPO.
        """
        ...


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            **{field: pd.Series(dtype="float64") for field in PRICE_FIELDS},
        }
    )


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce a provider/cache frame to the canonical PRICE_COLUMNS schema
    and dtypes, so concatenation never silently upcasts to object dtype."""
    if raw.empty:
        return _empty_price_frame()
    out = raw.reindex(columns=PRICE_COLUMNS).copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


class DataLoader:
    def __init__(self, provider: DataProvider, cache: CacheBackend) -> None:
        self._provider = provider
        self._cache = cache

    def get_prices(self, tickers: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """Return tidy OHLCV rows for `tickers` within [start, end] inclusive.

        `end` is inclusive: `end`'s own close is the newest data returned,
        which is what lets a future backtest engine request "everything
        known as of t" by passing `end=t`.

        Fails fast: if any ticker's fetch fails, the whole call raises
        DataProviderError and no partial results (from other tickers that
        did succeed) are returned.
        """
        frames = [self._get_tagged_ticker_frame(ticker, start, end) for ticker in tickers]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=TIDY_COLUMNS)
        result = pd.concat(frames, ignore_index=True)
        result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
        return result[TIDY_COLUMNS]

    def get_price_panel(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
        field: str = "adj_close",
    ) -> pd.DataFrame:
        """Pivot get_prices() into wide format: index=date, columns=ticker.

        Missing values (a ticker not trading on a given date, or having no
        data at all in the range) are left as NaN. No filling of any kind
        is performed here -- that is the backtest engine's responsibility.
        """
        tidy = self.get_prices(tickers, start, end)
        if tidy.empty:
            return pd.DataFrame(columns=list(tickers))
        panel = tidy.pivot(index="date", columns="ticker", values=field)
        return panel.reindex(columns=list(tickers))

    def _get_tagged_ticker_frame(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        data = self._get_ticker_data(ticker, start, end)
        if data.empty:
            return data
        tagged = data.copy()
        tagged.insert(1, "ticker", ticker)
        return tagged

    def _get_ticker_data(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        cached = self._cache.read(ticker)

        if cached is None:
            merged = _normalize(self._provider.fetch(ticker, start, end))
            covered_start, covered_end = start, end
        else:
            # Invariant: covered_start/covered_end always describe a single
            # contiguous queried interval, never a set of disjoint ranges.
            # This holds even if [start, end] does not overlap the previous
            # coverage at all: the head/tail branches below always bridge
            # from the *old* edge to the *new* edge, so any untouched middle
            # gap gets folded into whichever branch fires, and is queried
            # (not merely assumed covered) before covered_start/covered_end
            # are widened to include it.
            merged = cached.data
            covered_start, covered_end = cached.covered_start, cached.covered_end
            fetched_any = False

            if start < cached.covered_start:
                gap_end = cached.covered_start - timedelta(days=1)
                merged = pd.concat(
                    [merged, _normalize(self._provider.fetch(ticker, start, gap_end))],
                    ignore_index=True,
                )
                covered_start = start
                fetched_any = True

            if end > cached.covered_end:
                gap_start = cached.covered_end + timedelta(days=1)
                merged = pd.concat(
                    [merged, _normalize(self._provider.fetch(ticker, gap_start, end))],
                    ignore_index=True,
                )
                covered_end = end
                fetched_any = True

            if fetched_any:
                merged = merged.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

        self._cache.write(
            ticker,
            CachedSeries(data=merged, covered_start=covered_start, covered_end=covered_end),
        )

        mask = (merged["date"] >= pd.Timestamp(start)) & (merged["date"] <= pd.Timestamp(end))
        return merged.loc[mask].reset_index(drop=True)
