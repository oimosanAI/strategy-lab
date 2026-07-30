"""Data-source abstraction layer.

`DataLoader` is the single entry point strategies and the backtest engine
use to obtain price history. It knows nothing about yfinance (or any other
vendor) -- that is confined to whatever `DataProvider` implementation is
injected -- and it knows nothing about the storage format used to avoid
redundant API calls -- that is confined to whatever `CacheBackend` is
injected. Swapping data vendors or cache storage later means writing a new
adapter, not touching this module.

Known limitation (not yet implemented): no stale/frozen-price detection.
A ticker whose feed has stopped updating (halted, delisted, a vendor
outage) currently passes through unchanged -- a run of identical daily
closes looks, to every downstream consumer, like a real zero-volatility
asset. This matters concretely for
core.backtest.position_sizing.VolTargetSizer, which clips to
+/-max_leverage whenever realized_vol == 0.0 exactly, with no way to
distinguish "genuinely flat" from "feed broken" from realized_vol alone
(see that module's docstring). The correct fix is here, not in the
sizer: something like "flag any ticker with N consecutive identical
closes" is a data-quality check that belongs at the point prices enter
the system, the same way CacheBackend/DataProvider validation already
does, not a numeric threshold bolted onto a sizing formula downstream.
Deferred rather than implemented now for the same reason
core.backtest.portfolio.ExposureLimits leaves sector-concentration
limits out: it needs infrastructure (here, a staleness-tracking policy
decision: how many consecutive identical closes count as stale, per
asset class) that hasn't been scoped yet.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Protocol, Sequence

import pandas as pd
import yfinance as yf

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


_YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class YFinanceProvider:
    """Concrete DataProvider backed by yfinance.

    Empty (not an exception) is returned for a ticker with no data in the
    range -- e.g. a delisted/mistyped ticker or a range predating a
    listing -- matching yfinance's own behaviour and the DataProvider
    contract. Network/parsing failures are wrapped as DataProviderError.
    """

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        try:
            # yfinance's `end` is exclusive; DataProvider.fetch's is inclusive.
            raw = yf.download(
                ticker,
                start=start,
                end=end + timedelta(days=1),
                progress=False,
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted, undocumented types
            raise DataProviderError(f"Failed to fetch {ticker} from yfinance: {exc}") from exc

        if raw.empty:
            return _empty_price_frame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)

        out = pd.DataFrame({"date": pd.to_datetime(raw.index)})
        for yf_col, our_col in _YFINANCE_COLUMN_MAP.items():
            out[our_col] = raw[yf_col].to_numpy()
        return out.reset_index(drop=True)


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
