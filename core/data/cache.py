"""Local cache for OHLCV data, keyed per ticker.

The cache records not just the rows it has, but the date range that has
been queried (`covered_start`/`covered_end`), independent of whether any
rows exist in that range. Without this, a ticker with no data for a given
period (e.g. before its IPO) would look indistinguishable from a period
that has simply never been requested, and DataLoader would re-fetch it
from the provider on every call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class CachedSeries:
    data: pd.DataFrame
    covered_start: date
    covered_end: date


class CacheBackend(Protocol):
    def read(self, ticker: str) -> CachedSeries | None: ...

    def write(self, ticker: str, series: CachedSeries) -> None: ...


class ParquetCache:
    """Filesystem-backed CacheBackend: one parquet file + one JSON
    sidecar (holding covered_start/covered_end) per ticker."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)

    def read(self, ticker: str) -> CachedSeries | None:
        data_path = self._data_path(ticker)
        meta_path = self._meta_path(ticker)
        if not data_path.exists() or not meta_path.exists():
            return None
        data = pd.read_parquet(data_path)
        meta = json.loads(meta_path.read_text())
        return CachedSeries(
            data=data,
            covered_start=date.fromisoformat(meta["covered_start"]),
            covered_end=date.fromisoformat(meta["covered_end"]),
        )

    def write(self, ticker: str, series: CachedSeries) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        series.data.to_parquet(self._data_path(ticker))
        meta = {
            "covered_start": series.covered_start.isoformat(),
            "covered_end": series.covered_end.isoformat(),
        }
        self._meta_path(ticker).write_text(json.dumps(meta))

    def _data_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.parquet"

    def _meta_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.meta.json"
