from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

from .datasets import DatasetManifest, DatasetSpec
from .models import Bar


class DatasetStore(Protocol):
    def write_bars(self, spec: DatasetSpec, symbol: str, bars: Sequence[Bar]) -> Path:
        ...

    def read_bars(self, path: Path) -> list[Bar]:
        ...


class PyArrowParquetDatasetStore:
    """Filesystem-backed parquet store for validated research datasets."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def dataset_dir(self, spec: DatasetSpec) -> Path:
        return self.root / spec.exchange / spec.market / spec.timeframe / spec.dataset_id

    def write_bars(self, spec: DatasetSpec, symbol: str, bars: Sequence[Bar]) -> Path:
        pa, pq = self._import_pyarrow()
        dataset_dir = self.dataset_dir(spec)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        path = dataset_dir / f"{symbol}.parquet"
        table = pa.table(
            {
                "exchange": [bar.exchange for bar in bars],
                "symbol": [bar.symbol for bar in bars],
                "timeframe": [bar.timeframe for bar in bars],
                "timestamp": [bar.timestamp.isoformat() for bar in bars],
                "close_time": [
                    bar.close_time.isoformat() if bar.close_time is not None else None
                    for bar in bars
                ],
                "open": [bar.open for bar in bars],
                "high": [bar.high for bar in bars],
                "low": [bar.low for bar in bars],
                "close": [bar.close for bar in bars],
                "volume": [bar.volume for bar in bars],
                "trade_count": [bar.trade_count for bar in bars],
                "funding_rate": [bar.funding_rate for bar in bars],
                "open_interest": [bar.open_interest for bar in bars],
                "mark_price": [bar.mark_price for bar in bars],
                "index_price": [bar.index_price for bar in bars],
                "is_closed": [bar.is_closed for bar in bars],
            }
        )
        pq.write_table(table, path)
        return path

    def read_bars(self, path: Path) -> list[Bar]:
        _pa, pq = self._import_pyarrow()
        table = pq.read_table(path)
        rows = table.to_pylist()
        bars = []
        for row in rows:
            bars.append(
                Bar(
                    exchange=row["exchange"],
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    timestamp=self._parse_datetime(row["timestamp"]),
                    close_time=self._parse_datetime(row["close_time"])
                    if row["close_time"]
                    else None,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    trade_count=int(row["trade_count"]),
                    funding_rate=float(row["funding_rate"]),
                    open_interest=float(row["open_interest"]),
                    mark_price=float(row["mark_price"])
                    if row["mark_price"] is not None
                    else None,
                    index_price=float(row["index_price"])
                    if row["index_price"] is not None
                    else None,
                    is_closed=bool(row["is_closed"]),
                )
            )
        return bars

    def write_manifest(self, manifest: DatasetManifest, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
        return path

    def read_manifest(self, path: Path) -> DatasetManifest:
        return DatasetManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _import_pyarrow(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "pyarrow is required for parquet dataset storage. Install project dependencies first."
            ) from exc
        return pa, pq

    @staticmethod
    def _parse_datetime(value: str):
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
