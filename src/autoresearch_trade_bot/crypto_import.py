from __future__ import annotations

import csv
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .config import DataConfig
from .datasets import DatasetSpec, ValidatedDataset, ensure_utc
from .models import Bar
from .storage import PyArrowParquetDatasetStore
from .validation import DatasetValidator
from .data import HistoricalDatasetMaterializer


_FILENAME_RE = re.compile(
    r"^(?P<symbol>[A-Z0-9]+)-(?P<timeframe>\d+[mhd])-(?P<period>.+)\.(?P<suffix>csv|zip)$"
)


@dataclass(frozen=True)
class CryptoDataFile:
    path: Path
    symbol: str
    timeframe: str
    period: str
    source_priority: int
    sort_key: tuple[int, datetime]


def discover_crypto_data_files(
    *,
    source_root: str | Path,
    symbols: Sequence[str],
    timeframe: str,
) -> list[CryptoDataFile]:
    symbol_set = {symbol.upper() for symbol in symbols}
    files_by_key: dict[tuple[str, str], CryptoDataFile] = {}
    for path in sorted(Path(source_root).iterdir()):
        if not path.is_file():
            continue
        match = _FILENAME_RE.match(path.name)
        if match is None:
            continue
        symbol = match.group("symbol").upper()
        if symbol not in symbol_set:
            continue
        file_timeframe = match.group("timeframe")
        if file_timeframe != timeframe:
            continue
        period = match.group("period")
        suffix = match.group("suffix")
        source_priority = 2 if suffix == "csv" else 1
        candidate = CryptoDataFile(
            path=path,
            symbol=symbol,
            timeframe=file_timeframe,
            period=period,
            source_priority=source_priority,
            sort_key=_period_sort_key(period),
        )
        key = (symbol, period)
        existing = files_by_key.get(key)
        if existing is None or candidate.source_priority > existing.source_priority:
            files_by_key[key] = candidate
    return sorted(
        files_by_key.values(),
        key=lambda item: (item.symbol, item.sort_key[0], item.sort_key[1], item.path.name),
    )


def import_crypto_data_directory(
    *,
    source_root: str | Path,
    storage_root: str | Path,
    symbols: Sequence[str],
    timeframe: str,
    exchange: str = "binance",
    market: str = "usdm_futures",
    warning_stream=None,
) -> list[ValidatedDataset]:
    resolved_warning_stream = warning_stream or sys.stderr
    selected_files = discover_crypto_data_files(
        source_root=source_root,
        symbols=symbols,
        timeframe=timeframe,
    )
    if not selected_files:
        raise ValueError("no matching crypto_data files found")

    bars_by_symbol: dict[str, dict[datetime, Bar]] = {symbol: {} for symbol in symbols}
    for source_file in selected_files:
        try:
            for row in _iter_kline_rows(source_file.path):
                bar = _row_to_bar(
                    row=row,
                    exchange=exchange,
                    symbol=source_file.symbol,
                    timeframe=timeframe,
                )
                bars_by_symbol[source_file.symbol][bar.timestamp] = bar
        except zipfile.BadZipFile:
            print(
                f"warning: skipping invalid zip archive {source_file.path}",
                file=resolved_warning_stream,
            )
            continue
        except ValueError as exc:
            print(
                f"warning: skipping unreadable archive {source_file.path}: {exc}",
                file=resolved_warning_stream,
            )
            continue

    normalized_bars_by_symbol = {
        symbol: [bars_by_symbol[symbol][timestamp] for timestamp in sorted(bars_by_symbol[symbol])]
        for symbol in symbols
    }
    empty_symbols = [symbol for symbol, bars in normalized_bars_by_symbol.items() if not bars]
    if empty_symbols:
        raise ValueError(
            "missing imported bars for symbols: %s" % ", ".join(sorted(empty_symbols))
        )

    earliest_timestamp = min(bars[0].timestamp for bars in normalized_bars_by_symbol.values())
    latest_timestamp = max(bars[-1].timestamp for bars in normalized_bars_by_symbol.values())
    segmented_bars = _segment_bars_by_symbol(
        bars_by_symbol=normalized_bars_by_symbol,
        timeframe=timeframe,
    )
    data_config = DataConfig(
        exchange=exchange,
        market=market,
        timeframe=timeframe,
        storage_root=str(storage_root),
        strict_validation=True,
        default_start=earliest_timestamp,
        default_end=latest_timestamp + _timeframe_to_timedelta(timeframe),
    )
    materializer = HistoricalDatasetMaterializer(
        data_config=data_config,
        client=_StaticHistoricalClient(),
        store=PyArrowParquetDatasetStore(storage_root),
        validator=DatasetValidator(),
        normalizer=_PassthroughNormalizer(),
    )
    datasets: list[ValidatedDataset] = []
    for segment in segmented_bars:
        start = min(bars[0].timestamp for bars in segment.values())
        last_timestamp = max(bars[-1].timestamp for bars in segment.values())
        spec = DatasetSpec(
            exchange=exchange,
            market=market,
            timeframe=timeframe,
            start=start,
            end=last_timestamp + _timeframe_to_timedelta(timeframe),
            symbols=tuple(symbols),
        )
        datasets.append(materializer._write_validated_dataset(spec, segment))
    return datasets


class _StaticHistoricalClient:
    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str):
        raise NotImplementedError("static importer does not fetch remote history")


class _PassthroughNormalizer:
    def normalize(self, spec: DatasetSpec, raw):
        raise NotImplementedError("static importer bypasses raw normalization")


def _iter_kline_rows(path: Path) -> Iterator[list[str]]:
    if path.suffix == ".csv":
        yield from _iter_csv_lines(path)
        return
    if path.suffix != ".zip":
        raise ValueError(f"unsupported crypto_data file suffix for {path}")
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if not members:
            raise ValueError("zip archive contains no files")
        member = members[0]
        with archive.open(member) as handle:
            stream = io.TextIOWrapper(handle, encoding="utf-8", errors="ignore")
            reader = csv.reader(stream)
            for row in reader:
                if row:
                    yield row


def _iter_csv_lines(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row:
                yield row


def _row_to_bar(
    *,
    row: Sequence[str],
    exchange: str,
    symbol: str,
    timeframe: str,
) -> Bar:
    if len(row) < 12:
        raise ValueError(f"expected at least 12 columns, got {len(row)}")
    open_time = _timestamp_to_datetime(int(row[0]))
    close_time = _timestamp_to_datetime(int(row[6]))
    close_price = float(row[4])
    return Bar(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        timestamp=open_time,
        close_time=close_time,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=close_price,
        volume=float(row[5]),
        trade_count=int(row[8]),
        funding_rate=0.0,
        open_interest=0.0,
        mark_price=close_price,
        index_price=close_price,
        is_closed=True,
    )


def _timestamp_to_datetime(value: int) -> datetime:
    if value >= 10**15:
        return datetime.fromtimestamp(value / 1_000_000, tz=timezone.utc)
    if value >= 10**12:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _segment_bars_by_symbol(
    *,
    bars_by_symbol: dict[str, list[Bar]],
    timeframe: str,
) -> list[dict[str, list[Bar]]]:
    expected_step = _timeframe_to_timedelta(timeframe)
    segmented = {
        symbol: _split_contiguous_segments(bars, expected_step)
        for symbol, bars in bars_by_symbol.items()
    }
    symbols = tuple(bars_by_symbol)
    segment_counts = {symbol: len(segments) for symbol, segments in segmented.items()}
    if len(set(segment_counts.values())) != 1:
        raise ValueError(f"symbols do not share the same segment count: {segment_counts}")

    datasets: list[dict[str, list[Bar]]] = []
    segment_total = next(iter(segment_counts.values()))
    for segment_index in range(segment_total):
        current: dict[str, list[Bar]] = {}
        starts = set()
        ends = set()
        for symbol in symbols:
            bars = segmented[symbol][segment_index]
            current[symbol] = bars
            starts.add(bars[0].timestamp)
            ends.add(bars[-1].timestamp)
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError(
                "symbols do not align on contiguous segment boundaries for segment "
                f"{segment_index}: starts={sorted(starts)} ends={sorted(ends)}"
            )
        datasets.append(current)
    return datasets


def _split_contiguous_segments(bars: Sequence[Bar], expected_step: timedelta) -> list[list[Bar]]:
    if not bars:
        return []
    segments: list[list[Bar]] = []
    current_segment = [bars[0]]
    for bar in bars[1:]:
        if bar.timestamp - current_segment[-1].timestamp != expected_step:
            segments.append(current_segment)
            current_segment = [bar]
            continue
        current_segment.append(bar)
    segments.append(current_segment)
    return segments


def _period_sort_key(period: str) -> tuple[int, datetime]:
    normalized = period.replace(".csv", "").replace(".zip", "")
    for rank, fmt in (
        (0, "%Y-%m-%d"),
        (1, "%Y-%m"),
        (1, "%Y-%m"),
    ):
        try:
            return rank, ensure_utc(datetime.strptime(normalized, fmt))
        except ValueError:
            continue
    # Handle odd month naming like 2026-1 or 2026-2.
    parts = normalized.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        year = int(parts[0])
        month = int(parts[1])
        return 1, datetime(year, month, 1, tzinfo=timezone.utc)
    raise ValueError(f"unsupported crypto_data period format: {period}")


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    unit = timeframe[-1]
    magnitude = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    if unit == "d":
        return timedelta(days=magnitude)
    raise ValueError(f"unsupported timeframe: {timeframe}")
