from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .binance import BinanceBarNormalizer, BinanceUSDMHistoricalClient
from .bybit import BybitBarNormalizer, BybitLinearHistoricalClient
from .config import DataConfig
from .datasets import DatasetFile, DatasetManifest, DatasetSpec, RawSymbolHistory, ValidatedDataset
from .models import Bar
from .storage import DatasetStore, PyArrowParquetDatasetStore
from .validation import DatasetValidator


class HistoricalDataSource(Protocol):
    """Interface for validated historical datasets."""

    def load_bars(self, dataset_spec: DatasetSpec) -> Mapping[str, Sequence[Bar]]:
        ...


class RealtimeBarSource(Protocol):
    """Interface for a future paper/shadow market data stream."""

    def subscribe_bars(self, symbols: Sequence[str], timeframe: str) -> None:
        ...


class HistoricalMarketClient(Protocol):
    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        ...


class HistoricalBarNormalizer(Protocol):
    def normalize(self, spec: DatasetSpec, raw: RawSymbolHistory) -> list[Bar]:
        ...


class DataValidationError(ValueError):
    def __init__(self, issues) -> None:
        super().__init__("dataset validation failed")
        self.issues = issues


def _manifest_search_root(storage_root: str | Path, spec: DatasetSpec) -> Path:
    return Path(storage_root) / spec.exchange / spec.market / spec.timeframe


def _manifest_matches_spec(manifest: DatasetManifest, spec: DatasetSpec) -> bool:
    return (
        manifest.exchange == spec.exchange
        and manifest.market == spec.market
        and manifest.timeframe == spec.timeframe
        and tuple(manifest.symbols) == tuple(spec.symbols)
    )


def find_covering_manifest(
    storage_root: str | Path,
    spec: DatasetSpec,
    store: DatasetStore,
) -> Path | None:
    search_root = _manifest_search_root(storage_root, spec)
    if not search_root.exists():
        return None

    candidates: list[tuple[timedelta, datetime, Path]] = []
    for manifest_path in search_root.glob("**/manifest.json"):
        manifest = store.read_manifest(manifest_path)
        if not _manifest_matches_spec(manifest, spec):
            continue
        if manifest.start <= spec.start and manifest.end >= spec.end:
            candidates.append((manifest.end - manifest.start, manifest.generated_at, manifest_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=False)
    return candidates[0][2]


def find_latest_reusable_manifest(
    storage_root: str | Path,
    spec: DatasetSpec,
    store: DatasetStore,
) -> Path | None:
    search_root = _manifest_search_root(storage_root, spec)
    if not search_root.exists():
        return None

    candidates: list[tuple[datetime, datetime, Path]] = []
    for manifest_path in search_root.glob("**/manifest.json"):
        manifest = store.read_manifest(manifest_path)
        if not _manifest_matches_spec(manifest, spec):
            continue
        if manifest.start <= spec.start and spec.start < manifest.end < spec.end:
            candidates.append((manifest.end, manifest.generated_at, manifest_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


@dataclass
class HistoricalDatasetMaterializer:
    data_config: DataConfig
    client: HistoricalMarketClient
    store: DatasetStore
    validator: DatasetValidator
    normalizer: HistoricalBarNormalizer

    @classmethod
    def for_binance(
        cls,
        data_config: DataConfig,
        store: DatasetStore | None = None,
    ) -> "HistoricalDatasetMaterializer":
        return cls(
            data_config=data_config,
            client=BinanceUSDMHistoricalClient(data_config=data_config),
            store=store or PyArrowParquetDatasetStore(data_config.storage_root),
            validator=DatasetValidator(),
            normalizer=BinanceBarNormalizer(exchange=data_config.exchange),
        )

    @classmethod
    def for_bybit(
        cls,
        data_config: DataConfig,
        store: DatasetStore | None = None,
    ) -> "HistoricalDatasetMaterializer":
        return cls(
            data_config=data_config,
            client=BybitLinearHistoricalClient(data_config=data_config),
            store=store or PyArrowParquetDatasetStore(data_config.storage_root),
            validator=DatasetValidator(),
            normalizer=BybitBarNormalizer(exchange=data_config.exchange),
        )

    @classmethod
    def for_exchange(
        cls,
        data_config: DataConfig,
        store: DatasetStore | None = None,
    ) -> "HistoricalDatasetMaterializer":
        if data_config.exchange == "binance":
            return cls.for_binance(data_config, store=store)
        if data_config.exchange == "bybit":
            return cls.for_bybit(data_config, store=store)
        raise ValueError("unsupported historical exchange: %s" % data_config.exchange)

    def materialize(self, dataset_spec: DatasetSpec) -> ValidatedDataset:
        bars_by_symbol = {}
        for symbol in dataset_spec.symbols:
            raw = self.client.fetch_symbol_history(dataset_spec, symbol)
            bars = self.normalizer.normalize(dataset_spec, raw)
            bars_by_symbol[symbol] = bars

        return self._write_validated_dataset(dataset_spec, bars_by_symbol)

    def materialize_incremental(
        self,
        dataset_spec: DatasetSpec,
        base_manifest_path: str | Path,
    ) -> ValidatedDataset:
        manifest_path = Path(base_manifest_path)
        base_manifest = self.store.read_manifest(manifest_path)
        loader = ManifestHistoricalDataSource(
            manifest_path=manifest_path,
            store=self.store,
            manifest=base_manifest,
            storage_root=Path(self.data_config.storage_root),
        )
        cached_end = min(dataset_spec.end, base_manifest.end)
        cached_spec = DatasetSpec(
            exchange=dataset_spec.exchange,
            market=dataset_spec.market,
            timeframe=dataset_spec.timeframe,
            start=dataset_spec.start,
            end=cached_end,
            symbols=dataset_spec.symbols,
        )
        bars_by_symbol = {
            symbol: list(bars)
            for symbol, bars in loader.load_bars(cached_spec).items()
        }
        if cached_end < dataset_spec.end:
            tail_spec = DatasetSpec(
                exchange=dataset_spec.exchange,
                market=dataset_spec.market,
                timeframe=dataset_spec.timeframe,
                start=cached_end,
                end=dataset_spec.end,
                symbols=dataset_spec.symbols,
            )
            for symbol in dataset_spec.symbols:
                raw = self.client.fetch_symbol_history(tail_spec, symbol)
                tail_bars = self.normalizer.normalize(tail_spec, raw)
                bars_by_symbol[symbol] = self._merge_bars(
                    bars_by_symbol.get(symbol, []),
                    tail_bars,
                )

        return self._write_validated_dataset(dataset_spec, bars_by_symbol)

    def _write_validated_dataset(
        self,
        dataset_spec: DatasetSpec,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
    ) -> ValidatedDataset:
        files = {}

        report = self.validator.validate(dataset_spec, bars_by_symbol)
        if self.data_config.strict_validation and report.has_errors:
            raise DataValidationError(report.issues)

        for symbol, bars in bars_by_symbol.items():
            path = self.store.write_bars(dataset_spec, symbol, bars)
            files[symbol] = DatasetFile(
                symbol=symbol,
                relative_path=str(path.relative_to(Path(self.data_config.storage_root))),
                row_count=len(bars),
                first_timestamp=bars[0].timestamp,
                last_timestamp=bars[-1].timestamp,
            )

        manifest = DatasetManifest(
            dataset_id=dataset_spec.dataset_id,
            exchange=dataset_spec.exchange,
            market=dataset_spec.market,
            timeframe=dataset_spec.timeframe,
            start=dataset_spec.start,
            end=dataset_spec.end,
            generated_at=datetime.now(timezone.utc),
            symbols=dataset_spec.symbols,
            files=files,
            validation_issues=report.issues,
        )

        manifest_path = Path(self.data_config.storage_root) / dataset_spec.exchange / dataset_spec.market / dataset_spec.timeframe / dataset_spec.dataset_id / "manifest.json"
        if hasattr(self.store, "write_manifest"):
            self.store.write_manifest(manifest, manifest_path)
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
            )

        return ValidatedDataset(
            spec=dataset_spec,
            manifest=manifest,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _merge_bars(existing_bars: Sequence[Bar], new_bars: Sequence[Bar]) -> list[Bar]:
        merged = {bar.timestamp: bar for bar in existing_bars}
        for bar in new_bars:
            merged[bar.timestamp] = bar
        return [merged[timestamp] for timestamp in sorted(merged)]


@dataclass
class ManifestHistoricalDataSource(HistoricalDataSource):
    manifest_path: Path
    store: DatasetStore
    manifest: DatasetManifest
    storage_root: Path

    @classmethod
    def from_manifest_path(
        cls,
        manifest_path: str | Path,
        store: PyArrowParquetDatasetStore | None = None,
    ) -> "ManifestHistoricalDataSource":
        manifest_file = Path(manifest_path)
        inferred_root = manifest_file.parents[4]
        resolved_store = store or PyArrowParquetDatasetStore(inferred_root)
        manifest = resolved_store.read_manifest(manifest_file)
        return cls(
            manifest_path=manifest_file,
            store=resolved_store,
            manifest=manifest,
            storage_root=inferred_root,
        )

    def load_bars(self, dataset_spec: DatasetSpec) -> Mapping[str, Sequence[Bar]]:
        self._ensure_spec_is_compatible(dataset_spec)
        result = {}
        for symbol in dataset_spec.symbols:
            dataset_file = self.manifest.files[symbol]
            path = self.storage_root / dataset_file.relative_path
            bars = self.store.read_bars(path)
            result[symbol] = [
                bar
                for bar in bars
                if dataset_spec.start <= bar.timestamp < dataset_spec.end
            ]
        return result

    def _ensure_spec_is_compatible(self, dataset_spec: DatasetSpec) -> None:
        if dataset_spec.exchange != self.manifest.exchange:
            raise ValueError("dataset exchange mismatch")
        if dataset_spec.market != self.manifest.market:
            raise ValueError("dataset market mismatch")
        if dataset_spec.timeframe != self.manifest.timeframe:
            raise ValueError("dataset timeframe mismatch")
        if not set(dataset_spec.symbols).issubset(set(self.manifest.symbols)):
            raise ValueError("dataset symbol subset mismatch")
        if dataset_spec.start < self.manifest.start or dataset_spec.end > self.manifest.end:
            raise ValueError("requested range is outside the manifest window")
