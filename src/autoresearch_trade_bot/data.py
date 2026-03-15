from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .binance import BinanceBarNormalizer, BinanceUSDMHistoricalClient
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


class DataValidationError(ValueError):
    def __init__(self, issues) -> None:
        super().__init__("dataset validation failed")
        self.issues = issues


@dataclass
class HistoricalDatasetMaterializer:
    data_config: DataConfig
    client: HistoricalMarketClient
    store: DatasetStore
    validator: DatasetValidator
    normalizer: BinanceBarNormalizer

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

    def materialize(self, dataset_spec: DatasetSpec) -> ValidatedDataset:
        bars_by_symbol = {}
        files = {}

        for symbol in dataset_spec.symbols:
            raw = self.client.fetch_symbol_history(dataset_spec, symbol)
            bars = self.normalizer.normalize(dataset_spec, raw)
            bars_by_symbol[symbol] = bars

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
            manifest_path.write_text(str(manifest.to_dict()), encoding="utf-8")

        return ValidatedDataset(
            spec=dataset_spec,
            manifest=manifest,
            manifest_path=manifest_path,
        )


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
