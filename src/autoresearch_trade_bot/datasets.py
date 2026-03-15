from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    unit = timeframe[-1]
    magnitude = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    if unit == "d":
        return timedelta(days=magnitude)
    raise ValueError("unsupported timeframe: %s" % timeframe)


@dataclass(frozen=True)
class DatasetSpec:
    exchange: str
    market: str
    timeframe: str
    start: datetime
    end: datetime
    symbols: Tuple[str, ...]

    def __post_init__(self) -> None:
        start = ensure_utc(self.start)
        end = ensure_utc(self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        if start >= end:
            raise ValueError("dataset start must be before end")
        if not self.symbols:
            raise ValueError("dataset symbols must not be empty")

    @property
    def dataset_id(self) -> str:
        symbol_part = "-".join(sorted(self.symbols))
        start_part = self.start.strftime("%Y%m%dT%H%M%SZ")
        end_part = self.end.strftime("%Y%m%dT%H%M%SZ")
        return (
            f"{self.exchange}-{self.market}-{self.timeframe}-{start_part}-{end_part}-{symbol_part}"
        )

    @property
    def step(self) -> timedelta:
        return timeframe_to_timedelta(self.timeframe)


@dataclass(frozen=True)
class DatasetFile:
    symbol: str
    relative_path: str
    row_count: int
    first_timestamp: datetime
    last_timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "relative_path": self.relative_path,
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetFile":
        return cls(
            symbol=payload["symbol"],
            relative_path=payload["relative_path"],
            row_count=int(payload["row_count"]),
            first_timestamp=ensure_utc(datetime.fromisoformat(payload["first_timestamp"])),
            last_timestamp=ensure_utc(datetime.fromisoformat(payload["last_timestamp"])),
        )


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    symbol: str
    code: str
    message: str
    timestamp: datetime | None = None

    def to_dict(self) -> dict:
        payload = {
            "severity": self.severity,
            "symbol": self.symbol,
            "code": self.code,
            "message": self.message,
        }
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ValidationIssue":
        timestamp = payload.get("timestamp")
        return cls(
            severity=payload["severity"],
            symbol=payload["symbol"],
            code=payload["code"],
            message=payload["message"],
            timestamp=ensure_utc(datetime.fromisoformat(timestamp)) if timestamp else None,
        )


@dataclass(frozen=True)
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    exchange: str
    market: str
    timeframe: str
    start: datetime
    end: datetime
    generated_at: datetime
    symbols: Tuple[str, ...]
    files: Dict[str, DatasetFile]
    validation_issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "exchange": self.exchange,
            "market": self.market,
            "timeframe": self.timeframe,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "symbols": list(self.symbols),
            "files": {symbol: item.to_dict() for symbol, item in self.files.items()},
            "validation_issues": [issue.to_dict() for issue in self.validation_issues],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DatasetManifest":
        return cls(
            dataset_id=payload["dataset_id"],
            exchange=payload["exchange"],
            market=payload["market"],
            timeframe=payload["timeframe"],
            start=ensure_utc(datetime.fromisoformat(payload["start"])),
            end=ensure_utc(datetime.fromisoformat(payload["end"])),
            generated_at=ensure_utc(datetime.fromisoformat(payload["generated_at"])),
            symbols=tuple(payload["symbols"]),
            files={
                symbol: DatasetFile.from_dict(item)
                for symbol, item in payload["files"].items()
            },
            validation_issues=[
                ValidationIssue.from_dict(item)
                for item in payload.get("validation_issues", [])
            ],
        )


@dataclass(frozen=True)
class ValidatedDataset:
    spec: DatasetSpec
    manifest: DatasetManifest
    manifest_path: Path


@dataclass(frozen=True)
class RawSymbolHistory:
    symbol: str
    klines: List[list]
    funding_rates: List[dict]
    open_interest_stats: List[dict]
