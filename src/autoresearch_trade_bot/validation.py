from __future__ import annotations

from datetime import datetime
from typing import Mapping, Sequence

from .datasets import DatasetSpec, ValidationIssue, ValidationReport
from .models import Bar


class DatasetValidator:
    """Validates normalized market bars before they are promoted into research datasets."""

    def validate(
        self,
        spec: DatasetSpec,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
    ) -> ValidationReport:
        issues = []
        for symbol, bars in bars_by_symbol.items():
            issues.extend(self._validate_symbol(spec, symbol, bars))
        issues.extend(self._validate_alignment(spec, bars_by_symbol))
        return ValidationReport(issues=issues)

    def _validate_symbol(
        self,
        spec: DatasetSpec,
        symbol: str,
        bars: Sequence[Bar],
    ) -> list[ValidationIssue]:
        issues = []
        if not bars:
            return [
                ValidationIssue(
                    severity="error",
                    symbol=symbol,
                    code="empty_series",
                    message="symbol series is empty",
                )
            ]

        expected_step = spec.step
        previous_timestamp: datetime | None = None
        for bar in bars:
            if previous_timestamp is not None:
                delta = bar.timestamp - previous_timestamp
                if delta.total_seconds() <= 0:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            symbol=symbol,
                            code="non_monotonic_timestamp",
                            message="timestamps must be strictly increasing",
                            timestamp=bar.timestamp,
                        )
                    )
                elif delta != expected_step:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            symbol=symbol,
                            code="timestamp_gap",
                            message="unexpected gap for timeframe %s" % spec.timeframe,
                            timestamp=bar.timestamp,
                        )
                    )
            previous_timestamp = bar.timestamp

            if min(bar.open, bar.high, bar.low, bar.close) <= 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="non_positive_price",
                        message="all price fields must be positive",
                        timestamp=bar.timestamp,
                    )
                )
            if bar.high < max(bar.open, bar.close, bar.low):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="invalid_high",
                        message="high must be the maximum price in the bar",
                        timestamp=bar.timestamp,
                    )
                )
            if bar.low > min(bar.open, bar.close, bar.high):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="invalid_low",
                        message="low must be the minimum price in the bar",
                        timestamp=bar.timestamp,
                    )
                )
            if bar.volume < 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="negative_volume",
                        message="volume must not be negative",
                        timestamp=bar.timestamp,
                    )
                )

        return issues

    def _validate_alignment(
        self,
        spec: DatasetSpec,
        bars_by_symbol: Mapping[str, Sequence[Bar]],
    ) -> list[ValidationIssue]:
        if not bars_by_symbol:
            return [
                ValidationIssue(
                    severity="error",
                    symbol="*",
                    code="empty_dataset",
                    message="dataset contains no symbols",
                )
            ]

        expected = [bar.timestamp for bar in next(iter(bars_by_symbol.values()))]
        issues = []
        for symbol, bars in bars_by_symbol.items():
            timestamps = [bar.timestamp for bar in bars]
            if timestamps != expected:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="misaligned_series",
                        message="timestamps do not align across symbols",
                    )
                )
            if timestamps and timestamps[0] < spec.start:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="range_starts_too_early",
                        message="series starts before dataset start",
                        timestamp=timestamps[0],
                    )
                )
            if timestamps and timestamps[-1] >= spec.end:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        symbol=symbol,
                        code="range_ends_too_late",
                        message="series ends on or after dataset end",
                        timestamp=timestamps[-1],
                    )
                )
        return issues
