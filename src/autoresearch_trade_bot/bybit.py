from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen

from .config import DataConfig
from .datasets import DatasetSpec, RawSymbolHistory, timeframe_to_timedelta
from .models import Bar


BYBIT_LINEAR_INTERVAL_MAP = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "12h": "720",
    "1d": "D",
}

BYBIT_OPEN_INTEREST_INTERVAL_MAP = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


@dataclass
class BybitLinearHistoricalClient:
    data_config: DataConfig
    base_url: str = "https://api.bytick.com"

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        interval = self._interval_for(spec.timeframe)
        return RawSymbolHistory(
            symbol=symbol,
            klines=self.fetch_klines(symbol, interval, spec.start, spec.end),
            funding_rates=self.fetch_funding_rates(symbol, spec.start, spec.end),
            open_interest_stats=self.fetch_open_interest(symbol, spec.timeframe, spec.start, spec.end),
            mark_price_klines=self.fetch_mark_price_klines(symbol, interval, spec.start, spec.end),
            index_price_klines=self.fetch_index_price_klines(symbol, interval, spec.start, spec.end),
        )

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[list]:
        return self._paginate_time_series(
            path="/v5/market/kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(self.data_config.max_batch_size, 1000),
            },
            start=start,
            end=end,
            extract_rows=lambda payload: payload["result"]["list"],
            row_timestamp=lambda row: int(row[0]),
        )

    def fetch_mark_price_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[list]:
        return self._paginate_time_series(
            path="/v5/market/mark-price-kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(self.data_config.max_batch_size, 1000),
            },
            start=start,
            end=end,
            extract_rows=lambda payload: payload["result"]["list"],
            row_timestamp=lambda row: int(row[0]),
        )

    def fetch_index_price_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[list]:
        return self._paginate_time_series(
            path="/v5/market/index-price-kline",
            params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(self.data_config.max_batch_size, 1000),
            },
            start=start,
            end=end,
            extract_rows=lambda payload: payload["result"]["list"],
            row_timestamp=lambda row: int(row[0]),
        )

    def fetch_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        return self._paginate_time_series(
            path="/v5/market/funding/history",
            params={
                "category": "linear",
                "symbol": symbol,
                "limit": min(self.data_config.max_batch_size, 200),
            },
            start=start,
            end=end,
            extract_rows=lambda payload: payload["result"]["list"],
            row_timestamp=lambda row: int(row["fundingRateTimestamp"]),
        )

    def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        interval = BYBIT_OPEN_INTEREST_INTERVAL_MAP.get(timeframe)
        if interval is None:
            return []

        start_ms = self._to_millis(start)
        end_ms = self._to_millis(end)
        cursor: str | None = None
        rows: list[dict] = []

        while True:
            payload = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(self.data_config.max_batch_size, 200),
            }
            if cursor:
                payload["cursor"] = cursor
            response = self._request("/v5/market/open-interest", payload)
            batch = response["result"]["list"]
            if not batch:
                break
            rows.extend(batch)
            cursor = response["result"].get("nextPageCursor") or None
            if not cursor:
                break

        rows.sort(key=lambda item: int(item["timestamp"]))
        return rows

    def _paginate_time_series(
        self,
        path: str,
        params: dict[str, object],
        start: datetime,
        end: datetime,
        extract_rows: Callable[[dict], list],
        row_timestamp: Callable[[list | dict], int],
    ) -> list:
        start_ms = self._to_millis(start)
        end_ms = self._to_millis(end)
        next_end_ms = end_ms
        rows: list = []

        while next_end_ms > start_ms:
            payload = dict(params)
            payload["start"] = start_ms
            payload["end"] = next_end_ms
            response = self._request(path, payload)
            batch = extract_rows(response)
            if not batch:
                break
            rows.extend(batch)
            earliest = min(row_timestamp(item) for item in batch)
            if earliest <= start_ms or len(batch) < int(payload["limit"]):
                break
            next_end_ms = earliest - 1

        rows.sort(key=row_timestamp)
        return rows

    def _request(self, path: str, params: dict[str, object]) -> dict:
        query = urlencode(params)
        url = f"{self.base_url}{path}?{query}"
        with urlopen(url, timeout=self.data_config.request_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if int(payload.get("retCode", 0)) != 0:
            raise RuntimeError(
                "Bybit request failed for %s: %s"
                % (path, payload.get("retMsg", "unknown_error"))
            )
        return payload

    @staticmethod
    def _interval_for(timeframe: str) -> str:
        interval = BYBIT_LINEAR_INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError("unsupported Bybit timeframe: %s" % timeframe)
        return interval

    @staticmethod
    def _to_millis(value: datetime) -> int:
        return int(value.astimezone(timezone.utc).timestamp() * 1000)


class BybitBarNormalizer:
    def __init__(self, exchange: str = "bybit") -> None:
        self.exchange = exchange

    def normalize(self, spec: DatasetSpec, raw: RawSymbolHistory) -> list[Bar]:
        funding_by_timestamp = {
            self._from_millis(int(item["fundingRateTimestamp"])): float(item["fundingRate"])
            for item in raw.funding_rates
        }
        open_interest_by_timestamp = {
            self._from_millis(int(item["timestamp"])): float(item["openInterest"])
            for item in raw.open_interest_stats
        }
        mark_price_by_timestamp = self._build_price_map(raw.mark_price_klines)
        index_price_by_timestamp = self._build_price_map(raw.index_price_klines)

        bars = []
        for item in raw.klines:
            timestamp = self._from_millis(int(item[0]))
            if timestamp < spec.start or timestamp >= spec.end:
                continue
            close_time = timestamp + spec.step - timedelta(milliseconds=1)
            bars.append(
                Bar(
                    exchange=self.exchange,
                    symbol=raw.symbol,
                    timeframe=spec.timeframe,
                    timestamp=timestamp,
                    close_time=close_time,
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    trade_count=0,
                    funding_rate=funding_by_timestamp.get(timestamp, 0.0),
                    open_interest=open_interest_by_timestamp.get(timestamp, 0.0),
                    mark_price=mark_price_by_timestamp.get(timestamp),
                    index_price=index_price_by_timestamp.get(timestamp),
                    is_closed=True,
                )
            )
        return bars

    @staticmethod
    def _build_price_map(rows: list[list]) -> dict[datetime, float]:
        mapping: dict[datetime, float] = {}
        for item in rows:
            mapping[BybitBarNormalizer._from_millis(int(item[0]))] = float(item[4])
        return mapping

    @staticmethod
    def _from_millis(value: int) -> datetime:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
