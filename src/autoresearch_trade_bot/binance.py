from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urlencode
from urllib.request import urlopen

from .config import DataConfig
from .datasets import DatasetSpec, RawSymbolHistory
from .models import Bar


OPEN_INTEREST_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}


@dataclass
class BinanceUSDMHistoricalClient:
    data_config: DataConfig
    base_url: str = "https://fapi.binance.com"

    def fetch_symbol_history(self, spec: DatasetSpec, symbol: str) -> RawSymbolHistory:
        return RawSymbolHistory(
            symbol=symbol,
            klines=self.fetch_klines(symbol, spec.timeframe, spec.start, spec.end),
            funding_rates=self.fetch_funding_rates(symbol, spec.start, spec.end),
            open_interest_stats=self.fetch_open_interest(symbol, spec.timeframe, spec.start, spec.end),
        )

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[list]:
        return self._paginate(
            path="/fapi/v1/klines",
            params={"symbol": symbol, "interval": timeframe, "limit": min(self.data_config.max_batch_size, 1500)},
            start_key="startTime",
            end_key="endTime",
            cursor_key=lambda row: int(row[0]) + 1,
            start=start,
            end=end,
        )

    def fetch_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        return self._paginate(
            path="/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": min(self.data_config.max_batch_size, 1000)},
            start_key="startTime",
            end_key="endTime",
            cursor_key=lambda row: int(row["fundingTime"]) + 1,
            start=start,
            end=end,
        )

    def fetch_open_interest(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        if timeframe not in OPEN_INTEREST_PERIODS:
            raise ValueError("unsupported timeframe for open interest stats: %s" % timeframe)
        return self._paginate(
            path="/futures/data/openInterestHist",
            params={
                "symbol": symbol,
                "period": timeframe,
                "limit": min(self.data_config.max_batch_size, 500),
            },
            start_key="startTime",
            end_key="endTime",
            cursor_key=lambda row: int(row["timestamp"]) + 1,
            start=start,
            end=end,
        )

    def _paginate(
        self,
        path: str,
        params: dict,
        start_key: str,
        end_key: str,
        cursor_key,
        start: datetime,
        end: datetime,
    ) -> list:
        start_ms = int(start.astimezone(timezone.utc).timestamp() * 1000)
        end_ms = int(end.astimezone(timezone.utc).timestamp() * 1000)
        cursor = start_ms
        rows = []
        while cursor < end_ms:
            payload = dict(params)
            payload[start_key] = cursor
            payload[end_key] = end_ms
            batch = self._request(path, payload)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = cursor_key(batch[-1])
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < params["limit"]:
                break
        return rows

    def _request(self, path: str, params: dict):
        query = urlencode(params)
        url = f"{self.base_url}{path}?{query}"
        with urlopen(url, timeout=self.data_config.request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class BinanceBarNormalizer:
    """Normalizes Binance futures payloads into UTC-aware bars."""

    def __init__(self, exchange: str = "binance") -> None:
        self.exchange = exchange

    def normalize(self, spec: DatasetSpec, raw: RawSymbolHistory) -> list[Bar]:
        funding_by_timestamp = {
            self._from_millis(int(item["fundingTime"])): float(item["fundingRate"])
            for item in raw.funding_rates
        }
        open_interest_by_timestamp = {
            self._from_millis(int(item["timestamp"])): float(item["sumOpenInterest"])
            for item in raw.open_interest_stats
        }

        bars = []
        for item in raw.klines:
            timestamp = self._from_millis(int(item[0]))
            bars.append(
                Bar(
                    exchange=self.exchange,
                    symbol=raw.symbol,
                    timeframe=spec.timeframe,
                    timestamp=timestamp,
                    close_time=self._from_millis(int(item[6])),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    trade_count=int(item[8]),
                    funding_rate=funding_by_timestamp.get(timestamp, 0.0),
                    open_interest=open_interest_by_timestamp.get(timestamp, 0.0),
                    is_closed=True,
                )
            )
        return bars

    @staticmethod
    def _from_millis(value: int) -> datetime:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
