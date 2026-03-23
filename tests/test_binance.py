from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from autoresearch_trade_bot.binance import BinanceUSDMHistoricalClient
from autoresearch_trade_bot.config import DataConfig
from autoresearch_trade_bot.datasets import DatasetSpec


class BinanceHistoricalClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = DatasetSpec(
            exchange="binance",
            market="usdm_futures",
            timeframe="5m",
            start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            end=datetime(2026, 3, 2, tzinfo=timezone.utc),
            symbols=("BTCUSDT",),
        )

    def test_fetch_symbol_history_can_skip_open_interest(self) -> None:
        client = BinanceUSDMHistoricalClient(
            data_config=DataConfig(
                exchange="binance",
                market="usdm_futures",
                storage_root="data",
                include_open_interest=False,
            )
        )

        with patch.object(client, "fetch_klines", return_value=[]) as mocked_klines:
            with patch.object(client, "fetch_funding_rates", return_value=[]) as mocked_funding:
                with patch.object(client, "fetch_open_interest") as mocked_open_interest:
                    history = client.fetch_symbol_history(self.spec, "BTCUSDT")

        self.assertEqual(history.open_interest_stats, [])
        self.assertEqual(mocked_klines.call_count, 1)
        self.assertEqual(mocked_funding.call_count, 1)
        mocked_open_interest.assert_not_called()

    def test_fetch_symbol_history_includes_open_interest_when_enabled(self) -> None:
        client = BinanceUSDMHistoricalClient(
            data_config=DataConfig(
                exchange="binance",
                market="usdm_futures",
                storage_root="data",
                include_open_interest=True,
            )
        )
        open_interest_payload = [{"timestamp": "1710000000000", "sumOpenInterest": "123"}]

        with patch.object(client, "fetch_klines", return_value=[]):
            with patch.object(client, "fetch_funding_rates", return_value=[]):
                with patch.object(
                    client,
                    "fetch_open_interest",
                    return_value=open_interest_payload,
                ) as mocked_open_interest:
                    history = client.fetch_symbol_history(self.spec, "BTCUSDT")

        self.assertEqual(history.open_interest_stats, open_interest_payload)
        mocked_open_interest.assert_called_once_with(
            "BTCUSDT",
            self.spec.timeframe,
            self.spec.start,
            self.spec.end,
        )


if __name__ == "__main__":
    unittest.main()
