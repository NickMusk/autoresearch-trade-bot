from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_trade_bot.autoresearch import prepare_campaign


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a frozen crypto autoresearch campaign.")
    parser.add_argument("--campaign-name", default="crypto-bybit-5m")
    parser.add_argument("--exchange", default="bybit")
    parser.add_argument("--market", default="linear")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument(
        "--symbols",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT",
        help="Comma-separated symbols",
    )
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--window-count", type=int, default=2)
    parser.add_argument("--end", required=True, help="Frozen campaign anchor end, ISO-8601")
    parser.add_argument("--storage-root", default="data")
    parser.add_argument("--campaign-path")
    parser.add_argument("--campaigns-root", default=".autoresearch/campaigns")
    parser.add_argument("--active-campaign-path", default=".autoresearch/active_campaign.txt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    campaign_path = prepare_campaign(
        campaign_name=args.campaign_name,
        exchange=args.exchange,
        market=args.market,
        timeframe=args.timeframe,
        symbols=tuple(
            symbol.strip().upper()
            for symbol in args.symbols.split(",")
            if symbol.strip()
        ),
        anchor_end=parse_datetime(args.end),
        window_days=args.window_days,
        window_count=args.window_count,
        storage_root=args.storage_root,
        campaign_path=args.campaign_path,
        campaigns_root=args.campaigns_root,
        active_pointer_path=args.active_campaign_path,
    )
    print(campaign_path)


if __name__ == "__main__":
    main()
