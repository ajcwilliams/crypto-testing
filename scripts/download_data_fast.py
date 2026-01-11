#!/usr/bin/env python3
"""
Fast parallel data download using async.

~5-10x faster than sequential downloads.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.async_fetcher import async_fetch_data
from src.data.data_processor import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default coins (top by volume on Hyperliquid)
DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "MATIC",
    "UNI", "ATOM", "LTC", "FIL", "APT", "ARB", "OP", "INJ", "SUI", "SEI",
]


async def main_async(args):
    coins = args.coins.split(",") if args.coins else DEFAULT_COINS
    logger.info(f"Fetching {len(coins)} coins: {coins}")

    await async_fetch_data(
        coins=coins,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
    )

    if not args.skip_process:
        logger.info("Processing data...")
        process_data(raw_dir=args.data_dir, processed_dir="data/processed")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Fast parallel data download")
    parser.add_argument("--coins", type=str, default=None, help="Comma-separated coins (default: top 20)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="8h", help="Candle interval")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--skip-process", action="store_true", help="Skip processing step")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
