#!/usr/bin/env python3
"""
Download historical data from Hyperliquid.

Usage:
    python -m scripts.download_data [--coins BTC,ETH] [--start-date 2023-01-01] [--end-date 2025-01-01]

By default, downloads top 25% of coins by volume for the last 2 years.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_fetcher import fetch_data
from src.data.data_processor import process_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from Hyperliquid"
    )
    parser.add_argument(
        "--coins",
        type=str,
        default=None,
        help="Comma-separated list of coins (default: top 25%% by volume)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="8h",
        help="Candle interval (default: 8h)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw data (default: data/raw)",
    )
    parser.add_argument(
        "--skip-process",
        action="store_true",
        help="Skip processing step after download",
    )

    args = parser.parse_args()

    # Parse coins list
    coins = None
    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(",")]

    logger.info("Starting data download...")
    logger.info(f"Coins: {coins or 'top 25% by volume'}")
    logger.info(f"Start date: {args.start_date or 'default (2 years ago)'}")
    logger.info(f"End date: {args.end_date or 'default (today)'}")
    logger.info(f"Interval: {args.interval}")

    # Fetch data
    try:
        data = fetch_data(
            coins=coins,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            data_dir=args.data_dir,
        )
        logger.info(f"Downloaded data for {len(data)} coins")

        # Print summary
        for coin, coin_data in data.items():
            candles = coin_data["candles"]
            funding = coin_data["funding"]
            candle_info = f"{len(candles)} candles" if not candles.empty else "no candles"
            funding_info = f"{len(funding)} funding records" if not funding.empty else "no funding"
            logger.info(f"  {coin}: {candle_info}, {funding_info}")

    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

    # Process data
    if not args.skip_process:
        logger.info("Processing data...")
        try:
            processed = process_data(
                raw_dir=args.data_dir,
                processed_dir="data/processed",
            )
            logger.info(f"Processed data shape: prices={processed['prices'].shape}")
            logger.info(f"Date range: {processed['prices'].index.min()} to {processed['prices'].index.max()}")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    logger.info("Done!")


if __name__ == "__main__":
    main()
