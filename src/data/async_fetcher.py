"""
Async data fetcher for parallel downloads from Hyperliquid.

Uses aiohttp for concurrent API calls - significantly faster for multiple coins.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://api.hyperliquid.xyz/info"
MS_PER_HOUR = 3_600_000
MS_PER_DAY = 86_400_000
CANDLE_LIMIT = 5000
FUNDING_LIMIT_HOURS = 500

# Concurrency limits - Hyperliquid has strict rate limits
MAX_CONCURRENT_REQUESTS = 5  # Only 5 concurrent requests
RATE_LIMIT_DELAY = 0.2  # 200ms between requests


class AsyncHyperliquidFetcher:
    """Async fetcher for parallel data downloads."""

    def __init__(self, data_dir: Path | str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def _request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        max_retries: int = 3,
    ) -> dict:
        """Make rate-limited async request with retry on 429."""
        async with self.semaphore:
            for attempt in range(max_retries):
                await asyncio.sleep(RATE_LIMIT_DELAY)
                try:
                    async with session.post(BASE_URL, json=payload) as resp:
                        if resp.status == 429:
                            wait = 2 ** attempt  # Exponential backoff
                            logger.debug(f"Rate limited, waiting {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return await resp.json()
                except aiohttp.ClientError as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            return {}

    async def fetch_candles(
        self,
        session: aiohttp.ClientSession,
        coin: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> list[dict]:
        """Fetch all candles for a coin with pagination."""
        all_candles = []
        current_start = start_time
        interval_ms = self._interval_to_ms(interval)
        chunk_duration = CANDLE_LIMIT * interval_ms

        while current_start < end_time:
            chunk_end = min(current_start + chunk_duration, end_time)
            try:
                data = await self._request(session, {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": chunk_end,
                    }
                })
                if not data:
                    break
                for c in data:
                    all_candles.append({
                        "open_time": c["t"],
                        "close_time": c["T"],
                        "coin": c["s"],
                        "open": float(c["o"]),
                        "close": float(c["c"]),
                        "high": float(c["h"]),
                        "low": float(c["l"]),
                        "volume": float(c["v"]),
                    })
                current_start = data[-1]["T"] + 1
            except Exception as e:
                logger.warning(f"Error fetching candles for {coin}: {e}")
                break

        return all_candles

    async def fetch_funding(
        self,
        session: aiohttp.ClientSession,
        coin: str,
        start_time: int,
        end_time: int,
    ) -> list[dict]:
        """Fetch all funding rates for a coin with pagination."""
        all_funding = []
        current_start = start_time
        chunk_duration = FUNDING_LIMIT_HOURS * MS_PER_HOUR

        while current_start < end_time:
            chunk_end = min(current_start + chunk_duration, end_time)
            try:
                data = await self._request(session, {
                    "type": "fundingHistory",
                    "coin": coin,
                    "startTime": current_start,
                    "endTime": chunk_end,
                })
                if not data:
                    current_start = chunk_end + 1
                    continue
                for f in data:
                    all_funding.append({
                        "coin": f["coin"],
                        "funding_rate": float(f["fundingRate"]),
                        "premium": float(f["premium"]),
                        "time": f["time"],
                    })
                current_start = data[-1]["time"] + 1
            except Exception as e:
                logger.warning(f"Error fetching funding for {coin}: {e}")
                current_start = chunk_end + 1

        return all_funding

    async def fetch_coin(
        self,
        session: aiohttp.ClientSession,
        coin: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> dict:
        """Fetch both candles and funding for a single coin."""
        candles_task = self.fetch_candles(session, coin, interval, start_time, end_time)
        funding_task = self.fetch_funding(session, coin, start_time, end_time)

        candles, funding = await asyncio.gather(candles_task, funding_task)

        return {"coin": coin, "candles": candles, "funding": funding}

    async def fetch_all_coins(
        self,
        coins: list[str],
        interval: str = "8h",
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> dict[str, dict]:
        """Fetch data for all coins concurrently."""
        if end_time is None:
            end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        if start_time is None:
            start_time = end_time - (730 * MS_PER_DAY)

        results = {}

        async with aiohttp.ClientSession() as session:
            # Create tasks for all coins
            tasks = [
                self.fetch_coin(session, coin, interval, start_time, end_time)
                for coin in coins
            ]

            # Run with progress bar
            for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Fetching coins"):
                result = await coro
                coin = result["coin"]

                # Convert to DataFrames and save
                candles_df = pd.DataFrame(result["candles"])
                funding_df = pd.DataFrame(result["funding"])

                if not candles_df.empty:
                    candles_df["timestamp"] = pd.to_datetime(candles_df["open_time"], unit="ms", utc=True)
                    candles_df = candles_df.set_index("timestamp").sort_index()

                if not funding_df.empty:
                    funding_df["timestamp"] = pd.to_datetime(funding_df["time"], unit="ms", utc=True)
                    funding_df = funding_df.set_index("timestamp").sort_index()

                # Save to disk
                coin_dir = self.data_dir / coin
                coin_dir.mkdir(exist_ok=True)
                if not candles_df.empty:
                    candles_df.to_parquet(coin_dir / "candles.parquet")
                if not funding_df.empty:
                    funding_df.to_parquet(coin_dir / "funding.parquet")

                results[coin] = {"candles": candles_df, "funding": funding_df}
                logger.info(f"{coin}: {len(candles_df)} candles, {len(funding_df)} funding")

        # Save metadata
        metadata = {
            "coins": coins,
            "interval": interval,
            "start_time": start_time,
            "end_time": end_time,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return results

    @staticmethod
    def _interval_to_ms(interval: str) -> int:
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == "m":
            return value * 60 * 1000
        elif unit == "h":
            return value * 60 * 60 * 1000
        elif unit == "d":
            return value * 24 * 60 * 60 * 1000
        raise ValueError(f"Unknown interval: {interval}")


async def async_fetch_data(
    coins: list[str],
    interval: str = "8h",
    start_date: str | None = None,
    end_date: str | None = None,
    data_dir: str = "data/raw",
) -> dict:
    """
    Async convenience function for fast parallel downloads.

    Example:
        import asyncio
        data = asyncio.run(async_fetch_data(["BTC", "ETH", "SOL"]))
    """
    start_time = None
    end_time = None

    if start_date:
        dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_time = int(dt.timestamp() * 1000)

    if end_date:
        dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_time = int(dt.timestamp() * 1000)

    fetcher = AsyncHyperliquidFetcher(data_dir=data_dir)
    return await fetcher.fetch_all_coins(coins, interval, start_time, end_time)
