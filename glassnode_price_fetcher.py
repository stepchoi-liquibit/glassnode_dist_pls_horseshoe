#!/usr/bin/env python3
"""
Glassnode Price USD Fetcher with Local Caching
===============================================

Retrieves historical price_usd_close data from the Glassnode API at a specific
time of day (default: 8 AM UTC), using a separate Parquet cache from
glassnode_fetcher.py.

Internally fetches hourly (1h) data and filters to the requested hour,
providing daily price snapshots at a consistent time of day.

Usage (import):
    from glassnode_price_fetcher import fetch_glassnode_price

    df = fetch_glassnode_price(
        start_date="2024-01-01",
        end_date="2024-12-31",
        asset="BTC",
    )

    # Merge with fetch_glassnode_data output:
    from glassnode_fetcher import fetch_glassnode_data
    metrics_df = fetch_glassnode_data(asset="BTC", start_date="2024-01-01", end_date="2024-12-31")
    merged = metrics_df.merge(df, on=["asset", "date"], how="outer")

Usage (CLI):
    python glassnode_price_fetcher.py --asset BTC --start 2024-01-01 --end 2024-12-31
    python glassnode_price_fetcher.py --asset ETH --start 2024-01-01 --end 2024-12-31 --time-utc 12

Environment:
    export GLASSNODE_API_KEY="your-api-key"
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.glassnode.com"
ENDPOINT = "/v1/metrics/market/price_usd_close"

DEFAULT_PRICE_CACHE_PATH = Path("./glassnode_cache/price_usd_cache.parquet")

API_KEY = "38sAezXyubtsWOV9lJcXRD0WzNB"


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_unix(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' to UTC-midnight Unix timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _from_unix(ts: int) -> str:
    """Convert Unix timestamp to 'YYYY-MM-DD'."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# ──────────────────────────────────────────────────────────────────────────────
# Cache Layer  (separate Parquet from glassnode_fetcher.py)
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_COLUMNS = ["asset", "timestamp", "date", "hour", "price_usd"]


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=_CACHE_COLUMNS).astype(
        {"timestamp": "int64", "hour": "int64", "price_usd": "float32"}
    )


def _load_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if all(col in df.columns for col in _CACHE_COLUMNS):
                return df
            logger.warning("Price cache has unexpected schema -- rebuilding.")
        except Exception as exc:
            logger.warning(f"Could not read price cache ({exc}) -- starting fresh.")
    return _empty_cache()


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["asset", "timestamp"], keep="last")
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False, engine="pyarrow")


def _query_cache(
    cache: pd.DataFrame,
    asset: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    if cache.empty:
        return cache
    mask = (
        (cache["asset"] == asset)
        & (cache["timestamp"] >= start_ts)
        & (cache["timestamp"] <= end_ts)
    )
    return cache.loc[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Gap Detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_gaps(
    cached_timestamps: set[int],
    start_ts: int,
    end_ts: int,
    step: int = 3_600,
) -> list[tuple[int, int]]:
    """Return (gap_start, gap_end) tuples for missing hourly segments."""
    all_expected = set(range(start_ts, end_ts + 1, step))
    missing = sorted(all_expected - cached_timestamps)

    if not missing:
        return []

    gaps: list[tuple[int, int]] = []
    gap_start = prev = missing[0]
    for ts in missing[1:]:
        if ts - prev > step:
            gaps.append((gap_start, prev))
            gap_start = ts
        prev = ts
    gaps.append((gap_start, prev))
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
# Glassnode REST API
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_price_api(
    asset: str,
    start_ts: int,
    end_ts: int,
    resolution: str,
    api_key: str,
) -> list[dict]:
    """Single Glassnode API call for price_usd_close. Costs 1 credit."""
    url = f"{BASE_URL}{ENDPOINT}"
    params = {
        "a":       asset,
        "s":       str(start_ts),
        "u":       str(end_ts),
        "i":       resolution,
        "api_key": api_key,
    }

    logger.info(
        f"  API   price_usd_close  "
        f"{_from_unix(start_ts)} -> {_from_unix(end_ts)}  resolution={resolution}"
    )

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type ({type(data).__name__})")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main Public Function
# ──────────────────────────────────────────────────────────────────────────────

def fetch_glassnode_price(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    resolution: str = "24h",
    time_utc: int = 8,
    cache_path: Optional[str | Path] = None,
    api_key: Optional[str] = API_KEY,
) -> pd.DataFrame:
    """
    Fetch price_usd_close from Glassnode at a specific time of day.

    Internally fetches 1h-resolution data and filters to the specified UTC
    hour, providing price snapshots at a consistent daily time (default
    8 AM UTC).  Uses a separate Parquet cache from glassnode_fetcher.py.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format (inclusive).
    end_date : str
        End date in 'YYYY-MM-DD' format (inclusive).
    asset : str
        Cryptocurrency symbol (default: 'BTC').
    resolution : str
        Output resolution. '24h' returns one price per day at ``time_utc``
        hour; '1h' returns all hourly prices (default: '24h').
    time_utc : int
        Hour of day in UTC (0-23) to sample the price at.  Only used when
        resolution='24h'.  Default: 8 (8 AM UTC).
    cache_path : str | Path | None
        Override the default Parquet cache location.
    api_key : str | None
        Glassnode API key.  Falls back to GLASSNODE_API_KEY env var.

    Returns
    -------
    pd.DataFrame
        When resolution='24h': columns ``['asset', 'date', 'price_usd_<HH>utc']``
        When resolution='1h':  columns ``['asset', 'date', 'hour', 'price_usd']``
        Mergeable with ``fetch_glassnode_data()`` output on ``['asset', 'date']``.
    """
    if not 0 <= time_utc <= 23:
        raise ValueError(f"time_utc must be 0-23, got {time_utc}")

    _api_key = api_key or os.environ.get("GLASSNODE_API_KEY", "")
    if not _api_key:
        raise ValueError(
            "No API key provided.  Pass api_key= or set GLASSNODE_API_KEY env var."
        )

    _cache_path = Path(cache_path) if cache_path else DEFAULT_PRICE_CACHE_PATH

    # Always fetch at 1h from the API to enable time-of-day filtering
    api_resolution = "1h"
    step = 3_600  # 1 hour

    start_ts = _to_unix(start_date)
    end_ts = _to_unix(end_date) + 86_400 - step  # include all hours of last day

    # ── Load cache ───────────────────────────────────────────────────────
    cache = _load_cache(_cache_path)

    cached = _query_cache(cache, asset, start_ts, end_ts)
    cached_ts = set(cached["timestamp"].tolist()) if not cached.empty else set()

    gaps = _find_gaps(cached_ts, start_ts, end_ts, step)

    logger.info(
        f"Fetching price_usd_close for {asset}  "
        f"[{start_date} -> {end_date}]  time_utc={time_utc:02d}:00"
    )

    new_rows: list[dict] = []
    api_calls_made = 0

    if not gaps:
        logger.info("  CACHE  fully cached  (0 credits)")
    else:
        # Cache has gaps -- fetch the full requested range (1 credit).
        # Fresh data will overwrite overlapping cached entries on save.
        total_expected = len(range(start_ts, end_ts + 1, step))
        logger.info(
            f"  GAPS   {len(cached_ts)}/{total_expected} cached, "
            f"{total_expected - len(cached_ts)} missing -- fetching full range"
        )

        try:
            raw = _fetch_price_api(
                asset, start_ts, end_ts + step, api_resolution, _api_key
            )
            for point in raw:
                ts = point["t"]
                val = point["v"]
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                new_rows.append({
                    "asset":     asset,
                    "timestamp": ts,
                    "date":      dt.strftime("%Y-%m-%d"),
                    "hour":      dt.hour,
                    "price_usd": float(val) if val is not None else None,
                })
            api_calls_made += 1
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.error(f"  ERROR price_usd_close: HTTP {status} -- {exc}")
        except Exception as exc:
            logger.error(f"  ERROR price_usd_close: {exc}")

    # ── Persist new data ─────────────────────────────────────────────────
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        _save_cache(cache, _cache_path)
        logger.info(f"  SAVED +{len(new_rows)} rows -> {_cache_path}")

    # ── Build result ─────────────────────────────────────────────────────
    all_data = _query_cache(cache, asset, start_ts, end_ts)

    if all_data.empty:
        logger.warning("No price data available for the requested range.")
        col_name = f"price_usd_{time_utc:02d}utc"
        return pd.DataFrame(columns=["asset", "date", col_name])

    if resolution == "24h":
        # Filter to the requested hour -> one row per day
        result = all_data[all_data["hour"] == time_utc].copy()
        col_name = f"price_usd_{time_utc:02d}utc"
        result = (
            result[["asset", "date", "price_usd"]]
            .rename(columns={"price_usd": col_name})
        )
    else:
        # Return all hourly data
        result = all_data[["asset", "date", "hour", "price_usd"]].copy()

    result = result.sort_values("date").reset_index(drop=True)

    logger.info(
        f"\n{'=' * 55}\n"
        f"  Asset:          {asset}\n"
        f"  Date range:     {start_date} -> {end_date}\n"
        f"  Time (UTC):     {time_utc:02d}:00\n"
        f"  API calls:      {api_calls_made} ({api_calls_made} credits)\n"
        f"  Result shape:   {result.shape[0]} rows x {result.shape[1]} cols\n"
        f"  Cache:          {_cache_path.resolve()}\n"
        f"{'=' * 55}"
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Daily Close Price (true 24h bar)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CLOSE_CACHE_PATH = Path("./glassnode_cache/price_daily_close_cache.parquet")

_CLOSE_CACHE_COLUMNS = ["asset", "date", "price_usd_close"]


def _load_close_cache(cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if all(col in df.columns for col in _CLOSE_CACHE_COLUMNS):
                return df
            logger.warning("Daily-close cache has unexpected schema -- rebuilding.")
        except Exception as exc:
            logger.warning(f"Could not read daily-close cache ({exc}) -- starting fresh.")
    return pd.DataFrame(columns=_CLOSE_CACHE_COLUMNS).astype(
        {"price_usd_close": "float64"}
    )


def _save_close_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["asset", "date"], keep="last")
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False, engine="pyarrow")


def fetch_close_price(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    cache_path: Optional[str | Path] = None,
    api_key: Optional[str] = API_KEY,
) -> pd.DataFrame:
    """
    Fetch true daily close prices from Glassnode using 24h resolution.

    Unlike ``fetch_glassnode_price`` (which fetches 1h data and picks a
    specific UTC hour), this function queries the Glassnode API at native
    ``24h`` resolution.  Each data point represents the **end-of-day** price
    for the UTC calendar day (i.e. the price at ~23:59:59 UTC), which is
    the standard daily close used in financial analysis.

    Parameters
    ----------
    start_date : str
        Start date 'YYYY-MM-DD' (inclusive).
    end_date : str
        End date 'YYYY-MM-DD' (inclusive).
    asset : str
        Cryptocurrency symbol (default: 'BTC').
    cache_path : str | Path | None
        Override cache location.  Default: ``./glassnode_cache/price_daily_close_cache.parquet``
    api_key : str | None
        Glassnode API key.

    Returns
    -------
    pd.DataFrame
        Columns: ``['asset', 'date', 'price_usd_close']``
        One row per calendar day, sorted by date.
    """
    _api_key = api_key or os.environ.get("GLASSNODE_API_KEY", "")
    if not _api_key:
        raise ValueError(
            "No API key provided.  Pass api_key= or set GLASSNODE_API_KEY env var."
        )

    _cache_path = Path(cache_path) if cache_path else DEFAULT_CLOSE_CACHE_PATH

    start_ts = _to_unix(start_date)
    end_ts = _to_unix(end_date) + 86_400  # include end date

    # ── Load cache ────────────────────────────────────────────────────────
    cache = _load_close_cache(_cache_path)

    cached = cache[
        (cache["asset"] == asset)
        & (cache["date"] >= start_date)
        & (cache["date"] <= end_date)
    ] if not cache.empty else cache

    # Check for missing dates
    all_dates = set(
        pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d")
    )
    cached_dates = set(cached["date"].tolist()) if not cached.empty else set()
    missing_dates = all_dates - cached_dates

    logger.info(
        f"Fetching daily-close prices for {asset}  "
        f"[{start_date} -> {end_date}]  (24h resolution)"
    )

    new_rows: list[dict] = []
    api_calls_made = 0

    if not missing_dates:
        logger.info("  CACHE  fully cached  (0 credits)")
    else:
        logger.info(
            f"  GAPS   {len(cached_dates)}/{len(all_dates)} cached, "
            f"{len(missing_dates)} missing -- fetching full range"
        )

        try:
            raw = _fetch_price_api(
                asset, start_ts, end_ts, "24h", _api_key
            )
            for point in raw:
                ts = point["t"]
                val = point["v"]
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                new_rows.append({
                    "asset": asset,
                    "date": dt.strftime("%Y-%m-%d"),
                    "price_usd_close": float(val) if val is not None else None,
                })
            api_calls_made += 1
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.error(f"  ERROR daily-close price: HTTP {status} -- {exc}")
        except Exception as exc:
            logger.error(f"  ERROR daily-close price: {exc}")

    # ── Persist new data ──────────────────────────────────────────────────
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        cache = pd.concat([cache, new_df], ignore_index=True)
        _save_close_cache(cache, _cache_path)
        logger.info(f"  SAVED +{len(new_rows)} rows -> {_cache_path}")

    # ── Build result ──────────────────────────────────────────────────────
    result = cache[
        (cache["asset"] == asset)
        & (cache["date"] >= start_date)
        & (cache["date"] <= end_date)
    ].copy() if not cache.empty else pd.DataFrame(columns=_CLOSE_CACHE_COLUMNS)

    result = result[["asset", "date", "price_usd_close"]].copy()
    result = result.sort_values("date").reset_index(drop=True)

    logger.info(
        f"\n{'=' * 55}\n"
        f"  Asset:          {asset}\n"
        f"  Date range:     {start_date} -> {end_date}\n"
        f"  Resolution:     24h (daily close)\n"
        f"  API calls:      {api_calls_made} ({api_calls_made} credits)\n"
        f"  Result shape:   {result.shape[0]} rows x {result.shape[1]} cols\n"
        f"  Cache:          {_cache_path.resolve()}\n"
        f"{'=' * 55}"
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Fetch Glassnode price_usd_close at a specific UTC hour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python glassnode_price_fetcher.py --asset BTC --start 2024-01-01 --end 2024-06-30\n"
            "  python glassnode_price_fetcher.py --asset ETH --start 2024-01-01 --end 2024-12-31 "
            "--time-utc 12\n"
        ),
    )
    parser.add_argument("--asset",    default="BTC",        help="Asset symbol (default: BTC)")
    parser.add_argument("--start",    default="2026-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      default="2026-03-05", help="End date YYYY-MM-DD")
    parser.add_argument("--time-utc", type=int, default=8,  help="UTC hour 0-23 (default: 8)")
    parser.add_argument("--resolution", default="24h",      help="Output resolution: 24h or 1h")
    parser.add_argument("--cache",    default=None,         help="Cache file path override")
    parser.add_argument("--api-key",  default=None,         help="Glassnode API key")

    args = parser.parse_args()

    df = fetch_glassnode_price(
        start_date=args.start,
        end_date=args.end,
        asset=args.asset,
        resolution=args.resolution,
        time_utc=args.time_utc,
        cache_path=args.cache,
        api_key=args.api_key,
    )

    print(f"\n{df.to_string(index=False, max_rows=30)}\n")
