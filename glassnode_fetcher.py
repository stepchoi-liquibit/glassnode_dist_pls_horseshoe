#!/usr/bin/env python3
"""
Glassnode Historical Data Fetcher with Local Caching
=====================================================

Retrieves historical data from the Glassnode API for specified endpoints and
asset, using **per-endpoint Parquet files** as a persistent cache ("local
database") to minimize API credit usage AND memory consumption.

Credit model (https://docs.glassnode.com/basic-api/api-credits):
    - Each standard API call consumes 1 credit regardless of the date range.
    - Bulk endpoints cost 1 credit per asset per call.
    - Therefore the optimal strategy is to minimize the NUMBER of API calls
      while still covering only the missing date ranges.

Caching strategy:
    - Each (asset, endpoint) pair lives in its own lightweight Parquet file
      under ``<cache_dir>/<asset>/<sanitised_endpoint>.parquet``.
    - If the requested data is fully cached -> 0 credits, instant return.
    - If the cache has any gaps -> fetch the full requested date range in a
      single API call (1 credit).  Fresh data overwrites overlapping cached
      entries; non-overlapping cached entries are preserved.
    - On first run, if an old monolithic ``glassnode_data.parquet`` is found
      it is automatically split into per-endpoint files and the original is
      renamed to ``glassnode_data.parquet.migrated``.

Usage (import):
    from glassnode_fetcher import fetch_glassnode_data

    df = fetch_glassnode_data(
        asset="BTC",
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

Usage (CLI):
    python glassnode_fetcher.py --asset BTC --start 2024-01-01 --end 2024-12-31

Environment:
    export GLASSNODE_API_KEY="your-api-key"
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.glassnode.com"

# Now a *directory*; each endpoint gets its own file beneath it.
DEFAULT_CACHE_DIR = Path("./glassnode_cache")

# Legacy monolithic path (for auto-migration)
_LEGACY_CACHE_FILE = Path("./glassnode_cache/glassnode_data.parquet")

DEFAULT_RESOLUTION = "24h"

API_KEY = "38sAezXyubtsWOV9lJcXRD0WzNB"


# Friendly column names for the 5 standard endpoints
FRIENDLY_NAMES: dict[str, str] = {
    "price_usd_close":                       "price_usd",
    "realized_volatility_1_week":             "realized_vol_7d",
    "realized_volatility_1_month":            "realized_vol_30d",
    "options_atm_implied_volatility_1_week":  "implied_vol_atm_7d",
    "options_atm_implied_volatility_1_month": "implied_vol_atm_30d",
}

DEFAULT_ENDPOINTS: list[str] = [
    "/v1/metrics/market/price_usd_close",
    "/v1/metrics/market/realized_volatility_1_week",
    "/v1/metrics/market/realized_volatility_1_month",
    "/v1/metrics/derivatives/options_atm_implied_volatility_1_week",
    "/v1/metrics/derivatives/options_atm_implied_volatility_1_month",
]

RESOLUTION_TO_SECONDS: dict[str, int] = {
    "1h":     3_600,
    "24h":    86_400,
    "1w":     604_800,
    "1month": 2_592_000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
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


def _endpoint_short_name(endpoint: str) -> str:
    """Extract the last path component as a short metric name."""
    return endpoint.rstrip("/").split("/")[-1]


def _sanitise_filename(endpoint: str) -> str:
    """
    Turn an endpoint path like ``/v1/metrics/market/price_usd_close``
    into a safe filename stem: ``market__price_usd_close``.
    """
    # Strip leading /v1/metrics/ prefix
    cleaned = re.sub(r"^/v1/metrics/", "", endpoint.strip("/"))
    # Replace remaining slashes with double underscores
    cleaned = cleaned.replace("/", "__")
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", cleaned)
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Per-endpoint cache layer
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_COLUMNS = ["asset", "endpoint", "timestamp", "date", "value"]


def _endpoint_cache_path(cache_dir: Path, asset: str, endpoint: str) -> Path:
    """Return the Parquet path for a single (asset, endpoint) pair."""
    return cache_dir / asset / f"{_sanitise_filename(endpoint)}.parquet"


def _empty_cache() -> pd.DataFrame:
    """Return an empty DataFrame with the expected cache schema."""
    return pd.DataFrame(columns=_CACHE_COLUMNS).astype(
        {"timestamp": "int64", "value": "object"}
    )


def _load_endpoint_cache(cache_dir: Path, asset: str, endpoint: str) -> pd.DataFrame:
    """Load the Parquet cache for one (asset, endpoint). Tiny: ~KB not GB."""
    path = _endpoint_cache_path(cache_dir, asset, endpoint)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            if all(col in df.columns for col in _CACHE_COLUMNS):
                return df
            logger.warning(f"Unexpected schema in {path} -- rebuilding.")
        except Exception as exc:
            logger.warning(f"Could not read {path} ({exc}) -- starting fresh.")
    return _empty_cache()


def _save_endpoint_cache(df: pd.DataFrame, cache_dir: Path, asset: str, endpoint: str) -> None:
    """Deduplicate and persist cache for one (asset, endpoint)."""
    path = _endpoint_cache_path(cache_dir, asset, endpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop_duplicates(subset=["asset", "endpoint", "timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "value" in df.columns:
        df["value"] = df["value"].astype(str)
    df.to_parquet(path, index=False, engine="pyarrow")


def _query_endpoint_cache(
    cache: pd.DataFrame,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    """Filter a single-endpoint cache DataFrame to [start_ts, end_ts]."""
    if cache.empty:
        return cache
    mask = (cache["timestamp"] >= start_ts) & (cache["timestamp"] <= end_ts)
    return cache.loc[mask].copy()


# ──────────────────────────────────────────────────────────────────────────────
# Legacy monolithic-cache migration
# ──────────────────────────────────────────────────────────────────────────────

def _migrate_monolithic_cache(legacy_path: Path, cache_dir: Path) -> None:
    """
    Split the old single-file cache into per-endpoint Parquet files.

    Reads the monolithic file in chunks grouped by (asset, endpoint) to
    avoid loading 25 GB+ all at once, writes each group to its own file,
    then renames the original to ``*.migrated`` so this runs only once.
    """
    if not legacy_path.exists():
        return

    logger.info(f"Migrating monolithic cache {legacy_path} -> per-endpoint files...")

    try:
        # Read only the grouping columns first to get unique (asset, endpoint) pairs
        meta = pd.read_parquet(legacy_path, columns=["asset", "endpoint"])
        pairs = meta.drop_duplicates().values.tolist()
        del meta
        gc.collect()

        for asset, endpoint in pairs:
            # Read only this (asset, endpoint) slice using row-group filtering
            # pyarrow predicate pushdown keeps memory proportional to ONE endpoint
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(
                    legacy_path,
                    filters=[
                        ("asset", "=", asset),
                        ("endpoint", "=", endpoint),
                    ],
                )
                chunk = table.to_pandas()
                del table
            except Exception:
                # Fallback: full read + filter (slower but works)
                full = pd.read_parquet(legacy_path)
                chunk = full[(full["asset"] == asset) & (full["endpoint"] == endpoint)].copy()
                del full
                gc.collect()

            if not chunk.empty:
                _save_endpoint_cache(chunk, cache_dir, asset, endpoint)
                logger.info(
                    f"  Migrated {_endpoint_short_name(endpoint)}: "
                    f"{len(chunk)} rows -> {_endpoint_cache_path(cache_dir, asset, endpoint)}"
                )
            del chunk
            gc.collect()

        # Rename old file so migration doesn't re-run
        migrated_path = legacy_path.with_suffix(".parquet.migrated")
        legacy_path.rename(migrated_path)
        logger.info(f"  Migration complete. Old file renamed to {migrated_path}")

    except Exception as exc:
        logger.error(f"  Migration failed: {exc}  (will retry next run)")


# ──────────────────────────────────────────────────────────────────────────────
# Gap Detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_gaps(
    cached_timestamps: set[int],
    start_ts: int,
    end_ts: int,
    step: int = 86_400,
) -> list[tuple[int, int]]:
    """
    Return ``(gap_start, gap_end)`` tuples for contiguous missing segments
    within ``[start_ts, end_ts]`` at the given ``step``.
    """
    all_expected = set(range(start_ts, end_ts + 1, step))
    missing = sorted(all_expected - cached_timestamps)

    if not missing:
        return []

    gaps: list[tuple[int, int]] = []
    gap_start = prev = missing[0]
    for ts in missing[1:]:
        if ts - prev > step:  # new contiguous segment
            gaps.append((gap_start, prev))
            gap_start = ts
        prev = ts
    gaps.append((gap_start, prev))
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
# Glassnode REST API
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_api(
    endpoint: str,
    asset: str,
    start_ts: int,
    end_ts: int,
    resolution: str,
    api_key: str,
) -> list[dict]:
    """
    Single Glassnode API call.  Costs **1 credit**.

    Returns the raw JSON list ``[{"t": <unix>, "v": <value>}, ...]``.
    """
    url = f"{BASE_URL}{endpoint}"
    params = {
        "a":       asset,
        "s":       str(start_ts),
        "u":       str(end_ts),
        "i":       resolution,
        "api_key": api_key,
    }

    logger.info(
        f"  API   {_endpoint_short_name(endpoint):>40s}  "
        f"{_from_unix(start_ts)} -> {_from_unix(end_ts)}"
    )

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response type ({type(data).__name__})")
    return data


def _api_rows_to_dicts(
    raw: list[dict],
    asset: str,
    endpoint: str,
) -> list[dict]:
    """Convert raw API response items into cache-schema dicts."""
    rows: list[dict] = []
    for point in raw:
        ts = point["t"]
        # Some endpoints (e.g. options IV) use "o" instead of "v" for
        # structured values; fall back to "o" when "v" is absent.
        val = point.get("v")
        if val is None and "o" in point:
            val = point["o"]
        # Some metrics return nested objects (e.g. OHLC) -- serialise as JSON
        if isinstance(val, (dict, list)):
            val = json.dumps(val)
        rows.append(
            {
                "asset":     asset,
                "endpoint":  endpoint,
                "timestamp": ts,
                "date":      _from_unix(ts),
                "value":     val,
            }
        )
    return rows


def _try_expand_structured_values(series: pd.Series) -> Optional[pd.DataFrame]:
    """
    If values in *series* are JSON-encoded dicts (or lists), parse them and
    return a DataFrame with one column per dict key (or list index).

    Returns ``None`` when the values are plain scalars.
    """
    parsed: list[Optional[dict]] = []
    has_structured = False
    for val in series:
        if pd.isna(val) or val is None:
            parsed.append(None)
            continue
        if isinstance(val, str):
            try:
                obj = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                parsed.append(None)
                continue
            if isinstance(obj, dict):
                parsed.append(obj)
                has_structured = True
            elif isinstance(obj, list):
                parsed.append({str(i): v for i, v in enumerate(obj)})
                has_structured = True
            else:
                parsed.append(None)
        else:
            parsed.append(None)

    if not has_structured:
        return None

    return pd.DataFrame(parsed, index=series.index)


# ──────────────────────────────────────────────────────────────────────────────
# Single-endpoint fetch (low-memory building block)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_single_endpoint(
    endpoint: str,
    asset: str = "BTC",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    resolution: str = DEFAULT_RESOLUTION,
    cache_dir: str | Path | None = None,
    api_key: str | None = API_KEY,
    friendly_names: bool = True,
) -> pd.DataFrame:
    """
    Fetch ONE endpoint's data, using per-endpoint Parquet cache.

    Returns a small DataFrame with columns ``[date, <metric_col(s)>]``.
    Peak memory = O(rows for this single endpoint) — typically a few MB.
    """
    _api_key = api_key or os.environ.get("GLASSNODE_API_KEY", "")
    if not _api_key:
        raise ValueError("No API key provided.")

    _cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    step = RESOLUTION_TO_SECONDS.get(resolution, 86_400)
    start_ts = _to_unix(start_date)
    end_ts = _to_unix(end_date)

    short = _endpoint_short_name(endpoint)

    # ── Load per-endpoint cache (tiny file) ──────────────────────────────
    ep_cache = _load_endpoint_cache(_cache_dir, asset, endpoint)
    cached = _query_endpoint_cache(ep_cache, start_ts, end_ts)
    cached_ts = set(cached["timestamp"].tolist()) if not cached.empty else set()

    gaps = _find_gaps(cached_ts, start_ts, end_ts, step)

    if gaps:
        total_expected = len(range(start_ts, end_ts + 1, step))
        cached_count = len(cached_ts)
        logger.info(
            f"  GAPS  {short:>40s}  "
            f"{cached_count}/{total_expected} cached, "
            f"{total_expected - cached_count} missing -- fetching full range"
        )
        try:
            raw = _fetch_api(endpoint, asset, start_ts, end_ts + step, resolution, _api_key)
            new_rows = _api_rows_to_dicts(raw, asset, endpoint)
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                ep_cache = pd.concat([ep_cache, new_df], ignore_index=True)
                _save_endpoint_cache(ep_cache, _cache_dir, asset, endpoint)
                logger.info(f"  SAVED +{len(new_rows)} rows -> {_endpoint_cache_path(_cache_dir, asset, endpoint)}")
                del new_df
            del raw, new_rows
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            logger.error(f"  ERROR {short}: HTTP {status} -- {exc}")
        except Exception as exc:
            logger.error(f"  ERROR {short}: {exc}")
    else:
        logger.info(f"  CACHE {short:>40s}  fully cached  (0 credits)")

    # ── Build result for this endpoint ───────────────────────────────────
    ep_data = _query_endpoint_cache(ep_cache, start_ts, end_ts)
    del ep_cache
    gc.collect()

    if ep_data.empty:
        return pd.DataFrame()

    col = short
    if friendly_names:
        col = FRIENDLY_NAMES.get(col, col)

    ep_data = ep_data[["timestamp", "date", "value"]].copy()

    expanded = _try_expand_structured_values(ep_data["value"])
    if expanded is not None:
        for key in expanded.columns:
            sub_col = f"{col}_{key}"
            ep_data[sub_col] = pd.to_numeric(expanded[key], errors="coerce").astype(np.float32)
        ep_data = ep_data.drop(columns=["value"])
        del expanded
    else:
        ep_data = ep_data.rename(columns={"value": col})
        ep_data[col] = pd.to_numeric(ep_data[col], errors="coerce").astype(np.float32)

    ep_data = ep_data.sort_values("timestamp").reset_index(drop=True)
    return ep_data


# ──────────────────────────────────────────────────────────────────────────────
# Main Public Function (multi-endpoint, backward-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def fetch_glassnode_data(
    asset: str = "BTC",
    endpoints: Optional[list[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    resolution: str = DEFAULT_RESOLUTION,
    cache_path: Optional[str | Path] = None,
    api_key: Optional[str] = API_KEY,
    friendly_names: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical Glassnode data for *one asset* across *multiple endpoints*.

    Now uses **per-endpoint Parquet files** instead of a single monolithic
    cache.  Only one endpoint's data is in memory at a time during the
    fetch-and-merge loop.

    Parameters
    ----------
    asset : str
        Cryptocurrency symbol (e.g. ``"BTC"``, ``"ETH"``).
    endpoints : list[str] | None
        Glassnode metric paths.  Defaults to the 5 standard price / volatility
        endpoints when ``None``.
    start_date, end_date : str
        Inclusive date boundaries in ``"YYYY-MM-DD"`` format.
    resolution : str
        Candle / interval width (``"1h"``, ``"24h"``, ``"1w"``, ``"1month"``).
    cache_path : str | Path | None
        Override the default cache **directory** location.  (For backward
        compatibility, if this points to a file the parent directory is used.)
    api_key : str | None
        Glassnode API key.  Falls back to ``GLASSNODE_API_KEY`` env var.
    friendly_names : bool
        Rename known endpoint columns to short readable names (default True).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns ``asset``, ``date``, and one value
        column per endpoint.  Sorted by date ascending.
    """
    if endpoints is None:
        endpoints = list(DEFAULT_ENDPOINTS)

    # Resolve cache directory (backward-compat: if user passes a file path,
    # use its parent directory instead)
    if cache_path is not None:
        _cache_dir = Path(cache_path)
        if _cache_dir.suffix in (".parquet", ".pq"):
            _cache_dir = _cache_dir.parent
    else:
        _cache_dir = DEFAULT_CACHE_DIR

    # ── Auto-migrate old monolithic cache if present ─────────────────────
    _legacy = _cache_dir / "glassnode_data.parquet"
    if _legacy.exists():
        _migrate_monolithic_cache(_legacy, _cache_dir)

    logger.info(
        f"Fetching {len(endpoints)} endpoint(s) for {asset}  "
        f"[{start_date} -> {end_date}]  resolution={resolution}"
    )

    # ── Fetch each endpoint and merge incrementally ──────────────────────
    result: pd.DataFrame | None = None
    api_calls_made = 0
    credits_saved = 0

    for endpoint in endpoints:
        ep_df = fetch_single_endpoint(
            endpoint=endpoint,
            asset=asset,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution,
            cache_dir=_cache_dir,
            api_key=api_key,
            friendly_names=friendly_names,
        )

        if ep_df.empty:
            continue

        if result is None:
            result = ep_df
        else:
            result = result.merge(ep_df, on=["timestamp", "date"], how="outer")
        del ep_df
        gc.collect()

    if result is None or result.empty:
        logger.warning("No data available for the requested range.")
        return pd.DataFrame()

    result = result.sort_values("timestamp").reset_index(drop=True)
    result.insert(0, "asset", asset)
    result.drop(columns=["timestamp"], inplace=True)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info(
        f"\n{'=' * 65}\n"
        f"  Asset:            {asset}\n"
        f"  Date range:       {start_date}  ->  {end_date}\n"
        f"  Endpoints:        {len(endpoints)}\n"
        f"  Result shape:     {result.shape[0]} rows x {result.shape[1]} cols\n"
        f"  Cache location:   {_cache_dir.resolve()}\n"
        f"{'=' * 65}"
    )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Glassnode historical data with local Parquet caching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python glassnode_fetcher.py --asset BTC --start 2024-01-01 --end 2024-06-30\n"
            "  python glassnode_fetcher.py --asset ETH --start 2025-01-01 --end 2025-03-01 "
            "--resolution 1h\n"
        ),
    )
    parser.add_argument("--asset",      default="BTC",        help="Asset symbol (default: BTC)")
    parser.add_argument("--start",      default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--resolution", default="24h",        help="Resolution: 1h, 24h, 1w, 1month")
    parser.add_argument("--cache",      default=None,         help="Cache directory path override")
    parser.add_argument(
        "--endpoints", nargs="*", default=None,
        help="Endpoint paths (space-separated). Defaults to 5 standard metrics.",
    )
    parser.add_argument("--api-key", default=None, help="Glassnode API key (or use env var)")

    args = parser.parse_args()

    df = fetch_glassnode_data(
        asset=args.asset,
        endpoints=args.endpoints,
        start_date=args.start,
        end_date=args.end,
        resolution=args.resolution,
        cache_path=args.cache,
        api_key=args.api_key,
    )
    df.to_csv("temp.csv", index=False)
    print(f"\n{df.to_string(index=False, max_rows=30)}\n")
