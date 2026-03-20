"""
TWLGBM Data Pipeline
====================
Fetches and aligns daily-close prices, Glassnode on-chain metrics,
and FRED macro data into a single feature DataFrame aligned on a
configurable anchor day (Thursday, Friday, or Monday).

All data uses the anchor day's UTC daily close (≈23:59 UTC).
The pipeline is designed to run ≥1 day after the anchor so that the
full daily bar is available from every source.

Supported anchor days:
  - Thursday (default): run Friday ≥ 03:00 UTC
  - Friday: run Saturday ≥ 04:00 UTC
  - Monday: run Tuesday ≥ 04:00 UTC

This module is the sole data-ingestion layer for the TWLGBM system.
"""

from __future__ import annotations

import logging
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from glassnode_price_fetcher import fetch_close_price
from glassnode_fetcher import fetch_single_endpoint
from twlgbm_config import (
    FRED_API_KEY,
    FRED_SERIES_MACRO,
    FRED_SERIES_RISK,
    FRED_SERIES_LIQUIDITY,
    FRED_LOG_DIFF_SERIES,
    HORIZON_WEEKS,
    BUCKET_THRESHOLDS,
    NUM_BUCKETS,
    SUM_KEYWORDS,
    RATIO_KEYWORDS,
    CROSS_ASSETS,
    CROSS_ASSET_ENDPOINTS,
    STABLECOIN_ASSETS,
    STABLECOIN_ENDPOINTS,
    ANCHOR_DAY_WEEKDAY,
    ANCHOR_DAY_FREQ,
    DEFAULT_ANCHOR_DAY,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────────────────────

def _most_recent_anchor_day(
    ref_date: datetime | None = None,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> str:
    """Return the most recent anchor day as 'YYYY-MM-DD' (inclusive of today)."""
    if ref_date is None:
        ref_date = datetime.now(timezone.utc)
    weekday_target = ANCHOR_DAY_WEEKDAY[anchor_day]
    days_back = (ref_date.weekday() - weekday_target) % 7
    anchor = ref_date - timedelta(days=days_back)
    return anchor.strftime("%Y-%m-%d")


# Keep backward-compatible alias
def _most_recent_thursday(ref_date: datetime | None = None) -> str:
    """Return the most recent Thursday as 'YYYY-MM-DD' (inclusive of today if Thursday)."""
    return _most_recent_anchor_day(ref_date, anchor_day="thursday")


def _default_start_date(
    weeks_back: int = 210,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> str:
    """Return a date ~weeks_back weeks before the most recent anchor day."""
    ref = datetime.now(timezone.utc)
    weekday_target = ANCHOR_DAY_WEEKDAY[anchor_day]
    days_back = (ref.weekday() - weekday_target) % 7
    latest = ref - timedelta(days=days_back)
    start = latest - timedelta(weeks=weeks_back)
    # Snap to previous anchor day
    days_back_start = (start.weekday() - weekday_target) % 7
    start = start - timedelta(days=days_back_start)
    return start.strftime("%Y-%m-%d")


def _anchor_dates(
    start_date: str,
    end_date: str,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> pd.DatetimeIndex:
    """Generate a DatetimeIndex of all anchor days between start_date and end_date."""
    freq = ANCHOR_DAY_FREQ[anchor_day]
    return pd.date_range(start=start_date, end=end_date, freq=freq)


# Keep backward-compatible alias
def _thursday_dates(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Generate a DatetimeIndex of all Thursdays between start_date and end_date."""
    return _anchor_dates(start_date, end_date, anchor_day="thursday")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Thursday daily-close prices
# ──────────────────────────────────────────────────────────────────────────────

def fetch_anchor_day_prices(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> pd.DataFrame:
    """
    Retrieve daily-close prices on the anchor day (≈23:59 UTC).

    Uses the standard Glassnode daily bar (midnight-to-midnight UTC).
    The daily close aligns exactly with on-chain metrics, eliminating
    any intra-day lookahead bias.

    Returns DataFrame with columns: ['date', 'price_usd_close']
    where 'date' is a string 'YYYY-MM-DD' corresponding to anchor days only.
    """
    weekday_target = ANCHOR_DAY_WEEKDAY[anchor_day]
    logger.info(f"Fetching {anchor_day.capitalize()} daily-close prices for {asset}: {start_date} -> {end_date}")

    # Fetch true daily-close prices (24h resolution, end-of-day UTC)
    price_df = fetch_close_price(
        start_date=start_date,
        end_date=end_date,
        asset=asset,
    )

    # Filter to anchor day only
    price_df["date_dt"] = pd.to_datetime(price_df["date"])
    price_df = price_df[price_df["date_dt"].dt.weekday == weekday_target].copy()
    price_df = price_df.drop(columns=["date_dt"])

    if "asset" in price_df.columns:
        price_df = price_df.drop(columns=["asset"])

    logger.info(f"  Got {len(price_df)} {anchor_day.capitalize()} prices")
    return price_df.reset_index(drop=True)


# Keep backward-compatible alias
def fetch_thursday_prices(
    start_date: str, end_date: str, asset: str = "BTC",
) -> pd.DataFrame:
    """Retrieve daily-close prices on Thursdays."""
    return fetch_anchor_day_prices(start_date, end_date, asset, anchor_day="thursday")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Compute forward returns and bucket labels
# ──────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 1W, 4W, 13W forward log-returns from Thursday close prices.

    Parameters
    ----------
    price_df : DataFrame with columns ['date', 'price_usd_close']

    Returns
    -------
    DataFrame with columns: date, price, ret_1w, ret_4w, ret_13w
    """
    df = price_df.copy()
    price_col = [c for c in df.columns if c.startswith("price_usd")][0]
    df = df.rename(columns={price_col: "price"})
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt").reset_index(drop=True)

    for horizon, weeks in HORIZON_WEEKS.items():
        col = f"ret_{horizon}"
        df[col] = np.log(df["price"].shift(-weeks) / df["price"])

    df = df.drop(columns=["date_dt"])
    return df


def assign_bucket_labels(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Assign integer bucket labels {0..4} for each horizon based on thresholds.

    Uses a single calibration: boundaries computed from the full training window
    (start_date to end_date). Does NOT recalibrate.

    B1: r <= -outer   (big down)
    B2: -outer < r <= -inner
    B3: -inner < r <= +inner  (flat)
    B4: +inner < r <= +outer
    B5: r > +outer    (big up)
    """
    if thresholds is None:
        thresholds = BUCKET_THRESHOLDS

    df = df.copy()
    for horizon in HORIZON_WEEKS:
        ret_col = f"ret_{horizon}"
        label_col = f"label_{horizon}"
        bounds = thresholds[horizon]["boundaries"]  # [b1, b2, b3, b4]

        conditions = [
            df[ret_col] <= bounds[0],
            (df[ret_col] > bounds[0]) & (df[ret_col] <= bounds[1]),
            (df[ret_col] > bounds[1]) & (df[ret_col] <= bounds[2]),
            (df[ret_col] > bounds[2]) & (df[ret_col] <= bounds[3]),
            df[ret_col] > bounds[3],
        ]
        choices = [0, 1, 2, 3, 4]
        df[label_col] = np.select(conditions, choices, default=np.nan)
        # NaN where return is NaN (forward-looking data not yet available)
        df.loc[df[ret_col].isna(), label_col] = np.nan

    return df


def calibrate_bucket_thresholds(returns_series: pd.Series, horizon: str) -> dict:
    """
    Calibrate bucket boundaries from empirical return distribution.

    Pure quintile-based boundaries (20th/40th/60th/80th percentiles) so that
    each of the 5 buckets contains approximately 20% of observations.
    No center drift adjustment — let the data speak for itself.

    Returns dict with 'boundaries': [b1, b2, b3, b4] (4 cut-points).
    """
    clean = returns_series.dropna()
    if len(clean) < 30:
        logger.warning(f"Only {len(clean)} observations for {horizon}, using defaults")
        return BUCKET_THRESHOLDS[horizon]

    # Pure quintile boundaries for equal-weight buckets
    q20 = float(np.percentile(clean, 20))
    q40 = float(np.percentile(clean, 40))
    q60 = float(np.percentile(clean, 60))
    q80 = float(np.percentile(clean, 80))

    boundaries = [q20, q40, q60, q80]

    # Verify bucket weights
    n = len(clean)
    b1 = np.sum(clean <= boundaries[0]) / n
    b2 = np.sum((clean > boundaries[0]) & (clean <= boundaries[1])) / n
    b3 = np.sum((clean > boundaries[1]) & (clean <= boundaries[2])) / n
    b4 = np.sum((clean > boundaries[2]) & (clean <= boundaries[3])) / n
    b5 = np.sum(clean > boundaries[3]) / n

    center = (boundaries[1] + boundaries[2]) / 2
    logger.info(
        f"  Calibrated {horizon}: boundaries={[f'{b:.4f}' for b in boundaries]}  "
        f"center={center:.4f}  "
        f"bucket dist: [{b1:.1%}, {b2:.1%}, {b3:.1%}, {b4:.1%}, {b5:.1%}]"
    )

    # Constraint check: no bucket below 10%
    min_pct = min(b1, b2, b3, b4, b5)
    if min_pct < 0.10:
        logger.warning(
            f"  {horizon}: minimum bucket {min_pct:.1%} < 10%, falling back to defaults"
        )
        return BUCKET_THRESHOLDS[horizon]

    return {"boundaries": [round(b, 3) for b in boundaries]}


# ──────────────────────────────────────────────────────────────────────────────
# 3. Glassnode on-chain features
# ──────────────────────────────────────────────────────────────────────────────

def _load_metric_endpoints(
    csv_path: str = "btc_glassnode_metrics.csv",
) -> list[tuple[str, str]]:
    """Load (endpoint, category) pairs from the metrics CSV."""
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row.get("endpoint", "").strip()
            cat = row.get("category", "").strip()
            if ep and ep.startswith("/v1/metrics/"):
                results.append((ep, cat))
    return results


def _is_sum_metric(col_name: str) -> bool:
    """Check if a metric column should be weekly-summed (flow/volume metric)."""
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in SUM_KEYWORDS)


def _is_ratio_metric(col_name: str) -> bool:
    """Check if a metric is already stationary (ratio/percentage/index)."""
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in RATIO_KEYWORDS)


def _daily_to_weekly(
    daily_df: pd.DataFrame,
    metric_cols: list[str],
    anchor_dates: pd.DatetimeIndex,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> pd.DataFrame:
    """
    Transform daily Glassnode data to weekly frequency aligned on the anchor day.

    Weekly windows run from (anchor_day+1) of the previous week through
    anchor_day of the current week, matching the anchor day's daily-close
    price.  No intra-day lookahead exists because both the price and
    features use the same daily-close timestamp.

    - SUM_KEYWORDS metrics: sum 7 daily values in the weekly window
    - All other metrics: take the anchor day's daily close (last value)

    Then apply stationarity transforms:
    - RATIO_KEYWORDS metrics: keep as-is (already stationary)
    - All other metrics: week-over-week difference
    """
    weekday_target = ANCHOR_DAY_WEEKDAY[anchor_day]
    daily_df = daily_df.copy()
    daily_df["date_dt"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date_dt")

    # Assign each day to its week-ending anchor day
    daily_df["week_anchor"] = daily_df["date_dt"] + pd.to_timedelta(
        (weekday_target - daily_df["date_dt"].dt.weekday) % 7, unit="D"
    )

    # Separate sum vs last-value metrics
    sum_cols = [c for c in metric_cols if _is_sum_metric(c)]
    last_cols = [c for c in metric_cols if not _is_sum_metric(c)]

    weekly_parts = []

    if sum_cols:
        weekly_sum = (
            daily_df.groupby("week_anchor")[sum_cols]
            .sum()
            .reset_index()
            .rename(columns={"week_anchor": "date_dt"})
        )
        weekly_parts.append(weekly_sum)

    if last_cols:
        weekly_last = (
            daily_df.groupby("week_anchor")[last_cols]
            .last()
            .reset_index()
            .rename(columns={"week_anchor": "date_dt"})
        )
        weekly_parts.append(weekly_last)

    if not weekly_parts:
        return pd.DataFrame({"date_dt": anchor_dates})

    # Merge sum and last parts
    weekly = weekly_parts[0]
    for part in weekly_parts[1:]:
        weekly = weekly.merge(part, on="date_dt", how="outer")

    # Align to requested anchor dates
    weekly = weekly.sort_values("date_dt")
    thu_df = pd.DataFrame({"date_dt": anchor_dates}).sort_values("date_dt")
    aligned = pd.merge_asof(thu_df, weekly, on="date_dt", direction="backward")

    # Apply stationarity: difference non-ratio metrics
    all_cols = sum_cols + last_cols
    for col in all_cols:
        if col not in aligned.columns:
            continue
        if not _is_ratio_metric(col):
            aligned[col] = aligned[col].diff()

    return aligned


def _fetch_glassnode_endpoints(
    asset_endpoint_pairs: list[tuple[str, str]],
    start_date: str,
    end_date: str,
    category_fn: callable,
    col_prefix_fn: callable | None = None,
    label: str = "Glassnode",
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Generic fetcher: stream Glassnode endpoints one at a time, transform
    daily → weekly, and merge into a single anchor-day-aligned DataFrame.

    Parameters
    ----------
    asset_endpoint_pairs : list of (asset, endpoint) tuples
    category_fn : (asset, endpoint) -> category string
    col_prefix_fn : (asset, col) -> prefixed col name, or None for no prefix
    label : label for logging
    anchor_day : anchor day for weekly alignment
    """
    import gc

    anchor_day_dates = _anchor_dates(start_date, end_date, anchor_day)
    result = pd.DataFrame({"date_dt": anchor_day_dates}).sort_values("date_dt")
    col_to_category: dict[str, str] = {}
    n_metrics = 0

    for i, (asset, endpoint) in enumerate(asset_endpoint_pairs):
        try:
            ep_daily = fetch_single_endpoint(
                endpoint=endpoint, asset=asset,
                start_date=start_date, end_date=end_date,
                resolution="24h", friendly_names=False,
            )
            if ep_daily.empty:
                continue

            skip = {"timestamp", "date", "date_dt", "asset"}
            metric_cols = [c for c in ep_daily.columns if c not in skip]
            if not metric_cols:
                del ep_daily
                continue

            # Optional column prefix (for cross-asset / stablecoin)
            if col_prefix_fn is not None:
                rename_map = {c: col_prefix_fn(asset, c) for c in metric_cols}
                ep_daily = ep_daily.rename(columns=rename_map)
                metric_cols = list(rename_map.values())

            for col in metric_cols:
                ep_daily[col] = pd.to_numeric(ep_daily[col], errors="coerce")

            ep_weekly = _daily_to_weekly(ep_daily, metric_cols, anchor_day_dates, anchor_day)
            result = result.merge(ep_weekly, on="date_dt", how="left")

            for col in metric_cols:
                col_to_category[col] = category_fn(asset, endpoint)
            n_metrics += len(metric_cols)

            del ep_daily, ep_weekly
            gc.collect()

        except Exception as exc:
            logger.warning(f"  {label} #{i} ({asset} {endpoint}) failed: {exc}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"  {label}: processed {i + 1}/{len(asset_endpoint_pairs)} ({n_metrics} cols)")

    result["date"] = result["date_dt"].dt.strftime("%Y-%m-%d")
    result = result.drop(columns=["date_dt"]).reset_index(drop=True)

    logger.info(f"  {label}: {result.shape[0]} {anchor_day.capitalize()}s x {n_metrics} metrics")
    return result, col_to_category


def fetch_glassnode_features(
    start_date: str, end_date: str,
    asset: str = "BTC", metrics_csv: str = "btc_glassnode_metrics.csv",
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch all Glassnode on-chain metrics (daily→weekly, stationarity-transformed)."""
    endpoint_cats = _load_metric_endpoints(metrics_csv)
    logger.info(f"Fetching {len(endpoint_cats)} Glassnode metrics for {asset}")
    pairs = [(asset, ep) for ep, _cat in endpoint_cats]
    cat_lookup = {ep: cat for ep, cat in endpoint_cats}
    return _fetch_glassnode_endpoints(
        pairs, start_date, end_date,
        category_fn=lambda _a, ep: cat_lookup.get(ep, "unknown"),
        label=f"Glassnode({asset})",
        anchor_day=anchor_day,
    )


def fetch_cross_asset_features(
    start_date: str, end_date: str,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch price + derivatives for ETH, SOL from Glassnode."""
    pairs = [(a, ep) for a in CROSS_ASSETS for ep in CROSS_ASSET_ENDPOINTS]
    return _fetch_glassnode_endpoints(
        pairs, start_date, end_date,
        category_fn=lambda a, ep: f"{a.lower()}_{'derivatives' if 'derivatives' in ep else 'market'}",
        col_prefix_fn=lambda a, c: f"{a.lower()}_{c}",
        label="CrossAsset",
        anchor_day=anchor_day,
    )


def fetch_stablecoin_features(
    start_date: str, end_date: str,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch circulating supply for USDT, USDC from Glassnode."""
    pairs = [(a, ep) for a in STABLECOIN_ASSETS for ep in STABLECOIN_ENDPOINTS]
    return _fetch_glassnode_endpoints(
        pairs, start_date, end_date,
        category_fn=lambda a, _ep: f"{a.lower()}_supply",
        col_prefix_fn=lambda a, c: f"{a.lower()}_{c}",
        label="Stablecoin",
        anchor_day=anchor_day,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. FRED macro data
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a single FRED series and return DataFrame with date index."""
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        f"&observation_start={start_date}&observation_end={end_date}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    obs = resp.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    df = df.set_index("date")
    df = df.rename(columns={"value": series_id})
    return df


def fetch_fred_features(
    start_date: str, end_date: str,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> pd.DataFrame:
    """
    Fetch all FRED macro and risk series, align to anchor days between
    start_date and end_date using most-recent-available (as-of) logic.

    Returns DataFrame with 'date' column (string) and one column per series.
    """
    logger.info("Fetching FRED macro data")
    all_series = FRED_SERIES_MACRO + FRED_SERIES_RISK + FRED_SERIES_LIQUIDITY

    frames = []
    for series_id in all_series:
        try:
            df = _fetch_fred_series(series_id, start_date, end_date)
            frames.append(df)
            logger.info(f"  FRED {series_id}: {len(df)} observations")
        except Exception as exc:
            logger.warning(f"  FRED {series_id} failed: {exc}")

    if not frames:
        logger.warning("No FRED data returned")
        return pd.DataFrame()

    # Join all series on date, dropping all-NA columns to avoid FutureWarning
    frames = [f.dropna(axis=1, how="all") for f in frames]
    combined = pd.concat(frames, axis=1)
    # Forward-fill only — bfill would leak future values into early rows
    combined = combined.ffill()

    # Align to anchor days using merge_asof (avoids reindexing to every calendar day)
    thursdays = _anchor_dates(start_date, end_date, anchor_day)
    combined = combined.sort_index()
    combined = combined.reset_index().rename(columns={combined.index.name or "index": "date_dt"})
    thu_df = pd.DataFrame({"date_dt": thursdays}).sort_values("date_dt")
    fred_merged = pd.merge_asof(
        thu_df,
        combined,
        on="date_dt",
        direction="backward",
    )
    fred_weekly = fred_merged.set_index("date_dt")

    # Apply log-differencing to level series (SP500, Oil)
    for series_id in FRED_LOG_DIFF_SERIES:
        if series_id in fred_weekly.columns:
            diff_col = f"{series_id}_logret"
            fred_weekly[diff_col] = np.log(
                fred_weekly[series_id] / fred_weekly[series_id].shift(1)
            )
            fred_weekly = fred_weekly.drop(columns=[series_id])

    # Compute Net Liquidity = WALCL - TGA (WTREGEN) - RRPONTSYD
    # Measures actual USD liquidity flowing into financial markets
    liq_cols = {"WALCL", "WTREGEN", "RRPONTSYD"}
    if liq_cols.issubset(fred_weekly.columns):
        fred_weekly["net_liquidity"] = (
            fred_weekly["WALCL"] - fred_weekly["WTREGEN"] - fred_weekly["RRPONTSYD"]
        )
        # Also compute week-over-week change (momentum of liquidity)
        fred_weekly["net_liquidity_chg1w"] = fred_weekly["net_liquidity"].pct_change(fill_method=None)
        # Drop raw components — the composite is more informative and avoids
        # multicollinearity with its own constituents
        fred_weekly = fred_weekly.drop(columns=list(liq_cols))
        logger.info("  Computed net_liquidity composite feature (WALCL - TGA - RRP)")
    else:
        missing = liq_cols - set(fred_weekly.columns)
        logger.warning(f"  Net liquidity skipped — missing series: {missing}")

    fred_weekly["date"] = fred_weekly.index.strftime("%Y-%m-%d")
    fred_weekly = fred_weekly.reset_index(drop=True)

    logger.info(f"  FRED features: {fred_weekly.shape[0]} {anchor_day.capitalize()}s x {fred_weekly.shape[1]-1} series")
    return fred_weekly


# ──────────────────────────────────────────────────────────────────────────────
# 5. Assemble complete dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    asset: str = "BTC",
    metrics_csv: str = "btc_glassnode_metrics.csv",
    calibrate_from_data: bool = True,
    anchor_day: str = DEFAULT_ANCHOR_DAY,
) -> tuple[pd.DataFrame, dict, dict[str, str]]:
    """
    Build the complete anchor-day-aligned dataset with:
      - Forward returns and bucket labels
      - Glassnode on-chain features (daily→weekly, stationarity-transformed)
      - Cross-asset features (ETH, SOL price + derivatives)
      - Stablecoin supply features (USDT, USDC)
      - FRED macro features

    Parameters
    ----------
    start_date : str or None
        Start date. Defaults to 210 weeks before the most recent anchor day.
    end_date : str or None
        End date. Defaults to the most recent anchor day.
    asset : str
        Asset symbol (default 'BTC').
    metrics_csv : str
        Path to the CSV listing Glassnode metric endpoints.
    calibrate_from_data : bool
        If True, calibrate bucket thresholds from the data.
        If False, use static defaults from config.
    anchor_day : str
        Day of week for data alignment: "thursday", "friday", or "monday".

    Returns
    -------
    (df, thresholds, col_to_category) where df is the merged DataFrame,
    thresholds is the calibrated bucket boundary dict, and col_to_category
    maps feature columns to their Glassnode category.
    """
    if end_date is None:
        end_date = _most_recent_anchor_day(anchor_day=anchor_day)
    if start_date is None:
        start_date = _default_start_date(weeks_back=210, anchor_day=anchor_day)

    logger.info(f"Building TWLGBM dataset: {start_date} -> {end_date}, asset={asset}, anchor={anchor_day}")

    # 1. Anchor-day daily-close prices
    price_df = fetch_anchor_day_prices(start_date, end_date, asset, anchor_day)

    # 2. Forward returns
    ret_df = compute_forward_returns(price_df)

    # 3. Calibrate bucket boundaries (one-time, from full window)
    if calibrate_from_data:
        thresholds = {}
        for horizon in HORIZON_WEEKS:
            ret_col = f"ret_{horizon}"
            valid_returns = ret_df[ret_col].dropna()
            thresholds[horizon] = calibrate_bucket_thresholds(valid_returns, horizon)
    else:
        thresholds = BUCKET_THRESHOLDS

    # 4. Assign bucket labels
    ret_df = assign_bucket_labels(ret_df, thresholds)

    # 5. Glassnode features (daily→weekly with stationarity)
    gn_df, col_to_category = fetch_glassnode_features(
        start_date, end_date, asset, metrics_csv, anchor_day=anchor_day,
    )

    # 6. Cross-asset features (ETH, SOL price + derivatives)
    cross_df, cross_cats = fetch_cross_asset_features(start_date, end_date, anchor_day=anchor_day)
    col_to_category.update(cross_cats)

    # 7. Stablecoin supply features (USDT, USDC)
    stable_df, stable_cats = fetch_stablecoin_features(start_date, end_date, anchor_day=anchor_day)
    col_to_category.update(stable_cats)

    # 8. FRED features
    fred_df = fetch_fred_features(start_date, end_date, anchor_day=anchor_day)

    # 9. Map FRED columns to categories
    fred_macro_set = set(FRED_SERIES_MACRO)
    fred_risk_set = set(FRED_SERIES_RISK)
    for col in fred_df.columns:
        if col == "date":
            continue
        base = col.replace("_logret", "")
        if base in fred_macro_set:
            col_to_category[col] = "fred_macro"
        elif base in fred_risk_set:
            col_to_category[col] = "fred_risk"
        elif "liquidity" in col.lower() or col in {"WALCL", "RRPONTSYD", "WTREGEN"}:
            col_to_category[col] = "fred_liquidity"

    # 10. Merge everything on 'date'
    dataset = ret_df.copy()
    for extra_df in [gn_df, cross_df, stable_df, fred_df]:
        if not extra_df.empty:
            dataset = dataset.merge(extra_df, on="date", how="left")

    dataset = dataset.sort_values("date").reset_index(drop=True)

    # NOTE: Do NOT downcast to float32 here — rolling aggregations in
    # engineer_features() overflow float32 and trigger RuntimeWarnings
    # that stall the pipeline.  Downcasting happens once in twlgbm_main.py
    # after all feature engineering is complete.

    n_features = len([
        c for c in dataset.columns
        if c not in ["date", "price"] and not c.startswith("ret_") and not c.startswith("label_")
    ])
    logger.info(
        f"Dataset built: {dataset.shape[0]} rows, {n_features} raw features, "
        f"3 horizons"
    )

    return dataset, thresholds, col_to_category
