#!/usr/bin/env python3
"""
TWLGBM Main Orchestrator (PLS + Horseshoe)
============================================
5-Bucket Multiclass Return Classification for BTC
3 Horizons (1W, 4W, 13W) x 5 Buckets via LightGBM

Runs both regularization paths on the same data:
  - PLS regression per feature category -> TWLGBM
  - Horseshoe prior shrinkage + correlation clustering -> TWLGBM

Outputs side-by-side comparison of both methods.

Anchor day options (--anchor-day):
  - thursday (default): Thu-to-Thu returns, run Friday >= 03:00 UTC
  - friday:  Fri-to-Fri returns, run Saturday >= 04:00 UTC
  - monday:  Mon-to-Mon returns, run Tuesday >= 04:00 UTC

Usage:
    python twlgbm_main.py
    python twlgbm_main.py --anchor-day friday     # Friday-to-Friday returns
    python twlgbm_main.py --anchor-day monday      # Monday-to-Monday returns
    python twlgbm_main.py --start-date 2022-01-07 --end-date 2026-03-06 --asset BTC
    python twlgbm_main.py --no-cv           # skip walk-forward CV (faster)
    python twlgbm_main.py --no-shap         # skip SHAP attribution
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from twlgbm_config import HORIZONS, BUCKET_NAMES, HORIZON_WEEKS, NUM_BUCKETS, DEFAULT_ANCHOR_DAY, ANCHOR_DAY_WEEKDAY
from twlgbm_data import build_dataset, _most_recent_anchor_day, _most_recent_thursday, _default_start_date
from twlgbm_features import (
    engineer_features,
    clean_features,
    get_feature_columns,
    apply_pls_by_category,
    apply_horseshoe_shrinkage,
    cluster_correlated_features,
    select_features_by_importance,
)
from twlgbm_model import train_all_classifiers
from twlgbm_output import generate_forecast

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./twlgbm_output")


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 -> float32 and int64 -> int32 in-place."""
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == np.int64:
            df[col] = df[col].astype(np.int32)
    return df


def _compute_historical_bucket_dist(
    dataset: pd.DataFrame,
    thresholds: dict,
) -> dict[str, list[float]]:
    """Compute historical bucket distribution from the dataset labels."""
    dist = {}
    for horizon in HORIZONS:
        label_col = f"label_{horizon}"
        if label_col not in dataset.columns:
            continue
        labels = dataset[label_col].dropna()
        n = len(labels)
        if n == 0:
            dist[horizon] = [0.0] * NUM_BUCKETS
            continue
        bucket_counts = [float((labels == i).sum()) / n for i in range(NUM_BUCKETS)]
        dist[horizon] = bucket_counts
    return dist


# ──────────────────────────────────────────────────────────────────────────────
# Per-method pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_pls_path(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    col_to_category: dict[str, str],
    run_cv: bool,
    compute_shap: bool,
    thresholds: dict,
    recent_fraction: float = 1.0,
) -> dict:
    """Run PLS regularization -> TWLGBM pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("  PLS PATH")
    logger.info("=" * 70)

    # PLS regression per category
    logger.info("  Applying PLS regression per feature category...")
    ds_pls, pls_feat_cols, pls_info = apply_pls_by_category(
        dataset, feature_cols, col_to_category,
        recent_fraction=recent_fraction,
    )
    logger.info(f"  Features after PLS: {len(pls_feat_cols)}")

    # Train classifiers
    logger.info("  Training classifiers (PLS)...")
    classifiers, cv_results = train_all_classifiers(
        ds_pls, feature_cols=pls_feat_cols, run_cv=run_cv,
    )

    # Per-horizon feature selection
    logger.info("  Per-horizon feature selection (PLS)...")
    selected_per_horizon: dict[str, list[str]] = {}
    for horizon in HORIZONS:
        try:
            clf = classifiers[horizon]
            importance = clf.model.feature_importance(
                importance_type="gain",
            ).astype(np.float32)
            imp_series = pd.Series(importance, index=pls_feat_cols)
            sel = select_features_by_importance(imp_series)
            selected_per_horizon[horizon] = sel
            logger.info(f"  PLS {horizon}: selected {len(sel)} / {len(pls_feat_cols)} features")
        except Exception as exc:
            logger.warning(f"  PLS {horizon} feature selection skipped: {exc}")
            selected_per_horizon[horizon] = pls_feat_cols

    del classifiers, cv_results
    gc.collect()

    # Retrain with selected features
    classifiers, cv_results = train_all_classifiers(
        ds_pls, feature_cols=selected_per_horizon, run_cv=run_cv,
    )

    # Generate forecast
    all_selected = sorted(set().union(*selected_per_horizon.values()))
    last_idx = ds_pls.index[-1]
    X_latest = ds_pls.loc[[last_idx], all_selected].astype(np.float32)
    gc.collect()

    forecasts = generate_forecast(
        classifiers=classifiers,
        X_latest=X_latest,
        thresholds=thresholds,
        compute_shap=compute_shap,
    )

    return {
        "method": "PLS",
        "forecasts": forecasts,
        "cv_results": cv_results if cv_results else {},
        "classifiers": classifiers,
        "selected_per_horizon": selected_per_horizon,
        "pls_info": pls_info,
    }


def _run_horseshoe_path(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    col_to_category: dict[str, str],
    run_cv: bool,
    compute_shap: bool,
    thresholds: dict,
    recent_fraction: float = 1.0,
) -> dict:
    """Run Horseshoe shrinkage + correlation clustering -> TWLGBM pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("  HORSESHOE PATH")
    logger.info("=" * 70)

    # Horseshoe shrinkage
    logger.info("  Applying Horseshoe prior shrinkage...")
    ds_hs, hs_feat_cols, horseshoe_weights = apply_horseshoe_shrinkage(
        dataset, feature_cols,
        recent_fraction=recent_fraction,
    )
    logger.info(f"  Features after horseshoe: {len(hs_feat_cols)}")

    # Correlation clustering
    logger.info("  Clustering correlated features...")
    ds_hs, hs_feat_cols = cluster_correlated_features(ds_hs, hs_feat_cols)
    logger.info(f"  Features after correlation filter: {len(hs_feat_cols)}")

    # Train classifiers
    logger.info("  Training classifiers (Horseshoe)...")
    classifiers, cv_results = train_all_classifiers(
        ds_hs, feature_cols=hs_feat_cols, run_cv=run_cv,
    )

    # Per-horizon feature selection
    logger.info("  Per-horizon feature selection (Horseshoe)...")
    selected_per_horizon: dict[str, list[str]] = {}
    for horizon in HORIZONS:
        try:
            clf = classifiers[horizon]
            importance = clf.model.feature_importance(
                importance_type="gain",
            ).astype(np.float32)
            imp_series = pd.Series(importance, index=hs_feat_cols)
            sel = select_features_by_importance(imp_series)
            selected_per_horizon[horizon] = sel
            logger.info(f"  HS {horizon}: selected {len(sel)} / {len(hs_feat_cols)} features")
        except Exception as exc:
            logger.warning(f"  HS {horizon} feature selection skipped: {exc}")
            selected_per_horizon[horizon] = hs_feat_cols

    del classifiers, cv_results
    gc.collect()

    # Retrain with selected features
    classifiers, cv_results = train_all_classifiers(
        ds_hs, feature_cols=selected_per_horizon, run_cv=run_cv,
    )

    # Generate forecast
    all_selected = sorted(set().union(*selected_per_horizon.values()))
    last_idx = ds_hs.index[-1]
    X_latest = ds_hs.loc[[last_idx], all_selected].astype(np.float32)
    gc.collect()

    forecasts = generate_forecast(
        classifiers=classifiers,
        X_latest=X_latest,
        thresholds=thresholds,
        compute_shap=compute_shap,
    )

    return {
        "method": "Horseshoe",
        "forecasts": forecasts,
        "cv_results": cv_results if cv_results else {},
        "classifiers": classifiers,
        "selected_per_horizon": selected_per_horizon,
        "horseshoe_n_survived": len(horseshoe_weights[horseshoe_weights >= 0.01]),
        "horseshoe_median_weight": float(horseshoe_weights[horseshoe_weights >= 0.01].median())
        if len(horseshoe_weights) > 0 else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Report & comparison output
# ──────────────────────────────────────────────────────────────────────────────

def _save_combined_report(
    pls_result: dict,
    hs_result: dict,
    thresholds: dict,
    dataset_info: dict,
    output_dir: Path,
    historical_dist: dict[str, list[float]] | None = None,
) -> tuple[Path, Path]:
    """Save separate JSON reports for PLS and Horseshoe forecasts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    generated_utc = datetime.now(timezone.utc).isoformat()

    hist_block = {
        horizon: {name: float(p) for name, p in zip(BUCKET_NAMES, dist)}
        for horizon, dist in (historical_dist or {}).items()
    }

    saved_paths = []
    for result in [pls_result, hs_result]:
        method = result["method"]
        method_report = {"cv_results": {}, "forecasts": {}}

        for horizon, cv in result["cv_results"].items():
            rps_arr = np.array(cv.fold_rps)
            ll_arr = np.array(cv.fold_logloss)
            method_report["cv_results"][horizon] = {
                "mean_rps": float(cv.mean_rps) if not np.isnan(cv.mean_rps) else None,
                "std_rps": float(np.std(rps_arr)) if len(rps_arr) > 0 else None,
                "mean_logloss": float(cv.mean_logloss) if not np.isnan(cv.mean_logloss) else None,
                "std_logloss": float(np.std(ll_arr)) if len(ll_arr) > 0 else None,
                "n_folds": len(cv.fold_rps),
                "fold_rps": [float(x) for x in cv.fold_rps],
                "fold_logloss": [float(x) for x in cv.fold_logloss],
            }

        for horizon, fc in result["forecasts"].items():
            fc_dict = {
                "bucket_probs": {
                    name: float(p) for name, p in zip(BUCKET_NAMES, fc.bucket_probs)
                },
                "exceedance_probs": {
                    f"{k:+.4f}": float(v) for k, v in fc.exceedance_probs.items()
                },
                "kelly": fc.kelly_fractions,
            }
            if fc.johnson_su_params:
                fc_dict["johnson_su_params"] = fc.johnson_su_params
            if fc.shap_directional is not None:
                fc_dict["top_shap_features"] = {
                    k: float(v)
                    for k, v in fc.shap_directional.head(10).items()
                }
            method_report["forecasts"][horizon] = fc_dict

        if "pls_info" in result:
            method_report["pls_info"] = result["pls_info"]
        if "horseshoe_n_survived" in result:
            method_report["horseshoe_n_survived"] = result["horseshoe_n_survived"]
            method_report["horseshoe_median_weight"] = result["horseshoe_median_weight"]

        report = {
            "generated_utc": generated_utc,
            "dataset": dataset_info,
            "bucket_thresholds": thresholds,
            "historical_bucket_distribution": hist_block,
            "method": method,
            **method_report,
        }

        # e.g. forecast_pls_20260320_120000.json or forecast_horseshoe_20260320_120000.json
        filename = f"forecast_{method}_{timestamp}.json"
        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"{method.upper()} report saved to {filepath}")
        saved_paths.append(filepath)

    return tuple(saved_paths)


def _print_combined_summary(
    pls_result: dict,
    hs_result: dict,
    thresholds: dict,
    historical_dist: dict[str, list[float]] | None = None,
):
    """Print side-by-side comparison of PLS and Horseshoe forecasts."""
    print("\n" + "=" * 90)
    print("  TWLGBM 5-Bucket Return Forecast  --  PLS vs Horseshoe Comparison")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 90)

    for horizon in HORIZONS:
        fc_pls = pls_result["forecasts"][horizon]
        fc_hs = hs_result["forecasts"][horizon]
        th = thresholds[horizon]
        weeks = HORIZON_WEEKS[horizon]
        bounds = th["boundaries"]

        print(f"\n  --- {horizon.upper()} ({weeks}-week) horizon ---")
        print(f"  Boundaries: {[f'{b:+.2%}' for b in bounds]}")
        print()

        # Header
        has_hist = historical_dist and horizon in historical_dist
        if has_hist:
            print(f"  {'Bucket':<8} {'PLS':>8} {'Horseshoe':>10} {'Historical':>10} {'Return Range':>30}")
            print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*30}")
        else:
            print(f"  {'Bucket':<8} {'PLS':>8} {'Horseshoe':>10} {'Return Range':>30}")
            print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*30}")

        ranges = [
            f"r <= {bounds[0]:+.2%}",
            f"{bounds[0]:+.2%} < r <= {bounds[1]:+.2%}",
            f"{bounds[1]:+.2%} < r <= {bounds[2]:+.2%}",
            f"{bounds[2]:+.2%} < r <= {bounds[3]:+.2%}",
            f"r > {bounds[3]:+.2%}",
        ]

        for i, name in enumerate(BUCKET_NAMES):
            pls_prob = fc_pls.bucket_probs[i]
            hs_prob = fc_hs.bucket_probs[i]
            if has_hist:
                hist_pct = historical_dist[horizon][i]
                print(f"  {name:<8} {pls_prob:>7.1%}  {hs_prob:>9.1%}  {hist_pct:>9.1%}  {ranges[i]:>30}")
            else:
                print(f"  {name:<8} {pls_prob:>7.1%}  {hs_prob:>9.1%}  {ranges[i]:>30}")

        # Kelly comparison
        kelly_pls = fc_pls.kelly_fractions
        kelly_hs = fc_hs.kelly_fractions
        print()
        print(f"  {'':>20} {'PLS':>10} {'Horseshoe':>10}")
        print(f"  {'E[return]':>20} {kelly_pls['expected_return']:>+10.4f} {kelly_hs['expected_return']:>+10.4f}")
        print(f"  {'Kelly half':>20} {kelly_pls['kelly_half']:>+10.4f} {kelly_hs['kelly_half']:>+10.4f}")

        # CV comparison
        pls_cv = pls_result["cv_results"].get(horizon)
        hs_cv = hs_result["cv_results"].get(horizon)
        if pls_cv and hs_cv:
            pls_rps = pls_cv.mean_rps if not np.isnan(pls_cv.mean_rps) else None
            hs_rps = hs_cv.mean_rps if not np.isnan(hs_cv.mean_rps) else None
            if pls_rps is not None and hs_rps is not None:
                print(f"  {'CV RPS':>20} {pls_rps:>10.4f} {hs_rps:>10.4f}")

        # SHAP top features (abbreviated)
        if fc_pls.shap_directional is not None or fc_hs.shap_directional is not None:
            print(f"\n  Top SHAP features:")
            print(f"  {'PLS':>45}  {'Horseshoe':>45}")
            pls_shap = fc_pls.shap_directional.head(3) if fc_pls.shap_directional is not None else pd.Series(dtype=float)
            hs_shap = fc_hs.shap_directional.head(3) if fc_hs.shap_directional is not None else pd.Series(dtype=float)

            max_rows = max(len(pls_shap), len(hs_shap))
            for j in range(max_rows):
                pls_str = ""
                hs_str = ""
                if j < len(pls_shap):
                    feat = pls_shap.index[j]
                    val = pls_shap.iloc[j]
                    pls_str = f"{feat[:30]:>30s}: {val:+.4f}"
                if j < len(hs_shap):
                    feat = hs_shap.index[j]
                    val = hs_shap.iloc[j]
                    hs_str = f"{feat[:30]:>30s}: {val:+.4f}"
                print(f"  {pls_str:>45}  {hs_str:>45}")

    print("\n" + "=" * 90)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(
    start_date: str | None = None,
    end_date: str | None = None,
    asset: str = "BTC",
    run_cv: bool = True,
    compute_shap: bool = True,
    metrics_csv: str = "btc_glassnode_metrics.csv",
    anchor_day: str = DEFAULT_ANCHOR_DAY,
    recent_fraction: float = 1.0,
) -> dict:
    """
    Full TWLGBM pipeline with both PLS and Horseshoe regularization paths.

    Pipeline:
      1. Build dataset (shared)
      2. Engineer features (shared)
      3a. PLS -> train -> feature select -> retrain -> forecast
      3b. Horseshoe -> correlation filter -> train -> feature select -> retrain -> forecast
      4. Combined output

    Parameters
    ----------
    anchor_day : str
        Day of week for data alignment and return computation:
        "thursday" (default), "friday", or "monday".
    recent_fraction : float
        Fraction of data (most recent) to use for PLS/Horseshoe fitting.
        Default 1.0 uses all data. E.g. 0.5 fits on the most recent half.

    Returns dict with pls_result, hs_result, thresholds, dataset.
    """
    if end_date is None:
        end_date = _most_recent_anchor_day(anchor_day=anchor_day)
    if start_date is None:
        start_date = _default_start_date(weeks_back=210, anchor_day=anchor_day)

    logger.info(f"TWLGBM Pipeline (PLS + Horseshoe): {start_date} -> {end_date}, asset={asset}, anchor={anchor_day}")

    # ── Step 1: Build dataset (shared) ────────────────────────────────────
    logger.info("\n[Step 1] Building dataset...")
    dataset, thresholds, col_to_category = build_dataset(
        start_date=start_date,
        end_date=end_date,
        asset=asset,
        metrics_csv=metrics_csv,
        calibrate_from_data=True,
        anchor_day=anchor_day,
    )

    # ── Step 2: Feature engineering (shared) ──────────────────────────────
    logger.info("\n[Step 2] Engineering features...")
    dataset = engineer_features(dataset)
    dataset = clean_features(dataset)
    dataset = _downcast_dataframe(dataset)

    feature_cols = get_feature_columns(dataset)
    logger.info(f"  Total features after engineering: {len(feature_cols)}")

    # ── Step 3a: PLS path ─────────────────────────────────────────────────
    logger.info("\n[Step 3a] Running PLS regularization path...")
    pls_result = _run_pls_path(
        dataset, feature_cols, col_to_category,
        run_cv=run_cv, compute_shap=compute_shap, thresholds=thresholds,
        recent_fraction=recent_fraction,
    )

    gc.collect()

    # ── Step 3b: Horseshoe path ───────────────────────────────────────────
    logger.info("\n[Step 3b] Running Horseshoe regularization path...")
    hs_result = _run_horseshoe_path(
        dataset, feature_cols, col_to_category,
        run_cv=run_cv, compute_shap=compute_shap, thresholds=thresholds,
        recent_fraction=recent_fraction,
    )

    gc.collect()

    # ── Step 4: Combined output ───────────────────────────────────────────
    logger.info("\n[Step 4] Generating combined report...")

    dataset_info = {
        "start_date": start_date,
        "end_date": end_date,
        "asset": asset,
        "n_weeks": len(dataset),
        "n_features_engineered": len(feature_cols),
        "forecast_date": dataset.iloc[-1]["date"],
        "pls": {
            "n_features_per_horizon": {
                h: len(pls_result["selected_per_horizon"][h]) for h in HORIZONS
            },
        },
        "horseshoe": {
            "n_features_per_horizon": {
                h: len(hs_result["selected_per_horizon"][h]) for h in HORIZONS
            },
            "n_survived_shrinkage": hs_result.get("horseshoe_n_survived"),
            "median_weight": hs_result.get("horseshoe_median_weight"),
        },
    }

    historical_dist = _compute_historical_bucket_dist(dataset, thresholds)

    pls_path, hs_path = _save_combined_report(
        pls_result=pls_result,
        hs_result=hs_result,
        thresholds=thresholds,
        dataset_info=dataset_info,
        output_dir=OUTPUT_DIR,
        historical_dist=historical_dist,
    )

    _print_combined_summary(pls_result, hs_result, thresholds, historical_dist)

    return {
        "pls_result": pls_result,
        "hs_result": hs_result,
        "thresholds": thresholds,
        "dataset": dataset,
        "pls_report_path": pls_path,
        "hs_report_path": hs_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="TWLGBM 5-Bucket Return Classifier (PLS + Horseshoe)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date YYYY-MM-DD (default: 210 weeks before latest anchor day)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date YYYY-MM-DD (default: most recent anchor day)",
    )
    parser.add_argument("--asset", default="BTC", help="Asset symbol (default: BTC)")
    parser.add_argument("--no-cv", action="store_true", help="Skip walk-forward CV")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP attribution")
    parser.add_argument(
        "--metrics-csv", default="btc_glassnode_metrics.csv",
        help="Path to Glassnode metrics CSV",
    )
    parser.add_argument(
        "--anchor-day", default = 'thursday',   #default=DEFAULT_ANCHOR_DAY,
        choices=list(ANCHOR_DAY_WEEKDAY.keys()),
        help=(
            "Anchor day for weekly data alignment and return computation. "
            "'thursday' (default, run Fri 3AM UTC), "
            "'friday' (run Sat 4AM UTC), "
            "'monday' (run Tue 4AM UTC)"
        ),
    )
    parser.add_argument(
        "--recent-fraction", type=float, default=0.5,
        help=(
            "Fraction of data (most recent) to use for PLS/Horseshoe fitting. "
            "Default 1.0 uses all data. E.g. 0.5 fits on the most recent half."
        ),
    )

    args = parser.parse_args()

    result = run(
        start_date=args.start_date,
        end_date=args.end_date,
        asset=args.asset,
        run_cv=not args.no_cv,
        compute_shap=not args.no_shap,
        metrics_csv=args.metrics_csv,
        anchor_day=args.anchor_day,
        recent_fraction=args.recent_fraction,
    )

    logger.info(f"\nDone. Reports: {result['pls_report_path']}, {result['hs_report_path']}")


if __name__ == "__main__":
    main()
