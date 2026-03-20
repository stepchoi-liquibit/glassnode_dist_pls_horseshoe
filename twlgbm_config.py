"""
TWLGBM Configuration
====================
Bucket boundaries, LightGBM hyperparameters, FRED series, and constants
for the 5-bucket multiclass return classification system.

Supports both PLS and Horseshoe regularization paths.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Bucket boundaries (log-return thresholds)
# B1: big down  |  B2: down  |  B3: flat  |  B4: up  |  B5: big up
# ──────────────────────────────────────────────────────────────────────────────

BUCKET_THRESHOLDS: dict[str, dict] = {
    "1w": {
        "boundaries": [-0.07, -0.02, 0.02, 0.07],
    },
    "4w": {
        "boundaries": [-0.14, -0.04, 0.04, 0.14],
    },
    "13w": {
        "boundaries": [-0.28, -0.08, 0.08, 0.28],
    },
}

HORIZONS = ["1w", "4w", "13w"]
HORIZON_WEEKS = {"1w": 1, "4w": 4, "13w": 13}
NUM_BUCKETS = 5
BUCKET_NAMES = ["B1", "B2", "B3", "B4", "B5"]

# ──────────────────────────────────────────────────────────────────────────────
# Anchor day configuration
# ──────────────────────────────────────────────────────────────────────────────
# The anchor day determines which day-of-week is used for weekly data alignment,
# forward return computation, and data cutoff.
#
# "thursday" — Thursday-to-Thursday returns, run Friday 3 AM UTC (default)
# "friday"  — Friday-to-Friday returns, run Saturday 4 AM UTC
# "monday"  — Monday-to-Monday returns, run Tuesday 4 AM UTC

ANCHOR_DAY_WEEKDAY = {
    "thursday": 3,   # datetime.weekday() == 3
    "friday": 4,     # datetime.weekday() == 4
    "monday": 0,     # datetime.weekday() == 0
}

# pandas date_range freq offset aliases for each anchor day
ANCHOR_DAY_FREQ = {
    "thursday": "W-THU",
    "friday": "W-FRI",
    "monday": "W-MON",
}

DEFAULT_ANCHOR_DAY = "thursday"

# ──────────────────────────────────────────────────────────────────────────────
# LightGBM hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

LGBM_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": NUM_BUCKETS,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 8,
    "max_depth": 3,
    "learning_rate": 0.01,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "seed": 37,
    "verbosity": -1,
    "force_col_wise": True,
    "feature_pre_filter": False,
    "monotone_constraints": [],
}

NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 100

# ──────────────────────────────────────────────────────────────────────────────
# CV settings
# ──────────────────────────────────────────────────────────────────────────────

CV_N_FOLDS = 5
CV_N_REPEATS = 3
CV_PURGE_WEEKS = 13
MIN_TRAIN_WEEKS = 52

# ──────────────────────────────────────────────────────────────────────────────
# FRED macro series
# ──────────────────────────────────────────────────────────────────────────────

FRED_API_KEY = "ac6f670ee74102e4f03870510072acb9"

FRED_SERIES_MACRO = [
    "DFF",
    "DGS1MO",
    "DGS3MO",
    "DGS1",
    "DGS2",
    "DGS5",
    "DGS10",
    "AAA",
    "MORTGAGE30US",
    "IRLTLT01JPM156N",
]

FRED_SERIES_RISK = [
    "SP500",
    "VXVCLS",
    "BAMLH0A0HYM2",
    "DCOILWTICO",
    "OVXCLS",
    "GVZCLS",
    "NFCI",
    "DTWEXBGS",
    "DEXKOUS",
]

FRED_SERIES_LIQUIDITY = [
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
]

FRED_LOG_DIFF_SERIES = ["SP500", "DCOILWTICO"]

# ──────────────────────────────────────────────────────────────────────────────
# Glassnode daily -> weekly transformation keywords
# ──────────────────────────────────────────────────────────────────────────────

SUM_KEYWORDS = [
    'volume', '_sum', 'flow', 'issued', 'mined', 'liquidat', 'revenue',
    'spent_volume', 'inscriptions_count_sum', 'inscriptions_size_sum',
    'runes_count_sum',
]

RATIO_KEYWORDS = [
    'relative', 'ratio', 'rate', 'percent', 'mvrv', 'sopr', 'adoption',
    'skew', 'index', 'score', 'dominance', 'multiple', 'z_score',
    'profit_relative', 'fee_share', 'count_share', 'size_share',
    'drawdown', 'inflation',
]

# ──────────────────────────────────────────────────────────────────────────────
# Cross-asset and stablecoin Glassnode endpoints
# ──────────────────────────────────────────────────────────────────────────────

CROSS_ASSETS = ["ETH", "SOL"]

CROSS_ASSET_ENDPOINTS = [
    "/v1/metrics/market/price_usd_close",
    "/v1/metrics/derivatives/futures_volume_daily_sum",
    "/v1/metrics/derivatives/futures_open_interest_sum",
    "/v1/metrics/derivatives/futures_funding_rate_perpetual",
    "/v1/metrics/derivatives/options_volume_put_call_ratio",
    "/v1/metrics/derivatives/options_open_interest_put_call_ratio",
]

STABLECOIN_ASSETS = ["USDT", "USDC"]

STABLECOIN_ENDPOINTS = [
    "/v1/metrics/supply/current",
    "/v1/metrics/supply/issued",
]

# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────

TEMPERATURE_INIT = 1.0
TEMPERATURE_BOUNDS = (0.1, 10.0)
PROB_EPSILON = 1e-5
CALIBRATED_HORIZONS = {"1w"}

# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

ROLLING_WINDOWS = [4, 13]
MAX_NAN_FRACTION = 0.30
NAN_FILL_THRESHOLD = 0.10
MIN_FEATURE_IMPORTANCE_QUANTILE = 0.25
MAX_FEATURES = 250

# ──────────────────────────────────────────────────────────────────────────────
# PLS regression (per-category dimensionality reduction before LGBM)
# ──────────────────────────────────────────────────────────────────────────────

PLS_MAX_COMPONENTS = 10
PLS_CV_FOLDS = 3
PLS_MIN_FEATURES = 2

# ──────────────────────────────────────────────────────────────────────────────
# Horseshoe prior shrinkage (applied to features before LGBM)
# ──────────────────────────────────────────────────────────────────────────────

HORSESHOE_EXPECTED_SPARSITY = 0.10
HORSESHOE_WEIGHT_FLOOR = 0.01

# ──────────────────────────────────────────────────────────────────────────────
# Kelly sizing
# ──────────────────────────────────────────────────────────────────────────────

KELLY_TAIL_MULTIPLIER = 1.5
