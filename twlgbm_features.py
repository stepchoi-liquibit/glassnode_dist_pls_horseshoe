"""
TWLGBM Feature Engineering & Selection
=======================================
Rolling statistics, z-scores, momentum features, and
importance-based feature selection using SHAP.

Supports both PLS and Horseshoe regularization paths.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score

from twlgbm_config import (
    ROLLING_WINDOWS,
    MAX_NAN_FRACTION,
    NAN_FILL_THRESHOLD,
    MIN_FEATURE_IMPORTANCE_QUANTILE,
    MAX_FEATURES,
    HORIZON_WEEKS,
    PLS_MAX_COMPONENTS,
    PLS_CV_FOLDS,
    PLS_MIN_FEATURES,
    HORSESHOE_EXPECTED_SPARSITY,
    HORSESHOE_WEIGHT_FLOOR,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Feature column identification
# ──────────────────────────────────────────────────────────────────────────────

_META_COLS = {"date", "price"}
_LABEL_PREFIXES = ("ret_", "label_")


def _is_feature_col(col: str) -> bool:
    """Return True if column is a raw feature (not metadata or label)."""
    if col in _META_COLS:
        return False
    if col.startswith("_"):
        return False
    for prefix in _LABEL_PREFIXES:
        if col.startswith(prefix):
            return False
    return True


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature column names from a dataset DataFrame."""
    return [c for c in df.columns if _is_feature_col(c)]


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    batch_size: int = 50,
) -> pd.DataFrame:
    """
    Add derived features:
      - Rolling means and stds over ROLLING_WINDOWS
      - Z-scores (current value vs rolling mean/std)
      - Week-over-week percentage changes

    Pre-filters columns (NaN fraction, zero variance) before computing
    derived features, and processes in batches to limit peak RAM.

    All computations are strictly backward-looking (no future leakage).
    """
    df = df.copy()
    raw_features = get_feature_columns(df)

    logger.info(f"Engineering features from {len(raw_features)} raw columns")

    # ── Pre-filter: drop columns that are mostly NaN or constant ──────────
    eligible = []
    for col in raw_features:
        series = df[col]
        if series.isna().mean() > MAX_NAN_FRACTION:
            continue
        if series.dropna().nunique() <= 1:
            continue
        eligible.append(col)

    logger.info(
        f"  Pre-filter: {len(raw_features)} -> {len(eligible)} columns "
        f"eligible for derived features"
    )

    # ── Compute derived features in batches ───────────────────────────────
    n_new_total = 0
    for batch_start in range(0, len(eligible), batch_size):
        batch_cols = eligible[batch_start : batch_start + batch_size]
        batch_dict = {}

        for col in batch_cols:
            series = df[col]

            for window in ROLLING_WINDOWS:
                roll_mean = series.rolling(
                    window=window, min_periods=max(1, window // 2),
                ).mean()
                batch_dict[f"{col}_ma{window}"] = roll_mean

                roll_std = series.rolling(
                    window=window, min_periods=max(2, window // 2),
                ).std()
                batch_dict[f"{col}_std{window}"] = roll_std

                zscore = (series - roll_mean) / roll_std.replace(0, np.nan)
                batch_dict[f"{col}_z{window}"] = zscore

            batch_dict[f"{col}_chg1w"] = series.pct_change(periods=1, fill_method=None)

        batch_df = pd.DataFrame(batch_dict, index=df.index).astype(np.float32)
        batch_df = batch_df.dropna(axis=1, how="all")
        df = pd.concat([df, batch_df], axis=1)
        n_new_total += len(batch_dict)
        del batch_dict, batch_df

    logger.info(f"  Added {n_new_total} derived features")

    # ── Lagged bucket labels (regime persistence signals) ─────────────────
    n_label_feats = 0
    for horizon, weeks in HORIZON_WEEKS.items():
        label_col = f"label_{horizon}"
        if label_col not in df.columns:
            continue
        for lag_mult in [1, 2, 3]:
            lag = weeks * lag_mult
            feat_name = f"prev_bucket_{horizon}_lag{lag}"
            df[feat_name] = df[label_col].shift(lag).astype(np.float32)
            n_label_feats += 1
    logger.info(f"  Added {n_label_feats} lagged bucket label features")

    # ── Cross-horizon realized return features ────────────────────────────
    n_cross = 0
    for horizon, weeks in HORIZON_WEEKS.items():
        ret_col = f"ret_{horizon}"
        if ret_col not in df.columns:
            continue
        realized = df[ret_col].shift(weeks)
        for window in [4, 13, 26]:
            feat_name = f"realized_{horizon}_ma{window}"
            df[feat_name] = realized.rolling(
                window, min_periods=max(1, window // 2),
            ).mean().astype(np.float32)
            n_cross += 1
        df[f"realized_{horizon}_vol13"] = realized.rolling(
            13, min_periods=4,
        ).std().astype(np.float32)
        n_cross += 1
    logger.info(f"  Added {n_cross} cross-horizon return features")

    # ── Interaction features (explicit cross-terms) ───────────────────────
    if "ret_1w" in df.columns:
        df["_price_momentum"] = df["ret_1w"].shift(1).astype(np.float32)
    n_interactions = 0
    interaction_pairs = [
        ("_price_momentum", "VXVCLS", "price_x_vix"),
        ("_price_momentum", "BAMLH0A0HYM2", "price_x_hyspread"),
        ("_price_momentum", "net_liquidity", "price_x_netliq"),
        ("_price_momentum", "NFCI", "price_x_nfci"),
        ("VXVCLS", "BAMLH0A0HYM2", "vix_x_hyspread"),
    ]
    for base, cross, name in interaction_pairs:
        base_col = f"{base}_z13" if f"{base}_z13" in df.columns else base
        cross_col = f"{cross}_z13" if f"{cross}_z13" in df.columns else cross
        if base_col in df.columns and cross_col in df.columns:
            df[f"interact_{name}"] = (
                df[base_col] * df[cross_col]
            ).astype(np.float32)
            n_interactions += 1
    if "_price_momentum" in df.columns:
        df = df.drop(columns=["_price_momentum"])
    logger.info(f"  Added {n_interactions} interaction features")

    logger.info(f"  Total features after engineering: {len(get_feature_columns(df))}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature cleaning
# ──────────────────────────────────────────────────────────────────────────────

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with excessive NaNs, constant columns,
    and columns with inf values.
    """
    df = df.copy()

    if df.columns.duplicated().any():
        n_dupes = df.columns.duplicated(keep="last").sum()
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        logger.info(f"  Removed {n_dupes} duplicate columns")

    feature_cols = get_feature_columns(df)
    drop_cols = []
    keep_cols = []

    for col in feature_cols:
        series = df[col]
        if series.isna().mean() > MAX_NAN_FRACTION:
            drop_cols.append(col)
            continue
        if series.dropna().nunique() <= 1:
            drop_cols.append(col)
            continue
        keep_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"  Dropped {len(drop_cols)} low-quality feature columns")

    if keep_cols:
        df[keep_cols] = df[keep_cols].replace([np.inf, -np.inf], np.nan)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PLS regression (per-category dimensionality reduction)
# ──────────────────────────────────────────────────────────────────────────────

def _build_feature_category_map(
    feature_cols: list[str],
    col_to_category: dict[str, str],
) -> dict[str, str]:
    """
    Map each feature column (including derived features like rolling stats)
    to its source category.
    """
    result: dict[str, str] = {}
    base_cols = sorted(col_to_category.keys(), key=len, reverse=True)

    for feat in feature_cols:
        if feat in col_to_category:
            result[feat] = col_to_category[feat]
            continue
        for base in base_cols:
            if feat.startswith(base + "_"):
                result[feat] = col_to_category[base]
                break

    return result


def apply_pls_by_category(
    df: pd.DataFrame,
    feature_cols: list[str],
    col_to_category: dict[str, str],
    max_components: int = PLS_MAX_COMPONENTS,
    cv_folds: int = PLS_CV_FOLDS,
    min_features: int = PLS_MIN_FEATURES,
    recent_fraction: float = 1.0,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Replace raw features with PLS scores, one PLS per feature category.

    For each category with enough features:
      1. Fill NaN with column medians, standardize features
      2. Cross-validate to find optimal n_components (R^2 scoring)
      3. Fit PLS regression against [ret_1w, ret_4w, ret_13w]
      4. Transform all rows to PLS scores

    Features not belonging to any category (interaction terms, lagged
    bucket labels, cross-horizon return features) are kept as-is.

    Parameters
    ----------
    recent_fraction : float
        Fraction of data (most recent) to use for PLS fitting (CV + final fit).
        Default 1.0 uses all data. E.g. 0.5 fits PLS on the most recent half
        but still transforms all rows.
    """
    df = df.copy()

    feat_to_cat = _build_feature_category_map(feature_cols, col_to_category)

    groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for col in feature_cols:
        cat = feat_to_cat.get(col)
        if cat:
            groups.setdefault(cat, []).append(col)
        else:
            ungrouped.append(col)

    ret_cols = [f"ret_{h}" for h in HORIZON_WEEKS if f"ret_{h}" in df.columns]
    if not ret_cols:
        logger.warning("PLS: no return columns found, skipping")
        return df, feature_cols, {}

    pls_score_cols: list[str] = []
    pls_info: dict[str, dict] = {}

    if recent_fraction < 1.0:
        logger.info(f"  PLS: using most recent {recent_fraction:.0%} of data for fitting")

    for cat in sorted(groups.keys()):
        cols = groups[cat]
        if len(cols) < min_features:
            logger.info(f"  PLS skip {cat}: only {len(cols)} feature(s)")
            ungrouped.extend(cols)
            continue

        X_all = df[cols].values.astype(np.float64)
        col_medians = np.nanmedian(X_all, axis=0)
        nan_mask = np.isnan(X_all)
        if nan_mask.any():
            X_all[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        valid_mask = df[ret_cols].notna().all(axis=1).values

        # Apply recent_fraction: restrict fitting window to most recent portion
        if recent_fraction < 1.0:
            n_total = int(valid_mask.sum())
            n_recent = max(cv_folds * 5, int(n_total * recent_fraction))
            # Build a mask that is True only for the last n_recent valid rows
            valid_indices = np.where(valid_mask)[0]
            fit_indices = set(valid_indices[-n_recent:])
            fit_mask = np.array([i in fit_indices for i in range(len(valid_mask))])
        else:
            fit_mask = valid_mask

        X_fit = X_scaled[fit_mask]
        Y_fit = df.loc[fit_mask, ret_cols].values.astype(np.float64)

        if X_fit.shape[0] < cv_folds * 5:
            logger.info(f"  PLS skip {cat}: too few valid rows ({X_fit.shape[0]})")
            ungrouped.extend(cols)
            continue

        max_n = min(len(cols), max_components, X_fit.shape[0] // cv_folds)
        if max_n < 1:
            ungrouped.extend(cols)
            continue

        best_score = -np.inf
        best_n = 1
        min_n = max(1, max_n // 2 - 1)

        for n_comp in range(min_n, max_n + 1):
            try:
                pls = PLSRegression(n_components=n_comp)
                scores = cross_val_score(
                    pls, X_fit, Y_fit,
                    cv=KFold(cv_folds, shuffle=True, random_state=37),
                    scoring="r2",
                )
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_n = n_comp
            except Exception:
                continue

        pls = PLSRegression(n_components=best_n)
        pls.fit(X_fit, Y_fit)
        # Transform ALL rows (not just the fit window)
        X_scores = pls.transform(X_scaled)

        score_names = [f"{cat}_pls{i + 1}" for i in range(best_n)]
        for i, name in enumerate(score_names):
            df[name] = X_scores[:, i].astype(np.float32)

        pls_score_cols.extend(score_names)
        pls_info[cat] = {
            "n_components": best_n,
            "n_features": len(cols),
            "cv_r2": float(best_score),
        }

        logger.info(
            f"  PLS {cat}: {len(cols)} features -> {best_n} components "
            f"(CV R^2={best_score:.4f})"
        )

    cols_to_drop = [c for c in feature_cols if c not in ungrouped]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    final_features = pls_score_cols + ungrouped

    logger.info(
        f"  PLS complete: {len(feature_cols)} features -> "
        f"{len(pls_score_cols)} PLS scores + {len(ungrouped)} ungrouped = "
        f"{len(final_features)} total"
    )

    return df, final_features, pls_info


# ──────────────────────────────────────────────────────────────────────────────
# Horseshoe prior shrinkage (Carvalho, Polson & Scott 2010)
# ──────────────────────────────────────────────────────────────────────────────

def apply_horseshoe_shrinkage(
    df: pd.DataFrame,
    feature_cols: list[str],
    expected_sparsity: float = HORSESHOE_EXPECTED_SPARSITY,
    weight_floor: float = HORSESHOE_WEIGHT_FLOOR,
    recent_fraction: float = 1.0,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """
    Apply Horseshoe prior shrinkage to all feature columns before LGBM.

    For each feature, computes a marginal signal score (max absolute
    correlation with any target horizon label, converted to a z-statistic).
    The empirical Bayes horseshoe then assigns per-feature shrinkage weights:

        w_j = lambda_j^2 * tau^2 / (1 + lambda_j^2 * tau^2)

    where lambda_j^2 is proportional to z_j^2 (local scale) and tau is
    the global shrinkage parameter controlled by ``expected_sparsity``.

    Features with strong signal get w_j ~ 1 (preserved); noisy features
    get w_j ~ 0 (shrunk toward zero).  Features below ``weight_floor``
    are dropped entirely.

    Parameters
    ----------
    recent_fraction : float
        Fraction of data (most recent) to use for computing signal scores.
        Default 1.0 uses all data. E.g. 0.5 computes correlations on the
        most recent half only, but applies shrinkage decisions to all rows.
    """
    df = df.copy()
    label_cols = [f"label_{h}" for h in HORIZON_WEEKS if f"label_{h}" in df.columns]

    if not label_cols or len(feature_cols) == 0:
        logger.warning("Horseshoe: no label columns or features, skipping")
        return df, feature_cols, pd.Series(dtype=np.float64)

    X = df[feature_cols].values.astype(np.float64)
    n_rows, p = X.shape

    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X_std = (X - means) / stds

    # Apply recent_fraction: restrict signal scoring to most recent portion
    if recent_fraction < 1.0:
        n_recent = max(30, int(n_rows * recent_fraction))
        fit_start = n_rows - n_recent
        logger.info(
            f"  Horseshoe: using most recent {recent_fraction:.0%} of data "
            f"({n_recent}/{n_rows} rows) for signal scoring"
        )
    else:
        fit_start = 0

    max_abs_z = np.zeros(p, dtype=np.float64)

    for lcol in label_cols:
        y = df[lcol].values.astype(np.float64)
        # Restrict to recent window
        y_window = y[fit_start:]
        valid = ~np.isnan(y_window)
        n_valid = valid.sum()
        if n_valid < 30:
            continue

        y_clean = y_window[valid]
        y_mean = y_clean.mean()
        y_std = y_clean.std()
        if y_std < 1e-10:
            continue
        y_normed = (y_clean - y_mean) / y_std

        X_valid = X_std[fit_start:][valid]
        abs_corr = np.abs(X_valid.T @ y_normed) / n_valid
        z_scores = abs_corr * np.sqrt(n_valid)
        max_abs_z = np.maximum(max_abs_z, z_scores)

    p0 = max(1, int(p * expected_sparsity))
    tau0 = p0 / max(1, p - p0)

    z_sq_tau_sq = (max_abs_z ** 2) * (tau0 ** 2)
    weights = z_sq_tau_sq / (1.0 + z_sq_tau_sq)

    weight_series = pd.Series(weights, index=feature_cols)

    surviving_mask = weights >= weight_floor
    surviving_cols = [feature_cols[i] for i in range(p) if surviving_mask[i]]
    dropped_cols = [feature_cols[i] for i in range(p) if not surviving_mask[i]]

    if dropped_cols:
        df = df.drop(columns=dropped_cols)

    n_dropped = len(dropped_cols)
    n_survived = len(surviving_cols)
    median_w = float(np.median(weights[surviving_mask])) if n_survived > 0 else 0.0

    logger.info(
        f"  Horseshoe shrinkage (tau0={tau0:.4f}, sparsity={expected_sparsity:.0%}): "
        f"{p} -> {n_survived} features "
        f"(dropped {n_dropped} below weight floor {weight_floor})"
    )
    logger.info(
        f"  Horseshoe weight stats -- "
        f"median={median_w:.3f}, "
        f"min={float(weights[surviving_mask].min()) if n_survived else 0:.3f}, "
        f"max={float(weights[surviving_mask].max()) if n_survived else 0:.3f}"
    )

    return df, surviving_cols, weight_series


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical feature clustering (correlation-based)
# ──────────────────────────────────────────────────────────────────────────────

def cluster_correlated_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    corr_threshold: float = 0.70,
    max_sample_rows: int = 200,
    chunk_size: int = 1000,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fast correlation-based feature reduction via chunked greedy elimination.

    Sorts features by variance (descending) and greedily drops any feature
    whose absolute correlation with an already-kept feature >= corr_threshold.
    """
    if len(feature_cols) <= 1:
        return df, feature_cols

    n_features = len(feature_cols)
    logger.info(f"  Fast correlation filter on {n_features} features...")

    X = df[feature_cols].values.astype(np.float32)
    n_rows, n_cols = X.shape

    col_medians = np.nanmedian(X, axis=0)
    nan_inds = np.where(np.isnan(X))
    if len(nan_inds[0]) > 0:
        X[nan_inds] = col_medians[nan_inds[1]]

    n_sample = min(n_rows, max_sample_rows)
    X_sub = X[-n_sample:]

    means = X_sub.mean(axis=0)
    stds = X_sub.std(axis=0)
    stds[stds < 1e-8] = 1.0
    X_norm = (X_sub - means) / stds

    variances = stds ** 2
    order = np.argsort(-variances)

    kept_col_indices: list[int] = []
    kept_vectors = np.empty((n_sample, 0), dtype=np.float32)

    for chunk_start in range(0, n_cols, chunk_size):
        chunk_indices = order[chunk_start : chunk_start + chunk_size]
        chunk_vecs = X_norm[:, chunk_indices]

        if kept_vectors.shape[1] > 0:
            cross_corr = np.abs(chunk_vecs.T @ kept_vectors) / n_sample
            candidate_mask = cross_corr.max(axis=1) < corr_threshold
        else:
            candidate_mask = np.ones(len(chunk_indices), dtype=bool)

        candidate_positions = np.where(candidate_mask)[0]
        if len(candidate_positions) > 1:
            cand_vecs = chunk_vecs[:, candidate_positions]
            intra_corr = np.abs(cand_vecs.T @ cand_vecs) / n_sample
            np.fill_diagonal(intra_corr, 0.0)

            intra_keep = np.ones(len(candidate_positions), dtype=bool)
            for i in range(len(candidate_positions)):
                if not intra_keep[i]:
                    continue
                drop_mask = intra_corr[i] >= corr_threshold
                drop_mask[:i + 1] = False
                intra_keep[drop_mask] = False

            candidate_positions = candidate_positions[intra_keep]

        new_indices = chunk_indices[candidate_positions]
        if len(new_indices) > 0:
            kept_col_indices.extend(new_indices.tolist())
            kept_vectors = np.hstack([kept_vectors, X_norm[:, new_indices]])

    surviving = [feature_cols[i] for i in sorted(kept_col_indices)]

    logger.info(
        f"  Correlation filter (|corr|>={corr_threshold}): "
        f"{n_features} -> {len(surviving)} features"
    )
    return df, surviving


# ──────────────────────────────────────────────────────────────────────────────
# Feature selection (importance-based, post-training)
# ──────────────────────────────────────────────────────────────────────────────

def select_features_by_importance(
    feature_importances: pd.Series,
    min_quantile: float = MIN_FEATURE_IMPORTANCE_QUANTILE,
    max_features: int = MAX_FEATURES,
) -> list[str]:
    """
    Select features whose importance is above the min_quantile threshold,
    then enforce a hard cap of max_features (keeping the top ones).
    """
    threshold = feature_importances.quantile(min_quantile)
    selected = feature_importances[feature_importances >= threshold]

    if len(selected) > max_features:
        selected = selected.nlargest(max_features)

    selected = selected.index.tolist()
    logger.info(
        f"  Feature selection: {len(selected)}/{len(feature_importances)} features "
        f"(quantile={min_quantile:.0%}, cap={max_features})"
    )
    return selected


def prepare_train_data(
    df: pd.DataFrame,
    horizon: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X, y, sample_weights for a given horizon.

    Only rows with valid labels are included. Sample weights use
    linear time decay (more recent observations weighted higher).
    """
    label_col = f"label_{horizon}"
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    valid_mask = df[label_col].notna()
    df_valid = df.loc[valid_mask].copy()

    X = df_valid[feature_cols].astype(np.float32)
    y = df_valid[label_col].astype(int)

    n = len(y)
    weights = pd.Series(
        np.linspace(0.5, 1.0, n, dtype=np.float32),
        index=y.index,
    )

    nan_fracs = X.isna().mean()
    cols_to_fill = nan_fracs[nan_fracs < NAN_FILL_THRESHOLD].index
    if len(cols_to_fill) > 0:
        X[cols_to_fill] = X[cols_to_fill].fillna(X[cols_to_fill].median())

    return X, y, weights
