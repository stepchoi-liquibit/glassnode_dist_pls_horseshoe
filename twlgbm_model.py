"""
TWLGBM Model Training, Calibration & Evaluation
=================================================
Native multiclass LightGBM (softmax, 5 classes) with optional temperature
scaling (1w only).  4w/13w use raw softmax for small-sample stability.

Walk-forward expanding-window CV with per-horizon purge gap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

from twlgbm_config import (
    HORIZONS,
    HORIZON_WEEKS,
    NUM_BUCKETS,
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
    CV_N_FOLDS,
    MIN_TRAIN_WEEKS,
    PROB_EPSILON,
    TEMPERATURE_BOUNDS,
    CALIBRATED_HORIZONS,
)
from twlgbm_features import get_feature_columns, prepare_train_data

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainedClassifier:
    """Single multiclass LightGBM booster + optional temperature calibrator."""
    horizon: str
    model: lgb.Booster
    temperature: float | None          # only for calibrated horizons
    feature_cols: list[str]
    train_indices: list[int] = field(default_factory=list)


@dataclass
class CVResult:
    """Cross-validation results for one horizon."""
    horizon: str
    fold_rps: list[float]
    fold_logloss: list[float]
    mean_rps: float
    mean_logloss: float


# ──────────────────────────────────────────────────────────────────────────────
# RPS metric
# ──────────────────────────────────────────────────────────────────────────────

def ranked_probability_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean RPS for ordinal multiclass (lower is better)."""
    n_samples, n_classes = y_prob.shape
    rps_sum = 0.0
    for i in range(n_samples):
        cum_pred = np.cumsum(y_prob[i])
        cum_true = np.zeros(n_classes, dtype=np.float32)
        cum_true[int(y_true[i]):] = 1.0
        rps_sum += np.sum((cum_pred[:-1] - cum_true[:-1]) ** 2) / (n_classes - 1)
    return rps_sum / n_samples


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (shared by train_classifier and walk_forward_cv)
# ──────────────────────────────────────────────────────────────────────────────

def _make_dataset(X, y, weight=None, reference=None, params=None) -> lgb.Dataset:
    """Build a LightGBM Dataset that frees raw data after construction."""
    ds = lgb.Dataset(
        data=X, label=y, weight=weight, free_raw_data=True,
        feature_name=list(X.columns), reference=reference, params=params,
    )
    ds.construct()
    return ds


def _split_train_val_cal(
    X: pd.DataFrame, y: pd.Series, weights: pd.Series,
    calibrate: bool, val_frac: float = 0.10, cal_frac: float = 0.10,
) -> dict:
    """
    Chronological split: [train | val | cal].
    When calibrate=False, cal portion is merged into val for more data.
    Returns dict with X_train, y_train, w_train, X_val, y_val, X_cal, y_cal.
    """
    n = len(X)
    holdout_frac = val_frac + cal_frac

    if calibrate:
        n_cal = max(8, int(n * cal_frac))
        n_val = max(8, int(n * val_frac))
    else:
        n_cal = 0
        n_val = max(8, int(n * holdout_frac))

    n_train = n - n_val - n_cal

    if n_train < 30:
        # Fallback: shared val/cal
        n_holdout = max(10, int(n * holdout_frac))
        n_train = n - n_holdout
        return {
            "X_train": X.iloc[:n_train], "y_train": y.values[:n_train],
            "w_train": weights.iloc[:n_train],
            "X_val": X.iloc[n_train:], "y_val": y.values[n_train:],
            "X_cal": X.iloc[n_train:] if calibrate else None,
            "y_cal": y.values[n_train:] if calibrate else None,
        }

    split = {
        "X_train": X.iloc[:n_train], "y_train": y.values[:n_train],
        "w_train": weights.iloc[:n_train],
        "X_val": X.iloc[n_train:n_train + n_val],
        "y_val": y.values[n_train:n_train + n_val],
        "X_cal": None, "y_cal": None,
    }
    if calibrate and n_cal > 0:
        split["X_cal"] = X.iloc[n_train + n_val:]
        split["y_cal"] = y.values[n_train + n_val:]
    return split


def _train_booster(X_train, y_train, w_train, X_val, y_val, verbose=True) -> lgb.Booster:
    """Train a single multiclass LightGBM booster with early stopping."""
    train_ds = _make_dataset(X_train, y_train, weight=w_train, params=LGBM_PARAMS)
    val_ds = _make_dataset(X_val, y_val, reference=train_ds, params=LGBM_PARAMS)
    return lgb.train(
        params=LGBM_PARAMS, train_set=train_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_ds], valid_names=["validation"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=verbose)],
    )


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax (numerically stable)."""
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def _fit_temperature(model: lgb.Booster, X_cal: pd.DataFrame, y_cal: np.ndarray) -> float:
    """Fit temperature T minimizing NLL on calibration set (1 parameter)."""
    raw_logits = model.predict(X_cal, raw_score=True)

    def nll(T):
        probs = np.clip(_softmax(raw_logits, T), PROB_EPSILON, 1.0 - PROB_EPSILON)
        return -np.mean(np.log(probs[np.arange(len(y_cal)), y_cal.astype(int)]))

    result = minimize_scalar(nll, bounds=TEMPERATURE_BOUNDS, method="bounded")
    logger.info(f"  Temperature scaling: T={result.x:.4f} (NLL {result.fun:.4f})")
    return float(result.x)


def _predict_probs(model: lgb.Booster, X: pd.DataFrame, temperature: float | None) -> np.ndarray:
    """Predict probabilities, optionally temperature-scaled."""
    if temperature is not None:
        probs = _softmax(model.predict(X, raw_score=True), temperature)
    else:
        probs = model.predict(X)
    probs = np.clip(probs, PROB_EPSILON, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def predict_calibrated(classifier: TrainedClassifier, X: pd.DataFrame) -> np.ndarray:
    """Produce probability vectors from a TrainedClassifier. Shape (n, 5)."""
    return _predict_probs(classifier.model, X, classifier.temperature)


def train_classifier(
    df: pd.DataFrame, horizon: str,
    feature_cols: list[str] | None = None,
    calibrate: bool | None = None,
) -> TrainedClassifier:
    """Train a single native multiclass LightGBM model for one horizon."""
    X, y, weights = prepare_train_data(df, horizon, feature_cols)
    if feature_cols is None:
        feature_cols = list(X.columns)
    if calibrate is None:
        calibrate = horizon in CALIBRATED_HORIZONS

    logger.info(
        f"Training {horizon}: {X.shape[0]} samples, {X.shape[1]} features"
        f"{', +temp scaling' if calibrate else ''}"
    )

    s = _split_train_val_cal(X, y, weights, calibrate)
    booster = _train_booster(s["X_train"], s["y_train"], s["w_train"], s["X_val"], s["y_val"])
    logger.info(f"  {horizon}: stopped at round {booster.best_iteration}")

    temperature = None
    if calibrate and s["X_cal"] is not None and len(s["X_cal"]) >= 5:
        temperature = _fit_temperature(booster, s["X_cal"], s["y_cal"])

    return TrainedClassifier(
        horizon=horizon, model=booster, temperature=temperature,
        feature_cols=feature_cols, train_indices=list(X.index),
    )


def walk_forward_cv(
    df: pd.DataFrame, horizon: str,
    feature_cols: list[str] | None = None,
    n_folds: int = CV_N_FOLDS,
    min_train_weeks: int = MIN_TRAIN_WEEKS,
) -> CVResult:
    """
    Walk-forward expanding-window CV with per-horizon purge gap.

    Layout per fold:  train=[0, t)  gap=[t, t+h)  test=[t+h, ...)
    where h = HORIZON_WEEKS[horizon].
    """
    label_col = f"label_{horizon}"
    valid_idx = df.loc[df[label_col].notna()].index.tolist()
    n_valid = len(valid_idx)
    purge_weeks = HORIZON_WEEKS[horizon]
    calibrate = horizon in CALIBRATED_HORIZONS

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if n_valid < min_train_weeks + purge_weeks + 10:
        logger.warning(f"  {horizon} CV: too few samples ({n_valid}), skipping")
        return CVResult(horizon=horizon, fold_rps=[], fold_logloss=[],
                        mean_rps=np.nan, mean_logloss=np.nan)

    total_test = n_valid - min_train_weeks - purge_weeks
    fold_size = max(5, total_test // n_folds)

    fold_rps, fold_ll = [], []

    for fold_i in range(n_folds):
        train_end = min_train_weeks + fold_i * fold_size
        test_start = train_end + purge_weeks
        test_end = min(test_start + fold_size, n_valid)

        if test_end - test_start < 5:
            continue

        df_train = df.loc[valid_idx[:train_end]]
        df_test = df.loc[valid_idx[test_start:test_end]]

        X_train, y_train, w_train = prepare_train_data(df_train, horizon, feature_cols)
        X_test, y_test, _ = prepare_train_data(df_test, horizon, feature_cols)

        if len(X_train) < 30 or len(X_test) < 5:
            continue

        # Split train into fit/val(/cal)
        s = _split_train_val_cal(X_train, y_train, w_train, calibrate)
        if len(s["X_train"]) < 20:
            continue

        booster = _train_booster(s["X_train"], s["y_train"], s["w_train"],
                                 s["X_val"], s["y_val"], verbose=False)

        temperature = None
        if calibrate and s["X_cal"] is not None and len(s["X_cal"]) >= 5:
            temperature = _fit_temperature(booster, s["X_cal"], s["y_cal"])

        probs = _predict_probs(booster, X_test, temperature)

        rps = ranked_probability_score(y_test.values, probs)
        ll = log_loss(y_test.values, probs, labels=list(range(NUM_BUCKETS)))
        fold_rps.append(rps)
        fold_ll.append(ll)

        logger.info(
            f"  {horizon} fold {fold_i+1}/{n_folds}: "
            f"RPS={rps:.4f}, LL={ll:.4f}, train={len(s['X_train'])}, test={len(X_test)}"
        )

    mean_rps = np.mean(fold_rps) if fold_rps else np.nan
    mean_ll = np.mean(fold_ll) if fold_ll else np.nan
    logger.info(
        f"  {horizon} CV ({len(fold_rps)} folds): "
        f"RPS={mean_rps:.4f}±{np.std(fold_rps) if fold_rps else 0:.4f}"
    )

    return CVResult(
        horizon=horizon, fold_rps=fold_rps, fold_logloss=fold_ll,
        mean_rps=mean_rps, mean_logloss=mean_ll,
    )


def train_all_classifiers(
    df: pd.DataFrame,
    feature_cols: list[str] | dict[str, list[str]] | None = None,
    run_cv: bool = True,
) -> tuple[dict[str, TrainedClassifier], dict[str, CVResult]]:
    """Train one multiclass LightGBM classifier per horizon."""
    per_horizon = feature_cols if isinstance(feature_cols, dict) else {h: feature_cols for h in HORIZONS}
    classifiers, cv_results = {}, {}

    for horizon in HORIZONS:
        cols = per_horizon[horizon]
        logger.info(f"\n{'='*60}\nTraining {horizon}\n{'='*60}")
        if run_cv:
            cv_results[horizon] = walk_forward_cv(df, horizon, cols)
        classifiers[horizon] = train_classifier(df, horizon, cols)

    return classifiers, cv_results
