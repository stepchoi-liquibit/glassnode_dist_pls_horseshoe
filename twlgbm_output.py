"""
TWLGBM Output Applications
===========================
Johnson SU distribution fitting (Cramér-von Mises), exceedance curves,
Kelly sizing with Expected Shortfall, and SHAP attribution.

Implements Recommendations 1 & 2 from the review:
  - Cramér-von Mises fitting instead of MLE for Johnson SU stability
  - Expected Shortfall for tail-bucket representative returns
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from scipy import stats
from scipy.optimize import minimize

from twlgbm_config import (
    HORIZONS,
    BUCKET_THRESHOLDS,
    BUCKET_NAMES,
    NUM_BUCKETS,
    KELLY_TAIL_MULTIPLIER,
    NAN_FILL_THRESHOLD,
)
from twlgbm_model import TrainedClassifier, predict_calibrated

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data class for forecast output
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HorizonForecast:
    """Complete forecast output for one horizon."""
    horizon: str
    bucket_probs: np.ndarray         # shape (5,)
    exceedance_probs: dict[float, float]  # threshold -> P(r > threshold)
    johnson_su_params: dict | None   # {a, b, loc, scale} or None if fit failed
    kelly_fractions: dict[str, float]  # bucket -> Kelly fraction
    shap_values: pd.DataFrame | None  # features x classes SHAP matrix
    shap_directional: pd.Series | None  # feature -> net directional signal


# ──────────────────────────────────────────────────────────────────────────────
# 1. Exceedance curve from bucket probabilities (plan section 8.1)
# ──────────────────────────────────────────────────────────────────────────────

def compute_exceedance_probs(
    bucket_probs: np.ndarray,
    thresholds: dict,
) -> dict[float, float]:
    """
    Convert 5-bucket probability vector to exceedance probabilities.

    P(r > b1) = P(B2) + P(B3) + P(B4) + P(B5) = 1 - P(B1)
    P(r > b2) = P(B3) + P(B4) + P(B5)
    P(r > b3) = P(B4) + P(B5)
    P(r > b4) = P(B5)
    """
    bounds = thresholds["boundaries"]  # [b1, b2, b3, b4]

    p = bucket_probs  # [B1, B2, B3, B4, B5]
    exceedance = {
        bounds[0]: float(1 - p[0]),
        bounds[1]: float(p[2] + p[3] + p[4]),
        bounds[2]: float(p[3] + p[4]),
        bounds[3]: float(p[4]),
    }
    return exceedance


# ──────────────────────────────────────────────────────────────────────────────
# 2. Johnson SU Distribution Fitting (Recommendation 1: Cramér-von Mises)
# ──────────────────────────────────────────────────────────────────────────────

def fit_johnson_su_cramer_von_mises(
    exceedance_probs: dict[float, float],
) -> dict | None:
    """
    Fit Johnson SU distribution using Minimum Cramér-von Mises Distance.

    Instead of MLE on 4 points (which is unstable), we minimize the
    integrated squared distance between the model CDF and the implied
    quantile points from the exceedance curve.

    The initial scale parameter is dynamically derived from the standard
    deviation of the discrete bucket distribution implied by the exceedance
    curve, providing a mathematically coherent starting point.

    Returns dict with keys {a, b, loc, scale} or None on failure.
    """
    # Convert exceedance to (threshold, CDF_value) pairs
    # P(r > t) = 1 - CDF(t), so CDF(t) = 1 - exceedance(t)
    quantile_points = []
    for threshold, exc_prob in sorted(exceedance_probs.items()):
        cdf_val = 1.0 - exc_prob
        cdf_val = np.clip(cdf_val, 0.001, 0.999)
        quantile_points.append((threshold, cdf_val))

    thresholds = np.array([q[0] for q in quantile_points], dtype=np.float64)
    cdf_targets = np.array([q[1] for q in quantile_points], dtype=np.float64)

    # ── Dynamic initial scale from discrete bucket distribution ──
    # Reconstruct bucket probabilities from exceedance:
    #   P(B1) = CDF(b0),  P(B2) = CDF(b1)-CDF(b0), ..., P(B5) = 1-CDF(b3)
    cdf_vals = cdf_targets  # CDF at each boundary
    bucket_probs = np.zeros(len(cdf_vals) + 1)
    bucket_probs[0] = cdf_vals[0]
    for i in range(1, len(cdf_vals)):
        bucket_probs[i] = cdf_vals[i] - cdf_vals[i - 1]
    bucket_probs[-1] = 1.0 - cdf_vals[-1]
    bucket_probs = np.clip(bucket_probs, 0.0, 1.0)
    bucket_probs /= bucket_probs.sum()  # renormalise

    # Bucket midpoints: use boundaries as edges, extrapolate tails
    t = thresholds
    spread = t[-1] - t[0] if len(t) > 1 else abs(t[0]) * 2
    midpoints = np.zeros(len(bucket_probs))
    midpoints[0] = t[0] - spread * 0.25       # tail-left midpoint
    for i in range(1, len(t)):
        midpoints[i] = (t[i - 1] + t[i]) / 2
    midpoints[-1] = t[-1] + spread * 0.25     # tail-right midpoint

    # Discrete std of the implied distribution
    mu = np.dot(bucket_probs, midpoints)
    var = np.dot(bucket_probs, (midpoints - mu) ** 2)
    scale_init = max(0.005, float(np.sqrt(var)))
    loc_init = float(mu)

    def cramer_von_mises_loss(params):
        a, b, loc, scale = params
        if b <= 0 or scale <= 0:
            return 1e10
        try:
            cdf_model = stats.johnsonsu.cdf(thresholds, a, b, loc=loc, scale=scale)
            # Cramér-von Mises: sum of squared CDF differences
            return np.sum((cdf_model - cdf_targets) ** 2)
        except Exception:
            return 1e10

    # Multi-start optimization for robustness
    best_result = None
    best_loss = np.inf

    # Initial guesses: vary shape parameters, use data-driven loc & scale
    for a_init in [-0.5, 0.0, 0.5]:
        for b_init in [0.5, 1.0, 2.0]:
            try:
                result = minimize(
                    cramer_von_mises_loss,
                    x0=[a_init, b_init, loc_init, scale_init],
                    method="Nelder-Mead",
                    options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10},
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                continue

    if best_result is None or best_loss > 0.1:
        logger.warning("Johnson SU fit failed or poor quality")
        return None

    a, b, loc, scale = best_result.x
    if b <= 0 or scale <= 0:
        return None

    params = {"a": float(a), "b": float(b), "loc": float(loc), "scale": float(scale)}
    logger.info(
        f"  Johnson SU fit: a={a:.4f}, b={b:.4f}, loc={loc:.6f}, "
        f"scale={scale:.6f}, CvM loss={best_loss:.6f} "
        f"(scale_init={scale_init:.6f})"
    )
    return params


# ──────────────────────────────────────────────────────────────────────────────
# 3. Kelly Sizing with Expected Shortfall (Recommendation 2)
# ──────────────────────────────────────────────────────────────────────────────

def _expected_shortfall_from_johnson_su(
    params: dict,
    tail: str,
    threshold: float,
) -> float:
    """
    Compute Expected Shortfall (conditional tail mean) from fitted Johnson SU.

    For B1 (left tail): ES = E[r | r <= -outer]
    For B5 (right tail): ES = E[r | r >= +outer]
    """
    a, b, loc, scale = params["a"], params["b"], params["loc"], params["scale"]
    dist = stats.johnsonsu(a, b, loc=loc, scale=scale)

    n_samples = 50000
    samples = dist.rvs(size=n_samples, random_state=42)

    if tail == "left":
        tail_samples = samples[samples <= threshold]
    else:
        tail_samples = samples[samples >= threshold]

    if len(tail_samples) < 10:
        return threshold * KELLY_TAIL_MULTIPLIER

    es = float(np.mean(tail_samples))

    # Clamp to reasonable range: no more than 3x the threshold
    max_abs = abs(threshold) * 3.0
    es = np.clip(es, -max_abs, max_abs)

    return es


def compute_kelly_fractions(
    bucket_probs: np.ndarray,
    thresholds: dict,
    johnson_su_params: dict | None = None,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """
    Compute Kelly-optimal position fractions using discrete bucket distribution.

    For tail buckets (B1, B5):
      - If Johnson SU is available, use Expected Shortfall as representative return
      - Otherwise, use threshold * KELLY_TAIL_MULTIPLIER

    For inner buckets (B2, B3, B4): use midpoint of the bucket.
    """
    bounds = thresholds["boundaries"]  # [b1, b2, b3, b4]

    # Representative returns per bucket
    if johnson_su_params is not None:
        r_b1 = _expected_shortfall_from_johnson_su(johnson_su_params, "left", bounds[0])
        r_b5 = _expected_shortfall_from_johnson_su(johnson_su_params, "right", bounds[3])
    else:
        r_b1 = bounds[0] * KELLY_TAIL_MULTIPLIER
        r_b5 = bounds[3] * KELLY_TAIL_MULTIPLIER

    representative_returns = {
        "B1": r_b1,
        "B2": (bounds[0] + bounds[1]) / 2,
        "B3": (bounds[1] + bounds[2]) / 2,
        "B4": (bounds[2] + bounds[3]) / 2,
        "B5": r_b5,
    }

    # Expected return
    p = bucket_probs
    r_vals = np.array([representative_returns[b] for b in BUCKET_NAMES], dtype=np.float32)
    mean_return = np.dot(p, r_vals)              # raw mean μ
    expected_return = mean_return - risk_free     # excess return for Kelly numerator

    # Variance centered on the raw mean (not the excess return)
    variance = np.dot(p, (r_vals - mean_return) ** 2)

    if variance < 1e-12:
        kelly_full = 0.0
    else:
        kelly_full = expected_return / variance

    # Half-Kelly for prudence
    kelly_half = kelly_full / 2.0

    return {
        "expected_return": float(expected_return),
        "variance": float(variance),
        "kelly_full": float(kelly_full),
        "kelly_half": float(kelly_half),
        "representative_returns": representative_returns,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. SHAP Attribution (plan section 7)
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_attribution(
    classifier: TrainedClassifier,
    X_latest: pd.DataFrame,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Compute SHAP values for the latest observation using the native
    multiclass LightGBM model.

    TreeExplainer on a multiclass model returns SHAP values of shape
    (n_samples, n_features, n_classes).  The directional signal is
    computed as a signed weighted sum across classes: higher-bucket
    SHAP contributions are treated as bullish, lower as bearish.

    Returns
    -------
    (shap_df, directional) where:
      - shap_df: DataFrame of shape (top_n_features, n_classes) with
        SHAP per bucket class (columns: "B1", "B2", …, "B5")
      - directional: Series of net directional SHAP scores per feature
        (positive = bullish, negative = bearish)
    """
    n_features = X_latest.shape[1]
    feature_names = list(X_latest.columns)
    n_classes = NUM_BUCKETS

    explainer = shap.TreeExplainer(classifier.model)
    sv = explainer.shap_values(X_latest)

    # Multiclass SHAP: list of n_classes arrays each (n_samples, n_features)
    # OR ndarray of shape (n_samples, n_features, n_classes)
    if isinstance(sv, list):
        # sv[c] has shape (n_samples, n_features)
        shap_matrix = np.stack([sv[c][-1, :] for c in range(n_classes)], axis=1)
    elif sv.ndim == 3:
        # shape (n_samples, n_features, n_classes)
        shap_matrix = sv[-1, :, :]  # (n_features, n_classes)
    else:
        # Unexpected shape — try to reshape
        shap_matrix = sv[-1, :].reshape(n_features, -1)

    shap_df = pd.DataFrame(
        shap_matrix,
        index=feature_names,
        columns=BUCKET_NAMES[:n_classes],
    )

    # Directional aggregation: weight classes by ordinal position.
    # Signed weights: B1=-2, B2=-1, B3=0, B4=+1, B5=+2
    # A feature that pushes mass from B1 to B5 gets a positive score.
    ordinal_weights = np.array([-2, -1, 0, 1, 2], dtype=np.float32)[:n_classes]
    directional = pd.Series(
        shap_matrix @ ordinal_weights,
        index=feature_names,
        name="directional_shap",
    )

    # Sort by absolute directional importance and take top N
    abs_importance = directional.abs().sort_values(ascending=False)
    top_features = abs_importance.head(top_n).index.tolist()

    shap_df_top = shap_df.loc[top_features]
    directional_top = directional.loc[top_features]

    return shap_df_top, directional_top


def shap_sanity_checks(
    shap_df: pd.DataFrame,
    directional: pd.Series,
    bucket_probs: np.ndarray,
) -> list[str]:
    """
    Run 4 automated SHAP sanity checks (plan section 7).

    Returns list of warning messages (empty if all pass).
    """
    warnings = []

    # Check 1: SHAP values should sum to near-zero across features for each class
    class_sums = shap_df.sum(axis=0)
    for cls, val in class_sums.items():
        if abs(val) > 5.0:
            warnings.append(f"SHAP sum for {cls} is {val:.2f} (expected ~0)")

    # Check 2: Directional SHAP should be consistent with predicted bucket
    predicted_bucket = int(np.argmax(bucket_probs))
    mean_direction = directional.mean()
    if predicted_bucket <= 1 and mean_direction > 0.5:
        warnings.append(
            f"Mean SHAP direction ({mean_direction:.2f}) inconsistent with "
            f"bearish prediction (bucket {predicted_bucket})"
        )
    elif predicted_bucket >= 3 and mean_direction < -0.5:
        warnings.append(
            f"Mean SHAP direction ({mean_direction:.2f}) inconsistent with "
            f"bullish prediction (bucket {predicted_bucket})"
        )

    # Check 3: No single feature should dominate > 50% of total SHAP magnitude
    total_mag = directional.abs().sum()
    if total_mag > 0:
        max_feature = directional.abs().idxmax()
        max_pct = directional.abs().max() / total_mag
        if max_pct > 0.5:
            warnings.append(
                f"Feature '{max_feature}' dominates {max_pct:.0%} of SHAP magnitude"
            )

    # Check 4: At least 3 features should have non-trivial SHAP values
    non_trivial = (directional.abs() > 0.01).sum()
    if non_trivial < 3:
        warnings.append(f"Only {non_trivial} features with non-trivial SHAP values")

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# 5. Generate full forecast for one observation
# ──────────────────────────────────────────────────────────────────────────────

def generate_forecast(
    classifiers: dict[str, TrainedClassifier],
    X_latest: pd.DataFrame,
    thresholds: dict | None = None,
    compute_shap: bool = True,
) -> dict[str, HorizonForecast]:
    """
    Generate forecasts for all horizons for the latest observation.

    Parameters
    ----------
    classifiers : dict mapping horizon -> TrainedClassifier
    X_latest : DataFrame with one row of features
    thresholds : bucket thresholds (defaults to config)
    compute_shap : whether to compute SHAP attribution

    Returns
    -------
    dict mapping horizon -> HorizonForecast
    """
    if thresholds is None:
        thresholds = BUCKET_THRESHOLDS

    forecasts = {}

    for horizon in HORIZONS:
        clf = classifiers[horizon]

        # Ensure X_latest has the right columns
        X = X_latest[clf.feature_cols].copy()

        # Conditional NaN fill (consistent with prepare_train_data).
        # Single-row: NaN columns have 100% missing → left for LightGBM.
        nan_fracs = X.isna().mean()
        cols_to_fill = nan_fracs[nan_fracs < NAN_FILL_THRESHOLD].index
        if len(cols_to_fill) > 0:
            X[cols_to_fill] = X[cols_to_fill].fillna(X[cols_to_fill].median())

        probs = predict_calibrated(clf, X)
        bucket_probs = probs[0]  # single observation

        # Exceedance curve
        exc = compute_exceedance_probs(bucket_probs, thresholds[horizon])

        # Johnson SU fit
        jsu_params = fit_johnson_su_cramer_von_mises(exc)

        # Kelly sizing
        kelly = compute_kelly_fractions(bucket_probs, thresholds[horizon], jsu_params)

        # SHAP
        shap_df = None
        directional = None
        if compute_shap:
            try:
                shap_df, directional = compute_shap_attribution(clf, X)
                warnings = shap_sanity_checks(shap_df, directional, bucket_probs)
                for w in warnings:
                    logger.warning(f"  SHAP check ({horizon}): {w}")
            except Exception as exc_err:
                logger.warning(f"  SHAP computation failed for {horizon}: {exc_err}")

        forecasts[horizon] = HorizonForecast(
            horizon=horizon,
            bucket_probs=bucket_probs,
            exceedance_probs=exc,
            johnson_su_params=jsu_params,
            kelly_fractions=kelly,
            shap_values=shap_df,
            shap_directional=directional,
        )

        logger.info(
            f"\n  {horizon} forecast:"
            f"\n    Bucket probs: {dict(zip(BUCKET_NAMES, [f'{p:.3f}' for p in bucket_probs]))}"
            f"\n    Exceedance:   {dict(zip([f'{k:+.2f}' for k in exc], [f'{v:.3f}' for v in exc.values()]))}"
            f"\n    Kelly half:   {kelly['kelly_half']:.4f}"
            f"\n    E[r]:         {kelly['expected_return']:.4f}"
        )

    return forecasts
