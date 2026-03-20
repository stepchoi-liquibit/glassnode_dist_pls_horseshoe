#!/usr/bin/env python3
"""
BTC Return Distribution Estimator (Options / Risk-Neutral Density)
===================================================================

Estimates BTC return distributions from historical options vol surfaces
via the Breeden-Litzenberger method.

Pipeline per weekly observation (Thursday close):
    1. Fetch implied-vol delta surface from Glassnode
       (call/put delta 10, 25, 50 x tenors 1w, 1m, 3m)
    2. Fit a global SSVI surface (rho, eta, gamma, theta per tenor)
    3. For each tenor: evaluate smooth SVI on a fine log-strike grid,
       compute BS call prices, differentiate twice w.r.t. strike to
       extract the risk-neutral density (Breeden-Litzenberger)
    4. Fit a Student-t distribution to the RND

Produces **6 parameter sets**:
    - 3 tenors  x  full lookback   (all weekly obs averaged)
    - 3 tenors  x  trailing 1 month (last 5 weekly obs averaged)

Tenor mapping to match distributions_price.py:
    1w  ->  1w  (1 week)
    1m  ->  4w  (4 weeks)
    3m  ->  13w (13 weeks)

Usage (import):
    from distributions_options import estimate_distributions_options
    results = estimate_distributions_options()

Usage (CLI):
    python distributions_options.py
    python distributions_options.py --end-date 2026-03-09
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t

# NumPy 2.x renamed np.trapz -> np.trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")

from glassnode_fetcher import fetch_glassnode_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Option tenors to use (Glassnode returns 1w, 1m, 3m, 6m; we use the first 3)
OPTION_TENORS = ["1w", "1m", "3m"]
TENOR_DAYS = {"1w": 7, "1m": 30, "3m": 91, "6m": 182}

# Map option tenor labels to distributions_price.py horizon labels
TENOR_TO_HORIZON = {"1w": "1w", "1m": "4w", "3m": "13w"}

# Deltas and option types to fetch
DELTAS = [10, 25, 50]
OPTION_TYPES = ["call", "put"]

# Glassnode endpoints for IV surfaces
IV_ENDPOINTS = [
    f"/v1/metrics/options/iv_{otype}_delta_{d}"
    for otype in OPTION_TYPES
    for d in DELTAS
]

# SSVI fitting: use all call deltas, exclude put-50 (ATM duplicate)
FIT_CALL_DELTAS = {10, 25, 50}
FIT_PUT_DELTAS = {10, 25}

# Student-t bounds
NU_MIN = 2.0
NU_MAX = 30.0

# BTC funding rate for delta-to-moneyness conversion
BTC_FUNDING_RATE = 0.04

# RND computation grid (log-moneyness coordinates, F = 1)
K_MIN = -2.0
K_MAX = 2.0
N_K = 401


# ==============================================================================
# Helpers
# ==============================================================================

def _default_dates(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve default dates: end = yesterday (options data lags 1 day),
    start = 52 weeks before end."""
    if end_date is None:
        # Options IV data is typically not available for today's date;
        # default to yesterday to avoid futile API calls.
        end_dt = datetime.now(timezone.utc) - timedelta(days=1)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start_date is None:
        start_dt = end_dt - timedelta(weeks=52)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _tenor_to_years(tenor: str) -> float:
    """Convert tenor string ('1w', '1m', '3m') to fractional years."""
    days = TENOR_DAYS.get(tenor)
    if days is None:
        raise ValueError(f"Unknown tenor: {tenor}")
    return days / 365.0


# ==============================================================================
# Delta-to-log-moneyness (from glassnode_ssvi_calibration.py)
# ==============================================================================

def _delta_to_pseudo_k(
    delta_abs: float,
    option_type: str,
    T: float,
    r: float,
) -> float:
    """
    Map absolute delta to log-moneyness proxy k ~ -d1 using BS delta.

    For discounted forward delta:
        delta_call = exp(-rT) * N(d1)
        |delta_put| = exp(-rT) * (1 - N(d1))

    Returns k ~ -d1 as proxy for log-moneyness.
    """
    d = float(delta_abs)
    if not (0.0 < d < 1.0):
        return np.nan
    if T <= 0:
        return np.nan

    disc_inv = math.exp(float(r) * float(T))
    eps = 1e-12

    if option_type == "call":
        Nd1 = d * disc_inv
    elif option_type == "put":
        Nd1 = 1.0 - d * disc_inv
    else:
        return np.nan

    Nd1 = min(max(Nd1, eps), 1.0 - eps)
    d1 = norm.ppf(Nd1)
    return -float(d1)


# ==============================================================================
# SSVI Model (from glassnode_ssvi_calibration.py)
# ==============================================================================

def _ssvi_total_variance(
    k: np.ndarray,
    theta: np.ndarray,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """
    SSVI total variance (Gatheral-Jacquier):
        w(k, theta) = theta/2 * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1-rho^2))
        phi(theta)  = eta * theta^(-gamma)
    """
    theta = np.asarray(theta, dtype=float)
    k = np.asarray(k, dtype=float)
    phi = eta * np.power(theta, -gamma)
    z = phi * k + rho
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(z * z + 1.0 - rho * rho))


def _pack_monotone_thetas(u: np.ndarray) -> np.ndarray:
    """
    Build positive increasing theta grid from unconstrained u:
        inc_i = exp(u_i) > 0;  theta_i = cumsum(inc) => increasing > 0
    """
    return np.cumsum(np.exp(np.asarray(u, dtype=float)))


# ==============================================================================
# Black-Scholes (forward measure, F = 1)
# ==============================================================================

def _bs_call_forward(
    F: float,
    K: np.ndarray,
    T: float,
    vol: np.ndarray,
) -> np.ndarray:
    """
    Black-Scholes call in forward measure (discount = 1):
        C = F N(d1) - K N(d2)
    """
    K = np.asarray(K, dtype=float)
    vol = np.asarray(vol, dtype=float)
    out = np.maximum(F - K, 0.0)

    msk = (T > 0) & np.isfinite(K) & (K > 0) & np.isfinite(vol) & (vol > 0)
    if not np.any(msk):
        return out

    srt = vol[msk] * math.sqrt(T)
    lnFK = np.log(F / K[msk])
    d1 = (lnFK + 0.5 * (vol[msk] ** 2) * T) / srt
    d2 = d1 - srt
    out[msk] = F * norm.cdf(d1) - K[msk] * norm.cdf(d2)
    return out


# ==============================================================================
# IV Data Fetching & Reshaping
# ==============================================================================

def _fetch_iv_surface(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
) -> pd.DataFrame:
    """
    Fetch IV delta data via ``fetch_glassnode_data()`` and return a wide
    DataFrame with one column per (type, delta, tenor) combination.
    Columns like: ``iv_call_delta_50_1w``, ``iv_put_delta_10_3m``, ...
    """
    df = fetch_glassnode_data(
        asset=asset,
        endpoints=IV_ENDPOINTS,
        start_date=start_date,
        end_date=end_date,
        resolution="24h",
        friendly_names=False,
    )
    return df


def _wide_to_tidy(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide DataFrame (one column per type/delta/tenor) to tidy format:
        date, type, delta, tenor, iv

    Expected column names: ``iv_call_delta_10_1w``, ``iv_put_delta_25_3m``, ...
    """
    # Find IV columns
    iv_cols = [c for c in df_wide.columns
               if c.startswith("iv_") and any(t in c for t in TENOR_DAYS)]

    records: list[dict] = []
    for _, row in df_wide.iterrows():
        date_val = row["date"]
        for col in iv_cols:
            val = row[col]
            if pd.isna(val):
                continue
            # Parse: iv_call_delta_10_1w -> type=call, delta=10, tenor=1w
            parts = col.split("_")
            # parts: ['iv', 'call'|'put', 'delta', '10'|'25'|'50', '1w'|'1m'|...]
            if len(parts) != 5:
                continue
            opt_type = parts[1]
            try:
                delta = int(parts[3])
            except ValueError:
                continue
            tenor = parts[4]
            if tenor not in OPTION_TENORS:
                continue

            records.append({
                "date": date_val,
                "type": opt_type,
                "delta": delta,
                "tenor": tenor,
                "iv": float(val),
            })

    tidy = pd.DataFrame(records)
    if not tidy.empty:
        tidy = tidy.sort_values(["date", "tenor", "type", "delta"]).reset_index(drop=True)
    return tidy


# ==============================================================================
# SSVI Surface Fitting (one day)
# ==============================================================================

def _fit_ssvi(iv_day: pd.DataFrame) -> Optional[dict]:
    """
    Fit SSVI surface to one day's IV data.

    Parameters
    ----------
    iv_day : DataFrame
        Tidy IV data for a single date with columns [type, delta, tenor, iv].

    Returns
    -------
    dict with keys: rho, eta, gamma, thetas (dict tenor->float)
    or None on failure.
    """
    tenors = sorted(iv_day["tenor"].unique(), key=_tenor_to_years)
    T_by_tenor = {t: _tenor_to_years(t) for t in tenors}
    nT = len(tenors)

    if nT < 2:
        return None

    # Compute pseudo log-moneyness
    g = iv_day.copy()
    g["T_years"] = g["tenor"].map(T_by_tenor).astype(float)
    g["k"] = g.apply(
        lambda r: _delta_to_pseudo_k(
            float(r["delta"]) / 100.0,
            str(r["type"]),
            float(r["T_years"]),
            BTC_FUNDING_RATE,
        ),
        axis=1,
    )

    # Select points for fitting
    g["used"] = (
        ((g["type"] == "call") & (g["delta"].isin(FIT_CALL_DELTAS)))
        | ((g["type"] == "put") & (g["delta"].isin(FIT_PUT_DELTAS)))
    )
    g_fit = g[g["used"]].copy()

    # Drop invalid
    g_fit = g_fit[np.isfinite(g_fit["k"]) & np.isfinite(g_fit["iv"]) & (g_fit["iv"] > 0)]
    if len(g_fit) < max(6, nT + 3):
        return None

    k = g_fit["k"].to_numpy(dtype=float)
    iv = g_fit["iv"].to_numpy(dtype=float)
    tenor_arr = g_fit["tenor"].astype(str).to_numpy()
    T = g_fit["T_years"].to_numpy(dtype=float)
    w_obs = (iv ** 2) * T

    tenor_index = {t: i for i, t in enumerate(tenors)}
    idx_T = np.array([tenor_index[t] for t in tenor_arr], dtype=int)

    # Initialise theta from ATM (call delta-50) per tenor
    theta_init = []
    for tenor in tenors:
        sub = g_fit[g_fit["tenor"] == tenor]
        atm = sub[(sub["type"] == "call") & (sub["delta"] == 50)]["iv"]
        if len(atm) > 0 and np.isfinite(atm.iloc[-1]):
            th0 = float(atm.iloc[-1]) ** 2 * T_by_tenor[tenor]
        else:
            w_sub = (sub["iv"].to_numpy(dtype=float) ** 2) * T_by_tenor[tenor]
            th0 = float(np.nanmedian(w_sub)) if len(w_sub) > 0 else 0.01
        theta_init.append(max(th0, 1e-8))

    theta_init = np.maximum.accumulate(np.array(theta_init, dtype=float))
    incs = [theta_init[0]] + [
        max(theta_init[i] - theta_init[i - 1], 1e-8) for i in range(1, nT)
    ]
    u0 = np.log(np.array(incs, dtype=float))

    x0 = np.concatenate([u0, np.array([0.0, 1.0, 0.5], dtype=float)])
    bounds = [(-20, 20)] * nT + [(-0.999, 0.999), (1e-8, 50.0), (0.0, 1.0)]

    def objective(x: np.ndarray) -> float:
        u = x[:nT]
        rho, eta, gamma = float(x[nT]), float(x[nT + 1]), float(x[nT + 2])
        theta_grid = _pack_monotone_thetas(u)
        theta_pts = theta_grid[idx_T]
        w_fit = _ssvi_total_variance(k, theta_pts, rho, eta, gamma)
        if np.any(~np.isfinite(w_fit)) or np.any(w_fit <= 0):
            return 1e12
        mse = float(np.mean((w_fit - w_obs) ** 2))
        # Soft no-butterfly penalty: eta*(1+|rho|) <= 2
        viol = eta * (1.0 + abs(rho)) - 2.0
        return mse + (1e3 * max(viol, 0.0) ** 2)

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    u_hat = res.x[:nT]
    rho = float(res.x[nT])
    eta = float(res.x[nT + 1])
    gamma = float(res.x[nT + 2])
    theta_hat = _pack_monotone_thetas(u_hat)

    if np.any(~np.isfinite(theta_hat)) or np.any(theta_hat <= 0):
        return None

    return {
        "rho": rho,
        "eta": eta,
        "gamma": gamma,
        "thetas": {tenor: float(th) for tenor, th in zip(tenors, theta_hat)},
    }


# ==============================================================================
# Breeden-Litzenberger: SSVI slice -> risk-neutral density
# ==============================================================================

def _compute_rnd(
    rho: float,
    eta: float,
    gamma: float,
    theta: float,
    T: float,
    k_grid: np.ndarray,
) -> np.ndarray:
    """
    Extract risk-neutral density over log-returns from an SSVI slice using
    the Breeden-Litzenberger method.

    1. Evaluate SSVI implied vol on ``k_grid``
    2. Compute BS call prices (forward measure, F = 1)
    3. Differentiate twice w.r.t. strike K via k-derivatives:
           K^2 C_KK = C_kk - C_k
       Density over K:   q_K = C_KK = (C_kk - C_k) / K^2
       Density over k:   q_k = q_K * K = (C_kk - C_k) / K

    Parameters
    ----------
    theta : float
        SSVI theta for this tenor.
    T : float
        Time to expiry in years.
    k_grid : ndarray
        Fine grid of log-moneyness values (k = ln K/F).

    Returns
    -------
    q : ndarray
        Normalised risk-neutral density evaluated on ``k_grid``.
    """
    dk = float(k_grid[1] - k_grid[0])
    K_grid = np.exp(k_grid)

    # SSVI implied vol
    w = _ssvi_total_variance(
        k_grid, np.full_like(k_grid, theta), rho, eta, gamma,
    )
    iv = np.sqrt(np.maximum(w / T, 1e-12))

    # BS call prices
    C = _bs_call_forward(1.0, K_grid, T, iv)

    # k-derivatives (central difference)
    C_k = np.zeros_like(C)
    C_kk = np.zeros_like(C)
    C_k[1:-1] = (C[2:] - C[:-2]) / (2.0 * dk)
    C_kk[1:-1] = (C[2:] - 2.0 * C[1:-1] + C[:-2]) / (dk * dk)

    # Density over log-return k:  q(k) = (C_kk - C_k) / K
    q = (C_kk - C_k) / K_grid

    # Boundaries are invalid
    q[0] = 0.0
    q[-1] = 0.0

    # Ensure non-negative
    q = np.maximum(q, 0.0)

    # Normalise
    total = _trapezoid(q, k_grid)
    if total > 1e-12:
        q = q / total

    return q


# ==============================================================================
# Student-t fitting from density (weighted MLE / KL minimisation)
# ==============================================================================

def _fit_student_t_to_density(
    k_grid: np.ndarray,
    density: np.ndarray,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
) -> tuple[float, float, float]:
    """
    Fit Student-t to a discretised density by minimising KL divergence.

    Equivalent to weighted MLE:
        max  sum_i  w_i * log( t_pdf(k_i; mu, sigma, nu) )
    where  w_i = q(k_i) * dk  (probability mass in bin i).

    Returns (mu, sigma, nu).
    """
    dk = float(k_grid[1] - k_grid[0])
    weights = density * dk
    total = weights.sum()
    if total < 1e-12:
        raise ValueError("Density integrates to ~0; cannot fit Student-t")
    weights = weights / total  # normalise to sum = 1

    # Keep only bins with meaningful mass
    mask = weights > 1e-10
    k = k_grid[mask]
    w = weights[mask]

    # Moment estimates from the density
    mu0 = float(np.sum(w * k))
    sigma0 = float(np.sqrt(np.sum(w * (k - mu0) ** 2)))
    sigma0 = max(sigma0, 1e-6)

    def neg_weighted_ll(params):
        nu, mu, sigma = params
        if sigma <= 0:
            return 1e12
        log_pdf = student_t.logpdf(k, df=nu, loc=mu, scale=sigma)
        return -float(np.sum(w * log_pdf))

    best_result = None
    best_nll = np.inf

    for nu_init in [3.5, 7.0, 15.0, 25.0]:
        try:
            result = minimize(
                neg_weighted_ll,
                x0=[nu_init, mu0, sigma0],
                bounds=[(nu_min, nu_max), (None, None), (1e-12, None)],
                method="L-BFGS-B",
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("Student-t fitting to density failed")

    nu, mu, sigma = best_result.x
    return float(mu), float(sigma), float(nu)


# ==============================================================================
# Main Pipeline
# ==============================================================================

def estimate_distributions_options(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    asset: str = "BTC",
) -> pd.DataFrame:
    """
    Estimate Student-t distribution parameters from options vol surfaces
    via SSVI calibration + Breeden-Litzenberger RND extraction.

    Produces **6 parameter sets**:
        3 tenors (1w / 4w / 13w)  x  2 lookbacks (full / 1m)

    The ``mu``, ``sigma``, ``nu`` for each lookback are the arithmetic mean
    of the per-observation Student-t fits.

    Parameters
    ----------
    end_date : str or None
        End date 'YYYY-MM-DD'.  Defaults to today.
    start_date : str or None
        Start date 'YYYY-MM-DD'.  Defaults to 52 weeks before end_date.
    asset : str
        Asset symbol (default 'BTC').

    Returns
    -------
    pd.DataFrame
        Columns: [lookback, horizon, mu, sigma, nu, n_obs]
    """
    _start, _end = _default_dates(end_date, start_date)

    logger.info(f"Estimating option-implied distributions for {asset}")
    logger.info(f"  Date range:  {_start}  ->  {_end}")

    # -- Step 1: Fetch IV surface data -----------------------------------------
    logger.info("  Fetching IV delta surfaces from Glassnode...")
    iv_wide = _fetch_iv_surface(_start, _end, asset)
    if iv_wide.empty:
        raise RuntimeError("No IV data returned from Glassnode")

    logger.info(f"  Raw IV data: {iv_wide.shape[0]} rows x {iv_wide.shape[1]} cols")

    # -- Step 2: Filter to Thursdays (weekly observations) ---------------------
    iv_wide["_date"] = pd.to_datetime(iv_wide["date"])
    iv_wide = iv_wide[iv_wide["_date"].dt.weekday == 3].copy()  # Thursday = 3
    iv_wide = iv_wide.sort_values("_date").reset_index(drop=True)
    thursday_dates = iv_wide["_date"].unique()
    logger.info(f"  Thursday observations: {len(thursday_dates)}")

    if len(thursday_dates) == 0:
        raise RuntimeError("No Thursday observations found in IV data")

    # -- Step 3: Reshape to tidy format ----------------------------------------
    iv_tidy = _wide_to_tidy(iv_wide)
    if iv_tidy.empty:
        raise RuntimeError("No IV data after reshape (check column names)")
    iv_tidy["_date"] = pd.to_datetime(iv_tidy["date"])

    logger.info(f"  Tidy IV points: {len(iv_tidy)}")

    # -- Step 4: RND grid ------------------------------------------------------
    k_grid = np.linspace(K_MIN, K_MAX, N_K)

    # -- Step 5: Per-Thursday SSVI -> RND -> Student-t -------------------------
    all_fits: list[dict] = []  # each entry: {date, tenor, mu, sigma, nu}

    n_ssvi_fail = 0
    n_rnd_fail = 0
    n_t_fail = 0

    for date_val in thursday_dates:
        day_data = iv_tidy[iv_tidy["_date"] == date_val]
        date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

        # Check data availability for this Thursday
        n_tenors = day_data["tenor"].nunique()
        n_points = len(day_data)

        if n_points == 0:
            logger.warning(f"  {date_str}: no IV data points -- skipping")
            continue

        # Fit SSVI
        try:
            ssvi = _fit_ssvi(day_data)
        except Exception as exc:
            logger.warning(f"  {date_str}: SSVI fit raised {type(exc).__name__}: {exc}")
            n_ssvi_fail += 1
            continue

        if ssvi is None:
            logger.warning(
                f"  {date_str}: SSVI fit returned None "
                f"(tenors={n_tenors}, points={n_points})"
            )
            n_ssvi_fail += 1
            continue

        rho, eta, gamma = ssvi["rho"], ssvi["eta"], ssvi["gamma"]

        for tenor in OPTION_TENORS:
            theta = ssvi["thetas"].get(tenor)
            if theta is None or theta <= 0:
                logger.debug(f"  {date_str}/{tenor}: theta missing or <= 0")
                continue
            T = _tenor_to_years(tenor)

            # Compute RND
            try:
                q = _compute_rnd(rho, eta, gamma, theta, T, k_grid)
            except Exception as exc:
                logger.warning(
                    f"  {date_str}/{tenor}: RND failed -- "
                    f"{type(exc).__name__}: {exc}"
                )
                n_rnd_fail += 1
                continue

            # Sanity check: density should have meaningful mass
            if np.max(q) < 1e-6:
                logger.warning(
                    f"  {date_str}/{tenor}: RND max={np.max(q):.2e} too small"
                )
                n_rnd_fail += 1
                continue

            # Fit Student-t to RND
            try:
                mu, sigma, nu = _fit_student_t_to_density(k_grid, q)
            except Exception as exc:
                logger.warning(
                    f"  {date_str}/{tenor}: Student-t fit failed -- "
                    f"{type(exc).__name__}: {exc}"
                )
                n_t_fail += 1
                continue

            all_fits.append({
                "date": date_str,
                "tenor": tenor,
                "mu": mu,
                "sigma": sigma,
                "nu": nu,
            })

        # Quick log
        tenor_ok = [f["tenor"] for f in all_fits if f["date"] == date_str]
        if tenor_ok:
            logger.info(f"  {date_str}: SSVI OK, RND fitted for {tenor_ok}")

    if not all_fits:
        logger.error(
            f"  Failure summary: SSVI={n_ssvi_fail}, RND={n_rnd_fail}, "
            f"Student-t={n_t_fail} failures across {len(thursday_dates)} dates"
        )
        raise RuntimeError(
            f"No successful fits across all {len(thursday_dates)} dates. "
            f"Failures: SSVI={n_ssvi_fail}, RND={n_rnd_fail}, "
            f"Student-t={n_t_fail}. "
            f"Run with --log-level DEBUG or check the WARNING messages above."
        )

    fits_df = pd.DataFrame(all_fits)
    fits_df["_date"] = pd.to_datetime(fits_df["date"])

    # -- Step 6: Average across lookback windows -------------------------------
    end_dt = pd.to_datetime(_end)
    one_month_ago = end_dt - pd.DateOffset(weeks=5)

    results: list[dict] = []

    for lookback_label, date_mask in [
        ("full", fits_df["_date"] >= pd.Timestamp.min),
        ("1m",   fits_df["_date"] >= one_month_ago),
    ]:
        subset = fits_df[date_mask]
        for tenor in OPTION_TENORS:
            tenor_data = subset[subset["tenor"] == tenor]
            if tenor_data.empty:
                continue
            results.append({
                "lookback": lookback_label,
                "horizon":  TENOR_TO_HORIZON[tenor],
                "mu":       float(tenor_data["mu"].mean()),
                "sigma":    float(tenor_data["sigma"].mean()),
                "nu":       float(tenor_data["nu"].mean()),
                "n_obs":    len(tenor_data),
            })
            logger.info(
                f"  AVG   {TENOR_TO_HORIZON[tenor]:>4s} / {lookback_label:>4s}  "
                f"mu={tenor_data['mu'].mean():+.6f}  "
                f"sigma={tenor_data['sigma'].mean():.6f}  "
                f"nu={tenor_data['nu'].mean():.2f}  "
                f"(n={len(tenor_data)})"
            )

    return pd.DataFrame(results)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate BTC return distributions from options vol surfaces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python distributions_options.py\n"
            "  python distributions_options.py --end-date 2026-03-09\n"
            "  python distributions_options.py --asset BTC "
            "--start-date 2025-03-01 --end-date 2026-03-09\n"
        ),
    )
    parser.add_argument("--asset",      default="BTC", help="Asset symbol (default: BTC)")
    parser.add_argument("--start-date", default=None,  help="Start date YYYY-MM-DD (default: 52w before end)")
    parser.add_argument("--end-date",   default=None,  help="End date YYYY-MM-DD (default: today)")

    args = parser.parse_args()

    df = estimate_distributions_options(
        end_date=args.end_date,
        start_date=args.start_date,
        asset=args.asset,
    )

    # Pretty-print
    print(f"\n{'=' * 75}")
    print("  Option-Implied Student-t Parameters  (SSVI + Breeden-Litzenberger)")
    print(f"  nu bounded to [{NU_MIN:.0f}, {NU_MAX:.0f}]")
    print(f"{'=' * 75}\n")

    display = df.copy()
    display["mu"]    = display["mu"].map(lambda x: f"{x:+.6f}")
    display["sigma"] = display["sigma"].map(lambda x: f"{x:.6f}")
    display["nu"]    = display["nu"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))

    print(f"\n{'-' * 75}")
    print(f"  Total parameter sets: {len(df)}")
    print(f"{'-' * 75}\n")
