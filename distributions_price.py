#!/usr/bin/env python3
"""
BTC Return Distribution Estimator (Empirical Prices)
=====================================================

Estimates BTC return distributions from empirical historical Friday 8 AM UTC
prices fetched via ``glassnode_price_fetcher.py``.

For each return horizon (1w, 2w, 4w, 8w, 13w) a Student-t distribution is
MLE-fitted with degrees-of-freedom (ν) bounded to [3, 10].

Produces **10 parameter sets**:
    - 5 horizons  ×  full history   (start_date → end_date)
    - 5 horizons  ×  trailing 1 yr  (end_date − 1 year → end_date)

Usage (import):
    from distributions_price import estimate_distributions
    results = estimate_distributions()

Usage (CLI):
    python distributions_price.py
    python distributions_price.py --end-date 2026-03-09
    python distributions_price.py --asset BTC --start-date 2020-01-01 --end-date 2026-03-09
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

from glassnode_price_fetcher import fetch_glassnode_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

#  (weeks, overlapping?, display label)
HORIZONS: list[tuple[int, bool, str]] = [
    (1,  False, "1w"),
    (2,  True,  "2w"),
    (4,  True,  "4w"),
    (8,  True,  "8w"),
    (13, True,  "13w"),
]

NU_MIN = 2.0
NU_MAX = 30.0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _default_dates(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve default dates: end = yesterday, start = 210 weeks before end."""
    if end_date is None:
        # Default to yesterday to avoid fetching data that isn't published yet.
        end_dt = datetime.now(timezone.utc) - timedelta(days=1)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start_date is None:
        start_dt = end_dt - timedelta(weeks=210)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _get_friday_prices(
    start_date: str,
    end_date: str,
    asset: str = "BTC",
) -> pd.DataFrame:
    """
    Fetch daily 8 AM UTC prices and filter to Fridays only.

    Returns a DataFrame with columns ``['date', 'price']``, sorted by date,
    containing one row per Friday.
    """
    df = fetch_glassnode_price(
        start_date=start_date,
        end_date=end_date,
        asset=asset,
        resolution="24h",
        time_utc=8,
    )

    price_col = "price_usd_08utc"

    df["date"] = pd.to_datetime(df["date"])
    # Friday = weekday 4
    df = df[df["date"].dt.weekday == 4].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[price_col])

    return df[["date", price_col]].rename(columns={price_col: "price"})


# ──────────────────────────────────────────────────────────────────────────────
# Log-Return Computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_log_returns(
    friday_prices: pd.DataFrame,
    weeks: int,
    overlapping: bool = True,
) -> np.ndarray:
    """
    Compute log returns  r = ln(P_t / P_{t−n})  from a Friday price series.

    Parameters
    ----------
    friday_prices : DataFrame
        Must have a ``'price'`` column sorted by date (one row per Friday).
    weeks : int
        Return horizon in weeks.
    overlapping : bool
        If True, compute a rolling return at every Friday (stride = 1 week).
        If False, compute non-overlapping returns (stride = ``weeks``).

    Returns
    -------
    np.ndarray of log returns.
    """
    prices = friday_prices["price"].values

    if overlapping:
        # Rolling: at each index i ≥ weeks, r_i = ln(P_i / P_{i − weeks})
        returns = np.log(prices[weeks:] / prices[:-weeks])
    else:
        # Non-overlapping: step by `weeks` Fridays
        indices = list(range(0, len(prices), weeks))
        selected = prices[indices]
        returns = np.log(selected[1:] / selected[:-1])

    return returns


# ──────────────────────────────────────────────────────────────────────────────
# Student-t MLE Fitting
# ──────────────────────────────────────────────────────────────────────────────

def fit_student_t(
    returns: np.ndarray,
    nu_min: float = NU_MIN,
    nu_max: float = NU_MAX,
) -> tuple[float, float, float]:
    """
    MLE fit of a Student-t distribution with ν bounded to ``[nu_min, nu_max]``.

    Parameters
    ----------
    returns : array-like
        Observed log returns.
    nu_min, nu_max : float
        Bounds on degrees of freedom.

    Returns
    -------
    (mu, sigma, nu) : tuple[float, float, float]
        Location, scale, and degrees-of-freedom of the best-fit Student-t.
    """
    returns = returns[np.isfinite(returns)]

    if len(returns) < 5:
        raise ValueError(f"Insufficient data points ({len(returns)}) for fitting")

    def neg_log_likelihood(params):
        nu, mu, sigma = params
        if sigma <= 0:
            return 1e12
        return -np.sum(student_t.logpdf(returns, df=nu, loc=mu, scale=sigma))

    # Initial moment estimates
    mu0 = np.mean(returns)
    sigma0 = np.std(returns, ddof=1)

    # Try several starting ν to avoid local minima
    best_result = None
    best_nll = np.inf

    for nu_init in [3.5, 7.0, 12, 20]:
        try:
            result = minimize(
                neg_log_likelihood,
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
        raise RuntimeError("Student-t MLE fitting failed for all starting points")

    nu, mu, sigma = best_result.x
    return mu, sigma, nu


# ──────────────────────────────────────────────────────────────────────────────
# Main Public Function
# ──────────────────────────────────────────────────────────────────────────────

def estimate_distributions(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    asset: str = "BTC",
) -> pd.DataFrame:
    """
    Estimate Student-t distribution parameters for BTC log returns at five
    horizons (1w, 2w, 4w, 8w, 13w), each for two lookback windows
    (full history and trailing 1 year) → **10 parameter sets**.

    Parameters
    ----------
    end_date : str or None
        End date ``'YYYY-MM-DD'``.  Defaults to today.
    start_date : str or None
        Start date ``'YYYY-MM-DD'``.  Defaults to 210 weeks before end_date.
    asset : str
        Asset symbol (default ``'BTC'``).

    Returns
    -------
    pd.DataFrame
        Columns: ``[lookback, horizon, mu, sigma, nu, n_returns]``
    """
    _start, _end = _default_dates(end_date, start_date)

    logger.info(f"Estimating return distributions for {asset}")
    logger.info(f"  Full history:  {_start}  ->  {_end}")

    # ── Step 1: Fetch Friday 8 AM UTC prices ─────────────────────────────
    friday_prices = _get_friday_prices(_start, _end, asset)
    logger.info(f"  Friday prices: {len(friday_prices)} data points")

    if len(friday_prices) < 14:
        raise ValueError(
            f"Only {len(friday_prices)} Friday prices found.  "
            f"Need at least 14 for 13-week returns."
        )

    # ── Step 2: Derive the 1-year lookback subset ────────────────────────
    end_dt = pd.to_datetime(_end)
    one_year_ago = end_dt - pd.DateOffset(years=1)
    friday_prices_1y = friday_prices[
        friday_prices["date"] >= one_year_ago
    ].reset_index(drop=True)

    logger.info(
        f"  1-year subset: {len(friday_prices_1y)} data points  "
        f"(from {one_year_ago.strftime('%Y-%m-%d')})"
    )

    # ── Step 3: Fit Student-t for every (lookback × horizon) ─────────────
    results: list[dict] = []

    datasets = [
        ("full", friday_prices),
        ("1y",   friday_prices_1y),
    ]

    for lookback_label, price_df in datasets:
        for weeks, overlapping, horizon_label in HORIZONS:
            returns = _compute_log_returns(price_df, weeks, overlapping)

            if len(returns) < 5:
                logger.warning(
                    f"  SKIP  {horizon_label:>4s} / {lookback_label:>4s}  "
                    f"only {len(returns)} returns (need >= 5)"
                )
                continue

            mu, sigma, nu = fit_student_t(returns)

            results.append({
                "lookback":  lookback_label,
                "horizon":   horizon_label,
                "mu":        mu,
                "sigma":     sigma,
                "nu":        nu,
                "n_returns": len(returns),
            })

            logger.info(
                f"  FIT   {horizon_label:>4s} / {lookback_label:>4s}  "
                f"μ={mu:+.6f}  σ={sigma:.6f}  ν={nu:.2f}  "
                f"(n={len(returns)})"
            )

    result_df = pd.DataFrame(results)
    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

# Colour palette: one colour per horizon label, consistent across lookbacks
_HORIZON_COLOURS = {
    "1w":  "#1f77b4",
    "2w":  "#ff7f0e",
    "4w":  "#2ca02c",
    "8w":  "#d62728",
    "13w": "#9467bd",
}


def plot_distributions(
    results: pd.DataFrame,
    *,
    x_range: tuple[float, float] = (-0.6, 0.6),
    n_points: int = 500,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot the fitted Student-t PDFs from ``estimate_distributions()`` output.

    Creates one subplot per lookback window (e.g. *full* and *1y*), each
    containing the Student-t density curves for every horizon overlaid.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``estimate_distributions()`` with columns
        ``[lookback, horizon, mu, sigma, nu, n_returns]``.
    x_range : tuple[float, float]
        Log-return range for the x-axis (default: -0.6 to +0.6).
    n_points : int
        Number of evaluation points for the density curve.
    save_path : str or None
        If given, save the figure to this path (e.g. ``"dist_price.png"``).
    show : bool
        If True (default), display the plot interactively.
    """
    import matplotlib.pyplot as plt

    lookbacks = results["lookback"].unique()
    n_panels = len(lookbacks)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7 * n_panels, 5),
        sharey=False,
        squeeze=False,
    )

    x = np.linspace(x_range[0], x_range[1], n_points)

    for idx, lookback in enumerate(lookbacks):
        ax = axes[0, idx]
        subset = results[results["lookback"] == lookback]

        for _, row in subset.iterrows():
            mu = row["mu"]
            sigma = row["sigma"]
            nu = row["nu"]
            horizon = row["horizon"]
            n_ret = int(row["n_returns"])
            colour = _HORIZON_COLOURS.get(horizon, "#333333")

            y = student_t.pdf(x, df=nu, loc=mu, scale=sigma)
            label = (
                f"{horizon}  "
                f"(mu={mu:+.4f}, sig={sigma:.4f}, nu={nu:.1f}, n={n_ret})"
            )
            ax.plot(x, y, color=colour, linewidth=1.5, label=label)

        ax.set_title(f"Lookback: {lookback}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Log return", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    fig.suptitle(
        "Fitted Student-t Distributions (Empirical Prices)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Plot saved to {save_path}")

    if show:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate BTC return distributions from Friday 8 AM UTC prices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python distributions_price.py\n"
            "  python distributions_price.py --end-date 2026-03-09\n"
            "  python distributions_price.py --asset BTC "
            "--start-date 2022-01-01 --end-date 2026-03-09\n"
        ),
    )
    parser.add_argument("--asset",      default="BTC", help="Asset symbol (default: BTC)")
    parser.add_argument("--start-date", default=None,  help="Start date YYYY-MM-DD (default: 210w before end)")
    parser.add_argument("--end-date",   default=None,  help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--plot",       action="store_true",
                        help="Plot the fitted Student-t distributions")
    parser.add_argument("--plot-file",  default="dist.png",
                        help="Save plot to file (e.g. dist_price.png). Implies --plot")

    args = parser.parse_args()

    df = estimate_distributions(
        end_date=args.end_date,
        start_date=args.start_date,
        asset=args.asset,
    )

    # ── Pretty-print results ─────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  Student-t Distribution Parameters  (MLE, nu bounded to [{NU_MIN:.0f}, {NU_MAX:.0f}])")
    print(f"{'=' * 75}\n")

    # Format for display
    display = df.copy()
    display["mu"]    = display["mu"].map(lambda x: f"{x:+.6f}")
    display["sigma"] = display["sigma"].map(lambda x: f"{x:.6f}")
    display["nu"]    = display["nu"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))

    print(f"\n{'-' * 75}")
    print(f"  Total parameter sets: {len(df)}")
    print(f"{'-' * 75}\n")

    # ── Plot if requested ────────────────────────────────────────────────
    if args.plot or args.plot_file:
        plot_distributions(
            df,
            save_path=args.plot_file,
            show=args.plot_file is None,  # show interactively unless saving
        )
