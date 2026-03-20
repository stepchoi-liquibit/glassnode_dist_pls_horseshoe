#!/usr/bin/env python3
"""
Combined Distribution Estimator: Options RND vs Empirical Prices
=================================================================

Calls both ``distributions_options.estimate_distributions_options()`` and
``distributions_price.estimate_distributions()`` with identical date/asset
parameters, merges their Student-t parameter sets, and optionally plots
three overlay charts comparing options-implied vs empirical PDFs for the
1-week, 4-week, and 13-week horizons.

Produces up to **12 parameter sets** (6 options + 6 price, after filtering
price to the 3 shared horizons):

    Options RND:   3 horizons x 2 lookbacks (full, 1m)
    Empirical:     3 horizons x 2 lookbacks (full, 1y)

Usage (import):
    from distributions_combined import estimate_combined
    df = estimate_combined()

Usage (CLI):
    python distributions_combined.py
    python distributions_combined.py --plot
    python distributions_combined.py --plot-file dist_combined.png
    python distributions_combined.py --end-date 2026-03-09 --plot
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

from distributions_options import estimate_distributions_options
from distributions_price import estimate_distributions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# The three horizons shared by both pipelines
SHARED_HORIZONS = ["1w", "4w", "13w"]

# For overlay charts, use the "full" lookback (present in both sources).
CHART_LOOKBACK = "full"

# Visual style per source
SOURCE_STYLES = {
    "options": {"color": "#d62728", "linestyle": "-",  "label": "Options RND"},
    "price":   {"color": "#1f77b4", "linestyle": "--", "label": "Empirical"},
}


# ==============================================================================
# Helpers
# ==============================================================================

def _resolve_dates(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve default dates: end = today, start = 52 weeks before end."""
    if end_date is None:
        end_dt = datetime.now(timezone.utc)
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    if start_date is None:
        start_dt = end_dt - timedelta(weeks=52)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


# ==============================================================================
# Main Public Function
# ==============================================================================

def estimate_combined(
    end_date: Optional[str] = None,
    start_date: Optional[str] = None,
    asset: str = "BTC",
) -> pd.DataFrame:
    """
    Estimate Student-t parameters from both options-implied RND and
    empirical price returns, returning a single merged DataFrame.

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
        Columns: [source, lookback, horizon, mu, sigma, nu, n]
        ``source`` is ``"options"`` or ``"price"``.
    """
    _start, _end = _resolve_dates(end_date, start_date)

    logger.info(f"Combined estimation for {asset}  [{_start} -> {_end}]")

    frames: list[pd.DataFrame] = []

    # -- Options-implied (RND via SSVI + Breeden-Litzenberger) -----------------
    try:
        logger.info("--- Options-implied distributions ---")
        df_opt = estimate_distributions_options(
            end_date=_end, start_date=_start, asset=asset,
        )
        df_opt["source"] = "options"
        df_opt = df_opt.rename(columns={"n_obs": "n"})
        frames.append(df_opt)
    except Exception as exc:
        logger.warning(f"Options estimation failed: {type(exc).__name__}: {exc}")

    # -- Empirical price returns -----------------------------------------------
    try:
        logger.info("--- Empirical price distributions ---")
        df_price = estimate_distributions(
            end_date=_end, start_date=_start, asset=asset,
        )
        df_price["source"] = "price"
        df_price = df_price.rename(columns={"n_returns": "n"})
        # Keep only the shared horizons (drop 2w, 8w)
        df_price = df_price[df_price["horizon"].isin(SHARED_HORIZONS)]
        frames.append(df_price)
    except Exception as exc:
        logger.warning(f"Price estimation failed: {type(exc).__name__}: {exc}")

    if not frames:
        raise RuntimeError("Both options and price estimations failed")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["source", "lookback", "horizon", "mu", "sigma", "nu", "n"]]

    # Sort: options first, then price; within each, by lookback and horizon
    horizon_order = {h: i for i, h in enumerate(SHARED_HORIZONS)}
    combined["_h_ord"] = combined["horizon"].map(horizon_order)
    source_order = {"options": 0, "price": 1}
    combined["_s_ord"] = combined["source"].map(source_order)
    combined = (
        combined.sort_values(["_s_ord", "lookback", "_h_ord"])
        .drop(columns=["_h_ord", "_s_ord"])
        .reset_index(drop=True)
    )

    return combined


# ==============================================================================
# Plotting
# ==============================================================================

def plot_combined(
    results: pd.DataFrame,
    *,
    x_range: tuple[float, float] = (-0.6, 0.6),
    n_points: int = 500,
    save_path: Optional[str] = None,
    show: bool = True,
    lookback: str = CHART_LOOKBACK,
) -> None:
    """
    Plot three overlay charts (1w, 4w, 13w) comparing options-implied
    and empirical Student-t PDFs side by side.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``estimate_combined()`` with columns
        ``[source, lookback, horizon, mu, sigma, nu, n]``.
    x_range : tuple[float, float]
        Log-return range for the x-axis.
    n_points : int
        Number of evaluation points for each density curve.
    save_path : str or None
        If given, save the figure to this path.
    show : bool
        If True, display the plot interactively.
    lookback : str
        Which lookback window to plot (default ``"full"``).
    """
    import matplotlib.pyplot as plt

    subset = results[results["lookback"] == lookback]

    fig, axes = plt.subplots(
        1, len(SHARED_HORIZONS),
        figsize=(7 * len(SHARED_HORIZONS), 5),
        sharey=False,
        squeeze=False,
    )

    x = np.linspace(x_range[0], x_range[1], n_points)

    for idx, horizon in enumerate(SHARED_HORIZONS):
        ax = axes[0, idx]
        hz_data = subset[subset["horizon"] == horizon]

        for _, row in hz_data.iterrows():
            src = row["source"]
            mu = row["mu"]
            sigma = row["sigma"]
            nu = row["nu"]
            n = int(row["n"])
            style = SOURCE_STYLES.get(src, {})

            # Student-t PDF
            y = student_t.pdf(x, df=nu, loc=mu, scale=sigma)
            label = (
                f"{style.get('label', src)} t  "
                f"(mu={mu:+.4f}, sig={sigma:.4f}, nu={nu:.1f}, n={n})"
            )
            ax.plot(
                x, y,
                color=style.get("color", "#333333"),
                linestyle=style.get("linestyle", "-"),
                linewidth=1.5,
                label=label,
            )

            # Normal PDF with same mu, sigma
            y_norm = norm.pdf(x, loc=mu, scale=sigma)
            label_norm = (
                f"{style.get('label', src)} N  "
                f"(mu={mu:+.4f}, sig={sigma:.4f})"
            )
            ax.plot(
                x, y_norm,
                color=style.get("color", "#333333"),
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                label=label_norm,
            )

        ax.set_title(f"Horizon: {horizon}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Log return", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Density", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    fig.suptitle(
        f"Options-Implied vs Empirical Student-t  (lookback: {lookback})",
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


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Estimate BTC return distributions from both options vol surfaces "
            "and empirical prices, then optionally plot overlay comparisons."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python distributions_combined.py\n"
            "  python distributions_combined.py --plot\n"
            "  python distributions_combined.py --plot-file dist_combined.png\n"
            "  python distributions_combined.py --end-date 2026-03-09 --plot\n"
        ),
    )
    parser.add_argument(
        "--asset", default="BTC", help="Asset symbol (default: BTC)",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date YYYY-MM-DD (default: 52 weeks before end)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot overlay charts (options vs price per horizon)",
    )
    parser.add_argument(
        "--plot-file", default="comb_dist.png",
        help="Save plot to file (e.g. dist_combined.png). Implies --plot",
    )

    args = parser.parse_args()

    df = estimate_combined(
        end_date=args.end_date,
        start_date=args.start_date,
        asset=args.asset,
    )

    # -- Resolve dates for header display --------------------------------------
    _start, _end = _resolve_dates(args.end_date, args.start_date)

    # -- Pretty-print ----------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("  Combined Student-t Parameters: Options RND vs Empirical Prices")
    print(f"  Asset: {args.asset}    Date range: {_start} -> {_end}")
    print(f"{'=' * 75}\n")

    display = df.copy()
    display["mu"]    = display["mu"].map(lambda x: f"{x:+.6f}")
    display["sigma"] = display["sigma"].map(lambda x: f"{x:.6f}")
    display["nu"]    = display["nu"].map(lambda x: f"{x:.2f}")
    display["n"]     = display["n"].astype(int)
    print(display.to_string(index=False))

    print(f"\n{'-' * 75}")
    print(f"  Total parameter sets: {len(df)}")
    print(f"{'-' * 75}\n")

    # -- Plot if requested -----------------------------------------------------
    if args.plot or args.plot_file:
        plot_combined(
            df,
            save_path=args.plot_file,
            show=args.plot_file is None,
        )
