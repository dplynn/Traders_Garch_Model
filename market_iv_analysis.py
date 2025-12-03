"""
Visualize GARCH forecasts vs market implied volatility with forward-aligned realized volatility.

The realized volatility series is shifted 30 trading days earlier so each point lines up with
the 30-day-ahead forecasts from the same date (e.g., the realized vol ending 7/31 appears on 7/1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("main.csv")
OUTPUT_PATH = Path("garch_vs_market_iv_analysis.png")
REALIZED_WINDOW = 30  # trading days
TRADING_DAYS_PER_YEAR = 252
LONG_THRESHOLD = 0.05
SHORT_THRESHOLD = -0.05


def load_data(path: Path) -> pd.DataFrame:
    """Load main.csv with parsed dates and chronological ordering."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_forward_realized_vol(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute forward 30-day realized volatility aligned to the forecast start date.

    Uses future log returns so each realized value reflects the next `window` trading days,
    then shifts the series earlier by (window - 1) days to overlay with the forecast date.
    """
    df = df.copy()
    future_returns = df["LogReturn"].shift(-1)
    realized = future_returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Shift to the start of the forecast window (e.g., realized ending 7/31 maps to 7/1).
    df["RealizedVol_30D"] = realized.shift(-(window - 1))
    return df


def plot_garch_vs_iv(df: pd.DataFrame, output_path: Path) -> None:
    """Composite chart: top vol lines, middle vol spread signals, bottom signal distribution."""
    df = df.sort_values("Date")
    vol_df = df.dropna(subset=["GARCH_30D_Vol_Forecast", "Avg_IV", "RealizedVol_30D", "Vol_Spread"])

    garch_line = vol_df["GARCH_30D_Vol_Forecast"]
    market_iv_line = vol_df["Avg_IV"]

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(nrows=3, height_ratios=[3.5, 2.5, 2.2], hspace=0.55)

    # Top: volatility lines with forward realized overlay.
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(vol_df["Date"], garch_line, label="GARCH 30D Forecast", color="#8B0000")
    ax_top.plot(vol_df["Date"], market_iv_line, label="Market Implied Vol", color="#0b3c7f", alpha=0.95)
    ax_top.fill_between(vol_df["Date"], garch_line, market_iv_line, color="#0b3c7f", alpha=0.10, linewidth=0)
    ax_top.plot(
        vol_df["Date"],
        vol_df["RealizedVol_30D"],
        label="Forward Realized Vol (shifted -30d)",
        color="#000000",
        linewidth=1.5,
        alpha=0.50,
    )
    ax_top.set_title("CVNA: GARCH 30D Forecast vs Market Implied Volatility (with Forward Realized)")
    ax_top.set_ylabel("Annualized Volatility")
    ax_top.legend(loc="upper right")
    ax_top.grid(alpha=0.3)

    # Middle: vol spread trading signals (long/short straddle).
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    vol_spread = vol_df["Vol_Spread"]
    dates = vol_df["Date"]

    long_mask = vol_spread >= 0
    short_mask = vol_spread < 0

    ax_mid.bar(dates[long_mask], vol_spread[long_mask], width=3, color="#1fa83a", alpha=0.7, label="Long Straddle")
    ax_mid.bar(dates[short_mask], vol_spread[short_mask], width=3, color="#d62728", alpha=0.6, label="Short Straddle")
    ax_mid.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax_mid.axhline(LONG_THRESHOLD, color="#1fa83a", linestyle="--", linewidth=1.2, alpha=0.8, label="Long Threshold (+5%)")
    ax_mid.axhline(SHORT_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8, label="Short Threshold (-5%)")
    ax_mid.set_ylabel("Vol Spread (GARCH - Market IV)")
    ax_mid.set_title("Trading Signals: Green = Long Straddle, Red = Short Straddle")
    ax_mid.legend(loc="upper right")
    ax_mid.grid(alpha=0.25)

    # Bottom: distribution of trading signals.
    ax_bot = fig.add_subplot(gs[2, 0])
    signal_counts = df["Signal"].value_counts(normalize=True)
    labels = signal_counts.index.tolist()
    sizes = (signal_counts.values * 100).round(1)
    color_map = {"LONG_STRADDLE": "#1fa83a", "SHORT_STRADDLE": "#d62728", "NO_TRADE": "#bfbfbf"}
    colors = [color_map.get(lbl, "#cccccc") for lbl in labels]
    ax_bot.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90, wedgeprops={"linewidth": 0.7, "edgecolor": "white"})
    ax_bot.set_title("Distribution of Trading Signals")

    fig.tight_layout(h_pad=0.8)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data(DATA_PATH)
    df = compute_forward_realized_vol(df, REALIZED_WINDOW)
    plot_garch_vs_iv(df, OUTPUT_PATH)
    print(f"Saved GARCH vs Market IV analysis to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
