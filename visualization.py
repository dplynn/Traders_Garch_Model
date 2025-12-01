"""
Visualize the historical CVNA price series with the predicted GARCH volatility overlayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRICE_PATH = Path("CVNAPriceHistory - 5yr.csv")
VOL_PATH = Path("predicted_volatility.csv")
OUTPUT_PATH = Path("realvspred_volatility_overlay.png")
# Rolling window (in trading days) for realized volatility calculation.
REALIZED_WINDOW = 21


def load_price_history(path: Path) -> pd.DataFrame:
    """Load price history CSV with parsed dates and sorted chronologically."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Price"]]


def load_predicted_volatility(path: Path) -> pd.DataFrame:
    """Load predicted volatility output from the GARCH model run."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def compute_realized_volatility(price_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Compute realized volatility as rolling standard deviation of log returns.

    Returns a DataFrame with Date and RealizedVol columns, aligned to the end of the rolling window.
    """
    price_df = price_df.copy()
    price_df["LogPrice"] = np.log(price_df["Price"])
    price_df["LogReturn"] = price_df["LogPrice"].diff()

    # Rolling std of returns; multiplied by sqrt(window) to express comparable scale.
    realized = price_df["LogReturn"].rolling(window=window).std() * np.sqrt(window)
    out = pd.DataFrame({"Date": price_df["Date"], "RealizedVol": realized})
    return out.dropna().reset_index(drop=True)


def align_volatility_series(
    predicted_df: pd.DataFrame, realized_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Align predicted and realized volatility by date.

    Returns (dates, predicted_vol, realized_vol) over the intersection of dates.
    """
    merged = pd.merge(predicted_df, realized_df, on="Date", how="inner")
    return merged["Date"], merged["PredictedVolatility"], merged["RealizedVol"]


def plot_realized_vs_predicted(dates: pd.Series, predicted: pd.Series, realized: pd.Series, output_path: Path) -> None:
    """Create a line chart comparing realized volatility and predicted volatility."""
    plt.figure(figsize=(10, 6))
    plt.plot(dates, predicted, color="tab:orange", label="Predicted Volatility")
    plt.plot(dates, realized, color="tab:green", label=f"Realized Volatility ({REALIZED_WINDOW}-day)")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.title("Realized vs Predicted GARCH Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    price_df = load_price_history(PRICE_PATH)
    predicted_vol_df = load_predicted_volatility(VOL_PATH)
    realized_vol_df = compute_realized_volatility(price_df, REALIZED_WINDOW)

    dates, predicted, realized = align_volatility_series(predicted_vol_df, realized_vol_df)
    if dates.empty:
        raise ValueError("No overlapping dates between predicted and realized volatility series.")

    plot_realized_vs_predicted(dates, predicted, realized, OUTPUT_PATH)
    print(f"Saved realized vs predicted volatility chart to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
