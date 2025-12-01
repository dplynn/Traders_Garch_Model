"""
Visualize the historical CVNA price series with the predicted GARCH volatility overlayed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PRICE_PATH = Path("CVNAPriceHistory - 5yr.csv")
VOL_PATH = Path("predicted_volatility.csv")
OUTPUT_PATH = Path("volatility_overlay.png")


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


def plot_price_and_volatility(price_df: pd.DataFrame, vol_df: pd.DataFrame, output_path: Path) -> None:
    """Create a dual-axis line chart of price history and predicted volatility."""
    fig, ax_price = plt.subplots(figsize=(10, 6))

    ax_price.plot(price_df["Date"], price_df["Price"], color="tab:blue", label="CVNA Price")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price (USD)", color="tab:blue")
    ax_price.tick_params(axis="y", labelcolor="tab:blue")

    ax_vol = ax_price.twinx()
    ax_vol.plot(vol_df["Date"], vol_df["PredictedVolatility"], color="tab:orange", label="Predicted Volatility")
    ax_vol.set_ylabel("Predicted Volatility", color="tab:orange")
    ax_vol.tick_params(axis="y", labelcolor="tab:orange")

    # Align the date range for both series.
    min_date = min(price_df["Date"].min(), vol_df["Date"].min())
    max_date = max(price_df["Date"].max(), vol_df["Date"].max())
    ax_price.set_xlim(min_date, max_date)

    # Add a combined legend.
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_vol, labels_vol = ax_vol.get_legend_handles_labels()
    ax_price.legend(lines_price + lines_vol, labels_price + labels_vol, loc="upper left")

    ax_price.set_title("CVNA Price History with Predicted GARCH Volatility")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_volatility(price_df: pd.DataFrame, vol_df: pd.DataFrame, output_path: Path) -> None:
    """Create a line chart of predicted volatility over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(vol_df["Date"], vol_df["PredictedVolatility"], color="tab:orange", label="Predicted Volatility")
    plt.xlabel("Date")
    plt.ylabel("Predicted Volatility")
    plt.title("Predicted GARCH Volatility Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main() -> None:
    price_df = load_price_history(PRICE_PATH)
    vol_df = load_predicted_volatility(VOL_PATH)
    plot_volatility(price_df, vol_df, OUTPUT_PATH)
    print(f"Saved visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
