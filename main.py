"""
Fit a GARCH(1,1) model to the 5-year CVNA price history and
output the conditional volatility time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


DATA_PATH = Path("CVNAPriceHistory - 5yr.csv")
OUTPUT_PATH = Path("predicted_volatility.csv")


def load_price_history(path: Path) -> pd.DataFrame:
    """Load and clean the price history CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find price history CSV at {path}")

    df = pd.read_csv(path)

    # Ensure price is numeric and drop rows without a valid price.
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # Parse dates and sort oldest to newest.
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Price"]]


def _conditional_variance(params: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Recursively compute conditional variance for GARCH(1,1)."""
    mu, omega, alpha, beta = params
    n = residuals.shape[0]
    var = np.empty(n, dtype=float)

    # Start variance at the sample variance of the residuals to stabilize recursion.
    var[0] = np.var(residuals, ddof=1)
    for t in range(1, n):
        var[t] = omega + alpha * residuals[t - 1] ** 2 + beta * var[t - 1]
    return var


def _neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood under Gaussian innovations."""
    mu, omega, alpha, beta = params

    # Parameter constraints to keep the model stationary and variances positive.
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return np.inf

    residuals = returns - mu
    var = _conditional_variance(params, residuals)

    if np.any(var <= 0) or np.isnan(var).any():
        return np.inf

    ll = 0.5 * (np.log(2 * np.pi) + np.log(var) + (residuals**2) / var)
    return float(np.sum(ll))


def fit_garch_11(returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate GARCH(1,1) parameters and return conditional volatility."""
    returns_array = returns.to_numpy()
    mu_init = returns_array.mean()

    # Reasonable starting values for optimization.
    initial_params = np.array([mu_init, 1e-6, 0.05, 0.9], dtype=float)
    bounds = [
        (None, None),  # mu
        (1e-12, None),  # omega
        (1e-12, 0.999),  # alpha
        (1e-12, 0.999),  # beta
    ]

    result = minimize(
        _neg_log_likelihood,
        initial_params,
        args=(returns_array,),
        bounds=bounds,
        method="L-BFGS-B",
    )

    if not result.success:
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    params = result.x
    residuals = returns_array - params[0]
    conditional_var = _conditional_variance(params, residuals)
    conditional_vol = np.sqrt(conditional_var)
    return params, conditional_vol


def main() -> None:
    df = load_price_history(DATA_PATH)

    df["LogPrice"] = np.log(df["Price"])
    log_returns = df["LogPrice"].diff().dropna()

    if len(log_returns) < 30:
        raise ValueError("Not enough data to fit GARCH(1,1); need at least 30 return observations.")

    params, conditional_vol = fit_garch_11(log_returns)

    output = pd.DataFrame(
        {
            "Date": df.loc[log_returns.index, "Date"],
            "LogReturn": log_returns.values,
            "PredictedVolatility": conditional_vol,
        }
    )
    output.to_csv(OUTPUT_PATH, index=False)

    mu, omega, alpha, beta = params
    print("Estimated parameters:")
    print(f"  mu    (mean return): {mu:.6f}")
    print(f"  omega (constant)   : {omega:.6f}")
    print(f"  alpha (ARCH term)  : {alpha:.6f}")
    print(f"  beta  (GARCH term) : {beta:.6f}")
    print("\nSaved conditional volatility series to", OUTPUT_PATH)
    print("\nRecent predicted volatilities:")
    print(output.tail(5))


if __name__ == "__main__":
    main()
