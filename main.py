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
WINDOW_DAYS = 252  # number of trading days (~1 year) per rolling fit
EPS = 1e-8  # small floor to keep variances positive


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
    var[0] = max(np.var(residuals, ddof=1), EPS)
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


def fit_garch_11(returns: pd.Series, initial_params: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate GARCH(1,1) parameters and return conditional volatility."""
    returns_array = returns.to_numpy()
    mu_init = returns_array.mean()
    var_init = np.var(returns_array, ddof=1)
    if not np.isfinite(var_init) or var_init <= 0:
        raise ValueError("Variance of returns is not positive; cannot fit GARCH.")

    # Reasonable starting values for optimization.
    if initial_params is None:
        initial_params = np.array([mu_init, max(1e-6, 0.1 * var_init), 0.05, 0.9], dtype=float)
    bounds = [
        (None, None),  # mu
        (1e-12, None),  # omega
        (1e-12, 0.999),  # alpha
        (1e-12, 0.999),  # beta
    ]

    def _fit_once(start_params: np.ndarray, method: str) -> minimize:
        return minimize(
            _neg_log_likelihood,
            start_params,
            args=(returns_array,),
            bounds=bounds,
            method=method,
        )

    result = _fit_once(initial_params, "L-BFGS-B")

    if (not result.success) or (not np.isfinite(result.fun)):
        # Retry with a safer persistence guess and Powell, which is more forgiving.
        fallback_start = np.array([mu_init, max(1e-6, 0.05 * var_init), 0.05, 0.85], dtype=float)
        result = _fit_once(fallback_start, "Powell")

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"GARCH optimization failed: {result.message}")

    params = result.x
    residuals = returns_array - params[0]
    conditional_var = _conditional_variance(params, residuals)
    conditional_vol = np.sqrt(conditional_var)
    return params, conditional_vol


def rolling_garch_predictions(df: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fit a GARCH(1,1) model on a rolling window and return predicted volatility per day.

    For each date with at least `window` prior returns, fit on the trailing window and
    record the most recent conditional volatility estimate.
    """
    df = df.copy()
    df["LogPrice"] = np.log(df["Price"])
    log_returns = df["LogPrice"].diff().dropna()

    if len(log_returns) < window:
        raise ValueError(f"Need at least {window} return observations for rolling GARCH.")

    records = []
    last_params: np.ndarray | None = None
    for end in range(window, len(log_returns) + 1):
        window_returns = log_returns.iloc[end - window : end]
        params, conditional_vol = fit_garch_11(window_returns, initial_params=last_params)
        last_params = params

        # Align predicted volatility to the date of the last return in the window.
        date_idx = window_returns.index[-1]
        records.append(
            {
                "Date": df.loc[date_idx, "Date"],
                "LogReturn": window_returns.iloc[-1],
                "PredictedVolatility": conditional_vol[-1],
            }
        )

    return pd.DataFrame(records), last_params


def main() -> None:
    df = load_price_history(DATA_PATH)

    output, params = rolling_garch_predictions(df, WINDOW_DAYS)
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
