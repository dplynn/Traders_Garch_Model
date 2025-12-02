"""
Enhanced GARCH(1,1) Model with Complete Data Output

OUTPUT INCLUDES:
- Date
- Price (actual CVNA closing price)
- LogReturn (log return: ln(P_t / P_{t-1}))
- LogReturn_Pct (log return as percentage)
- GARCH_30D_Vol_Forecast (predicted 30-day volatility)
- GARCH_1D_Vol_Forecast (predicted 1-day volatility)
- Alpha (ARCH coefficient)
- Beta (GARCH coefficient)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FILE = "CVNAPriceHistory - 5yr.csv"
OUTPUT_FILE = "garch_complete_output.csv"

LOOKBACK_WINDOW = 252  # Use last 252 trading days to fit model
FORECAST_HORIZON = 30  # Forecast 30 days ahead


# ============================================================================
# GARCH(1,1) CORE FUNCTIONS
# ============================================================================

def _conditional_variance(params: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Recursively compute conditional variance for GARCH(1,1)."""
    mu, omega, alpha, beta = params
    n = len(residuals)
    var = np.empty(n, dtype=float)
    var[0] = np.var(residuals, ddof=1)
    
    for t in range(1, n):
        var[t] = omega + alpha * residuals[t-1]**2 + beta * var[t-1]
    
    return var


def _neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1) estimation."""
    mu, omega, alpha, beta = params
    
    if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
        return np.inf
    
    residuals = returns - mu
    var = _conditional_variance(params, residuals)
    
    if np.any(var <= 0) or np.isnan(var).any():
        return np.inf
    
    ll = 0.5 * (np.log(2 * np.pi) + np.log(var) + (residuals**2) / var)
    return float(np.sum(ll))


def fit_garch_11(returns: np.ndarray) -> np.ndarray | None:
    """Fit GARCH(1,1) model to return series."""
    mu_init = returns.mean()
    initial_params = np.array([mu_init, 1e-6, 0.05, 0.9], dtype=float)
    bounds = [
        (None, None),
        (1e-12, None),
        (1e-12, 0.999),
        (1e-12, 0.999),
    ]
    
    result = minimize(
        _neg_log_likelihood,
        initial_params,
        args=(returns,),
        bounds=bounds,
        method="L-BFGS-B",
    )
    
    if result.success:
        return result.x
    return None


def forecast_volatility(params: np.ndarray, returns: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast conditional volatility for next 'horizon' days (annualized)."""
    mu, omega, alpha, beta = params
    residuals = returns - mu
    conditional_var = _conditional_variance(params, residuals)
    
    last_var = conditional_var[-1]
    last_resid_sq = residuals[-1]**2
    
    forecast_var = np.empty(horizon)
    forecast_var[0] = omega + alpha * last_resid_sq + beta * last_var
    
    for t in range(1, horizon):
        forecast_var[t] = omega + (alpha + beta) * forecast_var[t-1]
    
    # Convert to annualized volatility
    annualized_vol = np.sqrt(forecast_var * 252)
    return annualized_vol


# ============================================================================
# ENHANCED OUTPUT GENERATION
# ============================================================================

def rolling_garch_with_returns(prices: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    """
    Generate GARCH forecasts AND include price/return data in output.
    
    Returns DataFrame with:
    - Date
    - Price
    - LogReturn
    - LogReturn_Pct
    - GARCH_30D_Vol_Forecast
    - GARCH_1D_Vol_Forecast
    - Alpha
    - Beta
    """
    prices = prices.copy()
    
    # Calculate log returns
    prices["LogPrice"] = np.log(prices["Price"])
    prices["LogReturn"] = prices["LogPrice"].diff()
    prices["LogReturn_Pct"] = prices["LogReturn"] * 100  # Convert to percentage
    
    results = []
    total_days = len(prices) - lookback
    
    print(f"Generating forecasts for {total_days} trading days...")
    
    for t in range(lookback, len(prices)):
        # Get rolling window of returns
        window_returns = prices.iloc[t-lookback:t]["LogReturn"].dropna().values
        
        if len(window_returns) < 30:
            continue
        
        # Fit GARCH
        params = fit_garch_11(window_returns)
        
        if params is None:
            # If GARCH fails, still output the row but with NaN forecasts
            results.append({
                "Date": prices.iloc[t]["Date"],
                "Price": prices.iloc[t]["Price"],
                "LogReturn": prices.iloc[t]["LogReturn"],
                "LogReturn_Pct": prices.iloc[t]["LogReturn_Pct"],
                "GARCH_30D_Vol_Forecast": np.nan,
                "GARCH_1D_Vol_Forecast": np.nan,
                "Alpha": np.nan,
                "Beta": np.nan,
            })
            continue
        
        # Forecast volatility
        vol_forecast = forecast_volatility(params, window_returns, horizon)
        
        # Combine everything
        results.append({
            "Date": prices.iloc[t]["Date"],
            "Price": prices.iloc[t]["Price"],
            "LogReturn": prices.iloc[t]["LogReturn"],
            "LogReturn_Pct": prices.iloc[t]["LogReturn_Pct"],
            "GARCH_30D_Vol_Forecast": vol_forecast.mean(),
            "GARCH_1D_Vol_Forecast": vol_forecast[0],
            "Alpha": params[2],
            "Beta": params[3],
        })
        
        # Progress indicator
        if (t - lookback) % 50 == 0:
            pct_complete = ((t - lookback + 1) / total_days) * 100
            print(f"Progress: {pct_complete:.1f}% ({t - lookback + 1}/{total_days} days)")
    
    return pd.DataFrame(results)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_price_history(path: Path) -> pd.DataFrame:
    """Load and clean price history CSV."""
    df = pd.read_csv(path)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date", "Price"]]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("GARCH(1,1) Model with Complete Data Output")
    print("="*70)
    print(f"Lookback window: {LOOKBACK_WINDOW} days")
    print(f"Forecast horizon: {FORECAST_HORIZON} days")
    print("="*70)
    
    # Load data
    print(f"\nLoading price data from {INPUT_FILE}...")
    df = load_price_history(Path(INPUT_FILE))
    print(f"Loaded {len(df)} days of price data")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Generate comprehensive output
    print(f"\nGenerating GARCH forecasts with returns...\n")
    results = rolling_garch_with_returns(df, LOOKBACK_WINDOW, FORECAST_HORIZON)
    
    if results.empty:
        raise ValueError("No forecasts generated.")
    
    # Save results
    results.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*70)
    print(f"SUCCESS! Saved {len(results)} rows to {OUTPUT_FILE}")
    print("="*70)
    
    # Display summary
    print("\nSummary Statistics:")
    print("-"*70)
    print("\nPrice Statistics:")
    print(results["Price"].describe())
    
    print("\nLog Return Statistics (%):")
    print(results["LogReturn_Pct"].describe())
    
    print("\n30-Day Volatility Forecast Statistics:")
    print(results["GARCH_30D_Vol_Forecast"].describe())
    
    print("\nSample Output (First 10 rows):")
    print("-"*70)
    print(results.head(10).to_string())
    
    print("\n\nSample Output (Last 10 rows):")
    print("-"*70)
    print(results.tail(10).to_string())
    
    print("\n" + "="*70)
    print("Output includes:")
    print("  - Price: CVNA closing price")
    print("  - LogReturn: ln(P_t / P_{t-1})")
    print("  - LogReturn_Pct: Log return as percentage")
    print("  - GARCH_30D_Vol_Forecast: Predicted 30-day volatility")
    print("  - GARCH_1D_Vol_Forecast: Predicted 1-day volatility")
    print("  - Alpha, Beta: GARCH model parameters")
    print("="*70)


if __name__ == "__main__":
    main()
