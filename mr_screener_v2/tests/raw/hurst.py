import numpy as np
import pandas as pd


def hurst_exponent(series, max_lag: int = 100) -> dict:
    """
    Estimate Hurst exponent via R/S (rescaled range) method.
    Applied to the static spread from EG cointegration.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]

    lags = range(2, min(max_lag, len(arr) // 2))
    tau = [np.std(arr[lag:] - arr[:-lag]) for lag in lags]

    # Filter out zero std (degenerate lags)
    valid = [(lag, t) for lag, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return {
            "hurst":     float("nan"),
            "verdict":   "insufficient_data",
            "tradeable": False,
            "space":     "raw",
        }

    log_lags = np.log([v[0] for v in valid])
    log_tau  = np.log([v[1] for v in valid])

    hurst = float(np.polyfit(log_lags, log_tau, 1)[0])

    if hurst < 0.45:
        verdict = "strong_mr"
    elif hurst < 0.50:
        verdict = "mild_mr"
    elif hurst < 0.55:
        verdict = "random_walk"
    else:
        verdict = "trending"

    return {
        "hurst":     round(hurst, 4),
        "verdict":   verdict,
        "tradeable": hurst < 0.50,
        "space":     "raw",
    }
