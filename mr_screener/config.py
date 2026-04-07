DATA = {
    "freq":         "hourly",
    "bars_per_day": 5.5,       # FCPO: 5.5 trading hours per day
                               # Change for your market: 24.0 crypto, 6.5 US stocks
}

ALIGNMENT = {
    "mode":        "intraday_positional",
    "min_overlap": 200,        # minimum matched hourly bars
    "warn_below":  500,
}

THRESHOLDS = {
    # Individual stationarity — warn only, never gate
    "adf_pvalue":          0.10,
    "adf_warn_only":       True,

    # Cointegration — EG p-value thresholds
    "coint_pvalue":        0.05,   # strict (STRONG tier)
    "coint_pvalue_loose":  0.20,   # loose  (MARGINAL tier)

    # Hurst
    "hurst_max":           0.50,   # strict (STRONG)
    "hurst_strong":        0.45,
    "hurst_max_loose":     0.60,   # loose  (MARGINAL)
    "hurst_warn_only":     True,
    "require_hurst":       False,

    # OU half-life — HOURLY thresholds (for intraday data)
    "half_life_min_hours":   2,
    "half_life_max_hours":   165,  # 30 days × 5.5h
    "half_life_sweet_min_h": 2,
    "half_life_sweet_max_h": 27,   # ~5 trading days

    # OU half-life — DAILY thresholds (fallback for daily data)
    "half_life_min":       1,
    "half_life_max":       60,
    "half_life_sweet_min": 2,
    "half_life_sweet_max": 10,

    "beta_ar1_min":        0.0,
    "beta_ar1_max":        1.0,

    # Data minimums
    "min_bars_warn":       100,
    "min_bars_error":      30,
}

KALMAN = {"delta": 1e-4, "Ve": 0.001}
