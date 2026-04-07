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
    "hurst_warn_only":     True,   # Hurst failure alone never causes REJECT if coint passes

    # Gate logic
    "require_hurst":       False,  # cointegration alone is enough to proceed

    # OU half-life (days)
    "half_life_min":       1,
    "half_life_max":       20,
    "half_life_sweet_min": 2,
    "half_life_sweet_max": 10,
    "beta_ar1_min":        0.0,
    "beta_ar1_max":        1.0,

    # Data minimums
    "min_bars_warn":       100,
    "min_bars_error":      30,
}

KALMAN = {"delta": 1e-4, "Ve": 0.001}
