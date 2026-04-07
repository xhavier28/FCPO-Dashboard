THRESHOLDS = {
    "adf_pvalue":          0.10,
    "coint_pvalue":        0.05,
    "hurst_max":           0.50,
    "hurst_strong":        0.45,
    "beta_ar1_min":        0.0,
    "beta_ar1_max":        1.0,
    "half_life_min":       1,    # days
    "half_life_max":       20,   # ~1 trading month
    "half_life_sweet_min": 2,
    "half_life_sweet_max": 10,
}

KALMAN = {"delta": 1e-4, "Ve": 0.001}
